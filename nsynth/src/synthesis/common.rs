use crate::benchmark::Problem;
use crate::differentiable::DifferentiableMetadata;
use crate::runtime::verify_problem_code_strict;
use crate::solver::SolveResult;
use std::cell::Cell;
use std::time::{Duration, Instant};

// ─── Gradient-training wall-clock budget ────────────────────────────────────────
//
// `train_program` runs hundreds of gradient steps and is invoked dozens of times
// per restart by the scalar/array gradient synthesizers. On data that never
// converges (e.g. a contradictory teacher-augmented re-fit, or a genuinely
// unsynthesizable problem) the stagnation early-stop is not always enough, so a
// caller could spend minutes sweeping every block to ultimately miss. A caller
// installs a [`TrainDeadline`] guard to cap the *total* time all `train_program`
// runs on this thread may take; once it elapses each run returns near-instantly.
// Default is "no deadline" so callers that don't opt in are unaffected.

thread_local! {
    static TRAIN_DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// RAII guard capping all [`train_program`] runs on this thread to `budget` from
/// now. Restores the previous deadline on drop (nesting-safe).
pub(crate) struct TrainDeadline {
    prev: Option<Instant>,
}
impl TrainDeadline {
    pub(crate) fn set(budget: Duration) -> Self {
        let prev = TRAIN_DEADLINE.with(|c| c.replace(Some(Instant::now() + budget)));
        TrainDeadline { prev }
    }

    /// Like [`set`], but NEVER loosens an already-installed deadline: the effective
    /// deadline becomes `min(now + budget, existing)`. Use this for a per-*attempt*
    /// cap nested inside a per-*query* budget — several sequential solve attempts each
    /// install their own attempt cap, but none may extend past the shared query
    /// deadline, so the whole query stays bounded (otherwise `set` would reset the
    /// clock on every attempt and the query total would be unbounded). Restores the
    /// previous deadline on drop, like `set`.
    pub(crate) fn set_min(budget: Duration) -> Self {
        let candidate = Instant::now() + budget;
        let prev = TRAIN_DEADLINE.with(|c| {
            let existing = c.get();
            let effective = match existing {
                Some(d) => d.min(candidate),
                None => candidate,
            };
            c.replace(Some(effective))
        });
        TrainDeadline { prev }
    }
}
impl Drop for TrainDeadline {
    fn drop(&mut self) {
        TRAIN_DEADLINE.with(|c| c.set(self.prev));
    }
}

/// True once an installed [`TrainDeadline`] has elapsed. Always false when no
/// deadline is set, so non-opted-in callers see no behavior change.
pub(crate) fn train_deadline_exceeded() -> bool {
    TRAIN_DEADLINE
        .with(|c| c.get())
        .is_some_and(|d| Instant::now() >= d)
}

// ─── Constants ────────────────────────────────────────────────────────────────

pub(crate) const N_OPS: usize = 5; // +, -, *, /, %
pub(crate) const N_CMPS: usize = 6; // >, <, >=, <=, ==, !=
pub(crate) const N_CONSTS: usize = 6; // [0, 1, -1, 2, -2, 10]
pub(crate) const MAX_LOOP_ITER: usize = 32;
pub(crate) const MAX_DIGIT_ITER: usize = 20;
pub(crate) const MAX_ARR: usize = 16; // max array length for soft array programs

// ─── Math utilities ───────────────────────────────────────────────────────────

pub(crate) fn softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
    // Single-pass: avoid an extra Vec allocation by fusing temp-scaling into exp.
    let inv_temp = 1.0 / temp;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scaled_max = max * inv_temp;
    let mut exps: Vec<f32> = logits
        .iter()
        .map(|&x| ((x * inv_temp) - scaled_max).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    let inv_sum = 1.0 / sum;
    for e in &mut exps {
        *e *= inv_sum;
    }
    exps
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

pub(crate) fn soft_read(storage: &[f32], weights: &[f32]) -> f32 {
    storage.iter().zip(weights).map(|(s, w)| s * w).sum()
}

/// Weighted mix of: a+b, a-b, a*b, a/b, a%b
pub(crate) fn soft_op(a: f32, b: f32, weights: &[f32]) -> f32 {
    let safe_b = if b.abs() < 1e-6 { 1.0 } else { b };
    let results = [
        a + b,
        a - b,
        a * b,
        a / safe_b,
        a - (a / safe_b).trunc() * safe_b,
    ];
    weights.iter().zip(&results).map(|(w, r)| w * r).sum()
}

/// Extended 7-op version: +, -, *, /, %, |a-b|, max(a,b)
/// Used by synthesize_scalar_expr_only for richer expression discovery.
pub(crate) const N_OPS7: usize = 7;
pub(crate) fn soft_op7(a: f32, b: f32, weights: &[f32]) -> f32 {
    let safe_b = if b.abs() < 1e-6 { 1.0 } else { b };
    let results = [
        a + b,
        a - b,
        a * b,
        a / safe_b,
        a - (a / safe_b).trunc() * safe_b,
        (a - b).abs(),                 // 5: abs_diff
        0.5 * (a + b + (a - b).abs()), // 6: max(a,b) smooth approx
    ];
    weights.iter().zip(&results).map(|(w, r)| w * r).sum()
}

/// Weighted mix of soft comparisons (returns [0,1]).
/// `cmp_temp` is the current training temperature — higher → wider sigmoid boundary
/// → gradients flow for large |d| values (needed for examples like abs_diff where d=±4).
/// The minimum is 0.5 so that comparison boundaries always have some gradient.
pub(crate) fn soft_cmp(a: f32, b: f32, weights: &[f32], cmp_temp: f32) -> f32 {
    let d = a - b;
    // t anneals with training temperature, clamped to [0.5, 2.0] so comparison
    // gradients flow even for large |d| while still sharpening over time.
    let t = cmp_temp.clamp(0.5, 2.0);
    let gauss_var = (t * t * 0.5).max(0.125);
    // Indices match cmp_names: ["<", "<=", "==", ">=", ">", "!="]
    let results = [
        sigmoid(-d / t),                    // 0: < (fires when a < b)
        sigmoid(-d / t),                    // 1: <= (approx same gradient)
        (-(d * d) / gauss_var).exp(),       // 2: == (Gaussian peak at d=0)
        sigmoid(d / t),                     // 3: >= (fires when a > b)
        sigmoid(d / t),                     // 4: > (approx same gradient)
        1.0 - (-(d * d) / gauss_var).exp(), // 5: != (inverse Gaussian)
    ];
    weights.iter().zip(&results).map(|(w, r)| w * r).sum()
}

/// soft_op extended with identity (index 5)
pub(crate) fn soft_op_ext(a: f32, b: f32, weights: &[f32]) -> f32 {
    let safe_b = if b.abs() < 1e-6 { 1.0 } else { b };
    let results = [
        a + b,
        a - b,
        a * b,
        a / safe_b,
        a - (a / safe_b).trunc() * safe_b,
        a, // identity last
    ];
    weights.iter().zip(&results).map(|(w, r)| w * r).sum()
}

pub(crate) fn mse(predicted: &[f32], targets: &[f32]) -> f32 {
    let n = predicted.len() as f32;
    predicted
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / n
}

pub(crate) fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─── Adam optimizer ───────────────────────────────────────────────────────────

pub(crate) struct Adam {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

impl Adam {
    pub(crate) fn new(n: usize, lr: f32) -> Self {
        Self {
            lr,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    pub(crate) fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let lr_t = self.lr * (1.0 - self.b2.powi(self.t as i32)).sqrt()
            / (1.0 - self.b1.powi(self.t as i32));
        for i in 0..params.len() {
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * grads[i];
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * grads[i] * grads[i];
            params[i] -= lr_t * self.m[i] / (self.v[i].sqrt() + self.eps);
        }
    }
}

// ─── Analytical gradient primitives ──────────────────────────────────────────
//
// Backprop through the soft execution primitives. Each function computes both
// the forward output and the gradient w.r.t. the logit parameters.
// This replaces the 2×N FD gradient with a single forward+backward pass.

/// Backward through softmax_temp: given dL/d(weights), compute dL/d(logits).
/// Uses the Jacobian: dw_i/dz_j = w_i * (delta_ij - w_j) / temp
pub(crate) fn softmax_temp_backward(weights: &[f32], d_weights: &[f32], temp: f32) -> Vec<f32> {
    let n = weights.len();
    let inv_temp = 1.0 / temp;
    let mut d_logits = vec![0.0f32; n];
    // dot = sum_j(dL/dw_j * w_j)
    let dot: f32 = d_weights.iter().zip(weights).map(|(dw, w)| dw * w).sum();
    for i in 0..n {
        d_logits[i] = inv_temp * weights[i] * (d_weights[i] - dot);
    }
    d_logits
}

/// Forward + backward for soft_op_ext: out = sum(op_weights[k] * results[k])
/// where results = [a+b, a-b, a*b, a/b, a%b, a]
pub(crate) struct SoftOpExtGrad {
    a: f32,
    b: f32,
    results: [f32; 6],
    op_weights: Vec<f32>,
}

impl SoftOpExtGrad {
    pub(crate) fn forward(a: f32, b: f32, op_logits: &[f32], temp: f32) -> Self {
        let safe_b = if b.abs() < 1e-6 { 1.0 } else { b };
        let results = [
            a + b,
            a - b,
            a * b,
            a / safe_b,
            a - (a / safe_b).trunc() * safe_b,
            a,
        ];
        let op_weights = softmax_temp(op_logits, temp);
        Self {
            a,
            b,
            results,
            op_weights,
        }
    }

    /// Given dL/d_output, return (dL/d_op_logits, dL/da, dL/db)
    pub(crate) fn backward(&self, d_out: f32, temp: f32) -> (Vec<f32>, f32, f32) {
        // dL/d_op_weights[k] = d_out * results[k]
        let d_op_w: Vec<f32> = self.results.iter().map(|&r| d_out * r).collect();
        let d_op_logits = softmax_temp_backward(&self.op_weights, &d_op_w, temp);

        let safe_b = if self.b.abs() < 1e-6 { 1.0 } else { self.b };
        // dL/da = d_out * sum_k(w_k * d_results_k/da)
        let da_results = [1.0, 1.0, self.b, 1.0 / safe_b, 1.0, 1.0]; // d(result_k)/da approx
        let d_a: f32 = d_out
            * self
                .op_weights
                .iter()
                .zip(&da_results)
                .map(|(w, dr)| w * dr)
                .sum::<f32>();
        // dL/db = d_out * sum_k(w_k * d_results_k/db)
        let db_results = [1.0, -1.0, self.a, -self.a / (safe_b * safe_b), -1.0, 0.0]; // approx
        let d_b: f32 = d_out
            * self
                .op_weights
                .iter()
                .zip(&db_results)
                .map(|(w, dr)| w * dr)
                .sum::<f32>();
        (d_op_logits, d_a, d_b)
    }
}

/// Forward + backward for soft_cmp: gate = sum(cmp_weights[k] * cmp_results[k])
pub(crate) struct SoftCmpGrad {
    d: f32,
    t: f32,
    gauss_var: f32,
    cmp_results: [f32; 6],
    cmp_weights: Vec<f32>,
}

impl SoftCmpGrad {
    pub(crate) fn forward(a: f32, b: f32, cmp_logits: &[f32], cmp_temp: f32, temp: f32) -> Self {
        let d = a - b;
        let t = cmp_temp.clamp(0.5, 2.0);
        let gauss_var = (t * t * 0.5).max(0.125);
        let gauss = (-(d * d) / gauss_var).exp();
        let cmp_results = [
            sigmoid(-d / t),
            sigmoid(-d / t),
            gauss,
            sigmoid(d / t),
            sigmoid(d / t),
            1.0 - gauss,
        ];
        let cmp_weights = softmax_temp(cmp_logits, temp);
        Self {
            d,
            t,
            gauss_var,
            cmp_results,
            cmp_weights,
        }
    }

    /// Given dL/d_gate, return (dL/d_cmp_logits, dL/da, dL/db)
    pub(crate) fn backward(&self, d_gate: f32, temp: f32) -> (Vec<f32>, f32, f32) {
        // dL/d_cmp_weights[k] = d_gate * cmp_results[k]
        let d_cmp_w: Vec<f32> = self.cmp_results.iter().map(|&r| d_gate * r).collect();
        let d_cmp_logits = softmax_temp_backward(&self.cmp_weights, &d_cmp_w, temp);

        // dL/dd (d = a - b)
        let sig_pos = sigmoid(self.d / self.t);
        let sig_neg = sigmoid(-self.d / self.t);
        let gauss = self.cmp_results[2];
        let d_gauss_dd = -2.0 * self.d / self.gauss_var * gauss;
        let d_sig_pos_dd = sig_pos * (1.0 - sig_pos) / self.t;
        let d_sig_neg_dd = -sig_neg * (1.0 - sig_neg) / self.t;
        let d_results_dd = [
            d_sig_neg_dd,
            d_sig_neg_dd,
            d_gauss_dd,
            d_sig_pos_dd,
            d_sig_pos_dd,
            -d_gauss_dd,
        ];
        let d_d: f32 = d_gate
            * self
                .cmp_weights
                .iter()
                .zip(&d_results_dd)
                .map(|(w, dr)| w * dr)
                .sum::<f32>();
        (d_cmp_logits, d_d, -d_d) // da = d_d, db = -d_d
    }
}

// ─── Finite-difference gradient ───────────────────────────────────────────────

pub(crate) fn fd_grad<F: Fn(&[f32], f32) -> f32>(
    params: &[f32],
    loss_fn: F,
    temp: f32,
) -> Vec<f32> {
    const EPS: f32 = 1e-3;
    let mut grad = vec![0.0f32; params.len()];
    let mut p = params.to_vec();
    for i in 0..params.len() {
        let orig = p[i];
        p[i] = orig + EPS;
        let lp = loss_fn(&p, temp);
        p[i] = orig - EPS;
        let lm = loss_fn(&p, temp);
        p[i] = orig;
        grad[i] = (lp - lm) / (2.0 * EPS);
    }
    grad
}

// ─── Generic training loop ────────────────────────────────────────────────────

pub(crate) fn try_emit_verify<G>(
    params: &[f32],
    emit_fn: &G,
    problem: &Problem,
    fn_name: &str,
    param_names: &[&str],
) -> Option<SolveResult>
where
    G: Fn(&[f32], &str, &[&str]) -> String,
{
    let code = emit_fn(params, fn_name, param_names);
    if verify_problem_code_strict(problem, &code).is_ok() {
        Some(SolveResult {
            success: true,
            code,
            method: "synth_gradient".to_string(),
            error: None,
            metadata: DifferentiableMetadata::default(),
        })
    } else {
        None
    }
}

pub(crate) fn train_program<F, G>(
    initial_params: Vec<f32>,
    loss_fn: F,
    emit_fn: G,
    problem: &Problem,
    param_names: &[&str],
    fn_name: &str,
    n_steps: usize,
) -> Option<SolveResult>
where
    F: Fn(&[f32], f32) -> f32,
    G: Fn(&[f32], &str, &[&str]) -> String,
{
    // Try the initial params directly (may already be correct after good init)
    if let Some(result) = try_emit_verify(&initial_params, &emit_fn, problem, fn_name, param_names)
    {
        return Some(result);
    }

    let mut params = initial_params;
    let n = params.len();
    let mut opt = Adam::new(n, 0.05);
    let mut best_loss = f32::MAX;
    let mut best_params = params.clone();
    // Track the lowest loss seen to trigger discretization checks more often
    let mut last_check_loss = f32::MAX;
    // Early stopping: track loss at two checkpoints to cut runs that aren't converging
    let chk1 = n_steps / 4; // 25% mark
    let chk2 = n_steps / 2; // 50% mark
    let mut loss_at_chk1 = f32::MAX;
    let mut loss_at_chk2 = f32::MAX;

    for step in 0..n_steps {
        // Bail out of a non-converging run once the caller's wall-clock budget is
        // spent (checked sparsely to keep `Instant::now()` off the hot path).
        // No-op when no `TrainDeadline` is installed.
        if step % 16 == 0 && train_deadline_exceeded() {
            break;
        }
        if step == chk1 {
            loss_at_chk1 = best_loss;
        }
        if step == chk2 {
            loss_at_chk2 = best_loss;
        }
        // At 50%: if loss improved <2% since 25% mark (truly stagnant), abort
        if step == chk2 && best_loss > loss_at_chk1 * 0.98 {
            break;
        }
        // At 75%: if loss improved <10% since 50% mark, abort
        if step > n_steps * 3 / 4 && best_loss > loss_at_chk2 * 0.90 {
            break;
        }

        let temp = (2.0f32 * (1.0 - step as f32 / n_steps as f32)).max(0.1);
        let loss = loss_fn(&params, temp);
        if loss < best_loss {
            best_loss = loss;
            best_params = params.clone();
        }
        // Try discretizing whenever we improve by >10% or loss is low
        let should_check = loss < 1.0 || (loss < last_check_loss * 0.9) || (step % 50 == 49); // also check every 50 steps
        if should_check {
            last_check_loss = loss.min(last_check_loss);
            if let Some(result) = try_emit_verify(&params, &emit_fn, problem, fn_name, param_names)
            {
                return Some(result);
            }
            // Also try best_params
            if best_loss < loss {
                if let Some(result) =
                    try_emit_verify(&best_params, &emit_fn, problem, fn_name, param_names)
                {
                    return Some(result);
                }
            }
        }
        let grads = fd_grad(&params, &loss_fn, temp);
        opt.step(&mut params, &grads);
    }

    // Final attempts with current and best params
    if let Some(result) = try_emit_verify(&params, &emit_fn, problem, fn_name, param_names) {
        return Some(result);
    }
    try_emit_verify(&best_params, &emit_fn, problem, fn_name, param_names)
}

// ─── Simple LCG pseudo-random ─────────────────────────────────────────────────

pub(crate) fn pseudo_rand(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((x >> 33) as f32) / (u32::MAX as f32)
}
