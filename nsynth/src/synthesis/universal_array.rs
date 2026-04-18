use super::*;
use std::fmt::Write as _;

const N_ARR_PRE: usize = 1;
const N_ARR_BODY: usize = 4;
const N_ARR_POST: usize = 1;
const N_ARR_SLOTS: usize = N_ARR_PRE + N_ARR_BODY + N_ARR_POST;
const N_ARR_FIXED: usize = 8;

/// Fallback constant pool when no problem-specific constants can be mined.
/// Keep {0, 1, -1} as anchors because essentially every nontrivial program
/// uses them; {2, -2, 10} are common magnitudes that cover loops/arithmetic.
const DEFAULT_CONSTS: [i64; N_CONSTS] = [0, 1, -1, 2, -2, 10];

/// Scan a problem's examples for integer values that would be useful as
/// constants in the emitted program. Returns `N_CONSTS` values, always
/// including the anchor set {0, 1, -1}, with remaining slots filled by the
/// most-frequently-appearing integers drawn from example inputs and
/// outputs. Falls back to [`DEFAULT_CONSTS`] when the problem doesn't
/// yield enough unique values.
///
/// This is the minimal "emergent vocabulary" move: the gradient brain's
/// constant pool stops being a hand-picked global and starts being a
/// per-problem prior mined from the problem itself.
pub(super) fn discover_useful_consts(examples: &[ArrExample]) -> [i64; N_CONSTS] {
    use std::collections::HashMap;
    let mut freq: HashMap<i64, usize> = HashMap::new();
    let mut bump = |v: i64| {
        *freq.entry(v).or_insert(0) += 1;
    };
    for ex in examples {
        bump(ex.arr_len as i64);
        bump(ex.expected as i64);
        for &v in &ex.arr {
            bump(v as i64);
        }
        for &s in &ex.scalar_args {
            bump(s as i64);
        }
        // Also harvest pairwise differences — they often reveal the
        // answer's "scale" for the problem (e.g. arr.len - 1 as a loop
        // bound, expected - arr.sum as a branch threshold).
        if ex.arr.len() >= 2 {
            for w in ex.arr.windows(2) {
                bump((w[1] - w[0]) as i64);
            }
        }
    }
    // Anchor values that should always be available regardless of the data.
    let anchors = [0i64, 1, -1];
    let mut chosen: Vec<i64> = anchors.to_vec();
    // Sort the mined values by (frequency desc, |value| asc) to prefer
    // recurring small constants. Anchors already in `chosen` are skipped.
    let mut mined: Vec<(i64, usize)> = freq.into_iter().collect();
    mined.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.abs().cmp(&b.0.abs())));
    for (v, _) in mined {
        if chosen.len() >= N_CONSTS {
            break;
        }
        if !chosen.contains(&v) {
            chosen.push(v);
        }
    }
    // Top up from DEFAULT_CONSTS if we still don't have enough unique vals.
    for &fallback in DEFAULT_CONSTS.iter() {
        if chosen.len() >= N_CONSTS {
            break;
        }
        if !chosen.contains(&fallback) {
            chosen.push(fallback);
        }
    }
    // Shouldn't need this, but be defensive.
    while chosen.len() < N_CONSTS {
        chosen.push(0);
    }
    let mut out = [0i64; N_CONSTS];
    for (i, v) in chosen.into_iter().take(N_CONSTS).enumerate() {
        out[i] = v;
    }
    out
}

#[inline]
fn uarr_pool(n_scalar: usize) -> usize {
    N_ARR_FIXED + N_CONSTS + n_scalar + N_ARR_SLOTS
}

#[inline]
fn uarr_sps(pool: usize) -> usize {
    (N_OPS + 1) + 5 * pool + N_CMPS
}

#[inline]
fn uarr_lip(n_scalar: usize) -> usize {
    1 + N_CONSTS + n_scalar
}

struct SoftUniversalArrayProgram {
    n_scalar: usize,
    params: Vec<f32>,
}

impl SoftUniversalArrayProgram {
    fn pool(&self) -> usize {
        uarr_pool(self.n_scalar)
    }

    fn sps(&self) -> usize {
        uarr_sps(self.pool())
    }

    fn lip(&self) -> usize {
        uarr_lip(self.n_scalar)
    }

    fn slot_off(&self, slot: usize) -> usize {
        slot * self.sps()
    }

    fn body_init_off(&self, bs: usize) -> usize {
        N_ARR_SLOTS * self.sps() + bs * self.lip()
    }

    fn return_off(&self) -> usize {
        self.body_init_off(N_ARR_BODY)
    }

    fn consts_off(&self) -> usize {
        self.return_off() + self.pool()
    }

    fn n_params_for(n_scalar: usize) -> usize {
        let pool = uarr_pool(n_scalar);
        let lip = uarr_lip(n_scalar);
        N_ARR_SLOTS * uarr_sps(pool) + N_ARR_BODY * lip + pool + N_CONSTS
    }

    #[allow(dead_code)]
    fn new(n_scalar: usize) -> Self {
        Self::new_with_consts(n_scalar, &DEFAULT_CONSTS)
    }

    /// Like `new()` but seeds the constant pool with `consts` instead of the
    /// hand-picked `DEFAULT_CONSTS = [0, 1, -1, 2, -2, 10]`. Used by the
    /// emergent-constant path so the gradient brain starts with constants
    /// discovered from the problem's own examples.
    fn new_with_consts(n_scalar: usize, consts: &[i64; N_CONSTS]) -> Self {
        let mut s = Self {
            n_scalar,
            params: vec![0f32; Self::n_params_for(n_scalar)],
        };
        let co = s.consts_off();
        for (i, &c) in consts.iter().enumerate() {
            s.params[co + i] = c as f32;
        }

        let pool = uarr_pool(n_scalar);
        let pre_start = N_ARR_FIXED + N_CONSTS + n_scalar;
        let body_start = pre_start + N_ARR_PRE;
        let c0_pool = N_ARR_FIXED;
        for slot in 0..N_ARR_SLOTS {
            let off = slot * uarr_sps(pool);
            let cb = off + N_OPS + 1 + 2 * pool;
            s.params[off + 5] = 1.0;
            let ref_idx = if slot < N_ARR_PRE {
                c0_pool
            } else if slot < N_ARR_PRE + N_ARR_BODY {
                body_start + (slot - N_ARR_PRE)
            } else {
                body_start
            };
            s.params[off + N_OPS + 1 + ref_idx] = 1.0;
            s.params[off + N_OPS + 1 + pool + ref_idx] = 1.0;
            s.params[cb + 1] = 1.0;
            s.params[cb + N_CMPS + 1] = 1.0;
            s.params[cb + N_CMPS + pool + 1] = 1.0;
            s.params[cb + N_CMPS + 2 * pool + ref_idx] = 1.0;
        }
        for bs in 0..N_ARR_BODY {
            let io = s.body_init_off(bs);
            s.params[io + 1] = 2.0;
        }
        let ro = s.return_off();
        s.params[ro + body_start] = 2.0;

        s
    }

    fn pool_names(_n_scalar: usize, consts: &[i64], scalar_names: &[&str]) -> Vec<String> {
        let mut pn = Vec::new();
        pn.push("item".to_string());
        pn.push("i".to_string());
        pn.push("parity".to_string());
        pn.push("arr.len".to_string());
        pn.push("item_even".to_string());
        pn.push("prev".to_string());
        pn.push("next".to_string());
        pn.push("mirror".to_string());
        for v in consts {
            pn.push(format!("{v}"));
        }
        for s in scalar_names {
            pn.push(s.to_string());
        }
        for i in 0..N_ARR_PRE {
            pn.push(format!("v{i}"));
        }
        for i in 0..N_ARR_BODY {
            pn.push(format!("s{i}"));
        }
        for i in 0..N_ARR_POST {
            pn.push(format!("p{i}"));
        }
        pn
    }

    fn pre_reg_start(&self) -> usize {
        N_ARR_FIXED + N_CONSTS + self.n_scalar
    }

    fn body_reg_start(&self) -> usize {
        self.pre_reg_start() + N_ARR_PRE
    }

    fn post_reg_start(&self) -> usize {
        self.body_reg_start() + N_ARR_BODY
    }

    fn exec_slot_backward(
        &self,
        slot: usize,
        r: &[f32],
        d_out: f32,
        temp: f32,
        d_params: &mut [f32],
    ) -> Vec<f32> {
        let pool = self.pool();
        let off = self.slot_off(slot);
        let cb = off + N_OPS + 1 + 2 * pool;

        let op_w = softmax_temp(&self.params[off..off + N_OPS + 1], temp);
        let s1_w = softmax_temp(&self.params[off + N_OPS + 1..off + N_OPS + 1 + pool], temp);
        let s2_w = softmax_temp(
            &self.params[off + N_OPS + 1 + pool..off + N_OPS + 1 + 2 * pool],
            temp,
        );
        let src1 = soft_read(r, &s1_w);
        let src2 = soft_read(r, &s2_w);
        let then_val = soft_op_ext(src1, src2, &op_w);
        let cmp_w = softmax_temp(&self.params[cb..cb + N_CMPS], temp);
        let gl_w = softmax_temp(&self.params[cb + N_CMPS..cb + N_CMPS + pool], temp);
        let gr_w = softmax_temp(
            &self.params[cb + N_CMPS + pool..cb + N_CMPS + 2 * pool],
            temp,
        );
        let gate_lhs = soft_read(r, &gl_w);
        let gate_rhs = soft_read(r, &gr_w);
        let gate = soft_cmp(gate_lhs, gate_rhs, &cmp_w, temp);
        let el_w = softmax_temp(
            &self.params[cb + N_CMPS + 2 * pool..cb + N_CMPS + 3 * pool],
            temp,
        );
        let else_val = soft_read(r, &el_w);

        let d_gate = d_out * (then_val - else_val);
        let d_then = d_out * gate;
        let d_else = d_out * (1.0 - gate);
        let mut d_reg = vec![0.0f32; pool];

        let d_el_w: Vec<f32> = r.iter().map(|&ri| d_else * ri).collect();
        let d_el_logits = softmax_temp_backward(&el_w, &d_el_w, temp);
        for (j, &g) in d_el_logits.iter().enumerate() {
            d_params[cb + N_CMPS + 2 * pool + j] += g;
        }
        for (j, &w) in el_w.iter().enumerate() {
            d_reg[j] += d_else * w;
        }

        let cmp_grad = SoftCmpGrad::forward(
            gate_lhs,
            gate_rhs,
            &self.params[cb..cb + N_CMPS],
            temp,
            temp,
        );
        let (d_cmp_logits, d_gl, d_gr) = cmp_grad.backward(d_gate, temp);
        for (j, &g) in d_cmp_logits.iter().enumerate() {
            d_params[cb + j] += g;
        }
        let d_gl_w: Vec<f32> = r.iter().map(|&ri| d_gl * ri).collect();
        for (j, &g) in softmax_temp_backward(&gl_w, &d_gl_w, temp)
            .iter()
            .enumerate()
        {
            d_params[cb + N_CMPS + j] += g;
        }
        for (j, &w) in gl_w.iter().enumerate() {
            d_reg[j] += d_gl * w;
        }
        let d_gr_w: Vec<f32> = r.iter().map(|&ri| d_gr * ri).collect();
        for (j, &g) in softmax_temp_backward(&gr_w, &d_gr_w, temp)
            .iter()
            .enumerate()
        {
            d_params[cb + N_CMPS + pool + j] += g;
        }
        for (j, &w) in gr_w.iter().enumerate() {
            d_reg[j] += d_gr * w;
        }

        let op_grad = SoftOpExtGrad::forward(src1, src2, &self.params[off..off + N_OPS + 1], temp);
        let (d_op_logits, d_src1, d_src2) = op_grad.backward(d_then, temp);
        for (j, &g) in d_op_logits.iter().enumerate() {
            d_params[off + j] += g;
        }
        let d_s1_w: Vec<f32> = r.iter().map(|&ri| d_src1 * ri).collect();
        for (j, &g) in softmax_temp_backward(&s1_w, &d_s1_w, temp)
            .iter()
            .enumerate()
        {
            d_params[off + N_OPS + 1 + j] += g;
        }
        for (j, &w) in s1_w.iter().enumerate() {
            d_reg[j] += d_src1 * w;
        }
        let d_s2_w: Vec<f32> = r.iter().map(|&ri| d_src2 * ri).collect();
        for (j, &g) in softmax_temp_backward(&s2_w, &d_s2_w, temp)
            .iter()
            .enumerate()
        {
            d_params[off + N_OPS + 1 + pool + j] += g;
        }
        for (j, &w) in s2_w.iter().enumerate() {
            d_reg[j] += d_src2 * w;
        }

        d_reg
    }

    fn exec_slot(&self, slot: usize, r: &[f32], temp: f32) -> f32 {
        let pool = self.pool();
        let off = self.slot_off(slot);
        let op_w = softmax_temp(&self.params[off..off + N_OPS + 1], temp);
        let s1_w = softmax_temp(&self.params[off + N_OPS + 1..off + N_OPS + 1 + pool], temp);
        let s2_w = softmax_temp(
            &self.params[off + N_OPS + 1 + pool..off + N_OPS + 1 + 2 * pool],
            temp,
        );
        let src1 = soft_read(r, &s1_w);
        let src2 = soft_read(r, &s2_w);
        let then_val = soft_op_ext(src1, src2, &op_w);
        let cb = off + N_OPS + 1 + 2 * pool;
        let cmp_w = softmax_temp(&self.params[cb..cb + N_CMPS], temp);
        let gl_w = softmax_temp(&self.params[cb + N_CMPS..cb + N_CMPS + pool], temp);
        let gr_w = softmax_temp(
            &self.params[cb + N_CMPS + pool..cb + N_CMPS + 2 * pool],
            temp,
        );
        let gate_lhs = soft_read(r, &gl_w);
        let gate_rhs = soft_read(r, &gr_w);
        let gate = soft_cmp(gate_lhs, gate_rhs, &cmp_w, temp);
        let el_w = softmax_temp(
            &self.params[cb + N_CMPS + 2 * pool..cb + N_CMPS + 3 * pool],
            temp,
        );
        let else_val = soft_read(r, &el_w);
        gate * then_val + (1.0 - gate) * else_val
    }

    #[allow(dead_code)]
    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let pool = self.pool();
        let lip = self.lip();
        let co = self.consts_off();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        let mut reg = vec![0f32; pool];
        reg[0] = arr[0];
        reg[1] = 0.0;
        reg[2] = 1.0;
        reg[3] = arr_len;
        reg[4] = (std::f32::consts::PI * arr[0]).cos();
        reg[5] = arr[0];
        reg[6] = arr[0];
        reg[7] = arr[0];
        for j in 0..N_CONSTS {
            reg[N_ARR_FIXED + j] = consts[j];
        }
        for j in 0..self.n_scalar {
            reg[N_ARR_FIXED + N_CONSTS + j] = scalar_args[j];
        }

        for slot in 0..N_ARR_PRE {
            reg[self.pre_reg_start() + slot] = self.exec_slot(slot, &reg, temp);
        }

        for bs in 0..N_ARR_BODY {
            let io = self.body_init_off(bs);
            let w = softmax_temp(&self.params[io..io + lip], temp);
            let mut init_pool = vec![0f32; lip];
            init_pool[0] = arr[0];
            for j in 0..N_CONSTS {
                init_pool[1 + j] = consts[j];
            }
            for j in 0..self.n_scalar {
                init_pool[1 + N_CONSTS + j] = scalar_args[j];
            }
            reg[self.body_reg_start() + bs] = soft_read(&init_pool, &w);
        }

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            if in_bounds < 1e-6 {
                break;
            }

            let len_i = arr_len.round() as usize;
            let prev_item = if i > 0 { arr[i - 1] } else { arr[0] };
            let next_item = if i + 1 < arr_len.round() as usize {
                arr[i + 1]
            } else {
                arr[i]
            };
            let mirror_item = if i < len_i {
                arr[len_i - 1 - i]
            } else {
                arr[0]
            };
            reg[0] = arr[i];
            reg[1] = i as f32;
            reg[2] = (std::f32::consts::PI * i as f32).cos();
            reg[4] = (std::f32::consts::PI * arr[i]).cos();
            reg[5] = prev_item;
            reg[6] = next_item;
            reg[7] = mirror_item;

            for bs in 0..N_ARR_BODY {
                let slot = N_ARR_PRE + bs;
                let out = self.exec_slot(slot, &reg, temp);
                let idx = self.body_reg_start() + bs;
                reg[idx] = in_bounds * out + (1.0 - in_bounds) * reg[idx];
            }
        }

        reg[0] = 0.0;
        reg[1] = 0.0;
        reg[2] = 0.0;
        reg[5] = 0.0;
        reg[6] = 0.0;
        reg[7] = 0.0;
        for ps in 0..N_ARR_POST {
            let slot = N_ARR_PRE + N_ARR_BODY + ps;
            reg[self.post_reg_start() + ps] = self.exec_slot(slot, &reg, temp);
        }

        let ro = self.return_off();
        let rw = softmax_temp(&self.params[ro..ro + pool], temp);
        soft_read(&reg, &rw)
    }

    // `loss(...)` and `grad(...)` were removed in favor of the fused
    // `grad_and_loss(...)` below — the old stand-alone `loss` cost a full
    // forward pass per step on top of the one `grad` already does. If a future
    // caller needs loss only, compute it via
    // `let (_, l) = prog.grad_and_loss(&[ex], 1.0);`.

    /// Returns both the parameter gradients and the MSE loss in a single
    /// forward+backward pass. Callers in the restart/step inner loop should
    /// prefer this over separate `grad` + `loss` calls — the `loss` computation
    /// would otherwise redo the entire forward pass for every step.
    fn grad_and_loss(&self, examples: &[ArrExample], temp: f32) -> (Vec<f32>, f32) {
        let n_scalar = self.n_scalar;
        let params = &self.params;
        let n_ex = examples.len() as f32;
        let n_params = params.len();
        let pool_sz = self.pool();
        let lip_sz = self.lip();
        let ro = self.return_off();
        let co = self.consts_off();
        let mut grad = vec![0.0f32; n_params];
        let mut loss_sum: f32 = 0.0;
        let _ = n_scalar;

        for ex in examples {
            let consts: Vec<f32> = (0..N_CONSTS).map(|i| params[co + i]).collect();

            let mut reg = vec![0f32; pool_sz];
            reg[0] = ex.arr[0];
            reg[1] = 0.0;
            reg[2] = 1.0;
            reg[3] = ex.arr_len;
            reg[4] = (std::f32::consts::PI * ex.arr[0]).cos();
            reg[5] = ex.arr[0];
            reg[6] = ex.arr[0];
            reg[7] = ex.arr[0];
            for j in 0..N_CONSTS {
                reg[N_ARR_FIXED + j] = consts[j];
            }
            for j in 0..n_scalar {
                reg[N_ARR_FIXED + N_CONSTS + j] = ex.scalar_args[j];
            }

            let mut reg_snapshots: Vec<Vec<f32>> = Vec::new();
            for slot in 0..N_ARR_PRE {
                reg_snapshots.push(reg.clone());
                reg[self.pre_reg_start() + slot] = self.exec_slot(slot, &reg, temp);
            }

            for bs in 0..N_ARR_BODY {
                let io = self.body_init_off(bs);
                let w = softmax_temp(&params[io..io + lip_sz], temp);
                let mut init_pool = vec![0f32; lip_sz];
                init_pool[0] = ex.arr[0];
                for j in 0..N_CONSTS {
                    init_pool[1 + j] = consts[j];
                }
                for j in 0..n_scalar {
                    init_pool[1 + N_CONSTS + j] = ex.scalar_args[j];
                }
                reg[self.body_reg_start() + bs] = soft_read(&init_pool, &w);
            }

            let mut iter_states: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut iter_bounds: Vec<f32> = Vec::new();
            let n_iters;
            {
                let mut count = 0usize;
                for i in 0..MAX_ARR {
                    let in_bounds = sigmoid((ex.arr_len - i as f32 - 0.5) / 0.3);
                    if in_bounds < 1e-6 {
                        break;
                    }
                    let len_i = ex.arr_len.round() as usize;
                    let prev_item = if i > 0 { ex.arr[i - 1] } else { ex.arr[0] };
                    let next_item = if i + 1 < ex.arr_len.round() as usize {
                        ex.arr[i + 1]
                    } else {
                        ex.arr[i]
                    };
                    let mirror_item = if i < len_i {
                        ex.arr[len_i - 1 - i]
                    } else {
                        ex.arr[0]
                    };
                    reg[0] = ex.arr[i];
                    reg[1] = i as f32;
                    reg[2] = (std::f32::consts::PI * i as f32).cos();
                    reg[4] = (std::f32::consts::PI * ex.arr[i]).cos();
                    reg[5] = prev_item;
                    reg[6] = next_item;
                    reg[7] = mirror_item;
                    let mut slot_snaps = Vec::new();
                    for bs in 0..N_ARR_BODY {
                        slot_snaps.push(reg.clone());
                        let slot = N_ARR_PRE + bs;
                        let out = self.exec_slot(slot, &reg, temp);
                        let idx = self.body_reg_start() + bs;
                        reg[idx] = in_bounds * out + (1.0 - in_bounds) * reg[idx];
                    }
                    iter_states.push(slot_snaps);
                    iter_bounds.push(in_bounds);
                    count += 1;
                }
                n_iters = count;
            }

            reg[0] = 0.0;
            reg[1] = 0.0;
            reg[2] = 0.0;
            reg[4] = 1.0;
            reg[5] = 0.0;
            reg[6] = 0.0;
            reg[7] = 0.0;
            let mut post_snaps = Vec::new();
            for ps in 0..N_ARR_POST {
                post_snaps.push(reg.clone());
                let slot = N_ARR_PRE + N_ARR_BODY + ps;
                reg[self.post_reg_start() + ps] = self.exec_slot(slot, &reg, temp);
            }

            let rw = softmax_temp(&params[ro..ro + pool_sz], temp);
            let output = soft_read(&reg, &rw);
            let diff = output - ex.expected;
            loss_sum += diff * diff;
            let d_output = 2.0 * diff / n_ex;

            let mut d_reg = vec![0.0f32; pool_sz];

            let d_rw: Vec<f32> = reg.iter().map(|&r| d_output * r).collect();
            let d_rw_logits = softmax_temp_backward(&rw, &d_rw, temp);
            for (j, &g) in d_rw_logits.iter().enumerate() {
                grad[ro + j] += g;
            }
            for (j, &w) in rw.iter().enumerate() {
                d_reg[j] += d_output * w;
            }

            for ps in (0..N_ARR_POST).rev() {
                let slot = N_ARR_PRE + N_ARR_BODY + ps;
                let reg_idx = self.post_reg_start() + ps;
                let d_slot_out = d_reg[reg_idx];
                d_reg[reg_idx] = 0.0;
                let d_r =
                    self.exec_slot_backward(slot, &post_snaps[ps], d_slot_out, temp, &mut grad);
                for (j, &dr) in d_r.iter().enumerate() {
                    d_reg[j] += dr;
                }
            }

            for iter_i in (0..n_iters).rev() {
                let ib = iter_bounds[iter_i];
                for bs in (0..N_ARR_BODY).rev() {
                    let slot = N_ARR_PRE + bs;
                    let reg_idx = self.body_reg_start() + bs;
                    let d_reg_val = d_reg[reg_idx];
                    let d_out = d_reg_val * ib;
                    d_reg[reg_idx] = d_reg_val * (1.0 - ib);
                    let d_r = self.exec_slot_backward(
                        slot,
                        &iter_states[iter_i][bs],
                        d_out,
                        temp,
                        &mut grad,
                    );
                    for (j, &dr) in d_r.iter().enumerate() {
                        if j != reg_idx {
                            d_reg[j] += dr;
                        }
                    }
                }
            }

            for bs in 0..N_ARR_BODY {
                let io = self.body_init_off(bs);
                let reg_idx = self.body_reg_start() + bs;
                let d_init = d_reg[reg_idx];
                let w = softmax_temp(&params[io..io + lip_sz], temp);
                let mut init_pool = vec![0f32; lip_sz];
                init_pool[0] = ex.arr[0];
                for j in 0..N_CONSTS {
                    init_pool[1 + j] = consts[j];
                }
                for j in 0..n_scalar {
                    init_pool[1 + N_CONSTS + j] = ex.scalar_args[j];
                }
                let d_w: Vec<f32> = init_pool.iter().map(|&v| d_init * v).collect();
                let d_logits = softmax_temp_backward(&w, &d_w, temp);
                for (j, &g) in d_logits.iter().enumerate() {
                    grad[io + j] += g;
                }
                for j in 0..N_CONSTS {
                    grad[co + j] += d_init * w[1 + j];
                }
            }

            for slot in (0..N_ARR_PRE).rev() {
                let reg_idx = self.pre_reg_start() + slot;
                let d_slot_out = d_reg[reg_idx];
                let d_r = self.exec_slot_backward(
                    slot,
                    &reg_snapshots[slot],
                    d_slot_out,
                    temp,
                    &mut grad,
                );
                for (j, &dr) in d_r.iter().enumerate() {
                    d_reg[j] += dr;
                }
            }

            for j in 0..N_CONSTS {
                grad[co + j] += d_reg[N_ARR_FIXED + j];
            }
        }
        (grad, loss_sum / n_ex)
    }

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let pool = self.pool();
        let lip = self.lip();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|i| self.params[co + i].round() as i64)
            .collect();

        let pn = Self::pool_names(self.n_scalar, &consts, scalar_names);
        let op_names = ["+", "-", "*", "/", "%"];
        let cmp_names = ["<", "<=", "==", ">=", ">", "!="];

        let mut lpn: Vec<String> = vec!["arr[0]".to_string()];
        for v in &consts {
            lpn.push(format!("{v}"));
        }
        for s in scalar_names {
            lpn.push(s.to_string());
        }

        let slot_line = |slot: usize, dest_name: &str, decl: bool| -> String {
            let off = self.slot_off(slot);
            let op_i = argmax(&self.params[off..off + N_OPS + 1]);
            let s1_i = argmax(&self.params[off + N_OPS + 1..off + N_OPS + 1 + pool]);
            let s2_i = argmax(&self.params[off + N_OPS + 1 + pool..off + N_OPS + 1 + 2 * pool]);
            let cb = off + N_OPS + 1 + 2 * pool;
            let cmp_i = argmax(&self.params[cb..cb + N_CMPS]);
            let gl_i = argmax(&self.params[cb + N_CMPS..cb + N_CMPS + pool]);
            let gr_i = argmax(&self.params[cb + N_CMPS + pool..cb + N_CMPS + 2 * pool]);
            let el_i = argmax(&self.params[cb + N_CMPS + 2 * pool..cb + N_CMPS + 3 * pool]);

            let s1 = &pn[s1_i];
            let s2 = &pn[s2_i];
            let then_expr = if op_i >= N_OPS {
                s1.clone()
            } else {
                format!("{s1} {} {s2}", op_names[op_i])
            };
            let else_expr = pn[el_i].clone();
            let gl = &pn[gl_i];
            let gr = &pn[gr_i];
            let cmp_s = cmp_names[cmp_i.min(5)];

            let trivially_true = gl_i == gr_i && matches!(cmp_i, 1 | 2 | 3);
            let trivially_false = gl_i == gr_i && matches!(cmp_i, 0 | 4 | 5);
            let no_diff = then_expr == else_expr;

            let prefix = if decl {
                format!("{dest_name}: i64")
            } else {
                dest_name.to_string()
            };
            if trivially_true || no_diff {
                format!("    {prefix} = {then_expr};")
            } else if trivially_false {
                format!("    {prefix} = {else_expr};")
            } else if decl {
                format!(
                    "    {dest_name}: i64 = 0;\n    if {gl} {cmp_s} {gr} {{ {dest_name} = {then_expr}; }} else {{ {dest_name} = {else_expr}; }}"
                )
            } else {
                format!(
                    "    if {gl} {cmp_s} {gr} {{ {dest_name} = {then_expr}; }} else {{ {dest_name} = {else_expr}; }}"
                )
            }
        };

        let scalar_params = scalar_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        let sig = if scalar_params.is_empty() {
            format!("fn {fn_name}(arr: [i64]) -> i64")
        } else {
            format!("fn {fn_name}(arr: [i64], {scalar_params}) -> i64")
        };
        let mut out = format!("{sig} {{\n");

        for i in 0..N_ARR_PRE {
            let line = slot_line(i, &format!("v{i}"), true);
            out.push_str(&line);
            out.push('\n');
        }

        for bs in 0..N_ARR_BODY {
            let io = self.body_init_off(bs);
            let src_i = argmax(&self.params[io..io + lip]);
            let src = &lpn[src_i];
            if src_i == 0 {
                writeln!(out, "    s{bs} := {src};").unwrap();
            } else {
                writeln!(out, "    s{bs}: i64 = {src};").unwrap();
            }
        }

        out.push_str("    i: i64 = 0;\n");
        out.push_str("    while i < arr.len {\n");
        out.push_str("        item: i64 = arr[i];\n");

        let mut uses_parity = false;
        let mut uses_item_even = false;
        let mut uses_prev = false;
        let mut uses_next = false;
        let mut uses_mirror = false;
        for bs in 0..N_ARR_BODY {
            let off = self.slot_off(N_ARR_PRE + bs);
            let cb = off + N_OPS + 1 + 2 * pool;
            for w_off in [
                off + N_OPS + 1,
                off + N_OPS + 1 + pool,
                cb + N_CMPS,
                cb + N_CMPS + pool,
                cb + N_CMPS + 2 * pool,
            ] {
                let idx = argmax(&self.params[w_off..w_off + pool]);
                if idx == 2 {
                    uses_parity = true;
                }
                if idx == 4 {
                    uses_item_even = true;
                }
                if idx == 5 {
                    uses_prev = true;
                }
                if idx == 6 {
                    uses_next = true;
                }
                if idx == 7 {
                    uses_mirror = true;
                }
            }
        }
        if uses_parity {
            out.push_str("        parity: i64 = 1 - 2 * (i % 2);\n");
        }
        if uses_item_even {
            out.push_str("        item_even: i64 = 1 - 2 * (item % 2);\n");
        }
        if uses_prev {
            out.push_str(
                "        prev: i64 = 0;\n        if i > 0 { prev = arr[i - 1]; }\n        if i == 0 { prev = arr[0]; }\n",
            );
        }
        if uses_next {
            out.push_str(
                "        next: i64 = item;\n        if i + 1 < arr.len { next = arr[i + 1]; }\n",
            );
        }
        if uses_mirror {
            out.push_str("        mirror: i64 = arr[arr.len - 1 - i];\n");
        }

        for bs in 0..N_ARR_BODY {
            let slot = N_ARR_PRE + bs;
            let line = slot_line(slot, &format!("s{bs}"), false);
            let indented = line
                .replace("\n    ", "\n        ")
                .replace("    s", "        s")
                .replace("    if", "        if");
            out.push_str(&indented);
            out.push('\n');
        }
        out.push_str("        i = i + 1;\n");
        out.push_str("    }\n");

        for pi in 0..N_ARR_POST {
            let slot = N_ARR_PRE + N_ARR_BODY + pi;
            let line = slot_line(slot, &format!("p{pi}"), true);
            out.push_str(&line);
            out.push('\n');
        }

        let ro = self.return_off();
        let ret_i = argmax(&self.params[ro..ro + pool]);
        writeln!(out, "    return {};", pn[ret_i]).unwrap();
        out.push_str("}\n");
        out
    }
}

/// Number of Adam refinement steps to run on each learned bias before
/// giving up. Zero-step replay only catches *exact* repeats; a few warm
/// steps let a bias from one problem shift to a similar one (cross-problem
/// transfer). Kept small because the whole point is to be cheap.
const WARM_REFINE_STEPS: usize = 120;

/// Generate a "random body-slot bias" — pick one program slot at random,
/// emphasize a random op/source/compare/else combination with a strong
/// weight, and zero-lean the return pointer toward a random pool index.
/// No hand-picked indices; purely uniform sampling. Successful random
/// biases get recorded to the learned-bias bank, so over time the system
/// accumulates priors it discovered emergently — not ones we baked in.
///
/// Returns the fully-configured `SoftUniversalArrayProgram`. Caller owns
/// the params vector for subsequent training.
fn random_bias_init(
    n_scalar: usize,
    seed: u64,
    consts: &[i64; N_CONSTS],
) -> SoftUniversalArrayProgram {
    let mut prog = SoftUniversalArrayProgram::new_with_consts(n_scalar, consts);
    let pool = prog.pool();
    // Small helper wrapping `pseudo_rand` into a 0..N integer picker.
    let pick = |k: u64, n: usize| -> usize {
        if n == 0 {
            0
        } else {
            ((pseudo_rand(seed.wrapping_mul(1_103_515_245).wrapping_add(k)) * n as f32) as usize)
                .min(n - 1)
        }
    };

    // Pick 1..=2 body slots to emphasize — single-slot biases cover simple
    // accumulators, two-slot biases cover branched/two-register programs.
    let n_biased_slots = 1 + pick(1, 2); // 1 or 2
    for s in 0..n_biased_slots {
        let bs = pick(10 + s as u64, N_ARR_BODY);
        let op = pick(20 + s as u64, N_OPS + 1);
        let src1 = pick(30 + s as u64, pool);
        let src2 = pick(40 + s as u64, pool);
        let cmp = pick(50 + s as u64, N_CMPS);
        let gl = pick(60 + s as u64, pool);
        let gr = pick(70 + s as u64, pool);
        let el = pick(80 + s as u64, pool);

        let slot = N_ARR_PRE + bs;
        let off = prog.slot_off(slot);
        prog.params[off + op] = 4.0;
        prog.params[off + N_OPS + 1 + src1] = 4.0;
        prog.params[off + N_OPS + 1 + pool + src2] = 4.0;
        let cb = off + N_OPS + 1 + 2 * pool;
        prog.params[cb + cmp] = 4.0;
        prog.params[cb + N_CMPS + gl] = 4.0;
        prog.params[cb + N_CMPS + pool + gr] = 4.0;
        prog.params[cb + N_CMPS + 2 * pool + el] = 4.0;
    }

    // Random body-init weight — pick a random pool source for each body slot.
    for bs in 0..N_ARR_BODY {
        let bio = prog.body_init_off(bs);
        let lip_sz = prog.lip();
        let w_idx = pick(100 + bs as u64, lip_sz);
        prog.params[bio + w_idx] = 4.0;
    }

    // Random return pointer.
    let ro = prog.return_off();
    let ret_idx = pick(200, pool);
    prog.params[ro + ret_idx] = 4.0;

    prog
}

/// How many *additional* pure-random-bias restarts to run after the
/// hand-coded ones. These discover priors the seed library doesn't cover;
/// successful ones get banked. Set via env var `NSYNTH_RANDOM_RESTARTS`
/// (default: 4, 0 disables). Setting it high makes the solver mostly
/// emergent; setting it to 0 restores pure hand-coded behavior.
fn random_restart_count() -> usize {
    std::env::var("NSYNTH_RANDOM_RESTARTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

/// Run the learned-bias replay step in isolation. Callers above
/// `synthesize_universal_array_fallback` can invoke this to short-circuit
/// the array-gradient path in milliseconds on exact repeats, or in a
/// couple of seconds via warm-refine on near-matches — saving the ~15s
/// native_array restart loop that would otherwise run first.
///
/// For each learned bias we:
/// 1. Discretize+verify verbatim (zero-step). Instant hit on exact repeats.
/// 2. If that misses, run up to [`WARM_REFINE_STEPS`] Adam steps from the
///    bias as initial params, periodically discretizing+verifying. This is
///    the cross-problem transfer path: a bias that solved `longest_plateau`
///    can drift a few steps and land on `max_abs`.
///
/// Any success — zero-step or warm-refined — promotes the bias's success
/// counter via the feedback loop.
pub(super) fn try_universal_array_replay(
    problem: &Problem,
    examples: &[ArrExample],
    n_scalar: usize,
    fn_name: &str,
    scalar_names: &[&str],
) -> Option<SolveResult> {
    let expected_n_params = SoftUniversalArrayProgram::n_params_for(n_scalar);
    let learned =
        crate::learned_biases::recent_biases(n_scalar, crate::learned_biases::REPLAY_WINDOW);
    for bias in learned {
        if bias.params.len() != expected_n_params {
            continue;
        }
        // Phase 1: zero-step verbatim replay.
        let prog = SoftUniversalArrayProgram {
            n_scalar,
            params: bias.params.clone(),
        };
        let code = prog.discretize_and_emit(fn_name, scalar_names);
        if verify_problem_code_strict(problem, &code).is_ok() {
            eprintln!(
                "[univ_arr_gradient] replay HIT (exact): {} (bank size {})",
                bias.origin,
                crate::learned_biases::len()
            );
            crate::learned_biases::note_replay_hit(&bias.origin);
            return Some(SolveResult {
                success: true,
                code,
                method: "univ_arr_gradient".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }

        // Phase 2: warm-refine — let Adam move the bias toward this
        // problem's examples. Cheap because WARM_REFINE_STEPS << full
        // N_UNIV_ARR_STEPS=1000. If none of the periodic discretize+verify
        // checks fire, we just drop the bias and move to the next.
        if let Some(result) = warm_refine_from_bias(
            &bias.params,
            problem,
            examples,
            n_scalar,
            fn_name,
            scalar_names,
        ) {
            eprintln!(
                "[univ_arr_gradient] replay HIT (warm): {} (bank size {})",
                bias.origin,
                crate::learned_biases::len()
            );
            crate::learned_biases::note_replay_hit(&bias.origin);
            // The *initial* bias that led here — not the refined params —
            // is what we'd want to retry next time with fewer steps. Record
            // the same bias under a "warm" tag so near-success attempts
            // accumulate their own score history.
            crate::learned_biases::record_success(
                n_scalar,
                bias.params.clone(),
                format!("warm:{}", bias.origin),
            );
            return Some(result);
        }
    }
    None
}

/// Run up to [`WARM_REFINE_STEPS`] of Adam descent from `initial` params,
/// periodically discretizing+verifying. Returns `Some(SolveResult)` on the
/// first verifying program, or `None` if the budget runs out.
fn warm_refine_from_bias(
    initial: &[f32],
    problem: &Problem,
    examples: &[ArrExample],
    n_scalar: usize,
    fn_name: &str,
    scalar_names: &[&str],
) -> Option<SolveResult> {
    let mut params: Vec<f32> = initial.to_vec();
    let mut opt = Adam::new(params.len(), 0.05);
    let mut prog_cur = SoftUniversalArrayProgram {
        n_scalar,
        params: Vec::new(),
    };
    let mut last_code: Option<String> = None;
    for step in 0..WARM_REFINE_STEPS {
        // Anneal from a moderate temperature so Adam can still escape the
        // starting bias's argmax pocket if it no longer fits.
        let temp = (1.5f32 * (1.0 - step as f32 / WARM_REFINE_STEPS as f32)).max(0.15);
        std::mem::swap(&mut prog_cur.params, &mut params);
        let (grads, _loss) = prog_cur.grad_and_loss(examples, temp);
        std::mem::swap(&mut prog_cur.params, &mut params);

        // Discretize+verify every 20 steps AND at step 0 so a close-to-exact
        // bias still gets an early shot. Skip duplicate codes so repeated
        // argmax-stable windows don't pay verify cost twice.
        if step % 20 == 0 {
            std::mem::swap(&mut prog_cur.params, &mut params);
            let code = prog_cur.discretize_and_emit(fn_name, scalar_names);
            std::mem::swap(&mut prog_cur.params, &mut params);
            if last_code.as_ref().map_or(true, |c| c != &code) {
                if verify_problem_code_strict(problem, &code).is_ok() {
                    return Some(SolveResult {
                        success: true,
                        code,
                        method: "univ_arr_gradient".to_string(),
                        error: None,
                        metadata: DifferentiableMetadata::default(),
                    });
                }
                last_code = Some(code);
            }
        }
        opt.step(&mut params, &grads);
    }
    None
}

pub(super) fn synthesize_universal_array_fallback(
    problem: &Problem,
    examples: &[ArrExample],
    n_scalar: usize,
    fn_name: &str,
    scalar_names: &[&str],
) -> Option<SolveResult> {
    const N_UNIV_ARR_STEPS: usize = 1000;
    const N_UNIV_ARR_RESTARTS: usize = 26;

    // Emergent constant vocabulary: mine the problem's own examples for the
    // six most-useful integer constants instead of hand-picking a global
    // pool. Gradient descent starts with problem-appropriate values in
    // `consts_off` and can still fine-tune from there.
    let discovered_consts = discover_useful_consts(examples);

    // Per-problem discretize cache. The inner step loop invokes
    // `discretize_and_emit` ~20 times, and many consecutive discretizations
    // produce identical code strings because the softmax argmax stays stable
    // while params drift slowly. Verification (which parses + runs Mog on
    // every example) dominates that cost, so we memoize by the exact emitted
    // code: if we've already rejected a code string this call, skip the
    // verify. A shared set across all 26 restarts wins even more because
    // close-by restart biases often converge to the same discrete program.
    let mut rejected_codes: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Phase 0: replay the K most-recent biases that *previously* led to a
    // successful solve of any compatible-shape problem. Each learned bias is
    // a full parameter vector; we try a zero-step discretize+verify, which
    // succeeds instantly when the argmax structure of the cached init fits
    // the new problem's examples. This is where the system stops relying
    // purely on hand-coded restart patterns and starts using priors it
    // discovered on its own. Misses fall through to the classical restart
    // loop below — no lost coverage.
    let expected_n_params = SoftUniversalArrayProgram::n_params_for(n_scalar);
    let learned =
        crate::learned_biases::recent_biases(n_scalar, crate::learned_biases::REPLAY_WINDOW);
    for bias in learned {
        if bias.params.len() != expected_n_params {
            continue;
        }
        let prog = SoftUniversalArrayProgram {
            n_scalar,
            params: bias.params.clone(),
        };
        let code = prog.discretize_and_emit(fn_name, scalar_names);
        if rejected_codes.contains(&code) {
            continue;
        }
        if verify_problem_code_strict(problem, &code).is_ok() {
            eprintln!(
                "[univ_arr_gradient] replay HIT: {} (bank size {})",
                bias.origin,
                crate::learned_biases::len()
            );
            crate::learned_biases::note_replay_hit(&bias.origin);
            return Some(SolveResult {
                success: true,
                code,
                method: "univ_arr_gradient".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
        rejected_codes.insert(code);
    }

    let bias_body_slot = |prog: &mut SoftUniversalArrayProgram,
                          bs: usize,
                          op: usize,
                          s1: usize,
                          s2: usize,
                          cmp: usize,
                          gl: usize,
                          gr: usize,
                          el: usize| {
        let pool = prog.pool();
        let slot = N_ARR_PRE + bs;
        let off = prog.slot_off(slot);
        prog.params[off + op] = 4.0;
        prog.params[off + N_OPS + 1 + s1] = 4.0;
        prog.params[off + N_OPS + 1 + pool + s2] = 4.0;
        let cb = off + N_OPS + 1 + 2 * pool;
        prog.params[cb + cmp] = 4.0;
        prog.params[cb + N_CMPS + gl] = 4.0;
        prog.params[cb + N_CMPS + pool + gr] = 4.0;
        prog.params[cb + N_CMPS + 2 * pool + el] = 4.0;
    };

    // Restarts run serially on purpose. The biased inits in restart 0..~5
    // target the most common program shapes (identity accumulator, branched
    // accumulator, two-reg loops) and usually succeed within a few seconds,
    // so the serial cascade short-circuits. An earlier rayon `find_map_any`
    // experiment (April 2026) measured 30-43s on problems that ran in 19-25s
    // serially, because the bias ordering is wasted under parallel launch
    // and per-step Adam state has non-trivial memory-bandwidth contention.
    let n_random_restarts = random_restart_count();
    let total_restarts = N_UNIV_ARR_RESTARTS + n_random_restarts;
    for restart in 0..total_restarts {
        let mut prog;
        let pool;
        let s0_idx;

        if restart >= N_UNIV_ARR_RESTARTS {
            // Emergent phase: pure random bias. No hand-picked slot / op /
            // source indices — uniform sampling over all legal positions.
            // Successful random biases get recorded below, building the
            // emergent prior library that replaces the hand-coded seeds.
            let seed = (restart as u64).wrapping_mul(0x9E3779B97F4A7C15)
                ^ (fn_name.len() as u64).wrapping_mul(0xBF58476D1CE4E5B9);
            prog = random_bias_init(n_scalar, seed, &discovered_consts);
            pool = prog.pool();
            s0_idx = prog.body_reg_start();
        } else {
            prog = SoftUniversalArrayProgram::new_with_consts(n_scalar, &discovered_consts);
            pool = prog.pool();
            s0_idx = prog.body_reg_start();
        }

        if restart == 1 {
            let bio = prog.body_init_off(0);
            prog.params[bio + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 0, s0_idx, 0, 1, 1, 1, s0_idx);
            let ro = prog.return_off();
            prog.params[ro + s0_idx] = 4.0;
        } else if restart == 2 {
            let bio = prog.body_init_off(0);
            prog.params[bio + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 0, s0_idx, 5, 4, 0, 4, s0_idx);
            let ro = prog.return_off();
            prog.params[ro + s0_idx] = 4.0;
        } else if restart == 3 {
            let bio = prog.body_init_off(0);
            prog.params[bio] = 4.0;
            bias_body_slot(&mut prog, 0, 5, 0, 0, 4, 0, s0_idx, s0_idx);
            let ro = prog.return_off();
            prog.params[ro + s0_idx] = 4.0;
        } else if restart == 4 {
            let s1_idx = prog.body_reg_start() + 1;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            prog.params[bio0] = 4.0;
            prog.params[bio1] = 4.0;
            bias_body_slot(&mut prog, 0, 5, 0, 0, 0, 0, s0_idx, s0_idx);
            bias_body_slot(&mut prog, 1, 5, 0, 0, 4, 0, s1_idx, s1_idx);
            let ro = prog.return_off();
            let post_slot = N_ARR_PRE + N_ARR_BODY;
            let p_off = prog.slot_off(post_slot);
            prog.params[p_off + 1] = 4.0;
            prog.params[p_off + N_OPS + 1 + s1_idx] = 4.0;
            prog.params[p_off + N_OPS + 1 + pool + s0_idx] = 4.0;
            let pcb = p_off + N_OPS + 1 + 2 * pool;
            prog.params[pcb + 1] = 4.0;
            prog.params[pcb + N_CMPS + 1] = 4.0;
            prog.params[pcb + N_CMPS + pool + 1] = 4.0;
            prog.params[pcb + N_CMPS + 2 * pool + s1_idx] = 4.0;
            let p0_idx = prog.post_reg_start();
            prog.params[ro + p0_idx] = 4.0;
        } else if restart == 5 {
            let s1_idx = prog.body_reg_start() + 1;
            let bio1 = prog.body_init_off(1);
            prog.params[bio1 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 2, 0, 0, 1, 1, 1, s0_idx);
            bias_body_slot(&mut prog, 1, 0, s1_idx, s0_idx, 1, 1, 1, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 6 && n_scalar >= 1 {
            let k_idx = N_ARR_FIXED + N_CONSTS;
            let bio = prog.body_init_off(0);
            prog.params[bio + 1] = 4.0;
            bias_body_slot(
                &mut prog,
                0,
                0,
                s0_idx,
                N_ARR_FIXED + 1,
                4,
                0,
                k_idx,
                s0_idx,
            );
            let ro = prog.return_off();
            prog.params[ro + s0_idx] = 4.0;
        } else if restart == 7 {
            let s1_idx = prog.body_reg_start() + 1;
            let c0_idx = N_ARR_FIXED;
            bias_body_slot(&mut prog, 0, 1, c0_idx, 0, 1, 1, 1, s0_idx);
            let s2_idx = prog.body_reg_start() + 2;
            let bio2 = prog.body_init_off(2);
            prog.params[bio2 + 1] = 4.0;
            bias_body_slot(&mut prog, 1, 5, 0, 0, 3, 0, c0_idx, s0_idx);
            bias_body_slot(&mut prog, 2, 0, s2_idx, s1_idx, 1, 1, 1, s2_idx);
            let ro = prog.return_off();
            prog.params[ro + s2_idx] = 4.0;
        } else if restart == 8 {
            let c1_idx = N_ARR_FIXED + 1;
            let bio = prog.body_init_off(0);
            prog.params[bio + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 0, s0_idx, c1_idx, 4, 4, N_ARR_FIXED, s0_idx);
            let ro = prog.return_off();
            prog.params[ro + s0_idx] = 4.0;
        } else if restart == 9 {
            let s1_idx = prog.body_reg_start() + 1;
            let bio1 = prog.body_init_off(1);
            prog.params[bio1 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 2, 2, 0, 1, 1, 1, s0_idx);
            bias_body_slot(&mut prog, 1, 0, s1_idx, s0_idx, 1, 1, 1, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 10 {
            let c3_idx = N_ARR_FIXED + 3;
            let s1_idx = prog.body_reg_start() + 1;
            let bio1 = prog.body_init_off(1);
            prog.params[bio1 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 2, 0, c3_idx, 1, 1, 1, s0_idx);
            bias_body_slot(&mut prog, 1, 0, s1_idx, s0_idx, 1, 1, 1, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 11 {
            let s1_idx = prog.body_reg_start() + 1;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            prog.params[bio0] = 4.0;
            prog.params[bio1 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 5, 0, 0, 4, 0, s0_idx, s0_idx);
            bias_body_slot(&mut prog, 1, 0, s1_idx, s0_idx, 1, 1, 1, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 12 {
            let s1_idx = prog.body_reg_start() + 1;
            let s2_idx = prog.body_reg_start() + 2;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            prog.params[bio0] = 4.0;
            prog.params[bio1 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 5, 0, 0, 0, 0, s0_idx, s0_idx);
            bias_body_slot(&mut prog, 1, 1, 0, s0_idx, 1, 1, 1, s2_idx);
            bias_body_slot(&mut prog, 2, 5, s2_idx, s2_idx, 4, s2_idx, s1_idx, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 13 {
            let s1_idx = prog.body_reg_start() + 1;
            let s2_idx = prog.body_reg_start() + 2;
            let c0_idx = N_ARR_FIXED;
            let bio2 = prog.body_init_off(2);
            prog.params[bio2 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 1, c0_idx, 0, 1, 1, 1, s0_idx);
            bias_body_slot(&mut prog, 1, 5, 0, 0, 3, 0, c0_idx, s0_idx);
            bias_body_slot(&mut prog, 2, 5, s1_idx, s1_idx, 4, s1_idx, s2_idx, s2_idx);
            let ro = prog.return_off();
            prog.params[ro + s2_idx] = 4.0;
        } else if restart == 14 && n_scalar >= 1 {
            let k_idx = N_ARR_FIXED + N_CONSTS;
            let bio = prog.body_init_off(0);
            prog.params[bio + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 0, s0_idx, 0, 0, 1, k_idx, s0_idx);
            let ro = prog.return_off();
            prog.params[ro + s0_idx] = 4.0;
        } else if restart == 15 {
            let bio = prog.body_init_off(0);
            prog.params[bio + 2] = 4.0;
            let c0_idx = N_ARR_FIXED;
            bias_body_slot(&mut prog, 0, 5, c0_idx, c0_idx, 0, 0, 5, s0_idx);
            let ro = prog.return_off();
            prog.params[ro + s0_idx] = 4.0;
        } else if restart == 16 {
            let s1_idx = prog.body_reg_start() + 1;
            let c0_idx = N_ARR_FIXED;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            prog.params[bio0 + 1] = 4.0;
            prog.params[bio1] = 4.0;
            bias_body_slot(&mut prog, 0, 0, s0_idx, 0, 4, s0_idx, c0_idx, 0);
            bias_body_slot(&mut prog, 1, 5, s0_idx, s0_idx, 4, s0_idx, s1_idx, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 17 {
            let s0_idx = prog.body_reg_start();
            let s1_idx = prog.body_reg_start() + 1;
            let s2_idx = prog.body_reg_start() + 2;
            let s3_idx = prog.body_reg_start() + 3;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            let bio2 = prog.body_init_off(2);
            let bio3 = prog.body_init_off(3);
            prog.params[bio0] = 4.0;
            prog.params[bio1] = 4.0;
            prog.params[bio2] = 4.0;
            prog.params[bio3] = 4.0;
            bias_body_slot(&mut prog, 0, 5, s2_idx, s2_idx, 1, 1, 1, s0_idx);
            bias_body_slot(&mut prog, 1, 5, 0, 0, 4, 0, s3_idx, s3_idx);
            bias_body_slot(&mut prog, 2, 5, 0, 0, 4, 0, s2_idx, s2_idx);
            bias_body_slot(&mut prog, 3, 5, s0_idx, s0_idx, 4, 0, s0_idx, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s3_idx] = 4.0;
        } else if restart == 18 {
            let s1_idx = prog.body_reg_start() + 1;
            let s2_idx = prog.body_reg_start() + 2;
            let bio0 = prog.body_init_off(0);
            let bio2 = prog.body_init_off(2);
            prog.params[bio0] = 4.0;
            prog.params[bio2 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 5, 0, 0, 0, 0, s0_idx, s0_idx);
            bias_body_slot(&mut prog, 1, 1, 0, s0_idx, 1, 1, 1, s1_idx);
            bias_body_slot(&mut prog, 2, 5, s1_idx, s1_idx, 4, s1_idx, s2_idx, s2_idx);
            let ro = prog.return_off();
            prog.params[ro + s2_idx] = 4.0;
        } else if restart == 19 {
            let co = prog.consts_off();
            prog.params[co + 5] = 1_000_000.0;
            let s0_idx = prog.body_reg_start();
            let s1_idx = prog.body_reg_start() + 1;
            let s2_idx = prog.body_reg_start() + 2;
            let s3_idx = prog.body_reg_start() + 3;
            let c0_idx = N_ARR_FIXED;
            let c_big_idx = N_ARR_FIXED + 5;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            let bio2 = prog.body_init_off(2);
            let bio3 = prog.body_init_off(3);
            prog.params[bio0 + 6] = 4.0;
            prog.params[bio1 + 6] = 4.0;
            prog.params[bio2 + 6] = 4.0;
            prog.params[bio3 + 1] = 4.0;
            bias_body_slot(
                &mut prog, 0, 5, s3_idx, s3_idx, 4, s3_idx, c0_idx, c_big_idx,
            );
            bias_body_slot(&mut prog, 1, 5, 0, 0, 4, 0, c0_idx, c_big_idx);
            bias_body_slot(&mut prog, 2, 5, s1_idx, s1_idx, 0, s1_idx, s0_idx, s0_idx);
            bias_body_slot(
                &mut prog, 3, 5, s2_idx, s2_idx, 0, s2_idx, c_big_idx, c0_idx,
            );
            let ro = prog.return_off();
            prog.params[ro + s3_idx] = 4.0;
        } else if restart == 20 {
            let s1_idx = prog.body_reg_start() + 1;
            let c0_idx = N_ARR_FIXED;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            prog.params[bio0 + 1] = 4.0;
            prog.params[bio1] = 4.0;
            bias_body_slot(&mut prog, 0, 0, s0_idx, 0, 0, s0_idx, c0_idx, 0);
            bias_body_slot(&mut prog, 1, 5, s0_idx, s0_idx, 0, s0_idx, s1_idx, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 21 {
            let s1_idx = prog.body_reg_start() + 1;
            let c1_idx = N_ARR_FIXED + 1;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            prog.params[bio0 + 2] = 4.0;
            prog.params[bio1 + 2] = 4.0;
            bias_body_slot(&mut prog, 0, 0, s0_idx, c1_idx, 4, 0, 5, c1_idx);
            bias_body_slot(&mut prog, 1, 5, s0_idx, s0_idx, 4, s0_idx, s1_idx, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 22 {
            let s1_idx = prog.body_reg_start() + 1;
            let c1_idx = N_ARR_FIXED + 1;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            prog.params[bio0 + 1] = 4.0;
            prog.params[bio1 + 2] = 4.0;
            bias_body_slot(&mut prog, 0, 0, s0_idx, c1_idx, 2, 0, 5, c1_idx);
            bias_body_slot(&mut prog, 1, 5, s0_idx, s0_idx, 4, s0_idx, s1_idx, s1_idx);
            let ro = prog.return_off();
            prog.params[ro + s1_idx] = 4.0;
        } else if restart == 23 {
            let s1_idx = prog.body_reg_start() + 1;
            let s2_idx = prog.body_reg_start() + 2;
            let c0_idx = N_ARR_FIXED;
            let bio2 = prog.body_init_off(2);
            prog.params[bio2 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 1, 0, 5, 1, 1, 1, s0_idx);
            bias_body_slot(&mut prog, 1, 1, c0_idx, s0_idx, 0, s0_idx, c0_idx, s0_idx);
            bias_body_slot(&mut prog, 2, 5, s1_idx, s1_idx, 4, s1_idx, s2_idx, s2_idx);
            let ro = prog.return_off();
            prog.params[ro + s2_idx] = 4.0;
        } else if restart == 24 {
            let s0_idx = prog.body_reg_start();
            let s1_idx = prog.body_reg_start() + 1;
            let s2_idx = prog.body_reg_start() + 2;
            let s3_idx = prog.body_reg_start() + 3;
            let c0_idx = N_ARR_FIXED;
            let c1_idx = N_ARR_FIXED + 1;
            let bio0 = prog.body_init_off(0);
            let bio1 = prog.body_init_off(1);
            let bio2 = prog.body_init_off(2);
            let bio3 = prog.body_init_off(3);
            prog.params[bio0 + 1] = 4.0;
            prog.params[bio1 + 1] = 4.0;
            prog.params[bio2 + 1] = 4.0;
            prog.params[bio3 + 1] = 4.0;
            bias_body_slot(&mut prog, 0, 5, c1_idx, c1_idx, 4, 0, 5, c0_idx);
            bias_body_slot(&mut prog, 1, 5, s0_idx, s0_idx, 4, 0, 6, c0_idx);
            bias_body_slot(&mut prog, 2, 5, s3_idx, s3_idx, 1, 1, 1, s2_idx);
            bias_body_slot(&mut prog, 3, 0, s2_idx, s1_idx, 1, 1, 1, s3_idx);
            let ro = prog.return_off();
            prog.params[ro + s3_idx] = 4.0;
        } else if restart == 25 {
            let c0_idx = N_ARR_FIXED;
            let bio0 = prog.body_init_off(0);
            prog.params[bio0 + 2] = 4.0;
            bias_body_slot(&mut prog, 0, 5, c0_idx, c0_idx, 5, 0, 7, s0_idx);
            let ro = prog.return_off();
            prog.params[ro + s0_idx] = 4.0;
        }

        if restart > 0 {
            let code = SoftUniversalArrayProgram {
                n_scalar,
                params: prog.params.clone(),
            }
            .discretize_and_emit(fn_name, scalar_names);
            let vr = verify_problem_code_strict(problem, &code);
            if cfg!(test)
                && restart <= 24
                && std::env::var_os("MOG_SYNTH_DEBUG_VERIFY").is_some()
                && vr.is_err()
            {
                eprintln!(
                    "[univ r={restart} {fn_name}] VERIFY FAIL: {:?}\nCODE:\n{code}",
                    vr.as_ref().err().unwrap()
                );
            }
            if vr.is_ok() {
                // Pre-step verify succeeded on the clean biased init — that's
                // a high-quality learned pattern (works without any gradient).
                crate::learned_biases::record_success(
                    n_scalar,
                    prog.params.clone(),
                    format!(
                        "{}:{}:prestep",
                        if restart >= N_UNIV_ARR_RESTARTS {
                            "random"
                        } else {
                            "hand"
                        },
                        restart
                    ),
                );
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "univ_arr_gradient".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
            for (idx, p) in prog.params.iter_mut().enumerate() {
                *p += (pseudo_rand(restart as u64 * 37000 + idx as u64) - 0.5) * 0.3;
            }
        }

        let mut params = prog.params;
        // Snapshot the init-plus-noise params that Adam is about to start
        // from. On any success inside the step loop we record this as a
        // learned bias so future solves can replay the same trajectory.
        let initial_params = params.clone();
        let n = params.len();
        let mut opt = Adam::new(n, 0.05);
        let mut best_loss = f32::MAX;
        let mut best_params = params.clone();
        let mut last_check_loss = f32::MAX;
        let chk1 = N_UNIV_ARR_STEPS / 4;
        let chk2 = N_UNIV_ARR_STEPS / 2;
        let mut loss_at_chk1 = f32::MAX;

        // Reusable program shell — params are swapped in/out each step so we
        // skip cloning the whole parameter vector per step (hot path).
        let mut prog_cur = SoftUniversalArrayProgram {
            n_scalar,
            params: Vec::new(),
        };

        for step in 0..N_UNIV_ARR_STEPS {
            if step == chk1 {
                loss_at_chk1 = best_loss;
            }
            if step == chk2 && best_loss > loss_at_chk1 * 0.98 {
                break;
            }

            let temp = (2.0f32 * (1.0 - step as f32 / N_UNIV_ARR_STEPS as f32)).max(0.1);
            // Swap params into the reusable prog (no alloc), compute, swap back.
            std::mem::swap(&mut prog_cur.params, &mut params);
            let (grads, loss) = prog_cur.grad_and_loss(examples, temp);
            std::mem::swap(&mut prog_cur.params, &mut params);

            if loss < best_loss {
                best_loss = loss;
                best_params.clone_from(&params);
            }
            let should_check = loss < 1.0 || (loss < last_check_loss * 0.9) || (step % 50 == 49);
            if should_check {
                last_check_loss = loss.min(last_check_loss);
                std::mem::swap(&mut prog_cur.params, &mut params);
                let code = prog_cur.discretize_and_emit(fn_name, scalar_names);
                std::mem::swap(&mut prog_cur.params, &mut params);
                if !rejected_codes.contains(&code) {
                    if verify_problem_code_strict(problem, &code).is_ok() {
                        crate::learned_biases::record_success(
                            n_scalar,
                            initial_params.clone(),
                            format!(
                                "{}:{}:step",
                                if restart >= N_UNIV_ARR_RESTARTS {
                                    "random"
                                } else {
                                    "hand"
                                },
                                restart
                            ),
                        );
                        return Some(SolveResult {
                            success: true,
                            code,
                            method: "univ_arr_gradient".to_string(),
                            error: None,
                            metadata: DifferentiableMetadata::default(),
                        });
                    }
                    rejected_codes.insert(code);
                }
                if best_loss < loss {
                    std::mem::swap(&mut prog_cur.params, &mut best_params);
                    let code2 = prog_cur.discretize_and_emit(fn_name, scalar_names);
                    std::mem::swap(&mut prog_cur.params, &mut best_params);
                    if !rejected_codes.contains(&code2) {
                        if verify_problem_code_strict(problem, &code2).is_ok() {
                            crate::learned_biases::record_success(
                                n_scalar,
                                initial_params.clone(),
                                format!(
                                    "{}:{}:best",
                                    if restart >= N_UNIV_ARR_RESTARTS {
                                        "random"
                                    } else {
                                        "hand"
                                    },
                                    restart
                                ),
                            );
                            return Some(SolveResult {
                                success: true,
                                code: code2,
                                method: "univ_arr_gradient".to_string(),
                                error: None,
                                metadata: DifferentiableMetadata::default(),
                            });
                        }
                        rejected_codes.insert(code2);
                    }
                }
            }
            opt.step(&mut params, &grads);
        }

        let code = SoftUniversalArrayProgram {
            n_scalar,
            params: best_params,
        }
        .discretize_and_emit(fn_name, scalar_names);
        if !rejected_codes.contains(&code) && verify_problem_code_strict(problem, &code).is_ok() {
            crate::learned_biases::record_success(
                n_scalar,
                initial_params.clone(),
                format!(
                    "{}:{}:final",
                    if restart >= N_UNIV_ARR_RESTARTS {
                        "random"
                    } else {
                        "hand"
                    },
                    restart
                ),
            );
            return Some(SolveResult {
                success: true,
                code,
                method: "univ_arr_gradient".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }

    None
}

#[cfg(test)]
mod emergent_const_tests {
    use super::*;

    fn mk_example(arr: &[i64], scalar_args: &[i64], expected: i64) -> ArrExample {
        let mut padded = vec![0.0f32; 16]; // MAX_ARR
        for (i, &v) in arr.iter().enumerate() {
            if i < padded.len() {
                padded[i] = v as f32;
            }
        }
        ArrExample {
            arr: padded,
            arr_len: arr.len() as f32,
            scalar_args: scalar_args.iter().map(|&v| v as f32).collect(),
            expected: expected as f32,
        }
    }

    #[test]
    fn discovery_always_includes_anchor_set() {
        // Even when no example has {0, 1, -1}, they must appear so basic
        // zero/identity patterns stay available.
        let examples = vec![
            mk_example(&[5, 5, 5], &[], 15),
            mk_example(&[7, 7, 7, 7], &[], 28),
        ];
        let consts = discover_useful_consts(&examples);
        for anchor in [0, 1, -1] {
            assert!(
                consts.contains(&anchor),
                "anchor {anchor} missing from {:?}",
                consts
            );
        }
    }

    #[test]
    fn discovery_mines_values_from_examples() {
        // A problem whose examples repeatedly use 100 should get 100 as a
        // learned constant even though it's not in DEFAULT_CONSTS.
        let examples = vec![
            mk_example(&[100, 50], &[], 150),
            mk_example(&[100, 100, 100], &[], 300),
            mk_example(&[100, 25], &[], 125),
        ];
        let consts = discover_useful_consts(&examples);
        assert!(
            consts.contains(&100),
            "frequent value 100 missing from {:?}",
            consts
        );
    }

    #[test]
    fn discovery_falls_back_when_starved() {
        // A problem with zero examples can't mine anything — should still
        // return a valid [N_CONSTS; _] array, filled from DEFAULT_CONSTS
        // after the anchors.
        let consts = discover_useful_consts(&[]);
        assert_eq!(consts.len(), N_CONSTS);
        for anchor in [0, 1, -1] {
            assert!(consts.contains(&anchor));
        }
        // No duplicates.
        let mut unique = consts.to_vec();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), N_CONSTS);
    }
}
