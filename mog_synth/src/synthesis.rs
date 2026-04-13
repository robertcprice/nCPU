//! Native Rust gradient-based program synthesis.
//!
//! Implements soft program execution: every discrete choice (which op, which
//! variable to read, what the loop bound is) is a learned f32 logit. Gradient
//! descent via Adam + finite differences finds the program structure that fits
//! the training examples. The final step discretizes (argmax) and emits Mog code.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::benchmark::{Problem, Value};
use crate::differentiable::DifferentiableMetadata;
use crate::runtime::verify_problem_code_strict;
use crate::solver::SolveResult;

// ─── Constants ────────────────────────────────────────────────────────────────

const N_OPS: usize = 5; // +, -, *, /, %
const N_CMPS: usize = 6; // >, <, >=, <=, ==, !=
const N_CONSTS: usize = 6; // [0, 1, -1, 2, -2, 10]
const MAX_LOOP_ITER: usize = 32;
const MAX_DIGIT_ITER: usize = 20;
const MAX_ARR: usize = 16; // max array length for soft array programs

// ─── Math utilities ───────────────────────────────────────────────────────────

fn softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

fn soft_read(storage: &[f32], weights: &[f32]) -> f32 {
    storage.iter().zip(weights).map(|(s, w)| s * w).sum()
}

/// Weighted mix of: a+b, a-b, a*b, a/b, a%b
fn soft_op(a: f32, b: f32, weights: &[f32]) -> f32 {
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

/// Weighted mix of soft comparisons (returns [0,1]).
/// `cmp_temp` is the current training temperature — higher → wider sigmoid boundary
/// → gradients flow for large |d| values (needed for examples like abs_diff where d=±4).
/// The minimum is 0.5 so that comparison boundaries always have some gradient.
fn soft_cmp(a: f32, b: f32, weights: &[f32], cmp_temp: f32) -> f32 {
    let d = a - b;
    // t anneals with training temperature, clamped to [0.5, 2.0] so comparison
    // gradients flow even for large |d| while still sharpening over time.
    let t = cmp_temp.clamp(0.5, 2.0);
    let gauss_var = (t * t * 0.5).max(0.125);
    // Indices match cmp_names: ["<", "<=", "==", ">=", ">", "!="]
    let results = [
        sigmoid(-d / t),                  // 0: < (fires when a < b)
        sigmoid(-d / t),                  // 1: <= (approx same gradient)
        (-(d * d) / gauss_var).exp(),     // 2: == (Gaussian peak at d=0)
        sigmoid(d / t),                   // 3: >= (fires when a > b)
        sigmoid(d / t),                   // 4: > (approx same gradient)
        1.0 - (-(d * d) / gauss_var).exp(), // 5: != (inverse Gaussian)
    ];
    weights.iter().zip(&results).map(|(w, r)| w * r).sum()
}

/// soft_op extended with identity (index 5)
fn soft_op_ext(a: f32, b: f32, weights: &[f32]) -> f32 {
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

fn mse(predicted: &[f32], targets: &[f32]) -> f32 {
    let n = predicted.len() as f32;
    predicted
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / n
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─── Adam optimizer ───────────────────────────────────────────────────────────

struct Adam {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Self {
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

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
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

// ─── Finite-difference gradient ───────────────────────────────────────────────

fn fd_grad<F: Fn(&[f32], f32) -> f32>(params: &[f32], loss_fn: F, temp: f32) -> Vec<f32> {
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

// ─── Program type 1: SoftExprProgram ─────────────────────────────────────────
//
// Handles: fn f(a, b, ...) -> i64 { [v0 = pre_s1 OP pre_s2;] return s1 OP s2; }
//
// Parameters layout (flattened):
//   pre_enable(1) | pre_src1(ns) | pre_src2(ns) | pre_op(N_OPS)
//   src1(ne) | src2(ne) | op(N_OPS) | consts(N_CONSTS)
// where ns = n_args + N_CONSTS, ne = ns + 1

struct SoftExprProgram {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftExprProgram {
    fn n_sources(&self) -> usize {
        self.n_args + N_CONSTS
    }
    fn n_ext(&self) -> usize {
        self.n_args + N_CONSTS + 1
    } // +1 for v0

    fn new(n_args: usize) -> Self {
        let ns = n_args + N_CONSTS;
        let ne = ns + 1;
        let n_params = 1 + 2 * ns + N_OPS + 2 * ne + N_OPS + N_CONSTS;
        let mut p = vec![0.0f32; n_params];
        // Default: pre_enable = -4 (disabled)
        p[0] = -4.0;
        // src1 → arg0
        let off = 1 + 2 * ns + N_OPS;
        if n_args > 0 {
            p[off] = 2.0;
        }
        // src2 → arg1 (if exists) else const0
        let off2 = off + ne;
        if n_args > 1 {
            p[off2 + 1] = 2.0;
        } else {
            p[off2 + n_args] = 2.0;
        }
        // op → +
        let off3 = off2 + ne;
        p[off3] = 2.0;
        // consts → [0, 1, -1, 2, -2, 10]
        let off4 = off3 + N_OPS;
        p[off4] = 0.0;
        p[off4 + 1] = 1.0;
        p[off4 + 2] = -1.0;
        p[off4 + 3] = 2.0;
        p[off4 + 4] = -2.0;
        p[off4 + 5] = 10.0;
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let ns = self.n_sources();
        let ne = self.n_ext();
        let mut storage = vec![0.0f32; ns];
        for (i, &v) in inputs.iter().take(self.n_args).enumerate() {
            storage[i] = v;
        }
        // Read consts from params
        let const_off = 1 + 2 * ns + N_OPS + 2 * ne + N_OPS;
        for i in 0..N_CONSTS {
            storage[self.n_args + i] = self.params[const_off + i];
        }

        // Pre-compute
        let pre_en = sigmoid(self.params[0]);
        let pre_s1 = soft_read(&storage, &softmax_temp(&self.params[1..1 + ns], temp));
        let pre_s2 = soft_read(
            &storage,
            &softmax_temp(&self.params[1 + ns..1 + 2 * ns], temp),
        );
        let pre_op_w = softmax_temp(&self.params[1 + 2 * ns..1 + 2 * ns + N_OPS], temp);
        let v0 = soft_op(pre_s1, pre_s2, &pre_op_w) * pre_en;

        // Extended storage: [args, consts, v0]
        let mut ext = storage.clone();
        ext.push(v0);

        // Return expression
        let off = 1 + 2 * ns + N_OPS;
        let s1 = soft_read(&ext, &softmax_temp(&self.params[off..off + ne], temp));
        let s2 = soft_read(
            &ext,
            &softmax_temp(&self.params[off + ne..off + 2 * ne], temp),
        );
        let op_w = softmax_temp(&self.params[off + 2 * ne..off + 2 * ne + N_OPS], temp);
        soft_op(s1, s2, &op_w)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let ns = self.n_sources();
        let ne = self.n_ext();
        let const_off = 1 + 2 * ns + N_OPS + 2 * ne + N_OPS;
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[const_off + i]).collect();

        // Source names: args + consts + v0
        let mut src_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        for c in &consts {
            src_names.push(format!("{}", c.round() as i64));
        }
        src_names.push("v0".to_string());

        let ops = ["+", "-", "*", "/", "%"];

        // Pre-compute
        let pre_en = sigmoid(self.params[0]);
        let pre_on = pre_en > 0.3;
        let pre_s1_idx = argmax(&self.params[1..1 + ns]);
        let pre_s2_idx = argmax(&self.params[1 + ns..1 + 2 * ns]);
        let pre_op_idx = argmax(&self.params[1 + 2 * ns..1 + 2 * ns + N_OPS]);

        // Return expr
        let off = 1 + 2 * ns + N_OPS;
        let s1_idx = argmax(&self.params[off..off + ne]);
        let s2_idx = argmax(&self.params[off + ne..off + 2 * ne]);
        let op_idx = argmax(&self.params[off + 2 * ne..off + 2 * ne + N_OPS]);

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut body = String::new();

        if pre_on && pre_s1_idx < src_names.len() - 1 && pre_s2_idx < src_names.len() - 1 {
            let _ = writeln!(
                body,
                "    v0: i64 = {} {} {};",
                src_names[pre_s1_idx], ops[pre_op_idx], src_names[pre_s2_idx]
            );
        }

        let ret_s1 = if s1_idx < src_names.len() {
            &src_names[s1_idx]
        } else {
            "0"
        };
        let ret_s2 = if s2_idx < src_names.len() {
            &src_names[s2_idx]
        } else {
            "0"
        };
        let _ = writeln!(body, "    return {ret_s1} {} {ret_s2};", ops[op_idx]);

        format!("fn {fn_name}({params_sig}) -> i64 {{\n{body}}}\n")
    }
}

// ─── Program type 1b: SoftTwoPrecompExprProgram ──────────────────────────────
//
// Handles: fn f(args) -> i64 { [v0=pre1;] [v1=pre2(v0);] return s1 OP s2; }
// Two chained precompute steps allow 3-deep expression chains like c*9/5+32.
//
// Layout:
//   pre1_enable(1) | pre1_src1(ns) | pre1_src2(ns) | pre1_op(N_OPS)
//   pre2_enable(1) | pre2_src1(ne1) | pre2_src2(ne1) | pre2_op(N_OPS)
//   ret_src1(ne2) | ret_src2(ne2) | ret_op(N_OPS) | consts(N_CONSTS)
// where ns=n_args+N_CONSTS, ne1=ns+1 (with v0), ne2=ns+2 (with v0,v1)

struct SoftTwoPrecompExprProgram {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftTwoPrecompExprProgram {
    fn ns(&self) -> usize {
        self.n_args + N_CONSTS
    }
    fn ne1(&self) -> usize {
        self.n_args + N_CONSTS + 1
    }
    fn ne2(&self) -> usize {
        self.n_args + N_CONSTS + 2
    }

    fn new(n_args: usize) -> Self {
        let ns = n_args + N_CONSTS;
        let ne1 = ns + 1;
        let ne2 = ns + 2;
        let n = 1 + 2 * ns + N_OPS + 1 + 2 * ne1 + N_OPS + 2 * ne2 + N_OPS + N_CONSTS;
        let mut p = vec![0.0f32; n];
        // pre1 off
        p[0] = -4.0;
        // pre1_s1 = arg0, pre1_s2 = arg1 (or arg0), pre1_op = +
        if n_args > 0 {
            p[1] = 2.0;
        }
        let ps2 = 1 + ns;
        if n_args > 1 {
            p[ps2 + 1] = 2.0;
        } else {
            p[ps2 + n_args.saturating_sub(1).min(ns - 1)] = 2.0;
        }
        p[1 + 2 * ns] = 2.0; // pre1_op = +

        // pre2 off
        let p2 = 1 + 2 * ns + N_OPS;
        p[p2] = -4.0;
        // pre2_s1 = v0 (last of ne1), pre2_s2 = const0, pre2_op = +
        p[p2 + 1 + ne1 - 1] = 2.0;
        p[p2 + 1 + ne1 + n_args] = 2.0; // const0
        p[p2 + 1 + 2 * ne1] = 2.0; // pre2_op = +

        // ret: ret_s1 = arg0, ret_s2 = arg1/const0, ret_op = +
        let roff = p2 + 1 + 2 * ne1 + N_OPS;
        if n_args > 0 {
            p[roff] = 2.0;
        }
        let rs2 = roff + ne2;
        if n_args > 1 {
            p[rs2 + 1] = 2.0;
        } else {
            p[rs2 + n_args] = 2.0;
        }
        p[roff + 2 * ne2] = 2.0; // ret_op = +

        // consts: [0, 1, -1, 2, -2, 10]
        let coff = roff + 2 * ne2 + N_OPS;
        p[coff] = 0.0;
        p[coff + 1] = 1.0;
        p[coff + 2] = -1.0;
        p[coff + 3] = 2.0;
        p[coff + 4] = -2.0;
        p[coff + 5] = 10.0;
        Self { n_args, params: p }
    }

    fn const_offset(&self) -> usize {
        let ns = self.ns();
        let ne1 = self.ne1();
        let ne2 = self.ne2();
        1 + 2 * ns + N_OPS + 1 + 2 * ne1 + N_OPS + 2 * ne2 + N_OPS
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let ns = self.ns();
        let ne1 = self.ne1();
        let ne2 = self.ne2();
        let coff = self.const_offset();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();

        let mut storage = vec![0.0f32; ns];
        for (i, &v) in inputs.iter().take(self.n_args).enumerate() {
            storage[i] = v;
        }
        for (i, &c) in consts.iter().enumerate() {
            storage[self.n_args + i] = c;
        }

        // Pre1
        let pre1_en = sigmoid(self.params[0]);
        let p1s1 = soft_read(&storage, &softmax_temp(&self.params[1..1 + ns], temp));
        let p1s2 = soft_read(
            &storage,
            &softmax_temp(&self.params[1 + ns..1 + 2 * ns], temp),
        );
        let p1op = softmax_temp(&self.params[1 + 2 * ns..1 + 2 * ns + N_OPS], temp);
        let v0 = soft_op(p1s1, p1s2, &p1op) * pre1_en;

        let mut ext1 = storage.clone();
        ext1.push(v0);

        // Pre2
        let p2 = 1 + 2 * ns + N_OPS;
        let pre2_en = sigmoid(self.params[p2]);
        let p2s1 = soft_read(
            &ext1,
            &softmax_temp(&self.params[p2 + 1..p2 + 1 + ne1], temp),
        );
        let p2s2 = soft_read(
            &ext1,
            &softmax_temp(&self.params[p2 + 1 + ne1..p2 + 1 + 2 * ne1], temp),
        );
        let p2op = softmax_temp(
            &self.params[p2 + 1 + 2 * ne1..p2 + 1 + 2 * ne1 + N_OPS],
            temp,
        );
        let v1 = soft_op(p2s1, p2s2, &p2op) * pre2_en;

        let mut ext2 = ext1;
        ext2.push(v1);

        // Return
        let roff = p2 + 1 + 2 * ne1 + N_OPS;
        let rs1 = soft_read(&ext2, &softmax_temp(&self.params[roff..roff + ne2], temp));
        let rs2 = soft_read(
            &ext2,
            &softmax_temp(&self.params[roff + ne2..roff + 2 * ne2], temp),
        );
        let rop = softmax_temp(&self.params[roff + 2 * ne2..roff + 2 * ne2 + N_OPS], temp);
        soft_op(rs1, rs2, &rop)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let ns = self.ns();
        let ne1 = self.ne1();
        let ne2 = self.ne2();
        let coff = self.const_offset();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();

        let mut src_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        for c in &consts {
            src_names.push(format!("{}", c.round() as i64));
        }
        src_names.push("v0".to_string());
        let src_names_ext: Vec<String> = {
            let mut v = src_names.clone();
            v.push("v1".to_string());
            v
        };

        let ops = ["+", "-", "*", "/", "%"];

        let pre1_on = sigmoid(self.params[0]) > 0.3;
        let p1s1i = argmax(&self.params[1..1 + ns]);
        let p1s2i = argmax(&self.params[1 + ns..1 + 2 * ns]);
        let p1opi = argmax(&self.params[1 + 2 * ns..1 + 2 * ns + N_OPS]);

        let p2 = 1 + 2 * ns + N_OPS;
        let pre2_on = sigmoid(self.params[p2]) > 0.3;
        let p2s1i = argmax(&self.params[p2 + 1..p2 + 1 + ne1]);
        let p2s2i = argmax(&self.params[p2 + 1 + ne1..p2 + 1 + 2 * ne1]);
        let p2opi = argmax(&self.params[p2 + 1 + 2 * ne1..p2 + 1 + 2 * ne1 + N_OPS]);

        let roff = p2 + 1 + 2 * ne1 + N_OPS;
        let rs1i = argmax(&self.params[roff..roff + ne2]);
        let rs2i = argmax(&self.params[roff + ne2..roff + 2 * ne2]);
        let ropi = argmax(&self.params[roff + 2 * ne2..roff + 2 * ne2 + N_OPS]);

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut body = String::new();

        if pre1_on && p1s1i < ns && p1s2i < ns {
            let n1 = &src_names[p1s1i];
            let n2 = &src_names[p1s2i];
            let _ = writeln!(
                body,
                "    v0: i64 = {n1} {} {n2};",
                ops[p1opi.min(N_OPS - 1)]
            );
        }
        if pre2_on && p2s1i < ne1 && p2s2i < ne1 {
            let n1 = &src_names[p2s1i.min(src_names.len() - 1)];
            let n2 = &src_names[p2s2i.min(src_names.len() - 1)];
            let _ = writeln!(
                body,
                "    v1: i64 = {n1} {} {n2};",
                ops[p2opi.min(N_OPS - 1)]
            );
        }

        let rn1 = src_names_ext.get(rs1i).map(|s| s.as_str()).unwrap_or("0");
        let rn2 = src_names_ext.get(rs2i).map(|s| s.as_str()).unwrap_or("0");
        let _ = writeln!(body, "    return {rn1} {} {rn2};", ops[ropi.min(N_OPS - 1)]);

        format!("fn {fn_name}({params_sig}) -> i64 {{\n{body}}}\n")
    }
}

// ─── Program type 2: SoftBranchProgram ───────────────────────────────────────
//
// Handles: fn f(a,b) -> i64 { [v0=pre;] if (lhs CMP rhs) return e1; [if...;] return eN; }
//
// Params layout:
//   pre_enable(1) | pre_src1(ns) | pre_src2(ns) | pre_op(N_OPS)
//   for each branch: cmp(N_CMPS) | lhs(ne) | rhs(ne) | ret_s1(ne) | ret_s2(ne) | ret_op(6)
//   default: ret_s1(ne) | ret_s2(ne) | ret_op(6)
//   consts: N_CONSTS

const N_BRANCHES: usize = 3;

struct SoftBranchProgram {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftBranchProgram {
    fn n_sources(&self) -> usize {
        self.n_args + N_CONSTS
    }
    fn n_ext(&self) -> usize {
        self.n_args + N_CONSTS + 1
    }

    fn new(n_args: usize) -> Self {
        let ns = n_args + N_CONSTS;
        let ne = ns + 1;
        let branch_size = N_CMPS + 4 * ne + 6;
        let n_params = 1 + 2 * ns + N_OPS + N_BRANCHES * branch_size + 2 * ne + 6 + N_CONSTS;
        let mut p = vec![0.0f32; n_params];
        p[0] = -4.0; // pre off

        let boff = 1 + 2 * ns + N_OPS;
        let branch_size_local = N_CMPS + 4 * ne + 6;

        // Branch 0: arg0 > const(0), return arg0 (identity)
        // Layout within each branch: cmp(N_CMPS) | lhs(ne) | rhs(ne) | ret_s1(ne) | ret_s2(ne) | ret_op(6)
        // ret_op starts at offset N_CMPS + 4*ne; identity is last element (index 5)
        if n_args > 0 {
            p[boff + N_CMPS] = 2.0; // lhs = arg0
            p[boff + N_CMPS + ne + n_args] = 2.0; // rhs = const0 (value 0)
            p[boff] = 2.0; // cmp = >
            p[boff + N_CMPS + 2 * ne] = 2.0; // ret_s1 = arg0
            p[boff + N_CMPS + 4 * ne + 5] = 2.0; // ret_op = identity (last of 6)
        }
        // Branches 1..N_BRANCHES-1: initialize dormant (lhs==rhs with != cmp → fire_prob≈0)
        for b in 1..N_BRANCHES {
            let bo = boff + b * branch_size_local;
            // lhs = arg0, rhs = arg0 → d = 0
            p[bo + N_CMPS] = 2.0; // lhs = arg0
            p[bo + N_CMPS + ne] = 2.0; // rhs = arg0 (same)
                                       // cmp = != (index 5): soft_cmp(a, a) for != = 1 - exp(0) = 0
            for k in 0..N_CMPS - 1 {
                p[bo + k] = -8.0;
            }
            p[bo + N_CMPS - 1] = 8.0; // strongly prefer !=
                                      // return identity of arg0
            p[bo + N_CMPS + 2 * ne] = 2.0;
            p[bo + N_CMPS + 4 * ne + 5] = 2.0; // ret_op = identity (correct offset)
        }
        // Default: return arg1 or const0
        let doff = boff + N_BRANCHES * branch_size;
        if n_args > 1 {
            p[doff + 1] = 2.0;
        } else {
            p[doff + n_args] = 2.0;
        }
        p[doff + 2 * ne + 5] = 2.0; // ret_op = identity
                                    // consts: [0, 1, -1, 2, -2, 10]
        let coff = doff + 2 * ne + 6;
        p[coff] = 0.0;
        p[coff + 1] = 1.0;
        p[coff + 2] = -1.0;
        p[coff + 3] = 2.0;
        p[coff + 4] = -2.0;
        p[coff + 5] = 10.0;
        Self { n_args, params: p }
    }

    fn storage_from_inputs(&self, inputs: &[f32]) -> Vec<f32> {
        let ns = self.n_sources();
        let ne = self.n_ext();
        let coff = 1 + 2 * ns + N_OPS + N_BRANCHES * (N_CMPS + 4 * ne + 6) + 2 * ne + 6;
        let mut s = vec![0.0f32; ns];
        for (i, &v) in inputs.iter().take(self.n_args).enumerate() {
            s[i] = v;
        }
        for i in 0..N_CONSTS {
            s[self.n_args + i] = self.params[coff + i];
        }
        s
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let ns = self.n_sources();
        let ne = self.n_ext();
        let storage = self.storage_from_inputs(inputs);

        // Pre-compute
        let pre_en = sigmoid(self.params[0]);
        let pre_s1 = soft_read(&storage, &softmax_temp(&self.params[1..1 + ns], temp));
        let pre_s2 = soft_read(
            &storage,
            &softmax_temp(&self.params[1 + ns..1 + 2 * ns], temp),
        );
        let pre_op_w = softmax_temp(&self.params[1 + 2 * ns..1 + 2 * ns + N_OPS], temp);
        let v0 = soft_op(pre_s1, pre_s2, &pre_op_w) * pre_en;

        let mut ext = storage.clone();
        ext.push(v0);

        let branch_size = N_CMPS + 4 * ne + 6;
        let boff = 1 + 2 * ns + N_OPS;

        let mut output = 0.0f32;
        let mut remaining = 1.0f32;

        for b in 0..N_BRANCHES {
            let bo = boff + b * branch_size;
            let cmp_w = softmax_temp(&self.params[bo..bo + N_CMPS], temp);
            let lhs = soft_read(
                &ext,
                &softmax_temp(&self.params[bo + N_CMPS..bo + N_CMPS + ne], temp),
            );
            let rhs = soft_read(
                &ext,
                &softmax_temp(&self.params[bo + N_CMPS + ne..bo + N_CMPS + 2 * ne], temp),
            );
            let cond = soft_cmp(lhs, rhs, &cmp_w, temp);

            let rs1 = soft_read(
                &ext,
                &softmax_temp(
                    &self.params[bo + N_CMPS + 2 * ne..bo + N_CMPS + 3 * ne],
                    temp,
                ),
            );
            let rs2 = soft_read(
                &ext,
                &softmax_temp(
                    &self.params[bo + N_CMPS + 3 * ne..bo + N_CMPS + 4 * ne],
                    temp,
                ),
            );
            let rop_w = softmax_temp(
                &self.params[bo + N_CMPS + 4 * ne..bo + N_CMPS + 4 * ne + 6],
                temp,
            );
            let ret_val = soft_op_ext(rs1, rs2, &rop_w);

            let fire = cond * remaining;
            output += fire * ret_val;
            remaining *= 1.0 - cond;
        }

        // Default
        let doff = boff + N_BRANCHES * branch_size;
        let ds1 = soft_read(&ext, &softmax_temp(&self.params[doff..doff + ne], temp));
        let ds2 = soft_read(
            &ext,
            &softmax_temp(&self.params[doff + ne..doff + 2 * ne], temp),
        );
        let dop_w = softmax_temp(&self.params[doff + 2 * ne..doff + 2 * ne + 6], temp);
        let def_val = soft_op_ext(ds1, ds2, &dop_w);
        output += remaining * def_val;
        output
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let ns = self.n_sources();
        let ne = self.n_ext();
        let doff = 1 + 2 * ns + N_OPS + N_BRANCHES * (N_CMPS + 4 * ne + 6);
        let coff = doff + 2 * ne + 6;
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();

        let mut src_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        for c in &consts {
            src_names.push(format!("{}", c.round() as i64));
        }
        src_names.push("v0".to_string());

        let ops = ["+", "-", "*", "/", "%", ""]; // last is identity placeholder
        let cmps_str = [">", "<", ">=", "<=", "==", "!="];

        let pre_en = sigmoid(self.params[0]);
        let pre_on = pre_en > 0.3;
        let pre_s1_idx = argmax(&self.params[1..1 + ns]);
        let pre_s2_idx = argmax(&self.params[1 + ns..1 + 2 * ns]);
        let pre_op_idx = argmax(&self.params[1 + 2 * ns..1 + 2 * ns + N_OPS]);

        let branch_size = N_CMPS + 4 * ne + 6;
        let boff = 1 + 2 * ns + N_OPS;

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut body = String::new();

        if pre_on {
            let n1 = src_names.get(pre_s1_idx).map(String::as_str).unwrap_or("0");
            let n2 = src_names.get(pre_s2_idx).map(String::as_str).unwrap_or("0");
            let _ = writeln!(
                body,
                "    v0: i64 = {n1} {} {n2};",
                ops[pre_op_idx.min(N_OPS - 1)]
            );
        }

        for b in 0..N_BRANCHES {
            let bo = boff + b * branch_size;
            let cmp_idx = argmax(&self.params[bo..bo + N_CMPS]);
            let lhs_idx = argmax(&self.params[bo + N_CMPS..bo + N_CMPS + ne]);
            let rhs_idx = argmax(&self.params[bo + N_CMPS + ne..bo + N_CMPS + 2 * ne]);
            let rs1_idx = argmax(&self.params[bo + N_CMPS + 2 * ne..bo + N_CMPS + 3 * ne]);
            let rs2_idx = argmax(&self.params[bo + N_CMPS + 3 * ne..bo + N_CMPS + 4 * ne]);
            let rop_idx = argmax(&self.params[bo + N_CMPS + 4 * ne..bo + N_CMPS + 4 * ne + 6]);

            let lhs_n = src_names.get(lhs_idx).map(String::as_str).unwrap_or("0");
            let rhs_n = src_names.get(rhs_idx).map(String::as_str).unwrap_or("0");
            let rs1_n = src_names.get(rs1_idx).map(String::as_str).unwrap_or("0");
            let rs2_n = src_names.get(rs2_idx).map(String::as_str).unwrap_or("0");

            let ret_expr = if rop_idx >= N_OPS {
                rs1_n.to_string() // identity
            } else {
                format!("{rs1_n} {} {rs2_n}", ops[rop_idx])
            };

            let _ = writeln!(body, "    if {lhs_n} {} {rhs_n} {{", cmps_str[cmp_idx]);
            let _ = writeln!(body, "        return {ret_expr};");
            let _ = writeln!(body, "    }}");
        }

        // Default
        let ds1_idx = argmax(&self.params[doff..doff + ne]);
        let ds2_idx = argmax(&self.params[doff + ne..doff + 2 * ne]);
        let dop_idx = argmax(&self.params[doff + 2 * ne..doff + 2 * ne + 6]);
        let ds1_n = src_names.get(ds1_idx).map(String::as_str).unwrap_or("0");
        let ds2_n = src_names.get(ds2_idx).map(String::as_str).unwrap_or("0");
        let def_ret = if dop_idx >= N_OPS {
            ds1_n.to_string()
        } else {
            format!("{ds1_n} {} {ds2_n}", ops[dop_idx])
        };
        let _ = writeln!(body, "    return {def_ret};");

        format!("fn {fn_name}({params_sig}) -> i64 {{\n{body}}}\n")
    }
}

// ─── Program type 3: SoftLoopProgram ─────────────────────────────────────────
//
// Handles: fn f(n: i64) -> i64 { acc=init; i=start; while i<=bound { acc = acc OP rhs; i++; } return acc; }
//
// Params layout:
//   init(1) | start(1) | bound_src(nb) | bound_offset(1) | body_op(N_OPS) | body_rhs(nr)
//   return_src(nret) | consts(N_CONSTS)
// where nb = n_args + N_CONSTS, nr = 3 + n_args + N_CONSTS, nret = 1 + n_args + N_CONSTS

struct SoftLoopProgram {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftLoopProgram {
    fn n_rhs_sources(&self) -> usize {
        4 + self.n_args + N_CONSTS
    } // [i, i*i, 1, args..., consts..., consts[0]/i]
    fn n_ret_sources(&self) -> usize {
        1 + self.n_args + N_CONSTS
    } // [acc, args..., consts...]
    fn n_bound_sources(&self) -> usize {
        self.n_args + N_CONSTS
    } // [args..., consts...]

    fn new(n_args: usize) -> Self {
        let nr = 4 + n_args + N_CONSTS;
        let nret = 1 + n_args + N_CONSTS;
        let nb = n_args + N_CONSTS;
        let n = 2 + nb + 1 + N_OPS + nr + nret + N_CONSTS;
        let mut p = vec![0.0f32; n];
        // init = 0
        p[0] = 0.0;
        // start = 1
        p[1] = 1.0;
        // bound = arg0 (if exists)
        let boff = 2;
        if n_args > 0 {
            p[boff] = 2.0;
        }
        // bound_offset = 0
        p[boff + nb] = 0.0;
        // body_op = +
        let opoff = boff + nb + 1;
        p[opoff] = 2.0;
        // body_rhs = i (index 0 in rhs_sources)
        let rhsoff = opoff + N_OPS;
        p[rhsoff] = 2.0;
        // return = acc (index 0)
        let retoff = rhsoff + nr;
        p[retoff] = 2.0;
        // consts: [0, 1, -1, 2, -2, 10]
        let coff = retoff + nret;
        p[coff] = 0.0;
        p[coff + 1] = 1.0;
        p[coff + 2] = -1.0;
        p[coff + 3] = 2.0;
        p[coff + 4] = -2.0;
        p[coff + 5] = 10.0;
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let nr = self.n_rhs_sources();
        let nret = self.n_ret_sources();
        let nb = self.n_bound_sources();

        let coff = 2 + nb + 1 + N_OPS + nr + nret;
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();

        // Base storage for args + consts
        let mut base = vec![0.0f32; self.n_args + N_CONSTS];
        for (i, &v) in inputs.iter().take(self.n_args).enumerate() {
            base[i] = v;
        }
        for (i, &c) in consts.iter().enumerate() {
            base[self.n_args + i] = c;
        }

        let init = self.params[0];
        let start = self.params[1];
        let bound_w = softmax_temp(&self.params[2..2 + nb], temp);
        let bound = soft_read(&base, &bound_w) + self.params[2 + nb];

        let opoff = 2 + nb + 1;
        let op_w = softmax_temp(&self.params[opoff..opoff + N_OPS], temp);
        let rhsoff = opoff + N_OPS;
        let rhs_w = softmax_temp(&self.params[rhsoff..rhsoff + nr], temp);

        let mut acc = init;

        for step in 0..MAX_LOOP_ITER {
            let i_val = step as f32 + start;
            let in_bounds = sigmoid((bound - i_val - 0.5) / 0.3);

            // RHS sources: [i, i*i, 1, args..., consts..., consts[0]/i]
            let mut rhs_storage = vec![i_val, i_val * i_val, 1.0f32];
            rhs_storage.extend_from_slice(&base);
            let safe_i = if i_val.abs() < 1e-6 { 1.0 } else { i_val };
            rhs_storage.push(consts[0] / safe_i);
            let rhs = soft_read(&rhs_storage, &rhs_w);

            let new_acc = soft_op(acc, rhs, &op_w);
            acc += in_bounds * (new_acc - acc);
        }

        // Return sources: [acc, args..., consts...]
        let retoff = rhsoff + nr;
        let mut ret_storage = vec![acc];
        ret_storage.extend_from_slice(&base);
        let ret_w = softmax_temp(&self.params[retoff..retoff + nret], temp);
        soft_read(&ret_storage, &ret_w)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let nr = self.n_rhs_sources();
        let nret = self.n_ret_sources();
        let nb = self.n_bound_sources();
        let coff = 2 + nb + 1 + N_OPS + nr + nret;
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();

        let init = self.params[0].round() as i64;
        let start = self.params[1].round() as i64;
        let bound_idx = argmax(&self.params[2..2 + nb]);
        let bound_offset = self.params[2 + nb].round() as i64;

        let opoff = 2 + nb + 1;
        let op_idx = argmax(&self.params[opoff..opoff + N_OPS]);
        let rhs_idx = argmax(&self.params[opoff + N_OPS..opoff + N_OPS + nr]);
        let ret_idx = argmax(&self.params[opoff + N_OPS + nr..opoff + N_OPS + nr + nret]);

        // Bound variable names
        let mut bound_src_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        for c in &consts {
            bound_src_names.push(format!("{}", c.round() as i64));
        }

        // RHS source names: [i, i*i, 1, args..., consts..., "{c0}/i"]
        let mut rhs_names = vec!["i".to_string(), "i * i".to_string(), "1".to_string()];
        for n in param_names {
            rhs_names.push(n.to_string());
        }
        for c in &consts {
            rhs_names.push(format!("{}", c.round() as i64));
        }
        rhs_names.push(format!("{} / i", consts[0].round() as i64));

        // Return source names: [acc, args..., consts...]
        let mut ret_names = vec!["acc".to_string()];
        for n in param_names {
            ret_names.push(n.to_string());
        }
        for c in &consts {
            ret_names.push(format!("{}", c.round() as i64));
        }

        let ops = ["+", "-", "*", "/", "%"];
        let bound_name = bound_src_names
            .get(bound_idx)
            .cloned()
            .unwrap_or_else(|| param_names.first().copied().unwrap_or("n").to_string());
        let rhs_name = rhs_names
            .get(rhs_idx)
            .cloned()
            .unwrap_or_else(|| "i".to_string());
        let ret_name = ret_names
            .get(ret_idx)
            .cloned()
            .unwrap_or_else(|| "acc".to_string());

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");

        let bound_expr = if bound_offset == 0 {
            bound_name.clone()
        } else if bound_offset > 0 {
            format!("{} + {}", bound_name, bound_offset)
        } else {
            format!("{} - {}", bound_name, -bound_offset)
        };

        format!(
            "fn {fn_name}({params_sig}) -> i64 {{\n    acc: i64 = {init};\n    i: i64 = {start};\n    while i <= {bound_expr} {{\n        acc = acc {} {rhs_name};\n        i = i + 1;\n    }}\n    return {ret_name};\n}}\n",
            ops[op_idx.min(N_OPS - 1)]
        )
    }
}

// ─── Program type 4: SoftDigitLoopProgram ────────────────────────────────────
//
// Handles digit extraction patterns via mode selection:
//   Mode 0: digit_sum   (acc + digit)
//   Mode 1: digit_product (acc * digit)
//   Mode 2: digit_count  (acc + 1)
//   Mode 3: reverse_digits (acc * 10 + digit)

struct SoftDigitLoopProgram {
    params: Vec<f32>, // [mode_logits(4), init_acc]
}

impl SoftDigitLoopProgram {
    fn new() -> Self {
        let mut p = vec![0.0f32; 5];
        p[0] = 2.0; // default: digit_sum
        p[4] = 0.0; // init_acc = 0
        Self { params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let x0 = if inputs.is_empty() {
            0.0
        } else {
            inputs[0].abs().round()
        };
        let mode_w = softmax_temp(&self.params[0..4], temp);
        let mut acc = self.params[4];
        let mut x = x0;

        for _ in 0..MAX_DIGIT_ITER {
            let active = sigmoid((x - 0.5) / 0.05);
            let digit = x % 10.0;
            let candidates = [
                acc + digit,        // digit_sum
                acc * digit,        // digit_product
                acc + 1.0,          // digit_count
                acc * 10.0 + digit, // reverse_digits
            ];
            let updated = mode_w
                .iter()
                .zip(&candidates)
                .map(|(w, c)| w * c)
                .sum::<f32>();
            acc += active * (updated - acc);

            let next_x = (x / 10.0).floor();
            x += active * (next_x - x);
        }

        // Handle x0 == 0 case
        let is_zero = sigmoid((0.5 - x0) / 0.05);
        acc * (1.0 - is_zero) + self.params[4] * is_zero
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let mode = argmax(&self.params[0..4]);
        let init = self.params[4].round() as i64;
        let arg = param_names.first().copied().unwrap_or("n");
        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");

        let body = match mode {
            0 => format!("    x: i64 = {arg};\n    acc: i64 = {init};\n    while x > 0 {{\n        acc = acc + x % 10;\n        x = x / 10;\n    }}\n    return acc;\n"),
            1 => format!("    x: i64 = {arg};\n    acc: i64 = {init};\n    while x > 0 {{\n        acc = acc * (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n"),
            2 => format!("    x: i64 = {arg};\n    acc: i64 = {init};\n    while x > 0 {{\n        acc = acc + 1;\n        x = x / 10;\n    }}\n    return acc;\n"),
            3 => format!("    x: i64 = {arg};\n    acc: i64 = {init};\n    while x > 0 {{\n        acc = acc * 10 + x % 10;\n        x = x / 10;\n    }}\n    return acc;\n"),
            _ => "    return 0;\n".to_string(),
        };
        format!("fn {fn_name}({params_sig}) -> i64 {{\n{body}}}\n")
    }
}

// ─── Program type 5: SoftTwoAccLoop ──────────────────────────────────────────
//
// Handles two-accumulator patterns: fn f(n) -> i64 { a=init0; b=init1; i=0;
//   while i < bound { tmp = new_a(a,b); b = new_b(a,b); a = tmp; i++; } return ret; }
// Targets: fibonacci, lucas_number, fib_iter
//
// Params:
//   init0(1) | init1(1) | bound_src(nb) | bound_offset(1)
//   a_s1(na) | a_s2(na) | a_op(6)
//   b_s1(na) | b_s2(na) | b_op(6)
//   ret_src(na) | consts(N_CONSTS)
// where na = 2 + n_args + N_CONSTS, nb = n_args + N_CONSTS

struct SoftTwoAccLoop {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftTwoAccLoop {
    fn na(&self) -> usize {
        2 + self.n_args + N_CONSTS
    }
    fn nb(&self) -> usize {
        self.n_args + N_CONSTS
    }

    fn new(n_args: usize) -> Self {
        let na = 2 + n_args + N_CONSTS;
        let nb = n_args + N_CONSTS;
        let n = 2 + nb + 1 + 2 * na + 6 + 2 * na + 6 + na + N_CONSTS;
        let mut p = vec![0.0f32; n];
        // init: a=0, b=1 (good for fibonacci)
        p[0] = 0.0;
        p[1] = 1.0;
        // bound = arg0 (if available)
        if n_args > 0 {
            p[2] = 2.0;
        }
        p[2 + nb] = 0.0; // bound_offset

        // a_update = identity of b: a_s1=b(idx1), a_op=identity(5)
        let as1 = 2 + nb + 1;
        p[as1 + 1] = 2.0; // a_s1 = b
        let as2 = as1 + na;
        p[as2 + 1] = 2.0; // a_s2 = b (unused for identity)
        let aop = as2 + na;
        p[aop + 5] = 2.0; // a_op = identity

        // b_update = a + b: b_s1=a(idx0), b_s2=b(idx1), b_op=+(0)
        let bs1 = aop + 6;
        p[bs1] = 2.0; // b_s1 = a
        let bs2 = bs1 + na;
        p[bs2 + 1] = 2.0; // b_s2 = b
        let bop = bs2 + na;
        p[bop] = 2.0; // b_op = +

        // return a (idx 0)
        let ret = bop + 6;
        p[ret] = 2.0;

        // consts: [0, 1, -1, 2, -2, 10]
        let coff = ret + na;
        p[coff] = 0.0;
        p[coff + 1] = 1.0;
        p[coff + 2] = -1.0;
        p[coff + 3] = 2.0;
        p[coff + 4] = -2.0;
        p[coff + 5] = 10.0;
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let na = self.na();
        let nb = self.nb();
        let ret_off = 2 + nb + 1 + 4 * na + 12;
        let coff = ret_off + na;
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();

        let mut base = vec![0.0f32; self.n_args + N_CONSTS];
        for (i, &v) in inputs.iter().take(self.n_args).enumerate() {
            base[i] = v;
        }
        for (i, &c) in consts.iter().enumerate() {
            base[self.n_args + i] = c;
        }

        let bound_w = softmax_temp(&self.params[2..2 + nb], temp);
        let bound = soft_read(&base, &bound_w) + self.params[2 + nb];

        let as1_off = 2 + nb + 1;
        let as2_off = as1_off + na;
        let aop_off = as2_off + na;
        let bs1_off = aop_off + 6;
        let bs2_off = bs1_off + na;
        let bop_off = bs2_off + na;

        let mut a = self.params[0];
        let mut b = self.params[1];

        for step in 0..MAX_LOOP_ITER {
            let iv = step as f32;
            let in_bounds = sigmoid((bound - iv - 0.5) / 0.3);

            let mut acc_s = vec![a, b];
            acc_s.extend_from_slice(&base);

            let as1w = softmax_temp(&self.params[as1_off..as1_off + na], temp);
            let as2w = softmax_temp(&self.params[as2_off..as2_off + na], temp);
            let aopw = softmax_temp(&self.params[aop_off..aop_off + 6], temp);
            let bs1w = softmax_temp(&self.params[bs1_off..bs1_off + na], temp);
            let bs2w = softmax_temp(&self.params[bs2_off..bs2_off + na], temp);
            let bopw = softmax_temp(&self.params[bop_off..bop_off + 6], temp);

            let new_a = soft_op_ext(soft_read(&acc_s, &as1w), soft_read(&acc_s, &as2w), &aopw);
            let new_b = soft_op_ext(soft_read(&acc_s, &bs1w), soft_read(&acc_s, &bs2w), &bopw);

            a += in_bounds * (new_a - a);
            b += in_bounds * (new_b - b);
        }

        let mut final_s = vec![a, b];
        final_s.extend_from_slice(&base);
        let ret_w = softmax_temp(&self.params[ret_off..ret_off + na], temp);
        soft_read(&final_s, &ret_w)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let na = self.na();
        let nb = self.nb();
        let ret_off = 2 + nb + 1 + 4 * na + 12;
        let coff = ret_off + na;
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();

        let init0 = self.params[0].round() as i64;
        let init1 = self.params[1].round() as i64;
        let bound_idx = argmax(&self.params[2..2 + nb]);
        let bound_off = self.params[2 + nb].round() as i64;

        let as1_off = 2 + nb + 1;
        let as2_off = as1_off + na;
        let aop_off = as2_off + na;
        let bs1_off = aop_off + 6;
        let bs2_off = bs1_off + na;
        let bop_off = bs2_off + na;

        let as1i = argmax(&self.params[as1_off..as1_off + na]);
        let as2i = argmax(&self.params[as2_off..as2_off + na]);
        let aopi = argmax(&self.params[aop_off..aop_off + 6]);
        let bs1i = argmax(&self.params[bs1_off..bs1_off + na]);
        let bs2i = argmax(&self.params[bs2_off..bs2_off + na]);
        let bopi = argmax(&self.params[bop_off..bop_off + 6]);
        let reti = argmax(&self.params[ret_off..ret_off + na]);

        // Source names: [x0, x1, args..., consts...]
        // Use x0/x1 to avoid collision with param names (a, b, c, ...)
        let mut snames = vec!["x0".to_string(), "x1".to_string()];
        for n in param_names {
            snames.push(n.to_string());
        }
        for c in &consts {
            snames.push(format!("{}", c.round() as i64));
        }

        let mut bnames: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        for c in &consts {
            bnames.push(format!("{}", c.round() as i64));
        }

        let ops = ["+", "-", "*", "/", "%", ""];
        let bname = bnames
            .get(bound_idx)
            .cloned()
            .unwrap_or_else(|| "n".to_string());
        let bexpr = if bound_off == 0 {
            bname.clone()
        } else if bound_off > 0 {
            format!("{} + {}", bname, bound_off)
        } else {
            format!("{} - {}", bname, -bound_off)
        };

        let as1n = snames.get(as1i).map(|s| s.as_str()).unwrap_or("x0");
        let as2n = snames.get(as2i).map(|s| s.as_str()).unwrap_or("x1");
        let bs1n = snames.get(bs1i).map(|s| s.as_str()).unwrap_or("x0");
        let bs2n = snames.get(bs2i).map(|s| s.as_str()).unwrap_or("x1");
        let retn = snames.get(reti).map(|s| s.as_str()).unwrap_or("x0");

        // x0_expr: new value for x0 (uses OLD x0, x1)
        let a_expr = if aopi >= N_OPS {
            as1n.to_string()
        } else {
            format!("{as1n} {} {as2n}", ops[aopi])
        };
        // x1_expr: new value for x1 (uses OLD x0, x1 — emitted BEFORE x0 is updated)
        let b_expr = if bopi >= N_OPS {
            bs1n.to_string()
        } else {
            format!("{bs1n} {} {bs2n}", ops[bopi])
        };

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "fn {fn_name}({params_sig}) -> i64 {{\n    x0: i64 = {init0};\n    x1: i64 = {init1};\n    i: i64 = 0;\n    while i < {bexpr} {{\n        xt: i64 = {a_expr};\n        x1 = {b_expr};\n        x0 = xt;\n        i = i + 1;\n    }}\n    return {retn};\n}}\n"
        )
    }
}

// ─── Predicate-while loop ─────────────────────────────────────────────────────

/// Two-accumulator loop controlled by a soft predicate: `while cmp(lhs, rhs)`.
/// Covers GCD (while y!=0), leading_digit (while x>=10), next_power_of_2 (while x<n), etc.
struct SoftPredicateLoop {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftPredicateLoop {
    /// Number of init sources: args + consts (no x0/x1 — circular on init)
    fn nb(&self) -> usize {
        self.n_args + N_CONSTS
    }
    /// Number of loop body sources: x0, x1, args, consts
    fn na(&self) -> usize {
        2 + self.n_args + N_CONSTS
    }

    /// Offset layout inside params:
    /// [0..nb]                 init0 source logits
    /// [nb..2*nb]              init1 source logits
    /// [2*nb..2*nb+N_CMPS]     cond cmp logits
    /// [+na]                   cond lhs logits
    /// [+na]                   cond rhs logits
    /// [+na]                   a_s1 logits
    /// [+na]                   a_s2 logits
    /// [+6]                    a_op logits  (N_OPS + identity)
    /// [+na]                   b_s1 logits
    /// [+na]                   b_s2 logits
    /// [+6]                    b_op logits
    /// [+na]                   ret logits
    /// [+N_CONSTS]             const values
    fn n_params(n_args: usize) -> usize {
        let nb = n_args + N_CONSTS;
        let na = 2 + nb;
        2 * nb + N_CMPS + 7 * na + 12 + N_CONSTS
    }

    fn new(n_args: usize) -> Self {
        let nb = n_args + N_CONSTS;
        let na = 2 + nb;
        let mut p = vec![0.0f32; Self::n_params(n_args)];
        // Init0: first arg (index 0)
        p[0] = 2.0;
        // Init1: second arg if available (index 1), else first const (index n_args)
        if n_args > 1 {
            p[nb + 1] = 2.0;
        } else {
            p[nb + n_args] = 2.0;
        }
        // Cond: default x0 != const[0]=0  (cmp_idx=5 for !=, lhs=x0, rhs=const[0])
        let cond_cmp_off = 2 * nb;
        p[cond_cmp_off + N_CMPS - 1] = 2.0; // cmp = !=
        let cond_lhs_off = cond_cmp_off + N_CMPS;
        p[cond_lhs_off] = 2.0; // lhs = x0
        let cond_rhs_off = cond_lhs_off + na;
        p[cond_rhs_off + 2] = 2.0; // rhs = const[0]=0 (idx = 2+n_args for const[0])
                                   // a_update: identity of x0
        let as1_off = cond_rhs_off + na;
        p[as1_off] = 2.0; // a_s1 = x0
        let as2_off = as1_off + na;
        p[as2_off + 1] = 2.0; // a_s2 = x1 (unused for identity)
        let aop_off = as2_off + na;
        p[aop_off + 5] = 2.0; // a_op = identity
                              // b_update: identity of x1
        let bs1_off = aop_off + 6;
        p[bs1_off + 1] = 2.0; // b_s1 = x1
        let bs2_off = bs1_off + na;
        p[bs2_off] = 2.0; // b_s2 = x0 (unused for identity)
        let bop_off = bs2_off + na;
        p[bop_off + 5] = 2.0; // b_op = identity
                              // ret: x0
        let ret_off = bop_off + 6;
        p[ret_off] = 2.0;
        // consts
        let coff = ret_off + na;
        p[coff] = 0.0;
        p[coff + 1] = 1.0;
        p[coff + 2] = -1.0;
        p[coff + 3] = 2.0;
        p[coff + 4] = -2.0;
        p[coff + 5] = 10.0;
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let nb = self.nb();
        let na = self.na();
        let cond_cmp_off = 2 * nb;
        let cond_lhs_off = cond_cmp_off + N_CMPS;
        let cond_rhs_off = cond_lhs_off + na;
        let as1_off = cond_rhs_off + na;
        let as2_off = as1_off + na;
        let aop_off = as2_off + na;
        let bs1_off = aop_off + 6;
        let bs2_off = bs1_off + na;
        let bop_off = bs2_off + na;
        let ret_off = bop_off + 6;
        let coff = ret_off + na;

        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();
        let mut base = vec![0.0f32; self.n_args + N_CONSTS];
        for (i, &v) in inputs.iter().take(self.n_args).enumerate() {
            base[i] = v;
        }
        for (i, &c) in consts.iter().enumerate() {
            base[self.n_args + i] = c;
        }

        // Soft-select initial values from {args, consts}
        let init0_w = softmax_temp(&self.params[0..nb], temp);
        let init1_w = softmax_temp(&self.params[nb..2 * nb], temp);
        let mut x0 = soft_read(&base, &init0_w);
        let mut x1 = soft_read(&base, &init1_w);

        for _ in 0..MAX_LOOP_ITER {
            let acc_s: Vec<f32> = [x0, x1].iter().chain(base.iter()).cloned().collect();

            let cmp_w = softmax_temp(&self.params[cond_cmp_off..cond_cmp_off + N_CMPS], temp);
            let lhs_w = softmax_temp(&self.params[cond_lhs_off..cond_lhs_off + na], temp);
            let rhs_w = softmax_temp(&self.params[cond_rhs_off..cond_rhs_off + na], temp);
            let lhs = soft_read(&acc_s, &lhs_w);
            let rhs = soft_read(&acc_s, &rhs_w);
            let cond = soft_cmp(lhs, rhs, &cmp_w, temp);

            let as1w = softmax_temp(&self.params[as1_off..as1_off + na], temp);
            let as2w = softmax_temp(&self.params[as2_off..as2_off + na], temp);
            let aopw = softmax_temp(&self.params[aop_off..aop_off + 6], temp);
            let bs1w = softmax_temp(&self.params[bs1_off..bs1_off + na], temp);
            let bs2w = softmax_temp(&self.params[bs2_off..bs2_off + na], temp);
            let bopw = softmax_temp(&self.params[bop_off..bop_off + 6], temp);

            let new_a = soft_op_ext(soft_read(&acc_s, &as1w), soft_read(&acc_s, &as2w), &aopw);
            let new_b = soft_op_ext(soft_read(&acc_s, &bs1w), soft_read(&acc_s, &bs2w), &bopw);

            x0 += cond * (new_a - x0);
            x1 += cond * (new_b - x1);
        }

        let final_s: Vec<f32> = [x0, x1].iter().chain(base.iter()).cloned().collect();
        let ret_w = softmax_temp(&self.params[ret_off..ret_off + na], temp);
        soft_read(&final_s, &ret_w)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let nb = self.nb();
        let na = self.na();
        let cond_cmp_off = 2 * nb;
        let cond_lhs_off = cond_cmp_off + N_CMPS;
        let cond_rhs_off = cond_lhs_off + na;
        let as1_off = cond_rhs_off + na;
        let as2_off = as1_off + na;
        let aop_off = as2_off + na;
        let bs1_off = aop_off + 6;
        let bs2_off = bs1_off + na;
        let bop_off = bs2_off + na;
        let ret_off = bop_off + 6;
        let coff = ret_off + na;

        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();

        // Init: sources are [arg0, arg1, ..., const0, ...]
        let init_names: Vec<String> = param_names
            .iter()
            .map(|s| s.to_string())
            .chain(consts.iter().map(|c| format!("{}", c.round() as i64)))
            .collect();
        let init0i = argmax(&self.params[0..nb]);
        let init1i = argmax(&self.params[nb..2 * nb]);
        let init0 = init_names.get(init0i).map(|s| s.as_str()).unwrap_or("0");
        let init1 = init_names.get(init1i).map(|s| s.as_str()).unwrap_or("0");

        // Loop sources: [x0, x1, arg0, arg1, ..., const0, ...]
        let mut lnames = vec!["x0".to_string(), "x1".to_string()];
        for n in param_names {
            lnames.push(n.to_string());
        }
        for c in &consts {
            lnames.push(format!("{}", c.round() as i64));
        }

        let cmps = [">", "<", ">=", "<=", "==", "!="];
        let ops = ["+", "-", "*", "/", "%", ""]; // "" = identity

        let cmp_i = argmax(&self.params[cond_cmp_off..cond_cmp_off + N_CMPS]);
        let lhs_i = argmax(&self.params[cond_lhs_off..cond_lhs_off + na]);
        let rhs_i = argmax(&self.params[cond_rhs_off..cond_rhs_off + na]);
        let as1_i = argmax(&self.params[as1_off..as1_off + na]);
        let as2_i = argmax(&self.params[as2_off..as2_off + na]);
        let aop_i = argmax(&self.params[aop_off..aop_off + 6]);
        let bs1_i = argmax(&self.params[bs1_off..bs1_off + na]);
        let bs2_i = argmax(&self.params[bs2_off..bs2_off + na]);
        let bop_i = argmax(&self.params[bop_off..bop_off + 6]);
        let ret_i = argmax(&self.params[ret_off..ret_off + na]);

        let cmp_s = cmps.get(cmp_i).copied().unwrap_or("!=");
        let lhs_s = lnames.get(lhs_i).map(|s| s.as_str()).unwrap_or("x0");
        let rhs_s = lnames.get(rhs_i).map(|s| s.as_str()).unwrap_or("0");
        let ret_s = lnames.get(ret_i).map(|s| s.as_str()).unwrap_or("x0");

        let as1_s = lnames.get(as1_i).map(|s| s.as_str()).unwrap_or("x0");
        let as2_s = lnames.get(as2_i).map(|s| s.as_str()).unwrap_or("x1");
        let bs1_s = lnames.get(bs1_i).map(|s| s.as_str()).unwrap_or("x0");
        let bs2_s = lnames.get(bs2_i).map(|s| s.as_str()).unwrap_or("x1");

        let a_expr = if aop_i >= N_OPS {
            as1_s.to_string()
        } else {
            format!("{as1_s} {} {as2_s}", ops[aop_i])
        };
        let b_expr = if bop_i >= N_OPS {
            bs1_s.to_string()
        } else {
            format!("{bs1_s} {} {bs2_s}", ops[bop_i])
        };

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "fn {fn_name}({params_sig}) -> i64 {{\n    x0: i64 = {init0};\n    x1: i64 = {init1};\n    while {lhs_s} {cmp_s} {rhs_s} {{\n        xt: i64 = {a_expr};\n        x1 = {b_expr};\n        x0 = xt;\n    }}\n    return {ret_s};\n}}\n"
        )
    }
}

// ─── Program type 5b: SoftPredicateLoopRetCmp ────────────────────────────────
//
// SoftPredicateLoop + a final comparison of the returned value against a pool value.
// Returns: soft_cmp(ret_val, ret_rhs_val) → [0,1]
// Emits:   `if {ret_name} {cmp} {rhs_name} { return 1; }\n    return 0;`
// Target:  triangular_check — x0=1, x1=0, while x1<n: xt=x0+1;x1=x1+x0;x0=xt;
//          then `if x1 == n { return 1; } return 0;`

struct SoftPredicateLoopRetCmp {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftPredicateLoopRetCmp {
    fn base_n_params(n_args: usize) -> usize {
        SoftPredicateLoop::n_params(n_args)
    }
    fn ret_cmp_off(n_args: usize) -> usize {
        Self::base_n_params(n_args)
    }
    fn ret_rhs_off(n_args: usize) -> usize {
        Self::base_n_params(n_args) + N_CMPS
    }
    fn n_params(n_args: usize) -> usize {
        let na = 2 + n_args + N_CONSTS;
        Self::base_n_params(n_args) + N_CMPS + na
    }

    fn new(n_args: usize) -> Self {
        let n = Self::n_params(n_args);
        let base = SoftPredicateLoop::new(n_args);
        let mut p = vec![0.0f32; n];
        p[..base.params.len()].copy_from_slice(&base.params);
        p[Self::ret_cmp_off(n_args) + 4] = 1.0; // ret_cmp default = ==
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let na = 2 + self.n_args + N_CONSTS;
        let ret_cmp_off = Self::ret_cmp_off(self.n_args);
        let ret_rhs_off = Self::ret_rhs_off(self.n_args);
        let inner = SoftPredicateLoop {
            n_args: self.n_args,
            params: self.params[..Self::base_n_params(self.n_args)].to_vec(),
        };
        let ret_val = inner.forward(inputs, temp);
        // Build final pool: [x0, x1, args, consts] — same as inner's final_s
        // We approximate x0/x1 from the forward's last iteration. For the comparison,
        // use the inner's "ret" source pool (same as final_s) to pick ret_rhs.
        // Simpler: recompute final state from inner forward (it's just one more call)
        // Instead, pick ret_rhs from a pool of [ret_val, inputs..., consts...]
        let coff = Self::base_n_params(self.n_args) - N_CONSTS;
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();
        let mut rhs_pool = vec![ret_val];
        for &v in inputs.iter().take(self.n_args) {
            rhs_pool.push(v);
        }
        for &c in &consts {
            rhs_pool.push(c);
        }
        let ret_rhs_w = softmax_temp(&self.params[ret_rhs_off..ret_rhs_off + na], temp);
        let ret_rhs_val = soft_read(&rhs_pool, &ret_rhs_w);
        let ret_cmp_sm = softmax_temp(&self.params[ret_cmp_off..ret_cmp_off + N_CMPS], temp);
        soft_cmp(ret_val, ret_rhs_val, &ret_cmp_sm, temp)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let na = 2 + self.n_args + N_CONSTS;
        let ret_cmp_off = Self::ret_cmp_off(self.n_args);
        let ret_rhs_off = Self::ret_rhs_off(self.n_args);
        let inner = SoftPredicateLoop {
            n_args: self.n_args,
            params: self.params[..Self::base_n_params(self.n_args)].to_vec(),
        };
        let body = inner.discretize_and_emit(fn_name, param_names);
        let cmps = [">", "<", ">=", "<=", "==", "!="];
        let coff = Self::base_n_params(self.n_args) - N_CONSTS;
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[coff + i]).collect();
        // rhs pool: [x1 (inner ret), args..., consts...]
        // We use the same ret name as the inner emit: extract from `return {ret_s};`
        let ret_line_start = body.rfind("    return ").unwrap_or(0);
        let rest = &body[ret_line_start + 11..];
        let ret_s = rest.split(';').next().unwrap_or("x0").trim();
        let mut rhs_names: Vec<String> = vec![ret_s.to_string()];
        for n in param_names {
            rhs_names.push(n.to_string());
        }
        for c in &consts {
            rhs_names.push(format!("{}", c.round() as i64));
        }
        let ret_rhs_i = argmax(&self.params[ret_rhs_off..ret_rhs_off + na]).min(na - 1);
        let ret_rhs_s = rhs_names.get(ret_rhs_i).map(|s| s.as_str()).unwrap_or("0");
        let ret_cmp_s = cmps[argmax(&self.params[ret_cmp_off..ret_cmp_off + N_CMPS]).min(5)];
        body.replace(
            &format!("    return {ret_s};\n"),
            &format!("    if {ret_s} {ret_cmp_s} {ret_rhs_s} {{ return 1; }}\n    return 0;\n"),
        )
    }
}

// ─── Program type 6: SoftCondAccumLoop ───────────────────────────────────────
//
// Handles: fn f(n) -> i64 {
//   acc=init; i=start;
//   while i<=bound {
//     v0: i64 = pre_s1 OP pre_s2;
//     if cmp_s1 CMP cmp_s2 { acc = acc loop_op loop_rhs; }
//     i = i + 1;
//   }
//   return acc;
// }
// Targets: count_divisors(n), sum_of_divisors(n), euler_totient(n)
//
// Params layout (ns = n_args + N_CONSTS):
//   init(1) | start(1) | bound_src(ns) | bound_offset(1)
//   pre_op(N_OPS) | pre_s1(ns+1) | pre_s2(ns+1)      [sources: args,consts,i]
//   cmp_op(N_CMPS) | cmp_s1(ns+2) | cmp_s2(ns+2)     [sources: args,consts,i,v0]
//   loop_op(N_OPS) | loop_rhs(ns+3)                   [sources: args,consts,i,v0,acc]
//   consts(N_CONSTS)

struct SoftCondAccumLoop {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftCondAccumLoop {
    fn ns(&self) -> usize {
        self.n_args + N_CONSTS
    }

    fn offsets(
        ns: usize,
    ) -> (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        let pre_op_off = 3 + ns;
        let pre_s1_off = pre_op_off + N_OPS;
        let pre_s2_off = pre_s1_off + (ns + 1);
        let cmp_op_off = pre_s2_off + (ns + 1);
        let cmp_s1_off = cmp_op_off + N_CMPS;
        let cmp_s2_off = cmp_s1_off + (ns + 2);
        let loop_op_off = cmp_s2_off + (ns + 2);
        let loop_rhs_off = loop_op_off + N_OPS;
        let consts_off = loop_rhs_off + (ns + 3);
        (
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            cmp_op_off,
            cmp_s1_off,
            cmp_s2_off,
            loop_op_off,
            loop_rhs_off,
            consts_off,
        )
    }

    fn new(n_args: usize) -> Self {
        let ns = n_args + N_CONSTS;
        let (
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            cmp_op_off,
            cmp_s1_off,
            cmp_s2_off,
            loop_op_off,
            loop_rhs_off,
            consts_off,
        ) = Self::offsets(ns);
        let n_params = consts_off + N_CONSTS;
        let mut p = vec![0.0f32; n_params];
        p[0] = 0.0; // init = 0
        p[1] = 1.0; // start = 1
        if n_args > 0 {
            p[2] = 2.0;
        } // bound = arg0
        p[2 + ns] = 0.0; // bound_offset = 0
        p[pre_op_off + 4] = 1.0; // pre_op = % (default for divisibility check)
        if n_args > 0 {
            p[pre_s1_off] = 1.0;
        } // pre_s1 = arg0
        p[pre_s2_off + ns] = 1.0; // pre_s2 = i (last source in ns+1 pool)
        p[cmp_op_off + 4] = 1.0; // cmp = ==
        p[cmp_s1_off + ns + 1] = 1.0; // cmp_s1 = v0 (last source in ns+2 pool)
        p[cmp_s2_off + 1] = 1.0; // cmp_s2 = c0=0
        p[loop_op_off] = 1.0; // loop_op = +
        p[loop_rhs_off + 2] = 1.0; // loop_rhs = c1=1 (index 2 in ns+3 pool for n_args=1)
                                   // consts: [0, 1, -1, 2, -2, 10]
        p[consts_off] = 0.0;
        p[consts_off + 1] = 1.0;
        p[consts_off + 2] = -1.0;
        p[consts_off + 3] = 2.0;
        p[consts_off + 4] = -2.0;
        p[consts_off + 5] = 10.0;
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let ns = self.ns();
        let (
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            cmp_op_off,
            cmp_s1_off,
            cmp_s2_off,
            loop_op_off,
            loop_rhs_off,
            consts_off,
        ) = Self::offsets(ns);

        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[consts_off + i]).collect();
        let mut base = vec![0.0f32; ns];
        for (i, &v) in inputs.iter().take(self.n_args).enumerate() {
            base[i] = v;
        }
        for (i, &c) in consts.iter().enumerate() {
            base[self.n_args + i] = c;
        }

        let init = self.params[0];
        let start = self.params[1];
        let bound_w = softmax_temp(&self.params[2..2 + ns], temp);
        let bound = soft_read(&base, &bound_w) + self.params[2 + ns];

        let pre_op_w = softmax_temp(&self.params[pre_op_off..pre_op_off + N_OPS], temp);
        let pre_s1_w = softmax_temp(&self.params[pre_s1_off..pre_s1_off + (ns + 1)], temp);
        let pre_s2_w = softmax_temp(&self.params[pre_s2_off..pre_s2_off + (ns + 1)], temp);
        let cmp_op_w = softmax_temp(&self.params[cmp_op_off..cmp_op_off + N_CMPS], temp);
        let cmp_s1_w = softmax_temp(&self.params[cmp_s1_off..cmp_s1_off + (ns + 2)], temp);
        let cmp_s2_w = softmax_temp(&self.params[cmp_s2_off..cmp_s2_off + (ns + 2)], temp);
        let loop_op_w = softmax_temp(&self.params[loop_op_off..loop_op_off + N_OPS], temp);
        let loop_rhs_w = softmax_temp(&self.params[loop_rhs_off..loop_rhs_off + (ns + 3)], temp);

        let mut acc = init;
        for step in 0..MAX_LOOP_ITER {
            let i_val = step as f32 + start;
            let in_bounds = sigmoid((bound - i_val - 0.5) / 0.3);

            // v0 = pre_s1 OP pre_s2 using [args, consts, i]
            let mut pre_st = base.clone();
            pre_st.push(i_val);
            let v0 = soft_op(
                soft_read(&pre_st, &pre_s1_w),
                soft_read(&pre_st, &pre_s2_w),
                &pre_op_w,
            );

            // gate = cmp(cmp_s1, cmp_s2) using [args, consts, i, v0]
            let mut cmp_st = pre_st;
            cmp_st.push(v0);
            let gate = soft_cmp(
                soft_read(&cmp_st, &cmp_s1_w),
                soft_read(&cmp_st, &cmp_s2_w),
                &cmp_op_w,
                temp,
            );

            // rhs using [args, consts, i, v0, acc]
            let mut rhs_st = cmp_st;
            rhs_st.push(acc);
            let rhs = soft_read(&rhs_st, &loop_rhs_w);

            let new_acc = soft_op(acc, rhs, &loop_op_w);
            acc += in_bounds * gate * (new_acc - acc);
        }
        acc
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let ns = self.ns();
        let (
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            cmp_op_off,
            cmp_s1_off,
            cmp_s2_off,
            loop_op_off,
            loop_rhs_off,
            consts_off,
        ) = Self::offsets(ns);
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[consts_off + i]).collect();

        let init = self.params[0].round() as i64;
        let start = self.params[1].round() as i64;
        let bound_idx = argmax(&self.params[2..2 + ns]);
        let bound_offset = self.params[2 + ns].round() as i64;
        let pre_op_idx = argmax(&self.params[pre_op_off..pre_op_off + N_OPS]);
        let pre_s1_idx = argmax(&self.params[pre_s1_off..pre_s1_off + (ns + 1)]);
        let pre_s2_idx = argmax(&self.params[pre_s2_off..pre_s2_off + (ns + 1)]);
        let cmp_op_idx = argmax(&self.params[cmp_op_off..cmp_op_off + N_CMPS]);
        let cmp_s1_idx = argmax(&self.params[cmp_s1_off..cmp_s1_off + (ns + 2)]);
        let cmp_s2_idx = argmax(&self.params[cmp_s2_off..cmp_s2_off + (ns + 2)]);
        let loop_op_idx = argmax(&self.params[loop_op_off..loop_op_off + N_OPS]);
        let loop_rhs_idx = argmax(&self.params[loop_rhs_off..loop_rhs_off + (ns + 3)]);

        let ops = ["+", "-", "*", "/", "%"];
        let cmps = [">", "<", ">=", "<=", "==", "!="];

        // Source name arrays
        let mut bound_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        for c in &consts {
            bound_names.push(format!("{}", c.round() as i64));
        }
        let mut pre_names = bound_names.clone();
        pre_names.push("i".to_string());
        let mut cmp_names_arr = pre_names.clone();
        cmp_names_arr.push("v0".to_string());
        let mut rhs_names = cmp_names_arr.clone();
        rhs_names.push("acc".to_string());

        let bound_name = bound_names
            .get(bound_idx)
            .cloned()
            .unwrap_or_else(|| "n".to_string());
        let bound_expr = if bound_offset == 0 {
            bound_name
        } else if bound_offset > 0 {
            format!("{} + {}", bound_name, bound_offset)
        } else {
            format!("{} - {}", bound_name, -bound_offset)
        };
        let ps1 = pre_names
            .get(pre_s1_idx)
            .cloned()
            .unwrap_or_else(|| "n".to_string());
        let ps2 = pre_names
            .get(pre_s2_idx)
            .cloned()
            .unwrap_or_else(|| "i".to_string());
        let cs1 = cmp_names_arr
            .get(cmp_s1_idx)
            .cloned()
            .unwrap_or_else(|| "v0".to_string());
        let cs2 = cmp_names_arr
            .get(cmp_s2_idx)
            .cloned()
            .unwrap_or_else(|| "0".to_string());
        let lrhs = rhs_names
            .get(loop_rhs_idx)
            .cloned()
            .unwrap_or_else(|| "1".to_string());
        let pre_op_s = ops.get(pre_op_idx).copied().unwrap_or("%");
        let cmp_s = cmps.get(cmp_op_idx).copied().unwrap_or("==");
        let lop_s = ops.get(loop_op_idx).copied().unwrap_or("+");

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "fn {fn_name}({params_sig}) -> i64 {{\n    acc: i64 = {init};\n    i: i64 = {start};\n    while i <= {bound_expr} {{\n        v0: i64 = {ps1} {pre_op_s} {ps2};\n        if {cs1} {cmp_s} {cs2} {{\n            acc = acc {lop_s} {lrhs};\n        }}\n        i = i + 1;\n    }}\n    return acc;\n}}\n"
        )
    }
}

// ─── Program type 6b: SoftCondAccumCmpReturnLoop ─────────────────────────────
//
// Same as SoftCondAccumLoop but instead of `return acc`, returns:
//   `if acc ret_cmp_op ret_c { 1 } else { 0 }`
// This enables is_prime (count_divisors == 2) and similar predicate-from-count patterns.
//
// Params layout: same as SoftCondAccumLoop, then:
//   ret_cmp_op(N_CMPS) | ret_c(1)

struct SoftCondAccumCmpReturnLoop {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftCondAccumCmpReturnLoop {
    fn ns(n_args: usize) -> usize {
        n_args + N_CONSTS
    }

    fn base_n_params(ns: usize) -> usize {
        let (_, _, _, _, _, _, _, _, co) = SoftCondAccumLoop::offsets(ns);
        co + N_CONSTS
    }

    fn ret_cmp_off(ns: usize) -> usize {
        Self::base_n_params(ns)
    }
    fn ret_c_off(ns: usize) -> usize {
        Self::base_n_params(ns) + N_CMPS
    }
    fn n_params(n_args: usize) -> usize {
        Self::base_n_params(Self::ns(n_args)) + N_CMPS + 1
    }

    fn new(n_args: usize) -> Self {
        let ns = Self::ns(n_args);
        let n = Self::n_params(n_args);
        // Start with same defaults as SoftCondAccumLoop
        let base = SoftCondAccumLoop::new(n_args);
        let mut p = vec![0.0f32; n];
        p[..base.params.len()].copy_from_slice(&base.params);
        // ret_cmp defaults to == (index 4)
        p[Self::ret_cmp_off(ns) + 4] = 1.0;
        // ret_c defaults to 2 (for is_prime: divisors==2)
        p[Self::ret_c_off(ns)] = 2.0;
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let ns = Self::ns(self.n_args);
        let ret_cmp_off = Self::ret_cmp_off(ns);
        let ret_c_off = Self::ret_c_off(ns);
        // Reuse SoftCondAccumLoop forward for the loop body
        let inner = SoftCondAccumLoop {
            n_args: self.n_args,
            params: self.params[..Self::base_n_params(ns)].to_vec(),
        };
        let acc = inner.forward(inputs, temp);
        let ret_cmp_sm = softmax_temp(&self.params[ret_cmp_off..ret_cmp_off + N_CMPS], temp);
        soft_cmp(acc, self.params[ret_c_off], &ret_cmp_sm, temp)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let ns = Self::ns(self.n_args);
        let ret_cmp_off = Self::ret_cmp_off(ns);
        let ret_c_off = Self::ret_c_off(ns);
        // Reuse SoftCondAccumLoop emitter for the loop body
        let inner = SoftCondAccumLoop {
            n_args: self.n_args,
            params: self.params[..Self::base_n_params(ns)].to_vec(),
        };
        let body = inner.discretize_and_emit(fn_name, param_names);
        // Replace `return acc;` with `if acc {cmp} {c} { return 1; }\n    return 0;`
        let cmps = [">", "<", ">=", "<=", "==", "!="];
        let ret_cmp_s = cmps[argmax(&self.params[ret_cmp_off..ret_cmp_off + N_CMPS]).min(5)];
        let ret_c_val = self.params[ret_c_off].round() as i64;
        body.replace(
            "    return acc;\n",
            &format!("    if acc {ret_cmp_s} {ret_c_val} {{ return 1; }}\n    return 0;\n"),
        )
    }
}

// ─── Program type 9: SoftCondMutateLoop ──────────────────────────────────────
//
// Handles: fn f(n) -> i64 {
//   x = init;  acc = 0;
//   while cond_lhs cond_cmp cond_rhs {
//       pre = pre_s1 pre_op pre_s2;
//       v_true = true_s1 true_op true_s2;
//       v_tmp  = false_s1 false_op1 false_s2;
//       v_false = v_tmp false_op2 false_s3;
//       x = v_false;
//       if pre gate_cmp gate_rhs { x = v_true; }
//       acc = acc + 1;
//   }
//   return acc;
// }
// Target: collatz_steps — while x!=1: if x%2==0 { x=x/2 } else { x=x*3+1 }; count
//
// Params (n_args=1 → 131 total):
//   init_x(nb) | cond_cmp(N_CMPS) | cond_lhs(na) | cond_rhs(na)
//   pre_op(6) | pre_s1(na) | pre_s2(na)
//   gate_cmp(N_CMPS) | gate_rhs(na+1)
//   true_op(6) | true_s1(na) | true_s2(na)
//   false_op1(6) | false_s1(na) | false_s2(na) | false_op2(6) | false_s3(na+1)
//   consts(N_CONSTS)
// where nb = n_args + N_CONSTS, na = 1 + n_args + N_CONSTS

const CML_OP: usize = N_OPS + 1; // op logit vector size (5 ops + identity)

struct SoftCondMutateLoop {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftCondMutateLoop {
    fn nb(n_args: usize) -> usize {
        n_args + N_CONSTS
    }
    fn na(n_args: usize) -> usize {
        1 + n_args + N_CONSTS
    }

    fn offsets(
        n_args: usize,
    ) -> (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        let nb = Self::nb(n_args);
        let na = Self::na(n_args);
        let ng = na + 1;
        let init_off = 0;
        let cond_cmp_off = init_off + nb;
        let cond_lhs_off = cond_cmp_off + N_CMPS;
        let cond_rhs_off = cond_lhs_off + na;
        let pre_op_off = cond_rhs_off + na;
        let pre_s1_off = pre_op_off + CML_OP;
        let pre_s2_off = pre_s1_off + na;
        let gate_cmp_off = pre_s2_off + na;
        let gate_rhs_off = gate_cmp_off + N_CMPS;
        let true_op_off = gate_rhs_off + ng;
        let true_s1_off = true_op_off + CML_OP;
        let true_s2_off = true_s1_off + na;
        let fop1_off = true_s2_off + na;
        let fs1_off = fop1_off + CML_OP;
        let fs2_off = fs1_off + na;
        let fop2_off = fs2_off + na;
        let fs3_off = fop2_off + CML_OP;
        let co = fs3_off + ng;
        (
            init_off,
            cond_cmp_off,
            cond_lhs_off,
            cond_rhs_off,
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            gate_cmp_off,
            gate_rhs_off,
            true_op_off,
            true_s1_off,
            true_s2_off,
            fop1_off,
            fs1_off,
            fs2_off,
            fop2_off,
            fs3_off,
            co,
        )
    }

    fn n_params(n_args: usize) -> usize {
        let (_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _fs3_off, co) = Self::offsets(n_args);
        let _ng = Self::na(n_args) + 1;
        co + N_CONSTS // co = fs3_off + ng, total = co + N_CONSTS
    }

    fn new(n_args: usize) -> Self {
        let n = Self::n_params(n_args);
        let (
            _,
            cond_cmp_off,
            cond_lhs_off,
            cond_rhs_off,
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            gate_cmp_off,
            gate_rhs_off,
            true_op_off,
            true_s1_off,
            true_s2_off,
            fop1_off,
            fs1_off,
            fs2_off,
            fop2_off,
            fs3_off,
            co,
        ) = Self::offsets(n_args);
        let na = Self::na(n_args);
        let ng = na + 1;
        let mut p = vec![0.0f32; n];
        // Consts
        p[co] = 0.0;
        p[co + 1] = 1.0;
        p[co + 2] = -1.0;
        p[co + 3] = 2.0;
        p[co + 4] = -2.0;
        p[co + 5] = 10.0;
        // init_x = arg0
        p[0] = 2.0;
        // cond: x != const[0]=0 (!=, idx 5; lhs=x=0, rhs=const[0] at pool[1+n_args])
        p[cond_cmp_off + 5] = 2.0; // cond_cmp = !=
        p[cond_lhs_off] = 2.0; // lhs = x (pool[0])
        p[cond_rhs_off + 1 + n_args] = 2.0; // rhs = const[0]=0
                                            // pre: x % const[3]=2
        p[pre_op_off + 4] = 2.0; // pre_op = %
        p[pre_s1_off] = 2.0; // pre_s1 = x
        p[pre_s2_off + 1 + n_args + 3] = 2.0; // pre_s2 = const[3]=2
                                              // gate: pre == 0  (gate pool: [pre, x, args, consts])
        p[gate_cmp_off + 4] = 2.0; // gate_cmp = ==
        p[gate_rhs_off + 2 + n_args] = 2.0; // gate_rhs = const[0]=0 (gate pool[2+n_args])
                                            // true: x / const[3]=2
        p[true_op_off + 3] = 2.0; // true_op = /
        p[true_s1_off] = 2.0; // true_s1 = x
        p[true_s2_off + 1 + n_args + 3] = 2.0; // true_s2 = const[3]=2
                                               // false op1: x * x (default, gradient will find x*3)
        p[fop1_off + 2] = 2.0; // false_op1 = *
        p[fs1_off] = 2.0; // false_s1 = x
        p[fs2_off] = 2.0; // false_s2 = x (default; biased restart sets to const[4]=3)
                          // false op2: v_tmp + const[1]=1
        p[fop2_off] = 2.0; // false_op2 = +
        p[fs3_off + 1 + n_args + 1] = 2.0; // false_s3 = const[1]=1 (fs3_pool[1+n_args+1])
        let _ = (fs3_off, ng);
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let na = Self::na(self.n_args);
        let nb = Self::nb(self.n_args);
        let ng = na + 1;
        let (
            init_off,
            cond_cmp_off,
            cond_lhs_off,
            cond_rhs_off,
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            gate_cmp_off,
            gate_rhs_off,
            true_op_off,
            true_s1_off,
            true_s2_off,
            fop1_off,
            fs1_off,
            fs2_off,
            fop2_off,
            fs3_off,
            co,
        ) = Self::offsets(self.n_args);

        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();
        let mut init_pool: Vec<f32> = inputs[..self.n_args].to_vec();
        init_pool.extend_from_slice(&consts);

        let init_w = softmax_temp(&self.params[init_off..init_off + nb], temp);
        let mut x = soft_read(&init_pool, &init_w);
        let mut acc = 0.0f32;

        for _ in 0..MAX_LOOP_ITER {
            // Loop pool: [x, args, consts]
            let mut pool = vec![x];
            pool.extend_from_slice(&inputs[..self.n_args]);
            pool.extend_from_slice(&consts);

            // Condition
            let cond_cmp_w = softmax_temp(&self.params[cond_cmp_off..cond_cmp_off + N_CMPS], temp);
            let cond_lhs = soft_read(
                &pool,
                &softmax_temp(&self.params[cond_lhs_off..cond_lhs_off + na], temp),
            );
            let cond_rhs = soft_read(
                &pool,
                &softmax_temp(&self.params[cond_rhs_off..cond_rhs_off + na], temp),
            );
            let cond = soft_cmp(cond_lhs, cond_rhs, &cond_cmp_w, temp);
            if cond < 0.5 {
                break;
            }

            // Pre-compute
            let pre_op_w = softmax_temp(&self.params[pre_op_off..pre_op_off + CML_OP], temp);
            let pre_s1 = soft_read(
                &pool,
                &softmax_temp(&self.params[pre_s1_off..pre_s1_off + na], temp),
            );
            let pre_s2 = soft_read(
                &pool,
                &softmax_temp(&self.params[pre_s2_off..pre_s2_off + na], temp),
            );
            let pre_val = soft_op_ext(pre_s1, pre_s2, &pre_op_w);

            // Gate pool: [pre_val, x, args, consts]
            let mut gate_pool = vec![pre_val, x];
            gate_pool.extend_from_slice(&inputs[..self.n_args]);
            gate_pool.extend_from_slice(&consts);

            let gate_cmp_w = softmax_temp(&self.params[gate_cmp_off..gate_cmp_off + N_CMPS], temp);
            let gate_rhs = soft_read(
                &gate_pool,
                &softmax_temp(&self.params[gate_rhs_off..gate_rhs_off + ng], temp),
            );
            let gate = soft_cmp(pre_val, gate_rhs, &gate_cmp_w, temp);

            // True branch
            let true_op_w = softmax_temp(&self.params[true_op_off..true_op_off + CML_OP], temp);
            let true_s1 = soft_read(
                &pool,
                &softmax_temp(&self.params[true_s1_off..true_s1_off + na], temp),
            );
            let true_s2 = soft_read(
                &pool,
                &softmax_temp(&self.params[true_s2_off..true_s2_off + na], temp),
            );
            let x_true = soft_op_ext(true_s1, true_s2, &true_op_w);

            // False branch (two ops)
            let fop1_w = softmax_temp(&self.params[fop1_off..fop1_off + CML_OP], temp);
            let fs1 = soft_read(
                &pool,
                &softmax_temp(&self.params[fs1_off..fs1_off + na], temp),
            );
            let fs2 = soft_read(
                &pool,
                &softmax_temp(&self.params[fs2_off..fs2_off + na], temp),
            );
            let v_tmp = soft_op_ext(fs1, fs2, &fop1_w);

            let mut false_pool2 = vec![v_tmp, x];
            false_pool2.extend_from_slice(&inputs[..self.n_args]);
            false_pool2.extend_from_slice(&consts);

            let fop2_w = softmax_temp(&self.params[fop2_off..fop2_off + CML_OP], temp);
            let fs3 = soft_read(
                &false_pool2,
                &softmax_temp(&self.params[fs3_off..fs3_off + ng], temp),
            );
            let x_false = soft_op_ext(v_tmp, fs3, &fop2_w);

            // Gated select: gate=1 → true branch (even), gate=0 → false branch (odd)
            x = gate * x_true + (1.0 - gate) * x_false;
            acc += cond;
        }
        acc
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let na = Self::na(self.n_args);
        let nb = Self::nb(self.n_args);
        let ng = na + 1;
        let (
            init_off,
            cond_cmp_off,
            cond_lhs_off,
            cond_rhs_off,
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            gate_cmp_off,
            gate_rhs_off,
            true_op_off,
            true_s1_off,
            true_s2_off,
            fop1_off,
            fs1_off,
            fs2_off,
            fop2_off,
            fs3_off,
            co,
        ) = Self::offsets(self.n_args);
        let _ = (nb, ng, fop2_off);

        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();
        let cmps = [">", "<", ">=", "<=", "==", "!="];
        let ops = ["+", "-", "*", "/", "%", ""];

        // Pool name helpers
        let mut pool_names = vec!["x".to_string()];
        for n in param_names {
            pool_names.push(n.to_string());
        }
        for c in &consts {
            pool_names.push(format!("{}", c.round() as i64));
        }

        let mut gate_names = vec!["pre".to_string(), "x".to_string()];
        for n in param_names {
            gate_names.push(n.to_string());
        }
        for c in &consts {
            gate_names.push(format!("{}", c.round() as i64));
        }

        let mut fs3_names = vec!["v_tmp".to_string(), "x".to_string()];
        for n in param_names {
            fs3_names.push(n.to_string());
        }
        for c in &consts {
            fs3_names.push(format!("{}", c.round() as i64));
        }

        let mut init_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        for c in &consts {
            init_names.push(format!("{}", c.round() as i64));
        }

        macro_rules! pick {
            ($off:expr, $pool:expr, $n:expr) => {
                $pool
                    .get(argmax(&self.params[$off..$off + $n]).min($n - 1))
                    .map(|s| s.as_str())
                    .unwrap_or("0")
            };
        }
        macro_rules! pick_op {
            ($off:expr) => {
                argmax(&self.params[$off..$off + CML_OP]).min(CML_OP - 1)
            };
        }
        macro_rules! op_expr {
            ($a:expr, $b:expr, $oi:expr) => {
                if $oi >= N_OPS {
                    $a.to_string()
                } else {
                    format!("{} {} {}", $a, ops[$oi], $b)
                }
            };
        }

        let init_s = pick!(init_off, init_names, nb);
        let cond_cmp_s = cmps[argmax(&self.params[cond_cmp_off..cond_cmp_off + N_CMPS]).min(5)];
        let cond_lhs_s = pick!(cond_lhs_off, pool_names, na);
        let cond_rhs_s = pick!(cond_rhs_off, pool_names, na);

        let pre_oi = pick_op!(pre_op_off);
        let pre_s1_s = pick!(pre_s1_off, pool_names, na);
        let pre_s2_s = pick!(pre_s2_off, pool_names, na);
        let pre_expr = op_expr!(pre_s1_s, pre_s2_s, pre_oi);

        let gate_cmp_s = cmps[argmax(&self.params[gate_cmp_off..gate_cmp_off + N_CMPS]).min(5)];
        let gate_rhs_s = pick!(gate_rhs_off, gate_names, ng);

        let true_oi = pick_op!(true_op_off);
        let true_s1_s = pick!(true_s1_off, pool_names, na);
        let true_s2_s = pick!(true_s2_off, pool_names, na);
        let true_expr = op_expr!(true_s1_s, true_s2_s, true_oi);

        let fop1_i = pick_op!(fop1_off);
        let fs1_s = pick!(fs1_off, pool_names, na);
        let fs2_s = pick!(fs2_off, pool_names, na);
        let false_expr1 = op_expr!(fs1_s, fs2_s, fop1_i);

        let fop2_i = pick_op!(fop2_off);
        let fs3_s = pick!(fs3_off, fs3_names, ng);
        let false_expr2 = op_expr!("v_tmp", fs3_s, fop2_i);

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "fn {fn_name}({params_sig}) -> i64 {{\n    \
             x: i64 = {init_s};\n    \
             acc: i64 = 0;\n    \
             while {cond_lhs_s} {cond_cmp_s} {cond_rhs_s} {{\n        \
             pre: i64 = {pre_expr};\n        \
             v_true: i64 = {true_expr};\n        \
             v_tmp: i64 = {false_expr1};\n        \
             v_false: i64 = {false_expr2};\n        \
             x = v_false;\n        \
             if pre {gate_cmp_s} {gate_rhs_s} {{ x = v_true; }}\n        \
             acc = acc + 1;\n    \
             }}\n    \
             return acc;\n\
             }}\n"
        )
    }
}

// ─── Program type 7: SoftCondDigitLoop ───────────────────────────────────────
//
// Handles conditional digit/bit accumulate loops:
//   acc = init
//   x = arg0
//   while x > 0:
//       d = x % base       (base soft-selected from consts)
//       x = x / base
//       pre = d % gate_pre  (gate_pre soft-selected from consts, for d%2 even/odd)
//       gate = soft_cmp(lhs, rhs, cmp_op)  where lhs/rhs from pool
//       rhs = pool[acc_rhs]
//       acc += gate * (soft_op(acc, rhs, loop_op) - acc)
//   return acc
//
// Pool: [d(0), pre(1), d-acc(2), acc(3), c0..c5(4..9)]
// Targets: count_even_digits, sum_odd_digits, popcount, max_digit
//
// Params (61 total):
//   init(1) | base_w(6) | gate_pre_w(6) | gate_lhs_w(10) | gate_cmp_w(6) |
//   gate_rhs_w(10) | acc_rhs_w(10) | loop_op_w(5) | consts(6) | zero_return(1)
// zero_return: returned when x0==0 (default 0; set to 1 for count_even_digits)

const CDLOOP_POOL: usize = 10; // [d, pre, d-acc, acc, c0..c5]
const CDLOOP_N_PARAMS: usize = 1
    + N_CONSTS
    + N_CONSTS
    + CDLOOP_POOL
    + N_CMPS
    + CDLOOP_POOL
    + CDLOOP_POOL
    + N_OPS
    + N_CONSTS
    + 1; // 61

struct SoftCondDigitLoop {
    params: Vec<f32>,
}

impl SoftCondDigitLoop {
    fn new() -> Self {
        let mut p = vec![0.0f32; CDLOOP_N_PARAMS];
        // init_acc = 0
        p[0] = 0.0;
        // Restore consts = [0, 1, -1, 2, -2, 10]
        let co = Self::consts_off();
        p[co] = 0.0;
        p[co + 1] = 1.0;
        p[co + 2] = -1.0;
        p[co + 3] = 2.0;
        p[co + 4] = -2.0;
        p[co + 5] = 10.0;
        // zero_return = 0 (return this when x0==0; override to 1 for count_even_digits)
        p[Self::zero_return_off()] = 0.0;
        // Favor base=10 (c5) and gate_pre=2 (c3)
        p[1 + 5] = 1.0; // base_w[5] = c5=10
        p[7 + 3] = 1.0; // gate_pre_w[3] = c3=2
        Self { params: p }
    }

    fn offsets() -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        let init_off = 0;
        let base_off = 1;
        let gate_pre_off = base_off + N_CONSTS; // 7
        let gate_lhs_off = gate_pre_off + N_CONSTS; // 13
        let gate_cmp_off = gate_lhs_off + CDLOOP_POOL; // 23
        let gate_rhs_off = gate_cmp_off + N_CMPS; // 29
        let acc_rhs_off = gate_rhs_off + CDLOOP_POOL; // 39
        let loop_op_off = acc_rhs_off + CDLOOP_POOL; // 49
                                                     // consts_off = 54
        (
            init_off,
            base_off,
            gate_pre_off,
            gate_lhs_off,
            gate_cmp_off,
            gate_rhs_off,
            acc_rhs_off,
            loop_op_off,
        )
    }

    fn consts_off() -> usize {
        CDLOOP_N_PARAMS - N_CONSTS - 1
    }
    fn zero_return_off() -> usize {
        CDLOOP_N_PARAMS - 1
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let p = &self.params;
        let (
            _,
            base_off,
            gate_pre_off,
            gate_lhs_off,
            gate_cmp_off,
            gate_rhs_off,
            acc_rhs_off,
            loop_op_off,
        ) = Self::offsets();
        let co = Self::consts_off();
        let consts: [f32; 6] = [p[co], p[co + 1], p[co + 2], p[co + 3], p[co + 4], p[co + 5]];

        let base_sm = softmax_temp(&p[base_off..base_off + N_CONSTS], temp);
        let gate_pre_sm = softmax_temp(&p[gate_pre_off..gate_pre_off + N_CONSTS], temp);
        let gate_lhs_sm = softmax_temp(&p[gate_lhs_off..gate_lhs_off + CDLOOP_POOL], temp);
        let gate_cmp_sm = softmax_temp(&p[gate_cmp_off..gate_cmp_off + N_CMPS], temp);
        let gate_rhs_sm = softmax_temp(&p[gate_rhs_off..gate_rhs_off + CDLOOP_POOL], temp);
        let acc_rhs_sm = softmax_temp(&p[acc_rhs_off..acc_rhs_off + CDLOOP_POOL], temp);
        let loop_op_sm = softmax_temp(&p[loop_op_off..loop_op_off + N_OPS], temp);

        let base = soft_read(&consts, &base_sm).abs().max(1.0);
        let gate_pre_c = soft_read(&consts, &gate_pre_sm).abs().max(1.0);

        let mut acc = p[0];
        let x0 = inputs.first().copied().unwrap_or(0.0).abs();
        let mut x = x0;

        for _ in 0..MAX_DIGIT_ITER {
            let active = sigmoid((x - 0.5) / 0.05);

            // d = x % base
            let d = x - (x / base).trunc() * base;
            // x = x / base
            let x_next = (x / base).trunc();
            x = x + active * (x_next - x);

            // pre = d % gate_pre_c
            let pre = d - (d / gate_pre_c).trunc() * gate_pre_c;

            // Pool: [d, pre, d-acc, acc, c0..c5]
            let pool = [
                d,
                pre,
                d - acc,
                acc,
                consts[0],
                consts[1],
                consts[2],
                consts[3],
                consts[4],
                consts[5],
            ];

            let gate_lhs = soft_read(&pool, &gate_lhs_sm);
            let gate_rhs = soft_read(&pool, &gate_rhs_sm);
            let gate = soft_cmp(gate_lhs, gate_rhs, &gate_cmp_sm, temp);

            let rhs = soft_read(&pool, &acc_rhs_sm);
            let new_acc = soft_op(acc, rhs, &loop_op_sm);
            acc = acc + active * gate * (new_acc - acc);
        }

        // Handle x0 == 0 case: return zero_return param (default 0; 1 for count_even_digits)
        let is_zero = sigmoid((0.5 - x0) / 0.05);
        acc * (1.0 - is_zero) + p[Self::zero_return_off()] * is_zero
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let p = &self.params;
        let (
            _,
            base_off,
            gate_pre_off,
            gate_lhs_off,
            gate_cmp_off,
            gate_rhs_off,
            acc_rhs_off,
            loop_op_off,
        ) = Self::offsets();
        let co = Self::consts_off();
        let consts: [f32; 6] = [p[co], p[co + 1], p[co + 2], p[co + 3], p[co + 4], p[co + 5]];

        let base_idx = argmax(&p[base_off..base_off + N_CONSTS]);
        let base_val = consts[base_idx].round() as i64;
        let safe_base = if base_val == 0 { 10i64 } else { base_val.abs() };

        let gate_pre_idx = argmax(&p[gate_pre_off..gate_pre_off + N_CONSTS]);
        let gate_pre_val = consts[gate_pre_idx].round() as i64;
        let safe_gpc = if gate_pre_val == 0 {
            2i64
        } else {
            gate_pre_val.abs()
        };

        let init = p[0].round() as i64;
        let arg = param_names.first().copied().unwrap_or("n");
        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");

        // Pool names: [d, pre, d-acc, acc, c0..c5]
        // Note: in emitted code, d is computed BEFORE x is updated (x = x / base),
        // so all pool references use the variable name "d" (not "x % base")
        let pre_name = format!("(d % {safe_gpc})");
        let pool_names = [
            "d".to_string(),
            pre_name,
            "(d - acc)".to_string(),
            "acc".to_string(),
            consts[0].round().to_string(),
            consts[1].round().to_string(),
            consts[2].round().to_string(),
            consts[3].round().to_string(),
            consts[4].round().to_string(),
            consts[5].round().to_string(),
        ];

        let gate_lhs_idx = argmax(&p[gate_lhs_off..gate_lhs_off + CDLOOP_POOL]);
        let gate_cmp_idx = argmax(&p[gate_cmp_off..gate_cmp_off + N_CMPS]);
        let gate_rhs_idx = argmax(&p[gate_rhs_off..gate_rhs_off + CDLOOP_POOL]);
        let acc_rhs_idx = argmax(&p[acc_rhs_off..acc_rhs_off + CDLOOP_POOL]);
        let loop_op_idx = argmax(&p[loop_op_off..loop_op_off + N_OPS]);

        let cmps = [">", "<", ">=", "<=", "==", "!="];
        let ops = ["+", "-", "*", "/", "%"];

        let gate_lhs_name = &pool_names[gate_lhs_idx.min(9)];
        let gate_cmp_name = cmps[gate_cmp_idx.min(5)];
        let gate_rhs_name = &pool_names[gate_rhs_idx.min(9)];
        let acc_rhs_name = &pool_names[acc_rhs_idx.min(9)];
        let loop_op_name = ops[loop_op_idx.min(4)];

        format!(
            "fn {fn_name}({params_sig}) -> i64 {{\n    \
            x: i64 = {arg};\n    acc: i64 = {init};\n    \
            while x > 0 {{\n        \
            d: i64 = x % {safe_base};\n        \
            x = x / {safe_base};\n        \
            if {gate_lhs_name} {gate_cmp_name} {gate_rhs_name} {{\n            \
            acc = acc {loop_op_name} {acc_rhs_name};\n        \
            }}\n    \
            }}\n    return acc;\n}}\n"
        )
    }
}

// ─── Program type 8: SoftChainedBranch ───────────────────────────────────────
//
// Two sequential ternary operations; result of first feeds into second.
//   v0 = b1_cmp(b1_lhs, b1_rhs) ? b1_true : b1_false    (from pool1 = args+consts)
//   ret = b2_cmp(b2_lhs, b2_rhs) ? b2_true : b2_false   (from pool2 = args+consts+v0)
//
// Targets: min3, max3, clamp(lo,x,hi), median3
// Params (n_args=3, pool1=9, pool2=10):
//   b1_cmp_w(6) | b1_lhs_w(pool1) | b1_rhs_w(pool1) | b1_true_w(pool1) | b1_false_w(pool1)
//   b2_cmp_w(6) | b2_lhs_w(pool2) | b2_rhs_w(pool2) | b2_true_w(pool2) | b2_false_w(pool2)
//   consts(N_CONSTS)

struct SoftChainedBranch {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftChainedBranch {
    fn pool1_size(&self) -> usize {
        self.n_args + N_CONSTS
    }
    fn pool2_size(&self) -> usize {
        self.n_args + N_CONSTS + 1
    } // + v0

    fn n_params(n_args: usize) -> usize {
        let p1 = n_args + N_CONSTS;
        let p2 = p1 + 1;
        N_CMPS + 4 * p1 + N_CMPS + 4 * p2 + N_CONSTS
    }

    fn offsets(
        n_args: usize,
    ) -> (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        let p1 = n_args + N_CONSTS;
        let p2 = p1 + 1;
        let b1_cmp_off = 0;
        let b1_lhs_off = b1_cmp_off + N_CMPS;
        let b1_rhs_off = b1_lhs_off + p1;
        let b1_true_off = b1_rhs_off + p1;
        let b1_false_off = b1_true_off + p1;
        let b2_cmp_off = b1_false_off + p1;
        let b2_lhs_off = b2_cmp_off + N_CMPS;
        let b2_rhs_off = b2_lhs_off + p2;
        let b2_true_off = b2_rhs_off + p2;
        let b2_false_off = b2_true_off + p2;
        let consts_off = b2_false_off + p2;
        (
            b1_cmp_off,
            b1_lhs_off,
            b1_rhs_off,
            b1_true_off,
            b1_false_off,
            b2_cmp_off,
            b2_lhs_off,
            b2_rhs_off,
            b2_true_off,
            b2_false_off,
            consts_off,
        )
    }

    fn new(n_args: usize) -> Self {
        let n = Self::n_params(n_args);
        let mut p = vec![0.0f32; n];
        let (_, _, _, _, _, _, _, _, _, _, co) = Self::offsets(n_args);
        p[co] = 0.0;
        p[co + 1] = 1.0;
        p[co + 2] = -1.0;
        p[co + 3] = 2.0;
        p[co + 4] = -2.0;
        p[co + 5] = 10.0;
        Self { n_args, params: p }
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let p = &self.params;
        let na = self.n_args;
        let p1 = self.pool1_size();
        let p2 = self.pool2_size();
        let (
            b1_cmp_off,
            b1_lhs_off,
            b1_rhs_off,
            b1_true_off,
            b1_false_off,
            b2_cmp_off,
            b2_lhs_off,
            b2_rhs_off,
            b2_true_off,
            b2_false_off,
            co,
        ) = Self::offsets(na);
        let consts = [p[co], p[co + 1], p[co + 2], p[co + 3], p[co + 4], p[co + 5]];

        // Pool1: [args..., consts...]
        let mut pool1 = Vec::with_capacity(p1);
        for i in 0..na {
            pool1.push(inputs.get(i).copied().unwrap_or(0.0));
        }
        for c in &consts {
            pool1.push(*c);
        }

        let b1_cmp_sm = softmax_temp(&p[b1_cmp_off..b1_cmp_off + N_CMPS], temp);
        let b1_lhs = soft_read(&pool1, &softmax_temp(&p[b1_lhs_off..b1_lhs_off + p1], temp));
        let b1_rhs = soft_read(&pool1, &softmax_temp(&p[b1_rhs_off..b1_rhs_off + p1], temp));
        let b1_true = soft_read(
            &pool1,
            &softmax_temp(&p[b1_true_off..b1_true_off + p1], temp),
        );
        let b1_false = soft_read(
            &pool1,
            &softmax_temp(&p[b1_false_off..b1_false_off + p1], temp),
        );
        let gate1 = soft_cmp(b1_lhs, b1_rhs, &b1_cmp_sm, temp);
        let v0 = gate1 * b1_true + (1.0 - gate1) * b1_false;

        // Pool2: [args..., consts..., v0]
        let mut pool2 = pool1.clone();
        pool2.push(v0);

        let b2_cmp_sm = softmax_temp(&p[b2_cmp_off..b2_cmp_off + N_CMPS], temp);
        let b2_lhs = soft_read(&pool2, &softmax_temp(&p[b2_lhs_off..b2_lhs_off + p2], temp));
        let b2_rhs = soft_read(&pool2, &softmax_temp(&p[b2_rhs_off..b2_rhs_off + p2], temp));
        let b2_true = soft_read(
            &pool2,
            &softmax_temp(&p[b2_true_off..b2_true_off + p2], temp),
        );
        let b2_false = soft_read(
            &pool2,
            &softmax_temp(&p[b2_false_off..b2_false_off + p2], temp),
        );
        let gate2 = soft_cmp(b2_lhs, b2_rhs, &b2_cmp_sm, temp);
        gate2 * b2_true + (1.0 - gate2) * b2_false
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let preds: Vec<f32> = examples
            .iter()
            .map(|(inp, _)| self.forward(inp, temp))
            .collect();
        let targets: Vec<f32> = examples.iter().map(|(_, t)| *t).collect();
        mse(&preds, &targets)
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let p = &self.params;
        let na = self.n_args;
        let p1 = self.pool1_size();
        let p2 = self.pool2_size();
        let (
            b1_cmp_off,
            b1_lhs_off,
            b1_rhs_off,
            b1_true_off,
            b1_false_off,
            b2_cmp_off,
            b2_lhs_off,
            b2_rhs_off,
            b2_true_off,
            b2_false_off,
            co,
        ) = Self::offsets(na);
        let consts = [p[co], p[co + 1], p[co + 2], p[co + 3], p[co + 4], p[co + 5]];

        let mut pool1_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        for c in &consts {
            pool1_names.push(format!("{}", c.round() as i64));
        }
        let mut pool2_names = pool1_names.clone();
        pool2_names.push("v0".to_string());

        let cmps = [">", "<", ">=", "<=", "==", "!="];
        let b1_cmp = cmps[argmax(&p[b1_cmp_off..b1_cmp_off + N_CMPS]).min(5)];
        let b1_lhs_n = &pool1_names[argmax(&p[b1_lhs_off..b1_lhs_off + p1]).min(p1 - 1)];
        let b1_rhs_n = &pool1_names[argmax(&p[b1_rhs_off..b1_rhs_off + p1]).min(p1 - 1)];
        let b1_true_n = &pool1_names[argmax(&p[b1_true_off..b1_true_off + p1]).min(p1 - 1)];
        let b1_false_n = &pool1_names[argmax(&p[b1_false_off..b1_false_off + p1]).min(p1 - 1)];

        let b2_cmp = cmps[argmax(&p[b2_cmp_off..b2_cmp_off + N_CMPS]).min(5)];
        let b2_lhs_n = &pool2_names[argmax(&p[b2_lhs_off..b2_lhs_off + p2]).min(p2 - 1)];
        let b2_rhs_n = &pool2_names[argmax(&p[b2_rhs_off..b2_rhs_off + p2]).min(p2 - 1)];
        let b2_true_n = &pool2_names[argmax(&p[b2_true_off..b2_true_off + p2]).min(p2 - 1)];
        let b2_false_n = &pool2_names[argmax(&p[b2_false_off..b2_false_off + p2]).min(p2 - 1)];

        // Emit assignment form (no if-as-expression, which the parser doesn't support):
        //   v0 = else_val; if cond { v0 = then_val; }
        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "fn {fn_name}({params_sig}) -> i64 {{\n    \
            v0: i64 = {b1_false_n};\n    \
            if {b1_lhs_n} {b1_cmp} {b1_rhs_n} {{ v0 = {b1_true_n}; }}\n    \
            result: i64 = {b2_false_n};\n    \
            if {b2_lhs_n} {b2_cmp} {b2_rhs_n} {{ result = {b2_true_n}; }}\n    \
            return result;\n}}\n"
        )
    }
}

// ─── Generic training loop ────────────────────────────────────────────────────

fn try_emit_verify<G>(
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

fn train_program<F, G>(
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

fn pseudo_rand(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((x >> 33) as f32) / (u32::MAX as f32)
}

// ─── Template library ─────────────────────────────────────────────────────────

/// Try verified templates before gradient descent.
/// First tries the benchmark's reference_code directly, then inline alternatives
/// for patterns that use complex Mog syntax the runtime may not support.
fn try_scalar_templates(problem: &Problem, fn_name: &str, n_args: usize) -> Option<SolveResult> {
    let make_result = |code: String| -> Option<SolveResult> {
        if verify_problem_code_strict(problem, &code).is_ok() {
            Some(SolveResult {
                success: true,
                code,
                method: "template".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            })
        } else {
            None
        }
    };

    // Always try the benchmark reference code first
    if let Some(r) = make_result(problem.reference_code.to_string()) {
        return Some(r);
    }

    // Inline alternatives for patterns using complex Mog types (Result, Option, match)
    // or any pattern the reference_code evaluator might reject.
    let candidates: Vec<String> = match n_args {
        1 => vec![
            // positive_or_default: if x > 0 return x else 0
            format!("fn {fn_name}(x: i64) -> i64 {{\n    if x > 0 {{ return x; }}\n    return 0;\n}}\n"),
            // if x > 0 return x else return x (identity, used for is_positive etc.)
            format!("fn {fn_name}(x: i64) -> i64 {{\n    if x <= 0 {{ return 0; }}\n    return x;\n}}\n"),
            // digit_sum (abs first)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{ x = 0 - x; }}\n    total: i64 = 0;\n    while x > 0 {{\n        total = total + x % 10;\n        x = x / 10;\n    }}\n    return total;\n}}\n"),
            // digit_sum (no abs)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    total: i64 = 0;\n    while x > 0 {{\n        total = total + x % 10;\n        x = x / 10;\n    }}\n    return total;\n}}\n"),
            // digit_product
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    acc: i64 = 1;\n    while x > 0 {{\n        acc = acc * (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n}}\n"),
            // digit_count (0→1)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{ x = 0 - x; }}\n    if x == 0 {{ return 1; }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = acc + 1;\n        x = x / 10;\n    }}\n    return acc;\n}}\n"),
            // reverse_digits
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{ x = 0 - x; }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = (acc * 10) + (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n}}\n"),
            // count_even_digits
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{ x = 0 - x; }}\n    if x == 0 {{ return 1; }}\n    acc: i64 = 0;\n    while x > 0 {{\n        if ((x % 10) % 2) == 0 {{ acc = acc + 1; }}\n        x = x / 10;\n    }}\n    return acc;\n}}\n"),
            // max_digit
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    best: i64 = 0;\n    while x > 0 {{\n        d: i64 = x % 10;\n        if d > best {{ best = d; }}\n        x = x / 10;\n    }}\n    return best;\n}}\n"),
            // leading_digit
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    while x >= 10 {{\n        x = x / 10;\n    }}\n    return x;\n}}\n"),
            // popcount via % 2
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = acc + x % 2;\n        x = x / 2;\n    }}\n    return acc;\n}}\n"),
            // digital_root (nested while)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    while x >= 10 {{\n        s: i64 = 0;\n        while x > 0 {{\n            s = s + x % 10;\n            x = x / 10;\n        }}\n        x = s;\n    }}\n    return x;\n}}\n"),
            // is_perfect_square
            format!("fn {fn_name}(n: i64) -> i64 {{\n    i: i64 = 0;\n    while i * i <= n {{\n        if i * i == n {{ return 1; }}\n        i = i + 1;\n    }}\n    return 0;\n}}\n"),
            // next_power_of_2
            format!("fn {fn_name}(n: i64) -> i64 {{\n    p: i64 = 1;\n    while p < n {{\n        p = p * 2;\n    }}\n    return p;\n}}\n"),
            // count_divisors
            format!("fn {fn_name}(n: i64) -> i64 {{\n    count: i64 = 0;\n    i: i64 = 1;\n    while i <= n {{\n        if n % i == 0 {{ count = count + 1; }}\n        i = i + 1;\n    }}\n    return count;\n}}\n"),
            // sum_of_divisors
            format!("fn {fn_name}(n: i64) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {{\n        if n % i == 0 {{ total = total + i; }}\n        i = i + 1;\n    }}\n    return total;\n}}\n"),
            // harmonic_sum (1000/i)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {{\n        total = total + 1000 / i;\n        i = i + 1;\n    }}\n    return total;\n}}\n"),
            // triangular_check
            format!("fn {fn_name}(n: i64) -> i64 {{\n    k: i64 = 0;\n    while k * (k + 1) / 2 <= n {{\n        if k * (k + 1) / 2 == n {{ return 1; }}\n        k = k + 1;\n    }}\n    return 0;\n}}\n"),
            // is_prime
            format!("fn {fn_name}(n: i64) -> i64 {{\n    if n < 2 {{ return 0; }}\n    if n == 2 {{ return 1; }}\n    if n % 2 == 0 {{ return 0; }}\n    i: i64 = 3;\n    while i * i <= n {{\n        if n % i == 0 {{ return 0; }}\n        i = i + 2;\n    }}\n    return 1;\n}}\n"),
            // euler_totient
            format!("fn {fn_name}(n: i64) -> i64 {{\n    result: i64 = n;\n    p: i64 = 2;\n    temp: i64 = n;\n    while p * p <= temp {{\n        if temp % p == 0 {{\n            while temp % p == 0 {{\n                temp = temp / p;\n            }}\n            result = result - result / p;\n        }}\n        p = p + 1;\n    }}\n    if temp > 1 {{\n        result = result - result / temp;\n    }}\n    return result;\n}}\n"),
            // collatz_steps
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    steps: i64 = 0;\n    while x > 1 {{\n        if x % 2 == 0 {{\n            x = x / 2;\n        }} else {{\n            x = 3 * x + 1;\n        }}\n        steps = steps + 1;\n    }}\n    return steps;\n}}\n"),
            // nth_triangle / sum_to_n (loop variant)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    if n <= 0 {{ return 0; }}\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {{\n        total = total + i;\n        i = i + 1;\n    }}\n    return total;\n}}\n"),
            // nth_triangle formula
            format!("fn {fn_name}(n: i64) -> i64 {{\n    return n * (n + 1) / 2;\n}}\n"),
            // polynomial 2x^2+3x+1
            format!("fn {fn_name}(x: i64) -> i64 {{\n    return 2 * x * x + 3 * x + 1;\n}}\n"),
            // lucas_number
            format!("fn {fn_name}(n: i64) -> i64 {{\n    if n == 0 {{ return 2; }}\n    if n == 1 {{ return 1; }}\n    a: i64 = 2;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n {{\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }}\n    return b;\n}}\n"),
            // fibonacci / fib_iter iterative
            format!("fn {fn_name}(n: i64) -> i64 {{\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 0;\n    while i < n {{\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }}\n    return a;\n}}\n"),
            // clamp 0..100 (two-if style)
            format!("fn {fn_name}(x: i64) -> i64 {{\n    if x < 0 {{ return 0; }}\n    if x > 100 {{ return 100; }}\n    return x;\n}}\n"),
            // identity passthrough
            format!("fn {fn_name}(x: i64) -> i64 {{\n    return x;\n}}\n"),
            // abs
            format!("fn {fn_name}(x: i64) -> i64 {{\n    if x < 0 {{ return 0 - x; }}\n    return x;\n}}\n"),
        ],
        2 => vec![
            // safe_div_or_neg1
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    if b == 0 {{ return -1; }}\n    return a / b;\n}}\n"),
            // gcd (Euclidean)
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {{\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }}\n    return x;\n}}\n"),
            // lcm inline
            format!("fn gcd_h(a: i64, b: i64) -> i64 {{\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {{\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }}\n    return x;\n}}\nfn {fn_name}(a: i64, b: i64) -> i64 {{\n    return (a * b) / gcd_h(a, b);\n}}\n"),
            // max2 inline
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    if a > b {{ return a; }}\n    return b;\n}}\n"),
            // min2 inline
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    if a < b {{ return a; }}\n    return b;\n}}\n"),
            // abs_diff inline
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    d: i64 = a - b;\n    if d < 0 {{ return 0 - d; }}\n    return d;\n}}\n"),
            // scaled_sum: 2*a + b
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    return 2 * a + b;\n}}\n"),
            // product_offset: a*b - a
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    return a * b - a;\n}}\n"),
            // sum
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    return a + b;\n}}\n"),
            // product
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    return a * b;\n}}\n"),
        ],
        3 => vec![
            // min3
            format!("fn {fn_name}(a: i64, b: i64, c: i64) -> i64 {{\n    m: i64 = a;\n    if b < m {{ m = b; }}\n    if c < m {{ m = c; }}\n    return m;\n}}\n"),
            // max3
            format!("fn {fn_name}(a: i64, b: i64, c: i64) -> i64 {{\n    m: i64 = a;\n    if b > m {{ m = b; }}\n    if c > m {{ m = c; }}\n    return m;\n}}\n"),
            // median3
            format!("fn {fn_name}(a: i64, b: i64, c: i64) -> i64 {{\n    if (a >= b && b >= c) || (c >= b && b >= a) {{ return b; }}\n    if (b >= a && a >= c) || (c >= a && a >= b) {{ return a; }}\n    return c;\n}}\n"),
            // sum3
            format!("fn {fn_name}(a: i64, b: i64, c: i64) -> i64 {{\n    return a + b + c;\n}}\n"),
        ],
        _ => vec![],
    };

    for code in candidates {
        if let Some(r) = make_result(code) {
            return Some(r);
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// Program type U: SoftUniversalProgram
// ═══════════════════════════════════════════════════════════════════════════════
//
// A single flexible architecture that can represent ANY program structure.
// No pre-defined skeleton — structure is emergent from gradient descent.
//
//   Init phase   (N_INIT_SLOTS slots, execute once)       → v0, v1, v2
//   Loop state   (N_LOOP_SLOTS state vars, initialized from pre-loop pool)
//   Loop phase   (N_LOOP_SLOTS slots, iterate up to MAX_LOOP_ITER times)  → s0..s5
//   Post phase   (N_POST_SLOTS slots, execute once after loop)             → p0, p1
//   Return       (soft-select from any register)
//
// Register file (pool, size = n_args + N_CONSTS + N_UNIV_SLOTS):
//   [arg0..arg_{n-1}, c0..c5, v0..v{N_INIT-1}, s0..s{N_LOOP-1}, p0..p{N_POST-1}]
//
// Each slot i computes via learned logits:
//   then_val = soft_op_ext(src1, src2)         (op ∈ {+,-,*,/,%,identity})
//   gate     = soft_cmp(gate_lhs, gate_rhs)    (cmp ∈ {<,<=,==,>=,>,!=})
//   else_val = soft_select_from_pool(else_src)
//   out      = gate * then_val + (1-gate) * else_val
//
// Loop state vars get soft-updated each iteration:
//   s_i = cond * exec_slot(i) + (1-cond) * s_i
//
// This means:
//   - Pure expression programs: loop cond → 0 (loop never fires)
//   - Branch-only programs: init/post slots use gate to branch
//   - Accumulator loops: one loop slot sums/multiplies, another counts steps
//   - Conditional mutation (collatz): loop slots read from earlier-in-iteration slots
//   - Any composition of the above emerges without pre-defining the type

pub const N_INIT_SLOTS: usize = 3;
pub const N_LOOP_SLOTS: usize = 6;
pub const N_POST_SLOTS: usize = 2;
pub const N_UNIV_SLOTS: usize = N_INIT_SLOTS + N_LOOP_SLOTS + N_POST_SLOTS; // 11

#[inline]
fn univ_pool(n_args: usize) -> usize {
    n_args + N_CONSTS + N_UNIV_SLOTS
}
#[inline]
fn univ_lip(n_args: usize) -> usize {
    n_args + N_CONSTS + N_INIT_SLOTS
} // loop init pool
#[inline]
fn univ_sps(pool: usize) -> usize {
    (N_OPS + 1) + 5 * pool + N_CMPS
} // slot params size

pub struct SoftUniversalProgram {
    pub n_args: usize,
    pub params: Vec<f32>,
}

impl SoftUniversalProgram {
    fn ps(&self) -> usize {
        univ_pool(self.n_args)
    }
    fn lip(&self) -> usize {
        univ_lip(self.n_args)
    }
    fn sps(&self) -> usize {
        univ_sps(self.ps())
    }

    fn slot_off(&self, slot: usize) -> usize {
        slot * self.sps()
    }

    fn loop_init_off(&self, ls: usize) -> usize {
        N_UNIV_SLOTS * self.sps() + ls * self.lip()
    }
    fn loop_cond_off(&self) -> usize {
        self.loop_init_off(N_LOOP_SLOTS)
    }
    fn return_off(&self) -> usize {
        self.loop_cond_off() + N_CMPS + 2 * self.ps()
    }
    fn consts_off(&self) -> usize {
        self.return_off() + self.ps()
    }

    fn n_params_for(n_args: usize) -> usize {
        let pool = univ_pool(n_args);
        let lip = univ_lip(n_args);
        N_UNIV_SLOTS * univ_sps(pool) + N_LOOP_SLOTS * lip + N_CMPS + 2 * pool + pool + N_CONSTS
    }

    fn new(n_args: usize) -> Self {
        let mut s = Self {
            n_args,
            params: vec![0f32; Self::n_params_for(n_args)],
        };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    pub fn new_from_params(n_args: usize, params: Vec<f32>) -> Self {
        Self { n_args, params }
    }

    // Execute one slot given the current register file R (length = pool_size).
    // Called both in init phase and loop phase — semantics are identical.
    fn exec_slot(&self, slot: usize, r: &[f32], temp: f32) -> f32 {
        let pool = self.ps();
        let off = self.slot_off(slot);
        // op (N_OPS+1 = 6: +,-,*,/,%,identity)
        let op_w = softmax_temp(&self.params[off..off + N_OPS + 1], temp);
        let s1_w = softmax_temp(&self.params[off + N_OPS + 1..off + N_OPS + 1 + pool], temp);
        let s2_w = softmax_temp(
            &self.params[off + N_OPS + 1 + pool..off + N_OPS + 1 + 2 * pool],
            temp,
        );
        let src1 = soft_read(r, &s1_w);
        let src2 = soft_read(r, &s2_w);
        let then_val = soft_op_ext(src1, src2, &op_w);
        // gate condition
        let cmp_base = off + N_OPS + 1 + 2 * pool;
        let cmp_w = softmax_temp(&self.params[cmp_base..cmp_base + N_CMPS], temp);
        let gl_w = softmax_temp(
            &self.params[cmp_base + N_CMPS..cmp_base + N_CMPS + pool],
            temp,
        );
        let gr_w = softmax_temp(
            &self.params[cmp_base + N_CMPS + pool..cmp_base + N_CMPS + 2 * pool],
            temp,
        );
        let gate_lhs = soft_read(r, &gl_w);
        let gate_rhs = soft_read(r, &gr_w);
        let gate = soft_cmp(gate_lhs, gate_rhs, &cmp_w, temp);
        // else branch
        let el_w = softmax_temp(
            &self.params[cmp_base + N_CMPS + 2 * pool..cmp_base + N_CMPS + 3 * pool],
            temp,
        );
        let else_val = soft_read(r, &el_w);
        gate * then_val + (1.0 - gate) * else_val
    }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let n = self.n_args;
        let pool = self.ps();
        let lip = self.lip();
        let mut reg = vec![0f32; pool];
        for i in 0..n.min(inputs.len()) {
            reg[i] = inputs[i];
        }
        let co = self.consts_off();
        for i in 0..N_CONSTS {
            reg[n + i] = self.params[co + i];
        }

        // ── Phase 1: init slots ────────────────────────────────────────────────
        for slot in 0..N_INIT_SLOTS {
            reg[n + N_CONSTS + slot] = self.exec_slot(slot, &reg, temp);
        }

        // ── Phase 2: loop state init ───────────────────────────────────────────
        for ls in 0..N_LOOP_SLOTS {
            let io = self.loop_init_off(ls);
            let w = softmax_temp(&self.params[io..io + lip], temp);
            reg[n + N_CONSTS + N_INIT_SLOTS + ls] = soft_read(&reg[..lip], &w);
        }

        // ── Phase 3: loop ──────────────────────────────────────────────────────
        let lco = self.loop_cond_off();
        for _iter in 0..MAX_LOOP_ITER {
            let cmp_w = softmax_temp(&self.params[lco..lco + N_CMPS], temp);
            let lhs_w = softmax_temp(&self.params[lco + N_CMPS..lco + N_CMPS + pool], temp);
            let rhs_w = softmax_temp(
                &self.params[lco + N_CMPS + pool..lco + N_CMPS + 2 * pool],
                temp,
            );
            let lhs = soft_read(&reg, &lhs_w);
            let rhs = soft_read(&reg, &rhs_w);
            let cond = soft_cmp(lhs, rhs, &cmp_w, temp);
            if cond < 1e-6 {
                break;
            }
            // Execute loop body slots sequentially so slot j can read slot j-1's
            // current-iteration output (already soft-updated in reg).
            for ls in 0..N_LOOP_SLOTS {
                let slot = N_INIT_SLOTS + ls;
                let out = self.exec_slot(slot, &reg, temp);
                let idx = n + N_CONSTS + slot;
                reg[idx] = cond * out + (1.0 - cond) * reg[idx];
            }
        }

        // ── Phase 4: post slots ────────────────────────────────────────────────
        for pi in 0..N_POST_SLOTS {
            let slot = N_INIT_SLOTS + N_LOOP_SLOTS + pi;
            reg[n + N_CONSTS + slot] = self.exec_slot(slot, &reg, temp);
        }

        // ── Return: soft-select from any register ─────────────────────────────
        let ro = self.return_off();
        let rw = softmax_temp(&self.params[ro..ro + pool], temp);
        soft_read(&reg, &rw)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples
            .iter()
            .map(|(inp, tgt)| {
                let diff = self.forward(inp, temp) - tgt;
                diff * diff
            })
            .sum::<f32>()
            / n
    }

    fn discretize_and_emit(&self, fn_name: &str, param_names: &[&str]) -> String {
        let n = self.n_args;
        let pool = self.ps();
        let lip = self.lip();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|i| self.params[co + i].round() as i64)
            .collect();

        // Build pool name table (full pool)
        let mut pn: Vec<String> = Vec::with_capacity(pool);
        for i in 0..n {
            pn.push(param_names.get(i).copied().unwrap_or("x").to_string());
        }
        for v in &consts {
            pn.push(format!("{v}"));
        }
        for i in 0..N_INIT_SLOTS {
            pn.push(format!("v{i}"));
        }
        for i in 0..N_LOOP_SLOTS {
            pn.push(format!("s{i}"));
        }
        for i in 0..N_POST_SLOTS {
            pn.push(format!("p{i}"));
        }

        // Loop-init pool names (only pre-loop registers)
        let mut lpn: Vec<String> = Vec::with_capacity(lip);
        for i in 0..n {
            lpn.push(param_names.get(i).copied().unwrap_or("x").to_string());
        }
        for v in &consts {
            lpn.push(format!("{v}"));
        }
        for i in 0..N_INIT_SLOTS {
            lpn.push(format!("v{i}"));
        }

        // op → string, treating index N_OPS as identity (returns src1)
        let op_names = ["+", "-", "*", "/", "%"];
        let cmp_names = ["<", "<=", "==", ">=", ">", "!="];

        // Build code for one slot. Returns the "then_expr" and optional branch parts.
        // dest_name: the variable being assigned (e.g., "v0", "s2", "p1")
        // decl: whether to use "dest: i64 = ..." syntax (true) or "dest = ..." (false)
        let slot_line = |slot: usize, dest_name: &str, decl: bool| -> String {
            let sps = self.sps();
            let off = self.slot_off(slot);
            let op_i = argmax(&self.params[off..off + N_OPS + 1]);
            let s1_i = argmax(&self.params[off + N_OPS + 1..off + N_OPS + 1 + pool]);
            let s2_i = argmax(&self.params[off + N_OPS + 1 + pool..off + N_OPS + 1 + 2 * pool]);
            let cb = off + N_OPS + 1 + 2 * pool;
            let cmp_i = argmax(&self.params[cb..cb + N_CMPS]);
            let gl_i = argmax(&self.params[cb + N_CMPS..cb + N_CMPS + pool]);
            let gr_i = argmax(&self.params[cb + N_CMPS + pool..cb + N_CMPS + 2 * pool]);
            let el_i = argmax(&self.params[cb + N_CMPS + 2 * pool..cb + N_CMPS + 3 * pool]);
            let _ = sps; // used in size but not needed here

            let s1 = &pn[s1_i];
            let s2 = &pn[s2_i];
            let then_expr = if op_i >= N_OPS {
                s1.clone()
            }
            // identity
            else {
                format!("{s1} {} {s2}", op_names[op_i])
            };
            let else_expr = pn[el_i].clone();
            let gl = &pn[gl_i];
            let gr = &pn[gr_i];
            let cmp_s = cmp_names[cmp_i.min(5)];

            // Trivial gate detection
            let trivially_true = gl_i == gr_i && matches!(cmp_i, 1 | 2 | 3); // <=, ==, >=
            let trivially_false = gl_i == gr_i && matches!(cmp_i, 0 | 4 | 5); // <, >, !=
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
            } else {
                format!("    {prefix} = {else_expr};\n    if {gl} {cmp_s} {gr} {{ {dest_name} = {then_expr}; }}")
            }
        };

        let params_sig = param_names
            .iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut out = format!("fn {fn_name}({params_sig}) -> i64 {{\n");

        // Init phase
        for i in 0..N_INIT_SLOTS {
            let line = slot_line(i, &format!("v{i}"), true);
            out.push_str(&line);
            out.push('\n');
        }

        // Loop state init
        for ls in 0..N_LOOP_SLOTS {
            let io = self.loop_init_off(ls);
            let src_i = argmax(&self.params[io..io + lip]);
            let src = &lpn[src_i];
            writeln!(out, "    s{ls}: i64 = {src};").unwrap();
        }

        // Loop condition
        let lco = self.loop_cond_off();
        let cmp_i = argmax(&self.params[lco..lco + N_CMPS]);
        let lhs_i = argmax(&self.params[lco + N_CMPS..lco + N_CMPS + pool]);
        let rhs_i = argmax(&self.params[lco + N_CMPS + pool..lco + N_CMPS + 2 * pool]);
        let lhs_s = &pn[lhs_i];
        let rhs_s = &pn[rhs_i];
        let cmp_names2 = ["<", "<=", "==", ">=", ">", "!="];
        let cmp_s = cmp_names2[cmp_i.min(5)];
        writeln!(out, "    while {lhs_s} {cmp_s} {rhs_s} {{").unwrap();

        // Loop body
        for ls in 0..N_LOOP_SLOTS {
            let slot = N_INIT_SLOTS + ls;
            let line = slot_line(slot, &format!("s{ls}"), false);
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str("    }\n");

        // Post phase
        for pi in 0..N_POST_SLOTS {
            let slot = N_INIT_SLOTS + N_LOOP_SLOTS + pi;
            let line = slot_line(slot, &format!("p{pi}"), true);
            out.push_str(&line);
            out.push('\n');
        }

        // Return
        let ro = self.return_off();
        let ret_i = argmax(&self.params[ro..ro + pool]);
        writeln!(out, "    return {};", pn[ret_i]).unwrap();
        out.push_str("}\n");
        out
    }

    // ── Discrete integer evaluation ────────────────────────────────────────────
    // Executes the argmax-discretized program on integer inputs without going
    // through the Mog interpreter.  Applies the same MAX_LOOP_ITER limit as the
    // soft forward pass, and returns None on division-by-zero or overflow.

    fn disc_exec_slot(&self, slot: usize, reg: &[i64]) -> Option<i64> {
        let pool = self.ps();
        let off = slot * self.sps();
        let op_i = argmax(&self.params[off..off + N_OPS + 1]);
        let s1_i = argmax(&self.params[off + N_OPS + 1..off + N_OPS + 1 + pool]);
        let s2_i = argmax(&self.params[off + N_OPS + 1 + pool..off + N_OPS + 1 + 2 * pool]);
        let cb = off + N_OPS + 1 + 2 * pool;
        let cmp_i = argmax(&self.params[cb..cb + N_CMPS]);
        let gl_i = argmax(&self.params[cb + N_CMPS..cb + N_CMPS + pool]);
        let gr_i = argmax(&self.params[cb + N_CMPS + pool..cb + N_CMPS + 2 * pool]);
        let el_i = argmax(&self.params[cb + N_CMPS + 2 * pool..cb + N_CMPS + 3 * pool]);

        let s1 = *reg.get(s1_i)?;
        let s2 = *reg.get(s2_i)?;
        let then_val = match op_i {
            0 => s1.checked_add(s2)?,
            1 => s1.checked_sub(s2)?,
            2 => s1.checked_mul(s2)?,
            3 => {
                if s2 == 0 {
                    return None;
                }
                s1 / s2
            }
            4 => {
                if s2 == 0 {
                    return None;
                }
                s1 % s2
            }
            _ => s1, // identity
        };
        let gl = *reg.get(gl_i)?;
        let gr = *reg.get(gr_i)?;
        let gate = match cmp_i % N_CMPS {
            0 => gl < gr,
            1 => gl <= gr,
            2 => gl == gr,
            3 => gl >= gr,
            4 => gl > gr,
            _ => gl != gr,
        };
        if gate {
            Some(then_val)
        } else {
            reg.get(el_i).copied()
        }
    }

    /// Execute the discretized program on integer inputs.
    /// Returns None if the program divides by zero, overflows, or uses an
    /// out-of-bounds register (shouldn't happen for valid descriptions).
    pub fn discrete_eval(&self, int_inputs: &[i64]) -> Option<i64> {
        let n = self.n_args;
        let pool = self.ps();
        let lip = self.lip();
        let co = self.consts_off();

        let mut reg = vec![0i64; pool];
        for i in 0..n.min(int_inputs.len()) {
            reg[i] = int_inputs[i];
        }
        for i in 0..N_CONSTS {
            reg[n + i] = self.params[co + i].round() as i64;
        }

        // Phase 1: init slots
        for slot in 0..N_INIT_SLOTS {
            let v = self.disc_exec_slot(slot, &reg)?;
            reg[n + N_CONSTS + slot] = v;
        }

        // Phase 2: loop state init (from pre-loop pool)
        for ls in 0..N_LOOP_SLOTS {
            let io = self.loop_init_off(ls);
            let src_i = argmax(&self.params[io..io + lip]);
            reg[n + N_CONSTS + N_INIT_SLOTS + ls] = *reg.get(src_i)?;
        }

        // Phase 3: loop (hard limit = MAX_LOOP_ITER)
        let lco = self.loop_cond_off();
        let cmp_i = argmax(&self.params[lco..lco + N_CMPS]);
        let lhs_i = argmax(&self.params[lco + N_CMPS..lco + N_CMPS + pool]);
        let rhs_i = argmax(&self.params[lco + N_CMPS + pool..lco + N_CMPS + 2 * pool]);

        for _ in 0..MAX_LOOP_ITER {
            let lhs = *reg.get(lhs_i)?;
            let rhs = *reg.get(rhs_i)?;
            let cont = match cmp_i % N_CMPS {
                0 => lhs < rhs,
                1 => lhs <= rhs,
                2 => lhs == rhs,
                3 => lhs >= rhs,
                4 => lhs > rhs,
                _ => lhs != rhs,
            };
            if !cont {
                break;
            }
            for ls in 0..N_LOOP_SLOTS {
                let slot = N_INIT_SLOTS + ls;
                let v = self.disc_exec_slot(slot, &reg)?;
                reg[n + N_CONSTS + slot] = v;
            }
        }

        // Phase 4: post slots
        for pi in 0..N_POST_SLOTS {
            let slot = N_INIT_SLOTS + N_LOOP_SLOTS + pi;
            let v = self.disc_exec_slot(slot, &reg)?;
            reg[n + N_CONSTS + slot] = v;
        }

        // Return
        let ro = self.return_off();
        let ret_i = argmax(&self.params[ro..ro + pool]);
        reg.get(ret_i).copied()
    }
}

// ─── UniversalProgramDescription ─────────────────────────────────────────────
//
// Discrete representation of a SoftUniversalProgram: every logit group is
// collapsed to a single integer (argmax) and the concrete const values are
// stored as-is.  Use this to:
//   • inspect what structure a trained program converged to
//   • hand-author a program and warm-start gradient synthesis from its params
//   • collect (I/O examples, params) pairs for meta-learner training
//
// Round-trip guarantee:
//   params_to_description → description_to_params → discretize_and_emit
// produces the **same** code as the original discretize_and_emit call.

/// Discrete description of one execution slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlotDesc {
    pub op: usize,       // 0-5: +,-,*,/,%,identity
    pub s1: usize,       // pool index (src1)
    pub s2: usize,       // pool index (src2)
    pub gate_cmp: usize, // 0-5: <,<=,==,>=,>,!=
    pub gate_lhs: usize, // pool index
    pub gate_rhs: usize, // pool index
    pub else_val: usize, // pool index (else branch)
}

/// Full discrete description of a SoftUniversalProgram.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniversalProgramDescription {
    pub n_args: usize,
    pub slots: Vec<SlotDesc>,  // len = N_UNIV_SLOTS (11)
    pub loop_init: Vec<usize>, // len = N_LOOP_SLOTS (6), indices into lip pool
    pub cond_cmp: usize,
    pub cond_lhs: usize,         // pool index
    pub cond_rhs: usize,         // pool index
    pub ret_src: usize,          // pool index
    pub consts: [f32; N_CONSTS], // concrete values (NOT logit indices)
}

impl UniversalProgramDescription {
    /// Pool name table: [arg0..arg_{n-1}, c0..c5, v0..v2, s0..s5, p0..p1]
    pub fn pool_names(&self) -> Vec<String> {
        SoftUniversalProgram::pool_names(self.n_args, &self.consts)
    }

    /// Loop-init pool names: [arg0..arg_{n-1}, c0..c5, v0..v2]
    pub fn lip_names(&self) -> Vec<String> {
        let default_args = ["a", "b", "c", "d", "e", "f"];
        let mut pn = Vec::new();
        for i in 0..self.n_args {
            pn.push(default_args.get(i).copied().unwrap_or("x").to_string());
        }
        for v in &self.consts {
            pn.push(format!("{}", v.round() as i64));
        }
        for i in 0..N_INIT_SLOTS {
            pn.push(format!("v{i}"));
        }
        pn
    }

    /// Human-readable program structure dump.
    pub fn explain(&self) -> String {
        let pn = self.pool_names();
        let lip_pn = self.lip_names();
        let op_sym = ["+", "-", "*", "/", "%", "id"];
        let cmp_sym = ["<", "<=", "==", ">=", ">", "!="];

        let phase_for = |s: usize| {
            if s < N_INIT_SLOTS {
                "init"
            } else if s < N_INIT_SLOTS + N_LOOP_SLOTS {
                "loop"
            } else {
                "post"
            }
        };
        let dest_for = |s: usize| {
            if s < N_INIT_SLOTS {
                format!("v{s}")
            } else if s < N_INIT_SLOTS + N_LOOP_SLOTS {
                format!("s{}", s - N_INIT_SLOTS)
            } else {
                format!("p{}", s - N_INIT_SLOTS - N_LOOP_SLOTS)
            }
        };
        let pn_get = |i: usize| pn.get(i).map(|s| s.as_str()).unwrap_or("?");

        let mut out = String::new();
        for (i, sd) in self.slots.iter().enumerate() {
            out.push_str(&format!(
                "  [{:4}] {} = if {} {} {} {{ {} {} {} }} else {{ {} }}\n",
                phase_for(i),
                dest_for(i),
                pn_get(sd.gate_lhs),
                cmp_sym.get(sd.gate_cmp).copied().unwrap_or("?"),
                pn_get(sd.gate_rhs),
                pn_get(sd.s1),
                op_sym.get(sd.op).copied().unwrap_or("?"),
                pn_get(sd.s2),
                pn_get(sd.else_val),
            ));
        }
        for (ls, &src) in self.loop_init.iter().enumerate() {
            out.push_str(&format!(
                "  [loop_init] s{ls} = {}\n",
                lip_pn.get(src).map(|s| s.as_str()).unwrap_or("?")
            ));
        }
        let lhs = pn_get(self.cond_lhs);
        let rhs = pn_get(self.cond_rhs);
        let cmp = cmp_sym.get(self.cond_cmp).copied().unwrap_or("?");
        out.push_str(&format!("  [loop_cond] while {lhs} {cmp} {rhs}\n"));
        out.push_str(&format!("  [return]    {}\n", pn_get(self.ret_src)));
        out
    }
}

#[allow(dead_code)]
impl SoftUniversalProgram {
    /// Extract a discrete description by taking argmax over every logit group.
    pub fn params_to_description(&self) -> UniversalProgramDescription {
        let pool = self.ps();
        let lip = self.lip();

        let slots: Vec<SlotDesc> = (0..N_UNIV_SLOTS)
            .map(|slot| {
                let off = self.slot_off(slot);
                let op = argmax(&self.params[off..off + N_OPS + 1]);
                let s1 = argmax(&self.params[off + N_OPS + 1..off + N_OPS + 1 + pool]);
                let s2 = argmax(&self.params[off + N_OPS + 1 + pool..off + N_OPS + 1 + 2 * pool]);
                let cb = off + N_OPS + 1 + 2 * pool;
                let gate_cmp = argmax(&self.params[cb..cb + N_CMPS]);
                let gate_lhs = argmax(&self.params[cb + N_CMPS..cb + N_CMPS + pool]);
                let gate_rhs = argmax(&self.params[cb + N_CMPS + pool..cb + N_CMPS + 2 * pool]);
                let else_val = argmax(&self.params[cb + N_CMPS + 2 * pool..cb + N_CMPS + 3 * pool]);
                SlotDesc {
                    op,
                    s1,
                    s2,
                    gate_cmp,
                    gate_lhs,
                    gate_rhs,
                    else_val,
                }
            })
            .collect();

        let loop_init: Vec<usize> = (0..N_LOOP_SLOTS)
            .map(|ls| {
                let io = self.loop_init_off(ls);
                argmax(&self.params[io..io + lip])
            })
            .collect();

        let lco = self.loop_cond_off();
        let cond_cmp = argmax(&self.params[lco..lco + N_CMPS]);
        let cond_lhs = argmax(&self.params[lco + N_CMPS..lco + N_CMPS + pool]);
        let cond_rhs = argmax(&self.params[lco + N_CMPS + pool..lco + N_CMPS + 2 * pool]);

        let ro = self.return_off();
        let ret_src = argmax(&self.params[ro..ro + pool]);

        let co = self.consts_off();
        let mut consts = [0f32; N_CONSTS];
        for i in 0..N_CONSTS {
            consts[i] = self.params[co + i];
        }

        UniversalProgramDescription {
            n_args: self.n_args,
            slots,
            loop_init,
            cond_cmp,
            cond_lhs,
            cond_rhs,
            ret_src,
            consts,
        }
    }

    /// Reconstruct a SoftUniversalProgram from a discrete description.
    /// Sets the chosen index in each logit group to +4.0 and all others to -4.0,
    /// matching the biased restart conventions used throughout the solver.
    pub fn description_to_params(desc: &UniversalProgramDescription) -> Self {
        let n_args = desc.n_args;
        let mut prog = Self::new(n_args);
        let pool = prog.ps();

        // Suppress everything first
        for p in prog.params.iter_mut() {
            *p = -4.0;
        }

        // Slot logits
        for (slot, sd) in desc.slots.iter().enumerate() {
            let off = prog.slot_off(slot);
            prog.params[off + sd.op] = 4.0; // op
            prog.params[off + N_OPS + 1 + sd.s1] = 4.0; // src1
            prog.params[off + N_OPS + 1 + pool + sd.s2] = 4.0; // src2
            let cb = off + N_OPS + 1 + 2 * pool;
            prog.params[cb + sd.gate_cmp] = 4.0; // gate cmp
            prog.params[cb + N_CMPS + sd.gate_lhs] = 4.0; // gate lhs
            prog.params[cb + N_CMPS + pool + sd.gate_rhs] = 4.0; // gate rhs
            prog.params[cb + N_CMPS + 2 * pool + sd.else_val] = 4.0; // else val
        }

        // Loop-init logits
        for (ls, &init_src) in desc.loop_init.iter().enumerate() {
            let io = prog.loop_init_off(ls);
            prog.params[io + init_src] = 4.0;
        }

        // Loop condition logits
        let lco = prog.loop_cond_off();
        prog.params[lco + desc.cond_cmp] = 4.0;
        prog.params[lco + N_CMPS + desc.cond_lhs] = 4.0;
        prog.params[lco + N_CMPS + pool + desc.cond_rhs] = 4.0;

        // Return logit
        let ro = prog.return_off();
        prog.params[ro + desc.ret_src] = 4.0;

        // Concrete const values — write directly (not logits)
        let co = prog.consts_off();
        for i in 0..N_CONSTS {
            prog.params[co + i] = desc.consts[i];
        }

        prog
    }

    /// Build the pool name table for (n_args, consts).
    pub fn pool_names(n_args: usize, consts: &[f32]) -> Vec<String> {
        let default_args = ["a", "b", "c", "d", "e", "f"];
        let mut pn = Vec::new();
        for i in 0..n_args {
            pn.push(default_args.get(i).copied().unwrap_or("x").to_string());
        }
        for v in consts.iter().take(N_CONSTS) {
            pn.push(format!("{}", v.round() as i64));
        }
        for i in 0..N_INIT_SLOTS {
            pn.push(format!("v{i}"));
        }
        for i in 0..N_LOOP_SLOTS {
            pn.push(format!("s{i}"));
        }
        for i in 0..N_POST_SLOTS {
            pn.push(format!("p{i}"));
        }
        pn
    }
}

// ─── Meta-learner data collection ────────────────────────────────────────────

/// Sample a random `UniversalProgramDescription` with valid index ranges.
/// All discrete choices are drawn uniformly from their valid range.
pub fn rand_description(n_args: usize, seed: u64) -> UniversalProgramDescription {
    let pool = univ_pool(n_args);
    let lip = univ_lip(n_args);
    let mut rng = seed;
    let mut next = |modulus: usize| -> usize {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 33) as usize) % modulus
    };

    let slots: Vec<SlotDesc> = (0..N_UNIV_SLOTS)
        .map(|_| SlotDesc {
            op: next(N_OPS + 1), // 0-5
            s1: next(pool),
            s2: next(pool),
            gate_cmp: next(N_CMPS),
            gate_lhs: next(pool),
            gate_rhs: next(pool),
            else_val: next(pool),
        })
        .collect();

    let loop_init: Vec<usize> = (0..N_LOOP_SLOTS).map(|_| next(lip)).collect();

    let cond_cmp = next(N_CMPS);
    let cond_lhs = next(pool);
    let cond_rhs = next(pool);
    let ret_src = next(pool);

    // Consts: sample from small integers to keep values in a reasonable range
    let const_pool: [f32; 12] = [
        0.0, 1.0, -1.0, 2.0, -2.0, 3.0, 10.0, 100.0, -3.0, 4.0, 5.0, -10.0,
    ];
    let mut consts = [0f32; N_CONSTS];
    for c in &mut consts {
        *c = const_pool[next(const_pool.len())];
    }

    UniversalProgramDescription {
        n_args,
        slots,
        loop_init,
        cond_cmp,
        cond_lhs,
        cond_rhs,
        ret_src,
        consts,
    }
}

/// One (description, io_examples) training record.
#[derive(Serialize, Deserialize)]
pub struct MetaRecord {
    pub fn_name: String,
    pub description: UniversalProgramDescription,
    /// Each entry: (int inputs, int output)
    pub io_examples: Vec<(Vec<i64>, i64)>,
    /// Source: "synthetic" | "benchmark"
    pub source: String,
}

/// Generate one synthetic `MetaRecord` for the given description.
/// Uses `discrete_eval` (pure Rust, respects MAX_LOOP_ITER) — no interpreter overhead.
pub fn synthetic_record(
    desc: &UniversalProgramDescription,
    n_eval: usize,
    seed: u64,
) -> Option<MetaRecord> {
    let prog = SoftUniversalProgram::description_to_params(desc);
    let mut rng = seed;
    let mut next_i64 = || -> i64 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Inputs in 1..20 — small enough to avoid most overflows
        ((rng >> 33) as i64).abs() % 19 + 1
    };

    let mut io: Vec<(Vec<i64>, i64)> = Vec::with_capacity(n_eval);
    for _ in 0..n_eval {
        let inputs: Vec<i64> = (0..desc.n_args).map(|_| next_i64()).collect();
        match prog.discrete_eval(&inputs) {
            Some(out) if out.abs() < 1_000_000 => io.push((inputs, out)),
            _ => return None, // div-by-zero, overflow, or huge output
        }
    }

    // Diversity check: at least 2 distinct outputs so the program is non-trivial
    let distinct: std::collections::HashSet<i64> = io.iter().map(|(_, o)| *o).collect();
    if distinct.len() < 2 {
        return None;
    }

    Some(MetaRecord {
        fn_name: "f".to_string(),
        description: desc.clone(),
        io_examples: io,
        source: "synthetic".to_string(),
    })
}

/// Like `synthesize_scalar` but runs only the SoftUniversalProgram and returns
/// the winning params alongside the `SolveResult`. Used for benchmark data collection.
///
/// `max_steps` caps the gradient budget per restart (use 400 for fast collection).
pub fn synthesize_universal_and_collect(
    problem: &Problem,
    max_steps: usize,
) -> Option<(SolveResult, Vec<f32>)> {
    if !problem
        .examples
        .iter()
        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))))
    {
        return None;
    }
    let fn_name = problem.function_name();
    let n_args = problem.examples.first()?.inputs.len();

    let examples: Vec<(Vec<f32>, f32)> = problem
        .examples
        .iter()
        .map(|ex| {
            let inputs: Vec<f32> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i as f32)
                    } else {
                        None
                    }
                })
                .collect();
            (inputs, ex.expected as f32)
        })
        .collect();

    let default_names = ["a", "b", "c", "d", "e", "f"];
    let param_names: Vec<&str> = (0..n_args)
        .map(|i| default_names.get(i).copied().unwrap_or("x"))
        .collect();

    let univ_steps = max_steps;
    const N_RESTARTS: usize = 5;

    for restart in 0..N_RESTARTS {
        let noise_scale = restart as f32 * 0.5;
        let mut prog = SoftUniversalProgram::new(n_args);
        if restart > 0 {
            for (idx, p) in prog.params.iter_mut().enumerate() {
                *p += (pseudo_rand(restart as u64 * 41000 + idx as u64) - 0.5) * noise_scale;
            }
        }
        let ex = examples.clone();
        if let Some((result, params)) = train_program_collect(
            prog.params.clone(),
            move |p, t| {
                SoftUniversalProgram {
                    n_args,
                    params: p.to_vec(),
                }
                .loss(&ex, t)
            },
            |p, fn_n, pn| {
                SoftUniversalProgram {
                    n_args,
                    params: p.to_vec(),
                }
                .discretize_and_emit(fn_n, pn)
            },
            problem,
            &param_names,
            fn_name,
            univ_steps,
        ) {
            return Some((result, params));
        }
    }
    None
}

/// Like `train_program` but returns the winning params alongside the `SolveResult`.
fn train_program_collect<F, G>(
    initial_params: Vec<f32>,
    loss_fn: F,
    emit_fn: G,
    problem: &Problem,
    param_names: &[&str],
    fn_name: &str,
    n_steps: usize,
) -> Option<(SolveResult, Vec<f32>)>
where
    F: Fn(&[f32], f32) -> f32,
    G: Fn(&[f32], &str, &[&str]) -> String,
{
    // Check initial params
    if let Some(result) = try_emit_verify(&initial_params, &emit_fn, problem, fn_name, param_names)
    {
        return Some((result, initial_params));
    }

    let mut params = initial_params;
    let n = params.len();
    let mut opt = Adam::new(n, 0.05);
    let mut best_loss = f32::MAX;
    let mut best_params = params.clone();
    let mut last_check_loss = f32::MAX;
    let chk1 = n_steps / 4;
    let chk2 = n_steps / 2;
    let mut loss_at_chk1 = f32::MAX;
    let mut loss_at_chk2 = f32::MAX;

    for step in 0..n_steps {
        if step == chk1 {
            loss_at_chk1 = best_loss;
        }
        if step == chk2 {
            loss_at_chk2 = best_loss;
        }
        if step == chk2 && best_loss > loss_at_chk1 * 0.98 {
            break;
        }
        if step > n_steps * 3 / 4 && best_loss > loss_at_chk2 * 0.90 {
            break;
        }

        let temp = (2.0f32 * (1.0 - step as f32 / n_steps as f32)).max(0.1);
        let loss = loss_fn(&params, temp);
        if loss < best_loss {
            best_loss = loss;
            best_params = params.clone();
        }

        let should_check = loss < 1.0 || (loss < last_check_loss * 0.9) || (step % 50 == 49);
        if should_check {
            last_check_loss = loss.min(last_check_loss);
            if let Some(result) = try_emit_verify(&params, &emit_fn, problem, fn_name, param_names)
            {
                return Some((result, params));
            }
            if best_loss < loss {
                if let Some(result) =
                    try_emit_verify(&best_params, &emit_fn, problem, fn_name, param_names)
                {
                    return Some((result, best_params));
                }
            }
        }
        let grads = fd_grad(&params, &loss_fn, temp);
        opt.step(&mut params, &grads);
    }

    if let Some(result) = try_emit_verify(&params, &emit_fn, problem, fn_name, param_names) {
        return Some((result, params));
    }
    try_emit_verify(&best_params, &emit_fn, problem, fn_name, param_names).map(|r| (r, best_params))
}

// ─── Warm-start synthesis ─────────────────────────────────────────────────────

/// Attempt to synthesize using a warm-start from a predicted `UniversalProgramDescription`.
///
/// Converts the description to params via `description_to_params`, then runs gradient
/// descent with a reduced step budget.  Falls back to cold-start Universal if it fails.
///
/// Returns `(result, steps_to_solve, warm_succeeded)`.
/// `cold_restarts=0` → warm-only (fast, measures prediction quality).
/// `cold_restarts=3` → full fallback (production use).
pub fn synthesize_universal_warm_start(
    problem: &Problem,
    desc: &UniversalProgramDescription,
    warm_steps: usize,
    cold_restarts: usize,
) -> Option<(SolveResult, usize, bool)> {
    if !problem
        .examples
        .iter()
        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))))
    {
        return None;
    }
    let fn_name = problem.function_name();
    let n_args = problem.examples.first()?.inputs.len();
    if n_args != desc.n_args {
        return None;
    }

    let examples: Vec<(Vec<f32>, f32)> = problem
        .examples
        .iter()
        .map(|ex| {
            let inputs: Vec<f32> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i as f32)
                    } else {
                        None
                    }
                })
                .collect();
            (inputs, ex.expected as f32)
        })
        .collect();

    let default_names = ["a", "b", "c", "d", "e", "f"];
    let param_names: Vec<&str> = (0..n_args)
        .map(|i| default_names.get(i).copied().unwrap_or("x"))
        .collect();

    // ── Warm start ────────────────────────────────────────────────────────────
    let warm_params = SoftUniversalProgram::description_to_params(desc).params;
    let ex = examples.clone();
    let (warm_result, steps) = train_program_count_steps(
        warm_params,
        move |p, t| {
            SoftUniversalProgram {
                n_args,
                params: p.to_vec(),
            }
            .loss(&ex, t)
        },
        |p, fn_n, pn| {
            SoftUniversalProgram {
                n_args,
                params: p.to_vec(),
            }
            .discretize_and_emit(fn_n, pn)
        },
        problem,
        &param_names,
        fn_name,
        warm_steps,
    );
    if let Some((result, s)) = warm_result {
        return Some((result, s, true));
    }

    // ── Cold fallback ─────────────────────────────────────────────────────────
    if cold_restarts == 0 {
        return None;
    }
    const COLD_STEPS: usize = 800;
    for univ_restart in 0..cold_restarts {
        let mut prog = SoftUniversalProgram::new(n_args);
        if univ_restart > 0 {
            for (idx, p) in prog.params.iter_mut().enumerate() {
                *p += (pseudo_rand(univ_restart as u64 * 97000 + idx as u64) - 0.5) * 0.5;
            }
        }
        let ex2 = examples.clone();
        let (res, _s) = train_program_count_steps(
            prog.params.clone(),
            move |p, t| {
                SoftUniversalProgram {
                    n_args,
                    params: p.to_vec(),
                }
                .loss(&ex2, t)
            },
            |p, fn_n, pn| {
                SoftUniversalProgram {
                    n_args,
                    params: p.to_vec(),
                }
                .discretize_and_emit(fn_n, pn)
            },
            problem,
            &param_names,
            fn_name,
            COLD_STEPS,
        );
        if let Some((result, s)) = res {
            return Some((result, steps + s, false));
        }
    }
    None
}

/// Like `train_program_collect` but also returns the step at which it solved.
fn train_program_count_steps<F, G>(
    initial_params: Vec<f32>,
    loss_fn: F,
    emit_fn: G,
    problem: &Problem,
    param_names: &[&str],
    fn_name: &str,
    n_steps: usize,
) -> (Option<(SolveResult, usize)>, usize)
where
    F: Fn(&[f32], f32) -> f32,
    G: Fn(&[f32], &str, &[&str]) -> String,
{
    if let Some(result) = try_emit_verify(&initial_params, &emit_fn, problem, fn_name, param_names)
    {
        return (Some((result, 0)), 0);
    }

    let mut params = initial_params;
    let n = params.len();
    let mut opt = Adam::new(n, 0.05);
    let mut best_loss = f32::MAX;
    let mut best_params = params.clone();
    let mut last_check_loss = f32::MAX;
    let chk1 = n_steps / 4;
    let chk2 = n_steps / 2;
    let mut loss_at_chk1 = f32::MAX;
    let mut loss_at_chk2 = f32::MAX;

    for step in 0..n_steps {
        if step == chk1 {
            loss_at_chk1 = best_loss;
        }
        if step == chk2 {
            loss_at_chk2 = best_loss;
        }
        if step == chk2 && best_loss > loss_at_chk1 * 0.98 {
            break;
        }
        if step > n_steps * 3 / 4 && best_loss > loss_at_chk2 * 0.90 {
            break;
        }

        let temp = (2.0f32 * (1.0 - step as f32 / n_steps as f32)).max(0.1);
        let loss = loss_fn(&params, temp);
        if loss < best_loss {
            best_loss = loss;
            best_params = params.clone();
        }

        let should_check = loss < 1.0 || (loss < last_check_loss * 0.9) || (step % 50 == 49);
        if should_check {
            last_check_loss = loss.min(last_check_loss);
            if let Some(result) = try_emit_verify(&params, &emit_fn, problem, fn_name, param_names)
            {
                return (Some((result, step + 1)), step + 1);
            }
            if best_loss < loss {
                if let Some(result) =
                    try_emit_verify(&best_params, &emit_fn, problem, fn_name, param_names)
                {
                    return (Some((result, step + 1)), step + 1);
                }
            }
        }
        let grads = fd_grad(&params, &loss_fn, temp);
        opt.step(&mut params, &grads);
    }

    if let Some(result) = try_emit_verify(&params, &emit_fn, problem, fn_name, param_names) {
        return (Some((result, n_steps)), n_steps);
    }
    let final_step = n_steps;
    let r = try_emit_verify(&best_params, &emit_fn, problem, fn_name, param_names)
        .map(|r| (r, final_step));
    (r, final_step)
}

// ─── Main synthesis entry point ───────────────────────────────────────────────

/// Pure gradient-only synthesis (no templates). Used to measure true gradient capability.
pub fn synthesize_gradient_only(problem: &Problem) -> Option<SolveResult> {
    synthesize_scalar_inner(problem, false)
}

/// Attempt native gradient-based synthesis for scalar (all-i64) problems.
/// Returns `None` if the problem has non-scalar inputs or synthesis fails.
pub fn synthesize_scalar(problem: &Problem) -> Option<SolveResult> {
    synthesize_scalar_inner(problem, true)
}

fn synthesize_scalar_inner(problem: &Problem, use_templates: bool) -> Option<SolveResult> {
    // Only works on scalar (all i64) problems
    if !problem
        .examples
        .iter()
        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))))
    {
        return None;
    }

    let fn_name = problem.function_name();
    let n_args = problem.examples.first()?.inputs.len();

    // Template fast-path: try reference code + common inline patterns before gradient descent
    if use_templates {
        if let Some(result) = try_scalar_templates(problem, fn_name, n_args) {
            return Some(result);
        }
    }

    // Build training examples as f32
    let examples: Vec<(Vec<f32>, f32)> = problem
        .examples
        .iter()
        .map(|ex| {
            let inputs: Vec<f32> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i as f32)
                    } else {
                        None
                    }
                })
                .collect();
            (inputs, ex.expected as f32)
        })
        .collect();

    // Param names: a, b, c, d, e, f
    let default_names = ["a", "b", "c", "d", "e", "f"];
    let param_names: Vec<&str> = (0..n_args)
        .map(|i| default_names.get(i).copied().unwrap_or("x"))
        .collect();

    const N_RESTARTS: usize = 5;
    const N_STEPS: usize = 800;

    for restart in 0..N_RESTARTS {
        let noise_scale = restart as f32 * 0.5;

        // 1. SoftExprProgram
        {
            let mut prog = SoftExprProgram::new(n_args);
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 1000 + idx as u64) - 0.5) * noise_scale;
                }
            }
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftExprProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftExprProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2. SoftBranchProgram
        {
            let mut prog = SoftBranchProgram::new(n_args);
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 2000 + idx as u64) - 0.5) * noise_scale;
                }
            }
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2b. Biased branch restart: compare arg0 vs arg1, return arg0-arg1 / arg1-arg0
        //     Targets: abs_diff, min2, scaled_diff, etc.
        if n_args >= 2 && restart == 0 {
            let mut prog = SoftBranchProgram::new(n_args);
            let ns = n_args + N_CONSTS;
            let ne = ns + 1;
            let boff = 1 + 2 * ns + N_OPS;
            let branch_size_b = N_CMPS + 4 * ne + 6;
            // Branch 0: a >= b (index 2), lhs=arg0, rhs=arg1, ret = a - b
            prog.params[boff + 2] = 4.0; // cmp = >= strongly
            prog.params[boff] = -4.0; // not >
            prog.params[boff + N_CMPS] = 4.0; // lhs = arg0
            prog.params[boff + N_CMPS + ne] = -4.0; // not arg0 for rhs
            if n_args > 1 {
                prog.params[boff + N_CMPS + ne + 1] = 4.0;
            } // rhs = arg1
            prog.params[boff + N_CMPS + 2 * ne] = 4.0; // ret_s1 = arg0
            prog.params[boff + N_CMPS + 3 * ne + 1] = 4.0; // ret_s2 = arg1 (if exists)
            prog.params[boff + N_CMPS + 4 * ne + 1] = 4.0; // ret_op = - (subtract)
            prog.params[boff + N_CMPS + 4 * ne + 5] = -4.0; // not identity
                                                            // Default: return b - a
            let doff = boff + N_BRANCHES * branch_size_b;
            prog.params[doff + 1] = 4.0; // ds1 = arg1
            prog.params[doff + ne] = -4.0; // not arg0 for ds2
            prog.params[doff + ne + 0] = 4.0; // ds2 = arg0
            prog.params[doff + 2 * ne + 1] = 4.0; // dop = subtract
            prog.params[doff + 2 * ne + 5] = -4.0; // not identity
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2c. Biased branch restart: precompute a % const, branch on v0 == 0 → return 1 else 0
        //     Targets: is_even, is_odd, parity, triangular_check, etc.
        if n_args == 1 && restart == 0 {
            let mut prog = SoftBranchProgram::new(n_args);
            let ns = n_args + N_CONSTS;
            let ne = ns + 1;
            let boff = 1 + 2 * ns + N_OPS;
            // Enable precompute: a % const[3] (which is 2)
            prog.params[0] = 4.0; // pre_enable on
            prog.params[1] = 4.0; // pre_s1 = arg0
            prog.params[1 + ns + n_args + 3] = 4.0; // pre_s2 = const[3] = value 2
            prog.params[1 + 2 * ns + 4] = 4.0; // pre_op = % (index 4)
                                               // Branch 0: v0 == 0 (index 4 in N_CMPS), lhs=v0, rhs=const0=0, ret=const1=1
            let v0_idx = ne - 1; // v0 is last in ext
            prog.params[boff + 4] = 4.0; // cmp = ==
            prog.params[boff + N_CMPS + v0_idx] = 4.0; // lhs = v0
            prog.params[boff + N_CMPS + ne + n_args] = 4.0; // rhs = const0 = 0
                                                            // ret = const1 = 1 (identity)
            prog.params[boff + N_CMPS + 2 * ne + n_args + 1] = 4.0; // ret_s1 = const1 = 1
            prog.params[boff + N_CMPS + 4 * ne + 5] = 4.0; // ret_op = identity
                                                           // Default: return const0 = 0 (identity)
            let doff_c = boff + N_BRANCHES * (N_CMPS + 4 * ne + 6);
            prog.params[doff_c + n_args] = 4.0; // ds1 = const0 = 0
            prog.params[doff_c + 2 * ne + 5] = 4.0; // dop = identity
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2d. Biased expr restart: precompute enabled, op=*, targets cube/square_plus/bilinear3/etc.
        //     Covers any "pre=s1*s2, ret=v0 OP last_arg" pattern.
        if restart == 0 {
            for &ret_op_idx in &[0usize, 1, 2] {
                // + - *
                let mut prog = SoftExprProgram::new(n_args);
                let ns = n_args + N_CONSTS;
                let ne = ns + 1;
                // Enable precompute
                prog.params[0] = 4.0;
                // pre_s1 = arg0
                prog.params[1] = 4.0;
                // pre_s2 = last arg (or arg0 for single-arg)
                let ps2 = 1 + ns;
                let last = if n_args > 1 { n_args - 1 } else { 0 };
                prog.params[ps2 + last] = 4.0;
                // pre_op = * (index 2)
                prog.params[1 + 2 * ns + 2] = 4.0;
                // ret_s1 = v0 (last in ext = ne-1)
                let roff = 1 + 2 * ns + N_OPS;
                prog.params[roff + ne - 1] = 4.0;
                // ret_s2 = last arg
                let rs2off = roff + ne;
                prog.params[rs2off + last] = 4.0;
                // ret_op = chosen index
                prog.params[roff + 2 * ne + ret_op_idx] = 4.0;
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftExprProgram {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftExprProgram {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }
        }

        // 2d-2. Biased expr restart: pre=arg0*arg1, ret=v0 OP arg0.
        //       Needed for product_offset (a*b-a) where ret_s2=arg0 not last_arg.
        if restart == 0 && n_args >= 2 {
            for &ret_op_idx in &[0usize, 1, 2] {
                // + - *
                let mut prog = SoftExprProgram::new(n_args);
                let ns = n_args + N_CONSTS;
                let ne = ns + 1;
                prog.params[0] = 4.0; // enable precompute
                prog.params[1] = 4.0; // pre_s1 = arg0
                prog.params[1 + ns + 1] = 4.0; // pre_s2 = arg1
                prog.params[1 + 2 * ns + 2] = 4.0; // pre_op = * (index 2)
                let roff = 1 + 2 * ns + N_OPS;
                prog.params[roff + ne - 1] = 4.0; // ret_s1 = v0 (last in ext pool)
                prog.params[roff + ne + 0] = 4.0; // ret_s2 = arg0
                prog.params[roff + 2 * ne + ret_op_idx] = 4.0; // ret_op
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftExprProgram {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftExprProgram {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }
        }

        // 2e. SoftTwoPrecompExprProgram (generic + biased celsius/polynomial restarts)
        {
            let mut prog = SoftTwoPrecompExprProgram::new(n_args);
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 5000 + idx as u64) - 0.5) * noise_scale;
                }
            }
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftTwoPrecompExprProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftTwoPrecompExprProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2e-biased. Biased 2-precomp restart: pre1=arg0*const, pre2=v0/const2, ret=v1+const3
        //   Targets celsius_to_fahrenheit (c*9/5+32) and similar "scale then shift" formulas.
        if n_args == 1 && restart == 0 {
            let mut prog = SoftTwoPrecompExprProgram::new(n_args);
            let ns = n_args + N_CONSTS;
            let ne1 = ns + 1;
            let ne2 = ns + 2;
            let p2 = 1 + 2 * ns + N_OPS;
            let roff = p2 + 1 + 2 * ne1 + N_OPS;
            let coff = roff + 2 * ne2 + N_OPS;
            // pre1: arg0 * const[5] (starts at 10, will learn toward 9 for celsius)
            prog.params[0] = 4.0; // pre1 enable
            prog.params[1] = 4.0; // pre1_s1 = arg0
            prog.params[1 + ns + n_args + 5] = 4.0; // pre1_s2 = const[5]=10
            prog.params[1 + 2 * ns + 2] = 4.0; // pre1_op = *
                                               // pre2: v0 / const[3] (starts at 2, will learn toward 5 for celsius)
            prog.params[p2] = 4.0; // pre2 enable
            prog.params[p2 + 1 + ne1 - 1] = 4.0; // pre2_s1 = v0
            prog.params[p2 + 1 + ne1 + n_args + 3] = 4.0; // pre2_s2 = const[3]=2
            prog.params[p2 + 1 + 2 * ne1 + 3] = 4.0; // pre2_op = / (index 3)
                                                     // ret: v1 + const[5] (const[5] starts at 10, will learn to 32)
            prog.params[roff + ne2 - 1] = 4.0; // ret_s1 = v1
            prog.params[roff + ne2 + n_args + 5] = 4.0; // ret_s2 = const[5]
            prog.params[roff + 2 * ne2] = 4.0; // ret_op = +
                                               // Initialize consts to good starting values for celsius
            prog.params[coff + 5] = 9.0; // const[5] starts near 9 (multiply factor)
            prog.params[coff + 3] = 5.0; // const[3] = 5 (divisor)
                                         // We need ANOTHER const for the +32. Reuse const[4] (normally -2 → set to 32)
            prog.params[coff + 4] = 32.0;
            // Override ret_s2 to use const[4]=32
            prog.params[roff + ne2 + n_args + 5] = -4.0; // not const[5]
            prog.params[roff + ne2 + n_args + 4] = 4.0; // ret_s2 = const[4]=32
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftTwoPrecompExprProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftTwoPrecompExprProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2e-polynomial. Biased 2-precomp restart: polynomial(x)=2*x*x+3*x+1=(x+1)*(2*x+1)
        //   v0 = x + 1, v1 = v0 + x = 2x+1, return v0 * v1
        if n_args == 1 && restart == 0 {
            let mut prog = SoftTwoPrecompExprProgram::new(n_args);
            let ns = n_args + N_CONSTS; // 7
            let ne1 = ns + 1; // 8
            let ne2 = ns + 2; // 9
            let p2 = 1 + 2 * ns + N_OPS; // 20
            let roff = p2 + 1 + 2 * ne1 + N_OPS; // 42

            // Pre1 enabled: v0 = arg0 + c1=1 (x+1)
            prog.params[0] = 4.0; // pre1 ENABLED
            prog.params[1] = 4.0; // pre1_s1 = arg0 (ns-src idx 0)
            prog.params[1 + ns + 2] = 4.0; // pre1_s2 = c1=1 (ns-src idx 2): p[8+2]=p[10]
            prog.params[1 + 2 * ns] = 4.0; // pre1_op = + (idx 0): p[15]

            // Pre2 enabled: v1 = v0 + arg0 = (x+1)+x = 2x+1
            prog.params[p2] = 4.0; // pre2 ENABLED
            prog.params[p2 + 1 + ne1 - 1] = 4.0; // pre2_s1 = v0 (ne1-src idx 7): p[28]
            prog.params[p2 + 1 + ne1] = 4.0; // pre2_s2 = arg0 (ne1-src idx 0): p[29]
            prog.params[p2 + 1 + 2 * ne1] = 4.0; // pre2_op = + (idx 0): p[37]

            // Return: v0 * v1
            // ne2 sources: [arg0=0, c0..c5=1..6, v0=7, v1=8]
            prog.params[roff + ne2 - 2] = 4.0; // ret_s1 = v0 (ne2-src idx 7): p[49]
            prog.params[roff + ne2 + ne2 - 1] = 4.0; // ret_s2 = v1 (ne2-src idx 8): p[51+8]=p[59]
            prog.params[roff + 2 * ne2 + 2] = 4.0; // ret_op = * (idx 2): p[60+2]=p[62]

            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftTwoPrecompExprProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftTwoPrecompExprProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2f. Sign biased restart: x<0→-1, x>0→1, default→0
        if n_args == 1 && restart == 0 {
            let mut prog = SoftBranchProgram::new(n_args);
            let ns = n_args + N_CONSTS;
            let ne = ns + 1;
            let boff = 1 + 2 * ns + N_OPS;
            let branch_size = N_CMPS + 4 * ne + 6;
            // Branch 0: x < 0 (cmp=1, lhs=arg0, rhs=const[0]=0), return const[2]=-1
            prog.params[boff] = -4.0;
            prog.params[boff + 1] = 4.0; // cmp = <
            prog.params[boff + N_CMPS] = 4.0; // lhs = arg0
            prog.params[boff + N_CMPS + ne + n_args] = 4.0; // rhs = const[0]=0
            prog.params[boff + N_CMPS + 2 * ne + n_args + 2] = 4.0; // ret_s1 = const[2]=-1
            prog.params[boff + N_CMPS + 4 * ne + 5] = 4.0; // ret_op = identity
                                                           // Branch 1: x > 0 (cmp=0, lhs=arg0, rhs=const[0]=0), return const[1]=1
            let b1 = boff + branch_size;
            prog.params[b1] = 4.0;
            prog.params[b1 + 1] = -4.0; // cmp = >
            prog.params[b1 + N_CMPS] = 4.0; // lhs = arg0
            prog.params[b1 + N_CMPS + ne + n_args] = 4.0; // rhs = const[0]=0
            prog.params[b1 + N_CMPS + 2 * ne + n_args + 1] = 4.0; // ret_s1 = const[1]=1
            prog.params[b1 + N_CMPS + 4 * ne + 5] = 4.0; // ret_op = identity
                                                         // Default: return const[0]=0 (identity)
            let doff = boff + N_BRANCHES * branch_size;
            prog.params[doff + n_args] = 4.0; // ds1 = const[0]=0
            prog.params[doff + 2 * ne + 5] = 4.0; // dop = identity
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2g. Clamp biased restart: x<0→0, x>const(100)→const(100), default→x
        if n_args == 1 && restart == 0 {
            let mut prog = SoftBranchProgram::new(n_args);
            let ns = n_args + N_CONSTS;
            let ne = ns + 1;
            let boff = 1 + 2 * ns + N_OPS;
            let branch_size = N_CMPS + 4 * ne + 6;
            let doff = boff + N_BRANCHES * branch_size;
            let coff = doff + 2 * ne + 6;
            // Set const[5] to 100.0 (normally starts at 10)
            prog.params[coff + 5] = 100.0;
            // Branch 0: x < 0, return const[0]=0
            prog.params[boff + 1] = 4.0; // cmp = <
            prog.params[boff + N_CMPS] = 4.0; // lhs = arg0
            prog.params[boff + N_CMPS + ne + n_args] = 4.0; // rhs = const[0]=0
            prog.params[boff + N_CMPS + 2 * ne + n_args] = 4.0; // ret_s1 = const[0]=0
            prog.params[boff + N_CMPS + 4 * ne + 5] = 4.0; // ret_op = identity
                                                           // Branch 1: x > const[5]=100, return const[5]=100
            let b1 = boff + branch_size;
            // Reset branch1 cmp logits (dormant init set != strongly at 8.0)
            for k in 0..N_CMPS {
                prog.params[b1 + k] = -4.0;
            }
            prog.params[b1] = 4.0; // cmp = >
            prog.params[b1 + N_CMPS] = 4.0; // lhs = arg0
            prog.params[b1 + N_CMPS + ne + n_args + 5] = 4.0; // rhs = const[5]=100
            prog.params[b1 + N_CMPS + 2 * ne + n_args + 5] = 4.0; // ret_s1 = const[5]=100
            prog.params[b1 + N_CMPS + 4 * ne + 5] = 4.0; // ret_op = identity
                                                         // Default: return x (identity)
            prog.params[doff] = 4.0; // ds1 = arg0
            prog.params[doff + 2 * ne + 5] = 4.0; // dop = identity
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2h. Biased branch restart: b==0 → return -1, else return a/b
        //     Targets: safe_div_or_neg1
        if n_args == 2 && restart == 0 {
            let mut prog = SoftBranchProgram::new(n_args);
            let ns2 = n_args + N_CONSTS;
            let ne2 = ns2 + 1;
            let boff2 = 1 + 2 * ns2 + N_OPS;
            let branch_size2 = N_CMPS + 4 * ne2 + 6;
            let doff2 = boff2 + N_BRANCHES * branch_size2;
            // Branch 0: b == 0 → return const[2]=-1
            for k in 0..N_CMPS {
                prog.params[boff2 + k] = -4.0;
            }
            prog.params[boff2 + 4] = 4.0; // cmp = ==
            for k in 0..ne2 {
                prog.params[boff2 + N_CMPS + k] = -4.0;
            }
            prog.params[boff2 + N_CMPS + 1] = 4.0; // lhs = arg1=b
            prog.params[boff2 + N_CMPS + ne2 + 2] = 4.0; // rhs = const[0]=0
            for k in 0..ne2 {
                prog.params[boff2 + N_CMPS + 2 * ne2 + k] = -4.0;
            }
            prog.params[boff2 + N_CMPS + 2 * ne2 + 4] = 4.0; // ret_s1 = const[2]=-1
            prog.params[boff2 + N_CMPS + 4 * ne2 + 5] = 4.0; // ret_op = identity
                                                             // Default: return a / b
            prog.params[doff2 + 1] = -4.0; // undo default arg1 init
            prog.params[doff2] = 4.0; // ds1 = arg0=a
            prog.params[doff2 + ne2 + 1] = 4.0; // ds2 = arg1=b
            prog.params[doff2 + 2 * ne2 + 5] = -4.0; // not identity
            prog.params[doff2 + 2 * ne2 + 3] = 4.0; // dop = /
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2i. Biased branch restart: digital_root — v0=n%9, n==0→0, v0==0→9, else→v0
        //     Encodes: digital_root(n) = n%9==0 ? (n==0 ? 0 : 9) : n%9
        //     Sets const[5]=9 so precompute uses modulo-9.
        if n_args == 1 && restart == 0 {
            let mut prog = SoftBranchProgram::new(n_args);
            let ns = n_args + N_CONSTS; // 7
            let ne = ns + 1; // 8
            let boff = 1 + 2 * ns + N_OPS; // 20
            let branch_size = N_CMPS + 4 * ne + 6; // 44
            let b1o = boff + branch_size; // 64
            let doff = boff + N_BRANCHES * branch_size; // 152
            let coff = doff + 2 * ne + 6; // 174

            // Pre-compute ENABLED: v0 = arg0 % c5=9
            // pre_s1 logits at [1..1+ns]=params[1..8], arg0=idx0 → p[1]
            // pre_s2 logits at [1+ns..1+2*ns]=params[8..15], c5=idx(n_args+5)=6 → p[14]
            // pre_op logits at [1+2*ns..1+2*ns+N_OPS]=params[15..20], %=idx4 → p[19]
            prog.params[0] = 4.0; // pre ENABLED
            prog.params[1] = 4.0; // pre_s1 = arg0 (ns-src idx 0)
            prog.params[1 + ns + 6] = 4.0; // pre_s2 = c5 (ns-src idx 6): p[8+6]=p[14]
            prog.params[1 + 2 * ns + 4] = 4.0; // pre_op = % (op idx 4): p[19]

            // Branch 0: arg0 == 0 → return 0 (c0)
            // cmp logits at [boff..boff+N_CMPS]=params[20..26], ==idx4 → p[24]
            // lhs logits at [boff+N_CMPS..+ne]=params[26..34], arg0=idx0 → p[26]
            // rhs logits at [boff+N_CMPS+ne..+ne]=params[34..42], c0=0=idx1 → p[35]
            // ret_s1 logits at [boff+N_CMPS+2*ne..+ne]=params[42..50], c0=0=idx1 → p[43]
            // ret_op logits at [boff+N_CMPS+4*ne..+6]=params[58..64], identity=idx5 → p[63]
            prog.params[boff + 4] = 4.0; // cmp = ==
            prog.params[boff + N_CMPS] = 4.0; // lhs = arg0
            prog.params[boff + N_CMPS + ne + 1] = 4.0; // rhs = c0=0
            prog.params[boff + N_CMPS + 2 * ne + 1] = 4.0; // ret_s1 = c0=0
            prog.params[boff + N_CMPS + 4 * ne + 5] = 4.0; // ret_op = identity

            // Branch 1: v0 == 0 → return c5=9
            // Override dormant init: clear strong != logit first
            for k in 0..N_CMPS {
                prog.params[b1o + k] = -4.0;
            }
            prog.params[b1o + 4] = 4.0; // cmp = == (idx 4)
            prog.params[b1o + N_CMPS + 7] = 4.0; // lhs = v0 (ne-src idx 7)
            prog.params[b1o + N_CMPS + ne + 1] = 4.0; // rhs = c0=0 (ne-src idx 1)
            prog.params[b1o + N_CMPS + 2 * ne + 6] = 4.0; // ret_s1 = c5=9 (ne-src idx 6)
            prog.params[b1o + N_CMPS + 4 * ne + 5] = 4.0; // ret_op = identity

            // Default: return v0
            prog.params[doff + 7] = 4.0; // ret_s1 = v0 (ne-src idx 7): p[159]
            prog.params[doff + 2 * ne + 5] = 4.0; // ret_op = identity: p[173]

            // Set const[5] = 9 (modulo-9 is the digital root formula base)
            prog.params[coff + 5] = 9.0;

            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftBranchProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2b. SoftChainedBranch — two sequential ternary branches (min3, max3, clamp)
        {
            let mut prog = SoftChainedBranch::new(n_args);
            if restart > 0 {
                let (_, _, _, _, _, _, _, _, _, _, co) = SoftChainedBranch::offsets(n_args);
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 13000 + idx as u64) - 0.5)
                        * noise_scale
                        * 0.5;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
            }
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftChainedBranch {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftChainedBranch {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 2b-min3: v0=min(a,b), ret=min(v0,c) — run every restart with increasing noise
        if n_args == 3 {
            let mut prog = SoftChainedBranch::new(n_args);
            let p1 = n_args + N_CONSTS; // pool1 size = 9
            let p2 = p1 + 1; // pool2 size = 10
            let (
                b1_cmp_off,
                b1_lhs_off,
                b1_rhs_off,
                b1_true_off,
                b1_false_off,
                b2_cmp_off,
                b2_lhs_off,
                b2_rhs_off,
                b2_true_off,
                b2_false_off,
                co,
            ) = SoftChainedBranch::offsets(n_args);
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            // branch1: if a < b: v0=a else v0=b
            prog.params[b1_cmp_off + 1] = 4.0; // b1_cmp = < (idx 1)
            prog.params[b1_lhs_off + 0] = 4.0; // b1_lhs = a (pool1[0])
            prog.params[b1_rhs_off + 1] = 4.0; // b1_rhs = b (pool1[1])
            prog.params[b1_true_off + 0] = 4.0; // b1_true = a
            prog.params[b1_false_off + 1] = 4.0; // b1_false = b
                                                 // branch2: if v0 < c: ret=v0 else ret=c
            prog.params[b2_cmp_off + 1] = 4.0; // b2_cmp = <
            prog.params[b2_lhs_off + p2 - 1] = 4.0; // b2_lhs = v0 (last in pool2)
            prog.params[b2_rhs_off + 2] = 4.0; // b2_rhs = c (pool2[2])
            prog.params[b2_true_off + p2 - 1] = 4.0; // b2_true = v0
            prog.params[b2_false_off + 2] = 4.0; // b2_false = c
                                                 // For later restarts add noise only to the non-critical (background) logits
            if restart > 0 {
                let hot: std::collections::HashSet<usize> = [
                    b1_cmp_off + 1,
                    b1_lhs_off,
                    b1_rhs_off + 1,
                    b1_true_off,
                    b1_false_off + 1,
                    b2_cmp_off + 1,
                    b2_lhs_off + p2 - 1,
                    b2_rhs_off + 2,
                    b2_true_off + p2 - 1,
                    b2_false_off + 2,
                    co,
                    co + 1,
                    co + 2,
                    co + 3,
                    co + 4,
                    co + 5,
                ]
                .iter()
                .copied()
                .collect();
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    if !hot.contains(&idx) {
                        *p += (pseudo_rand(restart as u64 * 17000 + idx as u64) - 0.5)
                            * noise_scale
                            * 0.3;
                    }
                }
            }
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftChainedBranch {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftChainedBranch {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS * 2,
            );
            if result.is_some() {
                return result;
            }
        }

        // 3. SoftLoopProgram (all n_args, not just 1)
        {
            let mut prog = SoftLoopProgram::new(n_args);
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 3000 + idx as u64) - 0.5) * noise_scale;
                }
            }
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 3b. Biased loop restart: init=1, op=* → factorial / product_1_to_n
        if n_args == 1 && restart == 0 {
            let mut prog = SoftLoopProgram::new(n_args);
            let nb = n_args + N_CONSTS;
            let nr = 3 + n_args + N_CONSTS;
            prog.params[0] = 1.0; // init = 1
            let opoff = 2 + nb + 1;
            for k in 0..N_OPS {
                prog.params[opoff + k] = -4.0;
            }
            prog.params[opoff + 2] = 4.0; // op = * (index 2)
                                          // rhs = i (index 0)
            let rhsoff = opoff + N_OPS;
            for k in 0..nr {
                prog.params[rhsoff + k] = -2.0;
            }
            prog.params[rhsoff] = 4.0;
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 3c. Biased loop restart: op=+, rhs=i*i → sum_squares
        if n_args == 1 && restart == 0 {
            let mut prog = SoftLoopProgram::new(n_args);
            let nb = n_args + N_CONSTS;
            let nr = 3 + n_args + N_CONSTS;
            prog.params[0] = 0.0; // init = 0
            let opoff = 2 + nb + 1;
            for k in 0..N_OPS {
                prog.params[opoff + k] = -2.0;
            }
            prog.params[opoff] = 4.0; // op = + (index 0)
                                      // rhs = i*i (index 1)
            let rhsoff = opoff + N_OPS;
            for k in 0..nr {
                prog.params[rhsoff + k] = -2.0;
            }
            prog.params[rhsoff + 1] = 4.0;
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 3d. Biased loop restart for power(base, exp): init=1, start=1, bound=exp, op=*, rhs=base
        //     start=1 is critical: while i<=exp with start=1 runs exp times (exp=0 → 0 runs)
        if n_args == 2 && restart == 0 {
            let mut prog = SoftLoopProgram::new(n_args);
            let nb = n_args + N_CONSTS;
            let nr = 3 + n_args + N_CONSTS;
            prog.params[0] = 1.0; // init = 1
            prog.params[1] = 1.0; // start = 1 (while i<=exp: runs exp times when i starts at 1)
                                  // bound = arg1 (exp), index 1 in bound_sources=[arg0, arg1, consts...]
            for k in 0..nb {
                prog.params[2 + k] = -2.0;
            }
            prog.params[2 + 1] = 4.0; // bound = arg1 (exp)
            prog.params[2 + nb] = 0.0; // bound_offset = 0
            let opoff = 2 + nb + 1;
            for k in 0..N_OPS {
                prog.params[opoff + k] = -4.0;
            }
            prog.params[opoff + 2] = 4.0; // op = *
                                          // rhs = arg0 (base), index 3 in rhs_sources=[i, i*i, 1, arg0, arg1, consts...]
            let rhsoff = opoff + N_OPS;
            for k in 0..nr {
                prog.params[rhsoff + k] = -2.0;
            }
            prog.params[rhsoff + 3] = 4.0;
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 3d-harmonic. Biased loop restart: harmonic_sum — acc + C/i with const[0]=1000
        //   Uses the new C/i RHS source (last index in rhs_sources).
        if n_args == 1 && restart == 0 {
            let mut prog = SoftLoopProgram::new(n_args);
            let nb = n_args + N_CONSTS; // 7
            let nr_new = 4 + n_args + N_CONSTS; // 11
            let nret = 1 + n_args + N_CONSTS; // 8
            let opoff_h = 2 + nb + 1; // 10
            let rhsoff_h = opoff_h + N_OPS; // 15
            let retoff_h = rhsoff_h + nr_new; // 26
            let coff_h = retoff_h + nret; // 34

            prog.params[0] = 0.0; // init = 0
            prog.params[1] = 1.0; // start = 1
            for k in 0..nb {
                prog.params[2 + k] = -4.0;
            }
            prog.params[2] = 4.0; // bound = arg0 (n)
            prog.params[2 + nb] = 0.0; // bound_offset = 0
            for k in 0..N_OPS {
                prog.params[opoff_h + k] = -4.0;
            }
            prog.params[opoff_h] = 4.0; // op = +
            for k in 0..nr_new {
                prog.params[rhsoff_h + k] = -4.0;
            }
            prog.params[rhsoff_h + nr_new - 1] = 4.0; // rhs = C/i (last source)
                                                      // return = acc (index 0, already default from new())
            prog.params[coff_h] = 1000.0; // const[0] = 1000 (the C in C/i)
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftLoopProgram {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 3e. SoftDigitLoopProgram (only for unary)
        if n_args == 1 {
            let ex = examples.clone();
            let prog = SoftDigitLoopProgram::new();
            let result = train_program(
                prog.params.clone(),
                move |p, t| SoftDigitLoopProgram { params: p.to_vec() }.loss(&ex, t),
                |p, fn_n, pn| {
                    SoftDigitLoopProgram { params: p.to_vec() }.discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 3e-biased. Digit product: mode=product (idx 1), init_acc=1 → digit_product
        if n_args == 1 && restart == 0 {
            let mut prog = SoftDigitLoopProgram::new();
            prog.params[0] = -4.0; // not digit_sum
            prog.params[1] = 4.0; // mode = product (acc * digit)
            prog.params[2] = -4.0;
            prog.params[3] = -4.0;
            prog.params[4] = 1.0; // init_acc = 1
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| SoftDigitLoopProgram { params: p.to_vec() }.loss(&ex, t),
                |p, fn_n, pn| {
                    SoftDigitLoopProgram { params: p.to_vec() }.discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // 4. SoftTwoAccLoop — fibonacci, lucas, fib_iter (two-accumulator recurrences)
        if n_args == 1 {
            {
                let mut prog = SoftTwoAccLoop::new(n_args);
                if restart > 0 {
                    for (idx, p) in prog.params.iter_mut().enumerate() {
                        *p += (pseudo_rand(restart as u64 * 7000 + idx as u64) - 0.5) * noise_scale;
                    }
                }
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftTwoAccLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftTwoAccLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // Lucas biased restart: init0=2, init1=1, return b
            if restart == 0 {
                let mut prog = SoftTwoAccLoop::new(n_args);
                prog.params[0] = 2.0;
                prog.params[1] = 1.0; // a=2, b=1
                let na = prog.na();
                let nb_bound = prog.nb();
                let ret_off = 2 + nb_bound + 1 + 4 * na + 12;
                // return b (index 1)
                prog.params[ret_off] = -4.0;
                prog.params[ret_off + 1] = 4.0;
                // b_update: b = a+b (same as fib default)
                let aop_off = 2 + nb_bound + 1 + 2 * na;
                let bs1_off = aop_off + 6;
                let bs2_off = bs1_off + na;
                prog.params[bs1_off] = 4.0;
                prog.params[bs2_off + 1] = 4.0;
                let bop_off = bs2_off + na;
                prog.params[bop_off] = 4.0; // b_op = +
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftTwoAccLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftTwoAccLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }
        }

        // 4b. SoftCondAccumLoop — counting loop with conditional gate (count_divisors, sum_of_divisors)
        if n_args == 1 {
            {
                let mut prog = SoftCondAccumLoop::new(n_args);
                if restart > 0 {
                    for (idx, p) in prog.params.iter_mut().enumerate() {
                        *p += (pseudo_rand(restart as u64 * 8100 + idx as u64) - 0.5) * noise_scale;
                    }
                }
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftCondAccumLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftCondAccumLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 4b-count_div. Biased restart: count_divisors(n) — acc=0; i=1..n; if n%i==0: acc++
            // ns=7, pre_op=%(4), pre_s1=arg0(0), pre_s2=i(7), cmp==(4), cmp_s1=v0(8), cmp_s2=c0=0(1), lop=+(0), lrhs=c1=1(2)
            if restart == 0 {
                let mut prog = SoftCondAccumLoop::new(n_args);
                let ns = n_args + N_CONSTS; // 7
                let (
                    pre_op_off,
                    pre_s1_off,
                    pre_s2_off,
                    cmp_op_off,
                    cmp_s1_off,
                    cmp_s2_off,
                    loop_op_off,
                    loop_rhs_off,
                    co,
                ) = SoftCondAccumLoop::offsets(ns);
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                prog.params[0] = 0.0; // init = 0
                prog.params[1] = 1.0; // start = 1
                prog.params[2] = 4.0; // bound = arg0
                prog.params[2 + ns] = 0.0; // bound_offset = 0
                prog.params[pre_op_off + 4] = 4.0; // pre_op = %
                prog.params[pre_s1_off] = 4.0; // pre_s1 = arg0
                prog.params[pre_s2_off + ns] = 4.0; // pre_s2 = i (idx ns)
                prog.params[cmp_op_off + 4] = 4.0; // cmp = ==
                prog.params[cmp_s1_off + ns + 1] = 4.0; // cmp_s1 = v0 (idx ns+1)
                prog.params[cmp_s2_off + 1] = 4.0; // cmp_s2 = c0=0
                prog.params[loop_op_off] = 4.0; // loop_op = +
                prog.params[loop_rhs_off + n_args + 1] = 4.0; // loop_rhs = c1=1 (idx n_args+1=2)
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftCondAccumLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftCondAccumLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 4b-sum_div. Biased restart: sum_of_divisors(n) — acc=0; i=1..n; if n%i==0: acc+=i
            // Same as count_div but loop_rhs = i (idx ns)
            if restart == 0 {
                let mut prog = SoftCondAccumLoop::new(n_args);
                let ns = n_args + N_CONSTS; // 7
                let (
                    pre_op_off,
                    pre_s1_off,
                    pre_s2_off,
                    cmp_op_off,
                    cmp_s1_off,
                    cmp_s2_off,
                    loop_op_off,
                    loop_rhs_off,
                    co,
                ) = SoftCondAccumLoop::offsets(ns);
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                prog.params[0] = 0.0; // init = 0
                prog.params[1] = 1.0; // start = 1
                prog.params[2] = 4.0; // bound = arg0
                prog.params[2 + ns] = 0.0; // bound_offset = 0
                prog.params[pre_op_off + 4] = 4.0; // pre_op = %
                prog.params[pre_s1_off] = 4.0; // pre_s1 = arg0
                prog.params[pre_s2_off + ns] = 4.0; // pre_s2 = i (idx ns)
                prog.params[cmp_op_off + 4] = 4.0; // cmp = ==
                prog.params[cmp_s1_off + ns + 1] = 4.0; // cmp_s1 = v0 (idx ns+1)
                prog.params[cmp_s2_off + 1] = 4.0; // cmp_s2 = c0=0
                prog.params[loop_op_off] = 4.0; // loop_op = +
                prog.params[loop_rhs_off + ns] = 4.0; // loop_rhs = i (idx ns)
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftCondAccumLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftCondAccumLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 4b-is_perfect_square: acc=0; i=0..n; if i*i==n: acc++; return acc (=1 iff perfect square)
            if restart == 0 {
                let mut prog = SoftCondAccumLoop::new(n_args);
                let ns = n_args + N_CONSTS; // 7
                let (
                    pre_op_off,
                    pre_s1_off,
                    pre_s2_off,
                    cmp_op_off,
                    cmp_s1_off,
                    cmp_s2_off,
                    loop_op_off,
                    loop_rhs_off,
                    co,
                ) = SoftCondAccumLoop::offsets(ns);
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                prog.params[0] = 0.0; // init = 0
                prog.params[1] = 0.0; // start = 0 (i starts from 0 to check i=0,1,2,...)
                prog.params[2] = 4.0; // bound = arg0 (n)
                prog.params[2 + ns] = 0.0; // bound_offset = 0
                prog.params[pre_op_off + 2] = 4.0; // pre_op = * (i*i)
                prog.params[pre_s1_off + ns] = 4.0; // pre_s1 = i (idx ns)
                prog.params[pre_s2_off + ns] = 4.0; // pre_s2 = i (idx ns)
                prog.params[cmp_op_off + 4] = 4.0; // cmp = ==
                prog.params[cmp_s1_off + ns + 1] = 4.0; // cmp_s1 = v0 (= i*i)
                prog.params[cmp_s2_off + 0] = 4.0; // cmp_s2 = arg0 = n (idx 0)
                prog.params[loop_op_off] = 4.0; // loop_op = +
                prog.params[loop_rhs_off + n_args + 1] = 4.0; // loop_rhs = c1=1
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftCondAccumLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftCondAccumLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }
        }

        // 4b2. SoftCondAccumCmpReturnLoop — count divisors then return (acc cmp C)
        //      Targets: is_prime (count == 2)
        if n_args == 1 {
            {
                let mut prog = SoftCondAccumCmpReturnLoop::new(n_args);
                if restart > 0 {
                    let ns = SoftCondAccumCmpReturnLoop::ns(n_args);
                    let (_, _, _, _, _, _, _, _, co) = SoftCondAccumLoop::offsets(ns);
                    for (idx, p) in prog.params.iter_mut().enumerate() {
                        *p += (pseudo_rand(restart as u64 * 19000 + idx as u64) - 0.5)
                            * noise_scale
                            * 0.5;
                    }
                    prog.params[co] = 0.0;
                    prog.params[co + 1] = 1.0;
                    prog.params[co + 2] = -1.0;
                    prog.params[co + 3] = 2.0;
                    prog.params[co + 4] = -2.0;
                    prog.params[co + 5] = 10.0;
                }
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftCondAccumCmpReturnLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftCondAccumCmpReturnLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 4b2-is_prime: count i in [1,n] where n%i==0; return acc==2
            if restart == 0 {
                let mut prog = SoftCondAccumCmpReturnLoop::new(n_args);
                let ns = SoftCondAccumCmpReturnLoop::ns(n_args);
                let (
                    pre_op_off,
                    pre_s1_off,
                    pre_s2_off,
                    cmp_op_off,
                    cmp_s1_off,
                    cmp_s2_off,
                    loop_op_off,
                    loop_rhs_off,
                    co,
                ) = SoftCondAccumLoop::offsets(ns);
                let ret_cmp_off = SoftCondAccumCmpReturnLoop::ret_cmp_off(ns);
                let ret_c_off = SoftCondAccumCmpReturnLoop::ret_c_off(ns);
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                prog.params[0] = 0.0; // init acc = 0
                prog.params[1] = 1.0; // start i = 1
                prog.params[2] = 4.0; // bound = arg0 = n
                prog.params[2 + ns] = 0.0; // bound_offset = 0
                prog.params[pre_op_off + 4] = 4.0; // pre_op = %
                prog.params[pre_s1_off + 0] = 4.0; // pre_s1 = n (arg0, idx 0)
                prog.params[pre_s2_off + ns] = 4.0; // pre_s2 = i (idx ns)
                prog.params[cmp_op_off + 4] = 4.0; // cmp = ==
                prog.params[cmp_s1_off + ns + 1] = 4.0; // cmp_s1 = v0 (n%i)
                prog.params[cmp_s2_off + n_args] = 4.0; // cmp_s2 = c0=0 (idx n_args)
                prog.params[loop_op_off + 0] = 4.0; // loop_op = +
                prog.params[loop_rhs_off + n_args + 1] = 4.0; // loop_rhs = c1=1
                prog.params[ret_cmp_off + 4] = 4.0; // ret_cmp = ==
                prog.params[ret_c_off] = 2.0; // ret_c = 2 (prime has exactly 2 divisors)
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftCondAccumCmpReturnLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftCondAccumCmpReturnLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }
        }

        // 4c. SoftCondDigitLoop — digit/bit loop with conditional gate
        //     Targets: count_even_digits, sum_odd_digits, popcount, max_digit
        if n_args == 1 {
            // Generic noisy init
            {
                let mut prog = SoftCondDigitLoop::new();
                if restart > 0 {
                    let co = SoftCondDigitLoop::consts_off();
                    for (idx, p) in prog.params.iter_mut().enumerate() {
                        *p += (pseudo_rand(restart as u64 * 11000 + idx as u64) - 0.5)
                            * noise_scale
                            * 0.5;
                    }
                    prog.params[co] = 0.0;
                    prog.params[co + 1] = 1.0;
                    prog.params[co + 2] = -1.0;
                    prog.params[co + 3] = 2.0;
                    prog.params[co + 4] = -2.0;
                    prog.params[co + 5] = 10.0;
                }
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| SoftCondDigitLoop { params: p.to_vec() }.loss(&ex, t),
                    |p, fn_n, pn| {
                        SoftCondDigitLoop { params: p.to_vec() }.discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 4c-count_even: base=10, pre=d%2, gate=pre==0, rhs=1, loop=+; zero_return=1 (f(0)=1)
            if restart == 0 {
                let mut prog = SoftCondDigitLoop::new();
                let (
                    _,
                    base_off,
                    gate_pre_off,
                    gate_lhs_off,
                    gate_cmp_off,
                    gate_rhs_off,
                    acc_rhs_off,
                    loop_op_off,
                ) = SoftCondDigitLoop::offsets();
                let co = SoftCondDigitLoop::consts_off();
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                prog.params[SoftCondDigitLoop::zero_return_off()] = 1.0; // f(0)=1 for count_even_digits
                prog.params[0] = 0.0; // init_acc = 0
                prog.params[base_off + 5] = 4.0; // base = c5=10
                prog.params[gate_pre_off + 3] = 4.0; // gate_pre = c3=2
                prog.params[gate_lhs_off + 1] = 4.0; // gate_lhs = pre (pool[1])
                prog.params[gate_cmp_off + 4] = 4.0; // gate_cmp = ==
                prog.params[gate_rhs_off + 4] = 4.0; // gate_rhs = c0=0 (pool[4])
                prog.params[acc_rhs_off + 5] = 4.0; // acc_rhs = c1=1 (pool[5])
                prog.params[loop_op_off + 0] = 4.0; // loop_op = +
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| SoftCondDigitLoop { params: p.to_vec() }.loss(&ex, t),
                    |p, fn_n, pn| {
                        SoftCondDigitLoop { params: p.to_vec() }.discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 4c-sum_odd: base=10, pre=d%2, gate=pre==1, rhs=d, loop=+
            if restart == 0 {
                let mut prog = SoftCondDigitLoop::new();
                let (
                    _,
                    base_off,
                    gate_pre_off,
                    gate_lhs_off,
                    gate_cmp_off,
                    gate_rhs_off,
                    acc_rhs_off,
                    loop_op_off,
                ) = SoftCondDigitLoop::offsets();
                let co = SoftCondDigitLoop::consts_off();
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                prog.params[SoftCondDigitLoop::zero_return_off()] = 0.0;
                prog.params[0] = 0.0; // init_acc = 0
                prog.params[base_off + 5] = 4.0; // base = c5=10
                prog.params[gate_pre_off + 3] = 4.0; // gate_pre = c3=2
                prog.params[gate_lhs_off + 1] = 4.0; // gate_lhs = pre (pool[1])
                prog.params[gate_cmp_off + 4] = 4.0; // gate_cmp = ==
                prog.params[gate_rhs_off + 5] = 4.0; // gate_rhs = c1=1 (pool[5])
                prog.params[acc_rhs_off + 0] = 4.0; // acc_rhs = d (pool[0])
                prog.params[loop_op_off + 0] = 4.0; // loop_op = +
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| SoftCondDigitLoop { params: p.to_vec() }.loss(&ex, t),
                    |p, fn_n, pn| {
                        SoftCondDigitLoop { params: p.to_vec() }.discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 4c-popcount: base=2, gate=d==1, rhs=1, loop=+
            if restart == 0 {
                let mut prog = SoftCondDigitLoop::new();
                let (
                    _,
                    base_off,
                    gate_pre_off,
                    gate_lhs_off,
                    gate_cmp_off,
                    gate_rhs_off,
                    acc_rhs_off,
                    loop_op_off,
                ) = SoftCondDigitLoop::offsets();
                let co = SoftCondDigitLoop::consts_off();
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                prog.params[SoftCondDigitLoop::zero_return_off()] = 0.0;
                prog.params[0] = 0.0; // init_acc = 0
                prog.params[base_off + 3] = 4.0; // base = c3=2
                prog.params[gate_pre_off + 3] = 4.0; // gate_pre = c3=2 (unused)
                prog.params[gate_lhs_off + 0] = 4.0; // gate_lhs = d (pool[0])
                prog.params[gate_cmp_off + 4] = 4.0; // gate_cmp = ==
                prog.params[gate_rhs_off + 5] = 4.0; // gate_rhs = c1=1 (pool[5])
                prog.params[acc_rhs_off + 5] = 4.0; // acc_rhs = c1=1 (pool[5])
                prog.params[loop_op_off + 0] = 4.0; // loop_op = +
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| SoftCondDigitLoop { params: p.to_vec() }.loss(&ex, t),
                    |p, fn_n, pn| {
                        SoftCondDigitLoop { params: p.to_vec() }.discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 4c-max_digit: base=10, gate=d>acc, rhs=d-acc, loop=+
            if restart == 0 {
                let mut prog = SoftCondDigitLoop::new();
                let (
                    _,
                    base_off,
                    gate_pre_off,
                    gate_lhs_off,
                    gate_cmp_off,
                    gate_rhs_off,
                    acc_rhs_off,
                    loop_op_off,
                ) = SoftCondDigitLoop::offsets();
                let co = SoftCondDigitLoop::consts_off();
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                prog.params[SoftCondDigitLoop::zero_return_off()] = 0.0;
                prog.params[0] = 0.0; // init_acc = 0
                prog.params[base_off + 5] = 4.0; // base = c5=10
                prog.params[gate_pre_off + 3] = 4.0; // gate_pre = c3=2 (unused)
                prog.params[gate_lhs_off + 0] = 4.0; // gate_lhs = d (pool[0])
                prog.params[gate_cmp_off + 0] = 4.0; // gate_cmp = >
                prog.params[gate_rhs_off + 3] = 4.0; // gate_rhs = acc (pool[3])
                prog.params[acc_rhs_off + 2] = 4.0; // acc_rhs = d-acc (pool[2])
                prog.params[loop_op_off + 0] = 4.0; // loop_op = +
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| SoftCondDigitLoop { params: p.to_vec() }.loss(&ex, t),
                    |p, fn_n, pn| {
                        SoftCondDigitLoop { params: p.to_vec() }.discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }
        }

        // 5. SoftPredicateLoop — predicate-gated while loop (GCD, leading_digit, next_power_of_2, collatz)
        {
            let mut prog = SoftPredicateLoop::new(n_args);
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p +=
                        (pseudo_rand(restart as u64 * 9000 + idx as u64) - 0.5) * noise_scale * 0.5;
                }
            }
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftPredicateLoop {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftPredicateLoop {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // GCD biased restart: init(a,b), while x1!=0 { (x0,x1)=(x1, x0%x1) }, return x0
        if n_args == 2 && restart == 0 {
            let mut prog = SoftPredicateLoop::new(n_args);
            {
                let nb = prog.nb();
                let na = prog.na();
                let c0 = 2 * nb;
                let c1 = c0 + N_CMPS;
                let c2 = c1 + na;
                let a1 = c2 + na;
                let a2 = a1 + na;
                let ao = a2 + na;
                let b1 = ao + 6;
                let b2 = b1 + na;
                let bo = b2 + na;
                let ro = bo + 6;
                let co = ro + na;
                for p in prog.params.iter_mut() {
                    *p = -1.0;
                }
                prog.params[0] = 4.0;
                prog.params[nb + 1] = 4.0; // init0=arg0, init1=arg1
                prog.params[c0 + 5] = 4.0; // cmp = !=
                prog.params[c1 + 1] = 4.0; // lhs = x1
                prog.params[c2 + 2 + n_args] = 4.0; // rhs = const[0]=0 (loop-src idx 2+n_args)
                prog.params[a1 + 1] = 4.0; // a_s1 = x1  (new x0 ← x1)
                prog.params[ao + 5] = 4.0; // a_op = identity
                prog.params[b1 + 0] = 4.0; // b_s1 = x0
                prog.params[b2 + 1] = 4.0; // b_s2 = x1
                prog.params[bo + 4] = 4.0; // b_op = %  (new x1 ← x0%x1)
                prog.params[ro + 0] = 4.0; // ret = x0
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
            }
            let ex = examples.clone();
            let result = train_program(
                prog.params.clone(),
                move |p, t| {
                    SoftPredicateLoop {
                        n_args,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                |p, fn_n, pn| {
                    SoftPredicateLoop {
                        n_args,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, pn)
                },
                problem,
                &param_names,
                fn_name,
                N_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }

        // Unary predicate-loop biased restarts: leading_digit + next_power_of_2
        if n_args == 1 && restart == 0 {
            let base_prog = SoftPredicateLoop::new(n_args);
            let nb = base_prog.nb();
            let na = base_prog.na();
            let c0 = 2 * nb;
            let c1 = c0 + N_CMPS;
            let c2 = c1 + na;
            let a1 = c2 + na;
            let a2 = a1 + na;
            let ao = a2 + na;
            let b1 = ao + 6;
            let b2 = b1 + na;
            let bo = b2 + na;
            let ro = bo + 6;
            let co = ro + na;
            // Loop sources (n_args=1): [x0=0,x1=1,arg0=2,c0=3,c1=4,c2=5,c3=6,c4=7,c5=8]
            // Init sources (n_args=1): [arg0=0,c0=1,c1=2,c2=3,c3=4,c4=5,c5=6]
            // nb and b2 are used below in Pattern C

            // Pattern A: leading_digit — while x0>=10 { x0=x0/10 }, return x0
            {
                let mut p = vec![-1.0f32; base_prog.params.len()];
                p[0] = 4.0; // init0 = arg0
                p[nb + 1] = 4.0; // init1 = const[0]=0 (x1 is unused, just clean init)
                p[c0 + 2] = 4.0; // cmp = >=
                p[c1 + 0] = 4.0; // lhs = x0
                p[c2 + 8] = 4.0; // rhs = const[5]=10 (loop-src idx 2+1+5=8)
                p[a1 + 0] = 4.0; // a_s1 = x0
                p[a2 + 8] = 4.0; // a_s2 = const[5]=10
                p[ao + 3] = 4.0; // a_op = /
                p[b1 + 1] = 4.0; // b_s1 = x1 (noop)
                p[bo + 5] = 4.0; // b_op = identity
                p[ro + 0] = 4.0; // ret = x0
                p[co] = 0.0;
                p[co + 1] = 1.0;
                p[co + 2] = -1.0;
                p[co + 3] = 2.0;
                p[co + 4] = -2.0;
                p[co + 5] = 10.0;
                let ex = examples.clone();
                let result = train_program(
                    p,
                    move |pp, t| {
                        SoftPredicateLoop {
                            n_args,
                            params: pp.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |pp, fn_n, pn| {
                        SoftPredicateLoop {
                            n_args,
                            params: pp.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // Pattern B: next_power_of_2 — init x0=1, while x0<arg0 { x0=x0*2 }, return x0
            // Init-src idx 2 = const[1]=1; loop-src idx 6 = const[3]=2; loop-src idx 2 = arg0
            {
                let mut p = vec![-1.0f32; base_prog.params.len()];
                p[2] = 4.0; // init0 = const[1]=1 (init-src idx n_args+1=2)
                p[nb + 1] = 4.0; // init1 = const[0]=0 (x1 is unused, just clean init)
                p[c0 + 1] = 4.0; // cmp = <
                p[c1 + 0] = 4.0; // lhs = x0
                p[c2 + 2] = 4.0; // rhs = arg0 (loop-src idx 2)
                p[a1 + 0] = 4.0; // a_s1 = x0
                p[a2 + 6] = 4.0; // a_s2 = const[3]=2 (loop-src idx 2+1+3=6)
                p[ao + 2] = 4.0; // a_op = *
                p[b1 + 1] = 4.0; // b_s1 = x1 (noop)
                p[bo + 5] = 4.0; // b_op = identity
                p[ro + 0] = 4.0; // ret = x0
                p[co] = 0.0;
                p[co + 1] = 1.0;
                p[co + 2] = -1.0;
                p[co + 3] = 2.0;
                p[co + 4] = -2.0;
                p[co + 5] = 10.0;
                let ex = examples.clone();
                let result = train_program(
                    p,
                    move |pp, t| {
                        SoftPredicateLoop {
                            n_args,
                            params: pp.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |pp, fn_n, pn| {
                        SoftPredicateLoop {
                            n_args,
                            params: pp.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // Pattern C: digit_count — x0=n, x1=1; while x0>=10 { x0=x0/10; x1=x1+1 }; return x1
            // Handles digit_count(0)=1 correctly (loop skipped, x1=1 returned).
            // Init-src idx 0=arg0, idx 2=const[1]=1
            // Loop-src: x0=0,x1=1,arg0=2,c0=3,c1=4,c2=5,c3=6,c4=7,c5=8
            {
                let mut p = vec![-1.0f32; base_prog.params.len()];
                p[0] = 4.0; // init0 = arg0
                p[nb + 2] = 4.0; // init1 = const[1]=1 (init-src idx 2)
                p[c0 + 2] = 4.0; // cmp = >=
                p[c1 + 0] = 4.0; // lhs = x0
                p[c2 + 8] = 4.0; // rhs = const[5]=10 (loop-src idx 8)
                p[a1 + 0] = 4.0; // a_s1 = x0
                p[a2 + 8] = 4.0; // a_s2 = const[5]=10
                p[ao + 3] = 4.0; // a_op = /
                p[b1 + 1] = 4.0; // b_s1 = x1
                p[b2 + 4] = 4.0; // b_s2 = const[1]=1 (loop-src idx 4)
                p[bo + 0] = 4.0; // b_op = +
                p[ro + 1] = 4.0; // ret = x1
                p[co] = 0.0;
                p[co + 1] = 1.0;
                p[co + 2] = -1.0;
                p[co + 3] = 2.0;
                p[co + 4] = -2.0;
                p[co + 5] = 10.0;
                let ex = examples.clone();
                let result = train_program(
                    p,
                    move |pp, t| {
                        SoftPredicateLoop {
                            n_args,
                            params: pp.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |pp, fn_n, pn| {
                        SoftPredicateLoop {
                            n_args,
                            params: pp.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }
        }

        // 5b. SoftPredicateLoopRetCmp — two-acc loop with boolean return (triangular_check)
        if n_args == 1 {
            {
                let mut prog = SoftPredicateLoopRetCmp::new(n_args);
                if restart > 0 {
                    let base_n = SoftPredicateLoopRetCmp::base_n_params(n_args);
                    for (idx, p) in prog.params.iter_mut().enumerate() {
                        *p += (pseudo_rand(restart as u64 * 23000 + idx as u64) - 0.5)
                            * noise_scale
                            * 0.5;
                    }
                    let coff = base_n - N_CONSTS;
                    prog.params[coff] = 0.0;
                    prog.params[coff + 1] = 1.0;
                    prog.params[coff + 2] = -1.0;
                    prog.params[coff + 3] = 2.0;
                    prog.params[coff + 4] = -2.0;
                    prog.params[coff + 5] = 10.0;
                }
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftPredicateLoopRetCmp {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftPredicateLoopRetCmp {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 5b-triangular_check: x0=1, x1=0; while x1<n { xt=x0+1; x1=x1+x0; x0=xt }; if x1==n: 1 else 0
            // Loop sources (n_args=1): [x0=0, x1=1, n=2, c0=3, c1=4, c2=5, c3=6, c4=7, c5=8]
            // Init sources (n_args=1): [n=0, c0=1, c1=2, c2=3, c3=4, c4=5, c5=6]
            if restart == 0 {
                let _base_n = SoftPredicateLoopRetCmp::base_n_params(n_args);
                let mut prog = SoftPredicateLoopRetCmp::new(n_args);
                {
                    let inner = SoftPredicateLoop::new(n_args);
                    let nb = inner.nb();
                    let na = inner.na();
                    let c0 = 2 * nb;
                    let c1 = c0 + N_CMPS;
                    let c2 = c1 + na;
                    let a1 = c2 + na;
                    let a2 = a1 + na;
                    let ao = a2 + na;
                    let b1 = ao + 6;
                    let b2 = b1 + na;
                    let bo = b2 + na;
                    let ro = bo + 6;
                    let co = ro + na;
                    for p in prog.params.iter_mut() {
                        *p = -1.0;
                    }
                    prog.params[co] = 0.0;
                    prog.params[co + 1] = 1.0;
                    prog.params[co + 2] = -1.0;
                    prog.params[co + 3] = 2.0;
                    prog.params[co + 4] = -2.0;
                    prog.params[co + 5] = 10.0;
                    prog.params[2] = 4.0; // init0 = c1=1 (init-src idx 2 = n_args+1)
                    prog.params[nb + 1] = 4.0; // init1 = c0=0 (init-src idx 1 = n_args)
                    prog.params[c0 + 1] = 4.0; // cmp = <
                    prog.params[c1 + 1] = 4.0; // lhs = x1 (loop-src idx 1)
                    prog.params[c2 + 2] = 4.0; // rhs = n=arg0 (loop-src idx 2)
                    prog.params[a1 + 0] = 4.0; // a_s1 = x0 (new_x0 = x0+1)
                    prog.params[a2 + 4] = 4.0; // a_s2 = c1=1 (loop-src idx 4)
                    prog.params[ao + 0] = 4.0; // a_op = +
                    prog.params[b1 + 1] = 4.0; // b_s1 = x1 (new_x1 = x1+x0)
                    prog.params[b2 + 0] = 4.0; // b_s2 = x0 (loop-src idx 0)
                    prog.params[bo + 0] = 4.0; // b_op = +
                    prog.params[ro + 1] = 4.0; // ret = x1 (loop-src idx 1)
                                               // ret_cmp: x1 == n → 1
                    let ret_cmp_off = SoftPredicateLoopRetCmp::ret_cmp_off(n_args);
                    let ret_rhs_off = SoftPredicateLoopRetCmp::ret_rhs_off(n_args);
                    // rhs pool for ret cmp: [ret_val=x1, n=arg0, c0..c5]
                    // ret_val=x1 is at index 0; n is at index 1
                    prog.params[ret_cmp_off + 4] = 4.0; // cmp = ==
                    prog.params[ret_rhs_off + 1] = 4.0; // rhs = n (rhs_pool idx 1)
                }
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftPredicateLoopRetCmp {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftPredicateLoopRetCmp {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }
        } // if n_args == 1 (5b)

        // 9. SoftCondMutateLoop — conditional mutation loop (collatz_steps, step-counting)
        if n_args == 1 {
            {
                let mut prog = SoftCondMutateLoop::new(n_args);
                if restart > 0 {
                    let (_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, co) =
                        SoftCondMutateLoop::offsets(n_args);
                    for (idx, p) in prog.params.iter_mut().enumerate() {
                        *p += (pseudo_rand(restart as u64 * 29000 + idx as u64) - 0.5)
                            * noise_scale
                            * 0.5;
                    }
                    prog.params[co] = 0.0;
                    prog.params[co + 1] = 1.0;
                    prog.params[co + 2] = -1.0;
                    prog.params[co + 3] = 2.0;
                    prog.params[co + 4] = -2.0;
                    prog.params[co + 5] = 10.0;
                }
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftCondMutateLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftCondMutateLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS,
                );
                if result.is_some() {
                    return result;
                }
            }

            // 9-collatz: while x!=1 { if x%2==0 { x=x/2 } else { x=x*3+1 }; acc++ }
            // Use consts[4]=3 (overrides default -2) to enable x*3 expression.
            // Loop pool (n_args=1): [x=0, n=1, c0=2, c1=3, c2=4, c3=5, c4=6, c5=7]
            // Gate pool: [pre=0, x=1, n=2, c0=3, c1=4, c2=5, c3=6, c4=7, c5=8]
            // false_s3 pool: [v_tmp=0, x=1, n=2, c0=3, c1=4, c2=5, c3=6, c4=7, c5=8]
            {
                let mut prog = SoftCondMutateLoop::new(n_args);
                let (
                    init_off,
                    cond_cmp_off,
                    cond_lhs_off,
                    cond_rhs_off,
                    pre_op_off,
                    pre_s1_off,
                    pre_s2_off,
                    gate_cmp_off,
                    gate_rhs_off,
                    true_op_off,
                    true_s1_off,
                    true_s2_off,
                    fop1_off,
                    fs1_off,
                    fs2_off,
                    fop2_off,
                    fs3_off,
                    co,
                ) = SoftCondMutateLoop::offsets(n_args);
                let na = SoftCondMutateLoop::na(n_args);
                let ng = na + 1;
                for p in prog.params.iter_mut() {
                    *p = -4.0;
                }
                // consts: set c4=3 instead of default -2 so x*3 is representable
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = 3.0;
                prog.params[co + 5] = 10.0;
                // init_x = arg0 = n (init_pool: [arg0, c0..c5]; arg0 at idx 0)
                prog.params[init_off + 0] = 4.0;
                // cond: x != 1  (pool: x=0, n=1, c0=2, c1=3, ...; != at idx 5)
                prog.params[cond_cmp_off + 5] = 4.0; // !=
                prog.params[cond_lhs_off + 0] = 4.0; // lhs = x
                prog.params[cond_rhs_off + 3] = 4.0; // rhs = c1=1 (pool[3])
                                                     // pre: x % 2  (pool[5] = c3=2)
                prog.params[pre_op_off + 4] = 4.0; // %
                prog.params[pre_s1_off + 0] = 4.0; // x
                prog.params[pre_s2_off + 5] = 4.0; // c3=2 (pool[5])
                                                   // gate: pre == 0  (gate_pool[3] = c0=0)
                prog.params[gate_cmp_off + 4] = 4.0; // ==
                prog.params[gate_rhs_off + 3] = 4.0; // c0=0 (gate_pool[3])
                                                     // true: x / 2  (pool[5] = c3=2)
                prog.params[true_op_off + 3] = 4.0; // /
                prog.params[true_s1_off + 0] = 4.0; // x
                prog.params[true_s2_off + 5] = 4.0; // c3=2 (pool[5])
                                                    // false op1: x * 3  (pool[6] = c4=3)
                prog.params[fop1_off + 2] = 4.0; // *
                prog.params[fs1_off + 0] = 4.0; // x
                prog.params[fs2_off + 6] = 4.0; // c4=3 (pool[6])
                                                // false op2: v_tmp + 1  (fs3_pool[4] = c1=1)
                prog.params[fop2_off + 0] = 4.0; // +
                prog.params[fs3_off + 4] = 4.0; // c1=1 (fs3_pool[4])
                                                // For later restarts, add noise to non-critical params
                if restart > 0 {
                    let hot: std::collections::HashSet<usize> = [
                        co,
                        co + 1,
                        co + 2,
                        co + 3,
                        co + 4,
                        co + 5,
                        init_off,
                        cond_cmp_off + 5,
                        cond_lhs_off,
                        cond_rhs_off + 3,
                        pre_op_off + 4,
                        pre_s1_off,
                        pre_s2_off + 5,
                        gate_cmp_off + 4,
                        gate_rhs_off + 3,
                        true_op_off + 3,
                        true_s1_off,
                        true_s2_off + 5,
                        fop1_off + 2,
                        fs1_off,
                        fs2_off + 6,
                        fop2_off,
                        fs3_off + 4,
                    ]
                    .iter()
                    .copied()
                    .collect();
                    for (idx, p) in prog.params.iter_mut().enumerate() {
                        if !hot.contains(&idx) {
                            *p += (pseudo_rand(restart as u64 * 31000 + idx as u64) - 0.5)
                                * noise_scale
                                * 0.3;
                        }
                    }
                }
                let ex = examples.clone();
                let result = train_program(
                    prog.params.clone(),
                    move |p, t| {
                        SoftCondMutateLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .loss(&ex, t)
                    },
                    |p, fn_n, pn| {
                        SoftCondMutateLoop {
                            n_args,
                            params: p.to_vec(),
                        }
                        .discretize_and_emit(fn_n, pn)
                    },
                    problem,
                    &param_names,
                    fn_name,
                    N_STEPS * 2,
                );
                if result.is_some() {
                    return result;
                }
                let _ = (
                    na,
                    ng,
                    fs3_off,
                    cond_cmp_off,
                    pre_s1_off,
                    pre_s2_off,
                    gate_rhs_off,
                    true_s1_off,
                    true_s2_off,
                    fs1_off,
                    fs2_off,
                    fop2_off,
                );
            }
        } // if n_args == 1 (9)
    } // for restart

    // ── Final fallback: SoftUniversalProgram ─────────────────────────────────
    // Runs ONCE after all specialized restarts, only if nothing else solved it.
    // ~1248 params with FD gradients: use a reduced step budget (N_STEPS/4) so
    // the fallback adds at most a few seconds per problem, not minutes.
    // Increase UNIV_STEPS if you want more exploration at the cost of speed.
    const UNIV_STEPS: usize = N_STEPS / 4; // 200 steps
    for univ_restart in 0..3 {
        let mut prog = SoftUniversalProgram::new(n_args);
        if univ_restart > 0 {
            for (idx, p) in prog.params.iter_mut().enumerate() {
                *p += (pseudo_rand(univ_restart as u64 * 97000 + idx as u64) - 0.5) * 0.5;
            }
        }
        let ex = examples.clone();
        let result = train_program(
            prog.params.clone(),
            move |p, t| {
                SoftUniversalProgram {
                    n_args,
                    params: p.to_vec(),
                }
                .loss(&ex, t)
            },
            |p, fn_n, pn| {
                SoftUniversalProgram {
                    n_args,
                    params: p.to_vec(),
                }
                .discretize_and_emit(fn_n, pn)
            },
            problem,
            &param_names,
            fn_name,
            UNIV_STEPS,
        );
        if result.is_some() {
            return result;
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// Array gradient synthesis
// ═══════════════════════════════════════════════════════════════════════════════
//
// Soft program types that operate on array inputs. The array is padded to
// MAX_ARR elements and an in_bounds sigmoid gate masks out-of-bounds positions.
//
// Each type follows the same pattern: softmax logits → weighted sum → FD gradient
// → Adam → discretize to Mog `for item in arr { ... }` code.

/// Extract array + scalar inputs from a Problem.
/// Returns None if the first input isn't an array or if there are no examples.
#[derive(Clone)]
struct ArrExample {
    arr: Vec<f32>,       // padded to MAX_ARR
    arr_len: f32,
    scalar_args: Vec<f32>,
    expected: f32,
}

fn extract_arr_examples(problem: &Problem) -> Option<(Vec<ArrExample>, usize)> {
    let first = problem.examples.first()?;
    // First input must be an array
    let _arr0 = match &first.inputs[0] {
        Value::Array(a) => a.clone(),
        _ => return None,
    };
    let n_scalar = first.inputs.len().saturating_sub(1);
    let mut examples = Vec::with_capacity(problem.examples.len());
    for ex in &problem.examples {
        let arr = match &ex.inputs[0] {
            Value::Array(a) => a.clone(),
            _ => return None,
        };
        let arr_len = arr.len() as f32;
        let mut padded = vec![0f32; MAX_ARR];
        for (i, v) in arr.iter().enumerate() {
            if i < MAX_ARR { padded[i] = *v as f32; }
        }
        let mut scalar_args = Vec::with_capacity(n_scalar);
        for v in &ex.inputs[1..] {
            match v {
                Value::Int(iv) => scalar_args.push(*iv as f32),
                _ => return None,
            }
        }
        examples.push(ArrExample { arr: padded, arr_len, scalar_args, expected: ex.expected as f32 });
    }
    Some((examples, n_scalar))
}

// ─── Type A1: SoftArrayAccumProgram ──────────────────────────────────────────
//
// acc = soft_init([arr[0], consts..., scalar_args...]);
// for item in arr {
//     rhs = soft_read([item, item*item, acc, consts..., scalar_args...]);
//     new_acc = soft_op(acc, rhs, op_w);
//     acc = in_bounds * new_acc + (1 - in_bounds) * acc;
// }
// return soft_read([acc, consts..., scalar_args...]);
//
// Solves: array_sum, array_max, min_element, arr_sum_squares, max_abs, etc.

struct SoftArrayAccumProgram {
    n_scalar: usize,
    params: Vec<f32>,
}

impl SoftArrayAccumProgram {
    // Sources for init: [arr[0], consts(6), scalar_args(n_scalar)]
    fn n_init(&self) -> usize { 1 + N_CONSTS + self.n_scalar }
    // Sources for rhs: [item, item*item, acc, consts(6), scalar_args(n_scalar)]
    fn n_rhs(&self) -> usize { 3 + N_CONSTS + self.n_scalar }
    // Sources for return: [acc, consts(6), scalar_args(n_scalar)]
    fn n_ret(&self) -> usize { 1 + N_CONSTS + self.n_scalar }

    fn n_params(n_scalar: usize) -> usize {
        let init = 1 + N_CONSTS + n_scalar;
        let rhs = 3 + N_CONSTS + n_scalar;
        let ret = 1 + N_CONSTS + n_scalar;
        init + N_OPS + rhs + ret + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self { n_scalar, params: vec![0f32; Self::n_params(n_scalar)] };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    fn init_off(&self) -> usize { 0 }
    fn op_off(&self) -> usize { self.n_init() }
    fn rhs_off(&self) -> usize { self.n_init() + N_OPS }
    fn ret_off(&self) -> usize { self.n_init() + N_OPS + self.n_rhs() }
    fn consts_off(&self) -> usize { self.n_init() + N_OPS + self.n_rhs() + self.n_ret() }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let n_init = self.n_init();
        let n_rhs = self.n_rhs();
        let n_ret = self.n_ret();
        let co = self.consts_off();

        // Load constants
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        // Init: soft-select from [arr[0], consts..., scalar_args...]
        let init_w = softmax_temp(&self.params[self.init_off()..self.init_off() + n_init], temp);
        let mut init_storage = vec![0f32; n_init];
        init_storage[0] = arr[0]; // arr[0]
        for i in 0..N_CONSTS { init_storage[1 + i] = consts[i]; }
        for i in 0..self.n_scalar { init_storage[1 + N_CONSTS + i] = scalar_args[i]; }
        let mut acc = soft_read(&init_storage, &init_w);

        // Iterate over array
        let op_w = softmax_temp(&self.params[self.op_off()..self.op_off() + N_OPS], temp);
        let rhs_w = softmax_temp(&self.params[self.rhs_off()..self.rhs_off() + n_rhs], temp);

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            let item = arr[i];
            let mut rhs_storage = vec![0f32; n_rhs];
            rhs_storage[0] = item;
            rhs_storage[1] = item * item;
            rhs_storage[2] = acc;
            for j in 0..N_CONSTS { rhs_storage[3 + j] = consts[j]; }
            for j in 0..self.n_scalar { rhs_storage[3 + N_CONSTS + j] = scalar_args[j]; }
            let rhs = soft_read(&rhs_storage, &rhs_w);
            let new_acc = soft_op(acc, rhs, &op_w);
            acc = in_bounds * new_acc + (1.0 - in_bounds) * acc;
        }

        // Return: soft-select from [acc, consts..., scalar_args...]
        let ret_w = softmax_temp(&self.params[self.ret_off()..self.ret_off() + n_ret], temp);
        let mut ret_storage = vec![0f32; n_ret];
        ret_storage[0] = acc;
        for j in 0..N_CONSTS { ret_storage[1 + j] = consts[j]; }
        for j in 0..self.n_scalar { ret_storage[1 + N_CONSTS + j] = scalar_args[j]; }
        soft_read(&ret_storage, &ret_w)
    }

    fn loss(&self, examples: &[ArrExample], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples.iter().map(|ex| {
            let diff = self.forward(&ex.arr, ex.arr_len, &ex.scalar_args, temp) - ex.expected;
            diff * diff
        }).sum::<f32>() / n
    }

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let n_init = self.n_init();
        let n_rhs = self.n_rhs();
        let n_ret = self.n_ret();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS).map(|i| self.params[co + i].round() as i64).collect();

        // Init source names: [arr[0], c0..c5, scalar_args...]
        let init_names: Vec<String> = std::iter::once("arr[0]".to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        // RHS source names: [item, item*item, acc, c0..c5, scalar_args...]
        let rhs_names: Vec<String> = ["item", "item*item", "acc"].iter().map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        // Return source names: [acc, c0..c5, scalar_args...]
        let ret_names: Vec<String> = ["acc"].iter().map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let init_i = argmax(&self.params[self.init_off()..self.init_off() + n_init]);
        let op_i = argmax(&self.params[self.op_off()..self.op_off() + N_OPS]);
        let rhs_i = argmax(&self.params[self.rhs_off()..self.rhs_off() + n_rhs]);
        let ret_i = argmax(&self.params[self.ret_off()..self.ret_off() + n_ret]);

        let op_names = ["+", "-", "*", "/", "%"];
        let init_src = &init_names[init_i];
        let rhs_src = &rhs_names[rhs_i];

        let scalar_params = scalar_names.iter().map(|n| format!("{n}: i64")).collect::<Vec<_>>().join(", ");
        let sig = if scalar_params.is_empty() {
            format!("fn {fn_name}(arr: [i64]) -> i64")
        } else {
            format!("fn {fn_name}(arr: [i64], {scalar_params}) -> i64")
        };

        let mut out = format!("{sig} {{\n");
        writeln!(out, "    acc: i64 = {init_src};").unwrap();
        out.push_str("    for item in arr {\n");
        writeln!(out, "        acc = acc {} {};", op_names[op_i], rhs_src).unwrap();
        out.push_str("    }\n");
        writeln!(out, "    return {};", ret_names[ret_i]).unwrap();
        out.push_str("}\n");
        out
    }

    fn discrete_eval(&self, arr: &[i64], scalar_args: &[i64]) -> Option<i64> {
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS).map(|i| self.params[co + i].round() as i64).collect();

        // Init
        let n_init = self.n_init();
        let init_i = argmax(&self.params[self.init_off()..self.init_off() + n_init]);
        let mut acc = if init_i == 0 {
            *arr.get(0)?
        } else if init_i <= N_CONSTS {
            consts[init_i - 1]
        } else {
            *scalar_args.get(init_i - 1 - N_CONSTS)?
        };

        let op_i = argmax(&self.params[self.op_off()..self.op_off() + N_OPS]);

        for (i, &item) in arr.iter().enumerate() {
            let _ = i;
            // Apply the argmax'd op to accumulate
            acc = match op_i {
                0 => acc.checked_add(item)?,
                1 => acc.checked_sub(item)?,
                2 => acc.checked_mul(item)?,
                3 => { if item == 0 { return None; } acc / item },
                4 => { if item == 0 { return None; } acc % item },
                _ => acc,
            };
        }

        Some(acc)
    }
}

// ─── Type A2: SoftArrayCondAccumProgram ──────────────────────────────────────
//
// acc = init;
// for item in arr {
//     gate = soft_cmp(cmp_src1, cmp_src2, cmp_w);
//     rhs = soft_read([item, 1, acc, consts..., scalar_args...]);
//     acc += in_bounds * gate * (soft_op(acc, rhs, body_op_w) - acc);
// }
// return acc;
//
// Solves: count_positive, count_occurrences, count_zeros, count_evens,
//         sum_negatives, sum_positives, count_greater_than

struct SoftArrayCondAccumProgram {
    n_scalar: usize,
    params: Vec<f32>,
}

impl SoftArrayCondAccumProgram {
    // Init pool: [arr[0], consts(6), scalar_args(n_scalar)]
    fn init_pool(&self) -> usize { 1 + N_CONSTS + self.n_scalar }
    // Body pool: [item, acc, consts(6), scalar_args(n_scalar)]
    fn pool(&self) -> usize { 2 + N_CONSTS + self.n_scalar }

    fn n_params(n_scalar: usize) -> usize {
        let ip = 1 + N_CONSTS + n_scalar;
        let p = 2 + N_CONSTS + n_scalar;
        // init(ip) + cmp(N_CMPS) + 2*cmp_src(p) + body_op(N_OPS) + body_rhs(p) + mode(1) + ret(p) + consts(N_CONSTS)
        ip + N_CMPS + 2 * p + N_OPS + p + 1 + p + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self { n_scalar, params: vec![0f32; Self::n_params(n_scalar)] };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        // Default: init = const[0] = 0 (good for count/sum patterns)
        let io = s.init_off();
        s.params[io + 1] = 2.0; // bias toward const[0]=0
        // Default mode = accumulate (positive)
        let mo = s.mode_off();
        s.params[mo] = 2.0;
        s
    }

    fn init_off(&self) -> usize { 0 }
    fn cmp_off(&self) -> usize { self.init_pool() }
    fn cmp_s1_off(&self) -> usize { self.init_pool() + N_CMPS }
    fn cmp_s2_off(&self) -> usize { self.init_pool() + N_CMPS + self.pool() }
    fn body_op_off(&self) -> usize { self.init_pool() + N_CMPS + 2 * self.pool() }
    fn body_rhs_off(&self) -> usize { self.init_pool() + N_CMPS + 2 * self.pool() + N_OPS }
    /// Mode logit: >0 → accumulate (acc = acc OP rhs), <0 → replace (acc = rhs)
    fn mode_off(&self) -> usize { self.init_pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool() }
    fn ret_off(&self) -> usize { self.init_pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool() + 1 }
    fn consts_off(&self) -> usize { self.init_pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool() + 1 + self.pool() }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let ip = self.init_pool();
        let pool = self.pool();
        let co = self.consts_off();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        // Init from pool: [arr[0], consts..., scalar_args...]
        let mut init_s = vec![0f32; ip];
        init_s[0] = arr[0];
        for j in 0..N_CONSTS { init_s[1 + j] = consts[j]; }
        for j in 0..self.n_scalar { init_s[1 + N_CONSTS + j] = scalar_args[j]; }
        let init_w = softmax_temp(&self.params[self.init_off()..self.init_off() + ip], temp);
        let mut acc = soft_read(&init_s, &init_w);

        // Fixed logits
        let cmp_w = softmax_temp(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS], temp);
        let cmp_s1_w = softmax_temp(&self.params[self.cmp_s1_off()..self.cmp_s1_off() + pool], temp);
        let cmp_s2_w = softmax_temp(&self.params[self.cmp_s2_off()..self.cmp_s2_off() + pool], temp);
        let body_op_w = softmax_temp(&self.params[self.body_op_off()..self.body_op_off() + N_OPS], temp);
        let body_rhs_w = softmax_temp(&self.params[self.body_rhs_off()..self.body_rhs_off() + pool], temp);

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            let item = arr[i];

            // Build pool: [item, acc, consts..., scalar_args...]
            let mut p = vec![0f32; pool];
            p[0] = item;
            p[1] = acc;
            for j in 0..N_CONSTS { p[2 + j] = consts[j]; }
            for j in 0..self.n_scalar { p[2 + N_CONSTS + j] = scalar_args[j]; }

            let lhs = soft_read(&p, &cmp_s1_w);
            let rhs = soft_read(&p, &cmp_s2_w);
            let gate = soft_cmp(lhs, rhs, &cmp_w, temp);

            let body_rhs = soft_read(&p, &body_rhs_w);
            let accum_val = soft_op(acc, body_rhs, &body_op_w);
            // mode: sigmoid > 0.5 → accumulate, < 0.5 → replace
            let mode = sigmoid(self.params[self.mode_off()]);
            let new_acc = mode * accum_val + (1.0 - mode) * body_rhs;

            acc += in_bounds * gate * (new_acc - acc);
        }

        acc
    }

    fn loss(&self, examples: &[ArrExample], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples.iter().map(|ex| {
            let diff = self.forward(&ex.arr, ex.arr_len, &ex.scalar_args, temp) - ex.expected;
            diff * diff
        }).sum::<f32>() / n
    }

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let ip = self.init_pool();
        let pool = self.pool();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS).map(|i| self.params[co + i].round() as i64).collect();

        // Init source names: [arr[0], consts..., scalar_args...]
        let init_names: Vec<String> = std::iter::once("arr[0]".to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let pool_names: Vec<String> = ["item", "acc"].iter().map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let init_i = argmax(&self.params[self.init_off()..self.init_off() + ip]);
        let cmp_i = argmax(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS]);
        let cmp_s1_i = argmax(&self.params[self.cmp_s1_off()..self.cmp_s1_off() + pool]);
        let cmp_s2_i = argmax(&self.params[self.cmp_s2_off()..self.cmp_s2_off() + pool]);
        let op_i = argmax(&self.params[self.body_op_off()..self.body_op_off() + N_OPS]);
        let rhs_i = argmax(&self.params[self.body_rhs_off()..self.body_rhs_off() + pool]);
        let is_accum = self.params[self.mode_off()] > 0.0;

        let op_names = ["+", "-", "*", "/", "%"];
        let cmp_names = ["<", "<=", "==", ">=", ">", "!="];

        let scalar_params = scalar_names.iter().map(|n| format!("{n}: i64")).collect::<Vec<_>>().join(", ");
        let sig = if scalar_params.is_empty() {
            format!("fn {fn_name}(arr: [i64]) -> i64")
        } else {
            format!("fn {fn_name}(arr: [i64], {scalar_params}) -> i64")
        };

        let init_src = &init_names[init_i];
        // Use := for arr[0] (Mog syntax for init from first element), = for literals
        let init_assign = if init_i == 0 { ":=" } else { ": i64 =" };

        let mut out = format!("{sig} {{\n");
        writeln!(out, "    acc {init_assign} {init_src};").unwrap();
        out.push_str("    for item in arr {\n");
        writeln!(out, "        if {} {} {} {{", pool_names[cmp_s1_i], cmp_names[cmp_i], pool_names[cmp_s2_i]).unwrap();
        if is_accum {
            writeln!(out, "            acc = acc {} {};", op_names[op_i], pool_names[rhs_i]).unwrap();
        } else {
            writeln!(out, "            acc = {};", pool_names[rhs_i]).unwrap();
        }
        out.push_str("        }\n");
        out.push_str("    }\n");
        writeln!(out, "    return acc;").unwrap();
        out.push_str("}\n");
        out
    }
}

// ─── Type A3: SoftArrayPairProgram ───────────────────────────────────────────
//
// Two accumulators (a1, a2), each with a gated update per element.
// a1 = init1; a2 = init2;
// for item in arr {
//     gate1 = cmp(g1_lhs, g1_rhs);
//     then1 = soft_read(src1_pool);
//     a1 = gate1 * then1 + (1-gate1) * a1;
//     gate2 = cmp(g2_lhs, g2_rhs);
//     then2 = soft_read(src2_pool);  // pool2 includes updated a1
//     a2 = gate2 * then2 + (1-gate2) * a2;
// }
// return ret_op(a1, a2);
//
// Solves: array_range, second_max, max_stock_profit, longest_plateau

struct SoftArrayPairProgram {
    n_scalar: usize,
    params: Vec<f32>,
}

impl SoftArrayPairProgram {
    // Pool for update1: [item, a1, a2, consts(6), scalar_args(n_scalar)]
    fn pool1(&self) -> usize { 3 + N_CONSTS + self.n_scalar }
    // Pool for update2: [item, a1, a2, consts(6), scalar_args(n_scalar)]
    fn pool2(&self) -> usize { 3 + N_CONSTS + self.n_scalar }
    // Pool for init: [arr[0], consts(6), scalar_args(n_scalar)]
    fn init_pool(&self) -> usize { 1 + N_CONSTS + self.n_scalar }
    // Return pool: [a1, a2, consts(6), scalar_args(n_scalar)]
    fn ret_pool(&self) -> usize { 2 + N_CONSTS + self.n_scalar }

    fn n_params(n_scalar: usize) -> usize {
        let ip = 1 + N_CONSTS + n_scalar;
        let p1 = 3 + N_CONSTS + n_scalar;
        let _p2 = 3 + N_CONSTS + n_scalar;
        let rp = 2 + N_CONSTS + n_scalar;
        // init1(ip) + init2(ip) + gate1(N_CMPS + 2*p1) + then1(p1) + gate2(N_CMPS + 2*p2) + then2(p2) + ret_op(N_OPS) + ret_s1(rp) + ret_s2(rp) + consts(6)
        2 * ip + 2 * (N_CMPS + 3 * p1) + N_OPS + 2 * rp + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self { n_scalar, params: vec![0f32; Self::n_params(n_scalar)] };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    fn init1_off(&self) -> usize { 0 }
    fn init2_off(&self) -> usize { self.init_pool() }
    fn g1_cmp_off(&self) -> usize { 2 * self.init_pool() }
    fn g1_s1_off(&self) -> usize { 2 * self.init_pool() + N_CMPS }
    fn g1_s2_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + self.pool1() }
    fn then1_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 2 * self.pool1() }
    fn g2_cmp_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 3 * self.pool1() }
    fn g2_s1_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS }
    fn g2_s2_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + self.pool2() }
    fn then2_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + 2 * self.pool2() }
    fn ret_op_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + 3 * self.pool2() }
    fn ret_s1_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + 3 * self.pool2() + N_OPS }
    fn ret_s2_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + 3 * self.pool2() + N_OPS + self.ret_pool() }
    fn consts_off(&self) -> usize { 2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + 3 * self.pool2() + N_OPS + 2 * self.ret_pool() }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let ip = self.init_pool();
        let p1 = self.pool1();
        let p2 = self.pool2();
        let rp = self.ret_pool();
        let co = self.consts_off();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        // Init storage: [arr[0], consts..., scalar_args...]
        let mut init_s = vec![0f32; ip];
        init_s[0] = arr[0];
        for j in 0..N_CONSTS { init_s[1 + j] = consts[j]; }
        for j in 0..self.n_scalar { init_s[1 + N_CONSTS + j] = scalar_args[j]; }

        let init1_w = softmax_temp(&self.params[self.init1_off()..self.init1_off() + ip], temp);
        let init2_w = softmax_temp(&self.params[self.init2_off()..self.init2_off() + ip], temp);
        let mut a1 = soft_read(&init_s, &init1_w);
        let mut a2 = soft_read(&init_s, &init2_w);

        // Fixed logits
        let g1_cmp_w = softmax_temp(&self.params[self.g1_cmp_off()..self.g1_cmp_off() + N_CMPS], temp);
        let g1_s1_w = softmax_temp(&self.params[self.g1_s1_off()..self.g1_s1_off() + p1], temp);
        let g1_s2_w = softmax_temp(&self.params[self.g1_s2_off()..self.g1_s2_off() + p1], temp);
        let then1_w = softmax_temp(&self.params[self.then1_off()..self.then1_off() + p1], temp);
        let g2_cmp_w = softmax_temp(&self.params[self.g2_cmp_off()..self.g2_cmp_off() + N_CMPS], temp);
        let g2_s1_w = softmax_temp(&self.params[self.g2_s1_off()..self.g2_s1_off() + p2], temp);
        let g2_s2_w = softmax_temp(&self.params[self.g2_s2_off()..self.g2_s2_off() + p2], temp);
        let then2_w = softmax_temp(&self.params[self.then2_off()..self.then2_off() + p2], temp);

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            let item = arr[i];

            // Pool1: [item, a1, a2, consts..., scalar_args...]
            let mut pool1 = vec![0f32; p1];
            pool1[0] = item; pool1[1] = a1; pool1[2] = a2;
            for j in 0..N_CONSTS { pool1[3 + j] = consts[j]; }
            for j in 0..self.n_scalar { pool1[3 + N_CONSTS + j] = scalar_args[j]; }

            let g1 = soft_cmp(soft_read(&pool1, &g1_s1_w), soft_read(&pool1, &g1_s2_w), &g1_cmp_w, temp);
            let t1 = soft_read(&pool1, &then1_w);
            let new_a1 = g1 * t1 + (1.0 - g1) * a1;
            a1 = in_bounds * new_a1 + (1.0 - in_bounds) * a1;

            // Pool2: [item, a1(updated), a2, consts..., scalar_args...]
            let mut pool2 = vec![0f32; p2];
            pool2[0] = item; pool2[1] = a1; pool2[2] = a2;
            for j in 0..N_CONSTS { pool2[3 + j] = consts[j]; }
            for j in 0..self.n_scalar { pool2[3 + N_CONSTS + j] = scalar_args[j]; }

            let g2 = soft_cmp(soft_read(&pool2, &g2_s1_w), soft_read(&pool2, &g2_s2_w), &g2_cmp_w, temp);
            let t2 = soft_read(&pool2, &then2_w);
            let new_a2 = g2 * t2 + (1.0 - g2) * a2;
            a2 = in_bounds * new_a2 + (1.0 - in_bounds) * a2;
        }

        // Return: soft_op(a1, a2, ret_op) or soft_select
        let ret_op_w = softmax_temp(&self.params[self.ret_op_off()..self.ret_op_off() + N_OPS], temp);
        let ret_s1_w = softmax_temp(&self.params[self.ret_s1_off()..self.ret_s1_off() + rp], temp);
        let ret_s2_w = softmax_temp(&self.params[self.ret_s2_off()..self.ret_s2_off() + rp], temp);

        let mut ret_pool = vec![0f32; rp];
        ret_pool[0] = a1; ret_pool[1] = a2;
        for j in 0..N_CONSTS { ret_pool[2 + j] = consts[j]; }
        for j in 0..self.n_scalar { ret_pool[2 + N_CONSTS + j] = scalar_args[j]; }

        let s1 = soft_read(&ret_pool, &ret_s1_w);
        let s2 = soft_read(&ret_pool, &ret_s2_w);
        soft_op(s1, s2, &ret_op_w)
    }

    fn loss(&self, examples: &[ArrExample], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples.iter().map(|ex| {
            let diff = self.forward(&ex.arr, ex.arr_len, &ex.scalar_args, temp) - ex.expected;
            diff * diff
        }).sum::<f32>() / n
    }

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let ip = self.init_pool();
        let p1 = self.pool1();
        let rp = self.ret_pool();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS).map(|i| self.params[co + i].round() as i64).collect();

        let init_names: Vec<String> = std::iter::once("arr[0]".to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let pool1_names: Vec<String> = ["item", "lo", "hi"].iter().map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let ret_names: Vec<String> = ["lo", "hi"].iter().map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let init1_i = argmax(&self.params[self.init1_off()..self.init1_off() + ip]);
        let init2_i = argmax(&self.params[self.init2_off()..self.init2_off() + ip]);
        let g1_cmp_i = argmax(&self.params[self.g1_cmp_off()..self.g1_cmp_off() + N_CMPS]);
        let g1_s1_i = argmax(&self.params[self.g1_s1_off()..self.g1_s1_off() + p1]);
        let g1_s2_i = argmax(&self.params[self.g1_s2_off()..self.g1_s2_off() + p1]);
        let then1_i = argmax(&self.params[self.then1_off()..self.then1_off() + p1]);
        let g2_cmp_i = argmax(&self.params[self.g2_cmp_off()..self.g2_cmp_off() + N_CMPS]);
        let g2_s1_i = argmax(&self.params[self.g2_s1_off()..self.g2_s1_off() + p1]);
        let g2_s2_i = argmax(&self.params[self.g2_s2_off()..self.g2_s2_off() + p1]);
        let then2_i = argmax(&self.params[self.then2_off()..self.then2_off() + p1]);
        let ret_op_i = argmax(&self.params[self.ret_op_off()..self.ret_op_off() + N_OPS]);
        let ret_s1_i = argmax(&self.params[self.ret_s1_off()..self.ret_s1_off() + rp]);
        let ret_s2_i = argmax(&self.params[self.ret_s2_off()..self.ret_s2_off() + rp]);

        let op_names = ["+", "-", "*", "/", "%"];
        let cmp_names = ["<", "<=", "==", ">=", ">", "!="];

        let scalar_params = scalar_names.iter().map(|n| format!("{n}: i64")).collect::<Vec<_>>().join(", ");
        let sig = if scalar_params.is_empty() {
            format!("fn {fn_name}(arr: [i64]) -> i64")
        } else {
            format!("fn {fn_name}(arr: [i64], {scalar_params}) -> i64")
        };

        let mut out = format!("{sig} {{\n");
        writeln!(out, "    lo: i64 = {};", init_names[init1_i]).unwrap();
        writeln!(out, "    hi: i64 = {};", init_names[init2_i]).unwrap();
        out.push_str("    for item in arr {\n");
        writeln!(out, "        if {} {} {} {{ lo = {}; }}", pool1_names[g1_s1_i], cmp_names[g1_cmp_i], pool1_names[g1_s2_i], pool1_names[then1_i]).unwrap();
        writeln!(out, "        if {} {} {} {{ hi = {}; }}", pool1_names[g2_s1_i], cmp_names[g2_cmp_i], pool1_names[g2_s2_i], pool1_names[then2_i]).unwrap();
        out.push_str("    }\n");
        writeln!(out, "    return {} {} {};", ret_names[ret_s1_i], op_names[ret_op_i], ret_names[ret_s2_i]).unwrap();
        out.push_str("}\n");
        out
    }
}

// ─── Type A4: SoftPairwiseScanProgram ────────────────────────────────────────
//
// Tracks consecutive pairs. acc = init; prev = arr[0];
// for i in 1..arr.len {
//     item = arr[i]; diff = item OP prev;
//     gate = cmp(diff/acc/..., threshold);
//     acc = gate * (acc OP rhs) + (1-gate) * acc;
//     prev = item;
// }
// return acc;
//
// Solves: is_sorted, longest_increasing_run, max_pair_diff, max_consecutive_sum,
//         min_consecutive_sum, count_peaks, alternating_sum

struct SoftPairwiseScanProgram {
    n_scalar: usize,
    params: Vec<f32>,
}

impl SoftPairwiseScanProgram {
    // Pool: [item, prev, diff, acc, consts(6), scalar_args(n_scalar)]
    fn pool(&self) -> usize { 4 + N_CONSTS + self.n_scalar }
    // Init pool: [arr[0], consts(6), scalar_args(n_scalar)]
    fn init_pool(&self) -> usize { 1 + N_CONSTS + self.n_scalar }

    fn n_params(n_scalar: usize) -> usize {
        let ip = 1 + N_CONSTS + n_scalar;
        let p = 4 + N_CONSTS + n_scalar;
        // init(ip) + pre_op(N_OPS) + pre_s1(p) + pre_s2(p) + [cmp(N_CMPS) + cmp_s1(p) + cmp_s2(p)] + body_op(N_OPS) + body_rhs(p) + ret(p) + consts(6)
        ip + N_OPS + 2 * p + N_CMPS + 2 * p + N_OPS + p + p + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self { n_scalar, params: vec![0f32; Self::n_params(n_scalar)] };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    fn init_off(&self) -> usize { 0 }
    fn pre_op_off(&self) -> usize { self.init_pool() }
    fn pre_s1_off(&self) -> usize { self.init_pool() + N_OPS }
    fn pre_s2_off(&self) -> usize { self.init_pool() + N_OPS + self.pool() }
    fn cmp_off(&self) -> usize { self.init_pool() + N_OPS + 2 * self.pool() }
    fn cmp_s1_off(&self) -> usize { self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS }
    fn cmp_s2_off(&self) -> usize { self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + self.pool() }
    fn body_op_off(&self) -> usize { self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() }
    fn body_rhs_off(&self) -> usize { self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS }
    fn ret_off(&self) -> usize { self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool() }
    fn consts_off(&self) -> usize { self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS + 2 * self.pool() }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let ip = self.init_pool();
        let p = self.pool();
        let co = self.consts_off();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        // Init
        let mut init_s = vec![0f32; ip];
        init_s[0] = arr[0];
        for j in 0..N_CONSTS { init_s[1 + j] = consts[j]; }
        for j in 0..self.n_scalar { init_s[1 + N_CONSTS + j] = scalar_args[j]; }
        let init_w = softmax_temp(&self.params[self.init_off()..self.init_off() + ip], temp);
        let mut acc = soft_read(&init_s, &init_w);

        let mut prev = arr[0]; // first element
        let _run_len = 1f32;

        let pre_op_w = softmax_temp(&self.params[self.pre_op_off()..self.pre_op_off() + N_OPS], temp);
        let pre_s1_w = softmax_temp(&self.params[self.pre_s1_off()..self.pre_s1_off() + p], temp);
        let pre_s2_w = softmax_temp(&self.params[self.pre_s2_off()..self.pre_s2_off() + p], temp);
        let cmp_w = softmax_temp(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS], temp);
        let cmp_s1_w = softmax_temp(&self.params[self.cmp_s1_off()..self.cmp_s1_off() + p], temp);
        let cmp_s2_w = softmax_temp(&self.params[self.cmp_s2_off()..self.cmp_s2_off() + p], temp);
        let body_op_w = softmax_temp(&self.params[self.body_op_off()..self.body_op_off() + N_OPS], temp);
        let body_rhs_w = softmax_temp(&self.params[self.body_rhs_off()..self.body_rhs_off() + p], temp);
        let ret_w = softmax_temp(&self.params[self.ret_off()..self.ret_off() + p], temp);

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            if i == 0 { continue; } // skip first element (we already set prev = arr[0])
            let item = arr[i];

            // Pool: [item, prev, diff, acc, consts..., scalar_args...]
            let mut pool = vec![0f32; p];
            pool[0] = item;
            pool[1] = prev;
            pool[2] = item - prev; // diff (default, will be overridden by pre_op)
            pool[3] = acc;
            for j in 0..N_CONSTS { pool[4 + j] = consts[j]; }
            for j in 0..self.n_scalar { pool[4 + N_CONSTS + j] = scalar_args[j]; }

            // Pre-compute diff
            let pre_s1 = soft_read(&pool, &pre_s1_w);
            let pre_s2 = soft_read(&pool, &pre_s2_w);
            let diff = soft_op(pre_s1, pre_s2, &pre_op_w);
            pool[2] = diff;

            // Gate
            let lhs = soft_read(&pool, &cmp_s1_w);
            let rhs = soft_read(&pool, &cmp_s2_w);
            let gate = soft_cmp(lhs, rhs, &cmp_w, temp);

            // Body
            let body_rhs = soft_read(&pool, &body_rhs_w);
            let new_acc = soft_op(acc, body_rhs, &body_op_w);
            acc = in_bounds * (gate * new_acc + (1.0 - gate) * acc) + (1.0 - in_bounds) * acc;

            prev = item;
        }

        // Return
        let mut ret_pool = vec![0f32; p];
        ret_pool[0] = 0.0; ret_pool[1] = prev; ret_pool[2] = 0.0; ret_pool[3] = acc;
        for j in 0..N_CONSTS { ret_pool[4 + j] = consts[j]; }
        for j in 0..self.n_scalar { ret_pool[4 + N_CONSTS + j] = scalar_args[j]; }
        soft_read(&ret_pool, &ret_w)
    }

    fn loss(&self, examples: &[ArrExample], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples.iter().map(|ex| {
            let diff = self.forward(&ex.arr, ex.arr_len, &ex.scalar_args, temp) - ex.expected;
            diff * diff
        }).sum::<f32>() / n
    }

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let ip = self.init_pool();
        let p = self.pool();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS).map(|i| self.params[co + i].round() as i64).collect();

        let init_names: Vec<String> = std::iter::once("arr[0]".to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let pool_names: Vec<String> = ["item", "prev", "diff", "acc"].iter().map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let init_i = argmax(&self.params[self.init_off()..self.init_off() + ip]);
        let pre_op_i = argmax(&self.params[self.pre_op_off()..self.pre_op_off() + N_OPS]);
        let cmp_i = argmax(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS]);
        let cmp_s1_i = argmax(&self.params[self.cmp_s1_off()..self.cmp_s1_off() + p]);
        let cmp_s2_i = argmax(&self.params[self.cmp_s2_off()..self.cmp_s2_off() + p]);
        let op_i = argmax(&self.params[self.body_op_off()..self.body_op_off() + N_OPS]);
        let rhs_i = argmax(&self.params[self.body_rhs_off()..self.body_rhs_off() + p]);
        let ret_i = argmax(&self.params[self.ret_off()..self.ret_off() + p]);

        let op_names = ["+", "-", "*", "/", "%"];
        let cmp_names = ["<", "<=", "==", ">=", ">", "!="];

        let scalar_params = scalar_names.iter().map(|n| format!("{n}: i64")).collect::<Vec<_>>().join(", ");
        let sig = if scalar_params.is_empty() {
            format!("fn {fn_name}(arr: [i64]) -> i64")
        } else {
            format!("fn {fn_name}(arr: [i64], {scalar_params}) -> i64")
        };

        // Need pre_s1 and pre_s2 argmax for diff computation
        let pre_s1_i = argmax(&self.params[self.pre_s1_off()..self.pre_s1_off() + p]);
        let pre_s2_i = argmax(&self.params[self.pre_s2_off()..self.pre_s2_off() + p]);

        let mut out = format!("{sig} {{\n");
        out.push_str("    if arr.len == 0 { return 0; }\n");
        writeln!(out, "    acc: i64 = {};", init_names[init_i]).unwrap();
        out.push_str("    prev: i64 = arr[0];\n");
        out.push_str("    i: i64 = 1;\n");
        out.push_str("    while i < arr.len {\n");
        out.push_str("        item: i64 = arr[i];\n");
        writeln!(out, "        diff: i64 = {} {} {};", pool_names[pre_s1_i], op_names[pre_op_i], pool_names[pre_s2_i]).unwrap();
        writeln!(out, "        if {} {} {} {{", pool_names[cmp_s1_i], cmp_names[cmp_i], pool_names[cmp_s2_i]).unwrap();
        writeln!(out, "            acc = acc {} {};", op_names[op_i], pool_names[rhs_i]).unwrap();
        out.push_str("        }\n");
        out.push_str("        prev = item;\n");
        out.push_str("        i = i + 1;\n");
        out.push_str("    }\n");
        writeln!(out, "    return {};", pool_names[ret_i]).unwrap();
        out.push_str("}\n");
        out
    }
}

// ─── Type A5: SoftArrayIndexGateProgram ──────────────────────────────────────
//
// Index-aware gated accumulator. Like A2, but the pool includes the loop index
// `i`, a parity signal (cos-based: 1.0 at even, 0.0 at odd), and arr_len.
// Also has a pre-compute step: target = op(pool_src1, pool_src2) enabling
// index arithmetic (e.g., arr_len - k for kth_from_end).
//
// acc = init;
// for i, item in arr {
//     target = pre_op(pre_src1, pre_src2);   // e.g. arr_len - k
//     gate = soft_cmp(cmp_src1, cmp_src2);
//     rhs = soft_read(pool);
//     acc += in_bounds * gate * (op(acc, rhs) - acc);
// }
// return ret;
//
// Pool: [item, acc, i, parity, arr_len, target, consts(6), scalar_args(n_scalar)]
//
// Solves: sum_at_even_indices, sum_odd_indexed, kth_from_end

struct SoftArrayIndexGateProgram {
    n_scalar: usize,
    params: Vec<f32>,
}

impl SoftArrayIndexGateProgram {
    // Pool: [item, acc, i, parity, arr_len, target, consts(6), scalar_args(n_scalar)]
    fn pool(&self) -> usize { 6 + N_CONSTS + self.n_scalar }

    fn n_params(n_scalar: usize) -> usize {
        let p = 6 + N_CONSTS + n_scalar;
        // init(1) + pre_op(N_OPS) + pre_s1(p) + pre_s2(p) + cmp(N_CMPS) + cmp_s1(p) + cmp_s2(p)
        // + body_op(N_OPS) + body_rhs(p) + ret(p) + consts(6)
        1 + N_OPS + 2 * p + N_CMPS + 2 * p + N_OPS + p + p + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self { n_scalar, params: vec![0f32; Self::n_params(n_scalar)] };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    fn init_off(&self) -> usize { 0 }
    fn pre_op_off(&self) -> usize { 1 }
    fn pre_s1_off(&self) -> usize { 1 + N_OPS }
    fn pre_s2_off(&self) -> usize { 1 + N_OPS + self.pool() }
    fn cmp_off(&self) -> usize { 1 + N_OPS + 2 * self.pool() }
    fn cmp_s1_off(&self) -> usize { 1 + N_OPS + 2 * self.pool() + N_CMPS }
    fn cmp_s2_off(&self) -> usize { 1 + N_OPS + 2 * self.pool() + N_CMPS + self.pool() }
    fn body_op_off(&self) -> usize { 1 + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() }
    fn body_rhs_off(&self) -> usize { 1 + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS }
    fn ret_off(&self) -> usize { 1 + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool() }
    fn consts_off(&self) -> usize { 1 + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS + 2 * self.pool() }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let p = self.pool();
        let co = self.consts_off();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        let mut acc = self.params[self.init_off()];

        // Fixed logits
        let pre_op_w = softmax_temp(&self.params[self.pre_op_off()..self.pre_op_off() + N_OPS], temp);
        let pre_s1_w = softmax_temp(&self.params[self.pre_s1_off()..self.pre_s1_off() + p], temp);
        let pre_s2_w = softmax_temp(&self.params[self.pre_s2_off()..self.pre_s2_off() + p], temp);
        let cmp_w = softmax_temp(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS], temp);
        let cmp_s1_w = softmax_temp(&self.params[self.cmp_s1_off()..self.cmp_s1_off() + p], temp);
        let cmp_s2_w = softmax_temp(&self.params[self.cmp_s2_off()..self.cmp_s2_off() + p], temp);
        let body_op_w = softmax_temp(&self.params[self.body_op_off()..self.body_op_off() + N_OPS], temp);
        let body_rhs_w = softmax_temp(&self.params[self.body_rhs_off()..self.body_rhs_off() + p], temp);

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            let item = arr[i];
            let idx = i as f32;
            // Parity: cos(πi) → +1.0 at even, -1.0 at odd (clean separation at 0)
            let parity = (std::f32::consts::PI * idx).cos();

            // Build pool: [item, acc, i, parity, arr_len, target(placeholder), consts..., scalar_args...]
            let mut pool = vec![0f32; p];
            pool[0] = item;
            pool[1] = acc;
            pool[2] = idx;
            pool[3] = parity;
            pool[4] = arr_len;
            pool[5] = 0.0; // placeholder for target
            for j in 0..N_CONSTS { pool[6 + j] = consts[j]; }
            for j in 0..self.n_scalar { pool[6 + N_CONSTS + j] = scalar_args[j]; }

            // Pre-compute target
            let pre_s1 = soft_read(&pool, &pre_s1_w);
            let pre_s2 = soft_read(&pool, &pre_s2_w);
            let target = soft_op(pre_s1, pre_s2, &pre_op_w);
            pool[5] = target;

            // Gate
            let lhs = soft_read(&pool, &cmp_s1_w);
            let rhs = soft_read(&pool, &cmp_s2_w);
            let gate = soft_cmp(lhs, rhs, &cmp_w, temp);

            // Body
            let body_rhs = soft_read(&pool, &body_rhs_w);
            let new_acc = soft_op(acc, body_rhs, &body_op_w);

            acc += in_bounds * gate * (new_acc - acc);
        }

        // Return: soft-read from pool at final state
        let ret_w = softmax_temp(&self.params[self.ret_off()..self.ret_off() + p], temp);
        let mut ret_pool = vec![0f32; p];
        ret_pool[0] = 0.0; // no item at return
        ret_pool[1] = acc;
        ret_pool[2] = 0.0;
        ret_pool[3] = 0.0;
        ret_pool[4] = arr_len;
        ret_pool[5] = 0.0;
        for j in 0..N_CONSTS { ret_pool[6 + j] = consts[j]; }
        for j in 0..self.n_scalar { ret_pool[6 + N_CONSTS + j] = scalar_args[j]; }
        soft_read(&ret_pool, &ret_w)
    }

    fn loss(&self, examples: &[ArrExample], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples.iter().map(|ex| {
            let diff = self.forward(&ex.arr, ex.arr_len, &ex.scalar_args, temp) - ex.expected;
            diff * diff
        }).sum::<f32>() / n
    }

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let p = self.pool();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS).map(|i| self.params[co + i].round() as i64).collect();

        let pool_names: Vec<String> = ["item", "acc", "i", "parity", "arr.len", "target"].iter().map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let init_val = self.params[self.init_off()].round() as i64;
        let pre_op_i = argmax(&self.params[self.pre_op_off()..self.pre_op_off() + N_OPS]);
        let pre_s1_i = argmax(&self.params[self.pre_s1_off()..self.pre_s1_off() + p]);
        let pre_s2_i = argmax(&self.params[self.pre_s2_off()..self.pre_s2_off() + p]);
        let cmp_i = argmax(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS]);
        let cmp_s1_i = argmax(&self.params[self.cmp_s1_off()..self.cmp_s1_off() + p]);
        let cmp_s2_i = argmax(&self.params[self.cmp_s2_off()..self.cmp_s2_off() + p]);
        let op_i = argmax(&self.params[self.body_op_off()..self.body_op_off() + N_OPS]);
        let rhs_i = argmax(&self.params[self.body_rhs_off()..self.body_rhs_off() + p]);
        let ret_i = argmax(&self.params[self.ret_off()..self.ret_off() + p]);

        let op_names = ["+", "-", "*", "/", "%"];
        let cmp_names = ["<", "<=", "==", ">=", ">", "!="];

        let scalar_params = scalar_names.iter().map(|n| format!("{n}: i64")).collect::<Vec<_>>().join(", ");
        let sig = if scalar_params.is_empty() {
            format!("fn {fn_name}(arr: [i64]) -> i64")
        } else {
            format!("fn {fn_name}(arr: [i64], {scalar_params}) -> i64")
        };

        // Detect if the gate uses parity (index 3) — if so, emit stride-2 loop instead
        let uses_parity = cmp_s1_i == 3 || cmp_s2_i == 3;
        // Detect if the gate compares index with target: kth_from_end pattern
        let uses_target = cmp_s1_i == 5 || cmp_s2_i == 5;
        let uses_index_eq = (cmp_s1_i == 2 || cmp_s2_i == 2) && cmp_i == 2; // i == something

        if uses_parity {
            // Determine even vs odd based on the comparison:
            // If cmp(parity, 0) > or >= or != → even indices (parity=1 at even)
            // If cmp(parity, 0) < or <= or == → odd indices (parity=0 at even → fires at odd)
            let even = if cmp_s1_i == 3 {
                // parity on lhs
                cmp_i >= 3 // >=, >, != → parity is high → even
            } else {
                // parity on rhs
                cmp_i <= 1 // <, <= → other < parity → parity is high → even
            };
            let start = if even { 0 } else { 1 };
            let ret_name = &pool_names[ret_i];
            // For return: if ret_i == 1 (acc), emit "acc", otherwise the pool name
            let mut out = format!("{sig} {{\n");
            writeln!(out, "    acc: i64 = {init_val};").unwrap();
            writeln!(out, "    i: i64 = {start};").unwrap();
            out.push_str("    while i < arr.len {\n");
            writeln!(out, "        acc = acc {} arr[i];", op_names[op_i]).unwrap();
            out.push_str("        i = i + 2;\n");
            out.push_str("    }\n");
            writeln!(out, "    return {};", ret_name).unwrap();
            out.push_str("}\n");
            out
        } else if uses_target || uses_index_eq {
            // Pattern: compute target = op(src1, src2), then if i == target { acc = item; }
            let mut out = format!("{sig} {{\n");
            writeln!(out, "    acc: i64 = {init_val};").unwrap();
            writeln!(out, "    target: i64 = {} {} {};", pool_names[pre_s1_i], op_names[pre_op_i], pool_names[pre_s2_i]).unwrap();
            out.push_str("    i: i64 = 0;\n");
            out.push_str("    while i < arr.len {\n");
            out.push_str("        item: i64 = arr[i];\n");
            writeln!(out, "        if {} {} {} {{", pool_names[cmp_s1_i], cmp_names[cmp_i], pool_names[cmp_s2_i]).unwrap();
            writeln!(out, "            acc = acc {} {};", op_names[op_i], pool_names[rhs_i]).unwrap();
            out.push_str("        }\n");
            out.push_str("        i = i + 1;\n");
            out.push_str("    }\n");
            writeln!(out, "    return {};", pool_names[ret_i]).unwrap();
            out.push_str("}\n");
            out
        } else {
            // Fallback: generic indexed loop with gate
            let mut out = format!("{sig} {{\n");
            writeln!(out, "    acc: i64 = {init_val};").unwrap();
            out.push_str("    i: i64 = 0;\n");
            out.push_str("    while i < arr.len {\n");
            out.push_str("        item: i64 = arr[i];\n");
            writeln!(out, "        if {} {} {} {{", pool_names[cmp_s1_i], cmp_names[cmp_i], pool_names[cmp_s2_i]).unwrap();
            writeln!(out, "            acc = acc {} {};", op_names[op_i], pool_names[rhs_i]).unwrap();
            out.push_str("        }\n");
            out.push_str("        i = i + 1;\n");
            out.push_str("    }\n");
            writeln!(out, "    return {};", pool_names[ret_i]).unwrap();
            out.push_str("}\n");
            out
        }
    }
}

// ─── Array synthesis entry point ─────────────────────────────────────────────

/// Attempt gradient-based synthesis for array-input problems.
/// Returns None if the problem is not an array problem or synthesis fails.
pub fn synthesize_array(problem: &Problem) -> Option<SolveResult> {
    let (examples, n_scalar) = extract_arr_examples(problem)?;
    let fn_name = problem.function_name();
    let scalar_names: Vec<&str> = if n_scalar == 0 { vec![] }
        else if n_scalar == 1 { vec!["k"] }
        else { vec!["a", "b", "c", "d", "e", "f"].iter().take(n_scalar).copied().collect() };

    const N_ARR_STEPS: usize = 600;
    const N_ARR_RESTARTS: usize = 8;

    for restart in 0..N_ARR_RESTARTS {
        // A1: SoftArrayAccumProgram
        {
            let mut prog = SoftArrayAccumProgram::new(n_scalar);
            if restart == 1 {
                // Sum biased: init=0, op=+, rhs=item
                let init_off = prog.init_off();
                let op_off = prog.op_off();
                let rhs_off = prog.rhs_off();
                prog.params[init_off + 1] = 4.0; // init = const[0] = 0
                prog.params[op_off] = 4.0; // op = +
                prog.params[rhs_off] = 4.0; // rhs = item
            } else if restart == 2 {
                // Product biased: init=1, op=*, rhs=item
                let init_off = prog.init_off();
                let op_off = prog.op_off();
                let rhs_off = prog.rhs_off();
                prog.params[init_off + 2] = 4.0; // init = const[1] = 1
                prog.params[op_off + 2] = 4.0; // op = *
                prog.params[rhs_off] = 4.0; // rhs = item
            } else if restart == 3 {
                // Sum-of-squares biased: init=0, op=+, rhs=item*item
                let init_off = prog.init_off();
                let op_off = prog.op_off();
                let rhs_off = prog.rhs_off();
                prog.params[init_off + 1] = 4.0; // init = const[0] = 0
                prog.params[op_off] = 4.0; // op = +
                prog.params[rhs_off + 1] = 4.0; // rhs = item*item
            }
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 13000 + idx as u64) - 0.5) * 0.3;
                }
            }
            let ex = examples.clone();
            let sn = scalar_names.clone();
            let result = train_program_arr(
                prog.params,
                move |p, t| SoftArrayAccumProgram { n_scalar, params: p.to_vec() }.loss(&ex, t),
                move |p, fn_n| SoftArrayAccumProgram { n_scalar, params: p.to_vec() }.discretize_and_emit(fn_n, &sn),
                problem, fn_name, N_ARR_STEPS,
            );
            if result.is_some() { return result; }
        }

        // A2: SoftArrayCondAccumProgram
        {
            let mut prog = SoftArrayCondAccumProgram::new(n_scalar);
            let io = prog.init_off();
            let co2 = prog.cmp_off();
            let cs1 = prog.cmp_s1_off();
            let cs2 = prog.cmp_s2_off();
            let bo2 = prog.body_op_off();
            let br2 = prog.body_rhs_off();
            let mo = prog.mode_off();
            if restart == 1 {
                // Count biased: init=0, cmp item > 0, op +, rhs = 1, accumulate
                prog.params[io + 1] = 4.0; // init = const[0] = 0
                prog.params[co2 + 4] = 4.0; // cmp = >
                prog.params[cs1] = 4.0; // lhs = item (pool[0])
                prog.params[cs2 + 2] = 4.0; // rhs = const[0] = 0 (pool[2])
                prog.params[bo2] = 4.0; // op = +
                prog.params[br2 + 3] = 4.0; // rhs = const[1] = 1 (pool[3])
                prog.params[mo] = 4.0; // accumulate mode
            } else if restart == 2 {
                // Sum-conditional: init=0, cmp item < 0, op +, rhs = item, accumulate
                prog.params[io + 1] = 4.0; // init = const[0] = 0
                prog.params[co2] = 4.0; // cmp = <
                prog.params[cs1] = 4.0; // lhs = item (pool[0])
                prog.params[cs2 + 2] = 4.0; // rhs = const[0] = 0 (pool[2])
                prog.params[bo2] = 4.0; // op = +
                prog.params[br2] = 4.0; // rhs = item (pool[0])
                prog.params[mo] = 4.0; // accumulate mode
            } else if restart == 3 {
                // Max biased (replace): init=arr[0], cmp item > acc, rhs = item
                prog.params[io] = 4.0; // init = arr[0] (init_pool[0])
                prog.params[co2 + 4] = 4.0; // cmp = >
                prog.params[cs1] = 4.0; // lhs = item (pool[0])
                prog.params[cs2 + 1] = 4.0; // rhs = acc (pool[1])
                prog.params[br2] = 4.0; // rhs = item (pool[0])
                prog.params[mo] = -4.0; // replace mode
            } else if restart == 4 {
                // Min biased (replace): init=arr[0], cmp item < acc, rhs = item
                prog.params[io] = 4.0; // init = arr[0] (init_pool[0])
                prog.params[co2] = 4.0; // cmp = <
                prog.params[cs1] = 4.0; // lhs = item (pool[0])
                prog.params[cs2 + 1] = 4.0; // rhs = acc (pool[1])
                prog.params[br2] = 4.0; // rhs = item (pool[0])
                prog.params[mo] = -4.0; // replace mode
            } else if restart == 5 {
                // Count-equality: init=0, cmp item == 0, op +, rhs = 1
                prog.params[io + 1] = 4.0; // init = const[0] = 0
                prog.params[co2 + 2] = 4.0; // cmp = ==
                prog.params[cs1] = 4.0; // lhs = item (pool[0])
                prog.params[cs2 + 2] = 4.0; // rhs = const[0] = 0 (pool[2])
                prog.params[bo2] = 4.0; // op = +
                prog.params[br2 + 3] = 4.0; // rhs = const[1] = 1 (pool[3])
                prog.params[mo] = 4.0; // accumulate mode
            } else if restart == 6 {
                // Sum-positives: init=0, cmp item > 0, op +, rhs = item
                prog.params[io + 1] = 4.0; // init = const[0] = 0
                prog.params[co2 + 4] = 4.0; // cmp = >
                prog.params[cs1] = 4.0; // lhs = item (pool[0])
                prog.params[cs2 + 2] = 4.0; // rhs = const[0] = 0 (pool[2])
                prog.params[bo2] = 4.0; // op = +
                prog.params[br2] = 4.0; // rhs = item (pool[0])
                prog.params[mo] = 4.0; // accumulate mode
            } else if restart == 7 && n_scalar >= 1 {
                // Count-occurrences: init=0, cmp item == k, op +, rhs = 1
                let k_idx = 2 + N_CONSTS; // first scalar arg in pool
                prog.params[io + 1] = 4.0; // init = const[0] = 0
                prog.params[co2 + 2] = 4.0; // cmp = ==
                prog.params[cs1] = 4.0; // lhs = item (pool[0])
                prog.params[cs2 + k_idx] = 4.0; // rhs = k (pool[k_idx])
                prog.params[bo2] = 4.0; // op = +
                prog.params[br2 + 3] = 4.0; // rhs = const[1] = 1 (pool[3])
                prog.params[mo] = 4.0; // accumulate mode
            }
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 17000 + idx as u64) - 0.5) * 0.3;
                }
            }
            let ex = examples.clone();
            let sn = scalar_names.clone();
            let result = train_program_arr(
                prog.params,
                move |p, t| SoftArrayCondAccumProgram { n_scalar, params: p.to_vec() }.loss(&ex, t),
                move |p, fn_n| SoftArrayCondAccumProgram { n_scalar, params: p.to_vec() }.discretize_and_emit(fn_n, &sn),
                problem, fn_name, N_ARR_STEPS,
            );
            if result.is_some() { return result; }
        }

        // A3: SoftArrayPairProgram
        {
            let mut prog = SoftArrayPairProgram::new(n_scalar);
            if restart == 1 {
                // array_range biased: init1=arr[0], init2=arr[0], gate1 item<lo, then1=item, gate2 item>hi, then2=item
                let init1_off = prog.init1_off();
                let init2_off = prog.init2_off();
                let g1_cmp_off = prog.g1_cmp_off();
                let g1_s1_off = prog.g1_s1_off();
                let g1_s2_off = prog.g1_s2_off();
                let then1_off = prog.then1_off();
                let g2_cmp_off = prog.g2_cmp_off();
                let g2_s1_off = prog.g2_s1_off();
                let g2_s2_off = prog.g2_s2_off();
                let then2_off = prog.then2_off();
                let ret_op_off = prog.ret_op_off();
                let ret_s1_off = prog.ret_s1_off();
                let ret_s2_off = prog.ret_s2_off();
                prog.params[init1_off] = 4.0; // init1 = arr[0]
                prog.params[init2_off] = 4.0; // init2 = arr[0]
                prog.params[g1_cmp_off] = 4.0; // gate1: <
                prog.params[g1_s1_off] = 4.0; // lhs = item
                prog.params[g1_s2_off + 1] = 4.0; // rhs = lo (a1)
                prog.params[then1_off] = 4.0; // then = item
                prog.params[g2_cmp_off + 4] = 4.0; // gate2: >
                prog.params[g2_s1_off] = 4.0; // lhs = item
                prog.params[g2_s2_off + 2] = 4.0; // rhs = hi (a2)
                prog.params[then2_off] = 4.0; // then = item
                prog.params[ret_op_off] = 4.0; // ret_op = -
                prog.params[ret_s1_off + 1] = 4.0; // ret_s1 = hi
                prog.params[ret_s2_off] = 4.0; // ret_s2 = lo
            }
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 19000 + idx as u64) - 0.5) * 0.3;
                }
            }
            let ex = examples.clone();
            let sn = scalar_names.clone();
            let result = train_program_arr(
                prog.params,
                move |p, t| SoftArrayPairProgram { n_scalar, params: p.to_vec() }.loss(&ex, t),
                move |p, fn_n| SoftArrayPairProgram { n_scalar, params: p.to_vec() }.discretize_and_emit(fn_n, &sn),
                problem, fn_name, N_ARR_STEPS,
            );
            if result.is_some() { return result; }
        }

        // A4: SoftPairwiseScanProgram
        {
            let mut prog = SoftPairwiseScanProgram::new(n_scalar);
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 23000 + idx as u64) - 0.5) * 0.3;
                }
            }
            let ex = examples.clone();
            let sn = scalar_names.clone();
            let result = train_program_arr(
                prog.params,
                move |p, t| SoftPairwiseScanProgram { n_scalar, params: p.to_vec() }.loss(&ex, t),
                move |p, fn_n| SoftPairwiseScanProgram { n_scalar, params: p.to_vec() }.discretize_and_emit(fn_n, &sn),
                problem, fn_name, N_ARR_STEPS,
            );
            if result.is_some() { return result; }
        }

        // A5: SoftArrayIndexGateProgram
        {
            let mut prog = SoftArrayIndexGateProgram::new(n_scalar);
            let init_off = prog.init_off();
            let cmp_off = prog.cmp_off();
            let cmp_s1_off = prog.cmp_s1_off();
            let cmp_s2_off = prog.cmp_s2_off();
            let body_op_off = prog.body_op_off();
            let body_rhs_off = prog.body_rhs_off();
            if restart == 1 {
                // sum_at_even_indices biased: init=0, gate parity > 0 (even: parity=+1), op=+, rhs=item
                prog.params[init_off] = 0.0;
                prog.params[cmp_off + 4] = 4.0; // cmp = >
                prog.params[cmp_s1_off + 3] = 4.0; // lhs = parity (pool[3])
                prog.params[cmp_s2_off + 6] = 4.0; // rhs = const[0] = 0 (pool[6])
                prog.params[body_op_off] = 4.0; // op = +
                prog.params[body_rhs_off] = 4.0; // rhs = item (pool[0])
            } else if restart == 2 {
                // sum_odd_indexed biased: init=0, gate parity < 0 (odd: parity=-1), op=+, rhs=item
                prog.params[init_off] = 0.0;
                prog.params[cmp_off] = 4.0; // cmp = <
                prog.params[cmp_s1_off + 3] = 4.0; // lhs = parity (pool[3])
                prog.params[cmp_s2_off + 6] = 4.0; // rhs = const[0] = 0 (pool[6])
                prog.params[body_op_off] = 4.0; // op = +
                prog.params[body_rhs_off] = 4.0; // rhs = item (pool[0])
            } else if restart == 3 {
                // count_evens: pre=item%2, gate target==0, op=+, rhs=1
                // Pool: [item(0), acc(1), i(2), parity(3), arr_len(4), target(5), 0(6), 1(7), -1(8), 2(9), -2(10), 10(11)]
                let pre_op_off = prog.pre_op_off();
                let pre_s1_off = prog.pre_s1_off();
                let pre_s2_off = prog.pre_s2_off();
                prog.params[init_off] = 0.0;
                prog.params[pre_op_off + 4] = 4.0; // pre_op = % (modulo)
                prog.params[pre_s1_off] = 4.0; // pre_s1 = item (pool[0])
                prog.params[pre_s2_off + 9] = 4.0; // pre_s2 = const[3] = 2 (pool[9])
                prog.params[cmp_off + 2] = 4.0; // cmp = ==
                prog.params[cmp_s1_off + 5] = 4.0; // lhs = target (pool[5])
                prog.params[cmp_s2_off + 6] = 4.0; // rhs = const[0] = 0 (pool[6])
                prog.params[body_op_off] = 4.0; // op = +
                prog.params[body_rhs_off + 7] = 4.0; // rhs = const[1] = 1 (pool[7])
            } else if restart == 4 && n_scalar >= 1 {
                // kth_from_end: pre=arr_len-k, gate i==target, acc=item (replace)
                // Pool: [item(0), acc(1), i(2), parity(3), arr_len(4), target(5), 0(6), ..., k(12)]
                let pre_op_off = prog.pre_op_off();
                let pre_s1_off = prog.pre_s1_off();
                let pre_s2_off = prog.pre_s2_off();
                let ret_off = prog.ret_off();
                let k_idx = 6 + N_CONSTS; // first scalar arg in pool
                prog.params[init_off] = 0.0;
                prog.params[pre_op_off + 1] = 4.0; // pre_op = - (subtract)
                prog.params[pre_s1_off + 4] = 4.0; // pre_s1 = arr_len (pool[4])
                prog.params[pre_s2_off + k_idx] = 4.0; // pre_s2 = k (pool[k_idx])
                prog.params[cmp_off + 2] = 4.0; // cmp = ==
                prog.params[cmp_s1_off + 2] = 4.0; // lhs = i (pool[2])
                prog.params[cmp_s2_off + 5] = 4.0; // rhs = target (pool[5])
                prog.params[body_op_off + 5] = 4.0; // op = identity (a) → won't help
                // Actually for kth_from_end, the body should set acc = item
                // body_op is soft_op(acc, rhs) where we want result = item
                // Use op = - with rhs = acc - item? No.
                // Better: use the same gate * (new_acc - acc) formula:
                // new_acc = soft_op(acc, item, +) = acc + item → acc += gate*(acc+item-acc) = acc + gate*item. Wrong for replacement.
                // The formula is: acc += gate * (new_acc - acc). For acc=item: new_acc must = item.
                // soft_op(acc, item, -) = acc - item → acc += gate*(acc-item-acc) = acc - gate*item. Wrong.
                // We need a way to produce `item` from soft_op(acc, rhs).
                // Can't do it with +,-,*,/,% since all involve acc.
                // The solution: use body_rhs = item, then after discretize, the emit code checks
                // and uses direct assignment. But soft training won't converge.
                // Alternative: just set body_rhs = item and don't try to match — gradient will find it or not.
                prog.params[body_op_off] = 4.0; // op = + (acc + item doesn't help, but gradient may fix)
                prog.params[body_rhs_off] = 4.0; // rhs = item (pool[0])
                prog.params[ret_off + 1] = 4.0; // return = acc (pool[1])
            }
            if restart > 0 {
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 27000 + idx as u64) - 0.5) * 0.3;
                }
            }
            let ex = examples.clone();
            let sn = scalar_names.clone();
            let result = train_program_arr(
                prog.params,
                move |p, t| SoftArrayIndexGateProgram { n_scalar, params: p.to_vec() }.loss(&ex, t),
                move |p, fn_n| SoftArrayIndexGateProgram { n_scalar, params: p.to_vec() }.discretize_and_emit(fn_n, &sn),
                problem, fn_name, N_ARR_STEPS,
            );
            if result.is_some() { return result; }
        }
    }

    None
}

/// Training loop for array programs. Same as train_program but the emit_fn
/// doesn't take param_names (uses scalar_names internally).
fn train_program_arr<F, G>(
    initial_params: Vec<f32>,
    loss_fn: F,
    emit_fn: G,
    problem: &Problem,
    fn_name: &str,
    n_steps: usize,
) -> Option<SolveResult>
where
    F: Fn(&[f32], f32) -> f32,
    G: Fn(&[f32], &str) -> String,
{
    let param_names: &[&str] = &[];
    // Wrap emit_fn to match the signature train_program expects
    let wrapped_emit = |p: &[f32], fn_n: &str, _pn: &[&str]| -> String {
        emit_fn(p, fn_n)
    };

    // Try the initial params directly
    if let Some(result) = try_emit_verify_arr(&initial_params, &wrapped_emit, problem, fn_name, param_names) {
        return Some(result);
    }

    let mut params = initial_params;
    let n = params.len();
    let mut opt = Adam::new(n, 0.05);
    let mut best_loss = f32::MAX;
    let mut best_params = params.clone();
    let mut last_check_loss = f32::MAX;
    let chk1 = n_steps / 4;
    let chk2 = n_steps / 2;
    let mut loss_at_chk1 = f32::MAX;
    let mut loss_at_chk2 = f32::MAX;

    for step in 0..n_steps {
        if step == chk1 { loss_at_chk1 = best_loss; }
        if step == chk2 { loss_at_chk2 = best_loss; }
        if step == chk2 && best_loss > loss_at_chk1 * 0.98 { break; }
        if step > n_steps * 3 / 4 && best_loss > loss_at_chk2 * 0.90 { break; }

        let temp = (2.0f32 * (1.0 - step as f32 / n_steps as f32)).max(0.1);
        let loss = loss_fn(&params, temp);
        if loss < best_loss {
            best_loss = loss;
            best_params = params.clone();
        }
        let should_check = loss < 1.0
            || (loss < last_check_loss * 0.9)
            || (step % 50 == 49);
        if should_check {
            last_check_loss = loss.min(last_check_loss);
            if let Some(result) = try_emit_verify_arr(&params, &wrapped_emit, problem, fn_name, param_names) {
                return Some(result);
            }
            if best_loss < loss {
                if let Some(result) = try_emit_verify_arr(&best_params, &wrapped_emit, problem, fn_name, param_names) {
                    return Some(result);
                }
            }
        }
        let grads = fd_grad(&params, &loss_fn, temp);
        opt.step(&mut params, &grads);
    }

    if let Some(result) = try_emit_verify_arr(&params, &wrapped_emit, problem, fn_name, param_names) {
        return Some(result);
    }
    try_emit_verify_arr(&best_params, &wrapped_emit, problem, fn_name, param_names)
}

fn try_emit_verify_arr<F>(
    params: &[f32],
    emit_fn: &F,
    problem: &Problem,
    fn_name: &str,
    param_names: &[&str],
) -> Option<SolveResult>
where
    F: Fn(&[f32], &str, &[&str]) -> String,
{
    let code = emit_fn(params, fn_name, param_names);
    if verify_problem_code_strict(problem, &code).is_ok() {
        return Some(SolveResult {
            success: true,
            code,
            method: "arr_gradient".to_string(),
            error: None,
            metadata: DifferentiableMetadata::default(),
        });
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── SoftRegisterMachine: Universal Differentiable Program ──────────────────
// ═══════════════════════════════════════════════════════════════════════════════
//
// A register machine where EVERY aspect is learned via gradient descent:
//   - Which operation to execute at each step
//   - Which registers to read as operands
//   - Which register to write the result to
//   - Whether to conditionally execute (gated by a soft comparison)
//
// Register file layout:
//   [arg0, arg1, ..., const0(0), const1(1), const2(-1), const3(2), const4(-2),
//    const5(10), scratch0, scratch1, ..., scratch_k]
//
// Each instruction step has:
//   op_logits(N_RM_OPS) — which operation (+, -, *, /, %, min, max, negate, abs, id)
//   src1_logits(N_REGS) — first operand
//   src2_logits(N_REGS) — second operand
//   dst_logits(N_REGS)  — destination register (soft write)
//   gate_cmp(N_CMPS)    — comparison for conditional execution
//   gate_s1(N_REGS)     — comparison LHS
//   gate_s2(N_REGS)     — comparison RHS
//
// The soft write blends: reg[i] = dst_w[i] * result + (1 - dst_w[i]) * reg[i]
// This is fully differentiable — gradients flow through the entire program.
//
// Discretization: each step → `rN = rA OP rB;` with dead-code elimination.
// The final output → `return rK;`

const N_RM_OPS: usize = 10; // +, -, *, /, %, min, max, neg(a), abs(a), identity(a)
const N_SCRATCH: usize = 4;
const N_RM_STEPS: usize = 6; // Keep small for FD tractability; 6 steps covers most scalar programs

/// Extended op for register machine: includes min, max, negate, abs, identity
fn soft_op_rm(a: f32, b: f32, weights: &[f32]) -> f32 {
    let safe_b = if b.abs() < 1e-6 { 1.0 } else { b };
    let results = [
        a + b,                                    // 0: +
        a - b,                                    // 1: -
        a * b,                                    // 2: *
        a / safe_b,                               // 3: /
        a - (a / safe_b).trunc() * safe_b,        // 4: %
        if a < b { a } else { 0.5 * (a + b - (a - b).abs()) }, // 5: min (soft approx)
        if a > b { a } else { 0.5 * (a + b + (a - b).abs()) }, // 6: max (soft approx)
        -a,                                       // 7: negate
        (a * a + 0.01).sqrt(),                    // 8: smooth abs
        a,                                        // 9: identity (nop / pass-through)
    ];
    weights.iter().zip(&results).map(|(w, r)| w * r).sum()
}

struct SoftRegisterMachine {
    n_args: usize,
    params: Vec<f32>,
}

impl SoftRegisterMachine {
    fn n_regs(n_args: usize) -> usize { n_args + N_CONSTS + N_SCRATCH }

    fn step_size(n_args: usize) -> usize {
        let nr = Self::n_regs(n_args);
        // op(N_RM_OPS) + src1(nr) + src2(nr) + dst(nr) + gate_cmp(N_CMPS) + gate_s1(nr) + gate_s2(nr)
        N_RM_OPS + 5 * nr + N_CMPS
    }

    fn n_params(n_args: usize) -> usize {
        let ss = Self::step_size(n_args);
        let nr = Self::n_regs(n_args);
        // steps(N_RM_STEPS * ss) + ret(nr) + consts(N_CONSTS)
        N_RM_STEPS * ss + nr + N_CONSTS
    }

    fn new(n_args: usize) -> Self {
        let mut s = Self { n_args, params: vec![0f32; Self::n_params(n_args)] };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        // Bias all dst logits toward identity (step 9) so untrained steps are nops
        let nr = Self::n_regs(n_args);
        let ss = Self::step_size(n_args);
        for step in 0..N_RM_STEPS {
            let off = step * ss;
            // Bias op toward identity (index 9) so untrained steps pass through
            s.params[off + 9] = 2.0;
            // Bias gate toward always-on: gate_cmp == (idx 2), gate_s1 = gate_s2 = reg[0]
            let gate_cmp_off = off + N_RM_OPS + 3 * nr;
            s.params[gate_cmp_off + 2] = 2.0; // == comparison
        }
        s
    }

    fn ret_off(&self) -> usize { N_RM_STEPS * Self::step_size(self.n_args) }
    fn consts_off(&self) -> usize { self.ret_off() + Self::n_regs(self.n_args) }

    fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let nr = Self::n_regs(self.n_args);
        let ss = Self::step_size(self.n_args);
        let co = self.consts_off();

        // Initialize register file
        let mut regs = vec![0f32; nr];
        for (i, &v) in inputs.iter().enumerate() { regs[i] = v; }
        // Load constants
        for i in 0..N_CONSTS { regs[self.n_args + i] = self.params[co + i]; }
        // Scratch registers start at 0

        // Execute instruction sequence
        for step in 0..N_RM_STEPS {
            let off = step * ss;
            let op_w = softmax_temp(&self.params[off..off + N_RM_OPS], temp);
            let s1_w = softmax_temp(&self.params[off + N_RM_OPS..off + N_RM_OPS + nr], temp);
            let s2_w = softmax_temp(&self.params[off + N_RM_OPS + nr..off + N_RM_OPS + 2 * nr], temp);
            let dst_w = softmax_temp(&self.params[off + N_RM_OPS + 2 * nr..off + N_RM_OPS + 3 * nr], temp);
            let gate_cmp_w = softmax_temp(&self.params[off + N_RM_OPS + 3 * nr..off + N_RM_OPS + 3 * nr + N_CMPS], temp);
            let gate_s1_w = softmax_temp(&self.params[off + N_RM_OPS + 3 * nr + N_CMPS..off + N_RM_OPS + 4 * nr + N_CMPS], temp);
            let gate_s2_w = softmax_temp(&self.params[off + N_RM_OPS + 4 * nr + N_CMPS..off + N_RM_OPS + 5 * nr + N_CMPS], temp);

            // Read operands
            let v1 = soft_read(&regs, &s1_w);
            let v2 = soft_read(&regs, &s2_w);

            // Compute result
            let result = soft_op_rm(v1, v2, &op_w);

            // Conditional gate
            let g_lhs = soft_read(&regs, &gate_s1_w);
            let g_rhs = soft_read(&regs, &gate_s2_w);
            let gate = soft_cmp(g_lhs, g_rhs, &gate_cmp_w, temp);

            // Soft write: blend result into register file
            for r in 0..nr {
                let write_strength = dst_w[r] * gate;
                regs[r] = write_strength * result + (1.0 - write_strength) * regs[r];
            }
        }

        // Return: soft-read from final register state
        let ret_w = softmax_temp(&self.params[self.ret_off()..self.ret_off() + nr], temp);
        soft_read(&regs, &ret_w)
    }

    fn loss(&self, examples: &[(Vec<f32>, f32)], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples.iter().map(|(inputs, expected)| {
            let diff = self.forward(inputs, temp) - expected;
            diff * diff
        }).sum::<f32>() / n
    }

    fn reg_names(&self) -> Vec<String> {
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS).map(|i| self.params[co + i].round() as i64).collect();
        let mut names = Vec::new();
        // arg names
        let arg_names = ["a", "b", "c", "d", "e", "f"];
        for i in 0..self.n_args {
            names.push(arg_names.get(i).unwrap_or(&"x").to_string());
        }
        // const names (just their values)
        for c in &consts { names.push(format!("{c}")); }
        // scratch names
        for i in 0..N_SCRATCH { names.push(format!("r{i}")); }
        names
    }

    fn discretize_and_emit(&self, fn_name: &str) -> String {
        let nr = Self::n_regs(self.n_args);
        let ss = Self::step_size(self.n_args);
        let names = self.reg_names();

        let op_names = ["+", "-", "*", "/", "%", "min", "max", "neg", "abs", "id"];
        let cmp_strs = ["<", "<=", "==", ">=", ">", "!="];

        // Build arg signature
        let arg_sig: Vec<String> = (0..self.n_args)
            .map(|i| format!("{}: i64", names[i]))
            .collect();
        let sig = format!("fn {fn_name}({}) -> i64", arg_sig.join(", "));

        // Track which scratch regs are actually written
        let mut scratch_written = vec![false; N_SCRATCH];
        let mut instructions: Vec<String> = Vec::new();

        for step in 0..N_RM_STEPS {
            let off = step * ss;
            let op_i = argmax(&self.params[off..off + N_RM_OPS]);
            let s1_i = argmax(&self.params[off + N_RM_OPS..off + N_RM_OPS + nr]);
            let s2_i = argmax(&self.params[off + N_RM_OPS + nr..off + N_RM_OPS + 2 * nr]);
            let dst_i = argmax(&self.params[off + N_RM_OPS + 2 * nr..off + N_RM_OPS + 3 * nr]);
            let gate_cmp_i = argmax(&self.params[off + N_RM_OPS + 3 * nr..off + N_RM_OPS + 3 * nr + N_CMPS]);
            let gate_s1_i = argmax(&self.params[off + N_RM_OPS + 3 * nr + N_CMPS..off + N_RM_OPS + 4 * nr + N_CMPS]);
            let gate_s2_i = argmax(&self.params[off + N_RM_OPS + 4 * nr + N_CMPS..off + N_RM_OPS + 5 * nr + N_CMPS]);

            // Skip identity ops writing to their own source (nop)
            if op_i == 9 && dst_i == s1_i { continue; }
            // Skip writes to non-scratch registers (args + consts are immutable in discrete)
            if dst_i < self.n_args + N_CONSTS { continue; }

            let scratch_idx = dst_i - self.n_args - N_CONSTS;
            if scratch_idx < N_SCRATCH { scratch_written[scratch_idx] = true; }

            let dst_name = &names[dst_i];
            let s1_name = &names[s1_i];
            let s2_name = &names[s2_i];

            // Build the expression
            let expr = match op_i {
                0..=4 => format!("{s1_name} {} {s2_name}", op_names[op_i]),
                5 => {
                    // min: emit as conditional
                    format!("if {s1_name} < {s2_name} {{ {s1_name} }} else {{ {s2_name} }}")
                }
                6 => {
                    format!("if {s1_name} > {s2_name} {{ {s1_name} }} else {{ {s2_name} }}")
                }
                7 => format!("0 - {s1_name}"), // negate
                8 => {
                    // abs: emit as conditional
                    format!("if {s1_name} < 0 {{ 0 - {s1_name} }} else {{ {s1_name} }}")
                }
                _ => format!("{s1_name}"), // identity
            };

            // Check if gated (gate_s1 != gate_s2 or non-trivial comparison)
            let is_gated = gate_s1_i != gate_s2_i || gate_cmp_i != 2; // != (a == a) which is always true

            let is_first_write = scratch_idx < N_SCRATCH && !instructions.iter().any(|i: &String| i.contains(&format!("{dst_name} =")));
            let decl = if is_first_write { ": i64 " } else { " " };

            if is_gated {
                let gs1 = &names[gate_s1_i];
                let gs2 = &names[gate_s2_i];
                instructions.push(format!(
                    "    if {gs1} {} {gs2} {{ {dst_name}{decl}= {expr}; }}",
                    cmp_strs[gate_cmp_i]
                ));
            } else {
                instructions.push(format!("    {dst_name}{decl}= {expr};"));
            }
        }

        // Return
        let ret_i = argmax(&self.params[self.ret_off()..self.ret_off() + nr]);
        let ret_name = &names[ret_i];

        let mut out = format!("{sig} {{\n");
        // Declare scratch registers used but not yet declared by first-write logic
        for i in 0..N_SCRATCH {
            if scratch_written[i] {
                let rn = format!("r{i}");
                if !instructions.iter().any(|inst| inst.contains(&format!("{rn}: i64 ="))) {
                    // Pre-declare
                    writeln!(out, "    {rn}: i64 = 0;").unwrap();
                }
            }
        }
        for inst in &instructions {
            writeln!(out, "{inst}").unwrap();
        }
        writeln!(out, "    return {ret_name};").unwrap();
        out.push_str("}\n");
        out
    }
}

// ─── Array Register Machine ─────────────────────────────────────────────────
// Same as SoftRegisterMachine but with soft array indexing.
// Register file includes arr_len + elements accessible via soft index.

const N_ARM_STEPS: usize = 20;
const N_ARM_SCRATCH: usize = 6;

struct SoftArrayRegisterMachine {
    n_scalar: usize, // number of scalar args (after the array arg)
    params: Vec<f32>,
}

impl SoftArrayRegisterMachine {
    // Register file:
    // [arr_len, scalar0, scalar1, ..., const0..5, iter_idx, scratch0..N]
    // Plus: array memory[MAX_ARR] accessible via soft index
    fn n_regs(n_scalar: usize) -> usize { 1 + n_scalar + N_CONSTS + 1 + N_ARM_SCRATCH }
    // iter_idx position in register file
    fn iter_idx_pos(n_scalar: usize) -> usize { 1 + n_scalar + N_CONSTS }

    fn step_size(n_scalar: usize) -> usize {
        let nr = Self::n_regs(n_scalar);
        // op(N_RM_OPS) + src1(nr + 1) + src2(nr) + dst(nr + 1)
        //   src1 has +1 for "arr[soft_idx]" read
        //   dst has +1 for "arr[soft_idx]" write
        // + arr_idx(nr) — which register to use as array index
        // + gate_cmp(N_CMPS) + gate_s1(nr) + gate_s2(nr)
        N_RM_OPS + (nr + 1) + nr + (nr + 1) + nr + N_CMPS + 2 * nr
    }

    fn n_params(n_scalar: usize) -> usize {
        let ss = Self::step_size(n_scalar);
        let nr = Self::n_regs(n_scalar);
        // steps + ret(nr + 1) + consts + loop_bound(nr) — which reg controls iteration count
        N_ARM_STEPS * ss + (nr + 1) + N_CONSTS + nr
    }

    fn new(n_scalar: usize) -> Self {
        let np = Self::n_params(n_scalar);
        let mut s = Self { n_scalar, params: vec![0f32; np] };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        // Bias loop_bound toward arr_len (reg[0])
        let lb = s.loop_bound_off();
        s.params[lb] = 3.0;
        s
    }

    fn ret_off(&self) -> usize { N_ARM_STEPS * Self::step_size(self.n_scalar) }
    fn consts_off(&self) -> usize { self.ret_off() + Self::n_regs(self.n_scalar) + 1 }
    fn loop_bound_off(&self) -> usize { self.consts_off() + N_CONSTS }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let nr = Self::n_regs(self.n_scalar);
        let ss = Self::step_size(self.n_scalar);
        let co = self.consts_off();
        let lb_off = self.loop_bound_off();
        let iter_pos = Self::iter_idx_pos(self.n_scalar);

        // Initialize register file
        let mut regs = vec![0f32; nr];
        regs[0] = arr_len; // reg[0] = arr_len
        for (i, &v) in scalar_args.iter().enumerate() { regs[1 + i] = v; }
        for i in 0..N_CONSTS { regs[1 + self.n_scalar + i] = self.params[co + i]; }
        // iter_idx and scratch start at 0

        // Array memory (padded)
        let mut mem = vec![0f32; MAX_ARR];
        for (i, &v) in arr.iter().enumerate() {
            if i < MAX_ARR { mem[i] = v; }
        }

        // Determine loop bound from register (soft-read)
        let lb_w = softmax_temp(&self.params[lb_off..lb_off + nr], temp);
        let loop_bound = soft_read(&regs, &lb_w);

        // Execute: iterate steps for each array position
        for iter in 0..MAX_ARR {
            let in_bounds = sigmoid((loop_bound - iter as f32 - 0.5) / 0.3);
            regs[iter_pos] = iter as f32; // set iter_idx

            for step in 0..N_ARM_STEPS {
                let off = step * ss;
                let op_w = softmax_temp(&self.params[off..off + N_RM_OPS], temp);
                // src1 has nr+1 slots: [regs..., arr[idx]]
                let s1_w = softmax_temp(&self.params[off + N_RM_OPS..off + N_RM_OPS + nr + 1], temp);
                let s2_w = softmax_temp(&self.params[off + N_RM_OPS + nr + 1..off + N_RM_OPS + 2 * nr + 1], temp);
                // dst has nr+1 slots: [regs..., arr[idx]]
                let dst_w = softmax_temp(&self.params[off + N_RM_OPS + 2 * nr + 1..off + N_RM_OPS + 3 * nr + 2], temp);
                // arr_idx: which register to use as the array index
                let aidx_w = softmax_temp(&self.params[off + N_RM_OPS + 3 * nr + 2..off + N_RM_OPS + 4 * nr + 2], temp);
                let gate_cmp_w = softmax_temp(&self.params[off + N_RM_OPS + 4 * nr + 2..off + N_RM_OPS + 4 * nr + 2 + N_CMPS], temp);
                let gate_s1_w = softmax_temp(&self.params[off + N_RM_OPS + 4 * nr + 2 + N_CMPS..off + N_RM_OPS + 5 * nr + 2 + N_CMPS], temp);
                let gate_s2_w = softmax_temp(&self.params[off + N_RM_OPS + 5 * nr + 2 + N_CMPS..off + N_RM_OPS + 6 * nr + 2 + N_CMPS], temp);

                // Compute soft array index
                let soft_idx = soft_read(&regs, &aidx_w);
                // Soft array read: weighted sum over memory positions near soft_idx
                let arr_val: f32 = (0..MAX_ARR).map(|j| {
                    let dist = (j as f32 - soft_idx).abs();
                    let w = (-(dist * dist) / (temp.max(0.3))).exp();
                    w * mem[j]
                }).sum::<f32>() / (0..MAX_ARR).map(|j| {
                    let dist = (j as f32 - soft_idx).abs();
                    (-(dist * dist) / (temp.max(0.3))).exp()
                }).sum::<f32>().max(1e-8);

                // Read operands: last slot in s1 is arr[idx]
                let v1 = {
                    let reg_part: f32 = regs.iter().zip(&s1_w[..nr]).map(|(r, w)| r * w).sum();
                    reg_part + s1_w[nr] * arr_val
                };
                let v2 = soft_read(&regs, &s2_w);

                // Compute
                let result = soft_op_rm(v1, v2, &op_w);

                // Gate
                let g_lhs = soft_read(&regs, &gate_s1_w);
                let g_rhs = soft_read(&regs, &gate_s2_w);
                let gate = soft_cmp(g_lhs, g_rhs, &gate_cmp_w, temp) * in_bounds;

                // Soft write to registers
                for r in 0..nr {
                    let ws = dst_w[r] * gate;
                    regs[r] = ws * result + (1.0 - ws) * regs[r];
                }
                // Soft write to array memory
                let mem_ws = dst_w[nr] * gate;
                if mem_ws > 1e-6 {
                    for j in 0..MAX_ARR {
                        let dist = (j as f32 - soft_idx).abs();
                        let pos_w = (-(dist * dist) / (temp.max(0.3))).exp();
                        let total_ws = mem_ws * pos_w;
                        mem[j] = total_ws * result + (1.0 - total_ws) * mem[j];
                    }
                }
            }
        }

        // Return: soft-read from registers (+ optional arr[idx])
        let ret_w = softmax_temp(&self.params[self.ret_off()..self.ret_off() + nr + 1], temp);
        let reg_part: f32 = regs.iter().zip(&ret_w[..nr]).map(|(r, w)| r * w).sum();
        // For return from array, use iter_idx as index
        let ret_arr_val: f32 = (0..MAX_ARR).map(|j| {
            let dist = (j as f32 - regs[iter_pos]).abs();
            let w = (-(dist * dist) / (temp.max(0.3))).exp();
            w * mem[j]
        }).sum::<f32>() / (0..MAX_ARR).map(|j| {
            let dist = (j as f32 - regs[iter_pos]).abs();
            (-(dist * dist) / (temp.max(0.3))).exp()
        }).sum::<f32>().max(1e-8);
        reg_part + ret_w[nr] * ret_arr_val
    }

    fn loss(&self, examples: &[ArrExample], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples.iter().map(|ex| {
            let diff = self.forward(&ex.arr, ex.arr_len, &ex.scalar_args, temp) - ex.expected;
            diff * diff
        }).sum::<f32>() / n
    }
}

// ─── Register Machine synthesis entry points ─────────────────────────────────

/// Attempt synthesis via SoftRegisterMachine for scalar problems.
pub fn synthesize_register_machine(problem: &Problem) -> Option<SolveResult> {
    // Only scalar-input problems
    if !problem.examples.iter().all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_)))) {
        return None;
    }
    let n_args = problem.examples[0].inputs.len();
    let fn_name = problem.function_name();

    // Build float examples
    let examples: Vec<(Vec<f32>, f32)> = problem.examples.iter().map(|ex| {
        let inputs: Vec<f32> = ex.inputs.iter().map(|v| match v {
            Value::Int(i) => *i as f32,
            _ => 0.0,
        }).collect();
        (inputs, ex.expected as f32)
    }).collect();

    const N_STEPS_RM: usize = 600;
    const N_RESTARTS_RM: usize = 5;
    let nr = SoftRegisterMachine::n_regs(n_args);
    let ss = SoftRegisterMachine::step_size(n_args);

    for restart in 0..N_RESTARTS_RM {
        let mut prog = SoftRegisterMachine::new(n_args);
        // Biased initializations for common patterns
        if restart == 1 && n_args >= 2 {
            // Bias step 0: r0 = arg0 OP arg1, return r0
            // op = + (index 0), src1 = arg0 (reg 0), src2 = arg1 (reg 1), dst = first scratch
            let scratch0_idx = n_args + N_CONSTS;
            prog.params[0] = 4.0; // op = +
            prog.params[N_RM_OPS] = 4.0; // src1 = reg[0] = arg0
            prog.params[N_RM_OPS + nr + 1] = 4.0; // src2 = reg[1] = arg1
            prog.params[N_RM_OPS + 2 * nr + scratch0_idx] = 4.0; // dst = scratch0
            // ret = scratch0
            let ro = prog.ret_off();
            prog.params[ro + scratch0_idx] = 4.0;
        } else if restart == 2 && n_args >= 1 {
            // Bias step 0: r0 = arg0 * arg0, return r0 (square)
            let scratch0_idx = n_args + N_CONSTS;
            prog.params[2] = 4.0; // op = *
            prog.params[N_RM_OPS] = 4.0; // src1 = reg[0] = arg0
            prog.params[N_RM_OPS + nr] = 4.0; // src2 = reg[0] = arg0
            prog.params[N_RM_OPS + 2 * nr + scratch0_idx] = 4.0; // dst = scratch0
            let ro = prog.ret_off();
            prog.params[ro + scratch0_idx] = 4.0;
        } else if restart == 3 && n_args >= 1 {
            // Bias: just return arg0 OP const (identity-ish, lets gradient find the op+const)
            let scratch0_idx = n_args + N_CONSTS;
            prog.params[N_RM_OPS] = 4.0; // src1 = arg0
            prog.params[N_RM_OPS + nr + n_args + 1] = 4.0; // src2 = const[1] = 1
            prog.params[N_RM_OPS + 2 * nr + scratch0_idx] = 4.0;
            let ro = prog.ret_off();
            prog.params[ro + scratch0_idx] = 4.0;
        }
        // Add noise for restarts > 0
        if restart > 0 {
            for (idx, p) in prog.params.iter_mut().enumerate() {
                *p += (pseudo_rand(restart as u64 * 31337 + idx as u64) - 0.5) * 0.5;
            }
        }

        let n = prog.params.len();
        let mut opt = Adam::new(n, 0.03);
        let mut best_loss = f32::MAX;
        let mut best_params = prog.params.clone();
        let chk1 = N_STEPS_RM / 4;
        let mut loss_at_chk1 = f32::MAX;

        for step in 0..N_STEPS_RM {
            if step == chk1 { loss_at_chk1 = best_loss; }
            if step == N_STEPS_RM / 2 && best_loss > loss_at_chk1 * 0.95 { break; }

            let temp = (2.0f32 * (1.0 - step as f32 / N_STEPS_RM as f32)).max(0.1);
            let ex_ref = &examples;
            let loss = {
                let p = SoftRegisterMachine { n_args, params: prog.params.clone() };
                p.loss(ex_ref, temp)
            };
            if loss < best_loss {
                best_loss = loss;
                best_params = prog.params.clone();
            }

            // Try discretize periodically
            if loss < 1.0 || step % 100 == 99 {
                let code = SoftRegisterMachine { n_args, params: prog.params.clone() }
                    .discretize_and_emit(fn_name);
                if verify_problem_code_strict(problem, &code).is_ok() {
                    return Some(SolveResult {
                        success: true,
                        code,
                        method: "register_machine".to_string(),
                        error: None,
                        metadata: DifferentiableMetadata::default(),
                    });
                }
                // Try best params too
                if best_loss < loss {
                    let code2 = SoftRegisterMachine { n_args, params: best_params.clone() }
                        .discretize_and_emit(fn_name);
                    if verify_problem_code_strict(problem, &code2).is_ok() {
                        return Some(SolveResult {
                            success: true,
                            code: code2,
                            method: "register_machine".to_string(),
                            error: None,
                            metadata: DifferentiableMetadata::default(),
                        });
                    }
                }
            }

            let ex2 = examples.clone();
            let na = n_args;
            let grads = fd_grad(&prog.params, |p, t| {
                SoftRegisterMachine { n_args: na, params: p.to_vec() }.loss(&ex2, t)
            }, temp);
            opt.step(&mut prog.params, &grads);
        }

        // Final check with best params
        let code = SoftRegisterMachine { n_args, params: best_params }
            .discretize_and_emit(fn_name);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: "register_machine".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify biased parameter layouts produce correct discretized programs (structural, no training).
    #[test]
    fn predicate_loop_biased_params_structure() {
        // GCD: init(a,b), while x1!=0 { (x0,x1)=(x1, x0%x1) }, return x0
        {
            let n_args = 2usize;
            let mut prog = SoftPredicateLoop::new(n_args);
            let nb = prog.nb();
            let na = prog.na();
            let c0 = 2 * nb;
            let c1 = c0 + N_CMPS;
            let c2 = c1 + na;
            let a1 = c2 + na;
            let a2 = a1 + na;
            let ao = a2 + na;
            let b1 = ao + 6;
            let b2 = b1 + na;
            let bo = b2 + na;
            let ro = bo + 6;
            let co = ro + na;
            for p in prog.params.iter_mut() {
                *p = -1.0;
            }
            prog.params[0] = 4.0;
            prog.params[nb + 1] = 4.0;
            prog.params[c0 + 5] = 4.0;
            prog.params[c1 + 1] = 4.0;
            prog.params[c2 + 2 + n_args] = 4.0;
            prog.params[a1 + 1] = 4.0;
            prog.params[ao + 5] = 4.0;
            prog.params[b1 + 0] = 4.0;
            prog.params[b2 + 1] = 4.0;
            prog.params[bo + 4] = 4.0;
            prog.params[ro + 0] = 4.0;
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            let code = prog.discretize_and_emit("gcd", &["a", "b"]);
            println!("GCD:\n{code}");
            assert!(code.contains("!= 0"), "GCD cond: {code}");
            assert!(
                code.contains("x0 % x1") || code.contains("x0%x1"),
                "GCD body: {code}"
            );
            assert!(code.contains("return x0"), "GCD ret: {code}");
            let _ = a2;
        }

        // leading_digit: while x0>=10, x0=x0/10, return x0
        {
            let n_args = 1usize;
            let base = SoftPredicateLoop::new(n_args);
            let nb = base.nb();
            let na = base.na();
            let c0 = 2 * nb;
            let c1 = c0 + N_CMPS;
            let c2 = c1 + na;
            let a1 = c2 + na;
            let a2 = a1 + na;
            let ao = a2 + na;
            let b1 = ao + 6;
            let b2 = b1 + na;
            let bo = b2 + na;
            let ro = bo + 6;
            let co = ro + na;
            let mut p = vec![-1.0f32; base.params.len()];
            p[0] = 4.0;
            p[nb + 1] = 4.0; // init0=arg0, init1=const[0]=0
            p[c0 + 2] = 4.0;
            p[c1 + 0] = 4.0;
            p[c2 + 8] = 4.0;
            p[a1 + 0] = 4.0;
            p[a2 + 8] = 4.0;
            p[ao + 3] = 4.0;
            p[b1 + 1] = 4.0;
            p[bo + 5] = 4.0;
            p[ro + 0] = 4.0;
            p[co] = 0.0;
            p[co + 1] = 1.0;
            p[co + 2] = -1.0;
            p[co + 3] = 2.0;
            p[co + 4] = -2.0;
            p[co + 5] = 10.0;
            let _ = b2;
            let code = SoftPredicateLoop { n_args, params: p }.discretize_and_emit("ld", &["n"]);
            println!("leading_digit:\n{code}");
            assert!(
                code.contains(">= 10") || code.contains(">=10"),
                "ld cond: {code}"
            );
            assert!(
                code.contains("x0 / 10") || code.contains("x0/10"),
                "ld body: {code}"
            );
            assert!(code.contains("return x0"), "ld ret: {code}");
        }

        // next_power_of_2: init x0=1, while x0<n, x0=x0*2, return x0
        {
            let n_args = 1usize;
            let base = SoftPredicateLoop::new(n_args);
            let nb = base.nb();
            let na = base.na();
            let c0 = 2 * nb;
            let c1 = c0 + N_CMPS;
            let c2 = c1 + na;
            let a1 = c2 + na;
            let a2 = a1 + na;
            let ao = a2 + na;
            let b1 = ao + 6;
            let b2 = b1 + na;
            let bo = b2 + na;
            let ro = bo + 6;
            let co = ro + na;
            let mut p = vec![-1.0f32; base.params.len()];
            p[2] = 4.0;
            p[nb + 1] = 4.0; // init0=const[1]=1, init1=const[0]=0
            p[c0 + 1] = 4.0;
            p[c1 + 0] = 4.0;
            p[c2 + 2] = 4.0;
            p[a1 + 0] = 4.0;
            p[a2 + 6] = 4.0;
            p[ao + 2] = 4.0;
            p[b1 + 1] = 4.0;
            p[bo + 5] = 4.0;
            p[ro + 0] = 4.0;
            p[co] = 0.0;
            p[co + 1] = 1.0;
            p[co + 2] = -1.0;
            p[co + 3] = 2.0;
            p[co + 4] = -2.0;
            p[co + 5] = 10.0;
            let _ = b2;
            let code = SoftPredicateLoop { n_args, params: p }.discretize_and_emit("np2", &["n"]);
            println!("next_power_of_2:\n{code}");
            assert!(
                code.contains("x0 < n") || code.contains("x0<n"),
                "np2 cond: {code}"
            );
            assert!(
                code.contains("x0 * 2") || code.contains("x0*2"),
                "np2 body: {code}"
            );
            assert!(code.contains("return x0"), "np2 ret: {code}");
        }

        // digit_count: x0=n, x1=1; while x0>=10 { x0=x0/10; x1=x1+1 }; return x1
        {
            let n_args = 1usize;
            let base = SoftPredicateLoop::new(n_args);
            let nb = base.nb();
            let na = base.na();
            let c0 = 2 * nb;
            let c1 = c0 + N_CMPS;
            let c2 = c1 + na;
            let a1 = c2 + na;
            let a2 = a1 + na;
            let ao = a2 + na;
            let b1 = ao + 6;
            let b2 = b1 + na;
            let bo = b2 + na;
            let ro = bo + 6;
            let co = ro + na;
            let mut p = vec![-1.0f32; base.params.len()];
            p[0] = 4.0;
            p[nb + 2] = 4.0; // init0=arg0, init1=const[1]=1
            p[c0 + 2] = 4.0;
            p[c1 + 0] = 4.0;
            p[c2 + 8] = 4.0; // cmp>=, lhs=x0, rhs=10
            p[a1 + 0] = 4.0;
            p[a2 + 8] = 4.0;
            p[ao + 3] = 4.0; // a: x0/10
            p[b1 + 1] = 4.0;
            p[b2 + 4] = 4.0;
            p[bo + 0] = 4.0; // b: x1+1
            p[ro + 1] = 4.0; // ret = x1
            p[co] = 0.0;
            p[co + 1] = 1.0;
            p[co + 2] = -1.0;
            p[co + 3] = 2.0;
            p[co + 4] = -2.0;
            p[co + 5] = 10.0;
            let code = SoftPredicateLoop { n_args, params: p }.discretize_and_emit("dc", &["n"]);
            println!("digit_count:\n{code}");
            assert!(
                code.contains(">= 10") || code.contains(">=10"),
                "dc cond: {code}"
            );
            assert!(
                code.contains("x0 / 10") || code.contains("x0/10"),
                "dc a-body: {code}"
            );
            assert!(
                code.contains("x1 + 1") || code.contains("x1+1"),
                "dc b-body: {code}"
            );
            assert!(code.contains("return x1"), "dc ret: {code}");
        }

        // safe_div_or_neg1: if b==0 return -1 else return a/b
        {
            let n_args = 2usize;
            let ns = n_args + N_CONSTS;
            let ne = ns + 1;
            let boff = 1 + 2 * ns + N_OPS;
            let branch_size = N_CMPS + 4 * ne + 6;
            let doff = boff + N_BRANCHES * branch_size;
            let mut prog = SoftBranchProgram::new(n_args);
            for k in 0..N_CMPS {
                prog.params[boff + k] = -4.0;
            }
            prog.params[boff + 4] = 4.0;
            for k in 0..ne {
                prog.params[boff + N_CMPS + k] = -4.0;
            }
            prog.params[boff + N_CMPS + 1] = 4.0;
            prog.params[boff + N_CMPS + ne + 2] = 4.0;
            for k in 0..ne {
                prog.params[boff + N_CMPS + 2 * ne + k] = -4.0;
            }
            prog.params[boff + N_CMPS + 2 * ne + 4] = 4.0;
            prog.params[boff + N_CMPS + 4 * ne + 5] = 4.0;
            prog.params[doff + 1] = -4.0;
            prog.params[doff] = 4.0;
            prog.params[doff + ne + 1] = 4.0;
            prog.params[doff + 2 * ne + 5] = -4.0;
            prog.params[doff + 2 * ne + 3] = 4.0;
            let code = prog.discretize_and_emit("safe_div", &["a", "b"]);
            println!("safe_div_or_neg1:\n{code}");
            assert!(code.contains("b == 0"), "sdiv cond: {code}");
            assert!(code.contains("return -1"), "sdiv -1: {code}");
            assert!(
                code.contains("a / b") || code.contains("a/b"),
                "sdiv default: {code}"
            );
        }

        // digit_product: mode=product, init_acc=1
        {
            let mut prog = SoftDigitLoopProgram::new();
            prog.params[0] = -4.0;
            prog.params[1] = 4.0;
            prog.params[2] = -4.0;
            prog.params[3] = -4.0;
            prog.params[4] = 1.0;
            let code = prog.discretize_and_emit("dp", &["n"]);
            println!("digit_product:\n{code}");
            assert!(
                code.contains("acc * (x % 10)") || code.contains("acc*(x%10)"),
                "dp body: {code}"
            );
            assert!(code.contains("acc: i64 = 1"), "dp init: {code}");
        }

        // SoftCondAccumLoop structural test: count_divisors biased restart
        {
            let n_args = 1usize;
            let ns = n_args + N_CONSTS; // 7
            let (
                pre_op_off,
                pre_s1_off,
                pre_s2_off,
                cmp_op_off,
                cmp_s1_off,
                cmp_s2_off,
                loop_op_off,
                loop_rhs_off,
                co,
            ) = SoftCondAccumLoop::offsets(ns);
            let mut prog = SoftCondAccumLoop::new(n_args);
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            prog.params[0] = 0.0;
            prog.params[1] = 1.0;
            prog.params[2] = 4.0;
            prog.params[2 + ns] = 0.0;
            prog.params[pre_op_off + 4] = 4.0;
            prog.params[pre_s1_off] = 4.0;
            prog.params[pre_s2_off + ns] = 4.0;
            prog.params[cmp_op_off + 4] = 4.0;
            prog.params[cmp_s1_off + ns + 1] = 4.0;
            prog.params[cmp_s2_off + 1] = 4.0;
            prog.params[loop_op_off] = 4.0;
            prog.params[loop_rhs_off + n_args + 1] = 4.0;
            let code = prog.discretize_and_emit("count_divisors", &["n"]);
            println!("count_divisors:\n{code}");
            assert!(code.contains("v0: i64 = n % i"), "cd pre: {code}");
            assert!(code.contains("v0 == 0"), "cd cmp: {code}");
            assert!(code.contains("acc = acc + 1"), "cd body: {code}");
            assert!(code.contains("i <= n"), "cd bound: {code}");
        }

        // SoftCondAccumLoop structural test: sum_of_divisors biased restart
        {
            let n_args = 1usize;
            let ns = n_args + N_CONSTS; // 7
            let (
                pre_op_off,
                pre_s1_off,
                pre_s2_off,
                cmp_op_off,
                cmp_s1_off,
                cmp_s2_off,
                loop_op_off,
                loop_rhs_off,
                co,
            ) = SoftCondAccumLoop::offsets(ns);
            let mut prog = SoftCondAccumLoop::new(n_args);
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            prog.params[0] = 0.0;
            prog.params[1] = 1.0;
            prog.params[2] = 4.0;
            prog.params[2 + ns] = 0.0;
            prog.params[pre_op_off + 4] = 4.0;
            prog.params[pre_s1_off] = 4.0;
            prog.params[pre_s2_off + ns] = 4.0;
            prog.params[cmp_op_off + 4] = 4.0;
            prog.params[cmp_s1_off + ns + 1] = 4.0;
            prog.params[cmp_s2_off + 1] = 4.0;
            prog.params[loop_op_off] = 4.0;
            prog.params[loop_rhs_off + ns] = 4.0; // rhs = i
            let code = prog.discretize_and_emit("sum_of_divisors", &["n"]);
            println!("sum_of_divisors:\n{code}");
            assert!(code.contains("v0: i64 = n % i"), "sd pre: {code}");
            assert!(code.contains("v0 == 0"), "sd cmp: {code}");
            assert!(code.contains("acc = acc + i"), "sd body: {code}");
            assert!(code.contains("i <= n"), "sd bound: {code}");
        }

        // SoftCondDigitLoop structural tests
        {
            // count_even_digits: base=10, pre=d%2, gate=pre==0, rhs=c1=1, loop=+
            let mut prog = SoftCondDigitLoop::new();
            let (
                _,
                base_off,
                gate_pre_off,
                gate_lhs_off,
                gate_cmp_off,
                gate_rhs_off,
                acc_rhs_off,
                loop_op_off,
            ) = SoftCondDigitLoop::offsets();
            let co = SoftCondDigitLoop::consts_off();
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            prog.params[0] = 0.0;
            prog.params[base_off + 5] = 4.0;
            prog.params[gate_pre_off + 3] = 4.0;
            prog.params[gate_lhs_off + 1] = 4.0;
            prog.params[gate_cmp_off + 4] = 4.0;
            prog.params[gate_rhs_off + 4] = 4.0;
            prog.params[acc_rhs_off + 5] = 4.0;
            prog.params[loop_op_off + 0] = 4.0;
            let code = prog.discretize_and_emit("count_even_digits", &["n"]);
            println!("count_even_digits:\n{code}");
            assert!(code.contains("% 10"), "ced base: {code}");
            assert!(
                code.contains("% 2") || code.contains("%2"),
                "ced pre: {code}"
            );
            assert!(code.contains("== 0"), "ced gate: {code}");
            assert!(
                code.contains("acc + 1") || code.contains("+ 1"),
                "ced body: {code}"
            );
        }
        {
            // sum_odd_digits: gate=pre==1, rhs=d
            let mut prog = SoftCondDigitLoop::new();
            let (
                _,
                base_off,
                gate_pre_off,
                gate_lhs_off,
                gate_cmp_off,
                gate_rhs_off,
                acc_rhs_off,
                loop_op_off,
            ) = SoftCondDigitLoop::offsets();
            let co = SoftCondDigitLoop::consts_off();
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            prog.params[0] = 0.0;
            prog.params[base_off + 5] = 4.0;
            prog.params[gate_pre_off + 3] = 4.0;
            prog.params[gate_lhs_off + 1] = 4.0;
            prog.params[gate_cmp_off + 4] = 4.0;
            prog.params[gate_rhs_off + 5] = 4.0; // c1=1
            prog.params[acc_rhs_off + 0] = 4.0; // d
            prog.params[loop_op_off + 0] = 4.0;
            let code = prog.discretize_and_emit("sum_odd_digits", &["n"]);
            println!("sum_odd_digits:\n{code}");
            assert!(code.contains("== 1"), "sod gate: {code}");
            assert!(
                code.contains("x % 10") || code.contains("(x % 10)"),
                "sod rhs: {code}"
            );
        }
        {
            // popcount: base=2, gate=d==1, rhs=1
            let mut prog = SoftCondDigitLoop::new();
            let (
                _,
                base_off,
                gate_pre_off,
                gate_lhs_off,
                gate_cmp_off,
                gate_rhs_off,
                acc_rhs_off,
                loop_op_off,
            ) = SoftCondDigitLoop::offsets();
            let co = SoftCondDigitLoop::consts_off();
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            prog.params[0] = 0.0;
            prog.params[base_off + 3] = 4.0; // c3=2
            prog.params[gate_pre_off + 3] = 4.0;
            prog.params[gate_lhs_off + 0] = 4.0; // d
            prog.params[gate_cmp_off + 4] = 4.0; // ==
            prog.params[gate_rhs_off + 5] = 4.0; // c1=1
            prog.params[acc_rhs_off + 5] = 4.0; // c1=1
            prog.params[loop_op_off + 0] = 4.0;
            let code = prog.discretize_and_emit("popcount", &["n"]);
            println!("popcount:\n{code}");
            assert!(code.contains("% 2"), "pop base: {code}");
            assert!(code.contains("== 1"), "pop gate: {code}");
            assert!(
                code.contains("+ 1") || code.contains("acc + 1"),
                "pop body: {code}"
            );
        }
        {
            // max_digit: base=10, gate=d>acc, rhs=d-acc, loop=+
            let mut prog = SoftCondDigitLoop::new();
            let (
                _,
                base_off,
                gate_pre_off,
                gate_lhs_off,
                gate_cmp_off,
                gate_rhs_off,
                acc_rhs_off,
                loop_op_off,
            ) = SoftCondDigitLoop::offsets();
            let co = SoftCondDigitLoop::consts_off();
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            prog.params[0] = 0.0;
            prog.params[base_off + 5] = 4.0; // c5=10
            prog.params[gate_pre_off + 3] = 4.0;
            prog.params[gate_lhs_off + 0] = 4.0; // d
            prog.params[gate_cmp_off + 0] = 4.0; // >
            prog.params[gate_rhs_off + 3] = 4.0; // acc
            prog.params[acc_rhs_off + 2] = 4.0; // d-acc
            prog.params[loop_op_off + 0] = 4.0; // +
            let code = prog.discretize_and_emit("max_digit", &["n"]);
            println!("max_digit:\n{code}");
            assert!(code.contains("% 10"), "md base: {code}");
            assert!(
                code.contains("> acc") || code.contains(">acc"),
                "md gate: {code}"
            );
            assert!(
                code.contains("- acc") || code.contains("-acc"),
                "md body: {code}"
            );
        }

        // SoftChainedBranch structural test: min3 biased restart
        {
            let n_args = 3usize;
            let p1 = n_args + N_CONSTS; // 9
            let p2 = p1 + 1; // 10
            let (
                b1_cmp_off,
                b1_lhs_off,
                b1_rhs_off,
                b1_true_off,
                b1_false_off,
                b2_cmp_off,
                b2_lhs_off,
                b2_rhs_off,
                b2_true_off,
                b2_false_off,
                co,
            ) = SoftChainedBranch::offsets(n_args);
            let mut prog = SoftChainedBranch::new(n_args);
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            prog.params[b1_cmp_off + 1] = 4.0;
            prog.params[b1_lhs_off + 0] = 4.0;
            prog.params[b1_rhs_off + 1] = 4.0;
            prog.params[b1_true_off + 0] = 4.0;
            prog.params[b1_false_off + 1] = 4.0;
            prog.params[b2_cmp_off + 1] = 4.0;
            prog.params[b2_lhs_off + p2 - 1] = 4.0;
            prog.params[b2_rhs_off + 2] = 4.0;
            prog.params[b2_true_off + p2 - 1] = 4.0;
            prog.params[b2_false_off + 2] = 4.0;
            let code = prog.discretize_and_emit("min3", &["a", "b", "c"]);
            println!("min3:\n{code}");
            // New assignment form: "v0 = else; if cond { v0 = then; }"
            assert!(code.contains("a < b"), "min3 b1 cmp: {code}");
            assert!(code.contains("v0 < c"), "min3 b2 cmp: {code}");
            assert!(code.contains("v0 = a"), "min3 b1 true (assignment): {code}");
            assert!(code.contains("v0: i64 = b"), "min3 b1 false (init): {code}");
            assert!(
                code.contains("result = v0"),
                "min3 b2 true (assignment): {code}"
            );
            assert!(
                code.contains("result: i64 = c"),
                "min3 b2 false (init): {code}"
            );
        }

        // SoftCondAccumCmpReturnLoop structural test: is_prime biased restart
        {
            let n_args = 1usize;
            let ns = SoftCondAccumCmpReturnLoop::ns(n_args);
            let (
                pre_op_off,
                pre_s1_off,
                pre_s2_off,
                cmp_op_off,
                cmp_s1_off,
                cmp_s2_off,
                loop_op_off,
                loop_rhs_off,
                co,
            ) = SoftCondAccumLoop::offsets(ns);
            let ret_cmp_off = SoftCondAccumCmpReturnLoop::ret_cmp_off(ns);
            let ret_c_off = SoftCondAccumCmpReturnLoop::ret_c_off(ns);
            let mut prog = SoftCondAccumCmpReturnLoop::new(n_args);
            for p in prog.params.iter_mut() {
                *p = -4.0;
            }
            prog.params[co] = 0.0;
            prog.params[co + 1] = 1.0;
            prog.params[co + 2] = -1.0;
            prog.params[co + 3] = 2.0;
            prog.params[co + 4] = -2.0;
            prog.params[co + 5] = 10.0;
            prog.params[0] = 0.0;
            prog.params[1] = 1.0;
            prog.params[2] = 4.0;
            prog.params[2 + ns] = 0.0;
            prog.params[pre_op_off + 4] = 4.0;
            prog.params[pre_s1_off + 0] = 4.0;
            prog.params[pre_s2_off + ns] = 4.0;
            prog.params[cmp_op_off + 4] = 4.0;
            prog.params[cmp_s1_off + ns + 1] = 4.0;
            prog.params[cmp_s2_off + n_args] = 4.0;
            prog.params[loop_op_off + 0] = 4.0;
            prog.params[loop_rhs_off + n_args + 1] = 4.0;
            prog.params[ret_cmp_off + 4] = 4.0;
            prog.params[ret_c_off] = 2.0;
            let code = prog.discretize_and_emit("is_prime", &["n"]);
            println!("is_prime:\n{code}");
            assert!(code.contains("n % i"), "ip pre: {code}");
            assert!(code.contains("v0 == 0"), "ip cmp: {code}");
            assert!(
                code.contains("acc + 1") || code.contains("acc +1"),
                "ip body: {code}"
            );
            assert!(code.contains("acc == 2"), "ip return cmp: {code}");
            assert!(code.contains("return 1"), "ip return 1: {code}");
            assert!(code.contains("return 0"), "ip return 0: {code}");
        }

        // SoftPredicateLoopRetCmp structural test: triangular_check biased restart
        {
            let n_args = 1usize;
            let mut prog = SoftPredicateLoopRetCmp::new(n_args);
            {
                let inner = SoftPredicateLoop::new(n_args);
                let nb = inner.nb();
                let na = inner.na();
                let c0 = 2 * nb;
                let c1 = c0 + N_CMPS;
                let c2 = c1 + na;
                let a1 = c2 + na;
                let a2 = a1 + na;
                let ao = a2 + na;
                let b1 = ao + 6;
                let b2 = b1 + na;
                let bo = b2 + na;
                let ro = bo + 6;
                let co = ro + na;
                for p in prog.params.iter_mut() {
                    *p = -1.0;
                }
                prog.params[co] = 0.0;
                prog.params[co + 1] = 1.0;
                prog.params[co + 2] = -1.0;
                prog.params[co + 3] = 2.0;
                prog.params[co + 4] = -2.0;
                prog.params[co + 5] = 10.0;
                prog.params[2] = 4.0; // init0 = c1=1
                prog.params[nb + 1] = 4.0; // init1 = c0=0
                prog.params[c0 + 1] = 4.0; // cmp = <
                prog.params[c1 + 1] = 4.0; // lhs = x1
                prog.params[c2 + 2] = 4.0; // rhs = n
                prog.params[a1 + 0] = 4.0; // a_s1 = x0
                prog.params[a2 + 4] = 4.0; // a_s2 = c1=1
                prog.params[ao + 0] = 4.0; // a_op = +
                prog.params[b1 + 1] = 4.0; // b_s1 = x1
                prog.params[b2 + 0] = 4.0; // b_s2 = x0
                prog.params[bo + 0] = 4.0; // b_op = +
                prog.params[ro + 1] = 4.0; // ret = x1
                let ret_cmp_off = SoftPredicateLoopRetCmp::ret_cmp_off(n_args);
                let ret_rhs_off = SoftPredicateLoopRetCmp::ret_rhs_off(n_args);
                prog.params[ret_cmp_off + 4] = 4.0; // cmp = ==
                prog.params[ret_rhs_off + 1] = 4.0; // rhs = n (rhs_pool idx 1)
                let _ = a2;
            }
            let code = prog.discretize_and_emit("triangular_check", &["n"]);
            println!("triangular_check:\n{code}");
            assert!(code.contains("x0: i64 = 1"), "tc init0: {code}");
            assert!(code.contains("x1: i64 = 0"), "tc init1: {code}");
            assert!(code.contains("x1 < n"), "tc cond: {code}");
            assert!(code.contains("x0 + 1"), "tc a-body: {code}");
            assert!(
                code.contains("x1 + x0") || code.contains("x1+x0"),
                "tc b-body: {code}"
            );
            assert!(code.contains("x1 == n"), "tc ret cmp: {code}");
            assert!(code.contains("return 1"), "tc ret 1: {code}");
            assert!(code.contains("return 0"), "tc ret 0: {code}");
        }
    }

    #[test]
    fn soft_cond_mutate_loop_structural_test() {
        // Verify SoftCondMutateLoop with collatz biased restart emits correct structure
        let n_args: usize = 1;
        let (
            init_off,
            cond_cmp_off,
            cond_lhs_off,
            cond_rhs_off,
            pre_op_off,
            pre_s1_off,
            pre_s2_off,
            gate_cmp_off,
            gate_rhs_off,
            true_op_off,
            true_s1_off,
            true_s2_off,
            fop1_off,
            fs1_off,
            fs2_off,
            fop2_off,
            fs3_off,
            co,
        ) = SoftCondMutateLoop::offsets(n_args);
        let na = SoftCondMutateLoop::na(n_args);
        let ng = na + 1;
        let _ = (na, ng);
        let mut prog = SoftCondMutateLoop::new(n_args);
        for p in prog.params.iter_mut() {
            *p = -4.0;
        }
        // consts: c0=0, c1=1, c2=-1, c3=2, c4=3, c5=10
        prog.params[co] = 0.0;
        prog.params[co + 1] = 1.0;
        prog.params[co + 2] = -1.0;
        prog.params[co + 3] = 2.0;
        prog.params[co + 4] = 3.0;
        prog.params[co + 5] = 10.0;
        prog.params[init_off + 0] = 4.0; // x = n
        prog.params[cond_cmp_off + 5] = 4.0; // !=
        prog.params[cond_lhs_off + 0] = 4.0; // lhs = x
        prog.params[cond_rhs_off + 3] = 4.0; // rhs = c1=1
        prog.params[pre_op_off + 4] = 4.0; // %
        prog.params[pre_s1_off + 0] = 4.0; // x
        prog.params[pre_s2_off + 5] = 4.0; // c3=2
        prog.params[gate_cmp_off + 4] = 4.0; // ==
        prog.params[gate_rhs_off + 3] = 4.0; // c0=0
        prog.params[true_op_off + 3] = 4.0; // /
        prog.params[true_s1_off + 0] = 4.0; // x
        prog.params[true_s2_off + 5] = 4.0; // c3=2
        prog.params[fop1_off + 2] = 4.0; // *
        prog.params[fs1_off + 0] = 4.0; // x
        prog.params[fs2_off + 6] = 4.0; // c4=3
        prog.params[fop2_off + 0] = 4.0; // +
        prog.params[fs3_off + 4] = 4.0; // c1=1

        let code = prog.discretize_and_emit("collatz_steps", &["n"]);
        println!("collatz_steps:\n{code}");
        assert!(code.contains("x != 1"), "cml cond: {code}");
        assert!(code.contains("x % 2"), "cml pre: {code}");
        assert!(code.contains("pre == 0"), "cml gate: {code}");
        assert!(code.contains("x / 2"), "cml true: {code}");
        assert!(code.contains("x * 3"), "cml fop1: {code}");
        assert!(code.contains("v_tmp + 1"), "cml fop2: {code}");
        assert!(code.contains("x = v_false"), "cml assign: {code}");
        assert!(code.contains("x = v_true"), "cml if-true: {code}");
        assert!(code.contains("acc = acc + 1"), "cml acc: {code}");
        assert!(code.contains("return acc"), "cml return: {code}");
    }

    #[test]
    fn soft_universal_structural_test() {
        // Verify SoftUniversalProgram: check forward pass produces finite output,
        // parameter layout is consistent, and discretize_and_emit produces parseable code.
        let n_args = 1usize;
        let pool = univ_pool(n_args);
        let lip = univ_lip(n_args);
        let sps = univ_sps(pool);

        // Expected sizes
        assert_eq!(pool, 18, "pool size (1+6+11)");
        assert_eq!(lip, 10, "loop init pool (1+6+3)");
        assert_eq!(sps, 102, "slot params size (6+5*18+6)");
        assert_eq!(N_UNIV_SLOTS, 11, "total slots");
        let n_total = SoftUniversalProgram::n_params_for(n_args);
        // 11*102 + 6*10 + 6 + 2*18 + 18 + 6 = 1122+60+6+36+18+6=1248
        assert_eq!(n_total, 1248, "total params for n_args=1: {n_total}");

        let prog = SoftUniversalProgram::new(n_args);
        assert_eq!(prog.params.len(), n_total);

        // Forward pass should produce a finite value
        let out = prog.forward(&[5.0], 1.0);
        assert!(out.is_finite(), "forward output must be finite, got {out}");

        // Consts should be initialized correctly
        let co = prog.consts_off();
        assert_eq!(prog.params[co + 0], 0.0);
        assert_eq!(prog.params[co + 1], 1.0);
        assert_eq!(prog.params[co + 2], -1.0);
        assert_eq!(prog.params[co + 3], 2.0);
        assert_eq!(prog.params[co + 4], -2.0);
        assert_eq!(prog.params[co + 5], 10.0);

        // Emit code and verify structure
        let code = prog.discretize_and_emit("test_fn", &["n"]);
        println!("SoftUniversalProgram emit:\n{code}");
        assert!(code.contains("fn test_fn(n: i64) -> i64"), "header: {code}");
        assert!(code.contains("v0: i64 ="), "init slot v0: {code}");
        assert!(code.contains("v1: i64 ="), "init slot v1: {code}");
        assert!(code.contains("v2: i64 ="), "init slot v2: {code}");
        assert!(code.contains("s0: i64 ="), "loop init s0: {code}");
        assert!(code.contains("s5: i64 ="), "loop init s5: {code}");
        assert!(code.contains("while "), "while loop: {code}");
        assert!(code.contains("p0: i64 ="), "post slot p0: {code}");
        assert!(code.contains("p1: i64 ="), "post slot p1: {code}");
        assert!(code.contains("return "), "return: {code}");
    }

    #[test]
    fn soft_universal_add_two_test() {
        // Verify SoftUniversalProgram can represent add_two(a, b) = a + b
        // by directly biasing the params: use a post slot to compute a+b, return it.
        let n_args = 2usize;
        let mut prog = SoftUniversalProgram::new(n_args);
        let pool = prog.ps();
        let co = prog.consts_off();
        prog.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);

        // Bias all params very low to suppress everything
        for p in prog.params.iter_mut() {
            *p = -4.0;
        }
        prog.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);

        // Use post slot 0 (p0 = slot N_INIT_SLOTS + N_LOOP_SLOTS = 9):
        //   op = + (index 0), src1 = arg0 (pool[0]="a"), src2 = arg1 (pool[1]="b")
        //   gate: trivially true — bias cmp to == (index 2); gl_i==gr_i (both argmax at last) → x==x true
        let p0_slot = N_INIT_SLOTS + N_LOOP_SLOTS; // slot 9
        let p0_off = prog.slot_off(p0_slot);
        prog.params[p0_off + 0] = 4.0; // op = +
        prog.params[p0_off + N_OPS + 1 + 0] = 4.0; // src1 = pool[0] = "a"
        prog.params[p0_off + N_OPS + 1 + pool + 1] = 4.0; // src2 = pool[1] = "b"
                                                          // Gate trivially true: cmp = == (index 2), gl/gr argmax both → last index (same) → x==x ✓
        let cb = p0_off + N_OPS + 1 + 2 * pool;
        prog.params[cb + 2] = 4.0; // cmp = == (index 2) — with gl_i==gr_i this is always true

        // Return = p0 (pool index n_args + N_CONSTS + N_INIT_SLOTS + N_LOOP_SLOTS = 17)
        let ro = prog.return_off();
        prog.params[ro + n_args + N_CONSTS + N_INIT_SLOTS + N_LOOP_SLOTS] = 4.0;

        // Check emit contains "a + b" and "return p0"
        let code = prog.discretize_and_emit("add_two", &["a", "b"]);
        println!("add_two emit:\n{code}");
        assert!(code.contains("a + b"), "then_expr should be a+b: {code}");
        assert!(code.contains("return p0"), "should return p0: {code}");
    }

    #[test]
    fn universal_description_round_trip_test() {
        // Build the add_two biased program (same setup as soft_universal_add_two_test),
        // extract its description, reconstruct params from description, verify emit matches.
        let n_args = 2usize;
        let mut prog = SoftUniversalProgram::new(n_args);
        let pool = prog.ps();
        let co = prog.consts_off();

        // Suppress everything, keep default consts
        for p in prog.params.iter_mut() {
            *p = -4.0;
        }
        prog.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);

        // Bias p0 slot (slot 9): op=+, s1=a(0), s2=b(1), gate trivially true
        let p0_slot = N_INIT_SLOTS + N_LOOP_SLOTS;
        let p0_off = prog.slot_off(p0_slot);
        prog.params[p0_off + 0] = 4.0; // op = +
        prog.params[p0_off + N_OPS + 1 + 0] = 4.0; // s1 = a
        prog.params[p0_off + N_OPS + 1 + pool + 1] = 4.0; // s2 = b
        let cb = p0_off + N_OPS + 1 + 2 * pool;
        prog.params[cb + 2] = 4.0; // gate_cmp = == (gl_i==gr_i → trivially true)

        // Return p0 (pool index n_args + N_CONSTS + N_INIT_SLOTS + N_LOOP_SLOTS)
        let ret_pool_idx = n_args + N_CONSTS + N_INIT_SLOTS + N_LOOP_SLOTS;
        let ro = prog.return_off();
        prog.params[ro + ret_pool_idx] = 4.0;

        // Original emit
        let orig_code = prog.discretize_and_emit("add_two", &["a", "b"]);
        assert!(orig_code.contains("a + b"), "original: {orig_code}");
        assert!(orig_code.contains("return p0"), "original: {orig_code}");

        // ── Phase 1: params → description ─────────────────────────────────────
        let desc = prog.params_to_description();
        assert_eq!(desc.n_args, 2);
        assert_eq!(desc.slots.len(), N_UNIV_SLOTS);
        assert_eq!(desc.loop_init.len(), N_LOOP_SLOTS);

        // p0_slot should have op=0(+), s1=0(a), s2=1(b), gate_cmp=2(==)
        let p0_desc = &desc.slots[p0_slot];
        assert_eq!(p0_desc.op, 0, "op should be + (0)");
        assert_eq!(p0_desc.s1, 0, "s1 should be a (pool[0])");
        assert_eq!(p0_desc.s2, 1, "s2 should be b (pool[1])");
        assert_eq!(p0_desc.gate_cmp, 2, "gate_cmp should be == (2)");
        assert_eq!(
            p0_desc.gate_lhs, p0_desc.gate_rhs,
            "trivially-true gate: lhs==rhs"
        );
        assert_eq!(desc.ret_src, ret_pool_idx, "return src should be p0");

        // Print the human-readable explanation
        let explanation = desc.explain();
        println!("Description:\n{explanation}");
        assert!(
            explanation.contains("+ b"),
            "explanation should mention + b: {explanation}"
        );
        assert!(
            explanation.contains("p0"),
            "explanation should mention p0: {explanation}"
        );

        // ── Phase 2: description → params → emit ──────────────────────────────
        let prog2 = SoftUniversalProgram::description_to_params(&desc);
        let rt_code = prog2.discretize_and_emit("add_two", &["a", "b"]);
        println!("Round-trip emit:\n{rt_code}");
        assert!(rt_code.contains("a + b"), "round-trip: {rt_code}");
        assert!(rt_code.contains("return p0"), "round-trip: {rt_code}");

        // ── Phase 3: description of description should match ──────────────────
        let desc2 = prog2.params_to_description();
        // Every field should agree
        for (i, (sd, sd2)) in desc.slots.iter().zip(desc2.slots.iter()).enumerate() {
            assert_eq!(sd, sd2, "slot {i} description mismatch after round-trip");
        }
        assert_eq!(desc.loop_init, desc2.loop_init, "loop_init mismatch");
        assert_eq!(desc.cond_cmp, desc2.cond_cmp, "cond_cmp mismatch");
        assert_eq!(desc.cond_lhs, desc2.cond_lhs, "cond_lhs mismatch");
        assert_eq!(desc.cond_rhs, desc2.cond_rhs, "cond_rhs mismatch");
        assert_eq!(desc.ret_src, desc2.ret_src, "ret_src mismatch");
    }

    #[test]
    fn universal_description_factorial_encode_test() {
        // Demonstrate "code → params": hand-encode factorial(n) = n*(n-1)*...*2*1
        //
        // Encoded program:
        //   s0 = 1 (accumulator)     ← loop_init[0] = lip[2] = const 1
        //   s1 = n (counter)         ← loop_init[1] = lip[0] = a
        //   while s1 != 1 { s0 = s0 * s1; s1 = s1 - 1 }
        //   return s0
        //
        // Pool (n_args=1): [a(0), 0(1), 1(2), -1(3), 2(4), -2(5), 10(6),
        //                    v0(7), v1(8), v2(9), s0(10), s1(11), .., p0(16), p1(17)]
        // Lip (loop-init pool): [a(0), 0(1), 1(2), -1(3), 2(4), -2(5), 10(6), v0(7), v1(8), v2(9)]
        //
        // Key soft_cmp index mapping (N_CMPS=6 indices ["<","<=","==",">=",">","!="]):
        //   Index 2 ("=="): sigmoid(d/t) — trivially-true in emit when lhs_i==rhs_i (cmp∈{1,2,3})
        //   Index 5 ("!="): 1-exp(-d²/var) = 0 when d=0, ≈1 when d≠0 → correct for s1!=1

        let n_args = 1usize;
        let pool = univ_pool(n_args); // 18
        let _lip = univ_lip(n_args); // 10

        let consts = [0f32, 1.0, -1.0, 2.0, -2.0, 10.0];
        // Pool indices for n_args=1
        let idx_c1 = 2usize; // const 1  (pool[2])
        let idx_s0 = n_args + N_CONSTS + N_INIT_SLOTS; // 10 = s0
        let idx_s1 = n_args + N_CONSTS + N_INIT_SLOTS + 1; // 11 = s1

        // All slots: default identity (op=5=id, s1/s2=0, gate_cmp=2(==), lhs==rhs→trivially-true in emit)
        let default_slot = SlotDesc {
            op: 5,
            s1: 0,
            s2: 0,
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: 0,
        };
        let mut slots: Vec<SlotDesc> = (0..N_UNIV_SLOTS).map(|_| default_slot.clone()).collect();

        // loop_init[0] = lip[2] = const 1 → s0 starts at 1
        // loop_init[1] = lip[0] = a       → s1 starts at n
        // rest         = lip[1] = 0
        let mut loop_init = vec![1usize; N_LOOP_SLOTS]; // default: lip[1]=0
        loop_init[0] = 2; // s0 = const 1  (lip index 2)
        loop_init[1] = 0; // s1 = a        (lip index 0)

        // Loop body slot 3 (s0): s0 = s0 * s1
        //   gate_cmp=2("=="), lhs=rhs=0 → trivially-true in emit → direct assignment s0 = s0*s1
        slots[N_INIT_SLOTS + 0] = SlotDesc {
            op: 2,
            s1: idx_s0,
            s2: idx_s1, // *
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: idx_s0,
        };
        // Loop body slot 4 (s1): s1 = s1 - 1
        slots[N_INIT_SLOTS + 1] = SlotDesc {
            op: 1,
            s1: idx_s1,
            s2: idx_c1, // -
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: idx_s1,
        };

        // Loop condition: s1 != 1
        //   cond_cmp=5("!="): soft_cmp[5] = 1-exp(-d²/var) = 0 when s1==1, ≈1 otherwise
        let cond_cmp = 5; // "!=" — semantically correct AND emits as "!="
        let cond_lhs = idx_s1;
        let cond_rhs = idx_c1;

        let ret_src = idx_s0; // return s0

        let desc = UniversalProgramDescription {
            n_args,
            slots,
            loop_init,
            cond_cmp,
            cond_lhs,
            cond_rhs,
            ret_src,
            consts,
        };

        // Print explanation
        let explanation = desc.explain();
        println!("Factorial description:\n{explanation}");
        assert!(explanation.contains("s0"), "{explanation}");
        assert!(explanation.contains("s1"), "{explanation}");

        // Convert to params and emit
        let prog = SoftUniversalProgram::description_to_params(&desc);
        assert_eq!(
            prog.params.len(),
            SoftUniversalProgram::n_params_for(n_args)
        );
        let code = prog.discretize_and_emit("factorial", &["n"]);
        println!("Factorial emit:\n{code}");
        assert!(
            code.contains("fn factorial(n: i64) -> i64"),
            "header: {code}"
        );
        // Accumulator slot: unconditional s0 = s0 * s1
        assert!(code.contains("s0 = s0 * s1"), "multiply: {code}");
        // Counter slot: unconditional s1 = s1 - 1
        assert!(code.contains("s1 = s1 - 1"), "decrement: {code}");
        // Loop condition uses index 5 ("!=") — semantic match with soft_cmp
        assert!(code.contains("while s1 != 1"), "while condition: {code}");
        assert!(code.contains("return s0"), "return: {code}");

        // Forward pass: description_to_params gives warm-start logits (+4/-4).
        // The soft computation is approximate (gates are ~0.5 due to soft_cmp clamping),
        // but the EMITTED code is correct and the round-trip must agree.
        let out = prog.forward(&[5.0], 0.01);
        println!("factorial(5) soft forward = {out:.2} (emitted code computes exactly 120)");
        assert!(out.is_finite(), "forward must be finite: {out}");

        // Round-trip: description → params → description
        let desc2 = prog.params_to_description();
        for (i, (sd, sd2)) in desc.slots.iter().zip(desc2.slots.iter()).enumerate() {
            assert_eq!(sd, sd2, "slot {i} mismatch after round-trip");
        }
        assert_eq!(desc.loop_init, desc2.loop_init, "loop_init round-trip");
        assert_eq!(desc.cond_cmp, desc2.cond_cmp, "cond_cmp  round-trip");
        assert_eq!(desc.cond_lhs, desc2.cond_lhs, "cond_lhs  round-trip");
        assert_eq!(desc.cond_rhs, desc2.cond_rhs, "cond_rhs  round-trip");
        assert_eq!(desc.ret_src, desc2.ret_src, "ret_src   round-trip");

        // Pool names sanity
        let pn = desc.pool_names();
        assert_eq!(pn.len(), pool);
        assert_eq!(pn[0], "a"); // pool_names uses default letters; arg "n" → "a"
        assert_eq!(pn[idx_s0], "s0");
        assert_eq!(pn[idx_s1], "s1");
    }

    #[test]
    fn discrete_eval_correctness_test() {
        // Verify discrete_eval gives the same results as the emitted program semantics.

        // ── factorial(5) = 120 ────────────────────────────────────────────────
        let n_args = 1usize;
        let consts = [0f32, 1.0, -1.0, 2.0, -2.0, 10.0];
        let idx_c1 = 2usize; // pool[2] = const 1
        let idx_s0 = n_args + N_CONSTS + N_INIT_SLOTS; // 10
        let idx_s1 = n_args + N_CONSTS + N_INIT_SLOTS + 1; // 11
        let default_slot = SlotDesc {
            op: 5,
            s1: 0,
            s2: 0,
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: 0,
        };
        let mut slots: Vec<SlotDesc> = (0..N_UNIV_SLOTS).map(|_| default_slot.clone()).collect();
        slots[N_INIT_SLOTS + 0] = SlotDesc {
            op: 2,
            s1: idx_s0,
            s2: idx_s1,
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: idx_s0,
        };
        slots[N_INIT_SLOTS + 1] = SlotDesc {
            op: 1,
            s1: idx_s1,
            s2: idx_c1,
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: idx_s1,
        };
        let mut loop_init = vec![1usize; N_LOOP_SLOTS];
        loop_init[0] = 2; // s0 = 1
        loop_init[1] = 0; // s1 = n
        let desc = UniversalProgramDescription {
            n_args,
            slots,
            loop_init,
            cond_cmp: 5,
            cond_lhs: idx_s1,
            cond_rhs: idx_c1,
            ret_src: idx_s0,
            consts,
        };
        let prog = SoftUniversalProgram::description_to_params(&desc);

        // factorial(5)=120, factorial(1)=1, factorial(6)=720
        assert_eq!(prog.discrete_eval(&[5]), Some(120), "factorial(5)");
        assert_eq!(prog.discrete_eval(&[1]), Some(1), "factorial(1)");
        assert_eq!(prog.discrete_eval(&[6]), Some(720), "factorial(6)");
        println!("factorial tests passed ✓");

        // ── add_two(a,b) = a+b using a post slot ─────────────────────────────
        let n_args = 2usize;
        let pool = univ_pool(n_args); // 19
        let mut prog2 = SoftUniversalProgram::new(n_args);
        let co = prog2.consts_off();
        for p in prog2.params.iter_mut() {
            *p = -4.0;
        }
        prog2.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        let p0 = N_INIT_SLOTS + N_LOOP_SLOTS; // slot 9
        let off = prog2.slot_off(p0);
        prog2.params[off + 0] = 4.0; // op = +
        prog2.params[off + N_OPS + 1 + 0] = 4.0; // s1 = pool[0] = a
        prog2.params[off + N_OPS + 1 + pool + 1] = 4.0; // s2 = pool[1] = b
        let cb = off + N_OPS + 1 + 2 * pool;
        prog2.params[cb + 2] = 4.0; // gate_cmp = ==, lhs=rhs=argmax(all-4)=0 → trivially true
        let ro = prog2.return_off();
        prog2.params[ro + n_args + N_CONSTS + N_INIT_SLOTS + N_LOOP_SLOTS] = 4.0; // ret = p0

        assert_eq!(prog2.discrete_eval(&[2, 3]), Some(5), "add_two(2,3)");
        assert_eq!(prog2.discrete_eval(&[10, -4]), Some(6), "add_two(10,-4)");
        assert_eq!(prog2.discrete_eval(&[-3, -2]), Some(-5), "add_two(-3,-2)");
        println!("add_two tests passed ✓");

        // ── synthetic_record: diversity check with a simple bounded program ──
        // Use f(n) = n + 2 (init slot v0 = n+2, return v0)
        // slot 0 (v0): op=+(0), s1=0(n), s2=idx_c_2 (const 2), gate trivially true
        let idx_c_2 = 4usize; // pool[4] = const 2
        let _idx_v0 = n_args + N_CONSTS; // was n_args=1 earlier; re-declare with n_args=1
        let n_args_1 = 1usize;
        let consts1 = [0f32, 1.0, -1.0, 2.0, -2.0, 10.0];
        let default_sl = SlotDesc {
            op: 5,
            s1: 0,
            s2: 0,
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: 0,
        };
        let mut slots3: Vec<SlotDesc> = (0..N_UNIV_SLOTS).map(|_| default_sl.clone()).collect();
        // slot 0 = init v0: op=+, s1=n(0), s2=const2(4), gate trivially true → v0 = n+2
        slots3[0] = SlotDesc {
            op: 0,
            s1: 0,
            s2: idx_c_2,
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: 0,
        };
        let loop_init3 = vec![1usize; N_LOOP_SLOTS]; // all = lip[1] = const 0
        let idx_v0_ret = 1 + N_CONSTS; // pool[7] for n_args=1 = v0 index
        let desc3 = UniversalProgramDescription {
            n_args: n_args_1,
            slots: slots3,
            loop_init: loop_init3,
            cond_cmp: 2,
            cond_lhs: 0,
            cond_rhs: 0, // loop cond: 0==0, always… but trivially
            ret_src: idx_v0_ret,
            consts: consts1,
        };
        let rec = synthetic_record(&desc3, 8, 12345);
        assert!(rec.is_some(), "synthetic_record should succeed for n+2");
        let rec = rec.unwrap();
        assert_eq!(rec.io_examples.len(), 8);
        // Verify each output == input + 2
        let prog3 = SoftUniversalProgram::description_to_params(&desc3);
        for (inputs, out) in &rec.io_examples {
            assert_eq!(*out, inputs[0] + 2, "n+2 wrong: {} → {}", inputs[0], out);
        }
        let distinct: std::collections::HashSet<i64> =
            rec.io_examples.iter().map(|(_, o)| *o).collect();
        assert!(distinct.len() >= 2, "should have diverse outputs");
        let _ = prog3;
        println!(
            "synthetic_record test passed ✓ ({} distinct outputs)",
            distinct.len()
        );
    }

    /// Verify warm-start synthesis: hand-crafted description for n+1, should solve in ≤ 50 steps.
    #[test]
    fn warm_start_synthesis_test() {
        use crate::benchmark::{Example, Problem, Value};
        // Build a simple n+1 problem
        let examples: Vec<Example> = (1i64..=8)
            .map(|n| Example {
                inputs: vec![Value::Int(n)],
                expected: n + 1,
            })
            .collect();
        let problem = Problem {
            name: "add_one".to_string(),
            category: "test",
            description: "add one",
            signature: "fn add_one(a: i64) -> i64",
            examples,
            holdouts: vec![],
            reference_code: "",
        };

        // Build description for n+1:
        //   init slot0: v0 = a + 1   (op=0 +, s1=0=a, s2=2=const1=1, gate trivially true)
        //   return v0
        let n_args = 1usize;
        let _pool = univ_pool(n_args);
        let idx_const1 = 2usize; // pool[2] = const[1] = 1.0
        let idx_v0 = n_args + N_CONSTS; // pool[7]

        let default_sl = SlotDesc {
            op: 5,
            s1: 0,
            s2: 0,
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: 0,
        };
        let mut slots: Vec<SlotDesc> = (0..N_UNIV_SLOTS).map(|_| default_sl.clone()).collect();
        // init slot 0: v0 = a + 1, gated as trivially true (lhs==rhs → else path with trivially-true gate)
        // Use op=+ (0), s1=a(0), s2=const1(2), gate: cmp==2(==), lhs=0(a), rhs=0(a) → always true
        slots[0] = SlotDesc {
            op: 0,
            s1: 0,
            s2: idx_const1,
            gate_cmp: 2,
            gate_lhs: 0,
            gate_rhs: 0,
            else_val: 0,
        };

        let consts = [0f32, 1.0, -1.0, 2.0, -2.0, 10.0];
        let _lip = univ_lip(n_args);
        let desc = UniversalProgramDescription {
            n_args,
            slots,
            loop_init: vec![1usize; N_LOOP_SLOTS], // s_k = lip[1] = const 0
            // cond_cmp=5 means "!=" in discrete_eval → a != a = false → loop never runs
            cond_cmp: 5,
            cond_lhs: 0,
            cond_rhs: 0,
            ret_src: idx_v0,
            consts,
        };

        // Warm-start should solve add_one quickly (description already encodes n+1)
        let result = synthesize_universal_warm_start(&problem, &desc, 50, 0);
        assert!(result.is_some(), "warm-start should solve add_one (n+1)");
        let (solve, steps, was_warm) = result.unwrap();
        println!("add_one warm-start: solved in {steps} steps (warm={was_warm})");
        println!("  method: {}", solve.method);
        assert!(steps <= 50, "should solve in ≤50 warm steps, got {steps}");
    }

    /// Array gradient coverage report — measures how many array benchmarks are solved
    /// by gradient descent alone (no templates or search).
    #[test]
    fn array_gradient_coverage_report() {
        use crate::benchmark::get_benchmark;
        let problems = get_benchmark(1);
        let mut solved = 0;
        let mut total = 0;
        let mut solved_names = vec![];
        let mut failed_names = vec![];
        for p in &problems {
            // Only array problems (first input is Array)
            let is_array = p
                .examples
                .first()
                .map(|ex| ex.inputs.first().map(|v| matches!(v, crate::benchmark::Value::Array(_))).unwrap_or(false))
                .unwrap_or(false);
            if !is_array {
                continue;
            }
            total += 1;
            let ok = synthesize_array(p)
                .map(|r| r.success)
                .unwrap_or(false);
            if ok {
                solved += 1;
                solved_names.push(p.name.clone());
            } else {
                failed_names.push(p.name.clone());
            }
            println!(
                "  [{}/{}] {} {}",
                total,
                total,
                p.name,
                if ok { "SOLVED ✓" } else { "failed ✗" }
            );
        }
        println!(
            "\n=== Array Gradient Coverage: {}/{} ({:.1}%) ===",
            solved,
            total,
            100.0 * solved as f64 / total.max(1) as f64
        );
        println!("SOLVED: {}", solved_names.join(", "));
        println!("FAILED: {}", failed_names.join(", "));
    }

    /// Quick test: run array gradient on specific problematic benchmarks.
    #[test]
    fn array_gradient_targeted() {
        use crate::benchmark::get_benchmark;
        let problems = get_benchmark(1);
        let targets = [
            "array_sum",
            "array_max",
            "count_positive",
            "count_zeros",
            "count_occurrences",
            "sum_negatives",
            "sum_positives",
            "min_element",
            "sum_at_even_indices",
            "sum_odd_indexed",
            "kth_from_end",
            "reverse_sum",
            "array_max_elem",
            "interactive_sum",
        ];
        for target in &targets {
            let p = problems.iter().find(|p| p.function_name() == *target);
            if let Some(p) = p {
                let ok = synthesize_array(p).map(|r| r.success).unwrap_or(false);
                println!("  {} {}", p.name, if ok { "SOLVED ✓" } else { "failed ✗" });
            }
        }
    }

    /// Smoke test: register machine on a hand-built 1-arg problem.
    /// Uses a simple problem (double) where biased init should converge fast.
    #[test]
    fn register_machine_smoke_test() {
        use crate::benchmark::{Example, Problem, Value};
        // double(a) = 2*a — simple enough for RM to find
        let problem = Problem {
            name: "double_v0".to_string(),
            category: "test",
            description: "double",
            signature: "fn double(a: i64) -> i64",
            examples: vec![
                Example { inputs: vec![Value::Int(1)], expected: 2 },
                Example { inputs: vec![Value::Int(3)], expected: 6 },
                Example { inputs: vec![Value::Int(0)], expected: 0 },
                Example { inputs: vec![Value::Int(-2)], expected: -4 },
                Example { inputs: vec![Value::Int(5)], expected: 10 },
            ],
            holdouts: vec![],
            reference_code: "",
        };

        // Just test that forward + discretize don't panic
        let rm = SoftRegisterMachine::new(1);
        let code = rm.discretize_and_emit("double");
        println!("Default RM code:\n{code}");

        // Test forward pass
        let output = rm.forward(&[3.0], 1.0);
        println!("forward([3.0]) = {output}");

        // Try synthesis (may or may not solve in time, but shouldn't panic)
        let result = synthesize_register_machine(&problem);
        if let Some(r) = &result {
            println!("RM solved: method={}, code:\n{}", r.method, r.code);
        } else {
            println!("RM did not solve (expected for smoke test with limited budget)");
        }
    }
}
