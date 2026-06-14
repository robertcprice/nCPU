use super::*;

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
        a + b,                             // 0: +
        a - b,                             // 1: -
        a * b,                             // 2: *
        a / safe_b,                        // 3: /
        a - (a / safe_b).trunc() * safe_b, // 4: %
        if a < b {
            a
        } else {
            0.5 * (a + b - (a - b).abs())
        }, // 5: min (soft approx)
        if a > b {
            a
        } else {
            0.5 * (a + b + (a - b).abs())
        }, // 6: max (soft approx)
        -a,                                // 7: negate
        (a * a + 0.01).sqrt(),             // 8: smooth abs
        a,                                 // 9: identity (nop / pass-through)
    ];
    weights.iter().zip(&results).map(|(w, r)| w * r).sum()
}

pub(super) struct SoftRegisterMachine {
    pub(super) n_args: usize,
    pub(super) params: Vec<f32>,
}

impl SoftRegisterMachine {
    pub(super) fn n_regs(n_args: usize) -> usize {
        n_args + N_CONSTS + N_SCRATCH
    }

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

    pub(super) fn new(n_args: usize) -> Self {
        let mut s = Self {
            n_args,
            params: vec![0f32; Self::n_params(n_args)],
        };
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

    fn ret_off(&self) -> usize {
        N_RM_STEPS * Self::step_size(self.n_args)
    }
    fn consts_off(&self) -> usize {
        self.ret_off() + Self::n_regs(self.n_args)
    }

    pub(super) fn forward(&self, inputs: &[f32], temp: f32) -> f32 {
        let nr = Self::n_regs(self.n_args);
        let ss = Self::step_size(self.n_args);
        let co = self.consts_off();

        // Initialize register file
        let mut regs = vec![0f32; nr];
        for (i, &v) in inputs.iter().enumerate() {
            regs[i] = v;
        }
        // Load constants
        for i in 0..N_CONSTS {
            regs[self.n_args + i] = self.params[co + i];
        }
        // Scratch registers start at 0

        // Execute instruction sequence
        for step in 0..N_RM_STEPS {
            let off = step * ss;
            let op_w = softmax_temp(&self.params[off..off + N_RM_OPS], temp);
            let s1_w = softmax_temp(&self.params[off + N_RM_OPS..off + N_RM_OPS + nr], temp);
            let s2_w = softmax_temp(
                &self.params[off + N_RM_OPS + nr..off + N_RM_OPS + 2 * nr],
                temp,
            );
            let dst_w = softmax_temp(
                &self.params[off + N_RM_OPS + 2 * nr..off + N_RM_OPS + 3 * nr],
                temp,
            );
            let gate_cmp_w = softmax_temp(
                &self.params[off + N_RM_OPS + 3 * nr..off + N_RM_OPS + 3 * nr + N_CMPS],
                temp,
            );
            let gate_s1_w = softmax_temp(
                &self.params[off + N_RM_OPS + 3 * nr + N_CMPS..off + N_RM_OPS + 4 * nr + N_CMPS],
                temp,
            );
            let gate_s2_w = softmax_temp(
                &self.params[off + N_RM_OPS + 4 * nr + N_CMPS..off + N_RM_OPS + 5 * nr + N_CMPS],
                temp,
            );

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
        examples
            .iter()
            .map(|(inputs, expected)| {
                let diff = self.forward(inputs, temp) - expected;
                diff * diff
            })
            .sum::<f32>()
            / n
    }

    fn reg_names(&self) -> Vec<String> {
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|i| self.params[co + i].round() as i64)
            .collect();
        let mut names = Vec::new();
        // arg names
        let arg_names = ["a", "b", "c", "d", "e", "f"];
        for i in 0..self.n_args {
            names.push(arg_names.get(i).unwrap_or(&"x").to_string());
        }
        // const names (just their values)
        for c in &consts {
            names.push(format!("{c}"));
        }
        // scratch names
        for i in 0..N_SCRATCH {
            names.push(format!("r{i}"));
        }
        names
    }

    pub(super) fn discretize_and_emit(&self, fn_name: &str) -> String {
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
            let gate_cmp_i =
                argmax(&self.params[off + N_RM_OPS + 3 * nr..off + N_RM_OPS + 3 * nr + N_CMPS]);
            let gate_s1_i = argmax(
                &self.params[off + N_RM_OPS + 3 * nr + N_CMPS..off + N_RM_OPS + 4 * nr + N_CMPS],
            );
            let gate_s2_i = argmax(
                &self.params[off + N_RM_OPS + 4 * nr + N_CMPS..off + N_RM_OPS + 5 * nr + N_CMPS],
            );

            // Skip identity ops writing to their own source (nop)
            if op_i == 9 && dst_i == s1_i {
                continue;
            }
            // Skip writes to non-scratch registers (args + consts are immutable in discrete)
            if dst_i < self.n_args + N_CONSTS {
                continue;
            }

            let scratch_idx = dst_i - self.n_args - N_CONSTS;
            if scratch_idx < N_SCRATCH {
                scratch_written[scratch_idx] = true;
            }

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

            let is_first_write = scratch_idx < N_SCRATCH
                && !instructions
                    .iter()
                    .any(|i: &String| i.contains(&format!("{dst_name} =")));
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
                if !instructions
                    .iter()
                    .any(|inst| inst.contains(&format!("{rn}: i64 =")))
                {
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

#[allow(dead_code)]
const N_ARM_STEPS: usize = 20;
#[allow(dead_code)]
const N_ARM_SCRATCH: usize = 6;

#[allow(dead_code)]
struct SoftArrayRegisterMachine {
    n_scalar: usize, // number of scalar args (after the array arg)
    params: Vec<f32>,
}

#[allow(dead_code)]
impl SoftArrayRegisterMachine {
    // Register file:
    // [arr_len, scalar0, scalar1, ..., const0..5, iter_idx, scratch0..N]
    // Plus: array memory[MAX_ARR] accessible via soft index
    fn n_regs(n_scalar: usize) -> usize {
        1 + n_scalar + N_CONSTS + 1 + N_ARM_SCRATCH
    }
    // iter_idx position in register file
    fn iter_idx_pos(n_scalar: usize) -> usize {
        1 + n_scalar + N_CONSTS
    }

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
        let mut s = Self {
            n_scalar,
            params: vec![0f32; np],
        };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        // Bias loop_bound toward arr_len (reg[0])
        let lb = s.loop_bound_off();
        s.params[lb] = 3.0;
        s
    }

    fn ret_off(&self) -> usize {
        N_ARM_STEPS * Self::step_size(self.n_scalar)
    }
    fn consts_off(&self) -> usize {
        self.ret_off() + Self::n_regs(self.n_scalar) + 1
    }
    fn loop_bound_off(&self) -> usize {
        self.consts_off() + N_CONSTS
    }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let nr = Self::n_regs(self.n_scalar);
        let ss = Self::step_size(self.n_scalar);
        let co = self.consts_off();
        let lb_off = self.loop_bound_off();
        let iter_pos = Self::iter_idx_pos(self.n_scalar);

        // Initialize register file
        let mut regs = vec![0f32; nr];
        regs[0] = arr_len; // reg[0] = arr_len
        for (i, &v) in scalar_args.iter().enumerate() {
            regs[1 + i] = v;
        }
        for i in 0..N_CONSTS {
            regs[1 + self.n_scalar + i] = self.params[co + i];
        }
        // iter_idx and scratch start at 0

        // Array memory (padded)
        let mut mem = vec![0f32; MAX_ARR];
        for (i, &v) in arr.iter().enumerate() {
            if i < MAX_ARR {
                mem[i] = v;
            }
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
                let s1_w =
                    softmax_temp(&self.params[off + N_RM_OPS..off + N_RM_OPS + nr + 1], temp);
                let s2_w = softmax_temp(
                    &self.params[off + N_RM_OPS + nr + 1..off + N_RM_OPS + 2 * nr + 1],
                    temp,
                );
                // dst has nr+1 slots: [regs..., arr[idx]]
                let dst_w = softmax_temp(
                    &self.params[off + N_RM_OPS + 2 * nr + 1..off + N_RM_OPS + 3 * nr + 2],
                    temp,
                );
                // arr_idx: which register to use as the array index
                let aidx_w = softmax_temp(
                    &self.params[off + N_RM_OPS + 3 * nr + 2..off + N_RM_OPS + 4 * nr + 2],
                    temp,
                );
                let gate_cmp_w = softmax_temp(
                    &self.params[off + N_RM_OPS + 4 * nr + 2..off + N_RM_OPS + 4 * nr + 2 + N_CMPS],
                    temp,
                );
                let gate_s1_w = softmax_temp(
                    &self.params[off + N_RM_OPS + 4 * nr + 2 + N_CMPS
                        ..off + N_RM_OPS + 5 * nr + 2 + N_CMPS],
                    temp,
                );
                let gate_s2_w = softmax_temp(
                    &self.params[off + N_RM_OPS + 5 * nr + 2 + N_CMPS
                        ..off + N_RM_OPS + 6 * nr + 2 + N_CMPS],
                    temp,
                );

                // Compute soft array index
                let soft_idx = soft_read(&regs, &aidx_w);
                // Soft array read: weighted sum over memory positions near soft_idx
                let arr_val: f32 = (0..MAX_ARR)
                    .map(|j| {
                        let dist = (j as f32 - soft_idx).abs();
                        let w = (-(dist * dist) / (temp.max(0.3))).exp();
                        w * mem[j]
                    })
                    .sum::<f32>()
                    / (0..MAX_ARR)
                        .map(|j| {
                            let dist = (j as f32 - soft_idx).abs();
                            (-(dist * dist) / (temp.max(0.3))).exp()
                        })
                        .sum::<f32>()
                        .max(1e-8);

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
        let ret_arr_val: f32 = (0..MAX_ARR)
            .map(|j| {
                let dist = (j as f32 - regs[iter_pos]).abs();
                let w = (-(dist * dist) / (temp.max(0.3))).exp();
                w * mem[j]
            })
            .sum::<f32>()
            / (0..MAX_ARR)
                .map(|j| {
                    let dist = (j as f32 - regs[iter_pos]).abs();
                    (-(dist * dist) / (temp.max(0.3))).exp()
                })
                .sum::<f32>()
                .max(1e-8);
        reg_part + ret_w[nr] * ret_arr_val
    }

    fn loss(&self, examples: &[ArrExample], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples
            .iter()
            .map(|ex| {
                let diff = self.forward(&ex.arr, ex.arr_len, &ex.scalar_args, temp) - ex.expected;
                diff * diff
            })
            .sum::<f32>()
            / n
    }
}

// ─── Register Machine synthesis entry points ─────────────────────────────────

/// Attempt synthesis via SoftRegisterMachine for scalar problems.
pub fn synthesize_register_machine(problem: &Problem) -> Option<SolveResult> {
    // Only scalar-input problems
    if !problem
        .examples
        .iter()
        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))))
    {
        return None;
    }
    let n_args = problem.examples[0].inputs.len();
    let fn_name = problem.function_name();

    // Build float examples
    let examples: Vec<(Vec<f32>, f32)> = problem
        .examples
        .iter()
        .map(|ex| {
            let inputs: Vec<f32> = ex
                .inputs
                .iter()
                .map(|v| match v {
                    Value::Int(i) => *i as f32,
                    _ => 0.0,
                })
                .collect();
            (inputs, ex.expected_int() as f32)
        })
        .collect();

    const N_STEPS_RM: usize = 600;
    const N_RESTARTS_RM: usize = 5;
    let nr = SoftRegisterMachine::n_regs(n_args);

    // Suppress unused-variable warnings now that the hand-coded biases are
    // gone. `nr` was used for the biased-restart layout maths.
    let _ = nr;
    for restart in 0..N_RESTARTS_RM {
        let mut prog = SoftRegisterMachine::new(n_args);
        // No hand-coded restart biases — restart diversity comes purely from
        // pseudo-random noise. Cross-problem warm starts are now the
        // responsibility of `strategy::CachedTeachers`, which feeds prior
        // solved programs back into the differentiable bridge as teachers.
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
            if step == chk1 {
                loss_at_chk1 = best_loss;
            }
            if step == N_STEPS_RM / 2 && best_loss > loss_at_chk1 * 0.95 {
                break;
            }

            let temp = (2.0f32 * (1.0 - step as f32 / N_STEPS_RM as f32)).max(0.1);
            let ex_ref = &examples;
            let loss = {
                let p = SoftRegisterMachine {
                    n_args,
                    params: prog.params.clone(),
                };
                p.loss(ex_ref, temp)
            };
            if loss < best_loss {
                best_loss = loss;
                best_params = prog.params.clone();
            }

            // Try discretize periodically
            if loss < 1.0 || step % 100 == 99 {
                let code = SoftRegisterMachine {
                    n_args,
                    params: prog.params.clone(),
                }
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
                    let code2 = SoftRegisterMachine {
                        n_args,
                        params: best_params.clone(),
                    }
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
            let grads = fd_grad(
                &prog.params,
                |p, t| {
                    SoftRegisterMachine {
                        n_args: na,
                        params: p.to_vec(),
                    }
                    .loss(&ex2, t)
                },
                temp,
            );
            opt.step(&mut prog.params, &grads);
        }

        // Final check with best params
        let code = SoftRegisterMachine {
            n_args,
            params: best_params,
        }
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
    use crate::benchmark::{Example, Problem, Value};

    /// Smoke test: register machine on a hand-built 1-arg problem.
    /// Uses a simple problem (double) where biased init should converge fast.
    #[test]
    fn register_machine_smoke_test() {
        // double(a) = 2*a — simple enough for RM to find
        let problem = Problem {
            name: "double_v0".to_string(),
            category: "test",
            description: "double",
            signature: "fn double(a: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: Value::Int(2),
                },
                Example {
                    inputs: vec![Value::Int(3)],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Int(0)],
                    expected: Value::Int(0),
                },
                Example {
                    inputs: vec![Value::Int(-2)],
                    expected: Value::Int(-4),
                },
                Example {
                    inputs: vec![Value::Int(5)],
                    expected: Value::Int(10),
                },
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
