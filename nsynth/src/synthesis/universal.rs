use super::*;

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

// Stage-1 scaling (April 2026): bump from (3,6,2)=11 to (4,8,3)=15.
// Motivated by Tier-2 game-adjacent problems that need more dispatch branches
// (score_tracker: 4-way), more post-processing (simulate_gravity: clamp+return),
// and more compositional state (count_adjacent_diff: item+prev+counter).
pub const N_INIT_SLOTS: usize = 4;
pub const N_LOOP_SLOTS: usize = 8;
pub const N_POST_SLOTS: usize = 3;
pub const N_UNIV_SLOTS: usize = N_INIT_SLOTS + N_LOOP_SLOTS + N_POST_SLOTS; // 15

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

/// Build a [`MetaRecord`] from the output of a completed universal synthesis.
///
/// Neither the description nor the I/O examples are hand-coded — the description
/// is argmax-discretised from learned soft parameters, and the examples are the
/// observed inputs/outputs that the synthesiser had to fit. Used by the corpus
/// harvester to turn every successful solve into meta-learner training data.
pub fn record_from_synthesis(
    fn_name: &str,
    n_args: usize,
    params: Vec<f32>,
    io_examples: Vec<(Vec<i64>, i64)>,
    source: &str,
) -> MetaRecord {
    let prog = SoftUniversalProgram::new_from_params(n_args, params);
    MetaRecord {
        fn_name: fn_name.to_string(),
        description: prog.params_to_description(),
        io_examples,
        source: source.to_string(),
    }
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
            (inputs, ex.expected_int() as f32)
        })
        .collect();

    let default_names = ["a", "b", "c", "d", "e", "f"];
    let param_names: Vec<&str> = (0..n_args)
        .map(|i| default_names.get(i).copied().unwrap_or("x"))
        .collect();

    let univ_steps = max_steps;
    const N_RESTARTS: usize = 5;

    for restart in 0..N_RESTARTS {
        // Bail between restarts once a wall-clock deadline is spent (QuerySolveBudget
        // / per-attempt cap) so the scalar universal route stays inside a bounded query.
        if crate::synthesis::common::train_deadline_exceeded() {
            break;
        }
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
        if step % 16 == 0 && crate::synthesis::common::train_deadline_exceeded() {
            break;
        }
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
            (inputs, ex.expected_int() as f32)
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
        if step % 16 == 0 && crate::synthesis::common::train_deadline_exceeded() {
            break;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_universal_structural_test() {
        // Verify SoftUniversalProgram: check forward pass produces finite output,
        // parameter layout is consistent, and discretize_and_emit produces parseable code.
        let n_args = 1usize;
        let pool = univ_pool(n_args);
        let lip = univ_lip(n_args);
        let sps = univ_sps(pool);

        // Expected sizes — recompute from current N_INIT / N_LOOP / N_POST to
        // survive slot-count scaling without manual updates every time.
        // Invariants that must still hold:
        //   pool = n_args + N_CONSTS + N_UNIV_SLOTS
        //   lip  = n_args + N_CONSTS + N_INIT_SLOTS
        //   sps  = (N_OPS+1) + 5*pool + N_CMPS
        assert_eq!(
            pool,
            n_args + N_CONSTS + N_UNIV_SLOTS,
            "pool = n_args + N_CONSTS + N_UNIV_SLOTS"
        );
        assert_eq!(
            lip,
            n_args + N_CONSTS + N_INIT_SLOTS,
            "lip = n_args + N_CONSTS + N_INIT_SLOTS"
        );
        assert_eq!(
            sps,
            (N_OPS + 1) + 5 * pool + N_CMPS,
            "sps = (N_OPS+1) + 5*pool + N_CMPS"
        );
        assert_eq!(
            N_UNIV_SLOTS,
            N_INIT_SLOTS + N_LOOP_SLOTS + N_POST_SLOTS,
            "N_UNIV_SLOTS decomposes into init+loop+post"
        );
        let n_total = SoftUniversalProgram::n_params_for(n_args);
        let expected =
            N_UNIV_SLOTS * sps + N_LOOP_SLOTS * lip + N_CMPS + 2 * pool + pool + N_CONSTS;
        assert_eq!(n_total, expected, "total params for n_args=1: {n_total}");

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
    fn warm_start_synthesis_test() {
        use crate::benchmark::{Example, Problem, Value};
        // Build a simple n+1 problem
        let examples: Vec<Example> = (1i64..=8)
            .map(|n| Example {
                inputs: vec![Value::Int(n)],
                expected: Value::Int(n + 1),
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

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,
            functions: vec![],
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
}
