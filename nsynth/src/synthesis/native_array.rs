use super::*;

// ─── Shared array example plumbing ───────────────────────────────────────────
// Used by every array-gradient synthesizer in this file plus the soft array
// register machine (register_machine.rs) and the universal-array fallback
// (universal_array.rs). Re-exported at synthesis module level via mod.rs.

/// Array + scalar inputs extracted from a Problem, padded to MAX_ARR.
#[derive(Clone)]
pub(crate) struct ArrExample {
    pub(crate) arr: Vec<f32>, // padded to MAX_ARR
    pub(crate) arr_len: f32,
    pub(crate) scalar_args: Vec<f32>,
    pub(crate) expected: f32,
}

/// Extract array + scalar inputs from a Problem.
/// Returns None if the first input isn't an array or if there are no examples.
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
        let arr = match ex.inputs[0].as_i64_slice() {
            Some(a) => a,
            None => return None,
        };
        // arr_len must not exceed the padded buffer width, or the soft-array
        // gradient loops index past MAX_ARR (panic). Arrays longer than MAX_ARR
        // are truncated to their first MAX_ARR elements for the gradient path.
        let arr_len = arr.len().min(MAX_ARR) as f32;
        let mut padded = vec![0f32; MAX_ARR];
        for (i, v) in arr.iter().enumerate() {
            if i < MAX_ARR {
                padded[i] = *v as f32;
            }
        }
        let mut scalar_args = Vec::with_capacity(n_scalar);
        for v in &ex.inputs[1..] {
            match v {
                Value::Int(iv) => scalar_args.push(*iv as f32),
                _ => return None,
            }
        }
        examples.push(ArrExample {
            arr: padded,
            arr_len,
            scalar_args,
            expected: ex.expected_int() as f32,
        });
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
    fn n_init(&self) -> usize {
        1 + N_CONSTS + self.n_scalar
    }
    // Sources for rhs: [item, item*item, acc, consts(6), scalar_args(n_scalar)]
    fn n_rhs(&self) -> usize {
        3 + N_CONSTS + self.n_scalar
    }
    // Sources for return: [acc, consts(6), scalar_args(n_scalar)]
    fn n_ret(&self) -> usize {
        1 + N_CONSTS + self.n_scalar
    }

    fn n_params(n_scalar: usize) -> usize {
        let init = 1 + N_CONSTS + n_scalar;
        let rhs = 3 + N_CONSTS + n_scalar;
        let ret = 1 + N_CONSTS + n_scalar;
        init + N_OPS + rhs + ret + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self {
            n_scalar,
            params: vec![0f32; Self::n_params(n_scalar)],
        };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    fn init_off(&self) -> usize {
        0
    }
    fn op_off(&self) -> usize {
        self.n_init()
    }
    fn rhs_off(&self) -> usize {
        self.n_init() + N_OPS
    }
    fn ret_off(&self) -> usize {
        self.n_init() + N_OPS + self.n_rhs()
    }
    fn consts_off(&self) -> usize {
        self.n_init() + N_OPS + self.n_rhs() + self.n_ret()
    }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let n_init = self.n_init();
        let n_rhs = self.n_rhs();
        let n_ret = self.n_ret();
        let co = self.consts_off();

        // Load constants
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        // Init: soft-select from [arr[0], consts..., scalar_args...]
        let init_w = softmax_temp(
            &self.params[self.init_off()..self.init_off() + n_init],
            temp,
        );
        let mut init_storage = vec![0f32; n_init];
        init_storage[0] = arr[0]; // arr[0]
        for i in 0..N_CONSTS {
            init_storage[1 + i] = consts[i];
        }
        for i in 0..self.n_scalar {
            init_storage[1 + N_CONSTS + i] = scalar_args[i];
        }
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
            for j in 0..N_CONSTS {
                rhs_storage[3 + j] = consts[j];
            }
            for j in 0..self.n_scalar {
                rhs_storage[3 + N_CONSTS + j] = scalar_args[j];
            }
            let rhs = soft_read(&rhs_storage, &rhs_w);
            let new_acc = soft_op(acc, rhs, &op_w);
            acc = in_bounds * new_acc + (1.0 - in_bounds) * acc;
        }

        // Return: soft-select from [acc, consts..., scalar_args...]
        let ret_w = softmax_temp(&self.params[self.ret_off()..self.ret_off() + n_ret], temp);
        let mut ret_storage = vec![0f32; n_ret];
        ret_storage[0] = acc;
        for j in 0..N_CONSTS {
            ret_storage[1 + j] = consts[j];
        }
        for j in 0..self.n_scalar {
            ret_storage[1 + N_CONSTS + j] = scalar_args[j];
        }
        soft_read(&ret_storage, &ret_w)
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

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let n_init = self.n_init();
        let n_rhs = self.n_rhs();
        let n_ret = self.n_ret();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|i| self.params[co + i].round() as i64)
            .collect();

        // Init source names: [arr[0], c0..c5, scalar_args...]
        let init_names: Vec<String> = std::iter::once("arr[0]".to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        // RHS source names: [item, item*item, acc, c0..c5, scalar_args...]
        let rhs_names: Vec<String> = ["item", "item*item", "acc"]
            .iter()
            .map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        // Return source names: [acc, c0..c5, scalar_args...]
        let ret_names: Vec<String> = ["acc"]
            .iter()
            .map(|s| s.to_string())
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
        writeln!(out, "    acc: i64 = {init_src};").unwrap();
        out.push_str("    for item in arr {\n");
        writeln!(out, "        acc = acc {} {};", op_names[op_i], rhs_src).unwrap();
        out.push_str("    }\n");
        writeln!(out, "    return {};", ret_names[ret_i]).unwrap();
        out.push_str("}\n");
        out
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
    fn init_pool(&self) -> usize {
        1 + N_CONSTS + self.n_scalar
    }
    // Body pool: [item, acc, consts(6), scalar_args(n_scalar)]
    fn pool(&self) -> usize {
        2 + N_CONSTS + self.n_scalar
    }

    fn n_params(n_scalar: usize) -> usize {
        let ip = 1 + N_CONSTS + n_scalar;
        let p = 2 + N_CONSTS + n_scalar;
        // init(ip) + cmp(N_CMPS) + 2*cmp_src(p) + body_op(N_OPS) + body_rhs(p) + mode(1) + ret(p) + consts(N_CONSTS)
        ip + N_CMPS + 2 * p + N_OPS + p + 1 + p + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self {
            n_scalar,
            params: vec![0f32; Self::n_params(n_scalar)],
        };
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

    fn init_off(&self) -> usize {
        0
    }
    fn cmp_off(&self) -> usize {
        self.init_pool()
    }
    fn cmp_s1_off(&self) -> usize {
        self.init_pool() + N_CMPS
    }
    fn cmp_s2_off(&self) -> usize {
        self.init_pool() + N_CMPS + self.pool()
    }
    fn body_op_off(&self) -> usize {
        self.init_pool() + N_CMPS + 2 * self.pool()
    }
    fn body_rhs_off(&self) -> usize {
        self.init_pool() + N_CMPS + 2 * self.pool() + N_OPS
    }
    /// Mode logit: >0 → accumulate (acc = acc OP rhs), <0 → replace (acc = rhs)
    fn mode_off(&self) -> usize {
        self.init_pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool()
    }
    fn consts_off(&self) -> usize {
        self.init_pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool() + 1 + self.pool()
    }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let ip = self.init_pool();
        let pool = self.pool();
        let co = self.consts_off();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        // Init from pool: [arr[0], consts..., scalar_args...]
        let mut init_s = vec![0f32; ip];
        init_s[0] = arr[0];
        for j in 0..N_CONSTS {
            init_s[1 + j] = consts[j];
        }
        for j in 0..self.n_scalar {
            init_s[1 + N_CONSTS + j] = scalar_args[j];
        }
        let init_w = softmax_temp(&self.params[self.init_off()..self.init_off() + ip], temp);
        let mut acc = soft_read(&init_s, &init_w);

        // Fixed logits
        let cmp_w = softmax_temp(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS], temp);
        let cmp_s1_w = softmax_temp(
            &self.params[self.cmp_s1_off()..self.cmp_s1_off() + pool],
            temp,
        );
        let cmp_s2_w = softmax_temp(
            &self.params[self.cmp_s2_off()..self.cmp_s2_off() + pool],
            temp,
        );
        let body_op_w = softmax_temp(
            &self.params[self.body_op_off()..self.body_op_off() + N_OPS],
            temp,
        );
        let body_rhs_w = softmax_temp(
            &self.params[self.body_rhs_off()..self.body_rhs_off() + pool],
            temp,
        );

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            let item = arr[i];

            // Build pool: [item, acc, consts..., scalar_args...]
            let mut p = vec![0f32; pool];
            p[0] = item;
            p[1] = acc;
            for j in 0..N_CONSTS {
                p[2 + j] = consts[j];
            }
            for j in 0..self.n_scalar {
                p[2 + N_CONSTS + j] = scalar_args[j];
            }

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
        examples
            .iter()
            .map(|ex| {
                let diff = self.forward(&ex.arr, ex.arr_len, &ex.scalar_args, temp) - ex.expected;
                diff * diff
            })
            .sum::<f32>()
            / n
    }

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let ip = self.init_pool();
        let pool = self.pool();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|i| self.params[co + i].round() as i64)
            .collect();

        // Init source names: [arr[0], consts..., scalar_args...]
        let init_names: Vec<String> = std::iter::once("arr[0]".to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let pool_names: Vec<String> = ["item", "acc"]
            .iter()
            .map(|s| s.to_string())
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

        let init_src = &init_names[init_i];
        // Use := for arr[0] (Mog syntax for init from first element), = for literals
        let init_assign = if init_i == 0 { ":=" } else { ": i64 =" };

        let mut out = format!("{sig} {{\n");
        writeln!(out, "    acc {init_assign} {init_src};").unwrap();
        out.push_str("    for item in arr {\n");
        writeln!(
            out,
            "        if {} {} {} {{",
            pool_names[cmp_s1_i], cmp_names[cmp_i], pool_names[cmp_s2_i]
        )
        .unwrap();
        if is_accum {
            writeln!(
                out,
                "            acc = acc {} {};",
                op_names[op_i], pool_names[rhs_i]
            )
            .unwrap();
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
    fn pool1(&self) -> usize {
        3 + N_CONSTS + self.n_scalar
    }
    // Pool for update2: [item, a1, a2, consts(6), scalar_args(n_scalar)]
    fn pool2(&self) -> usize {
        3 + N_CONSTS + self.n_scalar
    }
    // Pool for init: [arr[0], consts(6), scalar_args(n_scalar)]
    fn init_pool(&self) -> usize {
        1 + N_CONSTS + self.n_scalar
    }
    // Return pool: [a1, a2, consts(6), scalar_args(n_scalar)]
    fn ret_pool(&self) -> usize {
        2 + N_CONSTS + self.n_scalar
    }

    fn n_params(n_scalar: usize) -> usize {
        let ip = 1 + N_CONSTS + n_scalar;
        let p1 = 3 + N_CONSTS + n_scalar;
        let _p2 = 3 + N_CONSTS + n_scalar;
        let rp = 2 + N_CONSTS + n_scalar;
        // init1(ip) + init2(ip) + gate1(N_CMPS + 2*p1) + then1(p1) + gate2(N_CMPS + 2*p2) + then2(p2) + ret_op(N_OPS) + ret_s1(rp) + ret_s2(rp) + consts(6)
        2 * ip + 2 * (N_CMPS + 3 * p1) + N_OPS + 2 * rp + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self {
            n_scalar,
            params: vec![0f32; Self::n_params(n_scalar)],
        };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    fn init1_off(&self) -> usize {
        0
    }
    fn init2_off(&self) -> usize {
        self.init_pool()
    }
    fn g1_cmp_off(&self) -> usize {
        2 * self.init_pool()
    }
    fn g1_s1_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS
    }
    fn g1_s2_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS + self.pool1()
    }
    fn then1_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS + 2 * self.pool1()
    }
    fn g2_cmp_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS + 3 * self.pool1()
    }
    fn g2_s1_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS
    }
    fn g2_s2_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + self.pool2()
    }
    fn then2_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + 2 * self.pool2()
    }
    fn ret_op_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + 3 * self.pool2()
    }
    fn ret_s1_off(&self) -> usize {
        2 * self.init_pool() + N_CMPS + 3 * self.pool1() + N_CMPS + 3 * self.pool2() + N_OPS
    }
    fn ret_s2_off(&self) -> usize {
        2 * self.init_pool()
            + N_CMPS
            + 3 * self.pool1()
            + N_CMPS
            + 3 * self.pool2()
            + N_OPS
            + self.ret_pool()
    }
    fn consts_off(&self) -> usize {
        2 * self.init_pool()
            + N_CMPS
            + 3 * self.pool1()
            + N_CMPS
            + 3 * self.pool2()
            + N_OPS
            + 2 * self.ret_pool()
    }

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
        for j in 0..N_CONSTS {
            init_s[1 + j] = consts[j];
        }
        for j in 0..self.n_scalar {
            init_s[1 + N_CONSTS + j] = scalar_args[j];
        }

        let init1_w = softmax_temp(&self.params[self.init1_off()..self.init1_off() + ip], temp);
        let init2_w = softmax_temp(&self.params[self.init2_off()..self.init2_off() + ip], temp);
        let mut a1 = soft_read(&init_s, &init1_w);
        let mut a2 = soft_read(&init_s, &init2_w);

        // Fixed logits
        let g1_cmp_w = softmax_temp(
            &self.params[self.g1_cmp_off()..self.g1_cmp_off() + N_CMPS],
            temp,
        );
        let g1_s1_w = softmax_temp(&self.params[self.g1_s1_off()..self.g1_s1_off() + p1], temp);
        let g1_s2_w = softmax_temp(&self.params[self.g1_s2_off()..self.g1_s2_off() + p1], temp);
        let then1_w = softmax_temp(&self.params[self.then1_off()..self.then1_off() + p1], temp);
        let g2_cmp_w = softmax_temp(
            &self.params[self.g2_cmp_off()..self.g2_cmp_off() + N_CMPS],
            temp,
        );
        let g2_s1_w = softmax_temp(&self.params[self.g2_s1_off()..self.g2_s1_off() + p2], temp);
        let g2_s2_w = softmax_temp(&self.params[self.g2_s2_off()..self.g2_s2_off() + p2], temp);
        let then2_w = softmax_temp(&self.params[self.then2_off()..self.then2_off() + p2], temp);

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            let item = arr[i];

            // Pool1: [item, a1, a2, consts..., scalar_args...]
            let mut pool1 = vec![0f32; p1];
            pool1[0] = item;
            pool1[1] = a1;
            pool1[2] = a2;
            for j in 0..N_CONSTS {
                pool1[3 + j] = consts[j];
            }
            for j in 0..self.n_scalar {
                pool1[3 + N_CONSTS + j] = scalar_args[j];
            }

            let g1 = soft_cmp(
                soft_read(&pool1, &g1_s1_w),
                soft_read(&pool1, &g1_s2_w),
                &g1_cmp_w,
                temp,
            );
            let t1 = soft_read(&pool1, &then1_w);
            let new_a1 = g1 * t1 + (1.0 - g1) * a1;
            a1 = in_bounds * new_a1 + (1.0 - in_bounds) * a1;

            // Pool2: [item, a1(updated), a2, consts..., scalar_args...]
            let mut pool2 = vec![0f32; p2];
            pool2[0] = item;
            pool2[1] = a1;
            pool2[2] = a2;
            for j in 0..N_CONSTS {
                pool2[3 + j] = consts[j];
            }
            for j in 0..self.n_scalar {
                pool2[3 + N_CONSTS + j] = scalar_args[j];
            }

            let g2 = soft_cmp(
                soft_read(&pool2, &g2_s1_w),
                soft_read(&pool2, &g2_s2_w),
                &g2_cmp_w,
                temp,
            );
            let t2 = soft_read(&pool2, &then2_w);
            let new_a2 = g2 * t2 + (1.0 - g2) * a2;
            a2 = in_bounds * new_a2 + (1.0 - in_bounds) * a2;
        }

        // Return: soft_op(a1, a2, ret_op) or soft_select
        let ret_op_w = softmax_temp(
            &self.params[self.ret_op_off()..self.ret_op_off() + N_OPS],
            temp,
        );
        let ret_s1_w = softmax_temp(
            &self.params[self.ret_s1_off()..self.ret_s1_off() + rp],
            temp,
        );
        let ret_s2_w = softmax_temp(
            &self.params[self.ret_s2_off()..self.ret_s2_off() + rp],
            temp,
        );

        let mut ret_pool = vec![0f32; rp];
        ret_pool[0] = a1;
        ret_pool[1] = a2;
        for j in 0..N_CONSTS {
            ret_pool[2 + j] = consts[j];
        }
        for j in 0..self.n_scalar {
            ret_pool[2 + N_CONSTS + j] = scalar_args[j];
        }

        let s1 = soft_read(&ret_pool, &ret_s1_w);
        let s2 = soft_read(&ret_pool, &ret_s2_w);
        soft_op(s1, s2, &ret_op_w)
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

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let ip = self.init_pool();
        let p1 = self.pool1();
        let rp = self.ret_pool();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|i| self.params[co + i].round() as i64)
            .collect();

        let init_names: Vec<String> = std::iter::once("arr[0]".to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let pool1_names: Vec<String> = ["item", "lo", "hi"]
            .iter()
            .map(|s| s.to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let ret_names: Vec<String> = ["lo", "hi"]
            .iter()
            .map(|s| s.to_string())
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
        writeln!(out, "    lo: i64 = {};", init_names[init1_i]).unwrap();
        writeln!(out, "    hi: i64 = {};", init_names[init2_i]).unwrap();
        out.push_str("    for item in arr {\n");
        writeln!(
            out,
            "        if {} {} {} {{ lo = {}; }}",
            pool1_names[g1_s1_i], cmp_names[g1_cmp_i], pool1_names[g1_s2_i], pool1_names[then1_i]
        )
        .unwrap();
        writeln!(
            out,
            "        if {} {} {} {{ hi = {}; }}",
            pool1_names[g2_s1_i], cmp_names[g2_cmp_i], pool1_names[g2_s2_i], pool1_names[then2_i]
        )
        .unwrap();
        out.push_str("    }\n");
        writeln!(
            out,
            "    return {} {} {};",
            ret_names[ret_s1_i], op_names[ret_op_i], ret_names[ret_s2_i]
        )
        .unwrap();
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
    fn pool(&self) -> usize {
        4 + N_CONSTS + self.n_scalar
    }
    // Init pool: [arr[0], consts(6), scalar_args(n_scalar)]
    fn init_pool(&self) -> usize {
        1 + N_CONSTS + self.n_scalar
    }

    fn n_params(n_scalar: usize) -> usize {
        let ip = 1 + N_CONSTS + n_scalar;
        let p = 4 + N_CONSTS + n_scalar;
        // init(ip) + pre_op(N_OPS) + pre_s1(p) + pre_s2(p) + [cmp(N_CMPS) + cmp_s1(p) + cmp_s2(p)] + body_op(N_OPS) + body_rhs(p) + ret(p) + consts(6)
        ip + N_OPS + 2 * p + N_CMPS + 2 * p + N_OPS + p + p + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self {
            n_scalar,
            params: vec![0f32; Self::n_params(n_scalar)],
        };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    fn init_off(&self) -> usize {
        0
    }
    fn pre_op_off(&self) -> usize {
        self.init_pool()
    }
    fn pre_s1_off(&self) -> usize {
        self.init_pool() + N_OPS
    }
    fn pre_s2_off(&self) -> usize {
        self.init_pool() + N_OPS + self.pool()
    }
    fn cmp_off(&self) -> usize {
        self.init_pool() + N_OPS + 2 * self.pool()
    }
    fn cmp_s1_off(&self) -> usize {
        self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS
    }
    fn cmp_s2_off(&self) -> usize {
        self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + self.pool()
    }
    fn body_op_off(&self) -> usize {
        self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool()
    }
    fn body_rhs_off(&self) -> usize {
        self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS
    }
    fn ret_off(&self) -> usize {
        self.init_pool() + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool()
    }
    fn consts_off(&self) -> usize {
        self.init_pool()
            + N_OPS
            + 2 * self.pool()
            + N_CMPS
            + 2 * self.pool()
            + N_OPS
            + 2 * self.pool()
    }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let ip = self.init_pool();
        let p = self.pool();
        let co = self.consts_off();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        // Init
        let mut init_s = vec![0f32; ip];
        init_s[0] = arr[0];
        for j in 0..N_CONSTS {
            init_s[1 + j] = consts[j];
        }
        for j in 0..self.n_scalar {
            init_s[1 + N_CONSTS + j] = scalar_args[j];
        }
        let init_w = softmax_temp(&self.params[self.init_off()..self.init_off() + ip], temp);
        let mut acc = soft_read(&init_s, &init_w);

        let mut prev = arr[0]; // first element
        let _run_len = 1f32;

        let pre_op_w = softmax_temp(
            &self.params[self.pre_op_off()..self.pre_op_off() + N_OPS],
            temp,
        );
        let pre_s1_w = softmax_temp(&self.params[self.pre_s1_off()..self.pre_s1_off() + p], temp);
        let pre_s2_w = softmax_temp(&self.params[self.pre_s2_off()..self.pre_s2_off() + p], temp);
        let cmp_w = softmax_temp(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS], temp);
        let cmp_s1_w = softmax_temp(&self.params[self.cmp_s1_off()..self.cmp_s1_off() + p], temp);
        let cmp_s2_w = softmax_temp(&self.params[self.cmp_s2_off()..self.cmp_s2_off() + p], temp);
        let body_op_w = softmax_temp(
            &self.params[self.body_op_off()..self.body_op_off() + N_OPS],
            temp,
        );
        let body_rhs_w = softmax_temp(
            &self.params[self.body_rhs_off()..self.body_rhs_off() + p],
            temp,
        );
        let ret_w = softmax_temp(&self.params[self.ret_off()..self.ret_off() + p], temp);

        for i in 0..MAX_ARR {
            let in_bounds = sigmoid((arr_len - i as f32 - 0.5) / 0.3);
            if i == 0 {
                continue;
            } // skip first element (we already set prev = arr[0])
            let item = arr[i];

            // Pool: [item, prev, diff, acc, consts..., scalar_args...]
            let mut pool = vec![0f32; p];
            pool[0] = item;
            pool[1] = prev;
            pool[2] = item - prev; // diff (default, will be overridden by pre_op)
            pool[3] = acc;
            for j in 0..N_CONSTS {
                pool[4 + j] = consts[j];
            }
            for j in 0..self.n_scalar {
                pool[4 + N_CONSTS + j] = scalar_args[j];
            }

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
        ret_pool[0] = 0.0;
        ret_pool[1] = prev;
        ret_pool[2] = 0.0;
        ret_pool[3] = acc;
        for j in 0..N_CONSTS {
            ret_pool[4 + j] = consts[j];
        }
        for j in 0..self.n_scalar {
            ret_pool[4 + N_CONSTS + j] = scalar_args[j];
        }
        soft_read(&ret_pool, &ret_w)
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

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let ip = self.init_pool();
        let p = self.pool();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|i| self.params[co + i].round() as i64)
            .collect();

        let init_names: Vec<String> = std::iter::once("arr[0]".to_string())
            .chain(consts.iter().map(|v| format!("{v}")))
            .chain(scalar_names.iter().map(|s| s.to_string()))
            .collect();

        let pool_names: Vec<String> = ["item", "prev", "diff", "acc"]
            .iter()
            .map(|s| s.to_string())
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
        writeln!(
            out,
            "        diff: i64 = {} {} {};",
            pool_names[pre_s1_i], op_names[pre_op_i], pool_names[pre_s2_i]
        )
        .unwrap();
        writeln!(
            out,
            "        if {} {} {} {{",
            pool_names[cmp_s1_i], cmp_names[cmp_i], pool_names[cmp_s2_i]
        )
        .unwrap();
        writeln!(
            out,
            "            acc = acc {} {};",
            op_names[op_i], pool_names[rhs_i]
        )
        .unwrap();
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
    fn pool(&self) -> usize {
        6 + N_CONSTS + self.n_scalar
    }

    fn n_params(n_scalar: usize) -> usize {
        let p = 6 + N_CONSTS + n_scalar;
        // init(1) + pre_op(N_OPS) + pre_s1(p) + pre_s2(p) + cmp(N_CMPS) + cmp_s1(p) + cmp_s2(p)
        // + body_op(N_OPS) + body_rhs(p) + ret(p) + consts(6)
        1 + N_OPS + 2 * p + N_CMPS + 2 * p + N_OPS + p + p + N_CONSTS
    }

    fn new(n_scalar: usize) -> Self {
        let mut s = Self {
            n_scalar,
            params: vec![0f32; Self::n_params(n_scalar)],
        };
        let co = s.consts_off();
        s.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        s
    }

    fn init_off(&self) -> usize {
        0
    }
    fn pre_op_off(&self) -> usize {
        1
    }
    fn pre_s1_off(&self) -> usize {
        1 + N_OPS
    }
    fn pre_s2_off(&self) -> usize {
        1 + N_OPS + self.pool()
    }
    fn cmp_off(&self) -> usize {
        1 + N_OPS + 2 * self.pool()
    }
    fn cmp_s1_off(&self) -> usize {
        1 + N_OPS + 2 * self.pool() + N_CMPS
    }
    fn cmp_s2_off(&self) -> usize {
        1 + N_OPS + 2 * self.pool() + N_CMPS + self.pool()
    }
    fn body_op_off(&self) -> usize {
        1 + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool()
    }
    fn body_rhs_off(&self) -> usize {
        1 + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS
    }
    fn ret_off(&self) -> usize {
        1 + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS + self.pool()
    }
    fn consts_off(&self) -> usize {
        1 + N_OPS + 2 * self.pool() + N_CMPS + 2 * self.pool() + N_OPS + 2 * self.pool()
    }

    fn forward(&self, arr: &[f32], arr_len: f32, scalar_args: &[f32], temp: f32) -> f32 {
        let p = self.pool();
        let co = self.consts_off();
        let consts: Vec<f32> = (0..N_CONSTS).map(|i| self.params[co + i]).collect();

        let mut acc = self.params[self.init_off()];

        // Fixed logits
        let pre_op_w = softmax_temp(
            &self.params[self.pre_op_off()..self.pre_op_off() + N_OPS],
            temp,
        );
        let pre_s1_w = softmax_temp(&self.params[self.pre_s1_off()..self.pre_s1_off() + p], temp);
        let pre_s2_w = softmax_temp(&self.params[self.pre_s2_off()..self.pre_s2_off() + p], temp);
        let cmp_w = softmax_temp(&self.params[self.cmp_off()..self.cmp_off() + N_CMPS], temp);
        let cmp_s1_w = softmax_temp(&self.params[self.cmp_s1_off()..self.cmp_s1_off() + p], temp);
        let cmp_s2_w = softmax_temp(&self.params[self.cmp_s2_off()..self.cmp_s2_off() + p], temp);
        let body_op_w = softmax_temp(
            &self.params[self.body_op_off()..self.body_op_off() + N_OPS],
            temp,
        );
        let body_rhs_w = softmax_temp(
            &self.params[self.body_rhs_off()..self.body_rhs_off() + p],
            temp,
        );

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
            for j in 0..N_CONSTS {
                pool[6 + j] = consts[j];
            }
            for j in 0..self.n_scalar {
                pool[6 + N_CONSTS + j] = scalar_args[j];
            }

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
        for j in 0..N_CONSTS {
            ret_pool[6 + j] = consts[j];
        }
        for j in 0..self.n_scalar {
            ret_pool[6 + N_CONSTS + j] = scalar_args[j];
        }
        soft_read(&ret_pool, &ret_w)
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

    fn discretize_and_emit(&self, fn_name: &str, scalar_names: &[&str]) -> String {
        let p = self.pool();
        let co = self.consts_off();
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|i| self.params[co + i].round() as i64)
            .collect();

        let pool_names: Vec<String> = ["item", "acc", "i", "parity", "arr.len", "target"]
            .iter()
            .map(|s| s.to_string())
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
            writeln!(
                out,
                "    target: i64 = {} {} {};",
                pool_names[pre_s1_i], op_names[pre_op_i], pool_names[pre_s2_i]
            )
            .unwrap();
            out.push_str("    i: i64 = 0;\n");
            out.push_str("    while i < arr.len {\n");
            out.push_str("        item: i64 = arr[i];\n");
            writeln!(
                out,
                "        if {} {} {} {{",
                pool_names[cmp_s1_i], cmp_names[cmp_i], pool_names[cmp_s2_i]
            )
            .unwrap();
            writeln!(
                out,
                "            acc = acc {} {};",
                op_names[op_i], pool_names[rhs_i]
            )
            .unwrap();
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
            writeln!(
                out,
                "        if {} {} {} {{",
                pool_names[cmp_s1_i], cmp_names[cmp_i], pool_names[cmp_s2_i]
            )
            .unwrap();
            writeln!(
                out,
                "            acc = acc {} {};",
                op_names[op_i], pool_names[rhs_i]
            )
            .unwrap();
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

fn try_exact_array_warm_starts(
    problem: &Problem,
    fn_name: &str,
    n_scalar: usize,
    scalar_names: &[&str],
) -> Option<SolveResult> {
    for restart in 1..9 {
        let mut prog = SoftArrayAccumProgram::new(n_scalar);
        if restart == 1 {
            let init_off = prog.init_off();
            let op_off = prog.op_off();
            let rhs_off = prog.rhs_off();
            prog.params[init_off + 1] = 4.0;
            prog.params[op_off] = 4.0;
            prog.params[rhs_off] = 4.0;
        } else if restart == 2 {
            let init_off = prog.init_off();
            let op_off = prog.op_off();
            let rhs_off = prog.rhs_off();
            prog.params[init_off + 2] = 4.0;
            prog.params[op_off + 2] = 4.0;
            prog.params[rhs_off] = 4.0;
        } else if restart == 3 {
            let init_off = prog.init_off();
            let op_off = prog.op_off();
            let rhs_off = prog.rhs_off();
            prog.params[init_off + 1] = 4.0;
            prog.params[op_off] = 4.0;
            prog.params[rhs_off + 1] = 4.0;
        } else {
            continue;
        }

        let code = SoftArrayAccumProgram {
            n_scalar,
            params: prog.params.clone(),
        }
        .discretize_and_emit(fn_name, scalar_names);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: "arr_gradient".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }

    for restart in 1..9 {
        let mut prog = SoftArrayCondAccumProgram::new(n_scalar);
        let io = prog.init_off();
        let co2 = prog.cmp_off();
        let cs1 = prog.cmp_s1_off();
        let cs2 = prog.cmp_s2_off();
        let bo2 = prog.body_op_off();
        let br2 = prog.body_rhs_off();
        let mo = prog.mode_off();
        if restart == 1 {
            prog.params[io + 1] = 4.0;
            prog.params[co2 + 4] = 4.0;
            prog.params[cs1] = 4.0;
            prog.params[cs2 + 2] = 4.0;
            prog.params[bo2] = 4.0;
            prog.params[br2 + 3] = 4.0;
            prog.params[mo] = 4.0;
        } else if restart == 2 {
            prog.params[io + 1] = 4.0;
            prog.params[co2] = 4.0;
            prog.params[cs1] = 4.0;
            prog.params[cs2 + 2] = 4.0;
            prog.params[bo2] = 4.0;
            prog.params[br2] = 4.0;
            prog.params[mo] = 4.0;
        } else if restart == 3 {
            prog.params[io] = 4.0;
            prog.params[co2 + 4] = 4.0;
            prog.params[cs1] = 4.0;
            prog.params[cs2 + 1] = 4.0;
            prog.params[br2] = 4.0;
            prog.params[mo] = -4.0;
        } else if restart == 4 {
            prog.params[io] = 4.0;
            prog.params[co2] = 4.0;
            prog.params[cs1] = 4.0;
            prog.params[cs2 + 1] = 4.0;
            prog.params[br2] = 4.0;
            prog.params[mo] = -4.0;
        } else if restart == 5 {
            prog.params[io + 1] = 4.0;
            prog.params[co2 + 2] = 4.0;
            prog.params[cs1] = 4.0;
            prog.params[cs2 + 2] = 4.0;
            prog.params[bo2] = 4.0;
            prog.params[br2 + 3] = 4.0;
            prog.params[mo] = 4.0;
        } else if restart == 6 {
            prog.params[io + 1] = 4.0;
            prog.params[co2 + 4] = 4.0;
            prog.params[cs1] = 4.0;
            prog.params[cs2 + 2] = 4.0;
            prog.params[bo2] = 4.0;
            prog.params[br2] = 4.0;
            prog.params[mo] = 4.0;
        } else if restart == 7 && n_scalar >= 1 {
            let k_idx = 2 + N_CONSTS;
            prog.params[io + 1] = 4.0;
            prog.params[co2 + 2] = 4.0;
            prog.params[cs1] = 4.0;
            prog.params[cs2 + k_idx] = 4.0;
            prog.params[bo2] = 4.0;
            prog.params[br2 + 3] = 4.0;
            prog.params[mo] = 4.0;
        } else if restart == 8 && n_scalar >= 1 {
            let k_idx = 2 + N_CONSTS;
            prog.params[io + 1] = 4.0;
            prog.params[co2 + 4] = 4.0;
            prog.params[cs1] = 4.0;
            prog.params[cs2 + k_idx] = 4.0;
            prog.params[bo2] = 4.0;
            prog.params[br2 + 3] = 4.0;
            prog.params[mo] = 4.0;
        } else {
            continue;
        }

        let code = SoftArrayCondAccumProgram {
            n_scalar,
            params: prog.params.clone(),
        }
        .discretize_and_emit(fn_name, scalar_names);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: "arr_gradient".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }

    None
}

/// Attempt gradient-based synthesis for array-input problems.
/// Returns None if the problem is not an array problem or synthesis fails.
pub(super) fn synthesize_array_gradient_core(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let (examples, n_scalar) = extract_arr_examples(problem)?;

    // Wall-clock bound for the whole array-gradient sweep (9 restarts x ~13 archs x
    // 600 steps). Like synthesize_scalar_inner, install a generous default deadline
    // capped by the caller's NSYNTH_SOLVE_BUDGET_MS — so the tighter of the two wins
    // and this can never run unbounded. Without it a hard array task hung the
    // interactive path for minutes. train_program_arr honors it (checked every 16
    // steps). Opt-in on the env for the tight case; the 60s default just trims the
    // pathological tail and is far above any run that actually converges.
    let sweep_secs: f32 = std::env::var("NSYNTH_SOLVE_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or(60.0, |ms| 60.0_f32.min(ms as f32 / 1000.0));
    // set_min, not set: never loosen an outer per-query budget (QuerySolveBudget) —
    // keeps a multi-attempt query bounded (see universal_array for the rationale).
    let _arr_deadline = crate::synthesis::common::TrainDeadline::set_min(
        std::time::Duration::from_secs_f32(sweep_secs),
    );
    let scalar_names: Vec<&str> = if n_scalar == 0 {
        vec![]
    } else if n_scalar == 1 {
        vec!["k"]
    } else {
        vec!["a", "b", "c", "d", "e", "f"]
            .iter()
            .take(n_scalar)
            .copied()
            .collect()
    };

    if let Some(result) = try_exact_array_warm_starts(problem, fn_name, n_scalar, &scalar_names) {
        return Some(result);
    }

    // Learned-bias replay: if any previously-recorded successful bias
    // discretizes to a verifying program on this I/O, return in ~milliseconds
    // instead of running the full native-array restart loop. Every success
    // below records a new bias, so this early-exit gets stronger over time.
    if let Some(result) = super::universal_array::try_universal_array_replay(
        problem,
        &examples,
        n_scalar,
        fn_name,
        &scalar_names,
    ) {
        return Some(result);
    }

    const N_ARR_STEPS: usize = 600;
    const N_ARR_RESTARTS: usize = 9;

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
                let code = SoftArrayAccumProgram {
                    n_scalar,
                    params: prog.params.clone(),
                }
                .discretize_and_emit(fn_name, &scalar_names);
                if verify_problem_code_strict(problem, &code).is_ok() {
                    return Some(SolveResult {
                        success: true,
                        code,
                        method: "arr_gradient".to_string(),
                        error: None,
                        metadata: DifferentiableMetadata::default(),
                    });
                }
            }
            if restart > 0 && restart != 3 {
                // Add noise to explore after checking the discrete warm start directly.
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 13000 + idx as u64) - 0.5) * 0.3;
                }
            }
            let ex = examples.clone();
            let sn = scalar_names.clone();
            let result = train_program_arr(
                prog.params,
                move |p, t| {
                    SoftArrayAccumProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                move |p, fn_n| {
                    SoftArrayAccumProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, &sn)
                },
                problem,
                fn_name,
                N_ARR_STEPS,
            );
            if result.is_some() {
                return result;
            }
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
            } else if restart == 8 && n_scalar >= 1 {
                // Count-greater-than: init=0, cmp item > k, op +, rhs = 1
                let k_idx = 2 + N_CONSTS; // first scalar arg in pool
                prog.params[io + 1] = 4.0; // init = const[0] = 0
                prog.params[co2 + 4] = 4.0; // cmp = >
                prog.params[cs1] = 4.0; // lhs = item (pool[0])
                prog.params[cs2 + k_idx] = 4.0; // rhs = k (pool[k_idx])
                prog.params[bo2] = 4.0; // op = +
                prog.params[br2 + 3] = 4.0; // rhs = const[1] = 1 (pool[3])
                prog.params[mo] = 4.0; // accumulate mode
            }
            // Try exact biased params first (no noise), then perturbed
            if restart > 0 {
                let code = SoftArrayCondAccumProgram {
                    n_scalar,
                    params: prog.params.clone(),
                }
                .discretize_and_emit(fn_name, &scalar_names);
                if verify_problem_code_strict(problem, &code).is_ok() {
                    return Some(SolveResult {
                        success: true,
                        code,
                        method: "arr_gradient".to_string(),
                        error: None,
                        metadata: DifferentiableMetadata::default(),
                    });
                }
                for (idx, p) in prog.params.iter_mut().enumerate() {
                    *p += (pseudo_rand(restart as u64 * 17000 + idx as u64) - 0.5) * 0.3;
                }
            }
            let ex = examples.clone();
            let sn = scalar_names.clone();
            let result = train_program_arr(
                prog.params,
                move |p, t| {
                    SoftArrayCondAccumProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                move |p, fn_n| {
                    SoftArrayCondAccumProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, &sn)
                },
                problem,
                fn_name,
                N_ARR_STEPS,
            );
            if result.is_some() {
                return result;
            }
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
                move |p, t| {
                    SoftArrayPairProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                move |p, fn_n| {
                    SoftArrayPairProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, &sn)
                },
                problem,
                fn_name,
                N_ARR_STEPS,
            );
            if result.is_some() {
                return result;
            }
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
                move |p, t| {
                    SoftPairwiseScanProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                move |p, fn_n| {
                    SoftPairwiseScanProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, &sn)
                },
                problem,
                fn_name,
                N_ARR_STEPS,
            );
            if result.is_some() {
                return result;
            }
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
                move |p, t| {
                    SoftArrayIndexGateProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .loss(&ex, t)
                },
                move |p, fn_n| {
                    SoftArrayIndexGateProgram {
                        n_scalar,
                        params: p.to_vec(),
                    }
                    .discretize_and_emit(fn_n, &sn)
                },
                problem,
                fn_name,
                N_ARR_STEPS,
            );
            if result.is_some() {
                return result;
            }
        }
    }

    universal_array::synthesize_universal_array_fallback(
        problem,
        &examples,
        n_scalar,
        fn_name,
        &scalar_names,
    )
}

/// Training loop for array programs. Same as train_program but the emit_fn
/// doesn't take param_names (uses scalar_names internally).
pub(super) fn train_program_arr<F, G>(
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
    let wrapped_emit = |p: &[f32], fn_n: &str, _pn: &[&str]| -> String { emit_fn(p, fn_n) };

    // Try the initial params directly
    if let Some(result) = try_emit_verify_arr(
        &initial_params,
        &wrapped_emit,
        problem,
        fn_name,
        param_names,
    ) {
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
        // Cooperative wall-clock bound: honor an installed TrainDeadline (checked every
        // 16 steps, like train_program) so a doomed array-gradient sweep cannot run
        // unbounded. Without this the array core spun ~minutes on a hard array task
        // ("move all zeroes to the end") and hung the interactive handle_query path —
        // no deadline was ever consulted here. No-op when no deadline is set.
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
            if let Some(result) =
                try_emit_verify_arr(&params, &wrapped_emit, problem, fn_name, param_names)
            {
                return Some(result);
            }
            if best_loss < loss {
                if let Some(result) =
                    try_emit_verify_arr(&best_params, &wrapped_emit, problem, fn_name, param_names)
                {
                    return Some(result);
                }
            }
        }
        let grads = fd_grad(&params, &loss_fn, temp);
        opt.step(&mut params, &grads);
    }

    if let Some(result) = try_emit_verify_arr(&params, &wrapped_emit, problem, fn_name, param_names)
    {
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
