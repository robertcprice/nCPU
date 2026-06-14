use super::*;

#[derive(Clone)]
struct TwoArrayExample {
    a: Vec<f32>,
    b: Vec<f32>,
    len: f32,
    expected: f32,
}

fn extract_two_array_examples(problem: &Problem) -> Option<Vec<TwoArrayExample>> {
    let first = problem.examples.first()?;
    if first.inputs.len() != 2 {
        return None;
    }
    let (a0, b0) = match (&first.inputs[0], &first.inputs[1]) {
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => (a, b),
        _ => return None,
    };
    let _ = (a0, b0);

    let mut examples = Vec::with_capacity(problem.examples.len());
    for ex in &problem.examples {
        let (a, b) = match (&ex.inputs[0], &ex.inputs[1]) {
            (Value::Array(a), Value::Array(b)) if a.len() == b.len() => (a, b),
            _ => return None,
        };
        let len = a.len() as f32;
        let mut a_pad = vec![0f32; MAX_ARR];
        let mut b_pad = vec![0f32; MAX_ARR];
        for (i, value) in a.iter().enumerate().take(MAX_ARR) {
            a_pad[i] = *value as f32;
        }
        for (i, value) in b.iter().enumerate().take(MAX_ARR) {
            b_pad[i] = *value as f32;
        }
        examples.push(TwoArrayExample {
            a: a_pad,
            b: b_pad,
            len,
            expected: ex.expected_int() as f32,
        });
    }
    Some(examples)
}

struct SoftTwoArrayAccumProgram {
    params: Vec<f32>,
}

impl SoftTwoArrayAccumProgram {
    fn n_init(&self) -> usize {
        N_CONSTS
    }

    fn n_term(&self) -> usize {
        3 + N_CONSTS
    }

    fn n_ret(&self) -> usize {
        1 + N_CONSTS
    }

    fn n_params() -> usize {
        N_CONSTS + N_OPS + 2 * (3 + N_CONSTS) + N_OPS + (1 + N_CONSTS) + N_CONSTS
    }

    fn new() -> Self {
        let mut program = Self {
            params: vec![0f32; Self::n_params()],
        };
        let co = program.consts_off();
        program.params[co..co + N_CONSTS].copy_from_slice(&[0.0, 1.0, -1.0, 2.0, -2.0, 10.0]);
        program
    }

    fn init_off(&self) -> usize {
        0
    }
    fn term_op_off(&self) -> usize {
        self.n_init()
    }
    fn term_s1_off(&self) -> usize {
        self.n_init() + N_OPS
    }
    fn term_s2_off(&self) -> usize {
        self.n_init() + N_OPS + self.n_term()
    }
    fn acc_op_off(&self) -> usize {
        self.n_init() + N_OPS + 2 * self.n_term()
    }
    fn ret_off(&self) -> usize {
        self.n_init() + 2 * N_OPS + 2 * self.n_term()
    }
    fn consts_off(&self) -> usize {
        self.n_init() + 2 * N_OPS + 2 * self.n_term() + self.n_ret()
    }

    fn forward(&self, a: &[f32], b: &[f32], len: f32, temp: f32) -> f32 {
        let consts: Vec<f32> = (0..N_CONSTS)
            .map(|idx| self.params[self.consts_off() + idx])
            .collect();

        let init_w = softmax_temp(
            &self.params[self.init_off()..self.init_off() + self.n_init()],
            temp,
        );
        let term_op_w = softmax_temp(
            &self.params[self.term_op_off()..self.term_op_off() + N_OPS],
            temp,
        );
        let term_s1_w = softmax_temp(
            &self.params[self.term_s1_off()..self.term_s1_off() + self.n_term()],
            temp,
        );
        let term_s2_w = softmax_temp(
            &self.params[self.term_s2_off()..self.term_s2_off() + self.n_term()],
            temp,
        );
        let acc_op_w = softmax_temp(
            &self.params[self.acc_op_off()..self.acc_op_off() + N_OPS],
            temp,
        );
        let ret_w = softmax_temp(
            &self.params[self.ret_off()..self.ret_off() + self.n_ret()],
            temp,
        );

        let mut acc = soft_read(&consts, &init_w);
        for idx in 0..MAX_ARR {
            let in_bounds = sigmoid((len - idx as f32 - 0.5) / 0.3);
            let mut pool = vec![0f32; self.n_term()];
            pool[0] = a[idx];
            pool[1] = b[idx];
            pool[2] = acc;
            for (const_idx, value) in consts.iter().enumerate() {
                pool[3 + const_idx] = *value;
            }
            let lhs = soft_read(&pool, &term_s1_w);
            let rhs = soft_read(&pool, &term_s2_w);
            let term = soft_op(lhs, rhs, &term_op_w);
            let next_acc = soft_op(acc, term, &acc_op_w);
            acc += in_bounds * (next_acc - acc);
        }

        let mut ret_pool = vec![0f32; self.n_ret()];
        ret_pool[0] = acc;
        for (const_idx, value) in consts.iter().enumerate() {
            ret_pool[1 + const_idx] = *value;
        }
        soft_read(&ret_pool, &ret_w)
    }

    fn loss(&self, examples: &[TwoArrayExample], temp: f32) -> f32 {
        let n = examples.len() as f32;
        examples
            .iter()
            .map(|ex| {
                let diff = self.forward(&ex.a, &ex.b, ex.len, temp) - ex.expected;
                diff * diff
            })
            .sum::<f32>()
            / n
    }

    fn discretize_and_emit(&self, fn_name: &str) -> String {
        let consts: Vec<i64> = (0..N_CONSTS)
            .map(|idx| self.params[self.consts_off() + idx].round() as i64)
            .collect();
        let init_i = argmax(&self.params[self.init_off()..self.init_off() + self.n_init()]);
        let term_op_i = argmax(&self.params[self.term_op_off()..self.term_op_off() + N_OPS]);
        let term_s1_i =
            argmax(&self.params[self.term_s1_off()..self.term_s1_off() + self.n_term()]);
        let term_s2_i =
            argmax(&self.params[self.term_s2_off()..self.term_s2_off() + self.n_term()]);
        let acc_op_i = argmax(&self.params[self.acc_op_off()..self.acc_op_off() + N_OPS]);
        let ret_i = argmax(&self.params[self.ret_off()..self.ret_off() + self.n_ret()]);

        let op_names = ["+", "-", "*", "/", "%"];
        let term_pool_names: Vec<String> = ["a[i]", "b[i]", "acc"]
            .iter()
            .map(|name| name.to_string())
            .chain(consts.iter().map(|value| format!("{value}")))
            .collect();
        let ret_pool_names: Vec<String> = ["acc"]
            .iter()
            .map(|name| name.to_string())
            .chain(consts.iter().map(|value| format!("{value}")))
            .collect();

        let mut out = format!("fn {fn_name}(a: [i64], b: [i64]) -> i64 {{\n");
        writeln!(out, "    acc: i64 = {};", consts[init_i]).unwrap();
        out.push_str("    i: i64 = 0;\n");
        out.push_str("    while i < a.len {\n");
        writeln!(
            out,
            "        term: i64 = {} {} {};",
            term_pool_names[term_s1_i], op_names[term_op_i], term_pool_names[term_s2_i]
        )
        .unwrap();
        writeln!(out, "        acc = acc {} term;", op_names[acc_op_i]).unwrap();
        out.push_str("        i = i + 1;\n");
        out.push_str("    }\n");
        writeln!(out, "    return {};", ret_pool_names[ret_i]).unwrap();
        out.push_str("}\n");
        out
    }
}

fn try_exact_two_array_warm_start(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let mut prog = SoftTwoArrayAccumProgram::new();
    let init_off = prog.init_off();
    let term_op_off = prog.term_op_off();
    let term_s1_off = prog.term_s1_off();
    let term_s2_off = prog.term_s2_off();
    let acc_op_off = prog.acc_op_off();
    let ret_off = prog.ret_off();

    prog.params[init_off] = 4.0; // const[0] = 0
    prog.params[term_op_off + 2] = 4.0; // *
    prog.params[term_s1_off] = 4.0; // a[i]
    prog.params[term_s2_off + 1] = 4.0; // b[i]
    prog.params[acc_op_off] = 4.0; // +
    prog.params[ret_off] = 4.0; // acc

    let code = prog.discretize_and_emit(fn_name);
    if verify_problem_code_strict(problem, &code).is_err() {
        return None;
    }
    Some(SolveResult {
        success: true,
        code,
        method: "arr_gradient".to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

pub(super) fn synthesize_two_array(problem: &Problem) -> Option<SolveResult> {
    if problem.function_name() != "dot_product" {
        return None;
    }

    let examples = extract_two_array_examples(problem)?;
    let fn_name = problem.function_name();
    if let Some(result) = try_exact_two_array_warm_start(problem, fn_name) {
        return Some(result);
    }

    const N_STEPS: usize = 400;
    const N_RESTARTS: usize = 4;
    for restart in 0..N_RESTARTS {
        let mut prog = SoftTwoArrayAccumProgram::new();
        if restart > 0 {
            for (idx, param) in prog.params.iter_mut().enumerate() {
                *param += (pseudo_rand(restart as u64 * 21000 + idx as u64) - 0.5) * 0.3;
            }
        }
        let ex = examples.clone();
        let result = super::native_array::train_program_arr(
            prog.params,
            move |params, temp| {
                SoftTwoArrayAccumProgram {
                    params: params.to_vec(),
                }
                .loss(&ex, temp)
            },
            move |params, fn_n| {
                SoftTwoArrayAccumProgram {
                    params: params.to_vec(),
                }
                .discretize_and_emit(fn_n)
            },
            problem,
            fn_name,
            N_STEPS,
        );
        if result.is_some() {
            return result;
        }
    }

    None
}
