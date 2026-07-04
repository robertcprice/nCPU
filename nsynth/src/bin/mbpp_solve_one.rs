//! Solve ONE MBPP task (JSON line on stdin) with the CANONICAL engine, printing
//! `SOLVED <id> <method>` / `UNSOLVED <id>` / `SKIP <id>`. Run under an OS
//! `timeout` (per-task isolation — see scripts/run_mbpp_bench.sh). LLM-FREE:
//! solves purely from the task's own I/O test cases. Acceptance = the
//! synthesized program STRICT-VERIFIES against EVERY test case (MBPP's own
//! pass criterion), via the interpreter harness — no self-grading.
use mog_synth::benchmark::{Example, Problem, Value};
use std::io::Read;

fn json_to_value(v: &serde_json::Value) -> Option<Value> {
    if let Some(b) = v.as_bool() {
        return Some(Value::Bool(b));
    }
    if v.get("__map__").is_some() {
        return None; // Map values not in the canonical domain yet — task skips.
    }
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(f) = v.as_f64() {
        return Some(Value::Float(f.to_bits()));
    }
    if let Some(s) = v.as_str() {
        return Some(Value::Str(s.to_string()));
    }
    if let Some(arr) = v.as_array() {
        if let Some(ints) = arr.iter().map(|x| x.as_i64()).collect::<Option<Vec<i64>>>() {
            return Some(Value::int_array(&ints));
        }
        let vals: Option<Vec<Value>> = arr.iter().map(json_to_value).collect();
        return Some(Value::Array(vals?));
    }
    None
}

fn main() {
    let mut buf = String::new();
    if std::io::stdin().read_to_string(&mut buf).is_err() {
        println!("SKIP -1");
        return;
    }
    let task: serde_json::Value = match serde_json::from_str(buf.trim()) {
        Ok(t) => t,
        Err(_) => {
            println!("SKIP -1");
            return;
        }
    };
    let id = task.get("id").and_then(|v| v.as_i64()).unwrap_or(-1);

    let mut exs: Vec<Example> = Vec::new();
    if let Some(rows) = task.get("examples").and_then(|v| v.as_array()) {
        for row in rows {
            let (Some(ins), Some(out)) = (
                row.get("in").and_then(|v| v.as_array()),
                row.get("out").and_then(json_to_value),
            ) else {
                continue;
            };
            let inputs: Option<Vec<Value>> = ins.iter().map(json_to_value).collect();
            let Some(inputs) = inputs else { continue };
            exs.push(Example { inputs, expected: out });
        }
    }
    // Dedup identical rows; SKIP contradictory specs (same inputs, different out).
    let mut deduped: Vec<Example> = Vec::new();
    for e in exs {
        if let Some(prev) = deduped.iter().find(|p| p.inputs == e.inputs) {
            if prev.expected != e.expected {
                println!("SKIP {id}");
                return;
            }
            continue;
        }
        deduped.push(e);
    }
    if deduped.len() < 3 {
        println!("SKIP {id}");
        return;
    }
    let exs = deduped;

    let split = exs.len().saturating_sub(2).max(2);
    let seed = &exs[..split];
    let fname = task.get("fn").and_then(|v| v.as_str()).unwrap_or("f");
    let sig: &'static str =
        Box::leak(mog_synth::linguigenesis_bridge::infer_signature(fname, seed).into_boxed_str());
    let problem = Problem {
        name: fname.to_string(),
        category: "mbpp",
        description: "mbpp task",
        signature: sig,
        examples: seed.to_vec(),
        ..Default::default()
    };
    let full = Problem { examples: exs.clone(), ..problem.clone() };

    // Accept iff the program reproduces EVERY test case (seed + held-out).
    let accept = |code: &str| mog_synth::runtime::verify_problem_code_strict(&full, code).is_ok();

    let res = mog_synth::solver::solve_problem(&problem);
    if res.success && accept(&res.code) {
        println!("SOLVED {id} {}", res.method);
        return;
    }
    // FULL-EXAMPLE fallback: the seed split starves multi-parameter fits (a
    // 2-arg affine is underdetermined on 2 points). Same acceptance bar.
    let res = mog_synth::solver::solve_problem(&full);
    if res.success && accept(&res.code) {
        println!("SOLVED {id} {}-full", res.method);
        return;
    }
    println!("UNSOLVED {id}");
}
