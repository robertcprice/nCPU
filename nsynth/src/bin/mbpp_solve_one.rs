//! Solve ONE MBPP task (read as a single JSON line on stdin), print
//! `SOLVED <id>` / `UNSOLVED <id>` / `SKIP <id>`. Run under an OS `timeout` so a
//! pathological search is killed cleanly (per-task isolation for the benchmark
//! driver `scripts/run_mbpp_bench.sh`). LLM-FREE: solves purely from the task's
//! own I/O test cases (programming-by-example over a real model benchmark).
use mog_synth::benchmark::{dedup_consistent_examples, Example, Problem, Value};
use std::io::Read;

fn json_to_value(v: &serde_json::Value) -> Option<Value> {
    if let Some(b) = v.as_bool() {
        return Some(Value::Bool(b));
    }
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(s) = v.as_str() {
        return Some(Value::Str(s.to_string()));
    }
    if let Some(arr) = v.as_array() {
        // All-int -> the engine's dedicated int-array fast path.
        if let Some(ints) = arr.iter().map(|x| x.as_i64()).collect::<Option<Vec<i64>>>() {
            return Some(Value::int_array(&ints));
        }
        // Else a general array (strings, nested) -> recurse. Any element that
        // can't map (e.g. a float) fails the whole value -> the task is skipped.
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
    let Some(exs) = dedup_consistent_examples(&exs).filter(|e| e.len() >= 3) else {
        println!("SKIP {id}");
        return;
    };

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
    // NSYNTH_LLM_ONLY skips the LLM-free engine entirely -> straight to the repair
    // loop, so an ABLATION can isolate the model's own ability (LLM_ONLY + TRIES=1 =
    // raw single-shot) from what the engine + repair add.
    let llm_only = std::env::var("NSYNTH_LLM_ONLY").ok().filter(|s| !s.is_empty()).is_some();
    if !llm_only {
        let res = mog_synth::solver::solve_problem(&problem);
        // SOLVED = synthesized AND reproduces EVERY test case (seed + held-out).
        if res.success && mog_synth::runtime::code_reproduces_examples(&res.code, &exs) {
            // Report the winning method so the driver can attribute solves (library
            // vs search vs …) without a separate baseline run.
            println!("SOLVED {id} {}", res.method);
            return;
        }
    }
    // Mode D fallback (gated by NSYNTH_LOCAL_LLM_REPAIR + a served model): the LLM
    // writes a whole program from the DESCRIPTION + examples, verified against the
    // FULL example set with repair retries. Only accepted on a full reproduction.
    let desc = task.get("text").and_then(|v| v.as_str()).unwrap_or("solve the task");
    if let Some(r) =
        mog_synth::linguigenesis_bridge::LinguigenesisBridge::synthesize_via_repair_loop(desc, &exs)
    {
        if mog_synth::runtime::code_reproduces_examples(&r.code, &exs) {
            println!("SOLVED {id} {}", r.method);
            return;
        }
    }
    println!("UNSOLVED {id}");
}
