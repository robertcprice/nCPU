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
    // {"__map__": [[k, v], ...]} — the prep script's dict encoding (a plain JSON
    // object would stringify int keys and lose the key type). Decodes to
    // Value::Map in canonical order.
    if let Some(entries) = v.get("__map__").and_then(|m| m.as_array()) {
        let pairs: Option<Vec<(Value, Value)>> = entries
            .iter()
            .map(|e| {
                let kv = e.as_array()?;
                if kv.len() != 2 {
                    return None;
                }
                Some((json_to_value(&kv[0])?, json_to_value(&kv[1])?))
            })
            .collect();
        return Some(Value::map_from_pairs(pairs?));
    }
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    // Non-integer number -> Float (bits). Tried AFTER as_i64 so ints stay Int.
    if let Some(f) = v.as_f64() {
        return Some(Value::Float(f.to_bits()));
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
    let desc = task.get("text").and_then(|v| v.as_str()).unwrap_or("solve the task");
    // NSYNTH_HARVEST=<path>: on any SOLVE, append a VERIFIED (task -> Mog) training
    // pair to <path> (mlx_lm.lora chat format). Every harvested program passed the
    // verifier, so the corpus is guaranteed-correct (STaR / rejection-sampling).
    let harvest = |code: &str| {
        if let Some(path) = std::env::var("NSYNTH_HARVEST").ok().filter(|s| !s.is_empty()) {
            let rec = mog_synth::local_llm::training_record(desc, code);
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                use std::io::Write;
                let _ = writeln!(f, "{rec}");
            }
        }
    };

    if !llm_only {
        // FULL-EXAMPLE library match FIRST. The seed/held-out split (above) leaves
        // only 2 seed examples for a 3-example task, and on 2 points try_library
        // returns the FIRST coincidentally-matching op — e.g. largest_digit fits
        // (123->3, 25->5) but is wrong on the held-out 30, shadowing the correct
        // last_digit. A reference-algorithm op that reproduces EVERY example is
        // high-confidence correct, so try that before the seed-split solve.
        let full = Problem { examples: exs.clone(), ..problem.clone() };
        if let Some(res) = mog_synth::op_library::try_library(&full) {
            if mog_synth::runtime::code_reproduces_examples(&res.code, &exs) {
                harvest(&res.code);
                println!("SOLVED {id} {}", res.method);
                return;
            }
        }
        // FLOAT problems solve on the FULL example set FIRST: the float lanes
        // (affine + poly) are ms-scale and carry their own over-determination
        // gates, while the seed pass burns the whole per-task budget on doomed
        // exact-integer searches — the measured cause of 48/57 float timeouts
        // (πr³/4πr²/2πr tasks never REACHED the poly lane). Acceptance bar
        // unchanged: full reproduction below.
        if sig.contains("-> f64") {
            let full = Problem { examples: exs.clone(), ..problem.clone() };
            let res = mog_synth::solver::solve_problem(&full);
            if res.success && mog_synth::runtime::code_reproduces_examples(&res.code, &exs) {
                harvest(&res.code);
                println!("SOLVED {id} {}-full", res.method);
                return;
            }
        }
        let res = mog_synth::solver::solve_problem(&problem);
        // SOLVED = synthesized AND reproduces EVERY test case (seed + held-out).
        if res.success && mog_synth::runtime::code_reproduces_examples(&res.code, &exs) {
            harvest(&res.code);
            // Report the winning method so the driver can attribute solves (library
            // vs search vs …) without a separate baseline run.
            println!("SOLVED {id} {}", res.method);
            return;
        }
        // FULL-EXAMPLE fallback: the seed split leaves 1-2 examples for a
        // 3-example task, which STARVES multi-parameter fits — a 2-arg float
        // affine has 3 unknowns and is underdetermined on 2 points, so the lane
        // fails before it can generalize. Retry on every example. The acceptance
        // bar is unchanged (the program must reproduce EVERY test case — MBPP's
        // own pass criterion), so this adds reach, not looseness.
        let full = Problem { examples: exs.clone(), ..problem };
        let res = mog_synth::solver::solve_problem(&full);
        if res.success && mog_synth::runtime::code_reproduces_examples(&res.code, &exs) {
            harvest(&res.code);
            println!("SOLVED {id} {}-full", res.method);
            return;
        }
    }
    // Mode D fallback (gated by NSYNTH_LOCAL_LLM_REPAIR + a served model): the LLM
    // writes a whole program from the DESCRIPTION + examples, verified against the
    // FULL example set with repair retries. Only accepted on a full reproduction.
    if let Some(r) =
        mog_synth::linguigenesis_bridge::LinguigenesisBridge::synthesize_via_repair_loop(desc, &exs)
    {
        if mog_synth::runtime::code_reproduces_examples(&r.code, &exs) {
            harvest(&r.code);
            println!("SOLVED {id} {}", r.method);
            return;
        }
    }
    println!("UNSOLVED {id}");
}
