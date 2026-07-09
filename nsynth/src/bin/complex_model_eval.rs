//! Complex-algorithm capability WITH the gated model tier. For each task, hold out the
//! last 2 examples (the model never sees them), run answer_with_proposer using the served
//! local model as the proposer, then verify the returned code reproduces ALL examples
//! incl. the held-out. Reports solved / refused / WRONG. WRONG must stay 0 — the model
//! only widens REACH; the verify + distinguishing gates keep never-wrong.
//!
//!   NSYNTH_LOCAL_LLM_URL=http://127.0.0.1:8767/v1/chat/completions \
//!   NSYNTH_LOCAL_LLM_MODEL=mlx-community/Qwen3.5-9B-4bit \
//!   complex_model_eval bench/complex_algorithms.jsonl
use mog_synth::benchmark::{Example, Value};
use mog_synth::local_llm::propose_program;
use mog_synth::verified_nl_router::{answer_with_proposer, Answer};
use std::io::BufRead;

fn json_to_value(v: &serde_json::Value) -> Option<Value> {
    if let Some(b) = v.as_bool() {
        return Some(Value::Bool(b));
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

fn parse_examples(task: &serde_json::Value) -> Vec<Example> {
    let mut exs = Vec::new();
    if let Some(rows) = task.get("examples").and_then(|v| v.as_array()) {
        for row in rows {
            let (Some(ins), Some(out)) = (
                row.get("in").and_then(|v| v.as_array()),
                row.get("out").and_then(json_to_value),
            ) else {
                continue;
            };
            let Some(inputs) = ins.iter().map(json_to_value).collect::<Option<Vec<Value>>>() else {
                continue;
            };
            exs.push(Example { inputs, expected: out });
        }
    }
    exs
}

fn code_of(a: &Answer) -> Option<&str> {
    match a {
        Answer::Library { code, .. }
        | Answer::Composition { code }
        | Answer::Synthesized { code, .. }
        | Answer::Proposed { code, .. } => Some(code),
        Answer::Refused => None,
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: complex_model_eval <battery.jsonl>");
    let use_model = std::env::var("NSYNTH_LOCAL_LLM_URL").is_ok();
    eprintln!("[complex_model_eval] model_tier={} battery={path}", if use_model { "ON" } else { "OFF (set NSYNTH_LOCAL_LLM_URL)" });

    // Model-backed proposer: the served local model via propose_program (same call the
    // flywheel uses). answer_with_proposer only reaches it after the symbolic tiers refuse.
    let proposer = |req: &str, _seed: &[Example], prior: Option<(&str, &str)>| propose_program(req, prior, 0.2);

    let file = std::fs::File::open(&path).expect("open battery");
    let (mut total, mut solved, mut refused, mut wrong, mut by_model) = (0, 0, 0, 0, 0);
    for line in std::io::BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        let line = line.trim();
        if line.is_empty() { continue; }
        let Ok(task): Result<serde_json::Value, _> = serde_json::from_str(line) else { continue };
        let text = task.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let all = parse_examples(&task);
        if text.is_empty() || all.len() < 4 { continue; }
        total += 1;
        let id = task.get("fn").and_then(|v| v.as_str()).unwrap_or("?").to_string();

        // Hold out the last N (the model/search never sees them — generalization oracle).
        let holdout: usize = std::env::var("NSYNTH_HOLDOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let holdout = holdout.min(all.len().saturating_sub(1));
        let seed = &all[..all.len() - holdout];
        let held = &all[all.len() - holdout..];

        let a = answer_with_proposer(&text, seed, if use_model { Some(&proposer) } else { None });
        let method = match &a {
            Answer::Library { name, .. } => format!("library:{name}"),
            Answer::Composition { .. } => "composition".into(),
            Answer::Synthesized { method, .. } => format!("synth:{method}"),
            Answer::Proposed { method, .. } => format!("MODEL:{method}"),
            Answer::Refused => "refused".into(),
        };
        match code_of(&a) {
            None => { refused += 1; println!("REFUSED  {id:<18} {text}"); }
            Some(code) => {
                let ok = mog_synth::runtime::code_reproduces_examples(code, held)
                    && mog_synth::runtime::code_reproduces_examples(code, seed);
                if ok {
                    solved += 1;
                    if method.starts_with("MODEL:") { by_model += 1; }
                    println!("SOLVED   {id:<18} [{method}]");
                } else {
                    wrong += 1;
                    println!("WRONG    {id:<18} [{method}]  {text}\n----code----\n{code}\n------------");
                }
            }
        }
    }
    println!("\nCOMPLEX-ALGO (model tier {}): {solved}/{total} solved ({} via MODEL), {refused} refused, WRONG {wrong}",
        if use_model { "ON" } else { "OFF" }, by_model);
    if wrong > 0 { std::process::exit(1); }
}
