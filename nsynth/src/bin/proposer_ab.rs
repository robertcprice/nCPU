//! A/B harness for the KG semantic op proposer (the whack-a-mole cure).
//!
//! Feeds a battery of NOVEL-PHRASING tasks (prompt + examples) through the
//! never-wrong front door `verified_nl_router::answer`. For each task it HOLDS OUT
//! the last example, asks `answer` to solve from the rest, then re-checks the
//! returned code against the held-out example — so a confidently-WRONG answer is
//! detected, not hidden. Reports solved / refused / wrong.
//!
//! Run it twice on the same binary to measure the proposer's recall lift:
//!   NSYNTH_SEMANTIC_PROPOSER=0 proposer_ab battery.jsonl   # baseline (token+synonym)
//!   proposer_ab battery.jsonl                              # + KG semantic proposer
//! The delta in `solved` is recall gained; `wrong` MUST stay 0 in both (never-wrong).
use mog_synth::benchmark::{Example, Value};
use mog_synth::verified_nl_router::{answer, Answer};
use std::io::BufRead;

fn json_to_value(v: &serde_json::Value) -> Option<Value> {
    if let Some(b) = v.as_bool() {
        return Some(Value::Bool(b));
    }
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
    let path = std::env::args().nth(1).expect("usage: proposer_ab <battery.jsonl>");
    let on = std::env::var("NSYNTH_SEMANTIC_PROPOSER").map_or(true, |v| v != "0");
    let file = std::fs::File::open(&path).expect("open battery");
    let mut total = 0usize;
    let mut solved = 0usize;
    let mut refused = 0usize;
    let mut wrong = 0usize;
    let mut solved_ids: Vec<String> = Vec::new();

    eprintln!("[proposer_ab] semantic_proposer={} battery={path}", if on { "ON" } else { "OFF" });
    for line in std::io::BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(task): Result<serde_json::Value, _> = serde_json::from_str(line) else { continue };
        let text = task.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let all = parse_examples(&task);
        if text.is_empty() || all.len() < 3 {
            continue; // need a prompt and enough examples for the behavior gate
        }
        total += 1;
        let label = task
            .get("id")
            .and_then(|v| v.as_i64())
            .map(|i| i.to_string())
            .unwrap_or_else(|| text.chars().take(24).collect());

        // Hold out the last example when there are >=4 (leave >=3 for the gate);
        // otherwise verify only against the training set (can't measure wrong there).
        let (train, held): (&[Example], Option<&Example>) = if all.len() >= 4 {
            (&all[..all.len() - 1], all.last())
        } else {
            (&all[..], None)
        };

        let a = answer(&text, train);
        match code_of(&a) {
            None => refused += 1,
            Some(code) => {
                // Never-wrong check: the returned code must also reproduce the held-out
                // example (and, trivially, the training set). A miss is a real violation.
                let ok = match held {
                    Some(h) => mog_synth::runtime::code_reproduces_examples(code, std::slice::from_ref(h)),
                    None => true,
                };
                if ok {
                    solved += 1;
                    solved_ids.push(label);
                } else {
                    wrong += 1;
                    println!("WRONG  {label}  \"{text}\"");
                }
            }
        }
    }
    println!(
        "proposer={} total={total} solved={solved} refused={refused} WRONG={wrong}  ({:.0}% solved)",
        if on { "ON" } else { "OFF" },
        100.0 * solved as f32 / total.max(1) as f32
    );
    println!("solved: {}", solved_ids.join(", "));
    if wrong > 0 {
        std::process::exit(1);
    }
}
