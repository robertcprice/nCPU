//! Distillation pipeline (verify + bundle). Reads a JSONL of candidate ops — each
//! `{"name", "arity", "mog", "examples":[{"in":[...],"out":...}]}` where the examples are
//! FRESH (distinct from any the teacher saw) — fresh-verifies every candidate against its
//! examples, and appends only the passers to a learned-op store (dedup by program text).
//!
//! The Mog is authored by a strong TEACHER (a capable model via the OpenAI-compatible
//! endpoint, or a human/agent). This tool is the disciplined gate: NOTHING enters the store
//! that does not reproduce its fresh examples, so a bundled op is correct-by-verification,
//! never-wrong-safe at inference (routed by name, re-verified per query).
//!
//!   distill <candidates.jsonl> [--store bench/harvested_ops.jsonl]
use mog_synth::benchmark::{Example, Value};
use std::io::Write;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: distill <candidates.jsonl> [--store <path>]");
        std::process::exit(2);
    };
    let store = args
        .iter()
        .position(|a| a == "--store")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "bench/harvested_ops.jsonl".to_string());

    let existing: std::collections::HashSet<String> = std::fs::read_to_string(&store)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v.get("mog").and_then(|m| m.as_str()).map(str::to_string))
        .collect();

    let text = std::fs::read_to_string(path).expect("read candidates");
    let (mut total, mut verified, mut dup, mut bundled) = (0, 0, 0, 0);
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&store).expect("open store");
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(cand): Result<serde_json::Value, _> = serde_json::from_str(line) else {
            continue;
        };
        let name = cand.get("name").and_then(|v| v.as_str()).unwrap_or("?").to_string();
        let arity = cand.get("arity").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mog = cand.get("mog").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let exs = parse_examples(&cand);
        total += 1;
        if exs.len() < 4 {
            println!("SKIP   {name:<32} (<4 fresh examples — cannot verify)");
            continue;
        }
        if !mog_synth::runtime::code_reproduces_examples(&mog, &exs) {
            println!("REJECT {name:<32} (fails its fresh examples)");
            continue;
        }
        verified += 1;
        if existing.contains(&mog) {
            dup += 1;
            println!("DUP    {name:<32} (already in store)");
            continue;
        }
        let rec = serde_json::json!({"name": name, "arity": arity, "mog": mog});
        writeln!(out, "{rec}").expect("write store");
        bundled += 1;
        println!("BUNDLE {name:<32} ({} fresh examples) -> {store}", exs.len());
    }
    println!("\ndistill: {total} candidates | {verified} verified | {bundled} bundled | {dup} dup");
    if verified < total {
        std::process::exit(1);
    }
}
