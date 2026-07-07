//! Capability gap report: run a categorized battery of NL specs through the SAME
//! classifier as `nl_diag` (hand ONLY the `text` to `handle_query`, verify the
//! result against the hidden example asserts), then group SOLVED / WRONG / REFUSED
//! by category + hardness so the TRUE deterministic ceiling is visible.
//!
//!   cargo run --release --bin capability_report -- bench/failing_cases_core.jsonl [out.json]
//!
//! This is the measurement gate for the non-model-first plan: build only what the
//! data says is missing, and after each proposer lever re-run to confirm SOLVED
//! rises while WRONG stays 0 (no false accepts).

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy};
use mog_synth::benchmark::{dedup_consistent_examples, Example, Value};
use std::collections::BTreeMap;

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

/// One of SOLVED / WRONG / REFUSED / TENTATIVE_OK / TENTATIVE_MISS / SKIP —
/// identical taxonomy to `nl_diag`, computed in-process for the whole battery.
fn classify(id: i64, text: &str, exs: &[Example]) -> (&'static str, String) {
    let root = std::env::temp_dir().join(format!("caprep_{id}_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&root);
    let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r = session.handle_query(text);
    let _ = std::fs::remove_dir_all(&root);

    let method = r.synthesis_method.clone().unwrap_or_else(|| format!("{:?}", r.route));
    let reproduces = mog_synth::runtime::code_reproduces_examples(&r.response, exs);
    let tentative = method.contains(":tentative");
    let cat = if !r.success {
        "REFUSED"
    } else if tentative {
        if reproduces { "TENTATIVE_OK" } else { "TENTATIVE_MISS" }
    } else if reproduces {
        "SOLVED"
    } else {
        "WRONG"
    };
    (cat, method)
}

#[derive(Default, Clone)]
struct Tally {
    solved: usize,
    wrong: usize,
    refused: usize,
    tent_ok: usize,
    tent_miss: usize,
    skip: usize,
}
impl Tally {
    fn add(&mut self, outcome: &str) {
        match outcome {
            "SOLVED" => self.solved += 1,
            "WRONG" => self.wrong += 1,
            "REFUSED" => self.refused += 1,
            "TENTATIVE_OK" => self.tent_ok += 1,
            "TENTATIVE_MISS" => self.tent_miss += 1,
            _ => self.skip += 1,
        }
    }
    fn total(&self) -> usize {
        self.solved + self.wrong + self.refused + self.tent_ok + self.tent_miss + self.skip
    }
    fn row(&self, label: &str) -> String {
        format!(
            "| {label:<14} | {:>6} | {:>5} | {:>7} | {:>5} | {:>4} | {:>4} |",
            self.solved, self.wrong, self.refused, self.tent_ok, self.tent_miss, self.skip
        )
    }
}

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: capability_report <battery.jsonl> [out.json]");
            std::process::exit(2);
        }
    };
    let out_json = std::env::args().nth(2);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        }
    };

    let mut by_category: BTreeMap<String, Tally> = BTreeMap::new();
    let mut by_hardness: BTreeMap<String, Tally> = BTreeMap::new();
    let mut overall = Tally::default();
    let mut records: Vec<serde_json::Value> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let task: serde_json::Value = match serde_json::from_str(line) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let id = task.get("id").and_then(|v| v.as_i64()).unwrap_or(-1);
        let nl = task.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
        let category = task.get("category").and_then(|v| v.as_str()).unwrap_or("uncategorized").to_string();
        let hardness = task.get("hardness").and_then(|v| v.as_str()).unwrap_or("?").to_string();

        let mut exs: Vec<Example> = Vec::new();
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
        let (outcome, method) = match dedup_consistent_examples(&exs).filter(|e| e.len() >= 3) {
            Some(exs) => classify(id, nl, &exs),
            None => ("SKIP", "malformed".to_string()),
        };

        println!("CAT {id:<4} {:<8} {outcome:<14} {category:<12} {hardness:<7} {method}", "");
        by_category.entry(category.clone()).or_default().add(outcome);
        by_hardness.entry(hardness.clone()).or_default().add(outcome);
        overall.add(outcome);
        records.push(serde_json::json!({
            "id": id, "category": category, "hardness": hardness,
            "outcome": outcome, "method": method, "text": nl,
        }));
    }

    let header = "\n| group          | SOLVED | WRONG | REFUSED | T_OK | MISS | SKIP |\n|----------------|--------|-------|---------|------|------|------|";
    println!("\n## By category{header}");
    for (cat, t) in &by_category {
        println!("{}", t.row(cat));
    }
    println!("\n## By hardness{header}");
    for (h, t) in &by_hardness {
        println!("{}", t.row(h));
    }
    println!("\n## Overall{header}");
    println!("{}", overall.row("ALL"));
    println!(
        "\nSOLVED {}/{}  ({:.0}%)   WRONG {} (must stay 0)   REFUSED {}",
        overall.solved,
        overall.total(),
        100.0 * overall.solved as f64 / overall.total().max(1) as f64,
        overall.wrong,
        overall.refused,
    );

    if let Some(op) = out_json {
        let doc = serde_json::json!({
            "overall": {
                "solved": overall.solved, "wrong": overall.wrong, "refused": overall.refused,
                "tentative_ok": overall.tent_ok, "tentative_miss": overall.tent_miss,
                "skip": overall.skip, "total": overall.total(),
            },
            "records": records,
        });
        if let Err(e) = std::fs::write(&op, serde_json::to_string_pretty(&doc).unwrap_or_default()) {
            eprintln!("warning: could not write {op}: {e}");
        } else {
            eprintln!("wrote {op}");
        }
    }
}
