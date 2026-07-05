//! AGENTIC-NL variant of the MBPP driver: solve ONE task from ITS NATURAL-LANGUAGE
//! PROMPT ALONE (the MBPP `text`), with NO I/O examples handed to the engine — the
//! opposite of `mbpp_solve_one` (which is programming-by-example). The synthesized
//! program is then VERIFIED against the task's hidden asserts (`examples`), so a
//! `SOLVED` means the agent understood the English task AND produced code that
//! passes the real tests. Read one JSON task on stdin; print
//! `SOLVED <id> <method>` / `UNSOLVED <id>` / `SKIP <id>`. Run under an OS
//! `timeout` via `scripts/run_mbpp_bench.sh` for per-task isolation.
//!
//! This measures the AGENTIC dimension the PBE benchmark cannot: NL comprehension
//! is now the bottleneck, not example-fitting. LLM-FREE (the NL front door is the
//! emergent linguigenesis resolver, no model).
use mog_synth::agent::{CodingAgentSession, GuardrailPolicy};
use mog_synth::benchmark::{dedup_consistent_examples, Example, Value};
use std::io::Read;

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
    let text = task.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
    if text.is_empty() {
        println!("SKIP {id}");
        return;
    }

    // Parse the hidden asserts (used ONLY to verify the result — never given to the
    // engine).
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

    // AGENTIC: hand the ENGLISH prompt ALONE to the product entry point.
    let root = std::env::temp_dir().join(format!("mbpp_nl_{id}_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&root);
    let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let result = session.handle_query(text);
    let _ = std::fs::remove_dir_all(&root);

    if !result.success {
        println!("UNSOLVED {id}");
        return;
    }
    // VERIFY the synthesized program against the hidden asserts. A green result is
    // an HONEST agentic solve (NL understood + code passes the real tests).
    if mog_synth::runtime::code_reproduces_examples(&result.response, &exs) {
        let method = result.synthesis_method.as_deref().unwrap_or("nl");
        println!("SOLVED {id} {method}");
    } else {
        println!("UNSOLVED {id}");
    }
}
