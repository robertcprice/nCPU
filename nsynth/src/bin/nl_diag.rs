//! Failure-taxonomy diagnostic for the agentic-NL path. Reads one benchmark task
//! on stdin, hands ONLY the NL `text` to `handle_query`, and prints a category so
//! we can see WHERE the NL path breaks (comprehension vs synthesis vs silent-wrong):
//!   CAT <id> SOLVED   <method>   — NL understood + code passes the hidden asserts
//!   CAT <id> WRONG    <method>   — synthesized something, but it FAILS the asserts
//!                                  (mis-comprehended the task, or overfit)
//!   CAT <id> REFUSED  <route>    — handle_query returned success=false (no plan)
//!   CAT <id> SKIP
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
        println!("CAT -1 SKIP");
        return;
    }
    let task: serde_json::Value = match serde_json::from_str(buf.trim()) {
        Ok(t) => t,
        Err(_) => {
            println!("CAT -1 SKIP");
            return;
        }
    };
    let id = task.get("id").and_then(|v| v.as_i64()).unwrap_or(-1);
    let text = task.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
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
    let Some(exs) = dedup_consistent_examples(&exs).filter(|e| e.len() >= 3) else {
        println!("CAT {id} SKIP");
        return;
    };

    let root = std::env::temp_dir().join(format!("nldiag_{id}_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&root);
    let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r = session.handle_query(text);
    let _ = std::fs::remove_dir_all(&root);

    let method = r.synthesis_method.clone().unwrap_or_else(|| format!("{:?}", r.route));
    if !r.success {
        println!("CAT {id} REFUSED {:?}", r.route);
    } else if mog_synth::runtime::code_reproduces_examples(&r.response, &exs) {
        println!("CAT {id} SOLVED {method}");
    } else {
        println!("CAT {id} WRONG {method}");
    }
}
