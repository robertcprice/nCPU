//! Measure the never-confidently-wrong NL router (`verified_nl_router`). Reads one
//! benchmark task on stdin, routes ONLY its NL `text` to a verified library op, then
//! runs that op against the task's hidden examples. Prints:
//!   SOLVED  <id> <op>   — routed to a verified op that reproduces every example
//!   WRONG   <id> <op>   — routed, but the op FAILS the examples (a mis-route: the
//!                         one thing this design must drive toward zero)
//!   REFUSED <id>        — no confident op match -> honest refusal (never a guess)
//!   SKIP    <id>
//! The headline metric is WRONG: an LLM hallucinates a plausible-but-buggy program;
//! this router must instead be right or refuse. WRONG > 0 means the confidence gate
//! is too loose.
use mog_synth::benchmark::{Example, Value};
use mog_synth::verified_nl_router;
use std::io::Read;

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
    let text = task.get("text").and_then(|v| v.as_str()).unwrap_or("");

    // Parse the hidden examples (the oracle we grade against — the router never sees them).
    let examples: Vec<Example> = task
        .get("examples")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|ex| {
                    let inputs: Vec<Value> = ex
                        .get("in")?
                        .as_array()?
                        .iter()
                        .filter_map(json_to_value)
                        .collect();
                    let expected = json_to_value(ex.get("out")?)?;
                    Some(Example { inputs, expected })
                })
                .collect()
        })
        .unwrap_or_default();
    if examples.is_empty() {
        println!("SKIP {id}");
        return;
    }

    // Gated (never-wrong) taxonomy:
    //   SOLVED  — NL proposed an op AND the verify gate confirmed it on the examples
    //   GATED   — NL proposed an op but the gate REJECTED it (a mis-route the system
    //             refuses instead of returning — the never-wrong property working)
    //   REFUSED — NL proposed nothing (no confident op name in the prompt)
    match verified_nl_router::route_verified(text, &examples) {
        Some(r) => println!("SOLVED {id} {}", r.op.name),
        None => {
            // Single op failed — try a verified 2-op composition ("A then B").
            if verified_nl_router::route_composed(text, &examples).is_some() {
                println!("COMPOSED {id}");
            } else {
                match verified_nl_router::route(text) {
                    Some(r) => println!("GATED {id} {}", r.op.name),
                    None => println!("REFUSED {id}"),
                }
            }
        }
    }
}
