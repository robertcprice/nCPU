//! Verify a Mog program against examples. Reads {"code": "...", "examples":[{"in":[..],"out":..}]}
//! on stdin; prints "OK" if the program reproduces every example, else "FAIL".
//! Used to measure a RAW model's single-shot correctness (no nSynth gate/repair) so
//! it can be compared against the gated `answer()` path.
use mog_synth::benchmark::{Example, Value};
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
        println!("FAIL");
        return;
    }
    let Ok(task): Result<serde_json::Value, _> = serde_json::from_str(buf.trim()) else {
        println!("FAIL");
        return;
    };
    let code = task.get("code").and_then(|v| v.as_str()).unwrap_or("");
    let examples: Vec<Example> = task
        .get("examples")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|ex| {
                    let inputs = ex
                        .get("in")?
                        .as_array()?
                        .iter()
                        .filter_map(json_to_value)
                        .collect();
                    Some(Example { inputs, expected: json_to_value(ex.get("out")?)? })
                })
                .collect()
        })
        .unwrap_or_default();
    if examples.is_empty() {
        println!("FAIL");
        return;
    }
    if mog_synth::runtime::code_reproduces_examples(code, &examples) {
        println!("OK");
    } else {
        println!("FAIL");
    }
}
