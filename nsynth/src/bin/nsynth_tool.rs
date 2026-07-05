//! RL ENVIRONMENT for training a model to USE nsynth (see src/rlvr.rs +
//! docs/AGENTIC_NL_PLAYBOOK.md). Reads ONE model proposal as JSON on stdin, routes
//! it through nsynth's real synthesis + strict verification + trust gate, and
//! prints the verdict + the RLVR reward as JSON on stdout. One process per rollout
//! (per-proposal isolation; run under an OS timeout for pathological searches).
//!
//! stdin  : {"kind":"examples"|"reference"|"verify",
//!           "signature":"fn f(a: i64) -> i64",
//!           "examples":[{"in":[..],"out":..}, ...],   // examples|verify
//!           "code":"fn f(..){..}",                     // reference|verify
//!           "name":"f",                                // reference
//!           "hidden":[{"in":[..],"out":..}, ...]}      // held-out reward oracle
//! stdout : {"verdict":"verified"|"tentative"|"refused","reward":0.0,"code":".."}
use mog_synth::benchmark::{Example, Value};
use mog_synth::rlvr::{rlvr_reward, run_tool, ToolRequest, ToolResponse};
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

fn parse_examples(v: &serde_json::Value, key: &str) -> Vec<Example> {
    let mut out = Vec::new();
    if let Some(rows) = v.get(key).and_then(|x| x.as_array()) {
        for row in rows {
            let (Some(ins), Some(o)) = (
                row.get("in").and_then(|x| x.as_array()),
                row.get("out").and_then(json_to_value),
            ) else {
                continue;
            };
            let Some(inputs) = ins.iter().map(json_to_value).collect::<Option<Vec<Value>>>() else {
                continue;
            };
            out.push(Example { inputs, expected: o });
        }
    }
    out
}

fn emit(verdict: &str, reward: f32, code: Option<&str>) {
    println!(
        "{}",
        serde_json::json!({"verdict": verdict, "reward": reward, "code": code})
    );
}

fn main() {
    let mut buf = String::new();
    if std::io::stdin().read_to_string(&mut buf).is_err() {
        emit("refused", 0.0, None);
        return;
    }
    let task: serde_json::Value = match serde_json::from_str(buf.trim()) {
        Ok(t) => t,
        Err(_) => {
            emit("refused", 0.0, None);
            return;
        }
    };
    let kind = task.get("kind").and_then(|k| k.as_str()).unwrap_or("examples");
    let signature = task
        .get("signature")
        .and_then(|s| s.as_str())
        .unwrap_or("fn f(a: i64) -> i64")
        .to_string();
    let code = task.get("code").and_then(|c| c.as_str()).unwrap_or("").to_string();
    let name = task.get("name").and_then(|n| n.as_str()).unwrap_or("f").to_string();
    let examples = parse_examples(&task, "examples");
    let hidden = parse_examples(&task, "hidden");

    let req = match kind {
        "reference" => ToolRequest::Reference { name, signature, code },
        "verify" => ToolRequest::VerifyProgram { signature, code, examples },
        _ => ToolRequest::Examples { signature, examples },
    };

    let resp = run_tool(&req);
    // Reward requires the held-out oracle; without it, report 0 (verdict still
    // carries the trust decision).
    let reward = if hidden.is_empty() { 0.0 } else { rlvr_reward(&req, &hidden) };
    match &resp {
        ToolResponse::Verified { code, .. } => emit("verified", reward, Some(code)),
        ToolResponse::Tentative { code, .. } => emit("tentative", reward, Some(code)),
        ToolResponse::Refused { .. } => emit("refused", reward, None),
    }
}
