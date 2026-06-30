//! Optional LOCAL-LLM front door — an UNTRUSTED translator from arbitrary NL
//! prose to a single KNOWN op name, for the phrasing the symbolic comprehension
//! can't parse (e.g. "add up all the elements of an array" → array_sum, which the
//! symbolic path mis-resolves to scalar add).
//!
//! TRUST: the LLM NEVER emits code and NEVER bypasses verification. It only
//! proposes WHICH known op; the op's TRUSTED registry `example_cases` then drive
//! synthesis + strict-verify (the existing LLM-free path, unchanged). A
//! hallucinated / unknown op is rejected (validated against the live registry op
//! list — which is itself supplied by the caller, not hard-coded here). So a wrong
//! LLM guess can only mean "wrong known op" → the synthesized program still
//! strict-verifies against that op's spec, and consensus/clarify catch a true
//! mis-pick. The LLM is optional + off unless `NSYNTH_LOCAL_LLM_URL` names an
//! OpenAI-compatible endpoint (e.g. `mlx_lm.server` / LM Studio). HTTP via
//! subprocess `curl` — no http crate dependency, mirroring `hybrid::`.

use std::process::Command;

/// Translate `request` to one of `known_ops` via a local OpenAI-compatible chat
/// endpoint. `None` when disabled (no `NSYNTH_LOCAL_LLM_URL`), on any error, or
/// when the model's chosen op is NOT in `known_ops` (so only a real, synthesizable
/// op can ever reach the verified pipeline).
pub fn translate_op(request: &str, known_ops: &[String]) -> Option<String> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let ops = known_ops.join(", ");
    let sys = format!(
        "You convert a coding request into JSON for a program synthesizer. Output ONLY one \
         JSON object {{\"op\":\"...\"}}. Choose the single best op from this exact list, or \
         {{\"op\":\"\"}} if none fits: {ops}."
    );
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": format!("Request: \"{}\" ->", request.replace('"', "'"))}
        ],
        "temperature": 0.0,
        // Enough headroom that a long op menu never truncates the JSON mid-token.
        "max_tokens": 64
    });
    let out = Command::new("curl")
        .args([
            "-s",
            "-m",
            "30",
            &url,
            "-H",
            "Content-Type: application/json",
            "-d",
            &body.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let resp: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let content = resp["choices"][0]["message"]["content"].as_str()?;
    let op = extract_op(content)?;
    // Only a KNOWN op may pass — a hallucinated op never enters the pipeline.
    known_ops.iter().any(|k| k == &op).then_some(op)
}

/// Pull the `op` field out of the model's reply, tolerating ```json fences /
/// surrounding prose by scanning for the first `{...}` JSON object.
fn extract_op(content: &str) -> Option<String> {
    let s = content.trim();
    let start = s.find('{')?;
    let end = s[start..].find('}')? + start + 1;
    let parsed: serde_json::Value = serde_json::from_str(&s[start..end]).ok()?;
    let op = parsed.get("op")?.as_str()?.trim().to_string();
    (!op.is_empty()).then_some(op)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_op_handles_fences_and_prose() {
        assert_eq!(extract_op("{\"op\":\"array_sum\"}").as_deref(), Some("array_sum"));
        assert_eq!(
            extract_op("```json\n{\"op\":\"reverse\",\"x\":1}\n```").as_deref(),
            Some("reverse")
        );
        assert_eq!(extract_op("Sure: {\"op\":\"is_even\"}").as_deref(), Some("is_even"));
        assert_eq!(extract_op("{\"op\":\"\"}"), None);
        assert_eq!(extract_op("no json here"), None);
    }

    #[test]
    fn translate_op_disabled_without_url() {
        // With no NSYNTH_LOCAL_LLM_URL set, the path is inert (returns None).
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        assert_eq!(
            translate_op("add up all the elements", &["array_sum".to_string()]),
            None
        );
    }
}
