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

use std::process::{Command, Stdio};

/// Ensure the local model endpoint is reachable. If `NSYNTH_LOCAL_LLM_URL` already
/// responds → true. Otherwise, when `NSYNTH_LOCAL_LLM_AUTOSERVE` is set, spawn
/// `mlx_lm.server` (detached; persists across calls) for `NSYNTH_LOCAL_LLM_MODEL`
/// on the URL's port and wait for it to load. Best-effort; returns whether the
/// endpoint is now reachable. Without AUTOSERVE this is a fast no-op (returns
/// whether a server happens to already be up). So the lane can be fully
/// self-starting (`NSYNTH_LOCAL_LLM_AUTOSERVE=1`) or rely on an externally-managed
/// server (default).
pub fn ensure_server() -> bool {
    let Some(url) = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())
    else {
        return false;
    };
    if server_reachable(&url) {
        return true;
    }
    if std::env::var("NSYNTH_LOCAL_LLM_AUTOSERVE")
        .ok()
        .filter(|s| !s.is_empty())
        .is_none()
    {
        return false;
    }
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_default();
    if model.is_empty() {
        return false;
    }
    let port = url_port(&url).unwrap_or(8765);
    let _ = Command::new("python3")
        .args(["-m", "mlx_lm", "server", "--model", &model, "--port", &port.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    // Wait for the model to load (~16s observed; allow up to ~60s).
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_secs(2));
        if server_reachable(&url) {
            return true;
        }
    }
    false
}

/// GET `<base>/models` (base derived from the chat-completions URL) returns 2xx.
fn server_reachable(url: &str) -> bool {
    let base = url.rsplit_once("/chat/completions").map_or(url, |(b, _)| b);
    Command::new("curl")
        .args([
            "-s", "-m", "2", "-o", "/dev/null", "-w", "%{http_code}",
            &format!("{base}/models"),
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).starts_with('2'))
        .unwrap_or(false)
}

fn url_port(url: &str) -> Option<u16> {
    url.split("://")
        .nth(1)?
        .split('/')
        .next()?
        .rsplit(':')
        .next()?
        .parse()
        .ok()
}

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

/// Rewrite an arbitrary request into ONE short CANONICAL sentence the symbolic
/// comprehension reliably parses — a single op or a filter/map/reduce COMPOSITION
/// ("the sum of the positive values", "the maximum of the doubled values"). Used
/// as Mode A' (composition breadth) when the single-op menu doesn't fit. Still
/// comprehended + strict-verified downstream, so a bad rewrite fails closed.
/// `None` when disabled / on error / empty. Untrusted, like `translate_op`.
pub fn canonical_rephrase(request: &str) -> Option<String> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let sys = "Rewrite a coding request into ONE short canonical sentence a program synthesizer \
        understands. Use plain operation words (sum, maximum, minimum, average, product, count, \
        sort, reverse, double, square, negate; positive/negative/even/odd values). For a \
        collection operation say 'of the list'/'of the values'. For a filter+reduce say 'the SUM \
        of the POSITIVE values'. Output ONLY the rewritten sentence, no quotes.\n\
        Examples:\n\
        'add up only the positive ones' -> the sum of the positive values\n\
        'what is the biggest number' -> the maximum of the list\n\
        'multiply every item by itself' -> the squared values of the list\n\
        'total of the negatives' -> the sum of the negative values";
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": format!("{} ->", request.replace('"', "'"))}
        ],
        "temperature": 0.0,
        "max_tokens": 48
    });
    let out = Command::new("curl")
        .args([
            "-s", "-m", "30", &url, "-H", "Content-Type: application/json", "-d", &body.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let resp: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let content = resp["choices"][0]["message"]["content"].as_str()?.trim();
    let line = content.lines().next().unwrap_or(content).trim().trim_matches('"').to_string();
    (!line.is_empty() && line.len() < 200).then_some(line)
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

    #[test]
    fn url_port_parses() {
        assert_eq!(url_port("http://localhost:8765/v1/chat/completions"), Some(8765));
        assert_eq!(url_port("http://127.0.0.1:1234/v1/chat/completions"), Some(1234));
        assert_eq!(url_port("not a url"), None);
    }

    #[test]
    fn ensure_server_inert_without_url() {
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        assert!(!ensure_server());
    }
}
