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

/// Propose a full COMPONENT spec (JSON) for an arbitrary request, using ONLY the
/// given verified leaf ops. UNTRUSTED — the returned JSON is a hypothesis the
/// caller MUST run through the compile + behavior gates
/// (`component::verify_component_proposal`); a hallucinated leaf, mistyped glue, or
/// lying contract is rejected there. This is the PROPOSER half of RLVR; the model
/// may be as unreliable as it likes because the verifier ships only what survives.
/// `None` when disabled (no URL) / on error / empty.
pub fn propose_component(request: &str, known_leaves: &[String]) -> Option<String> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let leaves = known_leaves.join(", ");
    // MARKER protocol, not JSON-embedded code: small models can't reliably escape
    // quotes+newlines inside a JSON string. We take raw code between markers and
    // build the JSON ourselves (serde escapes it correctly), so the only failures
    // left are REAL ones the compile/behavior gates judge.
    let sys = "You design a tiny Rust component for a program synthesizer. Respond in EXACTLY this \
        format and nothing else:\n\
        NAME: <snake_case_name>\n\
        LEAVES: <comma-separated ops you use, from the allowed list>\n\
        CODE:\n\
        <rust here>\n\
        SMOKE:\n\
        <rust here>\n\
        Rules:\n\
        - Use ONLY the allowed ops as leaves. Each op is a free function imported with its name \
        TWICE: `use crate::negate::negate;` for op `negate`. Never import from the module you write.\n\
        - CODE: a short `pub struct` plus an `impl` whose methods call the leaf functions. Only i64 \
        and Vec<i64>. No external crates, no std collections beyond Vec.\n\
        - SMOKE: `#[cfg(test)] mod name_behaves { use super::*; #[test] fn t() { /* construct, call \
        methods, assert_eq! exact expected values */ } }` that PROVES the behavior.\n\
        - Write real newlines in the code (this is NOT JSON). No markdown fences, no prose.";
    let user = format!(
        "Allowed ops: {leaves}\nRequest: {}",
        request.replace('"', "'")
    );
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2,
        "max_tokens": 1400
    });
    let out = Command::new("curl")
        .args([
            "-s", "-m", "180", &url, "-H", "Content-Type: application/json", "-d", &body.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let resp: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let content = resp["choices"][0]["message"]["content"].as_str()?;
    // Preferred: marker protocol -> build valid JSON from raw fields.
    if let Some(json) = marker_component_to_json(content) {
        return Some(json);
    }
    // Fallback: the model emitted JSON directly (tolerate raw control chars).
    extract_json_array(content).map(|j| escape_control_chars_in_strings(&j))
}

/// Parse the NAME/LEAVES/CODE/SMOKE marker reply into a proper components JSON doc.
/// Raw code goes through serde (correct escaping), so the emitted JSON always
/// parses; the compile/behavior gates then judge the substance. `None` if required
/// markers are missing.
fn marker_component_to_json(content: &str) -> Option<String> {
    let name = marker_line(content, "NAME:")?;
    let name = sanitize_ident(&name);
    if name.is_empty() {
        return None;
    }
    let raw_leaves: Vec<String> = marker_line(content, "LEAVES:")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    // Dedup (models repeat a leaf).
    let mut seen = std::collections::HashSet::new();
    let leaves: Vec<String> = raw_leaves.into_iter().filter(|l| seen.insert(l.clone())).collect();
    let code = marker_block(content, "CODE:", &["SMOKE:"])?;
    let smoke = marker_block(content, "SMOKE:", &[]);
    if leaves.is_empty() || code.trim().is_empty() {
        return None;
    }
    // Auto-inject the `use crate::<op>::<op>;` imports the model reliably omits.
    // Each declared leaf is re-exported at crate root, so this always resolves; the
    // model supplies the LOGIC, the harness completes the wiring (like a linker).
    let mut prelude = String::new();
    for leaf in &leaves {
        if !code.contains(&format!("use crate::{leaf}")) {
            prelude.push_str(&format!("use crate::{leaf}::{leaf};\n"));
        }
    }
    let code = format!("{prelude}{code}");
    let glue = if smoke.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false) {
        serde_json::json!({ "module": name, "code": code, "smoke": smoke })
    } else {
        serde_json::json!({ "module": name, "code": code })
    };
    Some(
        serde_json::json!([{
            "name": name,
            "surfaces": [name],
            "leaves": leaves,
            "glue": glue,
        }])
        .to_string(),
    )
}

/// Value on the line beginning `marker` (inline value after the marker).
fn marker_line(content: &str, marker: &str) -> Option<String> {
    content
        .lines()
        .find(|l| l.trim_start().starts_with(marker))
        .map(|l| l.trim_start()[marker.len()..].trim().to_string())
}

/// Raw text from the line AFTER `marker` up to the first line starting with any of
/// `stops` (or end of input). Preserves interior newlines.
fn marker_block(content: &str, marker: &str, stops: &[&str]) -> Option<String> {
    let mut lines = content.lines();
    // advance to the marker line
    for l in lines.by_ref() {
        if l.trim_start().starts_with(marker) {
            break;
        }
    }
    let mut out: Vec<&str> = Vec::new();
    for l in lines {
        if stops.iter().any(|s| l.trim_start().starts_with(s)) {
            break;
        }
        out.push(l);
    }
    let joined = out.join("\n");
    let trimmed = joined.trim_matches('`').trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// Reduce to a valid snake identifier.
fn sanitize_ident(s: &str) -> String {
    let mut out: String = s
        .trim()
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect();
    out = out.trim_matches('_').to_string();
    if out.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        out.insert(0, '_');
    }
    out
}

/// Small models routinely emit multi-line Rust inside a JSON string value with RAW
/// newlines/tabs, which is invalid JSON. Escape any control char that appears while
/// INSIDE a `"`-delimited string (respecting backslash escapes) so the proposal
/// parses; content outside strings is untouched. This is a tolerance for the
/// PROPOSER's sloppiness only — the strict `parse_components_json` used for real
/// data files is unchanged, and the compile/behavior gates still judge the result.
fn escape_control_chars_in_strings(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_str = false;
    let mut escaped = false;
    for c in s.chars() {
        if in_str {
            if escaped {
                out.push(c);
                escaped = false;
            } else if c == '\\' {
                out.push(c);
                escaped = true;
            } else if c == '"' {
                out.push(c);
                in_str = false;
            } else if c == '\n' {
                out.push_str("\\n");
            } else if c == '\r' {
                out.push_str("\\r");
            } else if c == '\t' {
                out.push_str("\\t");
            } else if (c as u32) < 0x20 {
                // drop other unprintable control chars
            } else {
                out.push(c);
            }
        } else {
            if c == '"' {
                in_str = true;
            }
            out.push(c);
        }
    }
    out
}

/// Pull the first top-level JSON array out of the model's reply, tolerating
/// ```json fences / surrounding prose.
fn extract_json_array(content: &str) -> Option<String> {
    let s = content.trim();
    let start = s.find('[')?;
    let end = s.rfind(']')?;
    if end < start {
        return None;
    }
    Some(s[start..=end].to_string())
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
    fn marker_reply_becomes_valid_component_json() {
        let reply = "NAME: squarer\nLEAVES: square\nCODE:\nuse crate::square::square;\npub struct Squarer { v: i64 }\nimpl Squarer { pub fn new(v: i64) -> Self { Squarer { v } } pub fn go(&self) -> i64 { square(self.v) } }\nSMOKE:\n#[cfg(test)] mod m { use super::*; #[test] fn t() { assert_eq!(Squarer::new(4).go(), 16); } }\n";
        let json = marker_component_to_json(reply).expect("parsed");
        // Always valid JSON (serde escaped the raw code).
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(v[0]["name"], "squarer");
        assert_eq!(v[0]["leaves"][0], "square");
        assert!(v[0]["glue"]["code"].as_str().unwrap().contains("use crate::square::square;"));
        assert!(v[0]["glue"]["smoke"].as_str().unwrap().contains("assert_eq!"));
    }

    #[test]
    fn escape_control_chars_only_inside_strings() {
        // Raw newline inside a string value -> escaped; structure newline untouched.
        let raw = "[{\"code\":\"line1\nline2\"}]";
        let fixed = escape_control_chars_in_strings(raw);
        assert!(serde_json::from_str::<serde_json::Value>(&fixed).is_ok(), "{fixed}");
        assert!(fixed.contains("line1\\nline2"));
        // An already-escaped \n is preserved (backslash respected).
        let ok = "[{\"code\":\"a\\nb\"}]";
        assert_eq!(escape_control_chars_in_strings(ok), ok);
    }

    #[test]
    fn extract_json_array_handles_fences_and_prose() {
        assert_eq!(
            extract_json_array("[{\"name\":\"x\"}]").as_deref(),
            Some("[{\"name\":\"x\"}]")
        );
        assert_eq!(
            extract_json_array("```json\n[{\"a\":1}]\n```").as_deref(),
            Some("[{\"a\":1}]")
        );
        assert_eq!(extract_json_array("Sure, here: [1,2,3] done").as_deref(), Some("[1,2,3]"));
        assert_eq!(extract_json_array("no array"), None);
    }

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
