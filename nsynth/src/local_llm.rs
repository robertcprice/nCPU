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
pub fn translate_op(request: &str, ops: &[(String, String)]) -> Option<String> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    // Gloss menu: one "fn_name: definition" per line so the model matches a
    // PARAPHRASE to the exact registered op (bare names leave it guessing).
    let menu = ops
        .iter()
        .map(|(name, gloss)| {
            if gloss.is_empty() {
                name.clone()
            } else {
                format!("{name}: {gloss}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let sys = format!(
        "You convert a coding request into JSON for a program synthesizer. Output ONLY one \
         JSON object {{\"op\":\"...\"}} where op is the exact name (the text before the colon) \
         of the single best match, judged against the description after each colon, or \
         {{\"op\":\"\"}} if none fits:\n{menu}"
    );
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": format!("Request: \"{}\" ->", request.replace('"', "'"))}
        ],
        "temperature": 0.0,
        // Headroom for a REASONING model (Gemma 4): thinking can consume 100-150
        // tokens before the JSON, so a low cap returns empty content (finish=length).
        "max_tokens": 256
    });
    let out = Command::new("curl")
        .args([
            "-s",
            "-m",
            "40",
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
    let msg = &resp["choices"][0]["message"];
    // Prefer the answer channel; fall back to the reasoning channel (a reasoning
    // model that ran out of token budget may leave `content` empty but still have
    // emitted the JSON inside `reasoning`).
    let op = msg["content"]
        .as_str()
        .and_then(extract_op)
        .or_else(|| msg["reasoning"].as_str().and_then(extract_op))?;
    // Only a KNOWN op may pass — a hallucinated op never enters the pipeline.
    ops.iter().any(|(k, _)| k == &op).then_some(op)
}

/// One LLM-proposed I/O example (raw JSON values; the caller maps to runtime
/// values). Mode B's untrusted spec.
pub struct ProposedExample {
    pub inputs: Vec<serde_json::Value>,
    pub output: serde_json::Value,
}

/// Mode B (out-of-vocab, RISKIER tier): the LLM proposes I/O EXAMPLES for a task
/// with no known op. The spec is UNTRUSTED — the caller MUST synthesize from these
/// + strict-verify (and ideally hold some out): a program that merely matches the
/// examples can still be wrong if the LLM's examples are wrong. Returns >=4 parsed
/// examples, or None (disabled / error / too few). max_tokens is generous because
/// Gemma-class models pretty-print + may use reasoning tokens.
pub fn propose_examples(request: &str) -> Option<Vec<ProposedExample>> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let sys = "Output ONLY a JSON object, no prose: {\"examples\":[{\"in\":[ARG,...],\"out\":RESULT}]}. \
        Give 6 CORRECT examples for the request — compute each output carefully and cover edge \
        cases. Each ARG and RESULT is an integer, an array of integers, or a boolean.";
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": format!("Request: {}", request.replace('"', "'"))}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    });
    let out = Command::new("curl")
        .args([
            "-s", "-m", "60", &url, "-H", "Content-Type: application/json", "-d", &body.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let resp: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let content = resp["choices"][0]["message"]["content"].as_str()?;
    parse_examples(content)
}

/// Extract `{"examples":[{"in":[..],"out":..}]}` from the model text.
fn parse_examples(content: &str) -> Option<Vec<ProposedExample>> {
    let s = content.trim();
    let start = s.find('{')?;
    let end = s.rfind('}')? + 1;
    let parsed: serde_json::Value = serde_json::from_str(&s[start..end]).ok()?;
    let arr = parsed.get("examples")?.as_array()?;
    let mut out = Vec::new();
    for e in arr {
        let inputs = e.get("in")?.as_array()?.clone();
        let output = e.get("out")?.clone();
        if inputs.is_empty() {
            continue;
        }
        out.push(ProposedExample { inputs, output });
    }
    (out.len() >= 4).then_some(out)
}

/// One LLM-proposed sub-function in a project decomposition: a fn name + a single
/// NL description the per-component synthesis door will independently solve+verify.
pub struct ProposedComponent {
    pub name: String,
    pub description: String,
}

/// Mode C (project decomposition, RISKIER tier): the LLM breaks an open-ended
/// build request into a list of named, individually-synthesizable sub-functions.
/// UNTRUSTED — the LLM only proposes the DECOMPOSITION (names + NL descriptions);
/// each component is still synthesized + STRICT-VERIFIED downstream, so a bad plan
/// yields components that either verify (correct leaves) or are dropped, never an
/// unverified accept. Returns >=2 components, or None (disabled / error / too few).
/// NOTE: this verifies each PART, not the whole-artifact behavior — there is no
/// example oracle for "does the assembled program do what was asked".
pub fn propose_decomposition(request: &str) -> Option<Vec<ProposedComponent>> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let sys = "Break a programming request into small INDEPENDENT pure functions, each computing \
        ONE value from its inputs (integers, arrays of integers, or booleans). Output ONLY a JSON \
        object, no prose: {\"functions\":[{\"name\":\"snake_case\",\"description\":\"one clear \
        sentence describing exactly what this function computes\"}]}. Give 2 to 5 functions. Each \
        description must be self-contained and concrete (e.g. 'the sum of all elements in the \
        array', 'whether the number is even'), NOT a vague feature.";
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": format!("Request: {}", request.replace('"', "'"))}
        ],
        "temperature": 0.0,
        // Decomposition is a harder task than op-selection: Gemma 4's reasoning
        // channel alone runs ~1900 tokens, so a low cap returns empty content
        // (finish=length). Give room for reasoning AND the JSON answer.
        "max_tokens": 1024
    });
    let out = Command::new("curl")
        .args([
            "-s", "-m", "90", &url, "-H", "Content-Type: application/json", "-d", &body.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let resp: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let msg = &resp["choices"][0]["message"];
    let content = msg["content"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| msg["reasoning"].as_str())?;
    parse_decomposition(content)
}

/// Extract `{"functions":[{"name":..,"description":..}]}` from the model text.
fn parse_decomposition(content: &str) -> Option<Vec<ProposedComponent>> {
    let s = content.trim();
    let start = s.find('{')?;
    let end = s.rfind('}')? + 1;
    let parsed: serde_json::Value = serde_json::from_str(&s[start..end]).ok()?;
    let arr = parsed.get("functions")?.as_array()?;
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for e in arr {
        let name = e.get("name")?.as_str()?.trim().to_string();
        let description = e.get("description")?.as_str()?.trim().to_string();
        if name.is_empty() || description.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        out.push(ProposedComponent { name, description });
    }
    (out.len() >= 2).then_some(out)
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
            translate_op(
                "add up all the elements",
                &[("array_sum".to_string(), "sum of all elements".to_string())]
            ),
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

    #[test]
    fn parse_decomposition_extracts_functions() {
        let s = "```json\n{\"functions\":[{\"name\":\"sum_all\",\"description\":\"the sum of all elements\"},\
                 {\"name\":\"count_pos\",\"description\":\"how many are positive\"}]}\n```";
        let c = parse_decomposition(s).expect("2 components");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].name, "sum_all");
        assert_eq!(c[1].description, "how many are positive");
        // fewer than 2 distinct → None
        assert!(parse_decomposition("{\"functions\":[{\"name\":\"a\",\"description\":\"x\"}]}").is_none());
        // duplicate names dedup below the floor → None
        assert!(parse_decomposition(
            "{\"functions\":[{\"name\":\"a\",\"description\":\"x\"},{\"name\":\"a\",\"description\":\"y\"}]}"
        )
        .is_none());
    }

    #[test]
    fn parse_examples_extracts_in_out() {
        let s = "```json\n{\"examples\":[{\"in\":[1],\"out\":8},{\"in\":[2],\"out\":11},\
                 {\"in\":[0],\"out\":5},{\"in\":[3],\"out\":14}]}\n```";
        let ex = parse_examples(s).expect("4 examples");
        assert_eq!(ex.len(), 4);
        assert_eq!(ex[0].inputs, vec![serde_json::json!(1)]);
        assert_eq!(ex[0].output, serde_json::json!(8));
        // fewer than 4 → None
        assert!(parse_examples("{\"examples\":[{\"in\":[1],\"out\":2}]}").is_none());
    }
}
