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

/// Mode D (verify-and-repair): the LLM writes a WHOLE Mog program for a task no
/// known op / example-search can produce (arbitrary algorithms). UNTRUSTED — the
/// caller MUST run it against the tests and accept ONLY on a full pass; `prior` =
/// (previous code, concrete failure) drives a fix on the next iteration. Returns the
/// extracted Mog function source, or None. This is the lever that scales to
/// arbitrary algorithms (the model knows sort/DP/recursion); verification keeps the
/// guarantee (a wrong program never passes the tests).
/// The Mog-writing system prompt. Shared by `propose_program` (inference) AND the
/// training-corpus harvester (`training_record`), so fine-tuning optimizes the EXACT
/// inference distribution — the model learns to answer this prompt with valid Mog.
pub const MOG_SYSTEM_PROMPT: &str =
    "You write MOG, a small imperative language. RULES (critical — Mog is NOT Rust):\n\
        - Declare a variable with `name: TYPE = EXPR;` — NEVER `let`, NEVER `mut`. \
        TYPE is i64, [i64], bool, or string.\n\
        - Reassign with `name = EXPR;`. Use EXACTLY the given function signature.\n\
        - Control flow: `if C { } else { }`, `while C { }`, `for e in arr { }` (e = each element), \
        `for ch in s { }` (ch = a 1-char string), `return EXPR;`.\n\
        - Operators: + - * / % == != < > <= >= && || . Integer arithmetic only.\n\
        - Arrays: arr[i] (index), arr.len (length, NO parens). For index loops: \
        `i: i64 = 0; while i < arr.len { ... arr[i] ...; i = i + 1; }`. BUILD an array by \
        starting empty and pushing: `out: [i64] = []; for e in arr { out.push(e * 2); } return out;` \
        — or a fixed literal `[a, b]`. This is how you RETURN a list.\n\
        - Strings: s.len, s[i], s.upper(), s.lower(), s.reverse(), s.chars(), s.split(x), \
        s.contains(x), s.slice(a,b); per char: `for ch in s`, ch.is_vowel(), ch.is_digit(), \
        ch.is_alpha(), ch.ord(); a char is a 1-char string — compare with `ch == 'a'`.\n\
        - NO imports, NO `let`, NO helper functions — ONE self-contained function.\n\
        Output ONLY the function in a ```mog code block, no prose.\n\n\
        CRITICAL: ONE self-contained function. You may NOT call any other function \
        (no is_prime(), no helpers) — INLINE everything with nested loops.\n\n\
        EXAMPLE (count elements greater than ten):\n\
        ```mog\n\
        fn solve(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e > 10 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n\
        ```\n\n\
        EXAMPLE with an INLINED nested loop (count primes below n — the prime test is \
        inlined, NOT a helper call):\n\
        ```mog\n\
        fn solve(n: i64) -> i64 {\n    total: i64 = 0;\n    k: i64 = 2;\n    while k < n {\n        is_p: i64 = 1;\n        d: i64 = 2;\n        while (d * d) <= k {\n            if (k % d) == 0 {\n                is_p = 0;\n            }\n            d = d + 1;\n        }\n        if is_p == 1 {\n            total = total + 1;\n        }\n        k = k + 1;\n    }\n    return total;\n}\n\
        ```";

/// Build the user message for a Mog task (task text + optional signature + examples),
/// matching what the repair loop sends. Shared so a harvested training pair has the
/// SAME user turn the model will see at inference.
pub fn mog_user_message(request: &str) -> String {
    format!("Task:\n{request}\n\nWrite the Mog function.")
}

/// A single fine-tuning example in `mlx_lm.lora` chat format: the shared Mog system
/// prompt + the task user turn + the VERIFIED Mog program as the assistant answer.
/// Only ever built from a program that passed the verifier, so the corpus is
/// guaranteed-correct (the STaR / rejection-sampling property).
pub fn training_record(request: &str, verified_code: &str) -> serde_json::Value {
    serde_json::json!({
        "messages": [
            {"role": "system", "content": MOG_SYSTEM_PROMPT},
            {"role": "user", "content": mog_user_message(request)},
            {"role": "assistant", "content": format!("```mog\n{}\n```", verified_code.trim())}
        ]
    })
}

pub fn propose_program(
    request: &str,
    prior: Option<(&str, &str)>,
    temperature: f64,
) -> Option<String> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let sys = MOG_SYSTEM_PROMPT;
    let task = match prior {
        None => format!("Task:\n{request}\n\nWrite the Mog function."),
        Some((code, err)) => format!(
            "Task:\n{request}\n\nYour previous attempt:\n{code}\n\nIt FAILED: {err}\n\n\
             Fix the bug and output ONLY the corrected Mog function."
        ),
    };
    // Fold the system prompt into the USER turn: Gemma (and some other local models
    // via mlx_lm) reject a `system` role with "System role not supported" (HTTP 404).
    // A single user message works everywhere.
    let user = format!("{sys}\n\n{task}");
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "max_tokens": 2500,
        // Reasoning models (Qwen3.x) otherwise spend 2000+ tokens "thinking" — slow
        // (>2min, curl times out) and the code lands after a wall of prose. Disable
        // it: with thinking off + the in-prompt Mog example, Qwen writes correct Mog
        // in ~8s. Ignored by templates without this flag (e.g. Gemma).
        "chat_template_kwargs": {"enable_thinking": false}
    });
    let out = Command::new("curl")
        .args([
            "-s", "-m", "120", &url, "-H", "Content-Type: application/json", "-d", &body.to_string(),
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
    extract_code(content)
}

/// Propose a plain-**Rust** function (not Mog). Small local models write ordinary
/// Rust well but do not know the Mog DSL, so the repo-repair tier — whose oracle is
/// `cargo test` over real Rust — asks for Rust directly. Same gated, curl-only,
/// inert-without-`NSYNTH_LOCAL_LLM_URL` shape as [`propose_program`]; the caller
/// still reshapes the result to the repo signature and the cargo-test oracle still
/// decides, so a wrong or non-compiling proposal is discarded.
pub fn propose_rust_fn(request: &str, prior: Option<(&str, &str)>, temperature: f64) -> Option<String> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let sys = "You write ONE Rust function. Output ONLY the function definition in a \
               ```rust code block — no explanation, no tests, no main. Use plain Rust \
               (i64 for integers).";
    let task = match prior {
        None => format!("Task:\n{request}\n\nWrite the Rust function."),
        Some((code, err)) => format!(
            "Task:\n{request}\n\nYour previous attempt:\n{code}\n\nIt FAILED: {err}\n\n\
             Fix the bug and output ONLY the corrected Rust function."
        ),
    };
    // Fold the system instructions into a SINGLE user turn: some local models
    // (Gemma) reject a `system` role outright ("System role not supported"). Append
    // `/no_think` — Qwen3.x reasoning models skip their chain-of-thought and emit the
    // function directly (faster, and the answer lands in `content` instead of being
    // truncated mid-reasoning); harmless text to non-Qwen models.
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": format!("{sys}\n\n{task}\n\n/no_think")}],
        // Disable reasoning at the template level too (llama.cpp honors this for
        // Qwen); ignored by templates that don't support it.
        "chat_template_kwargs": {"enable_thinking": false},
        "temperature": temperature,
        // Generous cap: a reasoning model that still thinks needs room to finish the
        // function in `content` rather than truncating mid-thought.
        "max_tokens": 4096
    });
    let out = Command::new("curl")
        .args([
            "-s", "-m", "120", &url, "-H", "Content-Type: application/json", "-d", &body.to_string(),
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
    extract_code(content)
}

/// Extract a Mog function from the model output: prefer a fenced ``` block; else
/// take from the first `fn ` through its balanced closing brace.
fn extract_code(content: &str) -> Option<String> {
    // Fenced blocks. A REASONING model (Qwen3.5, etc.) emits DRAFT code blocks in its
    // thinking before the final answer, so scan ALL ``` fences and keep the LAST block
    // that defines a function — that is the finished program, not a scratch draft.
    // Strip an optional language tag on the fence line so it never leaks into the code.
    let parts: Vec<&str> = content.split("```").collect();
    // Blocks are the odd-indexed segments between fences.
    let mut best: Option<String> = None;
    for block in parts.iter().skip(1).step_by(2) {
        let body = match block.find('\n') {
            Some(nl)
                if !block[..nl].trim().is_empty()
                    && block[..nl].trim().chars().all(|c| c.is_ascii_alphanumeric()) =>
            {
                &block[nl + 1..]
            }
            _ => block.strip_prefix('\n').unwrap_or(block),
        };
        let code = body.trim();
        if code.contains("fn ") {
            best = Some(code.to_string());
        }
    }
    if best.is_some() {
        return best;
    }
    // Fallback: from `fn ` to the matching closing brace.
    let fn_pos = content.find("fn ")?;
    let rest = &content[fn_pos..];
    let mut depth = 0i32;
    let mut seen_open = false;
    for (i, c) in rest.char_indices() {
        match c {
            '{' => {
                depth += 1;
                seen_open = true;
            }
            '}' => {
                depth -= 1;
                if seen_open && depth == 0 {
                    return Some(rest[..=i].trim().to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// One LLM-proposed sub-function with an embedded I/O CONTRACT: a fn name, a NL
/// description, AND >=3 example rows. Unlike `ProposedComponent` (Mode C, which
/// re-derives examples downstream), this carries the examples directly so the
/// bridge can build a real Problem (seed + holdout), strict-verify, and emit
/// reproduction tests. UNTRUSTED — the examples are the LLM's claim, never an
/// oracle; the synthesis door is the sole authority.
pub struct ProposedComponentSpec {
    pub name: String,
    pub description: String,
    pub examples: Vec<ProposedExample>,
}

/// Mode C+ (contract-bearing decomposition). Successor to `propose_decomposition`:
/// asks for name + description + >=3 I/O examples per fn IN ONE CALL, so the bridge
/// can build a real Problem (seed+holdout) and strict-verify each component AND
/// emit reproduction tests. Untrusted; gated downstream by NSYNTH_LOCAL_LLM_PROJECT.
/// Returns >=2 functions (each with >=3 examples), or None (disabled / error /
/// too few).
pub fn propose_decomposition_with_contracts(request: &str) -> Option<Vec<ProposedComponentSpec>> {
    let url = std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let model = std::env::var("NSYNTH_LOCAL_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let sys = "Break a programming request into small INDEPENDENT pure functions, each computing \
        ONE value from its inputs. Inputs and outputs are integers, arrays of integers, or booleans. \
        For EACH function give a name (snake_case), a one-sentence description, AND at least 4 \
        correct input/output examples — compute each output carefully and include an edge case. \
        IMPORTANT: \"in\" is the ARGUMENT LIST. A function that takes a single LIST has ONE array \
        argument, written as a NESTED array: {\"in\":[[1,2,3]],\"out\":6} for the sum of a list. A \
        function of two numbers is {\"in\":[3,4],\"out\":7}. Never use an empty \"in\"; include an \
        edge case such as a single-element list or a negative number. Output ONLY a JSON object, no prose: \
        {\"functions\":[{\"name\":\"snake_case\",\"description\":\"one clear sentence\",\
        \"examples\":[{\"in\":[ARG,...],\"out\":RESULT}]}]}. Give 2 to 5 functions, each with at \
        least 4 examples.";
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": format!("Request: {}", request.replace('"', "'"))}
        ],
        "temperature": 0.0,
        // Decomposition WITH per-fn examples (nested-array list args, 4+ examples each)
        // is the heaviest Mode-C variant; Gemma 4 reasons longer here, so give ample
        // headroom or the JSON answer truncates (finish=length -> unparseable).
        "max_tokens": 2048
    });
    let out = Command::new("curl")
        .args([
            "-s", "-m", "150", &url, "-H", "Content-Type: application/json", "-d", &body.to_string(),
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
    parse_decomposition_with_contracts(content)
}

/// Extract `{"functions":[{"name":..,"description":..,"examples":[{"in":..,"out":..}]}]}`
/// from the model text. Keeps only functions with >=3 valid example rows; returns
/// >=2 functions or None. Values are left as raw `serde_json::Value` — the bridge's
/// `json_to_bench_value` is the sole type authority.
fn parse_decomposition_with_contracts(content: &str) -> Option<Vec<ProposedComponentSpec>> {
    let s = content.trim();
    let start = s.find('{')?;
    let end = s.rfind('}')? + 1;
    let parsed: serde_json::Value = serde_json::from_str(&s[start..end]).ok()?;
    let arr = parsed.get("functions")?.as_array()?;
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for e in arr {
        let name = match e.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.trim().to_string(),
            None => continue,
        };
        let description = match e.get("description").and_then(|v| v.as_str()) {
            Some(d) => d.trim().to_string(),
            None => continue,
        };
        if name.is_empty() || description.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        let rows = match e.get("examples").and_then(|v| v.as_array()) {
            Some(r) => r,
            None => continue,
        };
        let mut examples = Vec::new();
        for row in rows {
            let inputs = match row.get("in").and_then(|v| v.as_array()) {
                Some(i) if !i.is_empty() => i.clone(),
                _ => continue,
            };
            let output = match row.get("out") {
                Some(o) => o.clone(),
                None => continue,
            };
            examples.push(ProposedExample { inputs, output });
        }
        // A component without enough examples cannot form a seed+holdout split.
        if examples.len() < 3 {
            continue;
        }
        out.push(ProposedComponentSpec { name, description, examples });
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
    fn training_record_is_valid_chat_format() {
        let r = training_record("count vowels in a string", "fn f(s: string) -> i64 {\n    return 0;\n}");
        let msgs = r.get("messages").and_then(|m| m.as_array()).expect("messages array");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert!(msgs[0]["content"].as_str().unwrap().contains("MOG"));
        assert!(msgs[1]["content"].as_str().unwrap().contains("count vowels"));
        // assistant answer is the fenced verified program (the label).
        let a = msgs[2]["content"].as_str().unwrap();
        assert!(a.starts_with("```mog") && a.contains("fn f(") && a.trim_end().ends_with("```"));
    }

    #[test]
    fn extract_code_from_fence_and_bare() {
        let fenced = "Here:\n```mog\nfn f(x: i64) -> i64 {\n    return x + 1;\n}\n```\ndone";
        let c = extract_code(fenced).expect("fenced");
        assert!(c.starts_with("fn f(") && c.trim_end().ends_with('}'), "got: {c}");
        // bare (no fence) — balanced-brace extraction, trailing prose dropped.
        let bare = "fn g(a: i64) -> i64 { return a; } and then some words";
        let c2 = extract_code(bare).expect("bare");
        assert_eq!(c2, "fn g(a: i64) -> i64 { return a; }");
        // nested braces close correctly.
        let nested = "```\nfn h(a: i64) -> i64 {\n    if a > 0 {\n        return 1;\n    }\n    return 0;\n}\n```";
        assert!(extract_code(nested).unwrap().contains("if a > 0"));
        assert!(extract_code("no code here").is_none());
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
    fn parse_decomposition_with_contracts_extracts_functions_and_examples() {
        // Happy path: 2 functions, 3 examples each.
        let s = "```json\n{\"functions\":[\
            {\"name\":\"sum_all\",\"description\":\"the sum of all elements\",\"examples\":[\
                {\"in\":[[1,2,3]],\"out\":6},\
                {\"in\":[[4,5]],\"out\":9},\
                {\"in\":[[0]],\"out\":0}]},\
            {\"name\":\"is_positive\",\"description\":\"whether the value is positive\",\"examples\":[\
                {\"in\":[1],\"out\":true},\
                {\"in\":[-2],\"out\":false},\
                {\"in\":[5],\"out\":true}]}]}\n```";
        let specs = parse_decomposition_with_contracts(s).expect("2 components");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "sum_all");
        assert_eq!(specs[0].examples.len(), 3);
        assert_eq!(specs[0].examples[0].inputs, vec![serde_json::json!([1, 2, 3])]);
        assert_eq!(specs[0].examples[0].output, serde_json::json!(6));
        assert_eq!(specs[1].examples[2].output, serde_json::json!(true));

        // One function only → below the >=2 floor → None.
        assert!(parse_decomposition_with_contracts(
            "{\"functions\":[{\"name\":\"a\",\"description\":\"x\",\"examples\":[\
                {\"in\":[1],\"out\":1},{\"in\":[2],\"out\":2},{\"in\":[3],\"out\":3}]}]}"
        )
        .is_none());

        // Two functions but one has only 2 examples → that fn is dropped → <2 → None.
        assert!(parse_decomposition_with_contracts(
            "{\"functions\":[\
                {\"name\":\"a\",\"description\":\"x\",\"examples\":[\
                    {\"in\":[1],\"out\":1},{\"in\":[2],\"out\":2},{\"in\":[3],\"out\":3}]},\
                {\"name\":\"b\",\"description\":\"y\",\"examples\":[\
                    {\"in\":[1],\"out\":1},{\"in\":[2],\"out\":2}]}]}"
        )
        .is_none());

        // No JSON → None.
        assert!(parse_decomposition_with_contracts("no json here").is_none());
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
