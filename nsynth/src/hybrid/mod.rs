//! The hybrid proposer: an UNTRUSTED breadth source feeding the verified funnel.
//!
//! # Why this module exists
//!
//! nCPU is sound because everything it "knows" is a *verified, gated* synthesized
//! program. The price of that soundness is breadth: the engine can only learn a
//! class (e.g. "is `wizard` a person?") if some in-repo curriculum already lists
//! the members. A large language model has the opposite profile — vast breadth,
//! zero trustworthiness. This module marries the two **without** spending any of
//! nCPU's soundness.
//!
//! The contract is simple and load-bearing: an LLM (or any other proposer) is
//! treated as an **untrusted breadth source**. It never emits code, never touches
//! the engine, and never decides what nCPU believes. All it produces is *data* — a
//! list of candidate member words and non-member words for a class. That data is
//! converted into I/O [`Example`](crate::benchmark::Example)s and handed to the
//! existing self-extension funnel
//! ([`self_extend`](crate::self_improve::extend::self_extend)), where it must:
//!
//! 1. **Synthesize + verify** — `solve_problem` only succeeds if the recovered Mog
//!    program reproduces EVERY example. A hallucinated, contradictory, or
//!    unsynthesizable proposal fails here and the engine is untouched.
//! 2. **Pass the regression gate** — even a program that verifies must not regress
//!    any of the frozen golden behavioral cases nor break a soundness probe.
//! 3. **Be journaled** — accepted or rejected, the attempt is recorded.
//! 4. **Be domain-bounded on adoption** — a learned `<x>_class` classifier answers
//!    `Yes`/`No` ONLY within its verified example domain; unseen words stay
//!    open-world UNKNOWN, so an over-claiming LLM cannot leak a false `Yes`.
//!
//! Because the LLM's output is re-verified and re-gated by the SAME machinery the
//! rest of the crate trusts, **the LLM can never make nCPU unsound.** The worst a
//! malicious or hallucinating proposer can do is waste a synthesis attempt and
//! leave a gap open — identical to a synthesis failure. This is the cryptographic-
//! style guarantee the hybrid relies on: the trust boundary is `self_extend`, and
//! a proposer lives entirely on the untrusted side of it, producing nothing past a
//! [`MembershipProposal`].
//!
//! # The funnel
//!
//! ```text
//!   Proposer (UNTRUSTED) ─► MembershipProposal ─► lexicon_examples ─► Vec<Example>
//!                                                                          │
//!                            ┌─────────────────────────────────────────────┘
//!                            ▼
//!     self_extend: synthesize+verify ─► regression_gate ─► journal ─► persist
//!                            │
//!                            ▼  (only on a green gate)
//!                    engine adopts the verified, domain-bounded classifier
//! ```
//!
//! A [`MembershipProposal`] maps directly onto
//! [`crate::comprehension::lexicon_examples`] (the canonical
//! `(members, nonmembers) -> Vec<Example>` converter that produces a well-posed
//! string→int lookup), so a proposer that returns `{positives, negatives}` plugs
//! straight into the existing string-equality teacher with no new synthesis code.
//!
//! This module currently ships the [`Proposer`] trait, the [`MembershipProposal`]
//! payload, a hermetic [`MockProposer`] for tests, and a stubbed
//! [`OpenRouterProposer`] (the real subprocess-curl implementation lands next
//! phase). Nothing here can adopt a component on its own — adoption is, and stays,
//! the exclusive job of the gated funnel.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

/// A proposed string-membership classifier spec from an untrusted breadth source.
///
/// This is the ONLY thing a [`Proposer`] is allowed to produce. It is pure data —
/// a class name plus two word lists — and becomes a string→int classifier spec by
/// feeding the member/non-member lists through
/// [`crate::comprehension::lexicon_examples`], which dedups by surface string into
/// a well-posed lookup (members → 1, non-members → 0) that the existing
/// `noun_animacy`-style teacher recovers exactly.
///
/// # Fields
///
/// * `class_name` — the class the words are proposed to (in)habit, e.g. `"person"`
///   or `"agent"` for `wizard`. A later phase derives the component name
///   (`<class_name>_class`) and signature from this.
/// * `members` — words claimed to BE in the class (each becomes a `→ 1` example),
///   e.g. `["wizard", "sorcerer", "knight"]`.
/// * `nonmembers` — words claimed NOT to be in the class (each becomes a `→ 0`
///   example), e.g. `["report", "rock", "question"]`.
///
/// # Soundness note
///
/// The proposal is *untrusted*. If the LLM lists the same word as both a member
/// and a non-member, [`crate::comprehension::lexicon_examples`] keeps the
/// first-seen (positive) label so the underlying example vec stays well-posed and
/// synthesis is not spuriously broken — but even a genuinely contradictory or
/// over-claiming proposal cannot make nCPU unsound: it either fails to synthesize
/// (gap stays open) or is caught by the regression gate, and on adoption the
/// classifier answers only within its verified domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MembershipProposal {
    pub class_name: String,
    pub members: Vec<String>,
    pub nonmembers: Vec<String>,
}

/// An untrusted breadth source that proposes membership classifier specs.
///
/// A `Proposer` is a *teacher source*, nothing more. It constructs candidate
/// [`MembershipProposal`]s and touches nothing past that struct — it never emits
/// code, never mutates the engine, and never decides adoption. Everything it
/// produces flows through the verified + gated funnel
/// ([`self_extend`](crate::self_improve::extend::self_extend)) before nCPU believes
/// any of it, so a `Proposer` can be as untrusted (and as broad) as we like.
pub trait Proposer {
    /// A stable human-readable name for this proposer, used in journal/audit
    /// trails (e.g. `"mock"`, `"openrouter:anthropic/claude-opus-4.8"`).
    fn name(&self) -> String;

    /// Propose a membership classifier spec for `word`, or `None` if this proposer
    /// has no proposal (unknown word, unavailable backend, missing credentials,
    /// malformed response). Returning `None` is always safe: the caller simply
    /// leaves the gap open, exactly as a synthesis failure would.
    fn propose_membership(&self, word: &str) -> Option<MembershipProposal>;
}

/// A deterministic, fully hermetic [`Proposer`] for tests and reproducible demos.
///
/// `MockProposer` performs NO network or subprocess I/O. It is constructed with a
/// fixed `word -> MembershipProposal` map and returns the canned proposal for a
/// queried word (or `None` if the word is not in the map). This lets the whole
/// hybrid funnel be exercised end-to-end — proposal → examples → synthesize+verify
/// → gate → adopt — without any dependency on an external model.
pub struct MockProposer {
    name: String,
    canned: BTreeMap<String, MembershipProposal>,
}

impl MockProposer {
    /// Build a `MockProposer` from a `word -> MembershipProposal` map. The map is
    /// taken by value and consulted verbatim on each query.
    pub fn new_with(canned: BTreeMap<String, MembershipProposal>) -> Self {
        Self {
            name: "mock".to_string(),
            canned,
        }
    }

    /// Build a `MockProposer` with a custom audit name (otherwise `"mock"`).
    pub fn new_named(
        name: impl Into<String>,
        canned: BTreeMap<String, MembershipProposal>,
    ) -> Self {
        Self {
            name: name.into(),
            canned,
        }
    }
}

impl Proposer for MockProposer {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn propose_membership(&self, word: &str) -> Option<MembershipProposal> {
        self.canned.get(word).cloned()
    }
}

/// An OpenRouter-backed [`Proposer`] — the real breadth source.
///
/// This is the production proposer: it asks a frontier LLM, via OpenRouter's
/// OpenAI-compatible chat-completions API, for the member and non-member words of
/// a class. It remains an *untrusted* source — its answer is data fed to the gated
/// funnel, never code — so it cannot affect nCPU's soundness.
///
/// # How it works
///
/// Following the crate's established "untrusted helper runs as a subprocess"
/// idiom (mirroring [`crate::differentiable`]'s `execute_bridge_payload`), it:
///
/// * Reads the API key from the `OPENROUTER_API_KEY` environment variable
///   in-process via [`std::env::var`]. When the key is unset or empty,
///   [`propose_membership`](Proposer::propose_membership) returns `None` (the
///   proposer is simply unavailable) — no panic, the gap stays open exactly as a
///   synthesis failure would leave it. This is what lets CI without a key just
///   skip the proposer silently.
/// * Builds an OpenAI-compatible chat-completions request (model =
///   [`model`](Self::model)) with `serde_json` and POSTs it to
///   `https://openrouter.ai/api/v1/chat/completions` via
///   `std::process::Command::new("curl")`. The HTTP stays entirely OUT of the
///   Rust crate so the dependency-light `Cargo.toml` (serde + serde_json only) is
///   preserved and there is zero new HTTP attack surface in the linked library.
/// * Keeps secrets and payload **off the process argv** so they cannot leak to
///   `ps`/`/proc`. The `Authorization: Bearer <key>` header and the request URL
///   are written into a curl *config file* fed with `--config -` (i.e. read from
///   curl's stdin), and the JSON body is fed with `--data-binary @<tempfile>`.
///   Only the tempfile path (never its contents, never the key) appears in argv.
/// * Prompts Claude for STRICT JSON
///   `{"class_name": ..., "members": [...], "nonmembers": [...]}`, extracts the
///   first balanced JSON object from `choices[0].message.content` (stripping any
///   prose or ```json code fences the model may wrap it in), parses it with
///   `serde_json`, and builds a [`MembershipProposal`].
///
/// Every failure mode — key unset/empty, curl missing or non-zero exit, network
/// error, missing/garbled response field, unparseable JSON — maps to `None`,
/// identical to a synthesis failure. This proposer **never panics and never
/// blocks adoption**; the worst it can do is return `None` and leave a gap open.
pub struct OpenRouterProposer {
    /// The OpenRouter model id to query. Defaults to `"anthropic/claude-opus-4.8"`
    /// (see [`OpenRouterProposer::new`]).
    pub model: String,
}

impl OpenRouterProposer {
    /// The default OpenRouter model id used when none is specified.
    pub const DEFAULT_MODEL: &'static str = "anthropic/claude-opus-4.8";

    /// Build an `OpenRouterProposer` for the default model
    /// ([`DEFAULT_MODEL`](Self::DEFAULT_MODEL) = `"anthropic/claude-opus-4.8"`).
    pub fn new() -> Self {
        Self {
            model: Self::DEFAULT_MODEL.to_string(),
        }
    }

    /// Build an `OpenRouterProposer` for a specific OpenRouter model id.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl Default for OpenRouterProposer {
    fn default() -> Self {
        Self::new()
    }
}

/// The OpenRouter chat-completions endpoint (OpenAI-compatible).
const OPENROUTER_CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Environment variable holding the OpenRouter API key. Unset/empty ⇒ proposer
/// is unavailable and [`OpenRouterProposer::propose_membership`] returns `None`.
const OPENROUTER_KEY_ENV: &str = "OPENROUTER_API_KEY";

impl OpenRouterProposer {
    /// The system prompt: pins Claude to the classifier-spec task and to emitting
    /// nothing but a single strict-JSON object.
    fn system_prompt() -> &'static str {
        "You are a lexical-class oracle for a program-synthesis engine. Given a \
single English WORD, identify the most natural semantic class it belongs to \
(e.g. \"person\", \"animal\", \"tool\", \"emotion\", \"vehicle\"), then list real \
English words that ARE in that same class (members) and real English words that \
are clearly OUTSIDE it (nonmembers). Reply with STRICT JSON ONLY — no prose, no \
markdown, no code fences, no explanation — matching exactly this shape: \
{\"class_name\": \"<class>\", \"members\": [\"w1\", ...], \"nonmembers\": [\"w1\", ...]}. \
Provide 5 to 8 members and 5 to 8 nonmembers. Members must be real words in the \
same class as the given word; nonmembers must be real words clearly in different \
classes. Use lowercase single words only."
    }

    /// The user prompt for `word`.
    fn user_prompt(word: &str) -> String {
        format!(
            "WORD: {word}\n\nClassify this word and return the strict-JSON \
classifier spec described above. Members should be in the SAME class as \"{word}\"."
        )
    }

    /// Build the OpenAI-compatible chat-completions request body for `word`.
    fn request_body(&self, word: &str) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "temperature": 0,
            "messages": [
                { "role": "system", "content": Self::system_prompt() },
                { "role": "user", "content": Self::user_prompt(word) },
            ],
        })
    }

    /// Drive the live call: spawn `curl`, POST the request, return the assistant
    /// message text from `choices[0].message.content`. Returns `None` on ANY
    /// failure (curl missing, non-zero exit, network error, missing field).
    ///
    /// Secrets stay off argv: the `Authorization` header and URL are written to a
    /// curl config file passed via `--config -` (curl reads it from stdin), and
    /// the JSON body is passed via `--data-binary @<tempfile>` so only a tempfile
    /// PATH (never the key, never the body) is visible to `ps`.
    fn call_model(&self, api_key: &str, word: &str) -> Option<String> {
        let body = serde_json::to_vec(&self.request_body(word)).ok()?;

        // Write the request body to a tempfile so it never appears on argv.
        let mut body_path = std::env::temp_dir();
        let unique = format!(
            "nsynth_or_{}_{}.json",
            std::process::id(),
            // Cheap monotonic-ish suffix to avoid clobbering across calls.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        body_path.push(unique);
        if std::fs::write(&body_path, &body).is_err() {
            return None;
        }

        // The curl config (read from stdin via `--config -`) carries the secret
        // header + the URL, keeping both entirely off the argv.
        let curl_config = format!(
            "url = \"{url}\"\n\
             header = \"Authorization: Bearer {key}\"\n\
             header = \"Content-Type: application/json\"\n",
            url = OPENROUTER_CHAT_URL,
            key = api_key,
        );

        let body_arg = format!("@{}", body_path.display());
        let spawn = Command::new("curl")
            .arg("-s")
            .arg("--config")
            .arg("-")
            .arg("--data-binary")
            .arg(&body_arg)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let result = (|| {
            let mut child = spawn.ok()?;
            // Feed the config (with the secret header) through stdin, then close
            // it so curl proceeds.
            if let Some(stdin) = child.stdin.as_mut() {
                stdin.write_all(curl_config.as_bytes()).ok()?;
            }
            // Dropping the child's stdin handle closes the pipe.
            drop(child.stdin.take());
            let output = child.wait_with_output().ok()?;
            if !output.status.success() {
                return None;
            }
            let resp: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
            resp.get("choices")?
                .get(0)?
                .get("message")?
                .get("content")?
                .as_str()
                .map(|s| s.to_string())
        })();

        // Best-effort cleanup; ignore errors (tempdir GC will catch leftovers).
        let _ = std::fs::remove_file(&body_path);
        result
    }
}

/// Extract the first balanced top-level JSON object substring from `content`.
///
/// The model is instructed to reply with strict JSON only, but frontier models
/// sometimes wrap it in prose or a ```json … ``` fence. This finds the first `{`,
/// then scans forward tracking brace depth (and respecting string literals +
/// escapes so braces inside strings don't confuse the count) to the matching `}`,
/// returning that slice. Returns `None` if no balanced object is present.
fn extract_json_object(content: &str) -> Option<&str> {
    let bytes = content.as_bytes();
    let start = content.find('{')?;
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return content.get(start..=i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse the assistant message text into a [`MembershipProposal`], tolerating any
/// prose/code-fence wrapping. Returns `None` if the embedded JSON object is
/// missing, unparseable, or lacks the required fields.
fn parse_proposal(content: &str) -> Option<MembershipProposal> {
    let obj = extract_json_object(content)?;
    let parsed: serde_json::Value = serde_json::from_str(obj).ok()?;

    let class_name = parsed.get("class_name")?.as_str()?.trim().to_string();
    if class_name.is_empty() {
        return None;
    }

    let as_string_vec = |v: &serde_json::Value| -> Option<Vec<String>> {
        let arr = v.as_array()?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|item| item.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Some(out)
    };

    let members = as_string_vec(parsed.get("members")?)?;
    let nonmembers = as_string_vec(parsed.get("nonmembers")?)?;

    // A proposal with no members is useless to the synthesis funnel; reject it.
    if members.is_empty() {
        return None;
    }

    Some(MembershipProposal {
        class_name,
        members,
        nonmembers,
    })
}

impl Proposer for OpenRouterProposer {
    fn name(&self) -> String {
        format!("openrouter:{}", self.model)
    }

    fn propose_membership(&self, word: &str) -> Option<MembershipProposal> {
        // Graceful no-key behavior: with the key unset OR empty, the proposer is
        // simply unavailable — CI without a key just skips it, no panic, gap open.
        let api_key = std::env::var(OPENROUTER_KEY_ENV).ok()?;
        let api_key = api_key.trim();
        if api_key.is_empty() {
            return None;
        }

        // Live call → assistant text → strict-JSON proposal. Any failure → None.
        let content = self.call_model(api_key, word)?;
        parse_proposal(&content)
    }
}

/// THE SEAM. Convert an UNTRUSTED [`MembershipProposal`] into a verified-funnel
/// [`LearnRequest`](crate::self_improve::extend::LearnRequest) — the single,
/// load-bearing adapter that lets a proposer's *data* enter the synthesize → gate
/// → adopt pipeline without granting it a single grain of trust.
///
/// The request describes a string→int binary classifier named `<class>_class`
/// (e.g. `creature_class`), built from the proposal's word lists via the canonical
/// [`crate::comprehension::lexicon_examples`] converter (members → `1`,
/// non-members → `0`). The `gap_word` — the specific token the mind could not
/// classify — is folded into the member set so the resulting classifier, once
/// synthesized + verified + gated, recognizes exactly the word the gap was about.
///
/// # Well-posedness (the dedup contract)
///
/// A proposal is untrusted, so it may be sloppy or adversarial. Before building
/// examples we make the spec **well-posed by construction**:
///
/// * Whitespace is trimmed and surface strings are compared verbatim (no case
///   folding — the lexicon teacher keys on the exact surface form the parser sees).
/// * A word that appears in BOTH `members` and `nonmembers` is genuinely
///   contradictory (the proposer claims it is and isn't in the class), so it is
///   **dropped from both lists** — not silently coerced to one label. This is
///   stricter than [`crate::comprehension::lexicon_examples`]'s first-seen
///   tie-break: we never let an untrusted proposal's internal contradiction quietly
///   pick a label, we simply refuse to learn anything from that word. The
///   `gap_word` is treated as a member intent, so if the proposer ALSO listed it as
///   a non-member that conflict drops the gap word too (and, below, yields `None` —
///   we will not synthesize a classifier that contradicts the very thing we were
///   asked to learn).
/// * Duplicate words within a single list collapse to one example.
///
/// # Honest `None` (degeneracy)
///
/// Returns `None` — leaving the gap open, exactly as a synthesis failure would —
/// when the proposal cannot yield a meaningful, well-posed binary classifier:
///
/// * the `class_name` is empty/whitespace (no component name to derive), or
/// * after the conflict-dropping above there are **no** positive members, or
/// * after the conflict-dropping there are **no** negative non-members (a
///   classifier with only positives is degenerate — the string-equality teacher
///   needs disjoint counter-examples or it overfits to a constant `1`), or
/// * the `gap_word` itself was dropped as contradictory (the proposer both
///   asserted and denied the very word the gap is about — refusing is the honest
///   move; a later, cleaner proposal can still close the gap).
///
/// Crucially, `None` here is the SAFE outcome and never the only line of defense:
/// even a non-degenerate but *wrong* proposal that slips through is still
/// re-verified by `solve_problem` and re-gated by `regression_gate` downstream, so
/// this function's degeneracy checks are an ergonomic early-out, not the soundness
/// boundary. The soundness boundary remains
/// [`self_extend`](crate::self_improve::extend::self_extend).
pub fn proposal_to_learn_request(
    p: &MembershipProposal,
    gap_word: &str,
) -> Option<crate::self_improve::extend::LearnRequest> {
    let class = p.class_name.trim();
    if class.is_empty() {
        // No class name → no component name to derive. Honest decline.
        return None;
    }
    // Sanitize the UNTRUSTED class name into a valid Mog identifier for the
    // component name. An LLM may return a multi-word class like "magic user" with
    // spaces/punctuation/capitals, none of which are legal in a Mog function-name
    // token — synthesis would fail on the malformed signature. Lowercase, collapse
    // non-alphanumeric runs to a single '_', trim edge underscores, and prefix a
    // leading digit. If nothing usable remains, decline honestly (gap stays open).
    let class_id: String = {
        let mut s: String = class
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '_'
                }
            })
            .collect();
        while s.contains("__") {
            s = s.replace("__", "_");
        }
        let s = s.trim_matches('_').to_string();
        if s.is_empty() {
            return None;
        }
        if s.chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            format!("c_{s}")
        } else {
            s
        }
    };
    let gap = gap_word.trim();

    // --- Well-posedness: drop words that appear in BOTH lists --------------
    // Collect the trimmed surface forms. The `gap_word` is folded into the
    // member intent, so it participates in conflict detection like any member.
    let members_raw: Vec<&str> = std::iter::once(gap)
        .chain(p.members.iter().map(|s| s.trim()))
        .filter(|s| !s.is_empty())
        .collect();
    let nonmembers_raw: Vec<&str> = p
        .nonmembers
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    // A word claimed both a member AND a non-member is contradictory; drop it
    // from BOTH lists so an untrusted proposal's internal contradiction can never
    // silently pick a label (stricter than lexicon_examples' first-seen tie-break).
    let member_set: std::collections::BTreeSet<&str> = members_raw.iter().copied().collect();
    let nonmember_set: std::collections::BTreeSet<&str> = nonmembers_raw.iter().copied().collect();
    let conflicts: std::collections::BTreeSet<&str> =
        member_set.intersection(&nonmember_set).copied().collect();

    // If the gap word itself was contradicted, refuse: synthesizing a classifier
    // that both asserts and denies the very word we were asked to learn is
    // ill-posed. A cleaner proposal can still close the gap later.
    if !gap.is_empty() && conflicts.contains(gap) {
        return None;
    }

    // Dedup within each list (preserving first-seen order) while excluding any
    // conflicting word entirely.
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let members: Vec<&str> = members_raw
        .iter()
        .copied()
        .filter(|w| !conflicts.contains(w) && seen.insert(w))
        .collect();
    let mut seen_n: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let nonmembers: Vec<&str> = nonmembers_raw
        .iter()
        .copied()
        .filter(|w| !conflicts.contains(w) && seen_n.insert(w))
        .collect();

    // Degenerate spec → honest None (gap stays open, identical to a synth miss).
    // A binary classifier needs at least one positive AND one disjoint negative;
    // with only positives the string-equality teacher overfits to a constant.
    if members.is_empty() || nonmembers.is_empty() {
        return None;
    }

    // --- Build the verified-funnel request --------------------------------
    // `<class>_class` is the component name convention `try_extend` / the
    // store / `recognizes_word` all key on for a bounded membership classifier.
    let name = format!("{class_id}_class");
    // The Mog signature must be `&'static str` (see `LearnRequest::signature`).
    // Following the crate's established intentional-leak idiom for grafted
    // component metadata (see `Engine::try_extend`'s `Box::leak` of the component
    // name), we leak the per-proposal signature. The number of distinct proposals
    // is bounded by the number of self-extension attempts, so this is a negligible,
    // intentional leak — never a per-call leak in a hot loop.
    let signature: &'static str =
        Box::leak(format!("fn {name}(s: string) -> i64").into_boxed_str());

    let examples = crate::comprehension::lexicon_examples(&members, &nonmembers);

    Some(crate::self_improve::extend::LearnRequest {
        gap: format!(
            "untrusted proposer: classify `{gap}` for class `{class}` ({} members, {} non-members)",
            members.len(),
            nonmembers.len(),
        ),
        name,
        signature,
        examples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_proposal() -> MembershipProposal {
        MembershipProposal {
            class_name: "person".to_string(),
            members: vec!["wizard".to_string(), "sorcerer".to_string()],
            nonmembers: vec!["report".to_string(), "rock".to_string()],
        }
    }

    #[test]
    fn mock_returns_canned_proposal_for_known_word() {
        let mut canned = BTreeMap::new();
        canned.insert("wizard".to_string(), sample_proposal());
        let proposer = MockProposer::new_with(canned);

        let got = proposer.propose_membership("wizard");
        assert_eq!(got, Some(sample_proposal()));
        assert_eq!(proposer.name(), "mock");
    }

    #[test]
    fn mock_returns_none_for_unknown_word() {
        let proposer = MockProposer::new_with(BTreeMap::new());
        assert_eq!(proposer.propose_membership("dragon"), None);
    }

    #[test]
    fn mock_honors_custom_audit_name() {
        let proposer = MockProposer::new_named("mock:test", BTreeMap::new());
        assert_eq!(proposer.name(), "mock:test");
    }

    #[test]
    fn openrouter_defaults_to_opus() {
        let proposer = OpenRouterProposer::new();
        assert_eq!(proposer.model, "anthropic/claude-opus-4.8");
        assert_eq!(proposer.name(), "openrouter:anthropic/claude-opus-4.8");
    }

    #[test]
    fn openrouter_honors_a_custom_model() {
        let proposer = OpenRouterProposer::with_model("openai/gpt-4o");
        assert_eq!(proposer.model, "openai/gpt-4o");
        assert_eq!(proposer.name(), "openrouter:openai/gpt-4o");
    }

    /// With `OPENROUTER_API_KEY` unset, the proposer is gracefully unavailable
    /// (returns `None`) — never panics, never blocks adoption. This is what lets
    /// CI without a key just skip the proposer. Mutates a process-global env var,
    /// so it must not run concurrently with the live test (which reads the key);
    /// `--test-threads` aside, the live test is `#[ignore]` by default so only one
    /// of the two ever runs in normal CI.
    #[test]
    fn openrouter_returns_none_without_a_key() {
        // SAFETY: single-threaded within this test; we restore afterward.
        let saved = std::env::var(OPENROUTER_KEY_ENV).ok();
        std::env::remove_var(OPENROUTER_KEY_ENV);

        let proposer = OpenRouterProposer::new();
        assert_eq!(
            proposer.propose_membership("wizard"),
            None,
            "no key ⇒ gracefully None"
        );

        // An explicitly empty key is also treated as unavailable.
        std::env::set_var(OPENROUTER_KEY_ENV, "   ");
        assert_eq!(
            proposer.propose_membership("wizard"),
            None,
            "blank key ⇒ gracefully None"
        );

        // Restore the environment for any sibling tests.
        match saved {
            Some(v) => std::env::set_var(OPENROUTER_KEY_ENV, v),
            None => std::env::remove_var(OPENROUTER_KEY_ENV),
        }
    }

    #[test]
    fn extract_json_object_strips_prose_and_fences() {
        // Bare object.
        assert_eq!(extract_json_object("{\"a\":1}"), Some("{\"a\":1}"));
        // Wrapped in prose.
        assert_eq!(
            extract_json_object("Sure! Here you go:\n{\"a\":1}\nHope that helps."),
            Some("{\"a\":1}")
        );
        // Wrapped in a ```json fence.
        assert_eq!(
            extract_json_object("```json\n{\"a\":1}\n```"),
            Some("{\"a\":1}")
        );
        // Nested braces + a brace inside a string literal must not confuse depth.
        assert_eq!(
            extract_json_object("noise {\"a\": {\"b\": \"}\"}} trailing"),
            Some("{\"a\": {\"b\": \"}\"}}")
        );
        // No object at all.
        assert_eq!(extract_json_object("no json here"), None);
    }

    #[test]
    fn parse_proposal_accepts_strict_json() {
        let content = r#"{"class_name": "person",
            "members": ["wizard", "sorcerer", "knight"],
            "nonmembers": ["report", "rock", "question"]}"#;
        let got = parse_proposal(content).expect("well-formed strict JSON parses");
        assert_eq!(got.class_name, "person");
        assert_eq!(got.members, vec!["wizard", "sorcerer", "knight"]);
        assert_eq!(got.nonmembers, vec!["report", "rock", "question"]);
    }

    #[test]
    fn parse_proposal_tolerates_prose_and_fences() {
        let content = "Here's the classification:\n```json\n\
            {\"class_name\":\"animal\",\"members\":[\"dog\",\"cat\"],\
             \"nonmembers\":[\"car\",\"rock\"]}\n```\nLet me know if you need more.";
        let got = parse_proposal(content).expect("fenced JSON parses");
        assert_eq!(got.class_name, "animal");
        assert_eq!(got.members, vec!["dog", "cat"]);
        assert_eq!(got.nonmembers, vec!["car", "rock"]);
    }

    #[test]
    fn parse_proposal_rejects_malformed_or_empty() {
        // Not JSON at all.
        assert_eq!(parse_proposal("I cannot help with that."), None);
        // Missing required field (no members).
        assert_eq!(
            parse_proposal(r#"{"class_name":"x","nonmembers":["a"]}"#),
            None
        );
        // Empty class name.
        assert_eq!(
            parse_proposal(r#"{"class_name":"  ","members":["a"],"nonmembers":[]}"#),
            None
        );
        // Members present but all blank ⇒ no usable members.
        assert_eq!(
            parse_proposal(r#"{"class_name":"x","members":["",""],"nonmembers":["a"]}"#),
            None
        );
        // nonmembers may be empty as long as there is at least one member.
        let got = parse_proposal(r#"{"class_name":"x","members":["a"],"nonmembers":[]}"#).unwrap();
        assert_eq!(got.members, vec!["a"]);
        assert!(got.nonmembers.is_empty());
    }

    // ===================================================================
    // proposal_to_learn_request: THE SEAM — untrusted data → funnel request.
    // Pure, hermetic, no network: these only exercise the adapter's
    // well-posedness + degeneracy logic. The synthesize+gate END of the seam
    // is tested in `understanding::mind` (learn_with_proposer accept/reject).
    // ===================================================================

    /// A well-formed proposal builds a `<class>_class` request whose examples are
    /// the deduped union of {gap_word ∪ members} → 1 and nonmembers → 0, and whose
    /// name/signature follow the bounded-classifier convention.
    #[test]
    fn proposal_builds_a_well_posed_classifier_request() {
        use crate::benchmark::Value;
        let p = MembershipProposal {
            class_name: "creature".to_string(),
            members: vec!["griffin".to_string(), "phoenix".to_string()],
            nonmembers: vec!["report".to_string(), "book".to_string()],
        };
        let req =
            proposal_to_learn_request(&p, "wizard").expect("a clean proposal must yield a request");

        assert_eq!(req.name, "creature_class");
        assert_eq!(req.signature, "fn creature_class(s: string) -> i64");

        // The gap word is folded in as a member (→ 1); nonmembers are → 0.
        let label = |w: &str| {
            req.examples
                .iter()
                .find_map(|ex| match (ex.inputs.first(), &ex.expected) {
                    (Some(Value::Str(s)), Value::Int(n)) if s == w => Some(*n),
                    _ => None,
                })
        };
        assert_eq!(label("wizard"), Some(1), "gap word must be a member");
        assert_eq!(label("griffin"), Some(1));
        assert_eq!(label("phoenix"), Some(1));
        assert_eq!(label("report"), Some(0));
        assert_eq!(label("book"), Some(0));
        // Well-posed: each surface string appears exactly once.
        let mut surfaces: Vec<&str> = req
            .examples
            .iter()
            .filter_map(|ex| match ex.inputs.first() {
                Some(Value::Str(s)) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let n = surfaces.len();
        surfaces.sort_unstable();
        surfaces.dedup();
        assert_eq!(
            surfaces.len(),
            n,
            "every example word must be unique (well-posed)"
        );
        assert_eq!(n, 5, "wizard + 2 members + 2 nonmembers");
    }

    /// A word listed as BOTH a member and a non-member is contradictory: it is
    /// dropped from BOTH lists (stricter than lexicon_examples' first-seen
    /// tie-break), so the spec never silently picks a label for it.
    #[test]
    fn proposal_drops_a_word_claimed_member_and_nonmember() {
        use crate::benchmark::Value;
        let p = MembershipProposal {
            class_name: "creature".to_string(),
            members: vec!["griffin".to_string(), "hydra".to_string()],
            // "hydra" is ALSO claimed a non-member ⇒ contradictory ⇒ dropped.
            nonmembers: vec!["hydra".to_string(), "report".to_string()],
        };
        let req = proposal_to_learn_request(&p, "wizard").expect("still well-posed after drop");
        let surfaces: Vec<&str> = req
            .examples
            .iter()
            .filter_map(|ex| match ex.inputs.first() {
                Some(Value::Str(s)) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !surfaces.contains(&"hydra"),
            "contradictory word must be dropped from both lists"
        );
        assert!(surfaces.contains(&"griffin"));
        assert!(surfaces.contains(&"report"));
        assert!(surfaces.contains(&"wizard"));
    }

    /// Degenerate proposals yield an honest `None` (gap stays open, identical to a
    /// synthesis miss): empty class, no positives, no disjoint negatives, or the
    /// gap word itself contradicted.
    #[test]
    fn proposal_declines_degenerate_specs() {
        // Empty class name → None.
        assert!(proposal_to_learn_request(
            &MembershipProposal {
                class_name: "  ".to_string(),
                members: vec!["griffin".to_string()],
                nonmembers: vec!["report".to_string()],
            },
            "wizard",
        )
        .is_none());

        // No non-members ⇒ no disjoint counter-examples ⇒ degenerate ⇒ None.
        assert!(proposal_to_learn_request(
            &MembershipProposal {
                class_name: "creature".to_string(),
                members: vec!["griffin".to_string(), "phoenix".to_string()],
                nonmembers: vec![],
            },
            "wizard",
        )
        .is_none());

        // The gap word is itself contradicted (member AND non-member) ⇒ refuse.
        assert!(proposal_to_learn_request(
            &MembershipProposal {
                class_name: "creature".to_string(),
                members: vec!["griffin".to_string()],
                nonmembers: vec!["wizard".to_string(), "report".to_string()],
            },
            "wizard",
        )
        .is_none());

        // Only positives reachable (every non-member is blank) ⇒ degenerate ⇒ None.
        assert!(proposal_to_learn_request(
            &MembershipProposal {
                class_name: "creature".to_string(),
                members: vec!["griffin".to_string()],
                nonmembers: vec!["".to_string(), "   ".to_string()],
            },
            "wizard",
        )
        .is_none());
    }

    /// Live end-to-end call against OpenRouter. Requires a real key + network, so
    /// it is `#[ignore]` by default. Run with:
    ///   `OPENROUTER_API_KEY=sk-... cargo test --release --lib hybrid -- --ignored --nocapture`
    #[test]
    #[ignore = "needs OPENROUTER_API_KEY + network"]
    fn openrouter_live_proposes_for_a_real_word() {
        if std::env::var(OPENROUTER_KEY_ENV)
            .map(|k| k.trim().is_empty())
            .unwrap_or(true)
        {
            eprintln!("skipping: OPENROUTER_API_KEY unset/empty");
            return;
        }
        let proposer = OpenRouterProposer::new();
        let proposal = proposer
            .propose_membership("wizard")
            .expect("live model should return a parseable proposal for 'wizard'");
        eprintln!("live proposal for 'wizard': {proposal:?}");
        assert!(!proposal.class_name.is_empty());
        assert!(
            !proposal.members.is_empty(),
            "members must be non-empty on a successful live call"
        );
    }
}
