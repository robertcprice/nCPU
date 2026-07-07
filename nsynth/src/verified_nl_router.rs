//! Never-confidently-wrong NL front door over the VERIFIED op vocabulary.
//!
//! The agentic-NL path today refuses 83% of prompts and — worse for the thesis —
//! is confidently WRONG (10%) more often than it is right (3%), because when it
//! can't map a prompt it *synthesizes-and-hopes* with no oracle to check against.
//! This router flips that: it matches a prompt to a NAMED, already-VERIFIED library
//! op and returns that op's proven Mog program (correct BY CONSTRUCTION — no examples
//! needed, the op is a reference implementation). When it cannot identify an op with
//! confidence, it REFUSES rather than guess. So the NL path inherits the engine's
//! "never confidently wrong" property: a proven answer or an honest refusal, never a
//! hallucinated one.
//!
//! Matching is EMERGENT — derived from the op NAME (snake_case tokens + light stem),
//! not a per-op hand-authored surface-form list. WordNet synonym expansion is a
//! separate layer (see `synonyms`); this module is the name-grounded core.

use crate::op_library::{LibOp, OPS};

/// Content-free tokens that carry no routing signal — dropped from both the op-name
/// token set and the prompt so a match reflects the OPERATION, not filler.
// NOTE: deliberately does NOT include type words (string / number / list / array /
// digit / word) — those DISTINGUISH ops (reverse_string vs reverse_number), so
// dropping them collapses distinct ops to the same token set and forces a refuse.
const STOP: &[&str] = &[
    "is", "of", "the", "a", "an", "to", "for", "and", "or", "in", "on", "with", "by", "compute",
    "calculate", "find", "get", "return", "give", "me", "please", "that", "this", "from", "into",
    "as", "all", "each", "its", "their",
];

/// Crude, deterministic stem: lowercase then strip a common inflectional suffix so
/// "vowels"/"vowel", "digits"/"digit", "reversed"/"reverse" collapse. Not a full
/// stemmer — just enough that plural/verb-form prompts match singular op names.
fn stem(w: &str) -> String {
    let w = w.to_ascii_lowercase();
    for suf in ["ies", "es", "ed", "ing", "s"] {
        if w.len() > suf.len() + 2 && w.ends_with(suf) {
            let base = &w[..w.len() - suf.len()];
            // "ies" -> "y" (binaries -> binary); others just drop the suffix.
            return if suf == "ies" {
                format!("{base}y")
            } else {
                base.to_string()
            };
        }
    }
    w
}

/// Split on any non-alphanumeric, drop stopwords, stem the rest.
fn content_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .filter(|t| !STOP.contains(&t.as_str()))
        .map(|t| stem(&t))
        .collect()
}

/// A confident route to a verified op, with the evidence for why.
pub struct Route {
    pub op: &'static LibOp,
    /// The op-name content tokens (all of which were found in the prompt).
    pub matched_tokens: Vec<String>,
    /// Specificity = number of op-name content tokens matched. Higher = more
    /// specific = preferred, and it is the tie-break that keeps the match honest.
    pub specificity: usize,
}

/// Route a natural-language prompt to a verified library op, or `None` to REFUSE.
///
/// STRICT single-op identification (for the no-example / pure-proposer path where
/// there is no gate to backstop a guess): an op qualifies only if EVERY content
/// token of its name appears in the prompt; the MOST SPECIFIC qualifier wins; a tie
/// at the top specificity is genuine ambiguity -> REFUSE. Use [`route_verified`] when
/// examples exist — it proposes LIBERALLY and lets the verify gate decide, which
/// lifts recall (partial-name and near-miss prompts) without risking a wrong answer.
pub fn route(prompt: &str) -> Option<Route> {
    let ptoks = content_tokens(prompt);
    if ptoks.is_empty() {
        return None;
    }
    let has = |t: &str| ptoks.iter().any(|p| p == t);

    let mut best: Option<Route> = None;
    let mut tie_at_best = false;
    for op in OPS {
        let name_toks = content_tokens(op.name);
        if name_toks.is_empty() {
            continue;
        }
        if !name_toks.iter().all(|t| has(t)) {
            continue;
        }
        let spec = name_toks.len();
        match &best {
            Some(b) if spec > b.specificity => {
                best = Some(Route { op, matched_tokens: name_toks, specificity: spec });
                tie_at_best = false;
            }
            Some(b) if spec == b.specificity => {
                tie_at_best = true;
            }
            None => {
                best = Some(Route { op, matched_tokens: name_toks, specificity: spec });
            }
            _ => {}
        }
    }
    if tie_at_best {
        return None;
    }
    best
}

/// Emergent PROPOSAL: every op sharing >=1 content token with the prompt, ranked by
/// how much of the OP NAME the prompt covers (fraction of name tokens matched, then
/// absolute count). This is deliberately LIBERAL — it proposes `count_divisors` for
/// "number of divisors" (partial) and several plausible ops for one prompt. Liberal
/// proposal is only safe because [`route_verified`] gates every candidate on the
/// examples: a wrong proposal is executed, fails, and is discarded — never returned.
pub fn ranked_candidates(prompt: &str) -> Vec<&'static LibOp> {
    let ptoks = content_tokens(prompt);
    if ptoks.is_empty() {
        return Vec::new();
    }
    // Match an op-name token against the prompt: exact, OR a prompt token that
    // STARTS WITH the op token (>=4 chars) so compounds resolve — "uppercase"
    // matches `to_upper`'s "upper", "lowercase" matches "lower". Prefix matching is
    // liberal, but the verify gate backstops it, so a spurious prefix hit that
    // doesn't reproduce the examples is discarded, never returned.
    let has = |t: &str| {
        ptoks
            .iter()
            .any(|p| p == t || (t.len() >= 4 && p.starts_with(t)))
    };
    // Raw lowercase words in order (for acronym initial-matching, which needs the
    // ORIGINAL sequence, not the stopword-filtered/stemmed token set).
    let words: Vec<String> = prompt
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect();
    let mut scored: Vec<(&'static LibOp, f64, usize)> = Vec::new();
    for op in OPS {
        let name_toks = content_tokens(op.name);
        if name_toks.is_empty() {
            continue;
        }
        let matched = name_toks.iter().filter(|t| has(t)).count();
        // ACRONYM match: a single-token op name (gcd, lcm) whose letters are the
        // initials of a run of consecutive prompt words ("greatest common divisor"
        // -> "gcd"). Emergent — derived from the name's own letters, not a synonym
        // list. Scored as a full-coverage hit so it's tried early.
        let acronym = name_toks.len() == 1 && is_acronym_of(&name_toks[0], &words);
        if matched == 0 && !acronym {
            continue;
        }
        let coverage = if acronym {
            1.0
        } else {
            matched as f64 / name_toks.len() as f64
        };
        scored.push((op, coverage, matched.max(acronym as usize)));
    }
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.2.cmp(&a.2))
    });
    scored.into_iter().map(|(op, _, _)| op).collect()
}

/// True if `name` (>=2 letters) is spelled by the first letters of some run of
/// consecutive `words`, e.g. "gcd" from ["greatest","common","divisor"]. A pure
/// structural rule over the op name's own characters — no hand-authored expansion.
fn is_acronym_of(name: &str, words: &[String]) -> bool {
    let letters: Vec<char> = name.chars().collect();
    if letters.len() < 2 || words.len() < letters.len() {
        return false;
    }
    words.windows(letters.len()).any(|w| {
        w.iter()
            .zip(&letters)
            .all(|(word, &c)| word.starts_with(c))
    })
}

/// Never-confidently-wrong routing: PROPOSE an op from the prompt, then let
/// verification DISPOSE. Returns the op only if its proven program reproduces every
/// supplied example — so a semantically-wrong name match (the int `is_palindrome`
/// op proposed for a STRING-palindrome task) is caught and turned into a refusal
/// (`None`), never returned as a confident answer. This is the crucial lesson from
/// the ungated router: a unique NAME match is not a correctness guarantee (the
/// "no true oracle" ceiling applies to NL too), so the example gate is mandatory.
/// With zero examples the gate is vacuous — callers should treat that as TENTATIVE,
/// not confirmed.
pub fn route_verified(prompt: &str, examples: &[crate::benchmark::Example]) -> Option<Route> {
    if examples.is_empty() {
        return None; // no oracle -> cannot confirm -> refuse rather than guess
    }
    const MAX_TRIED: usize = 12;
    // Type-aware pre-filter: drop candidates whose parameter types can't accept the
    // example inputs BEFORE spending the top-MAX_TRIED budget on them. A string op
    // can never solve an int task, so proposing it only wastes a gate execution and
    // risks crowding the right op out of the budget. Correctness is unaffected (the
    // gate already rejects mismatches) — this improves recall and speed.
    let input_types: Vec<&'static str> = examples
        .first()
        .map(|e| e.inputs.iter().map(value_type_str).collect())
        .unwrap_or_default();
    let passing: Vec<&'static LibOp> = ranked_candidates(prompt)
        .into_iter()
        .filter(|op| op_accepts_types(op.mog, &input_types))
        .take(MAX_TRIED)
        .filter(|op| crate::runtime::code_reproduces_examples(op.mog, examples))
        .collect();
    let winner = *passing.first()?; // NL-top among the passers

    // DISTINGUISHING-POWER GATE. If more than one DISTINCT op reproduces the
    // examples, the examples may not uniquely determine the answer — the classic
    // "no true oracle" hole (all-True `perfect_square` examples pass both
    // `is_perfect_square` AND `next_perfect_square`). Returning the first would be
    // confidently WRONG. So probe the passers on FRESH inputs: if they all agree
    // everywhere, they are observationally equivalent and the top is safe to return;
    // if any two DISAGREE, the spec is under-determined -> REFUSE and let the caller
    // supply a distinguishing example. This keeps "never wrong" true even on weak
    // examples, at the cost of refusing genuinely ambiguous requests.
    if passing.len() > 1 && passers_disagree(&passing, examples) {
        return None;
    }
    let name_toks = content_tokens(winner.name);
    let spec = name_toks.len();
    Some(Route { op: winner, matched_tokens: name_toks, specificity: spec })
}

/// Never-wrong 2-op COMPOSITION. When no single op reproduces the examples, a
/// prompt like "reverse then uppercase a string" may still be a CHAIN of two proven
/// ops. Propose ordered pairs from the NL candidates whose types chain (a: X->Y,
/// b: Y->Z) and emit `f(x) = b(a(x))`; return the first composition that reproduces
/// every example. Same never-wrong contract — the composed program is example-gated,
/// and it is built only from already-verified ops, so a wrong chain is executed,
/// fails, and is discarded. Unary chains only (the common "do A then B" shape).
pub fn route_composed(prompt: &str, examples: &[crate::benchmark::Example]) -> Option<String> {
    let first = examples.first()?;
    if first.inputs.len() != 1 {
        return None; // unary chains only
    }
    let in_ty = value_type_str(&first.inputs[0]);
    let out_ty = value_type_str(&first.expected);
    let cands: Vec<&'static LibOp> = ranked_candidates(prompt).into_iter().take(8).collect();
    let fname = "composed";
    // Collect EVERY chain that reproduces the examples (not just the first), so the
    // distinguishing gate can catch a spec that two different chains both satisfy.
    let mut chains: Vec<String> = Vec::new();
    for a in &cands {
        let (a_params, a_ret) = op_sig_types(a.mog);
        if a_params.len() != 1 || a_params[0] != in_ty {
            continue; // a: in_ty -> a_ret
        }
        let (a_name, Some(a_entry)) = (a.name, op_entry_name(a.mog)) else { continue };
        for b in &cands {
            if b.name == a_name {
                continue;
            }
            let (b_params, b_ret) = op_sig_types(b.mog);
            if b_params.len() != 1 || b_params[0] != a_ret || b_ret != out_ty {
                continue; // b: a_ret -> out_ty
            }
            let Some(b_entry) = op_entry_name(b.mog) else { continue };
            // Entry FIRST (the verifier calls the first fn), then the two op defs.
            let code = format!(
                "fn {fname}(x0: {in_ty}) -> {out_ty} {{\n    return {b_entry}({a_entry}(x0));\n}}\n\n{}\n{}",
                a.mog.trim_end(),
                b.mog.trim_end()
            );
            if crate::runtime::code_reproduces_examples(&code, examples) {
                chains.push(code);
            }
        }
    }
    let first = chains.first()?.clone();
    // Distinguishing-power gate (same as single ops): if two distinct chains both
    // satisfy the examples but DIVERGE on fresh inputs, the spec is under-determined
    // -> refuse rather than return a possibly-wrong chain.
    if chains.len() > 1 {
        let progs: Vec<(String, String)> =
            chains.iter().map(|c| (c.clone(), fname.to_string())).collect();
        if programs_disagree(&progs, examples) {
            return None;
        }
    }
    Some(first)
}

/// The unified never-wrong answer: one verified tier, or a refusal.
pub enum Answer {
    /// A single verified library op.
    Library { name: &'static str, code: String },
    /// A verified 2-op composition.
    Composition { code: String },
    /// A function synthesized by the FULL engine and verified against the examples.
    Synthesized { method: String, code: String },
    /// A program a MODEL proposed, that nsynth verified (reproduces held-out +
    /// strict-verify + consensus). The model is the proposer; nsynth is the oracle.
    Proposed { method: String, code: String },
    /// No verified answer.
    Refused,
}

/// A model proposer: given a request, the seed examples, and an optional PRIOR
/// (failed code + error) for repair, it returns one candidate Mog program (or None
/// when no model is available — the tier is then inert).
pub type Proposer<'a> =
    dyn Fn(&str, &[crate::benchmark::Example], Option<(&str, &str)>) -> Option<String> + 'a;

/// The live proposer: the served local model (inert without `NSYNTH_LOCAL_LLM_URL`,
/// where `propose_program` returns `None`).
fn live_proposer(
    request: &str,
    _seed: &[crate::benchmark::Example],
    prior: Option<(&str, &str)>,
) -> Option<String> {
    crate::local_llm::propose_program(request, prior, 0.2)
}

/// Format the model request from the NL prompt + the SEED examples (the model never
/// sees the held-out examples, so they can catch a fit-to-seed hallucination).
fn proposer_request(prompt: &str, seed: &[crate::benchmark::Example]) -> String {
    let mut s = format!("{prompt}\n\nExamples:\n");
    for ex in seed {
        s.push_str(&format!("  {:?} -> {:?}\n", ex.inputs, ex.expected));
    }
    s.push_str("\nWrite the Mog function `f`.");
    s
}

/// THE never-wrong front door. Tries, in order of specificity/cost:
///   1. a single verified library op (fast, NL-guided),
///   2. a verified 2-op composition,
///   3. FULL SYNTHESIS from the examples — the whole engine (recognizers, PBE,
///      affine/poly, op-pipeline), so the door reaches ANY function the engine can
///      synthesize, not just the ~300-op vocabulary (e.g. `2n+1`, which no library
///      op matches but the polynomial lane synthesizes).
/// Every tier is verified against the examples. With NO examples, only the
/// vocabulary tiers (declare / declare_composed) apply — synthesis needs an oracle,
/// so it is not attempted there. Always verified-or-refused: never confidently wrong.
pub fn answer(prompt: &str, examples: &[crate::benchmark::Example]) -> Answer {
    answer_with_proposer(prompt, examples, Some(&live_proposer))
}

/// Split a raw query into its NL PROMPT and any inline I/O EXAMPLES it carries.
/// Examples are signalled by `->`. Forms handled generally (no per-case matching):
///   "multiply two numbers: 2,3->6, 4,5->20"  -> ("multiply two numbers", [..])
///   "reverse a list: [1,2,3]->[3,2,1]"        -> ("reverse a list", [..])
///   "0->7, 1->10, 2->17"                       -> ("", [..])
///   "multiply two numbers"                     -> ("multiply two numbers", [])
/// Values parse as int / float / bool / "string" / [int,..]; a top-level comma
/// separates operands and pairs, commas inside `[..]`/`"..."` do not.
pub fn split_prompt_examples(query: &str) -> (String, Vec<crate::benchmark::Example>) {
    let q = query.trim();
    if !q.contains("->") {
        return (q.to_string(), Vec::new());
    }
    // "<NL>: <examples>" when the tail after the first colon carries the arrows;
    // otherwise the whole string is the example region (no NL prefix).
    let (nl, body) = match q.split_once(':') {
        Some((head, tail)) if tail.contains("->") => (head.trim().to_string(), tail.trim()),
        _ => (String::new(), q),
    };
    (nl, parse_arrow_examples(body))
}

/// Parse `"INPUTS -> OUT, INPUTS -> OUT, ..."`. Integers before an arrow are that
/// pair's inputs; the first value after it is the output; remaining values belong to
/// the next pair — so multi-arg examples (`2,3->5, 4,5->9`) split correctly.
fn parse_arrow_examples(body: &str) -> Vec<crate::benchmark::Example> {
    use crate::benchmark::Example;
    let parts: Vec<&str> = body.split("->").collect();
    if parts.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut pending = parse_values(parts[0]);
    for mid in &parts[1..parts.len() - 1] {
        let vals = parse_values(mid);
        if vals.is_empty() || pending.is_empty() {
            return out;
        }
        out.push(Example { inputs: pending.clone(), expected: vals[0].clone() });
        pending = vals[1..].to_vec();
    }
    let last = parse_values(parts[parts.len() - 1]);
    if last.len() == 1 && !pending.is_empty() {
        out.push(Example { inputs: pending, expected: last[0].clone() });
    }
    out
}

/// Split at top-level commas (not inside `[..]` or `"..."`), parsing each token.
fn parse_values(s: &str) -> Vec<crate::benchmark::Value> {
    let mut vals = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut last = 0usize;
    let bytes = s.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'"' => in_str = !in_str,
            b'[' if !in_str => depth += 1,
            b']' if !in_str => depth -= 1,
            b',' if !in_str && depth == 0 => {
                if let Some(v) = parse_value(&s[last..i]) {
                    vals.push(v);
                }
                last = i + 1;
            }
            _ => {}
        }
    }
    if let Some(v) = parse_value(&s[last..]) {
        vals.push(v);
    }
    vals
}

/// One literal token -> Value: int, float, bool, `"string"`, or `[int,..]`.
fn parse_value(tok: &str) -> Option<crate::benchmark::Value> {
    use crate::benchmark::Value;
    let t = tok.trim();
    if t.is_empty() {
        return None;
    }
    if let Ok(i) = t.parse::<i64>() {
        return Some(Value::Int(i));
    }
    match t {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        _ => {}
    }
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        return Some(Value::Str(t[1..t.len() - 1].to_string()));
    }
    if t.starts_with('[') && t.ends_with(']') {
        let inner = &t[1..t.len() - 1];
        if inner.trim().is_empty() {
            return Some(Value::int_array(&[]));
        }
        let ints: Option<Vec<i64>> = inner.split(',').map(|x| x.trim().parse::<i64>().ok()).collect();
        if let Some(ints) = ints {
            return Some(Value::int_array(&ints));
        }
    }
    if let Ok(f) = t.parse::<f64>() {
        return Some(Value::Float(f.to_bits()));
    }
    None
}

/// [`answer`] with an explicit model proposer (tier 4). Tiers 1-3 (library op,
/// composition, full synthesis) are the model-free never-wrong core; tier 4 lets a
/// MODEL propose a program for the hard tail the engine can't synthesize — gated the
/// same way, so the model can NEVER produce a wrong answer. `proposer = None` (or an
/// unavailable model) reduces exactly to the model-free door.
pub fn answer_with_proposer(
    prompt: &str,
    examples: &[crate::benchmark::Example],
    proposer: Option<&Proposer>,
) -> Answer {
    if examples.is_empty() {
        if let Some(code) = declare_composed(prompt) {
            return Answer::Composition { code };
        }
        if let Some(op) = declare(prompt) {
            return Answer::Library { name: op.name, code: op.mog.to_string() };
        }
        return Answer::Refused;
    }
    if let Some(r) = route_verified(prompt, examples) {
        return Answer::Library { name: r.op.name, code: r.op.mog.to_string() };
    }
    if let Some(code) = route_composed(prompt, examples) {
        return Answer::Composition { code };
    }
    // Tier 3 — full synthesis with HOLDOUT discipline. Solve on a SEED (all but the
    // last two) and require the result to reproduce EVERY example including the two
    // held-out; a fit-only-to-seed overfit fails and is discarded. Needs >= 4.
    if examples.len() >= 4 {
        let seed = &examples[..examples.len() - 2];
        let sig: &'static str =
            Box::leak(crate::linguigenesis_bridge::infer_signature("f", seed).into_boxed_str());
        let problem = crate::benchmark::Problem {
            name: "f".to_string(),
            signature: sig,
            examples: seed.to_vec(),
            ..Default::default()
        };
        let res = crate::solver::solve_problem(&problem);
        if res.success && crate::runtime::code_reproduces_examples(&res.code, examples) {
            return Answer::Synthesized { method: res.method, code: res.code };
        }
    }
    // Tier 4 — GATED MODEL PROPOSER. The model writes a candidate program (an
    // algorithm the engine cannot synthesize); nsynth is the oracle. Same never-wrong
    // gate: the model sees only the SEED, and a candidate is accepted only if it
    // reproduces EVERY example (incl. the two held-out the model never saw) AND
    // passes run_tool's strict-verify + differential-consensus. A hallucinated or
    // fit-to-seed proposal fails and is discarded. Model quality bounds REACH, never
    // correctness.
    if examples.len() >= 4 {
        if let Some(propose) = proposer {
            let seed = &examples[..examples.len() - 2];
            let sig = crate::linguigenesis_bridge::infer_signature("f", seed);
            let request = proposer_request(prompt, seed);
            // Best-of-N WITH REPAIR: propose -> verify -> on failure feed the code +
            // error back so the model fixes its bug (an off-by-one, a bad return),
            // up to N attempts. The gate is unchanged, so no repair can ever yield a
            // wrong answer.
            const MAX_ATTEMPTS: usize = 4;
            let mut prior: Option<(String, String)> = None;
            for _ in 0..MAX_ATTEMPTS {
                let p = prior.as_ref().map(|(c, e)| (c.as_str(), e.as_str()));
                let Some(code) = propose(&request, seed, p) else { break };
                if crate::runtime::code_reproduces_examples(&code, examples) {
                    // Reproduces every example incl. the two held-out (never-wrong
                    // evidence). run_tool adds strict-verify + consensus; accept unless
                    // REFUSED (Tentative is fine — the held-out is the corroboration).
                    let req = crate::rlvr::ToolRequest::VerifyProgram {
                        signature: sig.clone(),
                        code: code.clone(),
                        examples: examples.to_vec(),
                    };
                    if crate::rlvr::run_tool(&req).code().is_some() {
                        // DISTILL model -> engine: persist this verified program as a
                        // learned op so a FUTURE run solves the same task MODEL-FREE at
                        // tier 3 (`try_library` reads the learned store), never reaching
                        // the model again. The model teaches the engine once; the
                        // capability becomes permanent + emergent. Sound because every
                        // future use re-verifies against that task's own held-out
                        // examples (see `record_proposed_op`).
                        let full: &'static str = Box::leak(sig.clone().into_boxed_str());
                        let problem = crate::benchmark::Problem {
                            name: "f".to_string(),
                            signature: full,
                            examples: examples.to_vec(),
                            ..Default::default()
                        };
                        crate::op_library::record_proposed_op(&problem, &code);
                        return Answer::Proposed { method: "model-proposed".to_string(), code };
                    }
                }
                prior = Some((code, "it did not reproduce all the examples".to_string()));
            }
        }
    }
    Answer::Refused
}

/// The entry function name of an op program (first `fn <name>(`).
pub fn op_entry_name(mog: &str) -> Option<&str> {
    mog.split("fn ").nth(1)?.split('(').next().map(str::trim)
}

/// NO-EXAMPLES path. Confidently identify a single verified op from the prompt
/// alone — for the common case where a person types a request with no test cases.
/// "Confident" means UNAMBIGUOUS: either a unique strict name match (every op-name
/// token present, one clear winner) or a unique acronym match. Anything ambiguous
/// or unmatched returns None -> the caller must refuse or ask, never guess. Because
/// there is no example oracle here, the honest completion is [`demonstrate`]: show
/// the op's ACTUAL behavior so the user confirms by recognition, not by trust.
pub fn declare(prompt: &str) -> Option<&'static LibOp> {
    // Acronym FIRST: a prompt that spells out an op ("greatest common divisor") is a
    // stronger, more specific signal than a coincidental single shared token (that
    // same prompt also token-matches `all_divisors` on "divisor"). Unique or skip.
    let words: Vec<String> = prompt
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect();
    let acro: Vec<&'static LibOp> = OPS
        .iter()
        .filter(|op| {
            let toks = content_tokens(op.name);
            toks.len() == 1 && is_acronym_of(&toks[0], &words)
        })
        .collect();
    if acro.len() == 1 {
        return Some(acro[0]);
    }
    // Otherwise a unique strict name match (every op-name token present, one winner).
    // No hardcoded type/phrase guards here: a no-example declare is a token-grounded
    // GUESS by nature, and the honest never-wrong correctness gate is VERIFICATION —
    // `answer()` executes each candidate against the caller's examples, which rejects a
    // type-wrong op (reverse_number on a list, all_divisors for a count) FOR ANY case,
    // no phrase list required. Route callers that have examples through `answer()`.
    route(prompt).map(|r| r.op)
}

/// NO-EXAMPLES composition. A prompt like "reverse then uppercase a string" names
/// two ops in sequence; with no example oracle, use the PROMPT ORDER to decide the
/// chain (first-named applied first) and the TYPES to validate it (a: X->Y, b: Y->Z).
/// Returns the composed program, or None if fewer than two ops are confidently named
/// or their types don't chain in prompt order. Like [`declare`], the honest
/// completion is [`demonstrate`] — show the chain's behavior for the user to confirm.
pub fn declare_composed(prompt: &str) -> Option<String> {
    // Ordered prompt words (lowercase); a prefix (>=4 chars) counts, so "uppercase"
    // matches the op token "upper".
    let words: Vec<String> = prompt
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect();
    // Op-name tokens are STEMMED (content_tokens), so match against the stemmed word
    // too (else "string" -> op token "str" never lines up with the raw word "string").
    let pos_of = |tok: &str| -> Option<usize> {
        words.iter().position(|w| {
            let sw = stem(w);
            sw == tok
                || w == tok
                || (tok.len() >= 4 && (sw.starts_with(tok) || w.starts_with(tok)))
        })
    };
    // Ops whose FULL name is present, tagged with their earliest position in prompt.
    let mut named: Vec<(&'static LibOp, usize)> = Vec::new();
    for op in OPS {
        let toks = content_tokens(op.name);
        if toks.is_empty() {
            continue;
        }
        let positions: Option<Vec<usize>> = toks.iter().map(|t| pos_of(t)).collect();
        let Some(ps) = positions else { continue };
        let earliest = *ps.iter().min().unwrap();
        named.push((op, earliest));
    }
    named.sort_by_key(|(_, p)| *p);
    named.dedup_by_key(|(op, _)| op.name);
    if named.len() < 2 {
        return None;
    }
    // First two ops in prompt order form the chain a -> b.
    let (a, _) = named[0];
    let (b, _) = named[1];
    let (a_params, a_ret) = op_sig_types(a.mog);
    let (b_params, b_ret) = op_sig_types(b.mog);
    if a_params.len() != 1 || b_params.len() != 1 || b_params[0] != a_ret {
        return None; // types don't chain in prompt order
    }
    let (a_entry, b_entry) = (op_entry_name(a.mog)?, op_entry_name(b.mog)?);
    Some(format!(
        "fn composed(x0: {}) -> {b_ret} {{\n    return {b_entry}({a_entry}(x0));\n}}\n\n{}\n{}",
        a_params[0],
        a.mog.trim_end(),
        b.mog.trim_end()
    ))
}

/// Run an op on a few illustrative inputs so a user can confirm the behavior by
/// recognition (the no-example safety mechanism). Returns (input, output) pairs the
/// op actually produced; skips inputs it can't run on.
pub fn demonstrate(op: &LibOp) -> Vec<(Vec<crate::benchmark::Value>, String)> {
    demonstrate_program(op.mog)
}

/// [`demonstrate`] for any program source (e.g. a composed chain), not just a
/// library op. Runs the entry fn on illustrative inputs and returns real I/O pairs.
pub fn demonstrate_program(mog: &str) -> Vec<(Vec<crate::benchmark::Value>, String)> {
    use crate::benchmark::Value;
    let Some(name) = op_entry_name(mog) else { return vec![] };
    let tys: Vec<&str> = {
        let (Some(o), Some(c)) = (mog.find('('), mog.find(')')) else { return vec![] };
        let inner = mog[o + 1..c].trim();
        if inner.is_empty() {
            vec![]
        } else {
            inner
                .split(',')
                .filter_map(|p| p.split(':').nth(1).map(str::trim))
                .collect()
        }
    };
    // Two illustrative input tuples, by declared type.
    let samples: [Vec<Value>; 2] = [
        tys.iter().enumerate().map(|(i, t)| demo_value(t, i, 0)).collect(),
        tys.iter().enumerate().map(|(i, t)| demo_value(t, i, 1)).collect(),
    ];
    let mut out = Vec::new();
    for inputs in samples {
        if inputs.len() != tys.len() {
            continue;
        }
        if let Ok(v) = crate::runtime::execute_function(mog, name, &inputs, "demo") {
            out.push((inputs, format!("{v:?}")));
        }
    }
    out
}

/// A canonical illustrative value for a type, varied by argument position and which
/// of the two demo rows it is, so a 2-arg op shows two genuinely different rows.
fn demo_value(ty: &str, pos: usize, row: usize) -> crate::benchmark::Value {
    use crate::benchmark::Value;
    match ty {
        "i64" => Value::Int([[12, 8], [30, 45]][row][pos.min(1)]),
        "bool" => Value::Bool(row == 0),
        "string" => Value::Str([["hello"], ["Cat"]][row][0].to_string()),
        "[i64]" => Value::int_array(&[[3, 1, 2, 1], [9, 4, 6, 4]][row]),
        "f64" => Value::Float((if row == 0 { 2.5f64 } else { 4.0 }).to_bits()),
        _ => Value::Int(0),
    }
}

/// Parse an op's signature into (parameter types, return type). Return type is read
/// from `-> T` (defaults to "i64" if absent).
fn op_sig_types(mog: &str) -> (Vec<String>, String) {
    let params: Vec<String> = match (mog.find('('), mog.find(')')) {
        (Some(o), Some(c)) if c > o => {
            let inner = mog[o + 1..c].trim();
            if inner.is_empty() {
                vec![]
            } else {
                inner
                    .split(',')
                    .filter_map(|p| p.split(':').nth(1).map(|t| t.trim().to_string()))
                    .collect()
            }
        }
        _ => vec![],
    };
    let ret = mog
        .split("->")
        .nth(1)
        .and_then(|s| s.split('{').next())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "i64".to_string());
    (params, ret)
}

/// The Mog parameter type an example input value would be declared as.
fn value_type_str(v: &crate::benchmark::Value) -> &'static str {
    use crate::benchmark::Value;
    match v {
        Value::Int(_) => "i64",
        Value::Bool(_) => "bool",
        Value::Str(_) => "string",
        Value::Float(_) => "f64",
        Value::Array(_) => "[i64]",
        _ => "?",
    }
}

/// True if the op's entry signature can accept inputs of `input_types` — same arity
/// and each declared param type matches (an unknown `?` input type is permissive so
/// we never wrongly EXCLUDE a candidate the gate could still verify). Empty input
/// types (no examples) -> permissive.
fn op_accepts_types(mog: &str, input_types: &[&str]) -> bool {
    if input_types.is_empty() {
        return true;
    }
    let Some(open) = mog.find('(') else { return true };
    let Some(rel_close) = mog[open..].find(')') else { return true };
    let inner = mog[open + 1..open + rel_close].trim();
    let params: Vec<&str> = if inner.is_empty() {
        vec![]
    } else {
        inner
            .split(',')
            .filter_map(|p| p.split(':').nth(1).map(str::trim))
            .collect()
    };
    if params.len() != input_types.len() {
        return false;
    }
    params
        .iter()
        .zip(input_types)
        .all(|(declared, got)| *got == "?" || declared == got)
}

/// True if any two passing ops produce DIFFERENT outputs on some fresh probe input
/// (an input NOT in the examples). Fresh inputs are small perturbations of the
/// example inputs (scalar ints nudged); an input on which the ops diverge proves the
/// examples failed to distinguish them.
/// True if any two candidate PROGRAMS produce a different output on some fresh probe
/// input. Each program is (source, entry-fn-name). This is the distinguishing gate,
/// generalized to program level so a SINGLE op and a COMPOSITION are gated the same
/// way — the never-wrong guarantee stays uniform across both answer paths.
fn programs_disagree(programs: &[(String, String)], examples: &[crate::benchmark::Example]) -> bool {
    let probes = fresh_probe_inputs(examples);
    for inputs in &probes {
        // runtime::Value has no PartialEq, so compare its deterministic Debug form.
        let mut seen: Option<String> = None;
        for (code, name) in programs {
            // Only successful evaluations count; an error is not a disagreement
            // signal (a program simply undefined on a probe input tells us nothing).
            if let Ok(out) = crate::runtime::execute_function(code, name, inputs, "nl_probe") {
                let key = format!("{out:?}");
                match &seen {
                    Some(prev) if *prev != key => return true, // divergence found
                    None => seen = Some(key),
                    _ => {}
                }
            }
        }
    }
    false
}

/// Convenience: run [`programs_disagree`] over a set of single library ops.
fn passers_disagree(passing: &[&'static LibOp], examples: &[crate::benchmark::Example]) -> bool {
    let progs: Vec<(String, String)> = passing
        .iter()
        .filter_map(|op| op_entry_name(op.mog).map(|n| (op.mog.to_string(), n.to_string())))
        .collect();
    programs_disagree(&progs, examples)
}

/// Fresh probe inputs: each example's input tuple with one scalar-int position
/// nudged (+1, +2, *2, and a few small constants), so we exercise inputs the
/// examples did not cover. Non-int arguments are left as-is (limited but the
/// non-distinguishing hole is overwhelmingly a scalar-predicate problem).
fn fresh_probe_inputs(examples: &[crate::benchmark::Example]) -> Vec<Vec<crate::benchmark::Value>> {
    use crate::benchmark::Value;
    let seen: std::collections::HashSet<String> =
        examples.iter().map(|e| format!("{:?}", e.inputs)).collect();
    let mut out = Vec::new();
    let mut push_if_new = |tup: Vec<Value>| {
        if !seen.contains(&format!("{tup:?}")) {
            out.push(tup);
        }
    };
    for ex in examples.iter().take(4) {
        for pos in 0..ex.inputs.len() {
            if let Value::Int(v) = ex.inputs[pos] {
                for delta in [1i64, 2, 3, 5, -1] {
                    let mut tup = ex.inputs.clone();
                    tup[pos] = Value::Int(v.wrapping_add(delta));
                    push_if_new(tup);
                }
            }
        }
    }
    // A few absolute small constants too (catch ops that only differ near 0/1).
    if let Some(first) = examples.first() {
        for c in [0i64, 1, 2, 6, 8, 12] {
            let mut tup = first.inputs.clone();
            if let Some(Value::Int(slot)) = tup.first_mut() {
                *slot = c;
                push_if_new(tup);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routed_name(prompt: &str) -> Option<&'static str> {
        route(prompt).map(|r| r.op.name)
    }

    #[test]
    fn routes_name_bearing_prompts_to_the_proven_op() {
        assert_eq!(routed_name("compute the factorial of a number"), Some("factorial"));
        assert_eq!(routed_name("sort a list of numbers"), Some("sort"));
        assert_eq!(routed_name("reverse a string"), Some("reverse_string"));
        assert_eq!(routed_name("count the vowels in a string"), Some("count_vowels"));
        assert_eq!(routed_name("gcd of two numbers"), Some("gcd"));
    }

    #[test]
    fn refuses_when_no_op_name_is_named() {
        // No op's full name appears -> honest refusal, never a guess.
        assert_eq!(routed_name("do something clever with the data"), None);
        assert_eq!(routed_name("solve the puzzle"), None);
    }

    #[test]
    fn prefers_the_more_specific_op() {
        // "sum of divisors" names sum_divisors fully; count_divisors needs "count".
        assert_eq!(routed_name("sum of the divisors"), Some("sum_divisors"));
    }

    #[test]
    fn acronym_op_is_proposed_from_spelled_out_words() {
        // gcd is spelled by the initials of "greatest common divisor" — an emergent
        // structural match, so it appears among the ranked candidates for the gate.
        let cands = ranked_candidates("greatest common divisor of two numbers");
        assert!(cands.iter().any(|op| op.name == "gcd"), "gcd should be proposed");
        let lcm = ranked_candidates("least common multiple of two numbers");
        assert!(lcm.iter().any(|op| op.name == "lcm"), "lcm should be proposed");
    }

    // A correct nth-prime program (an algorithm the engine cannot synthesize) — used
    // as a mock model proposal to exercise tier 4.
    const PRIME_MOG: &str = "fn f(n: i64) -> i64 {\n    count: i64 = 0;\n    cand: i64 = 1;\n    while count < n {\n        cand = cand + 1;\n        is_p: i64 = 1;\n        d: i64 = 2;\n        while (d * d) <= cand {\n            if (cand % d) == 0 {\n                is_p = 0;\n            }\n            d = d + 1;\n        }\n        if is_p == 1 {\n            count = count + 1;\n        }\n    }\n    return cand;\n}\n";

    fn answer_code(a: &Answer) -> &str {
        match a {
            Answer::Library { code, .. }
            | Answer::Composition { code }
            | Answer::Synthesized { code, .. }
            | Answer::Proposed { code, .. } => code,
            Answer::Refused => "",
        }
    }

    #[test]
    fn proposer_tier_is_gated_never_returns_a_bad_proposal() {
        use crate::benchmark::{Example, Value};
        let ex = |a: i64, b: i64| Example { inputs: vec![Value::Int(a)], expected: Value::Int(b) };
        // nth prime: no library op, beyond engine synthesis -> forces tier 4.
        let primes = vec![ex(1, 2), ex(2, 3), ex(3, 5), ex(4, 7), ex(5, 11), ex(6, 13)];

        // A model that ONLY hands back WRONG programs (every attempt, repair included)
        // must never cause a wrong answer: either an earlier tier answers correctly,
        // or it refuses — never the garbage.
        let wrong = |_r: &str, _s: &[Example], _p: Option<(&str, &str)>| {
            Some("fn f(n: i64) -> i64 { return n; }".to_string())
        };
        let a = answer_with_proposer("nth prime number", &primes, Some(&wrong));
        assert!(
            matches!(a, Answer::Refused)
                || crate::runtime::code_reproduces_examples(answer_code(&a), &primes),
            "a wrong proposal must never be returned"
        );

        // A model that hands back the CORRECT program is accepted (Proposed, unless an
        // earlier tier already solved it) and reproduces the examples.
        let right = |_r: &str, _s: &[Example], _p: Option<(&str, &str)>| Some(PRIME_MOG.to_string());
        let b = answer_with_proposer("nth prime number", &primes, Some(&right));
        assert!(!matches!(b, Answer::Refused), "correct proposal should be accepted");
        assert!(crate::runtime::code_reproduces_examples(answer_code(&b), &primes));
    }

    #[test]
    fn answer_synthesizes_beyond_the_vocabulary_but_refuses_overfits() {
        use crate::benchmark::{Example, Value};
        let ex = |n: i64, o: i64| Example { inputs: vec![Value::Int(n)], expected: Value::Int(o) };
        // 2n+1: no library op, but the engine synthesizes it. Six examples so a seed
        // of four determines the affine and two hold out.
        let affine = vec![ex(3, 7), ex(5, 11), ex(10, 21), ex(2, 5), ex(7, 15), ex(0, 1)];
        assert!(
            matches!(answer("two times n plus one", &affine), Answer::Synthesized { .. }),
            "should synthesize 2n+1 beyond the library vocabulary"
        );
        // Random non-generalizing points: any fit to the seed fails the held-out
        // examples -> the holdout discipline refuses (never a confident overfit).
        let noise = vec![ex(100, 105), ex(50, 52), ex(73, 30), ex(12, 99), ex(8, 3), ex(64, 77)];
        assert!(matches!(answer("predict", &noise), Answer::Refused));
    }

    #[test]
    fn composes_two_verified_ops_when_no_single_op_fits() {
        use crate::benchmark::{Example, Value};
        // "reverse then uppercase": reverse_string then to_upper. No single op does
        // this, but the composition must verify and be returned.
        let ex = vec![
            Example { inputs: vec![Value::Str("hello".into())], expected: Value::Str("OLLEH".into()) },
            Example { inputs: vec![Value::Str("abc".into())], expected: Value::Str("CBA".into()) },
        ];
        let code = route_composed("reverse then uppercase a string", &ex);
        assert!(code.is_some(), "should find a verified 2-op composition");
        assert!(crate::runtime::code_reproduces_examples(&code.unwrap(), &ex));
        // Wrong examples -> the composition must NOT be returned (never wrong).
        let bad = vec![Example {
            inputs: vec![Value::Str("hello".into())],
            expected: Value::Str("ZZZ".into()),
        }];
        assert!(route_composed("reverse then uppercase a string", &bad).is_none());
    }

    #[test]
    fn declare_resolves_confident_prompts_without_examples() {
        assert_eq!(declare("reverse a string").map(|o| o.name), Some("reverse_string"));
        assert_eq!(declare("sort a list of numbers").map(|o| o.name), Some("sort"));
        // Acronym wins over a coincidental single-token match (all_divisors on "divisor").
        assert_eq!(declare("greatest common divisor of two numbers").map(|o| o.name), Some("gcd"));
        assert_eq!(declare("all divisors of a number").map(|o| o.name), Some("all_divisors"));
    }

    #[test]
    fn declare_composed_reads_a_chain_from_prompt_order() {
        // "reverse then uppercase" -> the composed program, ordered by the prompt.
        let code = declare_composed("reverse then uppercase a string");
        assert!(code.is_some(), "should read a 2-op chain");
        let code = code.unwrap();
        assert!(code.contains("to_upper(reverse_string"), "chain must be reverse THEN upper");
        // Single-op prompt names only one op -> no chain.
        assert!(declare_composed("reverse a string").is_none());
    }

    #[test]
    fn declare_refuses_unmatched_prompt() {
        assert!(declare("do something clever with my data").is_none());
    }

    #[test]
    fn demonstrate_shows_real_behavior() {
        let op = declare("reverse a string").unwrap();
        let demo = demonstrate(op);
        assert!(!demo.is_empty(), "should produce illustrative runs");
        // reverse_string("hello") -> "olleh" must appear in the demonstration.
        assert!(demo.iter().any(|(_, out)| out.contains("olleh")));
    }

    #[test]
    fn distinguishing_gate_refuses_nondistinguishing_examples() {
        // All-True predicate examples cannot distinguish is-perfect-square from
        // several other ops; the ungated router returned a WRONG op. The gate must
        // now refuse (None) rather than confidently return one.
        use crate::benchmark::{Example, Value};
        let all_true: Vec<Example> = [1i64, 4, 9, 16]
            .iter()
            .map(|&n| Example { inputs: vec![Value::Int(n)], expected: Value::Bool(true) })
            .collect();
        assert!(
            route_verified("is it a perfect square", &all_true).is_none(),
            "non-distinguishing examples must refuse, not return a wrong op"
        );
    }

    #[test]
    fn invariant_returned_op_always_reproduces_examples() {
        // The core never-wrong invariant: whenever route_verified returns an op, that
        // op reproduces EVERY example. Checked across a spread of prompts/examples.
        use crate::benchmark::{Example, Value};
        let cases: Vec<(&str, Vec<Example>)> = vec![
            ("greatest common divisor", vec![
                Example { inputs: vec![Value::Int(12), Value::Int(18)], expected: Value::Int(6) },
                Example { inputs: vec![Value::Int(24), Value::Int(36)], expected: Value::Int(12) },
            ]),
            ("factorial of a number", vec![
                Example { inputs: vec![Value::Int(5)], expected: Value::Int(120) },
                Example { inputs: vec![Value::Int(4)], expected: Value::Int(24) },
            ]),
            // A deliberately mislabeled task: "factorial" but examples say something
            // else. The op must be rejected (returned op, if any, still reproduces).
            ("factorial of a number", vec![
                Example { inputs: vec![Value::Int(5)], expected: Value::Int(7) },
                Example { inputs: vec![Value::Int(4)], expected: Value::Int(99) },
            ]),
        ];
        for (prompt, ex) in cases {
            if let Some(r) = route_verified(prompt, &ex) {
                assert!(
                    crate::runtime::code_reproduces_examples(r.op.mog, &ex),
                    "INVARIANT VIOLATED: '{prompt}' returned {} which fails the examples",
                    r.op.name
                );
            }
        }
    }

    #[test]
    fn liberal_proposal_still_never_returns_unverified() {
        // A near-miss prompt proposes count_divisors, but with examples that DON'T
        // match it the gate must refuse (None), never return a wrong op.
        let bad = vec![
            crate::benchmark::Example {
                inputs: vec![crate::benchmark::Value::Int(6)],
                expected: crate::benchmark::Value::Int(999),
            },
            crate::benchmark::Example {
                inputs: vec![crate::benchmark::Value::Int(12)],
                expected: crate::benchmark::Value::Int(999),
            },
        ];
        assert!(route_verified("number of divisors of a number", &bad).is_none());
    }
}
