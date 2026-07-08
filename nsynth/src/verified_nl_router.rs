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
            return match suf {
                // "ies" -> "y" (binaries -> binary).
                "ies" => format!("{base}y"),
                // "-es" is the true plural suffix only after a sibilant (box->boxes,
                // dish->dishes); after any other consonant it is just "-e" + "-s", so
                // the base keeps its "e" (square->squares, value->values, name->names).
                // Without this, "squares" stemmed to "squar" and never matched the op
                // token "square".
                "es" => {
                    let sibilant = base.ends_with('s')
                        || base.ends_with('x')
                        || base.ends_with('z')
                        || base.ends_with("ch")
                        || base.ends_with("sh");
                    if sibilant { base.to_string() } else { format!("{base}e") }
                }
                _ => base.to_string(),
            };
        }
    }
    w
}

/// Generic TYPE / domain nouns that name the KIND of data, not the OPERATION. An op
/// that shares ONLY these with a prompt is not a real match — `tetrahedral_number`
/// for "double a NUMBER", `reverse_number` for "the absolute value of a NUMBER" —
/// so `ranked_candidates` requires at least one NON-generic (operation-bearing)
/// token before proposing an op. Tokens are already stemmed (plurals collapsed).
fn is_generic_type_token(t: &str) -> bool {
    // ONLY pure type-abstraction nouns that ops carry coincidentally as a suffix
    // (tetrahedral_NUMBER, reverse_NUMBER). Deliberately EXCLUDES list / array /
    // string: those are the sole matchable token for legitimate ops whose operation
    // word is an abbreviation the prompt spells out (list_MIN vs "minimum ... list";
    // "min" is 3 chars so it never prefix-matches "minimum"), so treating them as
    // generic would wrongly drop the correct op.
    matches!(t, "number" | "integer" | "value" | "element" | "item")
}

/// Domain / container TYPE nouns, for RANKING only (a superset of the candidacy set
/// above). A match on one of these describes the KIND of data ("in a STRING", "of a
/// LIST"), not the operation, so when two ops TIE on coverage the one matching a real
/// CONTENT token ("VOWEL") must outrank the one matching only a container noun
/// ("STRING") — `count_vowels` beats `string_length` for "number of vowels in a
/// string". Kept separate from `is_generic_type_token` so candidacy is unaffected:
/// `list_min`/`string_length`/`array_sum` stay candidates (their container token still
/// counts toward `matched`), they just lose the tiebreak to a content match.
fn is_domain_type_token(t: &str) -> bool {
    // NB: tokens are STEMMED before this check — `stem("string")` strips the "ing"
    // suffix to "str", so the container noun is matched as "str", not "string".
    is_generic_type_token(t)
        || matches!(t, "str" | "string" | "list" | "array" | "sequence")
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
    let mut scored: Vec<(&'static LibOp, f64, usize, usize, usize)> = Vec::new();
    for op in OPS {
        let name_toks = content_tokens(op.name);
        if name_toks.is_empty() {
            continue;
        }
        let matched = name_toks.iter().filter(|t| has(t)).count();
        // Candidacy needs at least one OPERATION-bearing token match — sharing only a
        // generic type noun ("number", "list") is not a real match and lets a
        // coincidental op (tetrahedral_number for "double a number") reproduce
        // degenerate examples and ship confident-wrong.
        let matched_specific = name_toks
            .iter()
            .filter(|t| has(t) && !is_generic_type_token(t))
            .count();
        // ACRONYM match: a single-token op name (gcd, lcm) whose letters are the
        // initials of a run of consecutive prompt words ("greatest common divisor"
        // -> "gcd"). Emergent — derived from the name's own letters, not a synonym
        // list. Scored as a full-coverage hit so it's tried early.
        let acronym = name_toks.len() == 1 && is_acronym_of(&name_toks[0], &words);
        if matched_specific == 0 && !acronym {
            continue;
        }
        let coverage = if acronym {
            1.0
        } else {
            matched as f64 / name_toks.len() as f64
        };
        // Content-token matches (a real OPERATION word) rank above matches on only a
        // container/type noun, breaking coverage ties toward the more specific op.
        let content_matched = name_toks
            .iter()
            .filter(|t| has(t) && !is_domain_type_token(t))
            .count();
        // Total characters of the matched content tokens. A FINAL tiebreak: when two
        // ops tie on coverage AND content-match COUNT, the one matching the LONGER (more
        // specific) word wins — "lowercase letters" resolves count_lowercase (matched
        // "lowercase", 9 chars) over count_letters (matched "letter", 6). Only reorders
        // already-tied candidates, so it cannot change any strict ranking.
        let content_len: usize = name_toks
            .iter()
            .filter(|t| has(t) && !is_domain_type_token(t))
            .map(|t| t.len())
            .sum();
        scored.push((op, coverage, content_matched, matched.max(acronym as usize), content_len));
    }
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.2.cmp(&a.2))
            .then(b.3.cmp(&a.3))
            .then(b.4.cmp(&a.4))
    });
    scored.into_iter().map(|(op, ..)| op).collect()
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

/// BEHAVIOUR-match fallback. `route_verified` only tries ops whose NAME the prompt
/// shares a token with, so an op the prompt DESCRIBES but never names is missed
/// ("a map from each value to how many times it appears" never says "frequency", yet
/// `element_frequency` computes exactly that). This tries EVERY type-compatible
/// library op by behaviour and keeps one only if the same distinguishing gate holds:
/// a known op that reproduces the examples is correct-by-construction (it cannot
/// overfit like a search), and if two ops reproduce but DIVERGE on fresh probes the
/// spec is under-determined -> refuse. A NL-matched passer is preferred as the winner
/// (better provenance) but any agreeing passer is behaviourally identical. Requires
/// >= 3 distinct examples so a lone coincidental passer has real points to clear.
pub fn route_by_behavior(prompt: &str, examples: &[crate::benchmark::Example]) -> Option<Route> {
    if examples.is_empty() {
        return None;
    }
    let distinct = crate::benchmark::dedup_consistent_examples(examples)
        .map(|v| v.len())
        .unwrap_or(0);
    if distinct < 3 {
        return None;
    }
    let input_types: Vec<&'static str> = examples
        .first()
        .map(|e| e.inputs.iter().map(value_type_str).collect())
        .unwrap_or_default();
    let passing: Vec<&'static LibOp> = OPS
        .iter()
        .filter(|op| op_accepts_types(op.mog, &input_types))
        .filter(|op| crate::runtime::code_reproduces_examples(op.mog, examples))
        .collect();
    if passing.is_empty() {
        return None;
    }
    // Distinguishing gate over ALL behaviour passers — the crucial soundness step:
    // e.g. abs given single-digit examples is reproduced by BOTH reverse_number and
    // sum_of_digits, which disagree on -99 -> refuse, never a confident coincidence.
    if passing.len() > 1 && passers_disagree(&passing, examples) {
        return None;
    }
    // Winner: prefer a passer the prompt actually names (provenance); else the first.
    let named: std::collections::HashSet<&str> =
        ranked_candidates(prompt).iter().map(|op| op.name).collect();
    let winner = *passing
        .iter()
        .find(|op| named.contains(op.name))
        .unwrap_or(&passing[0]);
    // SOLE-PASSER GUARD. When the winner is NOT name-matched (the prompt names no
    // operation this op performs), a behaviour match on a SCALAR/array output is a
    // likely COINCIDENCE — many ops produce ints, and a lone reproducer the prompt
    // doesn't name is as likely a look-alike (unset_bits for "trailing zeros") as the
    // intended op. Refuse. STRUCTURED output (Map/Struct -> "?") is kept: a frequency
    // map is structurally specific, so a behavioural match there (element_frequency
    // for "how many times each value appears") is trustworthy even un-named.
    let winner_named = named.contains(winner.name);
    let out_ty = examples
        .first()
        .map(|e| value_type_str(&e.expected))
        .unwrap_or("?");
    if !winner_named && out_ty != "?" {
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

/// BEHAVIOUR-match 2-op COMPOSITION. `route_composed` only chains NAME-matched
/// candidates, so a described pipeline whose ops the prompt never names is missed —
/// "double the sum of a list" is `times_two(array_sum(x))`, but "double" does not
/// name `times_two`. This chains EVERY type-compatible pair of library ops by
/// behaviour, gated by the same distinguishing power: a chain is kept only if it
/// reproduces every example, and if two surviving chains diverge on fresh probes the
/// spec is under-determined -> refuse. Requires >= 3 distinct examples (a 2-op chain
/// has more freedom to coincide). The O(n^2) surface is bounded by TYPE-chaining
/// (a: in->mid, b: mid->out) + an early reject on the first example, and it runs only
/// as a fallback after the named tiers, so cost is paid rarely.
pub fn route_composed_by_behavior(examples: &[crate::benchmark::Example]) -> Option<String> {
    let first = examples.first()?;
    if first.inputs.len() != 1 {
        return None; // unary chains only
    }
    let distinct = crate::benchmark::dedup_consistent_examples(examples)
        .map(|v| v.len())
        .unwrap_or(0);
    if distinct < 3 {
        return None;
    }
    let in_ty = value_type_str(&first.inputs[0]);
    let out_ty = value_type_str(&first.expected);
    let fname = "composed";
    let first_expected = &first.expected;
    // Cap the collected chains: past this many reproducers the spec is almost surely
    // under-determined, and we still gate whatever we gathered.
    const MAX_CHAINS: usize = 16;
    let mut chains: Vec<String> = Vec::new();
    'outer: for a in OPS {
        let (a_params, a_ret) = op_sig_types(a.mog);
        if a_params.len() != 1 || a_params[0] != in_ty {
            continue; // a: in_ty -> a_ret
        }
        let Some(a_entry) = op_entry_name(a.mog) else { continue };
        // Evaluate `a` ONCE on the first input; every candidate `b` is then rejected
        // with a single op-eval on this value before any chain string is built/parsed.
        let Ok(mid_rt) = crate::runtime::execute_function(a.mog, a_entry, &first.inputs, "c")
        else {
            continue;
        };
        let Ok(mid1) = crate::runtime::benchmark_value_from_runtime(&mid_rt) else { continue };
        for b in OPS {
            if b.name == a.name {
                continue;
            }
            let (b_params, b_ret) = op_sig_types(b.mog);
            if b_params.len() != 1 || b_params[0] != a_ret || b_ret != out_ty {
                continue; // b: a_ret -> out_ty
            }
            let Some(b_entry) = op_entry_name(b.mog) else { continue };
            // Early reject: b(a(first_input)) must equal the first expected output.
            let Ok(out_rt) = crate::runtime::execute_function(b.mog, b_entry, &[mid1.clone()], "c")
            else {
                continue;
            };
            match crate::runtime::benchmark_value_from_runtime(&out_rt) {
                Ok(out1) if &out1 == first_expected => {}
                _ => continue,
            }
            // Passed the first example — build the chain and verify EVERY example.
            let code = format!(
                "fn {fname}(x0: {in_ty}) -> {out_ty} {{\n    return {b_entry}({a_entry}(x0));\n}}\n\n{}\n{}",
                a.mog.trim_end(),
                b.mog.trim_end()
            );
            if crate::runtime::code_reproduces_examples(&code, examples) {
                chains.push(code);
                if chains.len() >= MAX_CHAINS {
                    break 'outer;
                }
            }
        }
    }
    let winner = chains.first()?.clone();
    if chains.len() > 1 {
        let progs: Vec<(String, String)> =
            chains.iter().map(|c| (c.clone(), fname.to_string())).collect();
        if programs_disagree(&progs, examples) {
            return None;
        }
    }
    Some(winner)
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
/// Split on TOP-LEVEL commas, respecting `[]` / `{}` nesting and `"..."` strings, so
/// an array or map argument is not broken apart. Returns the raw slices.
fn split_top(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut last = 0usize;
    let bytes = s.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'"' => in_str = !in_str,
            b'[' | b'{' if !in_str => depth += 1,
            b']' | b'}' if !in_str => depth -= 1,
            b',' if !in_str && depth == 0 => {
                parts.push(&s[last..i]);
                last = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[last..]);
    parts
}

fn parse_values(s: &str) -> Vec<crate::benchmark::Value> {
    split_top(s).into_iter().filter_map(parse_value).collect()
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
    // Map literal `{k:v, k:v}` (frequency maps and the like). Split on top-level
    // commas, then each entry on its FIRST colon; keys/values parse recursively.
    if t.starts_with('{') && t.ends_with('}') {
        let inner = &t[1..t.len() - 1];
        if inner.trim().is_empty() {
            return Some(Value::map_from_pairs(Vec::new()));
        }
        let mut pairs = Vec::new();
        for entry in split_top(inner) {
            let (k, v) = entry.split_once(':')?;
            pairs.push((parse_value(k)?, parse_value(v)?));
        }
        return Some(Value::map_from_pairs(pairs));
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
/// Every library op whose parameter types accept the example inputs AND which
/// reproduces every example. A >=2-op DISAGREEING result means the examples do not pin
/// down even a single op — used to keep the composition/synthesis tiers from re-shipping
/// one of the colliding ops as a confident coincidence.
fn library_ops_reproducing(examples: &[crate::benchmark::Example]) -> Vec<&'static LibOp> {
    let input_types: Vec<&'static str> = match examples.first() {
        Some(e) => e.inputs.iter().map(value_type_str).collect(),
        None => return Vec::new(),
    };
    OPS.iter()
        .filter(|op| op_accepts_types(op.mog, &input_types))
        .filter(|op| crate::runtime::code_reproduces_examples(op.mog, examples))
        .collect()
}

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
    // AMBIGUOUS single-op spec? If TWO OR MORE library ops reproduce the examples yet
    // DISAGREE on fresh inputs, the examples do not pin down even a single op — the
    // behaviour tier refuses for exactly this reason. Composition/synthesis must then NOT
    // run: a longer program would only RE-SHIP one of the colliding single ops as a
    // confident coincidence ("the second element" with a[1]==max examples: both
    // second_element and list_max fit, so a chain wrapping list_max ships the max, wrong
    // on [5,3,9]). Requiring TWO DISAGREEING reproducers is the key — a legitimate
    // composition/synthesis task may have ONE coincidental single-op fit that must still
    // reach the composition tiers; only genuine single-op ambiguity is blocked.
    let ambiguous_single_op = {
        let reproducing = library_ops_reproducing(examples);
        reproducing.len() >= 2 && passers_disagree(&reproducing, examples)
    };
    if !ambiguous_single_op {
        if let Some(code) = route_composed(prompt, examples) {
            return Answer::Composition { code };
        }
    }
    // Tier 2.5 — BEHAVIOUR-match any library op the prompt describes but does not
    // name (element_frequency for "how many times each value appears"). Distinguishing
    // -gated, so a coincidental match refuses instead of shipping wrong.
    if let Some(r) = route_by_behavior(prompt, examples) {
        return Answer::Library { name: r.op.name, code: r.op.mog.to_string() };
    }
    // The examples fit >=2 disagreeing single ops but the behaviour tier could not pick
    // one — refuse rather than let composition/synthesis re-ship the coincidence.
    if ambiguous_single_op {
        return Answer::Refused;
    }
    // Tier 2.75 — BEHAVIOUR-match a 2-op COMPOSITION the prompt describes but does not
    // name both ops of ("double the sum" = times_two(array_sum(x))). Distinguishing-
    // gated; runs only after the single-op tiers, so its O(n^2) surface is paid rarely.
    if let Some(code) = route_composed_by_behavior(examples) {
        return Answer::Composition { code };
    }
    // Tier 3 — full synthesis with HOLDOUT discipline. Solve on a SEED and require
    // the result to reproduce EVERY example including the held-out ones; a fit-only-
    // to-seed overfit fails and is discarded. Needs >= 4. The holdout is 2 when there
    // are enough examples, but shrinks to 1 at exactly 4 so the SEED stays >= 3:
    // reserving 2-of-4 left seed=2, too thin for the search to determine even simple
    // functions (abs, x*1.5), which then refused. seed>=3 keeps synthesis reachable
    // while the >=1 held-out example still catches a seed-overfit (fuzz-verified).
    // DISTINCT-example floor: five copies of {0->0, 1->1} are 5 examples but only
    // TWO points — too few to determine a function, so synthesis returns a
    // coincidental fit (a recurrence through 2 points) CONFIDENTLY wrong. Count
    // distinct consistent examples, not raw rows.
    let distinct = crate::benchmark::dedup_consistent_examples(examples)
        .map(|v| v.len())
        .unwrap_or(0);
    if distinct >= 4 {
        let holdout = (examples.len() - 3).clamp(1, 2);
        // Solve on a seed and require the result to reproduce EVERY example. Returns
        // (method, code, entry) or None.
        let solve_on = |seed: &[crate::benchmark::Example]| -> Option<(String, String, String)> {
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
                let entry =
                    crate::site::fn_name_from_mog(&res.code).unwrap_or_else(|| "f".to_string());
                Some((res.method, res.code, entry))
            } else {
                None
            }
        };
        let seed1 = &examples[..examples.len() - holdout];
        if let Some((method, code, entry)) = solve_on(seed1) {
            // TIER-3 DISTINGUISHING GATE. Tier 1/2 refuse when two candidate ops
            // reproduce the examples but DIVERGE on fresh inputs; tier 3 had no such
            // check — a lone synthesized program that fits under-determined examples
            // was returned CONFIDENT. Fix: independently synthesize on a DIFFERENT
            // seed subset; if the two programs disagree on fresh probes, the examples
            // don't pin the function down -> refuse rather than guess. A determined
            // task yields the same behaviour from both seeds and passes.
            // Hypotheses to corroborate the synthesis against on fresh probes:
            //  (1) an independent synthesis on a DIFFERENT seed subset (catches a lone
            //      overfit that fits under-determined examples), and
            //  (2) the NL-TOP library op that also reproduces the examples — the
            //      interpretation the prompt most names. If the synthesis disagrees
            //      with (2), the prompt points at a different function than the one
            //      synthesized ("sum of the SQUARES" whose 0/1 examples let a plain
            //      sum coincide) -> refuse. Only the SINGLE top passer is used, not
            //      every passer: a lower-ranked coincidental op (max_subarray_sum for
            //      "sum of a list") must NOT veto a correct synthesis the NL-top agrees
            //      with, else legit all-positive sum would over-refuse.
            let mut progs: Vec<(String, String)> = vec![(code.clone(), entry.clone())];
            if let Some((_, code2, entry2)) = solve_on(&examples[holdout..]) {
                progs.push((code2, entry2));
            }
            let input_types: Vec<&'static str> = examples
                .first()
                .map(|e| e.inputs.iter().map(value_type_str).collect())
                .unwrap_or_default();
            let mut library_corroborates = false;
            if let Some(op) = ranked_candidates(prompt)
                .into_iter()
                .filter(|op| op_accepts_types(op.mog, &input_types))
                .take(12)
                .find(|op| crate::runtime::code_reproduces_examples(op.mog, examples))
            {
                if let Some(en) = op_entry_name(op.mog) {
                    progs.push((op.mog.to_string(), en.to_string()));
                    library_corroborates = true;
                }
            }
            // STRUCTURAL BOOLEAN-PREDICATE OVERFIT GATE. A synthesized program whose outputs are
            // all boolean (0/1) over a small example set is the highest overfit-risk class: 2^k
            // functions fit k boolean points, and two INDEPENDENT seed-syntheses routinely
            // CONVERGE on the same coincidental pipeline (e.g. is_prime->is_even for "perfect
            // square", lowest_set_bit_pos->is_prime for "divisible by 6"), so programs_disagree
            // cannot see them diverge. The only independent semantic anchor is a LIBRARY op that
            // also reproduces the examples — the interpretation the prompt actually names. Without
            // one, a boolean synthesis is under-determined; refuse rather than ship a confident
            // guess. (Empirically the risk set is empty: no battery case resolves via an
            // uncorroborated boolean tier-3 synthesis, so this never regresses a solve.)
            let boolean_output = examples.iter().all(|e| {
                matches!(
                    e.expected,
                    crate::benchmark::Value::Int(0) | crate::benchmark::Value::Int(1)
                )
            });
            let boolean_overfit = boolean_output && !library_corroborates;
            // CONSTANT-OVERFIT GUARD. When the examples share an output-derived
            // constant (pairs all averaging 5 -> "return 5"; abs single-positive
            // "return 5"), synthesis can return a CONSTANT function the holdout cannot
            // catch (every held-out example shares the constant). A constant fn over
            // VARIED inputs is almost always an overfit: if the program returns the SAME
            // output on >= 3 fresh probes it succeeds on AND the examples have >= 2
            // distinct inputs, refuse. Require >= 3 SUCCESSFUL evals so a program that
            // errors on most probes and coincides on a couple is not mistaken for one.
            let constant_overfit = {
                let distinct_inputs: std::collections::HashSet<String> =
                    examples.iter().map(|e| format!("{:?}", e.inputs)).collect();
                if distinct_inputs.len() < 2 {
                    false
                } else {
                    let outs: Vec<String> = fresh_probe_inputs(examples)
                        .iter()
                        .filter_map(|inp| {
                            crate::runtime::execute_function(&code, &entry, inp, "cg").ok()
                        })
                        .map(|o| format!("{o:?}"))
                        .collect();
                    outs.len() >= 3 && outs.iter().all(|v| v == &outs[0])
                }
            };
            // A bare library-op match belongs to the gated behaviour tier
            // (route_by_behavior ran first with the distinguishing gate). If a library
            // op still reaches tier 3, it is one that tier REJECTED as coincidental
            // (sum_of_digits reproducing abs's single-digit examples, disagreeing with
            // reverse_number on -99) — solve_problem's internal try_library has no such
            // gate. Don't resurrect it: only a genuine SYNTHESIS result returns here.
            if !boolean_overfit
                && !constant_overfit
                && !method.contains("library:")
                && !programs_disagree(&progs, examples)
            {
                return Answer::Synthesized { method, code };
            }
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

/// Fresh probe inputs: each example's input tuple with one position perturbed into
/// inputs the examples did not cover, so the distinguishing gate can observe two
/// coincidental ops diverge. Scalar-int positions are nudged (+deltas, small
/// constants); ARRAY positions are perturbed to break regime coincidences (inject a
/// negative / a zero, negate, reverse, resize) — without which sum/max-subarray and
/// product/max-product-subarray look identical on all-positive examples and a wrong
/// op ships confident.
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
    for (ex_idx, ex) in examples.iter().take(4).enumerate() {
        for pos in 0..ex.inputs.len() {
            if let Value::Int(v) = ex.inputs[pos] {
                for delta in [1i64, 2, 3, 5, -1] {
                    let mut tup = ex.inputs.clone();
                    tup[pos] = Value::Int(v.wrapping_add(delta));
                    push_if_new(tup);
                }
            }
            // ARRAY probes. An all-positive example array cannot distinguish `sum`
            // from `max_subarray_sum`, `product` from `max_product_subarray`, or an
            // order-sensitive op from an order-insensitive one. Generate a few fresh
            // arrays that break those regimes so the distinguishing gate can observe
            // coincidental ops diverge. Only from the FIRST example (these run against
            // every candidate program, so keep the count small) with regime-breakers
            // chosen to cover sign, zero, and order at once.
            if ex_idx == 0 {
                if let Value::Array(elems) = &ex.inputs[pos] {
                    let ints: Option<Vec<i64>> = elems
                        .iter()
                        .map(|v| if let Value::Int(i) = v { Some(*i) } else { None })
                        .collect();
                    if let Some(base) = ints {
                        if !base.is_empty() {
                            let mut variants: Vec<Vec<i64>> = Vec::new();
                            // negative in front + a zero: breaks sum/max-subarray,
                            // product/max-product, and covers 0 all at once.
                            let mut mixed = vec![-3, 0];
                            mixed.extend_from_slice(&base);
                            variants.push(mixed);
                            // reverse: order-sensitive ops (reverse vs sort-desc) diverge
                            let mut rev = base.clone();
                            rev.reverse();
                            variants.push(rev);
                            // a fixed small mixed array, regime-independent catch-all
                            variants.push(vec![-2, 5, -1, 3]);
                            // an ALL-NEGATIVE array (negative maximum). Every other variant
                            // above leaves the maximum positive, so `|max|` (abs after max)
                            // and `max_of_abs_values` coincide and the gate cannot see them
                            // diverge -- a confident-wrong shipped for "absolute value of the
                            // maximum" (max_absolute_value returned max-of-abs). Negating each
                            // element gives a negative max where |max| != max-of-abs
                            // ([-5,-2,-8] -> |max|=2 vs max-of-abs 8), so the two passing ops
                            // diverge and the ambiguous prompt refuses instead of guessing.
                            let neg: Vec<i64> =
                                base.iter().map(|&x| if x > 0 { -x } else { x - 1 }).collect();
                            variants.push(neg);
                            // a DUPLICATE-containing array: dedup / identity /
                            // elements_once all coincide on distinct-valued inputs, so
                            // without repeats the gate can't tell them apart (a plain
                            // dedup task looked like identity). This distinguishes them.
                            variants.push(vec![4, 4, 1, 4, 7]);
                            for arr in variants {
                                let mut tup = ex.inputs.clone();
                                tup[pos] = Value::int_array(&arr);
                                push_if_new(tup);
                            }
                        }
                    }
                }
                // NOTE: string probes were tried here but REVERTED. They correctly made
                // the tier-1 gate refuse under-determined string tasks (already-upper
                // examples don't distinguish to_upper from identity), but that exposed a
                // SEPARATE tier-3 hole — `typed-enum-str` returns a lookup table that
                // memorizes the examples and falls back to identity for unseen input,
                // shipping a confident-wrong (abc->abc for "uppercase"). Adding string
                // probes without fixing that memorization regresses correct cases, so
                // strings stay unprobed until the tier-3 lookup-table overfit is gated.
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
        // 2n+1 is reached beyond the single-op vocabulary EITHER by full synthesis OR
        // by a verified 2-op composition (plus_one(times_two(n))) — the behaviour-
        // composition tier finds the chain first. Both are correct, verified solves.
        assert!(
            matches!(
                answer("two times n plus one", &affine),
                Answer::Synthesized { .. } | Answer::Composition { .. }
            ),
            "should reach 2n+1 beyond the single-op vocabulary (synthesis or composition)"
        );
        // Random non-generalizing points: any fit to the seed fails the held-out
        // examples -> the holdout discipline refuses (never a confident overfit).
        let noise = vec![ex(100, 105), ex(50, 52), ex(73, 30), ex(12, 99), ex(8, 3), ex(64, 77)];
        assert!(matches!(answer("predict", &noise), Answer::Refused));
    }

    #[test]
    fn adversarial_intent_sweep_is_never_confidently_wrong() {
        use crate::benchmark::{Example, Value};
        let iv = |x: i64| Value::Int(x);
        let av = |a: &[i64]| Value::int_array(a);
        let ex = |i: Vec<Value>, o: Value| Example { inputs: i, expected: o };
        // (prompt, NON-distinguishing examples, fresh DISTINGUISHING input, intended out).
        // Each may SOLVE-correct or REFUSE, but must NEVER return a program that
        // disagrees with the intended output on the fresh input (confident-wrong).
        // These are the coincidental-op traps found by the adversarial sweep.
        let cases: Vec<(&str, Vec<Example>, Vec<Value>, Value)> = vec![
            // sum coincides with max_subarray_sum on all-positive input
            (
                "the sum of a list of numbers",
                vec![ex(vec![av(&[1, 2, 3])], iv(6)), ex(vec![av(&[10, 20])], iv(30)), ex(vec![av(&[4, 4, 4])], iv(12)), ex(vec![av(&[5])], iv(5))],
                vec![av(&[-1, 5])],
                iv(4),
            ),
            // reverse coincides with sort-descending on ascending input
            (
                "reverse a list of numbers",
                vec![ex(vec![av(&[1, 2, 3])], av(&[3, 2, 1])), ex(vec![av(&[1, 2])], av(&[2, 1])), ex(vec![av(&[4, 5, 6, 7])], av(&[7, 6, 5, 4])), ex(vec![av(&[9])], av(&[9]))],
                vec![av(&[3, 1, 2])],
                av(&[2, 1, 3]),
            ),
            // double coincides with tetrahedral_number at {0,2}
            (
                "double a number",
                vec![ex(vec![iv(0)], iv(0)), ex(vec![iv(2)], iv(4)), ex(vec![iv(1)], iv(2)), ex(vec![iv(3)], iv(6))],
                vec![iv(5)],
                iv(10),
            ),
            // minimum coincides with `first` on ascending input
            (
                "the minimum value in a list",
                vec![ex(vec![av(&[1, 2, 3])], iv(1)), ex(vec![av(&[5, 9])], iv(5)), ex(vec![av(&[2, 4, 8])], iv(2)), ex(vec![av(&[0, 7])], iv(0))],
                vec![av(&[9, 1, 5])],
                iv(1),
            ),
            // sum-of-squares coincides with a plain sum when every element is 0/1
            // (x*x == x) — tier-3 must not ship the coincidence over the NL-top op.
            (
                "the sum of the squares of a list of numbers",
                vec![ex(vec![av(&[0, 1, 1])], iv(2)), ex(vec![av(&[1, 0])], iv(1)), ex(vec![av(&[1, 1, 1])], iv(3)), ex(vec![av(&[0, 0, 1])], iv(1))],
                vec![av(&[2, 3])],
                iv(13),
            ),
            // abs with single-digit examples: sum_of_digits AND reverse_number both
            // reproduce (they equal |x| for one digit) — solve_problem's ungated
            // try_library must not resurrect one after route_by_behavior refuses.
            (
                "the absolute value of a number",
                vec![ex(vec![iv(-3)], iv(3)), ex(vec![iv(5)], iv(5)), ex(vec![iv(-1)], iv(1)), ex(vec![iv(0)], iv(0))],
                vec![iv(-99)],
                iv(99),
            ),
            // trailing-zeros coincides with unset_bits (total 0-bits) on examples where
            // the low bits are the only zeros — the behaviour-match tier returned the
            // coincidental unset_bits (wrong on 10 = 1010: trailing=1, unset=2) until a
            // real trailing_zeros op made the name tier resolve it directly.
            (
                "the number of trailing zeros in binary",
                vec![ex(vec![iv(8)], iv(3)), ex(vec![iv(12)], iv(2)), ex(vec![iv(1)], iv(0)), ex(vec![iv(16)], iv(4)), ex(vec![iv(6)], iv(1))],
                vec![iv(10)],
                iv(1),
            ),
            // count_even_digits coincides with count_digits on all-even-digit numbers.
            (
                "the number of even digits in a number",
                vec![ex(vec![iv(2)], iv(1)), ex(vec![iv(24)], iv(2)), ex(vec![iv(468)], iv(3)), ex(vec![iv(6)], iv(1))],
                vec![iv(13)],
                iv(0),
            ),
            // abs_difference coincides with signed a-b when a >= b.
            (
                "the absolute difference between two numbers",
                vec![ex(vec![iv(7), iv(3)], iv(4)), ex(vec![iv(9), iv(2)], iv(7)), ex(vec![iv(5), iv(5)], iv(0)), ex(vec![iv(10), iv(4)], iv(6))],
                vec![iv(2), iv(10)],
                iv(8),
            ),
            // average_two overfits to a constant when all example pairs share an average.
            (
                "the average of two numbers",
                vec![ex(vec![iv(4), iv(6)], iv(5)), ex(vec![iv(2), iv(8)], iv(5)), ex(vec![iv(10), iv(0)], iv(5)), ex(vec![iv(3), iv(7)], iv(5)), ex(vec![iv(1), iv(9)], iv(5))],
                vec![iv(2), iv(4)],
                iv(3),
            ),
            // capitalize_first coincides with capitalize_first_last on single-char strings.
            (
                "capitalize the first letter of a string",
                vec![ex(vec![Value::Str("a".into())], Value::Str("A".into())), ex(vec![Value::Str("z".into())], Value::Str("Z".into())), ex(vec![Value::Str("x".into())], Value::Str("X".into()))],
                vec![Value::Str("hello".into())],
                Value::Str("Hello".into()),
            ),
            // uppercase_vowels coincides with to_upper on all-vowel strings.
            (
                "uppercase only the vowels in a string",
                vec![ex(vec![Value::Str("ae".into())], Value::Str("AE".into())), ex(vec![Value::Str("i".into())], Value::Str("I".into())), ex(vec![Value::Str("ou".into())], Value::Str("OU".into()))],
                vec![Value::Str("cat".into())],
                Value::Str("cAt".into()),
            ),
            // count_vowels coincides with string_length on all-vowel strings — the op
            // matches the CONTENT token "vowel" and must outrank string_length (matches
            // only the container noun "string", stemmed "str") on the coverage tie.
            (
                "the number of vowels in a string",
                vec![ex(vec![Value::Str("aei".into())], iv(3)), ex(vec![Value::Str("oo".into())], iv(2)), ex(vec![Value::Str("a".into())], iv(1))],
                vec![Value::Str("cat".into())],
                iv(1),
            ),
            // count_consonants coincides with string_length on all-consonant strings.
            (
                "the number of consonants in a string",
                vec![ex(vec![Value::Str("bcd".into())], iv(3)), ex(vec![Value::Str("xy".into())], iv(2)), ex(vec![Value::Str("rr".into())], iv(2))],
                vec![Value::Str("cat".into())],
                iv(2),
            ),
            // max_negative coincides with list_max on all-negative input.
            (
                "the largest negative number in a list",
                vec![ex(vec![av(&[-5, -1, -3])], iv(-1)), ex(vec![av(&[-9, -2])], iv(-2)), ex(vec![av(&[-4, -8, -2])], iv(-2))],
                vec![av(&[-4, 3, -1])],
                iv(-1),
            ),
            // min_even coincides with list_min on all-even input (min is the smallest
            // even too) — wrong once the min is odd ([5,3,8,4] -> smallest even 4, min 3).
            (
                "the smallest even number in a list",
                vec![ex(vec![av(&[4, 2, 6])], iv(2)), ex(vec![av(&[8, 10])], iv(8)), ex(vec![av(&[6, 2, 4])], iv(2))],
                vec![av(&[5, 3, 8, 4])],
                iv(4),
            ),
            // max_odd coincides with list_max on all-odd input — wrong once the max is
            // even ([2,7,8] -> largest odd 7, list_max 8).
            (
                "the largest odd number in a list",
                vec![ex(vec![av(&[1, 3, 5])], iv(5)), ex(vec![av(&[7, 9])], iv(9)), ex(vec![av(&[3, 1, 5])], iv(5))],
                vec![av(&[2, 7, 8])],
                iv(7),
            ),
            // min_odd coincides with list_min on all-odd input — wrong once a smaller
            // even is present ([2,7,3] -> smallest odd 3, list_min 2).
            (
                "the smallest odd number in a list",
                vec![ex(vec![av(&[3, 5, 7])], iv(3)), ex(vec![av(&[9, 1])], iv(1)), ex(vec![av(&[5, 3, 9])], iv(3))],
                vec![av(&[2, 7, 3])],
                iv(3),
            ),
            // product_evens coincides with array_product on all-even input.
            (
                "the product of the even numbers in a list",
                vec![ex(vec![av(&[2, 4])], iv(8)), ex(vec![av(&[2, 2, 2])], iv(8)), ex(vec![av(&[6])], iv(6))],
                vec![av(&[2, 3, 4])],
                iv(8),
            ),
            // second_smallest coincides with second_largest on 3-distinct examples (the
            // middle value is both) — wrong on 4+ elements.
            (
                "the second smallest value in a list",
                vec![ex(vec![av(&[1, 2, 3])], iv(2)), ex(vec![av(&[5, 7, 9])], iv(7)), ex(vec![av(&[2, 4, 8])], iv(4))],
                vec![av(&[9, 1, 5, 3])],
                iv(3),
            ),
            // Under-determined divisibility examples let TIER-3 synthesis ship a
            // coincidental library pipeline (binary_to_decimal->unset_bits, wrong on 15);
            // the named is_multiple_of_five op resolves before tier 3.
            (
                "whether a number is a multiple of five",
                vec![ex(vec![iv(5)], iv(1)), ex(vec![iv(10)], iv(1)), ex(vec![iv(7)], iv(0)), ex(vec![iv(3)], iv(0))],
                vec![iv(15)],
                iv(1),
            ),
            // filter_evens equals identity on all-even input; must stay never-wrong on a
            // fresh mixed input (locks the coincidence in case an identity op is added).
            (
                "keep only the even numbers in a list",
                vec![ex(vec![av(&[2, 4, 6])], av(&[2, 4, 6])), ex(vec![av(&[8])], av(&[8])), ex(vec![av(&[2, 10])], av(&[2, 10]))],
                vec![av(&[1, 2, 3, 4])],
                av(&[2, 4]),
            ),
            // Under-determined sign examples let the composition tier ship a coincidental
            // chain (wrong on -7: gave -3); the named sign op resolves before composition.
            (
                "the sign of a number",
                vec![ex(vec![iv(5)], iv(1)), ex(vec![iv(3)], iv(1)), ex(vec![iv(-2)], iv(-1)), ex(vec![iv(0)], iv(0))],
                vec![iv(-7)],
                iv(-1),
            ),
            // count_unique_chars coincides with string_length on all-unique strings.
            (
                "the number of unique characters in a string",
                vec![ex(vec![Value::Str("abc".into())], iv(3)), ex(vec![Value::Str("xy".into())], iv(2)), ex(vec![Value::Str("a".into())], iv(1))],
                vec![Value::Str("aab".into())],
                iv(2),
            ),
            // Under-determined 2-arg divisibility examples let tier-3 ship a coincidental
            // program wrong on (8,4); the named is_divisible op resolves before tier 3.
            (
                "whether a is divisible by b",
                vec![ex(vec![iv(6), iv(3)], iv(1)), ex(vec![iv(7), iv(2)], iv(0)), ex(vec![iv(10), iv(5)], iv(1)), ex(vec![iv(9), iv(4)], iv(0))],
                vec![iv(8), iv(4)],
                iv(1),
            ),
            // Same predicate phrased with "multiple" (which is_divisible does not name-
            // match) — tier-3 shipped a wrong program on (8,2); is_multiple resolves it.
            (
                "whether a is a multiple of b",
                vec![ex(vec![iv(6), iv(2)], iv(1)), ex(vec![iv(7), iv(3)], iv(0)), ex(vec![iv(9), iv(3)], iv(1)), ex(vec![iv(10), iv(4)], iv(0))],
                vec![iv(8), iv(2)],
                iv(1),
            ),
            // AMBIGUOUS-single-op guard: second_element and list_max both fit examples
            // where a[1] is the max ([1,9,2]->9); the composition tier used to re-ship the
            // max via a chain (wrong on [5,3,9]). The guard now refuses instead.
            (
                "the second element of a list",
                vec![ex(vec![av(&[1, 9, 2])], iv(9)), ex(vec![av(&[3, 8])], iv(8)), ex(vec![av(&[2, 7, 1])], iv(7))],
                vec![av(&[5, 3, 9])],
                iv(3),
            ),
            // third_element vs list_max, same ambiguous-single-op collision.
            (
                "the third element of a list",
                vec![ex(vec![av(&[1, 2, 3])], iv(3)), ex(vec![av(&[5, 6, 7])], iv(7)), ex(vec![av(&[2, 4, 8])], iv(8))],
                vec![av(&[9, 1, 2, 3])],
                iv(2),
            ),
            // count_odd_digits coincides with count_digits on all-odd-digit numbers.
            (
                "the number of odd digits in a number",
                vec![ex(vec![iv(1)], iv(1)), ex(vec![iv(13)], iv(2)), ex(vec![iv(357)], iv(3)), ex(vec![iv(5)], iv(1))],
                vec![iv(24)],
                iv(0),
            ),
            // all_same coincides with has_duplicates on all-equal / all-distinct examples;
            // wrong once a value repeats without ALL being equal ([7,7,8] -> 0 vs 1).
            (
                "whether all elements in a list are the same",
                vec![ex(vec![av(&[2, 2, 2])], iv(1)), ex(vec![av(&[1, 2])], iv(0)), ex(vec![av(&[5, 5])], iv(1)), ex(vec![av(&[3, 4, 5])], iv(0))],
                vec![av(&[7, 7, 8])],
                iv(0),
            ),
            // The composition tier shipped a wrong chain for first-minus-last; the named
            // first_last_diff op resolves it ([3,1,8] -> 3-8 = -5).
            (
                "the difference between the first and last element",
                vec![ex(vec![av(&[5, 1, 2])], iv(3)), ex(vec![av(&[10, 4])], iv(6)), ex(vec![av(&[7, 2, 9])], iv(-2))],
                vec![av(&[3, 1, 8])],
                iv(-5),
            ),
            // count_letters coincides with string_length on all-letter strings.
            (
                "the number of letters in a string",
                vec![ex(vec![Value::Str("abc".into())], iv(3)), ex(vec![Value::Str("hi".into())], iv(2)), ex(vec![Value::Str("xyz".into())], iv(3))],
                vec![Value::Str("a1b".into())],
                iv(2),
            ),
            // starts_with_uppercase coincides with count_uppercase on strings with <=1
            // uppercase; count_uppercase is wrong once there are 2+ ("AB" -> 1 vs 2).
            (
                "whether a string starts with an uppercase letter",
                vec![ex(vec![Value::Str("Hello".into())], iv(1)), ex(vec![Value::Str("world".into())], iv(0)), ex(vec![Value::Str("Xyz".into())], iv(1)), ex(vec![Value::Str("abc".into())], iv(0))],
                vec![Value::Str("AB".into())],
                iv(1),
            ),
            // contains_vowel coincides with count_vowels when examples have <=1 vowel;
            // count_vowels is wrong once there are 2+ ("bee" -> contains 1, count 2).
            (
                "whether a string contains a vowel",
                vec![ex(vec![Value::Str("cat".into())], iv(1)), ex(vec![Value::Str("xyz".into())], iv(0)), ex(vec![Value::Str("by".into())], iv(0)), ex(vec![Value::Str("dog".into())], iv(1))],
                vec![Value::Str("bee".into())],
                iv(1),
            ),
            // contains_negative coincides with count_negatives when examples have <=1
            // negative; count_negatives is wrong once there are 2+ ([-1,-2] -> 1 vs 2).
            (
                "whether a list contains a negative number",
                vec![ex(vec![av(&[1, -2, 3])], iv(1)), ex(vec![av(&[4, 5])], iv(0)), ex(vec![av(&[6, -1])], iv(1)), ex(vec![av(&[2, 3])], iv(0))],
                vec![av(&[-1, -2])],
                iv(1),
            ),
            // contains_even coincides with count_evens when examples have <=1 even.
            (
                "whether a list contains an even number",
                vec![ex(vec![av(&[1, 2, 3])], iv(1)), ex(vec![av(&[3, 5])], iv(0)), ex(vec![av(&[7, 4])], iv(1)), ex(vec![av(&[1, 9])], iv(0))],
                vec![av(&[2, 4])],
                iv(1),
            ),
            // contains_digit coincides with count_string_digits when examples have <=1 digit.
            (
                "whether a string contains a digit",
                vec![ex(vec![Value::Str("a1b".into())], iv(1)), ex(vec![Value::Str("abc".into())], iv(0)), ex(vec![Value::Str("x9".into())], iv(1)), ex(vec![Value::Str("yz".into())], iv(0))],
                vec![Value::Str("a12".into())],
                iv(1),
            ),
            // count_alphanumeric coincides with string_length on all-alnum strings.
            (
                "the number of alphanumeric characters in a string",
                vec![ex(vec![Value::Str("ab1".into())], iv(3)), ex(vec![Value::Str("xy".into())], iv(2)), ex(vec![Value::Str("z9".into())], iv(2))],
                vec![Value::Str("a b".into())],
                iv(2),
            ),
            // count_lowercase coincides with count_letters on all-lowercase strings; the
            // matched-token char-length tiebreak keeps count_lowercase ('lowercase' > 'letter').
            (
                "the number of lowercase letters in a string",
                vec![ex(vec![Value::Str("abc".into())], iv(3)), ex(vec![Value::Str("hi".into())], iv(2)), ex(vec![Value::Str("z".into())], iv(1))],
                vec![Value::Str("aBc".into())],
                iv(2),
            ),
            // 'less than zero' paraphrase: count_negatives reproduces but is un-named, so the
            // composition tier shipped a wrong chain ([-1,0,3] -> 2); count_less_than_zero
            // resolves it. (Note: 0 is NOT < 0, so the fresh input distinguishes.)
            (
                "how many numbers in a list are less than zero",
                vec![ex(vec![av(&[-1, 2, -3])], iv(2)), ex(vec![av(&[1, 2])], iv(0)), ex(vec![av(&[-5, -6])], iv(2))],
                vec![av(&[-1, 0, 3])],
                iv(1),
            ),
            // 'not zero' coincides with count_greater_than_zero on non-negative inputs
            // (all >0 are also !=0); wrong once a negative is present ([-2,0,3] -> not-zero
            // 2, but >0 count 1). count_not_zero ('not' + 'zero') resolves it.
            (
                "how many numbers in a list are not zero",
                vec![ex(vec![av(&[1, 0, 3])], iv(2)), ex(vec![av(&[0, 0])], iv(0)), ex(vec![av(&[5, 6])], iv(2))],
                vec![av(&[-2, 0, 3])],
                iv(2),
            ),
            // is-odd coincides with count_odd_digits on single-digit examples (one odd
            // digit == is-odd); wrong on multi-digit (13 -> is-odd 1, count_odd_digits 2).
            // is_odd (coverage 1.0 on 'odd') resolves it.
            (
                "whether a number is odd",
                vec![ex(vec![iv(1)], iv(1)), ex(vec![iv(4)], iv(0)), ex(vec![iv(7)], iv(1)), ex(vec![iv(2)], iv(0))],
                vec![iv(13)],
                iv(1),
            ),
            // count_distinct_digits coincides with count_digits on all-distinct-digit
            // numbers; wrong once a digit repeats (122 -> 2 distinct, count_digits 3).
            (
                "the number of distinct digits in a number",
                vec![ex(vec![iv(123)], iv(3)), ex(vec![iv(45)], iv(2)), ex(vec![iv(7)], iv(1))],
                vec![iv(122)],
                iv(2),
            ),
            // list_max (matching only the container noun 'list') coincides with
            // sum-of-all-but-last when max == that sum; wrong on [9,1,2] (10 vs 9).
            // sum_except_last resolves it by name.
            (
                "the sum of all elements except the last in a list",
                vec![ex(vec![av(&[1, 2, 3])], iv(3)), ex(vec![av(&[10, 4])], iv(10)), ex(vec![av(&[2, 3, 5])], iv(5))],
                vec![av(&[9, 1, 2])],
                iv(10),
            ),
            // list_max coincides with max-except-last when the max is not the last
            // element; wrong once it is ([1,2,9] -> 2 vs 9). max_except_last resolves it.
            (
                "the largest element except the last in a list",
                vec![ex(vec![av(&[9, 1, 2])], iv(9)), ex(vec![av(&[5, 3, 1])], iv(5)), ex(vec![av(&[8, 2, 4])], iv(8))],
                vec![av(&[1, 2, 9])],
                iv(2),
            ),
            // list_min coincides with min-except-last when the min is not the last
            // element; wrong once it is ([5,3,1] -> 3 vs 1). min_except_last resolves it.
            (
                "the smallest element except the last in a list",
                vec![ex(vec![av(&[1, 5, 9])], iv(1)), ex(vec![av(&[2, 8, 3])], iv(2)), ex(vec![av(&[4, 6, 7])], iv(4))],
                vec![av(&[5, 3, 1])],
                iv(3),
            ),
            // array_sum coincides with sum-except-largest when the max is 0 (dropping it
            // changes nothing); wrong once max != 0 ([2,3,4] -> 5 vs 9). sum_except_largest
            // out-covers array_sum and resolves it.
            (
                "the sum of all elements except the largest in a list",
                vec![ex(vec![av(&[-1, -2, 0])], iv(-3)), ex(vec![av(&[0, -4])], iv(-4)), ex(vec![av(&[-3, 0, -1])], iv(-4))],
                vec![av(&[2, 3, 4])],
                iv(5),
            ),
            // list_max coincides with max-except-first when the max is not the first
            // element; wrong once it is ([9,1,2] -> 2 vs 9). max_except_first resolves it.
            (
                "the largest element except the first in a list",
                vec![ex(vec![av(&[1, 9, 2])], iv(9)), ex(vec![av(&[3, 1, 5])], iv(5)), ex(vec![av(&[2, 8, 4])], iv(8))],
                vec![av(&[9, 1, 2])],
                iv(2),
            ),
            // string_length coincides with the non-letter count on strings with no letters.
            (
                "the number of non-letter characters in a string",
                vec![ex(vec![Value::Str("12".into())], iv(2)), ex(vec![Value::Str("!!".into())], iv(2)), ex(vec![Value::Str("3".into())], iv(1))],
                vec![Value::Str("a1".into())],
                iv(1),
            ),
            // 'capital letters' does not name 'uppercase', so count_letters coincided on
            // all-capital strings; count_capitals wins the char-length tiebreak ('capital'
            // > 'letter') and is wrong on mixed case only for count_letters ("aBc" -> 1).
            (
                "the number of capital letters in a string",
                vec![ex(vec![Value::Str("AB".into())], iv(2)), ex(vec![Value::Str("X".into())], iv(1)), ex(vec![Value::Str("QRS".into())], iv(3))],
                vec![Value::Str("aBc".into())],
                iv(1),
            ),
            // list_max coincides with the max ABSOLUTE value on all-nonnegative input;
            // wrong once a negative dominates ([-3,-8,1] -> 8 vs 1). max_absolute_value
            // matches 'absolute' (content), out-covering list_max's 'list' (container).
            (
                "the maximum absolute value in a list",
                vec![ex(vec![av(&[3, 5, 2])], iv(5)), ex(vec![av(&[1, 9])], iv(9)), ex(vec![av(&[4, 2])], iv(4))],
                vec![av(&[-3, -8, 1])],
                iv(8),
            ),
            // list_min coincides with the min ABSOLUTE value on all-nonnegative input;
            // wrong once a negative dominates ([-3,-8,1] -> 1 vs -8). min_absolute_value.
            (
                "the minimum absolute value in a list",
                vec![ex(vec![av(&[3, 5, 2])], iv(2)), ex(vec![av(&[1, 9])], iv(1)), ex(vec![av(&[4, 6])], iv(4))],
                vec![av(&[-3, -8, 1])],
                iv(1),
            ),
            // array_sum coincides with sum-of-negatives on all-negative input; wrong once a
            // positive is present ([1,-2,-3] -> -5 vs -4). sum_negatives out-covers array_sum.
            (
                "the sum of the negative numbers in a list",
                vec![ex(vec![av(&[-1, -2, -3])], iv(-6)), ex(vec![av(&[-4, -5])], iv(-9)), ex(vec![av(&[-2])], iv(-2))],
                vec![av(&[1, -2, -3])],
                iv(-5),
            ),
            // first_last_diff (signed) coincides with the ABSOLUTE first/last difference whenever
            // first >= last; wrong once last > first ([3,1,8] -> signed -5 vs |−5|=5). Guard refuses
            // (both diff ops reproduce and disagree) or absolute_first_last_diff out-covers.
            (
                "the absolute difference between the first and last element",
                vec![ex(vec![av(&[5, 1, 2])], iv(3)), ex(vec![av(&[10, 4])], iv(6)), ex(vec![av(&[8, 3])], iv(5))],
                vec![av(&[3, 1, 8])],
                iv(5),
            ),
            // array_product coincides with the ABSOLUTE product whenever the product is
            // non-negative; wrong once an odd count of negatives flips the sign ([-2,3] -> -6 vs 6).
            // absolute_product out-covers array_product via the 'absolute' token.
            (
                "the absolute value of the product of a list",
                vec![ex(vec![av(&[2, 3])], iv(6)), ex(vec![av(&[1, 5])], iv(5)), ex(vec![av(&[2, 2])], iv(4))],
                vec![av(&[-2, 3])],
                iv(6),
            ),
            // |max(arr)| (abs AFTER max) shares every token with max_absolute_value (max of the
            // abs values). On all-nonnegative examples both list_max and max_absolute_value pass
            // and the old probe set left the maximum positive, so they never diverged and a
            // confident-wrong shipped ([-3,-8] -> max_absolute_value returned 8, intended 3).
            // The all-negative probe now makes them diverge -> the gate refuses this token-order
            // ambiguity instead of guessing.
            (
                "the absolute value of the maximum value in a list",
                vec![ex(vec![av(&[3, 5, 2])], iv(5)), ex(vec![av(&[1, 9])], iv(9)), ex(vec![av(&[4, 7, 6])], iv(7))],
                vec![av(&[-3, -8])],
                iv(3),
            ),
            // The bare `first`/`last` ops coincide with |first|/|last| on a non-negative element,
            // wrong once it is negative ([-8,3] -> first -8 vs 8; [3,-8] -> last -8 vs 8).
            // absolute_first / absolute_last resolve the determined case; the ambiguous all-positive
            // set refuses (two passers diverge on the all-negative probe).
            (
                "the absolute value of the first element of a list",
                vec![ex(vec![av(&[3, 5, 2])], iv(3)), ex(vec![av(&[1, 9])], iv(1)), ex(vec![av(&[4, 7])], iv(4))],
                vec![av(&[-8, 3])],
                iv(8),
            ),
            (
                "the absolute value of the last element of a list",
                vec![ex(vec![av(&[3, 5, 2])], iv(2)), ex(vec![av(&[1, 9])], iv(9)), ex(vec![av(&[4, 7])], iv(7))],
                vec![av(&[3, -8])],
                iv(8),
            ),
            // SIGNED difference a-b: abs_difference coincides when a >= b, wrong on a < b
            // ([2,9] -> a-b = -7 vs |a-b| = 7). The single-token 'difference' op out-ranks it on
            // the bare prompt (coverage 1.0 vs 0.5).
            (
                "the difference between two numbers",
                vec![ex(vec![iv(5), iv(2)], iv(3)), ex(vec![iv(9), iv(4)], iv(5)), ex(vec![iv(8), iv(1)], iv(7))],
                vec![iv(2), iv(9)],
                iv(-7),
            ),
            // Perfect-square with a degenerate example set (every square is composite, every
            // non-square is prime) let tier-3 synthesis overfit the pipeline is_prime->is_even
            // (= 'is composite'), wrong on a composite non-square (6 -> 1 vs 0). The is_perfect_square
            // library op resolves before synthesis and computes the real predicate.
            (
                "whether a number is a perfect square",
                vec![ex(vec![iv(4)], iv(1)), ex(vec![iv(9)], iv(1)), ex(vec![iv(3)], iv(0)), ex(vec![iv(16)], iv(1)), ex(vec![iv(7)], iv(0))],
                vec![iv(6)],
                iv(0),
            ),
            // Coprime with a degenerate example set (coprime coincided with 'at least one operand
            // odd') let tier-3 synthesis ship a confident-wrong on two odds with a common factor
            // ((3,9) -> 1 vs gcd 3 => 0). is_coprime resolves before synthesis.
            (
                "whether two numbers are coprime",
                vec![ex(vec![iv(2), iv(3)], iv(1)), ex(vec![iv(4), iv(9)], iv(1)), ex(vec![iv(6), iv(8)], iv(0)), ex(vec![iv(2), iv(4)], iv(0))],
                vec![iv(3), iv(9)],
                iv(0),
            ),
            // starts_with_uppercase coincides with ALL-uppercase whenever the first character
            // already decides the outcome; wrong on a string that merely starts uppercase
            // ('Ab' -> starts-upper 1 vs all-upper 0). all_uppercase out-covers it via 'all'.
            (
                "whether a string is all uppercase",
                vec![ex(vec![Value::Str("AB".into())], iv(1)), ex(vec![Value::Str("XY".into())], iv(1)), ex(vec![Value::Str("ab".into())], iv(0)), ex(vec![Value::Str("aB".into())], iv(0))],
                vec![Value::Str("Ab".into())],
                iv(0),
            ),
            // The composition tier overfit an all-letters example set that coincided with
            // first-char-is-a-letter, wrong on a string that only starts with a letter
            // ('ab1' -> 1 vs 0). all_letters (library tier, before composition) is the real predicate.
            (
                "whether a string contains only letters",
                vec![ex(vec![Value::Str("abc".into())], iv(1)), ex(vec![Value::Str("xyz".into())], iv(1)), ex(vec![Value::Str("1bc".into())], iv(0)), ex(vec![Value::Str("9yz".into())], iv(0))],
                vec![Value::Str("ab1".into())],
                iv(0),
            ),
            // all_positive (every element positive) coincides with 'the sum is positive' across an
            // example set where the two happen to agree, but is wrong once a large positive
            // outweighs a negative ([-5,10] -> sum 5 > 0 => 1, all_positive 0). sum_is_positive is
            // the real aggregate-then-predicate op.
            (
                "whether the sum of a list is positive",
                vec![ex(vec![av(&[1, 2])], iv(1)), ex(vec![av(&[3, 4])], iv(1)), ex(vec![av(&[-5, -6])], iv(0)), ex(vec![av(&[-1, -2])], iv(0))],
                vec![av(&[-5, 10])],
                iv(1),
            ),
            // count_string_digits coincides with ENDS-WITH-a-digit when the digit count already
            // decides it, wrong on a mid-string digit ('a1b' -> count 1 vs ends-with-digit 0).
            // ends_with_digit resolves it.
            (
                "whether a string ends with a digit",
                vec![ex(vec![Value::Str("ab1".into())], iv(1)), ex(vec![Value::Str("xy9".into())], iv(1)), ex(vec![Value::Str("abc".into())], iv(0)), ex(vec![Value::Str("xyz".into())], iv(0))],
                vec![Value::Str("a1b".into())],
                iv(0),
            ),
            // STRUCTURAL: a boolean predicate with no library op (divisible by 6) fell to tier-3
            // synthesis, which overfit lowest_set_bit_pos->is_prime, wrong on 4. The boolean-
            // predicate overfit gate now refuses an uncorroborated boolean synthesis.
            (
                "whether a number is divisible by both two and three",
                vec![ex(vec![iv(6)], iv(1)), ex(vec![iv(12)], iv(1)), ex(vec![iv(8)], iv(0)), ex(vec![iv(9)], iv(0))],
                vec![iv(4)],
                iv(0),
            ),
        ];
        for (prompt, exs, fresh, intended) in cases {
            let code = match answer(prompt, &exs) {
                Answer::Library { code, .. }
                | Answer::Composition { code }
                | Answer::Synthesized { code, .. }
                | Answer::Proposed { code, .. } => code,
                Answer::Refused => continue, // honest refusal is allowed
            };
            let entry = crate::site::fn_name_from_mog(&code).unwrap_or_else(|| "f".to_string());
            if let Ok(got) = crate::runtime::execute_function(&code, &entry, &fresh, "probe") {
                assert_eq!(
                    format!("{got:?}"),
                    format!("{intended:?}"),
                    "CONFIDENT-WRONG on '{prompt}': fresh {fresh:?} -> {got:?}, intended {intended:?}"
                );
            }
        }
    }

    #[test]
    fn candidacy_requires_an_operation_token_not_a_generic_type_noun() {
        // "double a number": the only op-name token shared with tetrahedral_NUMBER is
        // the generic type noun "number" -> it must NOT be proposed (it coincides with
        // double at {0,2} and would ship confident-wrong on degenerate examples).
        let cands = ranked_candidates("double a number");
        assert!(
            !cands.iter().any(|op| op.name == "tetrahedral_number"),
            "generic 'number'-only match must not qualify tetrahedral_number"
        );
        // But list / array / string are NOT generic — list_min's operation word "min"
        // is a 3-char abbreviation that never prefix-matches "minimum", so "list" is
        // its only matchable token and it MUST still be proposed.
        let cands = ranked_candidates("the minimum value in a list");
        assert!(
            cands.iter().any(|op| op.name == "list_min"),
            "list_min must still be proposed via its 'list' token"
        );
    }

    #[test]
    fn tier3_refuses_under_determined_synthesis() {
        use crate::benchmark::{Example, Value};
        let ex = |n: i64, o: i64| Example { inputs: vec![Value::Int(n)], expected: Value::Int(o) };
        // Five rows but only TWO distinct points {0->0, 1->1}. Synthesis can fit a
        // recurrence through them and would return it CONFIDENTLY (f(2)=1, not 4),
        // but the distinct-example floor refuses: too few points to determine a fn.
        let degenerate = vec![ex(0, 0), ex(1, 1), ex(0, 0), ex(1, 1), ex(0, 0)];
        assert!(
            matches!(answer("the square of a number", &degenerate), Answer::Refused),
            "under-determined synthesis (2 distinct points) must refuse, not guess"
        );
    }

    #[test]
    fn behaviour_composition_reaches_a_pipeline_naming_neither_op() {
        use crate::benchmark::{Example, Value};
        let av = |a: &[i64]| Value::int_array(a);
        let ex = |i: Vec<Value>, o: i64| Example { inputs: i, expected: Value::Int(o) };
        // "double the sum" = times_two(array_sum(x)); "double" names neither op, so the
        // NAME-based route_composed misses it. Behaviour-composition must find it.
        let exs = vec![
            ex(vec![av(&[1, 2, 3])], 12),
            ex(vec![av(&[5])], 10),
            ex(vec![av(&[10, 20])], 60),
            ex(vec![av(&[-1, 3])], 4),
        ];
        let code = route_composed_by_behavior(&exs).expect("should find a 2-op chain");
        assert!(crate::runtime::code_reproduces_examples(&code, &exs));
        let entry = crate::site::fn_name_from_mog(&code).unwrap_or_else(|| "composed".into());
        // fresh distinguishing input: 2*(2+5)=14
        let out = crate::runtime::execute_function(&code, &entry, &[av(&[2, 5])], "p").unwrap();
        assert_eq!(format!("{out:?}"), "Int(14)", "chain must be times_two(sum), not a coincidence");
        // Random non-generalizing points: no 2-op chain reproduces them -> refuse.
        let noise = vec![
            ex(vec![av(&[1])], 99),
            ex(vec![av(&[2])], 3),
            ex(vec![av(&[3])], 41),
            ex(vec![av(&[4])], 8),
        ];
        assert!(route_composed_by_behavior(&noise).is_none(), "must not fabricate a chain for noise");
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
