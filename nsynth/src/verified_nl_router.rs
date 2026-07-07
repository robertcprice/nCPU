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
    route(prompt).map(|r| r.op)
}

/// Run an op on a few illustrative inputs so a user can confirm the behavior by
/// recognition (the no-example safety mechanism). Returns (input, output) pairs the
/// op actually produced; skips inputs it can't run on.
pub fn demonstrate(op: &LibOp) -> Vec<(Vec<crate::benchmark::Value>, String)> {
    use crate::benchmark::Value;
    let Some(name) = op_entry_name(op.mog) else { return vec![] };
    let tys: Vec<&str> = {
        let (Some(o), Some(c)) = (op.mog.find('('), op.mog.find(')')) else { return vec![] };
        let inner = op.mog[o + 1..c].trim();
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
        if let Ok(v) = crate::runtime::execute_function(op.mog, name, &inputs, "demo") {
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
fn passers_disagree(passing: &[&'static LibOp], examples: &[crate::benchmark::Example]) -> bool {
    let probes = fresh_probe_inputs(examples);
    let names: Vec<Option<&str>> = passing.iter().map(|op| op_entry_name(op.mog)).collect();
    for inputs in &probes {
        // runtime::Value has no PartialEq, so compare its deterministic Debug form.
        let mut seen: Option<String> = None;
        for (op, name) in passing.iter().zip(&names) {
            let Some(name) = name else { continue };
            // Only successful evaluations count; an error is not a disagreement
            // signal (an op simply undefined on a probe input tells us nothing).
            if let Ok(out) = crate::runtime::execute_function(op.mog, name, inputs, "nl_probe") {
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

    #[test]
    fn declare_resolves_confident_prompts_without_examples() {
        assert_eq!(declare("reverse a string").map(|o| o.name), Some("reverse_string"));
        assert_eq!(declare("sort a list of numbers").map(|o| o.name), Some("sort"));
        // Acronym wins over a coincidental single-token match (all_divisors on "divisor").
        assert_eq!(declare("greatest common divisor of two numbers").map(|o| o.name), Some("gcd"));
        assert_eq!(declare("all divisors of a number").map(|o| o.name), Some("all_divisors"));
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
