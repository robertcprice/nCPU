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
    let has = |t: &str| ptoks.iter().any(|p| p == t);
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
    // Try each NL-proposed candidate (best name-coverage first) through the gate;
    // return the FIRST whose proven program reproduces every example. The NL rank
    // is what makes this an oracle beyond the examples: on non-distinguishing
    // examples (e.g. all-True `perfect_square`) the semantically-closest op is tried
    // first, so we prefer the right op over a coincidental match. Bounded to the top
    // candidates so a vague prompt can't scan the whole library.
    const MAX_TRIED: usize = 12;
    for op in ranked_candidates(prompt).into_iter().take(MAX_TRIED) {
        if crate::runtime::code_reproduces_examples(op.mog, examples) {
            let name_toks = content_tokens(op.name);
            let spec = name_toks.len();
            return Some(Route { op, matched_tokens: name_toks, specificity: spec });
        }
    }
    None
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
