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
/// Rule: an op qualifies only if EVERY content token of its name appears (stemmed)
/// in the prompt — a partial name match is not enough (that is how you get
/// `count_divisors` firing on "divisor" when the user meant `sum_divisors`). Among
/// qualifiers the MOST SPECIFIC (most name tokens) wins; a tie at the top
/// specificity is genuine ambiguity, so we REFUSE rather than pick arbitrarily.
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
        // EVERY content token of the op name must be present in the prompt.
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
                tie_at_best = true; // two equally specific ops -> ambiguous
            }
            None => {
                best = Some(Route { op, matched_tokens: name_toks, specificity: spec });
            }
            _ => {}
        }
    }
    // Ambiguity at the top specificity -> refuse (never guess between two proven ops).
    if tie_at_best {
        return None;
    }
    best
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
pub fn route_verified<'a>(
    prompt: &str,
    examples: &'a [crate::benchmark::Example],
) -> Option<Route> {
    let r = route(prompt)?;
    if examples.is_empty() {
        return None; // no oracle -> cannot confirm -> refuse rather than guess
    }
    if crate::runtime::code_reproduces_examples(r.op.mog, examples) {
        Some(r)
    } else {
        None
    }
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
}
