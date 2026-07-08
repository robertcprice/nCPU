//! Semantic op proposer — the systemic cure for NL-comprehension "whack-a-mole".
//!
//! nSynth's never-wrong front door PROPOSES candidate ops from a prompt, then the
//! strict verify + distinguishing gates confirm-or-refuse. Today the proposer
//! (`verified_nl_router::ranked_candidates`) is TOKEN-ONLY over op names plus a
//! ~30-entry hand synonym table (`capability_miner::nl_surface`). A phrasing whose
//! words don't token-match an op name and isn't in the hand table is refused — so a
//! human hand-adds one synonym per phrasing (whack-a-mole).
//!
//! But the linguigenesis KG already ENCODES those synonyms: `probe_op_candidates`
//! grounds "invert"->reverse (0.82), "combine"->add (0.92), "biggest"->max (0.92),
//! "total"->sum (0.92) via lemma/morphology/synonym+hypernym graph walks. This module
//! turns that into an ADDITIVE proposer: for each content word, ask the resolver which
//! op-CONCEPTS it grounds to (>= min_score), then surface EVERY library op related to
//! that concept. It never decides correctness — it only widens PROPOSE; the caller's
//! verify gate (reproduce-all-incl-held-out) + distinguishing gate keep "never wrong".
//! So this generalises to unseen phrasings with ZERO hand-table edits and ZERO
//! never-wrong risk.

use crate::linguigenesis_bridge::LinguigenesisBridge;
use crate::op_library::{LibOp, OPS};
use std::collections::HashSet;
use std::sync::OnceLock;

/// Process-wide bridge (the 500k registry load is expensive; build once).
fn bridge() -> &'static LinguigenesisBridge {
    static B: OnceLock<LinguigenesisBridge> = OnceLock::new();
    B.get_or_init(LinguigenesisBridge::new)
}

/// Generic name fragments that carry no operation identity — matching on them alone
/// would flood candidates (every array op shares "array"). Dropped from overlap.
const GENERIC_FRAGMENTS: &[&str] = &[
    "array", "list", "number", "value", "element", "item", "string", "str", "seq",
    "sequence", "of", "the", "a", "an", "to", "in", "on", "at", "each", "all",
];

/// Split an op/concept identifier into its identity fragments (snake_case words,
/// generics dropped): "array_max" -> {max}, "max_of_three" -> {max, three},
/// "reverse_list" -> {reverse}. Used for concept<->library-op overlap so a resolved
/// CONCEPT name matches library ops that share a real operation word — bridging the
/// registry-concept vs library-op naming gap (probe returns "array_max"; the op is
/// "max_of_three").
fn identity_fragments(ident: &str) -> Vec<String> {
    ident
        .split('_')
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() >= 3 && !GENERIC_FRAGMENTS.contains(&w.as_str()))
        .collect()
}

/// A resolved op CONCEPT name maps to every library op sharing a non-generic identity
/// fragment — the concept "reverse" (or "array_reverse") surfaces `reverse_list`,
/// `reverse_string`, `reverse_number`; "array_max" surfaces `max_of_three`, `max_two`.
/// The verify gate then picks the one the examples determine. Broad on purpose (PROPOSE
/// widens; the gate is strict) but generics-filtered so it doesn't match everything.
fn ops_for_concept(concept: &str) -> impl Iterator<Item = &'static LibOp> {
    let frags = identity_fragments(concept);
    OPS.iter().filter(move |o| {
        if frags.is_empty() {
            return false;
        }
        let of = identity_fragments(o.name);
        of.iter().any(|f| frags.contains(f))
    })
}

/// Library ops the linguigenesis resolver semantically grounds the prompt's content
/// words to, above `min_score`. Deduped, capped. See module docs — ADDITIVE proposer.
pub fn semantic_op_candidates(prompt: &str, min_score: f32, cap: usize) -> Vec<&'static LibOp> {
    let b = bridge();
    if b.registry_load_error().is_some() {
        return Vec::new();
    }
    let mut seen_ops: HashSet<&'static str> = HashSet::new();
    let mut out: Vec<&'static LibOp> = Vec::new();
    for tok in prompt
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 3)
    {
        for (concept, score, _method) in b.probe_op_candidates(tok) {
            if score < min_score {
                continue;
            }
            for op in ops_for_concept(&concept) {
                if seen_ops.insert(op.name) {
                    out.push(op);
                    if out.len() >= cap {
                        return out;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(prompt: &str) -> Vec<&'static str> {
        semantic_op_candidates(prompt, 0.7, 24).iter().map(|o| o.name).collect()
    }

    /// The whack-a-mole cure: phrasings whose words are NOT in the hand synonym table
    /// and do NOT token-match an op name are still grounded to the right op family via
    /// the KG — so the verify gate has a real candidate to confirm. (Proposer only;
    /// correctness is the gate's job, so we assert the RIGHT op is PROPOSED.)
    #[test]
    fn grounds_novel_synonyms_to_the_right_op_family() {
        // "invert"/"flip" -> reverse_* ; "combine" -> add ; "biggest" -> a max op.
        // Each phrasing uses a synonym ABSENT from the hand table; assert the resolver
        // surfaces the right library-op FAMILY (the gate then picks by examples).
        // ("combine"->add is intentionally omitted: there is no bare `add` library op —
        // addition is the affine search's job, a different tier.)
        let inv = names("invert an array");
        assert!(
            inv.iter().any(|n| n.contains("reverse")),
            "invert should propose a reverse_* op, got {inv:?}"
        );
        let big = names("the biggest of three numbers");
        assert!(
            big.iter().any(|n| n.contains("max")),
            "biggest should propose a max op, got {big:?}"
        );
        let tot = names("the total of a list");
        assert!(
            tot.iter().any(|n| n.contains("sum")),
            "total should propose a sum op, got {tot:?}"
        );
        let sml = names("the smallest value in the list");
        assert!(
            sml.iter().any(|n| n.contains("min")),
            "smallest should propose a min op, got {sml:?}"
        );
    }

    /// Additive + honest: a prompt with no op-grounding content returns nothing (it does
    /// not manufacture candidates), so it can only widen recall, never mislead the gate.
    #[test]
    fn no_grounding_yields_no_candidates_not_garbage() {
        let out = names("qwx zzt");
        assert!(out.is_empty(), "nonsense should propose nothing, got {out:?}");
    }
}
