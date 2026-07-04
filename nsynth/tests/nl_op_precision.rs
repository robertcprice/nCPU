//! NL op-resolution PRECISION + COVERAGE guard (NL-OP-PRECISION).
//!
//! Two invariants over the live merged registry, at the real 0.80 synthesis gate:
//!
//!   * COVERAGE — every example-bearing op resolves from its OWN lemma to its own
//!     fn. A regression here means an op became NL-unreachable.
//!   * PRECISION — generic operand / data-type nouns (integer, string, value, …)
//!     do NOT resolve to an operation. This is the guard that would have caught
//!     "return the magnitude of an integer" -> bit_and (integer was missing from
//!     the operand-noun stop set).
//!
//! Data-driven: the op set comes from `op_lemmas()`; the operand list is generic
//! programming nouns, not op-specific. `sequence` is deliberately EXCLUDED — it is
//! a WordNet-seeded verb synonym of `sort` (to sequence = to arrange in order), an
//! intentional edge asserted by the WordNet-recall benchmark, not a precision leak.

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

const GATE: f32 = 0.80;

#[test]
fn every_op_resolves_from_its_own_lemma() {
    let bridge = LinguigenesisBridge::new();
    let ops = bridge.op_lemmas();
    assert!(!ops.is_empty(), "merged registry must expose ops");

    let mut misses = Vec::new();
    for (lemma, fnn) in &ops {
        match bridge.probe_resolution(lemma) {
            Some((f, s, _)) if &f == fnn && s >= GATE => {}
            other => misses.push(format!("{lemma} -> {fnn}, but resolved {other:?}")),
        }
    }
    assert!(
        misses.is_empty(),
        "{}/{} ops are NOT reachable from their own lemma at the {GATE} gate:\n{}",
        misses.len(),
        ops.len(),
        misses.join("\n")
    );
}

#[test]
fn known_synonyms_resolve_to_the_correct_op() {
    // POSITIVE coverage: common paraphrases must resolve to their intended op at
    // the gate. Locks in the WordNet-closure + cross-file-relink + derivational
    // wins against resolution regressions. Each pair is a distinct-word -> fn the
    // engine is expected to handle (curated synonym, WordNet edge, or morphology).
    let bridge = LinguigenesisBridge::new();
    const PAIRS: &[(&str, &str)] = &[
        // array reducers
        ("maximum", "array_max"), ("minimum", "array_min"),
        ("largest", "array_max"), ("smallest", "array_min"),
        ("total", "array_sum"), ("aggregate", "array_sum"),
        // cross-file relink (flip/invert live in coding_registry, reverse in mined)
        ("flip", "reverse"), ("invert", "reverse"),
        // WordNet synonym / derivational closure
        ("arrange", "sort"), ("arrangement", "sort"),
        ("quotient", "divide"), ("multiplication", "multiply"),
        ("deduct", "subtract"), ("increase", "increment"), ("decrease", "decrement"),
    ];
    let mut misses = Vec::new();
    for (w, expected) in PAIRS {
        match bridge.probe_resolution(w) {
            Some((f, s, _)) if &f == expected && s >= GATE => {}
            other => misses.push(format!("{w} -> {expected}, but resolved {other:?}")),
        }
    }
    assert!(
        misses.is_empty(),
        "{}/{} known synonyms no longer resolve to their op at the {GATE} gate:\n{}",
        misses.len(),
        PAIRS.len(),
        misses.join("\n")
    );
}

#[test]
fn generic_imperative_verbs_and_fillers_do_not_resolve_to_ops() {
    // Generic imperatives ("compute", "calculate", …) and structural fillers are
    // NOT operations; if one leaks to an op at the gate it pollutes composition
    // (e.g. "compute" -> "cmp" via the emergent root lens broke "compute the
    // total"). They must be stop-worded, not resolved.
    let bridge = LinguigenesisBridge::new();
    const FILLERS: &[&str] = &[
        "compute", "calculate", "determine", "return", "produce", "obtain",
        "perform", "give", "make", "use", "get", "find", "then", "some",
    ];
    let mut leaks = Vec::new();
    for w in FILLERS {
        if let Some((f, s, m)) = bridge.probe_resolution(w) {
            if s >= GATE {
                leaks.push(format!("{w} -> {f} @{s:.2} [{m}]"));
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "generic imperative/filler words leaked to an operation at the {GATE} gate \
         (add them to grammar_stop_words):\n{}",
        leaks.join("\n")
    );
}

#[test]
fn generic_operand_nouns_do_not_resolve_to_ops() {
    let bridge = LinguigenesisBridge::new();
    // Generic programming nouns a user names as OPERANDS, never as the operation.
    // `sequence` excluded: it is a deliberate WordNet verb-synonym of `sort`.
    const OPERANDS: &[&str] = &[
        "integer", "string", "number", "value", "list", "array", "element",
        "index", "item", "result", "output", "input", "data", "collection",
        "boolean", "float", "character", "variable", "argument", "parameter",
    ];
    let mut leaks = Vec::new();
    for w in OPERANDS {
        if let Some((f, s, m)) = bridge.probe_resolution(w) {
            if s >= GATE {
                leaks.push(format!("{w} -> {f} @{s:.2} [{m}]"));
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "operand/data-type nouns leaked to an operation at the {GATE} gate \
         (add them to grammar_stop_words):\n{}",
        leaks.join("\n")
    );
}
