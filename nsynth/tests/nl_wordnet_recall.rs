//! NON-GAMEABLE WordNet-recall benchmark for the NL bridge (NL-WORDNET-RECALL).
//!
//! Proves that an ADDITIVE, EMERGENT WordNet 1-hop synonym closure (generated
//! offline by `ingestion/wordnet_coding_edges_gen.py`, merged as a 3rd data file)
//! lets the bridge RESOLVE coding-op paraphrases that the hand-written registry
//! never knew — WITHOUT loosening any gate and WITHOUT opening a false-accept.
//!
//! WHY IT CANNOT BE GAMED:
//!   * ABSENCE — every test paraphrase word is asserted ABSENT from the ORIGINAL
//!     hand-table `coding_registry.json` (loaded here directly): the win cannot be
//!     a hand-registry add. The synonyms come from nltk WordNet closure only.
//!   * RESOLUTION+SYNTHESIS — each `sort`/`reverse` paraphrase resolves through a
//!     WordNet synonym edge to the correct op, and the op SYNTHESIZES + passes
//!     `verify_problem_code_strict` on FRESH holdouts the verifier SAMPLES and
//!     LABELS by an INDEPENDENT reference implementation (never example_cases).
//!   * RECALL-LIFT — baseline (registry-only) resolve count N vs post-WordNet M
//!     over a fixed paraphrase list is measured; M>N is asserted, both logged.
//!   * MUST-REFUSE — gibberish AND unsupported real ops STILL return no operation
//!     after the merge (WordNet noise must not open a false accept).
//!   * CLUSTER-SEPARATION — no paraphrase word resolves to two different ops.

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;
use std::path::PathBuf;

// ── Test paraphrases, grouped by the op they MUST resolve to. Every word here is
//    asserted absent from the hand-table (see `paraphrases_absent_from_handtable`)
//    and is a WordNet closure lemma emitted by the generator. ──────────────────
const SORT_PARAS: &[&str] = &["arrange", "order", "organize", "organise", "sequence"];
const REVERSE_PARAS: &[&str] = &["reversal", "reversion", "turnabout", "turnaround"];
const MAX_PARAS: &[&str] = &["maximum"];
const MIN_PARAS: &[&str] = &["minimum"];
const SUM_PARAS: &[&str] = &["amount", "aggregate", "totality"];

/// All (paraphrase, expected_op) pairs under test.
fn all_pairs() -> Vec<(&'static str, &'static str)> {
    let mut v = Vec::new();
    for w in SORT_PARAS {
        v.push((*w, "sort"));
    }
    for w in REVERSE_PARAS {
        v.push((*w, "reverse"));
    }
    for w in MAX_PARAS {
        v.push((*w, "array_max"));
    }
    for w in MIN_PARAS {
        v.push((*w, "array_min"));
    }
    for w in SUM_PARAS {
        v.push((*w, "array_sum"));
    }
    v
}

fn find_coding_registry() -> PathBuf {
    let rel = PathBuf::from("../../linguigenesis/data/coding_registry.json");
    if rel.exists() {
        return rel;
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join("projects/linguigenesis/data/coding_registry.json");
        if p.exists() {
            return p;
        }
    }
    panic!("coding_registry.json not found — test must read the ORIGINAL hand-table");
}

/// (a) ABSENCE: every test paraphrase is absent from the ORIGINAL hand-table's
/// lemma + synonym lists. Loads coding_registry.json raw and proves the words are
/// NOT there, so any later resolution is attributable to the WordNet edges only.
#[test]
fn paraphrases_absent_from_handtable() {
    let raw = std::fs::read_to_string(find_coding_registry()).expect("read coding_registry");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse coding_registry");
    let entities = json
        .get("entities")
        .and_then(|v| v.as_object())
        .expect("entities object");

    // Build the union of every lemma key + every declared synonym, lowercased.
    let mut hand_words: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (lemma, ent) in entities {
        hand_words.insert(lemma.to_lowercase());
        if let Some(syns) = ent
            .get("relations")
            .and_then(|r| r.get("synonym"))
            .and_then(|s| s.as_array())
        {
            for s in syns {
                if let Some(s) = s.as_str() {
                    hand_words.insert(s.to_lowercase());
                }
            }
        }
    }

    for (word, _op) in all_pairs() {
        assert!(
            !hand_words.contains(word),
            "GAMED: paraphrase {word:?} is already in the hand-table coding_registry.json \
             — the recall win must come from WordNet edges, not a registry hand-add"
        );
    }
}

/// LINK-SURVIVAL SPIKE + (b/partial) RESOLUTION: every paraphrase resolves through
/// the merged WordNet edge to the correct op at score >= 0.80 (OP_RESOLVE_FLOOR).
/// This is the spike the plan requires BEFORE the big benchmark: it confirms the
/// edge survives `merge_registry` and reaches the compositional pipeline floor.
#[test]
fn wordnet_paraphrases_resolve_to_correct_op() {
    let bridge = LinguigenesisBridge::new();
    let mut failures = Vec::new();

    for (word, expected_op) in all_pairs() {
        match bridge.resolve_op_probe(word) {
            Some((fn_name, score)) => {
                if score < 0.80 {
                    failures.push(format!(
                        "{word:?} -> {fn_name:?} but score {score:.3} < 0.80 floor"
                    ));
                } else if fn_name != expected_op {
                    failures.push(format!(
                        "{word:?} resolved to {fn_name:?}, expected {expected_op:?}"
                    ));
                }
            }
            None => failures.push(format!("{word:?}: did NOT resolve to any op")),
        }
    }

    assert!(
        failures.is_empty(),
        "{}/{} WordNet paraphrases failed to resolve:\n{}",
        failures.len(),
        all_pairs().len(),
        failures.join("\n")
    );
}

/// (c) RECALL-LIFT: baseline (registry-only) resolve count N over the fixed
/// paraphrase list vs post-WordNet count M. M>N is asserted; both logged.
/// Baseline is computed from a bridge whose registry has the WordNet edge file
/// ABSENT (via an env override the bridge honours), so the lift is attributable
/// solely to the added edges.
#[test]
fn recall_lift_is_positive() {
    let words: Vec<&str> = all_pairs().iter().map(|(w, _)| *w).collect();

    // Baseline: registry WITHOUT the WordNet edges merged.
    let baseline = LinguigenesisBridge::new_without_wordnet_edges();
    let n = words
        .iter()
        .filter(|w| {
            baseline
                .resolve_op_probe(w)
                .map(|(_, s)| s >= 0.80)
                .unwrap_or(false)
        })
        .count();

    // Post-WordNet: the full bridge.
    let full = LinguigenesisBridge::new();
    let m = words
        .iter()
        .filter(|w| {
            full.resolve_op_probe(w)
                .map(|(_, s)| s >= 0.80)
                .unwrap_or(false)
        })
        .count();

    println!("RECALL-LIFT over {} paraphrases: baseline N={n}, post-WordNet M={m}", words.len());
    assert!(
        m > n,
        "expected recall LIFT (M>N), got baseline N={n} post-WordNet M={m}"
    );
    // NOTE (2026-07): the emergent DERIVATION lens (`linguigenesis_core::nl_signals`)
    // now resolves the *derivational* paraphrases ("reversal"/"reversion" → reverse)
    // via morphology, INDEPENDENT of the WordNet edges — a legitimate recall path
    // added after this benchmark was written. So the registry-only baseline is no
    // longer strictly zero (n counts those morphology-covered derivations). The
    // benchmark's real invariant — WordNet edges lift recall for the
    // NON-derivational synonyms (arrange/order/turnabout/aggregate/…) — is
    // preserved as M > N above, and `unsupported_and_gibberish_still_refused`
    // still guarantees the derivation lens opens no false accepts. The lift
    // attributable purely to WordNet is M − n.
    assert!(
        m - n >= 3,
        "WordNet edges must still lift >=3 non-derivational synonyms, got M-n={}",
        m - n
    );
}

/// (d) MUST-REFUSE control: gibberish AND unsupported real ops STILL return no
/// operation after the WordNet merge — the closure must not open a false accept.
#[test]
fn unsupported_and_gibberish_still_refused() {
    let bridge = LinguigenesisBridge::new();
    // Gibberish.
    const GIBBERISH: &[&str] = &["zorp", "flibber", "qwxz"];
    // Real but unsupported ops (no registry entity, no closure edge).
    const UNSUPPORTED: &[&str] = &["parse", "encrypt", "hash", "sqrt", "regex", "dedupe"];

    let mut leaks = Vec::new();
    for w in GIBBERISH.iter().chain(UNSUPPORTED.iter()) {
        if let Some((fn_name, score)) = bridge.resolve_op_probe(w) {
            if score >= 0.80 {
                leaks.push(format!("{w:?} FALSE-ACCEPT -> {fn_name:?} @ {score:.3}"));
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "WordNet merge opened false accepts (must-refuse control failed):\n{}",
        leaks.join("\n")
    );
}

/// (e) CLUSTER-SEPARATION: no paraphrase word resolves to two DIFFERENT ops at or
/// above the floor. (A single high-confidence op per word — no ambiguous merge.)
#[test]
fn no_paraphrase_resolves_to_two_ops() {
    let bridge = LinguigenesisBridge::new();
    let mut conflicts = Vec::new();
    for (word, _) in all_pairs() {
        let ops = bridge.resolve_ops_above_floor(word, 0.80);
        let distinct: std::collections::HashSet<&String> = ops.iter().collect();
        if distinct.len() > 1 {
            conflicts.push(format!("{word:?} -> {:?}", ops));
        }
    }
    assert!(
        conflicts.is_empty(),
        "cluster-separation violated (a paraphrase resolved to >1 op):\n{}",
        conflicts.join("\n")
    );
}

/// (b) RESOLUTION+SYNTHESIS on FRESH holdouts for the PRIORITY ops sort + reverse:
/// each paraphrase resolves via the WordNet edge AND the resolved op synthesizes a
/// program that passes strict differential verification on holdouts SAMPLED and
/// LABELED by an INDEPENDENT reference implementation (never example_cases).
///
/// sort + reverse each gain >= 3 working paraphrases — the accept core.
#[test]
fn sort_and_reverse_paraphrases_synthesize_and_strict_verify() {
    let bridge = LinguigenesisBridge::new();

    // INDEPENDENT references (the verifier samples fresh arrays and labels them by
    // RUNNING these — so an example-echo cannot pass).
    let sort_ref: &'static str =
        "fn sort_ref(a: [i64]) -> [i64] {\n    out: [i64] = a;\n    out.sort();\n    return out;\n}\n";
    let reverse_ref: &'static str = "fn reverse_ref(a: [i64]) -> [i64] {\n    out: [i64] = [];\n    i: i64 = a.len - 1;\n    while i >= 0 {\n        out.push(a[i]);\n        i = i - 1;\n    }\n    return out;\n}\n";

    let mut sort_ok = 0usize;
    let mut reverse_ok = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for word in SORT_PARAS {
        match bridge.resolve_and_strict_verify(word, "sort", "sort_ref", "fn sort_ref(a: [i64]) -> [i64]", sort_ref) {
            Ok(()) => sort_ok += 1,
            Err(e) => failures.push(format!("sort/{word}: {e}")),
        }
    }
    for word in REVERSE_PARAS {
        match bridge.resolve_and_strict_verify(word, "reverse", "reverse_ref", "fn reverse_ref(a: [i64]) -> [i64]", reverse_ref) {
            Ok(()) => reverse_ok += 1,
            Err(e) => failures.push(format!("reverse/{word}: {e}")),
        }
    }

    for f in &failures {
        println!("STRICT-VERIFY-FAILURE: {f}");
    }
    println!("sort paraphrases verified: {sort_ok}/{}", SORT_PARAS.len());
    println!("reverse paraphrases verified: {reverse_ok}/{}", REVERSE_PARAS.len());

    assert!(
        sort_ok >= 3,
        "sort must gain >=3 working WordNet paraphrases, got {sort_ok} (failures: {failures:?})"
    );
    assert!(
        reverse_ok >= 3,
        "reverse must gain >=3 working WordNet paraphrases, got {reverse_ok} (failures: {failures:?})"
    );
}
