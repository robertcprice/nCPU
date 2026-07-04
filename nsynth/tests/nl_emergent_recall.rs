//! EMERGENT-LENS recall measurement (NL-EMERGENT-RECALL).
//!
//! Quantifies the recall lift that the four emergent lenses (frame / prime /
//! root / phonestheme, see `linguigenesis_core::nl_signals`) add to the live NL
//! resolver, and attributes each win to a lens. Non-gameable by construction:
//!
//!   * DATA-DRIVEN corpus — derivational surfaces are generated from the *real*
//!     op vocabulary (`bridge.op_lemmas()`), never a hand list, so the win can't
//!     be a fixture the resolver was pre-taught.
//!   * BASELINE vs EMERGENT in ONE pass — `probe_op_candidates` returns every op
//!     candidate with its `method`. Baseline = best candidate from a *curated*
//!     scorer (direct/morphology/graph/fuzzy/definition); emergent = best overall.
//!     A "lift" is a surface the emergent lens resolves at the auto-resolve floor
//!     that the curated scorers do not.
//!   * ADDITIVE INVARIANT — emergent must never resolve FEWER surfaces than
//!     baseline at the same floor (the lenses only add candidates).

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;
use std::collections::BTreeMap;

/// Production op-resolve gate (`OP_RESOLVE_FLOOR` in the bridge): a surface must
/// resolve at/above this to actually reach synthesis, so this is the honest
/// "deliverable recall" floor (not the looser 0.55 comprehension tier).
const FLOOR: f32 = 0.80;

const EMERGENT_METHODS: &[&str] = &["frame", "prime", "root", "phonestheme"];

/// Derivational suffixes appended to a real op lemma to synthesise a paraphrase
/// the hand morphology table (which knows only -ing/-ed/-ly/-es/-s) may miss.
const DERIV_SUFFIXES: &[&str] = &["al", "ion", "ation", "er", "ment", "ing", "s"];

fn is_emergent(method: &str) -> bool {
    EMERGENT_METHODS.contains(&method)
}

/// (fn_name, score) of the best op candidate from a curated (non-emergent) scorer.
fn baseline_best(cands: &[(String, f32, String)]) -> Option<(String, f32)> {
    cands
        .iter()
        .find(|(_, _, m)| !is_emergent(m))
        .map(|(f, s, _)| (f.clone(), *s))
}

#[test]
fn emergent_lift_is_measured_and_never_negative() {
    let bridge = LinguigenesisBridge::new();
    let op_lemmas = bridge.op_lemmas();
    assert!(!op_lemmas.is_empty(), "merged registry must expose ops");

    // Build a data-driven derivational corpus: (surface, expected_fn).
    let mut corpus: Vec<(String, String)> = Vec::new();
    for (lemma, fnn) in &op_lemmas {
        if lemma.len() < 4 || !lemma.chars().all(|c| c.is_ascii_alphabetic()) {
            continue;
        }
        for suf in DERIV_SUFFIXES {
            let surface = format!("{lemma}{suf}");
            if &surface == lemma {
                continue;
            }
            corpus.push((surface, fnn.clone()));
        }
    }
    // A small SEMANTIC set (targets frame/prime); kept only for fns that exist.
    let known_fns: std::collections::HashSet<&str> =
        op_lemmas.iter().map(|(_, f)| f.as_str()).collect();
    for (surface, fnn) in [
        ("aggregate", "array_sum"),
        ("tally", "array_sum"),
        ("largest", "array_max"),
        ("smallest", "array_min"),
        ("arrange", "sort"),
        ("organize", "sort"),
    ] {
        if known_fns.contains(fnn) {
            corpus.push((surface.to_string(), fnn.to_string()));
        }
    }

    let mut baseline_hits = 0usize;
    let mut emergent_hits = 0usize;
    let mut lift = 0usize;
    let mut hurt = 0usize;
    let mut per_lens: BTreeMap<String, usize> = BTreeMap::new();
    let mut examples: Vec<String> = Vec::new();

    for (surface, expected) in &corpus {
        let cands = bridge.probe_op_candidates(surface);
        let emergent_top = cands.first().cloned();
        let baseline_top = baseline_best(&cands);

        let baseline_ok = baseline_top
            .as_ref()
            .map(|(f, s)| f == expected && *s >= FLOOR)
            .unwrap_or(false);
        let emergent_ok = emergent_top
            .as_ref()
            .map(|(f, s, _)| f == expected && *s >= FLOOR)
            .unwrap_or(false);

        if baseline_ok {
            baseline_hits += 1;
        }
        if emergent_ok {
            emergent_hits += 1;
        }
        if emergent_ok && !baseline_ok {
            lift += 1;
            if let Some((f, s, m)) = &emergent_top {
                *per_lens.entry(m.clone()).or_default() += 1;
                if examples.len() < 12 {
                    examples.push(format!("  {surface:>18} -> {f} @{s:.3} [{m}]"));
                }
            }
        }
        if baseline_ok && !emergent_ok {
            hurt += 1;
        }
    }

    println!("\n=== NL-EMERGENT-RECALL over {} paraphrases (floor {FLOOR}) ===", corpus.len());
    println!("baseline (curated scorers) resolved : {baseline_hits}");
    println!("emergent (all lenses)      resolved : {emergent_hits}");
    println!("LIFT (emergent-only wins)           : {lift}");
    println!("HURT (emergent displaced a correct) : {hurt}");
    println!("lift by lens: {per_lens:?}");
    if !examples.is_empty() {
        println!("example emergent wins:");
        for e in &examples {
            println!("{e}");
        }
    }

    // Accounting identity always holds.
    assert_eq!(
        emergent_hits,
        baseline_hits + lift - hurt,
        "accounting mismatch: {emergent_hits} != {baseline_hits} + {lift} - {hurt}"
    );
    // NO-DISPLACEMENT GUARANTEE: the curated-ceiling cap means an emergent lens
    // can never outrank a resolution the curated scorers already got right.
    assert_eq!(hurt, 0, "emergent displaced {hurt} correct curated resolutions");
    // ADDITIVE INVARIANT: emergent resolves at least as many as baseline.
    assert!(
        emergent_hits >= baseline_hits,
        "emergent regressed recall: baseline={baseline_hits} emergent={emergent_hits}"
    );
    // The lenses add real recall.
    assert!(lift > 0, "expected positive emergent recall lift, got {lift}");
}
