//! Recall benchmark: how much NL-phrasing coverage does the untrusted LME lane
//! add OVER the symbolic baseline? Runs a phrasing corpus through (a) symbolic-only
//! and (b) symbolic+LME, and reports the recall lift. Every +LME win must be
//! BOTH strict-verified AND the semantically-right program — here an array FOLD
//! (`for ` over the array), NOT a scalar `add` — so this measures CORRECT recall,
//! not "any verified program". (Mode-B out-of-vocab synthesis is value-verified
//! separately in `local_llm_e2e.rs`; a loose substring check can't tell `3x` from
//! `3x+5`, so it is deliberately NOT measured here.)
//!
//! Gated: skips unless NSYNTH_LOCAL_LLM_URL points at a served model.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

/// (phrasing, substring the verified code/method MUST contain to count as correct).
/// `for ` ⇒ a genuine array fold — discriminates array_sum from a scalar `add`.
const CORPUS: &[(&str, &str)] = &[
    // --- sum-an-array family (symbolic historically mis-resolves to scalar add) ---
    ("add up all the elements of an array", "for "),
    ("total of all the numbers in the list", "for "),
    ("sum everything in the array", "for "),
    ("accumulate the values in the array", "for "),
    ("compute the sum of all entries", "for "),
    // --- filter+reduce composition ---
    ("add up only the positive numbers in the list", "for "),
    ("sum of just the positive values", "for "),
];

fn verified_correct(res: Option<mog_synth::solver::SolveResult>, expect: &str) -> Option<bool> {
    res.map(|r| r.success && (r.code.contains(expect) || r.method.contains(expect)))
}

#[test]
fn llm_recall_lift_over_symbolic() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[RECALL] skipped (no NSYNTH_LOCAL_LLM_URL — serve a model to measure lift)");
        return;
    }
    let bridge = LinguigenesisBridge::new();
    assert!(bridge.registry_load_error().is_none(), "registry must load");

    let rows: Vec<(&str, &str)> = CORPUS.to_vec();

    let (mut sym_ok, mut lme_ok, mut recovered) = (0usize, 0usize, 0usize);
    eprintln!("\n[RECALL] phrasing                                          symbolic  +LME");
    eprintln!("[RECALL] --------------------------------------------------  --------  ----");
    for (phrasing, expect) in &rows {
        // (a) symbolic baseline — NO llm.
        let sym = verified_correct(
            bridge.synthesize_from_description_symbolic(phrasing, None).ok(),
            expect,
        )
        .unwrap_or(false);
        // (b) +LME lane (Mode A → A′ → B).
        let lme = verified_correct(bridge.synthesize_via_local_llm(phrasing), expect).unwrap_or(false);
        // Net: symbolic OR lme (the production wrapper auto-falls-back).
        let net = sym || lme;
        if sym {
            sym_ok += 1;
        }
        if net {
            lme_ok += 1;
        }
        if !sym && lme {
            recovered += 1;
        }
        eprintln!(
            "[RECALL] {:54}  {:8}  {}",
            &phrasing[..phrasing.len().min(54)],
            if sym { "✓" } else { "·" },
            if lme { "✓" } else { "·" }
        );
    }
    let n = rows.len();
    eprintln!(
        "\n[RECALL] symbolic-only: {sym_ok}/{n}   with-LME: {lme_ok}/{n}   recovered by LME: {recovered}"
    );
    // The lane must NEVER regress net recall (auto-fallback only adds), and must
    // recover at least one phrasing the symbolic path misses — else it earns nothing.
    assert!(lme_ok >= sym_ok, "LME must not reduce net recall ({lme_ok} < {sym_ok})");
    assert!(recovered >= 1, "LME recovered no phrasings the symbolic path missed");
}
