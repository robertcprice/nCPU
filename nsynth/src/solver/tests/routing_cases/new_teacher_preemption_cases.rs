//! Regression guard: every new search teacher must be both *registered*
//! in `SEARCH_CANDIDATES` and *preempting* the slow native-gradient path.
//!
//! These two lists are the single source of truth for which verified
//! teachers are honored end to end. If a teacher is added to the
//! candidate list but not to the preemption whitelist, it would be
//! returned but then re-distilled (wasted work); if it is preempting but
//! not registered, the solver would never pick it. Both fail this
//! test — caught before any benchmark regression can sneak in.

use super::super::post_enumerative::search_result_preempts_native_gradient;
use super::super::search::enumerate_search_candidate_keys;

fn candidate_methods() -> Vec<&'static str> {
    let mut keys = enumerate_search_candidate_keys();
    keys.sort();
    keys
}

#[test]
fn search_array_feature_dnf_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_array_feature_dnf"),
        "search_array_feature_dnf missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_array_feature_dnf", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_array_feature_dnf not in preemption whitelist; it would \
         be returned only to be re-distilled through the slow gradient path.",
    );
}

#[test]
fn search_string_subsequence_class_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_string_subsequence_class"),
        "search_string_subsequence_class missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_string_subsequence_class", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_string_subsequence_class not in preemption whitelist; \
         it would be returned only to be re-distilled.",
    );
}

#[test]
fn every_preempting_method_has_a_registered_search_candidate() {
    // Inverse guard: no "preempting method" can be a phantom — every
    // method name that the preemption whitelist accepts must be in
    // SEARCH_CANDIDATES, otherwise the whitelist is lying.
    // Note: kept narrow on purpose. Many preemption entries come from
    // other routing families (e.g. affine / modular_cases / piecewise
    // affine) and are *not* in SEARCH_CANDIDATES — this test would be
    // noisy. It covers the newest search teachers only.
    let candidates = candidate_methods();
    for m in &[
        "search_array_feature_dnf",
        "search_string_subsequence_class",
        "search_strictly_increasing",
        "search_has_strictly_increasing_run",
    ] {
        assert!(
            candidates.contains(m),
            "preemption whitelist claims {m:?} but the method is not \
             registered in SEARCH_CANDIDATES — a phantom entry.",
        );
    }
}

#[test]
fn search_strictly_increasing_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_strictly_increasing"),
        "search_strictly_increasing missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_strictly_increasing", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_strictly_increasing not in preemption whitelist; it would \
         be returned only to be re-distilled through the slow gradient path.",
    );
}

#[test]
fn search_has_strictly_increasing_run_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_has_strictly_increasing_run"),
        "search_has_strictly_increasing_run missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_has_strictly_increasing_run", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_has_strictly_increasing_run not in preemption whitelist.",
    );
}

fn make_solve_result(method: &str, code: &str) -> crate::solver::SolveResult {
    use crate::differentiable::DifferentiableMetadata;
    crate::solver::SolveResult {
        success: true,
        code: code.to_string(),
        method: method.to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    }
}
