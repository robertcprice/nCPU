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
        "search_first_index_of",
        "search_last_index_of",
        "search_is_anagram",
        "search_longest_run",
        "search_intersects",
        "search_kth_smallest",
        "search_count_distinct",
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

#[test]
fn search_first_index_of_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_first_index_of"),
        "search_first_index_of missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_first_index_of", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_first_index_of not in preemption whitelist.",
    );
}

#[test]
fn search_last_index_of_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_last_index_of"),
        "search_last_index_of missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_last_index_of", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_last_index_of not in preemption whitelist.",
    );
}

#[test]
fn search_count_distinct_is_registered_and_preempts_gradient() {
    // The pre-existing search_count_distinct teacher already covers
    // count-unique. Regression here just keeps the contract honest.
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_count_distinct"),
        "search_count_distinct missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_count_distinct", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_count_distinct not in preemption whitelist.",
    );
}

#[test]
fn search_is_anagram_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_is_anagram"),
        "search_is_anagram missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_is_anagram", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_is_anagram not in preemption whitelist.",
    );
}

#[test]
fn search_longest_run_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_longest_run"),
        "search_longest_run missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_longest_run", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_longest_run not in preemption whitelist.",
    );
}

#[test]
fn search_intersects_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_intersects"),
        "search_intersects missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_intersects", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_intersects not in preemption whitelist.",
    );
}

#[test]
fn search_kth_smallest_is_registered_and_preempts_gradient() {
    // The pre-existing search_kth_smallest teacher takes (arr, k) as
    // binary parameters. It's already wired into SEARCH_CANDIDATES and
    // the preemption whitelist (since 8b08548). The regression here is
    // a no-op assertion that the contract still holds; a stronger
    // check would require mining example data, which is out of scope.
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_kth_smallest"),
        "search_kth_smallest missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_kth_smallest", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_kth_smallest not in preemption whitelist.",
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
