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

use crate::benchmark::{Example, Problem, Value as BmValue};
use crate::runtime::verify_problem_code_strict;
use crate::solver::{solve_problem, solve_problem_search_only};

fn candidate_methods() -> Vec<&'static str> {
    let mut keys = enumerate_search_candidate_keys();
    keys.sort();
    keys
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
fn search_stateful_reducer_is_registered_and_preempts_gradient() {
    // Stage-1 stateful synthesis: the (scalar, array) -> scalar
    // per-tick reducer. Wired into both the search candidate
    // list and the preemption whitelist so a verified reduce
    // is returned directly instead of being re-distilled.
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_stateful_reducer"),
        "search_stateful_reducer missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_stateful_reducer", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_stateful_reducer not in preemption whitelist; \
         it would be returned only to be re-distilled.",
    );
}

#[test]
fn search_stateful_reducer_solves_state_plus_array_sum() {
    // End-to-end: `f(state, arr) = state + sum(arr)`.
        let problem = Problem {
            name: "stateful_reducer_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn stateful_reducer_v0(state: i64, arr: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![BmValue::Int(0), BmValue::Array(vec![1, 2, 3])],
                expected: BmValue::Int(6),
            },
            Example {
                inputs: vec![BmValue::Int(10), BmValue::Array(vec![5, 0, 0])],
                expected: BmValue::Int(15),
            },
            Example {
                inputs: vec![BmValue::Int(-5), BmValue::Array(vec![2, 9, 1])],
                expected: BmValue::Int(7),
            },
            Example {
                inputs: vec![BmValue::Int(100), BmValue::Array(vec![-1, 0, 0])],
                expected: BmValue::Int(99),
            },
        ],
        holdouts: vec![],
        reference_code: "",
    };
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "stateful reducer failed: {:?}", result.error);
    assert_eq!(result.method, "search_stateful_reducer");
    verify_problem_code_strict(&problem, &result.code)
        .unwrap_or_else(|err| panic!("runtime verify failed: {err}"));
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
        "search_stateful_reducer",
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
