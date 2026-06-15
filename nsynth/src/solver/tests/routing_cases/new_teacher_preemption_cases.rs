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
        "search_stateful_reducer_dual",
        "search_stateful_reducer_event",
        "search_stateful_replace",
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
fn search_stateful_reducer_dual_is_registered_and_preempts_gradient() {
    // Stage-1.5 3-arg stateful reducer: (state, a, b) -> state with
    // `state = state OP1 r1(a) OP2 r2(b)`. Wired into both the search
    // candidate list and the preemption whitelist.
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_stateful_reducer_dual"),
        "search_stateful_reducer_dual missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_stateful_reducer_dual", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_stateful_reducer_dual not in preemption whitelist.",
    );
}

#[test]
fn search_stateful_reducer_dual_solves_delta_accumulator() {
    // End-to-end: `f(state, a, b) = state + sum(a) - sum(b)`.
    let problem = Problem {
        name: "delta_accumulator_v0".to_string(),
        category: "test",
        description: "test",
        signature: "fn delta_accumulator_v0(state: i64, a: [i64], b: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![
                    BmValue::Int(0),
                    BmValue::Array(vec![1, 2, 3]),
                    BmValue::Array(vec![1, 0, 0]),
                ],
                expected: BmValue::Int(5),
            },
            Example {
                inputs: vec![
                    BmValue::Int(10),
                    BmValue::Array(vec![5, 5]),
                    BmValue::Array(vec![2, 3]),
                ],
                expected: BmValue::Int(15),
            },
            Example {
                inputs: vec![
                    BmValue::Int(-5),
                    BmValue::Array(vec![3, 3, 3]),
                    BmValue::Array(vec![1, 1, 1]),
                ],
                expected: BmValue::Int(1),
            },
        ],
        holdouts: vec![],
        reference_code: "",
    };
    let result = solve_problem_search_only(&problem);
    assert!(
        result.success,
        "dual stateful reducer failed: {:?}",
        result.error
    );
    assert_eq!(result.method, "search_stateful_reducer_dual");
    verify_problem_code_strict(&problem, &result.code)
        .unwrap_or_else(|err| panic!("runtime verify failed: {err}"));
}

#[test]
fn search_stateful_replace_is_registered_and_preempts_gradient() {
    // Stage-1.5 2-arg stateful replace: (state, arr) -> state with
    // conditional update `if pred(arr) then state = g(arr) else state`.
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_stateful_replace"),
        "search_stateful_replace missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_stateful_replace", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_stateful_replace not in preemption whitelist.",
    );
}

#[test]
fn search_stateful_replace_solves_flip_on_positive() {
    // End-to-end: `f(state, arr) = if any(arr > 0) then -state else state`.
    let problem = Problem {
        name: "flip_on_positive_v0".to_string(),
        category: "test",
        description: "test",
        signature: "fn flip_on_positive_v0(state: i64, arr: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![BmValue::Int(1), BmValue::Array(vec![3, 7, 1])],
                expected: BmValue::Int(-1),
            },
            Example {
                inputs: vec![BmValue::Int(5), BmValue::Array(vec![-1, -2])],
                expected: BmValue::Int(5),
            },
            Example {
                inputs: vec![BmValue::Int(0), BmValue::Array(vec![-3])],
                expected: BmValue::Int(0),
            },
            Example {
                inputs: vec![BmValue::Int(-7), BmValue::Array(vec![0, 0, 1])],
                expected: BmValue::Int(7),
            },
        ],
        holdouts: vec![],
        reference_code: "",
    };
    let result = solve_problem_search_only(&problem);
    assert!(
        result.success,
        "stateful replace failed: {:?}",
        result.error
    );
    assert_eq!(result.method, "search_stateful_replace");
    verify_problem_code_strict(&problem, &result.code)
        .unwrap_or_else(|err| panic!("runtime verify failed: {err}"));
}

#[test]
fn search_stateful_reducer_event_is_registered_and_preempts_gradient() {
    // Stage-1.5 3-arg event-modulated stateful reducer:
    // (state, event, arr) -> state. Wired into both the search
    // candidate list and the preemption whitelist.
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_stateful_reducer_event"),
        "search_stateful_reducer_event missing from SEARCH_CANDIDATES.",
    );
    let fake = make_solve_result("search_stateful_reducer_event", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_stateful_reducer_event not in preemption whitelist.",
    );
}

#[test]
fn search_stateful_reducer_event_solves_event_modulated_sum() {
    // End-to-end: `f(state, event, arr) = state + event * sum(arr)`.
    let problem = Problem {
        name: "event_modulated_sum_v0".to_string(),
        category: "test",
        description: "test",
        signature: "fn event_modulated_sum_v0(state: i64, event: i64, arr: [i64]) -> i64",
        examples: vec![
            // state=0, event=3, sum=6  -> 0 + 3*6 = 18
            Example {
                inputs: vec![
                    BmValue::Int(0),
                    BmValue::Int(3),
                    BmValue::Array(vec![1, 2, 3]),
                ],
                expected: BmValue::Int(18),
            },
            // state=10, event=2, sum=5 -> 10 + 2*5 = 20
            Example {
                inputs: vec![
                    BmValue::Int(10),
                    BmValue::Int(2),
                    BmValue::Array(vec![1, 4]),
                ],
                expected: BmValue::Int(20),
            },
            // state=5, event=0, sum=4 -> 5 + 0*4 = 5  (event gates off)
            Example {
                inputs: vec![
                    BmValue::Int(5),
                    BmValue::Int(0),
                    BmValue::Array(vec![1, 3]),
                ],
                expected: BmValue::Int(5),
            },
            // state=-3, event=-2, sum=4 -> -3 + -2*4 = -11
            Example {
                inputs: vec![
                    BmValue::Int(-3),
                    BmValue::Int(-2),
                    BmValue::Array(vec![1, 3]),
                ],
                expected: BmValue::Int(-11),
            },
        ],
        holdouts: vec![],
        reference_code: "",
    };
    let result = solve_problem_search_only(&problem);
    assert!(
        result.success,
        "event-modulated reducer failed: {:?}",
        result.error
    );
    assert_eq!(result.method, "search_stateful_reducer_event");
    verify_problem_code_strict(&problem, &result.code)
        .unwrap_or_else(|err| panic!("runtime verify failed: {err}"));
}

#[test]
fn search_stateful_reducer_event_solves_gated_contribution() {
    // End-to-end: `f(state, event, arr) = if event > 0 then state + sum(arr) else state`.
    let problem = Problem {
        name: "gated_contribution_v0".to_string(),
        category: "test",
        description: "test",
        signature: "fn gated_contribution_v0(state: i64, event: i64, arr: [i64]) -> i64",
        examples: vec![
            // event=1, sum=6  -> 0 + 6 = 6
            Example {
                inputs: vec![
                    BmValue::Int(0),
                    BmValue::Int(1),
                    BmValue::Array(vec![1, 2, 3]),
                ],
                expected: BmValue::Int(6),
            },
            // event=0 -> state unchanged
            Example {
                inputs: vec![
                    BmValue::Int(7),
                    BmValue::Int(0),
                    BmValue::Array(vec![1, 2, 3]),
                ],
                expected: BmValue::Int(7),
            },
            // event=-1 -> state unchanged
            Example {
                inputs: vec![
                    BmValue::Int(-2),
                    BmValue::Int(-1),
                    BmValue::Array(vec![10, 20]),
                ],
                expected: BmValue::Int(-2),
            },
            // event=5, sum=9 -> 4 + 9 = 13
            Example {
                inputs: vec![
                    BmValue::Int(4),
                    BmValue::Int(5),
                    BmValue::Array(vec![4, 5]),
                ],
                expected: BmValue::Int(13),
            },
        ],
        holdouts: vec![],
        reference_code: "",
    };
    let result = solve_problem_search_only(&problem);
    assert!(
        result.success,
        "gated contribution failed: {:?}",
        result.error
    );
    assert_eq!(result.method, "search_stateful_reducer_event");
    verify_problem_code_strict(&problem, &result.code)
        .unwrap_or_else(|err| panic!("runtime verify failed: {err}"));
}
