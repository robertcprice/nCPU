use super::*;

#[test]
fn search_solves_second_max() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "second_max_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_second_max");
    assert!(result.code.contains("second = first;"));
    assert!(result.code.contains("fn second_max"));
}

#[test]
fn search_solves_array_range() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "array_range_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_array_range");
    assert!(result.code.contains("hi - lo"));
    assert!(result.code.contains("fn array_range"));
}

#[test]
fn search_solves_sum_of_divisors() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "sum_of_divisors_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_sum_of_divisors_loop");
    assert!(result.code.contains("total = total + i;"));
    assert!(result.code.contains("fn sum_of_divisors"));
}

#[test]
fn search_solves_sum_odd_digits() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "sum_odd_digits_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_sum_odd_digits_loop");
    assert!(result.code.contains("(d % 2) == 1"));
    assert!(result.code.contains("fn sum_odd_digits"));
}

#[test]
fn search_solves_count_zeros() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "count_zeros_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_count_zeros");
    assert!(result.code.contains("if item == 0"));
    assert!(result.code.contains("fn count_zeros"));
}

#[test]
fn search_solves_max_consecutive_sum() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "max_consecutive_sum_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_max_consecutive_sum");
    assert!(result.code.contains("current > 0"));
    assert!(result.code.contains("fn max_consecutive_sum"));
}

#[test]
fn search_solves_min_consecutive_sum() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "min_consecutive_sum_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_min_consecutive_sum");
    assert!(result.code.contains("current < 0"));
    assert!(result.code.contains("fn min_consecutive_sum"));
}

#[test]
fn search_solves_alternating_sum() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "alternating_sum_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_alternating_sum");
    assert!(result.code.contains("sign = 0 - sign"));
    assert!(result.code.contains("fn alternating_sum"));
}

#[test]
fn search_solves_count_greater_than() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "count_greater_than_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_array_count_greater_than");
    assert!(result.code.contains("item > k"));
    assert!(result.code.contains("fn count_greater_than"));
}

#[test]
fn search_solves_dot_product() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "dot_product_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_dot_product");
    assert!(result.code.contains("a[i] * b[i]"));
    assert!(result.code.contains("fn dot_product"));
}

#[test]
fn search_solves_leading_digit() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "leading_digit_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_leading_digit");
    assert!(result.code.contains("x >= 10"));
    assert!(result.code.contains("fn leading_digit"));
}

#[test]
fn search_solves_popcount() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "popcount_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_popcount");
    assert!(result.code.contains("x % 2"));
    assert!(result.code.contains("fn popcount"));
}

#[test]
fn search_solves_prefix_sum_k() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "prefix_sum_k_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_prefix_sum_k");
    assert!(result.code.contains("while i < k"));
    assert!(result.code.contains("fn prefix_sum_k"));
}

#[test]
fn search_solves_is_palindrome_arr() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "is_palindrome_arr_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_is_palindrome_arr");
    assert!(result.code.contains("arr.len - 1"));
    assert!(result.code.contains("fn is_palindrome_arr"));
}

#[test]
fn search_solves_sum_odd_indexed() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "sum_odd_indexed_v0")
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_sum_odd_indexed");
    assert!(result.code.contains("i = i + 2"));
    assert!(result.code.contains("fn sum_odd_indexed"));
}

#[test]
#[ignore = "exhaustive portfolio benchmark — use search_only_solves_full_benchmark in CI"]
fn solves_full_benchmark() {
    let problems = get_benchmark(1);
    let summary = solve_benchmark(&problems);
    assert_eq!(
        summary.solved,
        problems.len(),
        "failures: {:?}",
        summary.failures
    );
}

#[test]
#[ignore = "exhaustive portfolio benchmark — use search_only_solves_full_benchmark in CI"]
fn legacy_fallback_entrypoint_still_solves_full_benchmark() {
    let problems = get_benchmark(1);
    let summary = solve_benchmark_with_legacy_fallback(&problems);
    assert_eq!(
        summary.solved,
        problems.len(),
        "failures: {:?}",
        summary.failures
    );
}

#[test]
fn legacy_only_entrypoint_rejects_reference_oracles() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "add_two_v0")
        .unwrap();
    assert!(!problem.reference_code.is_empty());

    let result = solve_problem_legacy_only(&problem);
    assert!(!result.success);
    assert!(result.method.starts_with("legacy_"));
    assert!(result
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("evaluation-only"));
}

#[test]
fn search_output_is_invariant_to_evaluation_oracles() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "add_two_v0")
        .unwrap();
    let mut poisoned = problem.clone();
    poisoned.holdouts = vec![Example {
        inputs: vec![Value::Int(100), Value::Int(-37)],
        expected: Value::Int(999_999),
    }];
    poisoned.reference_code = "fn add_two(a: i64, b: i64) -> i64 { return 999999; }\n";

    let clean_result = solve_problem_search_only(&problem);
    let poisoned_result = solve_problem_search_only(&poisoned);
    assert!(clean_result.success, "{:?}", clean_result.error);
    assert_eq!(clean_result.success, poisoned_result.success);
    assert_eq!(clean_result.method, poisoned_result.method);
    assert_eq!(clean_result.code, poisoned_result.code);

    // Evaluation remains separate: the same candidate passes the real holdouts
    // and fails the deliberately poisoned evaluator oracle.
    crate::runtime::verify_problem_code_strict(&problem, &clean_result.code).unwrap();
    assert!(crate::runtime::verify_problem_code_strict(&poisoned, &clean_result.code).is_err());
}

#[test]
fn search_only_solves_full_benchmark() {
    let problems = get_benchmark(1);
    let summary = solve_benchmark_search_only(&problems);
    assert_eq!(
        summary.solved,
        problems.len(),
        "failures: {:?}",
        summary.failures
    );
    for problem in problems {
        let result = solve_problem_search_only(&problem);
        assert!(result.success, "search-only failed for {}", problem.name);
        assert!(
            result.method.starts_with("search_"),
            "non-search method {} for {}",
            result.method,
            problem.name
        );
        crate::runtime::verify_problem_code_strict(&problem, &result.code).unwrap_or_else(|err| {
            panic!(
                "search-only holdout verification failed for {}: {}",
                problem.name, err
            )
        });
    }
}
