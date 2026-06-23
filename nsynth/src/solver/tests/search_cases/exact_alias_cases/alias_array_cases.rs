use super::*;

#[test]
fn search_solves_aliased_array_sum_without_family_name() {
    let problem = aliased_problem(
        "array_sum",
        "mystery_reduce_v0",
        "fn mystery_reduce(xs: [i64]) -> i64",
        "array_search",
        "Return the total of all elements in xs.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_array_sum");
    assert!(result.code.contains("for item in arr"));
    assert!(result.code.contains("fn mystery_reduce"));
}

#[test]
fn search_solves_aliased_count_positive_without_family_name() {
    let problem = aliased_problem(
        "count_positive",
        "mystery_positive_counter_v0",
        "fn mystery_positive_counter(xs: [i64]) -> i64",
        "array_search",
        "Count how many entries in xs are strictly above zero.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_array_count_positive");
    assert!(result.code.contains("if item > 0"));
    assert!(result.code.contains("fn mystery_positive_counter"));
}

#[test]
fn search_solves_aliased_count_occurrences_without_family_name() {
    let problem = aliased_problem(
        "count_occurrences",
        "mystery_matches_v0",
        "fn mystery_matches(xs: [i64], needle: i64) -> i64",
        "array_search",
        "Count how many entries in xs equal needle.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_array_count_occurrences");
    assert!(result.code.contains("if item == target"));
    assert!(result.code.contains("fn mystery_matches"));
}

#[test]
fn search_solves_aliased_max_pair_diff_without_family_name() {
    let problem = aliased_problem(
        "max_pair_diff",
        "mystery_pair_diff_v0",
        "fn mystery_pair_diff(arr: [i64]) -> i64",
        "array_search",
        "Return the maximum absolute gap between consecutive elements.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_max_pair_diff");
    assert!(result.code.contains("arr[i] - arr[i - 1]"));
    assert!(result.code.contains("fn mystery_pair_diff"));
}
