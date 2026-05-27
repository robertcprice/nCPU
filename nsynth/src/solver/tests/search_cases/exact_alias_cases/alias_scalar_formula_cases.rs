use super::*;

#[test]
fn search_solves_aliased_lcm_without_family_name() {
    let problem = aliased_problem(
        "lcm",
        "mystery_lcm_v0",
        "fn mystery_lcm(a: i64, b: i64) -> i64",
        "scalar_search",
        "Return the least common multiple of a and b.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_lcm_formula");
    assert!(result.code.contains("gcd_inner"));
    assert!(result.code.contains("fn mystery_lcm"));
}

#[test]
fn search_solves_aliased_add_two_without_family_name() {
    let problem = aliased_problem(
        "add_two",
        "mystery_plus_v0",
        "fn mystery_plus(left: i64, right: i64) -> i64",
        "scalar_search",
        "Return the sum of the two inputs.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_scalar_expr");
    assert!(result.code.contains('+'));
    assert!(result.code.contains("fn mystery_plus"));
}

#[test]
fn search_solves_aliased_abs_diff_without_family_name() {
    let problem = aliased_problem(
        "abs_diff",
        "mystery_gap_v0",
        "fn mystery_gap(left: i64, right: i64) -> i64",
        "scalar_search",
        "Return the absolute difference between the two inputs.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_abs_diff_formula");
    assert!(result.code.contains("fn mystery_gap"));
}

#[test]
fn search_solves_aliased_polynomial_without_family_name() {
    let problem = aliased_problem(
        "polynomial",
        "mystery_quadratic_v0",
        "fn mystery_quadratic(x: i64) -> i64",
        "scalar_search",
        "Evaluate a small quadratic polynomial of x.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_polynomial_quadratic");
    assert!(result.code.contains("x * x"));
    assert!(result.code.contains("fn mystery_quadratic"));
}

#[test]
fn search_solves_aliased_min3_without_family_name() {
    let problem = aliased_problem(
        "min3",
        "mystery_min3_v0",
        "fn mystery_min3(a: i64, b: i64, c: i64) -> i64",
        "scalar_search",
        "Return the minimum of three integers.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_min3_branch");
    assert!(result.code.contains("if b < m"));
    assert!(result.code.contains("fn mystery_min3"));
}

#[test]
fn search_solves_aliased_safe_div_without_family_name() {
    let mut problem = aliased_problem(
        "safe_div_or_neg1",
        "mystery_safe_div_v0",
        "fn mystery_safe_div(a: i64, b: i64) -> i64",
        "scalar_search",
        "Return a divided by b, or -1 when b is zero.",
    );
    problem.examples.push(Example {
        inputs: vec![Value::Int(20), Value::Int(4)],
        expected: 5,
    });
    problem.examples.push(Example {
        inputs: vec![Value::Int(8), Value::Int(2)],
        expected: 4,
    });
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_safe_div_or_neg1_branch");
    assert!(result.code.contains("helper_div"));
    assert!(result.code.contains("=> -1"));
    assert!(result.code.contains(" / "));
    assert!(result.code.contains("fn mystery_safe_div"));
}

#[test]
fn search_solves_aliased_clamp_without_family_name() {
    let problem = aliased_problem(
        "clamp_0_100",
        "mystery_clamp_v0",
        "fn mystery_clamp(value: i64) -> i64",
        "scalar_search",
        "Clamp value into the inclusive range from 0 to 100.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_clamp_formula");
    assert!(result.code.matches("if ").count() >= 2);
    assert!(result.code.contains("return 100;"));
    assert!(result.code.contains("fn mystery_clamp"));
}

#[test]
fn search_solves_aliased_sign_without_family_name() {
    let problem = aliased_problem(
        "sign",
        "mystery_sign_v0",
        "fn mystery_sign(value: i64) -> i64",
        "scalar_search",
        "Return -1 for negative values, 0 for zero, and 1 for positive values.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_sign_branch");
    assert!(result.code.matches("if ").count() >= 2);
    assert!(result.code.contains("return -1;"));
    assert!(result.code.contains("return 1;"));
    assert!(result.code.contains("fn mystery_sign"));
}
