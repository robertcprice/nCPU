use super::*;

#[test]
fn default_solver_prefers_differentiable_when_supported() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "add_two_v0")
        .unwrap();
    let result = solve_problem(&problem);
    assert!(result.success, "{:?}", result.error);
    // Exact scalar expression search now preempts the gradient stack for
    // simple closed-form arithmetic, but older routes remain acceptable if
    // search heuristics ever change.
    assert!(
        result.method == "search_scalar_expr"
            || result.method == "search_abs_diff_formula"
            || result.method.starts_with("diff_gradient_")
            || result.method == "synth_gradient"
            || result.method == "enumerative"
            || result.method == "template"
            || result.method == "template_reference",
        "unexpected method: {}",
        result.method
    );
}

#[test]
fn default_solver_prefers_discovery_for_count_positive() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "count_positive_v0")
        .unwrap();
    let result = solve_problem(&problem);
    assert!(result.success, "{:?}", result.error);
    assert!(
        result.method == "enumerative-array"
            || result.method == "arr_gradient"
            || result.method == "univ_arr_gradient",
        "unexpected method: {}",
        result.method
    );
}

#[test]
fn default_solver_uses_gradient_for_dot_product() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "dot_product_v0")
        .unwrap();
    let result = solve_problem(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "arr_gradient");
    assert!(result.code.contains("a[i] * b[i]"), "{}", result.code);
}

#[test]
fn default_solver_uses_structured_gradient_for_hard_array_families() {
    let problems = get_benchmark(1);
    let targets = [
        ("kth_smallest_v0", "arr_gradient_kth_smallest"),
        ("two_sum_exists_v0", "arr_gradient_two_sum_exists"),
        ("count_distinct_v0", "arr_gradient_count_distinct"),
        ("binary_search_v0", "arr_gradient_binary_search"),
    ];
    for (name, expected_method) in targets {
        let problem = problems
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} not found"));
        let result = solve_problem(problem);
        assert!(result.success, "{name}: {:?}", result.error);
        assert_eq!(result.method, expected_method, "{name}: {}", result.code);
    }
}

#[test]
fn default_solver_short_circuits_lcm_to_search_teacher() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "lcm_v0")
        .unwrap();
    let result = solve_problem(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_lcm_formula");
    assert!(result.code.contains("gcd_inner"), "{}", result.code);
}

#[test]
fn default_solver_short_circuits_euler_totient_to_search_teacher() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "euler_totient_v0")
        .unwrap();
    let result = solve_problem(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "search_euler_totient_loop");
    assert!(
        result.code.contains("result = result - result / p;"),
        "{}",
        result.code
    );
}
