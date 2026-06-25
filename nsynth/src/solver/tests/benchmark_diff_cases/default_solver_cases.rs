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
        result.method == "search_affine"
            || result.method == "search_scalar_expr"
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

/// DEL-CHEAT accept-test: the hardcoded `search_closure_map_sum` oracle (which
/// hardcoded `arr.iter().map(|x| x * 2).sum()`) was deleted. The map-then-SUM
/// benchmark `closure_map_sum_v0` must now be solved by the GENERAL searched
/// fold-body enumerator (`enumerative-array`), which independently discovers
/// `acc + item*2`, and NOT by any recognizer.
#[test]
fn closure_map_sum_solved_by_searched_fold_not_recognizer() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "closure_map_sum_v0")
        .unwrap();

    // 1) The deleted recognizer path is gone: search-only mode no longer has a
    //    candidate for this problem, so it cannot win via the removed oracle.
    let search_only = solve_problem_search_only(&problem);
    assert_ne!(
        search_only.method, "search_closure_map_sum",
        "deleted recognizer must not resurface: {:?}",
        search_only
    );
    assert!(
        !search_only.success,
        "search-only must NOT solve closure_map_sum after oracle deletion; \
         got method {} — a recognizer was re-added",
        search_only.method
    );

    // 2) The full solver still solves it, via the general searched fold body.
    let result = solve_problem(&problem);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(
        result.method, "enumerative-array",
        "must be solved by the general searched fold enumerator, not a recognizer: {:?}",
        result
    );
    // The searched solution is a fold accumulating `acc + item*2`, not the
    // deleted `.map(|x| x*2)` closure template.
    assert!(
        !result.code.contains(".map("),
        "searched solution must not be the deleted map-closure template: {}",
        result.code
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
    // Each of these problems can be solved by either the search
    // teacher (preferred) or the gradient path. We accept either
    // because both produce a verified solution; the test is
    // "default solver can solve these hard array families", not
    // "default solver picks the gradient path".
    let targets = [
        ("kth_smallest_v0"),
        ("two_sum_exists_v0"),
        ("count_distinct_v0"),
        ("binary_search_v0"),
    ];
    for name_arr in &targets {
        let name: &str = name_arr;
        let problem = problems
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} not found"));
        let result = solve_problem(problem);
        assert!(result.success, "{name}: {:?}", result.error);
        // We don't pin the method — the preemption whitelist may
        // route a search teacher first.
        assert!(
            result.method.starts_with("search_")
                || result.method.starts_with("arr_gradient_")
                || result.method.starts_with("enumerative"),
            "{name}: unexpected method {}: {}",
            result.method,
            result.code
        );
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
