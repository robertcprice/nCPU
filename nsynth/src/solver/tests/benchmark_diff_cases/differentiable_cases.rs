use super::*;

#[test]
fn differentiable_only_solves_add_two() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "add_two_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert!(result.method.starts_with("diff_gradient_"));
    assert!(result.code.contains("return a + b;"));
}

#[test]
fn prefer_differentiable_keeps_gradient_for_supported_family() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "add_two_v0")
        .unwrap();
    let result = solve_problem_prefer_differentiable(&problem);
    assert!(result.success, "{:?}", result.error);
    assert!(
        result.method == "diff_gradient_arithmetic"
            || result.method == "search_scalar_expr"
            || result.method == "synth_gradient",
        "expected differentiable solve when the bridge is available, otherwise native/search fallback; got {}",
        result.method
    );
}

#[test]
fn prefer_differentiable_skips_probe_for_positive_or_default() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "positive_or_default_v0")
        .unwrap();
    let result = solve_problem_prefer_differentiable(&problem);
    assert!(result.success, "{:?}", result.error);
    assert!(
        result.method.starts_with("search_")
            || result.method.starts_with("diff_gradient_")
            || result.method == "synth_gradient",
        "expected search/native differentiable fallback, got {}",
        result.method
    );
}

#[test]
fn prefer_differentiable_skips_probe_for_is_prime() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "is_prime_v0")
        .unwrap();
    let result = solve_problem_prefer_differentiable(&problem);
    assert!(result.success, "{:?}", result.error);
    assert!(
        result.method == "search_is_prime_loop"
            || result.method.starts_with("diff_gradient_")
            || result.method == "synth_gradient",
        "expected search/native differentiable fallback, got {}",
        result.method
    );
}

#[test]
fn differentiable_only_solves_abs_diff() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "abs_diff_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_branch");
    assert!(result.code.contains("return a - b;"));
    assert!(result.code.contains("return b - a;"));
}

#[test]
fn differentiable_only_rejects_array_problem() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "array_sum_v0")
        .unwrap();
    let result = solve_problem_differentiable_only(&problem);
    assert!(!result.success);
    assert_eq!(result.method, "diff_gradient_unsupported");
}

#[test]
fn differentiable_only_solves_sign() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "sign_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_soft_multi_branch");

    for (input, expected) in [(-8, -1), (0, 0), (15, 1)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_clamp() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "clamp_0_100_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_soft_multi_branch");

    for (input, expected) in [(-1, 0), (42, 42), (101, 100)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_safe_div() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "safe_div_or_neg1_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_branch");

    for ((a, b), expected) in [((9, 0), -1), ((21, 7), 3)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(a), Value::Int(b)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_is_even() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "is_even_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_soft_multi_branch");

    for (input, expected) in [(-6, 1), (20, 1), (105, 0)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_sum_to_n() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "sum_to_n_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_loop");

    for (input, expected) in [(7, 28), (-3, 0)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_factorial() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "factorial_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_loop");

    for (input, expected) in [(3, 6), (8, 40320)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_digit_sum() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "digit_sum_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_digit_loop");

    for (input, expected) in [(405, 9), (7001, 8)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_reverse_digits() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "reverse_digits_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_digit_loop");

    for (input, expected) in [(81, 18), (12030, 3021)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_digit_count() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "digit_count_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_digit_loop");

    for (input, expected) in [(81, 2), (12030, 5)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}

#[test]
fn differentiable_only_solves_count_even_digits() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "count_even_digits_v0")
        .unwrap();
    let Some(result) = solve_problem_differentiable_only_or_skip(&problem) else {
        return;
    };
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "diff_gradient_digit_loop");

    for (input, expected) in [(81, 1), (12030, 3), (24680, 5)] {
        let exec = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &[Value::Int(input)],
            &problem,
        )
        .unwrap();
        match exec {
            crate::runtime::Value::Int(value) => assert_eq!(value, expected),
            other => panic!("expected int result, got {:?}", other),
        }
    }
}
