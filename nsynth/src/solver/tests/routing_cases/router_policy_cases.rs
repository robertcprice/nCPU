use super::*;

#[test]
fn solve_problem_prefers_native_gradient_before_scalar_templates() {
    let problem = Problem {
        name: "abs_diff_custom_v0".to_string(),
        category: "test",
        description: "Return the absolute difference between a and b.",
        signature: "fn abs_diff_custom(a: i64, b: i64) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Int(10), Value::Int(4)],
                expected: Value::Int(6),
            },
            Example {
                inputs: vec![Value::Int(4), Value::Int(10)],
                expected: Value::Int(6),
            },
            Example {
                inputs: vec![Value::Int(3), Value::Int(3)],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Int(-2), Value::Int(5)],
                expected: Value::Int(7),
            },
        ],
        holdouts: vec![],
        reference_code: "",

    synthetic_args: Vec::new(),

    synthetic_values: Vec::new(),

    recursive_allowed: false,

    tree_input: false,

    explicit_stack: false,

    };
    let result = solve_problem_after_enumeration(&problem, std::time::Instant::now(), None);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(
        result.method, "synth_gradient",
        "expected native gradient to beat template fallback, got {}",
        result.method
    );
}

#[test]
fn solve_problem_prefers_array_gradient_before_array_templates() {
    let problem = Problem {
        name: "count_positive_custom_v0".to_string(),
        category: "test",
        description: "Return the number of positive entries in the array.",
        signature: "fn count_positive_custom(arr: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Array(vec![1, 2, 3, 4])],
                expected: Value::Int(4),
            },
            Example {
                inputs: vec![Value::Array(vec![4, -3, 2, -1])],
                expected: Value::Int(2),
            },
            Example {
                inputs: vec![Value::Array(vec![-5])],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Array(vec![0, 0, 0])],
                expected: Value::Int(0),
            },
        ],
        holdouts: vec![],
        reference_code: "",

    synthetic_args: Vec::new(),

    synthetic_values: Vec::new(),

    recursive_allowed: false,

    tree_input: false,

    explicit_stack: false,

    };
    let stages = post_enumerative_stage_order(&problem);
    let arr_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::ArrayGradient)
        .expect("array gradient stage missing");
    let expr_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::ExprOnly)
        .expect("expr_only stage missing");
    assert!(
        arr_idx < expr_idx,
        "expected array gradient before expr_only, got {:?}",
        stages
    );
}

#[test]
fn method_router_promotes_learned_search_teacher_to_front() {
    with_scratch_method_router(|| {
        let problem = Problem {
            name: "router_order_custom_v0".to_string(),
            category: "test",
            description: "Return the absolute difference between a and b.",
            signature: "fn router_order_custom(a: i64, b: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(10), Value::Int(4)],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Int(4), Value::Int(10)],
                    expected: Value::Int(6),
                },
            ],
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        };

        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);

        let ctx = post_enumerative_context(&problem);
        let routes = planned_post_enumerative_routes(&problem, &ctx);
        assert_eq!(routes.first().copied(), Some(ROUTE_SEARCH_TEACHER));
        assert!(routes.contains(&ROUTE_SCALAR_GRADIENT));
    });
}

#[test]
fn method_router_normalizes_legacy_array_method_names() {
    with_scratch_method_router(|| {
        let problem = Problem {
            name: "router_array_custom_v0".to_string(),
            category: "test",
            description: "Return the sum of all entries in the array.",
            signature: "fn router_array_custom(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Array(vec![1, 2, 3])],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Array(vec![-5, 5])],
                    expected: Value::Int(0),
                },
            ],
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        };

        crate::method_router::record_win(&problem, "univ_arr_gradient");
        crate::method_router::record_win(&problem, "univ_arr_gradient");

        let ctx = post_enumerative_context(&problem);
        let routes = planned_post_enumerative_routes(&problem, &ctx);
        assert_eq!(routes.first().copied(), Some(ROUTE_ARRAY_GRADIENT));
    });
}

#[test]
fn preemptive_search_teacher_beats_router_array_gradient() {
    with_scratch_method_router(|| {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "run_length_decode_sum_v0")
            .expect("run_length_decode_sum_v0 not found");

        crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);
        crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);

        let ctx = post_enumerative_context(&problem);
        let routes = planned_post_enumerative_routes(&problem, &ctx);
        assert_eq!(routes.first().copied(), Some(ROUTE_SEARCH_TEACHER));

        let preemptive = solve_problem_from_preemptive_search_teacher(&problem);
        let result =
            solve_problem_after_enumeration(&problem, std::time::Instant::now(), preemptive);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "search_run_length_decode_sum");
    });
}

#[test]
fn exact_search_preemption_can_skip_enumerative_without_router_history() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "prefix_sum_k_v0")
        .expect("prefix_sum_k_v0 not found");
    let ctx = post_enumerative_context(&problem);
    assert!(!should_try_enumerative(&problem, &ctx, false, true));
}

#[test]
fn method_router_can_skip_enumerative_after_repeated_late_stage_wins() {
    with_scratch_method_router(|| {
        let problem = Problem {
            name: "router_skip_enum_custom_v0".to_string(),
            category: "test",
            description: "Return the absolute difference between a and b.",
            signature: "fn router_skip_enum_custom(a: i64, b: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(10), Value::Int(4)],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Int(4), Value::Int(10)],
                    expected: Value::Int(6),
                },
            ],
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        };

        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);

        let ctx = post_enumerative_context(&problem);
        assert!(!should_try_enumerative(&problem, &ctx, false, false));
    });
}

#[test]
fn method_router_keeps_enumerative_when_top_route_is_slow_array_gradient() {
    // Regression for the array-sum misroute: once the in-run router
    // accumulated several array_gradient wins for the (arrays, array-input)
    // bucket, `should_try_enumerative` used to skip the ~0.02s enumerative
    // guard in favour of the 20–60s array-gradient route. Skipping a free
    // guard for a slow route is strictly backwards — the guard must always
    // run, because if it solves we avoid the gradient grind entirely. This
    // pins that array_gradient (a `route_dwarfs_enumerative_guard` route)
    // can NEVER trigger the enumerative skip, regardless of win count.
    with_scratch_method_router(|| {
        let problem = Problem {
            name: "router_slow_route_keep_enum_v0".to_string(),
            category: "arrays",
            description: "Return the sum of all entries in the array.",
            signature: "fn router_slow_route_keep_enum(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Array(vec![1, 2, 3])],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Array(vec![-5, 5])],
                    expected: Value::Int(0),
                },
            ],
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        };

        // Many high-confidence array_gradient wins — would have crossed the
        // ENUMERATION_SKIP thresholds under the old policy.
        for _ in 0..8 {
            crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);
        }

        let ctx = post_enumerative_context(&problem);
        assert!(
            should_try_enumerative(&problem, &ctx, false, false),
            "enumerative guard must run even when the router strongly favors \
             the slow array_gradient route"
        );
    });
}

#[test]
fn method_router_keeps_enumerative_when_enum_has_the_bucket() {
    with_scratch_method_router(|| {
        let problem = Problem {
            name: "router_keep_enum_custom_v0".to_string(),
            category: "test",
            description: "Return the sum of all entries in the array.",
            signature: "fn router_keep_enum_custom(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Array(vec![1, 2, 3])],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Array(vec![-5, 5])],
                    expected: Value::Int(0),
                },
            ],
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        };

        crate::method_router::record_win(&problem, "enumerative-array");
        crate::method_router::record_win(&problem, "enumerative-array");
        crate::method_router::record_win(&problem, "enumerative-array");
        crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);
        crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);

        let ctx = post_enumerative_context(&problem);
        assert!(should_try_enumerative(&problem, &ctx, false, false));
    });
}

#[test]
fn cache_bypass_requires_a_stronger_general_route_than_cached_method() {
    with_scratch_method_router(|| {
        let problem = Problem {
            name: "router_cache_policy_custom_v0".to_string(),
            category: "test",
            description: "Return the absolute difference between a and b.",
            signature: "fn router_cache_policy_custom(a: i64, b: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(10), Value::Int(4)],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Int(4), Value::Int(10)],
                    expected: Value::Int(6),
                },
            ],
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        };

        for _ in 0..4 {
            crate::method_router::record_win(&problem, ROUTE_SCALAR_GRADIENT);
        }
        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
        crate::method_router::record_miss(&problem, ROUTE_SEARCH_TEACHER);
        crate::method_router::record_miss(&problem, ROUTE_SEARCH_TEACHER);

        let ctx = post_enumerative_context(&problem);
        let cached = crate::solved_cache::CachedSolution {
            code: code_abs_diff("router_cache_policy_custom"),
            method: "search_abs_diff_formula".to_string(),
            success_count: 0,
            last_used_at: 0,
        };
        assert!(should_bypass_solved_cache(&problem, &ctx, &cached));

        let cached_top_route = crate::solved_cache::CachedSolution {
            code: code_abs_diff("router_cache_policy_custom"),
            method: "synth_gradient".to_string(),
            success_count: 0,
            last_used_at: 0,
        };
        assert!(!should_bypass_solved_cache(
            &problem,
            &ctx,
            &cached_top_route
        ));
    });
}

#[test]
fn solve_problem_can_bypass_cache_and_upgrade_to_router_preferred_route() {
    with_scratch_router_and_cache(|| {
        let problem = Problem {
            name: "router_cache_upgrade_custom_v0".to_string(),
            category: "test",
            description: "Return the sum of all entries in the array.",
            signature: "fn router_cache_upgrade_custom(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Array(vec![1, 2, 3])],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Array(vec![4, -1, 2])],
                    expected: Value::Int(5),
                },
                Example {
                    inputs: vec![Value::Array(vec![-5, 5])],
                    expected: Value::Int(0),
                },
            ],
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        };

        crate::solved_cache::record(
            &problem,
            "search_array_sum",
            &code_array_sum("router_cache_upgrade_custom"),
        );
        for _ in 0..4 {
            crate::method_router::record_win(&problem, ROUTE_ARRAY_GRADIENT);
        }

        let result = solve_problem(&problem);
        assert!(result.success, "{:?}", result.error);
        // The router's 4 array_gradient wins make `solve_problem` bypass the
        // cached `search_array_sum` entry. The cheap enumerative guard then
        // runs *before* the slow array-gradient route (see
        // `route_dwarfs_enumerative_guard`) and recognises the closed-form
        // sum fold in microseconds — strictly better than grinding gradient
        // descent. The key assertion is that we did NOT return the cached
        // method, i.e. the bypass fired and a fresh route solved the problem.
        assert_ne!(
            result.method, "search_array_sum",
            "cache bypass should have re-solved with a non-cached route"
        );
        assert_eq!(
            result.method, "enumerative-array",
            "expected the cheap enumerative guard to preempt the slow \
             array-gradient route on a simple sum fold, got {}",
            result.method
        );
    });
}

#[test]
fn solve_problem_uses_array_gradient_for_simple_sum_fold() {
    let problem = Problem {
        name: "array_sum_custom_v0".to_string(),
        category: "test",
        description: "Return the sum of all entries in the array.",
        signature: "fn array_sum_custom(arr: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Array(vec![1, 2, 3])],
                expected: Value::Int(6),
            },
            Example {
                inputs: vec![Value::Array(vec![4, -1, 2])],
                expected: Value::Int(5),
            },
            Example {
                inputs: vec![Value::Array(vec![-5, 5])],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Array(vec![])],
                expected: Value::Int(0),
            },
        ],
        holdouts: vec![],
        reference_code: "",

    synthetic_args: Vec::new(),

    synthetic_values: Vec::new(),

    recursive_allowed: false,

    tree_input: false,

    explicit_stack: false,

    };
    let result = solve_problem_after_enumeration(&problem, std::time::Instant::now(), None);
    assert!(result.success, "{:?}", result.error);
    // Either native (`arr_gradient`) or universal (`univ_arr_gradient`) array
    // gradient is acceptable — both are the array-gradient family and the
    // warm-start path may discretize into either depending on which restart
    // converges first. (Matches the convention already used in
    // `synthesis::core_impl` tests.)
    assert!(
        result.method == "arr_gradient" || result.method == "univ_arr_gradient",
        "expected an array gradient method to solve the simple fold warm \
         start, got {}",
        result.method
    );
}
