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
                expected: 6,
            },
            Example {
                inputs: vec![Value::Int(4), Value::Int(10)],
                expected: 6,
            },
            Example {
                inputs: vec![Value::Int(3), Value::Int(3)],
                expected: 0,
            },
            Example {
                inputs: vec![Value::Int(-2), Value::Int(5)],
                expected: 7,
            },
        ],
        holdouts: vec![],
        reference_code: "",
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
                expected: 4,
            },
            Example {
                inputs: vec![Value::Array(vec![4, -3, 2, -1])],
                expected: 2,
            },
            Example {
                inputs: vec![Value::Array(vec![-5])],
                expected: 0,
            },
            Example {
                inputs: vec![Value::Array(vec![0, 0, 0])],
                expected: 0,
            },
        ],
        holdouts: vec![],
        reference_code: "",
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
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(4), Value::Int(10)],
                    expected: 6,
                },
            ],
            holdouts: vec![],
            reference_code: "",
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
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Array(vec![-5, 5])],
                    expected: 0,
                },
            ],
            holdouts: vec![],
            reference_code: "",
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
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(4), Value::Int(10)],
                    expected: 6,
                },
            ],
            holdouts: vec![],
            reference_code: "",
        };

        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);
        crate::method_router::record_win(&problem, ROUTE_SEARCH_TEACHER);

        let ctx = post_enumerative_context(&problem);
        assert!(!should_try_enumerative(&problem, &ctx, false, false));
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
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Array(vec![-5, 5])],
                    expected: 0,
                },
            ],
            holdouts: vec![],
            reference_code: "",
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
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Int(4), Value::Int(10)],
                    expected: 6,
                },
            ],
            holdouts: vec![],
            reference_code: "",
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
                    expected: 6,
                },
                Example {
                    inputs: vec![Value::Array(vec![4, -1, 2])],
                    expected: 5,
                },
                Example {
                    inputs: vec![Value::Array(vec![-5, 5])],
                    expected: 0,
                },
            ],
            holdouts: vec![],
            reference_code: "",
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
        assert_eq!(result.method, "arr_gradient");
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
                expected: 6,
            },
            Example {
                inputs: vec![Value::Array(vec![4, -1, 2])],
                expected: 5,
            },
            Example {
                inputs: vec![Value::Array(vec![-5, 5])],
                expected: 0,
            },
            Example {
                inputs: vec![Value::Array(vec![])],
                expected: 0,
            },
        ],
        holdouts: vec![],
        reference_code: "",
    };
    let result = solve_problem_after_enumeration(&problem, std::time::Instant::now(), None);
    assert!(result.success, "{:?}", result.error);
    assert_eq!(
        result.method, "arr_gradient",
        "expected array gradient to solve simple fold warm start, got {}",
        result.method
    );
}
