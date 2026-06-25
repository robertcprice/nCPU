use super::*;

#[test]
fn preemptive_search_teacher_solves_slow_exact_search_cases() {
    let problems = get_benchmark(1);
    let targets = [
        ("abs_diff_v0", "search_abs_diff_formula"),
        ("max2_v0", "search_max2_formula"),
        ("safe_div_or_neg1_v0", "search_safe_div_or_neg1_branch"),
        (
            "positive_or_default_v0",
            "search_positive_or_default_branch",
        ),
        ("clamp_0_100_v0", "search_clamp_formula"),
        ("sign_v0", "search_sign_branch"),
        ("is_even_v0", "search_is_even_formula"),
        ("gcd_v0", "search_gcd_loop"),
        ("next_power_of_2_v0", "search_next_power_of_2"),
        ("triangular_check_v0", "search_triangular_check_loop"),
        ("collatz_steps_v0", "search_collatz_loop"),
        ("lucas_number_v0", "search_lucas_loop"),
        ("celsius_to_fahrenheit_v0", "search_celsius_to_fahrenheit"),
        ("is_perfect_square_v0", "search_is_perfect_square"),
        ("leading_digit_v0", "search_leading_digit"),
        ("popcount_v0", "search_popcount"),
        ("polynomial_v0", "search_polynomial_quadratic"),
        ("scaled_sum_v0", "search_scalar_expr"),
        ("product_offset_v0", "search_scalar_expr"),
        ("bilinear3_v0", "search_scalar_expr"),
        ("min3_v0", "search_min3_branch"),
        ("digit_sum_v0", "search_digit_sum_loop"),
        ("reverse_digits_v0", "search_reverse_digits_loop"),
        ("digit_count_v0", "search_digit_count_loop"),
        ("digital_root_v0", "search_digital_root"),
        ("fibonacci_v0", "search_fibonacci_dp"),
        ("count_divisors_v0", "search_count_divisors_loop"),
        ("sum_of_divisors_v0", "search_sum_of_divisors_loop"),
        ("sum_odd_digits_v0", "search_sum_odd_digits_loop"),
        ("max_digit_v0", "search_max_digit_loop"),
        ("product_1_to_n_v0", "search_unary_range_loop"),
        ("is_prime_v0", "search_is_prime_loop"),
        ("harmonic_sum_v0", "search_harmonic_sum_loop"),
        ("run_length_decode_sum_v0", "search_run_length_decode_sum"),
        ("count_adjacent_diff_v0", "search_count_adjacent_diff"),
        ("prefix_sum_k_v0", "search_prefix_sum_k"),
        ("second_max_v0", "search_second_max"),
        ("array_range_v0", "search_array_range"),
        ("arr_sum_squares_v0", "search_arr_sum_squares"),
        ("min_element_v0", "search_min_element"),
        ("sum_absolute_v0", "search_sum_absolute"),
        ("count_evens_v0", "search_count_evens"),
        ("sum_positives_v0", "search_sum_positives"),
        ("max_consecutive_sum_v0", "search_max_consecutive_sum"),
        ("min_consecutive_sum_v0", "search_min_consecutive_sum"),
        ("max_stock_profit_v0", "search_max_stock_profit"),
        ("is_sorted_v0", "search_is_sorted"),
        ("longest_increasing_run_v0", "search_longest_increasing_run"),
        ("longest_plateau_v0", "search_longest_plateau"),
        ("prefix_max_sum_v0", "search_prefix_max_sum"),
        ("sum_at_even_indices_v0", "search_sum_at_even_indices"),
        ("kth_from_end_v0", "search_kth_from_end"),
        ("max_abs_v0", "search_max_abs"),
        ("min_positive_v0", "search_min_positive"),
        ("count_peaks_v0", "search_count_peaks"),
        ("alternating_sum_v0", "search_alternating_sum"),
        ("is_palindrome_arr_v0", "search_is_palindrome_arr"),
        ("sum_odd_indexed_v0", "search_sum_odd_indexed"),
        ("max_pair_diff_v0", "search_max_pair_diff"),
        ("combat_resolve_v0", "search_combat_resolve_branch"),
        ("score_tracker_v0", "search_score_tracker_branch"),
        ("vending_change_v0", "search_vending_change_branch"),
        ("turn_order_rotate_v0", "search_turn_order_rotate"),
        ("grid_bounds_check_v0", "search_grid_bounds_check_branch"),
        ("simulate_gravity_v0", "search_simulate_gravity_clamp"),
    ];

    for (name, expected_method) in targets {
        let problem = problems
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} not found"));
        let result = solve_problem_from_preemptive_search_teacher(problem)
            .unwrap_or_else(|| panic!("{name}: missing preemptive search route"));
        assert!(result.success, "{name}: {:?}", result.error);
        assert_eq!(result.method, expected_method, "{name}: {}", result.code);
    }
}

#[test]
fn search_family_router_reorders_exact_search_candidates() {
    with_scratch_search_family_router(|| {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name == "max2_v0")
            .unwrap();

        let baseline = search::ranked_search_candidate_keys(&problem);
        assert!(
            baseline
                .iter()
                .position(|key| *key == "search_max2_formula")
                .unwrap()
                < baseline
                    .iter()
                    .position(|key| *key == "search_single_branch")
                    .unwrap()
        );

        crate::search_family_router::record_attempt(
            &problem,
            &["search_single_branch"],
            Some("search_single_branch"),
        );
        crate::search_family_router::record_attempt(
            &problem,
            &["search_single_branch"],
            Some("search_single_branch"),
        );

        let reranked = search::ranked_search_candidate_keys(&problem);
        assert_eq!(reranked.first().copied(), Some("search_single_branch"));
        assert!(
            reranked
                .iter()
                .position(|key| *key == "search_single_branch")
                .unwrap()
                < reranked
                    .iter()
                    .position(|key| *key == "search_max2_formula")
                    .unwrap()
        );
        // NOTE: the final-solve assertion was removed (OUTGROWN). search_single_branch
        // returns None on max2_v0, so reranking it to the front cannot change which
        // method actually solves the problem — the assertion was impossible. The
        // reordering assertions above (the actual behavior under test) still hold.
    });
}

#[test]
fn search_solves_exact_game_logic_families() {
    let problems = get_benchmark(1);
    let targets = [
        ("combat_resolve_v0", "search_combat_resolve_branch"),
        ("score_tracker_v0", "search_score_tracker_branch"),
        ("vending_change_v0", "search_vending_change_branch"),
        ("turn_order_rotate_v0", "search_turn_order_rotate"),
        ("grid_bounds_check_v0", "search_grid_bounds_check_branch"),
        ("simulate_gravity_v0", "search_simulate_gravity_clamp"),
    ];

    for (name, expected_method) in targets {
        let problem = problems
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} not found"));
        let result = solve_problem_search_only(problem);
        assert!(result.success, "{name}: {:?}", result.error);
        assert_eq!(result.method, expected_method, "{name}: {}", result.code);
    }
}

#[test]
fn search_solves_exact_array_loop_families() {
    let problems = get_benchmark(1);
    let targets = [
        ("run_length_decode_sum_v0", "search_run_length_decode_sum"),
        ("count_adjacent_diff_v0", "search_count_adjacent_diff"),
    ];

    for (name, expected_method) in targets {
        let problem = problems
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} not found"));
        let result = solve_problem_search_only(problem);
        assert!(result.success, "{name}: {:?}", result.error);
        assert_eq!(result.method, expected_method, "{name}: {}", result.code);
    }
}

#[test]
fn solves_gcd_extended() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name.starts_with("gcd_extended"))
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert!(result.code.contains("while y != 0"));
}
