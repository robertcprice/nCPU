use crate::search_family_router;

use super::scalar_search::{
    build_deep_expr_candidates, code_scalar_return_expr, code_scalar_single_branch,
    code_scalar_two_branch, code_unary_range_loop, cond_is_total, cond_selection,
    cond_selection_on_mask, expr_matches_subset, expr_matches_target, extract_scalar_examples,
    render_scalar_expr, scalar_expr_complexity, scalar_search_context,
    score_single_branch_candidate, score_two_branch_candidate, simulate_unary_range_loop,
    RangeAccumOp, RangeLoopCmp, RangeLoopTerm,
};
use super::search_catalog::{
    search_alternating_sum, search_arr_sum_squares, search_binary_search,
    search_celsius_to_fahrenheit, search_count_distinct, search_count_evens, search_count_peaks,
    search_count_zeros, search_digital_root, search_dot_product, search_is_palindrome_arr,
    search_is_perfect_square, search_is_sorted, search_kth_from_end, search_kth_smallest,
    search_leading_digit, search_longest_increasing_run, search_longest_plateau, search_lucas_loop,
    search_max_abs, search_max_consecutive_sum, search_max_stock_profit,
    search_min_consecutive_sum, search_min_element, search_min_positive, search_next_power_of_2,
    search_popcount, search_prefix_max_sum, search_sum_absolute, search_sum_at_even_indices,
    search_sum_odd_indexed, search_sum_positives, search_two_sum_exists,
};
use super::*;

fn search_scalar_expr(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    let param_names = scalar_param_names(arity);
    let target: Vec<i64> = problem.examples.iter().map(|ex| ex.expected).collect();

    let mut candidates = build_deep_expr_candidates(arity, &examples);
    candidates.sort_by_key(|c| {
        (
            scalar_expr_complexity(&c.expr),
            render_scalar_expr(&c.expr, &param_names),
        )
    });

    let candidate = candidates
        .iter()
        .find(|c| expr_matches_target(&c.outputs, &target))?;
    let code = code_scalar_return_expr(fn_name, &param_names, &candidate.expr);
    verified_result(problem, code, "search_scalar_expr")
}

fn search_unary_range_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = extract_scalar_examples(problem)?;
    if examples.first()?.len() != 1 {
        return None;
    }

    let candidates = [
        (
            0,
            1,
            RangeLoopCmp::Le,
            RangeAccumOp::Add,
            RangeLoopTerm::Index,
        ),
        (
            0,
            1,
            RangeLoopCmp::Le,
            RangeAccumOp::Add,
            RangeLoopTerm::IndexSquared,
        ),
        (
            1,
            1,
            RangeLoopCmp::Le,
            RangeAccumOp::Mul,
            RangeLoopTerm::Index,
        ),
        (
            1,
            1,
            RangeLoopCmp::Le,
            RangeAccumOp::Mul,
            RangeLoopTerm::IndexSquared,
        ),
        (
            0,
            0,
            RangeLoopCmp::Lt,
            RangeAccumOp::Add,
            RangeLoopTerm::Index,
        ),
        (
            0,
            0,
            RangeLoopCmp::Lt,
            RangeAccumOp::Add,
            RangeLoopTerm::IndexSquared,
        ),
        (
            1,
            0,
            RangeLoopCmp::Lt,
            RangeAccumOp::Mul,
            RangeLoopTerm::Index,
        ),
        (
            1,
            0,
            RangeLoopCmp::Lt,
            RangeAccumOp::Mul,
            RangeLoopTerm::IndexSquared,
        ),
    ];

    for (init, start, cmp, op, term) in candidates {
        let matches = problem
            .examples
            .iter()
            .zip(examples.iter())
            .all(|(example, args)| {
                simulate_unary_range_loop(args[0], init, start, cmp, op, term)
                    == Some(example.expected)
            });
        if !matches {
            continue;
        }
        let code = code_unary_range_loop(fn_name, init, start, cmp, op, term);
        if let Some(result) = verified_result(problem, code, "search_unary_range_loop") {
            return Some(result);
        }
    }

    None
}

fn search_power_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(
        problem,
        |base, exp| {
            if exp < 0 {
                0
            } else {
                base.pow(exp as u32)
            }
        },
    ) {
        return None;
    }
    verified_result(
        problem,
        code_power_loop_search(fn_name),
        "search_power_loop",
    )
}

fn search_collatz_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, collatz_steps) {
        return None;
    }
    verified_result(problem, code_collatz_steps(fn_name), "search_collatz_loop")
}

fn search_is_prime_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, is_prime) {
        return None;
    }
    verified_result(problem, code_is_prime(fn_name), "search_is_prime_loop")
}

fn search_digit_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if validate_unary_int(problem, digit_sum) {
        return verified_result(
            problem,
            code_digit_sum_loop_search(fn_name),
            "search_digit_sum_loop",
        );
    }
    if validate_unary_int(problem, reverse_digits) {
        return verified_result(
            problem,
            code_reverse_digits_loop_search(fn_name),
            "search_reverse_digits_loop",
        );
    }
    if validate_unary_int(problem, digit_count) {
        return verified_result(
            problem,
            code_digit_count_loop_search(fn_name),
            "search_digit_count_loop",
        );
    }
    if validate_unary_int(problem, count_even_digits) {
        return verified_result(
            problem,
            code_count_even_digits_loop_search(fn_name),
            "search_count_even_digits_loop",
        );
    }
    if validate_unary_int(problem, |mut n| {
        let mut acc = 1i64;
        while n > 0 {
            acc *= n % 10;
            n /= 10;
        }
        acc
    }) {
        return verified_result(
            problem,
            code_digit_product(fn_name),
            "search_digit_product_loop",
        );
    }
    if validate_unary_int(problem, |mut n| {
        let mut best = 0i64;
        while n > 0 {
            let d = n % 10;
            if d > best {
                best = d;
            }
            n /= 10;
        }
        best
    }) {
        return verified_result(problem, code_max_digit(fn_name), "search_max_digit_loop");
    }
    None
}

fn search_fib_iter_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    // fibonacci overflows i64 for n > 92; skip problems with large inputs
    let max_input = problem
        .examples
        .iter()
        .filter_map(|ex| int_value(&ex.inputs[0]))
        .map(|v| v.abs())
        .max()
        .unwrap_or(0);
    if max_input > 92 {
        return None;
    }
    if !validate_unary_int(problem, fibonacci) {
        return None;
    }
    verified_result(
        problem,
        code_fib_iter_loop_search(fn_name),
        "search_fib_iter_loop",
    )
}

fn search_count_divisors_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |n| (1..=n).filter(|d| n % d == 0).count() as i64) {
        return None;
    }
    verified_result(
        problem,
        code_count_divisors(fn_name),
        "search_count_divisors_loop",
    )
}

fn search_harmonic_sum_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, harmonic_sum) {
        return None;
    }
    verified_result(
        problem,
        code_harmonic_sum(fn_name),
        "search_harmonic_sum_loop",
    )
}

fn search_triangular_check_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, triangular_check) {
        return None;
    }
    verified_result(
        problem,
        code_triangular_check(fn_name),
        "search_triangular_check_loop",
    )
}

fn search_euler_totient_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, euler_totient) {
        return None;
    }
    verified_result(
        problem,
        code_euler_totient(fn_name),
        "search_euler_totient_loop",
    )
}

fn search_lcm_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |a, b| (a * b) / gcd(a, b)) {
        return None;
    }
    verified_result(problem, code_lcm(fn_name), "search_lcm_formula")
}

fn search_polynomial_quadratic(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = extract_scalar_examples(problem)?;
    if examples.first()?.len() != 1 {
        return None;
    }

    for a in -5..=5 {
        for b in -5..=5 {
            for c in -10..=10 {
                let matches =
                    problem
                        .examples
                        .iter()
                        .zip(examples.iter())
                        .all(|(example, args)| {
                            let x = args[0];
                            a * x * x + b * x + c == example.expected
                        });
                if !matches {
                    continue;
                }
                let code = code_quadratic_search(fn_name, a, b, c);
                if let Some(result) = verified_result(problem, code, "search_polynomial_quadratic")
                {
                    return Some(result);
                }
            }
        }
    }
    None
}

fn search_min3_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_ternary_int(problem, |a, b, c| a.min(b).min(c)) {
        return None;
    }
    verified_result(problem, code_min3(fn_name), "search_min3_branch")
}

fn search_combat_resolve(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |attack, defense| {
        let damage = i128::from(attack) - i128::from(defense);
        damage.max(0).min(i128::from(i64::MAX)) as i64
    }) {
        return None;
    }
    verified_result(
        problem,
        code_combat_resolve(fn_name),
        "search_combat_resolve_branch",
    )
}

fn search_score_tracker(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |score, event| match event {
        0 => score + 1,
        1 => score + 5,
        2 => 0,
        _ => score,
    }) {
        return None;
    }
    verified_result(
        problem,
        code_score_tracker(fn_name),
        "search_score_tracker_branch",
    )
}

fn search_vending_change(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |coins_in, price| {
        if coins_in >= price {
            coins_in - price
        } else {
            -1
        }
    }) {
        return None;
    }
    verified_result(
        problem,
        code_vending_change(fn_name),
        "search_vending_change_branch",
    )
}

fn search_turn_order_rotate(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |current, num_players| {
        if num_players <= 0 {
            0
        } else {
            (current + 1) % num_players
        }
    }) {
        return None;
    }
    verified_result(
        problem,
        code_turn_order_rotate(fn_name),
        "search_turn_order_rotate",
    )
}

fn search_grid_bounds_check(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types
        != [
            ParamType::I64,
            ParamType::I64,
            ParamType::I64,
            ParamType::I64,
        ]
    {
        return None;
    }
    if !validate_quaternary_int(problem, |x, y, w, h| {
        if x < 0 || y < 0 || x >= w || y >= h {
            0
        } else {
            1
        }
    }) {
        return None;
    }
    verified_result(
        problem,
        code_grid_bounds_check(fn_name),
        "search_grid_bounds_check_branch",
    )
}

fn search_simulate_gravity(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_ternary_int(problem, |v, g, t| {
        let velocity = i128::from(v) + i128::from(g) * i128::from(t);
        velocity.clamp(0, 100) as i64
    }) {
        return None;
    }
    verified_result(
        problem,
        code_simulate_gravity(fn_name),
        "search_simulate_gravity_clamp",
    )
}

fn search_trimmed_len(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if unary_string_examples(problem).is_none() {
        return None;
    }
    if !validate_unary_str(problem, |s| s.trim().chars().count() as i64) {
        return None;
    }
    verified_result(problem, code_trimmed_len(fn_name), "search_trimmed_len")
}

fn search_contains_literal(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let strings = unary_string_examples(problem)?;
    let mut candidates = Vec::new();
    for (example, value) in problem.examples.iter().zip(strings.iter()) {
        if example.expected != 1 {
            continue;
        }
        let chars = value.chars().collect::<Vec<_>>();
        for start in 0..chars.len() {
            for end in (start + 1)..=chars.len().min(start + 4) {
                candidates.push(chars[start..end].iter().collect::<String>());
            }
        }
    }
    candidates.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    candidates.dedup();

    for candidate in candidates {
        let matches = problem
            .examples
            .iter()
            .zip(strings.iter())
            .all(|(example, value)| {
                (if value.contains(&candidate) { 1 } else { 0 }) == example.expected
            });
        if !matches {
            continue;
        }
        let code = code_contains_literal_search(fn_name, &candidate);
        if let Some(result) = verified_result(problem, code, "search_contains_literal") {
            return Some(result);
        }
    }
    None
}

fn search_starts_with_literal(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let strings = unary_string_examples(problem)?;
    let mut candidates = Vec::new();
    for (example, value) in problem.examples.iter().zip(strings.iter()) {
        if example.expected != 1 {
            continue;
        }
        let chars = value.chars().collect::<Vec<_>>();
        for end in 1..=chars.len().min(4) {
            candidates.push(chars[..end].iter().collect::<String>());
        }
    }
    candidates.sort();
    candidates.dedup();

    for candidate in candidates {
        let matches = problem
            .examples
            .iter()
            .zip(strings.iter())
            .all(|(example, value)| {
                (if value.starts_with(&candidate) { 1 } else { 0 }) == example.expected
            });
        if !matches {
            continue;
        }
        let code = code_starts_with_literal_search(fn_name, &candidate);
        if let Some(result) = verified_result(problem, code, "search_starts_with_literal") {
            return Some(result);
        }
    }
    None
}

fn search_vowel_count(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if unary_string_examples(problem).is_none() {
        return None;
    }
    if !validate_unary_str(problem, |s| {
        s.chars()
            .filter(|c| matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u'))
            .count() as i64
    }) {
        return None;
    }
    verified_result(problem, code_vowel_count(fn_name), "search_vowel_count")
}

fn search_count_words(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if unary_string_examples(problem).is_none() {
        return None;
    }
    if !validate_unary_str(problem, count_words) {
        return None;
    }
    verified_result(problem, code_count_words(fn_name), "search_count_words")
}

fn search_palindrome(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if unary_string_examples(problem).is_none() {
        return None;
    }
    if !validate_unary_str(problem, |s| {
        let chars: Vec<char> = s.chars().collect();
        if chars.iter().eq(chars.iter().rev()) {
            1
        } else {
            0
        }
    }) {
        return None;
    }
    verified_result(problem, code_palindrome_check(fn_name), "search_palindrome")
}

fn search_struct_pair_patterns(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    let ParamType::Other(type_name) = param_types.first()?.clone() else {
        return None;
    };
    let _pairs = unary_pair_examples(problem)?;

    if type_name == "Point" && validate_unary_pair(problem, |x, y| x + y) {
        return verified_result(problem, code_point_sum(fn_name), "search_struct_pair");
    }
    if type_name == "Rectangle" && validate_unary_pair(problem, |w, h| w * h) {
        return verified_result(problem, code_rectangle_area(fn_name), "search_struct_pair");
    }
    None
}

fn search_closure_map_sum(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| arr.iter().map(|x| x * 2).sum()) {
        return None;
    }
    verified_result(
        problem,
        code_closure_map_sum(fn_name),
        "search_closure_map_sum",
    )
}

fn search_max_pair_diff(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        arr.windows(2)
            .map(|w| (w[0] - w[1]).abs())
            .max()
            .unwrap_or(0)
    }) {
        return None;
    }
    verified_result(problem, code_max_pair_diff(fn_name), "search_max_pair_diff")
}

fn search_single_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let ctx = scalar_search_context(problem)?;
    let mut best: Option<((usize, usize, usize, usize, String), SolveResult)> = None;

    for cond in &ctx.cond_candidates {
        if !cond_is_total(&cond.outputs) {
            continue;
        }
        let Some(true_mask) = cond_selection(&cond.outputs, true) else {
            continue;
        };
        let Some(false_mask) = cond_selection(&cond.outputs, false) else {
            continue;
        };
        let Some(then_expr) = ctx
            .expr_candidates
            .iter()
            .find(|candidate| expr_matches_subset(&candidate.outputs, &ctx.target, &true_mask))
        else {
            continue;
        };
        let Some(else_expr) = ctx
            .expr_candidates
            .iter()
            .find(|candidate| expr_matches_subset(&candidate.outputs, &ctx.target, &false_mask))
        else {
            continue;
        };
        let code = code_scalar_single_branch(
            fn_name,
            &ctx.param_names,
            cond,
            &then_expr.expr,
            &else_expr.expr,
        );
        if let Some(result) = verified_result(problem, code, "search_single_branch") {
            let score = score_single_branch_candidate(
                &ctx.param_names,
                cond,
                &then_expr.expr,
                &else_expr.expr,
            );
            let replace = best
                .as_ref()
                .map(|(best_score, _)| score < *best_score)
                .unwrap_or(true);
            if replace {
                best = Some((score, result));
            }
        }
    }

    best.map(|(_, result)| result)
}

fn search_two_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let ctx = scalar_search_context(problem)?;
    let mut best: Option<((usize, usize, usize, usize, String), SolveResult)> = None;

    for first_cond in &ctx.cond_candidates {
        if !cond_is_total(&first_cond.outputs) {
            continue;
        }
        let Some(first_true_mask) = cond_selection(&first_cond.outputs, true) else {
            continue;
        };
        let Some(first_false_mask) = cond_selection(&first_cond.outputs, false) else {
            continue;
        };
        let Some(first_expr) = ctx.expr_candidates.iter().find(|candidate| {
            expr_matches_subset(&candidate.outputs, &ctx.target, &first_true_mask)
        }) else {
            continue;
        };

        for second_cond in &ctx.cond_candidates {
            let Some(second_true_mask) =
                cond_selection_on_mask(&second_cond.outputs, &first_false_mask, true)
            else {
                continue;
            };
            let Some(second_false_mask) =
                cond_selection_on_mask(&second_cond.outputs, &first_false_mask, false)
            else {
                continue;
            };
            let Some(second_expr) = ctx.expr_candidates.iter().find(|candidate| {
                expr_matches_subset(&candidate.outputs, &ctx.target, &second_true_mask)
            }) else {
                continue;
            };
            let Some(else_expr) = ctx.expr_candidates.iter().find(|candidate| {
                expr_matches_subset(&candidate.outputs, &ctx.target, &second_false_mask)
            }) else {
                continue;
            };
            let code = code_scalar_two_branch(
                fn_name,
                &ctx.param_names,
                first_cond,
                &first_expr.expr,
                second_cond,
                &second_expr.expr,
                &else_expr.expr,
            );
            if let Some(result) = verified_result(problem, code, "search_two_branch") {
                let score = score_two_branch_candidate(
                    &ctx.param_names,
                    first_cond,
                    &first_expr.expr,
                    second_cond,
                    &second_expr.expr,
                    &else_expr.expr,
                );
                let replace = best
                    .as_ref()
                    .map(|(best_score, _)| score < *best_score)
                    .unwrap_or(true);
                if replace {
                    best = Some((score, result));
                }
            }
        }
    }

    best.map(|(_, result)| result)
}

fn search_array_item_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);

    match param_types.as_slice() {
        [ParamType::ArrayI64] => {
            if validate_unary_array(problem, |arr| arr.iter().sum()) {
                return verified_result(problem, code_array_sum(fn_name), "search_array_sum");
            }
            if validate_unary_array(problem, |arr| *arr.iter().max().unwrap_or(&0)) {
                return verified_result(problem, code_array_max(fn_name), "search_array_max");
            }
            if validate_unary_array(problem, |arr| arr.iter().filter(|x| **x > 0).count() as i64) {
                return verified_result(
                    problem,
                    code_count_positive(fn_name),
                    "search_array_count_positive",
                );
            }
            if validate_unary_array(problem, |arr| arr.iter().filter(|x| **x < 0).sum()) {
                return verified_result(
                    problem,
                    code_sum_negatives(fn_name),
                    "search_array_sum_negatives",
                );
            }
        }
        [ParamType::ArrayI64, ParamType::I64] => {
            if validate_array_and_int(problem, |arr, target| {
                arr.iter().filter(|x| **x == target).count() as i64
            }) {
                return verified_result(
                    problem,
                    code_count_occurrences(fn_name),
                    "search_array_count_occurrences",
                );
            }
            if validate_array_and_int(problem, |arr, k| {
                arr.iter().filter(|&&x| x > k).count() as i64
            }) {
                return verified_result(
                    problem,
                    code_count_greater_than(fn_name),
                    "search_array_count_greater_than",
                );
            }
            if validate_array_and_int(problem, |arr, k| arr.iter().take(k as usize).sum()) {
                return verified_result(problem, code_prefix_sum_k(fn_name), "search_prefix_sum_k");
            }
        }
        _ => {}
    }

    None
}

fn search_run_length_decode_sum(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        let mut total = 0i64;
        let mut i = 0usize;
        while i + 1 < arr.len() {
            total += arr[i] * arr[i + 1];
            i += 2;
        }
        total
    }) {
        return None;
    }
    verified_result(
        problem,
        code_run_length_decode_sum(fn_name),
        "search_run_length_decode_sum",
    )
}

fn search_count_adjacent_diff(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, |arr| {
        let mut count = 0i64;
        for i in 1..arr.len() {
            if arr[i] != arr[i - 1] {
                count += 1;
            }
        }
        count
    }) {
        return None;
    }
    verified_result(
        problem,
        code_count_adjacent_diff(fn_name),
        "search_count_adjacent_diff",
    )
}

fn search_gcd_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, gcd) {
        return None;
    }
    verified_result(problem, code_gcd(fn_name), "search_gcd_loop")
}

fn search_abs_diff_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |a, b| (a - b).abs()) {
        return None;
    }
    verified_result(problem, code_abs_diff(fn_name), "search_abs_diff_formula")
}

fn search_max2_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |a, b| a.max(b)) {
        return None;
    }
    verified_result(problem, code_max2(fn_name), "search_max2_formula")
}

fn search_safe_div_or_neg1_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |a, b| if b == 0 { -1 } else { a / b }) {
        return None;
    }
    verified_result(
        problem,
        code_safe_div_or_neg1(fn_name),
        "search_safe_div_or_neg1_branch",
    )
}

fn search_clamp_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |x| x.clamp(0, 100)) {
        return None;
    }
    verified_result(problem, code_clamp(fn_name), "search_clamp_formula")
}

fn search_sign_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |x| {
        if x < 0 {
            -1
        } else if x > 0 {
            1
        } else {
            0
        }
    }) {
        return None;
    }
    verified_result(problem, code_sign(fn_name), "search_sign_branch")
}

fn search_positive_or_default_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |x| if x > 0 { x } else { 0 }) {
        return None;
    }
    verified_result(
        problem,
        code_positive_or_default(fn_name),
        "search_positive_or_default_branch",
    )
}

fn search_is_even_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |x| if x % 2 == 0 { 1 } else { 0 }) {
        return None;
    }
    verified_result(problem, code_is_even(fn_name), "search_is_even_formula")
}

fn search_second_max(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, second_max) {
        return None;
    }
    verified_result(problem, code_second_max(fn_name), "search_second_max")
}

fn search_array_range(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;
    }
    if !validate_unary_array(problem, array_range) {
        return None;
    }
    verified_result(problem, code_array_range(fn_name), "search_array_range")
}

fn search_sum_of_divisors_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, sum_of_divisors) {
        return None;
    }
    verified_result(
        problem,
        code_sum_of_divisors(fn_name),
        "search_sum_of_divisors_loop",
    )
}

fn search_sum_odd_digits_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, sum_odd_digits) {
        return None;
    }
    verified_result(
        problem,
        code_sum_odd_digits(fn_name),
        "search_sum_odd_digits_loop",
    )
}

type SearchFn = fn(&Problem, &str) -> Option<SolveResult>;

#[derive(Clone, Copy)]
struct SearchCandidate {
    key: &'static str,
    func: SearchFn,
}

const SEARCH_CANDIDATES: &[SearchCandidate] = &[
    SearchCandidate {
        key: "search_array_item_loop",
        func: search_array_item_loop,
    },
    SearchCandidate {
        key: "search_run_length_decode_sum",
        func: search_run_length_decode_sum,
    },
    SearchCandidate {
        key: "search_count_adjacent_diff",
        func: search_count_adjacent_diff,
    },
    SearchCandidate {
        key: "search_second_max",
        func: search_second_max,
    },
    SearchCandidate {
        key: "search_array_range",
        func: search_array_range,
    },
    SearchCandidate {
        key: "search_min_element",
        func: search_min_element,
    },
    SearchCandidate {
        key: "search_max_consecutive_sum",
        func: search_max_consecutive_sum,
    },
    SearchCandidate {
        key: "search_min_consecutive_sum",
        func: search_min_consecutive_sum,
    },
    SearchCandidate {
        key: "search_kth_smallest",
        func: search_kth_smallest,
    },
    SearchCandidate {
        key: "search_max_stock_profit",
        func: search_max_stock_profit,
    },
    SearchCandidate {
        key: "search_is_sorted",
        func: search_is_sorted,
    },
    SearchCandidate {
        key: "search_longest_increasing_run",
        func: search_longest_increasing_run,
    },
    SearchCandidate {
        key: "search_digital_root",
        func: search_digital_root,
    },
    SearchCandidate {
        key: "search_two_sum_exists",
        func: search_two_sum_exists,
    },
    SearchCandidate {
        key: "search_count_distinct",
        func: search_count_distinct,
    },
    SearchCandidate {
        key: "search_binary_search",
        func: search_binary_search,
    },
    SearchCandidate {
        key: "search_longest_plateau",
        func: search_longest_plateau,
    },
    SearchCandidate {
        key: "search_prefix_max_sum",
        func: search_prefix_max_sum,
    },
    SearchCandidate {
        key: "search_arr_sum_squares",
        func: search_arr_sum_squares,
    },
    SearchCandidate {
        key: "search_sum_absolute",
        func: search_sum_absolute,
    },
    SearchCandidate {
        key: "search_count_evens",
        func: search_count_evens,
    },
    SearchCandidate {
        key: "search_sum_positives",
        func: search_sum_positives,
    },
    SearchCandidate {
        key: "search_sum_at_even_indices",
        func: search_sum_at_even_indices,
    },
    SearchCandidate {
        key: "search_kth_from_end",
        func: search_kth_from_end,
    },
    SearchCandidate {
        key: "search_max_abs",
        func: search_max_abs,
    },
    SearchCandidate {
        key: "search_lucas_loop",
        func: search_lucas_loop,
    },
    SearchCandidate {
        key: "search_celsius_to_fahrenheit",
        func: search_celsius_to_fahrenheit,
    },
    SearchCandidate {
        key: "search_is_perfect_square",
        func: search_is_perfect_square,
    },
    SearchCandidate {
        key: "search_next_power_of_2",
        func: search_next_power_of_2,
    },
    SearchCandidate {
        key: "search_min_positive",
        func: search_min_positive,
    },
    SearchCandidate {
        key: "search_count_peaks",
        func: search_count_peaks,
    },
    SearchCandidate {
        key: "search_alternating_sum",
        func: search_alternating_sum,
    },
    SearchCandidate {
        key: "search_dot_product",
        func: search_dot_product,
    },
    SearchCandidate {
        key: "search_leading_digit",
        func: search_leading_digit,
    },
    SearchCandidate {
        key: "search_popcount",
        func: search_popcount,
    },
    SearchCandidate {
        key: "search_is_palindrome_arr",
        func: search_is_palindrome_arr,
    },
    SearchCandidate {
        key: "search_sum_odd_indexed",
        func: search_sum_odd_indexed,
    },
    SearchCandidate {
        key: "search_count_zeros",
        func: search_count_zeros,
    },
    SearchCandidate {
        key: "search_closure_map_sum",
        func: search_closure_map_sum,
    },
    SearchCandidate {
        key: "search_max_pair_diff",
        func: search_max_pair_diff,
    },
    SearchCandidate {
        key: "search_struct_pair_patterns",
        func: search_struct_pair_patterns,
    },
    SearchCandidate {
        key: "search_trimmed_len",
        func: search_trimmed_len,
    },
    SearchCandidate {
        key: "search_starts_with_literal",
        func: search_starts_with_literal,
    },
    SearchCandidate {
        key: "search_contains_literal",
        func: search_contains_literal,
    },
    SearchCandidate {
        key: "search_vowel_count",
        func: search_vowel_count,
    },
    SearchCandidate {
        key: "search_count_words",
        func: search_count_words,
    },
    SearchCandidate {
        key: "search_palindrome",
        func: search_palindrome,
    },
    SearchCandidate {
        key: "search_gcd_loop",
        func: search_gcd_loop,
    },
    SearchCandidate {
        key: "search_abs_diff_formula",
        func: search_abs_diff_formula,
    },
    SearchCandidate {
        key: "search_max2_formula",
        func: search_max2_formula,
    },
    SearchCandidate {
        key: "search_safe_div_or_neg1_branch",
        func: search_safe_div_or_neg1_branch,
    },
    SearchCandidate {
        key: "search_positive_or_default_branch",
        func: search_positive_or_default_branch,
    },
    SearchCandidate {
        key: "search_clamp_formula",
        func: search_clamp_formula,
    },
    SearchCandidate {
        key: "search_sign_branch",
        func: search_sign_branch,
    },
    SearchCandidate {
        key: "search_is_even_formula",
        func: search_is_even_formula,
    },
    SearchCandidate {
        key: "search_lcm_formula",
        func: search_lcm_formula,
    },
    SearchCandidate {
        key: "search_unary_range_loop",
        func: search_unary_range_loop,
    },
    SearchCandidate {
        key: "search_power_loop",
        func: search_power_loop,
    },
    SearchCandidate {
        key: "search_collatz_loop",
        func: search_collatz_loop,
    },
    SearchCandidate {
        key: "search_is_prime_loop",
        func: search_is_prime_loop,
    },
    SearchCandidate {
        key: "search_digit_loop",
        func: search_digit_loop,
    },
    SearchCandidate {
        key: "search_fib_iter_loop",
        func: search_fib_iter_loop,
    },
    SearchCandidate {
        key: "search_count_divisors_loop",
        func: search_count_divisors_loop,
    },
    SearchCandidate {
        key: "search_sum_of_divisors_loop",
        func: search_sum_of_divisors_loop,
    },
    SearchCandidate {
        key: "search_sum_odd_digits_loop",
        func: search_sum_odd_digits_loop,
    },
    SearchCandidate {
        key: "search_harmonic_sum_loop",
        func: search_harmonic_sum_loop,
    },
    SearchCandidate {
        key: "search_triangular_check_loop",
        func: search_triangular_check_loop,
    },
    SearchCandidate {
        key: "search_euler_totient_loop",
        func: search_euler_totient_loop,
    },
    SearchCandidate {
        key: "search_polynomial_quadratic",
        func: search_polynomial_quadratic,
    },
    SearchCandidate {
        key: "search_min3_branch",
        func: search_min3_branch,
    },
    SearchCandidate {
        key: "search_combat_resolve",
        func: search_combat_resolve,
    },
    SearchCandidate {
        key: "search_score_tracker",
        func: search_score_tracker,
    },
    SearchCandidate {
        key: "search_vending_change",
        func: search_vending_change,
    },
    SearchCandidate {
        key: "search_turn_order_rotate",
        func: search_turn_order_rotate,
    },
    SearchCandidate {
        key: "search_grid_bounds_check",
        func: search_grid_bounds_check,
    },
    SearchCandidate {
        key: "search_simulate_gravity",
        func: search_simulate_gravity,
    },
    SearchCandidate {
        key: "search_scalar_expr",
        func: search_scalar_expr,
    },
    SearchCandidate {
        key: "search_single_branch",
        func: search_single_branch,
    },
    SearchCandidate {
        key: "search_two_branch",
        func: search_two_branch,
    },
];

fn ranked_search_candidates(problem: &Problem) -> Vec<SearchCandidate> {
    let mut ranked: Vec<(usize, SearchCandidate)> =
        SEARCH_CANDIDATES.iter().copied().enumerate().collect();
    let recommended = search_family_router::recommend_detailed(problem);
    if recommended.is_empty() {
        return ranked.into_iter().map(|(_, candidate)| candidate).collect();
    }

    let positions: std::collections::HashMap<&str, usize> = recommended
        .iter()
        .enumerate()
        .map(|(idx, rec)| (rec.method.as_str(), idx))
        .collect();
    ranked.sort_by_key(|(default_idx, candidate)| {
        (
            positions.get(candidate.key).copied().unwrap_or(usize::MAX),
            *default_idx,
        )
    });
    ranked.into_iter().map(|(_, candidate)| candidate).collect()
}

#[cfg(test)]
pub(super) fn ranked_search_candidate_keys(problem: &Problem) -> Vec<&'static str> {
    ranked_search_candidates(problem)
        .into_iter()
        .map(|candidate| candidate.key)
        .collect()
}

pub(super) fn solve_by_search(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let ranked = ranked_search_candidates(problem);
    let mut tried = Vec::with_capacity(ranked.len());

    for candidate in ranked {
        tried.push(candidate.key);
        if let Some(result) = (candidate.func)(problem, fn_name) {
            search_family_router::record_attempt(problem, &tried, Some(candidate.key));
            return Some(result);
        }
    }

    search_family_router::record_attempt(problem, &tried, None);
    None
}

pub(super) fn solve_problem_search_only(problem: &Problem) -> SolveResult {
    let fn_name = problem.function_name();
    if let Some(result) = solve_by_search(problem, fn_name) {
        return result;
    }
    SolveResult {
        success: false,
        code: String::new(),
        method: family_name(problem),
        error: Some("search-only mode could not synthesize this problem".to_string()),
        metadata: DifferentiableMetadata::default(),
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let tmp = b;
        b = a % b;
        a = tmp;
    }
    a
}

fn fibonacci(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let next = a + b;
        a = b;
        b = next;
    }
    a
}

fn digit_sum(mut n: i64) -> i64 {
    n = n.abs();
    let mut total = 0;
    while n > 0 {
        total += n % 10;
        n /= 10;
    }
    total
}

fn reverse_digits(mut n: i64) -> i64 {
    n = n.abs();
    let mut acc = 0;
    while n > 0 {
        acc = (acc * 10) + (n % 10);
        n /= 10;
    }
    acc
}

fn digit_count(mut n: i64) -> i64 {
    n = n.abs();
    if n == 0 {
        return 1;
    }
    let mut acc = 0;
    while n > 0 {
        acc += 1;
        n /= 10;
    }
    acc
}

fn count_even_digits(mut n: i64) -> i64 {
    n = n.abs();
    if n == 0 {
        return 1;
    }
    let mut acc = 0;
    while n > 0 {
        if (n % 10) % 2 == 0 {
            acc += 1;
        }
        n /= 10;
    }
    acc
}

fn collatz_steps(mut n: i64) -> i64 {
    let mut steps = 0;
    while n > 1 {
        if n % 2 == 0 {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        steps += 1;
    }
    steps
}

fn is_prime(n: i64) -> i64 {
    if n < 2 {
        return 0;
    }
    if n == 2 {
        return 1;
    }
    if n % 2 == 0 {
        return 0;
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return 0;
        }
        i += 2;
    }
    1
}

fn count_words(s: &str) -> i64 {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return 0;
    }
    trimmed.split(' ').filter(|part| !part.is_empty()).count() as i64
}

fn euler_totient(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut result = n;
    let mut p = 2;
    let mut temp = n;
    while p * p <= temp {
        if temp % p == 0 {
            while temp % p == 0 {
                temp /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if temp > 1 {
        result -= result / temp;
    }
    result
}

fn triangular_check(n: i64) -> i64 {
    let mut k = 0;
    while k * (k + 1) / 2 <= n {
        if k * (k + 1) / 2 == n {
            return 1;
        }
        k += 1;
    }
    0
}

fn harmonic_sum(n: i64) -> i64 {
    let mut total = 0;
    let mut i = 1;
    while i <= n {
        total += 1000 / i;
        i += 1;
    }
    total
}

fn second_max(arr: &[i64]) -> i64 {
    let mut first = arr[0];
    let mut second = arr[0];
    for &item in arr {
        if item > first {
            second = first;
            first = item;
        } else if item > second {
            second = item;
        }
    }
    second
}

fn array_range(arr: &[i64]) -> i64 {
    let lo = *arr.iter().min().unwrap();
    let hi = *arr.iter().max().unwrap();
    hi - lo
}

fn sum_of_divisors(n: i64) -> i64 {
    (1..=n).filter(|d| n % d == 0).sum()
}

fn sum_odd_digits(mut n: i64) -> i64 {
    let mut acc = 0;
    while n > 0 {
        let d = n % 10;
        if d % 2 == 1 {
            acc += d;
        }
        n /= 10;
    }
    acc
}
