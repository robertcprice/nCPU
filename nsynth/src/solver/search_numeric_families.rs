use super::search_codegen::*;
use super::search_runtime::*;
use super::*;

pub(super) fn search_power_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |base, exp| {
        // Integer power, overflow-safe. An unchecked `base.pow(exp as u32)`
        // panics in debug builds for any example whose base^exp exceeds
        // i64 (and `exp as u32` silently truncates a huge exponent). A
        // power whose true result overflows i64 can never be the answer
        // to one of these benchmark problems (their outputs fit i64), so
        // map overflow / out-of-range exponent to a sentinel that simply
        // fails the match rather than crashing the whole pipeline.
        if exp < 0 {
            0
        } else if let Ok(e) = u32::try_from(exp) {
            base.checked_pow(e).unwrap_or(i64::MIN)
        } else {
            i64::MIN
        }
    }) {
        return None;
    }
    verified_result(
        problem,
        code_power_loop_search(fn_name),
        "search_power_loop",
    )
}

/// Iterative modular exponentiation `a^b mod m` (`modpow(a,b,m)`): the 3-arg gap
/// found validating against keon/algorithms `power(a,b,mod)`. The engine had
/// `search_power_loop` (a^b) but no MODULAR power, so real modpow only "solved" as
/// an overfit 2-branch. This is the dedicated verified recognizer (b multiplications,
/// mod each step — matches the emitted Mog exactly). i128 intermediate in the
/// reference avoids a validation-time overflow panic; out-of-domain (b<0 or m<=0)
/// maps to a sentinel that simply fails the match. Exact-by-construction: emitted
/// only when it reproduces every example, then `verified_result` re-verifies.
pub(super) fn search_modpow_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_ternary_int(problem, |a, b, m| {
        if b < 0 || m <= 0 {
            return i64::MIN;
        }
        let mut acc: i64 = 1;
        let mut i: i64 = 0;
        while i < b {
            acc = ((acc as i128 * a as i128) % m as i128) as i64;
            i += 1;
        }
        acc
    }) {
        return None;
    }
    verified_result(
        problem,
        code_modpow_loop_search(fn_name),
        "search_modpow_loop",
    )
}

pub(super) fn search_collatz_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, collatz_steps) {
        return None;
    }
    verified_result(problem, code_collatz_steps(fn_name), "search_collatz_loop")
}

pub(super) fn search_is_prime_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, is_prime) {
        return None;
    }
    verified_result(problem, code_is_prime(fn_name), "search_is_prime_loop")
}

pub(super) fn search_digit_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_fib_iter_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_count_divisors_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_harmonic_sum_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_triangular_check_loop(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
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

pub(super) fn search_euler_totient_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_lcm_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    // `gcd(0, 0) == 0`, so guard the division — an example with both inputs zero
    // would otherwise panic (divide by zero) and crash the whole binary. lcm is
    // conventionally 0 there. Use i128 for the product so a large pair does not
    // overflow before the divide.
    if !validate_binary_int(problem, |a, b| {
        let g = gcd(a, b);
        if g == 0 {
            0
        } else {
            ((a as i128 * b as i128) / g as i128) as i64
        }
    }) {
        return None;
    }
    verified_result(problem, code_lcm(fn_name), "search_lcm_formula")
}

pub(super) fn search_min3_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_ternary_int(problem, |a, b, c| a.min(b).min(c)) {
        return None;
    }
    verified_result(problem, code_min3(fn_name), "search_min3_branch")
}

pub(super) fn search_combat_resolve(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_score_tracker(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_vending_change(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_turn_order_rotate(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_grid_bounds_check(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_simulate_gravity(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_gcd_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, gcd) {
        return None;
    }
    verified_result(problem, code_gcd(fn_name), "search_gcd_loop")
}

pub(super) fn search_abs_diff_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |a, b| (a - b).abs()) {
        return None;
    }
    verified_result(problem, code_abs_diff(fn_name), "search_abs_diff_formula")
}

pub(super) fn search_max2_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64, ParamType::I64] {
        return None;
    }
    if !validate_binary_int(problem, |a, b| a.max(b)) {
        return None;
    }
    verified_result(problem, code_max2(fn_name), "search_max2_formula")
}

pub(super) fn search_safe_div_or_neg1_branch(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
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

pub(super) fn search_clamp_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |x| x.clamp(0, 100)) {
        return None;
    }
    verified_result(problem, code_clamp(fn_name), "search_clamp_formula")
}

pub(super) fn search_sign_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_positive_or_default_branch(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
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

pub(super) fn search_is_even_formula(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    if !validate_unary_int(problem, |x| if x % 2 == 0 { 1 } else { 0 }) {
        return None;
    }
    verified_result(problem, code_is_even(fn_name), "search_is_even_formula")
}

pub(super) fn search_sum_of_divisors_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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

pub(super) fn search_sum_odd_digits_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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
