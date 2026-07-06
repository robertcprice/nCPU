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

/// `is_power_of_two(n)` predicate (n>0 && n&(n-1)==0). The emitter
/// `code_bit_power_of_two_check` existed but was DEAD — no search wired it, so real
/// power-of-two predicates from OSS timed out in the general search. Mirrors
/// `search_is_even_formula` (bool validates as int 0/1). Exact, verified.
pub(super) fn search_power_of_two(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }
    // Bool predicate: validate against expected_bool (expected_int returns 0 for
    // ALL bools, so the int path can't check it) and emit a `-> bool` program.
    let ok = problem.examples.iter().all(|ex| {
        ex.inputs.len() == 1
            && match (int_value(&ex.inputs[0]), ex.expected_bool()) {
                (Some(n), Some(b)) => (n > 0 && (n & (n - 1)) == 0) == b,
                _ => false,
            }
    });
    if !ok {
        return None;
    }
    let code = format!(
        "fn {fn_name}(n: i64) -> bool {{\n    if n <= 0 {{\n        return false;\n    }}\n    if (n & (n - 1)) == 0 {{\n        return true;\n    }}\n    return false;\n}}\n"
    );
    verified_result(problem, code, "search_power_of_two")
}

/// General unary number-theory PREDICATE recognizer (int -> bool): tries a table of
/// exact predicates (perfect-square / pronic / integer-palindrome / automorphic) and
/// emits the first whose reference reproduces every example. Each Mog body is written
/// to match its Rust reference exactly. Covers the OSS bool-predicate timeout cluster
/// (validated on TheAlgorithms) that the general search cannot fit in time. Bool
/// validated via `expected_bool`; exact-by-construction.
pub(super) fn search_number_predicate(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if parse_param_types(problem.signature) != [ParamType::I64] {
        return None;
    }
    fn is_perfect_square(n: i64) -> bool {
        if n < 0 { return false; }
        let mut i = 0i64;
        while i * i < n { i += 1; }
        i * i == n
    }
    fn is_pronic(n: i64) -> bool {
        if n < 0 { return false; }
        let mut i = 0i64;
        while i * (i + 1) < n { i += 1; }
        i * (i + 1) == n
    }
    fn is_int_palindrome(n: i64) -> bool {
        if n < 0 { return false; }
        let (mut x, mut r) = (n, 0i64);
        while x > 0 { r = r * 10 + x % 10; x /= 10; }
        r == n
    }
    fn is_automorphic(n: i64) -> bool {
        if n < 0 { return false; }
        let (mut p, mut x) = (1i64, n);
        while x > 0 { p *= 10; x /= 10; }
        (n * n) % p == n
    }
    // (name, reference, Mog body)
    let table: &[(&str, fn(i64) -> bool, String)] = &[
        ("perfect_square", is_perfect_square, format!(
            "fn {fn_name}(n: i64) -> bool {{\n    if n < 0 {{ return false; }}\n    i: i64 = 0;\n    while (i * i) < n {{\n        i = i + 1;\n    }}\n    if (i * i) == n {{ return true; }}\n    return false;\n}}\n")),
        ("pronic", is_pronic, format!(
            "fn {fn_name}(n: i64) -> bool {{\n    if n < 0 {{ return false; }}\n    i: i64 = 0;\n    while (i * (i + 1)) < n {{\n        i = i + 1;\n    }}\n    if (i * (i + 1)) == n {{ return true; }}\n    return false;\n}}\n")),
        ("int_palindrome", is_int_palindrome, format!(
            "fn {fn_name}(n: i64) -> bool {{\n    if n < 0 {{ return false; }}\n    x: i64 = n;\n    r: i64 = 0;\n    while x > 0 {{\n        r = (r * 10) + (x % 10);\n        x = x / 10;\n    }}\n    if r == n {{ return true; }}\n    return false;\n}}\n")),
        ("automorphic", is_automorphic, format!(
            "fn {fn_name}(n: i64) -> bool {{\n    if n < 0 {{ return false; }}\n    p: i64 = 1;\n    x: i64 = n;\n    while x > 0 {{\n        p = p * 10;\n        x = x / 10;\n    }}\n    if ((n * n) % p) == n {{ return true; }}\n    return false;\n}}\n")),
    ];
    for (name, pred, code) in table {
        let ok = problem.examples.iter().all(|ex| {
            ex.inputs.len() == 1
                && match (int_value(&ex.inputs[0]), ex.expected_bool()) {
                    (Some(n), Some(b)) => pred(n) == b,
                    _ => false,
                }
        });
        if ok {
            let method = format!("search_number_predicate:{name}");
            return verified_result(problem, code.clone(), Box::leak(method.into_boxed_str()));
        }
    }
    None
}

/// Unary INT -> LIST recognizer: divisor lists and prime factorizations. Tries a
/// battery of exact list-generating references (all-divisors, prime factors with
/// multiplicity, distinct prime factors) and emits the first whose reference
/// reproduces EVERY example's list output. Covers the OSS factor-list timeout
/// cluster (factors_of_a_number / prime_factors / unique_prime_factors on
/// TheAlgorithms) that the generic list search cannot fit in time. Edge-case
/// variants that special-case n=0/1 (keon `factors` 1->[1], `prime_factorization`
/// 0->[0]) simply won't validate against the clean references -> None -> no false
/// solve. List validated via `array_value`; exact-by-construction; strict
/// re-verify in `verified_result` (so a Mog body that can't reproduce the examples
/// is rejected, never accepted).
pub(super) fn search_factor_list(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    if parse_param_types(problem.signature) != [ParamType::I64] {
        return None;
    }
    const BOUND: i64 = 2_000_000;
    // All positive divisors of n, ascending (incl. 1 and n).
    fn ref_all_divisors(n: i64) -> Option<Vec<i64>> {
        if !(1..=BOUND).contains(&n) {
            return None;
        }
        let mut v = Vec::new();
        let mut i = 1i64;
        while i <= n {
            if n % i == 0 {
                v.push(i);
            }
            i += 1;
        }
        Some(v)
    }
    // Prime factors WITH multiplicity, ascending (2560 -> [2;9, 5]).
    fn ref_prime_factors(n: i64) -> Option<Vec<i64>> {
        if !(0..=BOUND).contains(&n) {
            return None;
        }
        let mut v = Vec::new();
        let (mut m, mut d) = (n, 2i64);
        while d * d <= m {
            while m % d == 0 {
                v.push(d);
                m /= d;
            }
            d += 1;
        }
        if m > 1 {
            v.push(m);
        }
        Some(v)
    }
    // DISTINCT prime factors, ascending (2560 -> [2, 5]).
    fn ref_unique_prime_factors(n: i64) -> Option<Vec<i64>> {
        if !(0..=BOUND).contains(&n) {
            return None;
        }
        let mut v = Vec::new();
        let (mut m, mut d) = (n, 2i64);
        while d * d <= m {
            if m % d == 0 {
                v.push(d);
                while m % d == 0 {
                    m /= d;
                }
            }
            d += 1;
        }
        if m > 1 {
            v.push(m);
        }
        Some(v)
    }
    let all_div = format!(
        "fn {fn_name}(n: i64) -> [i64] {{\n    result: [i64] = [];\n    i: i64 = 1;\n    while i <= n {{\n        if (n % i) == 0 {{\n            result.push(i);\n        }}\n        i = i + 1;\n    }}\n    return result;\n}}\n"
    );
    let prime_mult = format!(
        "fn {fn_name}(n: i64) -> [i64] {{\n    result: [i64] = [];\n    m: i64 = n;\n    d: i64 = 2;\n    while (d * d) <= m {{\n        while (m % d) == 0 {{\n            result.push(d);\n            m = m / d;\n        }}\n        d = d + 1;\n    }}\n    if m > 1 {{\n        result.push(m);\n    }}\n    return result;\n}}\n"
    );
    let prime_uniq = format!(
        "fn {fn_name}(n: i64) -> [i64] {{\n    result: [i64] = [];\n    m: i64 = n;\n    d: i64 = 2;\n    while (d * d) <= m {{\n        if (m % d) == 0 {{\n            result.push(d);\n            while (m % d) == 0 {{\n                m = m / d;\n            }}\n        }}\n        d = d + 1;\n    }}\n    if m > 1 {{\n        result.push(m);\n    }}\n    return result;\n}}\n"
    );
    let table: [(&str, fn(i64) -> Option<Vec<i64>>, String); 3] = [
        ("all_divisors", ref_all_divisors, all_div),
        ("prime_factors", ref_prime_factors, prime_mult),
        ("unique_prime_factors", ref_unique_prime_factors, prime_uniq),
    ];
    for (name, reference, code) in &table {
        let ok = problem.examples.iter().all(|ex| {
            ex.inputs.len() == 1
                && match (int_value(&ex.inputs[0]), array_value(&ex.expected)) {
                    (Some(n), Some(exp)) => reference(n) == Some(exp),
                    _ => false,
                }
        });
        if ok {
            let method = format!("search_factor_list:{name}");
            return verified_result(problem, code.clone(), Box::leak(method.into_boxed_str()));
        }
    }
    None
}

/// INT -> base-N STRING recognizer. Two exact references keyed on arity:
///   * 1-arg, signed: binary with a `0b` prefix (decimal_to_binary_iterative /
///     _recursive — the recursive one carries a `-0b…` sign).
///   * 2-arg (n, base): arbitrary-base uppercase digit string, no prefix, 0 -> "0"
///     (int_to_base / decimal_to_any).
/// The emitted Mog builds the string with the string-INDEX primitive
/// (`"0123…Z"[m % base]` returns a 1-char string) prepended in a loop — there is
/// no int->char builtin, so indexing a digit lexicon is the mechanism. Covers the
/// OSS base-conversion timeout cluster (TheAlgorithms). String validated via
/// `str_value`; exact-by-construction; strict re-verify in `verified_result`.
pub(super) fn search_base_string(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let pts = parse_param_types(problem.signature);
    // ---- 1-arg: binary with a 0b prefix (signed) ----
    if pts == [ParamType::I64] {
        fn ref_bin(n: i64) -> String {
            if n == 0 {
                return "0b0".to_string();
            }
            let neg = n < 0;
            let mut m = (n as i128).unsigned_abs();
            let mut s = String::new();
            while m > 0 {
                s.insert(0, char::from(b'0' + (m % 2) as u8));
                m /= 2;
            }
            format!("{}0b{}", if neg { "-" } else { "" }, s)
        }
        let ok = problem.examples.iter().all(|ex| {
            ex.inputs.len() == 1
                && match (int_value(&ex.inputs[0]), str_value(&ex.expected)) {
                    (Some(n), Some(s)) => ref_bin(n) == s,
                    _ => false,
                }
        });
        if ok {
            let code = format!(
                "fn {fn_name}(n: i64) -> string {{\n    if n == 0 {{\n        return \"0b0\";\n    }}\n    digits: string = \"0123456789ABCDEF\";\n    m: i64 = n;\n    sign: string = \"\";\n    if n < 0 {{\n        sign = \"-\";\n        m = 0 - n;\n    }}\n    result: string = \"\";\n    while m > 0 {{\n        result = digits[m % 2] + result;\n        m = m / 2;\n    }}\n    return (sign + \"0b\") + result;\n}}\n"
            );
            return verified_result(problem, code, "search_base_string:binary_0b");
        }
        return None;
    }
    // ---- 2-arg (n, base): uppercase digit string, no prefix, 0 -> "0" ----
    if pts == [ParamType::I64, ParamType::I64] {
        fn ref_base(n: i64, b: i64) -> Option<String> {
            if !(2..=36).contains(&b) || n < 0 {
                return None;
            }
            if n == 0 {
                return Some("0".to_string());
            }
            let digits = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            let (mut m, mut s) = (n, String::new());
            while m > 0 {
                s.insert(0, digits[(m % b) as usize] as char);
                m /= b;
            }
            Some(s)
        }
        let ok = problem.examples.iter().all(|ex| {
            ex.inputs.len() == 2
                && match (
                    int_value(&ex.inputs[0]),
                    int_value(&ex.inputs[1]),
                    str_value(&ex.expected),
                ) {
                    (Some(n), Some(b), Some(s)) => ref_base(n, b).as_deref() == Some(s),
                    _ => false,
                }
        });
        if ok {
            // The `if b < 2` guard keeps the emitted program TOTAL: a base <= 1 would
            // divide-by-zero (b==0) or loop forever (b==1, `m / 1` never shrinks), which
            // the strict-verify robustness probe (perturbed inputs) would trip. Real
            // examples always carry b >= 2, so this guard never changes a real output —
            // it only makes the program execute cleanly on out-of-distribution probes.
            let code = format!(
                "fn {fn_name}(n: i64, b: i64) -> string {{\n    if b < 2 {{\n        return \"0\";\n    }}\n    if n == 0 {{\n        return \"0\";\n    }}\n    digits: string = \"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\";\n    m: i64 = n;\n    result: string = \"\";\n    while m > 0 {{\n        result = digits[m % b] + result;\n        m = m / b;\n    }}\n    return result;\n}}\n"
            );
            return verified_result(problem, code, "search_base_string:to_base");
        }
        return None;
    }
    None
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
