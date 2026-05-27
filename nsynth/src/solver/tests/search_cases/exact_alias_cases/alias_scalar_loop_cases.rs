use super::*;

#[test]
fn search_solves_aliased_sum_to_n_without_family_name() {
    let problem = aliased_problem(
        "sum_to_n",
        "mystery_series_v0",
        "fn mystery_series(value: i64) -> i64",
        "scalar_search",
        "Return the total from 1 through value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_unary_range_loop");
    assert!(result.code.contains("while i <= n"));
    assert!(result.code.contains("acc = acc + i;"));
    assert!(result.code.contains("fn mystery_series"));
}

#[test]
fn search_solves_aliased_sum_squares_without_family_name() {
    let problem = aliased_problem(
        "sum_squares",
        "mystery_square_series_v0",
        "fn mystery_square_series(value: i64) -> i64",
        "scalar_search",
        "Return the sum of the squares from 1 through value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_unary_range_loop");
    assert!(result.code.contains("acc = acc + (i * i);"));
    assert!(result.code.contains("fn mystery_square_series"));
}

#[test]
fn search_solves_aliased_product_without_family_name() {
    let problem = aliased_problem(
        "product_1_to_n",
        "mystery_product_v0",
        "fn mystery_product(value: i64) -> i64",
        "scalar_search",
        "Return the product of all integers from 1 through value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_unary_range_loop");
    assert!(result.code.contains("acc = acc * i;"));
    assert!(result.code.contains("fn mystery_product"));
}

#[test]
fn search_solves_aliased_power_without_family_name() {
    let problem = aliased_problem(
        "power",
        "mystery_power_v0",
        "fn mystery_power(base: i64, exp: i64) -> i64",
        "scalar_search",
        "Raise base to the non-negative exponent exp.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_power_loop");
    assert!(result.code.contains("while i < b"));
    assert!(result.code.contains("acc = acc * a;"));
    assert!(result.code.contains("fn mystery_power"));
}

#[test]
fn search_solves_aliased_collatz_without_family_name() {
    let problem = aliased_problem(
        "collatz_steps",
        "mystery_collatz_v0",
        "fn mystery_collatz(value: i64) -> i64",
        "scalar_search",
        "Return how many Collatz steps are needed for value to reach one.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_collatz_loop");
    assert!(result.code.contains("while x > 1"));
    assert!(result.code.contains("x = 3 * x + 1;"));
    assert!(result.code.contains("fn mystery_collatz"));
}

#[test]
fn search_solves_aliased_is_prime_without_family_name() {
    let problem = aliased_problem(
        "is_prime",
        "mystery_prime_v0",
        "fn mystery_prime(value: i64) -> i64",
        "scalar_search",
        "Return 1 when value is prime and 0 otherwise.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_is_prime_loop");
    assert!(result.code.contains("while i * i <= n"));
    assert!(result.code.contains("return 1;"));
    assert!(result.code.contains("fn mystery_prime"));
}

#[test]
fn search_solves_aliased_digit_sum_without_family_name() {
    let problem = aliased_problem(
        "digit_sum",
        "mystery_digits_v0",
        "fn mystery_digits(value: i64) -> i64",
        "scalar_search",
        "Return the sum of the base-10 digits of value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_digit_sum_loop");
    assert!(result.code.contains("x % 10"));
    assert!(result.code.contains("x = x / 10;"));
    assert!(result.code.contains("fn mystery_digits"));
}

#[test]
fn search_solves_aliased_reverse_digits_without_family_name() {
    let problem = aliased_problem(
        "reverse_digits",
        "mystery_reverse_digits_v0",
        "fn mystery_reverse_digits(value: i64) -> i64",
        "scalar_search",
        "Reverse the base-10 digits of value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_reverse_digits_loop");
    assert!(result.code.contains("acc = (acc * 10) + (x % 10);"));
    assert!(result.code.contains("fn mystery_reverse_digits"));
}

#[test]
fn search_solves_aliased_digit_count_without_family_name() {
    let problem = aliased_problem(
        "digit_count",
        "mystery_digit_count_v0",
        "fn mystery_digit_count(value: i64) -> i64",
        "scalar_search",
        "Count how many base-10 digits value contains.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_digit_count_loop");
    assert!(result.code.contains("acc = acc + 1;"));
    assert!(result.code.contains("fn mystery_digit_count"));
}

#[test]
fn search_solves_aliased_count_even_digits_without_family_name() {
    let problem = aliased_problem(
        "count_even_digits",
        "mystery_count_even_digits_v0",
        "fn mystery_count_even_digits(value: i64) -> i64",
        "scalar_search",
        "Count how many base-10 digits of value are even.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_count_even_digits_loop");
    assert!(result.code.contains("((x % 10) % 2) == 0"));
    assert!(result.code.contains("fn mystery_count_even_digits"));
}

#[test]
fn search_solves_aliased_gcd_without_family_name() {
    let problem = aliased_problem(
        "gcd_extended",
        "mystery_euclid_v0",
        "fn mystery_euclid(a: i64, b: i64) -> i64",
        "scalar_search",
        "Return the Euclidean greatest common divisor of a and b.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_gcd_loop");
    assert!(result.code.contains("while y != 0"));
    assert!(result.code.contains("fn mystery_euclid"));
}

#[test]
fn search_solves_aliased_count_divisors_without_family_name() {
    let problem = aliased_problem(
        "count_divisors",
        "mystery_divisors_v0",
        "fn mystery_divisors(value: i64) -> i64",
        "scalar_search",
        "Count the number of positive divisors of value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_count_divisors_loop");
    assert!(result.code.contains("while i <= n"));
    assert!(result.code.contains("if n % i == 0"));
    assert!(result.code.contains("fn mystery_divisors"));
}

#[test]
fn search_solves_aliased_fib_iter_without_family_name() {
    let problem = aliased_problem(
        "fib_iter",
        "mystery_fib_v0",
        "fn mystery_fib(value: i64) -> i64",
        "scalar_search",
        "Return the iterative Fibonacci number for value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_fib_iter_loop");
    assert!(result.code.contains("tmp: i64 = a + b;"));
    assert!(result.code.contains("while i <= n"));
    assert!(result.code.contains("fn mystery_fib"));
}

#[test]
fn search_solves_aliased_harmonic_sum_without_family_name() {
    let problem = aliased_problem(
        "harmonic_sum",
        "mystery_harmonic_v0",
        "fn mystery_harmonic(value: i64) -> i64",
        "scalar_search",
        "Return the scaled harmonic sum 1000/1 + ... + 1000/value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_harmonic_sum_loop");
    assert!(result.code.contains("total = total + 1000 / i;"));
    assert!(result.code.contains("fn mystery_harmonic"));
}

#[test]
fn search_solves_aliased_triangular_check_without_family_name() {
    let problem = aliased_problem(
        "triangular_check",
        "mystery_triangular_v0",
        "fn mystery_triangular(value: i64) -> i64",
        "scalar_search",
        "Return 1 when value is triangular and 0 otherwise.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_triangular_check_loop");
    assert!(result.code.contains("k * (k + 1) / 2"));
    assert!(result.code.contains("fn mystery_triangular"));
}

#[test]
fn search_solves_aliased_euler_totient_without_family_name() {
    let problem = aliased_problem(
        "euler_totient",
        "mystery_totient_v0",
        "fn mystery_totient(value: i64) -> i64",
        "scalar_search",
        "Compute Euler's totient function of value.",
    );
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert_eq!(result.method, "search_euler_totient_loop");
    assert!(result.code.contains("while p * p <= temp"));
    assert!(result.code.contains("result = result - result / p;"));
    assert!(result.code.contains("fn mystery_totient"));
}
