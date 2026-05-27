use super::*;

#[test]
fn search_only_generalizes_on_loop_and_formula_holdout_cases() {
    assert_search_generalizes(
        "sum_to_n_v0",
        vec![(vec![Value::Int(7)], 28), (vec![Value::Int(-3)], 0)],
    );
    assert_search_generalizes(
        "lcm_v0",
        vec![
            (vec![Value::Int(8), Value::Int(12)], 24),
            (vec![Value::Int(9), Value::Int(6)], 18),
        ],
    );
    assert_search_generalizes(
        "factorial_v0",
        vec![(vec![Value::Int(3)], 6), (vec![Value::Int(8)], 40320)],
    );
    assert_search_generalizes(
        "fibonacci_v0",
        vec![(vec![Value::Int(8)], 21), (vec![Value::Int(11)], 89)],
    );
    assert_search_generalizes(
        "digit_sum_v0",
        vec![(vec![Value::Int(1002)], 3), (vec![Value::Int(999)], 27)],
    );
    assert_search_generalizes(
        "reverse_digits_v0",
        vec![(vec![Value::Int(81)], 18), (vec![Value::Int(12030)], 3021)],
    );
    assert_search_generalizes(
        "digit_count_v0",
        vec![(vec![Value::Int(81)], 2), (vec![Value::Int(12030)], 5)],
    );
    assert_search_generalizes(
        "count_even_digits_v0",
        vec![
            (vec![Value::Int(81)], 1),
            (vec![Value::Int(12030)], 3),
            (vec![Value::Int(24680)], 5),
        ],
    );
    assert_search_generalizes(
        "power_v0",
        vec![
            (vec![Value::Int(4), Value::Int(3)], 64),
            (vec![Value::Int(2), Value::Int(5)], 32),
        ],
    );
    assert_search_generalizes(
        "polynomial_v0",
        vec![(vec![Value::Int(3)], 28), (vec![Value::Int(-2)], 3)],
    );
    assert_search_generalizes(
        "collatz_steps_v0",
        vec![(vec![Value::Int(6)], 8), (vec![Value::Int(7)], 16)],
    );
    assert_search_generalizes(
        "min3_v0",
        vec![
            (vec![Value::Int(5), Value::Int(1), Value::Int(9)], 1),
            (vec![Value::Int(-2), Value::Int(-8), Value::Int(-3)], -8),
        ],
    );
    assert_search_generalizes(
        "is_prime_v0",
        vec![(vec![Value::Int(17)], 1), (vec![Value::Int(21)], 0)],
    );
    assert_search_generalizes(
        "nth_triangle_v0",
        vec![(vec![Value::Int(7)], 28), (vec![Value::Int(8)], 36)],
    );
    assert_search_generalizes(
        "fib_iter_v0",
        vec![(vec![Value::Int(8)], 21), (vec![Value::Int(12)], 144)],
    );
    assert_search_generalizes(
        "euler_totient_v0",
        vec![(vec![Value::Int(10)], 4), (vec![Value::Int(13)], 12)],
    );
    assert_search_generalizes(
        "sum_squares_v0",
        vec![(vec![Value::Int(4)], 30), (vec![Value::Int(6)], 91)],
    );
    assert_search_generalizes(
        "product_1_to_n_v0",
        vec![(vec![Value::Int(5)], 120), (vec![Value::Int(7)], 5040)],
    );
    assert_search_generalizes(
        "count_divisors_v0",
        vec![(vec![Value::Int(16)], 5), (vec![Value::Int(18)], 6)],
    );
    assert_search_generalizes(
        "triangular_check_v0",
        vec![(vec![Value::Int(6)], 1), (vec![Value::Int(8)], 0)],
    );
    assert_search_generalizes(
        "harmonic_sum_v0",
        vec![(vec![Value::Int(3)], 1833), (vec![Value::Int(6)], 2449)],
    );
}
