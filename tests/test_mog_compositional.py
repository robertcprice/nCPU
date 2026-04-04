"""Test compositional synthesis: building complex programs from discovered sub-programs."""

from egdc.mog_execute import execute_mog


def test_compose_lcm_from_discovered_gcd():
    """LCM should be discovered by composing the already-known GCD sub-program."""
    from egdc.mog_compositional import CompositionalSolver

    solver = CompositionalSolver()
    # Register gcd as a known sub-program
    solver.register_subprogram(
        "gcd", ["a", "b"],
        "fn gcd(a: i64, b: i64) -> i64 { x: i64 = a; y: i64 = b; while y != 0 { tmp := y; y = x % y; x = tmp; } return x; }"
    )

    examples = [((3.0, 4.0), 12.0), ((6.0, 8.0), 24.0), ((5.0, 10.0), 10.0), ((7.0, 9.0), 63.0)]
    result = solver.solve("lcm", ["a", "b"], examples)
    assert result.success, f"loss={result.loss}"
    assert result.loss < 1e-6
    # Should contain a call to gcd
    assert "gcd" in result.code
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(lcm(12, 18)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "36"


def test_compose_is_prime_and_count():
    """Build count_primes by composing is_prime."""
    from egdc.mog_compositional import CompositionalSolver

    solver = CompositionalSolver()
    # Register is_prime
    solver.register_subprogram(
        "is_prime", ["n"],
        ("fn is_prime(n: i64) -> i64 {\n"
         "    if n < 2 { return 0; }\n"
         "    d: i64 = 2;\n"
         "    while d * d <= n {\n"
         "        if (n % d) == 0 { return 0; }\n"
         "        d = d + 1;\n"
         "    }\n"
         "    return 1;\n"
         "}\n")
    )

    examples = [
        ((10.0,), 4.0),   # primes below 10: 2,3,5,7
        ((2.0,), 0.0),    # primes below 2: none
        ((20.0,), 8.0),   # primes below 20: 2,3,5,7,11,13,17,19
    ]
    result = solver.solve("count_primes_below", ["n"], examples)
    assert result.success, f"loss={result.loss}"
    assert "is_prime" in result.code


def test_compose_sum_of_squares():
    """sum_of_squares = sum(i*i for i in range(1, n+1))."""
    from egdc.mog_compositional import CompositionalSolver

    solver = CompositionalSolver()
    examples = [
        ((1.0,), 1.0),
        ((3.0,), 14.0),   # 1+4+9
        ((5.0,), 55.0),   # 1+4+9+16+25
    ]
    result = solver.solve("sum_of_squares", ["n"], examples)
    assert result.success
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(sum_of_squares(4)); return 0; }")
    assert run.success
    assert run.stdout.strip() == "30"  # 1+4+9+16


def test_multi_step_pipeline():
    """Test a multi-step program: filter negatives, then sum."""
    from egdc.mog_compositional import CompositionalSolver

    solver = CompositionalSolver()
    # Register abs
    solver.register_subprogram(
        "my_abs", ["x"],
        "fn my_abs(x: i64) -> i64 { if x < 0 { return 0 - x; } return x; }"
    )

    examples = [
        ((5.0,), 15.0),   # abs(1)+abs(2)+abs(3)+abs(4)+abs(5)
        ((3.0,), 6.0),    # abs(1)+abs(2)+abs(3)
    ]
    result = solver.solve("sum_abs_to_n", ["n"], examples)
    assert result.success
