"""Test robustness features: held-out verification, counterexample generation,
richer loop bodies, and automatic stress testing."""

from egdc.mog_execute import execute_mog


def test_robust_search_verifies_on_held_out():
    from egdc.mog_program_search import robust_search_program

    train = [((2.0, 3.0), 5.0), ((10.0, -4.0), 6.0), ((7.0, 8.0), 15.0)]
    holdout = [((-3.0, -2.0), -5.0), ((0.0, 0.0), 0.0), ((100.0, 1.0), 101.0)]
    result = robust_search_program(
        arg_names=["a", "b"],
        train_examples=train,
        holdout_examples=holdout,
        function_name="add_two",
        seed=42,
    )
    assert result.success
    assert result.holdout_loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(add_two(50, 50)); return 0; }")
    assert run.success
    assert run.stdout.strip() == "100"


def test_robust_search_discovers_gcd():
    from egdc.mog_program_search import robust_search_program

    def gcd(a, b):
        while b: a, b = b, a % b
        return a

    train = [((12.0, 18.0), 6.0), ((21.0, 14.0), 7.0), ((48.0, 18.0), 6.0), ((9.0, 28.0), 1.0)]
    holdout = [((84.0, 30.0), 6.0), ((100.0, 75.0), 25.0)]
    result = robust_search_program(
        arg_names=["a", "b"],
        train_examples=train,
        holdout_examples=holdout,
        function_name="gcd",
        seed=42,
    )
    print(f"gcd: loss={result.loss}, holdout={result.holdout_loss}, code:\n{result.code}")
    assert result.success
    assert result.holdout_loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(gcd(84, 30)); return 0; }")
    assert run.success
    assert run.stdout.strip() == "6"


def test_robust_search_stress_tests_discovered_program():
    from egdc.mog_program_search import robust_search_program

    train = [((2.0, 3.0), 3.0), ((10.0, -4.0), 10.0), ((7.0, 7.0), 7.0), ((-3.0, -2.0), -2.0)]
    result = robust_search_program(
        arg_names=["a", "b"],
        train_examples=train,
        holdout_examples=[],
        function_name="max2",
        auto_stress_test=True,
        seed=42,
    )
    assert result.success
    assert result.stress_test_passed
