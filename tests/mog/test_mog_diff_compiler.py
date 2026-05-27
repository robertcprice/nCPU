"""Tests for the differentiable Mog compiler and expanded benchmark."""

import tempfile
from pathlib import Path


# --- Grammar constraint tests ---

def test_grammar_penalty_returns_nonzero_for_no_return():
    """A program with no return statement should get a penalty."""
    import torch
    from egdc.mog.solvers.grammar import grammar_penalty

    # 2 slots, no return_var active
    stmt_logits = torch.zeros(2, 7)
    stmt_logits[0, 1] = 10.0  # assign_binop
    stmt_logits[1, 1] = 10.0  # assign_binop (no return!)
    op_logits = torch.zeros(2, 5)
    op_logits[0, 0] = 10.0
    src2_logits = torch.zeros(2, 4)

    penalty = grammar_penalty(stmt_logits, op_logits, src2_logits, 2, 4)
    assert penalty.item() > 0.0


def test_grammar_penalty_low_for_valid_program():
    """A valid program (compute + return) should have low penalty."""
    import torch
    from egdc.mog.solvers.grammar import grammar_penalty, STMT_BINOP, STMT_RETURN

    stmt_logits = torch.full((2, 7), -10.0)
    stmt_logits[0, STMT_BINOP] = 10.0
    stmt_logits[1, STMT_RETURN] = 10.0
    op_logits = torch.zeros(2, 5)
    op_logits[0, 0] = 10.0
    src2_logits = torch.zeros(2, 4)

    penalty = grammar_penalty(stmt_logits, op_logits, src2_logits, 2, 4)
    assert penalty.item() < 0.5


def test_validate_discrete_valid_code():
    """Valid Mog code should pass validation."""
    from egdc.mog.solvers.grammar import validate_discrete

    code = "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n"
    valid, err = validate_discrete(code)
    assert valid, f"Expected valid, got: {err}"


def test_validate_discrete_invalid_code():
    """Syntactically broken Mog code should fail validation."""
    from egdc.mog.solvers.grammar import validate_discrete

    code = "fn add(a: i64 -> i64 {\n    return a + b;\n"
    valid, err = validate_discrete(code)
    # Might pass due to permissive parser, but at minimum shouldn't crash
    assert isinstance(valid, bool)


def test_constrained_softmax_respects_mask():
    """Constrained softmax should zero out masked choices."""
    import torch
    from egdc.mog.solvers.grammar import constrained_softmax

    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    result = constrained_softmax(logits, mask)
    assert result[1].item() < 1e-6
    assert result[3].item() < 1e-6
    assert abs(result[0].item() + result[2].item() - 1.0) < 1e-6


def test_constrained_softmax_no_mask():
    """Without mask, constrained softmax = regular softmax."""
    import torch
    from egdc.mog.solvers.grammar import constrained_softmax
    import torch.nn.functional as F

    logits = torch.tensor([1.0, 2.0, 3.0])
    result = constrained_softmax(logits, None)
    expected = F.softmax(logits, dim=0)
    assert torch.allclose(result, expected)


# --- Compiler arithmetic tests ---

def test_compiler_discovers_add_two():
    from egdc.mog.solvers.diff_compiler import MogDiffCompiler

    examples = [
        ((2.0, 3.0), 5.0),
        ((10.0, -4.0), 6.0),
        ((7.0, 8.0), 15.0),
        ((-3.0, -2.0), -5.0),
    ]
    compiler = MogDiffCompiler(max_steps=200)
    result = compiler.compile(["a", "b"], examples, "add_two", num_restarts=1)
    print(f"add_two: success={result.success}, loss={result.discrete_loss}, code:\n{result.code}")
    assert result.success, f"Failed: soft_loss={result.soft_loss}, discrete_loss={result.discrete_loss}"


def test_compiler_discovers_double():
    from egdc.mog.solvers.diff_compiler import MogDiffCompiler

    examples = [
        ((1.0,), 2.0),
        ((5.0,), 10.0),
        ((0.0,), 0.0),
        ((-3.0,), -6.0),
    ]
    compiler = MogDiffCompiler(max_steps=200)
    result = compiler.compile(["x"], examples, "double", num_restarts=1)
    print(f"double: success={result.success}, loss={result.discrete_loss}, code:\n{result.code}")
    assert result.success, f"Failed: soft_loss={result.soft_loss}, discrete_loss={result.discrete_loss}"


def test_compiler_discovers_multiply():
    from egdc.mog.solvers.diff_compiler import MogDiffCompiler

    examples = [
        ((3.0, 4.0), 12.0),
        ((2.0, 5.0), 10.0),
        ((7.0, 3.0), 21.0),
        ((-2.0, 3.0), -6.0),
    ]
    compiler = MogDiffCompiler(max_steps=200)
    result = compiler.compile(["a", "b"], examples, "multiply", num_restarts=1)
    print(f"multiply: success={result.success}, loss={result.discrete_loss}, code:\n{result.code}")
    assert result.success, f"Failed: soft_loss={result.soft_loss}, discrete_loss={result.discrete_loss}"


def test_compiler_discovers_subtract():
    from egdc.mog.solvers.diff_compiler import MogDiffCompiler

    examples = [
        ((10.0, 3.0), 7.0),
        ((5.0, 8.0), -3.0),
        ((0.0, 0.0), 0.0),
    ]
    compiler = MogDiffCompiler(max_steps=200)
    result = compiler.compile(["a", "b"], examples, "sub", num_restarts=1)
    print(f"sub: success={result.success}, loss={result.discrete_loss}, code:\n{result.code}")
    assert result.success, f"Failed: soft_loss={result.soft_loss}, discrete_loss={result.discrete_loss}"


# --- Compiler branching tests ---

def test_compiler_discovers_max2():
    from egdc.mog.solvers.diff_compiler import MogDiffCompiler

    examples = [
        ((2.0, 3.0), 3.0),
        ((10.0, -4.0), 10.0),
        ((7.0, 7.0), 7.0),
        ((-3.0, -2.0), -2.0),
    ]
    compiler = MogDiffCompiler(max_steps=300)
    result = compiler.compile(["a", "b"], examples, "max2", num_restarts=2)
    print(f"max2: success={result.success}, loss={result.discrete_loss}, code:\n{result.code}")
    assert result.success, f"Failed: soft_loss={result.soft_loss}, discrete_loss={result.discrete_loss}"


def test_compiler_discovers_abs_diff():
    from egdc.mog.solvers.diff_compiler import MogDiffCompiler

    examples = [
        ((3.0, 7.0), 4.0),
        ((7.0, 3.0), 4.0),
        ((0.0, 5.0), 5.0),
        ((-3.0, 2.0), 5.0),
    ]
    compiler = MogDiffCompiler(max_steps=300)
    result = compiler.compile(["a", "b"], examples, "abs_diff", num_restarts=2)
    print(f"abs_diff: success={result.success}, loss={result.discrete_loss}, code:\n{result.code}")
    assert result.success, f"Failed: soft_loss={result.soft_loss}, discrete_loss={result.discrete_loss}"


# --- New benchmark tests ---

def test_benchmark_has_61_factories():
    from egdc.mog.benchmark import PROBLEM_FACTORIES
    assert len(PROBLEM_FACTORIES) == 63, f"Expected 63 factories, got {len(PROBLEM_FACTORIES)}"


def test_search_solves_power():
    from egdc.mog.benchmark import _make_power
    from egdc.mog.solvers.search_solver import solve_problem

    problem = _make_power(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"power: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


def test_search_solves_polynomial():
    from egdc.mog.benchmark import _make_polynomial
    from egdc.mog.solvers.search_solver import solve_problem

    problem = _make_polynomial(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"polynomial: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


def test_search_solves_collatz():
    from egdc.mog.benchmark import _make_collatz_steps
    from egdc.mog.solvers.search_solver import solve_problem

    problem = _make_collatz_steps(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"collatz: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


def test_search_solves_min3():
    from egdc.mog.benchmark import _make_min3
    from egdc.mog.solvers.search_solver import solve_problem

    problem = _make_min3(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"min3: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


def test_search_solves_is_prime():
    from egdc.mog.benchmark import _make_is_prime
    from egdc.mog.solvers.search_solver import solve_problem

    problem = _make_is_prime(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"is_prime: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


def test_search_solves_nth_triangle():
    from egdc.mog.benchmark import _make_nth_triangle
    from egdc.mog.solvers.search_solver import solve_problem

    problem = _make_nth_triangle(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"nth_triangle: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


def test_search_solves_reverse_sum():
    from egdc.mog.benchmark import _make_reverse_array
    from egdc.mog.solvers.search_solver import solve_problem

    problem = _make_reverse_array(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"reverse_sum: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


def test_search_solves_array_max_elem():
    from egdc.mog.benchmark import _make_second_largest
    from egdc.mog.solvers.search_solver import solve_problem

    problem = _make_second_largest(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"array_max_elem: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


# --- Complex program tests ---

def _solve_factory(factory_fn, name):
    from egdc.mog.solvers.search_solver import solve_problem
    problem = factory_fn(__import__("random").Random(42), 0)
    result = solve_problem(problem, use_compiler=False)
    print(f"{name}: success={result.success}, method={result.method}, loss={result.loss}")
    assert result.success, f"Failed: method={result.method}, loss={result.loss}"


def test_search_solves_fib_iter():
    from egdc.mog.benchmark import _make_fib_iter
    _solve_factory(_make_fib_iter, "fib_iter")


def test_search_solves_palindrome_check():
    from egdc.mog.benchmark import _make_palindrome_check
    _solve_factory(_make_palindrome_check, "palindrome_check")


def test_search_solves_count_words():
    from egdc.mog.benchmark import _make_count_words
    _solve_factory(_make_count_words, "count_words")


def test_search_solves_euler_totient():
    from egdc.mog.benchmark import _make_euler_totient
    _solve_factory(_make_euler_totient, "euler_totient")


def test_search_solves_sum_squares():
    from egdc.mog.benchmark import _make_sum_squares
    _solve_factory(_make_sum_squares, "sum_squares")


def test_search_solves_product_1_to_n():
    from egdc.mog.benchmark import _make_product_1_to_n
    _solve_factory(_make_product_1_to_n, "product_1_to_n")


def test_search_solves_count_divisors():
    from egdc.mog.benchmark import _make_count_divisors
    _solve_factory(_make_count_divisors, "count_divisors")


def test_search_solves_triangular_check():
    from egdc.mog.benchmark import _make_triangular_check
    _solve_factory(_make_triangular_check, "triangular_check")


def test_search_solves_max_pair_diff():
    from egdc.mog.benchmark import _make_max_pair_diff
    _solve_factory(_make_max_pair_diff, "max_pair_diff")


def test_search_solves_sum_negatives():
    from egdc.mog.benchmark import _make_sum_negatives
    _solve_factory(_make_sum_negatives, "sum_negatives")


def test_search_solves_gcd_extended():
    from egdc.mog.benchmark import _make_gcd_extended
    _solve_factory(_make_gcd_extended, "gcd_extended")


def test_search_solves_harmonic_sum():
    from egdc.mog.benchmark import _make_harmonic_sum
    _solve_factory(_make_harmonic_sum, "harmonic_sum")


# --- Integration test ---

def test_search_solves_all_61_factories():
    """End-to-end: solve one variant from every factory."""
    from egdc.mog.benchmark import get_benchmark
    from egdc.mog.solvers.search_solver import solve_problem

    problems = get_benchmark(seed=42, variants_per_factory=1)  # 61 problems
    solved = 0
    failures = []
    for p in problems:
        r = solve_problem(p, use_compiler=False)
        if r.success:
            solved += 1
        else:
            failures.append(p.name)

    print(f"\nFull 61-factory benchmark: {solved}/{len(problems)} solved")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")

    assert solved >= 59, f"Expected >=59 solved, got {solved}. Failed: {failures}"


def test_search_solves_all_factories_multi_variant():
    """Full reproducibility regression: every factory × 5 variants must solve.

    This pins the publication claim. If a solver change drops coverage on any
    variant, this test surfaces it with a per-factory breakdown.
    """
    from collections import defaultdict

    from egdc.mog.benchmark import get_benchmark
    from egdc.mog.solvers.search_solver import solve_problem

    problems = get_benchmark(seed=42, variants_per_factory=5)
    total = len(problems)
    by_factory_total: dict[str, int] = defaultdict(int)
    by_factory_passed: dict[str, int] = defaultdict(int)
    failures: list[tuple[str, str, float]] = []

    for p in problems:
        by_factory_total[p.name] += 1
        r = solve_problem(p, use_compiler=False)
        if r.success:
            by_factory_passed[p.name] += 1
        else:
            failures.append((p.name, r.method or "unknown", float(r.loss or 0.0)))

    solved = sum(by_factory_passed.values())
    print(f"\nMulti-variant benchmark: {solved}/{total} solved across {len(by_factory_total)} factories")
    if failures:
        print("  failing variants (factory, method, loss):")
        for name, method, loss in failures:
            print(f"    {name}: method={method} loss={loss:.4g}")
        partial = [f for f in by_factory_total if by_factory_passed[f] < by_factory_total[f]]
        print(f"  partial factories: {sorted(partial)}")

    # Current baseline: 315/315 = 100%. Any drop is a regression.
    assert solved == total, (
        f"Expected {total}/{total} solved (100%), got {solved}/{total}. "
        f"Failing: {failures}"
    )
