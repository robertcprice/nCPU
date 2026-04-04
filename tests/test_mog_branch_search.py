"""Test that differentiable search reliably discovers branching programs."""

from egdc.mog_execute import execute_mog


def test_search_discovers_max2_exactly():
    from egdc.mog_program_search import search_program

    examples = [
        ((2.0, 3.0), 3.0),
        ((10.0, -4.0), 10.0),
        ((7.0, 7.0), 7.0),
        ((-3.0, -2.0), -2.0),
        ((0.0, 5.0), 5.0),
        ((5.0, 0.0), 5.0),
    ]
    result = search_program(
        arg_names=["a", "b"],
        examples=examples,
        function_name="max2",
        num_slots=6,
        steps=2000,
        lr=0.03,
        num_restarts=5,
        seed=42,
    )
    print(f"max2: loss={result.loss}, code:\n{result.code}")
    assert result.success, f"loss={result.loss}"
    assert result.loss < 1.0
    # Verify with real compiler on held-out inputs.
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(max2(4, 9)); println_i64(max2(11, 3)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "9\n11"


def test_search_discovers_abs_diff_exactly():
    from egdc.mog_program_search import search_program

    examples = [
        ((2.0, 3.0), 1.0),
        ((10.0, -4.0), 14.0),
        ((7.0, 7.0), 0.0),
        ((-3.0, -2.0), 1.0),
        ((0.0, 5.0), 5.0),
        ((5.0, 0.0), 5.0),
    ]
    result = search_program(
        arg_names=["a", "b"],
        examples=examples,
        function_name="abs_diff",
        num_slots=6,
        steps=2000,
        lr=0.03,
        num_restarts=5,
        seed=42,
    )
    print(f"abs_diff: loss={result.loss}, code:\n{result.code}")
    assert result.success, f"loss={result.loss}"
    assert result.loss < 1.0
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(abs_diff(8, 3)); println_i64(abs_diff(3, 8)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "5\n5"


def test_search_discovers_min2():
    from egdc.mog_program_search import search_program

    examples = [
        ((2.0, 3.0), 2.0),
        ((10.0, -4.0), -4.0),
        ((7.0, 7.0), 7.0),
        ((-3.0, -2.0), -3.0),
        ((0.0, 5.0), 0.0),
        ((5.0, 0.0), 0.0),
    ]
    result = search_program(
        arg_names=["a", "b"],
        examples=examples,
        function_name="min2",
        num_slots=6,
        steps=2000,
        lr=0.03,
        num_restarts=3,
        seed=42,
    )
    print(f"min2: loss={result.loss}, code:\n{result.code}")
    assert result.success, f"loss={result.loss}"
    assert result.loss < 1.0
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(min2(4, 9)); println_i64(min2(11, 3)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "4\n3"
