"""Test multi-branch and loop discovery via differentiable search."""

from egdc.mog.execute import execute_mog


def test_search_discovers_sign():
    from egdc.mog.solvers.program_search import search_program

    examples = [
        ((-5.0,), -1.0),
        ((0.0,), 0.0),
        ((7.0,), 1.0),
        ((3.0,), 1.0),
        ((-1.0,), -1.0),
    ]
    result = search_program(
        arg_names=["x"],
        examples=examples,
        function_name="sign",
        num_slots=8,
        steps=500,
        lr=0.03,
        num_restarts=3,
        seed=42,
    )
    print(f"sign: loss={result.loss}, code:\n{result.code}")
    assert result.success, f"loss={result.loss}"
    assert result.loss < 1.0
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(sign(-9)); println_i64(sign(0)); println_i64(sign(4)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "-1\n0\n1"


def test_search_discovers_sum_to_n():
    from egdc.mog.solvers.program_search import search_program

    examples = [
        ((0.0,), 0.0),
        ((1.0,), 1.0),
        ((5.0,), 15.0),
        ((10.0,), 55.0),
    ]
    result = search_program(
        arg_names=["n"],
        examples=examples,
        function_name="sum_to_n",
        num_slots=8,
        steps=500,
        lr=0.03,
        num_restarts=3,
        seed=42,
    )
    print(f"sum_to_n: loss={result.loss}, code:\n{result.code}")
    assert result.success, f"loss={result.loss}"
    assert result.loss < 1.0
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(sum_to_n(7)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "28"
