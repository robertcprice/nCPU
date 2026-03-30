from egdc.mog_execute import execute_mog


def test_direct_synth_recovers_sum_to_n_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((0.0,), 0.0),
        ((1.0,), 1.0),
        ((5.0,), 15.0),
        ((10.0,), 55.0),
    ]
    result = synthesize_expression_program(
        function_name="sum_to_n",
        arg_names=["n"],
        examples=examples,
        template="sum_to_n",
        steps=400,
        lr=0.05,
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(sum_to_n(7)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "28"


def test_direct_synth_recovers_is_even_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((0.0,), 1.0),
        ((1.0,), 0.0),
        ((8.0,), 1.0),
        ((11.0,), 0.0),
    ]
    result = synthesize_expression_program(
        function_name="is_even",
        arg_names=["x"],
        examples=examples,
        template="mod2_eq0",
        steps=200,
        lr=0.05,
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(is_even(4)); println_i64(is_even(9)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "1\n0"
