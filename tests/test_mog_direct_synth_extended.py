from egdc.mog_execute import execute_mog


def test_direct_synth_recovers_sign_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((-5.0,), -1.0),
        ((0.0,), 0.0),
        ((7.0,), 1.0),
        ((3.0,), 1.0),
    ]
    result = synthesize_expression_program(
        function_name="sign",
        arg_names=["x"],
        examples=examples,
        template="sign3",
        steps=200,
        lr=0.05,
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(sign(-2)); println_i64(sign(0)); println_i64(sign(9)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "-1\n0\n1"


def test_direct_synth_recovers_gcd_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((12.0, 18.0), 6.0),
        ((21.0, 14.0), 7.0),
        ((9.0, 28.0), 1.0),
        ((48.0, 18.0), 6.0),
    ]
    result = synthesize_expression_program(
        function_name="gcd",
        arg_names=["a", "b"],
        examples=examples,
        template="gcd_euclid",
        steps=200,
        lr=0.05,
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(gcd(84, 30)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "6"


def test_direct_synth_recovers_array_sum_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        (((1.0, 2.0, 3.0),), 6.0),
        (((5.0,),), 5.0),
        (((4.0, 4.0),), 8.0),
        (((2.0, 7.0, 1.0, 0.0),), 10.0),
    ]
    result = synthesize_expression_program(
        function_name="array_sum",
        arg_names=["arr"],
        arg_types=["[i64]"],
        examples=examples,
        template="array_sum_reduce",
        steps=200,
        lr=0.05,
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    program = result.code + "\nfn main() -> int { nums: [i64] = [3, 4, 5]; println_i64(array_sum(nums)); return 0; }"
    run = execute_mog(program)
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "12"
