from egdc.mog.execute import execute_mog


def test_direct_synth_recovers_clamp_program():
    from egdc.mog.solvers.direct_synth import synthesize_expression_program

    examples = [
        ((-5.0,), 0.0),
        ((0.0,), 0.0),
        ((37.0,), 37.0),
        ((140.0,), 100.0),
    ]
    result = synthesize_expression_program(
        function_name="clamp_0_100",
        arg_names=["x"],
        examples=examples,
        template="clamp_0_100",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(clamp_0_100(-2)); println_i64(clamp_0_100(55)); println_i64(clamp_0_100(250)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "0\n55\n100"


def test_direct_synth_recovers_array_max_program():
    from egdc.mog.solvers.direct_synth import synthesize_expression_program

    examples = [
        (((1.0, 2.0, 3.0),), 3.0),
        (((5.0,),), 5.0),
        (((4.0, 9.0),), 9.0),
        (((2.0, 7.0, 1.0, 0.0),), 7.0),
    ]
    result = synthesize_expression_program(
        function_name="array_max",
        arg_names=["arr"],
        arg_types=["[i64]"],
        examples=examples,
        template="array_max_reduce",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    program = result.code + "\nfn main() -> int { nums: [i64] = [3, 11, 5]; println_i64(array_max(nums)); return 0; }"
    run = execute_mog(program)
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "11"


def test_direct_synth_recovers_count_positive_program():
    from egdc.mog.solvers.direct_synth import synthesize_expression_program

    examples = [
        (((1.0, 2.0, 3.0),), 3.0),
        (((-5.0,),), 0.0),
        (((4.0, -9.0),), 1.0),
        (((2.0, 7.0, -1.0, 0.0),), 2.0),
    ]
    result = synthesize_expression_program(
        function_name="count_positive",
        arg_names=["arr"],
        arg_types=["[i64]"],
        examples=examples,
        template="count_positive_reduce",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    program = result.code + "\nfn main() -> int { nums: [i64] = [3, -2, 0, 9, 5]; println_i64(count_positive(nums)); return 0; }"
    run = execute_mog(program)
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "3"


def test_direct_synth_recovers_lcm_program():
    from egdc.mog.solvers.direct_synth import synthesize_expression_program

    examples = [
        ((3.0, 4.0), 12.0),
        ((6.0, 8.0), 24.0),
        ((5.0, 10.0), 10.0),
        ((7.0, 9.0), 63.0),
    ]
    result = synthesize_expression_program(
        function_name="lcm",
        arg_names=["a", "b"],
        examples=examples,
        template="lcm_via_gcd",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(lcm(12, 18)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "36"
