from egdc.mog_execute import execute_mog


def test_direct_synth_recovers_count_occurrences_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        (((1.0, 2.0, 1.0), 1.0), 2.0),
        (((5.0,), 5.0), 1.0),
        (((4.0, 9.0), 1.0), 0.0),
        (((2.0, 7.0, 2.0, 0.0), 2.0), 2.0),
    ]
    result = synthesize_expression_program(
        function_name="count_occurrences",
        arg_names=["arr", "target"],
        arg_types=["[i64]", "i64"],
        examples=examples,
        template="count_occurrences_reduce",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    program = result.code + "\nfn main() -> int { nums: [i64] = [3, 2, 2, 9, 2]; println_i64(count_occurrences(nums, 2)); return 0; }"
    run = execute_mog(program)
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "3"


def test_direct_synth_recovers_digit_sum_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((0.0,), 0.0),
        ((7.0,), 7.0),
        ((123.0,), 6.0),
        ((9081.0,), 18.0),
    ]
    result = synthesize_expression_program(
        function_name="digit_sum",
        arg_names=["n"],
        examples=examples,
        template="digit_sum_loop",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(digit_sum(507)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "12"


def test_direct_synth_recovers_safe_div_or_neg1_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((10.0, 2.0), 5.0),
        ((7.0, 0.0), -1.0),
        ((9.0, 3.0), 3.0),
        ((5.0, 0.0), -1.0),
    ]
    result = synthesize_expression_program(
        function_name="safe_div_or_neg1",
        arg_names=["a", "b"],
        examples=examples,
        template="safe_div_or_neg1",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(safe_div_or_neg1(20, 5)); println_i64(safe_div_or_neg1(20, 0)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "4\n-1"


def test_direct_synth_recovers_positive_or_default_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((10.0,), 10.0),
        ((0.0,), 0.0),
        ((-5.0,), 0.0),
        ((3.0,), 3.0),
    ]
    result = synthesize_expression_program(
        function_name="positive_or_default",
        arg_names=["x"],
        examples=examples,
        template="positive_or_default",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(positive_or_default(-2)); println_i64(positive_or_default(9)); return 0; }")
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "0\n9"


def test_direct_synth_recovers_point_sum_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((3.0, 4.0), 7.0),
        ((-1.0, 2.0), 1.0),
        ((0.0, 0.0), 0.0),
        ((9.0, 8.0), 17.0),
    ]
    result = synthesize_expression_program(
        function_name="point_sum",
        arg_names=["x", "y"],
        examples=examples,
        template="point_sum_struct",
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-6
    program = result.code + "\nfn main() -> int { p := Point { x: 5, y: 6 }; println_i64(point_sum(p)); return 0; }"
    run = execute_mog(program)
    assert run.success, run.stderr or run.compile_stderr or run.error
    assert run.stdout.strip() == "11"
