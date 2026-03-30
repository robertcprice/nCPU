from egdc.mog_execute import execute_mog


def test_direct_synth_recovers_add_two_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((2.0, 3.0), 5.0),
        ((10.0, -4.0), 6.0),
        ((7.0, 8.0), 15.0),
        ((-3.0, -2.0), -5.0),
    ]
    result = synthesize_expression_program(
        function_name="add_two",
        arg_names=["a", "b"],
        examples=examples,
        template="binary",
        steps=300,
        lr=0.1,
        seed=0,
    )
    assert result.success, result
    assert result.loss < 1e-2
    assert "fn add_two(a: i64, b: i64) -> i64" in result.code
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(add_two(4, 9)); return 0; }")
    assert run.success
    assert run.stdout.strip() == "13"


def test_direct_synth_recovers_max2_program():
    from egdc.mog_direct_synth import synthesize_expression_program

    examples = [
        ((2.0, 3.0), 3.0),
        ((10.0, -4.0), 10.0),
        ((7.0, 7.0), 7.0),
        ((-3.0, -2.0), -2.0),
    ]
    result = synthesize_expression_program(
        function_name="max2",
        arg_names=["a", "b"],
        examples=examples,
        template="if_cmp",
        steps=500,
        lr=0.1,
        seed=0,
    )
    assert result.success, result
    assert result.loss < 5e-2
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(max2(4, 9)); println_i64(max2(11, 3)); return 0; }")
    assert run.success
    assert run.stdout.strip() == "9\n11"
