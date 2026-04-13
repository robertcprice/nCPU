from pathlib import Path


def test_tfidf_retrieval_outperforms_word_overlap(tmp_path: Path):
    from egdc.mog.routing.pathways import PathwayMemory

    mem = PathwayMemory(root=tmp_path)
    mem.record_success("array_sum_v0", "array_sum_reduce",
        "fn array_sum(arr: [i64]) -> i64 { ... }",
        {"description": "Return the sum of all elements in an array of i64 values.", "signature": "fn array_sum(arr: [i64]) -> i64"})
    mem.record_success("array_max_v0", "array_max_reduce",
        "fn array_max(arr: [i64]) -> i64 { ... }",
        {"description": "Return the largest element in a non-empty array.", "signature": "fn array_max(arr: [i64]) -> i64"})
    mem.record_success("add_two_v0", "binary",
        "fn add_two(a: i64, b: i64) -> i64 { return a + b; }",
        {"description": "Return the sum of two i64 integers.", "signature": "fn add_two(a: i64, b: i64) -> i64"})
    mem.save()

    # "sum all items in array" should match array_sum better than add_two.
    hits = mem.retrieve_similar("Sum all items in the array.", "fn total(arr: [i64]) -> i64", top_k=3)
    assert hits[0]["family"] == "array_sum_reduce"


def test_auto_regression_creates_problem_from_failure(tmp_path: Path):
    from egdc.mog.routing.regression_bank import RegressionBank

    bank = RegressionBank(root=tmp_path)
    bank.add_regression(
        problem_name="weird_gcd_edge",
        description="gcd with one arg zero fails",
        code="fn gcd(a: i64, b: i64) -> i64 { ... }",
        error="division by zero at runtime",
        test_input="gcd(0, 5)",
        expected_output="5",
    )
    bank.save()

    bank2 = RegressionBank(root=tmp_path)
    bank2.load()
    assert len(bank2.regressions) == 1
    assert bank2.regressions[0].description == "gcd with one arg zero fails"


def test_composition_template_solves_lcm_from_gcd():
    from egdc.mog.solvers.direct_synth import synthesize_expression_program
    from egdc.mog.execute import execute_mog

    # LCM is already a single family, but test that a composed gcd+division
    # template also works as a concept.
    examples = [((4.0, 6.0), 12.0), ((3.0, 7.0), 21.0), ((12.0, 8.0), 24.0)]
    result = synthesize_expression_program(
        function_name="lcm2",
        arg_names=["a", "b"],
        examples=examples,
        template="lcm_via_gcd",
        seed=0,
    )
    assert result.success
    assert result.loss < 1e-6
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(lcm2(15, 20)); return 0; }")
    assert run.success
    assert run.stdout.strip() == "60"


def test_family_induction_detects_array_pattern(tmp_path: Path):
    from egdc.mog.routing.family_inductor import FamilyInductor

    inductor = FamilyInductor()
    solved_codes = [
        ("array_sum", "fn array_sum(arr: [i64]) -> i64 { total: i64 = 0; for item in arr { total = total + item; } return total; }"),
        ("array_max", "fn array_max(arr: [i64]) -> i64 { best := arr[0]; for item in arr { if item > best { best = item; } } return best; }"),
        ("count_positive", "fn count_positive(arr: [i64]) -> i64 { total: i64 = 0; for item in arr { if item > 0 { total = total + 1; } } return total; }"),
    ]
    patterns = inductor.detect_patterns(solved_codes)
    assert len(patterns) >= 1
    # Should detect "array reduction with for-in loop" pattern.
    assert any("for item in" in p["shared_structure"] for p in patterns)
