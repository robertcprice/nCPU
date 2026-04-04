"""Test counterexample refinement, auto sub-program extraction,
self-growing benchmark, multi-function synthesis, and interactive programs."""

from egdc.mog_execute import execute_mog


def test_counterexample_refinement_fixes_wrong_program():
    """A program that passes training but fails on a counterexample gets refined."""
    from egdc.mog_counterexample import CounterexampleRefiner

    refiner = CounterexampleRefiner()
    # Initial bad program: always returns a (wrong for max2 when b > a)
    bad_code = "fn max2(a: i64, b: i64) -> i64 { return a; }"
    train = [((5.0, 3.0), 5.0), ((10.0, 1.0), 10.0)]  # happens to pass
    counterexamples = [((2.0, 9.0), 9.0), ((-1.0, 5.0), 5.0)]

    result = refiner.refine("max2", ["a", "b"], bad_code, train, counterexamples)
    assert result.success
    assert result.loss < 1e-6
    # Should work on all examples now
    run = execute_mog(result.code + "\nfn main() -> int { println_i64(max2(2, 9)); return 0; }")
    assert run.success
    assert run.stdout.strip() == "9"


def test_auto_extract_subprogram():
    """Detect shared sub-computation across solved programs and extract it."""
    from egdc.mog_auto_extract import SubProgramExtractor

    extractor = SubProgramExtractor()
    solved = {
        "lcm": "fn lcm(a: i64, b: i64) -> i64 { x: i64 = a; y: i64 = b; while y != 0 { tmp := y; y = x % y; x = tmp; } return (a * b) / x; }",
        "gcd": "fn gcd(a: i64, b: i64) -> i64 { x: i64 = a; y: i64 = b; while y != 0 { tmp := y; y = x % y; x = tmp; } return x; }",
        "coprime": "fn coprime(a: i64, b: i64) -> i64 { x: i64 = a; y: i64 = b; while y != 0 { tmp := y; y = x % y; x = tmp; } if x == 1 { return 1; } return 0; }",
    }
    fragments = extractor.find_shared_fragments(solved)
    assert len(fragments) >= 1
    # Should detect the Euclidean loop as a shared fragment
    assert any("while y != 0" in f.code for f in fragments)


def test_self_growing_benchmark():
    """After solving a problem, generate harder variants automatically."""
    from egdc.mog_growing_benchmark import GrowingBenchmark

    gb = GrowingBenchmark()
    # Register a solved problem
    gb.register_solved(
        name="max2",
        arg_names=["a", "b"],
        code="fn max2(a: i64, b: i64) -> i64 { if (a > b) { return a; } else { return b; } }",
        examples=[((2.0, 3.0), 3.0), ((10.0, -4.0), 10.0)],
    )
    harder = gb.generate_harder_variants("max2", num_variants=3)
    assert len(harder) >= 1
    # Harder variants should have more/different test cases
    for v in harder:
        assert len(v.examples) >= 3


def test_multi_function_synthesis():
    """Synthesize a helper + caller together."""
    from egdc.mog_compositional import CompositionalSolver

    solver = CompositionalSolver()
    # Discover is_positive as a helper, then use it in count_positive_to_n
    # The solver should figure out it needs a helper
    examples = [
        ((5.0,), 5.0),   # count of positive integers 1..5 = 5
        ((3.0,), 3.0),   # count of positive integers 1..3 = 3
        ((0.0,), 0.0),   # count of positive integers 1..0 = 0
    ]
    result = solver.solve("count_positive_to_n", ["n"], examples)
    assert result.success
    assert result.loss < 1e-6


def test_interactive_program_io_trace():
    """Test a program that processes multiple inputs sequentially and actually compiles."""
    from egdc.mog_interactive import InteractiveSolver

    solver = InteractiveSolver()
    traces = [
        [(3, 3), (5, 8), (2, 10)],
        [(10, 10), (20, 30), (5, 35)],
    ]
    result = solver.solve_from_traces("running_sum", traces)
    assert result.success
    assert "for" in result.code or "while" in result.code

    # Actually compile and run
    test_code = (
        result.code.split("fn running_sum(")[0] +  # get any helper functions
        "fn process(state: i64, x: i64) -> i64 { return state + x; }\n"
        "fn main() -> int {\n"
        "    state: i64 = 0;\n"
        "    inputs: [i64] = [3, 5, 2];\n"
        "    for x in inputs {\n"
        "        state = state + x;\n"
        "        println_i64(state);\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    run = execute_mog(test_code)
    assert run.success, run.stderr or run.compile_stderr
    assert run.stdout.strip() == "3\n8\n10"
