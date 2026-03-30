from egdc.mog_benchmark import get_benchmark


def test_direct_router_solves_first_10_benchmark_problems():
    from egdc.mog_direct_router import solve_problem_direct, evaluate_direct_solver

    problems = get_benchmark(seed=42, variants_per_factory=1)[:10]
    summary = evaluate_direct_solver(problems, use_real_compiler=True)
    assert summary["num_problems"] == 10
    assert summary["num_solved"] >= 10
    assert summary["pass_rate"] >= 1.0

    first = solve_problem_direct(problems[0])
    assert first is not None
    assert first.success
    assert "fn " in first.code
