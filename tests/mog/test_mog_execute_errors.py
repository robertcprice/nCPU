from egdc.mog.benchmark import MogBenchmarkProblem, evaluate_solution_with_compiler
from egdc.mog.execute import ExecuteResult, compile_mog


def test_compile_mog_reports_missing_runtime_path():
    result = compile_mog(
        "fn main() -> int { println_i64(1); return 0; }",
        mogc="/bin/sh",
        runtime="/definitely/missing/libmog_runtime.a",
    )

    assert not result.success
    assert "mog runtime not found" in result.stderr


def test_compiler_evaluation_surfaces_compile_error(monkeypatch):
    problem = MogBenchmarkProblem(
        name="add_two_v0",
        category="arithmetic",
        description="Return the sum of two i64 integers.",
        signature="fn add_two(a: i64, b: i64) -> i64",
        test_cases=[((2, 3), "5")],
        wrapper_template="fn main() -> i64 { println_i64(add_two(2, 3)); return 0; }",
        reference_solution=None,
    )

    def fake_execute(_code: str) -> ExecuteResult:
        return ExecuteResult(
            compiled=False,
            compile_stderr="mogc not found at /tmp/mogc",
            success=False,
            stdout="",
            stderr="",
            returncode=-1,
            error="Compilation failed: mogc not found at /tmp/mogc",
        )

    monkeypatch.setattr("egdc.mog.benchmark.execute_mog", fake_execute)

    result = evaluate_solution_with_compiler(
        problem,
        "fn add_two(a: i64, b: i64) -> i64 { return a + b; }",
    )

    assert not result.passed
    assert result.error is not None
    assert "mogc not found" in result.error
