use super::*;

pub(super) fn run_benchmark(
    problems: &[Problem],
    solver: fn(&Problem) -> SolveResult,
) -> BenchmarkSummary {
    let mut solved = 0;
    let mut failures = Vec::new();

    for problem in problems {
        let result = solver(problem);
        if result.success {
            solved += 1;
        } else {
            failures.push(problem.name.clone());
        }
    }

    BenchmarkSummary {
        total: problems.len(),
        solved,
        failures,
    }
}

pub(super) fn find_python_warmstart_model(
    project_root: &std::path::Path,
) -> Option<std::path::PathBuf> {
    [
        "models/metalearner_1arg_v5.pt",
        "models/metalearner_1arg_v4.pt",
        "models/metalearner_1arg_v3.pt",
        "models/metalearner_1arg_known.pt",
        "models/metalearner_1arg.pt",
    ]
    .into_iter()
    .map(|rel| project_root.join(rel))
    .find(|path| path.exists())
}

pub(super) fn solve_benchmark_prefer_differentiable(problems: &[Problem]) -> BenchmarkSummary {
    run_benchmark(problems, super::solve_problem_prefer_differentiable)
}
