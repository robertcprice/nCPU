use std::io::Write;
use std::process::{Command, Stdio};

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

fn try_python_warmstart(problem: &Problem) -> Option<SolveResult> {
    let n_args = problem
        .examples
        .first()
        .map(|e| e.inputs.len())
        .unwrap_or(0);
    if n_args != 1 {
        return None;
    }

    let examples: Vec<serde_json::Value> = problem
        .examples
        .iter()
        .map(|ex| {
            let inputs: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            serde_json::json!([inputs, ex.expected])
        })
        .collect();

    let req = serde_json::json!({
        "name":     problem.name,
        "examples": examples,
        "n_args":   1,
    });

    let project_root = std::env::current_exe()
        .ok()
        .and_then(|p| {
            let mut dir = p;
            for _ in 0..6 {
                dir = dir.parent()?.to_path_buf();
                if dir.join("scripts/py_warmstart.py").exists() {
                    return Some(dir);
                }
            }
            None
        })
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    let script = project_root.join("scripts/py_warmstart.py");
    let model = find_python_warmstart_model(&project_root)?;
    if !script.exists() {
        return None;
    }

    let mut child = Command::new("python3")
        .arg(&script)
        .arg("--model")
        .arg(&model)
        .arg("--n-steps")
        .arg("400")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    if let Some(stdin) = child.stdin.take() {
        let mut stdin = stdin;
        let _ = stdin.write_all(req.to_string().as_bytes());
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }

    let resp: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    if resp.get("solved")?.as_bool()? {
        Some(SolveResult {
            success: true,
            code: resp.get("code")?.as_str().unwrap_or("").to_string(),
            method: resp
                .get("method")
                .and_then(|m| m.as_str())
                .unwrap_or("py_warmstart")
                .to_string(),
            error: None,
            metadata: DifferentiableMetadata::default(),
        })
    } else {
        None
    }
}

pub(super) fn solve_benchmark_prefer_differentiable(problems: &[Problem]) -> BenchmarkSummary {
    let mut solved = 0;
    let mut failures = Vec::new();

    for problem in problems {
        let result = super::solve_problem_prefer_differentiable(problem);
        if result.success {
            solved += 1;
        } else if let Some(py_result) = try_python_warmstart(problem) {
            eprintln!(
                "[py_fallback] {} -> SOLVED ({})",
                problem.name, py_result.method
            );
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
