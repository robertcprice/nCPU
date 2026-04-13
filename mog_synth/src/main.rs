use std::fs;
use std::path::PathBuf;

use mog_synth::benchmark::get_benchmark;
use mog_synth::interactive::{
    solve_interactive_problem, solve_interactive_problem_differentiable_only,
};
use mog_synth::orchestrator::Orchestrator;
use mog_synth::runtime::{execute_program, execute_program_with_input};
use mog_synth::solver::{
    solve_benchmark, solve_benchmark_differentiable_only, solve_benchmark_legacy_only,
    solve_benchmark_prefer_differentiable, solve_benchmark_with_legacy_fallback, solve_problem,
    solve_problem_differentiable_only, solve_problem_legacy_only,
    solve_problem_prefer_differentiable, solve_problem_with_legacy_fallback,
};

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn default_memory_root() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".mog_synth_memory")
}

fn parse_stdin_data(raw: &str) -> Vec<String> {
    raw.split(|ch: char| ch.is_ascii_whitespace() || ch == ',')
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_orchestrator = has_flag(&args, "--orchestrate");
    let use_interactive = has_flag(&args, "--interactive");
    let use_differentiable_only = has_flag(&args, "--differentiable-only");
    let prefer_differentiable = has_flag(&args, "--prefer-differentiable");
    let use_legacy_only = has_flag(&args, "--legacy-only");
    let use_legacy_fallback = has_flag(&args, "--legacy-fallback");
    let memory_root = arg_value(&args, "--memory-root")
        .map(PathBuf::from)
        .unwrap_or_else(default_memory_root);
    let stdin_data = arg_value(&args, "--stdin-data")
        .map(|raw| parse_stdin_data(&raw))
        .unwrap_or_default();

    if let Some(path) = arg_value(&args, "--run-file") {
        let code = fs::read_to_string(&path).unwrap_or_else(|err| {
            eprintln!("failed to read {path}: {err}");
            std::process::exit(1);
        });
        let result = if stdin_data.is_empty() {
            execute_program(&code)
        } else {
            execute_program_with_input(&code, stdin_data.clone())
        }
        .unwrap_or_else(|err| {
            eprintln!("execution failed for {path}: {err}");
            std::process::exit(1);
        });
        if !result.output.is_empty() {
            println!("{}", result.output);
        } else if let Some(value) = result.return_value {
            println!("{value}");
        }
        return;
    }

    if let Some(query) = arg_value(&args, "--problem") {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with(&query) || p.function_name() == query);

        match problem {
            Some(problem) => {
                if use_interactive {
                    if use_legacy_only || use_legacy_fallback {
                        eprintln!("interactive synthesis does not support legacy modes");
                        std::process::exit(1);
                    }
                    if use_orchestrator {
                        let mut orchestrator =
                            Orchestrator::new(&memory_root).unwrap_or_else(|err| {
                                eprintln!("failed to initialize orchestrator: {err}");
                                std::process::exit(1);
                            });
                        let result = orchestrator.solve_interactive(&problem);
                        if !result.success {
                            eprintln!(
                                "failed: {}",
                                result.error.unwrap_or_else(|| "unknown error".to_string())
                            );
                            std::process::exit(1);
                        }
                        println!("problem: interactive_{}", problem.name);
                        println!("method: {}", result.method);
                        println!("family: {}", result.family);
                        println!("memory_records: {}", orchestrator.memory.total_successes());
                        println!("{}", result.code);
                        return;
                    }
                    let result = if use_differentiable_only || prefer_differentiable {
                        solve_interactive_problem_differentiable_only(&problem)
                    } else {
                        solve_interactive_problem(&problem)
                    };
                    if !result.success {
                        eprintln!(
                            "failed: {}",
                            result.error.unwrap_or_else(|| "unknown error".to_string())
                        );
                        std::process::exit(1);
                    }
                    println!("problem: interactive_{}", problem.name);
                    println!("method: {}", result.method);
                    println!("{}", result.code);
                    return;
                }
                if use_orchestrator {
                    let mut orchestrator = Orchestrator::new(&memory_root).unwrap_or_else(|err| {
                        eprintln!("failed to initialize orchestrator: {err}");
                        std::process::exit(1);
                    });
                    let result = if use_differentiable_only {
                        orchestrator.solve_differentiable_only(&problem)
                    } else if prefer_differentiable {
                        orchestrator.solve_prefer_differentiable(&problem)
                    } else if use_legacy_only {
                        orchestrator.solve_legacy_only(&problem)
                    } else if use_legacy_fallback {
                        orchestrator.solve_with_legacy_fallback(&problem)
                    } else {
                        orchestrator.solve(&problem)
                    };
                    if result.success {
                        println!("problem: {}", problem.name);
                        println!("method: {}", result.method);
                        println!("family: {}", result.family);
                        println!("memory_records: {}", orchestrator.memory.total_successes());
                        println!("{}", result.code);
                    } else {
                        eprintln!(
                            "failed: {}",
                            result.error.unwrap_or_else(|| "unknown error".to_string())
                        );
                        std::process::exit(1);
                    }
                } else {
                    let result = if use_differentiable_only {
                        solve_problem_differentiable_only(&problem)
                    } else if prefer_differentiable {
                        solve_problem_prefer_differentiable(&problem)
                    } else if use_legacy_only {
                        solve_problem_legacy_only(&problem)
                    } else if use_legacy_fallback {
                        solve_problem_with_legacy_fallback(&problem)
                    } else {
                        solve_problem(&problem)
                    };
                    if !result.success {
                        eprintln!(
                            "failed: {}",
                            result.error.unwrap_or_else(|| "unknown error".to_string())
                        );
                        std::process::exit(1);
                    }
                    println!("problem: {}", problem.name);
                    println!("method: {}", result.method);
                    println!("{}", result.code);
                }
            }
            None => {
                eprintln!("unknown problem: {query}");
                std::process::exit(1);
            }
        }
        return;
    }

    let problems = get_benchmark(1);
    if use_interactive {
        if use_legacy_only || use_legacy_fallback {
            eprintln!("interactive batch synthesis does not support legacy modes");
            std::process::exit(1);
        }
        if use_orchestrator {
            let mut orchestrator = Orchestrator::new(&memory_root).unwrap_or_else(|err| {
                eprintln!("failed to initialize orchestrator: {err}");
                std::process::exit(1);
            });
            let results = orchestrator.solve_batch_interactive(&problems);
            let solved = results.iter().filter(|result| result.success).count();
            let failures = problems
                .iter()
                .zip(results.iter())
                .filter_map(|(problem, result)| {
                    if result.success {
                        None
                    } else {
                        Some(problem.name.clone())
                    }
                })
                .collect::<Vec<_>>();
            println!(
                "Solved {}/{} interactive problems via orchestrator",
                solved,
                problems.len()
            );
            println!("Memory records: {}", orchestrator.memory.total_successes());
            if !failures.is_empty() {
                println!("Failures: {}", failures.join(", "));
                std::process::exit(1);
            }
            return;
        }
        let liftable = problems
            .iter()
            .filter(|problem| {
                let result = if use_differentiable_only || prefer_differentiable {
                    solve_interactive_problem_differentiable_only(problem)
                } else {
                    solve_interactive_problem(problem)
                };
                result.success
            })
            .count();
        println!(
            "Solved {}/{} interactive problems",
            liftable,
            problems.len()
        );
        return;
    }
    if use_orchestrator {
        let mut orchestrator = Orchestrator::new(&memory_root).unwrap_or_else(|err| {
            eprintln!("failed to initialize orchestrator: {err}");
            std::process::exit(1);
        });
        let results = if use_differentiable_only {
            orchestrator.solve_batch_differentiable_only(&problems)
        } else if prefer_differentiable {
            orchestrator.solve_batch_prefer_differentiable(&problems)
        } else if use_legacy_only {
            orchestrator.solve_batch_legacy_only(&problems)
        } else if use_legacy_fallback {
            orchestrator.solve_batch_with_legacy_fallback(&problems)
        } else {
            orchestrator.solve_batch(&problems)
        };
        let solved = results.iter().filter(|result| result.success).count();
        let failures = problems
            .iter()
            .zip(results.iter())
            .filter_map(|(problem, result)| {
                if result.success {
                    None
                } else {
                    Some(problem.name.clone())
                }
            })
            .collect::<Vec<_>>();
        println!(
            "Solved {}/{} problems via orchestrator",
            solved,
            problems.len()
        );
        println!("Memory records: {}", orchestrator.memory.total_successes());
        if !failures.is_empty() {
            println!("Failures: {}", failures.join(", "));
            std::process::exit(1);
        }
    } else {
        let summary = if use_differentiable_only {
            solve_benchmark_differentiable_only(&problems)
        } else if prefer_differentiable {
            solve_benchmark_prefer_differentiable(&problems)
        } else if use_legacy_only {
            solve_benchmark_legacy_only(&problems)
        } else if use_legacy_fallback {
            solve_benchmark_with_legacy_fallback(&problems)
        } else {
            solve_benchmark(&problems)
        };
        println!("Solved {}/{} problems", summary.solved, summary.total);
        if !summary.failures.is_empty() {
            println!("Failures: {}", summary.failures.join(", "));
            std::process::exit(1);
        }
    }
}
