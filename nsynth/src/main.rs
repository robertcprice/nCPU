use std::fs;
use std::path::PathBuf;

use mog_synth::benchmark::{get_benchmark, Example, Problem, Value};
use mog_synth::enumerative;
use mog_synth::interactive::{
    solve_interactive_problem, solve_interactive_problem_differentiable_only,
};
use mog_synth::morph_transduce::{solve_morph_transduction, StrExample};
use mog_synth::orchestrator::Orchestrator;
use mog_synth::runtime::{execute_program, execute_program_with_input};
use mog_synth::solver::{
    solve_benchmark, solve_benchmark_differentiable_only, solve_benchmark_legacy_only,
    solve_benchmark_prefer_differentiable, solve_benchmark_with_legacy_fallback, solve_problem,
    solve_problem_differentiable_only, solve_problem_legacy_only,
    solve_problem_prefer_differentiable, solve_problem_with_legacy_fallback,
};
use mog_synth::string_synth::{synthesize_string_program, StrSynthExample};

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

/// Parse a Problem from JSON input (for --problem-json CLI).
fn parse_problem_json(json_str: &str) -> Result<Problem, String> {
    let v: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("invalid JSON: {e}"))?;

    let name = v["name"].as_str().unwrap_or("unknown").to_string();
    // Leak strings to get 'static lifetime (Problem uses &'static str for benchmark compat)
    let signature: &'static str = Box::leak(
        v["signature"]
            .as_str()
            .unwrap_or("fn unknown(a: i64) -> i64")
            .to_string()
            .into_boxed_str(),
    );

    fn parse_example(v: &serde_json::Value) -> Result<Example, String> {
        let inputs_arr = v["inputs"].as_array().ok_or("missing inputs")?;
        let mut inputs = Vec::new();
        for inp in inputs_arr {
            if let Some(n) = inp.as_i64() {
                inputs.push(Value::Int(n));
            } else if inp.is_f64() {
                // JSON numbers written with a decimal point are f64 (`as_i64`
                // returns None for them); integers keep their `Int` lane. Stored
                // as IEEE bits (Value keeps Eq/Ord).
                inputs.push(Value::Float(inp.as_f64().unwrap().to_bits()));
            } else if let Some(arr) = inp.as_array() {
                let vals: Vec<i64> = arr.iter().filter_map(|x| x.as_i64()).collect();
                inputs.push(Value::Array(vals));
            } else if let Some(s) = inp.as_str() {
                inputs.push(Value::Str(s.to_string()));
            } else {
                return Err(format!("unsupported input type: {inp}"));
            }
        }
        // Expected output may be int, string, or array — all first-class now.
        let exp = &v["expected"];
        let expected = if let Some(n) = exp.as_i64() {
            Value::Int(n)
        } else if exp.is_f64() {
            Value::Float(exp.as_f64().unwrap().to_bits())
        } else if let Some(s) = exp.as_str() {
            Value::Str(s.to_string())
        } else if let Some(arr) = exp.as_array() {
            Value::Array(arr.iter().filter_map(|x| x.as_i64()).collect())
        } else {
            return Err("missing/unsupported expected".to_string());
        };
        Ok(Example { inputs, expected })
    }

    let examples: Vec<Example> = v["examples"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|e| parse_example(e).ok())
        .collect();

    let holdouts: Vec<Example> = v["holdouts"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|e| parse_example(e).ok())
        .collect();

    if examples.is_empty() {
        return Err("no valid examples".to_string());
    }

    Ok(Problem {
        name,
        category: "external",
        description: "",
        signature,
        examples,
        holdouts,
        reference_code: "",
    })
}

/// If the problem-json describes a string-output problem (`-> string`), route it
/// to the string-program path: the fast morphology specialist for single-arg
/// suffix transduction, then the general enumerative string synthesizer. Returns
/// None when the problem is not string-output, so the normal i64 pipeline runs.
fn try_string_program(json_str: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let signature = v["signature"].as_str().unwrap_or("");
    if !signature.replace(' ', "").contains("->string") {
        return None;
    }
    let fn_name = signature
        .split_once("fn ")
        .and_then(|(_, rest)| rest.split_once('('))
        .map(|(name, _)| name.trim())
        .unwrap_or("transform")
        .to_string();
    // Parameter names from the signature (all string args).
    let params: Vec<String> = signature
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(p, _)| p)
        .unwrap_or("")
        .split(',')
        .filter_map(|p| p.split(':').next().map(|n| n.trim().to_string()))
        .filter(|n| !n.is_empty())
        .collect();

    fn rows_of(node: &serde_json::Value) -> Vec<(Vec<String>, String)> {
        node.as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| {
                        let inputs: Vec<String> = row["inputs"]
                            .as_array()?
                            .iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect();
                        let expected = row["expected"].as_str()?.to_string();
                        if inputs.is_empty() {
                            return None;
                        }
                        Some((inputs, expected))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    let train = rows_of(&v["examples"]);
    let holdouts = rows_of(&v["holdouts"]);
    let single_arg = train.iter().all(|(i, _)| i.len() == 1);
    eprintln!(
        "[string-program] {} train, {} holdout ({} args -> string)",
        train.len(),
        holdouts.len(),
        params.len().max(1)
    );

    // 1. Fast morphology specialist (single-arg suffix transduction).
    if single_arg {
        let to_morph = |rs: &[(Vec<String>, String)]| {
            rs.iter()
                .map(|(i, e)| StrExample {
                    input: i[0].clone(),
                    expected: e.clone(),
                })
                .collect::<Vec<_>>()
        };
        let m = solve_morph_transduction(&fn_name, &to_morph(&train), &to_morph(&holdouts));
        if m.success {
            return Some(result_json(m.success, m.code, m.method, m.error));
        }
    }

    // 2. General enumerative string synthesizer; verify on train + holdouts.
    let to_synth = |rs: &[(Vec<String>, String)]| {
        rs.iter()
            .map(|(i, e)| StrSynthExample {
                inputs: i.clone(),
                expected: e.clone(),
            })
            .collect::<Vec<_>>()
    };
    let all: Vec<StrSynthExample> = to_synth(&train)
        .into_iter()
        .chain(to_synth(&holdouts))
        .collect();
    let pnames = if params.is_empty() {
        vec!["s".to_string()]
    } else {
        params
    };
    let r = synthesize_string_program(&pnames, &all);
    // Rename the emitted `transform` to the requested function name.
    let code = r
        .code
        .replacen("fn transform(", &format!("fn {fn_name}("), 1);
    Some(result_json(r.success, code, r.method, r.error))
}

fn result_json(success: bool, code: String, method: String, error: Option<String>) -> String {
    serde_json::json!({
        "success": success,
        "code": if success { serde_json::Value::String(code) } else { serde_json::Value::Null },
        "method": method,
        "error": error,
    })
    .to_string()
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

    // --dream: run background enumeration to discover useful sub-expressions
    if has_flag(&args, "--dream") {
        let budget_ms: u64 = arg_value(&args, "--dream")
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        eprintln!("[dream] running dream mode with {budget_ms}ms budget...");
        let library = enumerative::dream(budget_ms);
        match library.save() {
            Ok(()) => eprintln!(
                "[dream] saved {} components to ~/.mog_synth_library.json",
                library.len()
            ),
            Err(e) => eprintln!("[dream] failed to save: {e}"),
        }
        eprintln!("[dream] complete");
        return;
    }

    // --transpile <python|rust|typescript>: read Mog source from stdin and
    // print the transpiled program on stdout. Additive flag for the
    // synthesis API server (ncpu/synthesis_api); touches nothing else.
    if let Some(target) = arg_value(&args, "--transpile") {
        use std::io::Read;
        let mut mog = String::new();
        std::io::stdin()
            .read_to_string(&mut mog)
            .unwrap_or_default();
        let out = match target.as_str() {
            "python" => mog_synth::mog_transpile::to_python(&mog),
            "rust" => mog_synth::mog_transpile::to_rust(&mog),
            "typescript" => mog_synth::mog_transpile::to_typescript(&mog),
            other => {
                eprintln!("unknown transpile target: {other} (expected python|rust|typescript)");
                std::process::exit(1);
            }
        };
        println!("{out}");
        return;
    }

    // --problem-json: accept arbitrary problem from stdin or argument
    if has_flag(&args, "--problem-json") {
        let json_str = if arg_value(&args, "--problem-json").as_deref() == Some("-") {
            // Read from stdin
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .unwrap_or_default();
            buf
        } else if let Some(val) = arg_value(&args, "--problem-json") {
            if val.starts_with('{') {
                val
            } else {
                fs::read_to_string(&val).unwrap_or_default()
            }
        } else {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .unwrap_or_default();
            buf
        };

        // String-output problems (generative morphology) take an additive path:
        // signature returns a string and `expected` fields are strings.
        if let Some(output) = try_string_program(&json_str) {
            println!("{output}");
            return;
        }

        // Parse JSON into a Problem
        eprintln!(
            "[problem-json] parsed {} bytes, calling solve_problem...",
            json_str.len()
        );
        match parse_problem_json(&json_str) {
            Ok(problem) => {
                eprintln!(
                    "[problem-json] problem: {} ({} examples, {} args)",
                    problem.name,
                    problem.examples.len(),
                    problem
                        .examples
                        .first()
                        .map(|e| e.inputs.len())
                        .unwrap_or(0)
                );
                let result = solve_problem(&problem);
                // Output as JSON
                let output = serde_json::json!({
                    "success": result.success,
                    "code": result.code,
                    "method": result.method,
                    "error": result.error,
                });
                println!("{output}");
            }
            Err(e) => {
                let output = serde_json::json!({
                    "success": false,
                    "code": null,
                    "method": "error",
                    "error": format!("Failed to parse problem JSON: {e}"),
                });
                println!("{output}");
            }
        }
        return;
    }

    // --per-problem-json: run every benchmark problem and emit one JSON row per problem,
    // followed by a single summary object. Deterministic, suitable for paper tables.
    if has_flag(&args, "--per-problem-json") {
        let variants: usize = arg_value(&args, "--variants")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let problems = get_benchmark(variants);
        let total = problems.len();
        let mut solved = 0usize;
        let mut by_method: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut failures: Vec<String> = Vec::new();
        let started = std::time::Instant::now();
        for problem in &problems {
            let t0 = std::time::Instant::now();
            let result = if use_differentiable_only {
                solve_problem_differentiable_only(problem)
            } else if prefer_differentiable {
                solve_problem_prefer_differentiable(problem)
            } else if use_legacy_only {
                solve_problem_legacy_only(problem)
            } else if use_legacy_fallback {
                solve_problem_with_legacy_fallback(problem)
            } else {
                solve_problem(problem)
            };
            let elapsed = t0.elapsed().as_secs_f64();
            if result.success {
                solved += 1;
                *by_method.entry(result.method.clone()).or_insert(0) += 1;
            } else {
                failures.push(problem.name.clone());
            }
            let row = serde_json::json!({
                "name": problem.name,
                "category": problem.category,
                "success": result.success,
                "method": result.method,
                "seconds": (elapsed * 10_000.0).round() / 10_000.0,
                "error": result.error,
            });
            println!("{row}");
        }
        let wall = started.elapsed().as_secs_f64();
        let summary = serde_json::json!({
            "summary": true,
            "variants_per_factory": variants,
            "problem_count": total,
            "passed": solved,
            "coverage": if total == 0 { 0.0 } else { solved as f64 / total as f64 },
            "wall_seconds": (wall * 1000.0).round() / 1000.0,
            "method_counts": by_method,
            "failures": failures,
        });
        println!("{summary}");
        let (n_cached, was_dirty) = mog_synth::solved_cache::flush();
        if was_dirty {
            eprintln!("[solved-cache] persisted {n_cached} entries");
        }
        let (n_biases, biases_dirty) = mog_synth::learned_biases::flush();
        if biases_dirty {
            eprintln!("[learned-biases] persisted {n_biases} biases");
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
    let (n_cached, was_dirty) = mog_synth::solved_cache::flush();
    if was_dirty {
        eprintln!("[solved-cache] persisted {n_cached} entries");
    }
    let (n_biases, biases_dirty) = mog_synth::learned_biases::flush();
    if biases_dirty {
        eprintln!("[learned-biases] persisted {n_biases} biases");
    }
}
