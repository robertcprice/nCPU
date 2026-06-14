//! Benchmark warm-start vs cold-start synthesis for SoftUniversalProgram.
//!
//! Usage:
//!   cargo run --release --bin warm_start_bench -- \
//!       --descriptions data/benchmark_descs.jsonl \
//!       [--warm-steps 200]
//!
//! Each JSONL line must be: { "problem_name": "...", "description": {...} }
//! where "description" matches `UniversalProgramDescription` (with n_args, slots, etc.)
//!
//! Reports per-problem: warm solve? | warm steps | cold steps | speedup
//!
//! Also supports --predict-py: calls scripts/infer_metalearner.py to predict
//! descriptions from benchmark I/O examples, then benchmarks those predictions.

use std::io::{BufRead, Write};

use mog_synth::benchmark::get_benchmark;
use mog_synth::synthesis::{synthesize_universal_warm_start, UniversalProgramDescription};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let warm_steps: usize = arg_value(&args, "--warm-steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let n_args_filter: Option<usize> = arg_value(&args, "--n-args").and_then(|s| s.parse().ok());

    let warm_only = has_flag(&args, "--warm-only");

    // ── Mode: predict via Python meta-learner, then benchmark ─────────────────
    if has_flag(&args, "--predict-py") {
        let model =
            arg_value(&args, "--model").unwrap_or_else(|| "models/metalearner_1arg.pt".to_string());
        run_predict_mode(&model, warm_steps, n_args_filter, warm_only);
        return;
    }

    // ── Mode: load descriptions from JSONL file ────────────────────────────────
    let desc_path = arg_value(&args, "--descriptions").unwrap_or_else(|| {
        eprintln!("Usage: warm_start_bench --descriptions FILE [--warm-steps N]");
        std::process::exit(1);
    });

    let file = std::fs::File::open(&desc_path).unwrap_or_else(|e| {
        eprintln!("Cannot open {desc_path}: {e}");
        std::process::exit(1);
    });

    #[derive(serde::Deserialize)]
    struct DescRecord {
        problem_name: String,
        description: UniversalProgramDescription,
    }

    let records: Vec<DescRecord> = std::io::BufReader::new(file)
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();

    if records.is_empty() {
        eprintln!("No records loaded from {desc_path}");
        std::process::exit(1);
    }

    let all_problems = get_benchmark(1);

    eprintln!(
        "Benchmarking {} warm-start descriptions (warm_steps={warm_steps})...",
        records.len()
    );
    let mut total_warm = 0usize;
    let mut warm_wins = 0usize;
    let mut total_warm_steps = 0usize;

    for rec in &records {
        let problem = match all_problems.iter().find(|p| p.name == rec.problem_name) {
            Some(p) => p,
            None => {
                eprintln!(
                    "  [SKIP] problem '{}' not found in benchmark",
                    rec.problem_name
                );
                continue;
            }
        };
        total_warm += 1;

        match synthesize_universal_warm_start(problem, &rec.description, warm_steps, 3) {
            Some((result, steps, warm_ok)) => {
                let warm_marker = if warm_ok { "WARM" } else { "COLD" };
                eprintln!(
                    "  [{}] {} → SOLVED in {} steps ({})",
                    warm_marker, rec.problem_name, steps, result.method
                );
                if warm_ok {
                    warm_wins += 1;
                    total_warm_steps += steps;
                }
            }
            None => {
                eprintln!("  [FAIL] {} → NOT SOLVED", rec.problem_name);
            }
        }
    }

    let warm_rate = if total_warm > 0 {
        warm_wins * 100 / total_warm
    } else {
        0
    };
    eprintln!("\nSummary:");
    eprintln!("  Problems tested:  {total_warm}");
    eprintln!("  Warm-start wins:  {warm_wins} ({warm_rate}%)");
    eprintln!(
        "  Avg warm steps:   {}",
        if warm_wins > 0 {
            total_warm_steps / warm_wins
        } else {
            0
        }
    );
}

/// Run Python meta-learner (single batch call) on all benchmark problems, then test warm-start.
fn run_predict_mode(
    model_path: &str,
    warm_steps: usize,
    n_args_filter: Option<usize>,
    warm_only: bool,
) {
    let n_args = n_args_filter.unwrap_or(1);
    let problems = get_benchmark(n_args);
    let total = problems.len();

    eprintln!("Predicting descriptions for {total} problems (n_args={n_args}, warm_steps={warm_steps})...");
    eprintln!("Model: {model_path}");

    // ── Build batch input JSONL ───────────────────────────────────────────────
    let batch_in = "/tmp/warm_bench_batch_in.jsonl";
    let batch_out = "/tmp/warm_bench_batch_out.jsonl";

    {
        let f = std::fs::File::create(batch_in).unwrap_or_else(|e| {
            eprintln!("Cannot create {batch_in}: {e}");
            std::process::exit(1);
        });
        let mut w = std::io::BufWriter::new(f);
        for problem in &problems {
            if let Some(io_rows) = build_io_rows(problem, n_args) {
                let rec = serde_json::json!({"name": problem.name, "io": io_rows});
                writeln!(w, "{rec}").unwrap();
            }
        }
    }

    // ── Single Python batch inference call ────────────────────────────────────
    eprintln!("  Calling Python batch inference...");
    let py_status = std::process::Command::new("python3")
        .args([
            "scripts/train_metalearner.py",
            "--infer",
            model_path,
            "--batch-in",
            batch_in,
            "--batch-out",
            batch_out,
        ])
        .status();

    match py_status {
        Ok(s) if !s.success() => {
            eprintln!("Python inference failed with status {s}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to spawn Python: {e}");
            std::process::exit(1);
        }
        _ => {}
    }

    // ── Load predicted descriptions ───────────────────────────────────────────
    #[derive(serde::Deserialize)]
    struct PredRec {
        name: String,
        description: UniversalProgramDescription,
    }
    let desc_file = std::fs::File::open(batch_out).unwrap_or_else(|e| {
        eprintln!("Cannot open {batch_out}: {e}");
        std::process::exit(1);
    });
    let desc_map: std::collections::HashMap<String, UniversalProgramDescription> =
        std::io::BufReader::new(desc_file)
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<PredRec>(&l).ok())
            .map(|r| (r.name, r.description))
            .collect();

    eprintln!(
        "  Got {} predictions. Running synthesis...\n",
        desc_map.len()
    );

    // ── Benchmark warm-start vs cold ─────────────────────────────────────────
    let mut warm_wins = 0usize;
    let mut total_tested = 0usize;

    for (i, problem) in problems.iter().enumerate() {
        let desc = match desc_map.get(&problem.name) {
            Some(d) => d,
            None => continue,
        };
        total_tested += 1;

        let cold_restarts = if warm_only { 0 } else { 3 };
        match synthesize_universal_warm_start(problem, desc, warm_steps, cold_restarts) {
            Some((result, steps, warm_ok)) => {
                let tag = if warm_ok { "WARM" } else { "COLD" };
                if warm_ok {
                    warm_wins += 1;
                }
                eprintln!(
                    "  [{}/{}] {} → {tag} {steps} steps — {}",
                    i + 1,
                    total,
                    problem.name,
                    result.method
                );
            }
            None => {
                let tag = if warm_only { "WRONG" } else { "FAIL" };
                eprintln!("  [{}/{}] {} → {tag}", i + 1, total, problem.name);
            }
        }
    }

    let pct = if total_tested > 0 {
        warm_wins as f64 / total_tested as f64 * 100.0
    } else {
        0.0
    };
    eprintln!("\n=== Warm-start benchmark ===");
    eprintln!("  Tested:       {total_tested}");
    eprintln!("  Warm correct: {warm_wins} / {total_tested}  ({pct:.1}%)");
    if !warm_only {
        eprintln!("  (Use --warm-only for fast exact-match benchmark)");
    }
}

fn build_io_rows(problem: &mog_synth::benchmark::Problem, n_args: usize) -> Option<Vec<Vec<i64>>> {
    use mog_synth::benchmark::Value;
    let mut rows = vec![];
    for ex in &problem.examples {
        if ex.inputs.len() != n_args {
            return None;
        }
        let mut row: Vec<i64> = ex
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
        if row.len() != n_args {
            return None;
        }
        row.push(ex.expected_int());
        rows.push(row);
    }
    Some(rows)
}
