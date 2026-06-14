//! Corpus harvester — emergent meta-learner training data pipeline.
//!
//! For every benchmark problem, run the fully-learned universal synthesiser and
//! emit a `MetaRecord` containing:
//!   - The I/O examples the synthesiser had to fit.
//!   - The `UniversalProgramDescription` it *discovered* via gradient descent
//!     (argmax-discretised from learned soft parameters).
//!
//! Nothing is hand-coded. Description, program structure, and training signal
//! all come from the I/O examples and the gradient flow. Output is line-delimited
//! JSON suitable for downstream meta-learner training.
//!
//! Usage:
//!     cargo run --release --bin corpus_harvest -- \
//!         --out corpus.jsonl            \
//!         [--variants N]                \
//!         [--max-steps N]               \
//!         [--limit N]                   \
//!         [--verbose]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use mog_synth::benchmark::{get_benchmark, Problem, Value};
use mog_synth::synthesis::{record_from_synthesis, synthesize_universal_and_collect};

// ─── CLI parsing ─────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Config {
    out_path: String,
    variants: usize,
    max_steps: usize,
    limit: Option<usize>,
    verbose: bool,
}

impl Config {
    fn from_args(args: &[String]) -> Self {
        Self {
            out_path: arg_value(args, "--out").unwrap_or_else(|| "corpus.jsonl".to_string()),
            variants: arg_value(args, "--variants")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            max_steps: arg_value(args, "--max-steps")
                .and_then(|v| v.parse().ok())
                .unwrap_or(400),
            limit: arg_value(args, "--limit").and_then(|v| v.parse().ok()),
            verbose: has_flag(args, "--verbose"),
        }
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

// ─── Harvesting ──────────────────────────────────────────────────────────────

/// Keep only scalar-input problems — cold-start universal synthesis requires
/// all-`Value::Int` inputs today. Array problems are skipped here; they need a
/// different learned synthesiser which is future work.
fn is_scalar_only(problem: &Problem) -> bool {
    problem
        .examples
        .iter()
        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))))
}

/// Convert a problem's `Example` list into the `(Vec<i64>, i64)` shape stored
/// inside a `MetaRecord`. Assumes scalar-only inputs (see `is_scalar_only`).
fn io_examples_from(problem: &Problem) -> Vec<(Vec<i64>, i64)> {
    problem
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
            (inputs, ex.expected_int())
        })
        .collect()
}

/// Run universal synthesis, and on success emit a JSONL line to `writer`.
/// Returns `true` on success, `false` otherwise.
fn harvest_one<W: Write>(problem: &Problem, cfg: &Config, writer: &mut W) -> bool {
    if !is_scalar_only(problem) {
        return false;
    }
    let n_args = match problem.examples.first() {
        Some(ex) => ex.inputs.len(),
        None => return false,
    };

    let t0 = Instant::now();
    let Some((result, params)) = synthesize_universal_and_collect(problem, cfg.max_steps) else {
        return false;
    };
    if !result.success {
        return false;
    }

    let record = record_from_synthesis(
        problem.function_name(),
        n_args,
        params,
        io_examples_from(problem),
        "harvested",
    );

    match serde_json::to_string(&record) {
        Ok(line) => {
            if writer.write_all(line.as_bytes()).is_err() || writer.write_all(b"\n").is_err() {
                eprintln!("[harvest] write error for {}", problem.name);
                return false;
            }
        }
        Err(err) => {
            eprintln!("[harvest] serialise error for {}: {err}", problem.name);
            return false;
        }
    }

    if cfg.verbose {
        eprintln!(
            "[harvest] ✓ {:40} via {:22} in {:.1}s",
            problem.name,
            result.method,
            t0.elapsed().as_secs_f32()
        );
    }
    true
}

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config::from_args(&args);

    let problems = get_benchmark(cfg.variants);
    let total_targets = cfg.limit.unwrap_or(problems.len());
    eprintln!(
        "[harvest] writing to {} — {} problems (variants={}, max_steps={}, limit={})",
        cfg.out_path,
        total_targets,
        cfg.variants,
        cfg.max_steps,
        cfg.limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "none".to_string()),
    );

    let file = match File::create(&cfg.out_path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[harvest] cannot open {}: {err}", cfg.out_path);
            std::process::exit(1);
        }
    };
    let mut writer = BufWriter::new(file);

    let t_start = Instant::now();
    let mut attempted = 0usize;
    let mut solved = 0usize;

    for problem in problems.iter().take(total_targets) {
        attempted += 1;
        if harvest_one(problem, &cfg, &mut writer) {
            solved += 1;
        }
    }

    if let Err(err) = writer.flush() {
        eprintln!("[harvest] flush error: {err}");
    }

    eprintln!(
        "[harvest] done: {}/{} solved in {:.1}s",
        solved,
        attempted,
        t_start.elapsed().as_secs_f32()
    );
}
