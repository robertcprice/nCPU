//! JSONL → synthesis harvester. Upstream handoff point for external I/O
//! sources (compiled binaries via the nCPU ARM64 emulator, fuzz captures,
//! trace-mined production logs, ...). Each input line is a self-contained
//! problem specification; the harvester runs the learned universal
//! synthesiser on each and emits JSONL meta-records suitable for training
//! the downstream meta-learner.
//!
//! Input schema (one JSON object per line):
//!   {
//!     "name": "fibonacci_v0",
//!     "signature": "fn fibonacci(n: i64) -> i64",
//!     "examples": [
//!       {"inputs": [{"Int": 0}], "expected": 0},
//!       {"inputs": [{"Int": 1}], "expected": 1},
//!       {"inputs": [{"Int": 5}], "expected": 5}
//!     ]
//!   }
//!
//! Only scalar-int inputs are accepted today (matches the universal
//! synthesiser's input requirement). Non-scalar problems are skipped with a
//! diagnostic line on stderr.
//!
//! Usage:
//!     cargo run --release --bin jsonl_harvest -- \
//!         --in problems.jsonl                    \
//!         --out corpus.jsonl                     \
//!         [--max-steps N]                        \
//!         [--verbose]

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use serde::Deserialize;

use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::synthesis::{record_from_synthesis, synthesize_universal_and_collect};

// ─── JSONL schema ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct InputExample {
    inputs: Vec<serde_json::Value>,
    expected: i64,
}

#[derive(Deserialize)]
struct InputProblem {
    name: String,
    signature: String,
    examples: Vec<InputExample>,
}

fn value_from_json(v: &serde_json::Value) -> Option<Value> {
    // Accept two shapes: raw `42` (interpreted as Int) and explicit
    // `{"Int": 42}` (matching the `Value` enum's serde tagging). The latter
    // is future-proof against Str / Array / Pair once those paths are
    // supported by the universal synthesiser.
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(obj) = v.as_object() {
        if let Some(n) = obj.get("Int").and_then(|x| x.as_i64()) {
            return Some(Value::Int(n));
        }
    }
    None
}

fn to_problem(input: InputProblem) -> Option<Problem> {
    let mut examples = Vec::with_capacity(input.examples.len());
    for (idx, ex) in input.examples.into_iter().enumerate() {
        let mut ins = Vec::with_capacity(ex.inputs.len());
        for v in ex.inputs {
            match value_from_json(&v) {
                Some(val) => ins.push(val),
                None => {
                    eprintln!(
                        "[jsonl_harvest] {} example {}: non-scalar input — skipping problem",
                        input.name, idx
                    );
                    return None;
                }
            }
        }
        // Scalar-only check: universal synthesiser cannot handle Str/Array/Pair.
        if !ins.iter().all(|v| matches!(v, Value::Int(_))) {
            eprintln!(
                "[jsonl_harvest] {} example {}: non-scalar Int values — skipping",
                input.name, idx
            );
            return None;
        }
        examples.push(Example {
            inputs: ins,
            expected: Value::Int(ex.expected),
        });
    }

    // The benchmark's Problem struct has `&'static str` fields because the
    // baked-in suite is all literals. Box::leak gives us equivalent lifetime
    // for runtime-loaded problems at the cost of a tiny allocation per
    // harvest — fine since this binary runs once per batch.
    let signature: &'static str = Box::leak(input.signature.into_boxed_str());
    Some(Problem {
        name: input.name,
        category: "jsonl",
        description: "",
        signature,
        examples,
        holdouts: vec![],
        reference_code: "",
    })
}

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

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Config {
    in_path: String,
    out_path: String,
    max_steps: usize,
    verbose: bool,
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config {
        in_path: arg_value(&args, "--in").unwrap_or_else(|| "problems.jsonl".to_string()),
        out_path: arg_value(&args, "--out").unwrap_or_else(|| "corpus.jsonl".to_string()),
        max_steps: arg_value(&args, "--max-steps")
            .and_then(|v| v.parse().ok())
            .unwrap_or(400),
        verbose: has_flag(&args, "--verbose"),
    };

    let in_file = match File::open(&cfg.in_path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[jsonl_harvest] cannot open {}: {err}", cfg.in_path);
            std::process::exit(1);
        }
    };
    let reader = BufReader::new(in_file);

    let out_file = match File::create(&cfg.out_path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[jsonl_harvest] cannot open {}: {err}", cfg.out_path);
            std::process::exit(1);
        }
    };
    let mut writer = BufWriter::new(out_file);

    let t_start = Instant::now();
    let mut attempted = 0usize;
    let mut solved = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(err) => {
                eprintln!("[jsonl_harvest] read error at line {line_no}: {err}");
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let parsed: InputProblem = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[jsonl_harvest] parse error at line {line_no}: {err}");
                continue;
            }
        };

        let Some(problem) = to_problem(parsed) else {
            continue;
        };
        let n_args = problem
            .examples
            .first()
            .map(|e| e.inputs.len())
            .unwrap_or(0);
        attempted += 1;

        let t0 = Instant::now();
        let Some((result, params)) = synthesize_universal_and_collect(&problem, cfg.max_steps)
        else {
            if cfg.verbose {
                eprintln!(
                    "[jsonl_harvest] ✗ {:40} no synthesis result ({:.1}s)",
                    problem.name,
                    t0.elapsed().as_secs_f32()
                );
            }
            continue;
        };
        if !result.success {
            if cfg.verbose {
                eprintln!(
                    "[jsonl_harvest] ✗ {:40} synthesis miss ({:.1}s)",
                    problem.name,
                    t0.elapsed().as_secs_f32()
                );
            }
            continue;
        }

        let record = record_from_synthesis(
            problem.function_name(),
            n_args,
            params,
            io_examples_from(&problem),
            "jsonl_harvest",
        );
        match serde_json::to_string(&record) {
            Ok(row) => {
                let _ = writer.write_all(row.as_bytes());
                let _ = writer.write_all(b"\n");
                solved += 1;
                if cfg.verbose {
                    eprintln!(
                        "[jsonl_harvest] ✓ {:40} via {:22} in {:.1}s",
                        problem.name,
                        result.method,
                        t0.elapsed().as_secs_f32()
                    );
                }
            }
            Err(err) => {
                eprintln!(
                    "[jsonl_harvest] serialise error for {}: {err}",
                    problem.name
                );
            }
        }
    }

    if let Err(err) = writer.flush() {
        eprintln!("[jsonl_harvest] flush error: {err}");
    }
    eprintln!(
        "[jsonl_harvest] done: {}/{} solved in {:.1}s",
        solved,
        attempted,
        t_start.elapsed().as_secs_f32()
    );
}
