//! Measure pure-emergent synthesis coverage.
//!
//! Runs `strategy::emergent_only_strategies()` (CachedTeachers + PureEmergent)
//! against the benchmark suite and reports per-problem outcomes plus a summary.
//!
//! This is the canonical metric for "how much hand-tuning have we removed?":
//!   - Coverage at any given moment = what the *learned* components solve
//!     with no hand-designed program skeletons, no templates, no reference
//!     code, no enumerative search.
//!   - As cache grows over runs and as learned warm-starts mature, coverage
//!     rises monotonically (in expectation). The curve over time is the
//!     direct measurement of the user's "everything emergent" mandate.
//!
//! Output: line-delimited JSON, one row per problem plus a final `{summary: ...}`
//! object. Use `--out path.jsonl` to capture for downstream analysis.
//!
//! Usage:
//!     cargo run --release --bin emergent_coverage -- \
//!         [--out coverage.jsonl]                     \
//!         [--variants N]                             \
//!         [--limit N]                                \
//!         [--quiet]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use mog_synth::benchmark::get_benchmark;
use mog_synth::strategy::{emergent_only_strategies, run_strategies};

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Config {
    out_path: Option<String>,
    variants: usize,
    limit: Option<usize>,
    quiet: bool,
}

impl Config {
    fn from_args(args: &[String]) -> Self {
        Self {
            out_path: arg_value(args, "--out"),
            variants: arg_value(args, "--variants")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            limit: arg_value(args, "--limit").and_then(|v| v.parse().ok()),
            quiet: has_flag(args, "--quiet"),
        }
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

// ─── JSON emission (no external dependency for log lines) ────────────────────

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn row_json(name: &str, success: bool, method: &str, time_s: f32) -> String {
    format!(
        r#"{{"name":"{}","success":{},"method":"{}","time_s":{:.3}}}"#,
        json_escape(name),
        success,
        json_escape(method),
        time_s,
    )
}

fn summary_json(total: usize, solved: usize, total_s: f32) -> String {
    let pct = if total > 0 {
        100.0 * solved as f32 / total as f32
    } else {
        0.0
    };
    format!(
        r#"{{"summary":{{"total":{},"solved":{},"coverage_pct":{:.1},"total_time_s":{:.1}}}}}"#,
        total, solved, pct, total_s,
    )
}

// ─── Entry ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config::from_args(&args);

    let problems = get_benchmark(cfg.variants);
    let total_targets = cfg.limit.unwrap_or(problems.len());
    let strats = emergent_only_strategies();

    let mut writer: Option<BufWriter<File>> = match &cfg.out_path {
        Some(path) => match File::create(path) {
            Ok(f) => Some(BufWriter::new(f)),
            Err(err) => {
                eprintln!("[coverage] cannot open {path}: {err}");
                std::process::exit(1);
            }
        },
        None => None,
    };

    let mut emit = |line: String| {
        if let Some(w) = writer.as_mut() {
            let _ = w.write_all(line.as_bytes());
            let _ = w.write_all(b"\n");
        }
        if !cfg.quiet {
            println!("{line}");
        }
    };

    let strategy_names: Vec<&str> = strats.iter().map(|s| s.name()).collect();
    eprintln!(
        "[coverage] running {} strategies over {} problems: {:?}",
        strategy_names.len(),
        total_targets,
        strategy_names,
    );

    let t_start = Instant::now();
    let mut solved = 0usize;

    for problem in problems.iter().take(total_targets) {
        let t_problem = Instant::now();
        let result = run_strategies(&strats, problem);
        let dt = t_problem.elapsed().as_secs_f32();
        let (success, method) = match &result {
            Some(r) if r.success => (true, r.method.as_str()),
            _ => (false, ""),
        };
        if success {
            solved += 1;
        }
        emit(row_json(&problem.name, success, method, dt));
    }

    let total_s = t_start.elapsed().as_secs_f32();
    emit(summary_json(total_targets, solved, total_s));

    if let Some(w) = writer.as_mut() {
        if let Err(err) = w.flush() {
            eprintln!("[coverage] flush error: {err}");
        }
    }

    eprintln!(
        "[coverage] done: {}/{} solved in {:.1}s ({:.1}%)",
        solved,
        total_targets,
        total_s,
        if total_targets > 0 {
            100.0 * solved as f32 / total_targets as f32
        } else {
            0.0
        },
    );
}
