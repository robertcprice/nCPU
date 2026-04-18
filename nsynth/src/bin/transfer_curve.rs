//! Transfer-curve harness — measures cross-run learning gains empirically.
//!
//! Runs the full solver (`solver::solve_problem`) over the benchmark suite
//! for N rounds and emits one JSONL row per (round, problem) capturing:
//!   - `round`: 0-indexed round number
//!   - `name`: problem name
//!   - `success`: whether the solver returned a passing program
//!   - `method`: which pipeline stage produced the win (cache_hit,
//!               cached_teachers, enumerative, synth_gradient, ...)
//!   - `time_ms`: wall-clock for this solve
//!   - `cache_size_before`: entries in the persistent solved cache at the
//!                          start of this solve
//!
//! Downstream plots:
//!   - (time_ms vs cache_size_before, coloured by round) — if cached_teachers
//!     is working, per-problem time should stay flat or drop as cache grows,
//!     not rise. That's the "bounded-cost self-improvement" signature.
//!   - rate-of(method == cache_hit) over rounds — exact-match cache effect.
//!   - rate-of(method == cached_teachers) over rounds — cross-problem
//!     transfer: solves that *required* a teacher to find, not just lookup.
//!
//! Two modes:
//!   - `--cumulative` (default): one cache file persists across all rounds,
//!     so later rounds benefit from everything earlier rounds discovered.
//!     This is the "growing knowledge base" curve.
//!   - `--fresh-cache`: each round runs with an empty cache. Baseline for
//!     "what does the solver do without any cross-run help?"
//!
//! Usage:
//!     cargo run --release --bin transfer_curve -- \
//!         --rounds 3                              \
//!         --out curve.jsonl                       \
//!         [--variants N]                          \
//!         [--limit N]                             \
//!         [--fresh-cache]                         \
//!         [--quiet]

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use mog_synth::benchmark::{get_benchmark, Problem};
use mog_synth::solved_cache;
use mog_synth::solver::solve_problem;

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Config {
    out_path: Option<String>,
    rounds: usize,
    variants: usize,
    offset: usize,
    limit: Option<usize>,
    fresh_cache: bool,
    quiet: bool,
}

impl Config {
    fn from_args(args: &[String]) -> Self {
        Self {
            out_path: arg_value(args, "--out"),
            rounds: arg_value(args, "--rounds")
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            variants: arg_value(args, "--variants")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            offset: arg_value(args, "--offset")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            limit: arg_value(args, "--limit").and_then(|v| v.parse().ok()),
            fresh_cache: has_flag(args, "--fresh-cache"),
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

// ─── JSON (no serde dependency here — one-line rows, simple schema) ─────────

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

fn row_json(
    round: usize,
    name: &str,
    success: bool,
    method: &str,
    time_ms: u64,
    cache_size_before: usize,
    via_cached_teachers: bool,
) -> String {
    format!(
        r#"{{"round":{},"name":"{}","success":{},"method":"{}","time_ms":{},"cache_size_before":{},"via_cached_teachers":{}}}"#,
        round,
        json_escape(name),
        success,
        json_escape(method),
        time_ms,
        cache_size_before,
        via_cached_teachers,
    )
}

fn summary_json(
    round: usize,
    total: usize,
    solved: usize,
    total_ms: u64,
    method_counts: &std::collections::BTreeMap<String, usize>,
) -> String {
    let pct = if total > 0 {
        100.0 * solved as f64 / total as f64
    } else {
        0.0
    };
    let mut methods = String::new();
    methods.push('{');
    for (i, (k, v)) in method_counts.iter().enumerate() {
        if i > 0 {
            methods.push(',');
        }
        methods.push_str(&format!(r#""{}":{}"#, json_escape(k), v));
    }
    methods.push('}');
    format!(
        r#"{{"summary_round":{},"total":{},"solved":{},"coverage_pct":{:.1},"total_ms":{},"methods":{}}}"#,
        round, total, solved, pct, total_ms, methods,
    )
}

// ─── Cache-mode helpers ──────────────────────────────────────────────────────

/// In fresh-cache mode we point `NSYNTH_CACHE_PATH` at a freshly-created
/// temp file before each round. The solver reads the env var at first use
/// per-process but caches the loaded state in-memory, so resetting requires
/// both the env var and `solved_cache::reset_for_tests` (which we don't
/// want to expose in production builds). For now: deliberately overwrite
/// the file with empty content between rounds, so the next lazy-reload picks
/// up an empty cache.
///
/// Returns the path used (for cleanup reporting). Creates the parent
/// directory if needed.
fn reset_fresh_cache_file(path: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Truncate-create. If the file existed with stale rows, the next lazy
    // reload will see it as empty.
    let _ = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    Ok(())
}

// ─── Runner ──────────────────────────────────────────────────────────────────

fn run_round<F: FnMut(String)>(
    round: usize,
    problems: &[Problem],
    offset: usize,
    limit: usize,
    emit: &mut F,
) -> (usize, u64) {
    let mut solved = 0usize;
    let mut method_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let t_start = Instant::now();

    for problem in problems.iter().skip(offset).take(limit) {
        let cache_size_before = solved_cache::entry_count();
        let t0 = Instant::now();
        let result = solve_problem(problem);
        let time_ms = t0.elapsed().as_millis() as u64;
        // Method tagged with `cached_teachers:` prefix when Stage 0.5 won —
        // see strategy::CachedTeachers::try_solve. This is the only wire-
        // level signal that a cross-problem transfer actually fired.
        let via_cached_teachers = result.method.starts_with("cached_teachers");
        let method = if result.success {
            solved += 1;
            result.method.clone()
        } else {
            result.method.clone()
        };
        *method_counts.entry(method.clone()).or_insert(0) += 1;
        emit(row_json(
            round,
            &problem.name,
            result.success,
            &method,
            time_ms,
            cache_size_before,
            via_cached_teachers,
        ));
    }

    let total_ms = t_start.elapsed().as_millis() as u64;
    emit(summary_json(round, limit, solved, total_ms, &method_counts));
    (solved, total_ms)
}

// ─── Entry ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config::from_args(&args);

    let problems = get_benchmark(cfg.variants);
    let remaining_after_offset = problems.len().saturating_sub(cfg.offset);
    let total_targets = cfg
        .limit
        .unwrap_or(remaining_after_offset)
        .min(remaining_after_offset);

    // Set up the cache policy. In fresh-cache mode we override the env var
    // to a per-binary-run path so the real user cache is never mutated.
    let cache_override_path: Option<PathBuf> = if cfg.fresh_cache {
        let p = std::env::temp_dir().join(format!(
            "nsynth_transfer_curve_{}.cache",
            std::process::id()
        ));
        unsafe { std::env::set_var("NSYNTH_CACHE_PATH", p.to_string_lossy().to_string()) };
        Some(p)
    } else {
        None
    };

    let mut writer: Option<BufWriter<File>> = match &cfg.out_path {
        Some(path) => match File::create(path) {
            Ok(f) => Some(BufWriter::new(f)),
            Err(err) => {
                eprintln!("[curve] cannot open {path}: {err}");
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

    let mode_label = if cfg.fresh_cache {
        "fresh-cache-per-round"
    } else {
        "cumulative"
    };
    eprintln!(
        "[curve] mode={mode_label} rounds={} offset={} problems={} variants={}",
        cfg.rounds, cfg.offset, total_targets, cfg.variants,
    );

    let t_start = Instant::now();
    let mut round_solves: Vec<usize> = Vec::with_capacity(cfg.rounds);

    for round in 0..cfg.rounds {
        if cfg.fresh_cache {
            if let Some(p) = cache_override_path.as_ref() {
                if let Err(err) = reset_fresh_cache_file(p) {
                    eprintln!("[curve] cannot reset cache file {}: {err}", p.display());
                }
            }
            // Drop the in-memory singleton so the next lookup lazily re-reads
            // the (now empty) file. Without this, the in-memory BTreeMap
            // would still hold prior-round solutions and Stage-0 would keep
            // firing — defeating the purpose of fresh-cache mode.
            solved_cache::reset_in_memory();
        }
        let (solved, _ms) = run_round(round, &problems, cfg.offset, total_targets, &mut emit);
        round_solves.push(solved);
    }

    let total_s = t_start.elapsed().as_secs_f32();

    if let Some(w) = writer.as_mut() {
        if let Err(err) = w.flush() {
            eprintln!("[curve] flush error: {err}");
        }
    }

    // Final human-readable summary to stderr.
    eprintln!("[curve] done in {:.1}s", total_s);
    for (round, solved) in round_solves.iter().enumerate() {
        let pct = if total_targets > 0 {
            100.0 * (*solved as f32) / (total_targets as f32)
        } else {
            0.0
        };
        eprintln!(
            "[curve] round {round}: {solved}/{total_targets} = {:.1}%",
            pct
        );
    }

    // Best-effort cleanup of the temp cache file if we created one.
    if let Some(p) = cache_override_path.as_ref() {
        let _ = std::fs::remove_file(p);
    }
}
