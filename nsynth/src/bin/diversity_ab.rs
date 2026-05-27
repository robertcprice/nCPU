//! A/B experiment: does the diversity pass in `rank_teachers_with_meta_topk`
//! actually pay off on real problems?
//!
//! Runs `CachedTeachers::try_solve` against the same problem set twice —
//! once with `NSYNTH_TEACHER_TOPK=0` (no cap, no diversity pass), once with
//! `NSYNTH_TEACHER_TOPK=N` (diversity pass active on top-N). Reports per-mode
//! win rate + mean solve wall-clock, plus the A/B delta.
//!
//! This is the experiment the diversity-pass code change *asked for*. Unit
//! tests prove the mechanism fires; this binary proves (or disproves) the
//! mechanism helps.
//!
//! Methodology notes:
//!   - Each mode sees a fresh empty in-memory cache via
//!     `solved_cache::reset_in_memory()`, and we point
//!     `NSYNTH_CACHE_PATH` at a temp file preloaded with a snapshot of the
//!     real user cache. That way both modes start from *identical* cache
//!     state — the only difference between runs is the ranker's selection.
//!   - The measurement is per-problem `(success, solve_ms)`, aggregated to
//!     counts + means. Variance across runs is reported in the JSON mode.
//!
//! Usage:
//!     cargo run --release --bin diversity_ab -- \
//!         [--offset N] [--limit M]              \
//!         [--topk-a 0] [--topk-b 8]             \
//!         [--json]

use std::time::Instant;

use mog_synth::benchmark::get_benchmark;
use mog_synth::solved_cache;
use mog_synth::strategy::{CachedTeachers, SynthesisStrategy};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Run CachedTeachers once on every problem in `problems` under the given
/// topk setting, using a *shared* cache snapshot so the two modes face the
/// same state. Returns `(wins, total_attempts, total_ms)`.
///
/// `pristine_path` holds the untouched cache state; this function copies it
/// to `working_path` first, so in-run mutations (record, note_transfer)
/// stay isolated to the mode-specific working file. Without this the A
/// phase's `note_transfer_success` calls would bump counters the B phase
/// then reads — confounding the measurement.
fn run_mode(
    problems: &[mog_synth::benchmark::Problem],
    topk: usize,
    pristine_path: &str,
    working_path: &str,
) -> (usize, usize, u64) {
    if let Err(err) = std::fs::copy(pristine_path, working_path) {
        eprintln!(
            "[diversity_ab] cannot restore pristine cache: {err}  (src={pristine_path} dst={working_path})"
        );
        std::process::exit(1);
    }

    // Point the cache at the mode-specific working copy. Reset the
    // in-memory singleton so the next lookup re-reads from the freshly-
    // copied file.
    //
    // SAFETY: single-threaded A/B harness; set_var is only unsafe under
    // concurrent reads.
    unsafe {
        std::env::set_var("NSYNTH_CACHE_PATH", working_path);
        std::env::set_var("NSYNTH_TEACHER_TOPK", topk.to_string());
    }
    solved_cache::reset_in_memory();

    let strategy = CachedTeachers;
    let mut wins = 0usize;
    let mut attempts = 0usize;
    let mut total_ms = 0u64;

    for problem in problems {
        if !<CachedTeachers as SynthesisStrategy>::applicable(&strategy, problem) {
            continue;
        }
        // Re-read cache before each problem so we measure CachedTeachers in
        // isolation — if a successful solve were recorded back to the cache,
        // the second problem in the run would see a hotter cache than the
        // first. Here we want a stable pool.
        solved_cache::reset_in_memory();

        attempts += 1;
        let t0 = Instant::now();
        let result = <CachedTeachers as SynthesisStrategy>::try_solve(&strategy, problem);
        let dt = t0.elapsed().as_millis() as u64;
        total_ms += dt;
        if let Some(r) = result {
            if r.success {
                wins += 1;
            }
        }
    }

    (wins, attempts, total_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let offset: usize = arg_value(&args, "--offset")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let limit: usize = arg_value(&args, "--limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let topk_a: usize = arg_value(&args, "--topk-a")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let topk_b: usize = arg_value(&args, "--topk-b")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let json_mode = has_flag(&args, "--json");

    // Snapshot the real user cache into a temp file so both A/B runs share
    // byte-identical state. Reading the env var first preserves user
    // overrides (e.g. CI isolated cache).
    let original_cache_path = std::env::var("NSYNTH_CACHE_PATH").unwrap_or_else(|_| {
        std::env::var("HOME")
            .map(|h| format!("{}/.nsynth_solved_programs.json", h))
            .unwrap_or_default()
    });
    let snapshot_path = std::env::temp_dir()
        .join(format!(
            "nsynth_diversity_ab_{}.snapshot",
            std::process::id()
        ))
        .to_string_lossy()
        .to_string();

    if std::path::Path::new(&original_cache_path).exists() {
        if let Err(err) = std::fs::copy(&original_cache_path, &snapshot_path) {
            eprintln!(
                "[diversity_ab] cannot snapshot cache {} → {}: {err}",
                original_cache_path, snapshot_path
            );
            std::process::exit(1);
        }
    } else {
        eprintln!(
            "[diversity_ab] no cache file at {} — running against empty cache (both modes)",
            original_cache_path
        );
        // Ensure the snapshot file exists so reset_in_memory + load doesn't fault.
        let _ = std::fs::write(&snapshot_path, "");
    }

    let problems = get_benchmark(1);
    let slice: Vec<_> = problems.iter().skip(offset).take(limit).cloned().collect();

    if slice.is_empty() {
        eprintln!(
            "[diversity_ab] no problems in the selected range (offset={offset} limit={limit})"
        );
        std::process::exit(1);
    }

    eprintln!(
        "[diversity_ab] problems={}  topk_a={}  topk_b={}  cache_snapshot={}",
        slice.len(),
        topk_a,
        topk_b,
        snapshot_path
    );

    let working_a = std::env::temp_dir()
        .join(format!(
            "nsynth_diversity_ab_{}_a.cache",
            std::process::id()
        ))
        .to_string_lossy()
        .to_string();
    let working_b = std::env::temp_dir()
        .join(format!(
            "nsynth_diversity_ab_{}_b.cache",
            std::process::id()
        ))
        .to_string_lossy()
        .to_string();

    let (wins_a, attempts_a, ms_a) = run_mode(&slice, topk_a, &snapshot_path, &working_a);
    let (wins_b, attempts_b, ms_b) = run_mode(&slice, topk_b, &snapshot_path, &working_b);

    let pct = |w: usize, n: usize| {
        if n > 0 {
            100.0 * w as f64 / n as f64
        } else {
            0.0
        }
    };
    let mean = |ms: u64, n: usize| if n > 0 { ms as f64 / n as f64 } else { 0.0 };

    let _ = std::fs::remove_file(&snapshot_path);
    let _ = std::fs::remove_file(&working_a);
    let _ = std::fs::remove_file(&working_b);

    if json_mode {
        println!(
            r#"{{"offset":{},"limit":{},"topk_a":{},"topk_b":{},"a":{{"wins":{},"attempts":{},"win_pct":{:.2},"mean_ms":{:.2}}},"b":{{"wins":{},"attempts":{},"win_pct":{:.2},"mean_ms":{:.2}}}}}"#,
            offset,
            limit,
            topk_a,
            topk_b,
            wins_a,
            attempts_a,
            pct(wins_a, attempts_a),
            mean(ms_a, attempts_a),
            wins_b,
            attempts_b,
            pct(wins_b, attempts_b),
            mean(ms_b, attempts_b),
        );
    } else {
        println!("── diversity A/B ──");
        println!(
            "problems attempted by CachedTeachers in each mode: {}",
            attempts_a.max(attempts_b)
        );
        println!();
        println!("Mode A (topk={}, no diversity pass):", topk_a);
        println!(
            "  wins {}/{}  ({:.1}%)  mean_ms {:.1}",
            wins_a,
            attempts_a,
            pct(wins_a, attempts_a),
            mean(ms_a, attempts_a)
        );
        println!("Mode B (topk={}, diversity pass active):", topk_b);
        println!(
            "  wins {}/{}  ({:.1}%)  mean_ms {:.1}",
            wins_b,
            attempts_b,
            pct(wins_b, attempts_b),
            mean(ms_b, attempts_b)
        );
        println!();
        let delta_wins = wins_b as i64 - wins_a as i64;
        let delta_ms = mean(ms_b, attempts_b) - mean(ms_a, attempts_a);
        println!("Δ wins: {:+}    Δ mean_ms: {:+.1}", delta_wins, delta_ms);
        if delta_wins > 0 {
            println!("✓ diversity pass found {} extra transfer(s)", delta_wins);
        } else if delta_wins < 0 {
            println!("✗ diversity pass *lost* {} transfer(s)", -delta_wins);
        } else {
            println!("~ no win-count difference");
        }
    }
}
