//! Demonstrate the cross-run self-improvement loop.
//!
//! Runs the benchmark suite twice through `solver::solve_problem`:
//!   - **Pass 1:** cache cold; every problem is solved from scratch by the
//!     full pipeline. The cache fills as we go.
//!   - **Pass 2:** identical input set, but now `solver` consults the
//!     persistent solved cache (memoized hit) and `CachedTeachers` (cross-
//!     problem transfer via gradient distillation). Most problems should
//!     return immediately.
//!
//! The reported speedup is the concrete measurement of "every successful
//! solve is a building block for the next one." No human curated which
//! prior solves are relevant; the cache + gradient flow do the work.
//!
//! Usage:
//!     cargo run --release --bin self_improve -- \
//!         [--variants N]                        \
//!         [--limit N]                           \
//!         [--cache PATH]                        \
//!         [--quiet]
//!
//! `--cache PATH` overrides `NSYNTH_CACHE_PATH`. Pass an empty string to
//! disable the cache (useful for measuring baseline pass-2 cost).

use std::time::Instant;

use mog_synth::benchmark::{get_benchmark, Problem};
use mog_synth::solver::solve_problem;

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Config {
    variants: usize,
    limit: Option<usize>,
    cache_path: Option<String>,
    quiet: bool,
}

impl Config {
    fn from_args(args: &[String]) -> Self {
        Self {
            variants: arg_value(args, "--variants")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            limit: arg_value(args, "--limit").and_then(|v| v.parse().ok()),
            cache_path: arg_value(args, "--cache"),
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

// ─── Pass execution ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PassStats {
    label: &'static str,
    total: usize,
    solved: usize,
    total_time_s: f32,
    /// Per-problem time and method, in benchmark order.
    rows: Vec<(String, bool, String, f32)>,
}

impl PassStats {
    fn coverage_pct(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            100.0 * self.solved as f32 / self.total as f32
        }
    }

    fn print_summary(&self) {
        eprintln!(
            "[self-improve][{}] {}/{} solved in {:.1}s ({:.1}% coverage)",
            self.label,
            self.solved,
            self.total,
            self.total_time_s,
            self.coverage_pct(),
        );
    }
}

fn run_pass(label: &'static str, problems: &[Problem], cfg: &Config) -> PassStats {
    let mut stats = PassStats {
        label,
        total: problems.len(),
        solved: 0,
        total_time_s: 0.0,
        rows: Vec::with_capacity(problems.len()),
    };
    let t_start = Instant::now();
    for p in problems {
        let t = Instant::now();
        let result = solve_problem(p);
        let dt = t.elapsed().as_secs_f32();
        if result.success {
            stats.solved += 1;
        }
        if !cfg.quiet {
            eprintln!(
                "[self-improve][{}] {:50} {:>22} in {:>6.2}s",
                label,
                p.name,
                if result.success {
                    result.method.as_str()
                } else {
                    "MISS"
                },
                dt,
            );
        }
        stats
            .rows
            .push((p.name.clone(), result.success, result.method.clone(), dt));
    }
    stats.total_time_s = t_start.elapsed().as_secs_f32();
    stats
}

// ─── Diff report ─────────────────────────────────────────────────────────────

fn print_speedup(p1: &PassStats, p2: &PassStats) {
    let speedup = if p2.total_time_s > 0.0 {
        p1.total_time_s / p2.total_time_s
    } else {
        0.0
    };
    eprintln!("─────────────────────────────────────────────────────────────");
    eprintln!(
        "[self-improve] pass1: {:.1}s  →  pass2: {:.1}s  ({}× speedup)",
        p1.total_time_s,
        p2.total_time_s,
        format_speedup(speedup),
    );
    eprintln!(
        "[self-improve] coverage: pass1 {:.1}%  →  pass2 {:.1}%",
        p1.coverage_pct(),
        p2.coverage_pct(),
    );

    let mut method_changes: Vec<(String, String, String)> = Vec::new();
    for ((n1, _, m1, _), (_, _, m2, _)) in p1.rows.iter().zip(p2.rows.iter()) {
        if m1 != m2 {
            method_changes.push((n1.clone(), m1.clone(), m2.clone()));
        }
    }
    if !method_changes.is_empty() {
        eprintln!(
            "[self-improve] {} problems switched method on pass 2 (cache transfer):",
            method_changes.len()
        );
        for (name, before, after) in method_changes.iter().take(20) {
            eprintln!("  {name:50} {before:22} → {after}");
        }
        if method_changes.len() > 20 {
            eprintln!("  ... ({} more)", method_changes.len() - 20);
        }
    }
}

fn format_speedup(s: f32) -> String {
    if s.is_finite() && s > 0.0 {
        format!("{s:.1}")
    } else {
        "∞".to_string()
    }
}

// ─── Entry ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config::from_args(&args);

    if let Some(path) = &cfg.cache_path {
        std::env::set_var("NSYNTH_CACHE_PATH", path);
        eprintln!("[self-improve] NSYNTH_CACHE_PATH={path}");
    } else if std::env::var("NSYNTH_CACHE_PATH").is_err() {
        // Default: a per-run cache so the demo is reproducible without
        // clobbering whatever the user has at the canonical location.
        let default = "/tmp/nsynth_self_improve_cache.tsv";
        std::env::set_var("NSYNTH_CACHE_PATH", default);
        // Wipe so pass-1 truly starts cold.
        let _ = std::fs::remove_file(default);
        eprintln!("[self-improve] NSYNTH_CACHE_PATH={default} (wiped for cold start)");
    }

    let mut problems = get_benchmark(cfg.variants);
    if let Some(n) = cfg.limit {
        problems.truncate(n);
    }
    eprintln!(
        "[self-improve] running {} problems × 2 passes",
        problems.len()
    );

    let pass1 = run_pass("pass1", &problems, &cfg);
    pass1.print_summary();

    // Make sure the cache is durable between passes.
    let (entries, dirty) = mog_synth::solved_cache::flush();
    eprintln!(
        "[self-improve] cache flushed: {} entries (dirty={})",
        entries, dirty,
    );

    let pass2 = run_pass("pass2", &problems, &cfg);
    pass2.print_summary();

    print_speedup(&pass1, &pass2);
}
