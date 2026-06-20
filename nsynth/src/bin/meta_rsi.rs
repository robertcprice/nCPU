//! `meta_rsi` — operator-invoked driver for Phase 5.1 bounded recursive
//! self-improvement.
//!
//! Tunes ONLY the meta-learner ranker weights against the full benchmark suite,
//! accepting a candidate only on a strict-superset coverage gain. Dry-run by
//! default; pass `--commit` to persist a winner to production weights.
//!
//! Usage:
//!   meta_rsi [--max-iters N] [--max-no-improve N] [--deadline-secs S]
//!            [--sigma F] [--lr F] [--seed N] [--audit-log PATH] [--commit]
//!
//! Safety: see `mog_synth::meta::recursive`. A crash or a non-`--commit` run
//! leaves production weights untouched; weights are clamped and snapshotted.

use mog_synth::meta::recursive::{run, Config};
use std::path::PathBuf;
use std::time::Duration;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}
fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if flag(&args, "--help") || flag(&args, "-h") {
        eprintln!(
            "meta_rsi: bounded recursive self-improvement (weights-only).\n\
             flags: --max-iters --max-no-improve --deadline-secs --sigma --lr --seed --audit-log --commit"
        );
        return;
    }

    let d = Config::default();
    let cfg = Config {
        max_iters: arg(&args, "--max-iters")
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.max_iters),
        max_no_improve: arg(&args, "--max-no-improve")
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.max_no_improve),
        deadline: arg(&args, "--deadline-secs")
            .and_then(|v| v.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(d.deadline),
        sigma: arg(&args, "--sigma")
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.sigma),
        lr: arg(&args, "--lr")
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.lr),
        seed: arg(&args, "--seed")
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.seed),
    };
    let commit = flag(&args, "--commit");
    let audit_log: Option<PathBuf> = arg(&args, "--audit-log").map(PathBuf::from);

    eprintln!(
        "[meta_rsi] start: max_iters={} patience={} deadline={}s sigma={} lr={} seed={} commit={}",
        cfg.max_iters,
        cfg.max_no_improve,
        cfg.deadline.as_secs(),
        cfg.sigma,
        cfg.lr,
        cfg.seed,
        commit
    );

    match run(&cfg, commit, audit_log.as_deref()) {
        Ok(r) => {
            let out = serde_json::json!({
                "baseline_solved": r.outcome.baseline_solved,
                "final_solved": r.outcome.final_solved,
                "total": r.outcome.total,
                "accepted": r.outcome.accepted,
                "iterations": r.outcome.iterations,
                "improved": r.outcome.improved,
                "committed": r.committed,
                "snapshot": r.snapshot.display().to_string(),
            });
            println!("{out}");
        }
        Err(e) => {
            eprintln!("[meta_rsi] error: {e}");
            std::process::exit(1);
        }
    }
}
