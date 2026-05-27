//! Analyze two transfer_curve JSONL files and emit the self-improvement rate.
//!
//! Reads a "baseline" curve (typically fresh-cache) and a "treatment" curve
//! (typically cumulative), joins per-problem rows by name, computes:
//!   - per-problem time ratio: treatment_ms / baseline_ms
//!   - method shift histogram: baseline_method → treatment_method, counts
//!   - via_cached_teachers rate in treatment
//!   - aggregate self-improvement rate (geomean of non-zero ratios)
//!
//! The geomean is chosen over the arithmetic mean because per-problem times
//! span several orders of magnitude (0ms cache-hit vs 10000ms gradient
//! fallback), and we want the metric to reflect "typical" speedup rather
//! than being dominated by the slowest outlier.
//!
//! Usage:
//!     cargo run --release --bin curve_analysis -- \
//!         --baseline curve_fresh.jsonl             \
//!         --treatment curve_cum.jsonl              \
//!         [--round N]                              \
//!         [--json]                                 \
//!         [--min-baseline-ms 1]    drop problems faster than this — their
//!                                  baseline is noise-dominated

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct Row {
    round: usize,
    name: String,
    success: bool,
    method: String,
    time_ms: u64,
    // Older curve files (before the via_cached_teachers field was added)
    // omit this — default to false so the analyser keeps working against
    // legacy artifacts.
    #[serde(default)]
    via_cached_teachers: bool,
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn load_rows(path: &str, round_filter: Option<usize>) -> Vec<Row> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[curve_analysis] cannot open {path}: {err}");
            std::process::exit(1);
        }
    };
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else {
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        // Skip summary_* rows — they don't have a `name` field.
        if line.contains("summary_round") {
            continue;
        }
        match serde_json::from_str::<Row>(&line) {
            Ok(row) => {
                if round_filter.map_or(true, |r| row.round == r) {
                    rows.push(row);
                }
            }
            Err(_) => {
                // Tolerate unparseable rows — a single bad line in a long
                // sweep shouldn't kill the analysis.
            }
        }
    }
    rows
}

/// Geometric mean of a slice of positive values. Skips zeros to avoid
/// `ln(0) = -inf`; returns NaN for empty input.
fn geomean(values: &[f64]) -> f64 {
    let mut count = 0usize;
    let mut log_sum = 0.0_f64;
    for v in values {
        if *v > 0.0 {
            log_sum += v.ln();
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        (log_sum / count as f64).exp()
    }
}

/// Median of a slice of `f64`. Returns NaN for empty input. Mutates via
/// `sort_by` on a local clone so the caller's order is preserved.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let baseline_path = match arg_value(&args, "--baseline") {
        Some(p) => p,
        None => {
            eprintln!("[curve_analysis] --baseline PATH required");
            std::process::exit(2);
        }
    };
    let treatment_path = match arg_value(&args, "--treatment") {
        Some(p) => p,
        None => {
            eprintln!("[curve_analysis] --treatment PATH required");
            std::process::exit(2);
        }
    };
    // `--round N` applies to both sides. For round-over-round comparisons
    // within a single file, pass `--baseline-round N --treatment-round M`
    // instead — that way baseline=file & treatment=same-file gives a real
    // comparison instead of comparing round N against itself.
    let shared_round: Option<usize> = arg_value(&args, "--round").and_then(|v| v.parse().ok());
    let baseline_round: Option<usize> = arg_value(&args, "--baseline-round")
        .and_then(|v| v.parse().ok())
        .or(shared_round);
    let treatment_round: Option<usize> = arg_value(&args, "--treatment-round")
        .and_then(|v| v.parse().ok())
        .or(shared_round);
    let min_baseline_ms: u64 = arg_value(&args, "--min-baseline-ms")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let json_mode = has_flag(&args, "--json");

    let baseline = load_rows(&baseline_path, baseline_round);
    let treatment = load_rows(&treatment_path, treatment_round);

    // Index baseline by name → row (most-recent round wins on duplicates).
    let mut baseline_by_name: BTreeMap<String, Row> = BTreeMap::new();
    for row in baseline {
        baseline_by_name.insert(row.name.clone(), row);
    }

    // Per-problem ratios + method shifts.
    let mut ratios: Vec<f64> = Vec::new();
    let mut method_shifts: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut transfer_count = 0usize;
    let mut treatment_count = 0usize;
    let mut joined_names: BTreeSet<String> = BTreeSet::new();
    let mut per_problem: Vec<(String, u64, u64, f64)> = Vec::new();

    for tr in &treatment {
        let Some(bl) = baseline_by_name.get(&tr.name) else {
            continue;
        };
        treatment_count += 1;
        joined_names.insert(tr.name.clone());
        if tr.via_cached_teachers {
            transfer_count += 1;
        }
        if bl.time_ms < min_baseline_ms {
            // Below the noise floor — ratio would be meaningless.
            continue;
        }
        let ratio = tr.time_ms as f64 / bl.time_ms as f64;
        ratios.push(ratio);
        per_problem.push((tr.name.clone(), bl.time_ms, tr.time_ms, ratio));
        *method_shifts
            .entry((bl.method.clone(), tr.method.clone()))
            .or_insert(0) += 1;
    }

    let rate = geomean(&ratios);
    let median_ratio = median(&ratios);
    // Instant-hit count: treatment_ms = 0 (Stage-0 cache hit or equivalent)
    // when the baseline wasn't. The most unambiguous "learning worked" signal
    // when ratios cluster at 0 and the geomean's log-space filter discards
    // them.
    let instant_hits: usize = per_problem.iter().filter(|(_, _, tr, _)| *tr == 0).count();
    let slowdowns: usize = ratios.iter().filter(|r| **r > 1.05).count();
    let transfer_pct = if treatment_count > 0 {
        100.0 * transfer_count as f64 / treatment_count as f64
    } else {
        0.0
    };

    if json_mode {
        println!(
            r#"{{"baseline":"{}","treatment":"{}","joined":{},"improvement_rate_geomean":{:.4},"improvement_rate_median":{:.4},"instant_hits":{},"slowdowns":{},"via_cached_teachers_pct":{:.2},"below_noise_floor":{}}}"#,
            baseline_path,
            treatment_path,
            joined_names.len(),
            rate,
            median_ratio,
            instant_hits,
            slowdowns,
            transfer_pct,
            treatment_count - ratios.len(),
        );
    } else {
        println!("── curve analysis ──");
        println!(
            "baseline:  {} ({} rows)",
            baseline_path,
            baseline_by_name.len()
        );
        println!("treatment: {} ({} rows)", treatment_path, treatment.len());
        println!("joined:    {} problems", joined_names.len());
        println!();
        println!(
            "self-improvement rate: geomean={:.3}  median={:.3}",
            rate, median_ratio,
        );
        if rate.is_finite() {
            let interpretation = if rate < 0.95 {
                "✓ cumulative is faster — learning is working"
            } else if rate > 1.05 {
                "✗ cumulative is slower — loop may be broken"
            } else {
                "~ indistinguishable from baseline (within noise)"
            };
            println!("                      {}", interpretation);
        }
        println!(
            "instant hits:         {} problems went to 0ms (baseline was nonzero)",
            instant_hits
        );
        println!(
            "slowdowns:            {} problems took >1.05× longer",
            slowdowns
        );
        println!();
        println!(
            "via_cached_teachers:  {:.1}% of treatment wins ({} / {})",
            transfer_pct, transfer_count, treatment_count
        );
        println!(
            "below_noise_floor:    {} problems skipped (baseline < {}ms)",
            treatment_count - ratios.len(),
            min_baseline_ms
        );

        // Top 10 biggest wins + top 10 biggest regressions, if any.
        let mut sorted = per_problem.clone();
        sorted.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
        if !sorted.is_empty() {
            println!("\nBiggest wins (lowest ratio):");
            for (name, bl, tr, ratio) in sorted.iter().take(10) {
                println!("  {:<35} {:>6}ms → {:>6}ms  ×{:.3}", name, bl, tr, ratio);
            }
            println!("\nBiggest regressions (highest ratio):");
            for (name, bl, tr, ratio) in sorted.iter().rev().take(5) {
                println!("  {:<35} {:>6}ms → {:>6}ms  ×{:.3}", name, bl, tr, ratio);
            }
        }

        // Method shift: only print shifts that changed.
        let actual_shifts: Vec<_> = method_shifts.iter().filter(|((b, t), _)| b != t).collect();
        if !actual_shifts.is_empty() {
            println!("\nMethod shifts (baseline → treatment):");
            for ((b, t), n) in &actual_shifts {
                println!("  {:<30} → {:<30}  ×{}", b, t, n);
            }
        } else {
            println!("\nNo method shifts observed (both sides used the same synthesis stages).");
        }
    }
}
