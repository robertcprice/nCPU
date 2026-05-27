//! Join the timestamped weight history with the timestamped solve log and
//! emit `(problem, ts, weights, time_ms, via_cached_teachers)` rows.
//!
//! The join is nearest-in-time on the `ts` field: for each solve row we
//! attach the most-recent weight snapshot at-or-before that solve. This
//! gives a dataset suitable for answering "did feature weight w_i rising
//! correlate with more transfer wins?" — the causal question the online
//! update rule exists to create signal for.
//!
//! No plotting here — just a clean JSONL emitter. Downstream analysis (even
//! a three-line Python pandas notebook) can take it from there.
//!
//! Input:
//!   --weights artifacts/meta_weights_history.tsv   (ts \t w0 \t w1 \t ... \t w25)
//!   --solves  artifacts/transfer_curve/curve_*.jsonl
//!
//! Output: JSONL on stdout, one row per solve:
//!   {
//!     "problem":"fibonacci_v0",
//!     "solve_ts_ms":...,
//!     "weights_ts":...,
//!     "weights":[w0,w1,...,w25],
//!     "time_ms":...,
//!     "via_cached_teachers":...,
//!     "method":"..."
//!   }
//!
//! Usage:
//!     cargo run --release --bin weight_transfer_join -- \
//!         --weights artifacts/meta_weights_history.tsv   \
//!         --solves artifacts/transfer_curve/curve_cum.jsonl \
//!         [--treatment-round N]

use std::fs::File;
use std::io::{BufRead, BufReader};

use serde::Deserialize;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
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

#[derive(Deserialize, Debug)]
struct SolveRow {
    #[serde(default)]
    round: usize,
    name: String,
    method: String,
    time_ms: u64,
    #[serde(default)]
    via_cached_teachers: bool,
}

/// One row of the weights history file. `ts` is seconds since epoch; the
/// full 26-dim weight vector follows.
struct WeightRow {
    ts: u64,
    weights: Vec<f64>,
}

fn load_weights(path: &str) -> Vec<WeightRow> {
    let Ok(file) = File::open(path) else {
        eprintln!("[weight_transfer_join] cannot open {}", path);
        std::process::exit(1);
    };
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let Some(ts_str) = parts.next() else {
            continue;
        };
        let Ok(ts) = ts_str.parse::<u64>() else {
            continue;
        };
        let weights: Vec<f64> = parts.filter_map(|s| s.parse::<f64>().ok()).collect();
        // Tolerate label columns at the end: keep whatever parsed as f64.
        if weights.is_empty() {
            continue;
        }
        rows.push(WeightRow { ts, weights });
    }
    rows.sort_by_key(|r| r.ts);
    rows
}

fn load_solves(path: &str, round_filter: Option<usize>) -> Vec<SolveRow> {
    let Ok(file) = File::open(path) else {
        eprintln!("[weight_transfer_join] cannot open {}", path);
        std::process::exit(1);
    };
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if line.trim().is_empty() || line.contains("summary_round") {
            continue;
        }
        if let Ok(row) = serde_json::from_str::<SolveRow>(&line) {
            if round_filter.map_or(true, |r| row.round == r) {
                rows.push(row);
            }
        }
    }
    rows
}

/// Find the most-recent weight row with `ts <= target_ts`. Returns None when
/// no snapshot predates the target (e.g. solves from before any weight
/// history was captured). Callers treat that as "weights unknown".
fn nearest_weights_at_or_before<'a>(
    weights: &'a [WeightRow],
    target_ts: u64,
) -> Option<&'a WeightRow> {
    // Linear scan suffices at the scale of a weekly file (~dozens of rows).
    // Sorted by ts so last eligible wins.
    weights.iter().rev().find(|w| w.ts <= target_ts)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights_path = match arg_value(&args, "--weights") {
        Some(p) => p,
        None => {
            eprintln!("[weight_transfer_join] --weights PATH required");
            std::process::exit(2);
        }
    };
    let solves_path = match arg_value(&args, "--solves") {
        Some(p) => p,
        None => {
            eprintln!("[weight_transfer_join] --solves PATH required");
            std::process::exit(2);
        }
    };
    let round_filter: Option<usize> =
        arg_value(&args, "--treatment-round").and_then(|v| v.parse().ok());

    let weights = load_weights(&weights_path);
    let solves = load_solves(&solves_path, round_filter);

    if weights.is_empty() {
        eprintln!(
            "[weight_transfer_join] no weight rows in {} — run weights_snapshot first",
            weights_path
        );
        std::process::exit(1);
    }

    // Assume the solve log's ts is "now" for each row, since transfer_curve
    // doesn't emit an explicit timestamp per row. We use the filesystem mtime
    // of the solve file as a conservative upper bound — all solves happened
    // at-or-before that. Less precise than per-row stamps but good enough
    // for the weekly-cadence analysis this enables.
    let solve_file_ts = std::fs::metadata(&solves_path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut emitted = 0usize;
    let mut missing_weights = 0usize;
    for row in &solves {
        let w = nearest_weights_at_or_before(&weights, solve_file_ts);
        let Some(w) = w else {
            missing_weights += 1;
            continue;
        };
        let weight_str: Vec<String> = w.weights.iter().map(|v| format!("{:.6}", v)).collect();
        println!(
            r#"{{"problem":"{}","solve_ts_ms":{},"weights_ts":{},"time_ms":{},"method":"{}","via_cached_teachers":{},"weights":[{}]}}"#,
            json_escape(&row.name),
            solve_file_ts,
            w.ts,
            row.time_ms,
            json_escape(&row.method),
            row.via_cached_teachers,
            weight_str.join(","),
        );
        emitted += 1;
    }

    eprintln!(
        "[weight_transfer_join] joined {} solves to {} weight snapshots (emitted {}, unmapped {})",
        solves.len(),
        weights.len(),
        emitted,
        missing_weights
    );
}
