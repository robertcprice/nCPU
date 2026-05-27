//! Snapshot the online-learned `MetaWeights` to a timestamped history file.
//!
//! Reads `~/.nsynth_meta_weights.tsv` (or the path in
//! `NSYNTH_META_WEIGHTS_PATH`), prepends a UNIX timestamp, appends to
//! `artifacts/meta_weights_history.tsv`. Designed to be cron-able or
//! git-committable — one line per snapshot, 27 tab-separated columns
//! (timestamp + 26 weights).
//!
//! Plotting weight trajectories over time answers "which features did the
//! ranker learn matter?" — a direct, data-driven window into what the
//! online update rule is actually doing to the feature space.
//!
//! Usage:
//!     cargo run --release --bin weights_snapshot -- \
//!         [--out artifacts/meta_weights_history.tsv]    default path
//!         [--label <tag>]                               optional column appended
//!         [--show]                                      print the row after append

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use mog_synth::meta_learner::{MetaWeights, FEATURE_DIM};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = arg_value(&args, "--out")
        .unwrap_or_else(|| "artifacts/meta_weights_history.tsv".to_string());
    let label = arg_value(&args, "--label").unwrap_or_default();
    let show = has_flag(&args, "--show");

    // Ensure parent dir exists so first-run writes don't fail.
    if let Some(parent) = Path::new(&out_path).parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            eprintln!("[weights_snapshot] cannot create {:?}: {err}", parent);
            std::process::exit(1);
        }
    }

    let weights = MetaWeights::load();
    let ts = now_epoch();

    let mut row = String::new();
    row.push_str(&ts.to_string());
    row.push('\t');
    for (i, w) in weights.w.iter().enumerate() {
        if i > 0 {
            row.push('\t');
        }
        row.push_str(&format!("{:.6}", w));
    }
    if !label.is_empty() {
        row.push('\t');
        row.push_str(&label);
    }
    row.push('\n');

    let mut file = match OpenOptions::new().append(true).create(true).open(&out_path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[weights_snapshot] cannot open {}: {err}", out_path);
            std::process::exit(1);
        }
    };
    if let Err(err) = file.write_all(row.as_bytes()) {
        eprintln!("[weights_snapshot] write error: {err}");
        std::process::exit(1);
    }

    eprintln!(
        "[weights_snapshot] appended ts={} ({} weight dims) → {}",
        ts, FEATURE_DIM, out_path,
    );
    if show {
        print!("{}", row);
    }
}
