//! Generate prior-net training data (ROADMAP Rung 9, Phase A, stage A1).
//!
//! Samples random universal-array programs (hand-coded restart shapes +
//! pure random biases), executes their discretized Mog code exactly, and
//! writes `(examples -> discrete program description)` JSONL rows.
//!
//! Usage:
//!   cargo run --release --bin gen_prior_data -- \
//!     --rows 100000 --seed 42 --out data/prior_net_train.jsonl

use mog_synth::synthesis::prior_gen::generate_prior_data;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = arg_value(&args, "--rows")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);
    let out = arg_value(&args, "--out").unwrap_or_else(|| "data/prior_net_train.jsonl".to_string());

    if let Some(parent) = std::path::Path::new(&out).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let t0 = std::time::Instant::now();
    match generate_prior_data(rows, seed, &out) {
        Ok(stats) => {
            let mut report = stats.to_json();
            report["seconds"] = serde_json::json!(t0.elapsed().as_secs_f64());
            report["out"] = serde_json::json!(out);
            report["seed"] = serde_json::json!(seed);
            println!("{report}");
        }
        Err(e) => {
            eprintln!("gen_prior_data failed: {e}");
            std::process::exit(1);
        }
    }
}
