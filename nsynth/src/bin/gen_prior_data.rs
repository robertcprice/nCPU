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

    // --eval-fallback: run synthesize_universal_array_fallback in isolation
    // on named benchmark problems (comma-separated via --problems; all
    // array problems if omitted) and emit one JSON row per problem. Honors
    // NSYNTH_PRIOR_NET, so OFF/ON diffs measure the tier-0 prior directly.
    if args.iter().any(|a| a == "--eval-fallback") {
        let names: Vec<String> = arg_value(&args, "--problems")
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default();
        let rows = mog_synth::synthesis::prior_gen::eval_fallback_direct(&names);
        for row in &rows {
            println!("{row}");
        }
        let solved = rows
            .iter()
            .filter(|r| r["solved"].as_bool() == Some(true))
            .count();
        let total_secs: f64 = rows.iter().filter_map(|r| r["seconds"].as_f64()).sum();
        println!(
            "{}",
            serde_json::json!({
                "summary": true,
                "problems": rows.len(),
                "solved": solved,
                "total_seconds": (total_secs * 100.0).round() / 100.0,
                "prior_net": std::env::var("NSYNTH_PRIOR_NET").unwrap_or_default() == "1",
            })
        );
        let _ = mog_synth::learned_biases::flush();
        return;
    }

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
