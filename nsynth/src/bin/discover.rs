//! Continuous component crawler daemon. Loops `crawl_once`, appending every novel
//! verified composition to a discoveries log until the composition space is
//! exhausted (or `--rounds N`). Resumable — restart and it continues from the log.
//!
//!   cargo run --release --bin discover -- --log discoveries.jsonl
//!   cargo run --release --bin discover -- --rounds 5 --per-round 6
use mog_synth::component_crawler::crawl_once;
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

fn arg_val(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log = std::path::PathBuf::from(
        arg_val(&args, "--log").unwrap_or_else(|| "component_discoveries.jsonl".to_string()),
    );
    let work = std::env::temp_dir().join("nsynth_discover_work");
    let rounds: usize = arg_val(&args, "--rounds")
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let per_round: usize = arg_val(&args, "--per-round")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let bridge = LinguigenesisBridge::new();
    let mut total = 0usize;
    for r in 0..rounds {
        let found = crawl_once(&bridge, &log, &work, per_round);
        if found.is_empty() {
            println!(
                "round {r}: no new behaviors — composition space exhausted for the current \
                 leaf set ({total} discoveries total in {})",
                log.display()
            );
            break;
        }
        for d in &found {
            println!(
                "DISCOVERED {}  ::  {}  ::  signature={:?}",
                d.name,
                d.chain.join(" -> "),
                d.signature
            );
        }
        total += found.len();
    }
    println!(
        "done: {total} novel verified compositions logged to {}",
        log.display()
    );

    // FLYWHEEL: --promote <components.json> pushes every logged discovery into
    // the live component-registry data file (merge by name), so discovered ops
    // become NL-resolvable + buildable exactly like seeds on the next boot.
    if let Some(out) = arg_val(&args, "--promote") {
        let out = std::path::PathBuf::from(out);
        match mog_synth::component_crawler::promote_discoveries(&log, &out) {
            Ok(n) => println!("promoted: {n} new component(s) merged into {}", out.display()),
            Err(e) => eprintln!("promotion failed: {e}"),
        }
    }
}
