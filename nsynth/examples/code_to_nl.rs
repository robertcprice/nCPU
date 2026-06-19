//! Code → Natural Language synthesis example
//!
//! Demonstrates reverse pipeline: Generated code → NL documentation

use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <rust code> or --file <filename>", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} 'fn add(a: i64, b: i64) -> i64 {{ a + b }}'", args[0]);
        eprintln!("  {} --file examples/code.rs", args[0]);
        std::process::exit(1);
    }

    let code = if args.get(1).map(|s| s.as_str()) == Some("--file") {
        let filename = args.get(2).ok_or("Expected filename after --file")?;
        std::fs::read_to_string(filename)?
    } else {
        args[1].clone()
    };

    println!("🔍 Analyzing code...");
    println!("---");
    println!("{}", code);
    println!("---");
    println!();

    let t0 = std::time::Instant::now();

    let nl = mog_synth::bidirectional::code_to_nl(&code)?;

    let elapsed = t0.elapsed();

    println!("✅ Analysis complete in {:.2}s", elapsed.as_secs_f64());
    println!();
    println!("Generated description:");
    println!("---");
    println!("{}", nl);
    println!("---");
    println!();

    let summary = mog_synth::bidirectional::generator::generate_summary(
        &mog_synth::bidirectional::analyzer::analyze_semantics(
            &mog_synth::bidirectional::parser::parse_code(&code)?
        )
    );

    println!("Summary: {}", summary);

    Ok(())
}
