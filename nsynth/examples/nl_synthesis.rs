//! Natural Language to Code synthesis example
//!
//! Demonstrates the Linguigenesis → nSynth pipeline:
//! NL description → Examples → Synthesized code

use std::env;
use std::time::Instant;

#[cfg(feature = "nl")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <natural language description>", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} 'add two numbers'", args[0]);
        eprintln!("  {} 'reverse an array'", args[0]);
        eprintln!("  {} 'filter even numbers from a list'", args[0]);
        eprintln!("  {} 'map elements by doubling'", args[0]);
        eprintln!("  {} 'sort an array'", args[0]);
        eprintln!("  {} 'calculate fibonacci'", args[0]);
        eprintln!();
        eprintln!("Or use --analyze to see the belief state:");
        eprintln!("  {} --analyze 'map elements'", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];
    let analyze = args.get(2).map(|s| s == "--analyze").unwrap_or(false);

    if analyze {
        // Show belief state analysis
        println!("🔍 Analyzing: '{}'", input);
        println!();

        let belief = mog_synth::solver::analyze_nl(input)?;
        println!("Intent Type: {}", belief.intent_type);
        println!("Entities: {}", belief.entities.join(", "));
        println!("Confidence: {:.2}", belief.confidence);
    } else {
        // Full synthesis
        println!("🎯 Synthesizing from: '{}'", input);
        println!();

        let t0 = Instant::now();

        // Extract function name from input or use default
        let fn_name = extract_function_name(input);

        let result = mog_synth::solver::solve_from_nl(input, fn_name.as_deref())?;

        let elapsed = t0.elapsed();

        if result.success {
            println!("✅ Success in {:.2}s", elapsed.as_secs_f64());
            println!();
            println!("Method: {}", result.method);
            println!();
            println!("Generated code:");
            println!("---");
            println!("{}", result.code);
            println!("---");
        } else {
            println!("❌ Failed to synthesize");
            if let Some(error) = result.error {
                println!("Error: {}", error);
            }
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Extract a reasonable function name from the description
fn extract_function_name(input: &str) -> Option<String> {
    let lower = input.to_lowercase();

    // Map common patterns to function names
    if lower.contains("add") || lower.contains("sum") {
        Some("add".to_string())
    } else if lower.contains("subtract") || lower.contains("difference") {
        Some("subtract".to_string())
    } else if lower.contains("multiply") || lower.contains("product") {
        Some("multiply".to_string())
    } else if lower.contains("divide") {
        Some("divide".to_string())
    } else if lower.contains("reverse") {
        Some("reverse".to_string())
    } else if lower.contains("filter") {
        Some("filter".to_string())
    } else if lower.contains("map") {
        Some("map".to_string())
    } else if lower.contains("sort") {
        Some("sort".to_string())
    } else if lower.contains("search") || lower.contains("find") {
        Some("search".to_string())
    } else if lower.contains("fibonacci") || lower.contains("fib") {
        Some("fibonacci".to_string())
    } else {
        None
    }
}

#[cfg(not(feature = "nl"))]
fn main() {
    eprintln!("Error: The 'nl' feature is required for natural language synthesis.");
    eprintln!("Run with: cargo run --example nl_synthesis --features nl -- <description>");
    std::process::exit(1);
}
