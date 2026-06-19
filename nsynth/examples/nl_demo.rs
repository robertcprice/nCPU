//! Natural Language Demo for nCPU/nSynth
//!
//! This example demonstrates the complete NL → code pipeline.
//! It takes natural language input, generates examples, and synthesizes
//! a verified program using the nCPU/nSynth solver.

use mog_synth::nl::NLPipeline;

fn main() {
    println!("=== nCPU/nSynth Natural Language → Code Pipeline ===\n");

    // Example 1: Simple addition
    let nl_input = "add two numbers together";
    println!("Input: \"{}\"", nl_input);
    println!();

    let pipeline = NLPipeline::new();
    let result = pipeline.synthesize_from_nl(nl_input);

    if result.success {
        println!("✓ Synthesis successful!");
        println!("  Method: {}", result.method);
        println!("  Generated code:");
        println!("  ```");
        for line in result.code.lines() {
            println!("    {}", line);
        }
        println!("  ```");
    } else {
        println!("✗ Synthesis failed");
        if let Some(error) = &result.error {
            println!("  Error: {}", error);
        }
    }
    println!();

    // Example 2: List operations
    let nl_input2 = "count the elements in a list";
    println!("Input: \"{}\"", nl_input2);
    println!();

    let result2 = pipeline.synthesize_from_nl(nl_input2);

    if result2.success {
        println!("✓ Synthesis successful!");
        println!("  Method: {}", result2.method);
        println!("  Generated code:");
        println!("  ```");
        for line in result2.code.lines().take(10) {
            println!("    {}", line);
        }
        if result2.code.lines().count() > 10 {
            println!("    ...");
        }
        println!("  ```");
    } else {
        println!("✗ Synthesis failed");
        if let Some(error) = &result2.error {
            println!("  Error: {}", error);
        }
    }
    println!();

    println!("=== Pipeline Data Flow ===");
    println!("1. NL Input → Example Generation (synthesizer)");
    println!("2. Examples → Problem Construction (signature inference)");
    println!("3. Problem → Solver Pipeline (enumerative, search, gradient)");
    println!("4. Solver → Verified Program");
    println!();

    println!("=== Demo Complete ===");
}
