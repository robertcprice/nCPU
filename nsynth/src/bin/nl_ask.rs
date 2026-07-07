//! The no-examples never-wrong front door: a person types a request, gets a verified
//! operation and a DEMONSTRATION of what it does — or an honest refusal. There is no
//! example oracle here, so the safety mechanism is the demonstration: the user
//! confirms by recognizing the behavior, not by trusting a claim.
//!
//! Usage: nl_ask "reverse a string"
use mog_synth::verified_nl_router::{declare, demonstrate};

fn fmt_inputs(inputs: &[mog_synth::benchmark::Value]) -> String {
    inputs
        .iter()
        .map(|v| format!("{v:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if prompt.trim().is_empty() {
        eprintln!("usage: nl_ask \"<request>\"");
        std::process::exit(2);
    }

    match declare(&prompt) {
        None => {
            println!("REFUSED");
            println!("  No verified operation confidently matches that request.");
            println!("  Rephrase, or give an example (input -> output) so it can be verified.");
        }
        Some(op) => {
            println!("UNDERSTOOD AS  {}", op.name);
            let demo = demonstrate(op);
            if demo.is_empty() {
                println!("  (could not produce an illustrative run)");
            } else {
                println!("  It does this:");
                for (inputs, out) in &demo {
                    println!("    {}  ->  {}", fmt_inputs(inputs), out);
                }
            }
            println!("  Confirm the behavior above is what you meant.");
            println!("\n  Verified program:");
            for line in op.mog.lines() {
                println!("    {line}");
            }
        }
    }
}
