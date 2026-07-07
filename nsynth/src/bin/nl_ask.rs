//! The no-examples never-wrong front door: a person types a request, gets a verified
//! operation and a DEMONSTRATION of what it does — or an honest refusal. There is no
//! example oracle here, so the safety mechanism is the demonstration: the user
//! confirms by recognizing the behavior, not by trusting a claim.
//!
//! Usage: nl_ask "reverse a string"
use mog_synth::mog_transpile;
use mog_synth::verified_nl_router::{declare, demonstrate};

fn fmt_inputs(inputs: &[mog_synth::benchmark::Value]) -> String {
    inputs
        .iter()
        .map(|v| format!("{v:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Transpile the verified Mog to a target language, or None for an unknown one.
fn emit(mog: &str, lang: &str) -> Option<String> {
    Some(match lang.to_ascii_lowercase().as_str() {
        "python" | "py" => mog_transpile::to_python(mog),
        "rust" | "rs" => mog_transpile::to_rust(mog),
        "typescript" | "ts" => mog_transpile::to_typescript(mog),
        "go" => mog_transpile::to_go(mog),
        "java" => mog_transpile::to_java(mog),
        _ => return None,
    })
}

fn main() {
    // Optional `--emit <lang>` flag: emit the verified op as device-native source.
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut emit_lang: Option<String> = None;
    if let Some(pos) = args.iter().position(|a| a == "--emit") {
        emit_lang = args.get(pos + 1).cloned();
        args.drain(pos..=(pos + 1).min(args.len() - 1));
    }
    let prompt = args.join(" ");
    if prompt.trim().is_empty() {
        eprintln!("usage: nl_ask \"<request>\" [--emit python|rust|typescript|go|java]");
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
            match &emit_lang {
                Some(lang) => match emit(op.mog, lang) {
                    Some(src) => {
                        // Honest labeling: the VERIFIED artifact is the Mog program
                        // (proven above by demonstration). The transpile is a
                        // structural translation — arithmetic/control-flow ports
                        // faithfully, but Mog builtins (e.g. s.reverse()) may not map
                        // to a valid target idiom. Never present it AS verified.
                        println!("\n  {lang} translation (from the verified Mog — review builtins before use):");
                        for line in src.lines() {
                            println!("    {line}");
                        }
                    }
                    None => println!("\n  (unknown --emit target '{lang}')"),
                },
                None => {
                    println!("\n  Verified program:");
                    for line in op.mog.lines() {
                        println!("    {line}");
                    }
                }
            }
        }
    }
}
