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

/// Conservative + TARGET-AWARE: is the transpile of `mog` to `lang` faithful by
/// construction? Two conditions must both hold:
///   1. Pure scalar int/bool arithmetic + control flow (no strings, arrays, floats,
///      or Mog builtins like `.reverse()` — those may not port to a valid idiom).
///   2. The target has Mog's i64 integer semantics. rust/go/java use fixed 64-bit
///      ints; python uses arbitrary-precision (bigint) and typescript/js use float64
///      `number` — so an arithmetic result that overflows i64 (e.g. factorial(25))
///      DIVERGES from Mog on those targets. Claiming "faithful" there would be a new
///      confidently-wrong path, so it isn't claimed.
fn transpile_faithful(mog: &str, lang: &str) -> bool {
    let pure_arith =
        !mog.contains("string") && !mog.contains('[') && !mog.contains('.') && !mog.contains("push");
    let i64_target = matches!(lang.to_ascii_lowercase().as_str(), "rust" | "rs" | "go" | "java");
    pure_arith && i64_target
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

    // Priority: a named 2-op CHAIN (more specific — two ops in sequence) beats a
    // single op; a single op beats a refusal. Both no-example paths confirm by
    // demonstration.
    if let Some(code) = mog_synth::verified_nl_router::declare_composed(&prompt) {
        println!("UNDERSTOOD AS  a 2-step chain");
        print_demo(&mog_synth::verified_nl_router::demonstrate_program(&code));
        println!("  Confirm the behavior above is what you meant.");
        println!("\n  Verified program:");
        for line in code.lines() {
            println!("    {line}");
        }
    } else if let Some(op) = declare(&prompt) {
        println!("UNDERSTOOD AS  {}", op.name);
        print_demo(&demonstrate(op));
        println!("  Confirm the behavior above is what you meant.");
        match &emit_lang {
            Some(lang) => match emit(op.mog, lang) {
                Some(src) => {
                    // Graded, honest labeling: pure i64 arithmetic ports 1:1 to
                    // rust/go/java and is trusted; anything else is a translation to
                    // review. Never present an unverified translation as verified.
                    if transpile_faithful(op.mog, lang) {
                        println!("\n  Verified {lang} source (faithful 1:1 transpile — i64 arithmetic):");
                    } else {
                        println!("\n  {lang} translation (from the verified Mog — review builtins / integer width before use):");
                    }
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
    } else {
        println!("REFUSED");
        println!("  No verified operation confidently matches that request.");
        println!("  Rephrase, or give an example (input -> output) so it can be verified.");
    }
}

/// Print the illustrative input->output rows of a demonstration.
fn print_demo(demo: &[(Vec<mog_synth::benchmark::Value>, String)]) {
    if demo.is_empty() {
        println!("  (could not produce an illustrative run)");
        return;
    }
    println!("  It does this:");
    for (inputs, out) in demo {
        println!("    {}  ->  {}", fmt_inputs(inputs), out);
    }
}
