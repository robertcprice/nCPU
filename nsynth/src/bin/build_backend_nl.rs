//! Build a generated local backend MVP from a synthesized rule contract.
//!
//! This is the LOOP-3B counterpart to `build_game_nl`: the HTTP shell is a
//! small stdlib artifact, while the business rule handler is synthesized and
//! injected only after solver success.

use mog_synth::backend_mvp::{default_out_path, write_default_backend};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "build_backend_nl — synthesize a rule-backed local Rust backend\n\
             --out PATH   write generated backend source here\n\
             default: {}",
            default_out_path().display()
        );
        return;
    }

    let out = arg_value(&args, "--out")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_out_path);

    match write_default_backend(&out) {
        Ok(generated) => {
            eprintln!(
                "[backend] wrote {} (rule method: {}, {} bytes)",
                out.display(),
                generated.rule_method,
                generated.source.len()
            );
            eprintln!(
                "[backend] compile with: rustc --edition=2021 {} -o /tmp/generated_rule_backend",
                out.display()
            );
        }
        Err(err) => {
            eprintln!("[backend] REFUSED: {err}");
            std::process::exit(1);
        }
    }
}
