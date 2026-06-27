//! Build a generated local backend MVP from a synthesized rule contract.
//!
//! This is the LOOP-3B/3C counterpart to `build_game_nl`: the HTTP shell is a
//! small stdlib artifact, while the business rule handler is synthesized and
//! injected only after solver success. Store mode selects in-memory, JSONL
//! file, or SQLite persistence in the generated artifact.

use mog_synth::backend_ir::StoreKind;
use mog_synth::backend_mvp::{default_out_path, write_backend, write_default_backend, default_rule_spec};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "build_backend_nl — synthesize a rule-backed local Rust backend\n\
             --out PATH      write generated backend source here\n\
             --store MODE    memory | file | sqlite (default: file)\n\
             default out: {}",
            default_out_path().display()
        );
        return;
    }

    let out = arg_value(&args, "--out")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_out_path);
    let store = arg_value(&args, "--store")
        .as_deref()
        .and_then(StoreKind::parse)
        .unwrap_or(StoreKind::File);

    let result = if store == StoreKind::File && !args.iter().any(|a| a == "--store") {
        write_default_backend(&out)
    } else {
        write_backend(&out, &default_rule_spec(), store)
    };

    match result {
        Ok(generated) => {
            eprintln!(
                "[backend] wrote {} (store: {}, rule method: {}, {} bytes)",
                out.display(),
                store.cli_name(),
                generated.rule_method,
                generated.source.len()
            );
            let link_hint = if store == StoreKind::Sqlite {
                " -l sqlite3"
            } else {
                ""
            };
            eprintln!(
                "[backend] compile with: rustc --edition=2021 {}{} -o /tmp/generated_rule_backend",
                out.display(),
                link_hint
            );
        }
        Err(err) => {
            eprintln!("[backend] REFUSED: {err}");
            std::process::exit(1);
        }
    }
}
