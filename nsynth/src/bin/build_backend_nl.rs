//! Build a generated local backend MVP from synthesized rule contracts.
//!
//! This is the LOOP-3B/3C/3D counterpart to `build_game_nl`: the HTTP shell is a
//! small stdlib artifact, while business rule handlers are synthesized and
//! injected only after solver success. Store mode selects in-memory, JSONL
//! file, or SQLite persistence in the generated artifact.

use mog_synth::backend_ir::StoreKind;
use mog_synth::backend_mvp::{
    default_out_path, default_rule_spec, default_rule_specs, write_backend,
    write_backend_app, write_default_backend,
};

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
             --single        synthesize only the default score_bonus rule\n\
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
    let single = args.iter().any(|a| a == "--single");

    let result = if single {
        write_backend(&out, &default_rule_spec(), store)
    } else if store == StoreKind::File && !args.iter().any(|a| a == "--store") && !single {
        write_default_backend(&out)
    } else {
        write_backend_app(&out, &default_rule_specs(), store)
    };

    match result {
        Ok(generated) => {
            eprintln!(
                "[backend] wrote {} (rules: {}, store: {}, {} bytes)",
                out.display(),
                generated.rules.len(),
                store.cli_name(),
                generated.source.len()
            );
            for rule in &generated.rules {
                eprintln!(
                    "[backend]   - {} via {}",
                    rule.name, rule.rule_method
                );
            }
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
