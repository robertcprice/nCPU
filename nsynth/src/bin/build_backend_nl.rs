//! Build a generated local backend MVP from synthesized rule contracts.
//!
//! LOOP-4 default path: English description with inline examples flows through
//! the real NL door (`synthesize_project`) into a dependency-free Rust HTTP
//! backend artifact. `--hand-specs` keeps the pre-authored spec path.

use mog_synth::backend_ir::StoreKind;
use mog_synth::backend_mvp::{
    default_out_path, default_rule_spec, default_rule_specs, write_backend,
    write_backend_app, write_default_backend,
};
use mog_synth::backend_nl::{
    default_required_rule_names, write_backend_from_english, DEFAULT_BACKEND_ENGLISH,
};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn read_text_file(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "build_backend_nl — synthesize a rule-backed local Rust backend\n\
             --out PATH        write generated backend source here\n\
             --store MODE      memory | file | sqlite (default: file)\n\
             --english PATH    read backend English contract from file\n\
             --text ENGLISH    inline English contract (default: built-in demo)\n\
             --hand-specs      use pre-authored rule specs (no NL door)\n\
             --single          synthesize only the default score_bonus rule\n\
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
    let hand_specs = args.iter().any(|a| a == "--hand-specs");
    let english = if let Some(path) = arg_value(&args, "--english") {
        read_text_file(&path).unwrap_or_else(|e| die(&e))
    } else {
        arg_value(&args, "--text").unwrap_or_else(|| DEFAULT_BACKEND_ENGLISH.to_string())
    };

    let result = if single {
        write_backend(&out, &default_rule_spec(), store)
    } else if hand_specs {
        if store == StoreKind::File && !args.iter().any(|a| a == "--store") {
            write_default_backend(&out)
        } else {
            write_backend_app(&out, &default_rule_specs(), store)
        }
    } else {
        write_backend_from_english(&out, &english, default_required_rule_names(), store)
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
                eprintln!("[backend]   - {} via {}", rule.name, rule.rule_method);
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

fn die(msg: &str) -> ! {
    eprintln!("[backend] REFUSED: {msg}");
    std::process::exit(1);
}
