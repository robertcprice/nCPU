//! WP6 — mine recurring Mog/Rust templates from a verified harvest JSONL.
//!
//! Usage: mine_schemas <path.jsonl> [top_k] [--out templates.json]
//!
//! Each line: `{"task":"...","code":"..."}` or `{"prompt":"...","program":"..."}`.
//! Identifiers + integer literals are hole-normalized; programs cluster by
//! statement sequence; top-k templates are printed (and optionally written for
//! `NSYNTH_MINED_TEMPLATES`).

use mog_synth::schema_miner::{
    cluster_templates, format_top_templates, load_harvest_jsonl, write_templates_json,
};
use std::path::Path;
use std::process;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut out_path: Option<String> = None;
    if let Some(i) = args.iter().position(|a| a == "--out") {
        if i + 1 < args.len() {
            out_path = Some(args[i + 1].clone());
            args.drain(i..=i + 1);
        } else {
            eprintln!("mine_schemas: --out requires a path");
            process::exit(2);
        }
    }
    let Some(path) = args.first().cloned() else {
        eprintln!("usage: mine_schemas <harvest.jsonl> [top_k] [--out templates.json]");
        process::exit(2);
    };
    let top_k: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let rows = match load_harvest_jsonl(Path::new(&path)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("mine_schemas: {e}");
            process::exit(1);
        }
    };
    eprintln!("mine_schemas: {} rows from {path}", rows.len());
    let templates = cluster_templates(&rows, top_k);
    print!("{}", format_top_templates(&templates));
    eprintln!("mine_schemas: {} templates (top_k={top_k})", templates.len());
    if let Some(out) = out_path {
        if let Err(e) = write_templates_json(Path::new(&out), &templates) {
            eprintln!("mine_schemas: {e}");
            process::exit(1);
        }
        eprintln!("mine_schemas: wrote {out} (set NSYNTH_MINED_TEMPLATES={out})");
    }
}
