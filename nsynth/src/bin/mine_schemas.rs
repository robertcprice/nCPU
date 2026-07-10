//! WP6 — mine recurring Mog/Rust templates from a verified harvest JSONL.
//!
//! Usage: mine_schemas <path.jsonl> [top_k]
//!
//! Each line: `{"task":"...","code":"..."}` or `{"prompt":"...","program":"..."}`.
//! Identifiers are hole-normalized; programs cluster by statement sequence; top-k
//! templates are printed.

use mog_synth::schema_miner::{
    cluster_templates, format_top_templates, load_harvest_jsonl,
};
use std::path::Path;
use std::process;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: mine_schemas <harvest.jsonl> [top_k]");
        process::exit(2);
    };
    let top_k: usize = args
        .next()
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
}
