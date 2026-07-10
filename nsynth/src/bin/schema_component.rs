//! SCHEMA -> VERIFIED TYPED COMPONENT (the novel Phase-2 front door).
//!
//! Thin CLI over [`mog_synth::schema_component`]. Prefer the product path:
//! `coding_agent --root <dir> query "a todo list where each task has …"` which
//! scaffolds + fills in one shot via [`mog_synth::whole_software`].
//!
//! Usage:
//!   schema_component "<prose>" [out_dir]
//! Then:  coding_agent --root <out_dir> query "fix the failing tests"

use mog_synth::schema_component;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prose = args.get(1).cloned().unwrap_or_default();
    if prose.trim().is_empty() {
        eprintln!("usage: schema_component \"<prose schema>\" [out_dir]");
        std::process::exit(2);
    }
    let out_dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("schema_component_out"));

    let Some(written) = schema_component::try_write_schema_crate(&out_dir, &prose) else {
        eprintln!("could not parse a schema (need e.g. \"... where each X has a, b and c\")");
        std::process::exit(1);
    };

    println!(
        "schema: {} {{ items: Vec<{}> }}  ({} fields, {} tests)",
        written.collection, written.record, written.n_fields, written.n_tests
    );
    println!("wrote crate to {}", out_dir.display());
    println!(
        "fill + verify:  coding_agent --root {} query \"fix the failing tests\"",
        out_dir.display()
    );
}
