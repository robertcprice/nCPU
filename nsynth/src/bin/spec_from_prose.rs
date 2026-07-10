//! PHASE 3 — prose -> SPEC (model) -> FILL (engine) -> verified.
//!
//! Thin CLI over [`mog_synth::whole_software::try_scaffold_from_spec`]. Prefer the
//! product path: `coding_agent query "build a bank account with …"` when
//! `NSYNTH_LOCAL_LLM_URL` is set (scaffolds + fills in one shot).
//!
//! Usage:  spec_from_prose "<prose>" [out_dir]      (needs NSYNTH_LOCAL_LLM_URL)
//! Then:   coding_agent --root <out_dir> query "fix the failing tests"

use mog_synth::whole_software;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prose = args.get(1).cloned().unwrap_or_default();
    if prose.trim().is_empty() {
        eprintln!("usage: spec_from_prose \"<prose>\" [out_dir]   (needs NSYNTH_LOCAL_LLM_URL)");
        std::process::exit(2);
    }
    if std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .is_none()
    {
        eprintln!(
            "Phase 3 needs a served model: set NSYNTH_LOCAL_LLM_URL. The model writes the SPEC \
             (tests + signatures, a checkable artifact); the engine then fills it and cargo test \
             verifies. Nothing untrusted is accepted."
        );
        std::process::exit(3);
    }
    let out_dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("spec_from_prose_out"));

    let Some(scaffolded) = whole_software::try_scaffold_from_spec(&out_dir, &prose) else {
        eprintln!(
            "the model did not return a usable spec (need a ```rust block with struct + empty-body methods + #[test]s)"
        );
        std::process::exit(1);
    };

    println!("{}", scaffolded.summary);
    println!("wrote crate to {}", out_dir.display());
    println!(
        "fill + verify:  coding_agent --root {} query \"fix the failing tests\"",
        out_dir.display()
    );
}
