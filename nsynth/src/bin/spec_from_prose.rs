//! PHASE 3 — prose -> SPEC (model) -> FILL (engine) -> verified. The oracle-for-prose step.
//!
//! For logic a fixed schema can't decide, the served model writes the SPECIFICATION — a Rust lib.rs
//! of struct(s) + method signatures with EMPTY bodies + a `#[cfg(test)]` module pinning the intended
//! behavior. The model implements nothing; its output is a CHECKABLE artifact. This bin just calls the
//! model and writes that crate. The engine (`coding_agent ... "fix the failing tests"`) then fills the
//! empty bodies via its multi-hole / mutation / synthesis tiers, and `cargo test` gates the result --
//! so the model proposes behavior but never writes trusted code (the verified guarantee holds).
//!
//! Usage:  spec_from_prose "<prose>" [out_dir]      (needs NSYNTH_LOCAL_LLM_URL)
//! Then:   coding_agent --root <out_dir> query "fix the failing tests"

use mog_synth::local_llm;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prose = args.get(1).cloned().unwrap_or_default();
    if prose.trim().is_empty() {
        eprintln!("usage: spec_from_prose \"<prose>\" [out_dir]   (needs NSYNTH_LOCAL_LLM_URL)");
        std::process::exit(2);
    }
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
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

    let Some(lib) = local_llm::propose_spec(&prose) else {
        eprintln!("the model did not return a usable spec (need a ```rust block with struct + empty-body methods + #[test]s)");
        std::process::exit(1);
    };

    let src = out_dir.join("src");
    if std::fs::create_dir_all(&src).is_err() {
        eprintln!("mkdir failed");
        std::process::exit(1);
    }
    let cargo = "[package]\nname = \"spec_crate\"\nversion = \"0.0.0\"\nedition = \"2021\"\n";
    if std::fs::write(out_dir.join("Cargo.toml"), cargo).is_err() || std::fs::write(src.join("lib.rs"), &lib).is_err() {
        eprintln!("write failed");
        std::process::exit(1);
    }
    let n_tests = lib.matches("#[test]").count();
    let n_methods = lib.matches("fn ").count();
    println!("model wrote a spec: {n_methods} fns, {n_tests} tests");
    println!("wrote crate to {}", out_dir.display());
    println!("fill + verify:  coding_agent --root {} query \"fix the failing tests\"", out_dir.display());
}
