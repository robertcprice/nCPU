//! Gated e2e: the FULL loop the user is asking for — a single English prompt →
//! the untrusted LLM decomposes it (Mode C) → each component is STRICT-VERIFIED →
//! the verified components are written as a multi-file Rust crate → `cargo check`
//! COMPILE GATE. Proves "prompt → verified, compiling, multi-file program".
//!
//! This is layered verification: each logic leaf is verified against examples; the
//! ASSEMBLY is verified by compilation. (Whole-artifact INTENT is not formally
//! verified — that needs acceptance tests, a separate layer.)
//!
//! Skips unless NSYNTH_LOCAL_LLM_URL is served AND NSYNTH_LOCAL_LLM_PROJECT is set.
use mog_synth::agent::repo::nl_fixture_harness::{write_synthesized_project, CompileStatus};
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;
use std::path::PathBuf;

#[test]
fn prompt_to_verified_compiling_multifile_crate() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[E2E] skipped (no url)");
        return;
    }
    std::env::set_var("NSYNTH_LOCAL_LLM_PROJECT", "1");
    let bridge = LinguigenesisBridge::new();
    assert!(bridge.registry_load_error().is_none(), "registry must load");

    let req = "build helpers to analyze a list of numbers: their total, the largest one, \
               and how many are positive";
    let (verified, failed) = bridge
        .synthesize_project_via_llm(req)
        .unwrap_or_else(|| panic!("[E2E] {req:?} → None"));
    eprintln!("[E2E] {} verified, {} failed", verified.len(), failed.len());
    assert!(verified.len() >= 2, "need >=2 verified components, got {}", verified.len());

    // Verified components → (name, mog_code) for the multi-file writer.
    let components: Vec<(String, String)> =
        verified.iter().map(|(name, r)| (name.clone(), r.code.clone())).collect();

    // Write under a unique temp dir (PID keeps parallel runs disjoint).
    let root: PathBuf =
        std::env::temp_dir().join(format!("nsynth_llm_proj_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("mkdir temp root");

    let outcome = write_synthesized_project(&root, "list_analyzer", &components)
        .unwrap_or_else(|e| panic!("[E2E] write failed: {e}"));
    eprintln!("[E2E] wrote {:?}", outcome.written);
    eprintln!("[E2E] compile = {:?}", outcome.compile);

    // The crate must COMPILE. Tolerate Unverified (cargo unavailable in a sandbox)
    // but a real compiler error is a hard failure.
    match &outcome.compile {
        CompileStatus::Ok => {}
        CompileStatus::Unverified(why) => {
            eprintln!("[E2E] compile gate could not run cargo ({why}) — tolerated");
        }
        CompileStatus::Failed(err) => panic!("[E2E] generated crate does NOT compile:\n{err}"),
    }
    let _ = std::fs::remove_dir_all(&root);
}
