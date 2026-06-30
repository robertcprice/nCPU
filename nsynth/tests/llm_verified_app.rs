// tests/llm_verified_app.rs
//
// PIECE 5 — gated end-to-end: prompt -> contract-driven decomposition ->
// strict-verified components -> written verified project -> compile + test gates.
//
// This test is INERT unless a live local LLM server is configured via
// NSYNTH_LOCAL_LLM_URL. When the URL is unset it prints SKIP and returns, so it
// still proves the whole flow compiles and links. The orchestrator's main loop
// runs it against the live server.
//
// Canonical re-export path (mod.rs:21): the verified-project writer + its status
// enums are exposed at `mog_synth::agent::repo::{...}`.

use mog_synth::agent::repo::{write_verified_project, CompileStatus, TestStatus};
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
fn prompt_to_verified_tested_app() {
    // Skip unless a live local LLM server is configured.
    if std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .is_none()
    {
        eprintln!("SKIP prompt_to_verified_tested_app: NSYNTH_LOCAL_LLM_URL unset");
        return;
    }
    std::env::set_var("NSYNTH_LOCAL_LLM_PROJECT", "1");
    let bridge = LinguigenesisBridge::default();
    let req = "build a small library of numeric list helpers: \
               the sum of all elements, and the maximum element";
    let Some((verified, failed)) = bridge.synthesize_project_with_contracts(req) else {
        panic!("contract lane returned None despite URL+PROJECT set");
    };
    eprintln!(
        "verified: {:?}",
        verified.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
    eprintln!("failed: {failed:?}");
    assert!(!verified.is_empty(), "expected >=1 verified component");

    let components: Vec<(String, String, Vec<_>)> = verified
        .iter()
        .map(|c| (c.name.clone(), c.result.code.clone(), c.examples.clone()))
        .collect();

    let dir = std::env::temp_dir().join(format!(
        "nsynth_verified_app_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let outcome = write_verified_project(&dir, "numeric_helpers", &components)
        .expect("write_verified_project must run");
    eprintln!("compile: {:?}", outcome.compile);
    eprintln!("test: {:?}", outcome.test);
    assert!(
        matches!(outcome.compile, CompileStatus::Ok),
        "crate must compile: {:?}",
        outcome.compile
    );
    assert!(
        !matches!(outcome.test, TestStatus::Failed(_)),
        "tests must not FAIL: {:?}",
        outcome.test
    );
    let _ = std::fs::remove_dir_all(&dir);
}
