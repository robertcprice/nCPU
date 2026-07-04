//! Component layer reached through the REAL session object (CodingAgentSession),
//! the same type the coding_agent binary drives. Proves an NL phrase produces a
//! verified component crate on disk via the product entry point, not a unit stub.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy};
use std::fs;
use std::path::PathBuf;

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_comp_session_{tag}_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create root");
    root
}

/// PRIMARY: "build a counter" -> the real session writes a verified crate with the
/// synthesized increment leaf + the Counter struct, and reports behavioral PASS.
#[test]
fn session_builds_a_counter_component_from_nl() {
    let root = fresh_root("counter");
    let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let result = session
        .try_build_components("build a counter")
        .expect("counter is a known component");
    assert!(result.success, "response: {}", result.response);
    assert_eq!(result.workflow, "component.build");
    // real files on disk
    assert!(root.join("src/counter.rs").is_file(), "Counter struct written");
    assert!(root.join("src/increment.rs").is_file(), "increment leaf written");
    let glue = fs::read_to_string(root.join("src/counter.rs")).unwrap();
    assert!(glue.contains("pub struct Counter"), "struct present: {glue}");
    // the trace shows compile + behavior gates both ran clean
    let traces: Vec<&str> = result.tool_trace.iter().map(|(k, _)| k.as_str()).collect();
    assert!(traces.contains(&"cargo.check"), "compile gate ran: {traces:?}");
    assert!(traces.contains(&"cargo.test"), "behavior gate ran: {traces:?}");
    assert!(result.response.contains("behavior: PASSED"), "{}", result.response);
    let _ = fs::remove_dir_all(&root);
}

/// MULTI: one phrase naming two concepts -> one crate with both.
#[test]
fn session_builds_multi_component_project_from_nl() {
    let root = fresh_root("multi");
    let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let result = session
        .try_build_components("a counter and some array statistics")
        .expect("both known");
    assert!(result.success, "response: {}", result.response);
    assert!(root.join("src/counter.rs").is_file());
    assert!(root.join("src/increment.rs").is_file());
    // an array_stats leaf is present too
    assert!(
        root.join("src/array_sum.rs").is_file() || root.join("src/average.rs").is_file(),
        "a stats leaf written"
    );
    let _ = fs::remove_dir_all(&root);
}

/// NEGATIVE: a phrase naming no known component returns None (caller falls back).
#[test]
fn session_declines_unknown_component() {
    let root = fresh_root("none");
    let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
    assert!(session.try_build_components("reverse an array").is_none());
    let _ = fs::remove_dir_all(&root);
}
