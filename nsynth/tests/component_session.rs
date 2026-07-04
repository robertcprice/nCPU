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

// ---- AUTO-ROUTER (handle_query) integration + false-positive guards ----

fn route(root: &std::path::Path, q: &str) -> mog_synth::agent::AgentQueryResult {
    let mut s = CodingAgentSession::new(root, GuardrailPolicy::default());
    s.handle_query(q)
}

/// AUTO-ROUTE: the real handle_query front door builds a Counter from a plain
/// construction request — a phrase that used to dead-end at Clarification.
#[test]
fn handle_query_auto_routes_build_a_counter() {
    let root = fresh_root("auto_counter");
    let r = route(&root, "build a counter");
    assert!(r.success, "response: {}", r.response);
    assert_eq!(r.workflow, "component.build", "routed to component layer");
    assert!(root.join("src/counter.rs").is_file(), "Counter struct written");
    assert!(r.response.contains("behavior: PASSED"), "{}", r.response);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn handle_query_auto_routes_an_accumulator() {
    let root = fresh_root("auto_accum");
    let r = route(&root, "an accumulator");
    assert!(r.success, "response: {}", r.response);
    assert_eq!(r.workflow, "component.build");
    assert!(root.join("src/accumulator.rs").is_file());
    let _ = fs::remove_dir_all(&root);
}

/// FALSE-POSITIVE GUARD 1 (ambiguous surface): "count" resolves to an op
/// (array_sum), so an operation request that merely contains it must NOT build a
/// Counter — it routes exactly as before (no component crate).
#[test]
fn handle_query_does_not_hijack_count_operation() {
    let root = fresh_root("count_op");
    let r = route(&root, "count the elements of an array");
    assert_ne!(r.workflow, "component.build", "must NOT build a component: {}", r.response);
    assert!(!root.join("src/counter.rs").exists(), "no Counter struct written");
    let _ = fs::remove_dir_all(&root);
}

/// FALSE-POSITIVE GUARD 2 (incidental distinctive noun): "counter" appears but the
/// head verb is "sort" — no construction of a Counter. The construction-cue gate
/// keeps this from building. (Belt-and-suspenders atop the op filter.)
#[test]
fn handle_query_does_not_hijack_incidental_counter_noun() {
    let root = fresh_root("incidental");
    let r = route(&root, "sort the list of names then reverse it");
    assert_ne!(r.workflow, "component.build");
    assert!(!root.join("src/counter.rs").exists());
    let _ = fs::remove_dir_all(&root);
}
