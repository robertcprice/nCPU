//! UNWALL-CALL-NODE-NL accept-test: an English "B that calls A" request writes a
//! MULTI-FILE crate where component B genuinely CALLS sibling component A.
//!
//! Proves the enumerator-level inter-function Call node (`enumerative::
//! synthesize_scalar_with_callees`, STEP7) is NL-reachable end-to-end through the
//! ACTUAL product path (`CodingAgentSession::handle_query`). lg-core
//! `comprehend_project` emits the emergent dependency edge (`detect_component_deps`:
//! a consumer clause that names a sibling's op AND carries a call/use cue), and the
//! nsynth bridge solves the consumer via the Call-node search
//! (`synthesize_consumer_with_call`), which discovers an `A(x)`-bearing body and
//! strict-verifies it against reference-derived composed examples.
//!
//! WHY IT CANNOT BE GAMED:
//!   * The consumer's `src/negate.rs` must `use crate::square::square;` AND call
//!     `square(` — asserted by the WRITTEN files, so an inlined re-derivation (the
//!     bridge REFUSES those via `body_calls_fn`) or a scaffold cannot satisfy it.
//!   * BEHAVIOUR: the producer `square` is a real `x*x` (verified), and the
//!     consumer composes `0 - square(a)` = `-(x^2)`, the behaviour the strict
//!     verifier accepted (a green multi-component solve means every component
//!     reproduced its reference-derived examples).
//!   * DIFFERENTIAL (no spurious edge): an INDEPENDENT two-function request (no
//!     call/use cue, each clause names only its own op) must NOT wire a
//!     cross-module call — the consumer file carries no `use crate::` of its
//!     sibling. Proves the dependency edge is the structural cue, not a phrase
//!     table that fires on any two-function request.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use std::fs;
use std::path::{Path, PathBuf};

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_callnode_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create root");
    root
}

fn run(root: &Path, query: &str) -> mog_synth::agent::AgentQueryResult {
    let mut session = CodingAgentSession::new(root, GuardrailPolicy::default());
    session.handle_query(query)
}

/// PRIMARY: "B that calls A" writes a crate where the consumer imports + calls the
/// producer.
#[test]
fn consumer_calls_producer_across_modules() {
    let root = fresh_root("calls");
    let r = run(
        &root,
        "a module with a function that squares a number \
         and a function that negates its square using square",
    );
    assert!(r.success, "response: {}", r.response);
    assert_eq!(r.route, QueryRoute::GreenfieldProject);

    // Producer: a real `x*x` square.
    let square_src = fs::read_to_string(root.join("src/square.rs")).unwrap();
    assert!(square_src.contains("pub fn square"), "square is pub: {square_src}");
    assert!(square_src.contains("x * x"), "square is x*x: {square_src}");

    // Consumer: imports AND calls the producer (a genuine cross-module call, not an
    // inlined re-derivation — the bridge refuses inlined results).
    let negate_src = fs::read_to_string(root.join("src/negate.rs")).unwrap();
    assert!(
        negate_src.contains("use crate::square::square"),
        "consumer imports the producer: {negate_src}"
    );
    assert!(
        negate_src.contains("square(") && negate_src.contains("pub fn negate"),
        "consumer calls square(...): {negate_src}"
    );

    // lib.rs wires both modules.
    let lib_src = fs::read_to_string(root.join("src/lib.rs")).unwrap();
    assert!(
        lib_src.contains("mod square;") && lib_src.contains("mod negate;"),
        "lib wires both: {lib_src}"
    );

    let _ = fs::remove_dir_all(&root);
}

/// DIFFERENTIAL (no spurious edge): two INDEPENDENT functions (no call/use cue)
/// must NOT wire a cross-module call — proving the dependency edge is the
/// structural cue, not a fire-on-any-two-functions table.
#[test]
fn independent_siblings_have_no_cross_call() {
    let root = fresh_root("indep");
    let r = run(
        &root,
        "a module with a function that negates a number \
         and a function that squares a number",
    );
    assert!(r.success, "response: {}", r.response);
    assert_eq!(r.route, QueryRoute::GreenfieldProject);

    // Neither component imports the other (independent siblings).
    let negate_src = fs::read_to_string(root.join("src/negate.rs")).unwrap();
    let square_src = fs::read_to_string(root.join("src/square.rs")).unwrap();
    assert!(
        !negate_src.contains("use crate::square") && !square_src.contains("use crate::negate"),
        "independent siblings must not cross-call;\nnegate:\n{negate_src}\nsquare:\n{square_src}"
    );

    let _ = fs::remove_dir_all(&root);
}
