//! UNWALL-3-REFERENCE-INTAKE-NL end-to-end accept-test.
//!
//! Proves the DIRECT user-supplied-reference path — a request that CARRIES a
//! runnable `fn` block ("behaves like THIS: <fn>") — is NL-reachable through the
//! ACTUAL product path (`CodingAgentSession::handle_query`, the same entry the
//! `coding_agent` binary uses) from a FRESH root each time. The reference's
//! BEHAVIOUR is the whole spec: `reference_nl::classify` extracts the `fn`,
//! `problem_from_reference` manufactures example pairs by RUNNING it (zero human
//! examples), and the synthesized program is strict-verified to differentially
//! agree with the reference on fresh reference-labelled holdouts.
//!
//! Distinct from `cli_p2c_widen.rs`, which covers the COMPOSITIONAL prose path
//! (`classify_compositional` → an internally-emitted reference). This file covers
//! the case where the USER hands over an actual `fn` body.
//!
//! WHY IT CANNOT BE GAMED:
//!   * NO EXAMPLES + NO OP-NAME in the query — only the `fn` body. The example
//!     pairs are auto-manufactured by executing the reference, so a memorized
//!     phrase/example table cannot satisfy it.
//!   * BEHAVIOUR over codegen: the synthesized program must reproduce the
//!     reference's mapping on FRESH inputs (a body of `3x+7` has no NL op name),
//!     checked via `runtime::code_reproduces_examples`.
//!   * DIFFERENTIAL (honest refusal): a request that SIGNALS a reference
//!     ("behaves like this:") but carries NO parseable Rust `fn` must be REFUSED
//!     (fail-closed), never fabricated.
//!   * DIFFERENTIAL (no over-routing): a plain request with neither a `fn` block
//!     nor a reference marker ("add two numbers") must NOT route to the reference
//!     path — it synthesizes normally.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use std::fs;
use std::path::{Path, PathBuf};

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_refintake_{tag}_{}_{}",
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

fn int_case(i: i64, o: i64) -> mog_synth::benchmark::Example {
    mog_synth::benchmark::Example {
        inputs: vec![mog_synth::benchmark::Value::Int(i)],
        expected: mog_synth::benchmark::Value::Int(o),
    }
}

/// NL-REACHABILITY (un-gameable): a user-supplied `fn` reference with an affine
/// body that has NO NL op-name (`3x + 7`) and NO examples in the query must
/// synthesize a program that reproduces the reference on FRESH inputs.
#[test]
fn user_reference_synthesizes_program_agreeing_with_reference() {
    let root = fresh_root("affine");
    let r = run(
        &root,
        "write a function that behaves like this: \
         fn f(x: i64) -> i64 { return x * 3 + 7; }",
    );
    assert!(
        r.success,
        "reference intake must synthesize; got: {}",
        r.response
    );
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    // The spec came ONLY from the reference body (no examples, no op name). The
    // synthesized program must reproduce 3x+7 on inputs the manufacturer may not
    // have sampled — a memorized table could not.
    let spec = [
        int_case(0, 7),
        int_case(1, 10),
        int_case(4, 19),
        int_case(-2, 1),
        int_case(100, 307),
    ];
    assert!(
        mog_synth::runtime::code_reproduces_examples(&r.response, &spec),
        "synthesized program must compute 3x+7 (the reference), got:\n{}",
        r.response
    );
    let _ = fs::remove_dir_all(&root);
}

/// DIFFERENTIAL (honest refusal): a "behaves like this:" marker with NO parseable
/// Rust `fn` must be refused fail-closed, never fabricated into a success.
#[test]
fn unparseable_reference_is_refused_not_fabricated() {
    let root = fresh_root("unparse");
    // Marker present (`behaves like this:`) but the body is Python, not a Rust
    // `fn NAME(params) -> RET { ... }`, so no runnable reference can be extracted.
    let r = run(
        &root,
        "write a function that behaves like this: def f(x): return x + 1",
    );
    assert!(
        !r.success,
        "an unparseable reference must be refused (fail-closed), got success:\n{}",
        r.response
    );
    assert!(
        r.response.to_lowercase().contains("reference"),
        "refusal must cite the reference intake, got:\n{}",
        r.response
    );
    let _ = fs::remove_dir_all(&root);
}

/// DIFFERENTIAL (no over-routing): a plain request with neither a `fn` block nor
/// a reference marker must NOT enter the reference path — it synthesizes normally
/// (i64 add), proving `classify` does not over-trigger.
#[test]
fn plain_request_does_not_route_to_reference() {
    let root = fresh_root("plain");
    let r = run(&root, "add two numbers");
    assert!(r.success, "plain add must still synthesize: {}", r.response);
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    // Did NOT go through the reference intake (that path tags its method
    // `reference-intake-*`); it took the ordinary synthesis lane.
    assert!(
        !r
            .synthesis_method
            .as_deref()
            .unwrap_or("")
            .starts_with("reference-intake"),
        "plain request must not route to reference intake, got method: {:?}",
        r.synthesis_method
    );
    // And it really adds.
    let add_spec = [
        mog_synth::benchmark::Example {
            inputs: vec![
                mog_synth::benchmark::Value::Int(2),
                mog_synth::benchmark::Value::Int(3),
            ],
            expected: mog_synth::benchmark::Value::Int(5),
        },
        mog_synth::benchmark::Example {
            inputs: vec![
                mog_synth::benchmark::Value::Int(6),
                mog_synth::benchmark::Value::Int(9),
            ],
            expected: mog_synth::benchmark::Value::Int(15),
        },
    ];
    assert!(
        mog_synth::runtime::code_reproduces_examples(&r.response, &add_spec),
        "plain synthesized program must add, got:\n{}",
        r.response
    );
    let _ = fs::remove_dir_all(&root);
}
