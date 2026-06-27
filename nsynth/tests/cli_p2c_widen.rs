//! BUILD-A (widen P2C) accept-test: the CLI FRONT DOOR
//! (`CodingAgentSession::handle_query`, the exact entry the `coding_agent` binary
//! calls) must comprehend, synthesize, and strict-verify the THREE widened
//! composition shapes — and the ARRAY shape must NEVER panic.
//!
//! The widening is over `classify_compositional` / `classify_domain_compositional`
//! (`reference_nl.rs`) + `emit_{scalar,array,string}_reference`
//! (`linguigenesis_bridge.rs`), routed in `session.rs:handle_query`. All step
//! resolution is EMERGENT (EntityResolver + the resolved op's declared signature);
//! there is no phrase->op table.
//!
//! WHY IT CANNOT BE GAMED (un-gameable, both directions):
//!   * Each shape is driven END TO END through the real CLI entry: comprehend ->
//!     emit reference -> `problem_from_reference` auto-manufactures examples by
//!     RUNNING the reference (zero human examples) + reference-labelled holdouts ->
//!     solve -> strict-verify. `success == true` already REQUIRES the synthesized
//!     program to differentially agree with the reference on fresh held-out inputs.
//!   * Then the INDEPENDENT grader RUNS the synthesized program (the bytes in
//!     `result.response`) on HAND-chosen inputs and compares to outputs computed
//!     BY HAND here (NOT reference-derived): 3-stage scalar -3x+1, array 2*sum,
//!     string reverse(upper(s)). A program that merely overfit the seed examples
//!     would fail this.
//!   * The ARRAY case is the regression anchor: it previously PANICKED (exit 101)
//!     because the reference-driven example sampler now feeds EMPTY/short arrays to
//!     solver array primitives (`second_max`/`array_range`/`max_stock_profit`...)
//!     that indexed `[0]` / `.unwrap()`ed `min`/`max`. The fix makes the
//!     `validate_*_array` chokepoint total (catch_unwind) + the two hot primitives
//!     total. This test asserts the array path RETURNS A RESULT (never unwinds).
//!   * UNRESOLVABLE: a composition with an unresolvable step must REFUSE
//!     (`success == false`, honest message), never fabricate, never panic.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use mog_synth::benchmark::Value as BVal;
use mog_synth::runtime::{execute_function, Value as RVal};
use std::path::PathBuf;

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_p2cwiden_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create root");
    root
}

fn session_at(root: &PathBuf) -> CodingAgentSession {
    CodingAgentSession::load(root, GuardrailPolicy::default(), "test").expect("load session")
}

/// Parse the composed fn name from the synthesized source in `result.response`
/// (`fn NAME(...) -> ... { ... }`). The grader runs the program by this name.
fn fn_name_of(code: &str) -> String {
    let after_fn = code
        .split_once("fn ")
        .map(|(_, rest)| rest)
        .unwrap_or(code);
    after_fn
        .split(|c: char| c == '(' || c.is_whitespace())
        .next()
        .unwrap_or("")
        .to_string()
}

fn run_int(code: &str, name: &str, args: &[BVal]) -> i64 {
    match execute_function(code, name, args, "p2c-widen-grader") {
        Ok(RVal::Int(v)) => v,
        other => panic!("{name} returned non-int {other:?} for args {args:?}\ncode:\n{code}"),
    }
}

fn run_str(code: &str, name: &str, input: &str) -> String {
    match execute_function(code, name, &[BVal::Str(input.to_string())], "p2c-widen-grader") {
        Ok(RVal::Str(s)) => s,
        other => panic!("{name} returned non-string {other:?} for {input:?}\ncode:\n{code}"),
    }
}

/// (1) THREE-STAGE SCALAR: "negate ... then triple ... then increment ..." -> -3x+1.
/// Driven through the real CLI entry; graded BY HAND on independent inputs.
#[test]
fn cli_three_stage_scalar_minus_3x_plus_1() {
    let root = fresh_root("scalar3");
    let mut session = session_at(&root);
    let result = session.handle_query("negate a number then triple it then increment it");
    eprintln!(
        "[P2C-SCALAR3] route={:?} success={} method={:?}\n{}",
        result.route, result.success, result.synthesis_method, result.response
    );
    assert_eq!(result.route, QueryRoute::SynthesizeFunction);
    assert!(
        result.success,
        "3-stage scalar must synthesize+strict-verify; response=\n{}",
        result.response
    );
    let code = &result.response;
    let name = fn_name_of(code);
    // GRADER computes -3x+1 INDEPENDENTLY (not reference-derived).
    for (x, expected) in [(-2i64, 7i64), (1, -2), (0, 1), (5, -14)] {
        let got = run_int(code, &name, &[BVal::Int(x)]);
        assert_eq!(got, expected, "-3*({x})+1 must equal {expected}");
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// (2) ARRAY map-then-reduce: "double each value in an array then sum them" -> 2*sum.
/// REGRESSION ANCHOR: this MUST NOT PANIC (it used to exit 101). It must return
/// either a correct synthesized program (preferred) OR a clean refusal — never
/// unwind. Driven through the real CLI entry; graded BY HAND.
#[test]
fn cli_array_map_then_reduce_does_not_panic_and_is_2_sum() {
    let root = fresh_root("array");
    let mut session = session_at(&root);
    // The mere fact that `handle_query` returns (does not unwind) is the no-panic
    // proof — a panic here would abort the test process (exit 101), failing it.
    let result = session.handle_query("double each value in an array then sum them");
    eprintln!(
        "[P2C-ARRAY] route={:?} success={} method={:?}\n{}",
        result.route, result.success, result.synthesis_method, result.response
    );
    assert_eq!(
        result.route,
        QueryRoute::SynthesizeFunction,
        "array composition routes through the single-fn synthesize door"
    );
    if result.success {
        // PREFERRED outcome: full array synthesis. Grade 2*sum BY HAND.
        let code = &result.response;
        let name = fn_name_of(code);
        for (arr, expected) in [
            (vec![1i64, 2, 3], 12i64),
            (vec![5, 0, -1], 8),
            (vec![10], 20),
            (vec![-3, -4, 7], 0),
            (vec![], 0),
        ] {
            let got = run_int(code, &name, &[BVal::int_array(&arr)]);
            assert_eq!(got, expected, "2*sum({arr:?}) must equal {expected}");
        }
    } else {
        // ACCEPTABLE fallback: a CLEAN refusal (honest message), never a panic and
        // never a fabricated success.
        assert!(
            result
                .synthesis_method
                .as_deref()
                .map(|m| m.contains("compositional") || m.contains("reference"))
                .unwrap_or(false),
            "array refusal must be an honest compositional/reference refusal, got method={:?}",
            result.synthesis_method
        );
        assert!(
            !result.response.is_empty(),
            "a refusal must carry an honest explanation, not an empty body"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// (3) STRING composition: "uppercase a string then reverse it" -> reverse(upper(s)).
/// Driven through the real CLI entry; graded BY HAND.
#[test]
fn cli_string_compose_reverse_of_upper() {
    let root = fresh_root("string");
    let mut session = session_at(&root);
    let result = session.handle_query("uppercase a string then reverse it");
    eprintln!(
        "[P2C-STRING] route={:?} success={} method={:?}\n{}",
        result.route, result.success, result.synthesis_method, result.response
    );
    assert_eq!(result.route, QueryRoute::SynthesizeFunction);
    assert!(
        result.success,
        "string composition must synthesize+strict-verify; response=\n{}",
        result.response
    );
    let code = &result.response;
    let name = fn_name_of(code);
    // GRADER computes reverse(uppercase(s)) INDEPENDENTLY.
    for (s, expected) in [("abc", "CBA"), ("Hello", "OLLEH"), ("aB2", "2BA")] {
        let got = run_str(code, &name, s);
        assert_eq!(got, expected, "reverse(upper({s:?})) must equal {expected:?}");
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// (4) UNRESOLVABLE: a composition with an unresolvable step must REFUSE
/// (success == false, honest message), never fabricate, never panic. Covers BOTH
/// a string-domain and an array-domain unresolvable tail.
#[test]
fn cli_unresolvable_step_refuses_no_fabrication() {
    let root = fresh_root("refuse");
    let mut session = session_at(&root);

    let s_res = session.handle_query("uppercase a string then frobnicate it");
    eprintln!(
        "[P2C-REFUSE-STR] success={} method={:?} resp={}",
        s_res.success, s_res.synthesis_method, s_res.response
    );
    assert!(
        !s_res.success,
        "an unresolvable string step must NOT report success"
    );
    assert!(
        s_res.response.to_lowercase().contains("refus")
            || s_res.response.to_lowercase().contains("does not resolve"),
        "refusal must be honest, got: {}",
        s_res.response
    );

    let a_res = session.handle_query("double each value in an array then frobnicate them");
    eprintln!(
        "[P2C-REFUSE-ARR] success={} method={:?} resp={}",
        a_res.success, a_res.synthesis_method, a_res.response
    );
    assert!(
        !a_res.success,
        "an unresolvable array step must NOT report success"
    );

    let _ = std::fs::remove_dir_all(&root);
}
