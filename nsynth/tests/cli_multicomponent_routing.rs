//! BUILD-B2-FIX-CLI-MULTICOMP-ROUTING accept-test: the CLI FRONT DOOR
//! (`CodingAgentSession::handle_query`, the exact entry the `coding_agent` binary
//! calls) must route a DESCRIBED multi-component program to the
//! `GreenfieldProject` multi-file door — NOT swallow it in the single-function
//! `classify_compositional` intercept.
//!
//! WHY THE BUG WAS CLI-ONLY (and why these tests drive `handle_query`, not
//! `synthesize_project` directly): the multi-component decomposition itself
//! already worked when `synthesize_project` was called directly (proved by
//! `nl_multicomponent_decomp.rs`). The defect lived purely in the CLI ENTRY
//! precedence — `handle_query` ran the single-function compositional intercept
//! (`classify_compositional`, which splits the WHOLE string on every `then`)
//! BEFORE the multi-component `synthesize_project` route, mashing two components
//! into one nonsense `max_then_triple_then_increment` chain. So the regression
//! can ONLY be pinned by driving `handle_query`.
//!
//! WHY IT CANNOT BE GAMED (differential, both directions):
//!   * A described MULTI-component request → route `GreenfieldProject`, a multi-file
//!     crate with >=2 component module files + lib.rs, `success == true` (which in
//!     `write_multifile_program` requires `CompileStatus::Ok` — the crate `cargo
//!     check`s clean), and the bytes of each component are correct BY HAND
//!     (max(3,7)*3 == 21, |-5|+1 == 6).
//!   * The SINGLE compositional request that legitimately uses the intercept
//!     ("the larger of two numbers then triples it") still routes to
//!     `SynthesizeFunction` (single-fn P2C), NOT `GreenfieldProject`. A fix that
//!     blanket-routed everything to the project door would FAIL this; the bug
//!     (route everything through the single intercept) FAILS the multi-component
//!     assertion. Only a guard that DISCRIMINATES on the structural
//!     component-head signal passes both.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use mog_synth::runtime::{execute_function, Value as RVal};
use mog_synth::benchmark::Value as BVal;
use std::path::PathBuf;

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_climc_{tag}_{}_{}",
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

/// PRIMARY (un-gameable): the multi-component request, driven through the REAL CLI
/// entry `handle_query`, routes to `GreenfieldProject`, writes a compiling
/// multi-file crate (>=2 component module files + lib.rs), and each component is
/// correct by hand.
#[test]
fn cli_multicomponent_request_routes_to_greenfield_multifile() {
    let root = fresh_root("multi");
    let mut session = session_at(&root);
    let request = "a module with a function that returns the larger of two numbers \
                   then triples it, and a function that returns the absolute value of \
                   a number then increments it";
    let result = session.handle_query(request);
    eprintln!(
        "[CLI-MC] route={:?} success={} response=\n{}",
        result.route, result.success, result.response
    );

    // ROUTE: the bug routed this to SynthesizeFunction (single intercept). Fixed →
    // GreenfieldProject.
    assert_eq!(
        result.route,
        QueryRoute::GreenfieldProject,
        "multi-component request must route to GreenfieldProject, got {:?}",
        result.route
    );

    // COMPILE GATE: in write_multifile_program, success == true ONLY when the
    // written crate's `cargo check` is CompileStatus::Ok. So this asserts the
    // multi-file crate COMPILES.
    assert!(
        result.success,
        "multi-file crate must compile clean (success requires CompileStatus::Ok); response=\n{}",
        result.response
    );

    // >=2 component module files + lib.rs were written to the sandbox root.
    let src = root.join("src");
    let modules: Vec<String> = std::fs::read_dir(&src)
        .expect("src dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".rs") && n != "lib.rs")
        .collect();
    eprintln!("[CLI-MC] module files = {modules:?}");
    assert!(
        modules.len() >= 2,
        ">=2 component module files must be written, got {modules:?}"
    );
    assert!(src.join("lib.rs").exists(), "lib.rs must wire the components");

    // BY-HAND GRADER over the written component source: identify each component by
    // behaviour (NOT name) and grade max(a,b)*3 and |x|+1 independently.
    let mut max_triple_ok = false;
    let mut abs_inc_ok = false;
    for m in &modules {
        let raw = std::fs::read_to_string(src.join(m)).expect("read module");
        // The writer prepends a `//!` module doc header; the Mog `parse_program`
        // the grader runs on the source wants the bare fn, so drop the leading
        // comment/blank lines.
        let code: String = raw
            .lines()
            .skip_while(|l| {
                let t = l.trim_start();
                t.starts_with("//") || t.is_empty()
            })
            .collect::<Vec<_>>()
            .join("\n")
            // The writer exposes each fn as `pub fn`; the Mog `parse_program` the
            // grader uses wants the bare `fn`. Strip the visibility modifier.
            .replace("pub fn", "fn");
        let fn_name = m.trim_end_matches(".rs");
        // 2-arg probe: max(3,7)*3 == 21, max(9,2)*3 == 27.
        if let Ok(RVal::Int(v)) =
            execute_function(&code, fn_name, &[BVal::Int(3), BVal::Int(7)], "probe")
        {
            if v == 21 {
                assert!(
                    matches!(
                        execute_function(&code, fn_name, &[BVal::Int(9), BVal::Int(2)], "probe"),
                        Ok(RVal::Int(27))
                    ),
                    "max(9,2)*3 must be 27"
                );
                max_triple_ok = true;
                continue;
            }
        }
        // 1-arg probe: |-5|+1 == 6, |3|+1 == 4.
        if let Ok(RVal::Int(v)) = execute_function(&code, fn_name, &[BVal::Int(-5)], "probe") {
            if v == 6 {
                assert!(
                    matches!(
                        execute_function(&code, fn_name, &[BVal::Int(3)], "probe"),
                        Ok(RVal::Int(4))
                    ),
                    "|3|+1 must be 4"
                );
                abs_inc_ok = true;
            }
        }
    }
    assert!(max_triple_ok, "a max(a,b)*3 component must be present and correct");
    assert!(abs_inc_ok, "an |x|+1 component must be present and correct");

    if std::env::var("NSYNTH_KEEP_CRATE").is_err() {
        let _ = std::fs::remove_dir_all(&root);
    }
}

/// DIFFERENTIAL (un-gameable, opposite direction): the SINGLE compositional request
/// that legitimately uses the single-fn intercept still routes to
/// `SynthesizeFunction` (single-fn P2C), NOT to the multi-file `GreenfieldProject`
/// door. This proves the guard DISCRIMINATES on the structural component-head
/// signal rather than blanket-routing.
#[test]
fn cli_single_compositional_request_still_routes_single_fn() {
    let root = fresh_root("single");
    let mut session = session_at(&root);
    let result = session.handle_query("the larger of two numbers then triples it");
    eprintln!("[CLI-SINGLE] route={:?} success={}", result.route, result.success);
    assert_ne!(
        result.route,
        QueryRoute::GreenfieldProject,
        "single compositional must NOT route to the multi-file door"
    );
    assert_eq!(
        result.route,
        QueryRoute::SynthesizeFunction,
        "single compositional must route to the single-fn P2C path, got {:?}",
        result.route
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// DIFFERENTIAL: an existing registry-op multi-file request (single ops, no `then`)
/// is UNCHANGED — still routes to `GreenfieldProject` and writes a multi-file crate.
#[test]
fn cli_registry_op_multifile_request_unchanged() {
    let root = fresh_root("regop");
    let mut session = session_at(&root);
    let result = session.handle_query(
        "a module with a function that negates a number and a function that triples a number",
    );
    eprintln!("[CLI-REGOP] route={:?} success={}", result.route, result.success);
    assert_eq!(
        result.route,
        QueryRoute::GreenfieldProject,
        "registry-op multi-file request must stay GreenfieldProject, got {:?}",
        result.route
    );
    let src = root.join("src");
    let modules = std::fs::read_dir(&src)
        .expect("src dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.ends_with(".rs") && n != "lib.rs"
        })
        .count();
    assert!(modules >= 2, "registry-op multi-file must write >=2 component files");
    let _ = std::fs::remove_dir_all(&root);
}

/// DIFFERENTIAL: a single op is unchanged — routes to `SynthesizeFunction`.
#[test]
fn cli_single_op_request_unchanged() {
    let root = fresh_root("singleop");
    let mut session = session_at(&root);
    let result = session.handle_query("add two numbers");
    eprintln!("[CLI-SINGLEOP] route={:?} success={}", result.route, result.success);
    assert_ne!(
        result.route,
        QueryRoute::GreenfieldProject,
        "a single op must not route to the multi-file door"
    );
    let _ = std::fs::remove_dir_all(&root);
}
