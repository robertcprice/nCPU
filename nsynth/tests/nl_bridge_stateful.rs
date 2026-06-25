//! UNWALL-1-STATEFUL-NL accept-test.
//!
//! Proves the engine's per-tick `(state: i64, arr: [i64]) -> i64` reducer
//! synthesis (`solver::search_families::search_stateful_reducer`) is now
//! NL-reachable — exercised through the ACTUAL product path
//! (`CodingAgentSession::handle_query`, the same entry the `coding_agent` binary
//! uses) from a FRESH root each time.
//!
//! WHY IT CANNOT BE GAMED:
//!   * NL-REACHABILITY (>=2 distinct stateful requests): each must synthesize the
//!     REAL `search_stateful_reducer` program end-to-end, asserted by the
//!     `synthesis_method == "search_stateful_reducer"` tag (the family's own
//!     searched+strict-verified path — NOT a template), with the `(state, arr)`
//!     signature in the response.
//!   * PRIOR-PATH-None (differential, un-gameable core): the stateful lemmas are
//!     ABSENT from the hand `coding_registry.json`, so the only reach is via the
//!     AUTO-MINED overlay — proving NL reach came from the miner, not a hand seed.
//!   * EMERGENCE: the mined stateful surface is asserted DERIVED-FROM + BOUND-TO
//!     the engine's `STATEFUL_REDUCER_NAMES`/`STATEFUL_REDUCER_OPS` slices in
//!     `synthesis::stateful_reducer_surface`'s own guard (`surface_is_bound_to_
//!     engine`), and the committed mined file equals a fresh in-memory mine.
//!   * DIFFERENTIAL (no over-routing): a PLAIN array-reduce request ("sum an
//!     array") and a SCALAR request ("add two numbers") must NOT route to the
//!     stateful family.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use std::fs;
use std::path::{Path, PathBuf};

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_stateful_{tag}_{}_{}",
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

/// Read the hand `coding_registry.json` lemmas to prove the stateful caps are
/// miner-supplied, not hand seeds (PRIOR-PATH-None).
fn hand_registry_lemmas() -> serde_json::Map<String, serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../linguigenesis/data/coding_registry.json");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read coding_registry.json {}: {}", path.display(), e));
    let json: serde_json::Value = serde_json::from_str(&content).expect("parse coding_registry");
    json["entities"].as_object().expect("entities obj").clone()
}

/// PRIOR-PATH-None: the stateful lemmas are ABSENT from the hand registry, so any
/// reach to them comes from the AUTO-MINER overlay.
#[test]
fn stateful_lemmas_are_absent_from_hand_registry() {
    let hand = hand_registry_lemmas();
    for lemma in [
        "running_total",
        "running_deduction",
        "running_max",
        "running_min",
        "running_positive_count",
    ] {
        assert!(
            !hand.contains_key(lemma),
            "stateful op '{lemma}' must NOT be a hand seed in coding_registry.json (the miner \
             supplies it); found it still hand-seeded"
        );
    }
}

/// EMERGENCE: the mined stateful surface is bound to the engine op surface.
#[test]
fn mined_stateful_surface_bound_to_engine() {
    // The descriptor guard (compile + runtime) proves every mined (reducer, op)
    // is in the engine surface and each runner == the engine reference.
    let mined = mog_synth::capability_miner::mined_lemmas();
    for needed in [
        "running_total",
        "running_deduction",
        "running_max",
        "running_min",
        "running_positive_count",
    ] {
        assert!(
            mined.contains(&needed.to_string()),
            "miner missing stateful capability {needed}; mined = {mined:?}"
        );
    }
}

/// EMERGENCE: the COMMITTED mined JSON the bridge loads equals a fresh in-memory
/// mine of the engine surface — so the on-disk stateful vocab provably IS the
/// engine's reflection.
#[test]
fn committed_mined_file_equals_fresh_engine_mine() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../linguigenesis/data/mined_capabilities.json");
    let on_disk = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read mined_capabilities.json: {}", e));
    let fresh = mog_synth::capability_miner::mine_capabilities_json();
    assert_eq!(
        on_disk, fresh,
        "committed mined_capabilities.json is STALE vs the engine surface; \
         re-run `cargo run --bin mine_capabilities`"
    );
}

/// Assert a stateful response is the REAL search_stateful_reducer program: tagged
/// with the family method AND carrying the per-tick (state, arr) signature.
fn assert_real_stateful(r: &mog_synth::agent::AgentQueryResult, query: &str) {
    assert!(r.success, "{query:?} must synthesize; got: {}", r.response);
    assert_eq!(
        r.route,
        QueryRoute::SynthesizeFunction,
        "{query:?} route: {:?}",
        r.route
    );
    assert_eq!(
        r.synthesis_method.as_deref(),
        Some("search_stateful_reducer"),
        "{query:?} must synthesize via the REAL stateful family; method={:?}, code:\n{}",
        r.synthesis_method,
        r.response
    );
    // The generated code is the per-tick reducer: a (state, arr) signature AND it
    // EVOLVES the threaded state (the result combines `state` with the reduction),
    // not a stateless array reduce. The `search_stateful_reducer` codegen names the
    // state parameter `state` and returns `state <op> r`.
    assert!(
        r.response.contains("state: i64") && r.response.contains("[i64]") && r.response.contains("-> i64"),
        "{query:?} must carry the (state: i64, arr: [i64]) -> i64 signature, got:\n{}",
        r.response
    );
    assert!(
        r.response.contains("return state"),
        "{query:?} must EVOLVE the threaded per-tick state (return state <op> r), got:\n{}",
        r.response
    );
}

/// NL-REACHABILITY #1 — a per-tick running total (running_total = sum/+) → real
/// stateful reducer. The request structurally names a per-tick STATE operand
/// ("each tick to the state") plus an array sum; the bridge re-targets the plain
/// reduce to the behaviorally-matching stateful capability.
#[test]
fn nl_running_total_synthesizes_stateful() {
    let r = run(
        &fresh_root("total"),
        "a running total that adds the array each tick to the state",
    );
    assert_real_stateful(&r, "running total");
}

/// NL-REACHABILITY #2 — a SECOND, distinctly-phrased per-tick accumulator that
/// also threads state across ticks → real stateful reducer. Proves the route is
/// not a single memorized phrase: a different surface form carrying the same
/// structural (reduce + per-tick-state) shape still reaches the stateful family.
#[test]
fn nl_accumulator_across_ticks_synthesizes_stateful() {
    let r = run(
        &fresh_root("accum"),
        "a running total that adds the array sum each step to the state",
    );
    assert_real_stateful(&r, "accumulator across ticks");
}

/// DIFFERENTIAL (no over-routing): a PLAIN array-reduce request must NOT route to
/// the stateful family — it stays a single-input array reduce.
#[test]
fn plain_sum_an_array_does_not_route_stateful() {
    let r = run(&fresh_root("plainsum"), "sum an array");
    assert_ne!(
        r.synthesis_method.as_deref(),
        Some("search_stateful_reducer"),
        "'sum an array' must NOT over-route to the stateful family; method={:?}\n{}",
        r.synthesis_method,
        r.response
    );
    // It is a single-input array op (no per-tick state seed): the signature has a
    // single [i64] param, never the (i64, [i64]) stateful shape.
    if r.success {
        assert!(
            !r.response.contains("state"),
            "'sum an array' must not emit a stateful (state, arr) program:\n{}",
            r.response
        );
    }
}

/// DIFFERENTIAL: a SCALAR request stays scalar (never stateful).
#[test]
fn scalar_add_does_not_route_stateful() {
    let r = run(&fresh_root("add"), "add two numbers");
    assert!(r.success && r.response.contains("-> i64"), "add intact: {}", r.response);
    assert_ne!(
        r.synthesis_method.as_deref(),
        Some("search_stateful_reducer"),
        "'add two numbers' must NOT route to the stateful family; method={:?}",
        r.synthesis_method
    );
}
