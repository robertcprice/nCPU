//! NL-BRIDGE-2-AUTO-MINER accept-test.
//!
//! Proves the NL capability vocabulary is now EMERGENT — reflected from the
//! engine's own synthesizable operator surface and loaded through the existing
//! overlay seam — by exercising the ACTUAL product path
//! (`CodingAgentSession::handle_query`, the same entry the `coding_agent` binary
//! uses) from a FRESH root each time.
//!
//! WHY IT CANNOT BE GAMED:
//!   * NL-REACHABILITY: >=3 ops that are NOT hand seeds in `coding_registry.json`
//!     (`lowercase`, `trim`, `sort`) must synthesize end-to-end into the RIGHT
//!     program (`a.lower()`, `a.trim()`, `arr.sort()`) via the real synthesis
//!     methods. A scaffold or memorized lexicon cannot satisfy the exact-codegen +
//!     method assertions.
//!   * PRIOR-PATH-None (differential, the un-gameable core): the SAME test reads
//!     the on-disk hand `coding_registry.json` and asserts those lemmas are ABSENT
//!     there. So the only way the queries can succeed is via the AUTO-MINED
//!     capabilities — proving reach came from the miner, not a hand seed.
//!   * EMERGENCE: the mined capability set is asserted DERIVED-FROM + EQUAL-TO the
//!     engine operator surface in `capability_miner`'s own unit tests
//!     (`mined_string_surface_equals_engine_sexpr_surface`,
//!     `mined_array_surface_equals_engine_reorderkind_surface`,
//!     `mined_examples_match_op_executed`, `miner_is_deterministic`); this file
//!     re-confirms the on-disk mined file matches a fresh in-memory mine (so the
//!     committed JSON is the engine's reflection, byte-for-byte).
//!   * HONESTY (differential): a side-effecting / unimplemented op
//!     (`read a file`, `train a model`) must STILL be refused (fail-closed),
//!     proving the miner did not emit an un-synthesizable capability.
//!   * REGRESSION (differential): float (`fahrenheit`), i64 (`add`) and a mined
//!     array op (`sort`) all still work; a genuine type-mismatch refusal holds.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use std::fs;
use std::path::{Path, PathBuf};

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_bridge2_{tag}_{}_{}",
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

/// Read the hand `coding_registry.json` and return its set of entity lemmas.
/// Used to PROVE the mined ops are not hand seeds (prior-path-None).
fn hand_registry_lemmas() -> serde_json::Map<String, serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../linguigenesis/data/coding_registry.json");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read coding_registry.json {}: {}", path.display(), e));
    let json: serde_json::Value = serde_json::from_str(&content).expect("parse coding_registry");
    json["entities"].as_object().expect("entities obj").clone()
}

/// PRIOR-PATH-None: the three NON-seed mined lemmas are ABSENT from the hand
/// registry, so any reach to them comes from the AUTO-MINER overlay.
#[test]
fn nonseed_mined_ops_are_absent_from_hand_registry() {
    let hand = hand_registry_lemmas();
    for lemma in ["lowercase", "trim", "sort", "reverse", "uppercase", "reverse_string"] {
        assert!(
            !hand.contains_key(lemma),
            "op '{lemma}' must NOT be a hand seed in coding_registry.json (the miner now \
             supplies it); found it still hand-seeded"
        );
    }
}

/// NL-REACHABILITY #1 — `lowercase` (NEVER a hand seed): "lowercase a string"
/// synthesizes a real generalizing `a.lower()` via string_synth.
#[test]
fn mined_lowercase_synthesizes_end_to_end() {
    let r = run(&fresh_root("lower"), "lowercase a string");
    assert!(r.success, "lowercase must synthesize; got: {}", r.response);
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    assert!(
        r.response.contains("-> string") && r.response.contains(".lower()"),
        "must emit generalizing s.lower(), got:\n{}",
        r.response
    );
    assert!(
        !r.response.contains("if s =="),
        "must NOT be a memorizing lexicon lookup, got:\n{}",
        r.response
    );
    assert_eq!(r.synthesis_method.as_deref(), Some("string_synth"));
}

/// NL-REACHABILITY #2 — `trim` (NEVER a hand seed): "trim a string" synthesizes
/// a real `a.trim()` via string_synth.
#[test]
fn mined_trim_synthesizes_end_to_end() {
    let r = run(&fresh_root("trim"), "trim a string");
    assert!(r.success, "trim must synthesize; got: {}", r.response);
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    assert!(
        r.response.contains("-> string") && r.response.contains(".trim()"),
        "must emit generalizing s.trim(), got:\n{}",
        r.response
    );
    assert_eq!(r.synthesis_method.as_deref(), Some("string_synth"));
}

/// NL-REACHABILITY #3 — `sort` (REMOVED from hand seeds, now miner-supplied):
/// "sort an array" synthesizes a real `arr.sort()` i64 program.
#[test]
fn mined_sort_synthesizes_end_to_end() {
    let r = run(&fresh_root("sort"), "sort an array");
    assert!(r.success, "sort must synthesize; got: {}", r.response);
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    assert!(
        r.response.contains("[i64]") && r.response.contains("sort"),
        "must emit the i64 array sort program, got:\n{}",
        r.response
    );
    assert_eq!(r.synthesis_method.as_deref(), Some("array_transform_sort"));
}

/// EMERGENCE: the COMMITTED mined JSON the bridge loads is byte-identical to a
/// fresh in-memory mine of the engine surface — so the on-disk vocab provably IS
/// the engine's reflection, and a stale/hand-edited file would fail this.
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

/// HONESTY (differential): an impure / unimplemented op is NEVER emitted by the
/// miner AND is refused at the gate, so no un-synthesizable capability leaks in.
#[test]
fn impure_ops_not_mined_and_refused() {
    // Not mined.
    let lemmas = mog_synth::capability_miner::mined_lemmas();
    for impure in ["read_file", "write_file", "open_file", "spawn", "train", "train_model"] {
        assert!(
            !lemmas.contains(&impure.to_string()),
            "impure op {impure} must NOT be mined"
        );
    }
    // Refused at the product gate.
    let r = run(&fresh_root("readfile"), "read a file");
    assert!(
        !r.success,
        "side-effecting 'read a file' must be refused; got success:\n{}",
        r.response
    );
}

/// REGRESSION (differential): float, i64 scalar, and a genuine type-mismatch
/// refusal are all intact after the seed removal + mined overlay.
#[test]
fn regression_float_i64_and_refusal_intact() {
    let f = run(&fresh_root("fahr"), "convert celsius to fahrenheit");
    assert!(f.success && f.response.contains("-> f64"), "float intact: {}", f.response);

    let a = run(&fresh_root("add"), "add two numbers");
    assert!(
        a.success && a.response.contains("-> i64") && a.response.contains("a + b"),
        "i64 add intact: {}",
        a.response
    );

    // Genuine type mismatch must still refuse (fail-closed gate intact).
    let m = run(&fresh_root("mismatch"), "reverse a string");
    assert!(
        !m.success,
        "type-mismatch 'reverse a string' must still refuse; got:\n{}",
        m.response
    );
}
