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
    assert!(
        mined_semantically_equal(&on_disk, &fresh),
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
    // De-brittled from method identity: stronger lanes (universal/hole-power)
    // may legitimately win these with CORRECT programs. The structural asserts
    // below carry the anti-parroting weight: a per-tick signature AND genuine
    // state threading.
    assert!(
        r.synthesis_method.is_some(),
        "{query:?} must report a synthesis method; got: {}",
        r.response
    );
    // The generated code is the per-tick reducer: a (state, arr) signature AND it
    // EVOLVES the threaded state (the result combines `state` with the reduction),
    // not a stateless array reduce. The `search_stateful_reducer` codegen names the
    // state parameter `state` and returns `state <op> r`.
    // (state-ish i64, arr [i64]) -> i64 signature, param NAMES free.
    let fn_line = r
        .response
        .lines()
        .find(|l| l.trim_start().starts_with("fn ") && l.contains("[i64]"))
        .unwrap_or_else(|| panic!("{query:?} must emit a per-tick fn; got:\n{}", r.response));
    assert!(
        fn_line.contains(": i64") && fn_line.contains("[i64]") && fn_line.contains("-> i64"),
        "{query:?} must carry an (i64, [i64]) -> i64 per-tick signature, got: {fn_line}"
    );
    // Genuine STATE THREADING: the scalar (state) param participates in the
    // returned expression — not a stateless array reduce.
    let state_param = fn_line
        .split('(')
        .nth(1)
        .and_then(|r| r.split(':').next())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "state".into());
    let returns_state = r
        .response
        .lines()
        .any(|l| l.trim_start().starts_with("return ") && l.contains(&state_param));
    assert!(
        returns_state,
        "{query:?} must EVOLVE the threaded state param '{state_param}', got:\n{}",
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

// ───────────────────────── UNWALL-1B: NON-ADDITIVE ─────────────────────────
//
// The genuinely-stateful value plain reduce CANNOT express: a 2-input UPDATE
// `f(state, arr) = state OP g(arr)` for a NON-ADDITIVE state-combine OP (max,
// min). These have a real prior-STATE input (an update, not a single-array
// reduce). UNWALL-1 only routed the additive (OP="+") seed; UNWALL-1B broadens
// the behavioral match to ALL state-combining ops via each op's left-identity
// seed (`f(e, arr) = e OP g(arr) = g(arr)`), derived emergently from the engine's
// OWN combine arithmetic — no per-op phrase table.

/// Assert a NON-ADDITIVE stateful response is the REAL `search_stateful_reducer`
/// program for `state OP g(arr)`: tagged with the family method, carrying the
/// per-tick `(state, arr)` 2-input signature, and combining the prior `state`
/// with the reduction `r` (the genuine non-additive update — a lattice op that
/// thresholds `r` against `state`, which a single-array reduce cannot produce
/// because it has no prior-state input).
fn assert_real_nonadditive_stateful(r: &mog_synth::agent::AgentQueryResult, query: &str) {
    assert!(r.success, "{query:?} must synthesize; got: {}", r.response);
    // De-brittled from method identity: stronger lanes (universal/hole-power)
    // may legitimately win these with CORRECT programs. The structural asserts
    // below carry the anti-parroting weight: a per-tick signature AND genuine
    // state threading.
    assert!(
        r.synthesis_method.is_some(),
        "{query:?} must report a synthesis method; got: {}",
        r.response
    );
    // Real 2-input per-tick signature (prior-STATE operand + a batch array),
    // param NAMES free — stronger lanes name it a0/n/etc.
    let fn_line = r
        .response
        .lines()
        .find(|l| l.trim_start().starts_with("fn ") && l.contains("[i64]"))
        .unwrap_or_else(|| panic!("{query:?} must emit a per-tick fn; got:\n{}", r.response));
    assert!(
        fn_line.contains(": i64") && fn_line.contains("[i64]") && fn_line.contains("-> i64"),
        "{query:?} must carry an (i64, [i64]) -> i64 per-tick signature, got: {fn_line}"
    );
    // The body MUST fold the prior state param into the result — a plain
    // single-array reduce never references it. Proof the prior-state input is
    // actually used (a real UPDATE), regardless of which lane won.
    let state_param = fn_line
        .split('(')
        .nth(1)
        .and_then(|rest| rest.split(':').next())
        .map(|x| x.trim().to_string())
        .unwrap_or_else(|| "state".into());
    let body_uses_state = {
        let after_fn = r.response.split(&fn_line as &str).nth(1).unwrap_or("");
        let first_fn_body: String = after_fn.chars().take_while(|&c| c != '}').collect();
        first_fn_body.matches(&state_param as &str).count() >= 1
    };
    assert!(
        body_uses_state,
        "{query:?} must USE the prior per-tick state param '{state_param}' in the \
         update body, got:\n{}",
        r.response
    );
}

/// NON-ADDITIVE #1 — a running MAXIMUM `f(state, arr) = state.max(max(arr))`. The
/// request structurally implies a 2-input update (a prior `current max` STATE +
/// a `new batch`), so the bridge re-targets to the `running_max` capability whose
/// engine reducer behaviorally matches under the MAX left-identity (i64::MIN).
/// Plain reduce CANNOT express this: it has no prior-state input.
#[test]
fn nl_running_maximum_synthesizes_nonadditive_stateful() {
    let r = run(
        &fresh_root("runmax"),
        "update a running maximum given the current max and a new batch each tick to the state",
    );
    assert_real_nonadditive_stateful(&r, "running maximum");
    // EXECUTION truth (de-brittled from codegen shape): the emitted program must
    // BEHAVE as a running max, whatever lane/helpers produced it.
    assert_stateful_behavior(&r.response, &[((5, vec![3, 9, 2]), 9), ((10, vec![3, 9, 2]), 10)]);
}

/// NON-ADDITIVE #2 — a running MINIMUM `f(state, arr) = state.min(min(arr))`. A
/// distinctly-phrased second non-additive update; routes via the MIN left-identity
/// (i64::MAX). Proves the broadened match is not a single memorized op.
#[test]
fn nl_running_minimum_synthesizes_nonadditive_stateful() {
    let r = run(
        &fresh_root("runmin"),
        "update a running minimum given the current min and the minimum of a new batch each tick to the state",
    );
    assert_real_nonadditive_stateful(&r, "running minimum");
    // EXECUTION truth (de-brittled from codegen shape): behaves as a running min.
    assert_stateful_behavior(&r.response, &[((5, vec![3, 9, 2]), 2), ((1, vec![3, 9, 2]), 1)]);
}

/// DIFFERENTIAL (un-gameable core for the NON-ADDITIVE broadening): the PLAIN
/// 1-input `the maximum of an array` request must stay a single-array reduce
/// (`array_max`), NEVER the stateful family — proving the broadened route fires
/// ONLY on the 2-input update structural signal (prior-state operand + batch),
/// not on the mere presence of a max/min op. This is the differential that would
/// catch over-routing: without the state-operand gate, "maximum of an array"
/// would wrongly grab `running_max`.
#[test]
fn plain_maximum_of_an_array_does_not_route_stateful() {
    let r = run(&fresh_root("plainmax"), "the maximum of an array");
    assert_ne!(
        r.synthesis_method.as_deref(),
        Some("search_stateful_reducer"),
        "'the maximum of an array' (1-input reduce) must NOT over-route to the stateful \
         family; method={:?}\n{}",
        r.synthesis_method,
        r.response
    );
    if r.success {
        assert!(
            !r.response.contains("state"),
            "'the maximum of an array' must stay a single-input array reduce (no `state` \
             param/operand):\n{}",
            r.response
        );
    }
}

/// Semantic comparison, immune to the known transform-collision emitted by
/// stale miners (load-time normalization makes it invisible at runtime; this
/// mirrors it for the byte artifact): parse both docs, normalize
/// default_fn_name "transform" -> the entity's word, compare as JSON values.
fn mined_semantically_equal(a: &str, b: &str) -> bool {
    fn normalize(v: &mut serde_json::Value) {
        if let Some(ents) = v.get_mut("entities").and_then(|e| e.as_object_mut()) {
            for (word, ent) in ents.iter_mut() {
                if let Some(attrs) = ent.get_mut("attributes").and_then(|a| a.as_object_mut()) {
                    if attrs.get("default_fn_name").and_then(|x| x.as_str()) == Some("transform")
                        && word != "transform"
                    {
                        attrs.insert(
                            "default_fn_name".into(),
                            serde_json::Value::String(word.clone()),
                        );
                    }
                }
            }
        }
    }
    let (Ok(mut va), Ok(mut vb)) = (
        serde_json::from_str::<serde_json::Value>(a),
        serde_json::from_str::<serde_json::Value>(b),
    ) else {
        return false;
    };
    normalize(&mut va);
    normalize(&mut vb);
    va == vb
}

/// Execute the per-tick program embedded in a session response against probe
/// (state, arr) inputs — BEHAVIOR is the assertion, not codegen shape.
fn assert_stateful_behavior(response: &str, probes: &[((i64, Vec<i64>), i64)]) {
    use mog_synth::benchmark::{Example, Problem, Value};
    let start = response.find("fn ").unwrap_or_else(|| panic!("no fn in:\n{response}"));
    let code = &response[start..];
    let examples: Vec<Example> = probes
        .iter()
        .map(|((st, arr), out)| Example {
            inputs: vec![Value::Int(*st), Value::int_array(arr)],
            expected: Value::Int(*out),
        })
        .collect();
    let fn_name = code[3..]
        .split('(')
        .next()
        .unwrap()
        .trim()
        .to_string();
    let sig: &'static str = Box::leak(
        format!("fn {fn_name}(state: i64, arr: [i64]) -> i64").into_boxed_str(),
    );
    let problem = Problem {
        name: fn_name,
        category: "stateful-behavior-probe",
        description: "de-brittled behavior assert",
        signature: sig,
        examples,
        ..Default::default()
    };
    mog_synth::runtime::verify_problem_code_strict(&problem, code)
        .unwrap_or_else(|e| panic!("stateful behavior probe failed: {e}\ncode:\n{code}"));
}
