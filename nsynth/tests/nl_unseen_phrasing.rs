//! NON-GAMEABLE compositional-comprehension benchmark for the NL bridge.
//!
//! This proves the two-op array pipeline (`reduce(map(arr))`) is built from what
//! each content word EMERGENTLY resolves to — not a phrase→plan table — and that
//! the composed program is accepted ONLY by strict differential verification on
//! FRESH holdouts (reference-labelled, never the registry's canned examples).
//!
//! WHY IT CANNOT BE GAMED:
//!   * Bucket A's map words (`negated`, `incremented`, `decremented`, `tripled`)
//!     are PROVABLY ABSENT from the coding registry's lemma/synonym lists (asserted
//!     by `bucket_a_content_words_absent_from_registry`). The bridge still resolves
//!     them — purely via emergent morphology to `negate`/`increment`/... — so the
//!     win cannot be faked by registering the phrase or a synonym for it.
//!   * Acceptance requires `verify_problem_code_strict` to pass on holdouts that the
//!     bridge SAMPLES FRESH and LABELS by running an INDEPENDENT reference
//!     composition (`problem_from_reference` path), so example-echo cannot pass.
//!   * Each accepted solve is asserted to be a genuine TWO-stage pipeline
//!     (`is_two_stage()`), rejecting a coincidental single-op match.
//!   * Bucket B (`totally bogus xyzzy`) must NOT be accepted as a pipeline AND must
//!     fail closed at the requirement gate.

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

/// Bucket-A content words whose presence in the live registry would mean the
/// benchmark is being gamed by adding vocabulary. These are the MAP-stage surface
/// forms the bridge must reach by morphology alone.
const BUCKET_A_NOVEL_WORDS: &[&str] = &["negated", "incremented", "decremented", "tripled"];

/// Bucket A: compositional requests whose map word is registry-absent (above) and
/// whose reduce word is a high-confidence aggregate. Each must synthesize a real
/// two-op pipeline AND pass strict holdout verification.
const BUCKET_A_PHRASES: &[&str] = &[
    "sum of the negated values",       // reduce=add(sum-fold) ∘ map=negate
    "sum of the incremented values",   // reduce=add(sum-fold) ∘ map=increment
    "total of the negated numbers",    // reduce=array_sum     ∘ map=negate
    "largest of the incremented values", // reduce=array_max   ∘ map=increment
    "smallest of the negated numbers", // reduce=array_min     ∘ map=negate
    "sum of the decremented values",   // reduce=add(sum-fold) ∘ map=decrement
];

/// ACCEPT-CRITERION core: >=4 compositional requests OUTSIDE any hand-table each
/// synthesize a genuine two-op pipeline AND pass strict (fresh-holdout) verify.
#[test]
fn bucket_a_compositions_synthesize_and_strict_verify() {
    let bridge = LinguigenesisBridge::new();
    let mut accepted = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for phrase in BUCKET_A_PHRASES {
        match bridge.try_compose_pipeline(phrase) {
            Some(Ok(outcome)) => {
                // Must be a genuine TWO-stage pipeline, not a degenerate single op.
                assert!(
                    outcome.is_two_stage(),
                    "{phrase:?}: accepted but NOT two-stage (map_fn={:?}, reduce_fn={})",
                    outcome.map_fn,
                    outcome.reduce_fn
                );
                // The method tag must mark it a composed pipeline (real 2-op shape).
                assert!(
                    outcome.method.starts_with("nl-compose-2op:"),
                    "{phrase:?}: method tag not a pipeline: {}",
                    outcome.method
                );
                accepted += 1;
            }
            Some(Err(e)) => failures.push(format!("{phrase:?}: recognised but rejected: {e}")),
            None => failures.push(format!("{phrase:?}: NOT recognised as a pipeline")),
        }
    }

    for f in &failures {
        println!("PIPELINE-FAILURE: {f}");
    }
    assert!(
        failures.is_empty(),
        "{}/{} Bucket-A compositions failed:\n{}",
        failures.len(),
        BUCKET_A_PHRASES.len(),
        failures.join("\n")
    );
    assert!(
        accepted >= 4,
        "ACCEPT-CRITERION needs >=4 compositional pipelines; only {accepted} accepted"
    );
}

/// Differential correctness spot-check: the accepted program (whatever the solver
/// found) must compute the intended composition on hand-chosen inputs the holdout
/// sampler did not necessarily hit, executed via the same runtime the verifier uses.
#[test]
fn bucket_a_compositions_compute_the_intended_function() {
    use mog_synth::benchmark::{Problem, Value};
    let bridge = LinguigenesisBridge::new();

    // (phrase, input array, expected output) — expected computed by hand from the
    // EMERGENT decomposition, NOT from any registry example.
    let cases: &[(&str, &[i64], i64)] = &[
        // sum of negated values: -(2) + -(-3) + -(4) = -2 + 3 - 4 = -3
        ("sum of the negated values", &[2, -3, 4], -3),
        // sum of incremented values: (1+1)+(2+1)+(10+1) = 2+3+11 = 16
        ("sum of the incremented values", &[1, 2, 10], 16),
        // largest of incremented values: max(1+1, 5+1, 3+1) = 6
        ("largest of the incremented values", &[1, 5, 3], 6),
        // smallest of negated numbers: min(-(1), -(5), -(3)) = min(-1,-5,-3) = -5
        ("smallest of the negated numbers", &[1, 5, 3], -5),
    ];

    for (phrase, input, expected) in cases {
        let outcome = bridge
            .try_compose_pipeline(phrase)
            .unwrap_or_else(|| panic!("{phrase:?}: not recognised as pipeline"))
            .unwrap_or_else(|e| panic!("{phrase:?}: rejected: {e}"));
        let problem = Problem {
            name: "probe".to_string(),
            category: "test",
            description: "intended-fn probe",
            signature: "fn probe(a: [i64]) -> i64",
            examples: vec![],
            ..Default::default()
        };
        let got = mog_synth::runtime::execute_function_for_problem(
            &outcome.code,
            &outcome.fn_name,
            &[Value::int_array(input)],
            &problem,
        )
        .unwrap_or_else(|e| panic!("{phrase:?}: execution failed: {e}\nCODE:\n{}", outcome.code));
        match got {
            mog_synth::runtime::Value::Int(v) => assert_eq!(
                v, *expected,
                "{phrase:?}: expected {expected} on {input:?}, got {v}\nCODE:\n{}",
                outcome.code
            ),
            other => panic!(
                "{phrase:?}: expected Int({expected}) on {input:?}, got {other:?}\nCODE:\n{}",
                outcome.code
            ),
        }
    }
}

/// NON-GAMEABLE GUARD: load the LIVE coding registry and assert NONE of the
/// Bucket-A map words appear as a lemma OR a synonym. If a future edit "wins" the
/// benchmark by registering one of these words, this guard fails — blocking
/// cheat-by-adding-synonym.
#[test]
fn bucket_a_content_words_absent_from_registry() {
    let path = locate_coding_registry();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read coding_registry.json at {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&raw).expect("coding_registry.json parses");
    let entities = json
        .get("entities")
        .and_then(|e| e.as_object())
        .expect("coding_registry.json has an `entities` object");

    let mut lemmas: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut synonyms: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (lemma, ent) in entities {
        lemmas.insert(lemma.to_lowercase());
        if let Some(word) = ent.get("word").and_then(|w| w.as_str()) {
            lemmas.insert(word.to_lowercase());
        }
        if let Some(rels) = ent.get("relations").and_then(|r| r.as_object()) {
            if let Some(syns) = rels.get("synonym").and_then(|s| s.as_array()) {
                for s in syns {
                    if let Some(s) = s.as_str() {
                        synonyms.insert(s.to_lowercase());
                    }
                }
            }
        }
    }

    for word in BUCKET_A_NOVEL_WORDS {
        let w = word.to_lowercase();
        assert!(
            !lemmas.contains(&w),
            "GAMED: Bucket-A word {word:?} is a registry LEMMA — the benchmark must use \
             words the bridge reaches only by emergent morphology"
        );
        assert!(
            !synonyms.contains(&w),
            "GAMED: Bucket-A word {word:?} is a registry SYNONYM — cheat-by-adding-synonym"
        );
    }
}

/// Bucket B (must-refuse): a genuine no-op nonsense request must NOT be accepted
/// as a pipeline AND must fail closed at the requirement gate.
#[test]
fn bucket_b_nonsense_is_refused() {
    let bridge = LinguigenesisBridge::new();
    let phrase = "totally bogus xyzzy";

    // Not recognised as a pipeline (no high-confidence ops resolve).
    assert!(
        bridge.try_compose_pipeline(phrase).is_none(),
        "{phrase:?} must NOT be recognised as a composable pipeline"
    );
    // And the normal requirement path fails closed (ClarificationNeeded / error).
    let req = bridge.nl_to_requirement(phrase);
    assert!(
        req.is_err(),
        "{phrase:?} must fail closed at the requirement gate, got Ok({:?})",
        req.ok().map(|r| r.function_name)
    );
}

/// Single-op behaviour is UNCHANGED: plain requests still resolve+synthesize and
/// are NOT hijacked by the pipeline path.
#[test]
fn single_op_requests_unchanged() {
    let bridge = LinguigenesisBridge::new();

    // A plain scalar op: not a pipeline, still synthesizes.
    assert!(
        bridge.try_compose_pipeline("square a number").is_none(),
        "single-op 'square a number' must not be treated as a pipeline"
    );
    let r = bridge
        .synthesize_from_description("square a number", Some("square"))
        .expect("square must still synthesize");
    assert!(r.success, "square failed: {:?}", r.error);

    // A plain array aggregate: reduce-only naming the resolved op is NOT a pipeline.
    assert!(
        bridge.try_compose_pipeline("compute the total of an array").is_none(),
        "single-op 'compute the total of an array' must not be a pipeline"
    );
    let r = bridge
        .synthesize_from_description("add two numbers", Some("add"))
        .expect("add must still synthesize");
    assert!(r.success, "add failed: {:?}", r.error);
}

fn locate_coding_registry() -> std::path::PathBuf {
    let candidates = [
        std::path::PathBuf::from("../../linguigenesis/data/coding_registry.json"),
        std::path::PathBuf::from("../linguigenesis/data/coding_registry.json"),
    ];
    for c in candidates {
        if c.exists() {
            return c;
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = std::path::PathBuf::from(home)
            .join("projects/linguigenesis/data/coding_registry.json");
        if p.exists() {
            return p;
        }
    }
    panic!("coding_registry.json not found for non-gameable guard");
}
