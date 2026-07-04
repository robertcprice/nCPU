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
    // (NL-COMPOSE-ARRTRANSFORM) BROADENED natural compositional phrasings. These
    // exercise more surface forms of the reduce∘map shape. `tripled` is
    // registry-absent morphology (∈ BUCKET_A_NOVEL_WORDS); `absolute` is a legit
    // registered synonym of `abs` (a content word, not a novel-word claim).
    "the total of the tripled values", // reduce=array_sum     ∘ map=triple
    "the sum of the absolute values",  // reduce=add(sum-fold) ∘ map=abs
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
                // Must be a genuine multi-stage pipeline, not a degenerate single op.
                assert!(
                    outcome.is_two_stage(),
                    "{phrase:?}: accepted but NOT multi-stage (map_fns={:?}, reduce_fn={:?})",
                    outcome.map_fns,
                    outcome.reduce_fn
                );
                // The method tag must mark it a composed pipeline.
                assert!(
                    outcome.method.starts_with("nl-compose-chain:"),
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
        // (broadened) total of tripled values: 3*1 + 3*2 + 3*3 = 18
        ("the total of the tripled values", &[1, 2, 3], 18),
        // (broadened) sum of absolute values: |-2| + |3| + |-4| = 9
        ("the sum of the absolute values", &[-2, 3, -4], 9),
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

// ─────────────────────────────────────────────────────────────────────────────
// TRANSFORM-CHAIN compositions (NL-COMPOSE-CHAIN): >=2 ScalarMaps fused into one
// element transform, in REQUEST ORDER (earlier word = OUTER). These prove the
// bridge composes an arbitrary-length map chain — `negate(increment(x))` — purely
// from what each content word EMERGENTLY resolves to (both map words are
// registry-ABSENT, reached only by morphology — see BUCKET_A_NOVEL_WORDS),
// accepts only via strict holdout verification, computes the ORDER-CORRECT
// function, and (shape b) can return an array when there is no reduce.
// ─────────────────────────────────────────────────────────────────────────────

/// (a) reduce(mapchain) → scalar: each must synthesize a >=2-map chain AND pass
/// strict (fresh-holdout) verification.
const CHAIN_REDUCE_PHRASES: &[&str] = &[
    "sum of the negated incremented values",   // reduce=sum ∘ negate ∘ increment
    "largest of the tripled negated values",   // reduce=max ∘ triple ∘ negate
];

#[test]
fn chain_reduce_compositions_synthesize_and_strict_verify() {
    let bridge = LinguigenesisBridge::new();
    let mut failures: Vec<String> = Vec::new();

    for phrase in CHAIN_REDUCE_PHRASES {
        match bridge.try_compose_pipeline(phrase) {
            Some(Ok(outcome)) => {
                // A genuine >=2-transform chain (not a single map).
                assert!(
                    outcome.map_chain_len() >= 2,
                    "{phrase:?}: expected a >=2-map chain, got {} (map_fns={:?})",
                    outcome.map_chain_len(),
                    outcome.map_fns
                );
                // Reduce present (scalar output) — this is shape (a).
                assert!(
                    outcome.reduce_fn.is_some(),
                    "{phrase:?}: expected a reduce stage, got none"
                );
                assert!(
                    outcome.is_two_stage(),
                    "{phrase:?}: accepted but NOT multi-stage"
                );
                assert!(
                    outcome.method.starts_with("nl-compose-chain:"),
                    "{phrase:?}: method tag not a chain pipeline: {}",
                    outcome.method
                );
            }
            Some(Err(e)) => failures.push(format!("{phrase:?}: recognised but rejected: {e}")),
            None => failures.push(format!("{phrase:?}: NOT recognised as a chain pipeline")),
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} chain-reduce compositions failed:\n{}",
        failures.len(),
        CHAIN_REDUCE_PHRASES.len(),
        failures.join("\n")
    );
}

/// (b) mapchain → ARRAY (no reduce): "the negated incremented values of the
/// array" must synthesize a >=2-map array→array transform, strict-verify on fresh
/// holdouts, and return the ORDER-CORRECT array.
#[test]
fn chain_array_output_composition_returns_correct_array() {
    use mog_synth::benchmark::{Problem, Value};
    let bridge = LinguigenesisBridge::new();
    let phrase = "the negated incremented values of the array";

    let outcome = bridge
        .try_compose_pipeline(phrase)
        .unwrap_or_else(|| panic!("{phrase:?}: not recognised as a chain pipeline"))
        .unwrap_or_else(|e| panic!("{phrase:?}: rejected: {e}"));

    // >=2-map chain, NO reduce → array output (shape b).
    assert!(
        outcome.map_chain_len() >= 2,
        "{phrase:?}: expected a >=2-map chain, got {} (map_fns={:?})",
        outcome.map_chain_len(),
        outcome.map_fns
    );
    assert!(
        outcome.reduce_fn.is_none(),
        "{phrase:?}: expected NO reduce (array output), got {:?}",
        outcome.reduce_fn
    );
    assert!(
        outcome.method.starts_with("nl-compose-chain:"),
        "{phrase:?}: method tag not a chain pipeline: {}",
        outcome.method
    );

    // Order-correct array: negate(increment(x)) = -(x+1). For [2, 3, -4] → [-3, -4, 3].
    // (increment(negate(x)) would give -x+1 = [-1, -2, 5] — proves order matters.)
    let problem = Problem {
        name: "probe".to_string(),
        category: "test",
        description: "array-output chain probe",
        signature: "fn probe(a: [i64]) -> [i64]",
        examples: vec![],
        ..Default::default()
    };
    let got = mog_synth::runtime::execute_function_for_problem(
        &outcome.code,
        &outcome.fn_name,
        &[Value::int_array(&[2, 3, -4])],
        &problem,
    )
    .unwrap_or_else(|e| panic!("{phrase:?}: execution failed: {e}\nCODE:\n{}", outcome.code));
    match got {
        mog_synth::runtime::Value::Array(v) => {
            let ints: Vec<i64> = v
                .iter()
                .map(|e| match e {
                    mog_synth::runtime::Value::Int(n) => *n,
                    other => panic!("{phrase:?}: non-int array element {other:?}"),
                })
                .collect();
            assert_eq!(
                ints,
                vec![-3, -4, 3],
                "{phrase:?}: expected negate(increment(x)) array, got {ints:?}\nCODE:\n{}",
                outcome.code
            );
        }
        other => panic!(
            "{phrase:?}: expected an array output, got {other:?}\nCODE:\n{}",
            outcome.code
        ),
    }
}

/// ORDER-CORRECT computation for chain-reduce: the composed function must apply
/// the maps in the COMPOSED order — negate(square(x)) ≠ square(negate(x)) where
/// they differ. Computed by EXECUTING the accepted program on hand inputs.
#[test]
fn chain_reduce_computes_order_correct_function() {
    use mog_synth::benchmark::{Problem, Value};
    let bridge = LinguigenesisBridge::new();

    // (phrase, input, expected) — expected from the EMERGENT decomposition. The
    // FIRST case is the ORDER discriminator: negate(increment(x)) = -(x+1) differs
    // from increment(negate(x)) = -x+1, so an order-swapped pipeline would compute
    // 0 instead of -6.
    //   "sum of the negated incremented values": sum of -(x+1):
    //     [2,-3,4] → -(3) + -(-2) + -(5) = -3 + 2 - 5 = -6. (swap → -1+4-3 = 0.)
    //   "largest of the tripled negated values": max of 3*(-x):
    //     [1,5,3] → max(3*-1, 3*-5, 3*-3) = max(-3,-15,-9) = -3.
    let cases: &[(&str, &[i64], i64)] = &[
        ("sum of the negated incremented values", &[2, -3, 4], -6),
        ("largest of the tripled negated values", &[1, 5, 3], -3),
    ];

    for (phrase, input, expected) in cases {
        let outcome = bridge
            .try_compose_pipeline(phrase)
            .unwrap_or_else(|| panic!("{phrase:?}: not recognised as pipeline"))
            .unwrap_or_else(|e| panic!("{phrase:?}: rejected: {e}"));
        let problem = Problem {
            name: "probe".to_string(),
            category: "test",
            description: "order-correct chain probe",
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

// ─────────────────────────────────────────────────────────────────────────────
// ARRAY-TRANSFORM compositions (NL-COMPOSE-ARRTRANSFORM): a ScalarMap chain
// followed by ONE array transform (sort / reverse: `Vec<i64> -> Vec<i64>`),
// returning an ARRAY where ORDER MATTERS. This is the genuinely-2-stage
// array→array shape the map-chain (which collapses to a single element fn) cannot
// express — sort/reverse REORDER the mapped array. The map word stays
// registry-ABSENT (reached only by morphology — `negated` ∈ BUCKET_A_NOVEL_WORDS),
// the array transform is classified by EXECUTION / its verified registry output
// spec (never its name), and acceptance is strict-verify on fresh holdouts.
// ─────────────────────────────────────────────────────────────────────────────

/// Each must synthesize a real >=2-stage pipeline (a ScalarMap + an array
/// transform), strict-verify on FRESH holdouts, and carry a behaviour-classified
/// array transform. `square`/`squared` is a registered synonym (a legit content
/// word, not a novel-word cheat); `negated` is registry-absent morphology.
const ARRTRANSFORM_PHRASES: &[&str] = &[
    "the sorted negated values",    // sort(map(negate))   -> array
    "reverse the squared values",   // reverse(map(square)) -> array
    "the reversed negated numbers", // reverse(map(negate)) -> array
    "the sorted incremented values", // sort(map(increment)) -> array
];

#[test]
fn arrtransform_compositions_synthesize_and_strict_verify() {
    let bridge = LinguigenesisBridge::new();
    let mut failures: Vec<String> = Vec::new();
    let mut accepted = 0usize;

    for phrase in ARRTRANSFORM_PHRASES {
        match bridge.try_compose_pipeline(phrase) {
            Some(Ok(outcome)) => {
                // A genuine array-transform stage (sort/reverse) over a map.
                assert!(
                    outcome.has_array_transform(),
                    "{phrase:?}: accepted but has NO array-transform stage (xfm_fns={:?})",
                    outcome.array_xfm_fns
                );
                assert!(
                    !outcome.array_xfms.is_empty(),
                    "{phrase:?}: array transform not behaviour-classified"
                );
                assert!(
                    !outcome.map_fns.is_empty(),
                    "{phrase:?}: expected a map stage composed with the array transform"
                );
                // Array output (no reduce → shape b).
                assert!(
                    outcome.reduce_fn.is_none(),
                    "{phrase:?}: expected array output (no reduce), got {:?}",
                    outcome.reduce_fn
                );
                assert!(
                    outcome.is_two_stage(),
                    "{phrase:?}: accepted but NOT multi-stage"
                );
                assert!(
                    outcome.method.starts_with("nl-compose-chain:"),
                    "{phrase:?}: method tag not a composed pipeline: {}",
                    outcome.method
                );
                accepted += 1;
            }
            Some(Err(e)) => failures.push(format!("{phrase:?}: recognised but rejected: {e}")),
            None => failures.push(format!("{phrase:?}: NOT recognised as a pipeline")),
        }
    }

    for f in &failures {
        println!("ARRTRANSFORM-FAILURE: {f}");
    }
    assert!(
        failures.is_empty(),
        "{}/{} array-transform compositions failed:\n{}",
        failures.len(),
        ARRTRANSFORM_PHRASES.len(),
        failures.join("\n")
    );
    // ACCEPT-CRITERION: >=2 genuine array-transform compositions.
    assert!(
        accepted >= 2,
        "ACCEPT-CRITERION needs >=2 array-transform compositions; only {accepted} accepted"
    );
}

/// ORDER/SEMANTICS-correct computation: the accepted program must REORDER the
/// MAPPED array exactly as sort/reverse demands. Computed by EXECUTING the
/// accepted program on hand inputs. Includes the task's canonical discriminator
/// `sorted-negated([3,-1,2]) == [-3,-2,1]`.
#[test]
fn arrtransform_compositions_compute_the_correct_array() {
    use mog_synth::benchmark::{Problem, Value};
    let bridge = LinguigenesisBridge::new();

    // (phrase, input, expected array) — expected derived by hand from the EMERGENT
    // decomposition. These distinguish sort/reverse AND map-then-transform ORDER:
    //   sort(negate([3,-1,2]))   = sort([-3,1,-2])  = [-3,-2,1]
    //   reverse(square([2,-3,4])) = reverse([4,9,16]) = [16,9,4]
    //   reverse(negate([1,2,3]))  = reverse([-1,-2,-3]) = [-3,-2,-1]
    //   sort(increment([3,1,2]))  = sort([4,2,3]) = [2,3,4]
    let cases: &[(&str, &[i64], &[i64])] = &[
        ("the sorted negated values", &[3, -1, 2], &[-3, -2, 1]),
        ("reverse the squared values", &[2, -3, 4], &[16, 9, 4]),
        ("the reversed negated numbers", &[1, 2, 3], &[-3, -2, -1]),
        ("the sorted incremented values", &[3, 1, 2], &[2, 3, 4]),
    ];

    for (phrase, input, expected) in cases {
        let outcome = bridge
            .try_compose_pipeline(phrase)
            .unwrap_or_else(|| panic!("{phrase:?}: not recognised as pipeline"))
            .unwrap_or_else(|e| panic!("{phrase:?}: rejected: {e}"));
        let problem = Problem {
            name: "probe".to_string(),
            category: "test",
            description: "array-transform order probe",
            signature: "fn probe(a: [i64]) -> [i64]",
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
            mog_synth::runtime::Value::Array(v) => {
                let ints: Vec<i64> = v
                    .iter()
                    .map(|e| match e {
                        mog_synth::runtime::Value::Int(n) => *n,
                        other => panic!("{phrase:?}: non-int array element {other:?}"),
                    })
                    .collect();
                assert_eq!(
                    ints,
                    expected.to_vec(),
                    "{phrase:?}: expected {expected:?} on {input:?}, got {ints:?}\nCODE:\n{}",
                    outcome.code
                );
            }
            other => panic!(
                "{phrase:?}: expected an array output, got {other:?}\nCODE:\n{}",
                outcome.code
            ),
        }
    }
}

/// A plain single-op array transform ("reverse the array", "sort the array") must
/// NOT be hijacked into the composed-pipeline path — it stays a single op (the
/// pipeline detector returns None) so the ordinary front door handles it.
#[test]
fn single_op_array_transform_not_hijacked() {
    let bridge = LinguigenesisBridge::new();
    for phrase in ["reverse the array", "sort the array"] {
        assert!(
            bridge.try_compose_pipeline(phrase).is_none(),
            "single-op {phrase:?} must not be treated as a composed pipeline"
        );
    }
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

// ── FILTER composition (predicate stage): reduce ∘ filter(predicate) ─────────
// The predicate adjectives (positive/negative/even) are NOT registry ops — they
// ground to a comparison the solver must independently re-discover + strict-verify.

// Comparison-predicate filters (item CMP 0) — the solver synthesizes these fast
// (the cond-sum fold / searched filter). MODULUS predicates ("even"/"odd") are a
// known follow-on: the cond-mod fold is harder and the router currently skips
// enumerative for them, so they are exercised separately, not gated here.
const FILTER_PHRASES: &[&str] = &[
    "sum of the positive values",       // array_sum ∘ filter(item > 0)
    "the total of the negative values", // array_sum ∘ filter(item < 0)
];

#[test]
fn filter_compositions_synthesize_and_strict_verify() {
    let bridge = LinguigenesisBridge::new();
    let mut accepted = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for phrase in FILTER_PHRASES {
        match bridge.try_compose_pipeline(phrase) {
            Some(Ok(outcome)) => {
                assert!(
                    outcome.method.starts_with("nl-compose-filter:"),
                    "{phrase:?}: accepted but not a FILTER pipeline: {}",
                    outcome.method
                );
                accepted += 1;
            }
            Some(Err(e)) => failures.push(format!("{phrase:?}: recognised but rejected: {e}")),
            None => failures.push(format!("{phrase:?}: NOT recognised as a filter pipeline")),
        }
    }
    for f in &failures {
        println!("FILTER-FAILURE: {f}");
    }
    assert!(
        failures.is_empty(),
        "{}/{} filter compositions failed:\n{}",
        failures.len(),
        FILTER_PHRASES.len(),
        failures.join("\n")
    );
    assert!(accepted >= 2, "need >=2 filter pipelines; only {accepted}");
}

#[test]
fn filter_compositions_compute_intended() {
    use mog_synth::benchmark::{Problem, Value};
    let bridge = LinguigenesisBridge::new();
    let cases: &[(&str, &[i64], i64)] = &[
        // sum of positives: 3 + 5 = 8
        ("sum of the positive values", &[-2, 3, -1, 5], 8),
        // total of negatives: -2 + -1 + -4 = -7
        ("the total of the negative values", &[-2, 3, -1, 5, -4], -7),
    ];
    for (phrase, input, expected) in cases {
        let outcome = bridge
            .try_compose_pipeline(phrase)
            .unwrap_or_else(|| panic!("{phrase:?}: not recognised as filter pipeline"))
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
        .unwrap_or_else(|e| panic!("{phrase:?}: exec failed: {e}\nCODE:\n{}", outcome.code));
        match got {
            mog_synth::runtime::Value::Int(v) => assert_eq!(
                v, *expected,
                "{phrase:?}: expected {expected} on {input:?}, got {v}\nCODE:\n{}",
                outcome.code
            ),
            other => panic!("{phrase:?}: expected Int, got {other:?}"),
        }
    }
}
