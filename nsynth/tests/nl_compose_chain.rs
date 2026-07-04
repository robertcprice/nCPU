//! Array-transform CHAIN composition (NL-COMPOSE-CHAIN-XFORM).
//!
//! Two array→array transforms in sequence ("sort then reverse") must comprehend
//! to a 2-transform pipeline, synthesize, and pass STRICT holdout verification
//! (`try_compose_pipeline` returns `Ok` only after `verify_problem_code_strict`).
//! This is the multi-transform generalization of the prior single-transform path
//! (`CompositionPlan.array_transforms` is now a Vec; the executor applies the
//! chain via `emit_transform_chain`).

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
fn transform_chains_synthesize_and_strict_verify() {
    let bridge = LinguigenesisBridge::new();
    // Each phrase names TWO array transforms in request order.
    const PHRASES: &[&str] = &[
        "sort then reverse the array",
        "reverse then sort the array",
    ];
    let mut failures = Vec::new();
    for phrase in PHRASES {
        match bridge.try_compose_pipeline(phrase) {
            Some(Ok(out)) => {
                // Genuine 2-transform chain (not a single op masquerading).
                if out.array_xfm_fns.len() < 2 {
                    failures.push(format!(
                        "{phrase:?}: expected >=2 array transforms, got {:?}",
                        out.array_xfm_fns
                    ));
                }
                if !out.has_array_transform() || !out.is_two_stage() {
                    failures.push(format!("{phrase:?}: not classified as a multi-stage pipeline"));
                }
            }
            Some(Err(e)) => failures.push(format!("{phrase:?}: {}", e.lines().next().unwrap_or(""))),
            None => failures.push(format!("{phrase:?}: not recognized as a pipeline")),
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} transform chains failed:\n{}",
        failures.len(),
        PHRASES.len(),
        failures.join("\n")
    );
}

#[test]
fn single_transform_is_not_a_chain() {
    // A lone transform that IS the resolved op stays a single op (not a pipeline) —
    // the chain generalization must not turn "reverse the array" into a pipeline.
    let bridge = LinguigenesisBridge::new();
    assert!(
        bridge.try_compose_pipeline("reverse the array").is_none(),
        "single-op 'reverse the array' must not be treated as a transform chain"
    );
}
