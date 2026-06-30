//! Pinpoint: does an operand-typed component ("sum a list") resolve to the
//! COLLECTION op (array_sum) or mis-resolve to scalar add? Single-component
//! requests isolate the per-component resolution from the decomposition.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

/// An operand-typed reduce verb resolves to the COLLECTION op when a collection
/// operand is present — incl. plural/verb forms via morphological root-matching.
#[test]
fn operand_typed_component_resolution() {
    let bridge = LinguigenesisBridge::new();
    let resolved = |request: &str| -> String {
        bridge
            .synthesize_project(request)
            .ok()
            .map(|(s, _)| s.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>().join(","))
            .unwrap_or_default()
    };
    // FIXED by morphological root-match (76454499): "sums" reaches array_sum.
    assert_eq!(resolved("a function that sums a list"), "array_sum", "sums a list");
    assert_eq!(
        resolved("a function that returns the sum of the numbers in a list"),
        "array_sum",
        "sum of numbers in a list"
    );
    // Known follow-on (NOT asserted): "adds the elements" still → scalar add
    // (needs add<->sum synonymy for the array reduce, separate from morphology).
    eprintln!(
        "[RES known-gap] 'adds the elements of an array' → {}",
        resolved("a function that adds the elements of an array")
    );
}
