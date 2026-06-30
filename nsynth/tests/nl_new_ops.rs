//! NL front-door widening: the new registry ops (bool predicates + product) must
//! resolve from natural language AND synthesize a verified program. Proves the
//! engine's existing bool/product capability is now REACHABLE from prose.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
fn nl_invokes_newly_exposed_ops() {
    let bridge = LinguigenesisBridge::new();
    assert!(bridge.registry_load_error().is_none(), "registry must load");

    // `even` (parity predicate) + `product` (array reduce) resolve from prose to
    // the CORRECT fn + synthesize a verified program. positive/negative are HELD
    // BACK: those words are sense-ambiguous (filter modifier in "the positive
    // values" / negation marker / standalone predicate) — exposing them as
    // standalone ops regresses filter composition (and "negative" panics via the
    // negation path). The real fix is context-based word-sense disambiguation, not
    // resolution priority (priority tweaks regressed array-op + filter resolution).
    let cases: &[(&str, &str)] = &[
        ("whether a number is even", "is_even"),
        ("whether a number is odd", "is_odd"),
        ("whether a number is positive", "is_positive"),
        ("whether a number is negative", "is_negative"),
        ("whether a number is zero", "is_zero"),
        ("the product of all the elements", "product"),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (phrase, want_fn) in cases {
        match bridge.synthesize_project(phrase) {
            Ok((solved, skipped)) => {
                match solved.iter().find(|(_, r)| r.success) {
                    Some((name, r)) => {
                        eprintln!("[NL-OP] {phrase:?} → fn={name} (want {want_fn}) method={}", r.method);
                        if !r.code.contains(want_fn) && name != want_fn {
                            failures.push(format!(
                                "{phrase:?}: resolved to {name} ({}), wanted {want_fn}",
                                r.method
                            ));
                        }
                    }
                    None => {
                        eprintln!("[NL-OP] {phrase:?} → NO SUCCESS (want {want_fn}) skipped={skipped:?}");
                        failures.push(format!("{phrase:?} (want {want_fn}): no success"));
                    }
                }
            }
            Err(e) => {
                eprintln!("[NL-OP] {phrase:?} → ERR {e}");
                failures.push(format!("{phrase:?}: bridge error {e}"));
            }
        }
    }
    for f in &failures {
        eprintln!("NL-OP-FAILURE: {f}");
    }
    assert!(
        failures.is_empty(),
        "newly-exposed ops must resolve to the CORRECT fn + synthesize:\n{}",
        failures.join("\n")
    );
}
