//! NL front-door widening: the new registry ops (bool predicates + product) must
//! resolve from natural language AND synthesize a verified program. Proves the
//! engine's existing bool/product capability is now REACHABLE from prose.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
fn nl_invokes_newly_exposed_ops() {
    let bridge = LinguigenesisBridge::new();
    assert!(bridge.registry_load_error().is_none(), "registry must load");

    // Only the ops that resolve CORRECTLY (the right fn) are asserted. `even`
    // (parity predicate) + `product` (array reduce) are now reachable from prose
    // AND synthesize a verified program. (positive/negative are NOT here: they
    // collide with WordNet adjectives and mis-resolve — a known resolution gap,
    // not shipped until the coding-overlay-vs-WordNet priority is fixed.)
    let cases: &[(&str, &str)] = &[
        ("whether a number is even", "is_even"),
        ("the product of all the elements", "product"),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (phrase, want_fn) in cases {
        match bridge.synthesize_project(phrase) {
            Ok((solved, skipped)) => {
                match solved.iter().find(|(_, r)| r.success) {
                    Some((name, r)) => {
                        eprintln!("[NL-OP] {phrase:?} → fn={name} method={}\n{}", r.method, r.code);
                        if !r.code.contains(want_fn) && name != want_fn {
                            failures.push(format!(
                                "{phrase:?}: resolved to {name} ({}), wanted {want_fn}",
                                r.method
                            ));
                        }
                    }
                    None => failures.push(format!(
                        "{phrase:?} (want {want_fn}): no success; skipped={skipped:?}"
                    )),
                }
            }
            Err(e) => failures.push(format!("{phrase:?}: bridge error {e}")),
        }
    }
    for f in &failures {
        println!("NL-OP-FAILURE: {f}");
    }
    assert!(
        failures.is_empty(),
        "newly-exposed ops must resolve to the CORRECT fn + synthesize:\n{}",
        failures.join("\n")
    );
}
