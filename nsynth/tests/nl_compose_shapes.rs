//! Composition SHAPE matrix (NL-COMPOSE-SHAPES) — a single regression surface for
//! every multi-stage pipeline shape the bridge composes, each strict-verified.
//!
//! Locks in the verified composition surface so a future change to resolution,
//! roles, or the pipeline code-generator cannot silently regress a working shape.
//! Uses negate/increment element-maps (which resolve cleanly) and sort/reverse
//! transforms. Every accepted `try_compose_pipeline` return is `Ok` ONLY after
//! `verify_problem_code_strict` on fresh reference-labelled holdouts, so a green
//! test means the composed program is behaviourally correct, not just recognised.

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

/// (phrase, expected map fns (sorted), expected reduce fn, expected #transforms).
struct Shape {
    phrase: &'static str,
    maps: &'static [&'static str],
    reduce: Option<&'static str>,
    n_xfms: usize,
}

const SHAPES: &[Shape] = &[
    // map + reduce
    Shape { phrase: "sum of the negated values", maps: &["negate"], reduce: Some("add"), n_xfms: 0 },
    Shape { phrase: "sum of the doubled values", maps: &["double"], reduce: Some("add"), n_xfms: 0 },
    // map + transform
    Shape { phrase: "double each then reverse the array", maps: &["double"], reduce: None, n_xfms: 1 },
    Shape { phrase: "largest of the negated values", maps: &["negate"], reduce: Some("array_max"), n_xfms: 0 },
    // map-chain + reduce
    Shape { phrase: "sum of the negated incremented values", maps: &["increment", "negate"], reduce: Some("add"), n_xfms: 0 },
    // transform + reduce
    Shape { phrase: "sort then sum the array", maps: &[], reduce: Some("add"), n_xfms: 1 },
    // transform chain (array output)
    Shape { phrase: "sort then reverse the array", maps: &[], reduce: None, n_xfms: 2 },
    // transform chain + reduce
    Shape { phrase: "sort then reverse then sum", maps: &[], reduce: Some("add"), n_xfms: 2 },
];

#[test]
fn every_composition_shape_synthesizes_and_verifies() {
    let bridge = LinguigenesisBridge::new();
    let mut failures = Vec::new();
    for s in SHAPES {
        match bridge.try_compose_pipeline(s.phrase) {
            Some(Ok(out)) => {
                let mut got_maps: Vec<&str> = out.map_fns.iter().map(|m| m.as_str()).collect();
                got_maps.sort_unstable();
                let mut want_maps: Vec<&str> = s.maps.to_vec();
                want_maps.sort_unstable();
                if got_maps != want_maps {
                    failures.push(format!("{:?}: maps {:?} != expected {:?}", s.phrase, got_maps, want_maps));
                }
                if out.reduce_fn.as_deref() != s.reduce {
                    failures.push(format!("{:?}: reduce {:?} != expected {:?}", s.phrase, out.reduce_fn, s.reduce));
                }
                if out.array_xfm_fns.len() != s.n_xfms {
                    failures.push(format!("{:?}: {} transforms != expected {}", s.phrase, out.array_xfm_fns.len(), s.n_xfms));
                }
            }
            Some(Err(e)) => failures.push(format!("{:?}: ERR {}", s.phrase, e.lines().next().unwrap_or(""))),
            None => failures.push(format!("{:?}: NOT recognized as a pipeline", s.phrase)),
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} composition shapes failed:\n{}",
        failures.len(),
        SHAPES.len(),
        failures.join("\n")
    );
}
