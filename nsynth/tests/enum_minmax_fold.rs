//! Regression guard for the array min/max fold (enumerative Strategy 4) and the
//! op-by-name synthesis path. `emit_mog_array` always seeds min/max folds with
//! `acc = arr[0]` (the correct reduce); the gate was fixed to accept on the TRUE
//! array min/max rather than a constant-init eval — which had let array_max
//! synthesize while array_min (and all-negative array_max) were wrongly rejected.
use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

fn arr_scalar_problem(name: &'static str, sig: &'static str, ex: Vec<Example>) -> Problem {
    Problem { name: name.to_string(), category: "structural", description: "minmax fold", signature: sig, examples: ex, ..Default::default() }
}

fn mk(a: &[i64], o: i64) -> Example {
    Example { inputs: vec![Value::int_array(a)], expected: Value::Int(o) }
}

#[test]
fn array_min_synthesizes() {
    let p = arr_scalar_problem(
        "array_min",
        "fn array_min(arr: [i64]) -> i64",
        vec![mk(&[3, 1, 2], 1), mk(&[5, 9, 1], 1), mk(&[8, 4, 6, 2], 2), mk(&[10, 20, 5], 5), mk(&[7, 7, 3], 3)],
    );
    let r = mog_synth::solver::solve_problem(&p);
    eprintln!("[array_min] success={} method={}", r.success, r.method);
    assert!(r.success, "array_min should synthesize (arr[0]-seeded min fold)");
}

#[test]
fn array_max_synthesizes_including_all_negative() {
    let p = arr_scalar_problem(
        "array_max",
        "fn array_max(arr: [i64]) -> i64",
        // The all-negative row [-3,-1,-2]→-1 is the case the OLD init=0 gate missed.
        vec![mk(&[3, 1, 2], 3), mk(&[5, 9, 1], 9), mk(&[8, 4, 6, 2], 8), mk(&[-3, -1, -2], -1), mk(&[7, 7, 3], 7)],
    );
    let r = mog_synth::solver::solve_problem(&p);
    eprintln!("[array_max] success={} method={}", r.success, r.method);
    assert!(r.success, "array_max should synthesize incl. all-negative arrays");
}

/// The Mode-A path builds the op's Problem from its REGISTRY examples; guard that
/// array_min synthesizes through it (not just from hand-built examples).
#[test]
fn synthesize_op_by_name_array_min() {
    let bridge = LinguigenesisBridge::new();
    if bridge.registry_load_error().is_some() {
        eprintln!("[op-by-name] registry failed to load — skipping");
        return;
    }
    let r = bridge.synthesize_op_by_name("array_min").expect("array_min op-by-name returned None");
    eprintln!("[op-by-name array_min] success={} method={}", r.success, r.method);
    assert!(r.success, "array_min must synthesize from its registry examples");
}
