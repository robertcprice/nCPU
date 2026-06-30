//! #2 program structure: patterns beyond the fixed map/filter/reduce shape.
//! Branch-on-structure (length/emptiness guard) is the first frontier pattern.
use mog_synth::benchmark::{Example, Problem, Value};

fn probe(name: &'static str, sig: &'static str, examples: Vec<Example>) -> mog_synth::solver::SolveResult {
    let problem = Problem {
        name: name.to_string(),
        category: "structural",
        description: "branch-on-structure probe",
        signature: sig,
        examples,
        ..Default::default()
    };
    let r = mog_synth::solver::solve_problem(&problem);
    eprintln!("[STRUCT {name}] success={} method={}\n{}", r.success, r.method, r.code);
    r
}

/// "max of the array, or 0 if empty" — a top-level branch on array length.
/// The reduce shape alone cannot express the empty-case default.
#[test]
fn max_or_zero_if_empty() {
    let mk = |a: &[i64], o: i64| Example {
        inputs: vec![Value::int_array(a)],
        expected: Value::Int(o),
    };
    let r = probe(
        "max_or_zero",
        "fn max_or_zero(a: [i64]) -> i64",
        vec![
            mk(&[], 0),
            mk(&[3, 1, 2], 3),
            mk(&[5, 2, 8, 1], 8),
            mk(&[7], 7),
            mk(&[], 0),
            mk(&[4, 9, 2], 9),
            mk(&[-3, -1, -2], -1),
        ],
    );
    assert!(r.success, "max-or-0-if-empty should synthesize (branch on length)");
}

/// General length-threshold branch, TOP-LEVEL (no empty case, so the emptiness
/// guard skips): "if len < 2 return the single element, else the sum". Two
/// genuinely different branch bodies selected by arr.len.
#[test]
fn single_element_else_sum() {
    let mk = |a: &[i64], o: i64| Example {
        inputs: vec![Value::int_array(a)],
        expected: Value::Int(o),
    };
    let r = probe(
        "small_zero_else_sum",
        "fn small_zero_else_sum(a: [i64]) -> i64",
        vec![
            mk(&[5], 0),
            mk(&[3], 0),
            mk(&[7], 0),
            mk(&[1, 2], 3),
            mk(&[4, 5, 6], 15),
            mk(&[2, 3], 5),
            mk(&[10, 20], 30),
            mk(&[1, 1, 1], 3),
        ],
    );
    assert!(r.success, "len<2->element else sum should synthesize (length branch)");
    assert!(
        r.method.contains("structural-length-branch:"),
        "wrong method (expected a length branch): {}",
        r.method
    );
}

/// Probe: top-level scalar branch with NON-affine sides — if n<0 { n*n } else { n+1 }.
/// Tests whether the enumerative IfExpr already covers expr-branches (to target the gap).
#[test]
fn scalar_branch_probe() {
    let mk = |n: i64, o: i64| Example { inputs: vec![Value::Int(n)], expected: Value::Int(o) };
    let r = probe(
        "sgnsq",
        "fn sgnsq(n: i64) -> i64",
        vec![
            mk(-2, 4), mk(-3, 9), mk(-1, 1), mk(0, 1),
            mk(1, 2), mk(5, 6), mk(3, 4), mk(-5, 25), mk(2, 3), mk(-4, 16),
        ],
    );
    // Already covered by search_single_branch (top-level scalar branch, arbitrary
    // expr sides) — lock it as a guard.
    assert!(r.success, "top-level scalar branch (non-affine sides) should synthesize");
}

/// GAP probe: array->array map whose body uses a WHOLE-ARRAY aggregate computed
/// once — "each element minus the max". out[k] = arr[k] - max(arr). The flat
/// elementwise map can't see max(arr); needs a reduce-then-map intermediate.
#[test]
fn aggregate_map_probe() {
    let mk = |a: &[i64]| {
        let mx = *a.iter().max().unwrap();
        Example {
            inputs: vec![Value::int_array(a)],
            expected: Value::int_array(&a.iter().map(|x| x - mx).collect::<Vec<_>>()),
        }
    };
    let r = probe(
        "minus_max",
        "fn minus_max(a: [i64]) -> [i64]",
        vec![mk(&[3, 1, 5]), mk(&[2, 8, 4]), mk(&[7, 7, 1]), mk(&[10, 2, 6]), mk(&[1, 2, 3, 4]), mk(&[9, 0, 5])],
    );
    assert!(r.success, "each-minus-max should synthesize (aggregate-parameterized map)");
    assert!(
        r.method.contains("aggregate-map"),
        "expected aggregate-map, got: {}",
        r.method
    );
}
