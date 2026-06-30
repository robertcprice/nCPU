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
