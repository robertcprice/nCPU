//! Probe: which primitive OUTPUT types does the solve path actually synthesize?
//! Evidence, not assumption — run before claiming coverage.
use mog_synth::benchmark::{Example, Problem, Value};

fn probe(name: &'static str, sig: &'static str, examples: Vec<Example>) -> mog_synth::solver::SolveResult {
    let problem = Problem {
        name: name.to_string(),
        category: "probe",
        description: "primitive output probe",
        signature: sig,
        examples,
        ..Default::default()
    };
    let r = mog_synth::solver::solve_problem(&problem);
    eprintln!("[PROBE {name}] success={} method={}\n{}", r.success, r.method, r.code);
    r
}

/// SOUNDNESS INVARIANT: a deterministic predicate must NEVER be "solved" by a
/// stochastic sampler. (Regression guard for the removed bool→Bernoulli
/// false-accept.) If a bool problem is reported solved, the code must not be a
/// random sampler.
fn assert_not_random_fake(r: &mog_synth::solver::SolveResult, name: &str) {
    if r.success {
        let c = r.code.to_lowercase();
        assert!(
            !c.contains("rand::") && !c.contains("rng") && !c.contains("bernoulli"),
            "{name}: deterministic problem FALSE-ACCEPTED by a random sampler:\n{}",
            r.code
        );
    }
}

#[test]
fn probe_bool_is_even() {
    let mk = |n: i64, b: bool| Example {
        inputs: vec![Value::Int(n)],
        expected: Value::Bool(b),
    };
    let r = probe(
        "is_even",
        "fn is_even(n: i64) -> bool",
        vec![
            mk(2, true), mk(3, false), mk(4, true), mk(7, false),
            mk(10, true), mk(0, true), mk(1, false), mk(8, true),
        ],
    );
    assert_not_random_fake(&r, "is_even");
    eprintln!("BOOL is_even synthesizes (for real): {}", r.success);
}

#[test]
fn probe_bool_is_positive() {
    let mk = |n: i64, b: bool| Example {
        inputs: vec![Value::Int(n)],
        expected: Value::Bool(b),
    };
    let r = probe(
        "is_positive",
        "fn is_positive(n: i64) -> bool",
        vec![
            mk(5, true), mk(-3, false), mk(0, false), mk(7, true),
            mk(-1, false), mk(100, true), mk(-50, false), mk(3, true),
        ],
    );
    assert_not_random_fake(&r, "is_positive");
    eprintln!("BOOL is_positive synthesizes (for real): {}", r.success);
}

#[test]
fn probe_array_product() {
    let mk = |a: &[i64], p: i64| Example {
        inputs: vec![Value::int_array(a)],
        expected: Value::Int(p),
    };
    let r = probe(
        "product",
        "fn product(a: [i64]) -> i64",
        vec![
            mk(&[2, 3, 4], 24), mk(&[1, 5], 5), mk(&[2, 2, 2], 8),
            mk(&[3, 3], 9), mk(&[10, 2], 20), mk(&[1, 1, 1, 7], 7),
        ],
    );
    assert!(r.success, "array product must synthesize (real fold)");
    assert!(r.code.contains('*'), "product must multiply: {}", r.code);
}
