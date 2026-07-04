//! FIELD-WISE STRUCT SYNTHESIS accept-tests: the engine PRODUCES struct-returning
//! programs from examples alone — previously representable/verifiable but never
//! emitted. Un-gameable: solved code must strict-verify through the interpreter
//! (runtime Struct vs wire Struct), the assembled program is checked on holdout
//! semantics via fresh assertions, and an unsynthesizable field declines the
//! whole (no fabrication).

use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::solver::solve_problem;

fn struct_ex(inputs: Vec<Value>, fields: Vec<(&str, Value)>) -> Example {
    Example {
        inputs,
        expected: Value::Struct(
            fields.into_iter().map(|(n, v)| (n.to_string(), v)).collect(),
        ),
    }
}

/// Two scalar fields of one input: Bounds { lo: x-1, hi: x+1 }.
#[test]
fn synthesizes_scalar_field_struct() {
    let ex = |x: i64| {
        struct_ex(
            vec![Value::Int(x)],
            vec![("lo", Value::Int(x - 1)), ("hi", Value::Int(x + 1))],
        )
    };
    let problem = Problem {
        name: "bounds".into(),
        category: "struct-accept",
        description: "bounds struct",
        signature: "fn bounds(x: i64) -> Bounds",
        examples: vec![ex(3), ex(10), ex(-2), ex(100)],
        ..Default::default()
    };
    let r = solve_problem(&problem);
    assert!(r.success, "err: {:?}", r.error);
    assert!(r.method.starts_with("struct_fieldwise"), "method: {}", r.method);
    // Genuinely a struct-producing program: decl + constructor present.
    assert!(r.code.contains("struct Bounds"), "{}", r.code);
    assert!(r.code.contains("return Bounds {"), "{}", r.code);
    // Whole-program strict verification (independent re-check).
    mog_synth::runtime::verify_problem_code_strict(&problem, &r.code).expect("strict");
}

/// Array-reducer fields: Stats { total: sum(arr), biggest: max(arr) } — proves
/// each field sub-problem runs through the FULL pipeline (array machinery).
#[test]
fn synthesizes_array_reducer_struct() {
    let ex = |arr: Vec<i64>| {
        let total: i64 = arr.iter().sum();
        let biggest: i64 = *arr.iter().max().unwrap();
        struct_ex(
            vec![Value::int_array(&arr)],
            vec![("total", Value::Int(total)), ("biggest", Value::Int(biggest))],
        )
    };
    let problem = Problem {
        name: "stats".into(),
        category: "struct-accept",
        description: "array stats struct",
        signature: "fn stats(arr: [i64]) -> Stats",
        examples: vec![
            ex(vec![1, 2, 3]),
            ex(vec![5, 5, 5, 5]),
            ex(vec![-3, 7, 0]),
            ex(vec![42]),
        ],
        ..Default::default()
    };
    let r = solve_problem(&problem);
    assert!(r.success, "err: {:?}", r.error);
    assert!(r.method.starts_with("struct_fieldwise"), "method: {}", r.method);
    assert!(r.code.contains("struct Stats"), "{}", r.code);
    mog_synth::runtime::verify_problem_code_strict(&problem, &r.code).expect("strict");
}

/// STRUCT-INPUT synthesis (flatten-and-wrap): area(Rect{h,w}) = w*h from
/// examples alone. The engine flattens the fields into a flat core solved by
/// the full pipeline, wraps it with field access, and strict-verifies the whole.
#[test]
fn synthesizes_struct_input_function() {
    let ex = |w: i64, h: i64| Example {
        inputs: vec![Value::Struct(vec![
            ("h".to_string(), Value::Int(h)),
            ("w".to_string(), Value::Int(w)),
        ])],
        expected: Value::Int(w * h),
    };
    let problem = Problem {
        name: "area".into(),
        category: "struct-accept",
        description: "rect area",
        signature: "fn area(r: Rect) -> i64",
        examples: vec![ex(3, 4), ex(2, 5), ex(7, 1), ex(6, 6)],
        ..Default::default()
    };
    let r = solve_problem(&problem);
    assert!(r.success, "err: {:?}", r.error);
    assert!(r.method.starts_with("struct_input_flatten"), "method: {}", r.method);
    assert!(r.code.contains("struct Rect"), "{}", r.code);
    assert!(r.code.contains("r.h") && r.code.contains("r.w"), "field access: {}", r.code);
    mog_synth::runtime::verify_problem_code_strict(&problem, &r.code).expect("strict");
}
