//! U5c accept-tests: SEARCHED linear-recursion synthesis.
//!
//! These tests prove the win is genuinely *searched*, not name/shape
//! recognised:
//!   1. The synthesized program is produced by `search_linear_recursion`
//!      (asserted via `result.method`), which is routed BEFORE the fixed
//!      `search_recursive_factorial` / `search_recursive_fibonacci`
//!      recognisers in `SEARCH_CANDIDATES` — so the recogniser path never
//!      claims these problems (the DEL-CHEAT preemption proof).
//!   2. Acceptance is via `verify_problem_code_strict` on FRESH
//!      reference-derived holdouts (asserted `HoldoutSource::Generated`,
//!      distinct from the visible example seeds).
//!   3. The synthesized recursive `Decl` executes to the correct values on
//!      hand inputs (factorial(5)=120, triangular(5)=15, sumsq(4)=30).
//!   4. The discovered `combine` body DIFFERS across targets (factorial uses
//!      `*`, triangular uses `+`), proving the combine op is searched, not one
//!      hardcoded body.

use crate::benchmark::{
    generated_holdouts_with_source, Example, HoldoutSource, Problem, Value as BmValue,
};
use crate::runtime::{execute_function_for_problem, verify_problem_code_strict};
use crate::solver::search::ranked_search_candidate_keys;
use crate::solver::solve_problem_search_only;

fn ex(n: i64, out: i64) -> Example {
    Example {
        inputs: vec![BmValue::Int(n)],
        expected: BmValue::Int(out),
    }
}

/// factorial(n) = n <= 1 ? 1 : n * factorial(n-1)  (combine = `n * acc`).
fn factorial_problem() -> Problem {
    Problem {
        name: "u5c_factorial".to_string(),
        category: "test",
        description: "searched linear recursion: factorial",
        signature: "fn u5c_factorial(n: i64) -> i64",
        // Small seeds (depth < 32): f(0..=6). Holdouts are sampled FRESH from
        // the reference, NOT from these.
        examples: vec![
            ex(0, 1),
            ex(1, 1),
            ex(2, 2),
            ex(3, 6),
            ex(4, 24),
            ex(5, 120),
            ex(6, 720),
        ],
        holdouts: vec![],
        // Recursive reference: the ONLY oracle for the generated holdouts.
        reference_code: "fn u5c_factorial(n: i64) -> i64 { if n <= 1 { return 1; } return n * u5c_factorial(n - 1); }",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: true,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    }
}

/// triangular(n) = n <= 0 ? 0 : n + triangular(n-1)  (combine = `n + acc`).
fn triangular_problem() -> Problem {
    Problem {
        name: "u5c_triangular".to_string(),
        category: "test",
        description: "searched linear recursion: sum-to-n / triangular",
        signature: "fn u5c_triangular(n: i64) -> i64",
        examples: vec![
            ex(0, 0),
            ex(1, 1),
            ex(2, 3),
            ex(3, 6),
            ex(4, 10),
            ex(5, 15),
            ex(6, 21),
        ],
        holdouts: vec![],
        reference_code: "fn u5c_triangular(n: i64) -> i64 { if n <= 0 { return 0; } return n + u5c_triangular(n - 1); }",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: true,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    }
}

/// sumsq(n) = n <= 0 ? 0 : n*n + sumsq(n-1)  (combine = `n * n + acc`).
fn sum_of_squares_problem() -> Problem {
    Problem {
        name: "u5c_sumsq".to_string(),
        category: "test",
        description: "searched linear recursion: sum of first n squares",
        signature: "fn u5c_sumsq(n: i64) -> i64",
        examples: vec![
            ex(0, 0),
            ex(1, 1),
            ex(2, 5),
            ex(3, 14),
            ex(4, 30),
            ex(5, 55),
        ],
        holdouts: vec![],
        reference_code: "fn u5c_sumsq(n: i64) -> i64 { if n <= 0 { return 0; } return n * n + u5c_sumsq(n - 1); }",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: true,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    }
}

/// The searched family must be routed BEFORE the fixed recognisers, so the
/// recogniser path is preempted (DEL-CHEAT: the win is searched, not
/// recognised). This is the structural guarantee behind every `method ==
/// "search_linear_recursion"` assertion below.
#[test]
fn searched_recursion_preempts_fixed_recognisers() {
    let problem = factorial_problem();
    let order = ranked_search_candidate_keys(&problem.synthesis_view());
    let searched = order
        .iter()
        .position(|k| *k == "search_linear_recursion")
        .expect("search_linear_recursion must be registered");
    let factorial = order
        .iter()
        .position(|k| *k == "search_recursive_factorial")
        .expect("search_recursive_factorial must be registered");
    let fibonacci = order
        .iter()
        .position(|k| *k == "search_recursive_fibonacci")
        .expect("search_recursive_fibonacci must be registered");
    assert!(
        searched < factorial && searched < fibonacci,
        "searched linear recursion ({searched}) must precede the fixed \
         factorial ({factorial}) and fibonacci ({fibonacci}) recognisers so \
         the win is searched, not recognised",
    );
}

/// End-to-end: factorial is SOLVED BY SEARCH, strict-verifies on FRESH
/// reference-derived holdouts, and the emitted recursive Decl executes to
/// factorial(5) = 120. The discovered combine is `n * acc`.
#[test]
fn factorial_is_searched_not_recognised() {
    let problem = factorial_problem();

    // Holdouts are GENERATED fresh from the reference (true differential
    // probes), not the hand-authored fallback.
    let (holdouts, source) = generated_holdouts_with_source(&problem);
    assert_eq!(
        source,
        HoldoutSource::Generated,
        "factorial holdouts must be reference-derived (differential), got {source:?}",
    );
    assert!(!holdouts.is_empty(), "generated holdouts must be non-empty");

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "factorial search failed: {:?}", result.error);

    // The win came from the SEARCHED family, not the fixed recogniser.
    assert_eq!(
        result.method, "search_linear_recursion",
        "factorial must be solved by searched linear recursion, not {}",
        result.method,
    );

    // The emitted program is a GENUINE self-recursive Decl (not an iterative
    // emit), and its discovered combine uses `*`.
    assert!(
        result.code.contains("u5c_factorial(n - 1)"),
        "expected a real recursive self-call, got:\n{}",
        result.code,
    );
    assert!(
        result.code.contains("n * acc"),
        "factorial combine must be discovered as `n * acc`, got:\n{}",
        result.code,
    );

    // Strict-verify on the fresh reference-derived holdouts.
    verify_problem_code_strict(&problem, &result.code)
        .unwrap_or_else(|e| panic!("factorial strict-verify failed: {e}"));

    // Execute the synthesized recursive Decl on a hand input: factorial(5)=120.
    let got = execute_function_for_problem(
        &result.code,
        "u5c_factorial",
        &[BmValue::Int(5)],
        &problem,
    )
    .expect("execute factorial(5)");
    match &got {
        crate::runtime::Value::Int(v) => assert_eq!(*v, 120, "factorial(5) must be 120, got {got:?}"),
        other => panic!("factorial(5) must be 120 (non-int result {other:?})"),
    }
}

/// End-to-end: triangular (sum-to-n) is SOLVED BY SEARCH, strict-verifies on
/// fresh holdouts, executes to triangular(5)=15, and discovers combine `n + acc`.
#[test]
fn triangular_is_searched_not_recognised() {
    let problem = triangular_problem();

    let (_holdouts, source) = generated_holdouts_with_source(&problem);
    assert_eq!(
        source,
        HoldoutSource::Generated,
        "triangular holdouts must be reference-derived, got {source:?}",
    );

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "triangular search failed: {:?}", result.error);
    assert_eq!(
        result.method, "search_linear_recursion",
        "triangular must be solved by searched linear recursion, not {}",
        result.method,
    );
    assert!(
        result.code.contains("u5c_triangular(n - 1)"),
        "expected a real recursive self-call, got:\n{}",
        result.code,
    );
    assert!(
        result.code.contains("n + acc"),
        "triangular combine must be discovered as `n + acc`, got:\n{}",
        result.code,
    );

    verify_problem_code_strict(&problem, &result.code)
        .unwrap_or_else(|e| panic!("triangular strict-verify failed: {e}"));

    let got = execute_function_for_problem(
        &result.code,
        "u5c_triangular",
        &[BmValue::Int(5)],
        &problem,
    )
    .expect("execute triangular(5)");
    match &got {
        crate::runtime::Value::Int(v) => assert_eq!(*v, 15, "triangular(5) must be 15, got {got:?}"),
        other => panic!("triangular(5) must be 15 (non-int result {other:?})"),
    }
}

/// Third target: sum of first n squares — SOLVED BY SEARCH, strict-verified,
/// executes to sumsq(4)=30, discovers combine `n * n + acc`.
#[test]
fn sum_of_squares_is_searched() {
    let problem = sum_of_squares_problem();

    let result = solve_problem_search_only(&problem);
    assert!(result.success, "sumsq search failed: {:?}", result.error);
    assert_eq!(
        result.method, "search_linear_recursion",
        "sumsq must be solved by searched linear recursion, not {}",
        result.method,
    );
    assert!(
        result.code.contains("u5c_sumsq(n - 1)"),
        "expected a real recursive self-call, got:\n{}",
        result.code,
    );
    assert!(
        result.code.contains("n * n + acc"),
        "sumsq combine must be discovered as `n * n + acc`, got:\n{}",
        result.code,
    );

    verify_problem_code_strict(&problem, &result.code)
        .unwrap_or_else(|e| panic!("sumsq strict-verify failed: {e}"));

    let got =
        execute_function_for_problem(&result.code, "u5c_sumsq", &[BmValue::Int(4)], &problem)
            .expect("execute sumsq(4)");
    match &got {
        crate::runtime::Value::Int(v) => assert_eq!(*v, 30, "sumsq(4) must be 30, got {got:?}"),
        other => panic!("sumsq(4) must be 30 (non-int result {other:?})"),
    }
}

/// The combine op is SEARCHED, not a single hardcoded body: two different
/// target functions yield two DIFFERENT discovered combine expressions
/// (factorial -> `n * acc`, triangular -> `n + acc`).
#[test]
fn discovered_combine_differs_across_targets() {
    let fact = solve_problem_search_only(&factorial_problem());
    let tri = solve_problem_search_only(&triangular_problem());
    assert!(fact.success && tri.success);
    assert_eq!(fact.method, "search_linear_recursion");
    assert_eq!(tri.method, "search_linear_recursion");

    assert!(fact.code.contains("n * acc"));
    assert!(tri.code.contains("n + acc"));
    // The two recursion bodies must genuinely differ (different combine op),
    // proving the search discovers the operator rather than emitting one fixed
    // recursion shape.
    assert!(
        fact.code != tri.code,
        "factorial and triangular must yield different searched bodies",
    );
    assert!(
        !fact.code.contains("n + acc"),
        "factorial body must not contain the `+` combine",
    );
    assert!(
        !tri.code.contains("n * acc"),
        "triangular body must not contain the `*` combine",
    );
}
