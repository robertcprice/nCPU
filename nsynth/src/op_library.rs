//! Verified reference-op library — the "known algorithms" tier.
//!
//! Some functions cannot be INDUCED from a handful of I/O examples (you can't learn
//! `is_prime` or `gcd` from 3 numbers). MBPP profiling proved this: the unsolved
//! tasks need algorithms, and more search time buys nothing. So this carries a small
//! set of correct reference implementations, and [`try_library`] returns one ONLY
//! when it reproduces EVERY example of the problem — matched by BEHAVIOR, never by
//! name/shape (so it is a verified standard library, not a hardcoded recognizer). A
//! wrong guess reproduces nothing and is dropped; downstream holdout checks catch a
//! coincidental seed-only match.
//!
//! This is the engine side of the thesis: the LLM (or the examples) name the task,
//! and a VERIFIED impl satisfies it. Ops here are pure Mog (run by the interpreter,
//! transpiled by `to_rust` for generated crates).

use crate::benchmark::Problem;
use crate::runtime::code_reproduces_examples;
use crate::solver::SolveResult;

/// One reference implementation: a Mog program whose entry fn is `name`.
pub struct LibOp {
    pub name: &'static str,
    pub arity: usize,
    pub mog: &'static str,
}

/// The library. Single-loop / while / early-return algorithms Mog expresses
/// directly. Each is validated by `tests::every_op_reproduces_its_probe`.
pub const OPS: &[LibOp] = &[
    // ── number theory (1-arg i64) ──────────────────────────────────────────
    LibOp { name: "is_prime", arity: 1, mog:
"fn is_prime(n: i64) -> i64 {\n    if n < 2 {\n        return 0;\n    }\n    d: i64 = 2;\n    while d * d <= n {\n        if n % d == 0 {\n            return 0;\n        }\n        d = d + 1;\n    }\n    return 1;\n}\n" },
    LibOp { name: "factorial", arity: 1, mog:
"fn factorial(n: i64) -> i64 {\n    acc: i64 = 1;\n    i: i64 = 2;\n    while i <= n {\n        acc = acc * i;\n        i = i + 1;\n    }\n    return acc;\n}\n" },
    LibOp { name: "sum_of_digits", arity: 1, mog:
"fn sum_of_digits(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    acc: i64 = 0;\n    while x > 0 {\n        acc = acc + x % 10;\n        x = x / 10;\n    }\n    return acc;\n}\n" },
    LibOp { name: "count_digits", arity: 1, mog:
"fn count_digits(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    c: i64 = 0;\n    while x > 0 {\n        c = c + 1;\n        x = x / 10;\n    }\n    if c == 0 {\n        c = 1;\n    }\n    return c;\n}\n" },
    LibOp { name: "reverse_number", arity: 1, mog:
"fn reverse_number(n: i64) -> i64 {\n    x: i64 = n;\n    r: i64 = 0;\n    while x > 0 {\n        r = r * 10 + x % 10;\n        x = x / 10;\n    }\n    return r;\n}\n" },
    LibOp { name: "fibonacci", arity: 1, mog:
"fn fibonacci(n: i64) -> i64 {\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 0;\n    while i < n {\n        t: i64 = a + b;\n        a = b;\n        b = t;\n        i = i + 1;\n    }\n    return a;\n}\n" },
    LibOp { name: "is_even", arity: 1, mog:
"fn is_even(n: i64) -> i64 {\n    if n % 2 == 0 {\n        return 1;\n    }\n    return 0;\n}\n" },
    // ── number theory (2-arg i64) ──────────────────────────────────────────
    LibOp { name: "gcd", arity: 2, mog:
"fn gcd(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        t: i64 = y;\n        y = x % y;\n        x = t;\n    }\n    return x;\n}\n" },
    LibOp { name: "lcm", arity: 2, mog:
"fn lcm(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        t: i64 = y;\n        y = x % y;\n        x = t;\n    }\n    return a / x * b;\n}\n" },
    LibOp { name: "power", arity: 2, mog:
"fn power(a: i64, b: i64) -> i64 {\n    acc: i64 = 1;\n    i: i64 = 0;\n    while i < b {\n        acc = acc * a;\n        i = i + 1;\n    }\n    return acc;\n}\n" },
    // ── array reductions the base engine misses (1-arg [i64]) ──────────────
    LibOp { name: "count_evens", arity: 1, mog:
"fn count_evens(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e % 2 == 0 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_odds", arity: 1, mog:
"fn count_odds(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e % 2 != 0 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "count_positives", arity: 1, mog:
"fn count_positives(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e > 0 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    LibOp { name: "sum_of_squares", arity: 1, mog:
"fn sum_of_squares(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for e in arr {\n        acc = acc + e * e;\n    }\n    return acc;\n}\n" },
    // ── array + scalar (2-arg) ─────────────────────────────────────────────
    LibOp { name: "count_value", arity: 2, mog:
"fn count_value(arr: [i64], x: i64) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e == x {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
    // ── batch 2: more number theory (1-arg i64) ────────────────────────────
    LibOp { name: "count_divisors", arity: 1, mog:
"fn count_divisors(n: i64) -> i64 {\n    c: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 {\n            c = c + 1;\n        }\n        i = i + 1;\n    }\n    return c;\n}\n" },
    LibOp { name: "sum_divisors", arity: 1, mog:
"fn sum_divisors(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 {\n            s = s + i;\n        }\n        i = i + 1;\n    }\n    return s;\n}\n" },
    LibOp { name: "is_perfect", arity: 1, mog:
"fn is_perfect(n: i64) -> i64 {\n    s: i64 = 0;\n    i: i64 = 1;\n    while i < n {\n        if n % i == 0 {\n            s = s + i;\n        }\n        i = i + 1;\n    }\n    if s == n {\n        return 1;\n    }\n    return 0;\n}\n" },
    LibOp { name: "digit_product", arity: 1, mog:
"fn digit_product(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    p: i64 = 1;\n    while x > 0 {\n        p = p * (x % 10);\n        x = x / 10;\n    }\n    return p;\n}\n" },
    LibOp { name: "largest_digit", arity: 1, mog:
"fn largest_digit(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    m: i64 = 0;\n    while x > 0 {\n        d: i64 = x % 10;\n        if d > m {\n            m = d;\n        }\n        x = x / 10;\n    }\n    return m;\n}\n" },
    LibOp { name: "count_primes_below", arity: 1, mog:
"fn count_primes_below(n: i64) -> i64 {\n    c: i64 = 0;\n    k: i64 = 2;\n    while k < n {\n        d: i64 = 2;\n        prime: i64 = 1;\n        while d * d <= k {\n            if k % d == 0 {\n                prime = 0;\n            }\n            d = d + 1;\n        }\n        if prime == 1 {\n            c = c + 1;\n        }\n        k = k + 1;\n    }\n    return c;\n}\n" },
    // ── batch 2: array aggregation (1-arg [i64]) ───────────────────────────
    LibOp { name: "array_range", arity: 1, mog:
"fn array_range(arr: [i64]) -> i64 {\n    mx: i64 = arr[0];\n    mn: i64 = arr[0];\n    for e in arr {\n        if e > mx {\n            mx = e;\n        }\n        if e < mn {\n            mn = e;\n        }\n    }\n    return mx - mn;\n}\n" },
    LibOp { name: "count_negatives", arity: 1, mog:
"fn count_negatives(arr: [i64]) -> i64 {\n    c: i64 = 0;\n    for e in arr {\n        if e < 0 {\n            c = c + 1;\n        }\n    }\n    return c;\n}\n" },
];

/// Return the first library impl that reproduces EVERY example of `problem`
/// (arity-compatible, behavior-matched, cheap: a few interpreter runs). `None` if
/// no known algorithm fits. The returned code is already verified against the spec.
pub fn try_library(problem: &Problem) -> Option<SolveResult> {
    let arity = problem.examples.first()?.inputs.len();
    for op in OPS {
        if op.arity != arity {
            continue;
        }
        if code_reproduces_examples(op.mog, &problem.examples) {
            return Some(SolveResult {
                success: true,
                code: op.mog.to_string(),
                method: format!("library:{}", op.name),
                error: None,
                metadata: Default::default(),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn runs_to(mog: &str, name: &str, args: Vec<Value>, expect: Value) -> bool {
        code_reproduces_examples(mog, &[Example { inputs: args, expected: expect }])
    }

    #[test]
    fn every_op_reproduces_its_probe() {
        // Each op checked on a hand-computed probe so a typo in a Mog impl fails here,
        // not silently in production.
        let iv = Value::int_array;
        let cases: &[(&str, Vec<Value>, Value)] = &[
            ("is_prime", vec![Value::Int(7)], Value::Int(1)),
            ("is_prime", vec![Value::Int(9)], Value::Int(0)),
            ("factorial", vec![Value::Int(5)], Value::Int(120)),
            ("sum_of_digits", vec![Value::Int(123)], Value::Int(6)),
            ("count_digits", vec![Value::Int(4056)], Value::Int(4)),
            ("reverse_number", vec![Value::Int(123)], Value::Int(321)),
            ("fibonacci", vec![Value::Int(7)], Value::Int(13)),
            ("is_even", vec![Value::Int(4)], Value::Int(1)),
            ("gcd", vec![Value::Int(12), Value::Int(18)], Value::Int(6)),
            ("lcm", vec![Value::Int(4), Value::Int(6)], Value::Int(12)),
            ("power", vec![Value::Int(2), Value::Int(10)], Value::Int(1024)),
            ("count_evens", vec![iv(&[1, 2, 3, 4, 6])], Value::Int(3)),
            ("count_odds", vec![iv(&[1, 2, 3, 4, 5])], Value::Int(3)),
            ("count_positives", vec![iv(&[-1, 2, -3, 4])], Value::Int(2)),
            ("sum_of_squares", vec![iv(&[1, 2, 3])], Value::Int(14)),
            ("count_value", vec![iv(&[1, 2, 2, 3, 2]), Value::Int(2)], Value::Int(3)),
            ("count_divisors", vec![Value::Int(12)], Value::Int(6)),
            ("sum_divisors", vec![Value::Int(6)], Value::Int(12)),
            ("is_perfect", vec![Value::Int(6)], Value::Int(1)),
            ("is_perfect", vec![Value::Int(8)], Value::Int(0)),
            ("digit_product", vec![Value::Int(234)], Value::Int(24)),
            ("largest_digit", vec![Value::Int(4092)], Value::Int(9)),
            ("count_primes_below", vec![Value::Int(10)], Value::Int(4)),
            ("array_range", vec![iv(&[3, 8, 1, 6])], Value::Int(7)),
            ("count_negatives", vec![iv(&[-1, 2, -3, -4, 5])], Value::Int(3)),
        ];
        for (name, args, expect) in cases {
            let op = OPS.iter().find(|o| o.name == *name).unwrap_or_else(|| panic!("no op {name}"));
            assert!(
                runs_to(op.mog, name, args.clone(), expect.clone()),
                "op {name} failed its probe (expected {expect:?})"
            );
        }
    }
}
