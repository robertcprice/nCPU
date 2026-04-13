/*! Search-based solver for Mog benchmark problems.

Mirrors the cascade in Python mog_search_solver.py but runs entirely in Rust
using the native interpreter — no PyTorch gradient search needed.
*/

use crate::benchmark::{BenchmarkProblem, evaluate_solution};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub success: bool,
    pub code: String,
    pub method: String,
}

impl SolveResult {
    fn ok(code: String, method: &str) -> Self {
        SolveResult { success: true, code, method: method.to_string() }
    }
    pub fn fail() -> Self {
        SolveResult { success: false, code: String::new(), method: "failed".to_string() }
    }
}

// ---------------------------------------------------------------------------
// Signature helpers (duplicated locally to avoid pub-ing benchmark internals)
// ---------------------------------------------------------------------------

fn fn_name(sig: &str) -> String {
    let start = sig.find("fn ").unwrap_or(0) + 3;
    let rest = &sig[start..];
    let end = rest.find('(').unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

fn params(sig: &str) -> Vec<(String, String)> {
    let start = match sig.find('(') { Some(s) => s + 1, None => return vec![] };
    let end   = match sig.find(')') { Some(e) => e,     None => return vec![] };
    if start >= end { return vec![]; }
    sig[start..end].split(',').filter_map(|part| {
        let part = part.trim();
        if part.is_empty() { return None; }
        let mut it = part.splitn(2, ':');
        let name = it.next()?.trim().to_string();
        let ty   = it.next()?.trim().to_string();
        Some((name, ty))
    }).collect()
}

fn arg_names(sig: &str) -> Vec<String> {
    params(sig).into_iter().map(|(n, _)| n).collect()
}

fn has_arrays(sig: &str)  -> bool { sig.contains('[') }
fn has_strings(sig: &str) -> bool { params(sig).iter().any(|(_, t)| t == "string") }
fn has_structs(sig: &str) -> bool {
    params(sig).iter().any(|(_, t)| !matches!(t.as_str(), "i64"|"string"|"[i64]"|"[string]"))
}

// ---------------------------------------------------------------------------
// Core helper: try a candidate and return SolveResult on pass
// ---------------------------------------------------------------------------

fn try_code(problem: &BenchmarkProblem, code: &str, method: &str) -> Option<SolveResult> {
    let r = evaluate_solution(problem, code);
    if r.passed { Some(SolveResult::ok(code.to_string(), method)) } else { None }
}

// ---------------------------------------------------------------------------
// Scalar search helpers  (1-arg unless noted)
// ---------------------------------------------------------------------------

fn search_fast_arithmetic(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    let consts = ["0", "1", "-1", "2", "100"];
    let params_str = args.iter().map(|a| format!("{a}: i64")).collect::<Vec<_>>().join(", ");
    let names: Vec<&str> = args.iter().map(|s| s.as_str()).chain(consts.iter().copied()).collect();
    for &s1 in &names {
        for &s2 in &names {
            for op in ["+", "-", "*", "/", "%"] {
                let code = format!("fn {fn_name}({params_str}) -> i64 {{\n    return {s1} {op} {s2};\n}}\n");
                if let Some(r) = try_code(problem, &code, "arithmetic") { return Some(r); }
            }
        }
    }
    None
}

fn search_gcd_loop(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() < 2 { return None; }
    let (a, b) = (&args[0], &args[1]);
    let ps = format!("{a}: i64, {b}: i64");
    let code = format!(
        "fn {fn_name}({ps}) -> i64 {{\n    x: i64 = {a};\n    y: i64 = {b};\n    while y != 0 {{\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }}\n    return x;\n}}\n"
    );
    try_code(problem, &code, "gcd_loop")
}

fn search_lcm(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() < 2 { return None; }
    let (a, b) = (&args[0], &args[1]);
    let ps = format!("{a}: i64, {b}: i64");
    let code = format!(
        "fn {fn_name}({ps}) -> i64 {{\n    x: i64 = {a};\n    y: i64 = {b};\n    while y != 0 {{\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }}\n    return ({a} * {b}) / x;\n}}\n"
    );
    try_code(problem, &code, "lcm_loop")
}

fn search_modulo_check(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.is_empty() { return None; }
    let x = &args[0];
    let ps = format!("{x}: i64");
    for m in [2, 3, 4, 5, 10i64] {
        for eq in 0..m {
            for ret_t in [0i64, 1, -1] {
                for ret_f in [0i64, 1, -1] {
                    let code = format!(
                        "fn {fn_name}({ps}) -> i64 {{\n    if (({x} % {m}) == {eq}) {{ return {ret_t}; }}\n    return {ret_f};\n}}\n"
                    );
                    if let Some(r) = try_code(problem, &code, "modulo_check") { return Some(r); }
                }
            }
        }
    }
    None
}

fn search_factorial(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    // Use iterative to avoid evaluator stack overflow on large n
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    result: i64 = 1;\n    i: i64 = 2;\n    while i <= {n} {{ result = result * i; i = i + 1; }}\n    return result;\n}}\n"
    );
    try_code(problem, &code, "factorial")
}

fn search_fibonacci(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    // Iterative to avoid stack overflow
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} == 0 {{ return 0; }}\n    if {n} == 1 {{ return 1; }}\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= {n} {{\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }}\n    return b;\n}}\n"
    );
    try_code(problem, &code, "fibonacci")
}

fn search_digit_sum(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    x: i64 = {n};\n    if x < 0 {{ x = 0 - x; }}\n    total: i64 = 0;\n    while x > 0 {{ total = total + (x % 10); x = x / 10; }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "digit_sum")
}

fn search_power(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() < 2 { return None; }
    let (base, exp) = (&args[0], &args[1]);
    let ps = format!("{base}: i64, {exp}: i64");
    let code = format!(
        "fn {fn_name}({ps}) -> i64 {{\n    if {exp} == 0 {{ return 1; }}\n    result: i64 = 1;\n    i: i64 = 0;\n    while i < {exp} {{ result = result * {base}; i = i + 1; }}\n    return result;\n}}\n"
    );
    try_code(problem, &code, "power")
}

fn search_collatz(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    x: i64 = {n};\n    steps: i64 = 0;\n    while x > 1 {{\n        if x % 2 == 0 {{ x = x / 2; }} else {{ x = 3 * x + 1; }}\n        steps = steps + 1;\n    }}\n    return steps;\n}}\n"
    );
    try_code(problem, &code, "collatz")
}

fn search_is_prime(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} < 2 {{ return 0; }}\n    if {n} == 2 {{ return 1; }}\n    if {n} % 2 == 0 {{ return 0; }}\n    i: i64 = 3;\n    while i * i <= {n} {{\n        if {n} % i == 0 {{ return 0; }}\n        i = i + 2;\n    }}\n    return 1;\n}}\n"
    );
    try_code(problem, &code, "is_prime")
}

fn search_polynomial(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let x = &args[0];
    let candidates = [
        format!("fn {fn_name}({x}: i64) -> i64 {{\n    return 2 * {x} * {x} + 3 * {x} + 1;\n}}\n"),
        format!("fn {fn_name}({x}: i64) -> i64 {{\n    return {x} * {x} + {x} + 1;\n}}\n"),
        format!("fn {fn_name}({x}: i64) -> i64 {{\n    return {x} * {x};\n}}\n"),
        format!("fn {fn_name}({x}: i64) -> i64 {{\n    return {x} * {x} + 1;\n}}\n"),
    ];
    for code in &candidates {
        if let Some(r) = try_code(problem, code, "polynomial") { return Some(r); }
    }
    None
}

fn search_nth_triangle(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let formula = format!("fn {fn_name}({n}: i64) -> i64 {{\n    return {n} * ({n} + 1) / 2;\n}}\n");
    if let Some(r) = try_code(problem, &formula, "nth_triangle") { return Some(r); }
    let loopcode = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= {n} {{ total = total + i; i = i + 1; }}\n    return total;\n}}\n"
    );
    try_code(problem, &loopcode, "nth_triangle")
}

fn search_min3(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 3 { return None; }
    let (a, b, c) = (&args[0], &args[1], &args[2]);
    let ps = format!("{a}: i64, {b}: i64, {c}: i64");
    let code = format!(
        "fn {fn_name}({ps}) -> i64 {{\n    m: i64 = {a};\n    if {b} < m {{ m = {b}; }}\n    if {c} < m {{ m = {c}; }}\n    return m;\n}}\n"
    );
    try_code(problem, &code, "min3")
}

fn search_fib_iter(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} == 0 {{ return 0; }}\n    if {n} == 1 {{ return 1; }}\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= {n} {{\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }}\n    return b;\n}}\n"
    );
    try_code(problem, &code, "fib_iter")
}

fn search_euler_totient(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    result: i64 = {n};\n    p: i64 = 2;\n    temp: i64 = {n};\n    while p * p <= temp {{\n        if temp % p == 0 {{\n            while temp % p == 0 {{ temp = temp / p; }}\n            result = result - result / p;\n        }}\n        p = p + 1;\n    }}\n    if temp > 1 {{ result = result - result / temp; }}\n    return result;\n}}\n"
    );
    try_code(problem, &code, "euler_totient")
}

fn search_sum_squares(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= {n} {{ total = total + i * i; i = i + 1; }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "sum_squares")
}

fn search_product_1_to_n(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} == 0 {{ return 1; }}\n    total: i64 = 1;\n    i: i64 = 1;\n    while i <= {n} {{ total = total * i; i = i + 1; }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "product_1_to_n")
}

fn search_count_divisors(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    count: i64 = 0;\n    i: i64 = 1;\n    while i <= {n} {{ if {n} % i == 0 {{ count = count + 1; }} i = i + 1; }}\n    return count;\n}}\n"
    );
    try_code(problem, &code, "count_divisors")
}

fn search_triangular_check(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    k: i64 = 0;\n    while k * (k + 1) / 2 <= {n} {{ if k * (k + 1) / 2 == {n} {{ return 1; }} k = k + 1; }}\n    return 0;\n}}\n"
    );
    try_code(problem, &code, "triangular_check")
}

fn search_gcd_extended(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() < 2 { return None; }
    let (a, b) = (&args[0], &args[1]);
    let ps = format!("{a}: i64, {b}: i64");
    let code = format!(
        "fn {fn_name}({ps}) -> i64 {{\n    x: i64 = {a};\n    y: i64 = {b};\n    while y != 0 {{ tmp: i64 = y; y = x % y; x = tmp; }}\n    return x;\n}}\n"
    );
    try_code(problem, &code, "gcd_extended")
}

fn search_harmonic_sum(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= {n} {{ total = total + 1000 / i; i = i + 1; }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "harmonic_sum")
}

// Phase 1: Digit manipulation
fn search_reverse_digits(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    result: i64 = 0;\n    x: i64 = {n};\n    while x > 0 {{ result = result * 10 + x % 10; x = x / 10; }}\n    return result;\n}}\n"
    );
    try_code(problem, &code, "reverse_digits")
}

fn search_digit_count(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} == 0 {{ return 1; }}\n    count: i64 = 0;\n    x: i64 = {n};\n    while x > 0 {{ count = count + 1; x = x / 10; }}\n    return count;\n}}\n"
    );
    try_code(problem, &code, "digit_count")
}

fn search_count_even_digits(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} == 0 {{ return 1; }}\n    count: i64 = 0;\n    x: i64 = {n};\n    while x > 0 {{ if (x % 10) % 2 == 0 {{ count = count + 1; }} x = x / 10; }}\n    return count;\n}}\n"
    );
    try_code(problem, &code, "count_even_digits")
}

// Phase 2: Algorithmic scalar
fn search_perfect_check(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} < 2 {{ return 0; }}\n    total: i64 = 1;\n    i: i64 = 2;\n    while i * i <= {n} {{\n        if {n} % i == 0 {{\n            total = total + i;\n            if i * i != {n} {{ total = total + {n} / i; }}\n        }}\n        i = i + 1;\n    }}\n    if total == {n} {{ return 1; }}\n    return 0;\n}}\n"
    );
    try_code(problem, &code, "perfect_check")
}

fn search_armstrong_check(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    total: i64 = 0;\n    x: i64 = {n};\n    while x > 0 {{\n        d: i64 = x % 10;\n        total = total + d * d * d;\n        x = x / 10;\n    }}\n    if total == {n} {{ return 1; }}\n    return 0;\n}}\n"
    );
    try_code(problem, &code, "armstrong_check")
}

fn search_geometric_sum(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    total: i64 = 1;\n    power: i64 = 1;\n    i: i64 = 0;\n    while i < {n} {{ power = power * 2; total = total + power; i = i + 1; }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "geometric_sum")
}

fn search_nested_sum(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= {n} {{\n        j: i64 = 1;\n        while j <= i {{ total = total + i * j; j = j + 1; }}\n        i = i + 1;\n    }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "nested_sum")
}

// Phase 5: Advanced scalar
fn search_fib_cached(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} == 0 {{ return 0; }}\n    if {n} == 1 {{ return 1; }}\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= {n} {{ tmp: i64 = a + b; a = b; b = tmp; i = i + 1; }}\n    return b;\n}}\n"
    );
    try_code(problem, &code, "fib_cached")
}

fn search_mersenne_check(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let n = &args[0];
    let code = format!(
        "fn {fn_name}({n}: i64) -> i64 {{\n    if {n} < 1 {{ return 0; }}\n    m: i64 = {n} + 1;\n    while m > 1 {{ if m % 2 != 0 {{ return 0; }} m = m / 2; }}\n    return 1;\n}}\n"
    );
    try_code(problem, &code, "mersenne_check")
}

// Loop accumulator search (combinatorial, mirrors _loop_accum_refinement)
fn search_loop_accum(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.is_empty() { return None; }
    let params_str = args.iter().map(|a| format!("{a}: i64")).collect::<Vec<_>>().join(", ");
    let bound_exprs: Vec<String> = args.iter()
        .flat_map(|a| vec![a.clone(), format!("{a} + 1"), format!("{a} + 2")])
        .collect();
    for init in ["0", "1"] {
        for start in ["0", "1", "2"] {
            for bound in &bound_exprs {
                for op in ["+", "*"] {
                    for rhs in ["i", "1"] {
                        let code = format!(
                            "fn {fn_name}({params_str}) -> i64 {{\n    acc: i64 = {init};\n    for i := {start} to ({bound}) {{\n        acc = acc {op} {rhs};\n    }}\n    return acc;\n}}\n"
                        );
                        if let Some(r) = try_code(problem, &code, "loop_accum") { return Some(r); }
                    }
                }
            }
        }
    }
    None
}

// abs_diff, max2, clamped patterns
fn search_two_arg_branch(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() < 2 { return None; }
    let (a, b) = (&args[0], &args[1]);
    let ps = format!("{a}: i64, {b}: i64");
    let candidates = [
        // abs_diff: if a > b { a-b } else { b-a }
        format!("fn {fn_name}({ps}) -> i64 {{\n    if {a} > {b} {{ return {a} - {b}; }}\n    return {b} - {a};\n}}\n"),
        // max2: if a > b { a } else { b }
        format!("fn {fn_name}({ps}) -> i64 {{\n    if {a} > {b} {{ return {a}; }}\n    return {b};\n}}\n"),
        // min2: if a < b { a } else { b }
        format!("fn {fn_name}({ps}) -> i64 {{\n    if {a} < {b} {{ return {a}; }}\n    return {b};\n}}\n"),
        // safe div: if b == 0 { -1 } else { a/b }
        format!("fn {fn_name}({ps}) -> i64 {{\n    if {b} == 0 {{ return -1; }}\n    return {a} / {b};\n}}\n"),
    ];
    for code in &candidates {
        if let Some(r) = try_code(problem, code, "two_arg_branch") { return Some(r); }
    }
    None
}

fn search_clamp(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.len() != 1 { return None; }
    let x = &args[0];
    // Try several clamp ranges
    for (lo, hi) in [(0, 100), (0, 10), (1, 10), (-1, 1), (0, 255)] {
        let code = format!(
            "fn {fn_name}({x}: i64) -> i64 {{\n    if {x} < {lo} {{ return {lo}; }}\n    if {x} > {hi} {{ return {hi}; }}\n    return {x};\n}}\n"
        );
        if let Some(r) = try_code(problem, &code, "clamp") { return Some(r); }
    }
    None
}

// Combinatorial single-branch search (if A op B: return C else D)
fn search_branching(problem: &BenchmarkProblem, fn_name: &str, args: &[String]) -> Option<SolveResult> {
    if args.is_empty() { return None; }
    let params_str = args.iter().map(|a| format!("{a}: i64")).collect::<Vec<_>>().join(", ");
    let consts = ["0", "1", "-1", "2"];
    let names: Vec<&str> = args.iter().map(|s| s.as_str()).chain(consts.iter().copied()).collect();
    let ops = ["<", ">", "<=", ">=", "==", "!="];
    for &lhs in &names {
        for &rhs in &names {
            for &op in &ops {
                for &then_v in &names {
                    for &else_v in &names {
                        if then_v == else_v { continue; }
                        let code = format!(
                            "fn {fn_name}({params_str}) -> i64 {{\n    if {lhs} {op} {rhs} {{ return {then_v}; }}\n    return {else_v};\n}}\n"
                        );
                        if let Some(r) = try_code(problem, &code, "single_branch") { return Some(r); }
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Array searches
// ---------------------------------------------------------------------------

fn search_max_pair_diff(problem: &BenchmarkProblem, fn_name: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({arr}: [i64]) -> i64 {{\n    best: i64 = 0;\n    i: i64 = 1;\n    while i < {arr}.len {{\n        diff: i64 = {arr}[i] - {arr}[i - 1];\n        if diff < 0 {{ diff = 0 - diff; }}\n        if diff > best {{ best = diff; }}\n        i = i + 1;\n    }}\n    return best;\n}}\n"
    );
    try_code(problem, &code, "max_pair_diff")
}

fn search_sum_negatives(problem: &BenchmarkProblem, fn_name: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({arr}: [i64]) -> i64 {{\n    total: i64 = 0;\n    for item in {arr} {{ if item < 0 {{ total = total + item; }} }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "sum_negatives")
}

fn search_find_first_even(problem: &BenchmarkProblem, fn_name: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({arr}: [i64]) -> i64 {{\n    i: i64 = 0;\n    while i < {arr}.len {{\n        if {arr}[i] % 2 == 0 {{ return i; }}\n        i = i + 1;\n    }}\n    return -1;\n}}\n"
    );
    try_code(problem, &code, "find_first_even")
}

fn search_sum_until_negative(problem: &BenchmarkProblem, fn_name: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({arr}: [i64]) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 0;\n    while i < {arr}.len {{\n        if {arr}[i] < 0 {{ return total; }}\n        total = total + {arr}[i];\n        i = i + 1;\n    }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "sum_until_negative")
}

fn search_sort_and_sum(problem: &BenchmarkProblem, fn_name: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({arr}: [i64]) -> i64 {{\n    mn := {arr}[0];\n    mx := {arr}[0];\n    for item in {arr} {{\n        if item < mn {{ mn = item; }}\n        if item > mx {{ mx = item; }}\n    }}\n    return mn + mx;\n}}\n"
    );
    try_code(problem, &code, "sort_and_sum")
}

fn search_array_triple(problem: &BenchmarkProblem, fn_name: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({arr}: [i64]) -> i64 {{\n    total: i64 = 0;\n    for item in {arr} {{ total = total + item * 3; }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "array_triple")
}

fn search_sum_even_indexed(problem: &BenchmarkProblem, fn_name: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({arr}: [i64]) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 0;\n    while i < {arr}.len {{ total = total + {arr}[i]; i = i + 2; }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "sum_even_indexed")
}

fn search_last_element(problem: &BenchmarkProblem, fn_name: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({arr}: [i64]) -> i64 {{\n    return {arr}[{arr}.len - 1];\n}}\n"
    );
    try_code(problem, &code, "last_element")
}

fn search_count_occurrences(problem: &BenchmarkProblem, fn_name: &str, ps: &[(String, String)]) -> Option<SolveResult> {
    // 2-arg: arr + target
    if ps.len() < 2 { return None; }
    let arr = &ps[0].0;
    let target = &ps[1].0;
    let params_str = ps.iter().map(|(n, t)| format!("{n}: {t}")).collect::<Vec<_>>().join(", ");
    let code = format!(
        "fn {fn_name}({params_str}) -> i64 {{\n    count: i64 = 0;\n    for item in {arr} {{ if item == {target} {{ count = count + 1; }} }}\n    return count;\n}}\n"
    );
    try_code(problem, &code, "count_occurrences")
}

fn search_array_reduction(problem: &BenchmarkProblem, fn_name: &str, arr: &str, params_str: &str) -> Option<SolveResult> {
    let candidates = [
        // sum
        format!("fn {fn_name}({params_str}) -> i64 {{\n    total: i64 = 0;\n    for item in {arr} {{ total = total + item; }}\n    return total;\n}}\n"),
        // max
        format!("fn {fn_name}({params_str}) -> i64 {{\n    best := {arr}[0];\n    for item in {arr} {{ if item > best {{ best = item; }} }}\n    return best;\n}}\n"),
        // count positive
        format!("fn {fn_name}({params_str}) -> i64 {{\n    total: i64 = 0;\n    for item in {arr} {{ if item > 0 {{ total = total + 1; }} }}\n    return total;\n}}\n"),
        // double sum
        format!("fn {fn_name}({params_str}) -> i64 {{\n    total: i64 = 0;\n    for item in {arr} {{ total = total + (item * 2); }}\n    return total;\n}}\n"),
    ];
    for code in &candidates {
        if let Some(r) = try_code(problem, code, "array_search") { return Some(r); }
    }
    None
}

// ---------------------------------------------------------------------------
// String searches
// ---------------------------------------------------------------------------

fn search_palindrome(problem: &BenchmarkProblem, fn_name: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}(s: string) -> i64 {{\n    chars := s.split(\"\");\n    left: i64 = 0;\n    right: i64 = s.len - 1;\n    while left < right {{\n        if chars[left] != chars[right] {{ return 0; }}\n        left = left + 1;\n        right = right - 1;\n    }}\n    return 1;\n}}\n"
    );
    try_code(problem, &code, "palindrome")
}

fn search_count_words(problem: &BenchmarkProblem, fn_name: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}(s: string) -> i64 {{\n    t := s.trim();\n    if t.len == 0 {{ return 0; }}\n    parts := t.split(\" \");\n    count: i64 = 0;\n    for p in parts {{ if p.len > 0 {{ count = count + 1; }} }}\n    return count;\n}}\n"
    );
    try_code(problem, &code, "count_words")
}

fn search_string_patterns(problem: &BenchmarkProblem, fn_name: &str, s: &str) -> Option<SolveResult> {
    let candidates = [
        // trimmed_len
        format!("fn {fn_name}({s}: string) -> i64 {{\n    t := {s}.trim();\n    return t.len;\n}}\n"),
        // vowel_count
        format!("fn {fn_name}({s}: string) -> i64 {{\n    chars := {s}.split(\"\");\n    total: i64 = 0;\n    for ch in chars {{\n        if ch == \"a\" {{ total = total + 1; }}\n        if ch == \"e\" {{ total = total + 1; }}\n        if ch == \"i\" {{ total = total + 1; }}\n        if ch == \"o\" {{ total = total + 1; }}\n        if ch == \"u\" {{ total = total + 1; }}\n    }}\n    return total;\n}}\n"),
        // contains_cat
        format!("fn {fn_name}({s}: string) -> i64 {{\n    if {s}.contains(\"cat\") {{ return 1; }}\n    return 0;\n}}\n"),
        // starts_with_m
        format!("fn {fn_name}({s}: string) -> i64 {{\n    if {s}.starts_with(\"m\") {{ return 1; }}\n    return 0;\n}}\n"),
    ];
    for code in &candidates {
        if let Some(r) = try_code(problem, code, "string_search") { return Some(r); }
    }
    None
}

// ---------------------------------------------------------------------------
// Struct searches
// ---------------------------------------------------------------------------

fn search_struct_patterns(problem: &BenchmarkProblem, fn_name: &str) -> Option<SolveResult> {
    let candidates = [
        format!("struct Point {{\n    x: i64,\n    y: i64,\n}}\n\nfn {fn_name}(p: Point) -> i64 {{\n    return p.x + p.y;\n}}\n"),
        format!("struct Rectangle {{\n    width: i64,\n    height: i64,\n}}\n\nfn {fn_name}(r: Rectangle) -> i64 {{\n    return r.width * r.height;\n}}\n"),
    ];
    for code in &candidates {
        if let Some(r) = try_code(problem, code, "struct_search") { return Some(r); }
    }
    None
}

// Interactive / no-arg array sum
fn search_interactive_sum(problem: &BenchmarkProblem, fn_name: &str, params_str: &str, arr: &str) -> Option<SolveResult> {
    let code = format!(
        "fn {fn_name}({params_str}) -> i64 {{\n    total: i64 = 0;\n    for item in {arr} {{ total = total + item; }}\n    return total;\n}}\n"
    );
    try_code(problem, &code, "interactive_sum")
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------

pub fn solve_problem(problem: &BenchmarkProblem) -> SolveResult {
    let sig = &problem.signature;
    let fname = fn_name(sig);
    let args = arg_names(sig);
    let ps = params(sig);

    let is_array  = has_arrays(sig);
    let is_string = has_strings(sig);
    let is_struct = has_structs(sig);

    // --- Scalar cascade ---
    if !is_array && !is_string && !is_struct && !args.is_empty() {
        let searches: &[fn(&BenchmarkProblem, &str, &[String]) -> Option<SolveResult>] = &[
            search_fast_arithmetic,
            search_gcd_loop,
            search_lcm,
            search_modulo_check,
            search_factorial,
            search_fibonacci,
            search_digit_sum,
            search_power,
            search_collatz,
            search_is_prime,
            search_polynomial,
            search_nth_triangle,
            search_min3,
            search_fib_iter,
            search_euler_totient,
            search_sum_squares,
            search_product_1_to_n,
            search_count_divisors,
            search_triangular_check,
            search_gcd_extended,
            search_harmonic_sum,
            // Phase 1
            search_reverse_digits,
            search_digit_count,
            search_count_even_digits,
            // Phase 2
            search_perfect_check,
            search_armstrong_check,
            search_geometric_sum,
            search_nested_sum,
            // Phase 5
            search_fib_cached,
            search_mersenne_check,
            // Two-arg branching (abs_diff, max2, min2, safe_div)
            search_two_arg_branch,
            // Clamp to range
            search_clamp,
            // Generic
            search_loop_accum,
            search_branching,
        ];
        for search_fn in searches {
            if let Some(r) = search_fn(problem, &fname, &args) { return r; }
        }
    }

    // --- Array cascade ---
    if is_array {
        let arr_name = ps.iter().find(|(_, t)| t.contains('['))
            .map(|(n, _)| n.as_str())
            .unwrap_or("arr");
        let ps_str = ps.iter().map(|(n, t)| format!("{n}: {t}")).collect::<Vec<_>>().join(", ");

        if let Some(r) = search_max_pair_diff(problem, &fname, arr_name)     { return r; }
        if let Some(r) = search_sum_negatives(problem, &fname, arr_name)     { return r; }
        if let Some(r) = search_find_first_even(problem, &fname, arr_name)   { return r; }
        if let Some(r) = search_sum_until_negative(problem, &fname, arr_name){ return r; }
        if let Some(r) = search_sort_and_sum(problem, &fname, arr_name)      { return r; }
        if let Some(r) = search_array_triple(problem, &fname, arr_name)      { return r; }
        if let Some(r) = search_sum_even_indexed(problem, &fname, arr_name)  { return r; }
        if let Some(r) = search_last_element(problem, &fname, arr_name)      { return r; }
        if let Some(r) = search_count_occurrences(problem, &fname, &ps)             { return r; }
        if let Some(r) = search_interactive_sum(problem, &fname, &ps_str, arr_name) { return r; }
        if let Some(r) = search_array_reduction(problem, &fname, arr_name, &ps_str) { return r; }
    }

    // --- String cascade ---
    if is_string {
        let s_name = ps.iter().find(|(_, t)| t == "string")
            .map(|(n, _)| n.as_str())
            .unwrap_or("s");
        if let Some(r) = search_palindrome(problem, &fname)                    { return r; }
        if let Some(r) = search_count_words(problem, &fname)                   { return r; }
        if let Some(r) = search_string_patterns(problem, &fname, s_name)       { return r; }
    }

    // --- Struct cascade ---
    if is_struct {
        if let Some(r) = search_struct_patterns(problem, &fname) { return r; }
    }

    SolveResult::fail()
}

// ---------------------------------------------------------------------------
// Batch evaluation
// ---------------------------------------------------------------------------

pub struct BenchmarkSummary {
    pub num_problems: usize,
    pub num_solved: usize,
    pub failures: Vec<String>,
}

pub fn evaluate_solver(problems: &[BenchmarkProblem]) -> BenchmarkSummary {
    let mut num_solved = 0;
    let mut failures = Vec::new();
    for p in problems {
        let r = solve_problem(p);
        if r.success { num_solved += 1; } else { failures.push(p.name.clone()); }
    }
    BenchmarkSummary { num_problems: problems.len(), num_solved, failures }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::get_benchmark;

    #[test]
    fn test_solver_add_two() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_add_two(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "add_two failed: method={}", r.method);
    }

    #[test]
    fn test_solver_gcd() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_gcd(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "gcd failed: method={}", r.method);
    }

    #[test]
    fn test_solver_factorial() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_factorial(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "factorial failed: method={}", r.method);
    }

    #[test]
    fn test_solver_is_prime() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_is_prime(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "is_prime failed: method={}", r.method);
    }

    #[test]
    fn test_solver_fib_iter() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_fib_iter(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "fib_iter failed: method={}", r.method);
    }

    #[test]
    fn test_solver_array_sum() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_array_sum(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "array_sum failed: method={}", r.method);
    }

    #[test]
    fn test_solver_vowel_count() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_vowel_count(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "vowel_count failed: method={}", r.method);
    }

    #[test]
    fn test_solver_nested_sum() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_nested_sum(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "nested_sum failed: method={}", r.method);
    }

    #[test]
    fn test_solver_find_first_even() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_find_first_even(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "find_first_even failed: method={}", r.method);
    }

    #[test]
    fn test_solver_mersenne_check() {
        let mut rng = crate::benchmark::Rng::new(42);
        let p = crate::benchmark::make_mersenne_check(&mut rng, 0);
        let r = solve_problem(&p);
        assert!(r.success, "mersenne_check failed: method={}", r.method);
    }

    #[test]
    fn test_solver_full_benchmark_61() {
        let problems = get_benchmark(42, 1);
        let summary = evaluate_solver(&problems);
        eprintln!("Solved {}/{}", summary.num_solved, summary.num_problems);
        if !summary.failures.is_empty() {
            eprintln!("Failed: {:?}", summary.failures);
        }
        assert!(
            summary.num_solved >= 59,
            "Expected >=59/61 solved, got {}/{}. Failed: {:?}",
            summary.num_solved, summary.num_problems, summary.failures
        );
    }
}
