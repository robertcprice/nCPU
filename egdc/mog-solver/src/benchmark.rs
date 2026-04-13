/*! Benchmark system for Mog program synthesis. */

use std::fmt;

use crate::interpreter;

// --- Core types ---

#[derive(Clone, Debug)]
pub enum MogLiteral {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MogLiteral>),
    RawCode(String),
}

impl fmt::Display for MogLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MogLiteral::Int(i) => write!(f, "{}", i),
            MogLiteral::Float(v) => {
                if *v == (*v as i64) as f64 {
                    write!(f, "{:.1}", v)
                } else {
                    write!(f, "{}", v)
                }
            }
            MogLiteral::Bool(b) => write!(f, "{}", if *b { 1 } else { 0 }),
            MogLiteral::Str(s) => {
                let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
                write!(f, "\"{}\"", escaped)
            }
            MogLiteral::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            MogLiteral::RawCode(code) => write!(f, "{}", code),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BenchmarkProblem {
    pub name: String,
    pub category: String,
    pub description: String,
    pub signature: String,
    pub test_cases: Vec<(Vec<MogLiteral>, String)>,
    pub wrapper_template: String,
    pub reference_solution: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    pub problem_name: String,
    pub passed: bool,
    pub expected_output: String,
    pub actual_output: String,
    pub error: Option<String>,
}

// --- Helper functions ---

fn parse_signature_params(signature: &str) -> Vec<(String, String)> {
    let start = match signature.find("fn ") {
        Some(s) => s + 3,
        None => return Vec::new(),
    };
    let rest = &signature[start..];
    let paren_start = match rest.find('(') {
        Some(s) => s,
        None => return Vec::new(),
    };
    let paren_end = match rest.find(")") {
        Some(s) => s,
        None => return Vec::new(),
    };
    if paren_start + 1 >= paren_end {
        return Vec::new();
    }
    let body = &rest[paren_start + 1..paren_end];
    body.split(',')
        .filter_map(|part| {
            let part = part.trim();
            if part.is_empty() { return None; }
            let mut parts = part.splitn(2, ':');
            let name = parts.next()?.trim().to_string();
            let type_ann = parts.next()?.trim().to_string();
            Some((name, type_ann))
        })
        .collect()
}

fn extract_fn_name(signature: &str) -> String {
    let start = signature.find("fn ").unwrap() + 3;
    let rest = &signature[start..];
    let end = rest.find('(').unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

fn build_wrapper(function_name: &str, signature: &str, test_cases: &[(Vec<MogLiteral>, String)]) -> String {
    let params = parse_signature_params(signature);
    let mut lines = vec!["fn main() -> i64 {".to_string()];

    for (case_idx, (args, _expected)) in test_cases.iter().enumerate() {
        let mut call_args: Vec<String> = Vec::new();
        for (arg_idx, arg) in args.iter().enumerate() {
            let param_type = if arg_idx < params.len() { Some(&params[arg_idx].1) } else { None };
            match (arg, param_type) {
                (MogLiteral::Array(_), Some(pt)) => {
                    let var_name = format!("arg_{}_{}", case_idx, arg_idx);
                    lines.push(format!("    {}: {} = {};", var_name, pt, arg));
                    call_args.push(var_name);
                }
                (MogLiteral::RawCode(code), Some(pt)) => {
                    let var_name = format!("arg_{}_{}", case_idx, arg_idx);
                    lines.push(format!("    {}: {} = {};", var_name, pt, code));
                    call_args.push(var_name);
                }
                _ => {
                    call_args.push(format!("{}", arg));
                }
            }
        }
        let arg_src = call_args.join(", ");
        lines.push(format!("    println_i64({}({}));", function_name, arg_src));
    }
    lines.push("    return 0;".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

fn expected_stdout(test_cases: &[(Vec<MogLiteral>, String)]) -> String {
    test_cases.iter().map(|(_, expected)| expected.as_str()).collect::<Vec<_>>().join("\n")
}

// --- Evaluation ---

pub fn evaluate_solution(problem: &BenchmarkProblem, generated_code: &str) -> BenchmarkResult {
    let program = format!("{}\n\n{}\n", generated_code.trim_end(), problem.wrapper_template);
    let result = interpreter::interpret(&program);
    let expected = expected_stdout(&problem.test_cases);
    let actual = result.output.trim().to_string();
    BenchmarkResult {
        problem_name: problem.name.clone(),
        passed: result.success && actual == expected.trim(),
        expected_output: expected.trim().to_string(),
        actual_output: actual,
        error: result.error,
    }
}

// --- Problem construction ---

fn make_problem(name: &str, category: &str, description: &str, signature: &str,
                test_cases: Vec<(Vec<MogLiteral>, String)>, reference_solution: &str) -> BenchmarkProblem {
    let fn_name = extract_fn_name(signature);
    BenchmarkProblem {
        name: name.to_string(),
        category: category.to_string(),
        description: description.to_string(),
        signature: signature.to_string(),
        wrapper_template: build_wrapper(&fn_name, signature, &test_cases),
        test_cases,
        reference_solution: Some(reference_solution.trim().to_string()),
    }
}

// --- Simple RNG for benchmark generation ---

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn next_i64(&mut self, min: i64, max: i64) -> i64 {
        let range = (max - min + 1) as u64;
        if range == 0 { return min; }
        (self.next_u64() % range) as i64 + min
    }

    pub fn next_bool(&mut self) -> bool {
        self.next_u64() % 2 == 0
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// --- Problem factories ---

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn factorial(n: i64) -> i64 {
    let mut r = 1i64;
    for i in 2..=n { r *= i; }
    r
}

fn fibonacci(n: i64) -> i64 {
    let (mut a, mut b) = (0i64, 1i64);
    for _ in 0..n { let t = b; b = a + b; a = t; }
    a
}

fn digit_sum(n: i64) -> i64 {
    let mut n = n.abs();
    let mut s = 0i64;
    while n > 0 { s += n % 10; n /= 10; }
    s
}

pub fn make_add_two(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let a = rng.next_i64(-50, 50);
        let b = rng.next_i64(-50, 50);
        tests.push((vec![MogLiteral::Int(a), MogLiteral::Int(b)], (a + b).to_string()));
    }
    make_problem(&format!("add_two_v{}", variant), "arithmetic",
        "Return the sum of two i64 integers.",
        "fn add_two(a: i64, b: i64) -> i64", tests,
        "fn add_two(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n")
}

pub fn make_abs_diff(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let a = rng.next_i64(-30, 30);
        let b = rng.next_i64(-30, 30);
        tests.push((vec![MogLiteral::Int(a), MogLiteral::Int(b)], (a - b).abs().to_string()));
    }
    make_problem(&format!("abs_diff_v{}", variant), "arithmetic",
        "Return the absolute difference between two integers.",
        "fn abs_diff(a: i64, b: i64) -> i64", tests,
        "fn abs_diff(a: i64, b: i64) -> i64 {\n    if a > b { return a - b; } else { return b - a; }\n}\n")
}

pub fn make_max2(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let a = rng.next_i64(-40, 40);
        let b = rng.next_i64(-40, 40);
        tests.push((vec![MogLiteral::Int(a), MogLiteral::Int(b)], a.max(b).to_string()));
    }
    make_problem(&format!("max2_v{}", variant), "control_flow",
        "Return the larger of two integers.",
        "fn max2(a: i64, b: i64) -> i64", tests,
        "fn max2(a: i64, b: i64) -> i64 {\n    if a > b { return a; } else { return b; }\n}\n")
}

pub fn make_clamp(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let x = rng.next_i64(-50, 150);
        tests.push((vec![MogLiteral::Int(x)], x.max(0).min(100).to_string()));
    }
    make_problem(&format!("clamp_0_100_v{}", variant), "control_flow",
        "Clamp x into the closed range [0, 100].",
        "fn clamp_0_100(x: i64) -> i64", tests,
        "fn clamp_0_100(x: i64) -> i64 {\n    if x < 0 { return 0; }\n    if x > 100 { return 100; }\n    return x;\n}\n")
}

pub fn make_sign(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let x = rng.next_i64(-20, 20);
        let s = if x < 0 { -1 } else if x > 0 { 1 } else { 0 };
        tests.push((vec![MogLiteral::Int(x)], s.to_string()));
    }
    make_problem(&format!("sign_v{}", variant), "control_flow",
        "Return -1 for negative, 0 for zero, and 1 for positive.",
        "fn sign(x: i64) -> i64", tests,
        "fn sign(x: i64) -> i64 {\n    if x < 0 { return -1; }\n    if x > 0 { return 1; }\n    return 0;\n}\n")
}

pub fn make_sum_to_n(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let n = rng.next_i64(0, 20);
        tests.push((vec![MogLiteral::Int(n)], (1..=n).sum::<i64>().to_string()));
    }
    make_problem(&format!("sum_to_n_v{}", variant), "arithmetic",
        "Return 1 + 2 + ... + n. For n <= 0 return 0.",
        "fn sum_to_n(n: i64) -> i64", tests,
        "fn sum_to_n(n: i64) -> i64 {\n    if n <= 0 { return 0; }\n    total: i64 = 0;\n    for i := 1 to (n + 1) { total = total + i; }\n    return total;\n}\n")
}

pub fn make_gcd(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let a = rng.next_i64(1, 60);
        let b = rng.next_i64(1, 60);
        tests.push((vec![MogLiteral::Int(a), MogLiteral::Int(b)], gcd(a, b).to_string()));
    }
    make_problem(&format!("gcd_v{}", variant), "arithmetic",
        "Return the greatest common divisor of two positive integers.",
        "fn gcd(a: i64, b: i64) -> i64", tests,
        "fn gcd(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n")
}

pub fn make_lcm(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let a = rng.next_i64(1, 20);
        let b = rng.next_i64(1, 20);
        tests.push((vec![MogLiteral::Int(a), MogLiteral::Int(b)],
            ((a * b) / gcd(a, b)).to_string()));
    }
    make_problem(&format!("lcm_v{}", variant), "arithmetic",
        "Return the least common multiple of two positive integers.",
        "fn lcm(a: i64, b: i64) -> i64", tests,
        "fn lcm(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return (a * b) / x;\n}\n")
}

pub fn make_array_sum(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let len = rng.next_i64(1, 6) as usize;
        let arr: Vec<MogLiteral> = (0..len).map(|_| MogLiteral::Int(rng.next_i64(0, 9))).collect();
        let sum: i64 = arr.iter().map(|v| if let MogLiteral::Int(i) = v { *i } else { 0 }).sum();
        tests.push((vec![MogLiteral::Array(arr)], sum.to_string()));
    }
    make_problem(&format!("array_sum_v{}", variant), "arrays",
        "Return the sum of all elements in an array of i64 values.",
        "fn array_sum(arr: [i64]) -> i64", tests,
        "fn array_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr { total = total + item; }\n    return total;\n}\n")
}

pub fn make_array_max(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let len = rng.next_i64(1, 6) as usize;
        let arr: Vec<MogLiteral> = (0..len).map(|_| MogLiteral::Int(rng.next_i64(-9, 20))).collect();
        let mx: i64 = arr.iter().map(|v| if let MogLiteral::Int(i) = v { *i } else { i64::MIN }).max().unwrap_or(0);
        tests.push((vec![MogLiteral::Array(arr)], mx.to_string()));
    }
    make_problem(&format!("array_max_v{}", variant), "arrays",
        "Return the largest element in a non-empty array.",
        "fn array_max(arr: [i64]) -> i64", tests,
        "fn array_max(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr { if item > best { best = item; } }\n    return best;\n}\n")
}

pub fn make_count_occurrences(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let len = rng.next_i64(3, 7) as usize;
        let arr: Vec<MogLiteral> = (0..len).map(|_| MogLiteral::Int(rng.next_i64(0, 4))).collect();
        let target = rng.next_i64(0, 4);
        let count = arr.iter().filter(|v| if let MogLiteral::Int(i) = v { *i == target } else { false }).count() as i64;
        tests.push((vec![MogLiteral::Array(arr), MogLiteral::Int(target)], count.to_string()));
    }
    make_problem(&format!("count_occurrences_v{}", variant), "arrays",
        "Count how many times target appears in arr.",
        "fn count_occurrences(arr: [i64], target: i64) -> i64", tests,
        "fn count_occurrences(arr: [i64], target: i64) -> i64 {\n    count: i64 = 0;\n    for item in arr { if item == target { count = count + 1; } }\n    return count;\n}\n")
}

pub fn make_trimmed_len(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut words = vec![" mog ", "  diffusion", "compiler  ", "  hello world  "];
    rng.shuffle(&mut words);
    let tests: Vec<(Vec<MogLiteral>, String)> = words[..4].iter()
        .map(|s| (vec![MogLiteral::Str(s.to_string())], s.trim().len().to_string()))
        .collect();
    make_problem(&format!("trimmed_len_v{}", variant), "strings",
        "Trim leading and trailing spaces and return the remaining length.",
        "fn trimmed_len(s: string) -> i64", tests,
        "fn trimmed_len(s: string) -> i64 {\n    t := s.trim();\n    return t.len;\n}\n")
}

pub fn make_vowel_count(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut samples = vec!["mog", "aeiou", "banana", "rhythm", "interpreter", "compiler"];
    rng.shuffle(&mut samples);
    let tests: Vec<(Vec<MogLiteral>, String)> = samples[..4].iter()
        .map(|s| {
            let count = s.chars().filter(|c| "aeiou".contains(*c)).count() as i64;
            (vec![MogLiteral::Str(s.to_string())], count.to_string())
        })
        .collect();
    make_problem(&format!("vowel_count_v{}", variant), "strings",
        "Count vowels (a, e, i, o, u) in a lowercase ASCII string.",
        "fn vowel_count(s: string) -> i64", tests,
        "fn vowel_count(s: string) -> i64 {\n    chars := s.split(\"\");\n    total: i64 = 0;\n    for ch in chars {\n        if ch == \"a\" { total = total + 1; }\n        if ch == \"e\" { total = total + 1; }\n        if ch == \"i\" { total = total + 1; }\n        if ch == \"o\" { total = total + 1; }\n        if ch == \"u\" { total = total + 1; }\n    }\n    return total;\n}\n")
}

pub fn make_contains_cat(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut samples = vec!["cat", "scatter", "dog", "catalog", "hello", "copycat"];
    rng.shuffle(&mut samples);
    let tests: Vec<(Vec<MogLiteral>, String)> = samples[..4].iter()
        .map(|s| (vec![MogLiteral::Str(s.to_string())], if s.contains("cat") { "1" } else { "0" }.to_string()))
        .collect();
    make_problem(&format!("contains_cat_v{}", variant), "strings",
        "Return 1 if the string contains the substring 'cat', else 0.",
        "fn contains_cat(s: string) -> i64", tests,
        "fn contains_cat(s: string) -> i64 {\n    if s.contains(\"cat\") { return 1; }\n    return 0;\n}\n")
}

pub fn make_point_sum(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let x = rng.next_i64(-10, 10);
        let y = rng.next_i64(-10, 10);
        tests.push((vec![MogLiteral::RawCode(format!("Point {{ x: {}, y: {} }}", x, y))], (x + y).to_string()));
    }
    make_problem(&format!("point_sum_v{}", variant), "structs",
        "Define struct Point { x: i64, y: i64 } and return x + y.",
        "fn point_sum(p: Point) -> i64", tests,
        "struct Point {\n    x: i64,\n    y: i64,\n}\n\nfn point_sum(p: Point) -> i64 {\n    return p.x + p.y;\n}\n")
}

pub fn make_safe_div_or_neg1(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let a = rng.next_i64(1, 50);
        let b = if rng.next_bool() { 0 } else { rng.next_i64(1, 9) };
        tests.push((vec![MogLiteral::Int(a), MogLiteral::Int(b)],
            if b == 0 { -1 } else { a / b }.to_string()));
    }
    make_problem(&format!("safe_div_or_neg1_v{}", variant), "control_flow",
        "Divide a by b. If b is zero, return -1.",
        "fn safe_div_or_neg1(a: i64, b: i64) -> i64", tests,
        "fn safe_div_or_neg1(a: i64, b: i64) -> i64 {\n    if b == 0 { return -1; }\n    return a / b;\n}\n")
}

pub fn make_positive_or_default(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let x = rng.next_i64(-20, 20);
        tests.push((vec![MogLiteral::Int(x)], if x > 0 { x } else { 0 }.to_string()));
    }
    make_problem(&format!("positive_or_default_v{}", variant), "control_flow",
        "Return x if x is positive, otherwise return 0.",
        "fn positive_or_default(x: i64) -> i64", tests,
        "fn positive_or_default(x: i64) -> i64 {\n    if x > 0 { return x; }\n    return 0;\n}\n")
}

pub fn make_factorial(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let n = rng.next_i64(0, 8);
        tests.push((vec![MogLiteral::Int(n)], factorial(n).to_string()));
    }
    make_problem(&format!("factorial_v{}", variant), "recursion",
        "Return n! recursively.",
        "fn factorial(n: i64) -> i64", tests,
        "fn factorial(n: i64) -> i64 {\n    if n <= 1 { return 1; }\n    return n * factorial(n - 1);\n}\n")
}

pub fn make_fibonacci(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let n = rng.next_i64(0, 12);
        tests.push((vec![MogLiteral::Int(n)], fibonacci(n).to_string()));
    }
    make_problem(&format!("fibonacci_v{}", variant), "recursion",
        "Return the nth Fibonacci number.",
        "fn fibonacci(n: i64) -> i64", tests,
        "fn fibonacci(n: i64) -> i64 {\n    if n <= 0 { return 0; }\n    if n == 1 { return 1; }\n    return fibonacci(n - 1) + fibonacci(n - 2);\n}\n")
}

pub fn make_closure_map_sum(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let arr: Vec<MogLiteral> = (0..3).map(|_| MogLiteral::Int(rng.next_i64(1, 5))).collect();
        let sum: i64 = arr.iter().map(|v| if let MogLiteral::Int(i) = v { i * 2 } else { 0 }).sum();
        tests.push((vec![MogLiteral::Array(arr)], sum.to_string()));
    }
    make_problem(&format!("closure_map_sum_v{}", variant), "higher_order",
        "Double every array element with .map() and return the sum.",
        "fn closure_map_sum(arr: [i64]) -> i64", tests,
        "fn closure_map_sum(arr: [i64]) -> i64 {\n    doubled := arr.map(fn(x: i64) -> i64 { return x * 2; });\n    total: i64 = 0;\n    for item in doubled { total = total + item; }\n    return total;\n}\n")
}

pub fn make_count_positive(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let arr: Vec<MogLiteral> = (0..5).map(|_| MogLiteral::Int(rng.next_i64(-4, 4))).collect();
        let count = arr.iter().filter(|v| if let MogLiteral::Int(i) = v { *i > 0 } else { false }).count() as i64;
        tests.push((vec![MogLiteral::Array(arr)], count.to_string()));
    }
    make_problem(&format!("count_positive_v{}", variant), "arrays",
        "Count how many elements in the array are greater than zero.",
        "fn count_positive(arr: [i64]) -> i64", tests,
        "fn count_positive(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr { if item > 0 { total = total + 1; } }\n    return total;\n}\n")
}

pub fn make_is_even(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let x = rng.next_i64(-20, 20);
        tests.push((vec![MogLiteral::Int(x)], if x % 2 == 0 { "1" } else { "0" }.to_string()));
    }
    make_problem(&format!("is_even_v{}", variant), "control_flow",
        "Return 1 if x is even, otherwise 0.",
        "fn is_even(x: i64) -> i64", tests,
        "fn is_even(x: i64) -> i64 {\n    if (x % 2) == 0 { return 1; }\n    return 0;\n}\n")
}

pub fn make_digit_sum(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let n = rng.next_i64(0, 9999);
        tests.push((vec![MogLiteral::Int(n)], digit_sum(n).to_string()));
    }
    make_problem(&format!("digit_sum_v{}", variant), "arithmetic",
        "Return the sum of the decimal digits of n.",
        "fn digit_sum(n: i64) -> i64", tests,
        "fn digit_sum(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 { x = 0 - x; }\n    total: i64 = 0;\n    while x > 0 { total = total + (x % 10); x = x / 10; }\n    return total;\n}\n")
}

pub fn make_starts_with_m(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut samples = vec!["mog", "metal", "apple", "mars", "code", "map"];
    rng.shuffle(&mut samples);
    let tests: Vec<(Vec<MogLiteral>, String)> = samples[..4].iter()
        .map(|s| (vec![MogLiteral::Str(s.to_string())], if s.starts_with('m') { "1" } else { "0" }.to_string()))
        .collect();
    make_problem(&format!("starts_with_m_v{}", variant), "strings",
        "Return 1 if s starts with the lowercase letter m, else 0.",
        "fn starts_with_m(s: string) -> i64", tests,
        "fn starts_with_m(s: string) -> i64 {\n    if s.starts_with(\"m\") { return 1; }\n    return 0;\n}\n")
}

pub fn make_rectangle_area(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let w = rng.next_i64(1, 10);
        let h = rng.next_i64(1, 10);
        tests.push((vec![MogLiteral::RawCode(format!("Rectangle {{ width: {}, height: {} }}", w, h))],
            (w * h).to_string()));
    }
    make_problem(&format!("rectangle_area_v{}", variant), "structs",
        "Define struct Rectangle { width: i64, height: i64 } and return its area.",
        "fn rectangle_area(r: Rectangle) -> i64", tests,
        "struct Rectangle {\n    width: i64,\n    height: i64,\n}\n\nfn rectangle_area(r: Rectangle) -> i64 {\n    return r.width * r.height;\n}\n")
}

pub fn make_power(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let base = rng.next_i64(2, 5);
        let exp = rng.next_i64(0, 4);
        tests.push((vec![MogLiteral::Int(base), MogLiteral::Int(exp)],
            base.pow(exp as u32).to_string()));
    }
    make_problem(&format!("power_v{}", variant), "arithmetic",
        "Compute base raised to the power exp (non-negative).",
        "fn power(base: i64, exp: i64) -> i64", tests,
        "fn power(base: i64, exp: i64) -> i64 {\n    if exp == 0 { return 1; }\n    result: i64 = 1;\n    i: i64 = 0;\n    while i < exp { result = result * base; i = i + 1; }\n    return result;\n}\n")
}

pub fn make_polynomial(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let x = rng.next_i64(0, 5);
        tests.push((vec![MogLiteral::Int(x)], (2 * x * x + 3 * x + 1).to_string()));
    }
    make_problem(&format!("polynomial_v{}", variant), "arithmetic",
        "Evaluate the polynomial 2*x*x + 3*x + 1.",
        "fn polynomial(x: i64) -> i64", tests,
        "fn polynomial(x: i64) -> i64 {\n    return 2 * x * x + 3 * x + 1;\n}\n")
}

pub fn make_collatz_steps(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for n in [1, 2, 3, 6, 7, 10, 27] {
        let (mut c, mut steps) = (n, 0i64);
        while c > 1 {
            c = if c % 2 == 0 { c / 2 } else { 3 * c + 1 };
            steps += 1;
        }
        tests.push((vec![MogLiteral::Int(n)], steps.to_string()));
    }
    make_problem(&format!("collatz_steps_v{}", variant), "loops",
        "Count how many steps it takes for the Collatz sequence starting at n to reach 1.",
        "fn collatz_steps(n: i64) -> i64", tests,
        "fn collatz_steps(n: i64) -> i64 {\n    x: i64 = n;\n    steps: i64 = 0;\n    while x > 1 {\n        if x % 2 == 0 { x = x / 2; } else { x = 3 * x + 1; }\n        steps = steps + 1;\n    }\n    return steps;\n}\n")
}

pub fn make_min3(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..5 {
        let a = rng.next_i64(-10, 10);
        let b = rng.next_i64(-10, 10);
        let c = rng.next_i64(-10, 10);
        tests.push((vec![MogLiteral::Int(a), MogLiteral::Int(b), MogLiteral::Int(c)],
            a.min(b).min(c).to_string()));
    }
    make_problem(&format!("min3_v{}", variant), "control_flow",
        "Return the minimum of three integers.",
        "fn min3(a: i64, b: i64, c: i64) -> i64", tests,
        "fn min3(a: i64, b: i64, c: i64) -> i64 {\n    m: i64 = a;\n    if b < m { m = b; }\n    if c < m { m = c; }\n    return m;\n}\n")
}

pub fn make_reverse_array(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let len = rng.next_i64(2, 5) as usize;
        let arr: Vec<MogLiteral> = (0..len).map(|_| MogLiteral::Int(rng.next_i64(1, 10))).collect();
        let sum: i64 = arr.iter().map(|v| if let MogLiteral::Int(i) = v { *i } else { 0 }).sum();
        tests.push((vec![MogLiteral::Array(arr)], sum.to_string()));
    }
    make_problem(&format!("reverse_sum_v{}", variant), "arrays",
        "Sum all elements of an array.",
        "fn reverse_sum(arr: [i64]) -> i64", tests,
        "fn reverse_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr { total = total + item; }\n    return total;\n}\n")
}

pub fn make_second_largest(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let len = rng.next_i64(3, 6) as usize;
        let arr: Vec<MogLiteral> = (0..len).map(|_| MogLiteral::Int(rng.next_i64(1, 20))).collect();
        let mx: i64 = arr.iter().map(|v| if let MogLiteral::Int(i) = v { *i } else { i64::MIN }).max().unwrap_or(0);
        tests.push((vec![MogLiteral::Array(arr)], mx.to_string()));
    }
    make_problem(&format!("array_max_v{}", variant), "arrays",
        "Find the maximum element in an array.",
        "fn array_max_elem(arr: [i64]) -> i64", tests,
        "fn array_max_elem(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr { if item > best { best = item; } }\n    return best;\n}\n")
}

pub fn make_is_prime(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for n in [2, 3, 4, 5, 7, 10, 11, 13, 15, 17] {
        let is_p = if n >= 2 { (2..=((n as f64).sqrt() as i64)).all(|d| n % d != 0) } else { false };
        tests.push((vec![MogLiteral::Int(n)], if is_p { "1" } else { "0" }.to_string()));
    }
    make_problem(&format!("is_prime_v{}", variant), "loops",
        "Return 1 if the number is prime, 0 otherwise.",
        "fn is_prime(n: i64) -> i64", tests,
        "fn is_prime(n: i64) -> i64 {\n    if n < 2 { return 0; }\n    if n == 2 { return 1; }\n    if n % 2 == 0 { return 0; }\n    i: i64 = 3;\n    while i * i <= n { if n % i == 0 { return 0; } i = i + 2; }\n    return 1;\n}\n")
}

pub fn make_nth_triangle(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for n in [0, 1, 2, 5, 10, 20] {
        tests.push((vec![MogLiteral::Int(n)], (n * (n + 1) / 2).to_string()));
    }
    make_problem(&format!("nth_triangle_v{}", variant), "loops",
        "Return the nth triangular number: 1+2+...+n.",
        "fn nth_triangle(n: i64) -> i64", tests,
        "fn nth_triangle(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n { total = total + i; i = i + 1; }\n    return total;\n}\n")
}

pub fn make_fib_iter(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for n in [0, 1, 2, 5, 7, 10] {
        tests.push((vec![MogLiteral::Int(n)], fibonacci(n).to_string()));
    }
    make_problem(&format!("fib_iter_v{}", variant), "loops",
        "Return the nth Fibonacci number using iterative multi-variable update.",
        "fn fib_iter(n: i64) -> i64", tests,
        "fn fib_iter(n: i64) -> i64 {\n    if n == 0 { return 0; }\n    if n == 1 { return 1; }\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n { tmp: i64 = a + b; a = b; b = tmp; i = i + 1; }\n    return b;\n}\n")
}

pub fn make_palindrome_check(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests = vec![
        (vec![MogLiteral::Str("racecar".into())], "1".into()),
        (vec![MogLiteral::Str("hello".into())], "0".into()),
        (vec![MogLiteral::Str("aba".into())], "1".into()),
        (vec![MogLiteral::Str("ab".into())], "0".into()),
        (vec![MogLiteral::Str("a".into())], "1".into()),
        (vec![MogLiteral::Str("".into())], "1".into()),
    ];
    make_problem(&format!("palindrome_check_v{}", variant), "strings",
        "Return 1 if the string is a palindrome, 0 otherwise.",
        "fn palindrome_check(s: string) -> i64", tests,
        "fn palindrome_check(s: string) -> i64 {\n    chars := s.split(\"\");\n    left: i64 = 0;\n    right: i64 = s.len - 1;\n    while left < right {\n        if chars[left] != chars[right] { return 0; }\n        left = left + 1;\n        right = right - 1;\n    }\n    return 1;\n}\n")
}

pub fn make_count_words(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests = vec![
        (vec![MogLiteral::Str("hello world".into())], "2".into()),
        (vec![MogLiteral::Str("one".into())], "1".into()),
        (vec![MogLiteral::Str("a b c d".into())], "4".into()),
        (vec![MogLiteral::Str("  two words  ".into())], "2".into()),
        (vec![MogLiteral::Str("".into())], "0".into()),
    ];
    make_problem(&format!("count_words_v{}", variant), "strings",
        "Count the number of space-separated words in a string.",
        "fn count_words(s: string) -> i64", tests,
        "fn count_words(s: string) -> i64 {\n    t := s.trim();\n    if t.len == 0 { return 0; }\n    parts := t.split(\" \");\n    count: i64 = 0;\n    for p in parts { if p.len > 0 { count = count + 1; } }\n    return count;\n}\n")
}

pub fn make_euler_totient(_rng: &mut Rng, _variant: i64) -> BenchmarkProblem {
    fn totient(n: i64) -> i64 {
        let (mut result, mut p, mut temp) = (n, 2, n);
        while p * p <= temp {
            if temp % p == 0 {
                while temp % p == 0 { temp /= p; }
                result -= result / p;
            }
            p += 1;
        }
        if temp > 1 { result -= result / temp; }
        result
    }
    let tests: Vec<(Vec<MogLiteral>, String)> = [1, 2, 3, 5, 6, 9, 10, 12].iter()
        .map(|&n| (vec![MogLiteral::Int(n)], totient(n).to_string()))
        .collect();
    let variant = _variant;
    make_problem(&format!("euler_totient_v{}", variant), "algorithms",
        "Compute Euler's totient function phi(n).",
        "fn euler_totient(n: i64) -> i64", tests,
        "fn euler_totient(n: i64) -> i64 {\n    result: i64 = n;\n    p: i64 = 2;\n    temp: i64 = n;\n    while p * p <= temp {\n        if temp % p == 0 {\n            while temp % p == 0 { temp = temp / p; }\n            result = result - result / p;\n        }\n        p = p + 1;\n    }\n    if temp > 1 { result = result - result / temp; }\n    return result;\n}\n")
}

pub fn make_sum_squares(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0, 1, 2, 3, 5, 10].iter()
        .map(|&n| (vec![MogLiteral::Int(n)], (1..=n).map(|i| i * i).sum::<i64>().to_string()))
        .collect();
    make_problem(&format!("sum_squares_v{}", variant), "loops",
        "Compute the sum of squares from 1 to n.",
        "fn sum_squares(n: i64) -> i64", tests,
        "fn sum_squares(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n { total = total + i * i; i = i + 1; }\n    return total;\n}\n")
}

pub fn make_product_1_to_n(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0, 1, 2, 4, 6].iter()
        .map(|&n| {
            let p: i64 = (1..=n).product();
            (vec![MogLiteral::Int(n)], p.to_string())
        })
        .collect();
    make_problem(&format!("product_1_to_n_v{}", variant), "loops",
        "Compute the product of all integers from 1 to n.",
        "fn product_1_to_n(n: i64) -> i64", tests,
        "fn product_1_to_n(n: i64) -> i64 {\n    if n == 0 { return 1; }\n    total: i64 = 1;\n    i: i64 = 1;\n    while i <= n { total = total * i; i = i + 1; }\n    return total;\n}\n")
}

pub fn make_count_divisors(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [1, 2, 6, 12, 7, 10].iter()
        .map(|&n| {
            let count = (1..=n).filter(|d| n % d == 0).count() as i64;
            (vec![MogLiteral::Int(n)], count.to_string())
        })
        .collect();
    make_problem(&format!("count_divisors_v{}", variant), "loops",
        "Count how many positive divisors n has.",
        "fn count_divisors(n: i64) -> i64", tests,
        "fn count_divisors(n: i64) -> i64 {\n    count: i64 = 0;\n    i: i64 = 1;\n    while i <= n { if n % i == 0 { count = count + 1; } i = i + 1; }\n    return count;\n}\n")
}

pub fn make_triangular_check(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0, 1, 3, 6, 10, 15, 2, 4, 7, 8].iter()
        .map(|&n| {
            let is_tri = (0..=n).any(|k| k * (k + 1) / 2 == n) as i64;
            (vec![MogLiteral::Int(n)], is_tri.to_string())
        })
        .collect();
    make_problem(&format!("triangular_check_v{}", variant), "algorithms",
        "Return 1 if n is a triangular number, 0 otherwise.",
        "fn triangular_check(n: i64) -> i64", tests,
        "fn triangular_check(n: i64) -> i64 {\n    k: i64 = 0;\n    while k * (k + 1) / 2 <= n {\n        if k * (k + 1) / 2 == n { return 1; }\n        k = k + 1;\n    }\n    return 0;\n}\n")
}

pub fn make_max_pair_diff(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let len = rng.next_i64(3, 6) as usize;
        let arr: Vec<MogLiteral> = (0..len.max(2)).map(|_| MogLiteral::Int(rng.next_i64(1, 20))).collect();
        let nums: Vec<i64> = arr.iter().map(|v| if let MogLiteral::Int(i) = v { *i } else { 0 }).collect();
        let max_diff = (0..nums.len()-1).map(|i| (nums[i] - nums[i+1]).abs()).max().unwrap_or(0);
        tests.push((vec![MogLiteral::Array(arr)], max_diff.to_string()));
    }
    make_problem(&format!("max_pair_diff_v{}", variant), "arrays",
        "Find the maximum absolute difference between consecutive elements.",
        "fn max_pair_diff(arr: [i64]) -> i64", tests,
        "fn max_pair_diff(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len {\n        diff: i64 = arr[i] - arr[i - 1];\n        if diff < 0 { diff = 0 - diff; }\n        if diff > best { best = diff; }\n        i = i + 1;\n    }\n    return best;\n}\n")
}

pub fn make_sum_negatives(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let arr: Vec<MogLiteral> = (0..rng.next_i64(3, 6) as usize)
            .map(|_| MogLiteral::Int(rng.next_i64(-10, 10))).collect();
        let sum: i64 = arr.iter().filter_map(|v| if let MogLiteral::Int(i) = v { if *i < 0 { Some(*i) } else { None } } else { None }).sum();
        tests.push((vec![MogLiteral::Array(arr)], sum.to_string()));
    }
    make_problem(&format!("sum_negatives_v{}", variant), "arrays",
        "Sum all negative numbers in the array.",
        "fn sum_negatives(arr: [i64]) -> i64", tests,
        "fn sum_negatives(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr { if item < 0 { total = total + item; } }\n    return total;\n}\n")
}

pub fn make_gcd_extended(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [(12, 8), (35, 14), (7, 13), (100, 75), (0, 5), (6, 0)].iter()
        .map(|&(a, b)| {
            let (mut x, mut y) = (a, b);
            while y != 0 { let t = y; y = x % y; x = t; }
            (vec![MogLiteral::Int(a), MogLiteral::Int(b)], x.to_string())
        })
        .collect();
    make_problem(&format!("gcd_extended_v{}", variant), "algorithms",
        "Compute the GCD of two non-negative integers using Euclidean algorithm.",
        "fn gcd_extended(a: i64, b: i64) -> i64", tests,
        "fn gcd_extended(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 { tmp: i64 = y; y = x % y; x = tmp; }\n    return x;\n}\n")
}

pub fn make_harmonic_sum(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [1, 2, 5, 10].iter()
        .map(|&n| {
            let h: i64 = (1..=n).map(|i| 1000 / i).sum();
            (vec![MogLiteral::Int(n)], h.to_string())
        })
        .collect();
    make_problem(&format!("harmonic_sum_v{}", variant), "loops",
        "Compute integer harmonic sum: sum of 1000/i for i from 1 to n.",
        "fn harmonic_sum(n: i64) -> i64", tests,
        "fn harmonic_sum(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n { total = total + 1000 / i; i = i + 1; }\n    return total;\n}\n")
}

pub fn make_interactive_sum(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..4 {
        let arr: Vec<MogLiteral> = (0..rng.next_i64(2, 5) as usize)
            .map(|_| MogLiteral::Int(rng.next_i64(1, 10))).collect();
        let sum: i64 = arr.iter().map(|v| if let MogLiteral::Int(i) = v { *i } else { 0 }).sum();
        tests.push((vec![MogLiteral::Array(arr)], sum.to_string()));
    }
    make_problem(&format!("interactive_sum_v{}", variant), "arrays",
        "Return the sum of all integers in an array.",
        "fn interactive_sum(arr: [i64]) -> i64", tests,
        "fn interactive_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr { total = total + item; }\n    return total;\n}\n")
}

// --- Phase 1: Scalar Digit Manipulation ---

pub fn make_reverse_digits(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0i64, 9, 12, 123, 100, 4321].iter()
        .map(|&n| {
            let mut result = 0i64;
            let mut x = n;
            while x > 0 { result = result * 10 + x % 10; x /= 10; }
            (vec![MogLiteral::Int(n)], result.to_string())
        })
        .collect();
    make_problem(&format!("reverse_digits_v{}", variant), "loops",
        "Reverse the digits of a non-negative integer (e.g. 123 -> 321).",
        "fn reverse_digits(n: i64) -> i64", tests,
        "fn reverse_digits(n: i64) -> i64 {\n    result: i64 = 0;\n    x: i64 = n;\n    while x > 0 {\n        result = result * 10 + x % 10;\n        x = x / 10;\n    }\n    return result;\n}\n")
}

pub fn make_digit_count(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0i64, 1, 9, 10, 99, 100, 999].iter()
        .map(|&n| {
            let count = if n == 0 { 1 } else {
                let mut c = 0i64; let mut x = n;
                while x > 0 { c += 1; x /= 10; } c
            };
            (vec![MogLiteral::Int(n)], count.to_string())
        })
        .collect();
    make_problem(&format!("digit_count_v{}", variant), "loops",
        "Count the number of digits in a non-negative integer.",
        "fn digit_count(n: i64) -> i64", tests,
        "fn digit_count(n: i64) -> i64 {\n    if n == 0 { return 1; }\n    count: i64 = 0;\n    x: i64 = n;\n    while x > 0 { count = count + 1; x = x / 10; }\n    return count;\n}\n")
}

pub fn make_count_even_digits(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0i64, 2, 13, 24, 135, 2468].iter()
        .map(|&n| {
            let count = if n == 0 { 1 } else {
                let mut c = 0i64; let mut x = n;
                while x > 0 { if (x % 10) % 2 == 0 { c += 1; } x /= 10; } c
            };
            (vec![MogLiteral::Int(n)], count.to_string())
        })
        .collect();
    make_problem(&format!("count_even_digits_v{}", variant), "loops",
        "Count how many digits of n are even.",
        "fn count_even_digits(n: i64) -> i64", tests,
        "fn count_even_digits(n: i64) -> i64 {\n    if n == 0 { return 1; }\n    count: i64 = 0;\n    x: i64 = n;\n    while x > 0 {\n        if (x % 10) % 2 == 0 { count = count + 1; }\n        x = x / 10;\n    }\n    return count;\n}\n")
}

// --- Phase 2: Algorithmic Scalar ---

pub fn make_perfect_check(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [1i64, 6, 12, 28, 30, 496].iter()
        .map(|&n| {
            let is_perfect = if n < 2 { 0 } else {
                let mut total = 1i64; let mut i = 2i64;
                while i * i <= n {
                    if n % i == 0 { total += i; if i * i != n { total += n / i; } }
                    i += 1;
                }
                if total == n { 1 } else { 0 }
            };
            (vec![MogLiteral::Int(n)], is_perfect.to_string())
        })
        .collect();
    make_problem(&format!("perfect_check_v{}", variant), "algorithms",
        "Return 1 if n is a perfect number (sum of proper divisors equals n), 0 otherwise.",
        "fn perfect_check(n: i64) -> i64", tests,
        "fn perfect_check(n: i64) -> i64 {\n    if n < 2 { return 0; }\n    total: i64 = 1;\n    i: i64 = 2;\n    while i * i <= n {\n        if n % i == 0 {\n            total = total + i;\n            if i * i != n { total = total + n / i; }\n        }\n        i = i + 1;\n    }\n    if total == n { return 1; }\n    return 0;\n}\n")
}

pub fn make_armstrong_check(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0i64, 1, 153, 370, 371, 100, 200].iter()
        .map(|&n| {
            let mut total = 0i64; let mut x = n;
            while x > 0 { let d = x % 10; total += d * d * d; x /= 10; }
            let is_arm = if total == n { 1 } else { 0 };
            (vec![MogLiteral::Int(n)], is_arm.to_string())
        })
        .collect();
    make_problem(&format!("armstrong_check_v{}", variant), "algorithms",
        "Return 1 if n is an Armstrong number (sum of cubes of its digits equals n), 0 otherwise.",
        "fn armstrong_check(n: i64) -> i64", tests,
        "fn armstrong_check(n: i64) -> i64 {\n    total: i64 = 0;\n    x: i64 = n;\n    while x > 0 {\n        d: i64 = x % 10;\n        total = total + d * d * d;\n        x = x / 10;\n    }\n    if total == n { return 1; }\n    return 0;\n}\n")
}

pub fn make_geometric_sum(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0i64, 1, 2, 3, 4, 6].iter()
        .map(|&n| {
            let result = (1i64 << (n + 1)) - 1;
            (vec![MogLiteral::Int(n)], result.to_string())
        })
        .collect();
    make_problem(&format!("geometric_sum_v{}", variant), "loops",
        "Compute 1 + 2 + 4 + ... + 2^n (geometric series with ratio 2).",
        "fn geometric_sum(n: i64) -> i64", tests,
        "fn geometric_sum(n: i64) -> i64 {\n    total: i64 = 1;\n    power: i64 = 1;\n    i: i64 = 0;\n    while i < n { power = power * 2; total = total + power; i = i + 1; }\n    return total;\n}\n")
}

pub fn make_nested_sum(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0i64, 1, 2, 3, 4].iter()
        .map(|&n| {
            let mut total = 0i64;
            for i in 1..=n { for j in 1..=i { total += i * j; } }
            (vec![MogLiteral::Int(n)], total.to_string())
        })
        .collect();
    make_problem(&format!("nested_sum_v{}", variant), "loops",
        "Compute sum(i*j for i=1..n, j=1..i) using nested loops.",
        "fn nested_sum(n: i64) -> i64", tests,
        "fn nested_sum(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        j: i64 = 1;\n        while j <= i { total = total + i * j; j = j + 1; }\n        i = i + 1;\n    }\n    return total;\n}\n")
}

// --- Phase 3: Early-Return Array Search ---

pub fn make_find_first_even(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let cases: Vec<Vec<i64>> = vec![
        vec![2, 3, 5], vec![1, 3, 4, 6], vec![1, 3, 5], vec![4], vec![7, 8, 9],
    ];
    let tests: Vec<(Vec<MogLiteral>, String)> = cases.iter()
        .map(|arr| {
            let idx = arr.iter().position(|&x| x % 2 == 0).map(|i| i as i64).unwrap_or(-1);
            let lit = arr.iter().map(|&x| MogLiteral::Int(x)).collect();
            (vec![MogLiteral::Array(lit)], idx.to_string())
        })
        .collect();
    make_problem(&format!("find_first_even_v{}", variant), "arrays",
        "Return the index of the first even element in arr, or -1 if no even element.",
        "fn find_first_even(arr: [i64]) -> i64", tests,
        "fn find_first_even(arr: [i64]) -> i64 {\n    i: i64 = 0;\n    while i < arr.len {\n        if arr[i] % 2 == 0 { return i; }\n        i = i + 1;\n    }\n    return -1;\n}\n")
}

pub fn make_sum_until_negative(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let cases: Vec<Vec<i64>> = vec![
        vec![1, 2, -3, 4], vec![5, 3, 1], vec![-1, 2, 3], vec![0, 5, -2, 7],
    ];
    let tests: Vec<(Vec<MogLiteral>, String)> = cases.iter()
        .map(|arr| {
            let mut total = 0i64;
            for &x in arr { if x < 0 { break; } total += x; }
            let lit = arr.iter().map(|&x| MogLiteral::Int(x)).collect();
            (vec![MogLiteral::Array(lit)], total.to_string())
        })
        .collect();
    make_problem(&format!("sum_until_negative_v{}", variant), "arrays",
        "Sum elements of arr until (not including) the first negative element.",
        "fn sum_until_negative(arr: [i64]) -> i64", tests,
        "fn sum_until_negative(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        if arr[i] < 0 { return total; }\n        total = total + arr[i];\n        i = i + 1;\n    }\n    return total;\n}\n")
}

// --- Phase 4: Array Access Patterns ---

pub fn make_sort_and_sum(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..5 {
        let arr: Vec<i64> = (0..rng.next_i64(2, 6) as usize).map(|_| rng.next_i64(1, 20)).collect();
        let mn = *arr.iter().min().unwrap();
        let mx = *arr.iter().max().unwrap();
        let lit: Vec<MogLiteral> = arr.iter().map(|&x| MogLiteral::Int(x)).collect();
        tests.push((vec![MogLiteral::Array(lit)], (mn + mx).to_string()));
    }
    make_problem(&format!("sort_and_sum_v{}", variant), "arrays",
        "Return the sum of the minimum and maximum elements of arr.",
        "fn sort_and_sum(arr: [i64]) -> i64", tests,
        "fn sort_and_sum(arr: [i64]) -> i64 {\n    mn := arr[0];\n    mx := arr[0];\n    for item in arr {\n        if item < mn { mn = item; }\n        if item > mx { mx = item; }\n    }\n    return mn + mx;\n}\n")
}

pub fn make_array_triple(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..5 {
        let arr: Vec<i64> = (0..rng.next_i64(2, 5) as usize).map(|_| rng.next_i64(1, 10)).collect();
        let total: i64 = arr.iter().map(|&x| x * 3).sum();
        let lit: Vec<MogLiteral> = arr.iter().map(|&x| MogLiteral::Int(x)).collect();
        tests.push((vec![MogLiteral::Array(lit)], total.to_string()));
    }
    make_problem(&format!("array_triple_v{}", variant), "arrays",
        "Return the sum of each element multiplied by 3.",
        "fn array_triple(arr: [i64]) -> i64", tests,
        "fn array_triple(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr { total = total + item * 3; }\n    return total;\n}\n")
}

pub fn make_sum_even_indexed(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let cases: Vec<Vec<i64>> = vec![
        vec![1, 2, 3, 4, 5], vec![10, 20, 30], vec![7], vec![4, 8, 2, 6],
    ];
    let tests: Vec<(Vec<MogLiteral>, String)> = cases.iter()
        .map(|arr| {
            let total: i64 = arr.iter().step_by(2).sum();
            let lit: Vec<MogLiteral> = arr.iter().map(|&x| MogLiteral::Int(x)).collect();
            (vec![MogLiteral::Array(lit)], total.to_string())
        })
        .collect();
    make_problem(&format!("sum_even_indexed_v{}", variant), "arrays",
        "Sum elements at even indices (0, 2, 4, ...) of arr.",
        "fn sum_even_indexed(arr: [i64]) -> i64", tests,
        "fn sum_even_indexed(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len { total = total + arr[i]; i = i + 2; }\n    return total;\n}\n")
}

pub fn make_last_element(rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let mut tests = Vec::new();
    for _ in 0..5 {
        let arr: Vec<i64> = (0..rng.next_i64(1, 6) as usize).map(|_| rng.next_i64(1, 20)).collect();
        let last = *arr.last().unwrap();
        let lit: Vec<MogLiteral> = arr.iter().map(|&x| MogLiteral::Int(x)).collect();
        tests.push((vec![MogLiteral::Array(lit)], last.to_string()));
    }
    make_problem(&format!("last_element_v{}", variant), "arrays",
        "Return the last element of arr.",
        "fn last_element(arr: [i64]) -> i64", tests,
        "fn last_element(arr: [i64]) -> i64 {\n    return arr[arr.len - 1];\n}\n")
}

// --- Phase 5: Advanced Scalar ---

pub fn make_fib_cached(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [0i64, 1, 2, 5, 7, 10].iter()
        .map(|&n| {
            let (mut a, mut b) = (0i64, 1i64);
            for _ in 0..n { let t = a + b; a = b; b = t; }
            (vec![MogLiteral::Int(n)], a.to_string())
        })
        .collect();
    make_problem(&format!("fib_cached_v{}", variant), "loops",
        "Compute the nth Fibonacci number using iterative two-variable state.",
        "fn fib_cached(n: i64) -> i64", tests,
        "fn fib_cached(n: i64) -> i64 {\n    if n == 0 { return 0; }\n    if n == 1 { return 1; }\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n { tmp: i64 = a + b; a = b; b = tmp; i = i + 1; }\n    return b;\n}\n")
}

pub fn make_mersenne_check(_rng: &mut Rng, variant: i64) -> BenchmarkProblem {
    let tests: Vec<(Vec<MogLiteral>, String)> = [1i64, 3, 5, 7, 6, 15, 14, 31].iter()
        .map(|&n| {
            let is_m = if n < 1 { 0 } else {
                let mut m = n + 1;
                let mut ok = true;
                while m > 1 { if m % 2 != 0 { ok = false; break; } m /= 2; }
                if ok { 1 } else { 0 }
            };
            (vec![MogLiteral::Int(n)], is_m.to_string())
        })
        .collect();
    make_problem(&format!("mersenne_check_v{}", variant), "algorithms",
        "Return 1 if n is a Mersenne number (n = 2^k - 1), 0 otherwise.",
        "fn mersenne_check(n: i64) -> i64", tests,
        "fn mersenne_check(n: i64) -> i64 {\n    if n < 1 { return 0; }\n    m: i64 = n + 1;\n    while m > 1 {\n        if m % 2 != 0 { return 0; }\n        m = m / 2;\n    }\n    return 1;\n}\n")
}

// --- Factory list and benchmark generation ---

pub type FactoryFn = fn(&mut Rng, i64) -> BenchmarkProblem;

pub static FACTORIES: &[FactoryFn] = &[
    make_add_two, make_abs_diff, make_max2, make_clamp, make_sign,
    make_sum_to_n, make_gcd, make_lcm, make_array_sum, make_array_max,
    make_count_occurrences, make_trimmed_len, make_vowel_count, make_contains_cat,
    make_point_sum, make_safe_div_or_neg1, make_positive_or_default,
    make_factorial, make_fibonacci, make_closure_map_sum, make_count_positive,
    make_is_even, make_digit_sum, make_starts_with_m, make_rectangle_area,
    make_power, make_polynomial, make_collatz_steps, make_min3,
    make_reverse_array, make_second_largest, make_is_prime, make_nth_triangle,
    make_fib_iter, make_palindrome_check, make_count_words, make_euler_totient,
    make_sum_squares, make_product_1_to_n, make_count_divisors, make_triangular_check,
    make_max_pair_diff, make_sum_negatives, make_gcd_extended, make_harmonic_sum,
    make_interactive_sum,
    // Phase 1: Scalar Digit Manipulation
    make_reverse_digits, make_digit_count, make_count_even_digits,
    // Phase 2: Algorithmic Scalar
    make_perfect_check, make_armstrong_check, make_geometric_sum, make_nested_sum,
    // Phase 3: Early-Return Array Search
    make_find_first_even, make_sum_until_negative,
    // Phase 4: Array Access Patterns
    make_sort_and_sum, make_array_triple, make_sum_even_indexed, make_last_element,
    // Phase 5: Advanced Scalar
    make_fib_cached, make_mersenne_check,
];

pub fn get_benchmark(seed: u64, variants_per_factory: usize) -> Vec<BenchmarkProblem> {
    let mut rng = Rng::new(seed);
    let mut problems = Vec::new();
    for factory in FACTORIES {
        for variant in 0..variants_per_factory {
            problems.push(factory(&mut rng, variant as i64));
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_count() {
        assert_eq!(FACTORIES.len(), 61, "Expected 61 factories");
    }

    #[test]
    fn test_reference_solutions_pass() {
        let problems = get_benchmark(42, 1);
        let mut passed = 0;
        for problem in &problems {
            if let Some(ref solution) = problem.reference_solution {
                let result = evaluate_solution(problem, solution);
                if !result.passed {
                    panic!("FAIL: {} expected='{}' actual='{}' error={:?}",
                        problem.name, result.expected_output, result.actual_output, result.error);
                }
                passed += 1;
            }
        }
        assert_eq!(passed, 61, "All 61 reference solutions should pass");
    }

    #[test]
    fn test_benchmark_generates_correct_count() {
        let problems = get_benchmark(42, 5);
        assert_eq!(problems.len(), 305, "61 factories * 5 variants = 305 problems");
    }
}
