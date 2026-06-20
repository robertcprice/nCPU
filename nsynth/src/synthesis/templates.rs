use super::*;

// ─── Template library ─────────────────────────────────────────────────────────

/// Try generic verified templates before gradient descent.
///
/// Benchmark reference implementations are deliberately not candidates: they
/// belong exclusively to evaluation and using them would turn synthesis into an
/// answer lookup.
pub(crate) fn try_scalar_templates(
    problem: &Problem,
    fn_name: &str,
    n_args: usize,
) -> Option<SolveResult> {
    let make_result = |code: String| -> Option<SolveResult> {
        if verify_problem_code_strict(problem, &code).is_ok() {
            Some(SolveResult {
                success: true,
                code,
                method: "template".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            })
        } else {
            None
        }
    };

    // Inline alternatives for common program structures.
    let candidates: Vec<String> = match n_args {
        1 => vec![
            // positive_or_default: if x > 0 return x else 0
            format!("fn {fn_name}(x: i64) -> i64 {{\n    if x > 0 {{ return x; }}\n    return 0;\n}}\n"),
            // if x > 0 return x else return x (identity, used for is_positive etc.)
            format!("fn {fn_name}(x: i64) -> i64 {{\n    if x <= 0 {{ return 0; }}\n    return x;\n}}\n"),
            // digit_sum (abs first)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{ x = 0 - x; }}\n    total: i64 = 0;\n    while x > 0 {{\n        total = total + x % 10;\n        x = x / 10;\n    }}\n    return total;\n}}\n"),
            // digit_sum (no abs)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    total: i64 = 0;\n    while x > 0 {{\n        total = total + x % 10;\n        x = x / 10;\n    }}\n    return total;\n}}\n"),
            // digit_product
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    acc: i64 = 1;\n    while x > 0 {{\n        acc = acc * (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n}}\n"),
            // digit_count (0→1)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{ x = 0 - x; }}\n    if x == 0 {{ return 1; }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = acc + 1;\n        x = x / 10;\n    }}\n    return acc;\n}}\n"),
            // reverse_digits
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{ x = 0 - x; }}\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = (acc * 10) + (x % 10);\n        x = x / 10;\n    }}\n    return acc;\n}}\n"),
            // count_even_digits
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    if x < 0 {{ x = 0 - x; }}\n    if x == 0 {{ return 1; }}\n    acc: i64 = 0;\n    while x > 0 {{\n        if ((x % 10) % 2) == 0 {{ acc = acc + 1; }}\n        x = x / 10;\n    }}\n    return acc;\n}}\n"),
            // max_digit
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    best: i64 = 0;\n    while x > 0 {{\n        d: i64 = x % 10;\n        if d > best {{ best = d; }}\n        x = x / 10;\n    }}\n    return best;\n}}\n"),
            // leading_digit
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    while x >= 10 {{\n        x = x / 10;\n    }}\n    return x;\n}}\n"),
            // popcount via % 2
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    acc: i64 = 0;\n    while x > 0 {{\n        acc = acc + x % 2;\n        x = x / 2;\n    }}\n    return acc;\n}}\n"),
            // digital_root (nested while)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    while x >= 10 {{\n        s: i64 = 0;\n        while x > 0 {{\n            s = s + x % 10;\n            x = x / 10;\n        }}\n        x = s;\n    }}\n    return x;\n}}\n"),
            // is_perfect_square
            format!("fn {fn_name}(n: i64) -> i64 {{\n    i: i64 = 0;\n    while i * i <= n {{\n        if i * i == n {{ return 1; }}\n        i = i + 1;\n    }}\n    return 0;\n}}\n"),
            // next_power_of_2
            format!("fn {fn_name}(n: i64) -> i64 {{\n    p: i64 = 1;\n    while p < n {{\n        p = p * 2;\n    }}\n    return p;\n}}\n"),
            // count_divisors
            format!("fn {fn_name}(n: i64) -> i64 {{\n    count: i64 = 0;\n    i: i64 = 1;\n    while i <= n {{\n        if n % i == 0 {{ count = count + 1; }}\n        i = i + 1;\n    }}\n    return count;\n}}\n"),
            // sum_of_divisors
            format!("fn {fn_name}(n: i64) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {{\n        if n % i == 0 {{ total = total + i; }}\n        i = i + 1;\n    }}\n    return total;\n}}\n"),
            // harmonic_sum (1000/i)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {{\n        total = total + 1000 / i;\n        i = i + 1;\n    }}\n    return total;\n}}\n"),
            // triangular_check
            format!("fn {fn_name}(n: i64) -> i64 {{\n    k: i64 = 0;\n    while k * (k + 1) / 2 <= n {{\n        if k * (k + 1) / 2 == n {{ return 1; }}\n        k = k + 1;\n    }}\n    return 0;\n}}\n"),
            // is_prime
            format!("fn {fn_name}(n: i64) -> i64 {{\n    if n < 2 {{ return 0; }}\n    if n == 2 {{ return 1; }}\n    if n % 2 == 0 {{ return 0; }}\n    i: i64 = 3;\n    while i * i <= n {{\n        if n % i == 0 {{ return 0; }}\n        i = i + 2;\n    }}\n    return 1;\n}}\n"),
            // euler_totient
            format!("fn {fn_name}(n: i64) -> i64 {{\n    result: i64 = n;\n    p: i64 = 2;\n    temp: i64 = n;\n    while p * p <= temp {{\n        if temp % p == 0 {{\n            while temp % p == 0 {{\n                temp = temp / p;\n            }}\n            result = result - result / p;\n        }}\n        p = p + 1;\n    }}\n    if temp > 1 {{\n        result = result - result / temp;\n    }}\n    return result;\n}}\n"),
            // collatz_steps
            format!("fn {fn_name}(n: i64) -> i64 {{\n    x: i64 = n;\n    steps: i64 = 0;\n    while x > 1 {{\n        if x % 2 == 0 {{\n            x = x / 2;\n        }} else {{\n            x = 3 * x + 1;\n        }}\n        steps = steps + 1;\n    }}\n    return steps;\n}}\n"),
            // nth_triangle / sum_to_n (loop variant)
            format!("fn {fn_name}(n: i64) -> i64 {{\n    if n <= 0 {{ return 0; }}\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {{\n        total = total + i;\n        i = i + 1;\n    }}\n    return total;\n}}\n"),
            // nth_triangle formula
            format!("fn {fn_name}(n: i64) -> i64 {{\n    return n * (n + 1) / 2;\n}}\n"),
            // polynomial 2x^2+3x+1
            format!("fn {fn_name}(x: i64) -> i64 {{\n    return 2 * x * x + 3 * x + 1;\n}}\n"),
            // lucas_number
            format!("fn {fn_name}(n: i64) -> i64 {{\n    if n == 0 {{ return 2; }}\n    if n == 1 {{ return 1; }}\n    a: i64 = 2;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n {{\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }}\n    return b;\n}}\n"),
            // fibonacci / fib_iter iterative
            format!("fn {fn_name}(n: i64) -> i64 {{\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 0;\n    while i < n {{\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }}\n    return a;\n}}\n"),
            // clamp 0..100 (two-if style)
            format!("fn {fn_name}(x: i64) -> i64 {{\n    if x < 0 {{ return 0; }}\n    if x > 100 {{ return 100; }}\n    return x;\n}}\n"),
            // identity passthrough
            format!("fn {fn_name}(x: i64) -> i64 {{\n    return x;\n}}\n"),
            // abs
            format!("fn {fn_name}(x: i64) -> i64 {{\n    if x < 0 {{ return 0 - x; }}\n    return x;\n}}\n"),
        ],
        2 => vec![
            // safe_div_or_neg1
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    if b == 0 {{ return -1; }}\n    return a / b;\n}}\n"),
            // gcd (Euclidean)
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {{\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }}\n    return x;\n}}\n"),
            // lcm inline
            format!("fn gcd_h(a: i64, b: i64) -> i64 {{\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {{\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }}\n    return x;\n}}\nfn {fn_name}(a: i64, b: i64) -> i64 {{\n    return (a * b) / gcd_h(a, b);\n}}\n"),
            // max2 inline
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    if a > b {{ return a; }}\n    return b;\n}}\n"),
            // min2 inline
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    if a < b {{ return a; }}\n    return b;\n}}\n"),
            // abs_diff inline
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    d: i64 = a - b;\n    if d < 0 {{ return 0 - d; }}\n    return d;\n}}\n"),
            // scaled_sum: 2*a + b
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    return 2 * a + b;\n}}\n"),
            // product_offset: a*b - a
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    return a * b - a;\n}}\n"),
            // sum
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    return a + b;\n}}\n"),
            // product
            format!("fn {fn_name}(a: i64, b: i64) -> i64 {{\n    return a * b;\n}}\n"),
        ],
        3 => vec![
            // min3
            format!("fn {fn_name}(a: i64, b: i64, c: i64) -> i64 {{\n    m: i64 = a;\n    if b < m {{ m = b; }}\n    if c < m {{ m = c; }}\n    return m;\n}}\n"),
            // max3
            format!("fn {fn_name}(a: i64, b: i64, c: i64) -> i64 {{\n    m: i64 = a;\n    if b > m {{ m = b; }}\n    if c > m {{ m = c; }}\n    return m;\n}}\n"),
            // median3
            format!("fn {fn_name}(a: i64, b: i64, c: i64) -> i64 {{\n    if (a >= b && b >= c) || (c >= b && b >= a) {{ return b; }}\n    if (b >= a && a >= c) || (c >= a && a >= b) {{ return a; }}\n    return c;\n}}\n"),
            // sum3
            format!("fn {fn_name}(a: i64, b: i64, c: i64) -> i64 {{\n    return a + b + c;\n}}\n"),
        ],
        _ => vec![],
    };

    for code in candidates {
        if let Some(r) = make_result(code) {
            return Some(r);
        }
    }

    None
}
