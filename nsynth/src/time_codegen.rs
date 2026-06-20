/// Stage 4 time-parameterized code generation.
/// Produces optimized Mog/PseudoCode for mathematical sequences and closed-form patterns
/// where output is a function of a single time/index parameter t: i64 → i64.

fn templ(template: &str, fn_name: &str) -> String {
    template.replace("__FN__", fn_name)
}

// ============================================================================
// POLYNOMIAL CODEGEN
// ============================================================================

/// Generate code for polynomial of degree 1 (linear): f(t) = c0 + c1*t
pub fn code_polynomial_linear(fn_name: &str, c0: i64, c1: i64) -> String {
    templ(
        &format!(
            r#"fn __FN__(t: i64) -> i64 {{
    return {c0} + ({c1} * t);
}}"#
        ),
        fn_name,
    )
}

/// Generate code for polynomial of degree 2 (quadratic): f(t) = c0 + c1*t + c2*t²
pub fn code_polynomial_quadratic(fn_name: &str, c0: i64, c1: i64, c2: i64) -> String {
    templ(
        &format!(
            r#"fn __FN__(t: i64) -> i64 {{
    return {c0} + ({c1} * t) + ({c2} * t * t);
}}"#
        ),
        fn_name,
    )
}

/// Generate code for polynomial of degree 3 (cubic): f(t) = c0 + c1*t + c2*t² + c3*t³
pub fn code_polynomial_cubic(fn_name: &str, c0: i64, c1: i64, c2: i64, c3: i64) -> String {
    templ(
        &format!(
            r#"fn __FN__(t: i64) -> i64 {{
    t2: i64 = t * t;
    t3: i64 = t2 * t;
    return {c0} + ({c1} * t) + ({c2} * t2) + ({c3} * t3);
}}"#
        ),
        fn_name,
    )
}

/// Generate code for arbitrary polynomial given coefficients.
/// coeffs[0] is highest degree, coeffs[n] is constant term.
/// degree = len(coeffs) - 1
pub fn code_polynomial_generic(fn_name: &str, coeffs: &[i64], degree: usize) -> String {
    if coeffs.is_empty() {
        return templ("fn __FN__(t: i64) -> i64 { return 0; }", fn_name);
    }

    let mut body = String::new();
    for i in 0..degree {
        body.push_str(&format!(
            "    t{}: i64 = t * {};\n",
            i + 1,
            if i == 0 {
                "t".to_string()
            } else {
                format!("t{}", i)
            }
        ));
    }

    let mut expr = String::new();
    for (i, &coeff) in coeffs.iter().enumerate() {
        let power = degree - i;
        let term = match power {
            0 => coeff.to_string(),
            1 => format!("({coeff} * t)"),
            n => format!("({coeff} * t{n})"),
        };

        if i == 0 {
            expr = term;
        } else if coeff >= 0 {
            expr = format!("{expr} + {term}");
        } else {
            expr = format!("{expr} - {}", term.trim_start_matches('-'));
        }
    }

    body.push_str(&format!("    return {expr};",));

    format!("fn __FN__(t: i64) -> i64 {{\n{body}\n}}",).replace("__FN__", fn_name)
}

// ============================================================================
// EXPONENTIAL CODEGEN
// ============================================================================

/// Generate code for 2^t using iterative multiplication.
pub fn code_power_of_2(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(t: i64) -> i64 {
    if t < 0 {
        return 1;
    }
    if t >= 63 {
        return -1;
    }
    acc: i64 = 1;
    i: i64 = 0;
    while i < t {
        acc = acc * 2;
        i = i + 1;
    }
    return acc;
}"#,
        fn_name,
    )
}

/// Generate code for 3^t using iterative multiplication.
pub fn code_power_of_3(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(t: i64) -> i64 {
    if t < 0 {
        return 1;
    }
    if t >= 40 {
        return -1;
    }
    acc: i64 = 1;
    i: i64 = 0;
    while i < t {
        acc = acc * 3;
        i = i + 1;
    }
    return acc;
}"#,
        fn_name,
    )
}

/// Generate code for base^t using iterative multiplication (generic base).
pub fn code_power_generic(fn_name: &str, base: i64) -> String {
    templ(
        &format!(
            r#"fn __FN__(t: i64) -> i64 {{
    if t < 0 {{
        return 1;
    }}
    acc: i64 = 1;
    i: i64 = 0;
    while i < t {{
        acc = acc * {base};
        i = i + 1;
    }}
    return acc;
}}"#
        ),
        fn_name,
    )
}

/// Generate code for Fibonacci sequence: fib(n) iteratively computed.
pub fn code_fibonacci_iter(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    a: i64 = 0;
    b: i64 = 1;
    i: i64 = 2;
    while i <= n {
        tmp: i64 = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}"#,
        fn_name,
    )
}

// ============================================================================
// FACTORIAL CODEGEN
// ============================================================================

/// Generate code for factorial: n! computed via loop.
pub fn code_factorial(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(n: i64) -> i64 {
    if n < 0 {
        return 1;
    }
    if n > 20 {
        return -1;
    }
    r: i64 = 1;
    i: i64 = 1;
    while i <= n {
        r = r * i;
        i = i + 1;
    }
    return r;
}"#,
        fn_name,
    )
}

// ============================================================================
// TRIANGULAR/CUMULATIVE SUM CODEGEN
// ============================================================================

/// Generate code for triangular number: t*(t+1)/2
pub fn code_triangular_closed_form(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(t: i64) -> i64 {
    if t < 0 {
        return 0;
    }
    return (t * (t + 1)) / 2;
}"#,
        fn_name,
    )
}

/// Generate code for cumulative sum via loop (more robust for large t).
pub fn code_triangular_loop(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(t: i64) -> i64 {
    if t < 0 {
        return 0;
    }
    r: i64 = 0;
    i: i64 = 1;
    while i <= t {
        r = r + i;
        i = i + 1;
    }
    return r;
}"#,
        fn_name,
    )
}

// ============================================================================
// PIECEWISE POLYNOMIAL CODEGEN
// ============================================================================

/// Generate code for piecewise polynomial (e.g., max(f₁(t), f₂(t))).
/// threshold: value of t where switch occurs
/// coeff1: coefficients for lower piece
/// coeff2: coefficients for upper piece
pub fn code_piecewise_polynomial(
    fn_name: &str,
    threshold: i64,
    coeff1: &[i64],
    coeff2: &[i64],
) -> String {
    let piece1 = code_polynomial_generic("f1", coeff1, coeff1.len() - 1);
    let piece2 = code_polynomial_generic("f2", coeff2, coeff2.len() - 1);

    format!(
        r#"fn {fn_name}(t: i64) -> i64 {{
    if t < {threshold} {{
        y1: i64 = {}(t);
        return y1;
    }} else {{
        y2: i64 = {}(t);
        return y2;
    }}
}}"#,
        piece1.lines().nth(0).unwrap_or("f1"),
        piece2.lines().nth(0).unwrap_or("f2")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_polynomial_linear() {
        let code = code_polynomial_linear("f", 1, 2);
        assert!(code.contains("return 1 + (2 * t)"));
    }

    #[test]
    fn test_code_polynomial_quadratic() {
        let code = code_polynomial_quadratic("g", 0, 0, 1);
        assert!(code.contains("t * t"));
    }

    #[test]
    fn test_code_polynomial_cubic() {
        let code = code_polynomial_cubic("h", 0, 1, 0, 1);
        assert!(code.contains("t3"));
    }

    #[test]
    fn test_code_power_of_2() {
        let code = code_power_of_2("pow2");
        assert!(code.contains("acc * 2"));
    }

    #[test]
    fn test_code_fibonacci() {
        let code = code_fibonacci_iter("fib");
        assert!(code.contains("a + b"));
    }

    #[test]
    fn test_code_factorial() {
        let code = code_factorial("fact");
        assert!(code.contains("r * i"));
    }

    #[test]
    fn test_code_triangular() {
        let code = code_triangular_closed_form("tri");
        assert!(code.contains("(t * (t + 1)) / 2"));
    }
}
