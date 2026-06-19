/// Stage 4 time-parameterized synthesis teachers.
/// Recognize mathematical sequences and closed-form patterns where output is a function of
/// a single time/index parameter t: i64 → i64 (no loops except for exponentiation/factorial).
///
/// Teachers:
/// 1. search_polynomial_time - Recognizes polynomial patterns (linear, quadratic, cubic)
/// 2. search_exponential_time - Recognizes exponential growth (2^t, 3^t, Fibonacci)
/// 3. search_factorial_time - Recognizes factorial pattern (n!)
/// 4. search_triangular_time - Recognizes triangular/cumsum pattern (t*(t+1)/2)
use super::search_codegen::*;
use super::*;
use crate::time_codegen::*;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Simple integer power function.
fn pow_i64(base: i64, exp: i64) -> Option<i64> {
    if exp < 0 {
        return None;
    }
    if exp == 0 {
        return Some(1);
    }
    let mut result = 1i64;
    for _ in 0..exp {
        result = result.checked_mul(base)?;
    }
    Some(result)
}

/// Compute the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, fib(3)=2, ...).
fn fibonacci(n: i64) -> i64 {
    if n < 0 {
        return 0;
    }
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut a = 0i64;
    let mut b = 1i64;
    for _ in 2..=n {
        let tmp = a + b;
        a = b;
        b = tmp;
    }
    b
}

// ============================================================================
// POLYNOMIAL TIME (Degree 1, 2, 3 via finite differences)
// ============================================================================

/// Finite differences to detect polynomial degree.
fn compute_finite_differences(pts: &[(i64, i64)]) -> Vec<Vec<i64>> {
    if pts.is_empty() {
        return vec![];
    }
    let mut levels = vec![pts.iter().map(|(_, y)| *y).collect::<Vec<_>>()];
    for level in 1..pts.len() {
        let prev = &levels[level - 1];
        if prev.len() < 2 {
            break;
        }
        let diffs = (0..prev.len() - 1)
            .map(|i| prev[i + 1] - prev[i])
            .collect::<Vec<_>>();
        levels.push(diffs);
    }
    levels
}

/// Check if a level of differences is constant.
fn is_constant_differences(diffs: &[i64]) -> bool {
    diffs.len() >= 2 && diffs.iter().all(|&d| d == diffs[0])
}

/// Fit a polynomial via Gaussian elimination for degree 1-3.
fn fit_polynomial(pts: &[(i64, i64)], degree: usize) -> Option<Vec<i64>> {
    if pts.len() < degree + 1 {
        return None;
    }
    let n = degree + 1;
    let mut normal = vec![vec![0i64; n]; n];
    let mut rhs = vec![0i64; n];

    for &(t, y) in pts {
        let mut powers = vec![1i64; n];
        for j in 1..n {
            powers[j] = pow_i64(t, (degree - j + 1) as i64).unwrap_or(0);
        }

        for i in 0..n {
            for j in 0..n {
                if let Some(prod) = powers[i].checked_mul(powers[j]) {
                    if let Some(new_val) = normal[i][j].checked_add(prod) {
                        normal[i][j] = new_val;
                    }
                }
            }
            if let Some(prod) = powers[i].checked_mul(y) {
                if let Some(new_val) = rhs[i].checked_add(prod) {
                    rhs[i] = new_val;
                }
            }
        }
    }

    gaussian_elimination_i64(&mut normal, &mut rhs).ok()
}

/// Simple Gaussian elimination over integers (with scaling).
fn gaussian_elimination_i64(a: &mut Vec<Vec<i64>>, b: &mut Vec<i64>) -> Result<Vec<i64>, String> {
    let n = a.len();
    for col in 0..n {
        let mut pivot_row = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[pivot_row][col].abs() {
                pivot_row = row;
            }
        }
        if a[pivot_row][col] == 0 {
            return Err("Singular matrix".to_string());
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        for row in col + 1..n {
            if a[row][col] == 0 {
                continue;
            }
            let factor = a[row][col];
            let divisor = a[col][col];
            for j in col..n {
                a[row][j] = a[row][j] * divisor - a[col][j] * factor;
            }
            b[row] = b[row] * divisor - b[col] * factor;
        }
    }
    let mut x = vec![0i64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in i + 1..n {
            sum -= a[i][j] * x[j];
        }
        if a[i][i] == 0 {
            return Err("Singular matrix".to_string());
        }
        if sum % a[i][i] != 0 {
            return Err("Non-integer solution".to_string());
        }
        x[i] = sum / a[i][i];
    }
    Ok(x)
}

/// Verify polynomial on a given (t, output) point.
fn verify_polynomial(t: i64, output: i64, coeffs: &[i64], degree: usize) -> bool {
    let mut result = 0i64;
    for (i, &coeff) in coeffs.iter().enumerate() {
        let power = degree - i;
        if let Some(t_pow) = pow_i64(t, power as i64) {
            if let Some(term) = coeff.checked_mul(t_pow) {
                if let Some(next) = result.checked_add(term) {
                    result = next;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    result == output
}

pub(super) fn search_polynomial_time(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }

    let mut pts: Vec<(i64, i64)> = problem
        .examples
        .iter()
        .filter_map(|ex| {
            if ex.inputs.len() == 1 {
                int_value(&ex.inputs[0]).map(|t| (t, ex.expected_int()))
            } else {
                None
            }
        })
        .collect();

    if pts.len() < 5 {
        return None;
    }

    pts.sort_unstable();
    pts.dedup();

    for degree in 1..=3 {
        if let Some(coeffs) = fit_polynomial(&pts, degree) {
            if !pts
                .iter()
                .all(|&(t, output)| verify_polynomial(t, output, &coeffs, degree))
            {
                continue;
            }

            if !coeffs.iter().all(|&c| c.abs() <= 10000) {
                continue;
            }

            let code = match degree {
                1 if coeffs.len() == 2 => code_polynomial_linear(fn_name, coeffs[1], coeffs[0]),
                2 if coeffs.len() == 3 => {
                    code_polynomial_quadratic(fn_name, coeffs[2], coeffs[1], coeffs[0])
                }
                3 if coeffs.len() == 4 => {
                    code_polynomial_cubic(fn_name, coeffs[3], coeffs[2], coeffs[1], coeffs[0])
                }
                _ => code_polynomial_generic(fn_name, &coeffs, degree),
            };

            if let Some(result) = verified_result(problem, code, "search_polynomial_time") {
                return Some(result);
            }
        }
    }
    None
}

// ============================================================================
// EXPONENTIAL TIME (2^t, 3^t, Fibonacci)
// ============================================================================

fn is_power_of_base(pts: &[(i64, i64)], base: i64) -> bool {
    pts.iter().all(|&(t, output)| {
        if t < 0 || t > 62 {
            return false;
        }
        pow_i64(base, t)
            .map(|expected| expected == output)
            .unwrap_or(false)
    })
}

fn is_fibonacci_pattern(pts: &[(i64, i64)]) -> bool {
    pts.iter().all(|&(t, output)| {
        if t < 0 || t > 92 {
            return false;
        }
        fibonacci(t) == output
    })
}

fn check_growth_ratio(pts: &[(i64, i64)], base: i64) -> bool {
    if pts.len() < 2 {
        return false;
    }
    for i in 0..pts.len() - 1 {
        let (_, y1) = pts[i];
        let (_, y2) = pts[i + 1];
        if y1 == 0 || y2 % y1 != 0 || y2 / y1 != base {
            return false;
        }
    }
    true
}

pub(super) fn search_exponential_time(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }

    let mut pts: Vec<(i64, i64)> = problem
        .examples
        .iter()
        .filter_map(|ex| {
            if ex.inputs.len() == 1 {
                int_value(&ex.inputs[0]).map(|t| (t, ex.expected_int()))
            } else {
                None
            }
        })
        .collect();

    if pts.len() < 4 {
        return None;
    }

    pts.sort_unstable();

    if is_fibonacci_pattern(&pts) {
        if let Some(result) = verified_result(
            problem,
            code_fibonacci_iter(fn_name),
            "search_exponential_time_fibonacci",
        ) {
            return Some(result);
        }
    }

    if check_growth_ratio(&pts, 2) && is_power_of_base(&pts, 2) {
        if let Some(result) = verified_result(
            problem,
            code_power_of_2(fn_name),
            "search_exponential_time_power2",
        ) {
            return Some(result);
        }
    }

    if check_growth_ratio(&pts, 3) && is_power_of_base(&pts, 3) {
        if let Some(result) = verified_result(
            problem,
            code_power_of_3(fn_name),
            "search_exponential_time_power3",
        ) {
            return Some(result);
        }
    }

    None
}

// ============================================================================
// FACTORIAL TIME (n!)
// ============================================================================

const FACTORIAL_TABLE: &[i64] = &[
    1,
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800,
    87178291200,
    1307674368000,
    20922789888000,
    355687428096000,
    6402373705728000,
    121645100408832000,
    2432902008176640000,
];

fn is_factorial_pattern(pts: &[(i64, i64)]) -> bool {
    pts.iter().all(|&(t, output)| {
        if t < 0 || t as usize >= FACTORIAL_TABLE.len() {
            return false;
        }
        FACTORIAL_TABLE[t as usize] == output
    })
}

pub(super) fn search_factorial_time(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }

    let mut pts: Vec<(i64, i64)> = problem
        .examples
        .iter()
        .filter_map(|ex| {
            if ex.inputs.len() == 1 {
                int_value(&ex.inputs[0]).map(|t| (t, ex.expected_int()))
            } else {
                None
            }
        })
        .collect();

    if pts.len() < 4 {
        return None;
    }

    pts.sort_unstable();

    if is_factorial_pattern(&pts) {
        if let Some(result) =
            verified_result(problem, code_factorial(fn_name), "search_factorial_time")
        {
            return Some(result);
        }
    }

    None
}

// ============================================================================
// TRIANGULAR TIME (Cumulative sum: 1 + 2 + 3 + ... + t = t*(t+1)/2)
// ============================================================================

fn is_triangular_pattern(pts: &[(i64, i64)]) -> bool {
    pts.iter().all(|&(t, output)| {
        if t < 0 || t > 1_000_000 {
            return false;
        }
        let expected = (t * (t + 1)) / 2;
        expected == output
    })
}

fn triangular_from_differences(pts: &[(i64, i64)]) -> bool {
    if pts.len() < 3 {
        return false;
    }
    let mut expected_diff = 1i64;
    for i in 1..pts.len() {
        let actual_diff = pts[i].1 - pts[i - 1].1;
        if actual_diff != expected_diff {
            return false;
        }
        expected_diff += 1;
    }
    true
}

pub(super) fn search_triangular_time(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::I64] {
        return None;
    }

    let mut pts: Vec<(i64, i64)> = problem
        .examples
        .iter()
        .filter_map(|ex| {
            if ex.inputs.len() == 1 {
                int_value(&ex.inputs[0]).map(|t| (t, ex.expected_int()))
            } else {
                None
            }
        })
        .collect();

    if pts.len() < 4 {
        return None;
    }

    pts.sort_unstable();

    if is_triangular_pattern(&pts) || triangular_from_differences(&pts) {
        if let Some(result) = verified_result(
            problem,
            code_triangular_closed_form(fn_name),
            "search_triangular_time",
        ) {
            return Some(result);
        }
        if let Some(result) = verified_result(
            problem,
            code_triangular_loop(fn_name),
            "search_triangular_time_loop",
        ) {
            return Some(result);
        }
    }

    None
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Polynomial Tests ==========

    #[test]
    fn test_polynomial_time_linear() {
        let pts = vec![(0, 1), (1, 3), (2, 5), (3, 7), (4, 9)];
        let coeffs = fit_polynomial(&pts, 1).unwrap();
        assert_eq!(coeffs.len(), 2);
    }

    #[test]
    fn test_polynomial_time_quadratic() {
        let pts = vec![(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)];
        let diffs = compute_finite_differences(&pts);
        assert!(is_constant_differences(&diffs[2]));
    }

    #[test]
    fn test_polynomial_time_cubic() {
        let pts = vec![(0, 0), (1, 1), (2, 8), (3, 27), (4, 64)];
        let coeffs = fit_polynomial(&pts, 3).unwrap();
        assert_eq!(coeffs.len(), 4);
    }

    #[test]
    fn test_pow_i64() {
        assert_eq!(pow_i64(2, 0), Some(1));
        assert_eq!(pow_i64(2, 1), Some(2));
        assert_eq!(pow_i64(2, 5), Some(32));
        assert_eq!(pow_i64(3, 3), Some(27));
        assert_eq!(pow_i64(-1, 2), Some(1));
        assert_eq!(pow_i64(2, -1), None);
    }

    // ========== Exponential/Fibonacci Tests ==========

    #[test]
    fn test_power_of_2() {
        let pts = vec![(0, 1), (1, 2), (2, 4), (3, 8), (4, 16)];
        assert!(is_power_of_base(&pts, 2));
    }

    #[test]
    fn test_power_of_3() {
        let pts = vec![(0, 1), (1, 3), (2, 9), (3, 27), (4, 81)];
        assert!(is_power_of_base(&pts, 3));
    }

    #[test]
    fn test_growth_ratio() {
        let pts = vec![(0, 1), (1, 2), (2, 4), (3, 8)];
        assert!(check_growth_ratio(&pts, 2));
        assert!(!check_growth_ratio(&pts, 3));
    }

    #[test]
    fn test_fibonacci_computation() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 1);
        assert_eq!(fibonacci(3), 2);
        assert_eq!(fibonacci(4), 3);
        assert_eq!(fibonacci(5), 5);
        assert_eq!(fibonacci(6), 8);
        assert_eq!(fibonacci(10), 55);
    }

    #[test]
    fn test_fibonacci_pattern() {
        let pts = vec![(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5)];
        assert!(is_fibonacci_pattern(&pts));
    }

    // ========== Factorial Tests ==========

    #[test]
    fn test_factorial_pattern() {
        let pts = vec![(0, 1), (1, 1), (2, 2), (3, 6), (4, 24)];
        assert!(is_factorial_pattern(&pts));
    }

    #[test]
    fn test_factorial_pattern_extended() {
        let pts = vec![(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120)];
        assert!(is_factorial_pattern(&pts));
    }

    // ========== Triangular Tests ==========

    #[test]
    fn test_triangular_pattern() {
        let pts = vec![(0, 0), (1, 1), (2, 3), (3, 6), (4, 10)];
        assert!(is_triangular_pattern(&pts));
    }

    #[test]
    fn test_triangular_from_differences() {
        let pts = vec![(0, 0), (1, 1), (2, 3), (3, 6), (4, 10), (5, 15)];
        assert!(triangular_from_differences(&pts));
    }

    #[test]
    fn test_triangular_larger_values() {
        let pts = vec![(0, 0), (10, 55), (20, 210)];
        assert!(is_triangular_pattern(&pts));
    }

    // ========== Gaussian Elimination Tests ==========

    #[test]
    fn test_gaussian_elimination_2x2() {
        // Solve: 2x + 1y = 5, 1x + 3y = 6
        // Solution: x=2, y=1
        let mut a = vec![vec![2, 1], vec![1, 3]];
        let mut b = vec![5, 6];
        let result = gaussian_elimination_i64(&mut a, &mut b);
        assert!(result.is_ok());
        let x = result.unwrap();
        assert_eq!(x.len(), 2);
    }

    #[test]
    fn test_finite_differences_degree_1() {
        let pts = vec![(0, 1), (1, 3), (2, 5), (3, 7)];
        let diffs = compute_finite_differences(&pts);
        assert!(diffs.len() >= 2);
        assert!(is_constant_differences(&diffs[1])); // First differences constant
    }

    #[test]
    fn test_finite_differences_degree_2() {
        let pts = vec![(0, 0), (1, 1), (2, 4), (3, 9)];
        let diffs = compute_finite_differences(&pts);
        assert!(diffs.len() >= 3);
        assert!(is_constant_differences(&diffs[2])); // Second differences constant
    }

    // ========== Code Generator Tests ==========

    #[test]
    fn test_code_polynomial_linear_generation() {
        let code = code_polynomial_linear("f", 1, 2);
        assert!(code.contains("return 1 + (2 * t)"));
    }

    #[test]
    fn test_code_polynomial_quadratic_generation() {
        let code = code_polynomial_quadratic("g", 0, 0, 1);
        assert!(code.contains("t * t"));
    }

    #[test]
    fn test_code_fibonacci_generation() {
        let code = code_fibonacci_iter("fib");
        assert!(code.contains("a + b"));
    }

    #[test]
    fn test_code_factorial_generation() {
        let code = code_factorial("fact");
        assert!(code.contains("r * i"));
    }

    #[test]
    fn test_code_triangular_generation() {
        let code = code_triangular_closed_form("tri");
        assert!(code.contains("(t * (t + 1)) / 2"));
    }

    #[test]
    fn test_code_power_of_2_generation() {
        let code = code_power_of_2("pow2");
        assert!(code.contains("acc * 2"));
    }
}
