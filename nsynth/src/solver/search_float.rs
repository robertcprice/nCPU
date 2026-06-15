//! Float (continuous) synthesis — the first lane that leaves the exact-integer
//! regime.
//!
//! The integer solvers REFUSE any rule with a non-integer coefficient. Real
//! formulas are full of them: `weight_kg = 0.4536 * pounds`, `celsius =
//! (fahrenheit - 32.0) / 1.8`, a calibration line `y = 2.5*x + 1.3`. This solver
//! recovers a multi-argument FLOAT affine `f(x) = c0 + Σ c_j·x_j` by ordinary
//! least squares over f64, then verifies the recovered model to a TOLERANCE
//! (continuous data is approximate by nature — exactness is the wrong test) and
//! emits a float Mog program (`-> f64`, float literals), now that the language
//! has an f64 value type, float arithmetic, and float division.
//!
//! Honest, recover-or-refuse like the rest of the crate: a model is returned
//! only if every example is reproduced within tolerance; over-determination
//! (more examples than coefficients) guards against fitting noise; and the
//! recovered coefficients are rendered at finite precision and RE-checked, so a
//! displayed formula always reproduces the data it claims.

use crate::benchmark::value_as_f64;
use crate::runtime::execute_function_for_problem;

use super::signature::{scalar_param_names, scalar_params_decl};
use super::*;

/// Extract `(rows, targets, arity)` of f64 from a problem whose signature returns
/// `f64` and whose inputs are all scalar numbers (int or float), 1–3 of them.
fn extract_float(problem: &Problem) -> Option<(Vec<Vec<f64>>, Vec<f64>, usize)> {
    if !problem.signature.contains("-> f64") {
        return None;
    }
    let arity = problem.examples.first()?.inputs.len();
    if !(1..=3).contains(&arity) {
        return None;
    }
    let mut rows = Vec::with_capacity(problem.examples.len());
    let mut targets = Vec::with_capacity(problem.examples.len());
    for ex in &problem.examples {
        if ex.inputs.len() != arity {
            return None;
        }
        let row: Option<Vec<f64>> = ex.inputs.iter().map(value_as_f64).collect();
        let row = row?;
        let y = ex.expected_f64()?;
        rows.push(row);
        targets.push(y);
    }
    Some((rows, targets, arity))
}

/// Solve the `m × m` linear system `A·w = b` by Gaussian elimination with partial
/// pivoting, in f64. Returns None if the matrix is singular (rank-deficient).
fn solve_f64(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let m = b.len();
    for col in 0..m {
        // partial pivot
        let mut piv = col;
        for r in (col + 1)..m {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in 0..m {
            if r != col {
                let factor = a[r][col] / a[col][col];
                for c in col..m {
                    a[r][c] -= factor * a[col][c];
                }
                b[r] -= factor * b[col];
            }
        }
    }
    Some((0..m).map(|i| b[i] / a[i][i]).collect())
}

/// Ordinary least squares for `c0 + Σ c_j·x_j`: build the normal equations
/// `(ΦᵀΦ) w = Φᵀy` with `φ = [1, x_1, …, x_n]` and solve them in f64.
fn least_squares_affine(rows: &[Vec<f64>], targets: &[f64], arity: usize) -> Option<Vec<f64>> {
    let m = arity + 1;
    let phi = |row: &[f64]| -> Vec<f64> {
        let mut v = Vec::with_capacity(m);
        v.push(1.0);
        v.extend_from_slice(row);
        v
    };
    let mut ata = vec![vec![0.0f64; m]; m];
    let mut atb = vec![0.0f64; m];
    for (row, &y) in rows.iter().zip(targets.iter()) {
        let p = phi(row);
        for i in 0..m {
            atb[i] += p[i] * y;
            for j in 0..m {
                ata[i][j] += p[i] * p[j];
            }
        }
    }
    solve_f64(ata, atb)
}

/// Format a float coefficient compactly: round to 6 decimals, trim trailing
/// zeros, and keep at least one fractional digit so it lexes as a float literal
/// (`2.0`, not `2`). Negative values are emitted bare; the renderer wraps terms.
fn fmt_coeff(c: f64) -> String {
    let r = (c * 1e6).round() / 1e6;
    let mut s = format!("{r:.6}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.push('0');
    }
    s
}

/// Predicted value of the affine at a row.
fn predict(coeffs: &[f64], row: &[f64]) -> f64 {
    let mut acc = coeffs[0];
    for (j, &x) in row.iter().enumerate() {
        acc += coeffs[j + 1] * x;
    }
    acc
}

/// A per-problem tolerance: a small relative slack scaled by the output
/// magnitude, with an absolute floor. Continuous fits are never bit-exact, so
/// "correct" means within this band on every example.
fn tolerance(targets: &[f64]) -> f64 {
    let max_abs = targets.iter().fold(0.0f64, |m, &y| m.max(y.abs()));
    (max_abs * 1e-6).max(1e-6)
}

/// Recover `f(x) = c0 + Σ c_j·x_j` over f64 by least squares, verify to
/// tolerance, and emit a float Mog program.
pub(super) fn search_float_affine(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let (rows, targets, arity) = extract_float(problem)?;
    let n = rows.len();
    if n < arity + 2 {
        return None; // over-determination: more examples than coefficients
    }
    let coeffs = least_squares_affine(&rows, &targets, arity)?;
    // Round the coefficients to the finite precision we will print, then verify
    // the ROUNDED model — so the formula we emit is exactly the one we checked.
    let rounded: Vec<f64> = coeffs.iter().map(|&c| (c * 1e6).round() / 1e6).collect();
    let eps = tolerance(&targets);
    for (row, &y) in rows.iter().zip(targets.iter()) {
        if (predict(&rounded, row) - y).abs() > eps {
            return None;
        }
    }
    // A pure constant or a degenerate all-zero slope is not interesting here.
    if rounded[1..].iter().all(|&c| c.abs() < 1e-9) {
        return None;
    }

    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names).replace(": i64", ": f64"); // float signature
                                                                             // Build `c0 + c1*x0 + c2*x1 + …`, dropping ~zero terms.
    let mut terms: Vec<String> = Vec::new();
    for (j, name) in param_names.iter().enumerate() {
        let c = rounded[j + 1];
        if c.abs() < 1e-9 {
            continue;
        }
        terms.push(format!("{} * {}", fmt_coeff(c), name));
    }
    if rounded[0].abs() >= 1e-9 || terms.is_empty() {
        terms.push(fmt_coeff(rounded[0]));
    }
    let body = terms.join(" + ");
    let code = format!("fn {fn_name}({params}) -> f64 {{\n    return {body};\n}}\n");

    // Final guard: run the emitted program through the (now float-capable) runtime
    // and confirm it reproduces every example within tolerance — the same
    // recover-or-refuse contract as verified_result, lifted to a float tolerance.
    for ex in &problem.examples {
        let out = execute_function_for_problem(&code, fn_name, &ex.inputs, problem).ok()?;
        let got = match out {
            crate::runtime::Value::Float(v) => v,
            crate::runtime::Value::Int(i) => i as f64,
            _ => return None,
        };
        let want = ex.expected_f64()?;
        if (got - want).abs() > eps {
            return None;
        }
    }

    Some(SolveResult {
        success: true,
        code,
        method: "search_float_affine".to_string(),
        error: None,
        metadata: Default::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    fn pf(sig: &'static str, rows: &[(Vec<f64>, f64)]) -> Problem {
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: sig,
            examples: rows
                .iter()
                .map(|(xs, y)| Example {
                    inputs: xs.iter().map(|x| Value::Float(x.to_bits())).collect(),
                    expected: Value::Float(y.to_bits()),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        }
    }

    #[test]
    fn float_affine_recovers_line() {
        // y = 2.5*x + 1.3
        let f = |x: f64| 2.5 * x + 1.3;
        let rows: Vec<(Vec<f64>, f64)> = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
            .iter()
            .map(|&x| (vec![x], f(x)))
            .collect();
        let p = pf("fn f(x: f64) -> f64", &rows);
        let r = search_float_affine(&p, "f").expect("must recover 2.5x + 1.3");
        assert!(
            r.code.contains("2.5") && r.code.contains("1.3"),
            "code: {}",
            r.code
        );
        assert!(r.code.contains("-> f64"), "must be a float fn: {}", r.code);
    }

    #[test]
    fn float_affine_recovers_two_arg() {
        // celsius-like: z = 1.8*a + 0.5*b - 2.0
        let f = |a: f64, b: f64| 1.8 * a + 0.5 * b - 2.0;
        let raw = [
            (1.0, 1.0),
            (2.0, 3.0),
            (0.0, 0.0),
            (5.0, 2.0),
            (3.0, 7.0),
            (10.0, 1.0),
            (4.0, 4.0),
        ];
        let rows: Vec<(Vec<f64>, f64)> = raw.iter().map(|&(a, b)| (vec![a, b], f(a, b))).collect();
        let p = pf("fn f(a: f64, b: f64) -> f64", &rows);
        let r = search_float_affine(&p, "f").expect("must recover the 2-arg float affine");
        assert!(r.code.contains("-> f64"), "code: {}", r.code);
    }

    #[test]
    fn float_affine_refuses_nonlinear() {
        // y = x^2 is not affine — least squares fits a line that misses, refuse.
        let f = |x: f64| x * x;
        let rows: Vec<(Vec<f64>, f64)> = (0..12).map(|i| (vec![i as f64], f(i as f64))).collect();
        let p = pf("fn f(x: f64) -> f64", &rows);
        assert!(
            search_float_affine(&p, "f").is_none(),
            "must refuse a parabola"
        );
    }
}
