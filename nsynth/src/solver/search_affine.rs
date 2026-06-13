//! Exact multi-argument affine synthesis.
//!
//! Real requirements are mostly multi-argument (`cost(base, units)`,
//! `ship(weight, zone)`), and the scalar branch/piecewise solvers are all
//! single-input. These two solvers close that gap for the linear family:
//!
//!   * `search_affine` — a global affine rule `c0 + c1·a + c2·b + …`, recovered
//!     exactly by solving the integer linear system the examples define.
//!   * `search_affine_threshold` — a single threshold on one argument with an
//!     affine piece on each side, `if a ≤ k { …affine… } else { …affine… }`.
//!
//! Both are exact (the candidate is always verified against every example before
//! it is returned), so they generalize by construction rather than by fitting.

use super::scalar_search::{render_scalar_expr, ScalarBinOp, ScalarExpr};
use super::search_codegen::verified_result;
use super::signature::{scalar_param_names, scalar_params_decl};
use super::*;

/// Build `[x1, …, xn]` example rows of integer arguments (2–3 args), or None if
/// the signature is not a small all-integer multi-arg function.
fn multi_arg_examples(problem: &Problem) -> Option<(Vec<Vec<i64>>, Vec<i64>, usize)> {
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if !(2..=3).contains(&arity) || examples.iter().any(|row| row.len() != arity) {
        return None;
    }
    let targets: Vec<i64> = problem.examples.iter().map(|example| example.expected).collect();
    if targets.len() != examples.len() {
        return None;
    }
    Some((examples, targets, arity))
}

/// Solve `c0 + Σ c_j·x_j = y` for integer coefficients `[c0, c1, …, cn]` from the
/// examples. Gaussian elimination in f64 with partial pivoting recovers the
/// coefficients; they are rounded to the nearest integer and rejected if they
/// are not (near-)integral. The final exactness check is the caller's
/// `verified_result`, so a wrong fit (non-affine data, ill-conditioning) is
/// caught there and discarded — this routine only proposes.
fn solve_affine(examples: &[Vec<i64>], targets: &[i64], arity: usize) -> Option<Vec<i64>> {
    let m = arity + 1; // unknowns: const + one per argument
    if examples.len() < m {
        return None;
    }
    // Augmented rows: [1, x_1, …, x_n | y].
    let mut a: Vec<Vec<f64>> = examples
        .iter()
        .zip(targets.iter())
        .map(|(row, &t)| {
            let mut r = Vec::with_capacity(m + 1);
            r.push(1.0);
            r.extend(row.iter().map(|&v| v as f64));
            r.push(t as f64);
            r
        })
        .collect();
    let rows = a.len();
    let mut pivot = 0;
    for col in 0..m {
        let mut sel = None;
        let mut best = 1e-6;
        for r in pivot..rows {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                sel = Some(r);
            }
        }
        let sel = sel?; // column has no pivot → rank-deficient, give up
        a.swap(pivot, sel);
        let pv = a[pivot][col];
        for r in 0..rows {
            if r != pivot && a[r][col].abs() > 1e-9 {
                let factor = a[r][col] / pv;
                for cc in col..=m {
                    a[r][cc] -= factor * a[pivot][cc];
                }
            }
        }
        pivot += 1;
        if pivot == m {
            break;
        }
    }
    if pivot < m {
        return None;
    }
    let mut coeffs = vec![0i64; m];
    for (row, slot) in coeffs.iter_mut().enumerate() {
        let value = a[row][m] / a[row][row];
        let rounded = value.round();
        if (value - rounded).abs() > 1e-3 || !rounded.is_finite() {
            return None;
        }
        *slot = rounded as i64;
    }
    Some(coeffs)
}

/// `coeffs = [c0, c1, …, cn]` → the expression `c0 + c1·v0 + c2·v1 + …`,
/// dropping zero terms and rendering `1·v` as just `v`.
fn affine_expr(coeffs: &[i64]) -> ScalarExpr {
    let mut terms: Vec<ScalarExpr> = Vec::new();
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        if c == 0 {
            continue;
        }
        let var = ScalarExpr::Var(i - 1);
        terms.push(if c == 1 {
            var
        } else {
            ScalarExpr::Bin(Box::new(ScalarExpr::Const(c)), ScalarBinOp::Mul, Box::new(var))
        });
    }
    if coeffs[0] != 0 || terms.is_empty() {
        terms.push(ScalarExpr::Const(coeffs[0]));
    }
    let mut iter = terms.into_iter();
    let first = iter.next().expect("at least one term");
    iter.fold(first, |acc, term| {
        ScalarExpr::Bin(Box::new(acc), ScalarBinOp::Add, Box::new(term))
    })
}

pub(super) fn search_affine(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let (examples, targets, arity) = multi_arg_examples(problem)?;
    let coeffs = solve_affine(&examples, &targets, arity)?;
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);
    let body = render_scalar_expr(&affine_expr(&coeffs), &param_names);
    let code = format!("fn {fn_name}({params}) -> i64 {{\n    return {body};\n}}\n");
    verified_result(problem, code, "search_affine")
}

pub(super) fn search_affine_threshold(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let (examples, targets, arity) = multi_arg_examples(problem)?;
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);
    // Try each argument as the thresholded one, against every distinct value it
    // takes (the breakpoint must fall between sampled values, so a data value is
    // always a valid split point to try).
    for ti in 0..arity {
        let mut thresholds: Vec<i64> = examples.iter().map(|row| row[ti]).collect();
        thresholds.sort_unstable();
        thresholds.dedup();
        for &k in &thresholds {
            let mut lo_x = Vec::new();
            let mut lo_y = Vec::new();
            let mut hi_x = Vec::new();
            let mut hi_y = Vec::new();
            for (row, &t) in examples.iter().zip(targets.iter()) {
                if row[ti] <= k {
                    lo_x.push(row.clone());
                    lo_y.push(t);
                } else {
                    hi_x.push(row.clone());
                    hi_y.push(t);
                }
            }
            if lo_x.len() < arity + 1 || hi_x.len() < arity + 1 {
                continue;
            }
            let Some(lo) = solve_affine(&lo_x, &lo_y, arity) else {
                continue;
            };
            let Some(hi) = solve_affine(&hi_x, &hi_y, arity) else {
                continue;
            };
            if lo == hi {
                continue; // not actually a piecewise rule
            }
            let cond = format!("{} <= {}", param_names[ti], k);
            let lo_body = render_scalar_expr(&affine_expr(&lo), &param_names);
            let hi_body = render_scalar_expr(&affine_expr(&hi), &param_names);
            let code = format!(
                "fn {fn_name}({params}) -> i64 {{\n    if {cond} {{\n        return {lo_body};\n    }}\n    return {hi_body};\n}}\n"
            );
            if let Some(result) = verified_result(problem, code, "search_affine_threshold") {
                return Some(result);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    fn p2(rows: &[((i64, i64), i64)]) -> Problem {
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: "fn f(a: i64, b: i64) -> i64",
            examples: rows
                .iter()
                .map(|((a, b), y)| Example { inputs: vec![Value::Int(*a), Value::Int(*b)], expected: *y })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        }
    }

    // A two-argument affine rule `5 + 3a + 2b` is recovered exactly by the
    // integer linear solve and verified correct on unseen points.
    #[test]
    fn affine_recovers_two_arg_linear() {
        let f = |a: i64, b: i64| 3 * a + 2 * b + 5;
        let rows: Vec<((i64, i64), i64)> =
            [(1, 1), (2, 1), (1, 2), (5, 2), (0, 0), (10, 10), (4, 7), (8, 3)]
                .iter()
                .map(|&(a, b)| ((a, b), f(a, b)))
                .collect();
        let p = p2(&rows);
        let r = search_affine(&p, "f").expect("affine must solve 3a+2b+5");
        let check = p2(&[((13, 4), f(13, 4)), ((99, 50), f(99, 50)), ((7, 200), f(7, 200))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("affine must be exact on unseen points");
    }

    // A single threshold on one argument with an affine piece on each side is
    // recovered exactly: `if a <= 100 { 3b } else { 2(a-100) + 3b }`.
    #[test]
    fn affine_threshold_recovers_single_breakpoint() {
        let f = |a: i64, b: i64| (if a > 100 { 2 * (a - 100) } else { 0 }) + 3 * b;
        let rows: Vec<((i64, i64), i64)> = [
            (0, 1), (50, 2), (90, 5), (100, 3), (101, 4), (150, 6), (300, 1), (500, 9), (40, 7),
            (700, 2), (95, 8), (250, 0),
        ]
        .iter()
        .map(|&(a, b)| ((a, b), f(a, b)))
        .collect();
        let p = p2(&rows);
        let r = search_affine_threshold(&p, "f").expect("threshold-affine must solve the rule");
        assert!(r.code.contains("if"), "expected a branch: {}", r.code);
        let check = p2(&[((110, 3), f(110, 3)), ((1000, 4), f(1000, 4)), ((10, 10), f(10, 10))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("threshold-affine must be exact on unseen points");
    }

    // It refuses a product (a·b) — not affine — rather than emitting a wrong fit.
    #[test]
    fn affine_refuses_nonlinear() {
        let rows: Vec<((i64, i64), i64)> =
            (1..14).map(|i| ((i, i + 1), i * (i + 1))).collect();
        let p = p2(&rows);
        assert!(search_affine(&p, "f").is_none(), "affine must refuse a product");
    }
}
