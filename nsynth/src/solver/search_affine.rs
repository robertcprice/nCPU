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

/// Solve the integer linear system `PHI · w = targets` for `w` (length `m`) from
/// pre-built feature rows (each `feature_rows[i]` is the length-`m` feature
/// vector `phi` for example `i`). Gaussian elimination in f64 with partial
/// pivoting recovers `w`; the entries are rounded to the nearest integer and
/// rejected if they are not (near-)integral. This is byte-for-byte the body of
/// the original `solve_affine` lifted to an arbitrary feature basis, so the
/// affine path that delegates to it (`feature = [1, x_1, …, x_n]`) is unchanged.
/// The final exactness check is the caller's `verified_result`, so a wrong fit
/// (wrong basis, ill-conditioning) is caught there — this routine only proposes.
fn solve_linear_features(feature_rows: &[Vec<i64>], targets: &[i64], m: usize) -> Option<Vec<i64>> {
    if feature_rows.len() < m {
        return None;
    }
    // Augmented rows: [phi_1, …, phi_m | y].
    let mut a: Vec<Vec<f64>> = feature_rows
        .iter()
        .zip(targets.iter())
        .map(|(row, &t)| {
            let mut r = Vec::with_capacity(m + 1);
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

/// Solve `c0 + Σ c_j·x_j = y` for integer coefficients `[c0, c1, …, cn]` from the
/// examples by building the affine feature rows `[1, x_1, …, x_n]` and delegating
/// to `solve_linear_features` (`m = arity + 1`). The body is identical to the
/// original direct implementation — only the elimination machinery has been
/// factored out so the quadratic solver can reuse it.
fn solve_affine(examples: &[Vec<i64>], targets: &[i64], arity: usize) -> Option<Vec<i64>> {
    let m = arity + 1; // unknowns: const + one per argument
    let feature_rows: Vec<Vec<i64>> = examples
        .iter()
        .map(|row| {
            let mut r = Vec::with_capacity(m);
            r.push(1);
            r.extend(row.iter().copied());
            r
        })
        .collect();
    solve_linear_features(&feature_rows, targets, m)
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

/// `w = [c0, c1, d1, c2, d2, …, cn, dn]` (the layout produced by
/// `search_polynomial_multi`: a const, then a linear+quadratic coefficient per
/// axis) → the expression `c0 + Σ_j (c_j·x_j + d_j·x_j²)`, dropping zero terms,
/// rendering `1·x` as `x` and `1·x²` as `(x·x)`. Reuses `affine_expr`'s constant
/// rule (append `c0` iff it is non-zero, or there are no other terms).
fn polynomial_expr(w: &[i64], arity: usize) -> ScalarExpr {
    let mut terms: Vec<ScalarExpr> = Vec::new();
    for j in 0..arity {
        let lin = w[1 + 2 * j];
        let quad = w[2 + 2 * j];
        if lin != 0 {
            let var = ScalarExpr::Var(j);
            terms.push(if lin == 1 {
                var
            } else {
                ScalarExpr::Bin(Box::new(ScalarExpr::Const(lin)), ScalarBinOp::Mul, Box::new(var))
            });
        }
        if quad != 0 {
            // x_j² rendered as (x_j · x_j).
            let sq = ScalarExpr::Bin(
                Box::new(ScalarExpr::Var(j)),
                ScalarBinOp::Mul,
                Box::new(ScalarExpr::Var(j)),
            );
            terms.push(if quad == 1 {
                sq
            } else {
                ScalarExpr::Bin(Box::new(ScalarExpr::Const(quad)), ScalarBinOp::Mul, Box::new(sq))
            });
        }
    }
    if w[0] != 0 || terms.is_empty() {
        terms.push(ScalarExpr::Const(w[0]));
    }
    let mut iter = terms.into_iter();
    let first = iter.next().expect("at least one term");
    iter.fold(first, |acc, term| {
        ScalarExpr::Bin(Box::new(acc), ScalarBinOp::Add, Box::new(term))
    })
}

/// Exact separable degree-2 (per-variable quadratic) integer rule over 1–3
/// integer arguments: `y = c0 + Σ_j (c_j·x_j + d_j·x_j²)`. Covers curved
/// single-argument rules `a·x² + b·x + c` that the affine solvers refuse, plus
/// mixed quadratic-per-axis multi-argument rules (e.g. `x² + 2y + 3`). The
/// cross-term `x_i·x_j` is deliberately excluded to keep the feature matrix
/// small and the solve deterministic — a documented gap, not a bug.
///
/// INVERSION: build the monomial feature row `phi = [1, x_1, x_1², …, x_n, x_n²]`
/// (length `m = 1 + 2·arity`), solving the integer linear system via the shared
/// `solve_linear_features`. The `x²` feature is formed in `i128` before the f64
/// cast so large inputs do not overflow during matrix construction; the
/// round-to-int gate plus `verified_result` reject any case where f64 precision
/// on `x²` lost integrality. Returns None when there are fewer than `m` examples.
///
/// EARLY-OUT: if every quadratic coefficient rounds to zero the program is pure
/// affine — for arity ≥ 2 return None and let `search_affine` own those (it runs
/// first). For arity == 1 the all-quad-zero (pure-linear `b·x + c`) result is
/// kept, because `search_affine` is gated to 2–3 args and cannot fire here.
pub(super) fn search_polynomial_multi(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if !(1..=3).contains(&arity) || examples.iter().any(|row| row.len() != arity) {
        return None;
    }
    let targets: Vec<i64> = problem.examples.iter().map(|example| example.expected).collect();
    if targets.len() != examples.len() {
        return None;
    }
    let m = 1 + 2 * arity;
    if examples.len() < m {
        return None;
    }
    // Feature rows phi = [1, x_1, x_1², …, x_n, x_n²]. x² is computed in i128 so
    // the construction does not overflow; values that lose integrality in the f64
    // solve are caught by the round-to-int gate and by verified_result.
    let feature_rows: Vec<Vec<i64>> = examples
        .iter()
        .map(|row| {
            let mut phi = Vec::with_capacity(m);
            phi.push(1);
            for &x in row {
                phi.push(x);
                let sq = (x as i128) * (x as i128);
                phi.push(sq as i64);
            }
            phi
        })
        .collect();
    let w = solve_linear_features(&feature_rows, &targets, m)?;
    // Early-out for degenerate (pure-affine) fits when search_affine owns the case.
    let all_quad_zero = (0..arity).all(|j| w[2 + 2 * j] == 0);
    if all_quad_zero && arity >= 2 {
        return None;
    }
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);
    let body = render_scalar_expr(&polynomial_expr(&w, arity), &param_names);
    let code = format!("fn {fn_name}({params}) -> i64 {{\n    return {body};\n}}\n");
    verified_result(problem, code, "search_polynomial_multi")
}

/// Exact clamped-affine rules: the whole output is an affine function of the
/// arguments saturated against constant bound(s):
///
///   * floor   `max(lo, A(x))`          — a minimum (ReLU / "never below lo")
///   * cap     `min(hi, A(x))`          — a saturation ceiling ("never above hi")
///   * band    `min(hi, max(lo, A(x)))` — a two-sided saturation band
///
/// with `A(x) = c0 + Σ c_j·x_j`. Clamping is everywhere in real rules — minimum
/// fees, spend caps, ReLU, physical saturation, control limits — yet every
/// affine solver refuses it the moment the data stops being one straight line.
///
/// INVERSION: a clamp's bound is a constant the output never crosses, so the
/// floor `lo` is exactly `min(targets)` and the cap `hi` is exactly
/// `max(targets)` whenever the rule is genuinely active (at least one sample
/// rests on each bound). The points strictly *inside* the band are pure affine,
/// so `A` is recovered by the integer linear solve on just those interior
/// points. Each candidate is checked against EVERY example by `verified_result`,
/// so a coincidental bound, a flat dataset, or an under-determined inner fit is
/// rejected rather than returned. The affine body is inlined (not bound to a
/// local) so the emitted program uses only `if`/comparison/`return` — the same
/// constructs the threshold solver already round-trips through the transpiler.
pub(super) fn search_clamp_affine(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if !(1..=3).contains(&arity) || examples.iter().any(|row| row.len() != arity) {
        return None;
    }
    let targets: Vec<i64> = problem.examples.iter().map(|example| example.expected).collect();
    if targets.len() != examples.len() {
        return None;
    }
    let m = arity + 1;
    let lo = *targets.iter().min()?;
    let hi = *targets.iter().max()?;
    if lo == hi {
        return None; // constant output — nothing for a clamp to recover
    }
    // A bound is only a real clamp if the output actually SATURATES there — i.e.
    // several distinct examples rest on exactly that value (a plateau). A lone
    // extreme is just the smallest/largest *active* sample, not a floor/ceiling;
    // treating it as one invents a clamp the data never shows and is wrong on
    // unseen inputs. Requiring an observed plateau is what keeps this solver
    // honest: it recovers clamps it can see and refuses the ones it cannot.
    let plateau = |bound: i64| targets.iter().filter(|&&t| t == bound).count();
    let lo_observed = plateau(lo) >= 2;
    let hi_observed = plateau(hi) >= 2;
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);

    // Fit the inner affine on the band interior: the examples whose target is
    // strictly above `gt` (when given) and strictly below `lt` (when given),
    // requiring at least `m` such points so the system is determined. Returns
    // the rendered affine body string. Taking explicit bounds (rather than a
    // predicate closure) keeps the call sites monomorphic — no `dyn Fn`.
    let fit_body = |gt: Option<i64>, lt: Option<i64>| -> Option<String> {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (row, &t) in examples.iter().zip(targets.iter()) {
            if gt.map_or(true, |g| t > g) && lt.map_or(true, |l| t < l) {
                xs.push(row.clone());
                ys.push(t);
            }
        }
        if xs.len() < m {
            return None;
        }
        let coeffs = solve_affine(&xs, &ys, arity)?;
        // The Gaussian solve reads the answer off the first `m` pivot rows and
        // never checks the remaining interior rows for consistency. For a true
        // clamp every interior point lies exactly on the inner line, so demand
        // that `coeffs` reproduces ALL of them. This refuses an under-determined
        // or f64-rounded fit (the band failure mode) instead of emitting a
        // saturation that is right on the samples but wrong between them.
        if !xs.iter().zip(ys.iter()).all(|(row, &y)| affine_predicts(&coeffs, row, y)) {
            return None;
        }
        Some(render_scalar_expr(&affine_expr(&coeffs), &param_names))
    };

    // floor: max(lo, A) — interior is the points strictly above the floor.
    if lo_observed {
        if let Some(body) = fit_body(Some(lo), None) {
            let code = format!(
                "fn {fn_name}({params}) -> i64 {{\n    if ({body}) < {lo} {{\n        return {lo};\n    }}\n    return {body};\n}}\n"
            );
            if let Some(result) = verified_result(problem, code, "search_clamp_affine") {
                return Some(result);
            }
        }
    }
    // cap: min(hi, A) — interior is the points strictly below the cap.
    if hi_observed {
        if let Some(body) = fit_body(None, Some(hi)) {
            let code = format!(
                "fn {fn_name}({params}) -> i64 {{\n    if ({body}) > {hi} {{\n        return {hi};\n    }}\n    return {body};\n}}\n"
            );
            if let Some(result) = verified_result(problem, code, "search_clamp_affine") {
                return Some(result);
            }
        }
    }
    // band: min(hi, max(lo, A)) — both bounds must be observed plateaus.
    if lo_observed && hi_observed {
        if let Some(body) = fit_body(Some(lo), Some(hi)) {
            let code = format!(
                "fn {fn_name}({params}) -> i64 {{\n    if ({body}) < {lo} {{\n        return {lo};\n    }}\n    if ({body}) > {hi} {{\n        return {hi};\n    }}\n    return {body};\n}}\n"
            );
            if let Some(result) = verified_result(problem, code, "search_clamp_affine") {
                return Some(result);
            }
        }
    }
    None
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

/// True iff `c0 + Σ c_{i+1}·row[i] == y` (i128 accumulation avoids overflow).
fn affine_predicts(coeffs: &[i64], row: &[i64], y: i64) -> bool {
    let mut acc = coeffs[0] as i128;
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        acc += c as i128 * row[i - 1] as i128;
    }
    acc == y as i128
}

struct MultiSeg {
    coeffs: Vec<i64>,
    x_last: i64,
    points: usize,
}

/// Greedily split examples (already sorted ascending by argument `ti`) into runs
/// where one multi-argument affine `c0 + Σ c_j·x_j` fits every point. Each run
/// is opened by solving the affine on its first `arity+1` points and extended
/// while subsequent points satisfy it; the next run begins at the first point
/// that does not — which is the start of the next tier, so a tier's opening
/// window never straddles a breakpoint as long as each tier has ≥ arity+1
/// sampled points. Returns None when a tier is too sparse to fit (honest:
/// under-sampled rather than guessed).
fn segment_multiarg(sorted: &[(&Vec<i64>, i64)], arity: usize, ti: usize) -> Option<Vec<MultiSeg>> {
    let m = arity + 1;
    let n = sorted.len();
    let mut segs: Vec<MultiSeg> = Vec::new();
    let mut i = 0;
    while i < n {
        if i + m > n {
            // Too few points left to fit a new affine. Only acceptable if they
            // all lie on the current segment (a slightly short final tier).
            let last = segs.last_mut()?;
            if (i..n).all(|k| affine_predicts(&last.coeffs, sorted[k].0, sorted[k].1)) {
                last.x_last = sorted[n - 1].0[ti];
                last.points += n - i;
                return Some(segs);
            }
            return None;
        }
        let win_x: Vec<Vec<i64>> = (i..i + m).map(|k| sorted[k].0.clone()).collect();
        let win_y: Vec<i64> = (i..i + m).map(|k| sorted[k].1).collect();
        let coeffs = solve_affine(&win_x, &win_y, arity)?;
        let mut j = i + m;
        while j < n && affine_predicts(&coeffs, sorted[j].0, sorted[j].1) {
            j += 1;
        }
        segs.push(MultiSeg { coeffs, x_last: sorted[j - 1].0[ti], points: j - i });
        i = j;
    }
    Some(segs)
}

/// Multi-argument piecewise-affine: a rule that is affine in all arguments
/// within each tier of *one* threshold argument (e.g. a shipping cost that is
/// linear in weight and zone, tiered by weight). Generalises
/// `search_affine_threshold` from two pieces to any number, recovering the tiers
/// by greedy segmentation along each candidate threshold argument.
pub(super) fn search_affine_piecewise(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let (examples, targets, arity) = multi_arg_examples(problem)?;
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);
    for ti in 0..arity {
        let mut order: Vec<usize> = (0..examples.len()).collect();
        order.sort_by_key(|&k| examples[k][ti]);
        let sorted: Vec<(&Vec<i64>, i64)> =
            order.iter().map(|&k| (&examples[k], targets[k])).collect();
        let Some(segs) = segment_multiarg(&sorted, arity, ti) else {
            continue;
        };
        if !(2..=5).contains(&segs.len()) || segs.iter().any(|s| s.points < arity + 1) {
            continue;
        }
        // Place each breakpoint at the value of `ti` where the two adjoining
        // pieces meet — the true threshold of a continuous tiered rule — rather
        // than at the last sampled point (which misclassifies the unsampled gap
        // up to the real breakpoint). The intersection is a clean threshold on
        // `ti` only when the other arguments' slopes match across the two pieces
        // (so their terms cancel); otherwise fall back to the last sample.
        let mut body = String::new();
        for k in 0..segs.len() - 1 {
            let s0 = &segs[k].coeffs;
            let s1 = &segs[k + 1].coeffs;
            let others_match = (0..arity).filter(|&j| j != ti).all(|j| s0[j + 1] == s1[j + 1]);
            let bp = if others_match && s0[ti + 1] != s1[ti + 1] {
                let num = s1[0] - s0[0];
                let den = s0[ti + 1] - s1[ti + 1];
                if num % den == 0 {
                    num / den
                } else {
                    segs[k].x_last
                }
            } else {
                segs[k].x_last
            };
            let piece = render_scalar_expr(&affine_expr(s0), &param_names);
            body.push_str(&format!(
                "    if {} <= {bp} {{\n        return {piece};\n    }}\n",
                param_names[ti]
            ));
        }
        let last = render_scalar_expr(&affine_expr(&segs.last().unwrap().coeffs), &param_names);
        body.push_str(&format!("    return {last};\n"));
        let code = format!("fn {fn_name}({params}) -> i64 {{\n{body}}}\n");
        if let Some(result) = verified_result(problem, code, "search_affine_piecewise") {
            return Some(result);
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

    fn p1(rows: &[(i64, i64)]) -> Problem {
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: "fn f(x: i64) -> i64",
            examples: rows
                .iter()
                .map(|&(x, y)| Example { inputs: vec![Value::Int(x)], expected: y })
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

    // A multi-argument, multi-TIER rule: affine in (a, b) within each of three
    // tiers of `a`. Recovered exactly and verified on unseen points.
    #[test]
    fn affine_piecewise_recovers_multiarg_tiers() {
        // tier0 a<=100: 3b ; tier1 100<a<=500: 2(a-100)+3b ; tier2 a>500: a+300+3b
        let f = |a: i64, b: i64| {
            let base = if a <= 100 {
                0
            } else if a <= 500 {
                2 * (a - 100)
            } else {
                800 + (a - 500)
            };
            base + 3 * b
        };
        let raw = [
            (10, 1), (40, 5), (90, 2), (100, 7), // tier0 (4)
            (150, 3), (200, 9), (300, 1), (450, 6), (500, 4), // tier1 (5)
            (600, 2), (800, 8), (1200, 5), (1500, 0), // tier2 (4)
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_affine_piecewise(&p, "f").expect("must solve the 3-tier multi-arg rule");
        assert!(r.code.matches("if").count() >= 2, "expected 3 pieces: {}", r.code);
        let check = p2(&[
            ((75, 3), f(75, 3)),
            ((250, 11), f(250, 11)),
            ((350, 0), f(350, 0)),
            ((900, 4), f(900, 4)),
            ((2000, 7), f(2000, 7)),
        ]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("multi-arg tiered rule must be exact on unseen points");
    }

    // It refuses a product (a·b) — not affine — rather than emitting a wrong fit.
    #[test]
    fn affine_refuses_nonlinear() {
        let rows: Vec<((i64, i64), i64)> =
            (1..14).map(|i| ((i, i + 1), i * (i + 1))).collect();
        let p = p2(&rows);
        assert!(search_affine(&p, "f").is_none(), "affine must refuse a product");
    }

    // A curved single-argument rule `2x² − 3x + 5` is recovered exactly by the
    // monomial-feature solve and verified correct on UNSEEN points — proving
    // generalization, not fit. (The affine solvers refuse this curvature.)
    #[test]
    fn polynomial_recovers_one_arg_quadratic() {
        let f = |x: i64| 2 * x * x - 3 * x + 5;
        let rows: Vec<(i64, i64)> =
            [0, 1, 2, 3, 4, 5, 7, 10].iter().map(|&x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_polynomial_multi(&p, "f").expect("must solve 2x^2 - 3x + 5");
        let check = p1(&[(13, f(13)), (50, f(50)), (99, f(99))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("quadratic must be exact on unseen points");
    }

    // A two-argument SEPARABLE quadratic `a² + 2b + 3` is recovered exactly and
    // verified on unseen points.
    #[test]
    fn polynomial_recovers_two_arg_separable() {
        let f = |a: i64, b: i64| a * a + 2 * b + 3;
        let rows: Vec<((i64, i64), i64)> =
            [(0, 0), (1, 1), (2, 3), (3, 5), (4, 2), (5, 7), (7, 1)]
                .iter()
                .map(|&(a, b)| ((a, b), f(a, b)))
                .collect();
        let p = p2(&rows);
        let r = search_polynomial_multi(&p, "f").expect("must solve a^2 + 2b + 3");
        let check = p2(&[((13, 4), f(13, 4)), ((99, 50), f(99, 50))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("separable quadratic must be exact on unseen points");
    }

    // It refuses a true cross-term `a·b` — the separable feature basis cannot
    // express it, so the round-to-int / rank check rejects rather than overfit.
    #[test]
    fn polynomial_refuses_cross_term() {
        let rows: Vec<((i64, i64), i64)> =
            (1..14).map(|i| ((i, i + 1), i * (i + 1))).collect();
        let p = p2(&rows);
        assert!(
            search_polynomial_multi(&p, "f").is_none(),
            "separable basis must refuse a cross-term product"
        );
    }

    // A one-argument FLOOR `max(0, 5x - 40)` (a minimum charge: nothing until
    // usage passes 8, then 5/unit) is recovered exactly and is correct on unseen
    // points both inside and below the floor.
    #[test]
    fn clamp_recovers_one_arg_floor() {
        let f = |x: i64| (5 * x - 40).max(0);
        let rows: Vec<(i64, i64)> =
            [0, 2, 4, 6, 8, 10, 14, 20, 30].iter().map(|&x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_clamp_affine(&p, "f").expect("must recover max(0, 5x-40)");
        let check = p1(&[(1, f(1)), (7, f(7)), (9, f(9)), (50, f(50)), (100, f(100))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("floor must be exact on unseen points");
    }

    // A two-argument CAP `min(500, 10a + 3b)` (a billing ceiling) is recovered
    // exactly and correct on unseen points above and below the cap.
    #[test]
    fn clamp_recovers_two_arg_cap() {
        let f = |a: i64, b: i64| (10 * a + 3 * b).min(500);
        let raw = [
            (0, 0), (1, 1), (5, 2), (10, 10), (20, 5), (3, 3), (8, 8), // below cap
            (60, 10), (90, 40), (100, 0), (200, 100), // at/above cap (clamped to 500)
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_clamp_affine(&p, "f").expect("must recover min(500, 10a+3b)");
        let check = p2(&[((4, 4), f(4, 4)), ((49, 3), f(49, 3)), ((300, 7), f(300, 7))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("cap must be exact on unseen points");
    }

    // It refuses data that is not a clamped line (a true product a·b): no
    // constant floor/cap fits, so it returns None rather than a wrong saturation.
    #[test]
    fn clamp_refuses_nonclamp() {
        let rows: Vec<((i64, i64), i64)> =
            (1..14).map(|i| ((i, i + 1), i * (i + 1))).collect();
        let p = p2(&rows);
        assert!(
            search_clamp_affine(&p, "f").is_none(),
            "clamp must refuse a non-saturated nonlinear rule"
        );
    }

    // A pure-linear single-argument rule `3x + 1` (all quadratic coeffs zero) is
    // still solved at arity 1, because search_affine is gated to 2–3 args and so
    // this legitimately lands in the polynomial solver's arity==1 branch.
    #[test]
    fn polynomial_one_arg_affine_still_lands() {
        let f = |x: i64| 3 * x + 1;
        let rows: Vec<(i64, i64)> =
            [0, 1, 2, 3, 4, 5, 7].iter().map(|&x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_polynomial_multi(&p, "f").expect("must solve 3x + 1 at arity 1");
        let check = p1(&[(13, f(13)), (88, f(88))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("linear arity-1 must be exact on unseen points");
    }
}
