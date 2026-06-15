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
    let targets: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
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
pub(super) fn solve_linear_features(
    feature_rows: &[Vec<i64>],
    targets: &[i64],
    m: usize,
) -> Option<Vec<i64>> {
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
            ScalarExpr::Bin(
                Box::new(ScalarExpr::Const(c)),
                ScalarBinOp::Mul,
                Box::new(var),
            )
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
                ScalarExpr::Bin(
                    Box::new(ScalarExpr::Const(lin)),
                    ScalarBinOp::Mul,
                    Box::new(var),
                )
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
                ScalarExpr::Bin(
                    Box::new(ScalarExpr::Const(quad)),
                    ScalarBinOp::Mul,
                    Box::new(sq),
                )
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
    let targets: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
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

/// A derived feature: a small expression over the raw arguments (the `expr`,
/// kept for rendering and final verification) together with its already-computed
/// value on every example (`col`, used for the fast in-Rust exact-fit check).
struct Feature {
    expr: ScalarExpr,
    col: Vec<i64>,
}

/// Evaluate a per-row i128 feature closure into an i64 column, or return an empty
/// column (the caller drops the feature) if any row overflows i64 or the closure
/// is undefined there. Generic over the closure so each call monomorphises — no
/// `dyn Fn` and no allocation per row beyond the column itself.
fn feature_column<F: Fn(&[i64]) -> Option<i128>>(examples: &[Vec<i64>], f: F) -> Vec<i64> {
    let mut col = Vec::with_capacity(examples.len());
    for row in examples {
        match f(row) {
            Some(v) if i64::try_from(v).is_ok() => col.push(v as i64),
            _ => return Vec::new(),
        }
    }
    col
}

/// Build the candidate-feature library for compositional synthesis: each entry
/// is a little program over the raw inputs whose output becomes a regression
/// column. This is the engine "thinking in code" — it proposes intermediate
/// computations, runs them, and lets the exact linear solve recover structure
/// over the *results* rather than the raw inputs. Features:
///
///   * raw          `x_j`
///   * square       `x_j · x_j`
///   * cross        `x_i · x_j`   (i < j) — the cross-terms the separable
///                                  polynomial solver deliberately cannot express
///   * modulo       `x_j % m`     (m a mined constant, 2 ≤ m, no mod-by-input)
///   * floor-div    `x_j / m`     (m a mined constant, 2 ≤ m)
///
/// A feature is dropped entirely if it overflows or divides/mods by zero on any
/// example (we never emit a program that can trap). Moduli/divisors are *mined
/// from the problem's own data* (`mine_scalar_constants`), not hand-picked, so
/// the vocabulary is data-driven. The set is capped so the subset search stays
/// fast and over-determined.
fn compose_features(examples: &[Vec<i64>], arity: usize) -> Vec<Feature> {
    let mut feats: Vec<Feature> = Vec::new();
    let mut add = |expr: ScalarExpr, col: Vec<i64>| {
        if col.len() == examples.len() {
            feats.push(Feature { expr, col });
        }
    };

    for j in 0..arity {
        // raw
        add(
            ScalarExpr::Var(j),
            feature_column(examples, |r| Some(r[j] as i128)),
        );
        // square
        let sq = ScalarExpr::Bin(
            Box::new(ScalarExpr::Var(j)),
            ScalarBinOp::Mul,
            Box::new(ScalarExpr::Var(j)),
        );
        add(
            sq,
            feature_column(examples, |r| Some(r[j] as i128 * r[j] as i128)),
        );
    }
    // cross terms x_i · x_j
    for i in 0..arity {
        for j in (i + 1)..arity {
            let cross = ScalarExpr::Bin(
                Box::new(ScalarExpr::Var(i)),
                ScalarBinOp::Mul,
                Box::new(ScalarExpr::Var(j)),
            );
            add(
                cross,
                feature_column(examples, |r| Some(r[i] as i128 * r[j] as i128)),
            );
        }
    }
    // modulo / floor-div bases. The UNIVERSAL arithmetic bases — parity (2),
    // mod-3, quarters (4), mod-5/7, decimal digit (10) — are always available:
    // these are the moduli/divisors of essentially every real periodic or banded
    // rule, foundational like `+`/`*` rather than problem-specific magic, so they
    // are included whether or not the literal value appears in the data (without
    // this, `x % 4` was unexpressible whenever no input happened to equal 4).
    // On top of them, any OTHER constant mined from the problem's own inputs is
    // appended for genuinely problem-specific periods. Universal-first ordering
    // also fixes the bug where a plain ascending-sort + truncate dropped a useful
    // base behind a run of small accidental ones.
    let universal = [2i64, 3, 4, 5, 7, 10];
    let mined: Vec<i64> = super::scalar_search::mine_scalar_constants(examples, &[])
        .into_iter()
        .filter(|&m| (2..=64).contains(&m))
        .collect();
    let mut bases: Vec<i64> = universal.to_vec();
    for &m in &mined {
        if !bases.contains(&m) {
            bases.push(m);
        }
    }
    bases.truncate(8); // keep the feature set bounded and over-determined
    for &m in &bases {
        for j in 0..arity {
            let md = ScalarExpr::Bin(
                Box::new(ScalarExpr::Var(j)),
                ScalarBinOp::Mod,
                Box::new(ScalarExpr::Const(m)),
            );
            add(
                md,
                feature_column(examples, |r| Some(r[j].rem_euclid(m) as i128)),
            );
            let dv = ScalarExpr::Bin(
                Box::new(ScalarExpr::Var(j)),
                ScalarBinOp::Div,
                Box::new(ScalarExpr::Const(m)),
            );
            add(
                dv,
                feature_column(examples, |r| Some(r[j].div_euclid(m) as i128)),
            );
        }
    }
    feats
}

/// `c0` plus the chosen features → the expression `c0 + Σ c_k·feature_k`,
/// dropping zero coefficients and rendering `1·feature` as the feature itself.
/// Reuses `affine_expr`'s constant rule.
fn composed_expr(c0: i64, picks: &[(&Feature, i64)]) -> ScalarExpr {
    let mut terms: Vec<ScalarExpr> = Vec::new();
    for &(feat, c) in picks {
        if c == 0 {
            continue;
        }
        terms.push(if c == 1 {
            feat.expr.clone()
        } else {
            ScalarExpr::Bin(
                Box::new(ScalarExpr::Const(c)),
                ScalarBinOp::Mul,
                Box::new(feat.expr.clone()),
            )
        });
    }
    if c0 != 0 || terms.is_empty() {
        terms.push(ScalarExpr::Const(c0));
    }
    let mut iter = terms.into_iter();
    let first = iter.next().expect("at least one term");
    iter.fold(first, |acc, term| {
        ScalarExpr::Bin(Box::new(acc), ScalarBinOp::Add, Box::new(term))
    })
}

/// True iff `c0 + Σ coeffs_k·feat_k[row] == targets[row]` for every example
/// (i128 accumulation). The cheap in-Rust gate that lets the subset search try
/// thousands of feature combinations without parsing/executing Mog — only the
/// surviving combination is handed to `verified_result`.
fn composed_predicts(c0: i64, picks: &[(&Feature, i64)], targets: &[i64]) -> bool {
    for (i, &t) in targets.iter().enumerate() {
        let mut acc = c0 as i128;
        for &(feat, c) in picks {
            acc += c as i128 * feat.col[i] as i128;
        }
        if acc != t as i128 {
            return false;
        }
    }
    true
}

/// Recover the rendered BODY of the simplest exact rule over a set of example
/// rows — an affine first, and if no affine fits, a sparse composed-feature rule
/// (one or two derived features: square, cross-term, modulo, floor-div). This is
/// the recursion that makes the branch solvers "think in code": a branch is no
/// longer limited to a straight line — its body can itself be a composed program
/// the engine recovers. Affine-first means every rule that was already affine
/// renders identically, so existing behaviour is unchanged; the composed path
/// only adds genuinely non-linear branch bodies.
///
/// Over-determination is STRICTER here than for a whole-program composed fit
/// (≥ k + 1 + 3 rows per composed body) because a branch's row set is small, and
/// the caller still verifies the entire reconstructed program against every
/// example — so a per-branch fit that does not generalise is rejected upstream.
fn recover_body(
    xs: &[Vec<i64>],
    ys: &[i64],
    arity: usize,
    param_names: &[String],
) -> Option<String> {
    // Affine first — preserves the exact output of the pre-existing affine-body
    // branch solvers. `solve_affine` reads its answer off the pivot rows and does
    // not check the rest, so demand that the recovered affine reproduces EVERY
    // row before accepting it; otherwise fall through to the composed search
    // (this is what lets a genuinely non-linear branch reach the composed path
    // instead of being masked by a bogus affine fit).
    if let Some(c) = solve_affine(xs, ys, arity) {
        if xs
            .iter()
            .zip(ys.iter())
            .all(|(row, &y)| affine_predicts(&c, row, y))
        {
            return Some(render_scalar_expr(&affine_expr(&c), param_names));
        }
    }
    // Composed fallback: the affine base (ALL raw variables) plus exactly ONE
    // derived feature — `affine + e·g(x)` for a single square, cross-term,
    // modulo, or floor-div `g`. This is the common non-linear branch body (one
    // curved/periodic/coupled term on top of a line); always including the raw
    // base means `a·b + 2a - 3` is one derived feature on top of the affine
    // rather than a 3-feature subset, which keeps the search cheap (linear in the
    // derived-feature count) — important because this runs once per candidate
    // branch partition. Higher-order branch bodies are deliberately out of scope.
    const BODY_MARGIN: usize = 3;
    let feats = compose_features(xs, arity);
    if feats.is_empty() {
        return None;
    }
    let n = xs.len();
    let raws: Vec<usize> = (0..feats.len())
        .filter(|&i| matches!(feats[i].expr, ScalarExpr::Var(_)))
        .collect();
    let derived: Vec<usize> = (0..feats.len())
        .filter(|&i| !matches!(feats[i].expr, ScalarExpr::Var(_)))
        .collect();
    for &d in &derived {
        let mut idxs = raws.clone();
        idxs.push(d);
        let k = idxs.len();
        if n < k + 1 + BODY_MARGIN {
            continue;
        }
        let m = k + 1;
        let rows: Vec<Vec<i64>> = (0..n)
            .map(|i| {
                let mut phi = Vec::with_capacity(m);
                phi.push(1);
                for &fi in &idxs {
                    phi.push(feats[fi].col[i]);
                }
                phi
            })
            .collect();
        let Some(w) = solve_linear_features(&rows, ys, m) else {
            continue;
        };
        let picks: Vec<(&Feature, i64)> = idxs
            .iter()
            .enumerate()
            .map(|(s, &fi)| (&feats[fi], w[s + 1]))
            .collect();
        if composed_predicts(w[0], &picks, ys) {
            return Some(render_scalar_expr(
                &composed_expr(w[0], &picks),
                param_names,
            ));
        }
    }
    None
}

/// Compositional ("think-in-code") synthesis: recover an exact rule of the form
///
///     f(x) = c0 + Σ_k c_k · g_k(x)
///
/// where each `g_k` is a derived feature (see `compose_features`) — a square, a
/// cross-term `x_i·x_j`, a modulo `x_j % m`, a floor-div `x_j / m`, or a raw
/// argument. This is what lifts the engine past one straight line: it composes
/// little intermediate programs and recovers the exact *linear combination* of
/// their outputs that reproduces the data, generalising by construction.
///
/// HONESTY (the whole point — a rich feature basis can fit noise, so the guards
/// are the contract):
///   * SPARSE-FIRST: subsets are tried smallest-and-simplest first (1 feature,
///     then 2, then 3) and the FIRST exact fit wins — the simplest explanation,
///     which is the one that generalises. A 3-feature fit is only reached when
///     nothing simpler is exact.
///   * OVER-DETERMINED: a k-feature fit is attempted only when there are at
///     least `k + 1 + MARGIN` examples, so the system is never square (which
///     would fit anything).
///   * EXACT INTEGER + FULL VERIFY: `solve_linear_features` rounds to integers
///     and rejects non-integral solutions; `composed_predicts` then requires the
///     fit to reproduce EVERY example before it is even rendered; finally
///     `verified_result` re-checks through the real runtime.
///   * NOT PURE AFFINE: a winning subset of only raw variables is left to
///     `search_affine` (it runs first); composition only claims genuinely
///     nonlinear/derived rules.
pub(super) fn search_composed_features(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    const MARGIN: usize = 2;
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if !(1..=3).contains(&arity) || examples.iter().any(|row| row.len() != arity) {
        return None;
    }
    let targets: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
    if targets.len() != examples.len() {
        return None;
    }
    let n = examples.len();
    let feats = compose_features(&examples, arity);
    if feats.is_empty() {
        return None;
    }
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);

    // A feature subset is "interesting" only if at least one picked feature is
    // not a raw variable — pure-affine combinations belong to search_affine.
    let is_raw = |f: &Feature| matches!(f.expr, ScalarExpr::Var(_));

    // Solve for [c0, c_1, …, c_k] over the chosen feature columns and, if it is
    // an exact integer fit reproducing every example, render + verify it.
    let try_subset = |idxs: &[usize]| -> Option<SolveResult> {
        let k = idxs.len();
        if n < k + 1 + MARGIN {
            return None;
        }
        if idxs.iter().all(|&fi| is_raw(&feats[fi])) {
            return None; // pure affine — not ours
        }
        let m = k + 1;
        let feature_rows: Vec<Vec<i64>> = (0..n)
            .map(|i| {
                let mut phi = Vec::with_capacity(m);
                phi.push(1);
                for &fi in idxs {
                    phi.push(feats[fi].col[i]);
                }
                phi
            })
            .collect();
        let w = solve_linear_features(&feature_rows, &targets, m)?;
        let picks: Vec<(&Feature, i64)> = idxs
            .iter()
            .enumerate()
            .map(|(slot, &fi)| (&feats[fi], w[slot + 1]))
            .collect();
        if !composed_predicts(w[0], &picks, &targets) {
            return None;
        }
        let body = render_scalar_expr(&composed_expr(w[0], &picks), &param_names);
        let code = format!("fn {fn_name}({params}) -> i64 {{\n    return {body};\n}}\n");
        verified_result(problem, code, "search_composed_features")
    };

    let fc = feats.len();
    // size 1
    for a in 0..fc {
        if let Some(r) = try_subset(&[a]) {
            return Some(r);
        }
    }
    // size 2
    for a in 0..fc {
        for b in (a + 1)..fc {
            if let Some(r) = try_subset(&[a, b]) {
                return Some(r);
            }
        }
    }
    // size 3
    for a in 0..fc {
        for b in (a + 1)..fc {
            for c in (b + 1)..fc {
                if let Some(r) = try_subset(&[a, b, c]) {
                    return Some(r);
                }
            }
        }
    }
    None
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
    let targets: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
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
        if !xs
            .iter()
            .zip(ys.iter())
            .all(|(row, &y)| affine_predicts(&coeffs, row, y))
        {
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
        segs.push(MultiSeg {
            coeffs,
            x_last: sorted[j - 1].0[ti],
            points: j - i,
        });
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
            let others_match = (0..arity)
                .filter(|&j| j != ti)
                .all(|j| s0[j + 1] == s1[j + 1]);
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

/// Conditional logic: recover `if P(x) { A(x) } else { B(x) }` where `P` is a
/// predicate over ONE argument — a modular test `x_j % m == r` (parity, mod-3,
/// quarters, …) or a threshold `x_j <= k` — and each branch body is an exact
/// affine rule over all arguments. Real programs branch; this is the first
/// solver that recovers a genuine `if/else` whose condition is a DERIVED
/// predicate (the existing threshold solvers split only on a raw argument
/// inequality, never on a modular/parity class).
///
/// INVERSION: enumerate candidate predicates simplest-first (modular before
/// threshold, smallest base/most-natural first); for each, partition the
/// examples into the two branches, recover an exact integer affine on EACH side
/// (`solve_affine`), and keep the first split whose reconstructed `if/else`
/// reproduces every example (`verified_result`). The modulus partition is
/// computed with `rem_euclid` to match the emitted `x % m`; any residual
/// mismatch on negative inputs is caught by the full verify, never returned.
///
/// HONESTY GUARDS:
///   * each branch must be OVER-DETERMINED — at least `arity + 2` examples, one
///     more than the affine's `arity + 1` unknowns — so neither side is a square
///     system that fits anything;
///   * the two branch affines must DIFFER (else it is not a branch and
///     `search_affine` owns it);
///   * a coincidental partition that does not reproduce all examples is rejected
///     by `verified_result`.
pub(super) fn search_predicate_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if !(1..=3).contains(&arity) || examples.iter().any(|row| row.len() != arity) {
        return None;
    }
    let targets: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
    if targets.len() != examples.len() {
        return None;
    }
    let min_side = arity + 2; // over-determined: one row more than the unknowns
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);

    // Fit affine on each side of a boolean partition, render `if cond { A } B`,
    // and verify against every example. Returns the verified result or None.
    let try_partition = |cond: &str, side_true: &[bool]| -> Option<SolveResult> {
        let mut tx = Vec::new();
        let mut ty = Vec::new();
        let mut fx = Vec::new();
        let mut fy = Vec::new();
        for (i, row) in examples.iter().enumerate() {
            if side_true[i] {
                tx.push(row.clone());
                ty.push(targets[i]);
            } else {
                fx.push(row.clone());
                fy.push(targets[i]);
            }
        }
        if tx.len() < min_side || fx.len() < min_side {
            return None;
        }
        // Each branch body is recovered by the engine itself (affine first, then
        // a composed-feature program) — so a branch can be non-linear, not just a
        // straight line.
        let a_body = recover_body(&tx, &ty, arity, &param_names)?;
        let b_body = recover_body(&fx, &fy, arity, &param_names)?;
        if a_body == b_body {
            return None; // not actually a branch
        }
        let code = format!(
            "fn {fn_name}({params}) -> i64 {{\n    if {cond} {{\n        return {a_body};\n    }}\n    return {b_body};\n}}\n"
        );
        verified_result(problem, code, "search_predicate_branch")
    };

    // Modular predicates first (the genuinely new capability): for each argument
    // and each natural-or-mined base, split on each residue class. Bases are the
    // universal arithmetic moduli plus any constant the data exhibits.
    let mut bases: Vec<i64> = vec![2, 3, 4, 5, 7, 10];
    for &m in &super::scalar_search::mine_scalar_constants(&examples, &targets) {
        if (2..=32).contains(&m) && !bases.contains(&m) {
            bases.push(m);
        }
    }
    for ti in 0..arity {
        for &m in &bases {
            for r in 0..m {
                let cond = format!("({} % {m}) == {r}", param_names[ti]);
                let side_true: Vec<bool> = examples
                    .iter()
                    .map(|row| row[ti].rem_euclid(m) == r)
                    .collect();
                // Skip a partition that puts everything on one side.
                let t = side_true.iter().filter(|&&b| b).count();
                if t == 0 || t == examples.len() {
                    continue;
                }
                if let Some(result) = try_partition(&cond, &side_true) {
                    return Some(result);
                }
            }
        }
    }
    // Threshold predicates (`x_j <= k`) are deliberately NOT emitted here: an
    // argument inequality is already owned by `search_affine_threshold` (multi-
    // arg) and the scalar single/two-branch solvers (1-arg), and a discontinuous
    // threshold whose exact boundary value is not sampled is off-by-one
    // ambiguous — exactly the breakpoint problem the piecewise solver handles
    // with intersection placement. This solver's unique, exact contribution is
    // the MODULAR/parity class split, which has no boundary ambiguity, so it
    // stays modular-only and refuses the rest.
    None
}

/// Full modular CASE ANALYSIS: recover `match x_j % m { 0 => A_0(x), 1 => A_1(x),
/// …, m-1 => A_{m-1}(x) }` — a distinct exact affine for EVERY residue class of
/// one argument. Generalises `search_predicate_branch` (which special-cases a
/// single residue against the rest) to the full cyclic case split that round-
/// robin, scheduling, and calendar rules use (`day % 7`, `phase % 3`, …).
/// Emitted as an `if x%m == r { … }` chain, so it stays inside the if/else
/// fragment the runtime and transpiler already round-trip.
///
/// INVERSION: for each argument and modulus, bucket the examples by
/// `x_j % m`, recover an exact integer affine on EACH bucket (`solve_affine`),
/// and keep the smallest modulus whose reconstructed chain reproduces every
/// example. HONESTY GUARDS: every one of the `m` residue classes must be present
/// AND over-determined (≥ arity+2 points — so no class is a square system), the
/// `m` affines must not be all identical (else it is plain affine, owned by
/// `search_affine`), and the whole chain is checked by `verified_result`. The
/// strong per-class data requirement (≥ m·(arity+2) examples) is itself a guard:
/// a coarser/finer modulus that coincidentally aligns will fail it or the verify.
/// Greedily split single-argument points (already sorted ascending by `x`) into
/// maximal runs on which one affine `c0 + c1·x` fits every point. Each run opens
/// by solving the affine on its first two points and extends while subsequent
/// points satisfy it; the next run begins at the first point that does not.
/// Returns `(coeffs, first_index, last_index)` per run, or None if a run is too
/// short to fit (it never can be for n ≥ 2 since two points always determine a
/// line, but the signature keeps the call site uniform).
fn segment_1arg(xs: &[i64], ys: &[i64]) -> Option<Vec<(Vec<i64>, usize, usize)>> {
    let n = xs.len();
    let mut segs: Vec<(Vec<i64>, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 2 > n {
            // One trailing point: it must lie on the current run, else the data
            // is not a clean piecewise-affine of this shape.
            let (coeffs, _, end) = segs.last_mut()?;
            if affine_predicts(coeffs, &[xs[n - 1]], ys[n - 1]) {
                *end = n - 1;
                return Some(segs);
            }
            return None;
        }
        let win_x = vec![vec![xs[i]], vec![xs[i + 1]]];
        let win_y = vec![ys[i], ys[i + 1]];
        let coeffs = solve_affine(&win_x, &win_y, 1)?;
        let mut j = i + 2;
        while j < n && affine_predicts(&coeffs, &[xs[j]], ys[j]) {
            j += 1;
        }
        segs.push((coeffs, i, j - 1));
        i = j;
    }
    Some(segs)
}

/// Single-argument closed-interval branch: `if lo <= x && x <= hi { A(x) } else
/// { B(x) }` — a value INSIDE a band picks one affine, outside picks another.
/// This is the first solver to emit the logical `&&` the language just gained,
/// and it owns 1-argument interval membership (the scalar single/two-branch
/// solvers do one-sided cuts; the multi-arg piecewise solver is gated to ≥ 2
/// args), so nothing else recovers this shape.
///
/// INVERSION (deterministic — no first-verified-wins ambiguity, which is what
/// made an earlier draft overfit): sort the points by `x` and segment them into
/// maximal affine runs. The interval shape is exactly THREE runs whose two outer
/// runs are the SAME affine and whose middle run differs — `B | A | B`. The
/// bounds are then the middle run's own `x`-range, refined to the integer
/// boundary where the two pieces meet when they are continuous (so an unsampled
/// gap between runs is placed correctly rather than at the last sample). The
/// reconstructed program is checked against EVERY example by `verified_result`.
///
/// HONESTY GUARDS: exactly three runs, each over-determined (≥ 3 points), outer
/// runs identical, middle run distinct; otherwise refuse.
pub(super) fn search_interval_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if arity != 1 || examples.iter().any(|row| row.len() != 1) {
        return None;
    }
    let targets: Vec<i64> = problem.examples.iter().map(|e| e.expected_int()).collect();
    let n = examples.len();
    if targets.len() != n || n < 9 {
        return None; // need three runs of at least three points each
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| examples[i][0]);
    let xs: Vec<i64> = order.iter().map(|&i| examples[i][0]).collect();
    let ys: Vec<i64> = order.iter().map(|&i| targets[i]).collect();

    let segs = segment_1arg(&xs, &ys)?;
    if segs.len() != 3 {
        return None;
    }
    let (outer_l, mid, outer_r) = (&segs[0], &segs[1], &segs[2]);
    let run_len = |s: &(Vec<i64>, usize, usize)| s.2 - s.1 + 1;
    if run_len(outer_l) < 3 || run_len(mid) < 3 || run_len(outer_r) < 3 {
        return None;
    }
    if outer_l.0 != outer_r.0 || mid.0 == outer_l.0 {
        return None; // not the B|A|B interval shape
    }
    let (a_coeffs, b_coeffs) = (&mid.0, &outer_l.0); // A = inside, B = outside

    // Integer boundary where the two lines meet: A(x) == B(x) at
    // x = (b0 - a0) / (a1 - b1). Use it when it divides evenly and lands in the
    // gap before the middle run (continuous interval); otherwise fall back to the
    // tightest in-data bound (the middle run's own endpoints).
    let intersect = |left: bool| -> i64 {
        let (a0, a1, b0, b1) = (a_coeffs[0], a_coeffs[1], b_coeffs[0], b_coeffs[1]);
        if a1 != b1 {
            let num = b0 - a0;
            let den = a1 - b1;
            if num % den == 0 {
                return num / den;
            }
        }
        if left {
            xs[mid.1]
        } else {
            xs[mid.2]
        }
    };
    // Left boundary: the smaller of (intersection, first middle x); right
    // boundary: the larger of (intersection-ish, last middle x). Keep it simple
    // and exact-on-data: lo = first middle x, hi = last middle x, but if the
    // continuous intersection sits just left of lo (in the gap), use it.
    let mut lo = xs[mid.1];
    let mut hi = xs[mid.2];
    let xi = intersect(true);
    if xi < lo && xi > xs[outer_l.2] {
        lo = xi;
    }
    let xj = intersect(false);
    if xj > hi && xj < xs[outer_r.1] {
        hi = xj;
    }

    let param_names = scalar_param_names(1);
    let params = scalar_params_decl(&param_names);
    let v = &param_names[0];
    let a_body = render_scalar_expr(&affine_expr(a_coeffs), &param_names);
    let b_body = render_scalar_expr(&affine_expr(b_coeffs), &param_names);
    let cond = format!("{v} >= {lo} && {v} <= {hi}");
    let code = format!(
        "fn {fn_name}({params}) -> i64 {{\n    if {cond} {{\n        return {a_body};\n    }}\n    return {b_body};\n}}\n"
    );
    verified_result(problem, code, "search_interval_branch")
}

/// Single-argument rational floor-division: `f(x) = (a·x + b) / d` (integer,
/// truncating) for a constant divisor `d ≥ 2` and non-negative `a, b`. The affine
/// lives INSIDE the division, which is exactly what no other solver can express:
/// `search_affine` is a plain line, and `search_composed_features`' `x / m`
/// feature divides the raw argument (`c·(x / m)`), never an affine of it. Real
/// rules of this shape are averages and bucketed rates — `(3x + 1) / 2`,
/// `(x + 5) / 3`, "every d units costs one more".
///
/// INVERSION: for each divisor `d`, the numerator's slope `a` is within a small
/// window of `d · (Δy / Δx)` (the observed step), so only a handful of `(d, a)`
/// pairs are tried. For a fixed `(d, a)`, `b` is forced: every example needs
/// `d·y ≤ a·x + b < d·y + d`, i.e. `b ≥ max_x(d·y − a·x)` and `b < min_x(…) + d`;
/// that interval is non-empty exactly when the spread of `d·y − a·x` is `< d`, and
/// then `b = max_x(d·y − a·x)`. HONESTY GUARDS: `a, b ≥ 0` and `x ≥ 0` (so Mog's
/// truncating `/` equals floor and the program is correct on unseen inputs, not
/// just where the samples happened to be non-negative); the division must be
/// genuinely lossy on some example (otherwise it is exact and belongs to
/// `search_affine`); and the whole program is re-checked by `verified_result`.
pub(super) fn search_rational_floor(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if arity != 1 || examples.iter().any(|row| row.len() != 1) {
        return None;
    }
    let targets: Vec<i64> = problem.examples.iter().map(|e| e.expected_int()).collect();
    let n = examples.len();
    if targets.len() != n || n < 6 {
        return None;
    }
    if examples.iter().any(|row| row[0] < 0) {
        return None; // keep truncating `/` == floor: only defined for x >= 0 here
    }
    // If a plain affine already reproduces every example, this rule is
    // `search_affine`'s (the minimal form) — refuse, even though a lossy floor
    // like `(3x + 7) / 3 == x + 2` could also represent it. This family only
    // claims genuinely non-affine floors (`(3x + 1) / 2`, whose first differences
    // are not constant).
    if let Some(c) = solve_affine(&examples, &targets, 1) {
        if examples
            .iter()
            .zip(targets.iter())
            .all(|(r, &y)| affine_predicts(&c, r, y))
        {
            return None;
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by_key(|&i| examples[i][0]);
    let (xmin, ymin) = (examples[idx[0]][0], targets[idx[0]]);
    let (xmax, ymax) = (examples[idx[n - 1]][0], targets[idx[n - 1]]);
    if xmax == xmin {
        return None;
    }
    let param_names = scalar_param_names(1);
    let params = scalar_params_decl(&param_names);

    for d in 2..=32i64 {
        let approx = (d as i128 * (ymax - ymin) as i128 / (xmax - xmin) as i128) as i64;
        for a in (approx - 2).max(0)..=(approx + 2).max(0) {
            // b is forced by the floor inequalities: b = max_x(d*y - a*x), valid
            // only when the spread of (d*y - a*x) is strictly less than d.
            let mut hmin = i128::MAX;
            let mut hmax = i128::MIN;
            for (row, &y) in examples.iter().zip(targets.iter()) {
                let h = d as i128 * y as i128 - a as i128 * row[0] as i128;
                hmin = hmin.min(h);
                hmax = hmax.max(h);
            }
            if hmax - hmin >= d as i128 || hmax < 0 {
                continue; // interval empty, or b would be negative
            }
            let Ok(b) = i64::try_from(hmax) else {
                continue;
            };
            // Reject exact division on every example — that is a plain affine,
            // owned by search_affine; this family is for the genuinely lossy floor.
            let lossy = examples.iter().zip(targets.iter()).any(|(row, _)| {
                (a as i128 * row[0] as i128 + b as i128).rem_euclid(d as i128) != 0
            });
            if !lossy {
                continue;
            }
            let num = render_scalar_expr(&affine_expr(&[b, a]), &param_names);
            let code = format!("fn {fn_name}({params}) -> i64 {{\n    return ({num}) / {d};\n}}\n");
            if let Some(result) = verified_result(problem, code, "search_rational_floor") {
                return Some(result);
            }
        }
    }
    None
}

pub(super) fn search_modular_cases(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if !(1..=3).contains(&arity) || examples.iter().any(|row| row.len() != arity) {
        return None;
    }
    let targets: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
    if targets.len() != examples.len() {
        return None;
    }
    let min_class = arity + 2; // over-determined per residue class
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);

    // Moduli to try, smallest-first (simplest case split wins): the natural
    // small bases plus any the data exhibits. Start at 3 — a 2-way split is
    // `search_predicate_branch`'s job and runs before this.
    let mut bases: Vec<i64> = vec![3, 4, 5, 6, 7, 8, 9, 10, 12];
    for &m in &super::scalar_search::mine_scalar_constants(&examples, &targets) {
        if (3..=16).contains(&m) && !bases.contains(&m) {
            bases.push(m);
        }
    }
    bases.sort_unstable();
    bases.dedup();

    for ti in 0..arity {
        'next_base: for &m in &bases {
            // Bucket the examples by residue class.
            let mut classes: Vec<(Vec<Vec<i64>>, Vec<i64>)> =
                (0..m).map(|_| (Vec::new(), Vec::new())).collect();
            for (row, &t) in examples.iter().zip(targets.iter()) {
                let r = row[ti].rem_euclid(m) as usize;
                classes[r].0.push(row.clone());
                classes[r].1.push(t);
            }
            // Every class must be present and over-determined.
            if classes.iter().any(|(xs, _)| xs.len() < min_class) {
                continue;
            }
            // Recover the body of each class via the engine itself (affine
            // first, composed program as fallback) — so a residue class can be
            // non-linear, not only a straight line.
            let mut bodies: Vec<String> = Vec::with_capacity(m as usize);
            for (xs, ys) in &classes {
                match recover_body(xs, ys, arity, &param_names) {
                    Some(b) => bodies.push(b),
                    None => continue 'next_base,
                }
            }
            // Reject the degenerate case where every class renders the same body
            // (that is a plain rule, owned by the non-branching solvers).
            if bodies.iter().all(|b| b == &bodies[0]) {
                continue;
            }
            // Render the `if x%m == r { … }` chain (last class as the fallthrough).
            let mut body = String::new();
            for (r, piece) in bodies.iter().enumerate().take(m as usize - 1) {
                body.push_str(&format!(
                    "    if ({} % {m}) == {r} {{\n        return {piece};\n    }}\n",
                    param_names[ti]
                ));
            }
            let last = bodies.last().unwrap();
            body.push_str(&format!("    return {last};\n"));
            let code = format!("fn {fn_name}({params}) -> i64 {{\n{body}}}\n");
            if let Some(result) = verified_result(problem, code, "search_modular_cases") {
                return Some(result);
            }
        }
    }
    None
}

/// `c0 + Σ c_{i+1}·row[i]` evaluated in i128 (no overflow), as the piecewise
/// value of an affine at a point.
fn affine_value(coeffs: &[i64], row: &[i64]) -> i128 {
    let mut acc = coeffs[0] as i128;
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        acc += c as i128 * row[i - 1] as i128;
    }
    acc
}

/// Value-based branching: recover `f(x) = max(A(x), B(x))` or `min(A(x), B(x))`
/// where A and B are BOTH non-constant affine — the upper/lower envelope of two
/// planes ("take the better of two formulas": `max(2a+b, a+3b)`). Distinct from
/// `search_clamp_affine`, which saturates one affine against a CONSTANT; here
/// both pieces vary, and the winning region is a half-space the data carves out
/// rather than an axis threshold, so the threshold/branch solvers cannot express
/// it.
///
/// INVERSION (the partition — which point is on which piece — is unknown): seed a
/// split from each axis threshold, fit an exact affine to each side, then
/// REASSIGN every example to the piece that wins there (argmax/argmin) and refit,
/// iterating to a fixpoint. Accept only when the reconstructed `max(A,B)` /
/// `min(A,B)` reproduces EVERY example exactly. Each piece must stay over-
/// determined (≥ arity+2 points) and be affine-exact on its own points at every
/// step (`solve_affine` reads off the pivot rows, so the explicit predicts-check
/// is what keeps a mixed assignment from yielding a bogus plane). The final
/// `verified_result` re-runs the emitted program, so a non-converged or
/// coincidental fit is rejected rather than returned.
pub(super) fn search_minmax_affine(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if !(1..=3).contains(&arity) || examples.iter().any(|row| row.len() != arity) {
        return None;
    }
    let targets: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
    if targets.len() != examples.len() {
        return None;
    }
    let n = examples.len();
    let min_side = arity + 2;
    let need = arity + 1; // points that determine one affine piece
    if n < 2 * min_side {
        return None; // too few examples to over-determine two pieces
    }
    let param_names = scalar_param_names(arity);
    let params = scalar_params_decl(&param_names);

    // Given an A-anchor set (arity+1 example indices) and the envelope direction,
    // recover the whole `max(A,B)` / `min(A,B)` if one fits, else None. A is the
    // affine through the anchors; for it to be a piece of a `max` it must lie at
    // or BELOW every target (a lower support), and its equality set is its
    // winning region; B is fit to the complement; the reconstructed envelope is
    // checked on every example and finally re-verified through the runtime.
    let try_anchor = |a_idx: &[usize], is_max: bool| -> Option<SolveResult> {
        let ax: Vec<Vec<i64>> = a_idx.iter().map(|&i| examples[i].clone()).collect();
        let ay: Vec<i64> = a_idx.iter().map(|&i| targets[i]).collect();
        let a = solve_affine(&ax, &ay, arity)?;
        if !ax
            .iter()
            .zip(ay.iter())
            .all(|(r, &y)| affine_predicts(&a, r, y))
        {
            return None; // anchors not affinely independent / not exact
        }
        // A must be a valid support: ≤ every target (max) or ≥ every target (min).
        let supports = examples.iter().zip(targets.iter()).all(|(r, &y)| {
            let v = affine_value(&a, r);
            if is_max {
                v <= y as i128
            } else {
                v >= y as i128
            }
        });
        if !supports {
            return None;
        }
        // A's winning region is where it meets the target; the rest is B's.
        let a_pts: Vec<bool> = examples
            .iter()
            .zip(targets.iter())
            .map(|(r, &y)| affine_value(&a, r) == y as i128)
            .collect();
        let a_count = a_pts.iter().filter(|&&b| b).count();
        if a_count < min_side || n - a_count < min_side {
            return None;
        }
        let (mut bx, mut by) = (Vec::new(), Vec::new());
        for (i, row) in examples.iter().enumerate() {
            if !a_pts[i] {
                bx.push(row.clone());
                by.push(targets[i]);
            }
        }
        let b = solve_affine(&bx, &by, arity)?;
        if a == b
            || !bx
                .iter()
                .zip(by.iter())
                .all(|(r, &y)| affine_predicts(&b, r, y))
        {
            return None;
        }
        // The reconstructed envelope must reproduce EVERY example.
        let ok = examples.iter().zip(targets.iter()).all(|(r, &y)| {
            let av = affine_value(&a, r);
            let bv = affine_value(&b, r);
            let want = if is_max { av.max(bv) } else { av.min(bv) };
            want == y as i128
        });
        if !ok {
            return None;
        }
        let a_body = render_scalar_expr(&affine_expr(&a), &param_names);
        let b_body = render_scalar_expr(&affine_expr(&b), &param_names);
        let cmp = if is_max { ">=" } else { "<=" };
        let code = format!(
            "fn {fn_name}({params}) -> i64 {{\n    if ({a_body}) {cmp} ({b_body}) {{\n        return {a_body};\n    }}\n    return {b_body};\n}}\n"
        );
        verified_result(problem, code, "search_minmax_affine")
    };

    // Enumerate A-anchor sets (arity+1 example indices), capped so the search
    // stays bounded, for both envelope directions. One affine piece is pinned by
    // any arity+1 of its own winning points, so a correct envelope is found as
    // soon as the anchors all land on the same piece.
    const MAX_ATTEMPTS: usize = 20_000;
    let mut attempts = 0usize;
    let mut idx = vec![0usize; need];
    // Iterative odometer over increasing index combinations of `need` out of n.
    for i in 0..need {
        idx[i] = i;
    }
    loop {
        for &is_max in &[true, false] {
            attempts += 1;
            if let Some(result) = try_anchor(&idx, is_max) {
                return Some(result);
            }
        }
        if attempts >= MAX_ATTEMPTS {
            break;
        }
        // advance the combination
        let mut p = need;
        while p > 0 {
            p -= 1;
            if idx[p] != p + n - need {
                idx[p] += 1;
                for q in (p + 1)..need {
                    idx[q] = idx[q - 1] + 1;
                }
                break;
            }
            if p == 0 {
                return None; // exhausted all combinations
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
                .map(|((a, b), y)| Example {
                    inputs: vec![Value::Int(*a), Value::Int(*b)],
                    expected: Value::Int(*y),
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

    fn p1(rows: &[(i64, i64)]) -> Problem {
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: "fn f(x: i64) -> i64",
            examples: rows
                .iter()
                .map(|&(x, y)| Example {
                    inputs: vec![Value::Int(x)],
                    expected: Value::Int(y),
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

    // A closed-interval branch `if 5 <= x <= 12 { 2x } else { x + 100 }` — emits
    // the `&&` operator, recovered by 3-run segmentation, exact on unseen points
    // inside and on both sides of the band.
    #[test]
    fn interval_branch_recovers_membership() {
        let f = |x: i64| {
            if (5..=12).contains(&x) {
                2 * x
            } else {
                x + 100
            }
        };
        let rows: Vec<(i64, i64)> = (0..22).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_interval_branch(&p, "f").expect("must recover the interval branch");
        assert!(
            r.code.contains("&&"),
            "expected a range condition: {}",
            r.code
        );
        let check = p1(&[(2, f(2)), (6, f(6)), (12, f(12)), (13, f(13)), (40, f(40))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("interval branch must be exact on unseen points");
    }

    // A rational floor rule `(3x + 1) / 2` — the affine is INSIDE the division,
    // which no affine/composition solver can express — recovered and exact on
    // unseen points.
    #[test]
    fn rational_floor_recovers_affine_over_d() {
        let f = |x: i64| (3 * x + 1) / 2;
        let rows: Vec<(i64, i64)> = (0..14).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_rational_floor(&p, "f").expect("must recover (3x+1)/2");
        assert!(r.code.contains('/'), "expected a division: {}", r.code);
        let check = p1(&[(20, f(20)), (33, f(33)), (50, f(50)), (101, f(101))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("rational floor must be exact on unseen points");
    }

    // It refuses an EXACT-division rule `(2x + 4) / 2 == x + 2` — that is a plain
    // affine, owned by search_affine, so the floor family must not claim it.
    #[test]
    fn rational_floor_refuses_exact_division() {
        let f = |x: i64| (2 * x + 4) / 2; // == x + 2 exactly, never lossy
        let rows: Vec<(i64, i64)> = (0..14).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        assert!(
            search_rational_floor(&p, "f").is_none(),
            "exact division is plain affine, not a lossy floor"
        );
    }

    // It refuses a single one-sided threshold (only two runs, not three) — that
    // belongs to the scalar branch solvers, not the interval solver.
    #[test]
    fn interval_branch_refuses_single_threshold() {
        let f = |x: i64| if x <= 10 { 2 * x } else { x + 5 };
        let rows: Vec<(i64, i64)> = (0..20).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        assert!(
            search_interval_branch(&p, "f").is_none(),
            "two-run threshold is not an interval"
        );
    }

    // A parity branch `if x even { 2x } else { 3x + 1 }` — the collatz-style
    // rule no affine or threshold solver can express — is recovered via the
    // `x % 2 == 0` predicate and is correct on unseen points.
    #[test]
    fn predicate_branch_recovers_parity_split() {
        let f = |x: i64| if x % 2 == 0 { 2 * x } else { 3 * x + 1 };
        let rows: Vec<(i64, i64)> = (0..16).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_predicate_branch(&p, "f").expect("must recover the parity branch");
        assert!(
            r.code.contains('%'),
            "expected a modular condition: {}",
            r.code
        );
        let check = p1(&[(20, f(20)), (21, f(21)), (50, f(50)), (99, f(99))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("parity branch must be exact on unseen points");
    }

    // A mod-3 residue branch over two arguments: `if a % 3 == 1 { a + 2b }
    // else { 2a - b }`. Recovered and exact on unseen points.
    #[test]
    fn predicate_branch_recovers_mod3_two_arg() {
        let f = |a: i64, b: i64| {
            if a.rem_euclid(3) == 1 {
                a + 2 * b
            } else {
                2 * a - b
            }
        };
        let raw = [
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 1),
            (5, 5),
            (6, 2),
            (7, 3),
            (9, 0),
            (10, 6),
            (12, 1),
            (13, 2),
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_predicate_branch(&p, "f").expect("must recover the mod-3 branch");
        let check = p2(&[
            ((22, 5), f(22, 5)),
            ((31, 9), f(31, 9)),
            ((40, 0), f(40, 0)),
        ]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("mod-3 branch must be exact on unseen points");
    }

    // Recursive branch bodies: `if a even { a·b + 1 } else { 2a - b }` — one
    // branch carries a CROSS-TERM, which a plain affine branch could never
    // express. Recovered (the body is itself a composed program) and exact on
    // unseen points. This is the "think-in-code" recursion: a branch body is a
    // program the engine recovers, not just a straight line.
    #[test]
    fn predicate_branch_recovers_composed_body() {
        let f = |a: i64, b: i64| if a % 2 == 0 { a * b + 1 } else { 2 * a - b };
        let raw = [
            (0, 1),
            (2, 3),
            (4, 0),
            (6, 5),
            (8, 2),
            (10, 4),
            (12, 1),
            (1, 2),
            (3, 4),
            (5, 0),
            (7, 6),
            (9, 1),
            (11, 3),
            (13, 5),
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_predicate_branch(&p, "f").expect("must recover the composed-body branch");
        let check = p2(&[
            ((20, 7), f(20, 7)),
            ((21, 9), f(21, 9)),
            ((14, 0), f(14, 0)),
        ]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("composed-body branch must be exact on unseen points");
    }

    // It refuses noise: targets with no exact two-affine branch over any modular
    // or threshold predicate yield None rather than a coincidental split.
    #[test]
    fn predicate_branch_refuses_noise() {
        let ys = [3i64, 7, 1, 9, 2, 8, 4, 6, 0, 5, 11, 13, 1, 7];
        let rows: Vec<(i64, i64)> = ys.iter().enumerate().map(|(i, &y)| (i as i64, y)).collect();
        let p = p1(&rows);
        assert!(
            search_predicate_branch(&p, "f").is_none(),
            "must refuse data with no exact branch explanation"
        );
    }

    // A two-argument MAX of two affines `max(2a + b, a + 3b)` — neither piece a
    // constant, the boundary not axis-aligned — recovered by partition
    // refinement and exact on unseen points.
    #[test]
    fn minmax_recovers_max_of_two_affine() {
        let f = |a: i64, b: i64| (2 * a + b).max(a + 3 * b);
        let raw = [
            (0, 0),
            (5, 1),
            (1, 5),
            (3, 3),
            (8, 2),
            (2, 8),
            (6, 4),
            (4, 6),
            (10, 0),
            (0, 10),
            (7, 7),
            (9, 1),
            (1, 9),
            (12, 3),
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_minmax_affine(&p, "f").expect("must recover max(2a+b, a+3b)");
        let check = p2(&[
            ((20, 5), f(20, 5)),
            ((5, 20), f(5, 20)),
            ((15, 15), f(15, 15)),
        ]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("max envelope must be exact on unseen points");
    }

    // A two-argument MIN of two affines `min(a + 2b, 3a - b)`, recovered and
    // exact on unseen points.
    #[test]
    fn minmax_recovers_min_of_two_affine() {
        let f = |a: i64, b: i64| (a + 2 * b).min(3 * a - b);
        let raw = [
            (0, 0),
            (5, 1),
            (1, 5),
            (3, 3),
            (8, 2),
            (2, 8),
            (6, 4),
            (4, 6),
            (10, 1),
            (1, 10),
            (7, 7),
            (9, 2),
            (2, 9),
            (11, 4),
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_minmax_affine(&p, "f").expect("must recover min(a+2b, 3a-b)");
        let check = p2(&[
            ((20, 6), f(20, 6)),
            ((6, 20), f(6, 20)),
            ((14, 14), f(14, 14)),
        ]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("min envelope must be exact on unseen points");
    }

    // Full mod-3 case analysis `match x%3 { 0 => 2x, 1 => x+5, 2 => 3x-1 }` —
    // three distinct affines, one per residue class — recovered and exact on
    // unseen points.
    #[test]
    fn modular_cases_recovers_three_way() {
        let f = |x: i64| match x.rem_euclid(3) {
            0 => 2 * x,
            1 => x + 5,
            _ => 3 * x - 1,
        };
        let rows: Vec<(i64, i64)> = (0..24).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_modular_cases(&p, "f").expect("must recover the mod-3 case split");
        assert!(
            r.code.matches("if").count() >= 2,
            "expected a 3-way chain: {}",
            r.code
        );
        let check = p1(&[(30, f(30)), (31, f(31)), (32, f(32)), (100, f(100))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("mod-3 case split must be exact on unseen points");
    }

    // It refuses a plain affine (all residue classes identical): that belongs to
    // search_affine, so modular_cases must NOT claim it.
    #[test]
    fn modular_cases_refuses_plain_affine() {
        let f = |x: i64| 3 * x + 2;
        let rows: Vec<(i64, i64)> = (0..24).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        assert!(
            search_modular_cases(&p, "f").is_none(),
            "modular_cases must refuse a rule with no per-residue difference"
        );
    }

    // A two-argument affine rule `5 + 3a + 2b` is recovered exactly by the
    // integer linear solve and verified correct on unseen points.
    #[test]
    fn affine_recovers_two_arg_linear() {
        let f = |a: i64, b: i64| 3 * a + 2 * b + 5;
        let rows: Vec<((i64, i64), i64)> = [
            (1, 1),
            (2, 1),
            (1, 2),
            (5, 2),
            (0, 0),
            (10, 10),
            (4, 7),
            (8, 3),
        ]
        .iter()
        .map(|&(a, b)| ((a, b), f(a, b)))
        .collect();
        let p = p2(&rows);
        let r = search_affine(&p, "f").expect("affine must solve 3a+2b+5");
        let check = p2(&[
            ((13, 4), f(13, 4)),
            ((99, 50), f(99, 50)),
            ((7, 200), f(7, 200)),
        ]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("affine must be exact on unseen points");
    }

    // A single threshold on one argument with an affine piece on each side is
    // recovered exactly: `if a <= 100 { 3b } else { 2(a-100) + 3b }`.
    #[test]
    fn affine_threshold_recovers_single_breakpoint() {
        let f = |a: i64, b: i64| (if a > 100 { 2 * (a - 100) } else { 0 }) + 3 * b;
        let rows: Vec<((i64, i64), i64)> = [
            (0, 1),
            (50, 2),
            (90, 5),
            (100, 3),
            (101, 4),
            (150, 6),
            (300, 1),
            (500, 9),
            (40, 7),
            (700, 2),
            (95, 8),
            (250, 0),
        ]
        .iter()
        .map(|&(a, b)| ((a, b), f(a, b)))
        .collect();
        let p = p2(&rows);
        let r = search_affine_threshold(&p, "f").expect("threshold-affine must solve the rule");
        assert!(r.code.contains("if"), "expected a branch: {}", r.code);
        let check = p2(&[
            ((110, 3), f(110, 3)),
            ((1000, 4), f(1000, 4)),
            ((10, 10), f(10, 10)),
        ]);
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
            (10, 1),
            (40, 5),
            (90, 2),
            (100, 7), // tier0 (4)
            (150, 3),
            (200, 9),
            (300, 1),
            (450, 6),
            (500, 4), // tier1 (5)
            (600, 2),
            (800, 8),
            (1200, 5),
            (1500, 0), // tier2 (4)
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_affine_piecewise(&p, "f").expect("must solve the 3-tier multi-arg rule");
        assert!(
            r.code.matches("if").count() >= 2,
            "expected 3 pieces: {}",
            r.code
        );
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
        let rows: Vec<((i64, i64), i64)> = (1..14).map(|i| ((i, i + 1), i * (i + 1))).collect();
        let p = p2(&rows);
        assert!(
            search_affine(&p, "f").is_none(),
            "affine must refuse a product"
        );
    }

    // A curved single-argument rule `2x² − 3x + 5` is recovered exactly by the
    // monomial-feature solve and verified correct on UNSEEN points — proving
    // generalization, not fit. (The affine solvers refuse this curvature.)
    #[test]
    fn polynomial_recovers_one_arg_quadratic() {
        let f = |x: i64| 2 * x * x - 3 * x + 5;
        let rows: Vec<(i64, i64)> = [0, 1, 2, 3, 4, 5, 7, 10]
            .iter()
            .map(|&x| (x, f(x)))
            .collect();
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
        let rows: Vec<((i64, i64), i64)> = [(0, 0), (1, 1), (2, 3), (3, 5), (4, 2), (5, 7), (7, 1)]
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
        let rows: Vec<((i64, i64), i64)> = (1..14).map(|i| ((i, i + 1), i * (i + 1))).collect();
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
        let rows: Vec<(i64, i64)> = [0, 2, 4, 6, 8, 10, 14, 20, 30]
            .iter()
            .map(|&x| (x, f(x)))
            .collect();
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
            (0, 0),
            (1, 1),
            (5, 2),
            (10, 10),
            (20, 5),
            (3, 3),
            (8, 8), // below cap
            (60, 10),
            (90, 40),
            (100, 0),
            (200, 100), // at/above cap (clamped to 500)
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_clamp_affine(&p, "f").expect("must recover min(500, 10a+3b)");
        let check = p2(&[
            ((4, 4), f(4, 4)),
            ((49, 3), f(49, 3)),
            ((300, 7), f(300, 7)),
        ]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("cap must be exact on unseen points");
    }

    // It refuses data that is not a clamped line (a true product a·b): no
    // constant floor/cap fits, so it returns None rather than a wrong saturation.
    #[test]
    fn clamp_refuses_nonclamp() {
        let rows: Vec<((i64, i64), i64)> = (1..14).map(|i| ((i, i + 1), i * (i + 1))).collect();
        let p = p2(&rows);
        assert!(
            search_clamp_affine(&p, "f").is_none(),
            "clamp must refuse a non-saturated nonlinear rule"
        );
    }

    // A true CROSS-TERM rule `a·b + 2a + 3` — which the separable polynomial
    // solver deliberately refuses — is recovered exactly by composition (the
    // `a·b` feature) and is correct on unseen points.
    #[test]
    fn composed_recovers_cross_term() {
        let f = |a: i64, b: i64| a * b + 2 * a + 3;
        let raw = [
            (0, 0),
            (1, 1),
            (2, 3),
            (3, 2),
            (4, 5),
            (5, 1),
            (2, 7),
            (6, 0),
            (1, 9),
            (8, 4),
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_composed_features(&p, "f").expect("must recover a·b + 2a + 3");
        let check = p2(&[
            ((11, 4), f(11, 4)),
            ((20, 7), f(20, 7)),
            ((3, 50), f(3, 50)),
        ]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("cross-term must be exact on unseen points");
    }

    // A modular rule `3·(x % 7) + 1` is recovered via the mined `x % 7` feature
    // and is correct on unseen points (it is periodic, so plain affine cannot fit
    // it — proving composition recovers genuine non-affine structure).
    #[test]
    fn composed_recovers_modular() {
        let f = |x: i64| 3 * (x % 7) + 1;
        let rows: Vec<(i64, i64)> = (0..14).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_composed_features(&p, "f").expect("must recover 3*(x%7) + 1");
        let check = p1(&[(20, f(20)), (35, f(35)), (48, f(48)), (100, f(100))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("modular rule must be exact on unseen points");
    }

    // It refuses noise: random targets with no exact composed-feature explanation
    // yield None rather than a coincidental over-parameterised fit.
    #[test]
    fn composed_refuses_noise() {
        let ys = [7i64, 2, 9, 1, 5, 8, 3, 6, 4, 0, 11, 13];
        let rows: Vec<((i64, i64), i64)> = ys
            .iter()
            .enumerate()
            .map(|(i, &y)| ((i as i64, (i * 2 + 1) as i64), y))
            .collect();
        let p = p2(&rows);
        assert!(
            search_composed_features(&p, "f").is_none(),
            "composition must refuse data with no exact feature explanation"
        );
    }

    // A pure-linear single-argument rule `3x + 1` (all quadratic coeffs zero) is
    // still solved at arity 1, because search_affine is gated to 2–3 args and so
    // this legitimately lands in the polynomial solver's arity==1 branch.
    #[test]
    fn polynomial_one_arg_affine_still_lands() {
        let f = |x: i64| 3 * x + 1;
        let rows: Vec<(i64, i64)> = [0, 1, 2, 3, 4, 5, 7].iter().map(|&x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_polynomial_multi(&p, "f").expect("must solve 3x + 1 at arity 1");
        let check = p1(&[(13, f(13)), (88, f(88))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("linear arity-1 must be exact on unseen points");
    }
}
