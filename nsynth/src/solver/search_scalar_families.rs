use super::scalar_search::{
    build_deep_expr_candidates, code_scalar_return_expr, code_scalar_single_branch,
    code_scalar_two_branch, code_unary_range_loop, cond_is_total, cond_selection,
    cond_selection_on_mask, expr_matches_subset, expr_matches_target, extract_scalar_examples,
    mine_scalar_constants, render_scalar_expr, scalar_expr_complexity, scalar_search_context,
    score_single_branch_candidate, score_two_branch_candidate, simulate_unary_range_loop,
    RangeAccumOp, RangeLoopCmp, RangeLoopTerm, ScalarBinOp, ScalarExpr,
};
use super::search_codegen::{code_quadratic_search, verified_result};
use super::signature::{scalar_param_names, scalar_params_decl};
use super::*;

/// One affine segment `y = a·x + b` supported by `points` colinear training
/// points; `x_last` is the largest input it covers (used to place a breakpoint
/// when two segments share a slope but differ in offset).
struct AffineSeg {
    a: i64,
    b: i64,
    points: usize,
    x_last: i64,
}

/// Greedily split sorted `(x, y)` points into maximal exact-affine runs. A
/// genuine piecewise-affine rule (tiered pricing, caps, clamps) yields a few
/// runs each supported by several colinear points; a curve (quadratic, modulo,
/// a loop) yields many two-point runs and is rejected by the caller's guards.
/// Returns `None` on a non-integer slope or arithmetic overflow.
fn segment_affine(pts: &[(i64, i64)]) -> Option<Vec<AffineSeg>> {
    let n = pts.len();
    let mut segs: Vec<AffineSeg> = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 1 >= n {
            // A lone trailing point only extends the previous run if it lies on
            // it; otherwise the last tier is unsupported and we refuse (honest
            // failure beats guessing a one-point segment that won't generalize).
            let (x, y) = pts[i];
            if let Some(last) = segs.last_mut() {
                if last.a.checked_mul(x)?.checked_add(last.b)? == y {
                    last.points += 1;
                    last.x_last = x;
                    i += 1;
                    continue;
                }
            }
            return None;
        }
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[i + 1];
        let dx = x1 - x0;
        if dx == 0 {
            return None;
        }
        if (y1 - y0).rem_euclid(dx.abs()) != 0 {
            return None;
        }
        let a = (y1 - y0) / dx;
        let b = y0.checked_sub(a.checked_mul(x0)?)?;
        let mut j = i + 1;
        while j + 1 < n {
            let (xn, yn) = pts[j + 1];
            if a.checked_mul(xn).and_then(|v| v.checked_add(b)) == Some(yn) {
                j += 1;
            } else {
                break;
            }
        }
        segs.push(AffineSeg {
            a,
            b,
            points: j - i + 1,
            x_last: pts[j].0,
        });
        i = j + 1;
    }
    Some(segs)
}

/// Breakpoints between consecutive segments. For differing slopes the breakpoint
/// is the integer `x` where the two affine pieces intersect — the *true*
/// threshold of a continuous tiered rule, recovered exactly rather than guessed.
/// For equal slopes (a discontinuous step) it is the last input of the left
/// segment.
fn piecewise_breakpoints(segs: &[AffineSeg]) -> Vec<i64> {
    let mut bps = Vec::with_capacity(segs.len().saturating_sub(1));
    for k in 0..segs.len().saturating_sub(1) {
        let s0 = &segs[k];
        let s1 = &segs[k + 1];
        let t = if s0.a != s1.a {
            let num = s1.b - s0.b;
            let den = s0.a - s1.a;
            if num % den == 0 {
                num / den
            } else {
                s0.x_last
            }
        } else {
            s0.x_last
        };
        bps.push(t);
    }
    bps
}

fn affine_expr(a: i64, b: i64) -> ScalarExpr {
    if a == 0 {
        return ScalarExpr::Const(b);
    }
    let term = if a == 1 {
        ScalarExpr::Var(0)
    } else {
        ScalarExpr::Bin(
            Box::new(ScalarExpr::Const(a)),
            ScalarBinOp::Mul,
            Box::new(ScalarExpr::Var(0)),
        )
    };
    if b == 0 {
        term
    } else {
        ScalarExpr::Bin(
            Box::new(term),
            ScalarBinOp::Add,
            Box::new(ScalarExpr::Const(b)),
        )
    }
}

fn code_piecewise(
    fn_name: &str,
    param_names: &[String],
    segs: &[AffineSeg],
    bps: &[i64],
    use_le: bool,
) -> String {
    let params = scalar_params_decl(param_names);
    let x = &param_names[0];
    let op = if use_le { "<=" } else { "<" };
    let mut body = String::new();
    for (k, bp) in bps.iter().enumerate() {
        let piece = render_scalar_expr(&affine_expr(segs[k].a, segs[k].b), param_names);
        body.push_str(&format!(
            "    if {x} {op} {bp} {{\n        return {piece};\n    }}\n"
        ));
    }
    let last = segs.last().unwrap();
    let last_piece = render_scalar_expr(&affine_expr(last.a, last.b), param_names);
    body.push_str(&format!("    return {last_piece};\n"));
    format!("fn {fn_name}({params}) -> i64 {{\n{body}}}\n")
}

/// Exact piecewise-affine synthesizer for a single integer argument. Recovers
/// tiered/threshold/clamp rules of any number of segments by reading the
/// breakpoints straight out of the data (where the slope changes) and placing
/// each threshold at the intersection of the adjoining pieces — so the program
/// generalizes by construction, not by fitting. It only commits when the data
/// is *confidently* piecewise-affine (2–6 segments, each backed by ≥2 colinear
/// points), so curves and loops fall through to the other solvers instead of
/// being faked with a per-point staircase.
pub(super) fn search_piecewise_affine(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = extract_scalar_examples(problem)?;
    if examples.first().map(Vec::len) != Some(1) {
        return None;
    }
    let targets: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
    let mut pts: Vec<(i64, i64)> = examples
        .iter()
        .zip(targets.iter())
        .map(|(row, &t)| (row[0], t))
        .collect();
    pts.sort_unstable();
    pts.dedup();
    if pts.len() < 4 {
        return None;
    }
    let segs = segment_affine(&pts)?;
    if segs.len() < 2 || segs.len() > 6 || segs.iter().any(|s| s.points < 2) {
        return None;
    }
    let bps = piecewise_breakpoints(&segs);
    let param_names = scalar_param_names(1);
    for use_le in [true, false] {
        let code = code_piecewise(fn_name, &param_names, &segs, &bps, use_le);
        if let Some(result) = verified_result(problem, code, "search_piecewise_affine") {
            return Some(result);
        }
    }
    None
}

pub(super) fn search_scalar_expr(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    let param_names = scalar_param_names(arity);
    let target: Vec<i64> = problem
        .examples
        .iter()
        .map(|ex| ex.expected_int())
        .collect();

    let constants = mine_scalar_constants(&examples, &target);
    let mut candidates = build_deep_expr_candidates(arity, &examples, &constants);
    candidates.sort_by_key(|c| {
        (
            scalar_expr_complexity(&c.expr),
            render_scalar_expr(&c.expr, &param_names),
        )
    });

    let candidate = candidates
        .iter()
        .find(|c| expr_matches_target(&c.outputs, &target))?;
    let code = code_scalar_return_expr(fn_name, &param_names, &candidate.expr);
    verified_result(problem, code, "search_scalar_expr")
}

pub(super) fn search_unary_range_loop(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = extract_scalar_examples(problem)?;
    if examples.first()?.len() != 1 {
        return None;
    }

    let candidates = [
        (
            0,
            1,
            RangeLoopCmp::Le,
            RangeAccumOp::Add,
            RangeLoopTerm::Index,
        ),
        (
            0,
            1,
            RangeLoopCmp::Le,
            RangeAccumOp::Add,
            RangeLoopTerm::IndexSquared,
        ),
        (
            1,
            1,
            RangeLoopCmp::Le,
            RangeAccumOp::Mul,
            RangeLoopTerm::Index,
        ),
        (
            1,
            1,
            RangeLoopCmp::Le,
            RangeAccumOp::Mul,
            RangeLoopTerm::IndexSquared,
        ),
        (
            0,
            0,
            RangeLoopCmp::Lt,
            RangeAccumOp::Add,
            RangeLoopTerm::Index,
        ),
        (
            0,
            0,
            RangeLoopCmp::Lt,
            RangeAccumOp::Add,
            RangeLoopTerm::IndexSquared,
        ),
        (
            1,
            0,
            RangeLoopCmp::Lt,
            RangeAccumOp::Mul,
            RangeLoopTerm::Index,
        ),
        (
            1,
            0,
            RangeLoopCmp::Lt,
            RangeAccumOp::Mul,
            RangeLoopTerm::IndexSquared,
        ),
    ];

    for (init, start, cmp, op, term) in candidates {
        let matches = problem
            .examples
            .iter()
            .zip(examples.iter())
            .all(|(example, args)| {
                simulate_unary_range_loop(args[0], init, start, cmp, op, term)
                    == Some(example.expected_int())
            });
        if !matches {
            continue;
        }
        let code = code_unary_range_loop(fn_name, init, start, cmp, op, term);
        if let Some(result) = verified_result(problem, code, "search_unary_range_loop") {
            return Some(result);
        }
    }

    None
}

pub(super) fn search_polynomial_quadratic(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = extract_scalar_examples(problem)?;
    if examples.first()?.len() != 1 {
        return None;
    }

    for a in -5..=5 {
        for b in -5..=5 {
            for c in -10..=10 {
                let matches =
                    problem
                        .examples
                        .iter()
                        .zip(examples.iter())
                        .all(|(example, args)| {
                            let x = args[0];
                            a * x * x + b * x + c == example.expected_int()
                        });
                if !matches {
                    continue;
                }
                let code = code_quadratic_search(fn_name, a, b, c);
                if let Some(result) = verified_result(problem, code, "search_polynomial_quadratic")
                {
                    return Some(result);
                }
            }
        }
    }
    None
}

pub(super) fn search_single_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let ctx = scalar_search_context(problem)?;
    let mut best: Option<((usize, usize, usize, usize, usize, String), SolveResult)> = None;

    for cond in &ctx.cond_candidates {
        if !cond_is_total(&cond.outputs) {
            continue;
        }
        let Some(true_mask) = cond_selection(&cond.outputs, true) else {
            continue;
        };
        let Some(false_mask) = cond_selection(&cond.outputs, false) else {
            continue;
        };
        let Some(then_expr) = ctx
            .branch_expr_candidates
            .iter()
            .find(|candidate| expr_matches_subset(&candidate.outputs, &ctx.target, &true_mask))
        else {
            continue;
        };
        let Some(else_expr) = ctx
            .branch_expr_candidates
            .iter()
            .find(|candidate| expr_matches_subset(&candidate.outputs, &ctx.target, &false_mask))
        else {
            continue;
        };
        // A branch whose two arms are the same expression is not really a
        // branch — it's a single expression dressed up with a vacuous guard.
        // Skip it so a clean unconditional form (search_scalar_expr) wins
        // instead of emitting `if c { e } else { e }`.
        if then_expr.expr == else_expr.expr {
            continue;
        }
        let code = code_scalar_single_branch(
            fn_name,
            &ctx.param_names,
            cond,
            &then_expr.expr,
            &else_expr.expr,
        );
        if let Some(result) = verified_result(problem, code, "search_single_branch") {
            let score = score_single_branch_candidate(
                &ctx.param_names,
                cond,
                &then_expr.expr,
                &else_expr.expr,
            );
            let replace = best
                .as_ref()
                .map(|(best_score, _)| score < *best_score)
                .unwrap_or(true);
            if replace {
                best = Some((score, result));
            }
        }
    }

    best.map(|(_, result)| result)
}

pub(super) fn search_two_branch(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let ctx = scalar_search_context(problem)?;
    let mut best: Option<((usize, usize, usize, usize, String), SolveResult)> = None;

    for first_cond in &ctx.cond_candidates {
        if !cond_is_total(&first_cond.outputs) {
            continue;
        }
        let Some(first_true_mask) = cond_selection(&first_cond.outputs, true) else {
            continue;
        };
        let Some(first_false_mask) = cond_selection(&first_cond.outputs, false) else {
            continue;
        };
        let Some(first_expr) = ctx.branch_expr_candidates.iter().find(|candidate| {
            expr_matches_subset(&candidate.outputs, &ctx.target, &first_true_mask)
        }) else {
            continue;
        };

        for second_cond in &ctx.cond_candidates {
            let Some(second_true_mask) =
                cond_selection_on_mask(&second_cond.outputs, &first_false_mask, true)
            else {
                continue;
            };
            let Some(second_false_mask) =
                cond_selection_on_mask(&second_cond.outputs, &first_false_mask, false)
            else {
                continue;
            };
            let Some(second_expr) = ctx.branch_expr_candidates.iter().find(|candidate| {
                expr_matches_subset(&candidate.outputs, &ctx.target, &second_true_mask)
            }) else {
                continue;
            };
            let Some(else_expr) = ctx.branch_expr_candidates.iter().find(|candidate| {
                expr_matches_subset(&candidate.outputs, &ctx.target, &second_false_mask)
            }) else {
                continue;
            };
            let code = code_scalar_two_branch(
                fn_name,
                &ctx.param_names,
                first_cond,
                &first_expr.expr,
                second_cond,
                &second_expr.expr,
                &else_expr.expr,
            );
            if let Some(result) = verified_result(problem, code, "search_two_branch") {
                let score = score_two_branch_candidate(
                    &ctx.param_names,
                    first_cond,
                    &first_expr.expr,
                    second_cond,
                    &second_expr.expr,
                    &else_expr.expr,
                );
                let replace = best
                    .as_ref()
                    .map(|(best_score, _)| score < *best_score)
                    .unwrap_or(true);
                if replace {
                    best = Some((score, result));
                }
            }
        }
    }

    best.map(|(_, result)| result)
}
