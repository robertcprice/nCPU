use super::scalar_search::{
    build_deep_expr_candidates, code_scalar_return_expr, code_scalar_single_branch,
    code_scalar_two_branch, code_unary_range_loop, cond_is_total, cond_selection,
    cond_selection_on_mask, expr_matches_subset, expr_matches_target, extract_scalar_examples,
    mine_scalar_constants, render_scalar_expr, scalar_expr_complexity, scalar_search_context,
    score_single_branch_candidate, score_two_branch_candidate, simulate_unary_range_loop,
    RangeAccumOp, RangeLoopCmp, RangeLoopTerm,
};
use super::search_codegen::{code_quadratic_search, verified_result};
use super::*;

pub(super) fn search_scalar_expr(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let examples = extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    let param_names = scalar_param_names(arity);
    let target: Vec<i64> = problem.examples.iter().map(|ex| ex.expected).collect();

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
                    == Some(example.expected)
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
                            a * x * x + b * x + c == example.expected
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
