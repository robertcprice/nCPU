use std::collections::HashMap;

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum ScalarBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum RangeLoopCmp {
    Lt,
    Le,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum RangeAccumOp {
    Add,
    Mul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum RangeLoopTerm {
    Index,
    IndexSquared,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum ScalarExpr {
    Var(usize),
    Const(i64),
    Bin(Box<ScalarExpr>, ScalarBinOp, Box<ScalarExpr>),
}

#[derive(Clone, Debug)]
pub(super) struct ExprCandidate {
    pub(super) expr: ScalarExpr,
    pub(super) outputs: Vec<Option<i64>>,
}

#[derive(Clone, Debug)]
pub(super) struct ConditionCandidate {
    pub(super) lhs: ScalarExpr,
    pub(super) op: CompareOp,
    pub(super) rhs: ScalarExpr,
    pub(super) outputs: Vec<Option<bool>>,
}

#[derive(Clone, Debug)]
pub(super) struct ScalarSearchContext {
    pub(super) param_names: Vec<String>,
    pub(super) target: Vec<i64>,
    pub(super) cond_candidates: Vec<ConditionCandidate>,
    /// Deeper expression pool used for branch then/else bodies (and any
    /// single-expression match). Replaces the old lean then/else pool so
    /// affine-with-threshold rules are reachable.
    pub(super) branch_expr_candidates: Vec<ExprCandidate>,
}

pub(super) fn render_scalar_expr(expr: &ScalarExpr, param_names: &[String]) -> String {
    match expr {
        ScalarExpr::Var(index) => param_names
            .get(*index)
            .cloned()
            .unwrap_or_else(|| format!("x{index}")),
        ScalarExpr::Const(value) => value.to_string(),
        ScalarExpr::Bin(lhs, op, rhs) => {
            let op = match op {
                ScalarBinOp::Add => "+",
                ScalarBinOp::Sub => "-",
                ScalarBinOp::Mul => "*",
                ScalarBinOp::Div => "/",
                ScalarBinOp::Mod => "%",
            };
            format!(
                "({} {} {})",
                render_scalar_expr(lhs, param_names),
                op,
                render_scalar_expr(rhs, param_names)
            )
        }
    }
}

fn render_compare_op(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "==",
        CompareOp::Ne => "!=",
        CompareOp::Lt => "<",
        CompareOp::Le => "<=",
        CompareOp::Gt => ">",
        CompareOp::Ge => ">=",
    }
}

pub(super) fn scalar_expr_complexity(expr: &ScalarExpr) -> usize {
    match expr {
        ScalarExpr::Var(_) | ScalarExpr::Const(_) => 1,
        ScalarExpr::Bin(lhs, _, rhs) => {
            1 + scalar_expr_complexity(lhs) + scalar_expr_complexity(rhs)
        }
    }
}

/// True if `expr` reads any input variable (vs. being a pure constant fold).
/// Used to forbid dividing or modding *by* a data-dependent value: `k % x`,
/// `expr / x`, etc. are almost never a real scalar rule — they are how branch
/// search overfits a piecewise/tiered function (e.g. faking the api_bill tiers
/// with `x - (-8000 % x)`). Restricting `/` and `%` to a data-independent
/// divisor keeps the legitimate `x % 10` / `x / 2` digit-and-parity forms while
/// removing the overfit class, so genuinely multi-tier data makes single-branch
/// search honestly fail and fall through to two-branch search.
fn scalar_expr_uses_var(expr: &ScalarExpr) -> bool {
    match expr {
        ScalarExpr::Var(_) => true,
        ScalarExpr::Const(_) => false,
        ScalarExpr::Bin(lhs, _, rhs) => scalar_expr_uses_var(lhs) || scalar_expr_uses_var(rhs),
    }
}

fn scalar_expr_collect_used_vars(expr: &ScalarExpr, used: &mut [bool]) {
    match expr {
        ScalarExpr::Var(index) => {
            if let Some(slot) = used.get_mut(*index) {
                *slot = true;
            }
        }
        ScalarExpr::Const(_) => {}
        ScalarExpr::Bin(lhs, _, rhs) => {
            scalar_expr_collect_used_vars(lhs, used);
            scalar_expr_collect_used_vars(rhs, used);
        }
    }
}

fn render_condition(cond: &ConditionCandidate, param_names: &[String]) -> String {
    format!(
        "{} {} {}",
        render_scalar_expr(&cond.lhs, param_names),
        render_compare_op(cond.op),
        render_scalar_expr(&cond.rhs, param_names)
    )
}

/// For a guard of the form `x </<=/>/>= k` (a single variable compared to a
/// constant), the penalty is how far the two arms are from meeting at the
/// boundary value `k`. Real-world piecewise rules are continuous there — the
/// free tier ends exactly where the paid rate begins — so when several guard
/// thresholds fit the training data equally well (which happens when the
/// boundary example lands in the holdout), the *continuous* one is the honest
/// generalization. Returns 0 for guards that aren't a simple variable-vs-
/// constant comparison, or for multi-argument functions.
fn boundary_continuity_penalty(
    cond: &ConditionCandidate,
    then_expr: &ScalarExpr,
    else_expr: &ScalarExpr,
    n_params: usize,
) -> usize {
    if n_params != 1 {
        return 0;
    }
    let k = match (&cond.lhs, &cond.rhs) {
        (ScalarExpr::Var(0), ScalarExpr::Const(k)) => *k,
        (ScalarExpr::Const(k), ScalarExpr::Var(0)) => *k,
        _ => return 0,
    };
    let args = [k];
    match (
        eval_scalar_expr(then_expr, args.as_slice()),
        eval_scalar_expr(else_expr, args.as_slice()),
    ) {
        (Some(a), Some(b)) => usize::try_from(a.abs_diff(b)).unwrap_or(usize::MAX),
        _ => 0,
    }
}

pub(super) fn score_single_branch_candidate(
    param_names: &[String],
    cond: &ConditionCandidate,
    then_expr: &ScalarExpr,
    else_expr: &ScalarExpr,
) -> (usize, usize, usize, usize, usize, String) {
    let mut used = vec![false; param_names.len()];
    scalar_expr_collect_used_vars(&cond.lhs, &mut used);
    scalar_expr_collect_used_vars(&cond.rhs, &mut used);
    scalar_expr_collect_used_vars(then_expr, &mut used);
    scalar_expr_collect_used_vars(else_expr, &mut used);
    let missing_params = used.iter().filter(|value| !**value).count();
    let constant_branches = [then_expr, else_expr]
        .into_iter()
        .filter(|expr| !matches!(expr, ScalarExpr::Var(_) | ScalarExpr::Bin(_, _, _)))
        .count();
    let total_complexity = scalar_expr_complexity(&cond.lhs)
        + scalar_expr_complexity(&cond.rhs)
        + scalar_expr_complexity(then_expr)
        + scalar_expr_complexity(else_expr);
    let branch_complexity = scalar_expr_complexity(then_expr) + scalar_expr_complexity(else_expr);
    let rendered = format!(
        "{} => {} | {}",
        render_condition(cond, param_names),
        render_scalar_expr(then_expr, param_names),
        render_scalar_expr(else_expr, param_names)
    );
    // Occam first: the simplest program that fits is the one most likely to
    // generalize. A flat region of a piecewise rule is legitimately a constant
    // (`else 0`); ranking total complexity ahead of the constant-arm count
    // stops the search from preferring a baroque `(45 * x) % 10` over a plain
    // `0` just because the former is non-constant. Then continuity at the
    // guard boundary (real piecewise rules don't jump there), then the const
    // count, all as tiebreaks among equally-simple candidates.
    let continuity = boundary_continuity_penalty(cond, then_expr, else_expr, param_names.len());
    (
        missing_params,
        total_complexity,
        continuity,
        constant_branches,
        branch_complexity,
        rendered,
    )
}

pub(super) fn score_two_branch_candidate(
    param_names: &[String],
    first_cond: &ConditionCandidate,
    first_expr: &ScalarExpr,
    second_cond: &ConditionCandidate,
    second_expr: &ScalarExpr,
    else_expr: &ScalarExpr,
) -> (usize, usize, usize, usize, String) {
    let mut used = vec![false; param_names.len()];
    scalar_expr_collect_used_vars(&first_cond.lhs, &mut used);
    scalar_expr_collect_used_vars(&first_cond.rhs, &mut used);
    scalar_expr_collect_used_vars(first_expr, &mut used);
    scalar_expr_collect_used_vars(&second_cond.lhs, &mut used);
    scalar_expr_collect_used_vars(&second_cond.rhs, &mut used);
    scalar_expr_collect_used_vars(second_expr, &mut used);
    scalar_expr_collect_used_vars(else_expr, &mut used);
    let missing_params = used.iter().filter(|value| !**value).count();
    let constant_branches = [first_expr, second_expr, else_expr]
        .into_iter()
        .filter(|expr| !matches!(expr, ScalarExpr::Var(_) | ScalarExpr::Bin(_, _, _)))
        .count();
    let total_complexity = scalar_expr_complexity(&first_cond.lhs)
        + scalar_expr_complexity(&first_cond.rhs)
        + scalar_expr_complexity(first_expr)
        + scalar_expr_complexity(&second_cond.lhs)
        + scalar_expr_complexity(&second_cond.rhs)
        + scalar_expr_complexity(second_expr)
        + scalar_expr_complexity(else_expr);
    let branch_complexity = scalar_expr_complexity(first_expr)
        + scalar_expr_complexity(second_expr)
        + scalar_expr_complexity(else_expr);
    let rendered = format!(
        "{} => {} | {} => {} | {}",
        render_condition(first_cond, param_names),
        render_scalar_expr(first_expr, param_names),
        render_condition(second_cond, param_names),
        render_scalar_expr(second_expr, param_names),
        render_scalar_expr(else_expr, param_names)
    );
    // Occam first (see score_single_branch_candidate): simplest-fits-best,
    // const-arm count is a tiebreak only.
    (
        missing_params,
        total_complexity,
        constant_branches,
        branch_complexity,
        rendered,
    )
}

fn eval_scalar_expr(expr: &ScalarExpr, args: &[i64]) -> Option<i64> {
    match expr {
        ScalarExpr::Var(index) => args.get(*index).copied(),
        ScalarExpr::Const(value) => Some(*value),
        ScalarExpr::Bin(lhs, op, rhs) => {
            let lhs = eval_scalar_expr(lhs, args)?;
            let rhs = eval_scalar_expr(rhs, args)?;
            match op {
                ScalarBinOp::Add => lhs.checked_add(rhs),
                ScalarBinOp::Sub => lhs.checked_sub(rhs),
                ScalarBinOp::Mul => lhs.checked_mul(rhs),
                ScalarBinOp::Div => {
                    if rhs == 0 {
                        None
                    } else {
                        Some(lhs / rhs)
                    }
                }
                ScalarBinOp::Mod => {
                    if rhs == 0 {
                        None
                    } else {
                        Some(lhs % rhs)
                    }
                }
            }
        }
    }
}

fn eval_compare(lhs: i64, op: CompareOp, rhs: i64) -> bool {
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::Ne => lhs != rhs,
        CompareOp::Lt => lhs < rhs,
        CompareOp::Le => lhs <= rhs,
        CompareOp::Gt => lhs > rhs,
        CompareOp::Ge => lhs >= rhs,
    }
}

fn render_range_loop_cmp(cmp: RangeLoopCmp) -> &'static str {
    match cmp {
        RangeLoopCmp::Lt => "<",
        RangeLoopCmp::Le => "<=",
    }
}

fn render_range_term(term: RangeLoopTerm) -> &'static str {
    match term {
        RangeLoopTerm::Index => "i",
        RangeLoopTerm::IndexSquared => "(i * i)",
    }
}

fn render_range_accum_op(op: RangeAccumOp) -> &'static str {
    match op {
        RangeAccumOp::Add => "+",
        RangeAccumOp::Mul => "*",
    }
}

fn apply_range_accum(acc: i64, op: RangeAccumOp, term: i64) -> Option<i64> {
    match op {
        RangeAccumOp::Add => acc.checked_add(term),
        RangeAccumOp::Mul => acc.checked_mul(term),
    }
}

fn eval_range_term(i: i64, term: RangeLoopTerm) -> Option<i64> {
    match term {
        RangeLoopTerm::Index => Some(i),
        RangeLoopTerm::IndexSquared => i.checked_mul(i),
    }
}

pub(super) fn simulate_unary_range_loop(
    n: i64,
    init: i64,
    start: i64,
    cmp: RangeLoopCmp,
    op: RangeAccumOp,
    term: RangeLoopTerm,
) -> Option<i64> {
    let mut acc = init;
    let mut i = start;
    let mut steps = 0usize;
    while match cmp {
        RangeLoopCmp::Lt => i < n,
        RangeLoopCmp::Le => i <= n,
    } {
        let term_value = eval_range_term(i, term)?;
        acc = apply_range_accum(acc, op, term_value)?;
        i = i.checked_add(1)?;
        steps += 1;
        if steps > 100_000 {
            return None;
        }
    }
    Some(acc)
}

pub(super) fn code_unary_range_loop(
    fn_name: &str,
    init: i64,
    start: i64,
    cmp: RangeLoopCmp,
    op: RangeAccumOp,
    term: RangeLoopTerm,
) -> String {
    let cmp = render_range_loop_cmp(cmp);
    let op = render_range_accum_op(op);
    let term = render_range_term(term);
    format!(
        "fn {fn_name}(n: i64) -> i64 {{\n    acc: i64 = {init};\n    i: i64 = {start};\n    while i {cmp} n {{\n        acc = acc {op} {term};\n        i = i + 1;\n    }}\n    return acc;\n}}\n"
    )
}

/// Order-independent canonical key for an expression, used to break complexity
/// ties when choosing the representative of an output-signature. Without it the
/// representative would be whichever expression a `HashMap` happened to visit
/// first, which varies with the hash seed and makes synthesis non-deterministic
/// (and lets a bounded-range modulo overfit like `x % 10001` stand in for the
/// simpler, correct `x + 8000` that has the same outputs on the samples).
fn scalar_expr_canon(expr: &ScalarExpr) -> String {
    match expr {
        ScalarExpr::Var(i) => format!("v{i}"),
        ScalarExpr::Const(c) => format!("c{c}"),
        ScalarExpr::Bin(lhs, op, rhs) => {
            let o = match op {
                ScalarBinOp::Add => '+',
                ScalarBinOp::Sub => '-',
                ScalarBinOp::Mul => '*',
                ScalarBinOp::Div => '/',
                ScalarBinOp::Mod => '%',
            };
            format!(
                "({} {} {})",
                scalar_expr_canon(lhs),
                o,
                scalar_expr_canon(rhs)
            )
        }
    }
}

fn insert_expr_candidate(
    seen: &mut HashMap<Vec<Option<i64>>, ScalarExpr>,
    expr: ScalarExpr,
    examples: &[Vec<i64>],
) {
    let outputs = examples
        .iter()
        .map(|args| eval_scalar_expr(&expr, args))
        .collect::<Vec<_>>();
    if outputs.iter().all(Option::is_none) {
        return;
    }
    // Keep the simplest representative for each distinct behaviour (then a
    // stable canonical tiebreak), so the candidate pool is deterministic and
    // Occam-ordered rather than hash-order-arbitrary.
    use std::collections::hash_map::Entry;
    match seen.entry(outputs) {
        Entry::Vacant(slot) => {
            slot.insert(expr);
        }
        Entry::Occupied(mut slot) => {
            let new_key = (scalar_expr_complexity(&expr), scalar_expr_canon(&expr));
            let cur_key = (
                scalar_expr_complexity(slot.get()),
                scalar_expr_canon(slot.get()),
            );
            if new_key < cur_key {
                slot.insert(expr);
            }
        }
    }
}

/// Mine a constant pool from the problem's own examples rather than a fixed
/// hand-picked list. Real-world thresholds (50 GB, 100 minutes, 1000 calls)
/// and slopes (5, 2) never appear in a global default list — but they *do*
/// appear in the inputs, the outputs, and their differences. Anchors keep the
/// cheap universal constants available; the mined values make the search
/// relevant to the actual problem. Capped (smallest magnitude first) so the
/// candidate enumeration stays bounded.
pub(super) fn mine_scalar_constants(examples: &[Vec<i64>], targets: &[i64]) -> Vec<i64> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<i64> = BTreeSet::new();
    for a in [-1i64, 0, 1, 2, 3, 10, 100] {
        set.insert(a);
    }
    for ex in examples {
        for &v in ex {
            set.insert(v);
        }
    }
    for &t in targets {
        set.insert(t);
    }
    // per-example intercept (out - in) and exact slope (out / in) when integral
    for (ex, &t) in examples.iter().zip(targets.iter()) {
        if let Some(&x) = ex.first() {
            set.insert(t - x);
            if x != 0 && t % x == 0 {
                set.insert(t / x);
            }
        }
    }
    // step sizes between sorted distinct outputs (piecewise slopes) and inputs
    let steps = |mut v: Vec<i64>, set: &mut BTreeSet<i64>| {
        v.sort_unstable();
        v.dedup();
        for w in v.windows(2) {
            set.insert(w[1] - w[0]);
        }
    };
    steps(targets.to_vec(), &mut set);
    steps(
        examples.iter().filter_map(|e| e.first().copied()).collect(),
        &mut set,
    );
    // Raw inputs and outputs are the breakpoints and tier values of a piecewise
    // rule; they must survive truncation even when large (a 10000-call tier
    // threshold has a big |value| and a fixed-size by-magnitude cap would drop
    // it, leaving tiered rules unexpressible). Keep them all, then fill the
    // remaining budget with the smaller derived constants (intercepts, slopes,
    // step sizes) ordered by magnitude.
    let mut guaranteed: BTreeSet<i64> = BTreeSet::new();
    for a in [-1i64, 0, 1] {
        guaranteed.insert(a);
    }
    for ex in examples {
        for &v in ex {
            guaranteed.insert(v);
        }
    }
    for &t in targets {
        guaranteed.insert(t);
    }
    let with_neg = |src: &BTreeSet<i64>| -> Vec<i64> {
        let mut v: Vec<i64> = Vec::new();
        for &x in src {
            v.push(x);
            v.push(-x);
        }
        v.sort_by_key(|x| (x.unsigned_abs(), *x));
        v.dedup();
        v
    };
    let guaranteed_vec = with_neg(&guaranteed);
    let derived: BTreeSet<i64> = set.difference(&guaranteed).copied().collect();
    let derived_vec = with_neg(&derived);

    const CAP: usize = 48;
    let mut out = guaranteed_vec;
    for v in derived_vec {
        if out.len() >= CAP {
            break;
        }
        out.push(v);
    }
    out.sort_by_key(|v| (v.unsigned_abs(), *v));
    out.dedup();
    out
}

fn build_expr_candidates(
    arity: usize,
    examples: &[Vec<i64>],
    constants: &[i64],
) -> (Vec<ExprCandidate>, Vec<ExprCandidate>) {
    let mut atoms = Vec::new();
    let mut seen = HashMap::<Vec<Option<i64>>, ScalarExpr>::new();

    for index in 0..arity {
        insert_expr_candidate(&mut seen, ScalarExpr::Var(index), examples);
    }
    for &constant in constants {
        insert_expr_candidate(&mut seen, ScalarExpr::Const(constant), examples);
    }

    for (outputs, expr) in &seen {
        atoms.push(ExprCandidate {
            expr: expr.clone(),
            outputs: outputs.clone(),
        });
    }

    let atom_exprs = atoms
        .iter()
        .map(|candidate| candidate.expr.clone())
        .collect::<Vec<_>>();
    for lhs in &atom_exprs {
        for rhs in &atom_exprs {
            let rhs_data_dependent = scalar_expr_uses_var(rhs);
            for op in [
                ScalarBinOp::Add,
                ScalarBinOp::Sub,
                ScalarBinOp::Mul,
                ScalarBinOp::Div,
                ScalarBinOp::Mod,
            ] {
                if rhs_data_dependent && matches!(op, ScalarBinOp::Div | ScalarBinOp::Mod) {
                    continue; // no dividing/modding by a data-dependent value
                }
                insert_expr_candidate(
                    &mut seen,
                    ScalarExpr::Bin(Box::new(lhs.clone()), op, Box::new(rhs.clone())),
                    examples,
                );
            }
        }
    }

    let exprs = seen
        .into_iter()
        .map(|(outputs, expr)| ExprCandidate { expr, outputs })
        .collect::<Vec<_>>();

    (atoms, exprs)
}

pub(super) fn build_deep_expr_candidates(
    arity: usize,
    examples: &[Vec<i64>],
    constants: &[i64],
) -> Vec<ExprCandidate> {
    let mut seen = HashMap::<Vec<Option<i64>>, ScalarExpr>::new();

    for index in 0..arity {
        insert_expr_candidate(&mut seen, ScalarExpr::Var(index), examples);
    }
    for &constant in constants {
        insert_expr_candidate(&mut seen, ScalarExpr::Const(constant), examples);
    }

    let atom_exprs: Vec<ScalarExpr> = seen.values().cloned().collect();

    for lhs in &atom_exprs {
        for rhs in &atom_exprs {
            let rhs_data_dependent = scalar_expr_uses_var(rhs);
            for op in [
                ScalarBinOp::Add,
                ScalarBinOp::Sub,
                ScalarBinOp::Mul,
                ScalarBinOp::Div,
                ScalarBinOp::Mod,
            ] {
                if rhs_data_dependent && matches!(op, ScalarBinOp::Div | ScalarBinOp::Mod) {
                    continue;
                }
                insert_expr_candidate(
                    &mut seen,
                    ScalarExpr::Bin(Box::new(lhs.clone()), op, Box::new(rhs.clone())),
                    examples,
                );
            }
        }
    }

    let d2_exprs: Vec<ScalarExpr> = seen.values().cloned().collect();
    for expr in &d2_exprs {
        let expr_data_dependent = scalar_expr_uses_var(expr);
        for atom in &atom_exprs {
            let atom_data_dependent = scalar_expr_uses_var(atom);
            for op in [
                ScalarBinOp::Add,
                ScalarBinOp::Sub,
                ScalarBinOp::Mul,
                ScalarBinOp::Div,
                ScalarBinOp::Mod,
            ] {
                let is_div_mod = matches!(op, ScalarBinOp::Div | ScalarBinOp::Mod);
                // expr op atom: divisor is `atom`
                if !(is_div_mod && atom_data_dependent) {
                    insert_expr_candidate(
                        &mut seen,
                        ScalarExpr::Bin(Box::new(expr.clone()), op, Box::new(atom.clone())),
                        examples,
                    );
                }
                // atom op expr: divisor is `expr`
                if !(is_div_mod && expr_data_dependent) {
                    insert_expr_candidate(
                        &mut seen,
                        ScalarExpr::Bin(Box::new(atom.clone()), op, Box::new(expr.clone())),
                        examples,
                    );
                }
            }
        }
    }

    seen.into_iter()
        .map(|(outputs, expr)| ExprCandidate { expr, outputs })
        .collect()
}

fn build_condition_candidates(
    expr_candidates: &[ExprCandidate],
    atom_candidates: &[ExprCandidate],
) -> Vec<ConditionCandidate> {
    let mut seen = HashMap::<Vec<Option<bool>>, ConditionCandidate>::new();

    for lhs in expr_candidates {
        for rhs in atom_candidates {
            for op in [
                CompareOp::Eq,
                CompareOp::Ne,
                CompareOp::Lt,
                CompareOp::Le,
                CompareOp::Gt,
                CompareOp::Ge,
            ] {
                let outputs = lhs
                    .outputs
                    .iter()
                    .zip(rhs.outputs.iter())
                    .map(|(lhs, rhs)| match (lhs, rhs) {
                        (Some(lhs), Some(rhs)) => Some(eval_compare(*lhs, op, *rhs)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if outputs.iter().all(|value| value != &Some(true))
                    || outputs.iter().all(|value| value != &Some(false))
                {
                    continue;
                }
                seen.entry(outputs.clone()).or_insert(ConditionCandidate {
                    lhs: lhs.expr.clone(),
                    op,
                    rhs: rhs.expr.clone(),
                    outputs,
                });
            }
        }
    }

    seen.into_values().collect()
}

pub(super) fn extract_scalar_examples(problem: &Problem) -> Option<Vec<Vec<i64>>> {
    let param_types = parse_param_types(problem.signature);
    if param_types.is_empty()
        || param_types.len() > 3
        || param_types.iter().any(|ty| ty != &ParamType::I64)
    {
        return None;
    }

    problem
        .examples
        .iter()
        .map(|example| {
            if example.inputs.len() != param_types.len() {
                return None;
            }
            example
                .inputs
                .iter()
                .map(int_value)
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

pub(super) fn scalar_search_context(problem: &Problem) -> Option<ScalarSearchContext> {
    let examples = extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    let param_names = scalar_param_names(arity);
    let target: Vec<i64> = problem
        .examples
        .iter()
        .map(|example| example.expected_int())
        .collect();
    let constants = mine_scalar_constants(&examples, &target);
    let (mut atom_candidates, mut expr_candidates) =
        build_expr_candidates(arity, &examples, &constants);
    atom_candidates.sort_by_key(|candidate| {
        (
            scalar_expr_complexity(&candidate.expr),
            render_scalar_expr(&candidate.expr, &param_names),
        )
    });
    expr_candidates.sort_by_key(|candidate| {
        (
            scalar_expr_complexity(&candidate.expr),
            render_scalar_expr(&candidate.expr, &param_names),
        )
    });
    // A deeper (two-level) pool used only as then/else branch expressions, so
    // affine-with-threshold rules like `(x - 50) * 5` are reachable without
    // blowing up the lean pool that feeds condition enumeration. Ranked by
    // *target agreement* first — how many example positions the expression
    // already reproduces the target on — so the pieces of a piecewise function
    // (each of which matches the target exactly on its own region) rank at the
    // top and survive the cap, instead of being lost behind lexically-smaller
    // but useless expressions. Then complexity, then render for determinism.
    let mut branch_expr_candidates = build_deep_expr_candidates(arity, &examples, &constants);
    branch_expr_candidates.sort_by_key(|candidate| {
        let agree = candidate
            .outputs
            .iter()
            .zip(target.iter())
            .filter(|(out, t)| **out == Some(**t))
            .count();
        (
            std::cmp::Reverse(agree),
            scalar_expr_complexity(&candidate.expr),
            render_scalar_expr(&candidate.expr, &param_names),
        )
    });
    branch_expr_candidates.truncate(800);
    // Agreement decided which expressions survive the cap (so the pieces of a
    // piecewise rule aren't lost). But branch *selection* should be Occam: when
    // several survivors satisfy a region's subset constraint, the `.find` below
    // returns the first, so order the survivors simplest-first. This is what
    // makes the flat region of `max(0, x-50)*5` resolve to a plain `0` instead
    // of an equally-valid-on-the-examples but non-generalizing `(x*45) % -10`.
    branch_expr_candidates.sort_by_key(|candidate| {
        (
            scalar_expr_complexity(&candidate.expr),
            render_scalar_expr(&candidate.expr, &param_names),
        )
    });
    let mut cond_candidates = build_condition_candidates(&expr_candidates, &atom_candidates);
    cond_candidates.sort_by_key(|candidate| {
        (
            scalar_expr_complexity(&candidate.lhs) + scalar_expr_complexity(&candidate.rhs),
            format!(
                "{} {} {}",
                render_scalar_expr(&candidate.lhs, &param_names),
                render_compare_op(candidate.op),
                render_scalar_expr(&candidate.rhs, &param_names)
            ),
        )
    });
    Some(ScalarSearchContext {
        param_names,
        target,
        cond_candidates,
        branch_expr_candidates,
    })
}

pub(super) fn expr_matches_target(outputs: &[Option<i64>], target: &[i64]) -> bool {
    outputs
        .iter()
        .zip(target.iter())
        .all(|(output, target)| *output == Some(*target))
}

pub(super) fn expr_matches_subset(
    outputs: &[Option<i64>],
    target: &[i64],
    selected: &[bool],
) -> bool {
    let mut matched_any = false;
    for ((output, target), is_selected) in outputs.iter().zip(target.iter()).zip(selected.iter()) {
        if *is_selected {
            matched_any = true;
            if *output != Some(*target) {
                return false;
            }
        }
    }
    matched_any
}

pub(super) fn cond_is_total(outputs: &[Option<bool>]) -> bool {
    outputs.iter().all(Option::is_some)
}

pub(super) fn cond_selection(outputs: &[Option<bool>], branch_value: bool) -> Option<Vec<bool>> {
    let mut selected = Vec::with_capacity(outputs.len());
    let mut any = false;
    for output in outputs {
        match output {
            Some(value) => {
                let keep = *value == branch_value;
                any |= keep;
                selected.push(keep);
            }
            None => return None,
        }
    }
    any.then_some(selected)
}

pub(super) fn cond_selection_on_mask(
    outputs: &[Option<bool>],
    active_mask: &[bool],
    branch_value: bool,
) -> Option<Vec<bool>> {
    let mut selected = Vec::with_capacity(outputs.len());
    let mut any = false;
    for (output, active) in outputs.iter().zip(active_mask.iter()) {
        if !active {
            selected.push(false);
            continue;
        }
        match output {
            Some(value) => {
                let keep = *value == branch_value;
                any |= keep;
                selected.push(keep);
            }
            None => return None,
        }
    }
    any.then_some(selected)
}

pub(super) fn code_scalar_return_expr(
    fn_name: &str,
    param_names: &[String],
    expr: &ScalarExpr,
) -> String {
    let params = scalar_params_decl(param_names);
    let expr = render_scalar_expr(expr, param_names);
    format!("fn {fn_name}({params}) -> i64 {{\n    return {expr};\n}}\n")
}

pub(super) fn code_scalar_single_branch(
    fn_name: &str,
    param_names: &[String],
    cond: &ConditionCandidate,
    then_expr: &ScalarExpr,
    else_expr: &ScalarExpr,
) -> String {
    let params = scalar_params_decl(param_names);
    let cond = format!(
        "{} {} {}",
        render_scalar_expr(&cond.lhs, param_names),
        render_compare_op(cond.op),
        render_scalar_expr(&cond.rhs, param_names)
    );
    let then_expr = render_scalar_expr(then_expr, param_names);
    let else_expr = render_scalar_expr(else_expr, param_names);
    format!(
        "fn {fn_name}({params}) -> i64 {{\n    if {cond} {{\n        return {then_expr};\n    }}\n    return {else_expr};\n}}\n"
    )
}

pub(super) fn code_scalar_two_branch(
    fn_name: &str,
    param_names: &[String],
    first_cond: &ConditionCandidate,
    first_expr: &ScalarExpr,
    second_cond: &ConditionCandidate,
    second_expr: &ScalarExpr,
    else_expr: &ScalarExpr,
) -> String {
    let params = scalar_params_decl(param_names);
    let first_cond = format!(
        "{} {} {}",
        render_scalar_expr(&first_cond.lhs, param_names),
        render_compare_op(first_cond.op),
        render_scalar_expr(&first_cond.rhs, param_names)
    );
    let second_cond = format!(
        "{} {} {}",
        render_scalar_expr(&second_cond.lhs, param_names),
        render_compare_op(second_cond.op),
        render_scalar_expr(&second_cond.rhs, param_names)
    );
    let first_expr = render_scalar_expr(first_expr, param_names);
    let second_expr = render_scalar_expr(second_expr, param_names);
    let else_expr = render_scalar_expr(else_expr, param_names);
    format!(
        "fn {fn_name}({params}) -> i64 {{\n    if {first_cond} {{\n        return {first_expr};\n    }}\n    if {second_cond} {{\n        return {second_expr};\n    }}\n    return {else_expr};\n}}\n"
    )
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    fn storage_problem() -> Problem {
        let rows = [
            (0, 0),
            (50, 0),
            (51, 5),
            (60, 50),
            (40, 0),
            (70, 100),
            (200, 750),
        ];
        Problem {
            name: "storage_overage".to_string(),
            category: "external",
            description: "",
            signature: "fn storage_overage(used_gb: i64) -> i64",
            examples: rows
                .iter()
                .map(|(i, o)| Example {
                    inputs: vec![Value::Int(*i)],
                    expected: Value::Int(*o),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        }
    }

    // Regression: a threshold-with-affine rule `max(0, x-50)*5` must be
    // solvable by generic single-branch search. Before example-mined
    // constants + the deeper agreement-ranked branch pool, this novel
    // (non-benchmark-named) problem could not be expressed at all.
    #[test]
    fn single_branch_solves_threshold_affine() {
        let p = storage_problem();
        let consts = mine_scalar_constants(
            &extract_scalar_examples(&p).unwrap(),
            &p.examples
                .iter()
                .map(|e| e.expected_int())
                .collect::<Vec<_>>(),
        );
        assert!(
            consts.contains(&50),
            "threshold 50 must be mined from examples"
        );
        // search_single_branch only returns Some after verified_result has
        // confirmed the program reproduces every example, so a Some here means
        // an exact, verified solution.
        let result = super::search_scalar_families::search_single_branch(&p, "storage_overage")
            .expect("single_branch must solve max(0,x-50)*5");
        assert!(result.success);
        assert!(result.code.contains("if"), "expected a branch program");
    }

    fn api_bill_problem() -> Problem {
        // 3-tier: free <=1000, 2c/call 1001..10000, 1c/call beyond.
        let rows = [
            (0, 0),
            (500, 0),
            (1000, 0),
            (1001, 2),
            (2000, 2000),
            (5000, 8000),
            (10000, 18000),
            (10001, 18001),
            (15000, 23000),
            (20000, 28000),
        ];
        Problem {
            name: "api_bill".to_string(),
            category: "external",
            description: "",
            signature: "fn api_bill(x: i64) -> i64",
            examples: rows
                .iter()
                .map(|(i, o)| Example {
                    inputs: vec![Value::Int(*i)],
                    expected: Value::Int(*o),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        }
    }

    fn api_bill_ref(x: i64) -> i64 {
        if x <= 1000 {
            0
        } else if x <= 10000 {
            2 * (x - 1000)
        } else {
            18000 + (x - 10000)
        }
    }

    fn dense_api_bill_problem() -> Problem {
        // Tier-spanning points chosen so the *only* program fitting all of them
        // is the exact tiered rule. The mid-tier values are deliberately not
        // multiples of any breakpoint, so an integer-division flooring overfit
        // (e.g. `(x/1001)*2000`) produces a different output on the training set
        // itself and therefore cannot fit — removing the equal-output ambiguity
        // that, on a sparse set, lets such an overfit be selected by hash order.
        // In the live pipeline CEGIS supplies exactly this kind of spread and
        // the dense-sweep gate rejects any survivor that is still wrong, so
        // MeterBill converges to the true rule reliably; this keeps the unit
        // test fast (few examples) while pinning the same outcome.
        let xs = [
            0, 317, 613, 1000, // tier 0
            1001, 1234, 1777, 2345, 3001, 4567, 5001, 6789, 7777, 8888, 9123, 9999,
            10000, // tier 1 (non-multiples)
            10001, 11234, 13567, 16789, 20000, // tier 2
        ];
        Problem {
            name: "api_bill_tiered_probe".to_string(),
            category: "external",
            description: "",
            signature: "fn api_bill_tiered_probe(x: i64) -> i64",
            examples: xs
                .iter()
                .map(|&x| Example {
                    inputs: vec![Value::Int(x)],
                    expected: Value::Int(api_bill_ref(x)),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        }
    }

    // Regression: large tier breakpoints must survive constant mining, and a
    // two-breakpoint tiered rule must be synthesizable *correctly* (not via a
    // modulo/division overfit that only fits the sampled points). Before the
    // var-as-divisor ban + breakpoint-preserving mining, the only fit found was
    // an integer-division overfit wrong between samples.
    #[test]
    fn two_breakpoint_tiered_rule_is_exact() {
        let sparse = api_bill_problem();
        let consts = mine_scalar_constants(
            &extract_scalar_examples(&sparse).unwrap(),
            &sparse
                .examples
                .iter()
                .map(|e| e.expected_int())
                .collect::<Vec<_>>(),
        );
        for want in [1000_i64, 1001, 10000] {
            assert!(
                consts.contains(&want),
                "breakpoint {} must survive mining",
                want
            );
        }
        // No `k % x` / `_ / x` overfit class: single-branch cannot fake the
        // tiers on the sparse set, so it honestly fails (or the program it
        // returns is *not* a divisor-by-variable hack).
        if let Some(r) = super::search_scalar_families::search_single_branch(&sparse, "api_bill") {
            assert!(
                !r.code.contains("% x") && !r.code.contains("/ x"),
                "single-branch must not overfit by dividing/modding by the input: {}",
                r.code
            );
        }
        // Given tier-spanning examples, two-branch search recovers the exact
        // rule — verified correct across the whole domain, including unseen
        // points between and beyond the samples.
        let dense = dense_api_bill_problem();
        let result =
            super::search_scalar_families::search_two_branch(&dense, "api_bill_tiered_probe")
                .expect("two_branch must solve the dense tiered rule");
        assert!(result.success);
        // Prove the synthesized program is correct on UNSEEN points (offset from
        // the training samples) across and beyond the training range.
        let check = Problem {
            name: "api_bill_tiered_probe".to_string(),
            category: "external",
            description: "",
            signature: "fn api_bill_tiered_probe(x: i64) -> i64",
            examples: (0..=30000)
                .step_by(137)
                .map(|x| Example {
                    inputs: vec![Value::Int(x)],
                    expected: Value::Int(api_bill_ref(x)),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        };
        crate::runtime::verify_problem_code_strict(&check, &result.code)
            .expect("tiered rule must be exact on unseen points, not an overfit");
    }

    fn scalar_problem(name: &'static str, rows: &[(i64, i64)]) -> Problem {
        Problem {
            name: name.to_string(),
            category: "external",
            description: "",
            signature: "fn f(x: i64) -> i64",
            examples: rows
                .iter()
                .map(|(i, o)| Example {
                    inputs: vec![Value::Int(*i)],
                    expected: Value::Int(*o),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        }
    }

    // The piecewise-affine solver recovers a 3-tier rule exactly from examples
    // and is verified correct on UNSEEN points — the capability that lifts the
    // generalization probe from 0% to most of 3/4-tier rules.
    #[test]
    fn piecewise_affine_recovers_three_tiers_exactly() {
        // 0 for x<=1000; 2(x-1000) for 1000<x<=5000; 8000+5(x-5000) beyond.
        let f = |x: i64| -> i64 {
            if x <= 1000 {
                0
            } else if x <= 5000 {
                2 * (x - 1000)
            } else {
                8000 + 5 * (x - 5000)
            }
        };
        let train: Vec<(i64, i64)> = [
            0, 500, 1000, 1001, 2000, 3000, 5000, 5001, 7000, 9000, 12000,
        ]
        .iter()
        .map(|&x| (x, f(x)))
        .collect();
        let p = scalar_problem("f", &train);
        let r = super::search_scalar_families::search_piecewise_affine(&p, "f")
            .expect("piecewise must solve the 3-tier rule");
        assert!(
            r.code.matches("if").count() >= 2,
            "expected >=3 pieces: {}",
            r.code
        );
        let check = scalar_problem(
            "f",
            &(0..=20000)
                .step_by(53)
                .map(|x| (x, f(x)))
                .collect::<Vec<_>>(),
        );
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("3-tier rule must be exact on unseen points");
    }

    // It must NOT fake a curve with a per-point staircase: a quadratic is not
    // piecewise-affine, so the solver refuses (returns None) and leaves it to
    // the other solvers, rather than emitting an overfit.
    #[test]
    fn piecewise_affine_refuses_quadratic() {
        let rows: Vec<(i64, i64)> = (0..12).map(|x| (x, x * x)).collect();
        let p = scalar_problem("sq", &rows);
        assert!(
            super::search_scalar_families::search_piecewise_affine(&p, "sq").is_none(),
            "piecewise must refuse a non-affine curve, not staircase-overfit it"
        );
    }
}
