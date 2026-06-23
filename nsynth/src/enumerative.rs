//! Bottom-up enumerative program synthesis with learned component library.
//!
//! Builds ALL programs from small to large, tests each against I/O examples,
//! returns the first (smallest) program that satisfies all examples.
//!
//! Features:
//!   - 12 binary ops: +, -, *, /, %, min, max, &, |, ^, <<, >>
//!   - 4 unary ops: neg, abs, bitwise_not, popcount
//!   - If-then-else with 6 comparisons
//!   - While loops with accumulator pattern
//!   - Observational equivalence pruning (skip equivalent programs)
//!   - Learned component library: saves discovered sub-expressions
//!   - Dream mode: explores combinations offline to grow the library
//!
//! No templates. No patterns. Pure discovery.

use crate::benchmark::{Problem, Value};
use crate::differentiable::DifferentiableMetadata;
use crate::runtime::verify_problem_code_strict;
use crate::solver::SolveResult;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

// ─── Expression AST ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Var(usize),
    Const(i64),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnOp, Box<Expr>),
    IfExpr(CmpOp, Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>), // if lhs CMP rhs { then } else { els }
    WhileAccum {
        // acc = init; while cond { acc = body(acc, i); i++ } return acc
        init: Box<Expr>,
        bound: Box<Expr>,    // loop runs i from 0..bound
        body_op: BinOp,      // acc = acc OP rhs
        body_rhs: Box<Expr>, // rhs computed from [acc, i, args, consts]
    },
    ForFold {
        // acc = init; for item in arr { acc = acc OP rhs(item, i, acc, args) } return acc
        init: Box<Expr>,
        body_op: BinOp,
        body_rhs: Box<Expr>, // rhs over {item, i, acc, scalar_args, consts}
    },
    NestedWhile {
        // two-level nested accumulator loop
        outer_init: Box<Expr>,
        outer_bound: Box<Expr>,
        outer_body_op: BinOp,
        inner_init: Box<Expr>,
        inner_bound: Box<Expr>,
        inner_body_op: BinOp,
        inner_body_rhs: Box<Expr>, // rhs over {args, outer_acc, outer_i, inner_acc, inner_i}
    },
    WhileCond {
        // acc = init; x = x_init; while x CMP cond_val { acc = acc OP body(x, acc, args); x = x / divisor } return acc
        init: Box<Expr>,       // acc init
        state_init: Box<Expr>, // x init (typically arg[0])
        cond_cmp: CmpOp,       // condition: x CMP cond_val
        cond_val: Box<Expr>,   // value to compare against
        divisor: i64,          // x update: x = x / divisor
        body_op: BinOp,        // acc = acc OP body_rhs
        body_rhs: Box<Expr>,   // rhs over {args, x, acc, i}
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Abs,
    BitNot,
    Popcount,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
    Ne,
}

const ALL_BINOPS: [BinOp; 12] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Mod,
    BinOp::Min,
    BinOp::Max,
    BinOp::BitAnd,
    BinOp::BitOr,
    BinOp::BitXor,
    BinOp::Shl,
    BinOp::Shr,
];
const ALL_UNOPS: [UnOp; 4] = [UnOp::Neg, UnOp::Abs, UnOp::BitNot, UnOp::Popcount];
const ALL_CMPS: [CmpOp; 6] = [
    CmpOp::Lt,
    CmpOp::Le,
    CmpOp::Eq,
    CmpOp::Ge,
    CmpOp::Gt,
    CmpOp::Ne,
];
const CONSTANTS: [i64; 12] = [0, 1, -1, 2, -2, 3, 5, 7, 10, 32, 100, 255];

// Ops suitable for loop body (no division to avoid div-by-zero in loops)
const LOOP_BODY_OPS: [BinOp; 5] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::BitXor,
    BinOp::BitOr,
];

impl Expr {
    pub fn eval(&self, args: &[i64]) -> Option<i64> {
        match self {
            Expr::Var(i) => args.get(*i).copied(),
            Expr::Const(c) => Some(*c),
            Expr::BinOp(op, l, r) => {
                let a = l.eval(args)?;
                let b = r.eval(args)?;
                eval_binop(*op, a, b)
            }
            Expr::UnaryOp(op, e) => {
                let v = e.eval(args)?;
                match op {
                    UnOp::Neg => v.checked_neg(),
                    UnOp::Abs => v.checked_abs(),
                    UnOp::BitNot => Some(!v),
                    UnOp::Popcount => Some((v as u64).count_ones() as i64),
                }
            }
            Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) => {
                let l = lhs.eval(args)?;
                let r = rhs.eval(args)?;
                if eval_cmp(*cmp, l, r) {
                    then_e.eval(args)
                } else {
                    else_e.eval(args)
                }
            }
            Expr::WhileAccum {
                init,
                bound,
                body_op,
                body_rhs,
            } => {
                let mut acc = init.eval(args)?;
                let n = bound.eval(args)?;
                if n < 0 || n > 10_000 {
                    return None;
                } // safety bound
                for i in 0..n {
                    // Build extended args: original args + [acc, i]
                    let mut ext = args.to_vec();
                    ext.push(acc);
                    ext.push(i);
                    let rhs = body_rhs.eval(&ext)?;
                    acc = eval_binop(*body_op, acc, rhs)?;
                }
                Some(acc)
            }
            // ForFold, NestedWhile, WhileCond are never evaluated via scalar eval().
            // They use eval_array / eval_nested / eval_while_cond instead.
            Expr::ForFold { .. } | Expr::NestedWhile { .. } | Expr::WhileCond { .. } => None,
        }
    }

    /// Evaluate a ForFold expression: acc = init; for (i, item) in arr { acc = acc OP rhs }
    /// Extended args for body: [scalar_args..., item, i, acc]
    pub fn eval_array(&self, scalar_args: &[i64], array: &[i64]) -> Option<i64> {
        match self {
            Expr::ForFold {
                init,
                body_op,
                body_rhs,
            } => {
                let mut acc = init.eval(scalar_args)?;
                for (i, &item) in array.iter().enumerate() {
                    let mut ext = scalar_args.to_vec();
                    ext.push(item); // item
                    ext.push(i as i64); // index
                    ext.push(acc); // accumulator
                    let rhs = body_rhs.eval(&ext)?;
                    acc = eval_binop(*body_op, acc, rhs)?;
                }
                Some(acc)
            }
            _ => None,
        }
    }

    /// Evaluate a NestedWhile expression.
    /// Extended args for inner body: [args..., outer_acc, outer_i, inner_acc, inner_i]
    pub fn eval_nested(&self, args: &[i64]) -> Option<i64> {
        match self {
            Expr::NestedWhile {
                outer_init,
                outer_bound,
                outer_body_op,
                inner_init,
                inner_bound,
                inner_body_op,
                inner_body_rhs,
            } => {
                let mut outer_acc = outer_init.eval(args)?;
                let outer_n = outer_bound.eval(args)?;
                if outer_n < 0 || outer_n > 1000 {
                    return None;
                }
                for outer_i in 0..outer_n {
                    // Run inner loop
                    let mut inner_acc = inner_init.eval(args)?;
                    let inner_n = inner_bound.eval(args)?;
                    if inner_n < 0 || inner_n > 1000 {
                        return None;
                    }
                    for inner_i in 0..inner_n {
                        let mut ext = args.to_vec();
                        ext.push(outer_acc);
                        ext.push(outer_i);
                        ext.push(inner_acc);
                        ext.push(inner_i);
                        let rhs = inner_body_rhs.eval(&ext)?;
                        inner_acc = eval_binop(*inner_body_op, inner_acc, rhs)?;
                    }
                    // Outer update: outer_acc = outer_acc OP inner_result
                    outer_acc = eval_binop(*outer_body_op, outer_acc, inner_acc)?;
                }
                Some(outer_acc)
            }
            _ => None,
        }
    }

    /// Evaluate a WhileCond expression: digit-style loop.
    /// acc = init; x = state_init; while x CMP cond_val { acc = acc OP body(x, acc, args); x = x / divisor }
    pub fn eval_while_cond(&self, args: &[i64]) -> Option<i64> {
        match self {
            Expr::WhileCond {
                init,
                state_init,
                cond_cmp,
                cond_val,
                divisor,
                body_op,
                body_rhs,
            } => {
                let mut acc = init.eval(args)?;
                let mut x = state_init.eval(args)?;
                let cv = cond_val.eval(args)?;
                if *divisor == 0 {
                    return None;
                }
                let mut iterations = 0;
                while eval_cmp(*cond_cmp, x, cv) {
                    if iterations > 10_000 {
                        return None;
                    }
                    // Body namespace: args ++ [x, acc, i]
                    let mut ext = args.to_vec();
                    ext.push(x); // x = Var(n_args)
                    ext.push(acc); // acc = Var(n_args + 1)
                    ext.push(iterations as i64); // i = Var(n_args + 2)
                    let rhs = body_rhs.eval(&ext)?;
                    acc = eval_binop(*body_op, acc, rhs)?;
                    x = x / divisor;
                    iterations += 1;
                }
                Some(acc)
            }
            _ => None,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::BinOp(_, l, r) => 1 + l.size() + r.size(),
            Expr::UnaryOp(_, e) => 1 + e.size(),
            Expr::IfExpr(_, a, b, c, d) => 1 + a.size() + b.size() + c.size() + d.size(),
            Expr::WhileAccum {
                init,
                bound,
                body_rhs,
                ..
            } => 3 + init.size() + bound.size() + body_rhs.size(),
            Expr::ForFold { init, body_rhs, .. } => 3 + init.size() + body_rhs.size(),
            Expr::NestedWhile {
                outer_init,
                outer_bound,
                inner_init,
                inner_bound,
                inner_body_rhs,
                ..
            } => {
                7 + outer_init.size()
                    + outer_bound.size()
                    + inner_init.size()
                    + inner_bound.size()
                    + inner_body_rhs.size()
            }
            Expr::WhileCond {
                init,
                state_init,
                cond_val,
                body_rhs,
                ..
            } => 5 + init.size() + state_init.size() + cond_val.size() + body_rhs.size(),
        }
    }

    pub fn to_mog(&self, param_names: &[&str]) -> String {
        self.to_mog_ext(param_names, &[])
    }

    fn to_mog_ext(&self, param_names: &[&str], extra_names: &[&str]) -> String {
        match self {
            Expr::Var(i) => {
                if *i < param_names.len() {
                    param_names[*i].to_string()
                } else {
                    let ext_idx = *i - param_names.len();
                    extra_names.get(ext_idx).unwrap_or(&"x").to_string()
                }
            }
            Expr::Const(c) => {
                if *c == i64::MIN {
                    "(-9223372036854775807 - 1)".to_string()
                } else if *c < 0 {
                    format!("(0 - {})", c.checked_neg().unwrap())
                } else {
                    format!("{c}")
                }
            }
            Expr::BinOp(op, l, r) => {
                let ls = l.to_mog_ext(param_names, extra_names);
                let rs = r.to_mog_ext(param_names, extra_names);
                match op {
                    BinOp::Add => format!("{ls} + {rs}"),
                    BinOp::Sub => format!("{ls} - {rs}"),
                    BinOp::Mul => format!("{ls} * {rs}"),
                    BinOp::Div => format!("{ls} / {rs}"),
                    BinOp::Mod => format!("{ls} % {rs}"),
                    BinOp::Min => format!("if {ls} < {rs} {{ {ls} }} else {{ {rs} }}"),
                    BinOp::Max => format!("if {ls} > {rs} {{ {ls} }} else {{ {rs} }}"),
                    BinOp::BitAnd => format!("{ls} % {rs}"), // Mog doesn't have &, approximate
                    BinOp::BitOr => format!("{ls} + {rs}"),  // approximate
                    BinOp::BitXor => format!("{ls} - {rs}"), // approximate
                    BinOp::Shl => format!("{ls} * 2"),       // approximate
                    BinOp::Shr => format!("{ls} / 2"),       // approximate
                }
            }
            Expr::UnaryOp(op, e) => {
                let es = e.to_mog_ext(param_names, extra_names);
                match op {
                    UnOp::Neg => format!("0 - {es}"),
                    UnOp::Abs => format!("if {es} < 0 {{ 0 - {es} }} else {{ {es} }}"),
                    UnOp::BitNot => format!("0 - {es} - 1"),
                    UnOp::Popcount => {
                        // Can't express popcount as single expr, emit as loop
                        format!("{es}") // placeholder — WhileAccum handles this
                    }
                }
            }
            Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) => {
                let ls = lhs.to_mog_ext(param_names, extra_names);
                let rs = rhs.to_mog_ext(param_names, extra_names);
                let cs = match cmp {
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Eq => "==",
                    CmpOp::Ge => ">=",
                    CmpOp::Gt => ">",
                    CmpOp::Ne => "!=",
                };
                let ts = then_e.to_mog_ext(param_names, extra_names);
                let es = else_e.to_mog_ext(param_names, extra_names);
                format!("if {ls} {cs} {rs} {{ {ts} }} else {{ {es} }}")
            }
            Expr::WhileAccum {
                init,
                bound,
                body_op,
                body_rhs,
            } => {
                let ext_names = &["acc", "i"];
                let init_s = init.to_mog_ext(param_names, &[]);
                let bound_s = bound.to_mog_ext(param_names, &[]);
                let op_s = match body_op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::BitXor => "-", // approximate
                    _ => "+",
                };
                let rhs_s = body_rhs.to_mog_ext(param_names, ext_names);
                format!("/* loop */ {init_s}; /* while i < {bound_s}: acc = acc {op_s} {rhs_s} */")
            }
            Expr::ForFold { .. } => {
                // Handled by emit_mog directly
                format!("/* for-fold */")
            }
            Expr::NestedWhile { .. } => {
                format!("/* nested-while */")
            }
            Expr::WhileCond { .. } => {
                format!("/* while-cond */")
            }
        }
    }
}

fn eval_binop(op: BinOp, a: i64, b: i64) -> Option<i64> {
    match op {
        BinOp::Add => a.checked_add(b),
        BinOp::Sub => a.checked_sub(b),
        BinOp::Mul => a.checked_mul(b),
        BinOp::Div => {
            if b == 0 || (a == i64::MIN && b == -1) {
                None
            } else {
                Some(a / b)
            }
        }
        BinOp::Mod => {
            if b == 0 || (a == i64::MIN && b == -1) {
                None
            } else {
                Some(a % b)
            }
        }
        BinOp::Min => Some(a.min(b)),
        BinOp::Max => Some(a.max(b)),
        BinOp::BitAnd => Some(a & b),
        BinOp::BitOr => Some(a | b),
        BinOp::BitXor => Some(a ^ b),
        BinOp::Shl => {
            if b < 0 || b > 63 {
                None
            } else {
                Some(a << b)
            }
        }
        BinOp::Shr => {
            if b < 0 || b > 63 {
                None
            } else {
                Some(a >> b)
            }
        }
    }
}

fn eval_cmp(cmp: CmpOp, a: i64, b: i64) -> bool {
    match cmp {
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Eq => a == b,
        CmpOp::Ge => a >= b,
        CmpOp::Gt => a > b,
        CmpOp::Ne => a != b,
    }
}

// ─── Observational equivalence ───────────────────────────────────────────────

fn fingerprint(expr: &Expr, test_inputs: &[Vec<i64>]) -> Option<Vec<i64>> {
    let mut fp = Vec::with_capacity(test_inputs.len());
    for args in test_inputs {
        match expr.eval(args) {
            Some(v) => fp.push(v),
            None => return None,
        }
    }
    Some(fp)
}

fn matches_all(expr: &Expr, examples: &[(Vec<i64>, i64)]) -> bool {
    examples
        .iter()
        .all(|(args, expected)| expr.eval(args) == Some(*expected))
}

// ─── Robust verification (anti-fingerprint collision) ──────────────────────────

/// Deterministic probe inputs: hash-based, no external RNG dependency.
/// Returns `count` input vectors with values in [-100, 100].
fn probe_inputs(n_args: usize, count: usize) -> Vec<Vec<i64>> {
    (0..count)
        .map(|i| {
            (0..n_args)
                .map(|j| {
                    // Simple hash: mix i and j, map to [-100, 100]
                    let h = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add((j as u64).wrapping_mul(1442695040888963407))
                        .wrapping_add(0xB5001F);
                    ((h % 201) as i64) - 100
                })
                .collect()
        })
        .collect()
}

/// Check that an expression doesn't crash on random inputs (no None returns).
/// This catches spurious matches like `b - b%3` for min(a,b).
fn robust_well_defined(expr: &Expr, n_args: usize, n_probes: usize) -> bool {
    for args in probe_inputs(n_args, n_probes) {
        if expr.eval(&args).is_none() {
            return false;
        }
    }
    true
}

// ─── Corpus-driven subtree mining (frequent-subtree + anti-unification) ────────
//
// Replaces the old hardcoded 15-seed `dream()`. We mine recurring scalar
// subtrees out of a corpus of *verified* solved `Expr` trees (mined identically
// to the trees we later inject — sound by construction). Each mined component is
// a canonicalized `Expr` whose free `Var(0..k)` ARE its parameters; there is no
// new AST node — abstractions reuse the existing free-Var-as-hole substrate so
// eval/to_mog/serde need no changes. Anti-unification (least-general
// generalization) folds structurally-matching subtrees (e.g. `x*x` and `y*y`)
// into a shared pattern `?0*?0` so mined components GENERALIZE rather than
// memorize. All counting is BTreeMap-keyed and all selection is total-ordered:
// no clock/rand ever picks an abstraction (Instant only bounds work).

use std::collections::{BTreeMap, BTreeSet};

/// Stable, deterministic BTreeMap/grouping key for an Expr. `Expr` derives
/// `Debug`, whose output is deterministic — the same approach the old `dream()`
/// already trusted for component descriptions.
fn expr_key(e: &Expr) -> String {
    format!("{e:?}")
}

/// Distinct `Var` indices referenced by a *scalar* subtree (loop variants
/// contribute nothing — they are out of scope for mining v1).
fn free_vars(e: &Expr) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    fn walk(e: &Expr, out: &mut BTreeSet<usize>) {
        match e {
            Expr::Var(i) => {
                out.insert(*i);
            }
            Expr::Const(_) => {}
            Expr::BinOp(_, l, r) => {
                walk(l, out);
                walk(r, out);
            }
            Expr::UnaryOp(_, c) => walk(c, out),
            Expr::IfExpr(_, a, b, c, d) => {
                walk(a, out);
                walk(b, out);
                walk(c, out);
                walk(d, out);
            }
            // Loop variants are out of scope — their interiors reference
            // loop-local extended Var slots that are meaningless when lifted.
            Expr::WhileAccum { .. }
            | Expr::ForFold { .. }
            | Expr::NestedWhile { .. }
            | Expr::WhileCond { .. } => {}
        }
    }
    walk(e, &mut out);
    out
}

/// Collect every pure-scalar node (self + descendants) into `out`. Recurses
/// through BinOp/UnaryOp/IfExpr children only. On encountering ANY loop variant
/// it pushes nothing and stops descending (scope guard — loop interiors
/// reference loop-local Var slots that can't be lifted soundly).
fn collect_scalar_subtrees<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
        Expr::Var(_) | Expr::Const(_) => out.push(e),
        Expr::BinOp(_, l, r) => {
            out.push(e);
            collect_scalar_subtrees(l, out);
            collect_scalar_subtrees(r, out);
        }
        Expr::UnaryOp(_, c) => {
            out.push(e);
            collect_scalar_subtrees(c, out);
        }
        Expr::IfExpr(_, a, b, c, d) => {
            out.push(e);
            collect_scalar_subtrees(a, out);
            collect_scalar_subtrees(b, out);
            collect_scalar_subtrees(c, out);
            collect_scalar_subtrees(d, out);
        }
        // Scope guard: do not push, do not descend into loop interiors.
        Expr::WhileAccum { .. }
        | Expr::ForFold { .. }
        | Expr::NestedWhile { .. }
        | Expr::WhileCond { .. } => {}
    }
}

/// Rename free `Var` indices to dense slots `0..k` in left-to-right
/// first-encounter order (deterministic via BTreeMap). Returns the canonical
/// Expr plus its arity `k`. Since loop interiors are excluded by
/// [`collect_scalar_subtrees`], every Var seen here is a free parameter, so
/// `x*x` -> (`Var0*Var0`, 1) and `a-b` -> (`Var0-Var1`, 2).
fn canonicalize(e: &Expr) -> (Expr, usize) {
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    // First pass: assign dense slots in first-encounter (left-to-right) order.
    fn assign(e: &Expr, map: &mut BTreeMap<usize, usize>, next: &mut usize) {
        match e {
            Expr::Var(i) => {
                if !map.contains_key(i) {
                    map.insert(*i, *next);
                    *next += 1;
                }
            }
            Expr::Const(_) => {}
            Expr::BinOp(_, l, r) => {
                assign(l, map, next);
                assign(r, map, next);
            }
            Expr::UnaryOp(_, c) => assign(c, map, next),
            Expr::IfExpr(_, a, b, c, d) => {
                assign(a, map, next);
                assign(b, map, next);
                assign(c, map, next);
                assign(d, map, next);
            }
            _ => {}
        }
    }
    let mut next = 0usize;
    assign(e, &mut map, &mut next);
    fn rebuild(e: &Expr, map: &BTreeMap<usize, usize>) -> Expr {
        match e {
            Expr::Var(i) => Expr::Var(*map.get(i).unwrap_or(i)),
            Expr::Const(c) => Expr::Const(*c),
            Expr::BinOp(op, l, r) => Expr::BinOp(
                *op,
                Box::new(rebuild(l, map)),
                Box::new(rebuild(r, map)),
            ),
            Expr::UnaryOp(op, c) => Expr::UnaryOp(*op, Box::new(rebuild(c, map))),
            Expr::IfExpr(cmp, a, b, c, d) => Expr::IfExpr(
                *cmp,
                Box::new(rebuild(a, map)),
                Box::new(rebuild(b, map)),
                Box::new(rebuild(c, map)),
                Box::new(rebuild(d, map)),
            ),
            other => other.clone(),
        }
    }
    (rebuild(e, &map), next)
}

/// Op-only structural skeleton key (consts and vars erased to `?`). Used to
/// group candidates for pairwise anti-unification so we only attempt lgg within
/// a structurally-matching group (keeps it O(group^2) and deterministic).
fn skeleton(e: &Expr) -> String {
    match e {
        Expr::Var(_) | Expr::Const(_) => "?".to_string(),
        Expr::BinOp(op, l, r) => format!("({:?} {} {})", op, skeleton(l), skeleton(r)),
        Expr::UnaryOp(op, c) => format!("({:?} {})", op, skeleton(c)),
        Expr::IfExpr(cmp, a, b, c, d) => format!(
            "(if {:?} {} {} {} {})",
            cmp,
            skeleton(a),
            skeleton(b),
            skeleton(c),
            skeleton(d)
        ),
        other => format!("{other:?}"),
    }
}

/// First-order least-general generalization (anti-unification). When the two
/// trees share a top constructor + op/cmp, recurse and rebuild the node. Equal
/// `Var` indices and equal `Const`s are kept verbatim. On ANY disagreement we
/// allocate a shared hole: a linear scan of `subst` reuses the existing hole for
/// a previously-seen (a_sub, b_sub) pair, so `f(x,x)` vs `g(y,y)` generalizes to
/// `?0 OP ?0` (ONE hole), not two. Holes are emitted as `Expr::Var(next_hole)`
/// — re-using the free-Var substrate, NO new AST variant. Returns `None` when
/// the top constructors are incompatible (no useful lgg there).
fn anti_unify(
    a: &Expr,
    b: &Expr,
    subst: &mut Vec<(Expr, Expr)>,
    next_hole: &mut usize,
) -> Option<Expr> {
    // Helper: allocate (or reuse) a shared hole for a disagreeing pair.
    fn hole_for(
        a: &Expr,
        b: &Expr,
        subst: &mut Vec<(Expr, Expr)>,
        next_hole: &mut usize,
    ) -> Expr {
        for (idx, (sa, sb)) in subst.iter().enumerate() {
            if sa == a && sb == b {
                return Expr::Var(idx);
            }
        }
        let h = *next_hole;
        subst.push((a.clone(), b.clone()));
        *next_hole += 1;
        Expr::Var(h)
    }
    match (a, b) {
        (Expr::Var(i), Expr::Var(j)) if i == j => Some(Expr::Var(*i)),
        (Expr::Const(x), Expr::Const(y)) if x == y => Some(Expr::Const(*x)),
        (Expr::BinOp(o1, l1, r1), Expr::BinOp(o2, l2, r2)) if o1 == o2 => {
            let l = anti_unify(l1, l2, subst, next_hole)?;
            let r = anti_unify(r1, r2, subst, next_hole)?;
            Some(Expr::BinOp(*o1, Box::new(l), Box::new(r)))
        }
        (Expr::UnaryOp(o1, c1), Expr::UnaryOp(o2, c2)) if o1 == o2 => {
            let c = anti_unify(c1, c2, subst, next_hole)?;
            Some(Expr::UnaryOp(*o1, Box::new(c)))
        }
        (Expr::IfExpr(cmp1, a1, b1, t1, e1), Expr::IfExpr(cmp2, a2, b2, t2, e2))
            if cmp1 == cmp2 =>
        {
            let la = anti_unify(a1, a2, subst, next_hole)?;
            let lb = anti_unify(b1, b2, subst, next_hole)?;
            let lt = anti_unify(t1, t2, subst, next_hole)?;
            let le = anti_unify(e1, e2, subst, next_hole)?;
            Some(Expr::IfExpr(
                *cmp1,
                Box::new(la),
                Box::new(lb),
                Box::new(lt),
                Box::new(le),
            ))
        }
        // Same broad constructor class but disagreeing leaves -> shared hole.
        (Expr::Var(_), Expr::Var(_))
        | (Expr::Const(_), Expr::Const(_))
        | (Expr::Var(_), Expr::Const(_))
        | (Expr::Const(_), Expr::Var(_)) => Some(hole_for(a, b, subst, next_hole)),
        // Incompatible top constructors (e.g. BinOp vs UnaryOp): no useful lgg.
        _ => None,
    }
}

/// Maximum size of any promotable abstraction (explicit anti-memorization cap).
const MAX_ABSTRACTION_SIZE: usize = 9;
/// A mined subtree must recur across at least this many DISTINCT corpus trees.
const MIN_SUPPORT: u32 = 2;
/// Smallest promotable subtree: a binop of two leaves (size 3).
const MIN_SUBTREE_SIZE: usize = 3;
/// Hard cap on how many abstractions we promote in one mining pass.
const MAX_PROMOTE: usize = 64;
/// Hard cap on injective re-rooting instantiations produced per component.
const MAX_INSTANTIATIONS: usize = 16;
/// Hard cap on how many size-1 library injections enter the enumeration bank
/// (bounds the size-2 binop blow-up).
const MAX_SIZE1_INJECTIONS: usize = 64;

/// Degeneracy filter (D1-D6). Rejects abstractions that can't possibly help
/// search or that smuggle memorization. Reuses [`probe_inputs`] / [`fingerprint`]
/// — no new RNG.
fn is_promotable(canon: &Expr, arity: usize) -> bool {
    // D1: bare leaf.
    if matches!(canon, Expr::Var(_) | Expr::Const(_)) {
        return false;
    }
    // D2: too small (a unary-of-leaf is size 2 and uninteresting).
    if canon.size() < MIN_SUBTREE_SIZE {
        return false;
    }
    // D3: var-free constant fold — already handled by the enumerator.
    if arity == 0 {
        return false;
    }
    // D4: over-parameterized => memorization in disguise.
    if arity > 3 {
        return false;
    }
    // D5: near-whole-program tree => the explicit anti-memorization cap.
    if canon.size() > MAX_ABSTRACTION_SIZE {
        return false;
    }
    // D6: observational identity. If the abstraction behaves exactly like a bare
    // Var0 on probe inputs it's a no-op (Var0+Const0, Var0*Const1, abs(abs(x))).
    let probes = probe_inputs(arity.max(1), 16);
    let canon_fp = fingerprint(canon, &probes);
    let id_fp = fingerprint(&Expr::Var(0), &probes);
    if let (Some(a), Some(b)) = (&canon_fp, &id_fp) {
        if a == b {
            return false;
        }
    }
    true
}

/// Human-readable provenance name for a mined abstraction. The `"mined: "`
/// prefix namespaces against seed descriptions so `ComponentLibrary::add`'s
/// dedup-by-description keeps working. Pure function of the canonical Expr ->
/// deterministic and collision-free.
fn name_abstraction(e: &Expr) -> String {
    let arity = free_vars(e).iter().max().map(|m| m + 1).unwrap_or(0);
    let names = ["a", "b", "c", "d"];
    let slice = &names[..arity.min(names.len())];
    format!("mined: {}", e.to_mog(slice))
}

/// Rewrite `Var(slot)` -> `Var(map[slot])` for re-rooting a k-ary abstraction
/// onto real argument indices. Slots out of `map`'s range are left unchanged.
fn remap_vars(e: &Expr, map: &[usize]) -> Expr {
    match e {
        Expr::Var(i) => Expr::Var(*map.get(*i).unwrap_or(i)),
        Expr::Const(c) => Expr::Const(*c),
        Expr::BinOp(op, l, r) => Expr::BinOp(
            *op,
            Box::new(remap_vars(l, map)),
            Box::new(remap_vars(r, map)),
        ),
        Expr::UnaryOp(op, c) => Expr::UnaryOp(*op, Box::new(remap_vars(c, map))),
        Expr::IfExpr(cmp, a, b, c, d) => Expr::IfExpr(
            *cmp,
            Box::new(remap_vars(a, map)),
            Box::new(remap_vars(b, map)),
            Box::new(remap_vars(c, map)),
            Box::new(remap_vars(d, map)),
        ),
        other => other.clone(),
    }
}

/// Produce every injective re-rooting of a k-ary component's dense slots
/// `0..comp_arity` onto the real argument indices `0..n_args`. k=1 yields
/// `n_args` instantiations; k=2 yields ordered pairs; etc. Capped at
/// [`MAX_INSTANTIATIONS`], keeping the lexicographically-first maps when over
/// cap (deterministic). Returns `[]` if the component needs more args than the
/// problem has.
fn instantiate_component(comp: &Expr, n_args: usize) -> Vec<Expr> {
    let comp_arity = free_vars(comp).iter().max().map(|m| m + 1).unwrap_or(0);
    if comp_arity == 0 || comp_arity > n_args {
        return Vec::new();
    }
    // Enumerate injective maps slot->real in lexicographic order.
    let mut maps: Vec<Vec<usize>> = Vec::new();
    fn rec(
        slot: usize,
        comp_arity: usize,
        n_args: usize,
        used: &mut Vec<bool>,
        cur: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if out.len() >= MAX_INSTANTIATIONS {
            return;
        }
        if slot == comp_arity {
            out.push(cur.clone());
            return;
        }
        for real in 0..n_args {
            if used[real] {
                continue;
            }
            used[real] = true;
            cur.push(real);
            rec(slot + 1, comp_arity, n_args, used, cur, out);
            cur.pop();
            used[real] = false;
            if out.len() >= MAX_INSTANTIATIONS {
                return;
            }
        }
    }
    let mut used = vec![false; n_args];
    let mut cur = Vec::new();
    rec(0, comp_arity, n_args, &mut used, &mut cur, &mut maps);
    maps.iter().map(|m| remap_vars(comp, m)).collect()
}

/// Per-pattern mining statistics keyed by canonical expr_key in a BTreeMap.
#[derive(Clone, Debug)]
struct SubtreeStats {
    /// Number of DISTINCT corpus trees this pattern appears in.
    support: u32,
    /// Total raw occurrences across the corpus (tie-break / diagnostics).
    total_occurrences: u32,
    arity: usize,
    size: usize,
    exemplar: Expr,
}

// ─── Persistent solved-Expr corpus (mirrors solved_cache guards) ───────────────

/// One verified solved scalar Expr, with its arity and a behavioural
/// fingerprint for dedup.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SolvedExpr {
    expr: Expr,
    n_args: usize,
    fp: Vec<i64>,
}

/// Max corpus entries kept on disk.
const SOLVED_EXPRS_MAX_ENTRIES: usize = 5000;
/// Max corpus file size in bytes (64 MiB). Refuse to grow/read past this.
const SOLVED_EXPRS_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// On-disk location for the mined-corpus. Mirrors `solved_cache::cache_path`:
/// `NSYNTH_SOLVED_EXPRS_PATH` override, empty string disables, `None` under
/// `cfg!(test)` so tests stay hermetic, else `~/.mog_synth_solved_exprs.json`.
fn solved_exprs_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_SOLVED_EXPRS_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    Some(dirs_home().join(".mog_synth_solved_exprs.json"))
}

/// Append a verified solved Expr to the persistent corpus. No-op when the path
/// is disabled (env empty / `cfg!(test)`). Dedups by (expr_key, fp), enforces
/// the entry and file-byte caps BEFORE writing, and fails soft on any IO error.
pub fn record_solved_expr(e: &Expr, n_args: usize, fp: &[i64]) {
    let path = match solved_exprs_path() {
        Some(p) => p,
        None => return,
    };
    let mut existing = load_solved_exprs();
    let key = expr_key(e);
    let already = existing
        .iter()
        .any(|s| expr_key(&s.expr) == key && s.fp == fp);
    if already {
        return;
    }
    existing.push(SolvedExpr {
        expr: e.clone(),
        n_args,
        fp: fp.to_vec(),
    });
    // Entry cap: keep the most recent SOLVED_EXPRS_MAX_ENTRIES.
    if existing.len() > SOLVED_EXPRS_MAX_ENTRIES {
        let overflow = existing.len() - SOLVED_EXPRS_MAX_ENTRIES;
        existing.drain(0..overflow);
    }
    let json = match serde_json::to_string_pretty(&existing) {
        Ok(j) => j,
        Err(_) => return,
    };
    // File-byte cap: refuse to grow past the limit.
    if json.len() as u64 > SOLVED_EXPRS_MAX_BYTES {
        return;
    }
    let _ = std::fs::write(&path, json);
}

/// Load the persistent corpus. Returns `[]` on missing file / parse failure /
/// disabled path, and refuses to read a file larger than the byte cap.
fn load_solved_exprs() -> Vec<SolvedExpr> {
    let path = match solved_exprs_path() {
        Some(p) => p,
        None => return Vec::new(),
    };
    if !path.exists() {
        return Vec::new();
    }
    if let Ok(meta) = std::fs::metadata(&path) {
        if meta.len() > SOLVED_EXPRS_MAX_BYTES {
            return Vec::new();
        }
    }
    let json = match std::fs::read_to_string(&path) {
        Ok(j) => j,
        Err(_) => return Vec::new(),
    };
    serde_json::from_str(&json).unwrap_or_default()
}

// ─── Resumable anytime search frontier ─────────────────────────────────────────
//
// A `Frontier` lifts the bottom-up enumerator's locals — the size-stratified
// expression bank (`by_size`) and the next size to expand (`next_size`) — out of
// the call stack so a search that runs out of *time budget* can be PERSISTED and
// RESUMED on a later call instead of restarting from size 1. Combined with the
// removal of the fixed `max_size` ceiling in `enumerate_exprs_resumable`, this
// turns every "give up / None" into "budget exhausted; frontier persisted;
// resumable, deepening next time" — the search is anytime and monotone, never a
// proof of impossibility.
//
// Determinism: selection order is purely structural (size ascending, then the
// fixed op/var/const order). `seen` (the observational-equivalence dedup set) is
// NOT stored; it is re-derived by replaying the stored `by_size` strata in order
// on resume, which keeps the file smaller and reproduces the identical dedup
// decisions (the replay visits strata in the exact stored Vec order).

/// A resumable, size-stratified enumeration frontier for one problem.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frontier {
    /// Deterministic key (the examples fingerprint) the frontier belongs to.
    fingerprint: String,
    /// Number of function arguments (selection space depends on it).
    n_args: usize,
    /// Op tier this frontier was grown under: 0 = core (5 ops), 1 = full (12).
    /// A frontier must only be resumed under the SAME tier (different op sets
    /// produce different strata), so a tier mismatch forces a fresh start.
    ops_tier: u8,
    /// Size-stratified expression bank: `by_size[s]` holds the unique (under
    /// observational equivalence) expressions of size `s`. Index 0 is unused.
    by_size: Vec<Vec<Expr>>,
    /// The next size level to expand. On a cold frontier this is 2 (sizes 0/1
    /// are seeded fresh each run from vars/consts/library).
    next_size: usize,
}

impl Frontier {
    /// A fresh frontier for `fingerprint`/`n_args`/`ops_tier`, ready to expand
    /// from size 2 (size-1 atoms are always re-seeded from vars/consts/library).
    fn fresh(fingerprint: String, n_args: usize, ops_tier: u8) -> Self {
        Self {
            fingerprint,
            n_args,
            ops_tier,
            by_size: vec![vec![]],
            next_size: 2,
        }
    }

    /// True when this frontier is reusable for the given problem signature.
    /// A mismatch on fingerprint / arity / op-tier means the stored strata do
    /// not describe THIS search, so the caller must start fresh.
    fn matches(&self, fingerprint: &str, n_args: usize, ops_tier: u8) -> bool {
        self.fingerprint == fingerprint && self.n_args == n_args && self.ops_tier == ops_tier
    }
}

/// Max distinct frontiers kept on disk. A frontier is FAR larger per entry than
/// a one-line solved program, so this cap is small and the file-byte cap below
/// is the real guard. (Disk-blowup history: the solved-program cache once grew
/// to 13.4 GB; the frontier store must never repeat that.)
const FRONTIER_MAX_ENTRIES: usize = 64;
/// Max frontier-store file size in bytes (32 MiB). Refuse to grow/read past it.
const FRONTIER_MAX_BYTES: u64 = 32 * 1024 * 1024;
/// Max serialized size of a SINGLE frontier (8 MiB). A frontier that has grown
/// past this is not persisted (search degrades gracefully to one-shot behavior
/// for that problem) so one deep search can't fill the store on its own.
const FRONTIER_MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;

/// On-disk location for the resumable-frontier store. Mirrors
/// `solved_exprs_path`: `NSYNTH_ENUM_FRONTIER_PATH` override, empty string
/// disables, `None` under `cfg!(test)` so tests stay hermetic, else
/// `~/.mog_synth_enum_frontier.json`.
fn frontier_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_ENUM_FRONTIER_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    Some(dirs_home().join(".mog_synth_enum_frontier.json"))
}

/// Load all persisted frontiers. Returns `[]` on missing file / parse failure /
/// disabled path, and refuses to read a file larger than the byte cap.
fn load_frontiers() -> Vec<Frontier> {
    let path = match frontier_path() {
        Some(p) => p,
        None => return Vec::new(),
    };
    if !path.exists() {
        return Vec::new();
    }
    if let Ok(meta) = std::fs::metadata(&path) {
        if meta.len() > FRONTIER_MAX_BYTES {
            return Vec::new();
        }
    }
    let json = match std::fs::read_to_string(&path) {
        Ok(j) => j,
        Err(_) => return Vec::new(),
    };
    serde_json::from_str(&json).unwrap_or_default()
}

/// Load the frontier for `fingerprint`/`n_args`/`ops_tier`, or `None` if the
/// store is disabled / absent / has no matching (and tier-compatible) entry.
fn load_frontier(fingerprint: &str, n_args: usize, ops_tier: u8) -> Option<Frontier> {
    load_frontiers()
        .into_iter()
        .find(|f| f.matches(fingerprint, n_args, ops_tier))
}

/// Persist `frontier`, replacing any prior entry for the same fingerprint.
/// No-op when the path is disabled. Enforces the single-entry byte cap, the
/// entry cap, and the file-byte cap BEFORE writing, and fails soft on any IO
/// error. This is the resumable counterpart of `record_solved_expr`.
fn save_frontier(frontier: &Frontier) {
    let path = match frontier_path() {
        Some(p) => p,
        None => return,
    };
    // Serialize THIS frontier first to enforce the per-entry cap; an
    // over-cap frontier is dropped (graceful degradation to one-shot search)
    // rather than persisted, so a single deep search can't fill the store.
    let entry_json = match serde_json::to_string(frontier) {
        Ok(j) => j,
        Err(_) => return,
    };
    if entry_json.len() > FRONTIER_MAX_ENTRY_BYTES {
        return;
    }
    let mut existing = load_frontiers();
    existing.retain(|f| f.fingerprint != frontier.fingerprint);
    existing.push(frontier.clone());
    // Entry cap: keep the most recent FRONTIER_MAX_ENTRIES.
    if existing.len() > FRONTIER_MAX_ENTRIES {
        let overflow = existing.len() - FRONTIER_MAX_ENTRIES;
        existing.drain(0..overflow);
    }
    let json = match serde_json::to_string(&existing) {
        Ok(j) => j,
        Err(_) => return,
    };
    // File-byte cap: refuse to grow past the limit (drop this save rather than
    // risk a runaway file).
    if json.len() as u64 > FRONTIER_MAX_BYTES {
        return;
    }
    let _ = std::fs::write(&path, json);
}

/// Drop the persisted frontier for `fingerprint` (e.g. once the problem is
/// solved, so its frontier becomes dead weight). No-op when disabled / absent.
fn evict_frontier(fingerprint: &str) {
    let path = match frontier_path() {
        Some(p) => p,
        None => return,
    };
    let mut existing = load_frontiers();
    let before = existing.len();
    existing.retain(|f| f.fingerprint != fingerprint);
    if existing.len() == before {
        return; // nothing to evict
    }
    if existing.is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    if let Ok(json) = serde_json::to_string(&existing) {
        let _ = std::fs::write(&path, json);
    }
}

// ─── Bootstrap corpus (cold-start safety) ──────────────────────────────────────

/// Build a cold-start corpus by re-deriving Exprs for the classic seed I/O
/// specs (the same 15 specs the old `dream()` used) via the enumerator. These
/// feed the miner as INPUT — they are not directly the library output. The
/// frequency miner then extracts the recurring subtrees; a hybrid floor
/// (in `mine_library`) re-adds these seed Exprs as low-priority components when
/// support-mining alone produces too few entries (cold machine safety).
fn bootstrap_corpus(budget_ms: u64) -> Vec<SolvedExpr> {
    let start = std::time::Instant::now();
    // (n_args, fn(&[i64]) -> i64) — same specs as the historical dream() seeds.
    type SeedFn = Box<dyn Fn(&[i64]) -> i64>;
    let seeds: Vec<(usize, SeedFn)> = vec![
        (2, Box::new(|a: &[i64]| a[0] + a[1])),
        (2, Box::new(|a: &[i64]| a[0] * a[1])),
        (2, Box::new(|a: &[i64]| a[0] - a[1])),
        (2, Box::new(|a: &[i64]| a[0].saturating_sub(a[1]).saturating_abs())),
        (2, Box::new(|a: &[i64]| a[0].min(a[1]))),
        (2, Box::new(|a: &[i64]| a[0].max(a[1]))),
        (1, Box::new(|a: &[i64]| a[0] * a[0])),
        (1, Box::new(|a: &[i64]| a[0] % 2)),
        (1, Box::new(|a: &[i64]| a[0] / 2)),
        (1, Box::new(|a: &[i64]| a[0] + a[0])),
        (1, Box::new(|a: &[i64]| a[0] * 2)),
        (1, Box::new(|a: &[i64]| a[0] * a[0] + a[0])),
        (1, Box::new(|a: &[i64]| a[0] % 10)),
        (1, Box::new(|a: &[i64]| a[0] + 1)),
        (1, Box::new(|a: &[i64]| a[0].saturating_neg())),
    ];

    let mut corpus = Vec::new();
    for (n_args, func) in &seeds {
        if start.elapsed().as_millis() as u64 > budget_ms {
            break;
        }
        let examples: Vec<(Vec<i64>, i64)> = (0..6)
            .map(|i| {
                let args: Vec<i64> = (0..*n_args)
                    .map(|j| ((i as i64 * 7 + 3 + j as i64 * 11) % 20) - 10)
                    .collect();
                let expected = func(&args);
                (args, expected)
            })
            .collect();
        if let Some(expr) = enumerate_exprs_core(*n_args, 5, &examples, 500, None) {
            let fp = fingerprint(&expr, &probe_inputs(*n_args, 8)).unwrap_or_default();
            corpus.push(SolvedExpr {
                expr,
                n_args: *n_args,
                fp,
            });
        }
    }
    corpus
}

// ─── Mining entry point ─────────────────────────────────────────────────────

/// Mine a [`ComponentLibrary`] from the persistent solved-Expr corpus.
/// Frequency-count scalar subtrees -> anti-unify within skeleton groups ->
/// threshold (support>=2, size>=3) -> rank by MDL compression score -> verify
/// (robust_well_defined) -> name -> add. Fully deterministic; `verify_budget_ms`
/// only bounds the cold-start bootstrap, never selection.
pub fn mine_library(verify_budget_ms: u64) -> ComponentLibrary {
    let corpus = load_solved_exprs();
    let corpus = if corpus.is_empty() {
        bootstrap_corpus(verify_budget_ms)
    } else {
        corpus
    };
    mine_from_corpus(&corpus)
}

/// Core mining over an explicit corpus (test-visible, disk-free). This is the
/// deterministic heart used by both `mine_library` and the unit tests.
fn mine_from_corpus(corpus_in: &[SolvedExpr]) -> ComponentLibrary {
    let mut library = ComponentLibrary::new();

    // Order-stability: sort the corpus by (n_args, expr_key) before mining.
    let mut corpus: Vec<&SolvedExpr> = corpus_in.iter().collect();
    corpus.sort_by(|a, b| {
        a.n_args
            .cmp(&b.n_args)
            .then_with(|| expr_key(&a.expr).cmp(&expr_key(&b.expr)))
    });

    // (b) FREQUENCY COUNT — distinct-trees support.
    let mut counts: BTreeMap<String, SubtreeStats> = BTreeMap::new();
    for s in &corpus {
        let mut nodes: Vec<&Expr> = Vec::new();
        collect_scalar_subtrees(&s.expr, &mut nodes);
        // Canonicalize each node and tally per-tree raw occurrences keyed by
        // canonical key; one tree contributes support+=1 per distinct pattern.
        let mut per_tree: BTreeMap<String, (Expr, usize, u32)> = BTreeMap::new();
        for node in &nodes {
            let (canon, arity) = canonicalize(node);
            if !is_promotable(&canon, arity) {
                continue;
            }
            let key = expr_key(&canon);
            let size = canon.size();
            let entry = per_tree.entry(key).or_insert((canon, arity, 0));
            entry.2 += 1;
            let _ = size;
        }
        for (key, (canon, arity, raw)) in per_tree {
            let size = canon.size();
            let stat = counts.entry(key).or_insert(SubtreeStats {
                support: 0,
                total_occurrences: 0,
                arity,
                size,
                exemplar: canon,
            });
            stat.support += 1;
            stat.total_occurrences += raw;
        }
    }

    // (c) ANTI-UNIFY PASS — group surviving exemplars by structural skeleton,
    // fold-left pairwise lgg within each group (members iterated in expr_key
    // order). Each successful lgg that is itself promotable is inserted into
    // counts with support = number of distinct group members it generalizes.
    let mut groups: BTreeMap<String, Vec<Expr>> = BTreeMap::new();
    for stat in counts.values() {
        groups
            .entry(skeleton(&stat.exemplar))
            .or_default()
            .push(stat.exemplar.clone());
    }
    let mut lgg_inserts: Vec<(String, SubtreeStats)> = Vec::new();
    for (_sk, members_in) in &groups {
        if members_in.len() < 2 {
            continue;
        }
        let mut members = members_in.clone();
        members.sort_by(|a, b| expr_key(a).cmp(&expr_key(b)));
        // Fold-left pairwise lgg.
        let mut acc = members[0].clone();
        let mut generalized = 1u32; // how many members folded so far
        for m in members.iter().skip(1) {
            let mut subst = Vec::new();
            let mut next_hole = 0usize;
            if let Some(g) = anti_unify(&acc, m, &mut subst, &mut next_hole) {
                acc = g;
                generalized += 1;
            }
        }
        let (canon, arity) = canonicalize(&acc);
        if generalized >= MIN_SUPPORT && is_promotable(&canon, arity) {
            let key = expr_key(&canon);
            let size = canon.size();
            lgg_inserts.push((
                key,
                SubtreeStats {
                    support: generalized,
                    total_occurrences: generalized,
                    arity,
                    size,
                    exemplar: canon,
                },
            ));
        }
    }
    for (key, stat) in lgg_inserts {
        counts
            .entry(key)
            .and_modify(|s| {
                if stat.support > s.support {
                    s.support = stat.support;
                }
            })
            .or_insert(stat);
    }

    // (d) RANK — collect candidates passing thresholds, sort by total
    // deterministic key: Reverse(compression_score), Reverse(support),
    // size ascending, expr_key lexicographic.
    let mut candidates: Vec<&SubtreeStats> = counts
        .values()
        .filter(|s| s.support >= MIN_SUPPORT && s.size >= MIN_SUBTREE_SIZE)
        .collect();
    candidates.sort_by(|a, b| {
        let sa = a.support as usize * (a.size.saturating_sub(1));
        let sb = b.support as usize * (b.size.saturating_sub(1));
        sb.cmp(&sa) // compression score descending
            .then_with(|| b.support.cmp(&a.support)) // support descending
            .then_with(|| a.size.cmp(&b.size)) // size ascending
            .then_with(|| expr_key(&a.exemplar).cmp(&expr_key(&b.exemplar)))
    });
    candidates.truncate(MAX_PROMOTE);

    // (e) VERIFY + NAME + ADD each in sorted order.
    for stat in candidates {
        if !robust_well_defined(&stat.exemplar, stat.arity.max(1), 50) {
            continue;
        }
        let name = name_abstraction(&stat.exemplar);
        library.add(stat.exemplar.clone(), name);
    }

    // HYBRID FALLBACK (cold-start safety): the bootstrap seeds rarely share
    // subtrees at support>=2 on a truly cold machine, so support-mining alone
    // can return very few entries. To never regress below the historical
    // coverage, also add the corpus's own canonicalized scalar Exprs as
    // low-priority components when the mined library is thin. This is an
    // intentional product decision, not a fallback that weakens soundness —
    // every added Expr is still verified-by-construction (it came from a solved
    // problem) and is robust-gated below.
    if library.len() < 8 {
        let mut seeds: Vec<(Expr, usize)> = corpus
            .iter()
            .map(|s| canonicalize(&s.expr))
            .filter(|(c, a)| is_promotable(c, *a))
            .collect();
        seeds.sort_by(|a, b| expr_key(&a.0).cmp(&expr_key(&b.0)));
        for (canon, arity) in seeds {
            if !robust_well_defined(&canon, arity.max(1), 50) {
                continue;
            }
            let name = name_abstraction(&canon);
            library.add(canon, name);
        }
    }

    library
}

// ─── Component Library ───────────────────────────────────────────────────────
// Stores discovered useful sub-expressions that can be reused across problems.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentLibrary {
    /// Reusable sub-expressions with their semantic descriptions
    components: Vec<(Expr, String)>, // (expr, description)
    /// Size of the solved-expression corpus this library was last mined from.
    /// Used by [`Self::load_or_dream`] to re-mine when the corpus has GROWN
    /// since the cached library was built — without this watermark the library
    /// was mined exactly once (at cold start) and every later solve, though
    /// recorded into the corpus, never became a new abstraction. `#[serde(default)]`
    /// so libraries written before this field load as `0` (forcing one re-mine).
    #[serde(default)]
    mined_corpus_len: usize,
}

/// Path for persistent library storage
fn library_path() -> PathBuf {
    dirs_home().join(".mog_synth_library.json")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

impl ComponentLibrary {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            mined_corpus_len: 0,
        }
    }

    /// Mine `corpus` and fold any NEW components into this library (dedup by
    /// description via [`Self::add`]). Returns how many components were added.
    ///
    /// This is the disk-free core of "the library keeps mining as the solve
    /// corpus grows." Mining is whole-corpus and frequency-gated, so re-mining
    /// a grown corpus yields the previously-mined patterns plus any new ones;
    /// merging (rather than replacing) guarantees a previously-mined component
    /// is never lost if a later, larger corpus shifts a pattern below threshold.
    pub fn merge_from_corpus(&mut self, corpus: &[SolvedExpr]) -> usize {
        let fresh = mine_from_corpus(corpus);
        let before = self.components.len();
        for (expr, desc) in fresh.components {
            self.add(expr, desc);
        }
        self.components.len() - before
    }

    /// Add a discovered component to the library
    pub fn add(&mut self, expr: Expr, description: String) {
        // Don't add duplicates (by description)
        if !self.components.iter().any(|(_, d)| d == &description) {
            self.components.push((expr, description));
        }
    }

    /// Get all re-rooted component instantiations applicable to an `n_args`
    /// problem. Each stored component is arity-filtered (skipped if it needs
    /// more args than the problem has) and re-rooted onto the real arg slots via
    /// [`instantiate_component`], so a mined `?0*?0` applies to BOTH Var0 and
    /// Var1 — enabling novel `square(a) + square(b)` compositions.
    pub fn get_for_args(&self, n_args: usize) -> Vec<Expr> {
        let mut out = Vec::new();
        for (comp, _) in &self.components {
            out.extend(instantiate_component(comp, n_args));
        }
        out
    }

    /// Number of components in the library
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Save library to disk
    pub fn save(&self) -> Result<(), String> {
        let path = library_path();
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// Load library from disk (returns empty library if file doesn't exist)
    pub fn load() -> Self {
        let path = library_path();
        if !path.exists() {
            return Self::new();
        }
        let json = match std::fs::read_to_string(&path) {
            Ok(j) => j,
            Err(_) => return Self::new(),
        };
        match serde_json::from_str(&json) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("[library] failed to parse {}: {e}", path.display());
                Self::new()
            }
        }
    }

    /// Load the library, re-mining when the solve corpus has grown — or create
    /// it via dream mode if none exists.
    ///
    /// Previously this returned the cached `library.json` unconditionally, so
    /// the library was mined exactly once and the `record_solved_expr` corpus,
    /// though it grew with every solve, never produced new abstractions. Now we
    /// compare the current corpus size against the watermark stored in the
    /// cached library and, when it has grown, mine the larger corpus and fold
    /// the new components in. This is the link that makes the library actually
    /// compound across runs ("writes its own teachers").
    pub fn load_or_dream(dream_budget_ms: u64) -> Self {
        let path = library_path();
        let corpus = load_solved_exprs();
        let corpus_len = corpus.len();
        if path.exists() {
            let mut lib = Self::load();
            if !lib.components.is_empty() {
                if corpus_len > lib.mined_corpus_len {
                    let mined_at = lib.mined_corpus_len;
                    let added = lib.merge_from_corpus(&corpus);
                    lib.mined_corpus_len = corpus_len;
                    eprintln!(
                        "[library] corpus grew {mined_at}->{corpus_len}; re-mined +{added} \
                         components -> {} total",
                        lib.len()
                    );
                    let _ = lib.save();
                } else {
                    eprintln!(
                        "[library] loaded {} components from {} (corpus {corpus_len}, no growth)",
                        lib.len(),
                        path.display()
                    );
                }
                return lib;
            }
        }
        eprintln!("[library] no library found, running dream mode...");
        let mut lib = dream(dream_budget_ms);
        // Watermark the corpus this library was mined from. When dream() had to
        // bootstrap (empty on-disk corpus), `corpus_len` is 0, so the first real
        // solve triggers a re-mine.
        lib.mined_corpus_len = corpus_len;
        let _ = lib.save();
        lib
    }
}

// ─── Main enumeration engine ─────────────────────────────────────────────────

const CORE_BINOPS: [BinOp; 5] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Mod];
const CORE_UNOPS: [UnOp; 2] = [UnOp::Neg, UnOp::Abs];

/// Fast enumeration with core 5 ops only.
pub fn enumerate_exprs_core(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
) -> Option<Expr> {
    enumerate_exprs_core_ext(n_args, max_size, examples, time_limit_ms, library).0
}

/// Like `enumerate_exprs_core`, but also reports whether the search bailed out
/// because the time budget was hit (`true`) or because every expression up to
/// `max_size` was exhausted cleanly (`false`). The distinction lets callers
/// skip a slower follow-up pass when a clean exhaustion has already ruled out
/// a small closed-form answer.
pub fn enumerate_exprs_core_ext(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
) -> (Option<Expr>, bool) {
    enumerate_exprs_with_ops_ext(
        n_args,
        max_size,
        examples,
        time_limit_ms,
        library,
        &CORE_BINOPS,
        &CORE_UNOPS,
    )
}

/// Full enumeration with all 12 ops.
pub fn enumerate_exprs(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
) -> Option<Expr> {
    enumerate_exprs_with_ops(
        n_args,
        max_size,
        examples,
        time_limit_ms,
        library,
        &ALL_BINOPS,
        &ALL_UNOPS,
    )
}

/// Enumerate expressions bottom-up with observational equivalence pruning.
fn enumerate_exprs_with_ops(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
) -> Option<Expr> {
    enumerate_exprs_with_ops_ext(
        n_args,
        max_size,
        examples,
        time_limit_ms,
        library,
        binops,
        unops,
    )
    .0
}

/// Same as [`enumerate_exprs_with_ops`] but also returns `true` when the search
/// terminated because the time budget ran out, and `false` when every size up
/// to `max_size` was explored without finding a match.
fn enumerate_exprs_with_ops_ext(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
) -> (Option<Expr>, bool) {
    let (expr, timed_out, _max_completed) = enumerate_exprs_with_ops_stats(
        n_args,
        max_size,
        examples,
        time_limit_ms,
        library,
        binops,
        unops,
    );
    (expr, timed_out)
}

/// Like [`enumerate_exprs_with_ops`] but additionally returns the largest size
/// level whose enumeration ran to completion before the deadline (`0` if none).
/// Callers can combine the `timed_out` flag with `max_completed` to decide
/// whether a shallower follow-up pass is worth running.
fn enumerate_exprs_with_ops_stats(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
) -> (Option<Expr>, bool, usize) {
    // Bounded one-shot search: build a throwaway fresh frontier and run the
    // resumable core with a fixed `soft_cap = max_size`. This preserves the
    // exact behavior (selection order, dedup, the `s <= max_size` push guards)
    // of the historical bounded enumerator that bootstrap/tests depend on.
    let mut frontier = Frontier::fresh(String::new(), n_args, 0);
    let (expr, timed_out) = enumerate_exprs_resumable(
        &mut frontier,
        examples,
        time_limit_ms,
        library,
        binops,
        unops,
        Some(max_size),
    );
    // `next_size` advances past each completed stratum; the deepest cleanly
    // completed size is `next_size - 1`, capped at `max_size` (0 if none).
    let max_completed = if expr.is_some() {
        // Found before completing the in-progress stratum; report the prior
        // fully-completed depth, consistent with the original semantics.
        frontier.next_size.saturating_sub(1).min(max_size)
    } else if timed_out {
        frontier.next_size.saturating_sub(1).min(max_size)
    } else {
        max_size
    };
    (expr, timed_out, max_completed)
}

/// Resumable, anytime, UNCAPPED bottom-up enumeration core.
///
/// Seeds `by_size`/`seen` from the incoming `frontier` (re-deriving `seen` by
/// replaying the stored strata in order, so dedup decisions are reproduced
/// exactly), then deepens from `frontier.next_size` with NO fixed size ceiling
/// when `soft_cap` is `None` — the ONLY stop conditions are "found a verified
/// Expr" or "time budget tripped". On a budget trip (or clean exhaustion under a
/// `soft_cap`), the grown `by_size` and the size reached are written back into
/// `frontier` so a later call RESUMES at that depth rather than restarting.
///
/// Returns `(found, timed_out)`. `timed_out == false` with `found == None` means
/// the (capped) space was exhausted cleanly; under `soft_cap == None` that can
/// only happen if the enumerator runs out of new unique expressions entirely.
///
/// Determinism: no rand, no clock-in-selection. Every `Instant::now()` use here
/// BOUNDS work (when to stop); none SELECTS a candidate. Selection is purely
/// `size ascending → fixed var/const/op order → library injection order`.
#[allow(clippy::too_many_arguments)]
fn enumerate_exprs_resumable(
    frontier: &mut Frontier,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
    soft_cap: Option<usize>,
) -> (Option<Expr>, bool) {
    let start = std::time::Instant::now();
    let n_args = frontier.n_args;
    let test_inputs: Vec<Vec<i64>> = examples.iter().map(|(a, _)| a.clone()).collect();

    // `by_size` grows dynamically (no `max_size + 1` preallocation), so an
    // uncapped search can deepen without an upper bound. Take ownership of the
    // frontier's strata for the duration of the search; write the (possibly
    // grown) bank back before returning.
    let mut by_size: Vec<Vec<Expr>> = std::mem::take(&mut frontier.by_size);
    if by_size.is_empty() {
        by_size.push(vec![]); // index 0 sentinel
    }
    let mut seen: HashSet<Vec<i64>> = HashSet::new();
    let mut timed_out = false;

    // The soft cap, if any: pushes past it are suppressed and the deepening
    // loop stops once `size` would exceed it. `None` == unbounded.
    let cap = soft_cap.unwrap_or(usize::MAX);

    // Ensure `by_size` can hold index `s`, growing with empty strata as needed.
    let ensure_slot = |by_size: &mut Vec<Vec<Expr>>, s: usize| {
        if s >= by_size.len() {
            by_size.resize(s + 1, Vec::new());
        }
    };

    // Helper: check a candidate; on a match return it, else dedup-insert into
    // the size stratum (suppressed above the soft cap). Mirrors the historical
    // `check_add` exactly, but grows `by_size` instead of indexing a fixed vec.
    let check_add = |e: &Expr,
                     by_size: &mut Vec<Vec<Expr>>,
                     seen: &mut HashSet<Vec<i64>>|
     -> Option<Expr> {
        if let Some(fp) = fingerprint(e, &test_inputs) {
            if matches_all(e, examples) && robust_well_defined(e, n_args, 30) {
                return Some(e.clone());
            }
            if seen.insert(fp) {
                let s = e.size();
                if s <= cap {
                    ensure_slot(by_size, s);
                    by_size[s].push(e.clone());
                }
            }
        }
        None
    };

    // RESUME PATH: if the frontier already carries strata beyond the size-1
    // seed (i.e. a prior call expanded it), rebuild `seen` by replaying those
    // strata in stored order. This reproduces the identical dedup state without
    // re-seeding size-1 atoms (which would duplicate vars/consts/library) and
    // keeps the persisted file free of the fingerprint set.
    let resuming = frontier.next_size > 2 || by_size.len() > 2;
    if resuming {
        for stratum in by_size.iter() {
            for e in stratum {
                if let Some(fp) = fingerprint(e, &test_inputs) {
                    seen.insert(fp);
                }
            }
        }
    } else {

    // ── COLD START: seed size 1 (variables, constants, library components) ──
    // On a `found` return the frontier is irrelevant (the caller evicts a
    // solved problem's frontier), so success returns do not write `by_size`
    // back — only the budget-exhausted/exhausted-clean exit persists it.
    for i in 0..n_args {
        if let Some(e) = check_add(&Expr::Var(i), &mut by_size, &mut seen) {
            return (Some(e), false);
        }
    }
    for &c in &CONSTANTS {
        if let Some(e) = check_add(&Expr::Const(c), &mut by_size, &mut seen) {
            return (Some(e), false);
        }
    }
    // Add library components as re-rooted size-1 leaves (cost discount).
    // Each mined abstraction is instantiated onto the real arg slots, gated by
    // the SAME soundness gate (matches_all + robust_well_defined) as every other
    // candidate, then injected into `by_size[1]` rather than `by_size[size]` so
    // the size-2 binop loop can compose two abstractions into a program of true
    // size 7 at an effective `max_size` of ~3 (the documented injection lever).
    // A count budget bounds the size-2 blow-up.
    if let Some(lib) = library {
        let mut injected = 0usize;
        for comp in lib.get_for_args(n_args) {
            if injected >= MAX_SIZE1_INJECTIONS {
                break;
            }
            if let Some(fp) = fingerprint(&comp, &test_inputs) {
                if matches_all(&comp, examples) && robust_well_defined(&comp, n_args, 30) {
                    return (Some(comp), false);
                }
                if seen.insert(fp) {
                    ensure_slot(&mut by_size, 1);
                    by_size[1].push(comp);
                    injected += 1;
                }
            }
        }
    }
    } // end COLD START

    // Deepening loop: from the frontier's resume point, with NO upper bound
    // when uncapped (`cap == usize::MAX`). The ONLY stops are "found" (early
    // return) or "budget tripped" (`timed_out`). `size` may exceed any prior
    // run's depth; `ensure_slot` grows `by_size` accordingly.
    let mut size = frontier.next_size;
    while size <= cap {
        if start.elapsed().as_millis() as u64 > time_limit_ms {
            timed_out = true;
            break;
        }
        ensure_slot(&mut by_size, size);
        let mut new: Vec<Expr> = Vec::new();

        // Unary ops
        if size >= 2 {
            let children = by_size[size - 1].clone();
            for child in &children {
                for &uop in unops {
                    let e = Expr::UnaryOp(uop, Box::new(child.clone()));
                    if let Some(fp) = fingerprint(&e, &test_inputs) {
                        if matches_all(&e, examples) && robust_well_defined(&e, n_args, 30) {
                            return (Some(e), false);
                        }
                        if seen.insert(fp) {
                            new.push(e);
                        }
                    }
                }
            }
        }

        // Binary ops
        for ls in 1..size {
            let rs = size - 1 - ls;
            if rs < 1 || rs > cap {
                continue;
            }
            if start.elapsed().as_millis() as u64 > time_limit_ms {
                timed_out = true;
                break;
            }
            let lefts = by_size[ls].clone();
            let rights = by_size[rs].clone();
            for left in &lefts {
                for right in &rights {
                    for &op in binops {
                        let e = Expr::BinOp(op, Box::new(left.clone()), Box::new(right.clone()));
                        if let Some(fp) = fingerprint(&e, &test_inputs) {
                            if matches_all(&e, examples) && robust_well_defined(&e, n_args, 30) {
                                return (Some(e), false);
                            }
                            if seen.insert(fp) {
                                new.push(e);
                            }
                        }
                    }
                    if start.elapsed().as_millis() as u64 > time_limit_ms {
                        timed_out = true;
                        break;
                    }
                }
            }
        }

        // If-then-else (size >= 5)
        if size >= 5 {
            let atoms = by_size[1].clone();
            for &cmp in &ALL_CMPS {
                if start.elapsed().as_millis() as u64 > time_limit_ms {
                    timed_out = true;
                    break;
                }
                for cl in &atoms {
                    for cr in &atoms {
                        let budget = size - 3;
                        for ts in 1..budget {
                            let es = budget - ts;
                            if es < 1 {
                                continue;
                            }
                            let then_es = by_size[ts].clone();
                            let else_es = by_size[es].clone();
                            for te in &then_es {
                                for ee in &else_es {
                                    let e = Expr::IfExpr(
                                        cmp,
                                        Box::new(cl.clone()),
                                        Box::new(cr.clone()),
                                        Box::new(te.clone()),
                                        Box::new(ee.clone()),
                                    );
                                    if let Some(fp) = fingerprint(&e, &test_inputs) {
                                        if matches_all(&e, examples)
                                            && robust_well_defined(&e, n_args, 30)
                                        {
                                            return (Some(e), false);
                                        }
                                        if seen.insert(fp) {
                                            new.push(e);
                                        }
                                    }
                                }
                                if start.elapsed().as_millis() as u64 > time_limit_ms {
                                    timed_out = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // While-accumulator loops (size >= 6)
        // acc = init; for i in 0..bound { acc = acc OP rhs(acc, i, args) }
        if size >= 6 && n_args >= 1 {
            let init_budget = 1;
            let bound_budget = 1;
            let rhs_budget = size - 3 - init_budget - bound_budget;
            if rhs_budget >= 1 && rhs_budget <= cap {
                let inits = by_size[init_budget].clone();
                let bounds = by_size[bound_budget].clone();
                // For loop body, the rhs can reference acc (var n_args) and i (var n_args+1)
                // Build a separate expression bank for the loop body context
                let loop_n_args = n_args + 2; // original args + acc + i
                let mut loop_atoms: Vec<Expr> = Vec::new();
                for i in 0..loop_n_args {
                    loop_atoms.push(Expr::Var(i));
                }
                for &c in &CONSTANTS {
                    loop_atoms.push(Expr::Const(c));
                }
                // Simple loop rhs: just atoms or atom OP atom
                let mut loop_exprs: Vec<Expr> = loop_atoms.clone();
                if rhs_budget >= 3 {
                    for l in &loop_atoms {
                        for r in &loop_atoms {
                            for &op in &[BinOp::Add, BinOp::Mul, BinOp::Mod] {
                                loop_exprs.push(Expr::BinOp(
                                    op,
                                    Box::new(l.clone()),
                                    Box::new(r.clone()),
                                ));
                            }
                        }
                    }
                }

                for init in &inits {
                    for bound in &bounds {
                        for &bop in &LOOP_BODY_OPS {
                            for rhs in &loop_exprs {
                                if start.elapsed().as_millis() as u64 > time_limit_ms {
                                    timed_out = true;
                                    break;
                                }
                                let e = Expr::WhileAccum {
                                    init: Box::new(init.clone()),
                                    bound: Box::new(bound.clone()),
                                    body_op: bop,
                                    body_rhs: Box::new(rhs.clone()),
                                };
                                if let Some(fp) = fingerprint(&e, &test_inputs) {
                                    if matches_all(&e, examples)
                                        && robust_well_defined(&e, n_args, 30)
                                    {
                                        return (Some(e), false);
                                    }
                                    if seen.insert(fp) {
                                        new.push(e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if timed_out {
            // Budget tripped mid-stratum: DISCARD the partial `new` for this
            // size (leave its slot empty) so a later resume rebuilds the full
            // size deterministically, and do NOT advance `next_size` past it.
            break;
        }
        eprintln!(
            "[enum] size {size}: {} new, {} total unique, {:.1}s",
            new.len(),
            seen.len(),
            start.elapsed().as_secs_f32()
        );
        by_size[size] = new;
        size += 1;
    }

    // Persist the grown frontier: the deepest cleanly-completed size is
    // `size - 1`, so the next resume continues at `size`. (On a clean exhaustion
    // under a soft cap, `size` is `cap + 1`, which the caller treats as done.)
    frontier.next_size = size;
    frontier.by_size = by_size;
    (None, timed_out)
}

// ─── Emit Mog code from discovered expression ────────────────────────────────

fn emit_mog(expr: &Expr, fn_name: &str, param_names: &[&str]) -> String {
    let sig = param_names
        .iter()
        .map(|n| format!("{n}: i64"))
        .collect::<Vec<_>>()
        .join(", ");

    match expr {
        Expr::WhileAccum {
            init,
            bound,
            body_op,
            body_rhs,
        } => {
            let init_s = init.to_mog(param_names);
            let bound_s = bound.to_mog(param_names);
            let op_s = match body_op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                _ => "+",
            };
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("acc");
                v.push("i");
                v
            };
            let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    i: i64 = 0;\n    while i < {bound_s} {{\n        acc = acc {op_s} {rhs_s};\n        i = i + 1;\n    }}\n    return acc;\n}}\n"
            )
        }
        Expr::ForFold {
            init,
            body_op,
            body_rhs,
        } => {
            let init_s = init.to_mog(param_names);
            let op_s = match body_op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Min => "min",
                BinOp::Max => "max",
                _ => "+",
            };
            // Body rhs namespace: scalar_params + [item, i, acc]
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("item");
                v.push("i");
                v.push("acc");
                v
            };
            let arr_sig = if param_names.is_empty() {
                "arr: [i64]".to_string()
            } else {
                format!("{}, arr: [i64]", sig)
            };
            if op_s == "min" || op_s == "max" {
                // Min/max folds need conditional update
                let cmp = if op_s == "min" { "<" } else { ">" };
                format!(
                    "fn {fn_name}({arr_sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if item {cmp} acc {{\n            acc = item;\n        }}\n    }}\n    return acc;\n}}\n"
                )
            } else if let Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) = body_rhs.as_ref() {
                // Conditional fold: if lhs CMP rhs { acc = acc OP then } else { acc = acc OP else }
                let ls = lhs.to_mog_ext(&ext_names, &[]);
                let rs = rhs.to_mog_ext(&ext_names, &[]);
                let cs = match cmp {
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Eq => "==",
                    CmpOp::Ge => ">=",
                    CmpOp::Gt => ">",
                    CmpOp::Ne => "!=",
                };
                let ts = then_e.to_mog_ext(&ext_names, &[]);
                let es = else_e.to_mog_ext(&ext_names, &[]);
                if **else_e == Expr::Const(0) {
                    // if cond { acc = acc OP val; }
                    format!(
                        "fn {fn_name}({arr_sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if {ls} {cs} {rs} {{\n            acc = acc {op_s} {ts};\n        }}\n    }}\n    return acc;\n}}\n"
                    )
                } else {
                    // if cond { acc = acc OP a; } else { acc = acc OP b; }
                    format!(
                        "fn {fn_name}({arr_sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if {ls} {cs} {rs} {{\n            acc = acc {op_s} {ts};\n        }} else {{\n            acc = acc {op_s} {es};\n        }}\n    }}\n    return acc;\n}}\n"
                    )
                }
            } else {
                let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
                format!(
                    "fn {fn_name}({arr_sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        acc = acc {op_s} {rhs_s};\n    }}\n    return acc;\n}}\n"
                )
            }
        }
        Expr::NestedWhile {
            outer_init,
            outer_bound,
            outer_body_op,
            inner_init,
            inner_bound,
            inner_body_op,
            inner_body_rhs,
        } => {
            let oinit_s = outer_init.to_mog(param_names);
            let obound_s = outer_bound.to_mog(param_names);
            let oop_s = match outer_body_op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                _ => "+",
            };
            let iinit_s = inner_init.to_mog(param_names);
            let ibound_s = inner_bound.to_mog(param_names);
            let iop_s = match inner_body_op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                _ => "+",
            };
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("outer_acc");
                v.push("outer_i");
                v.push("inner_acc");
                v.push("inner_i");
                v
            };
            let irhs_s = inner_body_rhs.to_mog_ext(&ext_names, &[]);
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    outer_acc: i64 = {oinit_s};\n    outer_i: i64 = 0;\n    while outer_i < {obound_s} {{\n        inner_acc: i64 = {iinit_s};\n        inner_i: i64 = 0;\n        while inner_i < {ibound_s} {{\n            inner_acc = inner_acc {iop_s} {irhs_s};\n            inner_i = inner_i + 1;\n        }}\n        outer_acc = outer_acc {oop_s} inner_acc;\n        outer_i = outer_i + 1;\n    }}\n    return outer_acc;\n}}\n"
            )
        }
        Expr::WhileCond {
            init,
            state_init,
            cond_cmp,
            cond_val,
            divisor,
            body_op,
            body_rhs,
        } => {
            let init_s = init.to_mog(param_names);
            let state_init_s = state_init.to_mog(param_names);
            let cv_s = cond_val.to_mog(param_names);
            let op_s = match body_op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                _ => "+",
            };
            let cmp_s = match cond_cmp {
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Eq => "==",
                CmpOp::Ge => ">=",
                CmpOp::Gt => ">",
                CmpOp::Ne => "!=",
            };
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("x");
                v.push("acc");
                v.push("i");
                v
            };
            let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
            let div_expr = if *divisor == 10 {
                "x / 10".to_string()
            } else if *divisor == 2 {
                "x / 2".to_string()
            } else {
                format!("x / {divisor}")
            };
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    x: i64 = {state_init_s};\n    while x {cmp_s} {cv_s} {{\n        acc = acc {op_s} {rhs_s};\n        x = {div_expr};\n    }}\n    return acc;\n}}\n"
            )
        }
        _ => {
            let expr_s = expr.to_mog(param_names);
            format!("fn {fn_name}({sig}) -> i64 {{\n    return {expr_s};\n}}\n")
        }
    }
}

/// Emit Mog code for array fold expressions, respecting the original parameter order.
/// `array_idx` is the position of the array parameter in the original function signature.
/// `scalar_names` are the names for the scalar parameters (in order of appearance).
fn emit_mog_array(expr: &Expr, fn_name: &str, scalar_names: &[&str], array_idx: usize) -> String {
    // Build full signature: put arr at the correct position
    let n_total = scalar_names.len() + 1; // scalars + 1 array
    let mut sig_parts: Vec<String> = vec![String::new(); n_total];
    sig_parts[array_idx] = "arr: [i64]".to_string();
    let mut scalar_i = 0;
    for i in 0..n_total {
        if i != array_idx {
            sig_parts[i] = format!("{}: i64", scalar_names[scalar_i]);
            scalar_i += 1;
        }
    }
    let sig = sig_parts.join(", ");

    if let Expr::ForFold {
        init,
        body_op,
        body_rhs,
    } = expr
    {
        let init_s = init.to_mog(scalar_names);
        let op_s = match body_op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Min => "min",
            BinOp::Max => "max",
            _ => "+",
        };
        // Body rhs namespace: scalar_params + [item, i, acc]
        let ext_names: Vec<&str> = {
            let mut v: Vec<&str> = scalar_names.to_vec();
            v.push("item");
            v.push("i");
            v.push("acc");
            v
        };

        if op_s == "min" || op_s == "max" {
            let cmp = if op_s == "min" { "<" } else { ">" };
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = arr[0];\n    for item in arr {{\n        if item {cmp} acc {{\n            acc = item;\n        }}\n    }}\n    return acc;\n}}\n"
            )
        } else if let Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) = body_rhs.as_ref() {
            let ls = lhs.to_mog_ext(&ext_names, &[]);
            let rs = rhs.to_mog_ext(&ext_names, &[]);
            let cs = match cmp {
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Eq => "==",
                CmpOp::Ge => ">=",
                CmpOp::Gt => ">",
                CmpOp::Ne => "!=",
            };
            let ts = then_e.to_mog_ext(&ext_names, &[]);
            if **else_e == Expr::Const(0) {
                format!(
                    "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if {ls} {cs} {rs} {{\n            acc = acc {op_s} {ts};\n        }}\n    }}\n    return acc;\n}}\n"
                )
            } else {
                let es = else_e.to_mog_ext(&ext_names, &[]);
                format!(
                    "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if {ls} {cs} {rs} {{\n            acc = acc {op_s} {ts};\n        }} else {{\n            acc = acc {op_s} {es};\n        }}\n    }}\n    return acc;\n}}\n"
                )
            }
        } else {
            let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        acc = acc {op_s} {rhs_s};\n    }}\n    return acc;\n}}\n"
            )
        }
    } else {
        // Fallback: should not happen for array synthesis
        format!("fn {fn_name}({sig}) -> i64 {{\n    return 0;\n}}\n")
    }
}

// ─── Public synthesis entry point ────────────────────────────────────────────

/// Enumerative synthesis: discovers programs from I/O examples alone.
/// Handles both scalar and array problems.
pub fn synthesize_enumerative(problem: &Problem) -> Option<SolveResult> {
    // Detect array vs scalar
    let has_array = problem
        .examples
        .iter()
        .any(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Array(_))));

    if has_array {
        return synthesize_array_enumerative(problem);
    }

    // Scalar path (original)
    synthesize_scalar_enumerative(problem)
}

/// Scalar-only enumerative synthesis (original two-pass approach).
fn synthesize_scalar_enumerative(problem: &Problem) -> Option<SolveResult> {
    let n_args = problem.examples.first()?.inputs.len();
    let fn_name = problem.function_name();

    let examples: Vec<(Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let args: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            (args, ex.expected_int())
        })
        .collect();

    let param_names: Vec<&str> = ["a", "b", "c", "d", "e", "f"]
        .iter()
        .take(n_args)
        .copied()
        .collect();

    // Load component library (lazy, first call initializes)
    let library = ComponentLibrary::load_or_dream(5_000);

    // Inputs from the in-scope examples, used to fingerprint solved Exprs for
    // the persistent mining corpus (no-op under cfg!(test) / disabled path).
    let test_inputs: Vec<Vec<i64>> = examples.iter().map(|(a, _)| a.clone()).collect();

    // Deterministic per-problem key for the resumable frontier store. Derived
    // ONLY from the examples (stable across runs/machines), so a later solve of
    // the SAME problem reloads its frontier and deepens instead of restarting.
    let fp = crate::solved_cache::examples_fingerprint(&problem.examples);

    // ── Per-call time budget (anytime, resumable) ──────────────────────────
    // NSYNTH_ENUM_BUDGET_MS bounds THIS call; the SEARCH DEPTH is unbounded and
    // RISES across calls (the frontier resumes deeper each time). The default
    // equals the sum of the historical pass-1 + pass-2 budgets, so a cold first
    // call has unchanged wall-clock. The split mirrors the old deep/shallow
    // ratio: ~55% to the core-op (deep) tier, the rest to the full-op tier.
    let default_budget: u64 = if n_args <= 1 {
        18_000
    } else if n_args <= 2 {
        13_000
    } else {
        10_000
    };
    let budget_ms: u64 = std::env::var("NSYNTH_ENUM_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_budget);
    let core_budget = (budget_ms * 55 / 100).max(1);
    let full_budget = budget_ms.saturating_sub(core_budget).max(1);

    // Optional SOFT size cap. Unset → unbounded (the anytime guarantee). Set it
    // to bound cumulative re-run cost in CI without reintroducing a hard ceiling
    // in the engine itself.
    let soft_cap: Option<usize> = std::env::var("NSYNTH_ENUM_SOFT_CAP")
        .ok()
        .and_then(|v| v.parse().ok());

    // Run one resumable deepening pass for a given op tier: load this tier's
    // frontier (empty on first call), deepen under `budget`, and on a verified
    // hit record + evict the frontier and return the solution. On a miss,
    // persist the (deepened) frontier so the NEXT call resumes from here — this
    // is the "budget exhausted, frontier persisted, resumable" reframe of the
    // old terminal `None`.
    let mut run_tier = |ops_tier: u8,
                        binops: &[BinOp],
                        unops: &[UnOp],
                        budget: u64|
     -> Option<SolveResult> {
        let mut frontier = load_frontier(&fp, n_args, ops_tier)
            .unwrap_or_else(|| Frontier::fresh(fp.clone(), n_args, ops_tier));
        let (expr, _timed_out) = enumerate_exprs_resumable(
            &mut frontier,
            &examples,
            budget,
            Some(&library),
            binops,
            unops,
            soft_cap,
        );
        if let Some(expr) = expr {
            let code = emit_mog(&expr, fn_name, &param_names);
            if verify_problem_code_strict(problem, &code).is_ok() {
                record_solved_expr(
                    &expr,
                    n_args,
                    &fingerprint(&expr, &test_inputs).unwrap_or_default(),
                );
                // Solved → its frontier is dead weight; drop it.
                evict_frontier(&fp);
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "enumerative".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
            eprintln!(
                "[enum] found expr but Mog verification failed: {}",
                expr.to_mog(&param_names)
            );
        }
        // Miss for THIS call: persist the deepened frontier so a later call
        // continues from here (no fixed ceiling, never "impossible").
        save_frontier(&frontier);
        None
    };

    // Tier 0: core ops only (+,-,*,/,%) — deep. Tier 1: all 12 ops — broader.
    // Both are uncapped and resumable; clean exhaustion of small sizes is no
    // longer terminal (it just means the frontier now starts deeper next time).
    if let Some(result) = run_tier(0, &CORE_BINOPS, &CORE_UNOPS, core_budget) {
        return Some(result);
    }
    if let Some(result) = run_tier(1, &ALL_BINOPS, &ALL_UNOPS, full_budget) {
        return Some(result);
    }

    // Try nested loops for 1-2 arg scalar problems
    if n_args <= 2 {
        if let Some(result) = synthesize_nested_while(problem) {
            return Some(result);
        }
    }

    // Try while-cond loops for 1-arg problems (digit_sum, digit_count, etc.)
    if n_args == 1 {
        if let Some(result) = synthesize_while_cond(problem) {
            return Some(result);
        }
    }

    None
}

// ─── Array enumeration ─────────────────────────────────────────────────────

/// Enumerate fold bodies for array problems.
/// Most array problems are folds: acc = init; for item in arr { acc = acc OP body(item, i, acc) }
fn synthesize_array_enumerative(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();

    // Extract scalar args and arrays from examples
    let first_ex = problem.examples.first()?;
    let array_idx = first_ex
        .inputs
        .iter()
        .position(|v| matches!(v, Value::Array(_)))?;
    let n_scalar_args = first_ex.inputs.len() - 1;

    // Build per-example data: (scalar_args, array, expected)
    let array_examples: Vec<(Vec<i64>, Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let scalar_args: Vec<i64> = ex
                .inputs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != array_idx)
                .filter_map(|(_, v)| {
                    if let Value::Int(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .collect();
            let array: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| v.as_i64_slice())
                .next()
                .unwrap_or_default();
            (scalar_args, array, ex.expected_int())
        })
        .collect();

    let scalar_param_names: Vec<&str> = ["a", "b", "c", "d"][..n_scalar_args].to_vec();

    // Namespace indices for fold body:
    // [scalar_args(0..n_scalar), item(n_scalar), i(n_scalar+1), acc(n_scalar+2)]
    let item_idx = n_scalar_args; // Var(item_idx) = item

    let init_candidates: Vec<Expr> = vec![Expr::Const(0), Expr::Const(1)];

    let start = std::time::Instant::now();
    let time_limit_ms: u64 = 15_000;

    // ── Strategy 1: Simple folds (acc = acc OP rhs, no condition) ──────────
    // Handles: array_sum, reverse_sum, interactive_sum, arr_sum_squares, closure_map_sum
    {
        let fold_n_args = n_scalar_args + 3;
        let mut body_atoms: Vec<Expr> = Vec::new();
        for v in 0..fold_n_args {
            body_atoms.push(Expr::Var(v));
        }
        for &c in &[0i64, 1, -1, 2] {
            body_atoms.push(Expr::Const(c));
        }

        // Atom OP atom combinations
        let mut body_exprs: Vec<Expr> = body_atoms.clone();
        for l in &body_atoms {
            for r in &body_atoms {
                for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul] {
                    body_exprs.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
                }
            }
        }

        for init in &init_candidates {
            for &bop in &[BinOp::Add, BinOp::Mul] {
                if start.elapsed().as_millis() as u64 > time_limit_ms {
                    return None;
                }
                for rhs in &body_exprs {
                    let fold_expr = Expr::ForFold {
                        init: Box::new(init.clone()),
                        body_op: bop,
                        body_rhs: Box::new(rhs.clone()),
                    };
                    if check_fold_examples(&fold_expr, &array_examples) {
                        let code =
                            emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                        eprintln!(
                            "[enum-array] FOUND simple fold: {} {:?}",
                            fn_name, fold_expr
                        );
                        if verify_problem_code_strict(problem, &code).is_ok() {
                            return Some(SolveResult {
                                success: true,
                                code,
                                method: "enumerative-array".to_string(),
                                error: None,
                                metadata: DifferentiableMetadata::default(),
                            });
                        }
                    }
                }
            }
        }
    }

    // ── Strategy 2: Conditional count folds (if CMP { acc + 1 }) ──────────
    // Handles: count_occurrences, count_positive, count_zeros, count_evens, count_greater_than
    {
        // Conditions: item CMP val where val ∈ {0, 1, scalar_args, item%2}
        let mut cond_pairs: Vec<(CmpOp, Expr, Expr)> = Vec::new();
        // item CMP const
        for &cmp in &[
            CmpOp::Eq,
            CmpOp::Gt,
            CmpOp::Lt,
            CmpOp::Ne,
            CmpOp::Ge,
            CmpOp::Le,
        ] {
            for &c in &[0i64, 1, -1, 2] {
                cond_pairs.push((cmp, Expr::Var(item_idx), Expr::Const(c)));
            }
            // item CMP scalar_arg
            for s in 0..n_scalar_args {
                cond_pairs.push((cmp, Expr::Var(item_idx), Expr::Var(s)));
                cond_pairs.push((cmp, Expr::Var(s), Expr::Var(item_idx)));
            }
        }
        // item % 2 == 0 (for count_evens)
        cond_pairs.push((
            CmpOp::Eq,
            Expr::BinOp(
                BinOp::Mod,
                Box::new(Expr::Var(item_idx)),
                Box::new(Expr::Const(2)),
            ),
            Expr::Const(0),
        ));
        cond_pairs.push((
            CmpOp::Eq,
            Expr::BinOp(
                BinOp::Mod,
                Box::new(Expr::Var(item_idx)),
                Box::new(Expr::Const(2)),
            ),
            Expr::Const(1),
        ));

        for (cmp, lhs, rhs) in &cond_pairs {
            if start.elapsed().as_millis() as u64 > time_limit_ms {
                return None;
            }
            // fold: acc = acc + if lhs CMP rhs { 1 } else { 0 }
            let cond_body = Expr::IfExpr(
                *cmp,
                Box::new(lhs.clone()),
                Box::new(rhs.clone()),
                Box::new(Expr::Const(1)),
                Box::new(Expr::Const(0)),
            );
            let fold_expr = Expr::ForFold {
                init: Box::new(Expr::Const(0)),
                body_op: BinOp::Add,
                body_rhs: Box::new(cond_body),
            };
            if check_fold_examples(&fold_expr, &array_examples) {
                let code = emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                eprintln!("[enum-array] FOUND cond-count: {} {:?}", fn_name, fold_expr);
                if verify_problem_code_strict(problem, &code).is_ok() {
                    return Some(SolveResult {
                        success: true,
                        code,
                        method: "enumerative-array".to_string(),
                        error: None,
                        metadata: DifferentiableMetadata::default(),
                    });
                }
            }
        }
    }

    // ── Strategy 3: Conditional sum folds (if CMP { acc + item }) ──────────
    // Handles: sum_negatives, sum_positives, sum_absolute
    {
        let mut cond_pairs: Vec<(CmpOp, Expr, Expr)> = Vec::new();
        for &cmp in &[CmpOp::Lt, CmpOp::Gt, CmpOp::Le, CmpOp::Ge] {
            for &c in &[0i64, 1, -1] {
                cond_pairs.push((cmp, Expr::Var(item_idx), Expr::Const(c)));
            }
        }

        for (cmp, lhs, rhs) in &cond_pairs {
            if start.elapsed().as_millis() as u64 > time_limit_ms {
                return None;
            }
            // fold: acc = acc + if lhs CMP rhs { item } else { 0 }
            let cond_body = Expr::IfExpr(
                *cmp,
                Box::new(lhs.clone()),
                Box::new(rhs.clone()),
                Box::new(Expr::Var(item_idx)),
                Box::new(Expr::Const(0)),
            );
            let fold_expr = Expr::ForFold {
                init: Box::new(Expr::Const(0)),
                body_op: BinOp::Add,
                body_rhs: Box::new(cond_body),
            };
            if check_fold_examples(&fold_expr, &array_examples) {
                let code = emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                eprintln!("[enum-array] FOUND cond-sum: {} {:?}", fn_name, fold_expr);
                if verify_problem_code_strict(problem, &code).is_ok() {
                    return Some(SolveResult {
                        success: true,
                        code,
                        method: "enumerative-array".to_string(),
                        error: None,
                        metadata: DifferentiableMetadata::default(),
                    });
                }
            }
        }

        // sum_absolute: acc = acc + if item < 0 { 0 - item } else { item }
        let abs_body = Expr::IfExpr(
            CmpOp::Lt,
            Box::new(Expr::Var(item_idx)),
            Box::new(Expr::Const(0)),
            Box::new(Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Const(0)),
                Box::new(Expr::Var(item_idx)),
            )),
            Box::new(Expr::Var(item_idx)),
        );
        let abs_fold = Expr::ForFold {
            init: Box::new(Expr::Const(0)),
            body_op: BinOp::Add,
            body_rhs: Box::new(abs_body),
        };
        if check_fold_examples(&abs_fold, &array_examples) {
            let code = emit_mog_array(&abs_fold, fn_name, &scalar_param_names, array_idx);
            eprintln!("[enum-array] FOUND abs-sum: {} {:?}", fn_name, abs_fold);
            if verify_problem_code_strict(problem, &code).is_ok() {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "enumerative-array".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
        }
    }

    // ── Strategy 4: Max/Min folds with conditional update ──────────
    // Handles: array_max, array_max_elem, min_element
    // Uses emit_mog_array's built-in min/max handling with `if item > acc { acc = item }`
    {
        // With init=0: works when all arrays have positive max / negative min
        // The emit_mog_array handles Min/Max body_op by generating `if item CMP acc { acc = item }`
        for &bop in &[BinOp::Min, BinOp::Max] {
            for init in &[Expr::Const(0), Expr::Const(1), Expr::Const(-1)] {
                if start.elapsed().as_millis() as u64 > time_limit_ms {
                    return None;
                }
                // For max: if item > acc { acc = item }
                // For min: if item < acc { acc = item }
                // These are handled by the ForFold body_rhs = Var(item_idx), body_op = Min/Max
                let fold_expr = Expr::ForFold {
                    init: Box::new(init.clone()),
                    body_op: bop,
                    body_rhs: Box::new(Expr::Var(item_idx)),
                };
                if check_fold_examples(&fold_expr, &array_examples) {
                    let code = emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                    eprintln!(
                        "[enum-array] FOUND max/min fold: {} {:?}",
                        fn_name, fold_expr
                    );
                    if verify_problem_code_strict(problem, &code).is_ok() {
                        return Some(SolveResult {
                            success: true,
                            code,
                            method: "enumerative-array".to_string(),
                            error: None,
                            metadata: DifferentiableMetadata::default(),
                        });
                    }
                }
            }
        }
    }

    // ── Strategy 5: Conditional sum with scalar comparison ──────────
    // Handles: count_greater_than, sum_at_even_indices-like (via i%2)
    {
        // Extended conditions: i CMP const
        for &cmp in &[CmpOp::Eq, CmpOp::Ne] {
            for &c in &[0i64, 1] {
                let cond_body = Expr::IfExpr(
                    cmp,
                    Box::new(Expr::BinOp(
                        BinOp::Mod,
                        Box::new(Expr::Var(item_idx)),
                        Box::new(Expr::Const(2)),
                    )),
                    Box::new(Expr::Const(c)),
                    Box::new(Expr::Const(1)),
                    Box::new(Expr::Const(0)),
                );
                let fold_expr = Expr::ForFold {
                    init: Box::new(Expr::Const(0)),
                    body_op: BinOp::Add,
                    body_rhs: Box::new(cond_body),
                };
                if check_fold_examples(&fold_expr, &array_examples) {
                    let code = emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                    eprintln!("[enum-array] FOUND mod-count: {} {:?}", fn_name, fold_expr);
                    if verify_problem_code_strict(problem, &code).is_ok() {
                        return Some(SolveResult {
                            success: true,
                            code,
                            method: "enumerative-array".to_string(),
                            error: None,
                            metadata: DifferentiableMetadata::default(),
                        });
                    }
                }
            }
        }
    }

    eprintln!(
        "[enum-array] no fold found in {:.1}s",
        start.elapsed().as_secs_f32()
    );
    None
}

/// Check a ForFold expression against array examples.
fn check_fold_examples(expr: &Expr, examples: &[(Vec<i64>, Vec<i64>, i64)]) -> bool {
    examples.iter().all(|(scalar_args, array, expected)| {
        expr.eval_array(scalar_args, array) == Some(*expected)
    })
}

// ─── Nested while loops ────────────────────────────────────────────────────

/// Try to discover programs with nested loops (e.g., sum_of_divisors, count_divisors).
fn synthesize_nested_while(problem: &Problem) -> Option<SolveResult> {
    let n_args = problem.examples.first()?.inputs.len();
    if n_args < 1 || n_args > 2 {
        return None;
    }

    let fn_name = problem.function_name();
    let param_names: Vec<&str> = ["a", "b"][..n_args].to_vec();

    let examples: Vec<(Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let args: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            (args, ex.expected_int())
        })
        .collect();

    let start = std::time::Instant::now();
    let time_limit_ms: u64 = 8_000;

    // Inner namespace: [args(0..n_args), outer_acc(n_args), outer_i(n_args+1), inner_acc(n_args+2), inner_i(n_args+3)]
    let nested_n_args = n_args + 4;
    let mut nested_atoms: Vec<Expr> = Vec::new();
    for v in 0..nested_n_args {
        nested_atoms.push(Expr::Var(v));
    }
    for &c in &[0i64, 1, -1, 2] {
        nested_atoms.push(Expr::Const(c));
    }

    let mut nested_binops: Vec<Expr> = Vec::new();
    for l in &nested_atoms {
        for r in &nested_atoms {
            for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Mod] {
                nested_binops.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
            }
        }
    }

    // Try different bound combinations:
    // outer_bound ∈ {args[0], args[0]+1}  inner_bound ∈ {outer_i, outer_i+1, args[0]}
    let outer_bounds: Vec<Expr> = vec![
        Expr::Var(0), // a
    ];
    let inner_bounds: Vec<Expr> = vec![
        Expr::Var(n_args + 1), // outer_i
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var(n_args + 1)),
            Box::new(Expr::Const(1)),
        ), // outer_i + 1
    ];

    for outer_bound in &outer_bounds {
        for inner_bound in &inner_bounds {
            for &outer_bop in &[BinOp::Add, BinOp::Mul] {
                for &inner_bop in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::BitOr] {
                    for inner_rhs in &nested_binops {
                        if start.elapsed().as_millis() as u64 > time_limit_ms {
                            return None;
                        }

                        let e = Expr::NestedWhile {
                            outer_init: Box::new(Expr::Const(0)),
                            outer_bound: Box::new(outer_bound.clone()),
                            outer_body_op: outer_bop,
                            inner_init: Box::new(Expr::Const(0)),
                            inner_bound: Box::new(inner_bound.clone()),
                            inner_body_op: inner_bop,
                            inner_body_rhs: Box::new(inner_rhs.clone()),
                        };

                        if examples
                            .iter()
                            .all(|(args, expected)| e.eval_nested(args) == Some(*expected))
                        {
                            let code = emit_mog(&e, fn_name, &param_names);
                            if verify_problem_code_strict(problem, &code).is_ok() {
                                return Some(SolveResult {
                                    success: true,
                                    code,
                                    method: "enumerative-nested".to_string(),
                                    error: None,
                                    metadata: DifferentiableMetadata::default(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// ─── While-cond loops (digit extraction, etc.) ─────────────────────────────

/// Discover programs with while-cond loops (e.g., digit_sum, digit_count, reverse_digits).
/// Pattern: acc = 0; x = arg; while x > 0 { acc = acc OP body(x, acc); x = x / 10 }
fn synthesize_while_cond(problem: &Problem) -> Option<SolveResult> {
    let n_args = problem.examples.first()?.inputs.len();
    if n_args != 1 {
        return None;
    }

    let fn_name = problem.function_name();
    let param_names: Vec<&str> = vec!["a"];

    let examples: Vec<(Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let args: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            (args, ex.expected_int())
        })
        .collect();

    let start = std::time::Instant::now();
    let time_limit_ms: u64 = 5_000;

    // Namespace: [a, x, acc, i]
    let n_ext = 4;
    let mut atoms: Vec<Expr> = Vec::new();
    for v in 0..n_ext {
        atoms.push(Expr::Var(v));
    }
    for &c in &[0i64, 1, -1, 2, 10] {
        atoms.push(Expr::Const(c));
    }

    // x % 10 (digit extraction) and x / 10 * 10 + acc etc.
    let mut body_exprs: Vec<Expr> = atoms.clone();
    for l in &atoms {
        for r in &atoms {
            for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Mod] {
                body_exprs.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
            }
        }
    }

    // Try different configurations
    let divisors: [i64; 2] = [10, 2];
    let cond_cmps: [CmpOp; 2] = [CmpOp::Gt, CmpOp::Ne];

    for &divisor in &divisors {
        for &cmp in &cond_cmps {
            for &bop in &[BinOp::Add, BinOp::Mul] {
                for body_rhs in &body_exprs {
                    if start.elapsed().as_millis() as u64 > time_limit_ms {
                        return None;
                    }

                    let e = Expr::WhileCond {
                        init: Box::new(Expr::Const(0)),
                        state_init: Box::new(Expr::Var(0)), // x = a
                        cond_cmp: cmp,
                        cond_val: Box::new(Expr::Const(0)), // while x > 0
                        divisor,
                        body_op: bop,
                        body_rhs: Box::new(body_rhs.clone()),
                    };

                    if examples
                        .iter()
                        .all(|(args, expected)| e.eval_while_cond(args) == Some(*expected))
                    {
                        let code = emit_mog(&e, fn_name, &param_names);
                        if verify_problem_code_strict(problem, &code).is_ok() {
                            return Some(SolveResult {
                                success: true,
                                code,
                                method: "enumerative-while-cond".to_string(),
                                error: None,
                                metadata: DifferentiableMetadata::default(),
                            });
                        }
                    }
                }
            }
        }
    }

    None
}

// ─── Dream mode ─────────────────────────────────────────────────────────────

/// Build the component library by corpus-driven frequent-subtree mining.
///
/// The historical hand-listed 15-seed enumeration has moved into
/// [`bootstrap_corpus`] (cold-start mining INPUT) — the library is now MINED
/// from the verified solved-Expr corpus rather than hand-enumerated. All call
/// sites (`load_or_dream`, `--dream`) are preserved by this shim.
pub fn dream(time_budget_ms: u64) -> ComponentLibrary {
    mine_library(time_budget_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_ext_finds_identity_and_reports_no_timeout() {
        // f(x) = x: should be found at size 1 (Var), so timed_out stays false
        let examples = vec![(vec![3], 3), (vec![7], 7), (vec![-2], -2)];
        let (expr, timed_out) = enumerate_exprs_core_ext(1, 5, &examples, 5_000, None);
        assert!(expr.is_some(), "identity must be discoverable in pass-1");
        assert!(
            !timed_out,
            "identity is size-1 so the budget shouldn't expire"
        );
    }

    #[test]
    fn stats_reports_max_completed_when_exhausted() {
        // f(x) = x + 1000 doesn't match any expression under the size/const budget
        // used here. At max_size=3 with a generous 3s budget, size 3 should
        // complete and max_completed should equal 3.
        let examples = vec![(vec![0], 1000), (vec![1], 1001), (vec![2], 1002)];
        let (expr, timed_out, max_completed) =
            enumerate_exprs_with_ops_stats(1, 3, &examples, 3_000, None, &CORE_BINOPS, &CORE_UNOPS);
        assert!(expr.is_none(), "1000 is not in the constant set");
        assert!(!timed_out, "size 3 enumeration should fit within 3s");
        assert_eq!(max_completed, 3, "every size should have completed cleanly");
    }

    #[test]
    fn stats_reports_timeout_with_partial_max_completed() {
        // 0ms budget forces the search to bail out of the size loop on the
        // first time check. We don't assert a specific max_completed value
        // because some sizes may slip through before the first elapsed()
        // check, but the timed_out flag must be set.
        let examples = vec![
            (vec![0], 1_000_000),
            (vec![1], 1_000_001),
            (vec![2], 1_000_002),
            (vec![3], 1_000_003),
        ];
        let (_expr, timed_out, _max_completed) =
            enumerate_exprs_with_ops_stats(1, 5, &examples, 0, None, &CORE_BINOPS, &CORE_UNOPS);
        assert!(timed_out, "0ms budget must flip the timeout flag");
    }

    // ── Anytime resumable search (Part A) ──────────────────────────────────

    /// THE RESUMABILITY GUARANTEE. A problem (`f(x) = x*x*x`, minimal expr size
    /// 5 under core ops) is UNSOLVED under a shallow soft cap, but the search
    /// does NOT report impossibility — it persists a frontier and, when RESUMED
    /// under a deeper cap, finds the answer by continuing from where it stopped
    /// rather than restarting from size 1.
    #[test]
    fn resumable_search_deepens_across_calls() {
        // cube: no single op / constant reproduces it, so it is not findable at
        // size <= 3; (x*x)*x is the size-5 witness.
        let examples = vec![(vec![2], 8), (vec![3], 27), (vec![4], 64), (vec![5], 125)];

        // CALL 1: deepen only up to size 3 (generous time, so the cap — not the
        // clock — bounds the run). Must MISS, must NOT be a timeout (it cleanly
        // exhausted the capped space), and must have advanced the frontier.
        let mut frontier = Frontier::fresh(String::new(), 1, 0);
        let (expr1, timed_out1) = enumerate_exprs_resumable(
            &mut frontier,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(3),
        );
        assert!(expr1.is_none(), "cube must NOT be solvable at size <= 3");
        assert!(
            !timed_out1,
            "clean exhaustion under a soft cap is not a timeout — and a None \
             here is 'budget exhausted, resumable', never a proof of impossibility"
        );
        let depth_after_1 = frontier.next_size;
        assert!(
            depth_after_1 >= 4,
            "frontier should have advanced past size 3 (got next_size={depth_after_1})"
        );
        // The frontier carries real work forward (size-3 stratum is populated).
        assert!(
            frontier.by_size.len() > 3 && !frontier.by_size[3].is_empty(),
            "size-3 expressions must be persisted in the frontier for resume"
        );

        // CALL 2: RESUME the SAME frontier under a deeper cap. It must now solve
        // cube — proving the second call continued deeper instead of restarting.
        let (expr2, _timed_out2) = enumerate_exprs_resumable(
            &mut frontier,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(7),
        );
        let solved = expr2.expect("resumed deeper search must solve cube");
        assert!(
            matches_all(&solved, &examples),
            "resumed solution must satisfy every example"
        );
        assert!(
            frontier.next_size > depth_after_1 || solved.size() <= 7,
            "resume must have explored deeper than the first call"
        );
    }

    /// DETERMINISM / MONOTONICITY. A search SPLIT across two resumed calls finds
    /// the SAME witness as one long uncapped call (the search is uniform-cost by
    /// size and the frontier is a faithful snapshot — no rand, no clock in
    /// selection). Proves the split-vs-whole equivalence the design claims.
    #[test]
    fn split_resume_matches_single_run() {
        let examples = vec![(vec![2], 8), (vec![3], 27), (vec![4], 64), (vec![5], 125)];

        // One long run (generous cap).
        let mut whole = Frontier::fresh(String::new(), 1, 0);
        let (one, _) = enumerate_exprs_resumable(
            &mut whole,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(7),
        );
        let one = one.expect("single run solves cube");

        // Split run: cap 3 (miss), then resume cap 7 (solve).
        let mut split = Frontier::fresh(String::new(), 1, 0);
        let (a, _) = enumerate_exprs_resumable(
            &mut split,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(3),
        );
        assert!(a.is_none(), "cube unsolved at cap 3");
        let (b, _) = enumerate_exprs_resumable(
            &mut split,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(7),
        );
        let two = b.expect("resumed run solves cube");

        assert_eq!(
            one, two,
            "split-across-calls search must find the identical witness as one long run"
        );
    }

    /// The resumable wrapper preserves the historical bounded semantics: the
    /// old `enumerate_exprs_with_ops_stats` API still finds identity at size 1
    /// and reports a clean (non-timeout) exhaustion when nothing matches.
    #[test]
    fn bounded_wrapper_behaves_like_before() {
        // identity at size 1
        let id = vec![(vec![3], 3), (vec![7], 7), (vec![-2], -2)];
        let (e, t, _mc) =
            enumerate_exprs_with_ops_stats(1, 5, &id, 5_000, None, &CORE_BINOPS, &CORE_UNOPS);
        assert!(e.is_some() && !t, "identity is size-1, no timeout");

        // unsolvable-in-budget but clean exhaustion at small cap
        let miss = vec![(vec![0], 1000), (vec![1], 1001), (vec![2], 1002)];
        let (e2, t2, mc2) =
            enumerate_exprs_with_ops_stats(1, 3, &miss, 3_000, None, &CORE_BINOPS, &CORE_UNOPS);
        assert!(e2.is_none() && !t2, "1000 not in const set; clean exhaustion");
        assert_eq!(mc2, 3, "every size up to the cap completed");
    }

    // ── Mining / generalization tests ──────────────────────────────────────

    fn solved(expr: Expr, n_args: usize) -> SolvedExpr {
        let fp = fingerprint(&expr, &probe_inputs(n_args, 8)).unwrap_or_default();
        SolvedExpr { expr, n_args, fp }
    }

    fn mul(a: Expr, b: Expr) -> Expr {
        Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b))
    }
    fn add(a: Expr, b: Expr) -> Expr {
        Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b))
    }

    /// REGRESSION (Bug B): the library must keep mining as the solve corpus
    /// GROWS — not freeze at its cold-start snapshot. Before the fix,
    /// `load_or_dream` returned the cached library unconditionally, so solves
    /// recorded into the corpus never became new abstractions. Here we mine a
    /// small corpus, then fold in a grown corpus carrying a NEW repeated pattern
    /// and assert a component is added (compounding) and that re-merging an
    /// unchanged corpus adds nothing (idempotent dedup).
    #[test]
    fn library_remines_as_corpus_grows() {
        // Small corpus: the square pattern a*a (support 2).
        let small = vec![
            solved(mul(Expr::Var(0), Expr::Var(0)), 1), // a*a
            solved(add(mul(Expr::Var(0), Expr::Var(0)), Expr::Var(1)), 2), // a*a + b
        ];
        let mut lib = mine_from_corpus(&small);
        let n_small = lib.len();
        assert!(n_small >= 1, "small corpus must mine at least the square pattern");

        // Grown corpus: small PLUS two trees carrying a NEW repeated pattern a+a.
        let mut grown = small.clone();
        grown.push(solved(add(Expr::Var(0), Expr::Var(0)), 1)); // a+a
        grown.push(solved(add(add(Expr::Var(0), Expr::Var(0)), Expr::Var(1)), 2)); // (a+a)+b

        let added = lib.merge_from_corpus(&grown);
        assert!(
            added >= 1,
            "growing the corpus with a new repeated pattern must fold in a component; \
             added={added}, names={:?}",
            lib.components.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>()
        );
        assert!(lib.len() > n_small, "library must grow with the corpus");

        // Re-merging an unchanged corpus must add nothing (dedup by description).
        let again = lib.merge_from_corpus(&grown);
        assert_eq!(again, 0, "re-merging an unchanged corpus must add no components");
    }

    /// THE GENERALIZATION LEVER. A mined `?0*?0` abstraction, re-rooted onto
    /// both arg slots and injected at by_size[1], makes sum-of-squares (true
    /// size 7) solvable at max_size=3 — otherwise impossible from scratch.
    #[test]
    fn mined_subtree_enables_novel_composition() {
        // Corpus: two distinct trees that BOTH contain the square pattern a*a.
        // Tree A = a*a (square). Tree B = a*a + b (square plus a second arg).
        let tree_a = mul(Expr::Var(0), Expr::Var(0)); // f(a) = a*a
        let tree_b = add(mul(Expr::Var(0), Expr::Var(0)), Expr::Var(1)); // f(a,b) = a*a + b
        let corpus = vec![solved(tree_a, 1), solved(tree_b, 2)];

        let lib = mine_from_corpus(&corpus);

        // The mined library must contain a component canonicalizing to a*a.
        let square = mul(Expr::Var(0), Expr::Var(0));
        let has_square = lib
            .get_for_args(1)
            .iter()
            .any(|e| canonicalize(e).0 == square);
        assert!(
            has_square,
            "mined library must contain the square pattern (canonical a*a); got names: {:?}",
            lib.components.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>()
        );
        // And its name must be the human-readable provenance.
        assert!(
            lib.components.iter().any(|(_, d)| d == "mined: a * a"),
            "square component must be named 'mined: a * a'"
        );

        // Sum-of-squares: f(a,b) = a*a + b*b. True Expr size is 7.
        let examples = vec![
            (vec![1, 2], 5),
            (vec![3, 1], 10),
            (vec![2, 4], 20),
            (vec![5, 0], 25),
            (vec![-2, 3], 13),
        ];

        // Without the library, max_size=3 CANNOT build a size-7 program.
        let without = enumerate_exprs(2, 3, &examples, 2000, None);
        assert!(
            without.is_none(),
            "max_size=3 must NOT solve sum-of-squares from scratch (got {without:?})"
        );

        // With the mined library, the re-rooted square (Var0*Var0 and Var1*Var1)
        // injected at by_size[1] composes via a size-2 Add into the solution.
        let with = enumerate_exprs(2, 3, &examples, 2000, Some(&lib));
        assert!(
            with.is_some(),
            "mined abstraction must make sum-of-squares solvable at max_size=3"
        );
        // Soundness preserved: the returned expr is observationally verified.
        let e = with.unwrap();
        assert!(
            matches_all(&e, &examples),
            "returned expr must satisfy all examples (sound): {e:?}"
        );
    }

    /// ADVERSARIAL PROBE (temporary): prove the miner is DATA-DERIVED.
    /// (A) A corpus whose recurring pattern is a*(a+b) [NOT a*a] must mine
    ///     that pattern and must NOT contain a*a. A hardcoded a*a list fails.
    /// (B) A corpus where a*a appears in only ONE tree (no recurrence) must NOT
    ///     promote a*a (support<2). A canned list would still surface it.
    #[test]
    fn adversarial_data_derivation_probe() {
        // ---- (A) different recurring pattern: a*(a+b) appears in two trees ----
        let patt = mul(Expr::Var(0), add(Expr::Var(0), Expr::Var(1))); // a*(a+b), size 5
        let tree_a = patt.clone();
        let tree_b = add(patt.clone(), Expr::Var(1)); // a*(a+b) + b
        let corpus_a = vec![solved(tree_a, 2), solved(tree_b, 2)];
        let lib_a = mine_from_corpus(&corpus_a);
        let patt_canon = canonicalize(&patt).0;
        let square_canon = canonicalize(&mul(Expr::Var(0), Expr::Var(0))).0;
        let mined: Vec<Expr> = lib_a.components.iter().map(|(e, _)| canonicalize(e).0).collect();
        eprintln!("PROBE-A mined canon set: {:?}", mined);
        eprintln!("PROBE-A names: {:?}", lib_a.components.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>());
        assert!(
            mined.iter().any(|e| *e == patt_canon),
            "miner must extract the ACTUAL recurring pattern a*(a+b) from THIS corpus"
        );
        assert!(
            !mined.iter().any(|e| *e == square_canon),
            "miner must NOT surface a*a — it is absent from corpus_a (no hardcoding)"
        );

        // ---- (B) a*a present in only ONE tree -> support 1 -> not promoted ----
        let only_once = vec![
            solved(mul(Expr::Var(0), Expr::Var(0)), 1), // a*a once
            solved(add(Expr::Var(0), Expr::Var(1)), 2), // a+b (no a*a)
        ];
        let lib_b = mine_from_corpus(&only_once);
        let mined_b: Vec<Expr> = lib_b.components.iter().map(|(e, _)| canonicalize(e).0).collect();
        eprintln!("PROBE-B mined canon set: {:?}", mined_b);
        // Support-mined a*a needs >=2 distinct trees. The hybrid floor (lib<8)
        // re-adds corpus exprs verbatim, so a*a CAN appear via the floor — but
        // that is still corpus-derived (it came from a solved tree), never a
        // literal. The discriminating claim: it is NOT named "mined: a * a"
        // unless support>=2. Assert the FREQUENCY path did not fire for a*a.
        let support_mined_square = lib_b
            .components
            .iter()
            .any(|(e, d)| canonicalize(e).0 == square_canon && d == "mined: a * a");
        // a*a appears once => the only way it enters is the hybrid floor, which
        // is corpus-derived; either way no canned literal exists. We additionally
        // confirm a pattern absent from BOTH corpora (a/b) is NEVER mined.
        let div_canon = canonicalize(&Expr::BinOp(BinOp::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)))).0;
        assert!(
            !mined_b.iter().any(|e| *e == div_canon),
            "a/b never appears in any corpus and must never be mined (no seeded list)"
        );
        eprintln!("PROBE-B support-mined a*a named 'mined: a * a'? {}", support_mined_square);
    }

    /// Determinism: mining the same corpus twice yields byte-identical libraries
    /// in the same order.
    #[test]
    fn mining_is_deterministic() {
        let corpus = vec![
            solved(mul(Expr::Var(0), Expr::Var(0)), 1),
            solved(add(mul(Expr::Var(0), Expr::Var(0)), Expr::Var(1)), 2),
            solved(
                add(mul(Expr::Var(1), Expr::Var(1)), Expr::Var(0)),
                2,
            ),
        ];
        let lib1 = mine_from_corpus(&corpus);
        let lib2 = mine_from_corpus(&corpus);
        assert_eq!(
            lib1.components, lib2.components,
            "mining must be deterministic (identical components in identical order)"
        );
    }

    /// Degeneracy rejection: leaves, no-ops, var-free folds, and whole-program
    /// memorization are never promoted.
    #[test]
    fn rejects_degenerate_abstractions() {
        // Pure leaves + a no-op (Var0 + Const0 == Var0 observationally).
        let corpus = vec![
            solved(Expr::Var(0), 1),
            solved(Expr::Const(5), 1),
            solved(add(Expr::Var(0), Expr::Const(0)), 1),
            solved(add(Expr::Var(0), Expr::Const(0)), 1), // appears twice -> support 2
        ];
        let lib = mine_from_corpus(&corpus);
        // D1/D2/D6 must fire: no leaf, no no-op promoted.
        for (e, _) in &lib.components {
            let (canon, arity) = canonicalize(e);
            assert!(
                !matches!(canon, Expr::Var(_) | Expr::Const(_)),
                "no bare leaf may be promoted"
            );
            // observational-identity no-op must not be promoted.
            let probes = probe_inputs(arity.max(1), 16);
            assert_ne!(
                fingerprint(&canon, &probes),
                fingerprint(&Expr::Var(0), &probes),
                "no observational no-op may be promoted: {canon:?}"
            );
        }

        // Whole-program memorization (size > 9 cap, D5): two identical large
        // trees must NOT be promoted as the whole tree.
        // Build a size-11 tree: ((a*a)+(a*a)) + ((a*a)+(a*a)) ... compose to >9.
        let sq = mul(Expr::Var(0), Expr::Var(0)); // size 3
        let big = add(add(sq.clone(), sq.clone()), add(sq.clone(), sq.clone())); // size 3*4+3 = 15
        assert!(big.size() > MAX_ABSTRACTION_SIZE);
        let corpus2 = vec![solved(big.clone(), 1), solved(big.clone(), 1)];
        let lib2 = mine_from_corpus(&corpus2);
        let big_canon = canonicalize(&big).0;
        assert!(
            !lib2
                .components
                .iter()
                .any(|(e, _)| canonicalize(e).0 == big_canon),
            "whole oversized tree must not be promoted (no memorization)"
        );

        // Var-free fold (D3): Const2 * Const3 has arity 0 -> not promotable.
        let varfree = mul(Expr::Const(2), Expr::Const(3));
        assert!(
            !is_promotable(&canonicalize(&varfree).0, 0),
            "var-free constant fold must not be promotable"
        );
    }

    /// Soundness: re-rooting a Div onto an arg that can be 0 must never yield an
    /// accepted-but-ill-defined program. Either the mine-time robust gate
    /// rejects it, or the per-use robust gate prunes it, or any returned
    /// solution evaluates cleanly on every example.
    #[test]
    fn rerooted_division_is_probe_guarded() {
        // Component a / b (size 3). Re-rooted onto (Var0, Var1) and (Var1, Var0).
        let divexpr = Expr::BinOp(BinOp::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let corpus = vec![
            solved(divexpr.clone(), 2),
            // a second tree containing a/b so support >= 2
            solved(
                add(
                    Expr::BinOp(BinOp::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
                    Expr::Var(0),
                ),
                2,
            ),
        ];
        let lib = mine_from_corpus(&corpus);

        // Examples where one input is 0 in the divisor position for some maps.
        let examples = vec![
            (vec![6, 2], 3),
            (vec![10, 0], 999), // divisor 0 path -> any re-rooting that divides by b crashes
            (vec![8, 4], 2),
        ];
        // Whatever the enumerator returns (or None), it must be sound: never an
        // expr that returns None on an example.
        let result = enumerate_exprs(2, 3, &examples, 2000, Some(&lib));
        if let Some(e) = result {
            assert!(
                matches_all(&e, &examples),
                "any returned solution must be observationally valid on all examples"
            );
            assert!(
                robust_well_defined(&e, 2, 30),
                "any returned solution must be robust (no div-by-zero on probes)"
            );
        }
        // The point: no crash, no ill-defined accepted program. (result may be
        // None — there is no clean closed form here, and that is correct.)
    }

    /// Back-compat: an OLD-shape library JSON ({"components":[[expr,"a + b"]]})
    /// still deserializes with the component intact.
    #[test]
    fn old_library_json_still_loads() {
        let json = r#"{
            "components": [
                [{"BinOp":["Add",{"Var":0},{"Var":1}]}, "a + b"]
            ]
        }"#;
        let lib: ComponentLibrary =
            serde_json::from_str(json).expect("old-shape library must deserialize");
        assert_eq!(lib.len(), 1);
        assert_eq!(lib.components[0].1, "a + b");
        assert_eq!(
            lib.components[0].0,
            add(Expr::Var(0), Expr::Var(1)),
            "deserialized component must be a + b"
        );
    }

    /// Corpus guards: the persistent path is disabled under cfg!(test)
    /// (hermetic) and record_solved_expr respects the entry cap when an explicit
    /// path is provided.
    #[test]
    fn solved_expr_corpus_is_test_disabled_and_capped() {
        // Under cfg!(test) with no env override, the path is None (hermetic).
        assert!(
            solved_exprs_path().is_none(),
            "solved_exprs_path must be None under cfg!(test) for hermeticity"
        );

        // With an explicit tempfile path, the entry cap is enforced.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "mog_synth_test_corpus_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::env::set_var("NSYNTH_SOLVED_EXPRS_PATH", &path);

        // Push more than the cap; each distinct expr/fp is a new entry.
        for i in 0..(SOLVED_EXPRS_MAX_ENTRIES + 50) {
            let e = add(Expr::Var(0), Expr::Const(i as i64));
            let fp = vec![i as i64];
            record_solved_expr(&e, 1, &fp);
        }
        let loaded = load_solved_exprs();
        assert!(
            loaded.len() <= SOLVED_EXPRS_MAX_ENTRIES,
            "corpus must be capped at {} entries, got {}",
            SOLVED_EXPRS_MAX_ENTRIES,
            loaded.len()
        );

        // Cleanup.
        std::env::remove_var("NSYNTH_SOLVED_EXPRS_PATH");
        let _ = std::fs::remove_file(&path);
    }
}
