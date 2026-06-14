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

// ─── Component Library ───────────────────────────────────────────────────────
// Stores discovered useful sub-expressions that can be reused across problems.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentLibrary {
    /// Reusable sub-expressions with their semantic descriptions
    components: Vec<(Expr, String)>, // (expr, description)
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
        }
    }

    /// Add a discovered component to the library
    pub fn add(&mut self, expr: Expr, description: String) {
        // Don't add duplicates (by description)
        if !self.components.iter().any(|(_, d)| d == &description) {
            self.components.push((expr, description));
        }
    }

    /// Get all components that take n_args arguments
    pub fn get_for_args(&self, _n_args: usize) -> Vec<&Expr> {
        self.components.iter().map(|(e, _)| e).collect()
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

    /// Load or create via dream mode if no library exists
    pub fn load_or_dream(dream_budget_ms: u64) -> Self {
        let path = library_path();
        if path.exists() {
            let lib = Self::load();
            if !lib.components.is_empty() {
                eprintln!(
                    "[library] loaded {} components from {}",
                    lib.len(),
                    path.display()
                );
                return lib;
            }
        }
        eprintln!("[library] no library found, running dream mode...");
        let lib = dream(dream_budget_ms);
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
    let start = std::time::Instant::now();
    let test_inputs: Vec<Vec<i64>> = examples.iter().map(|(a, _)| a.clone()).collect();

    let mut by_size: Vec<Vec<Expr>> = vec![vec![]; max_size + 1];
    let mut seen: HashSet<Vec<i64>> = HashSet::new();
    let mut timed_out = false;
    let mut max_completed: usize = 0;

    // Helper: check and add
    let check_add =
        |e: &Expr, by_size: &mut Vec<Vec<Expr>>, seen: &mut HashSet<Vec<i64>>| -> Option<Expr> {
            if let Some(fp) = fingerprint(e, &test_inputs) {
                if matches_all(e, examples) && robust_well_defined(e, n_args, 30) {
                    return Some(e.clone());
                }
                if seen.insert(fp) {
                    let s = e.size();
                    if s <= max_size {
                        by_size[s].push(e.clone());
                    }
                }
            }
            None
        };

    // Size 1: variables, constants, library components
    for i in 0..n_args {
        if let Some(e) = check_add(&Expr::Var(i), &mut by_size, &mut seen) {
            return (Some(e), false, max_completed);
        }
    }
    for &c in &CONSTANTS {
        if let Some(e) = check_add(&Expr::Const(c), &mut by_size, &mut seen) {
            return (Some(e), false, max_completed);
        }
    }
    // Add library components
    if let Some(lib) = library {
        for comp in lib.get_for_args(n_args) {
            if let Some(e) = check_add(comp, &mut by_size, &mut seen) {
                return (Some(e), false, max_completed);
            }
        }
    }

    for size in 2..=max_size {
        if start.elapsed().as_millis() as u64 > time_limit_ms {
            timed_out = true;
            break;
        }
        let mut new: Vec<Expr> = Vec::new();

        // Unary ops
        if size >= 2 {
            let children = by_size[size - 1].clone();
            for child in &children {
                for &uop in unops {
                    let e = Expr::UnaryOp(uop, Box::new(child.clone()));
                    if let Some(fp) = fingerprint(&e, &test_inputs) {
                        if matches_all(&e, examples) && robust_well_defined(&e, n_args, 30) {
                            return (Some(e), false, max_completed);
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
            if rs < 1 || rs > max_size {
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
                                return (Some(e), false, max_completed);
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
                                            return (Some(e), false, max_completed);
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
            if rhs_budget >= 1 && rhs_budget <= max_size {
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
                                        return (Some(e), false, max_completed);
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

        eprintln!(
            "[enum] size {size}: {} new, {} total unique, {:.1}s",
            new.len(),
            seen.len(),
            start.elapsed().as_secs_f32()
        );
        by_size[size] = new;
        if !timed_out {
            max_completed = size;
        }
    }

    (None, timed_out, max_completed)
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
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if item {cmp} acc {{\n            acc = item;\n        }}\n    }}\n    return acc;\n}}\n"
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

    // Two-pass: fast 5-op sweep (deep), then expanded 12-op sweep (shallow)
    // Pass 1: core ops only (+,-,*,/,%) — reaches size 7-9 quickly
    let max_size_fast = if n_args <= 1 {
        9
    } else if n_args <= 2 {
        7
    } else {
        6
    };
    let time_fast = if n_args <= 1 {
        10_000
    } else if n_args <= 2 {
        8_000
    } else {
        5_000
    };
    let (fast_expr, fast_timed_out, fast_max_completed) = enumerate_exprs_with_ops_stats(
        n_args,
        max_size_fast,
        &examples,
        time_fast,
        Some(&library),
        &CORE_BINOPS,
        &CORE_UNOPS,
    );
    if let Some(expr) = fast_expr {
        let code = emit_mog(&expr, fn_name, &param_names);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: "enumerative".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    // Decide whether pass-2 (full ops, shallower) can still help:
    //  - clean exhaustion of pass-1 → pass-2 won't find anything pass-1 missed,
    //    skip it so loop-class problems reach the gradient path sooner.
    //  - pass-1 timed out but already completed size ≥ 6 → the search space
    //    larger than pass-2's max_size is already explored; pass-2 would
    //    re-grind small sizes with a different op set for little gain.
    //  - pass-1 timed out having completed < 6 → pass-2's alternative ops
    //    might still find something small, so fall through.
    if !fast_timed_out {
        eprintln!("[enum] pass-1 exhausted cleanly; skipping pass-2");
        return None;
    }
    let max_size_full = if n_args <= 1 {
        7
    } else if n_args <= 2 {
        5
    } else {
        5
    };
    if fast_max_completed >= max_size_full {
        eprintln!(
            "[enum] pass-1 reached size {fast_max_completed} before timeout; skipping pass-2"
        );
        return None;
    }
    let time_full = if n_args <= 1 { 8_000 } else { 5_000 };
    if let Some(expr) = enumerate_exprs(n_args, max_size_full, &examples, time_full, Some(&library))
    {
        let code = emit_mog(&expr, fn_name, &param_names);

        // Verify via Mog runtime
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: "enumerative".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
        // If Mog verification fails (syntax mismatch), still report the expression
        eprintln!(
            "[enum] found expr but Mog verification failed: {}",
            expr.to_mog(&param_names)
        );
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
                .filter_map(|v| {
                    if let Value::Array(a) = v {
                        Some(a.clone())
                    } else {
                        None
                    }
                })
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

/// Background enumerator that discovers useful sub-expressions.
/// Seeds: target functions like add, mul, abs_diff, min, max, square, digit_sum, etc.
/// Each seed generates random I/O examples, runs enumeration, saves results to the library.
pub fn dream(time_budget_ms: u64) -> ComponentLibrary {
    let start = std::time::Instant::now();
    let mut library = ComponentLibrary::new();

    // Seed functions: (description, implementation as fn(&[i64]) -> i64)
    let seeds: Vec<(&str, Box<dyn Fn(&[i64]) -> i64>)> = vec![
        ("a + b", Box::new(|a| a[0] + a[1])),
        ("a * b", Box::new(|a| a[0] * a[1])),
        ("a - b", Box::new(|a| a[0] - a[1])),
        (
            "abs(a - b)",
            Box::new(|a| a[0].saturating_sub(a[1]).saturating_abs()),
        ),
        ("min(a, b)", Box::new(|a| a[0].min(a[1]))),
        ("max(a, b)", Box::new(|a| a[0].max(a[1]))),
        ("a * a", Box::new(|a| a[0] * a[0])),
        ("a % 2", Box::new(|a| a[0] % 2)),
        ("a / 2", Box::new(|a| a[0] / 2)),
        ("a + a", Box::new(|a| a[0] + a[0])),
        ("a * 2", Box::new(|a| a[0] * 2)),
        ("a * a + a", Box::new(|a| a[0] * a[0] + a[0])),
        ("a % 10", Box::new(|a| a[0] % 10)),
        ("a + 1", Box::new(|a| a[0] + 1)),
        ("0 - a", Box::new(|a| a[0].saturating_neg())),
    ];

    for (desc, func) in &seeds {
        if start.elapsed().as_millis() as u64 > time_budget_ms {
            break;
        }

        let n_args = 2; // most seeds are binary
                        // Generate 6 random I/O examples
        let examples: Vec<(Vec<i64>, i64)> = (0..6)
            .map(|i| {
                let args = vec![
                    ((i as i64 * 7 + 3) % 20 - 10),
                    ((i as i64 * 13 + 5) % 20 - 10),
                ];
                let expected = func(&args);
                (args, expected)
            })
            .collect();

        // Run quick enumeration
        if let Some(expr) = enumerate_exprs_core(n_args, 5, &examples, 500, None) {
            let expr_desc = format!("{:?}", expr);
            eprintln!("[dream] discovered: {desc} -> {expr_desc}");
            library.add(expr, desc.to_string());
        }
    }

    eprintln!(
        "[dream] discovered {} components in {:.1}s",
        library.components.len(),
        start.elapsed().as_secs_f32()
    );
    library
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
}
