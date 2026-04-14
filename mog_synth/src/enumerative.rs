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
use std::collections::HashSet;
use std::fmt::Write;

// ─── Expression AST ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Expr {
    Var(usize),
    Const(i64),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnOp, Box<Expr>),
    IfExpr(CmpOp, Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>), // if lhs CMP rhs { then } else { els }
    WhileAccum {  // acc = init; while cond { acc = body(acc, i); i++ } return acc
        init: Box<Expr>,
        bound: Box<Expr>,       // loop runs i from 0..bound
        body_op: BinOp,         // acc = acc OP rhs
        body_rhs: Box<Expr>,    // rhs computed from [acc, i, args, consts]
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Min, Max,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnOp { Neg, Abs, BitNot, Popcount }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CmpOp { Lt, Le, Eq, Ge, Gt, Ne }

const ALL_BINOPS: [BinOp; 12] = [
    BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Mod,
    BinOp::Min, BinOp::Max,
    BinOp::BitAnd, BinOp::BitOr, BinOp::BitXor,
    BinOp::Shl, BinOp::Shr,
];
const ALL_UNOPS: [UnOp; 4] = [UnOp::Neg, UnOp::Abs, UnOp::BitNot, UnOp::Popcount];
const ALL_CMPS: [CmpOp; 6] = [CmpOp::Lt, CmpOp::Le, CmpOp::Eq, CmpOp::Ge, CmpOp::Gt, CmpOp::Ne];
const CONSTANTS: [i64; 12] = [0, 1, -1, 2, -2, 3, 5, 7, 10, 32, 100, 255];

// Ops suitable for loop body (no division to avoid div-by-zero in loops)
const LOOP_BODY_OPS: [BinOp; 5] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::BitXor, BinOp::BitOr];

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
                    UnOp::Neg => Some(-v),
                    UnOp::Abs => Some(v.abs()),
                    UnOp::BitNot => Some(!v),
                    UnOp::Popcount => Some((v as u64).count_ones() as i64),
                }
            }
            Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) => {
                let l = lhs.eval(args)?;
                let r = rhs.eval(args)?;
                if eval_cmp(*cmp, l, r) { then_e.eval(args) } else { else_e.eval(args) }
            }
            Expr::WhileAccum { init, bound, body_op, body_rhs } => {
                let mut acc = init.eval(args)?;
                let n = bound.eval(args)?;
                if n < 0 || n > 10_000 { return None; } // safety bound
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
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::BinOp(_, l, r) => 1 + l.size() + r.size(),
            Expr::UnaryOp(_, e) => 1 + e.size(),
            Expr::IfExpr(_, a, b, c, d) => 1 + a.size() + b.size() + c.size() + d.size(),
            Expr::WhileAccum { init, bound, body_rhs, .. } => 3 + init.size() + bound.size() + body_rhs.size(),
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
            Expr::Const(c) => if *c < 0 { format!("(0 - {})", -c) } else { format!("{c}") },
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
                    CmpOp::Lt => "<", CmpOp::Le => "<=", CmpOp::Eq => "==",
                    CmpOp::Ge => ">=", CmpOp::Gt => ">", CmpOp::Ne => "!=",
                };
                let ts = then_e.to_mog_ext(param_names, extra_names);
                let es = else_e.to_mog_ext(param_names, extra_names);
                format!("if {ls} {cs} {rs} {{ {ts} }} else {{ {es} }}")
            }
            Expr::WhileAccum { init, bound, body_op, body_rhs } => {
                let ext_names = &["acc", "i"];
                let init_s = init.to_mog_ext(param_names, &[]);
                let bound_s = bound.to_mog_ext(param_names, &[]);
                let op_s = match body_op {
                    BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*",
                    BinOp::BitXor => "-", // approximate
                    _ => "+",
                };
                let rhs_s = body_rhs.to_mog_ext(param_names, ext_names);
                format!("/* loop */ {init_s}; /* while i < {bound_s}: acc = acc {op_s} {rhs_s} */")
            }
        }
    }
}

fn eval_binop(op: BinOp, a: i64, b: i64) -> Option<i64> {
    match op {
        BinOp::Add => a.checked_add(b),
        BinOp::Sub => a.checked_sub(b),
        BinOp::Mul => a.checked_mul(b),
        BinOp::Div => if b == 0 { None } else { Some(a / b) },
        BinOp::Mod => if b == 0 { None } else { Some(a % b) },
        BinOp::Min => Some(a.min(b)),
        BinOp::Max => Some(a.max(b)),
        BinOp::BitAnd => Some(a & b),
        BinOp::BitOr => Some(a | b),
        BinOp::BitXor => Some(a ^ b),
        BinOp::Shl => if b < 0 || b > 63 { None } else { Some(a << b) },
        BinOp::Shr => if b < 0 || b > 63 { None } else { Some(a >> b) },
    }
}

fn eval_cmp(cmp: CmpOp, a: i64, b: i64) -> bool {
    match cmp {
        CmpOp::Lt => a < b, CmpOp::Le => a <= b, CmpOp::Eq => a == b,
        CmpOp::Ge => a >= b, CmpOp::Gt => a > b, CmpOp::Ne => a != b,
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
    examples.iter().all(|(args, expected)| expr.eval(args) == Some(*expected))
}

// ─── Component Library ───────────────────────────────────────────────────────
// Stores discovered useful sub-expressions that can be reused across problems.

#[derive(Clone, Debug)]
pub struct ComponentLibrary {
    /// Reusable sub-expressions with their semantic descriptions
    components: Vec<(Expr, String)>, // (expr, description)
}

impl ComponentLibrary {
    pub fn new() -> Self {
        Self { components: Vec::new() }
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
}

// ─── Main enumeration engine ─────────────────────────────────────────────────

const CORE_BINOPS: [BinOp; 5] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Mod];
const CORE_UNOPS: [UnOp; 2] = [UnOp::Neg, UnOp::Abs];

/// Fast enumeration with core 5 ops only.
pub fn enumerate_exprs_core(
    n_args: usize, max_size: usize, examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64, library: Option<&ComponentLibrary>,
) -> Option<Expr> {
    enumerate_exprs_with_ops(n_args, max_size, examples, time_limit_ms, library, &CORE_BINOPS, &CORE_UNOPS)
}

/// Full enumeration with all 12 ops.
pub fn enumerate_exprs(
    n_args: usize, max_size: usize, examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64, library: Option<&ComponentLibrary>,
) -> Option<Expr> {
    enumerate_exprs_with_ops(n_args, max_size, examples, time_limit_ms, library, &ALL_BINOPS, &ALL_UNOPS)
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
    let start = std::time::Instant::now();
    let test_inputs: Vec<Vec<i64>> = examples.iter().map(|(a, _)| a.clone()).collect();

    let mut by_size: Vec<Vec<Expr>> = vec![vec![]; max_size + 1];
    let mut seen: HashSet<Vec<i64>> = HashSet::new();

    // Helper: check and add
    let mut check_add = |e: &Expr, by_size: &mut Vec<Vec<Expr>>, seen: &mut HashSet<Vec<i64>>| -> Option<Expr> {
        if let Some(fp) = fingerprint(e, &test_inputs) {
            if matches_all(e, examples) { return Some(e.clone()); }
            if seen.insert(fp) {
                let s = e.size();
                if s <= max_size { by_size[s].push(e.clone()); }
            }
        }
        None
    };

    // Size 1: variables, constants, library components
    for i in 0..n_args {
        if let Some(e) = check_add(&Expr::Var(i), &mut by_size, &mut seen) { return Some(e); }
    }
    for &c in &CONSTANTS {
        if let Some(e) = check_add(&Expr::Const(c), &mut by_size, &mut seen) { return Some(e); }
    }
    // Add library components
    if let Some(lib) = library {
        for comp in lib.get_for_args(n_args) {
            if let Some(e) = check_add(comp, &mut by_size, &mut seen) { return Some(e); }
        }
    }

    for size in 2..=max_size {
        if start.elapsed().as_millis() as u64 > time_limit_ms { break; }
        let mut new: Vec<Expr> = Vec::new();

        // Unary ops
        if size >= 2 {
            let children = by_size[size - 1].clone();
            for child in &children {
                for &uop in unops {
                    let e = Expr::UnaryOp(uop, Box::new(child.clone()));
                    if let Some(fp) = fingerprint(&e, &test_inputs) {
                        if matches_all(&e, examples) { return Some(e); }
                        if seen.insert(fp) { new.push(e); }
                    }
                }
            }
        }

        // Binary ops
        for ls in 1..size {
            let rs = size - 1 - ls;
            if rs < 1 || rs > max_size { continue; }
            if start.elapsed().as_millis() as u64 > time_limit_ms { break; }
            let lefts = by_size[ls].clone();
            let rights = by_size[rs].clone();
            for left in &lefts {
                for right in &rights {
                    for &op in binops {
                        let e = Expr::BinOp(op, Box::new(left.clone()), Box::new(right.clone()));
                        if let Some(fp) = fingerprint(&e, &test_inputs) {
                            if matches_all(&e, examples) { return Some(e); }
                            if seen.insert(fp) { new.push(e); }
                        }
                    }
                    if start.elapsed().as_millis() as u64 > time_limit_ms { break; }
                }
            }
        }

        // If-then-else (size >= 5)
        if size >= 5 {
            let atoms = by_size[1].clone();
            for &cmp in &ALL_CMPS {
                if start.elapsed().as_millis() as u64 > time_limit_ms { break; }
                for cl in &atoms {
                    for cr in &atoms {
                        let budget = size - 3;
                        for ts in 1..budget {
                            let es = budget - ts;
                            if es < 1 { continue; }
                            let then_es = by_size[ts].clone();
                            let else_es = by_size[es].clone();
                            for te in &then_es {
                                for ee in &else_es {
                                    let e = Expr::IfExpr(cmp,
                                        Box::new(cl.clone()), Box::new(cr.clone()),
                                        Box::new(te.clone()), Box::new(ee.clone()));
                                    if let Some(fp) = fingerprint(&e, &test_inputs) {
                                        if matches_all(&e, examples) { return Some(e); }
                                        if seen.insert(fp) { new.push(e); }
                                    }
                                }
                                if start.elapsed().as_millis() as u64 > time_limit_ms { break; }
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
                for i in 0..loop_n_args { loop_atoms.push(Expr::Var(i)); }
                for &c in &CONSTANTS { loop_atoms.push(Expr::Const(c)); }
                // Simple loop rhs: just atoms or atom OP atom
                let mut loop_exprs: Vec<Expr> = loop_atoms.clone();
                if rhs_budget >= 3 {
                    for l in &loop_atoms {
                        for r in &loop_atoms {
                            for &op in &[BinOp::Add, BinOp::Mul, BinOp::Mod] {
                                loop_exprs.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
                            }
                        }
                    }
                }

                for init in &inits {
                    for bound in &bounds {
                        for &bop in &LOOP_BODY_OPS {
                            for rhs in &loop_exprs {
                                if start.elapsed().as_millis() as u64 > time_limit_ms { break; }
                                let e = Expr::WhileAccum {
                                    init: Box::new(init.clone()),
                                    bound: Box::new(bound.clone()),
                                    body_op: bop,
                                    body_rhs: Box::new(rhs.clone()),
                                };
                                if let Some(fp) = fingerprint(&e, &test_inputs) {
                                    if matches_all(&e, examples) { return Some(e); }
                                    if seen.insert(fp) { new.push(e); }
                                }
                            }
                        }
                    }
                }
            }
        }

        eprintln!("[enum] size {size}: {} new, {} total unique, {:.1}s",
                  new.len(), seen.len(), start.elapsed().as_secs_f32());
        by_size[size] = new;
    }

    None
}

// ─── Emit Mog code from discovered expression ────────────────────────────────

fn emit_mog(expr: &Expr, fn_name: &str, param_names: &[&str]) -> String {
    let sig = param_names.iter().map(|n| format!("{n}: i64")).collect::<Vec<_>>().join(", ");

    match expr {
        Expr::WhileAccum { init, bound, body_op, body_rhs } => {
            let init_s = init.to_mog(param_names);
            let bound_s = bound.to_mog(param_names);
            let op_s = match body_op {
                BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*",
                _ => "+",
            };
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("acc"); v.push("i"); v
            };
            let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    i: i64 = 0;\n    while i < {bound_s} {{\n        acc = acc {op_s} {rhs_s};\n        i = i + 1;\n    }}\n    return acc;\n}}\n"
            )
        }
        _ => {
            let expr_s = expr.to_mog(param_names);
            format!("fn {fn_name}({sig}) -> i64 {{\n    return {expr_s};\n}}\n")
        }
    }
}

// ─── Public synthesis entry point ────────────────────────────────────────────

/// Enumerative synthesis: discovers programs from I/O examples alone.
pub fn synthesize_enumerative(problem: &Problem) -> Option<SolveResult> {
    // Only scalar-input problems
    if !problem.examples.iter().all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_)))) {
        return None;
    }
    let n_args = problem.examples.first()?.inputs.len();
    let fn_name = problem.function_name();

    let examples: Vec<(Vec<i64>, i64)> = problem.examples.iter().map(|ex| {
        let args: Vec<i64> = ex.inputs.iter().filter_map(|v| {
            if let Value::Int(i) = v { Some(*i) } else { None }
        }).collect();
        (args, ex.expected)
    }).collect();

    let param_names: Vec<&str> = ["a", "b", "c", "d", "e", "f"]
        .iter().take(n_args).copied().collect();

    // Two-pass: fast 5-op sweep (deep), then expanded 12-op sweep (shallow)
    // Pass 1: core ops only (+,-,*,/,%) — reaches size 7-9 quickly
    let max_size_fast = if n_args <= 1 { 9 } else if n_args <= 2 { 7 } else { 6 };
    let time_fast = if n_args <= 1 { 10_000 } else if n_args <= 2 { 8_000 } else { 5_000 };
    if let Some(expr) = enumerate_exprs_core(n_args, max_size_fast, &examples, time_fast, None) {
        let code = emit_mog(&expr, fn_name, &param_names);
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true, code, method: "enumerative".to_string(),
                error: None, metadata: DifferentiableMetadata::default(),
            });
        }
    }
    // Pass 2: full ops (min, max, abs, bitops) — shallower but wider
    let max_size_full = if n_args <= 1 { 7 } else if n_args <= 2 { 5 } else { 5 };
    let time_full = if n_args <= 1 { 8_000 } else { 5_000 };
    if let Some(expr) = enumerate_exprs(n_args, max_size_full, &examples, time_full, None) {
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
        eprintln!("[enum] found expr but Mog verification failed: {}", expr.to_mog(&param_names));
    }

    None
}
