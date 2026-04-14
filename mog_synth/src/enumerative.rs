//! Bottom-up enumerative program synthesis.
//!
//! Builds ALL programs from small to large, tests each against I/O examples,
//! returns the first (smallest) program that satisfies all examples.
//!
//! No templates. No patterns. Pure discovery.
//!
//! The key insight: for small programs (up to ~size 7), exhaustive enumeration
//! is fast. A program of size 1 is just a variable or constant. Size 2 is
//! `a OP b`. Size 3 is `(a OP b) OP c` or `a OP (b OP c)`. With observational
//! equivalence pruning, the actual search space is much smaller than the
//! theoretical maximum.
//!
//! Performance: ~1M candidates/sec for scalar programs in release mode.

use crate::benchmark::{Example, Problem, Value};
use crate::runtime::verify_problem_code_strict;
use crate::solver::SolveResult;
use crate::differentiable::DifferentiableMetadata;
use std::fmt::Write;

/// An expression in our mini-language.
#[derive(Clone, Debug)]
enum Expr {
    Var(usize),           // argument index
    Const(i64),           // literal constant
    BinOp(Op, Box<Expr>, Box<Expr>),
    UnaryOp(UnOp, Box<Expr>),
    IfThenElse(Cmp, Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>), // if lhs CMP rhs then a else b
}

#[derive(Clone, Copy, Debug)]
enum Op { Add, Sub, Mul, Div, Mod }

#[derive(Clone, Copy, Debug)]
enum UnOp { Neg, Abs }

#[derive(Clone, Copy, Debug)]
enum Cmp { Lt, Le, Eq, Ge, Gt, Ne }

impl Expr {
    fn eval(&self, args: &[i64]) -> Option<i64> {
        match self {
            Expr::Var(i) => args.get(*i).copied(),
            Expr::Const(c) => Some(*c),
            Expr::BinOp(op, l, r) => {
                let lv = l.eval(args)?;
                let rv = r.eval(args)?;
                match op {
                    Op::Add => lv.checked_add(rv),
                    Op::Sub => lv.checked_sub(rv),
                    Op::Mul => lv.checked_mul(rv),
                    Op::Div => if rv == 0 { None } else { Some(lv / rv) },
                    Op::Mod => if rv == 0 { None } else { Some(lv % rv) },
                }
            }
            Expr::UnaryOp(op, e) => {
                let v = e.eval(args)?;
                match op {
                    UnOp::Neg => Some(-v),
                    UnOp::Abs => Some(v.abs()),
                }
            }
            Expr::IfThenElse(cmp, lhs, rhs, then_e, else_e) => {
                let l = lhs.eval(args)?;
                let r = rhs.eval(args)?;
                let cond = match cmp {
                    Cmp::Lt => l < r, Cmp::Le => l <= r, Cmp::Eq => l == r,
                    Cmp::Ge => l >= r, Cmp::Gt => l > r, Cmp::Ne => l != r,
                };
                if cond { then_e.eval(args) } else { else_e.eval(args) }
            }
        }
    }

    fn size(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::BinOp(_, l, r) => 1 + l.size() + r.size(),
            Expr::UnaryOp(_, e) => 1 + e.size(),
            Expr::IfThenElse(_, a, b, c, d) => 1 + a.size() + b.size() + c.size() + d.size(),
        }
    }

    fn to_mog(&self, param_names: &[&str]) -> String {
        match self {
            Expr::Var(i) => param_names.get(*i).unwrap_or(&"x").to_string(),
            Expr::Const(c) => {
                if *c < 0 { format!("(0 - {})", -c) } else { format!("{c}") }
            }
            Expr::BinOp(op, l, r) => {
                let ls = l.to_mog(param_names);
                let rs = r.to_mog(param_names);
                let ops = match op {
                    Op::Add => "+", Op::Sub => "-", Op::Mul => "*",
                    Op::Div => "/", Op::Mod => "%",
                };
                format!("{ls} {ops} {rs}")
            }
            Expr::UnaryOp(UnOp::Neg, e) => format!("0 - {}", e.to_mog(param_names)),
            Expr::UnaryOp(UnOp::Abs, e) => {
                let es = e.to_mog(param_names);
                format!("if {es} < 0 {{ 0 - {es} }} else {{ {es} }}")
            }
            Expr::IfThenElse(cmp, lhs, rhs, then_e, else_e) => {
                let ls = lhs.to_mog(param_names);
                let rs = rhs.to_mog(param_names);
                let cs = match cmp {
                    Cmp::Lt => "<", Cmp::Le => "<=", Cmp::Eq => "==",
                    Cmp::Ge => ">=", Cmp::Gt => ">", Cmp::Ne => "!=",
                };
                let ts = then_e.to_mog(param_names);
                let es = else_e.to_mog(param_names);
                format!("if {ls} {cs} {rs} {{ {ts} }} else {{ {es} }}")
            }
        }
    }
}

/// Check if an expression matches all examples.
fn matches_all(expr: &Expr, examples: &[(Vec<i64>, i64)]) -> bool {
    for (args, expected) in examples {
        match expr.eval(args) {
            Some(v) if v == *expected => {}
            _ => return false,
        }
    }
    true
}

/// Observational equivalence: compute outputs on a set of test inputs.
/// Two expressions with the same outputs are equivalent — skip duplicates.
fn fingerprint(expr: &Expr, test_inputs: &[Vec<i64>]) -> Option<Vec<i64>> {
    let mut fp = Vec::with_capacity(test_inputs.len());
    for args in test_inputs {
        match expr.eval(args) {
            Some(v) => fp.push(v),
            None => return None, // division by zero etc
        }
    }
    Some(fp)
}

/// Bottom-up enumeration of all expressions up to a given size.
fn enumerate_exprs(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
) -> Option<Expr> {
    let start = std::time::Instant::now();

    // Constants to try
    let constants: Vec<i64> = vec![0, 1, -1, 2, -2, 3, 5, 10, 100];
    let ops = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Mod];
    let cmps = [Cmp::Lt, Cmp::Le, Cmp::Eq, Cmp::Ge, Cmp::Gt, Cmp::Ne];

    // Test inputs for observational equivalence
    let test_inputs: Vec<Vec<i64>> = examples.iter().map(|(args, _)| args.clone()).collect();

    // Build expressions by size, using observational equivalence to prune
    let mut by_size: Vec<Vec<Expr>> = vec![vec![]; max_size + 1];
    let mut seen_fingerprints: std::collections::HashSet<Vec<i64>> = std::collections::HashSet::new();

    // Size 1: variables and constants
    for i in 0..n_args {
        let e = Expr::Var(i);
        if let Some(fp) = fingerprint(&e, &test_inputs) {
            if matches_all(&e, examples) { return Some(e); }
            if seen_fingerprints.insert(fp) { by_size[1].push(e); }
        }
    }
    for &c in &constants {
        let e = Expr::Const(c);
        if let Some(fp) = fingerprint(&e, &test_inputs) {
            if matches_all(&e, examples) { return Some(e); }
            if seen_fingerprints.insert(fp) { by_size[1].push(e); }
        }
    }

    // Size 2+: compose from smaller
    for size in 2..=max_size {
        if start.elapsed().as_millis() as u64 > time_limit_ms { break; }
        let mut new_exprs: Vec<Expr> = Vec::new();

        // Unary ops: size = 1 + child_size
        if size >= 2 {
            let children: Vec<Expr> = by_size[size - 1].clone();
            for child in &children {
                for &uop in &[UnOp::Neg, UnOp::Abs] {
                    let e = Expr::UnaryOp(uop, Box::new(child.clone()));
                    if let Some(fp) = fingerprint(&e, &test_inputs) {
                        if matches_all(&e, examples) { return Some(e); }
                        if seen_fingerprints.insert(fp) { new_exprs.push(e); }
                    }
                }
            }
        }

        // Binary ops: size = 1 + left_size + right_size
        for left_size in 1..size {
            let right_size = size - 1 - left_size;
            if right_size < 1 || right_size >= size { continue; }
            if start.elapsed().as_millis() as u64 > time_limit_ms { break; }

            let lefts: Vec<Expr> = by_size[left_size].clone();
            let rights: Vec<Expr> = by_size[right_size].clone();
            for left in &lefts {
                for right in &rights {
                    for &op in &ops {
                        let e = Expr::BinOp(op, Box::new(left.clone()), Box::new(right.clone()));
                        if let Some(fp) = fingerprint(&e, &test_inputs) {
                            if matches_all(&e, examples) { return Some(e); }
                            if seen_fingerprints.insert(fp) { new_exprs.push(e); }
                        }
                    }
                }
            }
        }

        // If-then-else: minimum size 5 (if var CMP var then var else var)
        if size >= 5 {
            let atoms: Vec<Expr> = by_size[1].clone();
            for &cmp in &cmps {
                for cmp_l in &atoms {
                    for cmp_r in &atoms {
                        let branch_budget = size - 3;
                        for then_size in 1..branch_budget {
                            let else_size = branch_budget - then_size;
                            if else_size < 1 { continue; }
                            if start.elapsed().as_millis() as u64 > time_limit_ms { break; }
                            let then_es: Vec<Expr> = by_size[then_size].clone();
                            let else_es: Vec<Expr> = by_size[else_size].clone();
                            for then_e in &then_es {
                                for else_e in &else_es {
                                    let e = Expr::IfThenElse(
                                        cmp,
                                        Box::new(cmp_l.clone()),
                                        Box::new(cmp_r.clone()),
                                        Box::new(then_e.clone()),
                                        Box::new(else_e.clone()),
                                    );
                                    if let Some(fp) = fingerprint(&e, &test_inputs) {
                                        if matches_all(&e, examples) { return Some(e); }
                                        if seen_fingerprints.insert(fp) { new_exprs.push(e); }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        eprintln!("[enum] size {size}: {} new exprs, {} total fingerprints, {:.1}s",
                  new_exprs.len(), seen_fingerprints.len(),
                  start.elapsed().as_secs_f32());
        by_size[size] = new_exprs;
    }

    None
}

/// Main entry point: enumerative synthesis for scalar problems.
pub fn synthesize_enumerative(problem: &Problem) -> Option<SolveResult> {
    // Only scalar-input problems for now
    if !problem.examples.iter().all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_)))) {
        return None;
    }
    let n_args = problem.examples.first()?.inputs.len();
    let fn_name = problem.function_name();

    // Build integer examples
    let examples: Vec<(Vec<i64>, i64)> = problem.examples.iter().map(|ex| {
        let args: Vec<i64> = ex.inputs.iter().filter_map(|v| {
            if let Value::Int(i) = v { Some(*i) } else { None }
        }).collect();
        (args, ex.expected)
    }).collect();

    let param_names: Vec<&str> = ["a", "b", "c", "d", "e", "f"]
        .iter().take(n_args).copied().collect();

    // Enumerate expressions up to size 7 with 10s time limit
    let max_size = if n_args <= 2 { 7 } else { 5 };
    let time_limit = if n_args <= 2 { 10_000 } else { 5_000 };

    if let Some(expr) = enumerate_exprs(n_args, max_size, &examples, time_limit) {
        let expr_str = expr.to_mog(&param_names);
        let sig = param_names.iter()
            .map(|n| format!("{n}: i64"))
            .collect::<Vec<_>>().join(", ");
        let code = format!("fn {fn_name}({sig}) -> i64 {{\n    return {expr_str};\n}}\n");

        // Verify via Mog runtime (handles edge cases the simple eval might miss)
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

    None
}
