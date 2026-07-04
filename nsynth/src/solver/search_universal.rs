//! ARITY-POLYMORPHIC TYPED SEARCH — composition that is NOT limited by the
//! number of arguments, inputs, or outputs.
//!
//! `search_combinator` composes atoms over a SINGLE int-list input. Real tasks
//! take many arguments of mixed type — `add(x, y)`, `triangle_area(a, h)`,
//! `compare_one(a, b)`, `add_elements(arr, k)` — and the composition that solves
//! them binds SEVERAL leaves at once. This module lifts the same guided
//! bottom-up enumeration to a heterogeneous, arity-general leaf set: every input
//! argument becomes a typed leaf (`a0, a1, …`, any count), data-mined constants
//! join them, and a typed operator basis (scalar arithmetic, comparison, boolean
//! logic, list reductions, list transforms) composes them. The output type is
//! whatever the examples demand — scalar, bool, float, or list.
//!
//! Same discipline as the other searchers: observational-equivalence dedup, a
//! best-first goal heuristic that keeps the wide surface affordable, a
//! deterministic node budget (not wall-clock), exact-match acceptance (float
//! close), and end-to-end re-verification of the emitted Mog before acceptance.

use crate::benchmark::{Problem, Value};
use crate::solver::SolveResult;
use std::collections::HashSet;

/// Typed value in the search. One `UV` per example is carried per sub-expression
/// (the "output vector"); observational equivalence is decided on that vector.
#[derive(Clone, Debug, PartialEq)]
enum UV {
    I(i64),
    F(f64),
    B(bool),
    LI(Vec<i64>),
}

impl UV {
    /// Structural key for OE dedup (floats bucketed to avoid FP noise).
    fn key(&self) -> String {
        match self {
            UV::I(i) => format!("I{i}"),
            UV::F(f) => format!("F{}", (f * 1e6).round() as i64),
            UV::B(b) => format!("B{b}"),
            UV::LI(l) => format!("L{l:?}"),
        }
    }
}

#[derive(Clone)]
struct Expr {
    outs: Vec<UV>,
    ir: String,
    depth: usize,
}

const MAX_DEPTH: usize = 4;
const MAX_NODES: usize = 9000;
const KEEP_PER_DEPTH: usize = 220;

fn to_uv(v: &Value) -> Option<UV> {
    match v {
        Value::Int(i) => Some(UV::I(*i)),
        Value::Float(b) => Some(UV::F(f64::from_bits(*b))),
        Value::Bool(b) => Some(UV::B(*b)),
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::Int(i) => Some(*i),
                _ => None,
            })
            .collect::<Option<Vec<i64>>>()
            .map(UV::LI),
        _ => None,
    }
}

fn uv_expected(v: &Value) -> Option<UV> {
    match v {
        Value::Int(i) => Some(UV::I(*i)),
        Value::Float(b) => Some(UV::F(f64::from_bits(*b))),
        Value::Bool(b) => Some(UV::B(*b)),
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::Int(i) => Some(*i),
                _ => None,
            })
            .collect::<Option<Vec<i64>>>()
            .map(UV::LI),
        _ => None,
    }
}

fn as_f(v: &UV) -> Option<f64> {
    match v {
        UV::I(i) => Some(*i as f64),
        UV::F(f) => Some(*f),
        _ => None,
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * (1.0 + a.abs().max(b.abs()))
}

fn uv_eq(a: &UV, b: &UV) -> bool {
    match (a, b) {
        (UV::F(x), UV::F(y)) => close(*x, *y),
        (UV::F(x), UV::I(y)) | (UV::I(y), UV::F(x)) => close(*x, *y as f64),
        _ => a == b,
    }
}

/// Try to solve an arbitrary-arity problem by typed composition. Fires as a
/// fallback tier; returns a verified Mog program or None.
pub(super) fn try_universal(problem: &Problem, name: &str) -> Option<SolveResult> {
    let n_args = problem.examples.first()?.inputs.len();
    if n_args == 0 || n_args > 4 {
        return None;
    }
    // Every argument of every example must be a supported type, same arity.
    let arg_cols: Vec<Vec<UV>> = (0..n_args)
        .map(|i| {
            problem
                .examples
                .iter()
                .map(|ex| ex.inputs.get(i).and_then(to_uv))
                .collect::<Option<Vec<UV>>>()
        })
        .collect::<Option<_>>()?;
    let expected: Vec<UV> = problem
        .examples
        .iter()
        .map(|ex| uv_expected(&ex.expected))
        .collect::<Option<_>>()?;

    // Param type declarations, taken from the first example's arg shapes.
    let param_ty: Vec<&'static str> = arg_cols
        .iter()
        .map(|col| match &col[0] {
            UV::I(_) => "i64",
            UV::F(_) => "f64",
            UV::B(_) => "bool",
            UV::LI(_) => "[i64]",
        })
        .collect();
    let ret_ty = match &expected[0] {
        UV::I(_) => "i64",
        UV::F(_) => "f64",
        UV::B(_) => "bool",
        UV::LI(_) => "[i64]",
    };

    // Seed pool: one leaf per argument.
    let mut pool: Vec<Expr> = Vec::new();
    for (i, col) in arg_cols.iter().enumerate() {
        pool.push(Expr { outs: col.clone(), ir: format!("a{i}"), depth: 1 });
    }
    // Constant leaves: small fixed ints, data-mined ints, and float 0.5 (halving
    // shows up in geometry/averages). Constants are the same across examples.
    let mut consts: Vec<i64> = vec![0, 1, 2];
    for col in &arg_cols {
        for v in col {
            if let UV::I(i) = v {
                if i.abs() <= 1000 && !consts.contains(i) {
                    consts.push(*i);
                }
            }
            if let UV::LI(l) = v {
                for &x in l {
                    if x.abs() <= 1000 && !consts.contains(&x) {
                        consts.push(x);
                    }
                }
            }
        }
    }
    for e in &expected {
        if let UV::I(i) = e {
            if i.abs() <= 1000 && !consts.contains(i) {
                consts.push(*i);
            }
        }
    }
    consts.sort_unstable();
    consts.dedup();
    consts.truncate(8);
    for &k in &consts {
        pool.push(Expr {
            outs: vec![UV::I(k); problem.examples.len()],
            ir: format!("#{k}"),
            depth: 1,
        });
    }
    pool.push(Expr {
        outs: vec![UV::F(0.5); problem.examples.len()],
        ir: "~0.5".to_string(),
        depth: 1,
    });

    let mut seen: HashSet<String> = HashSet::new();
    for e in &pool {
        seen.insert(e.outs.iter().map(|v| v.key()).collect::<Vec<_>>().join("|"));
    }

    // Immediate check on the seed pool (e.g. answer is a single argument).
    if let Some(win) = pool.iter().find(|e| vec_match(&e.outs, &expected)).cloned() {
        if let Some(r) = finish(name, &win.ir, &param_ty, ret_ty, problem) {
            return Some(r);
        }
    }

    let mut nodes = 0usize;
    for depth in 2..=MAX_DEPTH {
        if nodes >= MAX_NODES {
            break;
        }
        let prev: Vec<Expr> = pool.iter().filter(|e| e.depth < depth).cloned().collect();
        let mut fresh: Vec<Expr> = Vec::new();

        for (ai, a) in prev.iter().enumerate() {
            if nodes >= MAX_NODES {
                break;
            }
            // ---- unary over scalars ----
            match &a.outs[0] {
                UV::I(_) | UV::F(_) => {
                    // negate
                    let outs = map_scalar(a, |x| Some(neg(x)));
                    push(&mut fresh, outs, format!("Bsub(#0§{})", a.ir), depth);
                }
                UV::B(_) => {
                    let outs: Option<Vec<UV>> = a
                        .outs
                        .iter()
                        .map(|v| if let UV::B(b) = v { Some(UV::B(!b)) } else { None })
                        .collect();
                    if let Some(o) = outs {
                        push(&mut fresh, Some(o), format!("N({})", a.ir), depth);
                    }
                }
                _ => {}
            }
            // ---- list ops: leaf list -> reductions/transforms ----
            if let UV::LI(_) = &a.outs[0] {
                for op in ["sum", "product", "max", "min", "count"] {
                    let outs: Option<Vec<UV>> = a
                        .outs
                        .iter()
                        .map(|v| match v {
                            UV::LI(l) => fold(l, op).map(UV::I),
                            _ => None,
                        })
                        .collect();
                    push(&mut fresh, outs, format!("R{op}({})", a.ir), depth);
                }
                for t in ["sortasc", "sortdesc", "reverse", "unique"] {
                    let outs: Option<Vec<UV>> = a
                        .outs
                        .iter()
                        .map(|v| match v {
                            UV::LI(l) => Some(UV::LI(transform(l, t))),
                            _ => None,
                        })
                        .collect();
                    push(&mut fresh, outs, format!("T{t}({})", a.ir), depth);
                }
            }
            // ---- binary over scalar pairs ----
            for b in prev.iter().skip(ai) {
                if nodes >= MAX_NODES {
                    break;
                }
                nodes += 1;
                let sa = matches!(a.outs[0], UV::I(_) | UV::F(_));
                let sb = matches!(b.outs[0], UV::I(_) | UV::F(_));
                if sa && sb {
                    for (tag, f) in [
                        ("add", bin_add as fn(&UV, &UV) -> Option<UV>),
                        ("sub", bin_sub),
                        ("mul", bin_mul),
                        ("div", bin_div),
                        ("mod", bin_mod),
                    ] {
                        // sub/div/mod are not commutative — try both orders.
                        let comm = matches!(tag, "add" | "mul");
                        let outs = zip_bin(a, b, f);
                        push(&mut fresh, outs, format!("B{tag}({}§{})", a.ir, b.ir), depth);
                        if !comm {
                            let outs = zip_bin(b, a, f);
                            push(&mut fresh, outs, format!("B{tag}({}§{})", b.ir, a.ir), depth);
                        }
                    }
                    for (tag, hi) in [("max", true), ("min", false)] {
                        let outs = zip_bin(a, b, move |x, y| bin_minmax(x, y, hi));
                        push(&mut fresh, outs, format!("M{tag}({}§{})", a.ir, b.ir), depth);
                    }
                    // comparisons -> bool (both orders for lt/gt)
                    for (tag, f) in [
                        ("eq", cmp_eq as fn(&UV, &UV) -> Option<UV>),
                        ("lt", cmp_lt),
                        ("gt", cmp_gt),
                    ] {
                        let outs = zip_bin(a, b, f);
                        push(&mut fresh, outs, format!("C{tag}({}§{})", a.ir, b.ir), depth);
                    }
                }
                // ---- boolean combination ----
                if matches!(a.outs[0], UV::B(_)) && matches!(b.outs[0], UV::B(_)) {
                    for tag in ["and", "or"] {
                        let outs: Option<Vec<UV>> = a
                            .outs
                            .iter()
                            .zip(b.outs.iter())
                            .map(|(x, y)| match (x, y) {
                                (UV::B(p), UV::B(q)) => Some(UV::B(if tag == "and" {
                                    *p && *q
                                } else {
                                    *p || *q
                                })),
                                _ => None,
                            })
                            .collect();
                        push(&mut fresh, outs, format!("L{tag}({}§{})", a.ir, b.ir), depth);
                    }
                }
            }
        }

        // Exact winner bypasses the prune.
        if let Some(win) = fresh.iter().find(|e| vec_match(&e.outs, &expected)).cloned() {
            if let Some(r) = finish(name, &win.ir, &param_ty, ret_ty, problem) {
                return Some(r);
            }
        }
        fresh.retain(|e| {
            seen.insert(e.outs.iter().map(|v| v.key()).collect::<Vec<_>>().join("|"))
        });
        fresh.sort_by(|x, y| goal(&y.outs, &expected).cmp(&goal(&x.outs, &expected)));
        fresh.truncate(KEEP_PER_DEPTH);
        pool.append(&mut fresh);

        if let Some(win) = pool.iter().find(|e| vec_match(&e.outs, &expected)).cloned() {
            if let Some(r) = finish(name, &win.ir, &param_ty, ret_ty, problem) {
                return Some(r);
            }
        }
    }
    None
}

fn vec_match(outs: &[UV], expected: &[UV]) -> bool {
    outs.len() == expected.len() && outs.iter().zip(expected).all(|(a, b)| uv_eq(a, b))
}

/// Higher = better. Count of exactly-matching example slots dominates; a small
/// magnitude term breaks ties toward the closer numeric shape.
fn goal(outs: &[UV], expected: &[UV]) -> i64 {
    if outs.len() != expected.len() {
        return i64::MIN / 2;
    }
    let mut score = 0i64;
    for (o, e) in outs.iter().zip(expected) {
        if uv_eq(o, e) {
            score += 1000;
        } else if let (Some(a), Some(b)) = (as_f(o), as_f(e)) {
            let d = (a - b).abs();
            score -= (d.min(1e6) as i64).min(999);
        } else {
            score -= 500;
        }
    }
    score
}

fn push(fresh: &mut Vec<Expr>, outs: Option<Vec<UV>>, ir: String, depth: usize) {
    if let Some(outs) = outs {
        fresh.push(Expr { outs, ir, depth });
    }
}

fn map_scalar(a: &Expr, f: impl Fn(&UV) -> Option<UV>) -> Option<Vec<UV>> {
    a.outs.iter().map(|v| f(v)).collect()
}

fn zip_bin(a: &Expr, b: &Expr, f: impl Fn(&UV, &UV) -> Option<UV>) -> Option<Vec<UV>> {
    a.outs.iter().zip(b.outs.iter()).map(|(x, y)| f(x, y)).collect()
}

fn neg(x: &UV) -> UV {
    match x {
        UV::I(i) => UV::I(-i),
        UV::F(f) => UV::F(-f),
        _ => UV::I(0),
    }
}

fn both_int(a: &UV, b: &UV) -> Option<(i64, i64)> {
    if let (UV::I(x), UV::I(y)) = (a, b) {
        Some((*x, *y))
    } else {
        None
    }
}

fn bin_add(a: &UV, b: &UV) -> Option<UV> {
    if let Some((x, y)) = both_int(a, b) {
        return x.checked_add(y).map(UV::I);
    }
    Some(UV::F(as_f(a)? + as_f(b)?))
}
fn bin_sub(a: &UV, b: &UV) -> Option<UV> {
    if let Some((x, y)) = both_int(a, b) {
        return x.checked_sub(y).map(UV::I);
    }
    Some(UV::F(as_f(a)? - as_f(b)?))
}
fn bin_mul(a: &UV, b: &UV) -> Option<UV> {
    if let Some((x, y)) = both_int(a, b) {
        return x.checked_mul(y).map(UV::I);
    }
    Some(UV::F(as_f(a)? * as_f(b)?))
}
fn bin_div(a: &UV, b: &UV) -> Option<UV> {
    if let Some((x, y)) = both_int(a, b) {
        if y == 0 {
            return None;
        }
        return Some(UV::I(x / y));
    }
    let d = as_f(b)?;
    if d == 0.0 {
        return None;
    }
    Some(UV::F(as_f(a)? / d))
}
fn bin_mod(a: &UV, b: &UV) -> Option<UV> {
    let (x, y) = both_int(a, b)?;
    if y == 0 {
        return None;
    }
    Some(UV::I(x % y))
}
fn bin_minmax(a: &UV, b: &UV, hi: bool) -> Option<UV> {
    if let Some((x, y)) = both_int(a, b) {
        return Some(UV::I(if hi { x.max(y) } else { x.min(y) }));
    }
    let (x, y) = (as_f(a)?, as_f(b)?);
    // Preserve the original typed value of the winner (compare_one keeps type).
    let a_wins = if hi { x >= y } else { x <= y };
    Some(if a_wins { a.clone() } else { b.clone() })
}
fn cmp_eq(a: &UV, b: &UV) -> Option<UV> {
    Some(UV::B(uv_eq(a, b)))
}
fn cmp_lt(a: &UV, b: &UV) -> Option<UV> {
    Some(UV::B(as_f(a)? < as_f(b)?))
}
fn cmp_gt(a: &UV, b: &UV) -> Option<UV> {
    Some(UV::B(as_f(a)? > as_f(b)?))
}

fn fold(l: &[i64], op: &str) -> Option<i64> {
    match op {
        "sum" => Some(l.iter().sum()),
        "product" => Some(l.iter().product()),
        "count" => Some(l.len() as i64),
        "max" => l.iter().copied().max(),
        "min" => l.iter().copied().min(),
        _ => None,
    }
}

fn transform(l: &[i64], t: &str) -> Vec<i64> {
    match t {
        "sortasc" => {
            let mut v = l.to_vec();
            v.sort_unstable();
            v
        }
        "sortdesc" => {
            let mut v = l.to_vec();
            v.sort_unstable();
            v.reverse();
            v
        }
        "reverse" => l.iter().rev().copied().collect(),
        "unique" => {
            let mut out = Vec::new();
            for &x in l {
                if !out.contains(&x) {
                    out.push(x);
                }
            }
            out
        }
        _ => l.to_vec(),
    }
}

/// Emit + re-verify. Returns a SolveResult only if the emitted Mog reproduces
/// every example.
fn finish(
    name: &str,
    ir: &str,
    param_ty: &[&str],
    ret_ty: &str,
    problem: &Problem,
) -> Option<SolveResult> {
    let code = emit(name, ir, param_ty, ret_ty);
    if crate::runtime::code_reproduces_examples(&code, &problem.examples) {
        Some(SolveResult {
            success: true,
            code,
            method: "universal".to_string(),
            error: None,
            metadata: Default::default(),
        })
    } else {
        None
    }
}

fn emit(name: &str, ir: &str, param_ty: &[&str], ret_ty: &str) -> String {
    let params: Vec<String> = param_ty
        .iter()
        .enumerate()
        .map(|(i, t)| format!("a{i}: {t}"))
        .collect();
    let mut helpers: Vec<String> = Vec::new();
    let expr = compile(ir, &mut helpers);
    let mut code = format!(
        "fn {name}({}) -> {ret_ty} {{\n    return {expr};\n}}\n",
        params.join(", ")
    );
    for h in helpers {
        code.push('\n');
        code.push_str(&h);
    }
    code
}

/// Split an IR argument list `L§R` at the TOP-LEVEL separator, respecting
/// nested `(...)`.
fn split_top(s: &str) -> (String, String) {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            '§' if depth == 0 => return (s[..i].to_string(), s[i + '§'.len_utf8()..].to_string()),
            _ => {}
        }
    }
    (s.to_string(), String::new())
}

fn compile(ir: &str, helpers: &mut Vec<String>) -> String {
    if let Some(rest) = ir.strip_prefix('a') {
        if rest.chars().all(|c| c.is_ascii_digit()) {
            return format!("a{rest}");
        }
    }
    if let Some(n) = ir.strip_prefix('#') {
        return n.to_string();
    }
    if let Some(f) = ir.strip_prefix('~') {
        return if f.contains('.') { f.to_string() } else { format!("{f}.0") };
    }
    // Prefix tag then `(...)`.
    let lp = ir.find('(').unwrap_or(ir.len());
    let tag = &ir[..lp];
    let inner = &ir[lp + 1..ir.len() - 1];
    let scalar_bin = |sym: &str, helpers: &mut Vec<String>| {
        let (l, r) = split_top(inner);
        format!("({} {sym} {})", compile(&l, helpers), compile(&r, helpers))
    };
    match tag {
        "Badd" => scalar_bin("+", helpers),
        "Bsub" => scalar_bin("-", helpers),
        "Bmul" => scalar_bin("*", helpers),
        "Bdiv" => scalar_bin("/", helpers),
        "Bmod" => scalar_bin("%", helpers),
        "Ceq" => scalar_bin("==", helpers),
        "Clt" => scalar_bin("<", helpers),
        "Cgt" => scalar_bin(">", helpers),
        "Land" => scalar_bin("&&", helpers),
        "Lor" => scalar_bin("||", helpers),
        "N" => format!("(!{})", compile(inner, helpers)),
        "Mmax" | "Mmin" => {
            let (l, r) = split_top(inner);
            let (lc, rc) = (compile(&l, helpers), compile(&r, helpers));
            let id = helpers.len();
            let fname = format!("mm{id}");
            let cmp = if tag == "Mmax" { ">=" } else { "<=" };
            helpers.push(format!(
                "fn {fname}(p: i64, q: i64) -> i64 {{\n    if p {cmp} q {{\n        return p;\n    }}\n    return q;\n}}\n"
            ));
            format!("{fname}({lc}, {rc})")
        }
        "Rsum" | "Rproduct" | "Rmax" | "Rmin" | "Rcount" => {
            let arg = compile(inner, helpers);
            let id = helpers.len();
            let fname = format!("rd{id}");
            let body = match tag {
                "Rsum" => "acc: i64 = 0;\n    for e in xs {\n        acc = acc + e;\n    }\n    return acc;",
                "Rproduct" => "acc: i64 = 1;\n    for e in xs {\n        acc = acc * e;\n    }\n    return acc;",
                "Rcount" => "return xs.len;",
                "Rmax" => "acc: i64 = xs[0];\n    for e in xs {\n        if e > acc {\n            acc = e;\n        }\n    }\n    return acc;",
                _ => "acc: i64 = xs[0];\n    for e in xs {\n        if e < acc {\n            acc = e;\n        }\n    }\n    return acc;",
            };
            helpers.push(format!("fn {fname}(xs: [i64]) -> i64 {{\n    {body}\n}}\n"));
            format!("{fname}({arg})")
        }
        "Tsortasc" | "Tsortdesc" | "Treverse" | "Tunique" => {
            let arg = compile(inner, helpers);
            let id = helpers.len();
            let fname = format!("tf{id}");
            let body = match tag {
                "Tsortasc" => "out: [i64] = [];\n    for e in xs {\n        out.push(e);\n    }\n    out.sort();\n    return out;",
                "Tsortdesc" => "out: [i64] = [];\n    for e in xs {\n        out.push(e);\n    }\n    out.sort();\n    r: [i64] = [];\n    i: i64 = out.len - 1;\n    while i >= 0 {\n        r.push(out[i]);\n        i = i - 1;\n    }\n    return r;",
                "Treverse" => "out: [i64] = [];\n    i: i64 = xs.len - 1;\n    while i >= 0 {\n        out.push(xs[i]);\n        i = i - 1;\n    }\n    return out;",
                _ => "out: [i64] = [];\n    for e in xs {\n        hit: i64 = 0;\n        for k in out {\n            if k == e {\n                hit = 1;\n            }\n        }\n        if hit == 0 {\n            out.push(e);\n        }\n    }\n    return out;",
            };
            helpers.push(format!("fn {fname}(xs: [i64]) -> [i64] {{\n    {body}\n}}\n"));
            format!("{fname}({arg})")
        }
        _ => compile(inner, helpers),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Example;

    fn prob(rows: Vec<(Vec<Value>, Value)>) -> Problem {
        Problem {
            name: "t".to_string(),
            examples: rows
                .into_iter()
                .map(|(inputs, expected)| Example { inputs, expected })
                .collect(),
            ..Problem::default()
        }
    }

    #[test]
    fn two_arg_add() {
        // add(x, y) = x + y — TWO scalar args, impossible for the single-list combinator.
        let p = prob(vec![
            (vec![Value::Int(0), Value::Int(1)], Value::Int(1)),
            (vec![Value::Int(2), Value::Int(3)], Value::Int(5)),
            (vec![Value::Int(5), Value::Int(7)], Value::Int(12)),
        ]);
        let r = try_universal(&p, "add").expect("must find a0 + a1");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn triangle_area_half_base_height() {
        // area = a * h * 0.5 — float output from int args (Mog promotes on the 0.5).
        let p = prob(vec![
            (vec![Value::Int(5), Value::Int(3)], Value::Float((7.5f64).to_bits())),
            (vec![Value::Int(2), Value::Int(2)], Value::Float((2.0f64).to_bits())),
            (vec![Value::Int(10), Value::Int(8)], Value::Float((40.0f64).to_bits())),
        ]);
        let r = try_universal(&p, "tri").expect("must find a0 * a1 * 0.5");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn compare_one_max() {
        // max(a, b) preserving type.
        let p = prob(vec![
            (vec![Value::Int(1), Value::Int(2)], Value::Int(2)),
            (vec![Value::Int(2), Value::Int(3)], Value::Int(3)),
            (vec![Value::Int(9), Value::Int(4)], Value::Int(9)),
        ]);
        let r = try_universal(&p, "cmp").expect("must find max(a0, a1)");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn refuses_unrelated() {
        let p = prob(vec![
            (vec![Value::Int(1), Value::Int(2)], Value::Int(99)),
            (vec![Value::Int(3), Value::Int(1)], Value::Int(7)),
            (vec![Value::Int(4), Value::Int(5)], Value::Int(2)),
        ]);
        assert!(try_universal(&p, "f").is_none());
    }
}
