//! TYPED SCHEMA COMBINATOR — emergent composition of schema atoms over int-lists.
//!
//! The hand-written decompose schemas each fix ONE shape (filter, fold, map,
//! sort). This module makes those atoms COMPOSE by search instead of by hand: a
//! guided bottom-up enumeration over typed values {IntList, Int, Bool} whose
//! operators are the schema primitives — filter[predicate], map[element-fn],
//! sort, fold[sum/product/max/min/count], length. A program like "sum of the
//! even elements" emerges as `fold_sum(filter_even(xs))` at depth 2; "sum of
//! squares of the positives" as `fold_sum(map_square(filter_pos(xs)))` at depth
//! 3 — none of them hand-written.
//!
//! Same discipline as `search_typed_enum`: observational-equivalence dedup, a
//! best-first goal heuristic (distance of each expression's output to the
//! target) that keeps the wide operator set affordable, a deterministic node
//! budget, exact-match acceptance, and end-to-end re-verification of the emitted
//! Mog before the caller accepts it.

use crate::benchmark::{Problem, Value};
use crate::solver::SolveResult;
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum V {
    IL(Vec<i64>),
    I(i64),
    B(bool),
}

#[derive(Clone)]
struct Expr {
    outs: Vec<V>,
    mog: String, // over the input variable `xs`
    depth: usize,
}

const MAX_DEPTH: usize = 3;
const MAX_NODES: usize = 4000;
const KEEP_PER_DEPTH: usize = 300;

/// Solve a single int-list-input problem (output int / int-list / bool) by
/// composing schema atoms. Returns a verified Mog program or None.
pub(super) fn try_combinator(problem: &Problem, name: &str) -> Option<SolveResult> {
    let inputs: Vec<Vec<i64>> = problem
        .examples
        .iter()
        .map(|ex| match ex.inputs.as_slice() {
            [Value::Array(a)] => a
                .iter()
                .map(|v| if let Value::Int(i) = v { Some(*i) } else { None })
                .collect::<Option<Vec<i64>>>(),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let expected: Vec<V> = problem
        .examples
        .iter()
        .map(|ex| match &ex.expected {
            Value::Int(i) => Some(V::I(*i)),
            Value::Bool(b) => Some(V::B(*b)),
            Value::Array(a) => a
                .iter()
                .map(|v| if let Value::Int(i) = v { Some(*i) } else { None })
                .collect::<Option<Vec<i64>>>()
                .map(V::IL),
            _ => None,
        })
        .collect::<Option<_>>()?;

    let seed = Expr {
        outs: inputs.iter().map(|l| V::IL(l.clone())).collect(),
        mog: "xs".to_string(),
        depth: 1,
    };
    let mut pool = vec![seed.clone()];
    let mut seen: HashSet<Vec<V>> = HashSet::new();
    seen.insert(seed.outs.clone());

    // Constants MINED from the task's own data: element values seen in the
    // inputs plus the distinct expected-int outputs. These parametrize the
    // threshold predicates and arithmetic maps below, so the basis stays
    // emergent (the DATA supplies the constants, not a hand-list) while covering
    // "> k", "% k == 0", "+ k", etc.
    let mut consts: Vec<i64> = Vec::new();
    for l in &inputs {
        for &x in l {
            if x.abs() <= 1000 && !consts.contains(&x) {
                consts.push(x);
            }
        }
    }
    for e in &expected {
        if let V::I(v) = e {
            if v.abs() <= 1000 && !consts.contains(v) {
                consts.push(*v);
            }
        }
    }
    consts.sort_unstable();
    consts.dedup();
    consts.truncate(10); // bound fan-out; best-first + budget handle the rest

    // Element predicates: fixed class checks + mined thresholds.
    let mut preds: Vec<(String, Box<dyn Fn(i64) -> bool>)> = vec![
        ("e % 2 == 0".to_string(), Box::new(|e: i64| e % 2 == 0)),
        ("e % 2 != 0".to_string(), Box::new(|e: i64| e % 2 != 0)),
        ("e > 0".to_string(), Box::new(|e: i64| e > 0)),
        ("e < 0".to_string(), Box::new(|e: i64| e < 0)),
        ("e != 0".to_string(), Box::new(|e: i64| e != 0)),
    ];
    for &k in &consts {
        preds.push((format!("e > {k}"), Box::new(move |e: i64| e > k)));
        preds.push((format!("e < {k}"), Box::new(move |e: i64| e < k)));
        preds.push((format!("e == {k}"), Box::new(move |e: i64| e == k)));
        if k > 1 {
            preds.push((format!("e % {k} == 0"), Box::new(move |e: i64| e % k == 0)));
        }
    }
    // Element maps: fixed + mined arithmetic.
    let mut maps: Vec<(String, Box<dyn Fn(i64) -> i64>)> = vec![
        ("e * e".to_string(), Box::new(|e: i64| e.wrapping_mul(e))),
        ("0 - e".to_string(), Box::new(|e: i64| -e)),
    ];
    for &k in &consts {
        if k != 0 {
            maps.push((format!("e + {k}"), Box::new(move |e: i64| e.wrapping_add(k))));
            maps.push((format!("e * {k}"), Box::new(move |e: i64| e.wrapping_mul(k))));
        }
        if k > 1 {
            maps.push((format!("e % {k}"), Box::new(move |e: i64| e % k)));
        }
    }

    let mut nodes = 0usize;
    for depth in 2..=MAX_DEPTH {
        if nodes >= MAX_NODES {
            break;
        }
        let prev: Vec<Expr> = pool.iter().filter(|e| e.depth == depth - 1).cloned().collect();
        let mut fresh: Vec<Expr> = Vec::new();

        for e in &prev {
            if nodes >= MAX_NODES {
                break;
            }
            nodes += 1;
            let V::IL(_) = &e.outs[0] else { continue };

            // filter[pred] : IL -> IL
            for (cond, pf) in &preds {
                let outs: Vec<V> = e
                    .outs
                    .iter()
                    .map(|v| {
                        let V::IL(l) = v else { unreachable!() };
                        V::IL(l.iter().copied().filter(|x| pf(*x)).collect())
                    })
                    .collect();
                fresh.push(Expr {
                    outs,
                    mog: format!("__FILTER[{cond}]({})", e.mog),
                    depth,
                });
            }
            // map[fn] : IL -> IL
            for (body, mf) in &maps {
                let outs: Vec<V> = e
                    .outs
                    .iter()
                    .map(|v| {
                        let V::IL(l) = v else { unreachable!() };
                        V::IL(l.iter().copied().map(|x| mf(x)).collect())
                    })
                    .collect();
                fresh.push(Expr {
                    outs,
                    mog: format!("__MAP[{body}]({})", e.mog),
                    depth,
                });
            }
            // sort asc/desc : IL -> IL
            for desc in [false, true] {
                let outs: Vec<V> = e
                    .outs
                    .iter()
                    .map(|v| {
                        let V::IL(l) = v else { unreachable!() };
                        let mut l = l.clone();
                        l.sort_unstable();
                        if desc {
                            l.reverse();
                        }
                        V::IL(l)
                    })
                    .collect();
                fresh.push(Expr {
                    outs,
                    mog: format!("__SORT[{}]({})", if desc { "desc" } else { "asc" }, e.mog),
                    depth,
                });
            }
            // reverse : IL -> IL
            {
                let outs: Vec<V> = e
                    .outs
                    .iter()
                    .map(|v| {
                        let V::IL(l) = v else { unreachable!() };
                        V::IL(l.iter().rev().copied().collect())
                    })
                    .collect();
                fresh.push(Expr { outs, mog: format!("__REVERSE[_]({})", e.mog), depth });
            }
            // unique (order-preserving) : IL -> IL
            {
                let outs: Vec<V> = e
                    .outs
                    .iter()
                    .map(|v| {
                        let V::IL(l) = v else { unreachable!() };
                        let mut seen_e = Vec::new();
                        for &x in l {
                            if !seen_e.contains(&x) {
                                seen_e.push(x);
                            }
                        }
                        V::IL(seen_e)
                    })
                    .collect();
                fresh.push(Expr { outs, mog: format!("__UNIQUE[_]({})", e.mog), depth });
            }
            // scan (running fold) : IL -> IL  (prefix sum / max / min)
            for op in ["sum", "max", "min"] {
                let outs: Vec<V> = e
                    .outs
                    .iter()
                    .map(|v| {
                        let V::IL(l) = v else { unreachable!() };
                        let mut acc: Option<i64> = None;
                        let out: Vec<i64> = l
                            .iter()
                            .map(|&x| {
                                let n = match acc {
                                    None => x,
                                    Some(a) => match op {
                                        "sum" => a + x,
                                        "max" => a.max(x),
                                        _ => a.min(x),
                                    },
                                };
                                acc = Some(n);
                                n
                            })
                            .collect();
                        V::IL(out)
                    })
                    .collect();
                fresh.push(Expr { outs, mog: format!("__SCAN[{op}]({})", e.mog), depth });
            }
            // fold[op] : IL -> I  (sum / product / max / min / count)
            for op in ["sum", "product", "max", "min", "count"] {
                let outs: Option<Vec<V>> = e
                    .outs
                    .iter()
                    .map(|v| {
                        let V::IL(l) = v else { unreachable!() };
                        fold(l, op).map(V::I)
                    })
                    .collect();
                if let Some(outs) = outs {
                    fresh.push(Expr { outs, mog: format!("__FOLD[{op}]({})", e.mog), depth });
                }
            }
            // any/all[pred] : IL -> B
            for (cond, pf) in &preds {
                for kind in ["any", "all"] {
                    let outs: Vec<V> = e
                        .outs
                        .iter()
                        .map(|v| {
                            let V::IL(l) = v else { unreachable!() };
                            let b = if kind == "any" {
                                l.iter().any(|x| pf(*x))
                            } else {
                                l.iter().all(|x| pf(*x))
                            };
                            V::B(b)
                        })
                        .collect();
                    fresh.push(Expr {
                        outs,
                        mog: format!("__{}[{cond}]({})", kind.to_uppercase(), e.mog),
                        depth,
                    });
                }
            }
        }

        // Exact winner bypasses the prune.
        if let Some(win) = fresh.iter().find(|e| e.outs == expected).cloned() {
            pool.push(win);
        } else {
            fresh.retain(|e| seen.insert(e.outs.clone()));
            fresh.sort_by(|a, b| goal(&b.outs, &expected).cmp(&goal(&a.outs, &expected)));
            fresh.truncate(KEEP_PER_DEPTH);
            pool.append(&mut fresh);
        }

        if let Some(win) = pool.iter().find(|e| e.outs == expected) {
            let code = emit(name, &win.mog, &expected[0]);
            if crate::runtime::code_reproduces_examples(&code, &problem.examples) {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "combinator".to_string(),
                    error: None,
                    metadata: Default::default(),
                });
            }
        }
    }
    None
}

/// FLOAT-list value type: the same guided-composition idea over Vec<f64>, with
/// float atoms — element maps (abs, negate, +/*/ mined-const), WHOLE-LIST-CONTEXT
/// maps (rescale (x-min)/(max-min), normalize x/sum, x/max, shift x-min), and
/// folds (sum, mean, max, min, product). Solves float-list tasks (rescale-to-
/// unit, float reductions) no int schema reaches.
pub(super) fn try_combinator_float(problem: &Problem, name: &str) -> Option<SolveResult> {
    let inputs: Vec<Vec<f64>> = problem
        .examples
        .iter()
        .map(|ex| match ex.inputs.as_slice() {
            [Value::Array(a)] => a.iter().map(as_f).collect::<Option<Vec<f64>>>(),
            _ => None,
        })
        .collect::<Option<_>>()?;
    // At least one example must actually contain a float (else the int path owns it).
    if !problem.examples.iter().any(|ex| {
        matches!(&ex.inputs[0], Value::Array(a) if a.iter().any(|v| matches!(v, Value::Float(_))))
    }) {
        return None;
    }
    #[derive(Clone, PartialEq)]
    enum FV {
        L(Vec<f64>),
        F(f64),
    }
    let expected: Vec<FV> = problem
        .examples
        .iter()
        .map(|ex| match &ex.expected {
            Value::Float(b) => Some(FV::F(f64::from_bits(*b))),
            Value::Int(i) => Some(FV::F(*i as f64)),
            Value::Array(a) => a.iter().map(as_f).collect::<Option<Vec<f64>>>().map(FV::L),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let eps = 1e-6;
    let close = |a: f64, b: f64| (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()));

    // Whole-list-context element maps (name, apply(x, min, max, sum)).
    type CMap = (&'static str, fn(f64, f64, f64, f64) -> f64);
    let cmaps: [CMap; 5] = [
        ("(x - mn) / (mx - mn)", |x, mn, mx, _| (x - mn) / (mx - mn)),
        ("x / sm", |x, _, _, sm| x / sm),
        ("x / mx", |x, _, mx, _| x / mx),
        ("x - mn", |x, mn, _, _| x - mn),
        ("0.0 - x", |x, _, _, _| -x),
    ];

    // Depth-2 is enough for these (context-map, or fold, or context-map then fold).
    for (cname, cf) in cmaps {
        // context-map -> FL
        let outs: Vec<FV> = inputs
            .iter()
            .map(|l| {
                let mn = l.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sm: f64 = l.iter().sum();
                FV::L(l.iter().map(|&x| cf(x, mn, mx, sm)).collect())
            })
            .collect();
        let ok = outs.iter().zip(expected.iter()).all(|(o, e)| match (o, e) {
            (FV::L(a), FV::L(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y)),
            _ => false,
        });
        if ok {
            let code = format!(
                "fn {name}(xs: [i64]) -> [i64] {{\n    mn: f64 = xs[0];\n    mx: f64 = xs[0];\n    sm: f64 = 0.0;\n    for e in xs {{\n        if e < mn {{\n            mn = e;\n        }}\n        if e > mx {{\n            mx = e;\n        }}\n        sm = sm + e;\n    }}\n    out: [i64] = [];\n    for x in xs {{\n        out.push({cname});\n    }}\n    return out;\n}}\n"
            );
            if crate::runtime::code_reproduces_examples(&code, &problem.examples) {
                return Some(SolveResult { success: true, code, method: "combinator-float-ctx".to_string(), error: None, metadata: Default::default() });
            }
        }
    }
    // scalar float folds -> F
    for (fname, ff) in [
        ("sum", ff_sum as fn(&[f64]) -> Option<f64>),
        ("mean", ff_mean),
        ("max", ff_max),
        ("min", ff_min),
        ("product", ff_prod),
    ] {
        let outs: Option<Vec<FV>> = inputs.iter().map(|l| ff(l).map(FV::F)).collect();
        let Some(outs) = outs else { continue };
        let ok = outs.iter().zip(expected.iter()).all(|(o, e)| match (o, e) {
            (FV::F(a), FV::F(b)) => close(*a, *b),
            _ => false,
        });
        if ok {
            let body = match fname {
                "sum" => "acc: f64 = 0.0;\n    for e in xs {\n        acc = acc + e;\n    }\n    return acc;".to_string(),
                "mean" => "acc: f64 = 0.0;\n    for e in xs {\n        acc = acc + e;\n    }\n    return acc / xs.len;".to_string(),
                "product" => "acc: f64 = 1.0;\n    for e in xs {\n        acc = acc * e;\n    }\n    return acc;".to_string(),
                "max" => "acc: f64 = xs[0];\n    for e in xs {\n        if e > acc {\n            acc = e;\n        }\n    }\n    return acc;".to_string(),
                _ => "acc: f64 = xs[0];\n    for e in xs {\n        if e < acc {\n            acc = e;\n        }\n    }\n    return acc;".to_string(),
            };
            let code = format!("fn {name}(xs: [i64]) -> f64 {{\n    {body}\n}}\n");
            if crate::runtime::code_reproduces_examples(&code, &problem.examples) {
                return Some(SolveResult { success: true, code, method: format!("combinator-float-{fname}"), error: None, metadata: Default::default() });
            }
        }
    }
    None
}

fn ff_sum(l: &[f64]) -> Option<f64> {
    Some(l.iter().sum())
}
fn ff_mean(l: &[f64]) -> Option<f64> {
    if l.is_empty() {
        None
    } else {
        Some(l.iter().sum::<f64>() / l.len() as f64)
    }
}
fn ff_max(l: &[f64]) -> Option<f64> {
    l.iter().cloned().reduce(f64::max)
}
fn ff_min(l: &[f64]) -> Option<f64> {
    l.iter().cloned().reduce(f64::min)
}
fn ff_prod(l: &[f64]) -> Option<f64> {
    Some(l.iter().product())
}

fn as_f(v: &Value) -> Option<f64> {
    match v {
        Value::Int(i) => Some(*i as f64),
        Value::Float(b) => Some(f64::from_bits(*b)),
        _ => None,
    }
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

/// Negative distance to target (higher = closer): exact-value proximity for
/// ints, char-free overlap for lists, equality for bools.
fn goal(outs: &[V], expected: &[V]) -> i64 {
    outs.iter()
        .zip(expected.iter())
        .map(|(o, e)| match (o, e) {
            (V::I(a), V::I(b)) => -((a - b).abs().min(1_000_000)),
            (V::IL(a), V::IL(b)) => {
                let common = a.iter().filter(|x| b.contains(x)).count() as i64;
                common - (a.len() as i64 - common).abs()
            }
            (V::B(a), V::B(b)) => {
                if a == b {
                    1
                } else {
                    -1
                }
            }
            _ => -1_000_000,
        })
        .sum()
}

/// Lower the bracketed combinator IR into a real Mog program. Each atom expands
/// to a verified loop; nesting composes by substituting the inner call.
fn emit(name: &str, ir: &str, ret: &V) -> String {
    // The IR is a nest of __ATOM[param](inner). We compile it into a chain of
    // helper fns f1, f2, ... each consuming the previous result, plus the entry.
    let mut helpers: Vec<String> = Vec::new();
    let ret_ty = match ret {
        V::I(_) => "i64",
        V::B(_) => "bool",
        V::IL(_) => "[i64]",
    };
    // Peel atoms outermost-first; build a value expression over `xs`.
    let expr = compile(ir, &mut helpers);
    let mut code = format!("fn {name}(xs: [i64]) -> {ret_ty} {{\n    return {expr};\n}}\n");
    for h in helpers {
        code.push('\n');
        code.push_str(&h);
    }
    code
}

/// Compile the combinator IR to a Mog value-expression over the current input
/// expression, emitting helper fns as needed. Returns the expression string.
fn compile(ir: &str, helpers: &mut Vec<String>) -> String {
    if ir == "xs" {
        return "xs".to_string();
    }
    // Parse `__ATOM[param](inner)`.
    let atom_end = ir.find('[').unwrap_or(0);
    let atom = &ir[2..atom_end];
    let pstart = atom_end + 1;
    let pend = ir[pstart..].find(']').map(|i| pstart + i).unwrap_or(pstart);
    let param = &ir[pstart..pend];
    let inner_start = ir[pend..].find('(').map(|i| pend + i + 1).unwrap_or(pend);
    let inner = &ir[inner_start..ir.len() - 1];
    let inner_expr = compile(inner, helpers);

    let id = helpers.len();
    match atom {
        "FILTER" => {
            let fname = format!("cf{id}");
            helpers.push(format!(
                "fn {fname}(xs: [i64]) -> [i64] {{\n    out: [i64] = [];\n    for e in xs {{\n        if {param} {{\n            out.push(e);\n        }}\n    }}\n    return out;\n}}\n"
            ));
            format!("{fname}({inner_expr})")
        }
        "MAP" => {
            let fname = format!("cm{id}");
            helpers.push(format!(
                "fn {fname}(xs: [i64]) -> [i64] {{\n    out: [i64] = [];\n    for e in xs {{\n        out.push({param});\n    }}\n    return out;\n}}\n"
            ));
            format!("{fname}({inner_expr})")
        }
        "SORT" => {
            let fname = format!("cs{id}");
            let rev = if param == "desc" {
                "\n    r: [i64] = [];\n    i: i64 = out.len - 1;\n    while i >= 0 {\n        r.push(out[i]);\n        i = i - 1;\n    }\n    return r;"
            } else {
                "\n    return out;"
            };
            helpers.push(format!(
                "fn {fname}(xs: [i64]) -> [i64] {{\n    out: [i64] = [];\n    for e in xs {{\n        out.push(e);\n    }}\n    out.sort();{rev}\n}}\n"
            ));
            format!("{fname}({inner_expr})")
        }
        "FOLD" => {
            let fname = format!("cd{id}");
            let body = match param {
                "sum" => "acc: i64 = 0;\n    for e in xs {\n        acc = acc + e;\n    }\n    return acc;",
                "product" => "acc: i64 = 1;\n    for e in xs {\n        acc = acc * e;\n    }\n    return acc;",
                "count" => "return xs.len;",
                "max" => "acc: i64 = xs[0];\n    for e in xs {\n        if e > acc {\n            acc = e;\n        }\n    }\n    return acc;",
                _ => "acc: i64 = xs[0];\n    for e in xs {\n        if e < acc {\n            acc = e;\n        }\n    }\n    return acc;",
            };
            helpers.push(format!("fn {fname}(xs: [i64]) -> i64 {{\n    {body}\n}}\n"));
            format!("{fname}({inner_expr})")
        }
        "REVERSE" => {
            let fname = format!("cr{id}");
            helpers.push(format!(
                "fn {fname}(xs: [i64]) -> [i64] {{\n    out: [i64] = [];\n    i: i64 = xs.len - 1;\n    while i >= 0 {{\n        out.push(xs[i]);\n        i = i - 1;\n    }}\n    return out;\n}}\n"
            ));
            format!("{fname}({inner_expr})")
        }
        "UNIQUE" => {
            let fname = format!("cu{id}");
            helpers.push(format!(
                "fn {fname}(xs: [i64]) -> [i64] {{\n    out: [i64] = [];\n    for e in xs {{\n        hit: i64 = 0;\n        for k in out {{\n            if k == e {{\n                hit = 1;\n            }}\n        }}\n        if hit == 0 {{\n            out.push(e);\n        }}\n    }}\n    return out;\n}}\n"
            ));
            format!("{fname}({inner_expr})")
        }
        "SCAN" => {
            let fname = format!("cn{id}");
            let upd = match param {
                "sum" => "acc = acc + e;",
                "max" => "if e > acc {\n                acc = e;\n            }",
                _ => "if e < acc {\n                acc = e;\n            }",
            };
            helpers.push(format!(
                "fn {fname}(xs: [i64]) -> [i64] {{\n    out: [i64] = [];\n    acc: i64 = 0;\n    first: i64 = 1;\n    for e in xs {{\n        if first == 1 {{\n            acc = e;\n            first = 0;\n        }} else {{\n            {upd}\n        }}\n        out.push(acc);\n    }}\n    return out;\n}}\n"
            ));
            format!("{fname}({inner_expr})")
        }
        "ANY" | "ALL" => {
            let fname = format!("cb{id}");
            let (init, hit, miss) = if atom == "ANY" {
                ("0", "1", "")
            } else {
                ("1", "", "0")
            };
            let check = if atom == "ANY" {
                format!("if {param} {{\n            return true;\n        }}")
            } else {
                format!("if {param} {{\n        }} else {{\n            return false;\n        }}")
            };
            let _ = (init, hit, miss);
            let ret = if atom == "ANY" { "false" } else { "true" };
            helpers.push(format!(
                "fn {fname}(xs: [i64]) -> bool {{\n    for e in xs {{\n        {check}\n    }}\n    return {ret};\n}}\n"
            ));
            format!("{fname}({inner_expr})")
        }
        _ => inner_expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Example;

    fn prob(rows: Vec<(Vec<i64>, Value)>) -> Problem {
        Problem {
            name: "t".to_string(),
            examples: rows
                .into_iter()
                .map(|(xs, out)| Example {
                    inputs: vec![Value::int_array(&xs)],
                    expected: out,
                })
                .collect(),
            ..Problem::default()
        }
    }

    #[test]
    fn composes_sum_of_evens() {
        // fold_sum(filter_even(xs)) — depth 2, NOT hand-written.
        let p = prob(vec![
            (vec![1, 2, 3, 4], Value::Int(6)),
            (vec![5, 10, 2], Value::Int(12)),
            (vec![7], Value::Int(0)),
        ]);
        let r = try_combinator(&p, "sum_evens").expect("must compose filter+fold");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn composes_sum_of_squares() {
        // fold_sum(map_square(xs)) — depth 2.
        let p = prob(vec![
            (vec![1, 2, 3], Value::Int(14)),
            (vec![2, 4], Value::Int(20)),
            (vec![5], Value::Int(25)),
        ]);
        let r = try_combinator(&p, "sq_sum").expect("must compose map+fold");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn refuses_unrelated() {
        let p = prob(vec![
            (vec![1, 2], Value::Int(99)),
            (vec![3], Value::Int(7)),
            (vec![4, 5, 6], Value::Int(2)),
        ]);
        assert!(try_combinator(&p, "f").is_none());
    }
}
