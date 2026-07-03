//! Sketch synthesis (NO MODEL) — supply the program STRUCTURE the flat enumerator
//! lacks (nested loops), then fill the small holes SYSTEMATICALLY.
//!
//! MBPP's unsolved tail is not deep expressions (the enumerator already covers
//! those); it is missing CONTROL STRUCTURE. A huge cluster is nested-pair counting:
//!
//! ```text
//! fn f(arr[, k]) -> i64 { c = 0; for i { for j in i+1.. { if <pred(a[i],a[j],k)> { c += 1 } } } return c }
//! ```
//!
//! — count_pairs (|a−b|==k), get_Pairs_Count (a+b==k), find_even/odd_Pair (parity of
//! a+b), inversions (a>b), same-pair (a==b), … The SKETCH is the loop nest; the only
//! hole is a small predicate, enumerated exhaustively and example-matched. Reliable
//! and fast — no stochastic search needed for a hole this small. Every full match is
//! still strict-verified on held-out, and discoveries feed the self-growing library.
//!
//! Gated on `NSYNTH_SKETCH` for clean A/B while it proves out.

use crate::benchmark::{Problem, Value as BV};
use crate::runtime::verify_problem_code_strict;
use crate::solver::SolveResult;

#[derive(Clone, Copy)]
enum Cmp { Eq, Ne, Lt, Le, Gt, Ge }

/// Left-hand feature of the pair predicate (compared against a right-hand value).
#[derive(Clone, Copy)]
enum Feat {
    Sum,      // a + b
    Diff,     // a - b
    AbsDiff,  // |a - b|
    SumMod2,  // (a + b) % 2
    XorMod2,  // (a ^ b) % 2
    ProdMod2, // (a * b) % 2
    First,    // a   (for a==b style: compare a to b directly)
}

/// Right-hand side of the predicate.
#[derive(Clone, Copy)]
enum Rhs {
    K,        // the scalar arg (2-arg tasks)
    Second,   // b   (compare the feature to the other element)
    Const(i64),
}

struct Pred {
    feat: Feat,
    cmp: Cmp,
    rhs: Rhs,
}

fn cmp(op: Cmp, a: i64, b: i64) -> bool {
    match op {
        Cmp::Eq => a == b,
        Cmp::Ne => a != b,
        Cmp::Lt => a < b,
        Cmp::Le => a <= b,
        Cmp::Gt => a > b,
        Cmp::Ge => a >= b,
    }
}

impl Pred {
    fn holds(&self, a: i64, b: i64, k: i64) -> Option<bool> {
        let lhs = match self.feat {
            Feat::Sum => a.checked_add(b)?,
            Feat::Diff => a.checked_sub(b)?,
            Feat::AbsDiff => a.checked_sub(b)?.abs(),
            Feat::SumMod2 => a.checked_add(b)?.rem_euclid(2),
            Feat::XorMod2 => (a ^ b).rem_euclid(2),
            Feat::ProdMod2 => a.checked_mul(b)?.rem_euclid(2),
            Feat::First => a,
        };
        let rhs = match self.rhs {
            Rhs::K => k,
            Rhs::Second => b,
            Rhs::Const(c) => c,
        };
        Some(cmp(self.cmp, lhs, rhs))
    }

    /// Count matching ordered pairs (i < j) over one array.
    fn count(&self, arr: &[i64], k: i64) -> Option<i64> {
        let mut c = 0i64;
        for i in 0..arr.len() {
            for j in i + 1..arr.len() {
                if self.holds(arr[i], arr[j], k)? {
                    c += 1;
                }
            }
        }
        Some(c)
    }

    fn to_mog(&self, fn_name: &str, two_arg: bool) -> String {
        let params = if two_arg { "arr: [i64], k: i64" } else { "arr: [i64]" };
        let lhs = match self.feat {
            Feat::Sum => "(arr[i] + arr[j])",
            Feat::Diff => "(arr[i] - arr[j])",
            Feat::AbsDiff => "d", // computed into a local below
            Feat::SumMod2 => "((arr[i] + arr[j]) % 2)",
            Feat::XorMod2 => "((arr[i] ^ arr[j]) % 2)",
            Feat::ProdMod2 => "((arr[i] * arr[j]) % 2)",
            Feat::First => "arr[i]",
        };
        let rhs = match self.rhs {
            Rhs::K => "k".to_string(),
            Rhs::Second => "arr[j]".to_string(),
            Rhs::Const(c) => {
                if c < 0 { format!("(0 - {})", -c) } else { c.to_string() }
            }
        };
        let op = match self.cmp {
            Cmp::Eq => "==",
            Cmp::Ne => "!=",
            Cmp::Lt => "<",
            Cmp::Le => "<=",
            Cmp::Gt => ">",
            Cmp::Ge => ">=",
        };
        // AbsDiff needs a local (Mog has no abs on an inline expr here).
        let absdiff_local = if matches!(self.feat, Feat::AbsDiff) {
            "            d: i64 = arr[i] - arr[j];\n            if d < 0 {\n                d = 0 - d;\n            }\n"
        } else {
            ""
        };
        format!(
            "fn {fn_name}({params}) -> i64 {{\n    c: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {{\n        j: i64 = i + 1;\n        while j < arr.len {{\n{absdiff_local}            if {lhs} {op} {rhs} {{\n                c = c + 1;\n            }}\n            j = j + 1;\n        }}\n        i = i + 1;\n    }}\n    return c;\n}}\n"
        )
    }
}

const FEATS: [Feat; 7] = [
    Feat::Sum,
    Feat::Diff,
    Feat::AbsDiff,
    Feat::SumMod2,
    Feat::XorMod2,
    Feat::ProdMod2,
    Feat::First,
];
const CMPS: [Cmp; 6] = [Cmp::Eq, Cmp::Ne, Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge];
const PRED_CONSTS: [i64; 5] = [0, 1, 2, -1, 10];

/// `(arrays, k_or_none, outputs)` for a `[i64](,i64) -> i64` task, or None.
struct CountTask {
    rows: Vec<(Vec<i64>, i64, i64)>, // (arr, k, out); k=0 when 1-arg
    two_arg: bool,
}

fn count_task(problem: &Problem) -> Option<CountTask> {
    let first = problem.examples.first()?;
    let two_arg = match first.inputs.len() {
        1 => false,
        2 => true,
        _ => return None,
    };
    let mut rows = Vec::new();
    for ex in &problem.examples {
        if ex.inputs.len() != first.inputs.len() {
            return None;
        }
        let BV::Array(elems) = &ex.inputs[0] else { return None };
        let arr: Option<Vec<i64>> =
            elems.iter().map(|v| if let BV::Int(n) = v { Some(*n) } else { None }).collect();
        let arr = arr?;
        let k = if two_arg {
            if let BV::Int(k) = ex.inputs[1] { k } else { return None }
        } else {
            0
        };
        let BV::Int(out) = ex.expected else { return None };
        rows.push((arr, k, out));
    }
    Some(CountTask { rows, two_arg })
}

/// Enumerate pair predicates; return a strict-verified program for the first that
/// reproduces every example. `None` unless `NSYNTH_SKETCH` is set / not a count task.
pub fn synthesize_evolve(problem: &Problem) -> Option<SolveResult> {
    if std::env::var_os("NSYNTH_SKETCH").is_none() {
        return None;
    }
    let task = count_task(problem)?;
    for &feat in &FEATS {
        for &cmpop in &CMPS {
            // RHS candidates: k (2-arg only), the other element, and small consts.
            let mut rhss: Vec<Rhs> = vec![Rhs::Second];
            if task.two_arg {
                rhss.push(Rhs::K);
            }
            rhss.extend(PRED_CONSTS.iter().map(|&c| Rhs::Const(c)));
            for rhs in rhss {
                let pred = Pred { feat, cmp: cmpop, rhs };
                let all = task
                    .rows
                    .iter()
                    .all(|(arr, k, out)| pred.count(arr, *k) == Some(*out));
                if all {
                    let code = pred.to_mog(problem.function_name(), task.two_arg);
                    if verify_problem_code_strict(problem, &code).is_ok() {
                        return Some(SolveResult {
                            success: true,
                            code,
                            method: "sketch-pair-count".to_string(),
                            error: None,
                            metadata: Default::default(),
                        });
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Example;

    fn arr_problem(name: &'static str, sig: &'static str, rows: Vec<(Vec<i64>, Option<i64>, i64)>) -> Problem {
        let mut p = Problem::default();
        p.name = name.to_string();
        p.signature = sig;
        p.examples = rows
            .into_iter()
            .map(|(a, k, o)| {
                let mut inputs = vec![BV::Array(a.into_iter().map(BV::Int).collect())];
                if let Some(k) = k {
                    inputs.push(BV::Int(k));
                }
                Example { inputs, expected: BV::Int(o) }
            })
            .collect();
        p
    }

    #[test]
    fn sketch_solves_pair_diff_count() {
        std::env::set_var("NSYNTH_SKETCH", "1");
        // count pairs (i<j) with |a[i]-a[j]| == k  (MBPP count_pairs shape).
        let f = |a: &[i64], k: i64| -> i64 {
            let mut c = 0;
            for i in 0..a.len() {
                for j in i + 1..a.len() {
                    if (a[i] - a[j]).abs() == k {
                        c += 1;
                    }
                }
            }
            c
        };
        let cases: &[(&[i64], i64)] =
            &[(&[1, 5, 3, 4, 2], 3), (&[8, 12, 16, 4, 0, 20], 4), (&[1, 2, 3, 4, 5], 1)];
        let rows: Vec<(Vec<i64>, Option<i64>, i64)> =
            cases.iter().map(|(a, k)| (a.to_vec(), Some(*k), f(a, *k))).collect();
        let p = arr_problem("count_pairs", "fn count_pairs(arr: [i64], k: i64) -> i64", rows);
        let r = synthesize_evolve(&p);
        std::env::remove_var("NSYNTH_SKETCH");
        let r = r.expect("sketch should solve pair-diff counting");
        assert_eq!(r.method, "sketch-pair-count");
        // held-out
        assert!(crate::runtime::code_reproduces_examples(
            &r.code,
            &[Example {
                inputs: vec![BV::Array(vec![10, 7, 13, 4].into_iter().map(BV::Int).collect()), BV::Int(3)],
                expected: BV::Int(f(&[10, 7, 13, 4], 3)),
            }]
        ), "must generalise:\n{}", r.code);
    }

    #[test]
    fn sketch_solves_inversions() {
        std::env::set_var("NSYNTH_SKETCH", "1");
        // count pairs (i<j) with a[i] > a[j]  (inversions, 1-arg).
        let f = |a: &[i64]| -> i64 {
            let mut c = 0;
            for i in 0..a.len() {
                for j in i + 1..a.len() {
                    if a[i] > a[j] {
                        c += 1;
                    }
                }
            }
            c
        };
        let cases: &[&[i64]] = &[&[1, 20, 6, 4, 5], &[3, 1, 2], &[5, 4, 3, 2, 1], &[1, 2, 3]];
        let rows: Vec<(Vec<i64>, Option<i64>, i64)> =
            cases.iter().map(|a| (a.to_vec(), None, f(a))).collect();
        let p = arr_problem("inversions", "fn inversions(arr: [i64]) -> i64", rows);
        let r = synthesize_evolve(&p);
        std::env::remove_var("NSYNTH_SKETCH");
        let r = r.expect("sketch should solve inversions");
        assert_eq!(r.method, "sketch-pair-count");
    }
}
