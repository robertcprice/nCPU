//! Emergent invariant oracle — a REFERENCE-FREE overfit catcher for array transforms.
//!
//! The distinguishing gate refuses when TWO programs reproduce the examples but DIVERGE on
//! fresh inputs. It is blind to a SOLE overfit: one program that reproduces the examples yet
//! computes the wrong function, with no second program to disagree with. This oracle closes
//! that gap for array→array tasks WITHOUT a reference implementation: it DISCOVERS structural
//! invariants that hold across every example (and are non-trivial — not vacuously true), then
//! requires the candidate to maintain them on FRESH inputs. A program that reproduces the
//! examples but breaks a discovered invariant on a fresh probe is provably not the intended
//! function → refuse. The invariants are discovered from the data, not a hand table, so this
//! generalises to any array transform (emergent, not per-op).
use crate::benchmark::{Example, Value};

#[derive(Clone, Copy, PartialEq, Debug)]
enum ArrayInvariant {
    /// len(out) == len(in) — sort, reverse, map-each.
    LengthPreserved,
    /// out is a permutation of in (same multiset) — sort, reverse, rotate.
    MultisetPreserved,
    /// every element of out occurs in in — filter, dedup, take, select.
    OutputSubsetInput,
    /// out is sorted ascending — sort.
    OutputSortedAsc,
}

/// A single `[i64]` value as a Vec, else None.
fn int_vec(v: &Value) -> Option<Vec<i64>> {
    match v {
        Value::Array(a) => a
            .iter()
            .map(|e| match e {
                Value::Int(i) => Some(*i),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn is_sorted_asc(xs: &[i64]) -> bool {
    xs.windows(2).all(|w| w[0] <= w[1])
}

fn same_multiset(a: &[i64], b: &[i64]) -> bool {
    let (mut x, mut y) = (a.to_vec(), b.to_vec());
    x.sort_unstable();
    y.sort_unstable();
    x == y
}

/// Structural invariants that hold across EVERY (in,out) example and are non-trivial.
/// Only fires for UNARY `[i64] -> [i64]` tasks; returns empty otherwise (no false gate).
fn discover(examples: &[Example]) -> Vec<ArrayInvariant> {
    let rows: Vec<(Vec<i64>, Vec<i64>)> = examples
        .iter()
        .filter_map(|e| {
            if e.inputs.len() != 1 {
                return None;
            }
            Some((int_vec(&e.inputs[0])?, int_vec(&e.expected)?))
        })
        .collect();
    // Require ALL examples to be [i64]->[i64] (else this is not an int-array transform).
    if rows.len() != examples.len() || rows.is_empty() {
        return Vec::new();
    }
    let mut invs = Vec::new();

    // LengthPreserved — non-trivial only if input lengths VARY (else "all length k" is weak).
    let lengths: std::collections::HashSet<usize> = rows.iter().map(|(i, _)| i.len()).collect();
    if lengths.len() > 1 && rows.iter().all(|(i, o)| i.len() == o.len()) {
        invs.push(ArrayInvariant::LengthPreserved);
    }

    // MultisetPreserved — a permutation. Strong; require a non-empty, non-degenerate case.
    if rows.iter().all(|(i, o)| same_multiset(i, o)) && rows.iter().any(|(i, _)| i.len() >= 2) {
        invs.push(ArrayInvariant::MultisetPreserved);
    }

    // OutputSubsetInput — filter/select. Non-trivial only if some output actually DROPS an
    // element (o != i somewhere) and it is not already a permutation (that's MultisetPreserved).
    if !invs.contains(&ArrayInvariant::MultisetPreserved)
        && rows.iter().all(|(i, o)| o.iter().all(|x| i.contains(x)))
        && rows.iter().any(|(i, o)| o.len() < i.len())
    {
        invs.push(ArrayInvariant::OutputSubsetInput);
    }

    // OutputSortedAsc — non-trivial only if some INPUT is not already sorted (else vacuous).
    if rows.iter().all(|(_, o)| is_sorted_asc(o)) && rows.iter().any(|(i, _)| !is_sorted_asc(i)) {
        invs.push(ArrayInvariant::OutputSortedAsc);
    }
    invs
}

fn holds(inv: ArrayInvariant, inp: &[i64], out: &[i64]) -> bool {
    match inv {
        ArrayInvariant::LengthPreserved => out.len() == inp.len(),
        ArrayInvariant::MultisetPreserved => same_multiset(inp, out),
        ArrayInvariant::OutputSubsetInput => out.iter().all(|x| inp.contains(x)),
        ArrayInvariant::OutputSortedAsc => is_sorted_asc(out),
    }
}

/// TRUE when `code` (entry fn `entry`) reproduces the examples but VIOLATES a discovered
/// structural invariant on a fresh probe input — i.e. it is an array-transform OVERFIT the
/// caller must REFUSE. FALSE when there are no discoverable invariants, or the candidate
/// maintains all of them (so this only ever adds refusals for genuine violations — it never
/// forces a solve). Reference-free; complements the distinguishing gate (no second program
/// needed).
pub fn array_transform_overfit(code: &str, entry: &str, examples: &[Example]) -> bool {
    let invs = discover(examples);
    if invs.is_empty() {
        return false;
    }
    // Fresh probes: the example inputs with the first array perturbed (append/reorder/negate),
    // plus a couple of fixed shapes — inputs NOT among the examples.
    for probe in fresh_probes(examples) {
        let Some(inp) = probe.first().and_then(int_vec) else {
            continue;
        };
        let Ok(out_rt) = crate::runtime::execute_function(code, entry, &probe, "inv_oracle") else {
            continue; // undefined on a probe tells us nothing
        };
        let Ok(out_bv) = crate::runtime::benchmark_value_from_runtime(&out_rt) else {
            continue;
        };
        let Some(out) = int_vec(&out_bv) else {
            // An array transform that returns a non-array on a fresh array input is broken.
            return true;
        };
        if invs.iter().any(|&inv| !holds(inv, &inp, &out)) {
            return true;
        }
    }
    false
}

/// Fresh `[i64]` probe inputs for a unary array task: perturbations of the first example's
/// array that are NOT any example input (append a distinct value, reverse, add a negative,
/// and a fixed mixed shape). Small + deterministic (runs against a candidate on the hot path).
fn fresh_probes(examples: &[Example]) -> Vec<Vec<Value>> {
    let seen: std::collections::HashSet<String> =
        examples.iter().map(|e| format!("{:?}", e.inputs)).collect();
    let base: Vec<i64> = examples
        .iter()
        .find_map(|e| e.inputs.first().and_then(int_vec))
        .unwrap_or_default();
    let mut cands: Vec<Vec<i64>> = Vec::new();
    if !base.is_empty() {
        let mut appended = base.clone();
        appended.push(base.iter().max().copied().unwrap_or(0) + 7); // a value likely not present
        cands.push(appended);
        let mut rev = base.clone();
        rev.reverse();
        cands.push(rev);
        let mut neg = base.clone();
        neg.insert(0, -3);
        cands.push(neg);
    }
    cands.push(vec![4, 1, 3, 1, 2]); // fixed mixed shape with a duplicate + unsorted
    cands
        .into_iter()
        .map(|xs| vec![Value::int_array(&xs)])
        .filter(|inp| !seen.contains(&format!("{inp:?}")))
        .collect()
}

/// Monotonic direction of a scalar `i64 -> i64` function, discovered from examples.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Monotonic {
    NonDecreasing,
    NonIncreasing,
}

/// Discover monotonicity of a UNARY `i64 -> i64` task from its examples. Returns a direction
/// only with STRONG evidence: >= 5 DISTINCT inputs spanning a range, outputs strictly follow
/// one direction as inputs increase, and outputs actually vary (not constant). Conservative
/// on purpose — a spurious monotonicity would only ever OVER-refuse (never make a wrong
/// answer), but we still want it rare, so a periodic/modular fn (whose wide-span examples are
/// not monotonic) is not mistaken for monotonic.
fn discover_monotonic(examples: &[Example]) -> Option<Monotonic> {
    let mut pts: Vec<(i64, i64)> = examples
        .iter()
        .filter_map(|e| {
            if e.inputs.len() != 1 {
                return None;
            }
            match (&e.inputs[0], &e.expected) {
                (Value::Int(i), Value::Int(o)) => Some((*i, *o)),
                _ => None,
            }
        })
        .collect();
    if pts.len() != examples.len() {
        return None; // not all i64 -> i64
    }
    pts.sort_unstable();
    pts.dedup_by_key(|p| p.0);
    if pts.len() < 5 {
        return None; // too few distinct inputs to trust a direction
    }
    let nondec = pts.windows(2).all(|w| w[1].1 >= w[0].1);
    let noninc = pts.windows(2).all(|w| w[1].1 <= w[0].1);
    let varies = pts.first().map(|p| p.1) != pts.last().map(|p| p.1);
    if !varies {
        return None; // constant output — no useful monotonic bound
    }
    match (nondec, noninc) {
        (true, false) => Some(Monotonic::NonDecreasing),
        (false, true) => Some(Monotonic::NonIncreasing),
        _ => None,
    }
}

/// TRUE when `code` reproduces the examples of a monotonic `i64 -> i64` task but VIOLATES that
/// monotonicity on a fresh probe: for non-decreasing f, `f(x)` must be >= every example output
/// at an input <= x and <= every example output at an input >= x. A candidate outside those
/// example-derived bounds is not the intended monotonic function -> overfit -> refuse. This
/// catches a SOLE number-theoretic coincidence (e.g. sum_of_primes fit that spikes on a fresh
/// input) the distinguishing gate cannot see. Reference-free; discovered from data.
pub fn scalar_monotonicity_overfit(code: &str, entry: &str, examples: &[Example]) -> bool {
    let Some(dir) = discover_monotonic(examples) else {
        return false;
    };
    let pts: Vec<(i64, i64)> = examples
        .iter()
        .filter_map(|e| match (&e.inputs[0], &e.expected) {
            (Value::Int(i), Value::Int(o)) => Some((*i, *o)),
            _ => None,
        })
        .collect();
    for probe in scalar_fresh_probes(examples) {
        let Some(&Value::Int(x)) = probe.first() else {
            continue;
        };
        let Ok(out_rt) = crate::runtime::execute_function(code, entry, &probe, "inv_mono") else {
            continue;
        };
        let Ok(out_bv) = crate::runtime::benchmark_value_from_runtime(&out_rt) else {
            continue;
        };
        let Value::Int(fx) = out_bv else {
            return true; // a scalar-int task returning a non-int on a fresh input is broken
        };
        // Bounds from the examples that bracket x.
        let below = pts.iter().filter(|(i, _)| *i <= x).map(|(_, o)| *o);
        let above = pts.iter().filter(|(i, _)| *i >= x).map(|(_, o)| *o);
        let violated = match dir {
            Monotonic::NonDecreasing => {
                below.clone().max().is_some_and(|lo| fx < lo)
                    || above.clone().min().is_some_and(|hi| fx > hi)
            }
            Monotonic::NonIncreasing => {
                below.clone().min().is_some_and(|lo| fx > lo)
                    || above.clone().max().is_some_and(|hi| fx < hi)
            }
        };
        if violated {
            return true;
        }
    }
    false
}

/// Fresh single-`i64` probe inputs for a scalar task: values BETWEEN and just BEYOND the
/// example inputs (where a monotonic bound bites), excluding any example input.
fn scalar_fresh_probes(examples: &[Example]) -> Vec<Vec<Value>> {
    let mut ins: Vec<i64> = examples
        .iter()
        .filter_map(|e| match e.inputs.first() {
            Some(Value::Int(i)) => Some(*i),
            _ => None,
        })
        .collect();
    ins.sort_unstable();
    ins.dedup();
    let seen: std::collections::HashSet<i64> = ins.iter().copied().collect();
    let mut cands: Vec<i64> = Vec::new();
    // midpoints between consecutive example inputs
    for w in ins.windows(2) {
        let mid = w[0] + (w[1] - w[0]) / 2;
        if mid != w[0] && mid != w[1] {
            cands.push(mid);
        }
    }
    // just beyond the range
    if let (Some(&lo), Some(&hi)) = (ins.first(), ins.last()) {
        cands.push(lo - 1);
        cands.push(hi + 1);
        cands.push(hi + 3);
    }
    cands
        .into_iter()
        .filter(|x| !seen.contains(x))
        .map(|x| vec![Value::Int(x)])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Value;

    fn ex(inp: &[i64], out: &[i64]) -> Example {
        Example {
            inputs: vec![Value::int_array(inp)],
            expected: Value::int_array(out),
        }
    }

    const SORT: &str =
        "fn f(a: [i64]) -> [i64] {\n    out: [i64] = a;\n    n: i64 = out.len;\n    i: i64 = 0;\n    while i < n {\n        j: i64 = 0;\n        while j < n - 1 {\n            if out[j] > out[j + 1] {\n                t: i64 = out[j];\n                out[j] = out[j + 1];\n                out[j + 1] = t;\n            }\n            j = j + 1;\n        }\n        i = i + 1;\n    }\n    return out;\n}\n";

    #[test]
    fn discovers_sort_invariants_and_passes_a_correct_sort() {
        // Examples span varied lengths + unsorted inputs so the discovery is non-trivial.
        let exs = vec![
            ex(&[3, 1, 2], &[1, 2, 3]),
            ex(&[5, 4], &[4, 5]),
            ex(&[2, 2, 1], &[1, 2, 2]),
            ex(&[9], &[9]),
        ];
        let invs = discover(&exs);
        assert!(invs.contains(&ArrayInvariant::MultisetPreserved), "sort permutes: {invs:?}");
        assert!(invs.contains(&ArrayInvariant::OutputSortedAsc), "sort output sorted: {invs:?}");
        // A CORRECT sort maintains every invariant on fresh probes -> not an overfit.
        assert!(!array_transform_overfit(SORT, "f", &exs), "correct sort must pass");
    }

    #[test]
    fn catches_a_sole_overfit_the_distinguishing_gate_cannot() {
        // "reverse a list" shown only on length-2 inputs. A `swap-first-two` program
        // reproduces every example and is a SOLE plausible passer (the distinguishing gate,
        // needing a SECOND disagreeing program, is blind), yet it is not reverse: on a
        // fresh longer probe it returns a length-2 array — violating MultisetPreserved.
        let exs = vec![ex(&[1, 2], &[2, 1]), ex(&[3, 4], &[4, 3]), ex(&[5, 9], &[9, 5])];
        assert!(discover(&exs).contains(&ArrayInvariant::MultisetPreserved), "reverse permutes");
        let swap = "fn f(a: [i64]) -> [i64] {\n    out: [i64] = [];\n    out.push(a[1]);\n    out.push(a[0]);\n    return out;\n}\n";
        assert!(crate::runtime::code_reproduces_examples(swap, &exs), "swap fits length-2 reverse");
        assert!(
            array_transform_overfit(swap, "f", &exs),
            "swap-first-two overfit must be caught: a fresh longer input breaks MultisetPreserved"
        );
    }

    fn si(inp: i64, out: i64) -> Example {
        Example { inputs: vec![Value::Int(inp)], expected: Value::Int(out) }
    }

    #[test]
    fn catches_a_monotonic_spike_overfit() {
        // A monotonic non-decreasing task (identity on 1..5). An overfit reproduces the
        // examples but collapses to 0 beyond them -> below the f(5)=5 lower bound on a fresh
        // input -> caught. The distinguishing gate (needing a second program) is blind to it.
        let exs = vec![si(1, 1), si(2, 2), si(3, 3), si(4, 4), si(5, 5)];
        assert_eq!(discover_monotonic(&exs), Some(Monotonic::NonDecreasing));
        let overfit = "fn f(x: i64) -> i64 {\n    if x <= 5 {\n        return x;\n    }\n    return 0;\n}\n";
        assert!(crate::runtime::code_reproduces_examples(overfit, &exs), "overfit fits 1..5");
        assert!(scalar_monotonicity_overfit(overfit, "f", &exs), "spike-to-0 breaks monotonicity");
        // A correct monotonic identity passes.
        assert!(!scalar_monotonicity_overfit("fn f(x: i64) -> i64 { return x; }", "f", &exs));
    }

    #[test]
    fn non_monotonic_discovers_nothing() {
        // Not monotonic (dips at 3) -> no direction discovered -> the gate never fires, so a
        // correct non-monotonic function is never over-refused by this oracle.
        let exs = vec![si(0, 0), si(1, 1), si(2, 2), si(3, 0), si(4, 1)];
        assert_eq!(discover_monotonic(&exs), None);
        assert!(!scalar_monotonicity_overfit("fn f(x: i64) -> i64 { return x; }", "f", &exs));
    }

    #[test]
    fn no_invariants_means_no_false_gate() {
        // A scalar task (no array output) discovers nothing -> never gates.
        let scalar = vec![
            Example { inputs: vec![Value::Int(1)], expected: Value::Int(2) },
            Example { inputs: vec![Value::Int(2)], expected: Value::Int(4) },
        ];
        assert!(discover(&scalar).is_empty());
        assert!(!array_transform_overfit("fn f(x: i64) -> i64 { return x * 2; }", "f", &scalar));
    }
}
