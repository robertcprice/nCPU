//! Exact bitwise-structured synthesis.
//!
//! Now that the Mog language has bitwise operators (`& | ^ << >>`), this solver
//! recovers rules whose structure is a bit operation: a mask `a & m`, a set
//! `a | m`, a toggle `a ^ m`, or a pairwise combine `a & b` / `a | b` / `a ^ b`,
//! optionally wrapped in an affine `c0 + c1·(a & m)`. These are everywhere in
//! real code (flags, permission masks, parity checks, packing) and were
//! previously inexpressible. Shifts and low-bit masks are deliberately NOT a
//! separate family here: `a << k`, `a >> k`, `a & (2^k-1)` equal `a·2^k`,
//! `a / 2^k`, `a % 2^k`, which `search_composed_features` already recovers.
//!
//! Every candidate is checked against EVERY example by `verified_result`, and
//! the affine-over-feature solve is over-determined, so a coincidental fit is
//! rejected rather than returned — the same recover-or-refuse contract as the
//! rest of the exact family.

use super::search_affine::solve_linear_features;
use super::search_codegen::verified_result;
use super::signature::{scalar_param_names, scalar_params_decl};
use super::*;

/// A bitwise-derived feature: the Mog expression that computes it (for codegen
/// and final verification) plus its value on every example (for the fast in-Rust
/// exact-fit check).
struct BitFeature {
    expr: String,
    col: Vec<i64>,
}

/// Mine candidate mask constants from the data: small positive values that
/// appear as inputs/outputs, the canonical low-bit masks (1,3,7,…,255), and a
/// few derived from output magnitudes. Bounded so the search stays cheap.
fn mine_masks(examples: &[Vec<i64>], targets: &[i64]) -> Vec<i64> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<i64> = BTreeSet::new();
    for m in [1i64, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 127, 255] {
        set.insert(m);
    }
    for row in examples {
        for &v in row {
            if (1..=1024).contains(&v) {
                set.insert(v);
            }
        }
    }
    for &t in targets {
        if (1..=1024).contains(&t) {
            set.insert(t);
        }
    }
    set.into_iter().take(20).collect()
}

/// Build the bitwise feature library: per-argument masks/sets/toggles by mined
/// constant, and pairwise bit-combines. Each feature's column is computed
/// directly; a feature is kept only if it is defined on every example.
fn build_bit_features(examples: &[Vec<i64>], arity: usize, masks: &[i64]) -> Vec<BitFeature> {
    let names = scalar_param_names(arity);
    let n = examples.len();
    let mut feats: Vec<BitFeature> = Vec::new();
    let mut add = |expr: String, col: Vec<i64>| {
        if col.len() == n {
            feats.push(BitFeature { expr, col });
        }
    };
    for j in 0..arity {
        let v = &names[j];
        for &m in masks {
            add(format!("({v} & {m})"), examples.iter().map(|r| r[j] & m).collect());
            add(format!("({v} | {m})"), examples.iter().map(|r| r[j] | m).collect());
            add(format!("({v} ^ {m})"), examples.iter().map(|r| r[j] ^ m).collect());
        }
    }
    // pairwise bit-combines (the genuinely two-argument bit ops)
    for i in 0..arity {
        for j in (i + 1)..arity {
            let (a, b) = (&names[i], &names[j]);
            add(format!("({a} & {b})"), examples.iter().map(|r| r[i] & r[j]).collect());
            add(format!("({a} | {b})"), examples.iter().map(|r| r[i] | r[j]).collect());
            add(format!("({a} ^ {b})"), examples.iter().map(|r| r[i] ^ r[j]).collect());
        }
    }
    feats
}

/// Render `c0 + Σ c_k·term_k` over already-stringified feature terms, dropping
/// zero coefficients, rendering `1·t` as `t`, `-1·t` as `(0 - t)`, `w·t` as
/// `(w) * t` (parenthesised so a negative `w` does not parse as binary minus),
/// and appending `c0` iff non-zero or there is no other term.
fn render_affine_over(c0: i64, terms: &[(&str, i64)]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (t, w) in terms {
        match *w {
            0 => continue,
            1 => parts.push((*t).to_string()),
            -1 => parts.push(format!("(0 - {t})")),
            w => parts.push(format!("({w}) * {t}")),
        }
    }
    if c0 != 0 || parts.is_empty() {
        parts.push(c0.to_string());
    }
    parts.join(" + ")
}

/// Exact bitwise rule: `f(x) = c0 + Σ c_k·g_k(x)` for a SPARSE set of bitwise
/// features `g_k` (1 or 2 of them) plus the raw-argument affine base. Sparse-
/// first and over-determined; the first exact-and-verified fit wins.
pub(super) fn search_bitwise(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    const MARGIN: usize = 2;
    let examples = super::scalar_search::extract_scalar_examples(problem)?;
    let arity = examples.first()?.len();
    if !(1..=2).contains(&arity) || examples.iter().any(|row| row.len() != arity) {
        return None;
    }
    let targets: Vec<i64> = problem.examples.iter().map(|e| e.expected).collect();
    if targets.len() != examples.len() {
        return None;
    }
    let n = examples.len();
    let masks = mine_masks(&examples, &targets);
    let feats = build_bit_features(&examples, arity, &masks);
    if feats.is_empty() {
        return None;
    }
    let names = scalar_param_names(arity);
    let params = scalar_params_decl(&names);

    // Raw-argument columns form the affine base, always available so a rule like
    // `(a & m) + b` is one bitwise feature on top of the line rather than needing
    // the raw var as a separate "bitwise" feature.
    let raw_terms: Vec<(String, Vec<i64>)> =
        (0..arity).map(|j| (names[j].clone(), examples.iter().map(|r| r[j]).collect())).collect();

    // Try `c0 + (raw affine) + Σ chosen bitwise features`, for 1 then 2 features.
    let try_combo = |bit_idx: &[usize]| -> Option<SolveResult> {
        let k = raw_terms.len() + bit_idx.len();
        if n < k + 1 + MARGIN {
            return None;
        }
        let m = k + 1;
        let feature_rows: Vec<Vec<i64>> = (0..n)
            .map(|i| {
                let mut phi = Vec::with_capacity(m);
                phi.push(1);
                for (_, col) in &raw_terms {
                    phi.push(col[i]);
                }
                for &bi in bit_idx {
                    phi.push(feats[bi].col[i]);
                }
                phi
            })
            .collect();
        let w = solve_linear_features(&feature_rows, &targets, m)?;
        // Numeric exact check across all examples.
        for (i, &t) in targets.iter().enumerate() {
            let mut acc = w[0] as i128;
            let mut slot = 1;
            for (_, col) in &raw_terms {
                acc += w[slot] as i128 * col[i] as i128;
                slot += 1;
            }
            for &bi in bit_idx {
                acc += w[slot] as i128 * feats[bi].col[i] as i128;
                slot += 1;
            }
            if acc != t as i128 {
                return None;
            }
        }
        // At least one bitwise feature must carry a non-zero coefficient,
        // otherwise this is a plain affine (search_affine's job).
        let bit_nonzero = bit_idx
            .iter()
            .enumerate()
            .any(|(s, _)| w[1 + raw_terms.len() + s] != 0);
        if !bit_nonzero {
            return None;
        }
        let mut terms: Vec<(&str, i64)> = Vec::with_capacity(m);
        let mut slot = 1;
        for (name, _) in &raw_terms {
            terms.push((name.as_str(), w[slot]));
            slot += 1;
        }
        for &bi in bit_idx {
            terms.push((feats[bi].expr.as_str(), w[slot]));
            slot += 1;
        }
        let body = render_affine_over(w[0], &terms);
        let code = format!("fn {fn_name}({params}) -> i64 {{\n    return {body};\n}}\n");
        verified_result(problem, code, "search_bitwise")
    };

    let fc = feats.len();
    for a in 0..fc {
        if let Some(r) = try_combo(&[a]) {
            return Some(r);
        }
    }
    for a in 0..fc {
        for b in (a + 1)..fc {
            if let Some(r) = try_combo(&[a, b]) {
                return Some(r);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    fn p1(rows: &[(i64, i64)]) -> Problem {
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: "fn f(x: i64) -> i64",
            examples: rows
                .iter()
                .map(|&(x, y)| Example { inputs: vec![Value::Int(x)], expected: y })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        }
    }

    fn p2(rows: &[((i64, i64), i64)]) -> Problem {
        Problem {
            name: "f".to_string(),
            category: "external",
            description: "",
            signature: "fn f(a: i64, b: i64) -> i64",
            examples: rows
                .iter()
                .map(|((a, b), y)| Example {
                    inputs: vec![Value::Int(*a), Value::Int(*b)],
                    expected: *y,
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",
        }
    }

    // A low-bit mask `x & 7` (keep low 3 bits) — recovered and exact on unseen.
    #[test]
    fn bitwise_recovers_mask() {
        let f = |x: i64| x & 7;
        let rows: Vec<(i64, i64)> = (0..20).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_bitwise(&p, "f").expect("must recover x & 7");
        assert!(r.code.contains('&'), "expected a bitwise-and: {}", r.code);
        let check = p1(&[(33, f(33)), (100, f(100)), (255, f(255))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("mask must be exact on unseen points");
    }

    // A pairwise XOR `a ^ b` — recovered and exact on unseen.
    #[test]
    fn bitwise_recovers_pairwise_xor() {
        let f = |a: i64, b: i64| a ^ b;
        let raw = [
            (0, 0), (1, 2), (3, 5), (6, 1), (7, 7), (8, 4), (10, 3), (12, 9), (5, 14), (15, 0),
        ];
        let rows: Vec<((i64, i64), i64)> = raw.iter().map(|&(a, b)| ((a, b), f(a, b))).collect();
        let p = p2(&rows);
        let r = search_bitwise(&p, "f").expect("must recover a ^ b");
        let check = p2(&[((20, 13), f(20, 13)), ((31, 8), f(31, 8)), ((9, 22), f(9, 22))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("xor must be exact on unseen points");
    }

    // An affine wrapping a mask `2*(x & 1) + 5` (even/odd → 5 or 7) — recovered.
    #[test]
    fn bitwise_recovers_affine_of_mask() {
        let f = |x: i64| 2 * (x & 1) + 5;
        let rows: Vec<(i64, i64)> = (0..16).map(|x| (x, f(x))).collect();
        let p = p1(&rows);
        let r = search_bitwise(&p, "f").expect("must recover 2*(x&1)+5");
        let check = p1(&[(20, f(20)), (21, f(21)), (100, f(100)), (101, f(101))]);
        crate::runtime::verify_problem_code_strict(&check, &r.code)
            .expect("affine-of-mask must be exact on unseen points");
    }

    // It refuses noise: no exact bitwise rule, so None rather than a wrong fit.
    #[test]
    fn bitwise_refuses_noise() {
        let ys = [3i64, 7, 1, 9, 2, 8, 4, 6, 0, 5, 11, 13];
        let rows: Vec<(i64, i64)> = ys.iter().enumerate().map(|(i, &y)| (i as i64, y)).collect();
        let p = p1(&rows);
        assert!(search_bitwise(&p, "f").is_none(), "must refuse data with no exact bit rule");
    }
}
