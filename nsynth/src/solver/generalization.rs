//! Universal overfit-resistance guards for the exact-fit search families.
//!
//! ## The problem
//! Every exact-fit searcher (affine, threshold, polynomial, the scalar
//! expression enumerator, …) ends by handing its candidate to
//! [`super::search_codegen::verified_result`], which checks the program against
//! the problem's *given* examples (and any holdouts). For an inline natural-
//! language problem there are **no holdouts** — so a candidate that merely
//! *fits* the handful of supplied points passes the gate even when it is wrong
//! everywhere else. Two failure shapes were verified by prior probes:
//!
//!   * `min(a, b)` from 3 examples → `search_affine` solves the 3×3 linear
//!     system `(-6·a) + (-7·b) + 70` — an exact fit to 3 points that is not
//!     `min` at all.
//!   * `min(a, b)` from 7 examples → `search_affine_threshold` emits
//!     `if a <= 3 { a } else { b }` (the `3` copied from the first example),
//!     fitting all 7 training points yet failing `min(5, 100)`.
//!
//! ## The universal principle (no per-operation logic, no oracle)
//! An exactly-determined fit is worthless as evidence of generalization. A
//! linear model with `p` free parameters fit through exactly `p` points passes
//! through *any* `p` targets — there are infinitely many distinct rules that
//! agree on those points and disagree everywhere else. The honest requirement is
//! **strict over-determination**: a fit may only be returned as a confident
//! success when the spec pins it down with *more* independent constraints than
//! the model has free parameters. This is approach (2) ("example-sufficiency vs
//! model DOF") from the mission brief — exact, cheap, and applicable to every
//! family because every family knows its own parameter count.
//!
//! ## Why the gate counts EXAMPLES ONLY (not holdouts)
//! The examples are the *specification*; the holdouts (and `reference_code`) are
//! the hidden *evaluation oracle*. A core invariant of this engine is that search
//! output must be **invariant to the oracle** — synthesising from the same
//! examples must yield the same program whether the holdout set is empty, real,
//! or even adversarially poisoned (see
//! `search_output_is_invariant_to_evaluation_oracles`). If the DOF gate counted
//! holdouts, the number of holdouts would change which fits are accepted and thus
//! the emitted code, breaking that invariant. So the gate measures only whether
//! the *examples* determine the model. This is also the more honest reading of
//! "example-sufficiency": a square *example* system is underdetermined regardless
//! of how many holdouts happen to corroborate it, because the searcher never sees
//! the holdouts while fitting.
//!
//! The companion mechanism for the open-ended scalar **expression** search —
//! where "free parameters" is not a fixed count — is hypothesis disagreement
//! (approach (1)): if two equally-simple programs both reproduce every example
//! but disagree on a fresh in-domain input, the spec is underdetermined and the
//! engine must decline rather than guess. The deterministic in-domain input
//! pool used for that check lives here as [`scalar_probe_inputs`].

use crate::benchmark::Problem;

/// A fit must carry strictly MORE independent constraints than it has free
/// parameters. `1` is the minimum honest margin: it rejects the exactly-
/// determined (square) system — which fits any target and so proves nothing —
/// while still accepting a fit backed by even a single redundant, corroborating
/// point. Larger margins were considered but would reject legitimately thin
/// (margin-1) benchmark specs (e.g. a 4-point plane in two variables), so the
/// guard stays at the principled minimum.
pub(super) const DOF_MARGIN: usize = 1;

/// The number of independent constraints the *specification* places on a
/// candidate: the given examples. Holdouts are deliberately excluded — they are
/// the evaluation oracle, invisible to the searcher while it fits, and counting
/// them would make the emitted program depend on the oracle (violating
/// `search_output_is_invariant_to_evaluation_oracles`). See the module header.
pub(super) fn verification_points(problem: &Problem) -> usize {
    problem.examples.len()
}

/// True iff `points` constraints strictly over-determine a model with `params`
/// free parameters (`points >= params + DOF_MARGIN`). A square
/// (`points == params`) or under-determined system can fit ANY target exactly
/// and therefore carries no evidence of generalization, so it is rejected.
pub(super) fn over_determined(points: usize, params: usize) -> bool {
    points >= params.saturating_add(DOF_MARGIN)
}

/// Whole-problem DOF gate: are there strictly more example constraints than the
/// model's `params` free parameters? This is the single universal acceptance
/// gate the exact-fit families consult before returning a candidate as a
/// confident success.
pub(super) fn problem_over_determined(problem: &Problem, params: usize) -> bool {
    over_determined(verification_points(problem), params)
}

/// FNV-1a hash of a string — a deterministic seed for the probe-input LCG.
/// Never the clock or `rand`, so two synthesis runs of the same problem probe
/// the exact same inputs (reproducible disagreement detection).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash | 1
}

/// One step of a 64-bit linear congruential generator (Numerical-Recipes
/// constants). Deterministic given the seed; used only to sample probe inputs,
/// never for control flow.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn draw_in_range(state: &mut u64, lo: i64, hi: i64) -> i64 {
    if lo >= hi {
        return lo;
    }
    let span = (hi - lo) as u64 + 1;
    lo + (lcg_next(state) % span) as i64
}

/// Build a deterministic pool of in-domain probe inputs for an all-int scalar
/// problem (1–3 arguments), used by the hypothesis-disagreement check. The pool
/// mixes structured edge values (0, ±1, observed min/max and one step beyond
/// each) varied one column at a time around the first example, plus a handful of
/// fully-random rows inside the observed value range. The inputs deliberately
/// land INSIDE (and just outside) the observed domain: a disagreement there
/// means the spec genuinely fails to determine the answer on inputs the caller
/// could plausibly supply, which is the only kind of disagreement that should
/// make the engine decline.
///
/// Determinism: the random rows are driven by an FNV-1a-seeded LCG over the
/// concatenated example bytes, so the pool reproduces exactly across runs and
/// machines. Returns an empty pool for a non-scalar or empty `examples` shape.
pub(super) fn scalar_probe_inputs(examples: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let Some(first) = examples.first() else {
        return Vec::new();
    };
    let arity = first.len();
    if arity == 0 || arity > 3 || examples.iter().any(|row| row.len() != arity) {
        return Vec::new();
    }

    // Observed per-problem value range across every column.
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    for row in examples {
        for &v in row {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if lo > hi {
        lo = -8;
        hi = 8;
    }

    // Seed the LCG from the example bytes so the draws are reproducible.
    let mut seed = {
        let mut bytes = Vec::new();
        for row in examples {
            for &v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        fnv1a(&bytes)
    };

    let mut edge_values = vec![
        0,
        1,
        -1,
        lo,
        hi,
        lo.saturating_sub(1),
        hi.saturating_add(1),
    ];
    edge_values.sort_unstable();
    edge_values.dedup();

    let base = first.clone();
    let mut rows: Vec<Vec<i64>> = Vec::new();
    // Vary one column at a time around the base row.
    for col in 0..arity {
        for &edge in &edge_values {
            let mut row = base.clone();
            row[col] = edge;
            rows.push(row);
        }
    }
    // A few fully-random rows inside the observed range.
    for _ in 0..8 {
        rows.push((0..arity).map(|_| draw_in_range(&mut seed, lo, hi)).collect());
    }

    // Drop rows that coincide with a given example (they carry no new
    // information) and intra-pool duplicates.
    let mut seen: Vec<Vec<i64>> = Vec::new();
    rows.retain(|row| {
        if examples.iter().any(|e| e == row) || seen.iter().any(|s| s == row) {
            false
        } else {
            seen.push(row.clone());
            true
        }
    });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_system_is_not_over_determined() {
        // 3 points, 3 params — square. Fits any target, proves nothing.
        assert!(!over_determined(3, 3));
        // One redundant corroborating point is the minimum honest evidence.
        assert!(over_determined(4, 3));
        assert!(over_determined(5, 3));
    }

    #[test]
    fn probe_inputs_are_deterministic_and_in_domain() {
        let examples = vec![vec![1i64], vec![5], vec![9]];
        let a = scalar_probe_inputs(&examples);
        let b = scalar_probe_inputs(&examples);
        assert_eq!(a, b, "probe pool must reproduce exactly");
        assert!(!a.is_empty());
        // No probe duplicates a given example.
        for probe in &a {
            assert!(!examples.contains(probe));
        }
    }

    #[test]
    fn probe_inputs_reject_bad_shapes() {
        assert!(scalar_probe_inputs(&[]).is_empty());
        // ragged rows
        assert!(scalar_probe_inputs(&[vec![1, 2], vec![3]]).is_empty());
        // arity > 3
        assert!(scalar_probe_inputs(&[vec![1, 2, 3, 4]]).is_empty());
    }
}
