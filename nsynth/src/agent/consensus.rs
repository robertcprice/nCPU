//! Differential-consensus gate for examples-only specs (no oracle).
//!
//! When a spec arrives as examples ONLY — no runnable `reference_code`, no
//! hand-authored `holdouts` — strict verification has no ground truth. The
//! robustness floor in [`crate::runtime::verify_problem_code_strict`] can then
//! only require the candidate to execute cleanly on fresh in-distribution inputs;
//! it cannot catch a TOTAL-but-WRONG overfit (e.g. a constant that matches the
//! visible examples and never crashes).
//!
//! This gate adds the missing correctness signal WITHOUT an oracle: synthesize
//! one or more INDEPENDENT candidates and require them to AGREE with the accepted
//! candidate on fresh inputs. Independence is essential — two programs from the
//! same deterministic path are identical and agree vacuously. We obtain it two
//! ways:
//!   (a) the bottom-up [`crate::enumerative::synthesize_enumerative`] engine,
//!       independent of the search portfolio that typically produces the agent's
//!       accepted candidate; and
//!   (b) leave-one-out re-synthesis on each `(n-1)`-example subset — a candidate
//!       that changes behavior when one example is withheld is overfitting it.
//!
//! Verdicts:
//!   - [`ConsensusVerdict::Verified`]   — >= 1 independent candidate, all agree
//!     with the accepted candidate on every probe both could execute.
//!   - [`ConsensusVerdict::Ambiguous`]  — an independent candidate DISAGREES on a
//!     probe → the spec is underdetermined or the accepted candidate overfits →
//!     the caller should fail closed / pivot to clarification.
//!   - [`ConsensusVerdict::NoConsensus`] — no independent candidate could be found
//!     (a tight or hard spec): no extra evidence either way → the caller decides
//!     (honest examples-only label, or fail closed under a strict policy).

use crate::benchmark::{robustness_probe_inputs, Problem, Value};
use crate::runtime::{execute_function_for_problem, outputs_equal, verify_problem_code};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsensusVerdict {
    /// `agreeing` independent candidates agreed with the accepted candidate over
    /// `probes` co-executed probe inputs.
    Verified { agreeing: usize, probes: usize },
    /// An independent candidate produced a DIFFERENT output on `witness`.
    Ambiguous { witness: Vec<Value> },
    /// No independent candidate was found (or none co-executed on any probe).
    NoConsensus,
}

/// Whether `problem` is an examples-only spec with NO oracle — the only regime
/// where this gate adds value. With a reference/property, the strict verifier's
/// reference-derived holdouts already give a real differential correctness check.
pub fn is_examples_only(problem: &Problem) -> bool {
    problem.reference_code.is_empty() && problem.holdouts.is_empty()
}

/// Wall-clock budget (ms) for gathering independent candidates. The leave-one-out
/// pass re-runs the full solver once PER example, and a single solve on a hard task
/// can burn ~30s (a search STAGE runs to completion internally, past the between-
/// stages `NSYNTH_SOLVE_BUDGET_MS` check) — so N examples = N×30s, e.g. a measured
/// 250s stall on nth_composite that froze the model-tier front door
/// (`verified_nl_router`→rlvr verify→consensus) for minutes. Bounding gathering by
/// wall clock is SOUND: fewer corroborators can only reduce a would-be `Verified`
/// to `NoConsensus` (→ Tentative) or miss an `Ambiguous`, never MANUFACTURE a false
/// `Verified` — the gate can only get MORE conservative, never confidently wrong.
/// Default 20s (override `NSYNTH_CONSENSUS_BUDGET_MS`); the cheap enumerative
/// candidate (a) always runs first since it is the usual fast corroborator.
fn consensus_budget_ms() -> u128 {
    std::env::var("NSYNTH_CONSENSUS_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000)
}

/// Gather INDEPENDENT candidates that also satisfy every visible example and
/// differ textually from the accepted candidate.
fn independent_candidates(problem: &Problem, accepted_code: &str) -> Vec<String> {
    let mut raw: Vec<String> = Vec::new();
    let start = std::time::Instant::now();
    let budget = consensus_budget_ms();

    // (a) bottom-up enumerative engine.
    if let Some(r) = crate::enumerative::synthesize_enumerative(problem) {
        if r.success {
            raw.push(r.code);
        }
    }

    // (b) leave-one-out re-synthesis (only meaningful with >= 2 examples), bounded
    // by the wall-clock budget so one pathological subset-solve cannot stall the
    // whole gate. Checked BEFORE each solve; the in-flight solve is not interrupted
    // (no cooperative deadline inside the search stages yet), so the true bound is
    // budget + one-solve overshoot — still bounded, unlike the unbounded N×solve.
    if problem.examples.len() >= 2 {
        for omit in 0..problem.examples.len() {
            if start.elapsed().as_millis() > budget {
                break;
            }
            let mut sub = problem.clone();
            sub.examples = problem
                .examples
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != omit)
                .map(|(_, e)| e.clone())
                .collect();
            let r = crate::solver::solve_problem(&sub);
            if r.success {
                raw.push(r.code);
            }
        }
    }

    // Keep only distinct candidates that (i) differ from the accepted one and
    // (ii) still fit ALL visible examples — a leave-one-out solve must not have
    // drifted off the full spec.
    let mut out: Vec<String> = Vec::new();
    for code in raw {
        if code != accepted_code
            && !out.contains(&code)
            && verify_problem_code(problem, &code).is_ok()
        {
            out.push(code);
        }
    }
    out
}

/// Run the differential-consensus gate. See the module docs for the verdicts.
pub fn differential_consensus(problem: &Problem, accepted_code: &str) -> ConsensusVerdict {
    let independents = independent_candidates(problem, accepted_code);
    if independents.is_empty() {
        return ConsensusVerdict::NoConsensus;
    }

    let probes = robustness_probe_inputs(problem);
    let fn_name = problem.function_name();
    let mut checked = 0usize;
    for inputs in &probes {
        // If the accepted candidate is undefined on this probe, skip it — we have
        // nothing to compare against (the robustness floor already ran clean-exec
        // on the accepted candidate before this gate).
        let base = match execute_function_for_problem(accepted_code, fn_name, inputs, problem) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mut compared_here = false;
        for cand in &independents {
            match execute_function_for_problem(cand, fn_name, inputs, problem) {
                Ok(other) => {
                    compared_here = true;
                    if !outputs_equal(&base, &other) {
                        return ConsensusVerdict::Ambiguous {
                            witness: inputs.clone(),
                        };
                    }
                }
                // Independent candidate undefined here while the accepted one is
                // defined: a definedness divergence. Conservatively SKIP (not a
                // disagreement) so a genuine partial function is not false-
                // rejected — VALUE divergence is the sound overfit signal.
                Err(_) => {}
            }
        }
        if compared_here {
            checked += 1;
        }
    }

    if checked == 0 {
        return ConsensusVerdict::NoConsensus;
    }
    ConsensusVerdict::Verified {
        agreeing: independents.len(),
        probes: checked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value as BmValue};

    fn affine_problem(name: &str, pairs: &[(i64, i64)]) -> Problem {
        let leaked: &'static str =
            Box::leak(format!("fn {name}(a: i64) -> i64").into_boxed_str());
        Problem {
            name: name.to_string(),
            category: "test",
            description: "test",
            signature: leaked,
            examples: pairs
                .iter()
                .map(|(i, o)| Example {
                    inputs: vec![BmValue::Int(*i)],
                    expected: BmValue::Int(*o),
                })
                .collect(),
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn is_examples_only_detects_regime() {
        let mut p = affine_problem("eo_detect_v0", &[(1, 2)]);
        assert!(is_examples_only(&p), "no reference/holdouts → examples-only");
        p.reference_code = "fn eo_detect_v0(a: i64) -> i64 { return a + 1; }";
        assert!(!is_examples_only(&p), "a reference oracle exits the regime");
    }

    /// THE VALUE OVER TIER 1: a determined multi-example affine spec gets a
    /// CONSENSUS verdict — leave-one-out / enumerative independents agree with a
    /// correct candidate on fresh probes.
    #[test]
    fn consensus_verifies_determined_affine() {
        let problem = affine_problem(
            "consensus_affine_ok_v0",
            &[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        );
        let correct = "fn consensus_affine_ok_v0(a: i64) -> i64 { return a + 1; }";
        let verdict = differential_consensus(&problem, correct);
        assert!(
            matches!(verdict, ConsensusVerdict::Verified { .. }),
            "a determined affine spec with a correct candidate must reach \
             consensus, got {verdict:?}"
        );
    }

    /// THE OVERFIT TIER 1 MISSES: a candidate that matches every visible example
    /// but DIVERGES off-distribution is caught as Ambiguous (an independent
    /// candidate disagrees). The robustness floor passes this (it never crashes);
    /// only consensus catches it.
    #[test]
    fn consensus_catches_offspec_divergent_overfit() {
        let problem = affine_problem(
            "consensus_overfit_v0",
            &[(1, 2), (2, 3), (3, 4), (4, 5)],
        );
        // Matches all four examples (a in 1..=4 → a+1) but returns a wrong
        // constant elsewhere — a total function, so the robustness floor accepts.
        let overfit = "fn consensus_overfit_v0(a: i64) -> i64 { \
                       if a <= 4 { return a + 1; } return 0; }";
        // Sanity: it really does satisfy the visible examples and the floor.
        verify_problem_code(&problem, overfit).expect("overfit matches examples");
        crate::runtime::verify_problem_code_strict(&problem, overfit)
            .expect("robustness floor (no oracle) does not catch a total overfit");
        // Consensus does: an independent candidate (a+1) disagrees off-window.
        let verdict = differential_consensus(&problem, overfit);
        assert!(
            matches!(verdict, ConsensusVerdict::Ambiguous { .. }),
            "consensus must flag an off-window divergent overfit, got {verdict:?}"
        );
    }
}
