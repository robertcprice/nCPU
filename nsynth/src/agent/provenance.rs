//! Provenance certificate + trust verdict (Robustness T1, task #7).
//!
//! A structured, honest answer to "WHY should this synthesized program be
//! trusted?" — surfacing the verification EVIDENCE that the bare
//! `verify_problem_code_strict -> Result<(), String>` throws away. Built entirely
//! from the EXISTING soundness machinery (strict holdout verification + the
//! holdout-source honesty tag + differential consensus), so a certificate can
//! only attest to checks that actually ran — never a rubber stamp. Accepted Mog
//! code uses [`certify`]; exact Rust process artifacts use the narrower
//! crate-internal [`certify_exact_rust_execution`] after their bound sandbox
//! reports and cross-runtime consensus have completed.
//!
//! This is the self-contained trust primitive. Any synthesis path can adopt it by
//! calling [`certify`] with the problem + accepted code + method label; wiring the
//! resulting certificate onto the product-boundary result type is an incremental
//! follow-up (kept out of here so the primitive stays free of that 53-site change).

use crate::agent::consensus::{differential_consensus, ConsensusVerdict};
use crate::benchmark::{generated_holdouts_with_source, HoldoutSource, Problem};
use crate::execution::VerificationReport;
use serde::{Deserialize, Serialize};

/// Runtime that executed the accepted artifact for the evidence in this
/// certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactVerificationPath {
    /// The accepted source was parsed and executed by the internal Mog runtime.
    InternalMog,
    /// The exact Rust source was compiled and executed in the OS process sandbox;
    /// independent candidates remained in Mog for cross-runtime consensus.
    ExactRustProcess,
}

impl Default for ArtifactVerificationPath {
    fn default() -> Self {
        Self::InternalMog
    }
}

impl ArtifactVerificationPath {
    fn is_internal_mog(path: &Self) -> bool {
        matches!(path, Self::InternalMog)
    }
}

/// Evidence that a synthesized program reproduced its spec, with provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceCertificate {
    /// The synthesis method that produced the accepted program.
    pub method: String,
    /// How the holdouts were obtained: `Generated` (labelled by running a reference
    /// over freshly sampled inputs — a real generalization probe) vs `HandFallback`
    /// (degraded to hand-authored holdouts, or examples-only). The honesty tag.
    pub holdout_source: HoldoutSource,
    /// Count of seed examples the program reproduced.
    pub n_examples: usize,
    /// Count of fresh holdouts the program reproduced. `0` means the acceptance
    /// rested on the examples-only robustness floor (no generalization holdouts).
    pub n_holdouts: usize,
    /// Independent-candidate agreement over co-executed probes (differential
    /// consensus). `Ambiguous` here is a soundness WARNING even when strict verify
    /// passed: a second, independently synthesized candidate diverged on a probe.
    pub consensus: ConsensusVerdict,
    /// Which runtime executed the accepted artifact. The default is omitted from
    /// serialized legacy certificates so persisted v1 trace digests remain stable.
    #[serde(
        default,
        skip_serializing_if = "ArtifactVerificationPath::is_internal_mog"
    )]
    pub artifact_verification: ArtifactVerificationPath,
}

impl ProvenanceCertificate {
    /// True when the certificate rests on GENERATED generalization holdouts (the
    /// strongest evidence), not the examples-only / hand-fallback floor.
    pub fn has_generalization_holdouts(&self) -> bool {
        self.n_holdouts > 0 && matches!(self.holdout_source, HoldoutSource::Generated)
    }

    /// True when an independent candidate agreed (or none was found to disagree) —
    /// i.e. the differential consensus surfaced no divergence witness.
    pub fn consensus_clean(&self) -> bool {
        !matches!(self.consensus, ConsensusVerdict::Ambiguous { .. })
    }
}

/// The trust verdict for an accepted candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Reproduced every example + holdout under STRICT verification. Carries the
    /// provenance certificate (holdout source/counts + differential consensus).
    Verified(ProvenanceCertificate),
    /// Failed strict verification (did not reproduce the examples/holdouts).
    Refuted { reason: String },
}

impl Verdict {
    pub fn is_verified(&self) -> bool {
        matches!(self, Verdict::Verified(_))
    }
    pub fn certificate(&self) -> Option<&ProvenanceCertificate> {
        match self {
            Verdict::Verified(c) => Some(c),
            Verdict::Refuted { .. } => None,
        }
    }
}

/// Certify `code` against `problem`: run STRICT holdout verification (the same
/// oracle every engine certifies through) and, on success, build a certificate
/// recording the holdout SOURCE + COUNTS + differential-consensus result. On
/// failure return [`Verdict::Refuted`] with the reason. `method` is the synthesis
/// method label carried through for provenance.
///
/// certify NEVER weakens acceptance: strict verify remains the gate. The consensus
/// result is RECORDED (not enforced) so a strict-pass-but-consensus-ambiguous
/// candidate is visibly flagged rather than silently rubber-stamped.
pub fn certify(problem: &Problem, code: &str, method: &str) -> Verdict {
    if let Err(reason) = crate::runtime::verify_problem_code_strict(problem, code) {
        return Verdict::Refuted { reason };
    }
    let (holdouts, holdout_source) = generated_holdouts_with_source(problem);
    let consensus = differential_consensus(problem, code);
    Verdict::Verified(ProvenanceCertificate {
        method: method.to_string(),
        holdout_source,
        n_examples: problem.examples.len(),
        n_holdouts: holdouts.len(),
        consensus,
        artifact_verification: ArtifactVerificationPath::InternalMog,
    })
}

/// Build provenance for an exact artifact already executed by the process
/// sandbox on the evaluator's visible examples and non-visible holdouts.
///
/// This validates that the concrete reports cover the exact evaluator
/// cardinalities and passed before recording separately computed cross-runtime
/// consensus. The caller binds the reports and artifact digest into an
/// [`crate::agent::runtime::ExecutionTrace`].
pub(crate) fn certify_exact_rust_execution(
    problem: &Problem,
    method: &str,
    visible_report: &VerificationReport,
    holdout_report: &VerificationReport,
    consensus: ConsensusVerdict,
) -> Result<ProvenanceCertificate, String> {
    if method.trim().is_empty() {
        return Err("exact Rust provenance method is empty".into());
    }
    if !visible_report.all_passed() || visible_report.total != problem.examples.len() {
        return Err(format!(
            "exact Rust visible report covers {}/{} passing examples, evaluator requires {}",
            visible_report.passed,
            visible_report.total,
            problem.examples.len()
        ));
    }
    let (holdouts, holdout_source) = generated_holdouts_with_source(problem);
    if !holdout_report.all_passed() || holdout_report.total != holdouts.len() {
        return Err(format!(
            "exact Rust holdout report covers {}/{} passing holdouts, evaluator requires {}",
            holdout_report.passed,
            holdout_report.total,
            holdouts.len()
        ));
    }
    Ok(ProvenanceCertificate {
        method: method.to_string(),
        holdout_source,
        n_examples: visible_report.total,
        n_holdouts: holdout_report.total,
        consensus,
        artifact_verification: ArtifactVerificationPath::ExactRustProcess,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    /// A `square` problem with a runnable reference (so holdouts are GENERATED over
    /// fresh inputs) and multi-digit examples that pin `x*x`.
    fn square_problem() -> Problem {
        let ex = |i: i64, o: i64| Example {
            inputs: vec![Value::Int(i)],
            expected: Value::Int(o),
        };
        Problem {
            name: "square".to_string(),
            category: "arithmetic",
            description: "square a number",
            signature: "fn square(x: i64) -> i64",
            examples: vec![ex(2, 4), ex(3, 9), ex(4, 16), ex(12, 144)],
            reference_code: "fn square(x: i64) -> i64 { return x * x; }",
            ..Default::default()
        }
    }

    #[test]
    fn certifies_correct_program_with_generalization_holdouts() {
        let p = square_problem();
        let v = certify(
            &p,
            "fn square(x: i64) -> i64 { return x * x; }",
            "test-method",
        );
        let cert = v.certificate().expect("x*x must verify");
        assert_eq!(cert.method, "test-method");
        assert_eq!(cert.n_examples, 4);
        // The reference makes holdouts GENERATED over fresh inputs (real
        // generalization evidence), not the examples-only floor.
        assert!(
            cert.has_generalization_holdouts(),
            "expected generated holdouts, got source={:?} n={}",
            cert.holdout_source,
            cert.n_holdouts
        );
    }

    #[test]
    fn refutes_a_wrong_program() {
        let p = square_problem();
        // `x + x` (doubling) reproduces only x=2 (4) and x=0; it fails 3->9 etc., so
        // strict verify must REFUTE it — no certificate.
        let v = certify(
            &p,
            "fn square(x: i64) -> i64 { return x + x; }",
            "test-method",
        );
        assert!(!v.is_verified(), "x+x must be refuted, got {v:?}");
        assert!(v.certificate().is_none());
    }

    #[test]
    fn refutes_the_single_digit_overfit_via_generated_holdouts() {
        // The historical square overfit: sum-of-squares-of-digits agrees with x*x on
        // every SINGLE-digit example but diverges on 12 (5 vs 144). Because the
        // reference makes holdouts GENERATED over fresh (incl. multi-digit) inputs,
        // strict verify REFUTES the overfit — the certificate is never issued.
        let p = square_problem();
        let overfit = "fn square(x: i64) -> i64 { \
                       let mut n: i64 = x; if n < 0 { n = 0 - n; } \
                       let mut s: i64 = 0; \
                       while n > 0 { let d: i64 = n % 10; s = s + d * d; n = n / 10; } \
                       return s; }";
        let v = certify(&p, overfit, "library-pipeline");
        assert!(
            !v.is_verified(),
            "single-digit overfit must be refuted by generated holdouts, got {v:?}"
        );
    }
}
