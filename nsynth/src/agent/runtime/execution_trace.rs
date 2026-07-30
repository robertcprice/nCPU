//! Durable verifier trace and canonical capability-admission proposal.
//!
//! A trace records what the bounded executor and independent verifier actually
//! observed. It cannot promote a capability merely because execution returned
//! successfully: canonical admission additionally requires generated holdouts,
//! affirmative differential consensus, explicit LinguaGenesis lineage, and a
//! budget that did not overrun. The admission object is a proposal for the
//! canonical LinguaGenesis graph, not a second local knowledge store.

use super::{
    AgentRunBudget, AgentRunId, CapsuleError, ContentDigest, ExecutionCapsule, SCHEMA_VERSION,
};
use crate::agent::consensus::ConsensusVerdict;
use crate::agent::provenance::ProvenanceCertificate;
use crate::benchmark::HoldoutSource;
use crate::execution::VerificationReport;
use linguigenesis_core::entity::EntityId;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionTraceOutcome {
    Verified,
    Refuted,
    ExecutionFailed,
    BudgetExhausted,
    PolicyDenied,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionFailureKind {
    CapsuleIntegrity,
    Compilation,
    Sandbox,
    Timeout,
    OutputLimit,
    Verification,
    BudgetExhausted,
    PolicyDenied,
    UnsupportedInterface,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionFailure {
    pub kind: ExecutionFailureKind,
    pub message: String,
}

impl ExecutionFailure {
    pub fn new(kind: ExecutionFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// Content-addressed result of executing and verifying one capsule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub schema_version: u32,
    pub run_id: AgentRunId,
    pub capsule_digest: ContentDigest,
    pub artifact_digest: ContentDigest,
    pub evidence_entity_ids: Vec<EntityId>,
    pub outcome: ExecutionTraceOutcome,
    pub verification: Option<VerificationReport>,
    pub generalization_verification: Option<VerificationReport>,
    pub provenance: Option<ProvenanceCertificate>,
    pub failure: Option<ExecutionFailure>,
    pub budget_after: AgentRunBudget,
    pub used_capabilities: Vec<String>,
    pub trace_digest: ContentDigest,
}

#[derive(Serialize)]
struct TracePayload<'a> {
    schema_version: u32,
    run_id: &'a AgentRunId,
    capsule_digest: &'a ContentDigest,
    artifact_digest: &'a ContentDigest,
    evidence_entity_ids: &'a [EntityId],
    outcome: ExecutionTraceOutcome,
    verification: &'a Option<VerificationReport>,
    generalization_verification: &'a Option<VerificationReport>,
    provenance: &'a Option<ProvenanceCertificate>,
    failure: &'a Option<ExecutionFailure>,
    budget_after: &'a AgentRunBudget,
    used_capabilities: &'a [String],
}

impl ExecutionTrace {
    /// Build a verified or refuted trace from the sandbox report. A passing
    /// report must carry the independent strict-verification certificate.
    pub fn from_report(
        capsule: &ExecutionCapsule,
        verification: VerificationReport,
        generalization_verification: Option<VerificationReport>,
        provenance: Option<ProvenanceCertificate>,
        budget_after: AgentRunBudget,
        mut used_capabilities: Vec<String>,
    ) -> Result<Self, TraceError> {
        capsule.validate_integrity().map_err(TraceError::Capsule)?;
        canonicalize_capabilities(&mut used_capabilities);
        validate_used_capabilities(capsule, &used_capabilities)?;
        validate_report_shape(capsule, &verification)?;
        if let Some(report) = &generalization_verification {
            validate_trace_report_shape(report)?;
        }

        let all_passed = verification.all_passed()
            && generalization_verification
                .as_ref()
                .map_or(true, VerificationReport::all_passed);
        let outcome = if all_passed {
            let certificate = provenance.as_ref().ok_or(TraceError::MissingProvenance)?;
            if certificate.method != capsule.artifact.synthesis_method {
                return Err(TraceError::ProvenanceMethodMismatch);
            }
            if certificate.n_examples != verification.total {
                return Err(TraceError::ProvenanceExampleMismatch);
            }
            let executed_holdouts = generalization_verification
                .as_ref()
                .map_or(0, |report| report.total);
            if certificate.n_holdouts != executed_holdouts {
                return Err(TraceError::ProvenanceHoldoutMismatch);
            }
            ExecutionTraceOutcome::Verified
        } else {
            if provenance.is_some() {
                return Err(TraceError::ProvenanceOnRefutedTrace);
            }
            ExecutionTraceOutcome::Refuted
        };

        let mut trace = Self {
            schema_version: SCHEMA_VERSION,
            run_id: capsule.run_id.clone(),
            capsule_digest: capsule.capsule_digest.clone(),
            artifact_digest: capsule.artifact.source_digest.clone(),
            evidence_entity_ids: capsule.artifact.evidence_entity_ids.clone(),
            outcome,
            verification: Some(verification),
            generalization_verification,
            provenance,
            failure: None,
            budget_after,
            used_capabilities,
            trace_digest: ContentDigest::sha256(&[]),
        };
        trace.trace_digest = trace.recompute_digest()?;
        trace.validate_integrity()?;
        Ok(trace)
    }

    /// Record a typed fail-closed outcome when no verification report exists.
    pub fn from_failure(
        capsule: &ExecutionCapsule,
        failure: ExecutionFailure,
        budget_after: AgentRunBudget,
        mut used_capabilities: Vec<String>,
    ) -> Result<Self, TraceError> {
        capsule.validate_integrity().map_err(TraceError::Capsule)?;
        if failure.message.trim().is_empty() {
            return Err(TraceError::EmptyFailure);
        }
        canonicalize_capabilities(&mut used_capabilities);
        validate_used_capabilities(capsule, &used_capabilities)?;
        let outcome = match failure.kind {
            ExecutionFailureKind::BudgetExhausted => ExecutionTraceOutcome::BudgetExhausted,
            ExecutionFailureKind::PolicyDenied => ExecutionTraceOutcome::PolicyDenied,
            _ => ExecutionTraceOutcome::ExecutionFailed,
        };
        let mut trace = Self {
            schema_version: SCHEMA_VERSION,
            run_id: capsule.run_id.clone(),
            capsule_digest: capsule.capsule_digest.clone(),
            artifact_digest: capsule.artifact.source_digest.clone(),
            evidence_entity_ids: capsule.artifact.evidence_entity_ids.clone(),
            outcome,
            verification: None,
            generalization_verification: None,
            provenance: None,
            failure: Some(failure),
            budget_after,
            used_capabilities,
            trace_digest: ContentDigest::sha256(&[]),
        };
        trace.trace_digest = trace.recompute_digest()?;
        trace.validate_integrity()?;
        Ok(trace)
    }

    /// Strong admission boundary, deliberately narrower than "tests passed."
    pub fn admission_eligible(&self) -> bool {
        if self.validate_integrity().is_err()
            || self.outcome != ExecutionTraceOutcome::Verified
            || self.evidence_entity_ids.is_empty()
            || !budget_within_limits(&self.budget_after)
        {
            return false;
        }
        self.verification
            .as_ref()
            .is_some_and(VerificationReport::all_passed)
            && self
                .generalization_verification
                .as_ref()
                .is_some_and(VerificationReport::all_passed)
            && self.provenance.as_ref().is_some_and(|certificate| {
                certificate.n_holdouts > 0
                    && self
                        .generalization_verification
                        .as_ref()
                        .is_some_and(|report| report.total == certificate.n_holdouts)
                    && certificate.holdout_source == HoldoutSource::Generated
                    && matches!(
                        certificate.consensus,
                        ConsensusVerdict::Verified { agreeing, probes }
                            if agreeing > 0 && probes > 0
                    )
            })
    }

    pub fn validate_integrity(&self) -> Result<(), TraceError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(TraceError::SchemaVersion {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.run_id.0.trim().is_empty() {
            return Err(TraceError::EmptyRunId);
        }
        if !strictly_sorted_unique(&self.evidence_entity_ids)
            || !strictly_sorted_unique(&self.used_capabilities)
        {
            return Err(TraceError::NonCanonicalCollections);
        }
        if !self.capsule_digest.is_well_formed() || !self.artifact_digest.is_well_formed() {
            return Err(TraceError::DigestMismatch);
        }
        match self.outcome {
            ExecutionTraceOutcome::Verified => {
                let report = self
                    .verification
                    .as_ref()
                    .ok_or(TraceError::MissingVerification)?;
                let certificate = self
                    .provenance
                    .as_ref()
                    .ok_or(TraceError::MissingProvenance)?;
                if !report.all_passed()
                    || self.failure.is_some()
                    || certificate.n_examples != report.total
                {
                    return Err(TraceError::OutcomeInvariant);
                }
                validate_trace_report_shape(report)?;
                let generalization = self
                    .generalization_verification
                    .as_ref()
                    .ok_or(TraceError::MissingGeneralizationVerification)?;
                validate_trace_report_shape(generalization)?;
                if !generalization.all_passed()
                    || generalization.total == 0
                    || certificate.n_holdouts != generalization.total
                {
                    return Err(TraceError::OutcomeInvariant);
                }
            }
            ExecutionTraceOutcome::Refuted => {
                let report = self
                    .verification
                    .as_ref()
                    .ok_or(TraceError::MissingVerification)?;
                if (report.all_passed()
                    && self
                        .generalization_verification
                        .as_ref()
                        .map_or(true, VerificationReport::all_passed))
                    || self.provenance.is_some()
                    || self.failure.is_some()
                {
                    return Err(TraceError::OutcomeInvariant);
                }
                validate_trace_report_shape(report)?;
                if let Some(generalization) = &self.generalization_verification {
                    validate_trace_report_shape(generalization)?;
                }
            }
            outcome @ (ExecutionTraceOutcome::ExecutionFailed
            | ExecutionTraceOutcome::BudgetExhausted
            | ExecutionTraceOutcome::PolicyDenied) => {
                let failure = self.failure.as_ref().ok_or(TraceError::OutcomeInvariant)?;
                if self.verification.is_some()
                    || self.generalization_verification.is_some()
                    || self.provenance.is_some()
                    || failure.message.trim().is_empty()
                    || !failure_matches_outcome(failure.kind, outcome)
                {
                    return Err(TraceError::OutcomeInvariant);
                }
            }
        }
        let expected = self.recompute_digest()?;
        if !self.trace_digest.is_well_formed() || self.trace_digest != expected {
            return Err(TraceError::DigestMismatch);
        }
        Ok(())
    }

    /// Re-bind a standalone trace to the exact capsule it claims to report.
    /// This is the load/persistence boundary: a valid trace digest alone is not
    /// enough if it was paired with a different task, artifact, or policy.
    pub fn validate_against_capsule(&self, capsule: &ExecutionCapsule) -> Result<(), TraceError> {
        capsule.validate_integrity().map_err(TraceError::Capsule)?;
        self.validate_integrity()?;
        if self.run_id != capsule.run_id
            || self.capsule_digest != capsule.capsule_digest
            || self.artifact_digest != capsule.artifact.source_digest
            || self.evidence_entity_ids != capsule.artifact.evidence_entity_ids
        {
            return Err(TraceError::TraceCapsuleMismatch);
        }
        validate_used_capabilities(capsule, &self.used_capabilities)?;
        if let Some(report) = &self.verification {
            validate_report_shape(capsule, report)?;
        }
        Ok(())
    }

    pub fn recompute_digest(&self) -> Result<ContentDigest, TraceError> {
        let payload = TracePayload {
            schema_version: self.schema_version,
            run_id: &self.run_id,
            capsule_digest: &self.capsule_digest,
            artifact_digest: &self.artifact_digest,
            evidence_entity_ids: &self.evidence_entity_ids,
            outcome: self.outcome,
            verification: &self.verification,
            generalization_verification: &self.generalization_verification,
            provenance: &self.provenance,
            failure: &self.failure,
            budget_after: &self.budget_after,
            used_capabilities: &self.used_capabilities,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| TraceError::Encoding(error.to_string()))?;
        Ok(ContentDigest::sha256(&bytes))
    }
}

#[derive(Debug)]
pub enum TraceError {
    Capsule(CapsuleError),
    SchemaVersion { found: u32, expected: u32 },
    EmptyRunId,
    EmptyFailure,
    EmptyAdmissionField,
    EvaluatorMismatch(String),
    UnsupportedEvaluatorValue,
    NonCanonicalCollections,
    CapabilityOutsidePolicy(String),
    ReportShape,
    MissingVerification,
    MissingGeneralizationVerification,
    MissingProvenance,
    ProvenanceMethodMismatch,
    ProvenanceExampleMismatch,
    ProvenanceHoldoutMismatch,
    ProvenanceOnRefutedTrace,
    OutcomeInvariant,
    NotAdmissionEligible,
    AdmissionTargetOutsideLineage,
    AdmissionInvariant,
    TraceCapsuleMismatch,
    DigestMismatch,
    AdmissionDigestMismatch,
    Encoding(String),
}

impl fmt::Display for TraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capsule(error) => write!(formatter, "invalid capsule: {error}"),
            Self::SchemaVersion { found, expected } => {
                write!(
                    formatter,
                    "schema version {found} does not match {expected}"
                )
            }
            Self::EmptyRunId => formatter.write_str("trace run ID is empty"),
            Self::EmptyFailure => formatter.write_str("execution failure message is empty"),
            Self::EmptyAdmissionField => formatter.write_str("capability admission field is empty"),
            Self::EvaluatorMismatch(reason) => write!(formatter, "evaluator mismatch: {reason}"),
            Self::UnsupportedEvaluatorValue => {
                formatter.write_str("evaluator value is unsupported by the typed sandbox")
            }
            Self::NonCanonicalCollections => {
                formatter.write_str("trace collections are not sorted and unique")
            }
            Self::CapabilityOutsidePolicy(capability) => {
                write!(
                    formatter,
                    "capability {capability:?} was outside the capsule policy"
                )
            }
            Self::ReportShape => formatter.write_str("verification report shape is inconsistent"),
            Self::MissingVerification => formatter.write_str("verification report is missing"),
            Self::MissingGeneralizationVerification => {
                formatter.write_str("passing trace lacks exact-artifact holdout execution")
            }
            Self::MissingProvenance => formatter.write_str("passing report lacks provenance"),
            Self::ProvenanceMethodMismatch => {
                formatter.write_str("provenance method does not match the artifact")
            }
            Self::ProvenanceExampleMismatch => {
                formatter.write_str("provenance and sandbox example counts differ")
            }
            Self::ProvenanceHoldoutMismatch => {
                formatter.write_str("provenance and sandbox holdout counts differ")
            }
            Self::ProvenanceOnRefutedTrace => {
                formatter.write_str("refuted trace cannot carry verified provenance")
            }
            Self::OutcomeInvariant => formatter.write_str("trace outcome fields are inconsistent"),
            Self::NotAdmissionEligible => {
                formatter.write_str("trace does not satisfy canonical admission evidence")
            }
            Self::AdmissionTargetOutsideLineage => {
                formatter.write_str("capability target is outside the trace evidence lineage")
            }
            Self::AdmissionInvariant => {
                formatter.write_str("capability admission does not match its trace")
            }
            Self::TraceCapsuleMismatch => {
                formatter.write_str("execution trace does not match its capsule")
            }
            Self::DigestMismatch => formatter.write_str("trace digest mismatch"),
            Self::AdmissionDigestMismatch => formatter.write_str("admission digest mismatch"),
            Self::Encoding(error) => write!(formatter, "trace encoding failed: {error}"),
        }
    }
}

impl std::error::Error for TraceError {}

fn validate_report_shape(
    capsule: &ExecutionCapsule,
    report: &VerificationReport,
) -> Result<(), TraceError> {
    let expected_total = capsule.examples.len();
    if report.total != expected_total || validate_trace_report_shape(report).is_err() {
        return Err(TraceError::ReportShape);
    }
    Ok(())
}

fn validate_trace_report_shape(report: &VerificationReport) -> Result<(), TraceError> {
    if report.results.len() != report.total
        || report.metrics.examples_executed != report.total
        || report.passed > report.total
        || report.passed != report.results.iter().filter(|result| result.passed).count()
        || report.success != (report.passed == report.total)
        || report
            .results
            .iter()
            .enumerate()
            .any(|(index, result)| result.index != index)
        || report.results.iter().any(|result| {
            result.failure_kind.is_some()
                && (result.passed || result.actual.is_some() || result.error.is_none())
        })
    {
        return Err(TraceError::ReportShape);
    }
    Ok(())
}

fn failure_matches_outcome(failure: ExecutionFailureKind, outcome: ExecutionTraceOutcome) -> bool {
    match outcome {
        ExecutionTraceOutcome::BudgetExhausted => failure == ExecutionFailureKind::BudgetExhausted,
        ExecutionTraceOutcome::PolicyDenied => failure == ExecutionFailureKind::PolicyDenied,
        ExecutionTraceOutcome::ExecutionFailed => !matches!(
            failure,
            ExecutionFailureKind::BudgetExhausted | ExecutionFailureKind::PolicyDenied
        ),
        ExecutionTraceOutcome::Verified | ExecutionTraceOutcome::Refuted => false,
    }
}

fn validate_used_capabilities(
    capsule: &ExecutionCapsule,
    capabilities: &[String],
) -> Result<(), TraceError> {
    for capability in capabilities {
        if !capsule.policy.allows(capability) {
            return Err(TraceError::CapabilityOutsidePolicy(capability.clone()));
        }
    }
    Ok(())
}

fn budget_within_limits(budget: &AgentRunBudget) -> bool {
    budget.attempts_used <= budget.max_attempts
        && budget.wall_ms_used <= budget.max_wall_ms
        && budget.synthesis_candidates_used <= budget.max_synthesis_candidates
}

fn canonicalize_capabilities(capabilities: &mut Vec<String>) {
    capabilities.sort();
    capabilities.dedup();
}

fn strictly_sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

#[cfg(test)]
mod tests {
    use super::super::{CapabilityAdmission, CodeTaskSpec, ExecutableArtifact, ExecutionPolicy};
    use super::*;
    use crate::agent::capability_registry::CapabilityStatus;
    use crate::agent::coding_intent::CodingIntent;
    use crate::execution::sandbox::ExampleResult;
    use crate::execution::{Example, ExecutionMetrics, InputValue, Language, VerificationReport};

    fn evaluator() -> crate::benchmark::Problem {
        crate::benchmark::Problem {
            name: "add".into(),
            category: "arithmetic",
            description: "add two numbers",
            signature: "fn add(a: i64, b: i64) -> i64",
            examples: vec![crate::benchmark::Example {
                inputs: vec![
                    crate::benchmark::Value::Int(2),
                    crate::benchmark::Value::Int(3),
                ],
                expected: crate::benchmark::Value::Int(5),
            }],
            ..Default::default()
        }
    }

    fn capsule() -> ExecutionCapsule {
        let intent = CodingIntent::from_nl("add two numbers").expect("intent");
        let task = CodeTaskSpec::from_nl(
            "/tmp/repo",
            "add two numbers",
            intent,
            "cargo test add",
            vec!["src/lib.rs".into()],
            2,
        );
        ExecutionCapsule::new(
            task,
            ExecutableArtifact::new(
                "add",
                "fn add(a: i64, b: i64) -> i64 { a + b }",
                Language::Rust,
                "inductive-solver",
                vec![11],
            ),
            &evaluator(),
            vec![Example {
                inputs: vec![InputValue::Int(2), InputValue::Int(3)],
                expected: InputValue::Int(5),
            }],
            ExecutionPolicy::new(vec!["fs.read".into()], 1_000, 1024 * 1024, 4096),
        )
        .expect("capsule")
    }

    fn report(success: bool) -> VerificationReport {
        report_n(success, 1)
    }

    fn report_n(success: bool, total: usize) -> VerificationReport {
        VerificationReport {
            success,
            total,
            passed: if success { total } else { 0 },
            results: (0..total)
                .map(|index| ExampleResult {
                    index,
                    passed: success,
                    actual: Some(InputValue::Int(if success { 5 } else { 6 })),
                    expected: InputValue::Int(5),
                    error: (!success).then(|| "mismatch".into()),
                    failure_kind: None,
                    duration_ms: 1,
                })
                .collect(),
            metrics: ExecutionMetrics {
                total_duration_ms: 1,
                examples_executed: total,
                ..Default::default()
            },
        }
    }

    fn certificate(consensus: ConsensusVerdict) -> ProvenanceCertificate {
        ProvenanceCertificate {
            method: "inductive-solver".into(),
            holdout_source: HoldoutSource::Generated,
            n_examples: 1,
            n_holdouts: 8,
            consensus,
            artifact_verification: crate::agent::provenance::ArtifactVerificationPath::InternalMog,
        }
    }

    #[test]
    fn verified_trace_roundtrips_and_is_admission_eligible() {
        let trace = ExecutionTrace::from_report(
            &capsule(),
            report(true),
            Some(report_n(true, 8)),
            Some(certificate(ConsensusVerdict::Verified {
                agreeing: 1,
                probes: 8,
            })),
            AgentRunBudget::default(),
            vec!["fs.read".into()],
        )
        .expect("trace");
        assert!(trace.admission_eligible());
        let json = serde_json::to_string(&trace).expect("serialize");
        let restored: ExecutionTrace = serde_json::from_str(&json).expect("deserialize");
        restored.validate_integrity().expect("trace integrity");
        assert_eq!(restored.trace_digest, trace.trace_digest);
    }

    #[test]
    fn passing_examples_without_affirmative_consensus_cannot_be_admitted() {
        let trace = ExecutionTrace::from_report(
            &capsule(),
            report(true),
            Some(report_n(true, 8)),
            Some(certificate(ConsensusVerdict::NoConsensus)),
            AgentRunBudget::default(),
            vec![],
        )
        .expect("honest verified-example trace");
        assert_eq!(trace.outcome, ExecutionTraceOutcome::Verified);
        assert!(!trace.admission_eligible());
        assert!(matches!(
            CapabilityAdmission::from_verified_trace(&trace, 99, "add", "add_conformance"),
            Err(TraceError::NotAdmissionEligible)
        ));
    }

    #[test]
    fn refuted_and_out_of_policy_traces_fail_closed() {
        let refuted = ExecutionTrace::from_report(
            &capsule(),
            report(false),
            None,
            None,
            AgentRunBudget::default(),
            vec![],
        )
        .expect("refuted trace");
        assert_eq!(refuted.outcome, ExecutionTraceOutcome::Refuted);
        assert!(!refuted.admission_eligible());

        assert!(matches!(
            ExecutionTrace::from_failure(
                &capsule(),
                ExecutionFailure::new(ExecutionFailureKind::PolicyDenied, "not allowed"),
                AgentRunBudget::default(),
                vec!["shell.run".into()]
            ),
            Err(TraceError::CapabilityOutsidePolicy(_))
        ));
    }

    #[test]
    fn trace_and_admission_tampering_is_detected() {
        let trace = ExecutionTrace::from_report(
            &capsule(),
            report(true),
            Some(report_n(true, 8)),
            Some(certificate(ConsensusVerdict::Verified {
                agreeing: 2,
                probes: 8,
            })),
            AgentRunBudget::default(),
            vec![],
        )
        .expect("trace");
        assert!(matches!(
            CapabilityAdmission::from_verified_trace(&trace, 99, "add", "add_conformance"),
            Err(TraceError::AdmissionTargetOutsideLineage)
        ));
        let mut admission =
            CapabilityAdmission::from_verified_trace(&trace, 11, "add", "add_conformance")
                .expect("admission");
        admission.validate_against(&trace).expect("bound admission");
        assert_eq!(admission.record.status, CapabilityStatus::Verified);

        admission.canonical_entity_id = 100;
        assert!(matches!(
            admission.validate_against(&trace),
            Err(TraceError::AdmissionTargetOutsideLineage)
        ));

        let mut tampered_trace = trace;
        tampered_trace.budget_after.attempts_used = 1;
        assert!(matches!(
            tampered_trace.validate_integrity(),
            Err(TraceError::DigestMismatch)
        ));
    }

    #[test]
    fn recomputed_digests_cannot_hide_semantic_tampering() {
        let mut failure_trace = ExecutionTrace::from_failure(
            &capsule(),
            ExecutionFailure::new(ExecutionFailureKind::PolicyDenied, "not allowed"),
            AgentRunBudget::default(),
            vec![],
        )
        .expect("failure trace");
        failure_trace.outcome = ExecutionTraceOutcome::BudgetExhausted;
        failure_trace.trace_digest = failure_trace.recompute_digest().expect("digest");
        assert!(matches!(
            failure_trace.validate_integrity(),
            Err(TraceError::OutcomeInvariant)
        ));

        let trace = ExecutionTrace::from_report(
            &capsule(),
            report(true),
            Some(report_n(true, 8)),
            Some(certificate(ConsensusVerdict::Verified {
                agreeing: 2,
                probes: 8,
            })),
            AgentRunBudget::default(),
            vec![],
        )
        .expect("trace");
        let mut admission =
            CapabilityAdmission::from_verified_trace(&trace, 11, "add", "add_conformance")
                .expect("admission");
        admission.record.evidence = "execution-trace:sha256:forged".into();
        admission.admission_digest = admission.recompute_digest_for_test().expect("digest");
        assert!(matches!(
            admission.validate_against(&trace),
            Err(TraceError::AdmissionInvariant)
        ));
    }
}
