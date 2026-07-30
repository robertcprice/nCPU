//! Trace-bound proposal for admission into the canonical LinguaGenesis graph.
//!
//! This module deliberately owns no persistent registry. It converts a strong
//! verifier trace into an existing [`CapabilityRecord`] that the canonical
//! knowledge authority may accept or reject.

use super::{ContentDigest, ExecutionTrace, TraceError, SCHEMA_VERSION};
use crate::agent::capability_registry::{CapabilityRecord, CapabilityStatus};
use linguigenesis_core::capability_learning::{
    admit_verified_capability, CanonicalCapabilityAdmissionError,
    CanonicalCapabilityAdmissionReceipt, SemanticGraph, VerifiedCapabilityAdmissionRequest,
    CAPABILITY_ADMISSION_SCHEMA_VERSION,
};
use linguigenesis_core::entity::EntityId;
use linguigenesis_core::registry::Registry;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Proof-bound proposal for admission into the canonical LinguaGenesis graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityAdmission {
    pub schema_version: u32,
    pub canonical_entity_id: EntityId,
    pub run_id: super::AgentRunId,
    pub verification_trace_digest: ContentDigest,
    pub artifact_digest: ContentDigest,
    pub evidence_entity_ids: Vec<EntityId>,
    pub record: CapabilityRecord,
    pub admission_digest: ContentDigest,
}

#[derive(Serialize)]
struct AdmissionPayload<'a> {
    schema_version: u32,
    canonical_entity_id: EntityId,
    run_id: &'a super::AgentRunId,
    verification_trace_digest: &'a ContentDigest,
    artifact_digest: &'a ContentDigest,
    evidence_entity_ids: &'a [EntityId],
    record: &'a CapabilityRecord,
}

impl CapabilityAdmission {
    pub fn from_verified_trace(
        trace: &ExecutionTrace,
        canonical_entity_id: EntityId,
        capability_name: impl Into<String>,
        conformance_test: impl Into<String>,
    ) -> Result<Self, TraceError> {
        trace.validate_integrity()?;
        if !trace.admission_eligible() {
            return Err(TraceError::NotAdmissionEligible);
        }
        if trace
            .evidence_entity_ids
            .binary_search(&canonical_entity_id)
            .is_err()
        {
            return Err(TraceError::AdmissionTargetOutsideLineage);
        }
        let capability_name = capability_name.into();
        let conformance_test = conformance_test.into();
        if capability_name.trim().is_empty() || conformance_test.trim().is_empty() {
            return Err(TraceError::EmptyAdmissionField);
        }
        let record = CapabilityRecord {
            name: capability_name,
            status: CapabilityStatus::Verified,
            evidence: format!("execution-trace:sha256:{}", trace.trace_digest),
            conformance_test: Some(conformance_test),
        };
        let mut admission = Self {
            schema_version: SCHEMA_VERSION,
            canonical_entity_id,
            run_id: trace.run_id.clone(),
            verification_trace_digest: trace.trace_digest.clone(),
            artifact_digest: trace.artifact_digest.clone(),
            evidence_entity_ids: trace.evidence_entity_ids.clone(),
            record,
            admission_digest: ContentDigest::sha256(&[]),
        };
        admission.admission_digest = admission.recompute_digest()?;
        admission.validate_against(trace)?;
        Ok(admission)
    }

    pub fn validate_against(&self, trace: &ExecutionTrace) -> Result<(), TraceError> {
        trace.validate_integrity()?;
        if trace
            .evidence_entity_ids
            .binary_search(&self.canonical_entity_id)
            .is_err()
        {
            return Err(TraceError::AdmissionTargetOutsideLineage);
        }
        let expected_evidence = format!("execution-trace:sha256:{}", trace.trace_digest);
        if !trace.admission_eligible()
            || self.schema_version != SCHEMA_VERSION
            || self.run_id != trace.run_id
            || self.verification_trace_digest != trace.trace_digest
            || self.artifact_digest != trace.artifact_digest
            || self.evidence_entity_ids != trace.evidence_entity_ids
            || self.record.status != CapabilityStatus::Verified
            || self.record.name.trim().is_empty()
            || self.record.evidence != expected_evidence
            || self
                .record
                .conformance_test
                .as_deref()
                .map_or(true, |test| test.trim().is_empty())
        {
            return Err(TraceError::AdmissionInvariant);
        }
        let expected = self.recompute_digest()?;
        if !self.admission_digest.is_well_formed() || self.admission_digest != expected {
            return Err(TraceError::AdmissionDigestMismatch);
        }
        Ok(())
    }

    /// Admit this already trace-validated proposal through the LinguaGenesis
    /// canonical-USG boundary. The graph remains caller-owned; nCPU does not
    /// create or retain a capability registry.
    pub fn admit_into_canonical_graph(
        &self,
        trace: &ExecutionTrace,
        registry: &Registry,
        graph: &mut SemanticGraph,
    ) -> Result<CanonicalCapabilityAdmissionReceipt, CapabilityGraphAdmissionError> {
        self.validate_against(trace)
            .map_err(CapabilityGraphAdmissionError::Trace)?;
        let request = VerifiedCapabilityAdmissionRequest {
            schema_version: CAPABILITY_ADMISSION_SCHEMA_VERSION,
            canonical_entity_id: self.canonical_entity_id,
            run_id: self.run_id.0.clone(),
            verification_trace_digest: self.verification_trace_digest.to_string(),
            artifact_digest: self.artifact_digest.to_string(),
            evidence_entity_ids: self.evidence_entity_ids.clone(),
            capability_name: self.record.name.clone(),
            conformance_test: self
                .record
                .conformance_test
                .clone()
                .ok_or(TraceError::AdmissionInvariant)
                .map_err(CapabilityGraphAdmissionError::Trace)?,
            proposal_digest: self.admission_digest.to_string(),
        };
        admit_verified_capability(registry, graph, &request)
            .map_err(CapabilityGraphAdmissionError::Canonical)
    }

    #[cfg(test)]
    pub(super) fn recompute_digest_for_test(&self) -> Result<ContentDigest, TraceError> {
        self.recompute_digest()
    }

    fn recompute_digest(&self) -> Result<ContentDigest, TraceError> {
        let payload = AdmissionPayload {
            schema_version: self.schema_version,
            canonical_entity_id: self.canonical_entity_id,
            run_id: &self.run_id,
            verification_trace_digest: &self.verification_trace_digest,
            artifact_digest: &self.artifact_digest,
            evidence_entity_ids: &self.evidence_entity_ids,
            record: &self.record,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| TraceError::Encoding(error.to_string()))?;
        Ok(ContentDigest::sha256(&bytes))
    }
}

#[derive(Debug)]
pub enum CapabilityGraphAdmissionError {
    Trace(TraceError),
    Canonical(CanonicalCapabilityAdmissionError),
}

impl fmt::Display for CapabilityGraphAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trace(error) => write!(formatter, "invalid nCPU capability proposal: {error}"),
            Self::Canonical(error) => write!(formatter, "LinguaGenesis admission failed: {error}"),
        }
    }
}

impl std::error::Error for CapabilityGraphAdmissionError {}
