use super::plan::{ToolArgumentBinding, ToolExecutionPlan, ToolPlanError};
use crate::agent::runtime::{AgentRunBudget, AgentRunId, ContentDigest, SCHEMA_VERSION};
use crate::agent::tools::{ToolError, ToolOutput};
use linguigenesis_core::entity::EntityId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedToolOutput {
    pub content: String,
    pub metadata: BTreeMap<String, String>,
}

impl From<ToolOutput> for ObservedToolOutput {
    fn from(output: ToolOutput) -> Self {
        Self {
            content: output.content,
            metadata: output.metadata.into_iter().collect(),
        }
    }
}

impl ObservedToolOutput {
    pub(crate) fn encoded_size(&self) -> usize {
        self.content.len()
            + self
                .metadata
                .iter()
                .map(|(key, value)| key.len().saturating_add(value.len()))
                .sum::<usize>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservedToolFailureKind {
    UnknownTool,
    UnknownAction,
    MissingArgument,
    InvalidArgument,
    PermissionDenied,
    Io,
    Execution,
    OutputLimit,
    TimeoutExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedToolFailure {
    pub kind: ObservedToolFailureKind,
    pub message: String,
}

impl From<ToolError> for ObservedToolFailure {
    fn from(error: ToolError) -> Self {
        let kind = match &error {
            ToolError::UnknownTool(_) => ObservedToolFailureKind::UnknownTool,
            ToolError::UnknownAction { .. } => ObservedToolFailureKind::UnknownAction,
            ToolError::MissingArg(_) => ObservedToolFailureKind::MissingArgument,
            ToolError::InvalidArg(_, _) => ObservedToolFailureKind::InvalidArgument,
            ToolError::PermissionDenied(_) => ObservedToolFailureKind::PermissionDenied,
            ToolError::Io(_) => ObservedToolFailureKind::Io,
            ToolError::Execution(_) => ObservedToolFailureKind::Execution,
        };
        Self {
            kind,
            message: error.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolStepResult {
    Succeeded(ObservedToolOutput),
    Failed(ObservedToolFailure),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolStepObservation {
    pub schema_version: u32,
    pub plan_digest: ContentDigest,
    pub step_index: usize,
    pub step_id: String,
    pub capability: String,
    pub resolved_arguments: BTreeMap<String, String>,
    pub prior_observation_digests: Vec<ContentDigest>,
    pub result: ToolStepResult,
    pub duration_ms: u64,
    pub observation_digest: ContentDigest,
}

#[derive(Serialize)]
struct ObservationPayload<'a> {
    schema_version: u32,
    plan_digest: &'a ContentDigest,
    step_index: usize,
    step_id: &'a str,
    capability: &'a str,
    resolved_arguments: &'a BTreeMap<String, String>,
    prior_observation_digests: &'a [ContentDigest],
    result: &'a ToolStepResult,
    duration_ms: u64,
}

impl ToolStepObservation {
    pub(crate) fn new(
        plan: &ToolExecutionPlan,
        step_index: usize,
        resolved_arguments: BTreeMap<String, String>,
        prior_observation_digests: Vec<ContentDigest>,
        result: ToolStepResult,
        duration_ms: u64,
    ) -> Result<Self, ToolChainError> {
        let step = plan
            .steps
            .get(step_index)
            .ok_or(ToolChainError::ObservationShape)?;
        let mut observation = Self {
            schema_version: SCHEMA_VERSION,
            plan_digest: plan.plan_digest.clone(),
            step_index,
            step_id: step.step_id.clone(),
            capability: step.capability(),
            resolved_arguments,
            prior_observation_digests,
            result,
            duration_ms,
            observation_digest: ContentDigest::sha256(&[]),
        };
        observation.observation_digest = observation.recompute_digest()?;
        Ok(observation)
    }

    pub fn validate_integrity(&self) -> Result<(), ToolChainError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ToolChainError::SchemaVersion {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.step_id.trim().is_empty()
            || self.capability.trim().is_empty()
            || self
                .resolved_arguments
                .keys()
                .any(|key| key.trim().is_empty())
            || self
                .prior_observation_digests
                .iter()
                .any(|digest| !digest.is_well_formed())
            || !strictly_sorted_unique(&self.prior_observation_digests)
        {
            return Err(ToolChainError::ObservationShape);
        }
        if matches!(
            &self.result,
            ToolStepResult::Failed(ObservedToolFailure { message, .. }) if message.trim().is_empty()
        ) {
            return Err(ToolChainError::ObservationShape);
        }
        let expected = self.recompute_digest()?;
        if !self.plan_digest.is_well_formed()
            || !self.observation_digest.is_well_formed()
            || self.observation_digest != expected
        {
            return Err(ToolChainError::DigestMismatch);
        }
        Ok(())
    }

    pub fn recompute_digest(&self) -> Result<ContentDigest, ToolChainError> {
        let payload = ObservationPayload {
            schema_version: self.schema_version,
            plan_digest: &self.plan_digest,
            step_index: self.step_index,
            step_id: &self.step_id,
            capability: &self.capability,
            resolved_arguments: &self.resolved_arguments,
            prior_observation_digests: &self.prior_observation_digests,
            result: &self.result,
            duration_ms: self.duration_ms,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| ToolChainError::Encoding(error.to_string()))?;
        Ok(ContentDigest::sha256(&bytes))
    }

    pub fn output(&self) -> Option<&ObservedToolOutput> {
        match &self.result {
            ToolStepResult::Succeeded(output) => Some(output),
            ToolStepResult::Failed(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChainOutcome {
    Succeeded,
    Failed { step_id: String },
    BudgetExhausted { next_step_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolExecutionChain {
    pub schema_version: u32,
    pub run_id: AgentRunId,
    pub plan_digest: ContentDigest,
    pub evidence_entity_ids: Vec<EntityId>,
    pub observations: Vec<ToolStepObservation>,
    pub outcome: ToolChainOutcome,
    pub budget_after: AgentRunBudget,
    pub used_capabilities: Vec<String>,
    pub chain_digest: ContentDigest,
}

#[derive(Serialize)]
struct ChainPayload<'a> {
    schema_version: u32,
    run_id: &'a AgentRunId,
    plan_digest: &'a ContentDigest,
    evidence_entity_ids: &'a [EntityId],
    observations: &'a [ToolStepObservation],
    outcome: &'a ToolChainOutcome,
    budget_after: &'a AgentRunBudget,
    used_capabilities: &'a [String],
}

impl ToolExecutionChain {
    pub(crate) fn new(
        plan: &ToolExecutionPlan,
        observations: Vec<ToolStepObservation>,
        outcome: ToolChainOutcome,
        budget_after: AgentRunBudget,
    ) -> Result<Self, ToolChainError> {
        let used_capabilities = observations
            .iter()
            .map(|observation| observation.capability.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut chain = Self {
            schema_version: SCHEMA_VERSION,
            run_id: plan.run_id.clone(),
            plan_digest: plan.plan_digest.clone(),
            evidence_entity_ids: plan.evidence_entity_ids.clone(),
            observations,
            outcome,
            budget_after,
            used_capabilities,
            chain_digest: ContentDigest::sha256(&[]),
        };
        chain.chain_digest = chain.recompute_digest()?;
        chain.validate_against_plan(plan)?;
        Ok(chain)
    }

    pub fn validate_against_plan(&self, plan: &ToolExecutionPlan) -> Result<(), ToolChainError> {
        plan.validate_integrity().map_err(ToolChainError::Plan)?;
        if self.schema_version != SCHEMA_VERSION {
            return Err(ToolChainError::SchemaVersion {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.run_id != plan.run_id
            || self.plan_digest != plan.plan_digest
            || self.evidence_entity_ids != plan.evidence_entity_ids
            || self.observations.len() > plan.steps.len()
        {
            return Err(ToolChainError::PlanMismatch);
        }

        let mut prior_by_id = BTreeMap::<String, &ToolStepObservation>::new();
        for (index, observation) in self.observations.iter().enumerate() {
            observation.validate_integrity()?;
            let step = &plan.steps[index];
            let expected_arguments = resolve_arguments(step, &prior_by_id)?;
            let expected_prior_digests = dependency_digests(step, &prior_by_id)?;
            if observation.plan_digest != plan.plan_digest
                || observation.step_index != index
                || observation.step_id != step.step_id
                || observation.capability != step.capability()
                || observation.resolved_arguments != expected_arguments
                || observation.prior_observation_digests != expected_prior_digests
            {
                return Err(ToolChainError::ObservationShape);
            }
            if matches!(observation.result, ToolStepResult::Failed(_))
                && index + 1 != self.observations.len()
            {
                return Err(ToolChainError::OutcomeInvariant);
            }
            prior_by_id.insert(step.step_id.clone(), observation);
        }

        match &self.outcome {
            ToolChainOutcome::Succeeded => {
                if self.observations.len() != plan.steps.len()
                    || self.observations.iter().any(|observation| {
                        !matches!(observation.result, ToolStepResult::Succeeded(_))
                    })
                {
                    return Err(ToolChainError::OutcomeInvariant);
                }
            }
            ToolChainOutcome::Failed { step_id } => {
                let Some(last) = self.observations.last() else {
                    return Err(ToolChainError::OutcomeInvariant);
                };
                if &last.step_id != step_id || !matches!(last.result, ToolStepResult::Failed(_)) {
                    return Err(ToolChainError::OutcomeInvariant);
                }
            }
            ToolChainOutcome::BudgetExhausted { next_step_id } => {
                if self.observations.len() >= plan.steps.len()
                    || plan.steps[self.observations.len()].step_id != *next_step_id
                    || self.observations.iter().any(|observation| {
                        !matches!(observation.result, ToolStepResult::Succeeded(_))
                    })
                {
                    return Err(ToolChainError::OutcomeInvariant);
                }
            }
        }

        let expected_capabilities = self
            .observations
            .iter()
            .map(|observation| observation.capability.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if self.used_capabilities != expected_capabilities
            || !strictly_sorted_unique(&self.used_capabilities)
        {
            return Err(ToolChainError::OutcomeInvariant);
        }
        let expected = self.recompute_digest()?;
        if !self.chain_digest.is_well_formed() || self.chain_digest != expected {
            return Err(ToolChainError::DigestMismatch);
        }
        Ok(())
    }

    pub fn recompute_digest(&self) -> Result<ContentDigest, ToolChainError> {
        let payload = ChainPayload {
            schema_version: self.schema_version,
            run_id: &self.run_id,
            plan_digest: &self.plan_digest,
            evidence_entity_ids: &self.evidence_entity_ids,
            observations: &self.observations,
            outcome: &self.outcome,
            budget_after: &self.budget_after,
            used_capabilities: &self.used_capabilities,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| ToolChainError::Encoding(error.to_string()))?;
        Ok(ContentDigest::sha256(&bytes))
    }
}

pub(crate) fn resolve_arguments(
    step: &super::plan::ToolStepSpec,
    prior_by_id: &BTreeMap<String, &ToolStepObservation>,
) -> Result<BTreeMap<String, String>, ToolChainError> {
    step.arguments
        .iter()
        .map(|(name, binding)| {
            let value = match binding {
                ToolArgumentBinding::Literal(value) => value.clone(),
                ToolArgumentBinding::PriorContent { step_id } => {
                    prior_output(prior_by_id, step_id)?.content.clone()
                }
                ToolArgumentBinding::PriorMetadata { step_id, key } => {
                    prior_output(prior_by_id, step_id)?
                        .metadata
                        .get(key)
                        .cloned()
                        .ok_or_else(|| ToolChainError::MissingPriorMetadata {
                            step_id: step_id.clone(),
                            key: key.clone(),
                        })?
                }
            };
            Ok((name.clone(), value))
        })
        .collect()
}

pub(crate) fn dependency_digests(
    step: &super::plan::ToolStepSpec,
    prior_by_id: &BTreeMap<String, &ToolStepObservation>,
) -> Result<Vec<ContentDigest>, ToolChainError> {
    step.dependencies()
        .iter()
        .map(|step_id| {
            prior_by_id
                .get(step_id)
                .map(|observation| observation.observation_digest.clone())
                .ok_or_else(|| ToolChainError::MissingPriorObservation(step_id.clone()))
        })
        .collect::<Result<BTreeSet<_>, _>>()
        .map(|digests| digests.into_iter().collect())
}

fn prior_output<'a>(
    prior_by_id: &'a BTreeMap<String, &ToolStepObservation>,
    step_id: &str,
) -> Result<&'a ObservedToolOutput, ToolChainError> {
    prior_by_id
        .get(step_id)
        .and_then(|observation| observation.output())
        .ok_or_else(|| ToolChainError::MissingPriorObservation(step_id.to_string()))
}

fn strictly_sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

#[derive(Debug)]
pub enum ToolChainError {
    Plan(ToolPlanError),
    SchemaVersion { found: u32, expected: u32 },
    PlanMismatch,
    ObservationShape,
    OutcomeInvariant,
    MissingPriorObservation(String),
    MissingPriorMetadata { step_id: String, key: String },
    DigestMismatch,
    Encoding(String),
}

impl fmt::Display for ToolChainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => write!(formatter, "invalid tool plan: {error}"),
            Self::SchemaVersion { found, expected } => {
                write!(
                    formatter,
                    "schema version {found} does not match {expected}"
                )
            }
            Self::PlanMismatch => formatter.write_str("tool chain does not match its plan"),
            Self::ObservationShape => formatter.write_str("tool observation shape is invalid"),
            Self::OutcomeInvariant => formatter.write_str("tool chain outcome invariant failed"),
            Self::MissingPriorObservation(step_id) => {
                write!(
                    formatter,
                    "missing successful prior observation {step_id:?}"
                )
            }
            Self::MissingPriorMetadata { step_id, key } => {
                write!(formatter, "step {step_id:?} has no metadata key {key:?}")
            }
            Self::DigestMismatch => formatter.write_str("tool execution digest mismatch"),
            Self::Encoding(error) => write!(formatter, "tool execution encoding failed: {error}"),
        }
    }
}

impl std::error::Error for ToolChainError {}
