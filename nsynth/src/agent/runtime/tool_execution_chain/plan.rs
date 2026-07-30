use crate::agent::runtime::{AgentRunId, ContentDigest, SCHEMA_VERSION};
use crate::agent::tools::SecureToolRuntime;
use linguigenesis_core::entity::EntityId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Source of one concrete tool argument.
///
/// Bindings are structural data. No source text, prompt phrase, or intent label
/// is inspected to decide how information flows between steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolArgumentBinding {
    Literal(String),
    PriorContent { step_id: String },
    PriorMetadata { step_id: String, key: String },
}

impl ToolArgumentBinding {
    pub fn literal(value: impl Into<String>) -> Self {
        Self::Literal(value.into())
    }

    pub fn prior_content(step_id: impl Into<String>) -> Self {
        Self::PriorContent {
            step_id: step_id.into(),
        }
    }

    pub fn prior_metadata(step_id: impl Into<String>, key: impl Into<String>) -> Self {
        Self::PriorMetadata {
            step_id: step_id.into(),
            key: key.into(),
        }
    }

    pub(crate) fn dependency(&self) -> Option<&str> {
        match self {
            Self::Literal(_) => None,
            Self::PriorContent { step_id } | Self::PriorMetadata { step_id, .. } => Some(step_id),
        }
    }
}

/// One policy-addressable tool action and its typed argument bindings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolStepSpec {
    pub step_id: String,
    pub tool: String,
    pub action: String,
    pub arguments: BTreeMap<String, ToolArgumentBinding>,
}

impl ToolStepSpec {
    pub fn new(
        step_id: impl Into<String>,
        tool: impl Into<String>,
        action: impl Into<String>,
    ) -> Self {
        Self {
            step_id: step_id.into(),
            tool: tool.into(),
            action: action.into(),
            arguments: BTreeMap::new(),
        }
    }

    pub fn argument(mut self, name: impl Into<String>, binding: ToolArgumentBinding) -> Self {
        self.arguments.insert(name.into(), binding);
        self
    }

    pub fn capability(&self) -> String {
        format!("{}.{}", self.tool, self.action)
    }

    pub(crate) fn dependencies(&self) -> Vec<String> {
        self.arguments
            .values()
            .filter_map(ToolArgumentBinding::dependency)
            .map(str::to_string)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Capability and evidence limits that the tool-chain executor actually
/// enforces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolExecutionPolicy {
    pub allowed_capabilities: Vec<String>,
    pub max_step_wall_ms: u64,
    pub max_output_bytes: usize,
}

impl ToolExecutionPolicy {
    pub fn new(
        mut allowed_capabilities: Vec<String>,
        max_step_wall_ms: u64,
        max_output_bytes: usize,
    ) -> Self {
        allowed_capabilities.sort();
        allowed_capabilities.dedup();
        Self {
            allowed_capabilities,
            max_step_wall_ms,
            max_output_bytes,
        }
    }

    pub fn from_secure_runtime(
        runtime: &SecureToolRuntime,
        max_step_wall_ms: u64,
        max_output_bytes: usize,
    ) -> Self {
        Self::new(
            runtime.allowed_capabilities(),
            max_step_wall_ms,
            max_output_bytes,
        )
    }

    pub fn allows(&self, capability: &str) -> bool {
        self.allowed_capabilities
            .binary_search_by(|candidate| candidate.as_str().cmp(capability))
            .is_ok()
    }

    fn validate(&self) -> Result<(), ToolPlanError> {
        if self.max_step_wall_ms == 0 {
            return Err(ToolPlanError::InvalidLimit("policy.max_step_wall_ms"));
        }
        if self.max_output_bytes == 0 {
            return Err(ToolPlanError::InvalidLimit("policy.max_output_bytes"));
        }
        if !strictly_sorted_unique(&self.allowed_capabilities)
            || self
                .allowed_capabilities
                .iter()
                .any(|capability| capability.trim().is_empty())
        {
            return Err(ToolPlanError::NonCanonicalCapabilities);
        }
        Ok(())
    }
}

/// Immutable, content-addressed multi-step tool plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolExecutionPlan {
    pub schema_version: u32,
    pub run_id: AgentRunId,
    pub evidence_entity_ids: Vec<EntityId>,
    pub policy: ToolExecutionPolicy,
    pub steps: Vec<ToolStepSpec>,
    pub plan_digest: ContentDigest,
}

#[derive(Serialize)]
struct PlanPayload<'a> {
    schema_version: u32,
    run_id: &'a AgentRunId,
    evidence_entity_ids: &'a [EntityId],
    policy: &'a ToolExecutionPolicy,
    steps: &'a [ToolStepSpec],
}

impl ToolExecutionPlan {
    pub fn new(
        run_id: AgentRunId,
        mut evidence_entity_ids: Vec<EntityId>,
        policy: ToolExecutionPolicy,
        steps: Vec<ToolStepSpec>,
    ) -> Result<Self, ToolPlanError> {
        evidence_entity_ids.sort_unstable();
        evidence_entity_ids.dedup();
        let mut plan = Self {
            schema_version: SCHEMA_VERSION,
            run_id,
            evidence_entity_ids,
            policy,
            steps,
            plan_digest: ContentDigest::sha256(&[]),
        };
        plan.plan_digest = plan.recompute_digest()?;
        plan.validate_integrity()?;
        Ok(plan)
    }

    pub fn validate_integrity(&self) -> Result<(), ToolPlanError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ToolPlanError::SchemaVersion {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.run_id.0.trim().is_empty() {
            return Err(ToolPlanError::EmptyField("run_id"));
        }
        if self.evidence_entity_ids.is_empty() || !strictly_sorted_unique(&self.evidence_entity_ids)
        {
            return Err(ToolPlanError::NonCanonicalLineage);
        }
        self.policy.validate()?;
        if self.steps.is_empty() {
            return Err(ToolPlanError::EmptySteps);
        }

        let mut prior_ids = BTreeSet::new();
        for step in &self.steps {
            if step.step_id.trim().is_empty() {
                return Err(ToolPlanError::EmptyField("step.step_id"));
            }
            if step.tool.trim().is_empty() {
                return Err(ToolPlanError::EmptyField("step.tool"));
            }
            if step.action.trim().is_empty() {
                return Err(ToolPlanError::EmptyField("step.action"));
            }
            if prior_ids.contains(&step.step_id) {
                return Err(ToolPlanError::DuplicateStep(step.step_id.clone()));
            }
            if step.arguments.keys().any(|key| key.trim().is_empty()) {
                return Err(ToolPlanError::EmptyField("step.argument"));
            }
            for binding in step.arguments.values() {
                match binding {
                    ToolArgumentBinding::Literal(_) => {}
                    ToolArgumentBinding::PriorContent { step_id } => {
                        require_prior_step(step_id, &prior_ids, &step.step_id)?;
                    }
                    ToolArgumentBinding::PriorMetadata { step_id, key } => {
                        require_prior_step(step_id, &prior_ids, &step.step_id)?;
                        if key.trim().is_empty() {
                            return Err(ToolPlanError::EmptyField("prior_metadata.key"));
                        }
                    }
                }
            }
            prior_ids.insert(step.step_id.clone());
            let capability = step.capability();
            if !self.policy.allows(&capability) {
                return Err(ToolPlanError::CapabilityOutsidePolicy(capability));
            }
        }

        let expected = self.recompute_digest()?;
        if !self.plan_digest.is_well_formed() || self.plan_digest != expected {
            return Err(ToolPlanError::DigestMismatch);
        }
        Ok(())
    }

    pub fn recompute_digest(&self) -> Result<ContentDigest, ToolPlanError> {
        let payload = PlanPayload {
            schema_version: self.schema_version,
            run_id: &self.run_id,
            evidence_entity_ids: &self.evidence_entity_ids,
            policy: &self.policy,
            steps: &self.steps,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| ToolPlanError::Encoding(error.to_string()))?;
        Ok(ContentDigest::sha256(&bytes))
    }
}

fn require_prior_step(
    dependency: &str,
    prior_ids: &BTreeSet<String>,
    current: &str,
) -> Result<(), ToolPlanError> {
    if dependency.trim().is_empty() || !prior_ids.contains(dependency) {
        return Err(ToolPlanError::InvalidDependency {
            step_id: current.to_string(),
            dependency: dependency.to_string(),
        });
    }
    Ok(())
}

fn strictly_sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolPlanError {
    SchemaVersion { found: u32, expected: u32 },
    EmptyField(&'static str),
    EmptySteps,
    InvalidLimit(&'static str),
    NonCanonicalLineage,
    NonCanonicalCapabilities,
    DuplicateStep(String),
    InvalidDependency { step_id: String, dependency: String },
    CapabilityOutsidePolicy(String),
    DigestMismatch,
    Encoding(String),
}

impl fmt::Display for ToolPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaVersion { found, expected } => {
                write!(
                    formatter,
                    "schema version {found} does not match {expected}"
                )
            }
            Self::EmptyField(field) => write!(formatter, "{field} is empty"),
            Self::EmptySteps => formatter.write_str("tool execution plan has no steps"),
            Self::InvalidLimit(field) => write!(formatter, "{field} must be greater than zero"),
            Self::NonCanonicalLineage => {
                formatter.write_str("tool plan lineage must be nonempty, sorted, and unique")
            }
            Self::NonCanonicalCapabilities => {
                formatter.write_str("tool plan capabilities must be sorted, unique, and nonempty")
            }
            Self::DuplicateStep(step_id) => write!(formatter, "duplicate step ID {step_id:?}"),
            Self::InvalidDependency {
                step_id,
                dependency,
            } => write!(
                formatter,
                "step {step_id:?} references unavailable prior step {dependency:?}"
            ),
            Self::CapabilityOutsidePolicy(capability) => {
                write!(
                    formatter,
                    "capability {capability:?} is outside plan policy"
                )
            }
            Self::DigestMismatch => formatter.write_str("tool plan digest mismatch"),
            Self::Encoding(error) => write!(formatter, "tool plan encoding failed: {error}"),
        }
    }
}

impl std::error::Error for ToolPlanError {}
