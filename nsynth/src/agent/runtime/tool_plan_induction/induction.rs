use crate::agent::runtime::{
    AgentRunId, ToolArgumentBinding, ToolExecutionPlan, ToolExecutionPolicy, ToolPlanError,
    ToolStepSpec,
};
use linguigenesis_core::capability_learning::{
    data_kind_admits, read_verified_capability_contracts, AdmittedCapabilityContract, SemanticGraph,
};
use linguigenesis_core::entity::EntityId;
use linguigenesis_core::registry::Registry;
use std::collections::BTreeMap;
use std::fmt;

/// One caller-supplied literal that is already available before any step runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InductionSeed {
    pub value: String,
    pub data_kind_entity_id: EntityId,
}

impl InductionSeed {
    pub fn new(value: impl Into<String>, data_kind_entity_id: EntityId) -> Self {
        Self {
            value: value.into(),
            data_kind_entity_id,
        }
    }
}

/// What to induce, and the exact limits the induction must respect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepGraphInductionRequest {
    pub run_id: AgentRunId,
    pub evidence_entity_ids: Vec<EntityId>,
    pub policy: ToolExecutionPolicy,
    /// `tool.action` capability every induced plan must end with.
    pub goal_capability: String,
    pub seeds: Vec<InductionSeed>,
    pub max_steps: usize,
    pub max_plans: usize,
    /// Hard ceiling on partial chains explored, so induction always terminates.
    pub max_explored_chains: usize,
}

/// A capability that exists canonically but cannot enter an induced plan, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UncomposableCapability {
    pub capability_name: String,
    pub reason: UncomposableReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncomposableReason {
    /// Admitted without a typed contract, so its data flow is unknown.
    NoDeclaredContract,
    /// The capability anchor is not a single `tool.action` pair.
    NotToolAddressable,
    /// Outside the requested execution policy.
    OutsidePolicy,
}

/// Induced candidates plus an honest account of what was skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InducedStepGraphs {
    pub plans: Vec<ToolExecutionPlan>,
    pub composable_capabilities: Vec<String>,
    pub uncomposable_capabilities: Vec<UncomposableCapability>,
    pub explored_chains: usize,
    /// True when a limit stopped the search before it was exhausted.
    pub search_truncated: bool,
}

/// Structural composer over canonical capability contracts.
pub struct ToolStepGraphInducer;

/// One capability that survived the composability filter.
struct ComposableCapability {
    capability: String,
    tool: String,
    action: String,
    consumed: Vec<(String, EntityId)>,
    produced: Vec<(ProducedSource, EntityId)>,
}

#[derive(Clone)]
enum ProducedSource {
    Content,
    Metadata(String),
}

#[derive(Clone)]
struct ChosenStep {
    capability_index: usize,
    step_id: String,
    arguments: BTreeMap<String, ToolArgumentBinding>,
}

impl ToolStepGraphInducer {
    pub fn induce(
        graph: &SemanticGraph,
        registry: &Registry,
        request: &StepGraphInductionRequest,
    ) -> Result<InducedStepGraphs, InductionError> {
        request.validate()?;
        let contracts = read_verified_capability_contracts(graph);
        let (composable, uncomposable) = partition_capabilities(&contracts, &request.policy);
        if !composable
            .iter()
            .any(|capability| capability.capability == request.goal_capability)
        {
            return Err(InductionError::GoalNotComposable(
                request.goal_capability.clone(),
            ));
        }

        let mut plans: Vec<ToolExecutionPlan> = Vec::new();
        let mut frontier: Vec<Vec<ChosenStep>> = vec![Vec::new()];
        let mut explored = 0usize;
        let mut truncated = false;

        for _depth in 0..request.max_steps {
            let mut next_frontier = Vec::new();
            for chain in &frontier {
                for (capability_index, capability) in composable.iter().enumerate() {
                    for arguments in
                        binding_combinations(capability, chain, &composable, registry, request)
                    {
                        let consumes_prior = arguments
                            .values()
                            .any(|binding| binding.dependency().is_some());
                        if !chain.is_empty() && !consumes_prior {
                            // An appended step that reads nothing from the chain
                            // is not a causal continuation of it.
                            continue;
                        }
                        explored += 1;
                        if explored > request.max_explored_chains {
                            truncated = true;
                            break;
                        }
                        let mut extended = chain.clone();
                        extended.push(ChosenStep {
                            capability_index,
                            step_id: step_id(chain.len(), capability),
                            arguments,
                        });
                        if capability.capability == request.goal_capability {
                            match build_plan(&extended, &composable, request) {
                                Ok(plan) => {
                                    if !plans
                                        .iter()
                                        .any(|known| known.plan_digest == plan.plan_digest)
                                    {
                                        plans.push(plan);
                                    }
                                }
                                Err(error) => return Err(InductionError::Plan(error)),
                            }
                            if plans.len() >= request.max_plans {
                                truncated = true;
                                break;
                            }
                        } else {
                            next_frontier.push(extended);
                        }
                    }
                    if truncated {
                        break;
                    }
                }
                if truncated {
                    break;
                }
            }
            if truncated || next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        plans.sort_by(|left, right| {
            (left.steps.len(), left.plan_digest.to_string())
                .cmp(&(right.steps.len(), right.plan_digest.to_string()))
        });
        Ok(InducedStepGraphs {
            plans,
            composable_capabilities: composable
                .iter()
                .map(|capability| capability.capability.clone())
                .collect(),
            uncomposable_capabilities: uncomposable,
            explored_chains: explored,
            search_truncated: truncated,
        })
    }
}

impl StepGraphInductionRequest {
    fn validate(&self) -> Result<(), InductionError> {
        if self.max_steps == 0 {
            return Err(InductionError::InvalidLimit("max_steps"));
        }
        if self.max_plans == 0 {
            return Err(InductionError::InvalidLimit("max_plans"));
        }
        if self.max_explored_chains == 0 {
            return Err(InductionError::InvalidLimit("max_explored_chains"));
        }
        if self.goal_capability.trim().is_empty() {
            return Err(InductionError::EmptyField("goal_capability"));
        }
        if self.evidence_entity_ids.is_empty() {
            return Err(InductionError::EmptyField("evidence_entity_ids"));
        }
        Ok(())
    }
}

fn partition_capabilities(
    contracts: &[AdmittedCapabilityContract],
    policy: &ToolExecutionPolicy,
) -> (Vec<ComposableCapability>, Vec<UncomposableCapability>) {
    let mut composable = Vec::new();
    let mut uncomposable = Vec::new();
    for contract in contracts {
        if contract.slots.is_empty() {
            uncomposable.push(UncomposableCapability {
                capability_name: contract.capability_name.clone(),
                reason: UncomposableReason::NoDeclaredContract,
            });
            continue;
        }
        let Some((tool, action)) = split_capability(&contract.capability_name) else {
            uncomposable.push(UncomposableCapability {
                capability_name: contract.capability_name.clone(),
                reason: UncomposableReason::NotToolAddressable,
            });
            continue;
        };
        if !policy.allows(&contract.capability_name) {
            uncomposable.push(UncomposableCapability {
                capability_name: contract.capability_name.clone(),
                reason: UncomposableReason::OutsidePolicy,
            });
            continue;
        }
        let mut produced = Vec::new();
        if let Some(content) = contract.produced_content() {
            produced.push((ProducedSource::Content, content.data_kind_entity_id));
        }
        for metadata in contract.produced_metadata() {
            produced.push((
                ProducedSource::Metadata(metadata.name.clone()),
                metadata.data_kind_entity_id,
            ));
        }
        composable.push(ComposableCapability {
            capability: contract.capability_name.clone(),
            tool,
            action,
            consumed: contract
                .consumed()
                .map(|slot| (slot.name.clone(), slot.data_kind_entity_id))
                .collect(),
            produced,
        });
    }
    (composable, uncomposable)
}

/// Split a capability anchor into exactly one `tool.action` pair.
fn split_capability(capability: &str) -> Option<(String, String)> {
    let mut parts = capability.split('.');
    let tool = parts.next()?.trim();
    let action = parts.next()?.trim();
    if parts.next().is_some() || tool.is_empty() || action.is_empty() {
        return None;
    }
    Some((tool.to_string(), action.to_string()))
}

fn step_id(position: usize, capability: &ComposableCapability) -> String {
    format!(
        "step-{}-{}-{}",
        position + 1,
        capability.tool,
        capability.action
    )
}

/// Every way this capability's consumed slots can be satisfied from the seeds
/// and the observations the chain will already have produced.
fn binding_combinations(
    capability: &ComposableCapability,
    chain: &[ChosenStep],
    composable: &[ComposableCapability],
    registry: &Registry,
    request: &StepGraphInductionRequest,
) -> Vec<BTreeMap<String, ToolArgumentBinding>> {
    let mut combinations = vec![BTreeMap::new()];
    for (name, consumed_kind) in &capability.consumed {
        let mut candidates = Vec::new();
        for seed in &request.seeds {
            if data_kind_admits(registry, seed.data_kind_entity_id, *consumed_kind) {
                candidates.push(ToolArgumentBinding::literal(seed.value.clone()));
            }
        }
        for prior in chain {
            let producer = &composable[prior.capability_index];
            for (source, produced_kind) in &producer.produced {
                if !data_kind_admits(registry, *produced_kind, *consumed_kind) {
                    continue;
                }
                candidates.push(match source {
                    ProducedSource::Content => {
                        ToolArgumentBinding::prior_content(prior.step_id.clone())
                    }
                    ProducedSource::Metadata(key) => {
                        ToolArgumentBinding::prior_metadata(prior.step_id.clone(), key.clone())
                    }
                });
            }
        }
        if candidates.is_empty() {
            return Vec::new();
        }
        combinations = combinations
            .into_iter()
            .flat_map(|partial| {
                candidates.iter().map(move |candidate| {
                    let mut extended = partial.clone();
                    extended.insert(name.clone(), candidate.clone());
                    extended
                })
            })
            .collect();
    }
    combinations
}

fn build_plan(
    chain: &[ChosenStep],
    composable: &[ComposableCapability],
    request: &StepGraphInductionRequest,
) -> Result<ToolExecutionPlan, ToolPlanError> {
    let steps = chain
        .iter()
        .map(|chosen| {
            let capability = &composable[chosen.capability_index];
            let mut step = ToolStepSpec::new(&chosen.step_id, &capability.tool, &capability.action);
            step.arguments = chosen.arguments.clone();
            step
        })
        .collect::<Vec<_>>();
    ToolExecutionPlan::new(
        request.run_id.clone(),
        request.evidence_entity_ids.clone(),
        request.policy.clone(),
        steps,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InductionError {
    EmptyField(&'static str),
    InvalidLimit(&'static str),
    GoalNotComposable(String),
    Plan(ToolPlanError),
}

impl fmt::Display for InductionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "{field} is empty"),
            Self::InvalidLimit(field) => write!(formatter, "{field} must be greater than zero"),
            Self::GoalNotComposable(capability) => write!(
                formatter,
                "goal capability {capability:?} has no composable canonical contract"
            ),
            Self::Plan(error) => write!(formatter, "induced plan rejected: {error}"),
        }
    }
}

impl std::error::Error for InductionError {}
