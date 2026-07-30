use super::observation::{
    dependency_digests, resolve_arguments, ObservedToolFailure, ObservedToolFailureKind,
    ObservedToolOutput, ToolChainError, ToolChainOutcome, ToolExecutionChain, ToolStepObservation,
    ToolStepResult,
};
use super::plan::ToolExecutionPlan;
use crate::agent::runtime::AgentRunBudget;
use crate::agent::tools::{SecureToolRuntime, ToolCall};
use std::collections::BTreeMap;
use std::time::Instant;

/// Execute a pre-grounded tool plan through the existing deny-by-default runtime.
pub struct ToolChainExecutor;

impl ToolChainExecutor {
    pub fn execute(
        plan: &ToolExecutionPlan,
        runtime: &SecureToolRuntime,
        mut budget: AgentRunBudget,
    ) -> Result<ToolExecutionChain, ToolChainError> {
        plan.validate_integrity().map_err(ToolChainError::Plan)?;
        let started = Instant::now();
        let initial_wall_ms = budget.wall_ms_used;
        let mut observations = Vec::with_capacity(plan.steps.len());

        for (step_index, step) in plan.steps.iter().enumerate() {
            update_wall(&mut budget, initial_wall_ms, started);
            if budget.wall_ms_used >= budget.max_wall_ms
                || budget.attempts_used >= budget.max_attempts
            {
                return ToolExecutionChain::new(
                    plan,
                    observations,
                    ToolChainOutcome::BudgetExhausted {
                        next_step_id: step.step_id.clone(),
                    },
                    budget,
                );
            }
            budget
                .record_attempt()
                .expect("preflight guarantees an available attempt");

            let prior_by_id = observations
                .iter()
                .map(|observation: &ToolStepObservation| (observation.step_id.clone(), observation))
                .collect::<BTreeMap<_, _>>();
            let resolved_arguments = resolve_arguments(step, &prior_by_id)?;
            let prior_digests = dependency_digests(step, &prior_by_id)?;
            let mut call = ToolCall::new(&step.action);
            for (name, value) in &resolved_arguments {
                call = call.arg(name, value);
            }

            let step_started = Instant::now();
            let invoked = runtime.invoke(&step.tool, &call);
            let step_duration_ms = step_started.elapsed().as_millis() as u64;
            update_wall(&mut budget, initial_wall_ms, started);

            let result = match invoked {
                Ok(output) if step_duration_ms > plan.policy.max_step_wall_ms => {
                    let _ = output;
                    ToolStepResult::Failed(ObservedToolFailure {
                        kind: ObservedToolFailureKind::TimeoutExceeded,
                        message: format!(
                            "tool step exceeded post-execution timeout evidence ({}ms / {}ms)",
                            step_duration_ms, plan.policy.max_step_wall_ms
                        ),
                    })
                }
                Ok(output) => {
                    let output = ObservedToolOutput::from(output);
                    if output.encoded_size() > plan.policy.max_output_bytes {
                        ToolStepResult::Failed(ObservedToolFailure {
                            kind: ObservedToolFailureKind::OutputLimit,
                            message: format!(
                                "tool output exceeded policy limit ({} / {} bytes)",
                                output.encoded_size(),
                                plan.policy.max_output_bytes
                            ),
                        })
                    } else if budget.wall_ms_used > budget.max_wall_ms {
                        ToolStepResult::Failed(ObservedToolFailure {
                            kind: ObservedToolFailureKind::TimeoutExceeded,
                            message: format!(
                                "tool chain wall budget exhausted ({}ms / {}ms)",
                                budget.wall_ms_used, budget.max_wall_ms
                            ),
                        })
                    } else {
                        ToolStepResult::Succeeded(output)
                    }
                }
                Err(error) => ToolStepResult::Failed(error.into()),
            };
            let failed = matches!(result, ToolStepResult::Failed(_));
            observations.push(ToolStepObservation::new(
                plan,
                step_index,
                resolved_arguments,
                prior_digests,
                result,
                step_duration_ms,
            )?);
            if failed {
                return ToolExecutionChain::new(
                    plan,
                    observations,
                    ToolChainOutcome::Failed {
                        step_id: step.step_id.clone(),
                    },
                    budget,
                );
            }
        }

        ToolExecutionChain::new(plan, observations, ToolChainOutcome::Succeeded, budget)
    }
}

fn update_wall(budget: &mut AgentRunBudget, initial_wall_ms: u64, started: Instant) {
    budget.wall_ms_used = initial_wall_ms.saturating_add(started.elapsed().as_millis() as u64);
}
