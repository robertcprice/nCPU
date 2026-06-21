//! Canonical agent runtime contracts (Package B / Phase 1).

mod budget;
mod code_task_spec;
mod state_machine;

pub use budget::{AgentRunBudget, BudgetExhausted};
pub use code_task_spec::CodeTaskSpec;
pub use state_machine::{
    legal_transition, transition, AgentRunEvent, AgentRunId, AgentRunStatus, SCHEMA_VERSION,
};
