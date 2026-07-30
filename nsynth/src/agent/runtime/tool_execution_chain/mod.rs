//! Proof-bound execution of data-driven multi-step tool plans.
//!
//! This module is an execution/audit substrate, not a planner and not a
//! knowledge store. A caller supplies a content-bound plan whose typed argument
//! bindings make prior observations causally available to later steps. The
//! executor applies the existing deny-by-default
//! [`SecureToolRuntime`](crate::agent::tools::SecureToolRuntime) policy and emits
//! a replayable digest chain. Canonical capability learning remains a
//! LinguaGenesis USG responsibility.

mod executor;
mod observation;
mod plan;

pub use executor::ToolChainExecutor;
pub use observation::{
    ObservedToolFailure, ObservedToolFailureKind, ObservedToolOutput, ToolChainError,
    ToolChainOutcome, ToolExecutionChain, ToolStepObservation, ToolStepResult,
};
pub use plan::{
    ToolArgumentBinding, ToolExecutionPlan, ToolExecutionPolicy, ToolPlanError, ToolStepSpec,
};

#[cfg(test)]
mod tests;
