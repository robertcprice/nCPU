//! Induce candidate multi-step tool plans from canonical capability contracts.
//!
//! This module owns no knowledge. Every capability it can compose was admitted
//! into a caller-owned canonical LinguaGenesis USG with a declared typed
//! contract, and every decision about whether one capability's output satisfies
//! another's input is answered by the canonical registry's own relations. The
//! output is a set of *candidate*
//! [`ToolExecutionPlan`](crate::agent::runtime::ToolExecutionPlan)s: the
//! deny-by-default runtime, the plan invariants, and the verifiers still decide
//! what actually runs and what is believed.

mod induction;

pub use induction::{
    InducedStepGraphs, InductionError, InductionSeed, StepGraphInductionRequest,
    ToolStepGraphInducer, UncomposableCapability, UncomposableReason,
};

#[cfg(test)]
mod tests;
