//! Canonical agent runtime contracts (Package B / Phase 1).

mod budget;
mod capability_admission;
mod capsule_executor;
mod code_task_spec;
mod content_digest;
mod execution_capsule;
mod execution_evidence_persistence;
mod execution_trace;
mod state_machine;
mod tool_execution_chain;

pub use budget::{AgentRunBudget, BudgetExhausted};
pub use capability_admission::{CapabilityAdmission, CapabilityGraphAdmissionError};
pub use capsule_executor::{CapsuleExecutor, SANDBOX_EXECUTION_CAPABILITY};
pub use code_task_spec::CodeTaskSpec;
pub use content_digest::ContentDigest;
pub use execution_capsule::{CapsuleError, ExecutableArtifact, ExecutionCapsule, ExecutionPolicy};
pub use execution_evidence_persistence::{
    execution_evidence_dir, execution_evidence_path, load_execution_evidence,
    save_execution_evidence, EvidencePersistenceError, ExecutionEvidenceBundle,
};
pub use execution_trace::{
    ExecutionFailure, ExecutionFailureKind, ExecutionTrace, ExecutionTraceOutcome, TraceError,
};
pub use state_machine::{
    legal_transition, transition, AgentRunEvent, AgentRunId, AgentRunStatus, SCHEMA_VERSION,
};
pub use tool_execution_chain::{
    ObservedToolFailure, ObservedToolFailureKind, ObservedToolOutput, ToolArgumentBinding,
    ToolChainError, ToolChainExecutor, ToolChainOutcome, ToolExecutionChain, ToolExecutionPlan,
    ToolExecutionPolicy, ToolPlanError, ToolStepObservation, ToolStepResult, ToolStepSpec,
};
