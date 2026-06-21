// Multi-Agent System Module
// Contains orchestrator for collaborative synthesis, adversarial debate, plan executor, goal hierarchy, and task decomposition

pub mod agent_run;
pub mod agent_run_persistence;
pub mod capability_registry;
pub mod coding_intent;
pub mod debate;
pub mod dependencies;
pub mod edit;
pub mod executor;
pub mod hierarchy;
pub mod orchestrator;
pub mod package_b_gate;
pub mod planning;
pub mod repo;
pub mod repository;
pub mod runtime;
pub mod session;
pub mod session_persistence;
pub mod synthesis;
pub mod synthesis_proposer;
pub mod tools;

pub use agent_run::AgentRun;
pub use capability_registry::{CapabilityRecord, CapabilityRegistry, CapabilityStatus};
pub use coding_intent::{CodingExample, CodingIntent, CodingValue};
pub use debate::{
    AdversarialAgent, DebateAgent, DebateAgentId, DebateConfig, DebateError, DebateResult,
    DebateSystem, DefenderAgent, ProposerAgent,
};
pub use dependencies::{
    CycleDetectionResult, Dependency, DependencyEdge, DependencyId, DependencyResolver,
    DependencyStats, DependencyType, DependencyUpdate, TopologicalOrder, UpdateResult,
    ValidationError, ValidationErrorType, ValidationResult, ValidationWarning,
    ValidationWarningType, VisualizationFormat,
};
pub use executor::{
    ExecutionProgress, ExecutionSummary, Executor, ExecutorConfig, Milestone, Plan, RollbackAction,
    Task, TaskId, TaskResult, TaskStatus,
};
pub use hierarchy::{Goal, GoalId, GoalStatus, GoalTree, Priority};
pub use orchestrator::{
    Agent, AgentId, AgentMessage, CollaborativeResult, CommunicationChannel, Orchestrator,
    OrchestratorConfig, SolutionProposal,
};
pub use planning::{
    DecomposerStats, DecompositionStrategy, Task as PlanningTask, TaskDecomposer,
    TaskId as PlanningTaskId, TaskMetadata, TaskOptimization, TaskPriority,
    TaskResult as PlanningTaskResult, TaskStatus as PlanningTaskStatus,
};
pub use repo::{
    nl_fixture_code_specs, nl_synthesis_fixture_suite, AgentStep, AgentTrace, BenchmarkValidation,
    CreditAssignment, CreditCategory, CreditLedger, FailureAnalysis, FailureKind, FailureParser,
    GuardrailDecision, GuardrailPolicy, HardnessMiner, HardnessProfile, HardnessTier,
    LocalBenchmarkTask, PatchGate, PatchGateResult, RepairAgent, RepairContext, RepairEdit,
    RepairFile, RepairLoop, RepairLoopResult, RepairPatch, RepairVerification, RepairVerifier,
    RepoAgent, RepoAgentRunResult, RepoBenchmark, RepoRunOutcome, RepoRunSupervisor, RepoSignal,
    RepoTaskKind, RepoTaskSpec, RepoWorkflowRunner, WorkflowRunReport,
};
pub use runtime::{
    legal_transition, transition, AgentRunBudget, AgentRunEvent, AgentRunId, AgentRunStatus,
    CodeTaskSpec, SCHEMA_VERSION,
};
pub use session::{AgentQueryResult, CodingAgentSession, QueryRoute};
pub use session_persistence::{load_session_snapshot, save_session_snapshot, SessionSnapshot};
pub use synthesis::solve_compositional;
pub use synthesis_proposer::{
    nl_description_from_issue, nl_synthesis_proposer, repair_patch_from_synthesis,
    task_uses_nl_synthesis,
};
pub use tools::{
    DbTool, FsTool, GitTool, HttpTool, SecureToolRuntime, ShellTool, Tool, ToolCall, ToolError,
    ToolOutput, ToolRegistry,
};
