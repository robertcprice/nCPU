// Multi-Agent System Module
// Contains orchestrator for collaborative synthesis, adversarial debate, plan executor, goal hierarchy, and task decomposition

pub mod debate;
pub mod dependencies;
pub mod executor;
pub mod hierarchy;
pub mod orchestrator;
pub mod planning;
pub mod repo;
pub mod synthesis;
pub mod tools;

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
    AgentStep, AgentTrace, BenchmarkValidation, CreditAssignment, CreditCategory, CreditLedger,
    FailureAnalysis, FailureKind, FailureParser, GuardrailDecision, GuardrailPolicy, HardnessMiner,
    HardnessProfile, HardnessTier, LocalBenchmarkTask, PatchGate, PatchGateResult, RepoAgent,
    RepoBenchmark, RepoSignal, RepoTaskKind, RepoTaskSpec,
};
pub use synthesis::solve_compositional;
pub use tools::{
    DbTool, FsTool, GitTool, HttpTool, ShellTool, Tool, ToolCall, ToolError, ToolOutput,
    ToolRegistry,
};
