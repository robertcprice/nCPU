pub mod agent;
pub mod benchmark;
pub mod credit;
pub mod failure_parser;
pub mod gencode_normalize;
pub mod guardrails;
pub mod hardness;
pub mod nl_fixture_harness;
pub mod patch_gate;
pub mod repo_agent;
pub mod repair_loop;
pub mod run_supervisor;
pub mod trace;
pub mod workflow_runner;

pub use benchmark::{
    nl_fixture_code_specs, nl_fixture_wrong_stub, nl_synthesis_fixture_suite, BenchmarkValidation,
    LocalBenchmarkTask, RepoBenchmark,
};
pub use nl_fixture_harness::{
    nl_fixture_cargo_test_command, write_nl_fixture_crate, write_synthesized_project,
    write_tensor_program, CompileStatus, WriteOutcome,
};

pub use agent::RepairAgent;
pub use credit::{CreditAssignment, CreditCategory, CreditLedger};
pub use failure_parser::{FailureAnalysis, FailureKind, FailureParser};
pub use guardrails::{GuardrailDecision, GuardrailPolicy};
pub use hardness::{
    HardnessMiner, HardnessProfile, HardnessTier, RepoSignal, RepoTaskKind, RepoTaskSpec,
};
pub use patch_gate::{PatchGate, PatchGateResult};
pub use repair_loop::{
    RepairContext, RepairEdit, RepairFile, RepairLoop, RepairLoopResult, RepairPatch,
    RepairVerification, RepairVerifier,
};
pub use repo_agent::{RepoAgent, RepoAgentRunResult};
pub use run_supervisor::{RepoRunOutcome, RepoRunSupervisor};
pub use trace::{AgentStep, AgentTrace};
pub use workflow_runner::{RepoWorkflowRunner, WorkflowRunReport};
