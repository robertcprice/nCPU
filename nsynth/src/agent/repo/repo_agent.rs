//! Closed-loop repository agent (Phase 7 / Package G5 owner).

use crate::agent::edit::IsolatedRepoSession;
use crate::agent::repo::{
    CreditCategory, GuardrailPolicy, RepairAgent, RepairVerifier, RepoRunOutcome,
    RepoRunSupervisor,
};
use crate::agent::repository::{retrieve_paths, RepoIndex};
use crate::agent::runtime::{AgentRunBudget, AgentRunId, CodeTaskSpec};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Result of `RepoAgent::run` — oracle-gated, not proposal-gated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoAgentRunResult {
    pub run_id: AgentRunId,
    pub success: bool,
    pub phases_completed: Vec<String>,
    pub baseline_passed: bool,
    pub final_passed: bool,
    pub retrieved_files: Vec<String>,
    pub indexed_files: usize,
    pub repair_iterations: usize,
    pub supervisor_outcome: Option<RepoRunOutcome>,
    pub trace_len: usize,
    pub credit_summary: Vec<(CreditCategory, f64)>,
    pub budget: AgentRunBudget,
    pub error: Option<String>,
}

/// Bounded repository repair loop owner.
pub struct RepoAgent {
    root: PathBuf,
    policy: GuardrailPolicy,
    agent: RepairAgent,
}

impl RepoAgent {
    pub fn new(root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        let root = root.into();
        Self {
            agent: RepairAgent::with_policy(&root, policy.clone()),
            root,
            policy,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn repair_agent(&self) -> &RepairAgent {
        &self.agent
    }

    /// Run index → retrieve → baseline oracle → repair → final oracle.
    pub fn run(&mut self, spec: &CodeTaskSpec) -> RepoAgentRunResult {
        let started = Instant::now();
        let mut phases_completed = Vec::new();
        let mut budget = spec.budget.clone();
        // Bound the inner synthesis to the task's own wall budget: the phase-level
        // tick_wall checks below only fire BETWEEN phases, but a single solve_problem
        // inside the synthesis phase can spin unbounded in the array-gradient. Install a
        // QuerySolveBudget from max_wall_ms so that solve degrades to a refusal within
        // budget instead of overrunning it (repo_workflow --fixtures overran ~61s on a
        // hard task). Held for the whole run; no-op when no wall budget is set.
        let _solve_budget = (budget.max_wall_ms > 0)
            .then(|| crate::synthesis::QuerySolveBudget::millis(budget.max_wall_ms));
        let mut retrieved_files = Vec::new();
        let mut indexed_files = 0usize;
        let mut baseline_passed = false;
        let mut final_passed = false;

        if let Err(exhausted) = budget.tick_wall(started) {
            return self.finish_run(
                spec,
                phases_completed,
                budget,
                retrieved_files,
                indexed_files,
                baseline_passed,
                final_passed,
                0,
                None,
                Some(exhausted.reason),
            );
        }

        let index = match RepoIndex::build(&self.root, &self.policy) {
            Ok(index) => index,
            Err(error) => {
                return self.finish_run(
                    spec,
                    phases_completed,
                    budget,
                    retrieved_files,
                    indexed_files,
                    baseline_passed,
                    final_passed,
                    0,
                    None,
                    Some(error),
                );
            }
        };
        indexed_files = index.files.len();
        phases_completed.push("index".into());
        self.agent.assign_credit(
            CreditCategory::Retrieval,
            1.0,
            format!("indexed {} files", indexed_files),
        );

        retrieved_files = retrieve_paths(&index, &spec.nl_input, 8);
        phases_completed.push("retrieve".into());
        let localization_confidence =
            crate::agent::repository::localization_confidence(&retrieved_files, indexed_files);
        self.agent.trace_mut().push(
            "retrieve",
            spec.nl_input.clone(),
            format!(
                "hits={:?} localization_confidence={:.2}",
                retrieved_files, localization_confidence
            ),
            "ok",
        );

        let task_id = spec.run_id.0.clone();
        let max_iterations = spec
            .max_iterations
            .min(spec.budget.max_attempts as usize)
            .max(1);
        let mut repo_task = spec.to_repo_task_spec(&task_id);
        repo_task.max_iterations = max_iterations;
        let mut hardness = repo_task.hardness.clone();
        hardness.apply_retrieval_localization(localization_confidence);
        repo_task.hardness = hardness;

        let verifier = RepairVerifier::new(&self.root, self.policy.clone());
        let baseline = match verifier.verify(&repo_task.test_command) {
            Ok(report) => report,
            Err(error) => {
                return self.finish_run(
                    spec,
                    phases_completed,
                    budget,
                    retrieved_files,
                    indexed_files,
                    baseline_passed,
                    final_passed,
                    0,
                    None,
                    Some(error),
                );
            }
        };
        baseline_passed = baseline.success;
        phases_completed.push("baseline_verify".into());
        self.agent.trace_mut().push(
            "baseline_verify",
            repo_task.test_command.clone(),
            format!("exit={:?}", baseline.exit_code),
            if baseline_passed { "ok" } else { "failed" }.to_string(),
        );

        if baseline_passed {
            self.agent.assign_credit(
                CreditCategory::Verification,
                1.0,
                "baseline oracle already satisfied",
            );
            if let Err(exhausted) = budget.tick_wall(started) {
                return self.finish_run(
                    spec,
                    phases_completed,
                    budget,
                    retrieved_files,
                    indexed_files,
                    baseline_passed,
                    true,
                    0,
                    None,
                    Some(exhausted.reason),
                );
            }
            return self.finish_run(
                spec,
                phases_completed,
                budget,
                retrieved_files,
                indexed_files,
                true,
                true,
                0,
                None,
                None,
            );
        }

        if let Err(exhausted) = budget.record_attempt() {
            return self.finish_run(
                spec,
                phases_completed,
                budget,
                retrieved_files,
                indexed_files,
                baseline_passed,
                final_passed,
                0,
                None,
                Some(exhausted.reason),
            );
        }
        if let Err(exhausted) = budget.tick_wall(started) {
            return self.finish_run(
                spec,
                phases_completed,
                budget,
                retrieved_files,
                indexed_files,
                baseline_passed,
                final_passed,
                0,
                None,
                Some(exhausted.reason),
            );
        }

        phases_completed.push("isolate".into());
        let session = match IsolatedRepoSession::open(&self.root) {
            Ok(session) => session,
            Err(error) => {
                return self.finish_run(
                    spec,
                    phases_completed,
                    budget,
                    retrieved_files,
                    indexed_files,
                    baseline_passed,
                    final_passed,
                    0,
                    None,
                    Some(error),
                );
            }
        };
        let work_root = session.work_root().to_path_buf();
        repo_task.repo = work_root.to_string_lossy().to_string();

        phases_completed.push("repair".into());
        let work_verifier = RepairVerifier::new(&work_root, self.policy.clone());
        let mut supervisor = RepoRunSupervisor::new(&work_root, self.policy.clone());
        let supervisor_outcome = match supervisor.execute_task_with_budget(&repo_task, &mut budget) {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = session.discard();
                return self.finish_run(
                    spec,
                    phases_completed,
                    budget,
                    retrieved_files,
                    indexed_files,
                    baseline_passed,
                    final_passed,
                    0,
                    None,
                    Some(error),
                );
            }
        };

        let work_final = match work_verifier.verify(&repo_task.test_command) {
            Ok(report) => report,
            Err(error) => {
                let _ = session.discard();
                return self.finish_run(
                    spec,
                    phases_completed,
                    budget,
                    retrieved_files,
                    indexed_files,
                    baseline_passed,
                    final_passed,
                    supervisor_outcome.repair_iterations,
                    Some(supervisor_outcome),
                    Some(error),
                );
            }
        };

        if work_final.success {
            if let Err(error) = session.promote() {
                let _ = session.discard();
                return self.finish_run(
                    spec,
                    phases_completed,
                    budget,
                    retrieved_files,
                    indexed_files,
                    baseline_passed,
                    false,
                    supervisor_outcome.repair_iterations,
                    Some(supervisor_outcome),
                    Some(error),
                );
            }
            phases_completed.push("promote".into());
        }
        let _ = session.discard();

        let final_report = match verifier.verify(&repo_task.test_command) {
            Ok(report) => report,
            Err(error) => {
                return self.finish_run(
                    spec,
                    phases_completed,
                    budget,
                    retrieved_files,
                    indexed_files,
                    baseline_passed,
                    final_passed,
                    supervisor_outcome.repair_iterations,
                    Some(supervisor_outcome),
                    Some(error),
                );
            }
        };
        final_passed = final_report.success;
        phases_completed.push("final_verify".into());
        self.agent.trace_mut().push(
            "final_verify",
            repo_task.test_command.clone(),
            format!("exit={:?}", final_report.exit_code),
            if final_passed { "ok" } else { "failed" }.to_string(),
        );

        if final_passed && !baseline_passed {
            self.agent.assign_credit(
                CreditCategory::Verification,
                1.0,
                "acceptance oracle improved after repair",
            );
        } else if !final_passed {
            self.agent.assign_credit(
                CreditCategory::Verification,
                0.0,
                "acceptance oracle still failing",
            );
        }

        if let Err(exhausted) = budget.tick_wall(started) {
            let repair_iterations = supervisor_outcome.repair_iterations;
            return self.finish_run(
                spec,
                phases_completed,
                budget,
                retrieved_files,
                indexed_files,
                baseline_passed,
                final_passed,
                repair_iterations,
                Some(supervisor_outcome),
                Some(exhausted.reason),
            );
        }

        let repair_iterations = supervisor_outcome.repair_iterations;
        let repair_error = supervisor_outcome.error.clone();
        self.finish_run(
            spec,
            phases_completed,
            budget,
            retrieved_files,
            indexed_files,
            baseline_passed,
            final_passed,
            repair_iterations,
            Some(supervisor_outcome),
            if final_passed {
                None
            } else {
                repair_error
            },
        )
    }

    fn finish_run(
        &self,
        spec: &CodeTaskSpec,
        phases_completed: Vec<String>,
        budget: AgentRunBudget,
        retrieved_files: Vec<String>,
        indexed_files: usize,
        baseline_passed: bool,
        final_passed: bool,
        repair_iterations: usize,
        supervisor_outcome: Option<RepoRunOutcome>,
        error: Option<String>,
    ) -> RepoAgentRunResult {
        RepoAgentRunResult {
            run_id: spec.run_id.clone(),
            success: final_passed,
            phases_completed,
            baseline_passed,
            final_passed,
            retrieved_files,
            indexed_files,
            repair_iterations,
            supervisor_outcome,
            trace_len: self.agent.trace().len(),
            credit_summary: self.agent.credit().summary(),
            budget,
            error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::coding_intent::CodingIntent;
    use crate::agent::repo::{
        nl_fixture_cargo_test_command, write_nl_fixture_crate, GuardrailPolicy,
    };
    use crate::agent::runtime::CodeTaskSpec;
    use std::fs;
    use std::sync::Mutex;

    static REPO_AGENT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn nl_code_task(root: &Path, fixture_id: &str, issue: &str) -> CodeTaskSpec {
        let intent = CodingIntent::from_nl(issue).expect("intent");
        CodeTaskSpec::from_nl(
            root.to_string_lossy(),
            issue,
            intent,
            nl_fixture_cargo_test_command(fixture_id).expect("cmd"),
            vec!["src/**".into()],
            3,
        )
    }

    #[test]
    fn repo_agent_runs_nl_fixture_with_cargo_oracle() {
        let _guard = REPO_AGENT_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_repo_agent_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_multiply").expect("write");
        let spec = nl_code_task(&root, "nl_fixture_multiply", "multiply two numbers");
        let mut agent = RepoAgent::new(&root, GuardrailPolicy::default());
        let result = agent.run(&spec);
        assert!(!result.baseline_passed);
        assert!(result.final_passed);
        assert!(result.success);
        assert!(result.phases_completed.contains(&"repair".to_string()));
        assert!(result.phases_completed.contains(&"isolate".to_string()));
        assert!(result.phases_completed.contains(&"promote".to_string()));
        assert!(result
            .retrieved_files
            .iter()
            .any(|path| path.ends_with(".rs")));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repo_agent_negative_control_does_not_false_pass() {
        let _guard = REPO_AGENT_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_repo_agent_neg_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_multiply").expect("write");
        let spec = CodeTaskSpec::from_nl(
            root.to_string_lossy(),
            "factorial of n",
            CodingIntent::from_nl("compute factorial").unwrap_or_else(|_| {
                CodingIntent {
                    function_name: "factorial".into(),
                    signature: "i64 -> i64".into(),
                    category: "math".into(),
                    description: "compute factorial".into(),
                    examples: Vec::new(),
                    constraints: Vec::new(),
                    confidence: 0.5,
                    unresolved: vec!["no registry match".into()],
                    evidence_entity_ids: Vec::new(),
                    reference_code: String::new(),
                }
            }),
            "cargo test nl_fixture_multiply --lib",
            vec!["src/**".into()],
            2,
        );
        let mut agent = RepoAgent::new(&root, GuardrailPolicy::default());
        let result = agent.run(&spec);
        assert!(!result.success);
        assert!(!result.final_passed);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repo_agent_repairs_multifile_multiply_fixture() {
        let _guard = REPO_AGENT_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_repo_agent_mf_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_multifile_multiply").expect("write");
        let spec = nl_code_task(
            &root,
            "nl_fixture_multifile_multiply",
            "multiply two numbers",
        );
        let mut agent = RepoAgent::new(&root, GuardrailPolicy::default());
        let result = agent.run(&spec);
        assert!(!result.baseline_passed);
        assert!(result.final_passed);
        assert!(result.success);
        assert!(result
            .retrieved_files
            .iter()
            .any(|path| path.contains("ops.rs")));
        let ops = fs::read_to_string(root.join("src/ops.rs")).expect("ops");
        assert!(ops.contains("a * b"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repo_agent_repairs_gcd_via_general_synthesis() {
        let _guard = REPO_AGENT_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_repo_agent_gcd_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_gcd").expect("write");
        let spec = nl_code_task(&root, "nl_fixture_gcd", "greatest common divisor");
        let mut agent = RepoAgent::new(&root, GuardrailPolicy::default());
        let result = agent.run(&spec);
        assert!(!result.baseline_passed);
        assert!(result.final_passed, "error: {:?}", result.error);
        assert!(result.success);
        let lib = fs::read_to_string(root.join("src/lib.rs")).expect("lib");
        assert!(!lib.contains("if a < b { a } else { b }"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repo_agent_respects_exhausted_wall_budget() {
        let _guard = REPO_AGENT_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_repo_agent_budget_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_add").expect("write");
        let mut spec = nl_code_task(&root, "nl_fixture_add", "add two numbers");
        spec.budget.max_wall_ms = 0;
        let mut agent = RepoAgent::new(&root, GuardrailPolicy::default());
        let result = agent.run(&spec);
        assert!(!result.success);
        assert!(result.error.as_ref().is_some_and(|e| e.contains("wall")));
        let _ = fs::remove_dir_all(root);
    }
}
