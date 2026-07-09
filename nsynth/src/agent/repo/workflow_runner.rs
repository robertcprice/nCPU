//! Resumable typed workflow runner over `RepoAgent` + supervisor (Package I).

use crate::agent::repo::{
    nl_fixture_code_specs, nl_synthesis_fixture_suite, GuardrailPolicy, RepoAgent,
    RepoAgentRunResult, RepoBenchmark, RepoRunOutcome, RepoRunSupervisor, RepoTaskSpec,
};
use crate::agent::runtime::CodeTaskSpec;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowRunReport {
    pub root: String,
    pub total: usize,
    pub succeeded: usize,
    pub outcomes: Vec<RepoRunOutcome>,
    pub agent_results: Vec<RepoAgentRunResult>,
}

/// Runs a batch of repo tasks with durable NL resume semantics.
pub struct RepoWorkflowRunner {
    supervisor: RepoRunSupervisor,
}

impl RepoWorkflowRunner {
    pub fn new(root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        Self {
            supervisor: RepoRunSupervisor::new(root, policy),
        }
    }

    pub fn root(&self) -> &Path {
        self.supervisor.root()
    }

    pub fn supervisor(&self) -> &RepoRunSupervisor {
        &self.supervisor
    }

    pub fn supervisor_mut(&mut self) -> &mut RepoRunSupervisor {
        &mut self.supervisor
    }

    /// Run NL synthesis tasks from a benchmark suite via `RepoAgent` (oracle-gated loop).
    pub fn run_nl_benchmark(&mut self, benchmark: &RepoBenchmark) -> WorkflowRunReport {
        let specs: Vec<CodeTaskSpec> = benchmark
            .tasks()
            .iter()
            .filter(|task| task.issue.starts_with("synthesize:") || task.issue.starts_with("nl:"))
            .map(|task| {
                let nl = task
                    .issue
                    .strip_prefix("synthesize:")
                    .or_else(|| task.issue.strip_prefix("nl:"))
                    .map(str::trim)
                    .unwrap_or(&task.issue);
                let intent = crate::agent::coding_intent::CodingIntent::from_nl(nl)
                    .expect("benchmark nl intent");
                CodeTaskSpec::from_nl(
                    benchmark.root().to_string_lossy(),
                    nl,
                    intent,
                    task.test_command.clone(),
                    task.allowed_files.clone(),
                    task.max_iterations,
                )
            })
            .collect();
        self.run_code_specs(&specs)
    }

    /// Run canonical code tasks through `RepoAgent::run`.
    pub fn run_code_specs(&mut self, specs: &[CodeTaskSpec]) -> WorkflowRunReport {
        let mut agent = RepoAgent::new(self.supervisor.root(), self.supervisor.policy().clone());
        let mut agent_results = Vec::new();
        let mut outcomes = Vec::new();
        for spec in specs {
            let result = agent.run(spec);
            outcomes.push(RepoRunOutcome {
                task_id: spec.run_id.0.clone(),
                repair_success: result.success,
                repair_iterations: result.repair_iterations,
                agent_run_status: None,
                agent_run_path: Some(
                    self.supervisor
                        .agent_run_path(&spec.run_id.0)
                        .to_string_lossy()
                        .to_string(),
                ),
                error: result.error.clone(),
            });
            agent_results.push(result);
        }
        let succeeded = agent_results.iter().filter(|r| r.success).count();
        WorkflowRunReport {
            root: self.supervisor.root().to_string_lossy().to_string(),
            total: agent_results.len(),
            succeeded,
            outcomes,
            agent_results,
        }
    }

    /// Run explicit repo task specs via supervisor (legacy resume path).
    pub fn run_specs(&mut self, specs: &[RepoTaskSpec]) -> WorkflowRunReport {
        let outcomes = self.supervisor.execute_tasks(specs);
        let succeeded = outcomes.iter().filter(|o| o.repair_success).count();
        WorkflowRunReport {
            root: self.supervisor.root().to_string_lossy().to_string(),
            total: outcomes.len(),
            succeeded,
            outcomes,
            agent_results: Vec::new(),
        }
    }

    /// Run local NL fixture tasks through `RepoAgent` + cargo-test oracles.
    pub fn run_nl_fixtures(&mut self) -> WorkflowRunReport {
        let specs = nl_fixture_code_specs(self.supervisor.root());
        self.run_code_specs(&specs)
    }

    /// Run a single NL query through the universal session router.
    pub fn run_query(&mut self, query: &str) -> crate::agent::session::AgentQueryResult {
        let mut session =
            crate::agent::session::CodingAgentSession::new(self.supervisor.root(), self.supervisor.policy().clone());
        // Per-task wall-clock bound so a hard repo task degrades to a refusal instead of
        // spinning in the unbounded array-gradient (the same liveness fix the product
        // bins carry). Default 20s, NSYNTH_QUERY_BUDGET_MS override. Held for this solve.
        let budget_ms: u64 = std::env::var("NSYNTH_QUERY_BUDGET_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20000);
        let _budget = crate::synthesis::QuerySolveBudget::millis(budget_ms);
        session.handle_query(query)
    }

    /// Persist workflow report JSON under `.nsynth/workflows/`.
    pub fn save_report(&self, report: &WorkflowRunReport, name: &str) -> Result<PathBuf, String> {
        let dir = self.supervisor.root().join(".nsynth").join("workflows");
        fs::create_dir_all(&dir).map_err(|e| format!("create workflow dir: {e}"))?;
        let path = dir.join(format!("{}.json", sanitize_workflow_name(name)));
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| format!("serialize workflow report: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("write workflow report: {e}"))?;
        Ok(path)
    }
}

fn sanitize_workflow_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::repo::benchmark::nl_synthesis_fixture_ci_subset;
    use crate::agent::repo::write_nl_fixture_crate;
    use std::fs;
    use std::sync::Mutex;

    static WORKFLOW_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn run_nl_fixture_tasks(tasks: &[crate::agent::repo::LocalBenchmarkTask]) {
        let root =
            std::env::temp_dir().join(format!("nsynth_workflow_nl_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        for task in tasks {
            let _ = fs::remove_dir_all(&root);
            write_nl_fixture_crate(&root, &task.id).expect("write fixture crate");
            let nl = task
                .issue
                .strip_prefix("synthesize:")
                .map(str::trim)
                .unwrap_or(&task.issue);
            let spec = nl_fixture_code_specs(&root)
                .into_iter()
                .find(|spec| spec.nl_input == nl)
                .expect("code spec");

            let mut runner = RepoWorkflowRunner::new(&root, GuardrailPolicy::default());
            let report = runner.run_code_specs(&[spec]);
            assert_eq!(report.total, 1, "task {}", task.id);
            assert!(report.agent_results[0].success, "task {}", task.id);
        }

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn workflow_runner_executes_nl_fixture_ci_subset() {
        let _guard = WORKFLOW_TEST_LOCK.lock().unwrap();
        run_nl_fixture_tasks(&nl_synthesis_fixture_ci_subset());
    }

    #[test]
    #[ignore = "full 17-fixture Gate G5 corpus (~276s release); verified nightly via scripts/nsynth_g5_nightly.sh"]
    fn workflow_runner_executes_nl_fixture_suite() {
        let _guard = WORKFLOW_TEST_LOCK.lock().unwrap();
        run_nl_fixture_tasks(&nl_synthesis_fixture_suite());
    }

    #[test]
    fn workflow_runner_run_query_synthesis() {
        let _guard = WORKFLOW_TEST_LOCK.lock().unwrap();
        let root =
            std::env::temp_dir().join(format!("nsynth_workflow_query_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut runner = RepoWorkflowRunner::new(&root, GuardrailPolicy::default());
        let result = runner.run_query("add two numbers");
        assert_eq!(result.route, crate::agent::session::QueryRoute::SynthesizeFunction);
        assert!(result.success);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn workflow_runner_nl_fixtures_use_repo_agent_phases() {
        let _guard = WORKFLOW_TEST_LOCK.lock().unwrap();
        let root =
            std::env::temp_dir().join(format!("nsynth_workflow_agent_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_subtract").expect("write");
        let mut runner = RepoWorkflowRunner::new(&root, GuardrailPolicy::default());
        let spec = nl_fixture_code_specs(&root)
            .into_iter()
            .find(|spec| spec.nl_input == "subtract two numbers")
            .expect("spec");
        let report = runner.run_code_specs(&[spec]);
        assert!(report.agent_results[0].success);
        assert!(report
            .agent_results[0]
            .phases_completed
            .contains(&"final_verify".to_string()));
        let _ = fs::remove_dir_all(root);
    }
}
