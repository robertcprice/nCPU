//! Durable repo task supervisor: `AgentRun` persistence + repair loop (Package B/I).

use crate::agent::agent_run::AgentRun;
use crate::agent::repo::{
    GuardrailPolicy, RepairContext, RepairLoop, RepairLoopResult, RepoTaskSpec,
};
use crate::agent::runtime::AgentRunStatus;
use crate::agent::synthesis_proposer::{
    nl_description_from_issue, nl_synthesis_proposer, task_uses_nl_synthesis,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Outcome of a supervised repo run (repair loop + optional NL agent run).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoRunOutcome {
    pub task_id: String,
    pub repair_success: bool,
    pub repair_iterations: usize,
    pub agent_run_status: Option<AgentRunStatus>,
    pub agent_run_path: Option<String>,
    pub error: Option<String>,
}

/// Runs repair tasks with durable `AgentRun` snapshots for NL synthesis tasks.
pub struct RepoRunSupervisor {
    root: PathBuf,
    policy: GuardrailPolicy,
    state_root: PathBuf,
}

impl RepoRunSupervisor {
    pub fn new(root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        let root = root.into();
        let state_root = root.join(".nsynth").join("runs");
        Self {
            root,
            policy,
            state_root,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn policy(&self) -> &GuardrailPolicy {
        &self.policy
    }

    pub fn agent_run_path(&self, task_id: &str) -> PathBuf {
        self.state_root
            .join(format!("{}.json", sanitize_id(task_id)))
    }

    /// Execute many tasks sequentially (resumes durable runs when snapshots exist).
    pub fn execute_tasks(&mut self, tasks: &[RepoTaskSpec]) -> Vec<RepoRunOutcome> {
        tasks
            .iter()
            .map(|task| {
                self.resume_task(task).unwrap_or_else(|e| RepoRunOutcome {
                    task_id: task.id.clone(),
                    repair_success: false,
                    repair_iterations: 0,
                    agent_run_status: None,
                    agent_run_path: Some(
                        self.agent_run_path(&task.id).to_string_lossy().to_string(),
                    ),
                    error: Some(e),
                })
            })
            .collect()
    }

    /// Execute a task, auto-selecting the NL synthesis proposer when appropriate.
    pub fn execute_task(&mut self, task: &RepoTaskSpec) -> Result<RepoRunOutcome, String> {
        let mut budget = crate::agent::runtime::AgentRunBudget::default();
        self.execute_task_with_budget(task, &mut budget)
    }

    /// Execute with shared runtime budget (synthesis attempt limits).
    pub fn execute_task_with_budget(
        &mut self,
        task: &RepoTaskSpec,
        budget: &mut crate::agent::runtime::AgentRunBudget,
    ) -> Result<RepoRunOutcome, String> {
        fs::create_dir_all(&self.state_root).map_err(|e| format!("create run state dir: {e}"))?;
        if task_uses_nl_synthesis(task) {
            self.execute_nl_task(task, Some(budget))
        } else {
            Err(format!(
                "task {} has no supported proposer (use nl: or synthesize: prefix)",
                task.id
            ))
        }
    }

    /// Resume a task from a persisted `AgentRun` snapshot when present.
    pub fn resume_task(&mut self, task: &RepoTaskSpec) -> Result<RepoRunOutcome, String> {
        fs::create_dir_all(&self.state_root).map_err(|e| format!("create run state dir: {e}"))?;
        let run_path = self.agent_run_path(&task.id);
        if !run_path.exists() {
            return self.execute_task(task);
        }
        self.execute_nl_task(task, None)
    }

    /// Apply a clarification answer and continue an in-progress NL task.
    pub fn clarify_and_resume(
        &mut self,
        task: &RepoTaskSpec,
        field: linguigenesis_core::coding_dialogue::ClarificationField,
        answer: &str,
    ) -> Result<RepoRunOutcome, String> {
        let run_path = self.agent_run_path(&task.id);
        let mut run = if run_path.exists() {
            AgentRun::load(&run_path)?
        } else {
            let description =
                nl_description_from_issue(&task.issue).unwrap_or_else(|| task.issue.clone());
            let mut fresh = AgentRun::start(description);
            fresh.comprehend().map_err(|e| e.to_string())?;
            fresh.save(&run_path)?;
            fresh
        };
        if !run.needs_clarification() {
            return Err(format!(
                "task {} is not awaiting clarification (status={:?})",
                task.id, run.status
            ));
        }
        run.clarify(field, answer)?;
        run.save(&run_path)?;
        self.execute_nl_task(task, None)
    }

    fn execute_nl_task(
        &mut self,
        task: &RepoTaskSpec,
        mut budget: Option<&mut crate::agent::runtime::AgentRunBudget>,
    ) -> Result<RepoRunOutcome, String> {
        let run_path = self.agent_run_path(&task.id);
        if let Some(budget) = budget.as_mut() {
            self.seed_agent_run_budget(&run_path, task, budget)?;
        }
        let mut loop_runner = RepairLoop::new(&self.root, self.policy.clone());
        let context = loop_runner.context()?;
        let run_path_capture = run_path.clone();

        let proposer =
            |task: &RepoTaskSpec,
             context: &RepairContext,
             _iteration: usize,
             analysis: Option<&crate::agent::repo::FailureAnalysis>| {
                nl_synthesis_proposer_with_run(&run_path_capture, task, context, analysis)
            };

        let repair = loop_runner
            .run_with_context(task, &context, &proposer)
            .map_err(|e| e.to_string())?;

        let agent_status = if run_path.exists() {
            Some(AgentRun::load(&run_path)?.status)
        } else {
            None
        };

        if let Some(budget) = budget {
            if run_path.exists() {
                let run = AgentRun::load(&run_path)?;
                budget.synthesis_candidates_used = run.budget.synthesis_candidates_used;
            }
        }

        Ok(RepoRunOutcome {
            task_id: task.id.clone(),
            repair_success: repair.success,
            repair_iterations: repair.iterations,
            agent_run_status: agent_status,
            agent_run_path: Some(run_path.to_string_lossy().to_string()),
            error: if repair.success {
                None
            } else {
                repair
                    .last_failure
                    .as_ref()
                    .map(|f| f.message.clone())
                    .or_else(|| {
                        repair
                            .last_verification
                            .as_ref()
                            .map(|v| v.failure_output())
                    })
            },
        })
    }

    fn seed_agent_run_budget(
        &self,
        run_path: &Path,
        task: &RepoTaskSpec,
        budget: &mut crate::agent::runtime::AgentRunBudget,
    ) -> Result<(), String> {
        let description =
            nl_description_from_issue(&task.issue).unwrap_or_else(|| task.issue.clone());
        let mut run = if run_path.exists() {
            AgentRun::load(run_path)?
        } else {
            AgentRun::start(description)
        };
        run.budget.max_synthesis_candidates = budget.max_synthesis_candidates;
        run.budget.synthesis_candidates_used = budget.synthesis_candidates_used;
        run.save(run_path)?;
        Ok(())
    }
}

/// NL proposer that persists `AgentRun` state for resume across repair iterations.
pub fn nl_synthesis_proposer_with_run(
    run_path: &Path,
    task: &RepoTaskSpec,
    context: &RepairContext,
    analysis: Option<&crate::agent::repo::FailureAnalysis>,
) -> Result<crate::agent::repo::RepairPatch, String> {
    let description = nl_description_from_issue(&task.issue).unwrap_or_else(|| task.issue.clone());

    // Cheap deterministic edits first (no synthesis budget): a RENAME refactor or an ADD-PARAM
    // request. Present in the standalone nl_synthesis_proposer but MISSING from this supervisor
    // ladder (which was an incomplete copy), so repo repairs never reached them.
    if let Some(patch) = crate::agent::synthesis_proposer::try_rename_patch(context, &description) {
        return Ok(patch);
    }
    if let Some(patch) =
        crate::agent::synthesis_proposer::try_add_param_patch(context, &description)
    {
        return Ok(patch);
    }

    // Primary path: genuine verified synthesis (bridge + solver), generalizing
    // to any demonstrated function (registry op or inline I/O examples) rather
    // than the canned keyword shapes. Real synthesis *is* a synthesis candidate,
    // so it is gated by the persisted run budget: when exhausted we skip it and
    // let the free keyword fast-patch (or the paid-synthesis rejection) govern.
    // Falls through when the solver output is not directly compilable for the
    // repo's concrete signature.
    let budget_allows_synthesis = if run_path.exists() {
        AgentRun::load(run_path)
            .map(|run| !run.budget.exhausted())
            .unwrap_or(true)
    } else {
        true
    };
    if budget_allows_synthesis {
        // Two synthesis candidates, both budget-gated (bypassing the budget here would defeat the
        // exhaustion guard): (1) REAL synthesis from prose examples, and (2) TEST-MINED synthesis
        // — a bare "fix the failing tests" request carries no prose examples, but the failing
        // test's `assert_eq!` calls ARE I/O examples; mine them and solve the real function
        // (deterministic, verified, no model). This step was present in the standalone
        // `nl_synthesis_proposer` but MISSING here, so every bare-NL repo repair used to fall
        // straight through to the empty-intent fallback and fail with "CodingIntent has no
        // examples".
        // Synthesis candidates in the standalone-ladder order, all budget-gated: (1) EMERGENT
        // synthesis of an existing fn from bare NL, (2) FEATURE-ADD — synthesize a MISSING function
        // referenced by a failing test and append it (was missing here, so feature-add never fired
        // via the repo loop), (3) REAL synthesis from prose examples, (4) TEST-MINED synthesis from
        // the failing test's asserts.
        let synth = crate::agent::synthesis_proposer::try_emergent_synthesis_patch(
            task,
            context,
            &description,
            analysis,
        )
        .or_else(|| {
            crate::agent::synthesis_proposer::try_emergent_addition_patch(
                task,
                context,
                &description,
                analysis,
            )
        })
        .or_else(|| {
            crate::agent::synthesis_proposer::try_real_synthesis_patch(task, context, &description)
        })
        .or_else(|| {
            crate::agent::synthesis_proposer::try_test_mined_synthesis_patch(
                task,
                context,
                &description,
            )
        });
        if let Some(patch) = synth {
            if run_path.exists() {
                if let Ok(mut run) = AgentRun::load(run_path) {
                    let _ = run.budget.record_synthesis_candidate();
                    let _ = run.save(run_path);
                }
            }
            return Ok(patch);
        }
    }

    if let Some(patch) = crate::agent::synthesis_proposer::try_nl_repo_fast_patch(
        task,
        context,
        &description,
        analysis,
    ) {
        return Ok(patch);
    }

    let mut run = if run_path.exists() {
        AgentRun::load(run_path)?
    } else {
        AgentRun::start(description)
    };

    if run.status == AgentRunStatus::Created {
        run.comprehend().map_err(|e| e.to_string())?;
        run.save(run_path)?;
    }

    if run.needs_clarification() {
        return Err(format!(
            "clarification required: {:?}",
            run.clarification_questions
        ));
    }

    if let Some(intent) = run.intent.as_ref() {
        let target_hint =
            match crate::agent::synthesis_proposer::pick_target_path(task, context, Some(intent)) {
                Ok(target) => {
                    crate::agent::synthesis_proposer::read_relative_file(context, &target).ok()
                }
                Err(_) => None,
            };
        if let Some(rust_body) = crate::agent::synthesis_proposer::repo_rust_body_for_nl(
            intent,
            "",
            target_hint.as_deref(),
        ) {
            let stub = crate::solver::SolveResult {
                success: true,
                code: rust_body,
                method: "nl_rust_repo_stub".to_string(),
                error: None,
                metadata: Default::default(),
            };
            return crate::agent::synthesis_proposer::repair_patch_from_synthesis(
                task,
                context,
                Some(intent),
                &stub,
            );
        }
    }

    if run.status != AgentRunStatus::Succeeded && run.synthesis.is_none() {
        run.synthesize()?;
        run.save(run_path)?;
    }

    let synthesis = run
        .synthesis
        .as_ref()
        .ok_or_else(|| "no synthesis result after agent run".to_string())?;
    crate::agent::synthesis_proposer::repair_patch_from_synthesis(
        task,
        context,
        run.intent.as_ref(),
        synthesis,
    )
}

fn sanitize_id(id: &str) -> String {
    id.chars()
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
    use crate::agent::repo::{
        nl_fixture_cargo_test_command, nl_synthesis_fixture_suite, write_nl_fixture_crate,
        HardnessProfile, HardnessTier, RepairVerifier, RepoTaskKind,
    };
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static NL_SUPERVISOR_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn supervisor_task_from_fixture(fixture_id: &str, task_id: &str, root: &Path) -> RepoTaskSpec {
        let fixture = nl_synthesis_fixture_suite()
            .into_iter()
            .find(|task| task.id == fixture_id)
            .expect("fixture");
        RepoTaskSpec {
            id: task_id.to_string(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::Feature,
            issue: fixture.issue,
            test_command: nl_fixture_cargo_test_command(fixture_id).expect("cargo test cmd"),
            allowed_files: fixture.allowed_files,
            max_iterations: fixture.max_iterations,
            hardness: HardnessProfile::for_expected_tier(fixture.expected_tier_min),
            signals: Vec::new(),
        }
    }

    fn run_supervisor_nl_fixture(fixture_id: &str, task_id: &str) {
        let _guard = NL_SUPERVISOR_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!(
            "nsynth_super_{}_{}_{}",
            fixture_id,
            task_id,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, fixture_id).expect("write fixture");
        let task = supervisor_task_from_fixture(fixture_id, task_id, &root);
        let mut supervisor = RepoRunSupervisor::new(&root, GuardrailPolicy::default());
        let outcome = supervisor.execute_task(&task).expect("execute");
        assert!(outcome.repair_success, "fixture {}", fixture_id);
        let verification = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify");
        assert!(verification.success, "fixture {}", fixture_id);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn supervisor_executes_nl_add_task() {
        run_supervisor_nl_fixture("nl_fixture_add", "nl-add-supervisor");
    }

    #[test]
    fn supervisor_resumes_from_agent_run_snapshot() {
        let _guard = NL_SUPERVISOR_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_super_resume_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_add").expect("write");
        let task = supervisor_task_from_fixture("nl_fixture_add", "nl-resume", &root);

        let mut supervisor = RepoRunSupervisor::new(&root, GuardrailPolicy::default());
        let mut run = AgentRun::start("add two numbers");
        run.comprehend().expect("comprehend");
        run.synthesize().expect("synthesize");
        let path = supervisor.agent_run_path("nl-resume");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        run.save(&path).expect("save");

        let outcome = supervisor.resume_task(&task).expect("resume");
        assert!(outcome.repair_success);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn supervisor_executes_nl_subtract_task() {
        run_supervisor_nl_fixture("nl_fixture_subtract", "nl-sub-supervisor");
    }

    #[test]
    fn supervisor_executes_nl_multiply_task() {
        run_supervisor_nl_fixture("nl_fixture_multiply", "nl-mul-supervisor");
    }

    #[test]
    fn supervisor_executes_nl_reverse_task() {
        run_supervisor_nl_fixture("nl_fixture_reverse", "nl-rev-supervisor");
    }

    #[test]
    fn supervisor_executes_nl_divide_task() {
        run_supervisor_nl_fixture("nl_fixture_divide", "nl-div-supervisor");
    }

    #[test]
    fn supervisor_executes_nl_max_task() {
        run_supervisor_nl_fixture("nl_fixture_max", "nl-max-supervisor");
    }

    #[test]
    fn supervisor_executes_multifile_multiply_fixture() {
        run_supervisor_nl_fixture("nl_fixture_multifile_multiply", "nl-mf-multiply-supervisor");
    }

    #[test]
    fn supervisor_syncs_synthesis_budget_from_agent_run() {
        let _guard = NL_SUPERVISOR_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_super_budget_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_add").expect("write");
        let task = supervisor_task_from_fixture("nl_fixture_add", "nl-budget", &root);
        let mut supervisor = RepoRunSupervisor::new(&root, GuardrailPolicy::default());
        let mut budget = crate::agent::runtime::AgentRunBudget {
            max_synthesis_candidates: 0,
            synthesis_candidates_used: 0,
            ..Default::default()
        };
        let outcome = supervisor
            .execute_task_with_budget(&task, &mut budget)
            .expect("execute");
        assert!(outcome.repair_success);
        assert_eq!(budget.synthesis_candidates_used, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn supervisor_rejects_exhausted_synthesis_budget_for_gcd() {
        let _guard = NL_SUPERVISOR_TEST_LOCK.lock().unwrap();
        let root =
            std::env::temp_dir().join(format!("nsynth_super_gcd_budget_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_gcd").expect("write");
        let task = supervisor_task_from_fixture("nl_fixture_gcd", "nl-gcd-budget", &root);
        let mut supervisor = RepoRunSupervisor::new(&root, GuardrailPolicy::default());
        let mut budget = crate::agent::runtime::AgentRunBudget {
            max_synthesis_candidates: 0,
            synthesis_candidates_used: 1,
            ..Default::default()
        };
        let outcome = supervisor.execute_task_with_budget(&task, &mut budget);
        assert!(outcome.is_err());
        assert!(outcome
            .unwrap_err()
            .contains("synthesis candidate budget exhausted"));
        let verification = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify");
        assert!(!verification.success);
        let _ = fs::remove_dir_all(root);
    }
}
