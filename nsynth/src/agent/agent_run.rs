//! Durable agent run state for NL → synthesis workflows (Package B).

use crate::agent::coding_intent::CodingIntent;
use crate::agent::runtime::{
    transition, AgentRunBudget, AgentRunEvent, AgentRunId, AgentRunStatus, SCHEMA_VERSION,
};
use crate::linguigenesis_bridge::{BridgeError, LinguigenesisBridge};
use crate::solver::{solve_problem, SolveResult};
use linguigenesis_core::coding_comprehension::{CodingComprehension, ComprehensionOutcome};
use linguigenesis_core::coding_dialogue::{
    apply_clarification, needs_clarification, ClarificationField, ClarificationQuestion,
};
use std::time::Instant;

/// One end-to-end NL synthesis run with clarification support.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentRun {
    pub run_id: AgentRunId,
    pub schema_version: u32,
    pub status: AgentRunStatus,
    pub nl_input: String,
    pub intent: Option<CodingIntent>,
    pub clarification_questions: Vec<ClarificationQuestion>,
    pub synthesis: Option<SolveResult>,
    pub error: Option<String>,
    pub budget: AgentRunBudget,
    pub events: Vec<AgentRunEvent>,
    started_at: Instant,
}

impl AgentRun {
    /// Start a run from natural language.
    pub fn start(nl_input: impl Into<String>) -> Self {
        Self {
            run_id: AgentRunId::new(),
            schema_version: SCHEMA_VERSION,
            status: AgentRunStatus::Created,
            nl_input: nl_input.into(),
            intent: None,
            clarification_questions: Vec::new(),
            synthesis: None,
            error: None,
            budget: AgentRunBudget::default(),
            events: Vec::new(),
            started_at: Instant::now(),
        }
    }

    pub fn needs_clarification(&self) -> bool {
        self.status
            .needs_clarification(self.clarification_questions.len())
    }

    fn apply_transition(&mut self, next: AgentRunStatus, detail: &str) -> Result<(), String> {
        self.budget
            .tick_wall(self.started_at)
            .map_err(|e| e.reason)?;
        let (status, event) = transition(self.status, next, detail)?;
        self.status = status;
        self.events.push(event);
        Ok(())
    }

    /// Cancel an in-progress run.
    pub fn cancel(&mut self) -> Result<AgentRunStatus, String> {
        if self.status.is_terminal() {
            return Err(format!("cannot cancel terminal status {:?}", self.status));
        }
        self.apply_transition(AgentRunStatus::Cancelled, "cancelled by operator")?;
        Ok(self.status)
    }

    /// Comprehend NL and advance toward synthesis or clarification.
    pub fn comprehend(&mut self) -> Result<AgentRunStatus, BridgeError> {
        self.budget
            .record_attempt()
            .map_err(|e| BridgeError::InvalidInput(e.reason))?;
        if self.status == AgentRunStatus::Created {
            self.apply_transition(AgentRunStatus::Understanding, "begin comprehension")
                .map_err(|e| BridgeError::InvalidInput(e))?;
        }
        let bridge = LinguigenesisBridge::new();
        match bridge.comprehend_outcome(&self.nl_input) {
            Ok(ComprehensionOutcome::Ready(req)) => {
                self.intent = Some(CodingIntent::from_requirement(&req));
                self.clarification_questions.clear();
                if needs_clarification(&req) {
                    Ok(self.status)
                } else {
                    self.apply_transition(AgentRunStatus::Executing, "comprehension ready")
                        .map_err(|e| BridgeError::InvalidInput(e))?;
                    Ok(self.status)
                }
            }
            Ok(ComprehensionOutcome::NeedsClarification(req, questions)) => {
                self.intent = Some(CodingIntent::from_requirement(&req));
                self.clarification_questions = questions;
                Ok(self.status)
            }
            Err(e) => {
                self.status = AgentRunStatus::Failed;
                self.error = Some(e.to_string());
                Err(e)
            }
        }
    }

    /// Apply a clarification answer and advance toward synthesis.
    pub fn clarify(&mut self, field: ClarificationField, answer: &str) -> Result<(), String> {
        self.budget.record_attempt().map_err(|e| e.reason)?;
        let registry = LinguigenesisBridge::new()
            .registry_clone()
            .map_err(|e| e.to_string())?;
        let mut cc = CodingComprehension::new(registry);
        let mut req = cc.comprehend(&self.nl_input);
        if !apply_clarification(&mut req, &field, answer, cc.registry()) {
            return Err(format!("failed to apply clarification '{}'", answer));
        }
        self.intent = Some(CodingIntent::from_requirement(&req));
        self.clarification_questions.clear();
        if self.status == AgentRunStatus::Understanding {
            self.apply_transition(AgentRunStatus::Executing, "clarification applied")?;
        }
        Ok(())
    }

    /// Run synthesis when intent is ready.
    pub fn synthesize(&mut self) -> Result<&SolveResult, String> {
        self.budget
            .record_synthesis_candidate()
            .map_err(|e| e.reason)?;
        if self.needs_clarification() {
            return Err("clarification required before synthesis".to_string());
        }
        if self.status == AgentRunStatus::Understanding {
            self.apply_transition(AgentRunStatus::Executing, "begin synthesis")?;
        }
        if self.status != AgentRunStatus::Executing {
            return Err(format!(
                "synthesis requires Executing status, got {:?}",
                self.status
            ));
        }
        let intent = self
            .intent
            .as_ref()
            .ok_or_else(|| "no coding intent — call comprehend() first".to_string())?;
        let problem = intent.to_problem()?;
        let result = solve_problem(&problem);
        self.synthesis = Some(result.clone());
        if result.success {
            self.apply_transition(AgentRunStatus::Succeeded, "synthesis succeeded")?;
        } else {
            self.status = AgentRunStatus::Failed;
            self.error = result.error.clone();
        }
        Ok(self.synthesis.as_ref().expect("synthesis stored"))
    }

    /// Restore from a persisted snapshot.
    pub fn from_snapshot(snapshot: crate::agent::agent_run_persistence::AgentRunSnapshot) -> Self {
        use crate::solver::SolveResult;
        let status = snapshot.resolved_status();
        let synthesis = snapshot.synthesis_code.map(|code| SolveResult {
            success: snapshot.synthesis_success.unwrap_or(false),
            code,
            method: snapshot
                .synthesis_method
                .unwrap_or_else(|| "restored".to_string()),
            error: snapshot.error.clone(),
            metadata: Default::default(),
        });
        Self {
            run_id: snapshot.run_id,
            schema_version: snapshot.schema_version.max(SCHEMA_VERSION),
            status,
            nl_input: snapshot.nl_input,
            intent: snapshot.intent,
            clarification_questions: snapshot.clarification_questions,
            synthesis,
            error: snapshot.error,
            budget: snapshot.budget,
            events: snapshot.events,
            started_at: Instant::now(),
        }
    }

    /// Build canonical `CodeTaskSpec` when intent is available.
    pub fn to_code_task_spec(
        &self,
        repo_root: impl Into<String>,
        test_command: impl Into<String>,
        allowed_files: Vec<String>,
        max_iterations: usize,
    ) -> Result<crate::agent::runtime::CodeTaskSpec, String> {
        let intent = self
            .intent
            .as_ref()
            .ok_or_else(|| "no intent".to_string())?
            .clone();
        Ok(crate::agent::runtime::CodeTaskSpec {
            schema_version: SCHEMA_VERSION,
            run_id: self.run_id.clone(),
            repo_root: repo_root.into(),
            kind: crate::agent::repo::RepoTaskKind::Feature,
            intent,
            test_command: test_command.into(),
            allowed_files,
            max_iterations,
            budget: self.budget.clone(),
            nl_input: self.nl_input.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::runtime::legal_transition;

    #[test]
    fn agent_run_comprehend_and_synthesize_add() {
        let mut run = AgentRun::start("add two numbers");
        run.comprehend().expect("comprehend");
        let result = run.synthesize().expect("synthesize");
        assert!(result.success);
        assert_eq!(run.status, AgentRunStatus::Succeeded);
    }

    #[test]
    fn agent_run_cancel_from_understanding() {
        let mut run = AgentRun::start("add two numbers");
        run.comprehend().expect("comprehend");
        let status = run.cancel().expect("cancel");
        assert_eq!(status, AgentRunStatus::Cancelled);
        assert!(!legal_transition(run.status, AgentRunStatus::Executing));
    }

    #[test]
    fn agent_run_budget_blocks_extra_synthesis() {
        let mut run = AgentRun::start("add two numbers");
        run.budget.max_synthesis_candidates = 0;
        run.comprehend().expect("comprehend");
        assert!(run.synthesize().is_err());
    }
}
