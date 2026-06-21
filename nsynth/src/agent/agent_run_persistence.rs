//! Serializable agent run snapshot for resume (Package B/J).

use crate::agent::agent_run::AgentRun;
use crate::agent::coding_intent::CodingIntent;
use crate::agent::runtime::{
    AgentRunBudget, AgentRunEvent, AgentRunId, AgentRunStatus, SCHEMA_VERSION,
};
use linguigenesis_core::coding_dialogue::ClarificationQuestion;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Legacy phase labels for snapshot migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum LegacyAgentRunPhase {
    Comprehending,
    NeedsClarification,
    Synthesizing,
    Complete,
    Failed,
}

/// Persisted agent run state (without non-serializable `SolveResult` internals).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRunSnapshot {
    #[serde(default)]
    pub schema_version: u32,
    pub run_id: AgentRunId,
    pub status: AgentRunStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_phase: Option<LegacyAgentRunPhase>,
    pub nl_input: String,
    pub intent: Option<CodingIntent>,
    pub clarification_questions: Vec<ClarificationQuestion>,
    pub synthesis_success: Option<bool>,
    pub synthesis_code: Option<String>,
    pub synthesis_method: Option<String>,
    pub error: Option<String>,
    #[serde(default)]
    pub budget: AgentRunBudget,
    #[serde(default)]
    pub events: Vec<AgentRunEvent>,
}

fn migrate_legacy_phase(phase: LegacyAgentRunPhase) -> AgentRunStatus {
    match phase {
        LegacyAgentRunPhase::Comprehending | LegacyAgentRunPhase::NeedsClarification => {
            AgentRunStatus::Understanding
        }
        LegacyAgentRunPhase::Synthesizing => AgentRunStatus::Executing,
        LegacyAgentRunPhase::Complete => AgentRunStatus::Succeeded,
        LegacyAgentRunPhase::Failed => AgentRunStatus::Failed,
    }
}

impl AgentRunSnapshot {
    pub fn resolved_status(&self) -> AgentRunStatus {
        if self.schema_version == 0 {
            self.legacy_phase
                .map(migrate_legacy_phase)
                .unwrap_or(self.status)
        } else {
            self.status
        }
    }
}

impl AgentRun {
    /// Export a JSON-serializable snapshot.
    pub fn snapshot(&self) -> AgentRunSnapshot {
        AgentRunSnapshot {
            schema_version: self.schema_version,
            run_id: self.run_id.clone(),
            status: self.status,
            legacy_phase: None,
            nl_input: self.nl_input.clone(),
            intent: self.intent.clone(),
            clarification_questions: self.clarification_questions.clone(),
            synthesis_success: self.synthesis.as_ref().map(|r| r.success),
            synthesis_code: self.synthesis.as_ref().map(|r| r.code.clone()),
            synthesis_method: self.synthesis.as_ref().map(|r| r.method.clone()),
            error: self.error.clone(),
            budget: self.budget.clone(),
            events: self.events.clone(),
        }
    }

    /// Load run state from JSON.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content =
            fs::read_to_string(path.as_ref()).map_err(|e| format!("read agent run: {}", e))?;
        let snapshot: AgentRunSnapshot =
            serde_json::from_str(&content).map_err(|e| format!("parse agent run: {}", e))?;
        Ok(AgentRun::from_snapshot(snapshot))
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.snapshot())
            .map_err(|e| format!("serialize agent run: {}", e))?;
        fs::write(path.as_ref(), json).map_err(|e| format!("write agent run: {}", e))?;
        Ok(())
    }

    /// Resume: load snapshot and continue from current phase.
    pub fn resume<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        Self::load(path)
    }

    /// Advance run to completion when possible (comprehend → synthesize).
    pub fn run_to_completion(&mut self) -> Result<(), String> {
        if self.status == AgentRunStatus::Created {
            self.comprehend().map_err(|e| e.to_string())?;
        }
        if self.needs_clarification() {
            return Err("clarification required before completion".to_string());
        }
        if self.status == AgentRunStatus::Understanding || self.status == AgentRunStatus::Executing
        {
            self.synthesize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn agent_run_save_load_roundtrip() {
        let mut run = AgentRun::start("add two numbers");
        run.comprehend().expect("comprehend");
        run.synthesize().expect("synthesize");

        let path = env::temp_dir().join(format!("agent_run_snapshot_{}.json", std::process::id()));
        run.save(&path).expect("save");
        let restored = AgentRun::load(&path).expect("load");
        assert_eq!(restored.status, AgentRunStatus::Succeeded);
        assert!(restored.synthesis.as_ref().unwrap().success);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn legacy_snapshot_migration() {
        let legacy = r#"{
            "schema_version": 0,
            "run_id": "run-legacy",
            "status": "Created",
            "legacy_phase": "Complete",
            "nl_input": "add two numbers",
            "intent": null,
            "clarification_questions": [],
            "synthesis_success": true,
            "synthesis_code": "fn add() {}",
            "synthesis_method": "test",
            "error": null,
            "budget": {
                "max_attempts": 8,
                "attempts_used": 1,
                "max_wall_ms": 120000,
                "wall_ms_used": 0,
                "max_synthesis_candidates": 4,
                "synthesis_candidates_used": 1
            },
            "events": []
        }"#;
        let snapshot: AgentRunSnapshot = serde_json::from_str(legacy).expect("parse");
        let run = AgentRun::from_snapshot(snapshot);
        assert_eq!(run.status, AgentRunStatus::Succeeded);
    }
}
