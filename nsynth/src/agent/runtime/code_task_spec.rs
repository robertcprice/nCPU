use crate::agent::coding_intent::CodingIntent;
use crate::agent::repo::{RepoTaskKind, RepoTaskSpec};
use crate::agent::runtime::{AgentRunBudget, AgentRunId, SCHEMA_VERSION};
use serde::{Deserialize, Serialize};

/// Grounded coding task contract (Section 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeTaskSpec {
    pub schema_version: u32,
    pub run_id: AgentRunId,
    pub repo_root: String,
    pub kind: RepoTaskKind,
    pub intent: CodingIntent,
    pub test_command: String,
    pub allowed_files: Vec<String>,
    pub max_iterations: usize,
    pub budget: AgentRunBudget,
    pub nl_input: String,
}

impl CodeTaskSpec {
    pub fn from_nl(
        repo_root: impl Into<String>,
        nl_input: impl Into<String>,
        intent: CodingIntent,
        test_command: impl Into<String>,
        allowed_files: Vec<String>,
        max_iterations: usize,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            run_id: AgentRunId::new(),
            repo_root: repo_root.into(),
            kind: RepoTaskKind::Feature,
            intent,
            test_command: test_command.into(),
            allowed_files,
            max_iterations,
            budget: AgentRunBudget::default(),
            nl_input: nl_input.into(),
        }
    }

    pub fn to_repo_task_spec(&self, task_id: &str) -> RepoTaskSpec {
        use crate::agent::repo::{HardnessProfile, HardnessTier};
        RepoTaskSpec {
            id: task_id.to_string(),
            repo: self.repo_root.clone(),
            kind: self.kind,
            issue: format!("synthesize: {}", self.nl_input),
            test_command: self.test_command.clone(),
            allowed_files: self.allowed_files.clone(),
            max_iterations: self.max_iterations,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::coding_intent::CodingIntent;

    #[test]
    fn code_task_spec_roundtrip_json() {
        let intent = CodingIntent::from_nl("add two numbers").expect("intent");
        let spec = CodeTaskSpec::from_nl(
            "/tmp/repo",
            "add two numbers",
            intent,
            "cargo test add",
            vec!["src/**".into()],
            3,
        );
        let json = serde_json::to_string(&spec).expect("serialize");
        let restored: CodeTaskSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.schema_version, SCHEMA_VERSION);
        assert_eq!(restored.nl_input, "add two numbers");
        let repo = restored.to_repo_task_spec("t1");
        assert!(repo.issue.contains("synthesize:"));
    }
}
