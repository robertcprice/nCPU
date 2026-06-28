//! Capability registry — honest labels for agent features (Package B).

use serde::{Deserialize, Serialize};

/// Capability maturity label (roadmap §0.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityStatus {
    Absent,
    Scaffold,
    Experimental,
    Implemented,
    Verified,
    Blocked,
}

/// Named agent capability with evidence pointer and conformance test hook.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityRecord {
    pub name: String,
    pub status: CapabilityStatus,
    pub evidence: String,
    /// Executable conformance test name (cargo test filter).
    pub conformance_test: Option<String>,
}

/// Registry of known agent capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityRegistry {
    pub capabilities: Vec<CapabilityRecord>,
}

impl CapabilityRegistry {
    pub fn package_b_native_runtime() -> Self {
        Self {
            capabilities: vec![
                CapabilityRecord {
                    name: "nl_to_requirement".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "linguigenesis_bridge + coding_registry.json".into(),
                    conformance_test: Some("linguigenesis_bridge".into()),
                },
                CapabilityRecord {
                    name: "nl_clarification_dialogue".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "coding_dialogue + BridgeError::ClarificationNeeded".into(),
                    conformance_test: Some("coding_dialogue".into()),
                },
                CapabilityRecord {
                    name: "coding_intent_contract".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/coding_intent.rs".into(),
                    conformance_test: Some("coding_intent_from_add_nl".into()),
                },
                CapabilityRecord {
                    name: "agent_run_state_machine".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/runtime/state_machine.rs canonical transitions".into(),
                    conformance_test: Some("legal_nl_synthesis_path".into()),
                },
                CapabilityRecord {
                    name: "agent_run_lifecycle".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/agent_run.rs comprehend→synthesize".into(),
                    conformance_test: Some("agent_run_comprehend_and_synthesize_add".into()),
                },
                CapabilityRecord {
                    name: "agent_run_persistence".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/agent_run_persistence.rs JSON snapshot v1".into(),
                    conformance_test: Some("agent_run_save_load_roundtrip".into()),
                },
                CapabilityRecord {
                    name: "code_task_spec".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/runtime/code_task_spec.rs".into(),
                    conformance_test: Some("code_task_spec_roundtrip_json".into()),
                },
                CapabilityRecord {
                    name: "repo_run_supervisor".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/repo/run_supervisor.rs durable NL repair".into(),
                    conformance_test: Some("supervisor_executes_nl_add_task".into()),
                },
                CapabilityRecord {
                    name: "repo_workflow_runner".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/repo/workflow_runner.rs RepoAgent batch + report JSON; 17/17 nl_fixture nightly".into(),
                    conformance_test: Some("workflow_runner_executes_nl_fixture_ci_subset".into()),
                },
                CapabilityRecord {
                    name: "repo_agent_closed_loop".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/repo/repo_agent.rs oracle-gated index→repair loop".into(),
                    conformance_test: Some("repo_agent_runs_nl_fixture_with_cargo_oracle".into()),
                },
                CapabilityRecord {
                    name: "transactional_edits".into(),
                    status: CapabilityStatus::Experimental,
                    evidence: "agent/edit/transaction.rs + worktree.rs isolated promote/discard"
                        .into(),
                    conformance_test: Some(
                        "isolated_session_promote_applies_repair_to_parent".into(),
                    ),
                },
                CapabilityRecord {
                    name: "durable_session_resume".into(),
                    status: CapabilityStatus::Experimental,
                    evidence: "agent/session_persistence.rs + CodingAgentSession::load/persist"
                        .into(),
                    conformance_test: Some("session_persist_and_resume".into()),
                },
                CapabilityRecord {
                    name: "universal_query_session".into(),
                    status: CapabilityStatus::Experimental,
                    evidence: "agent/session.rs KVRM workflow routing + full secure tools"
                        .into(),
                    conformance_test: Some("session_routes_synthesis_query".into()),
                },
                CapabilityRecord {
                    name: "secure_tool_runtime".into(),
                    status: CapabilityStatus::Experimental,
                    evidence: "agent/tools/secure_runtime.rs deny-by-default + verification allowlist"
                        .into(),
                    conformance_test: Some("deny_by_default_rejects_unlisted_tool".into()),
                },
                CapabilityRecord {
                    name: "repo_index_retrieval".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/repository retrieval benchmark + localization_confidence".into(),
                    conformance_test: Some("retrieval_benchmark_hits_expected_paths".into()),
                },
                CapabilityRecord {
                    name: "nl_synthesis_repair_proposer".into(),
                    status: CapabilityStatus::Implemented,
                    evidence: "agent/synthesis_proposer.rs real synthesis primary; 17/17 nl_fixture workflow nightly".into(),
                    conformance_test: Some("real_synthesis_repairs_divide_result_idiom".into()),
                },
                CapabilityRecord {
                    name: "legacy_nl_llm_module".into(),
                    status: CapabilityStatus::Scaffold,
                    evidence: "nl/mod.rs quarantined; delegates to linguigenesis_bridge".into(),
                    conformance_test: None,
                },
            ],
        }
    }

    pub fn native_nl_synthesis() -> Self {
        Self::package_b_native_runtime()
    }

    pub fn get(&self, name: &str) -> Option<&CapabilityRecord> {
        self.capabilities.iter().find(|c| c.name == name)
    }

    pub fn implemented_without_conformance_test(&self) -> Vec<&CapabilityRecord> {
        self.capabilities
            .iter()
            .filter(|c| c.status == CapabilityStatus::Implemented && c.conformance_test.is_none())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_nl_capabilities_implemented() {
        let reg = CapabilityRegistry::native_nl_synthesis();
        let nl = reg.get("nl_to_requirement").expect("nl");
        assert_eq!(nl.status, CapabilityStatus::Implemented);
        assert!(nl.conformance_test.is_some());
    }

    #[test]
    fn implemented_capabilities_link_conformance_tests() {
        let reg = CapabilityRegistry::package_b_native_runtime();
        let missing = reg.implemented_without_conformance_test();
        assert!(
            missing.is_empty(),
            "implemented capabilities missing conformance tests: {:?}",
            missing
        );
    }
}
