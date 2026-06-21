//! Package B gate conformance suite (Phase 1 runtime contracts).

#[cfg(test)]
mod gate {
    use crate::agent::capability_registry::{CapabilityRegistry, CapabilityStatus};
    use crate::agent::coding_intent::CodingIntent;
    use crate::agent::runtime::{legal_transition, AgentRunStatus, SCHEMA_VERSION};

    #[test]
    fn package_b_schema_version_is_one() {
        assert_eq!(SCHEMA_VERSION, 1);
    }

    #[test]
    fn package_b_capability_registry_covers_runtime_contracts() {
        let reg = CapabilityRegistry::package_b_native_runtime();
        for name in [
            "coding_intent_contract",
            "agent_run_state_machine",
            "agent_run_lifecycle",
            "agent_run_persistence",
            "code_task_spec",
            "repo_run_supervisor",
            "repo_workflow_runner",
        ] {
            let cap = reg.get(name).expect(name);
            assert_eq!(
                cap.status,
                CapabilityStatus::Implemented,
                "{name} not implemented"
            );
        }
    }

    #[test]
    fn package_b_nl_intent_to_problem() {
        let intent = CodingIntent::from_nl("add two numbers").expect("intent");
        let problem = intent.to_problem().expect("problem");
        assert!(!problem.examples.is_empty());
    }

    #[test]
    fn package_b_state_machine_has_terminal_guard() {
        assert!(!legal_transition(
            AgentRunStatus::Succeeded,
            AgentRunStatus::Executing
        ));
        assert!(!legal_transition(
            AgentRunStatus::Failed,
            AgentRunStatus::Understanding
        ));
    }
}
