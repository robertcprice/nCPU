//! Gate G5 — closed repair loop MVP sign-off checks (Package H).

#[cfg(test)]
mod tests {
    use crate::agent::capability_registry::{CapabilityRegistry, CapabilityStatus};
    use crate::agent::repo::benchmark::{
        nl_synthesis_fixture_ci_subset, nl_synthesis_fixture_suite,
    };

    #[test]
    fn g5_nl_fixture_corpus_size() {
        assert_eq!(nl_synthesis_fixture_suite().len(), 17);
        assert_eq!(nl_synthesis_fixture_ci_subset().len(), 3);
    }

    #[test]
    fn g5_repair_capabilities_are_implemented_or_experimental() {
        let reg = CapabilityRegistry::package_b_native_runtime();
        for name in [
            "repo_agent_closed_loop",
            "repo_workflow_runner",
            "nl_synthesis_repair_proposer",
        ] {
            let cap = reg.get(name).expect(name);
            assert!(
                matches!(
                    cap.status,
                    CapabilityStatus::Implemented | CapabilityStatus::Experimental
                ),
                "{name} status {:?}",
                cap.status
            );
        }
    }

    #[test]
    fn g5_real_synthesis_proposer_is_implemented() {
        let reg = CapabilityRegistry::package_b_native_runtime();
        let cap = reg
            .get("nl_synthesis_repair_proposer")
            .expect("proposer");
        assert_eq!(cap.status, CapabilityStatus::Implemented);
        assert!(cap.conformance_test.is_some());
    }
}
