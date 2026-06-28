//! Gate G6 — durable agency MVP checks (Packages I/J).

#[cfg(test)]
mod tests {
    use crate::agent::capability_registry::{CapabilityRegistry, CapabilityStatus};
    use crate::agent::session_persistence::{load_session_snapshot, session_path};

    #[test]
    fn g6_durable_session_capability_is_experimental_or_implemented() {
        let reg = CapabilityRegistry::package_b_native_runtime();
        let cap = reg.get("durable_session_resume").expect("durable_session_resume");
        assert!(matches!(
            cap.status,
            CapabilityStatus::Implemented | CapabilityStatus::Experimental
        ));
    }

    #[test]
    fn g6_universal_query_session_wired() {
        let reg = CapabilityRegistry::package_b_native_runtime();
        let cap = reg.get("universal_query_session").expect("universal_query_session");
        assert_eq!(cap.status, CapabilityStatus::Experimental);
        assert!(cap.conformance_test.is_some());
    }

    #[test]
    fn g6_session_snapshot_schema_has_pending_slot() {
        use crate::agent::session_persistence::SessionSnapshot;
        let snap = SessionSnapshot::default();
        assert_eq!(snap.schema_version, crate::agent::runtime::SCHEMA_VERSION);
        assert!(snap.pending.is_none());
    }
}
