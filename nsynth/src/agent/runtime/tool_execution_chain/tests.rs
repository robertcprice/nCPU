use super::*;
use crate::agent::repo::GuardrailPolicy;
use crate::agent::runtime::{AgentRunBudget, AgentRunId};
use crate::agent::tools::SecureToolRuntime;
use std::fs;
use std::path::PathBuf;

fn temp_root(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "nsynth-tool-chain-{label}-{}-{}",
        std::process::id(),
        AgentRunId::new().0
    ))
}

fn fs_plan(runtime: &SecureToolRuntime) -> ToolExecutionPlan {
    ToolExecutionPlan::new(
        AgentRunId::new(),
        vec![9, 4, 9],
        ToolExecutionPolicy::from_secure_runtime(runtime, 2_000, 4096),
        vec![
            ToolStepSpec::new("read-seed", "fs", "read")
                .argument("path", ToolArgumentBinding::literal("seed.txt")),
            ToolStepSpec::new("copy-content", "fs", "write")
                .argument("path", ToolArgumentBinding::literal("copy.txt"))
                .argument("content", ToolArgumentBinding::prior_content("read-seed")),
            ToolStepSpec::new("record-size", "fs", "write")
                .argument("path", ToolArgumentBinding::literal("size.txt"))
                .argument(
                    "content",
                    ToolArgumentBinding::prior_metadata("copy-content", "bytes"),
                ),
            ToolStepSpec::new("read-copy", "fs", "read")
                .argument("path", ToolArgumentBinding::literal("copy.txt")),
        ],
    )
    .expect("plan")
}

#[test]
fn prior_content_and_metadata_drive_real_later_tool_steps() {
    let root = temp_root("success");
    fs::create_dir_all(&root).expect("root");
    fs::write(root.join("seed.txt"), "emergent λ\n").expect("seed");
    let runtime = SecureToolRuntime::for_repo_repair(&root, GuardrailPolicy::default());
    let plan = fs_plan(&runtime);

    let chain =
        ToolChainExecutor::execute(&plan, &runtime, AgentRunBudget::default()).expect("chain");
    assert_eq!(chain.outcome, ToolChainOutcome::Succeeded);
    assert_eq!(chain.observations.len(), 4);
    assert_eq!(
        fs::read_to_string(root.join("copy.txt")).unwrap(),
        "emergent λ\n"
    );
    assert_eq!(
        fs::read_to_string(root.join("size.txt")).unwrap(),
        "emergent λ\n".len().to_string()
    );
    assert_eq!(
        chain.observations[3].output().expect("read output").content,
        "emergent λ\n"
    );
    assert_eq!(plan.evidence_entity_ids, vec![4, 9]);
    assert_eq!(
        chain.used_capabilities,
        vec!["fs.read".to_string(), "fs.write".to_string()]
    );
    chain.validate_against_plan(&plan).expect("integrity");

    let json = serde_json::to_string(&chain).expect("serialize");
    let restored: ToolExecutionChain = serde_json::from_str(&json).expect("deserialize");
    restored.validate_against_plan(&plan).expect("restored");

    let mut tampered = restored;
    match &mut tampered.observations[0].result {
        ToolStepResult::Succeeded(output) => output.content.push_str("tampered"),
        ToolStepResult::Failed(_) => panic!("fixture succeeded"),
    }
    assert!(matches!(
        tampered.validate_against_plan(&plan),
        Err(ToolChainError::DigestMismatch)
    ));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn plan_refuses_future_dependencies_and_capabilities_outside_policy() {
    let policy = ToolExecutionPolicy::new(vec!["fs.read".into()], 1_000, 1024);
    let future_dependency = ToolExecutionPlan::new(
        AgentRunId::new(),
        vec![1],
        policy.clone(),
        vec![
            ToolStepSpec::new("first", "fs", "read")
                .argument("path", ToolArgumentBinding::prior_content("second")),
            ToolStepSpec::new("second", "fs", "read")
                .argument("path", ToolArgumentBinding::literal("x")),
        ],
    );
    assert!(matches!(
        future_dependency,
        Err(ToolPlanError::InvalidDependency { .. })
    ));

    let outside_policy = ToolExecutionPlan::new(
        AgentRunId::new(),
        vec![1],
        policy,
        vec![ToolStepSpec::new("write", "fs", "write")
            .argument("path", ToolArgumentBinding::literal("x"))
            .argument("content", ToolArgumentBinding::literal("x"))],
    );
    assert!(matches!(
        outside_policy,
        Err(ToolPlanError::CapabilityOutsidePolicy(_))
    ));
}

#[test]
fn runtime_denial_and_attempt_exhaustion_stop_the_chain() {
    let root = temp_root("denial");
    fs::create_dir_all(&root).expect("root");
    fs::write(root.join("seed.txt"), "seed").expect("seed");
    let permitted = SecureToolRuntime::for_repo_repair(&root, GuardrailPolicy::default());
    let plan = fs_plan(&permitted);

    let denied = SecureToolRuntime::deny_by_default(&root);
    let denied_chain = ToolChainExecutor::execute(&plan, &denied, AgentRunBudget::default())
        .expect("denied chain");
    assert_eq!(
        denied_chain.outcome,
        ToolChainOutcome::Failed {
            step_id: "read-seed".into()
        }
    );
    assert!(matches!(
        denied_chain.observations[0].result,
        ToolStepResult::Failed(ObservedToolFailure {
            kind: ObservedToolFailureKind::PermissionDenied,
            ..
        })
    ));
    assert!(!root.join("copy.txt").exists());

    let budget = AgentRunBudget {
        max_attempts: 2,
        ..AgentRunBudget::default()
    };
    let budget_chain = ToolChainExecutor::execute(&plan, &permitted, budget).expect("budget chain");
    assert_eq!(
        budget_chain.outcome,
        ToolChainOutcome::BudgetExhausted {
            next_step_id: "record-size".into()
        }
    );
    assert_eq!(budget_chain.observations.len(), 2);
    assert_eq!(budget_chain.budget_after.attempts_used, 2);
    assert!(root.join("copy.txt").exists());
    assert!(!root.join("size.txt").exists());
    budget_chain
        .validate_against_plan(&plan)
        .expect("budget integrity");
    let _ = fs::remove_dir_all(root);
}
