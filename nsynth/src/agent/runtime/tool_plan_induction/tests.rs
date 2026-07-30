use super::*;
use crate::agent::repo::GuardrailPolicy;
use crate::agent::runtime::{
    AgentRunBudget, AgentRunId, ToolArgumentBinding, ToolChainExecutor, ToolChainOutcome,
    ToolExecutionPolicy,
};
use crate::agent::tools::SecureToolRuntime;
use linguigenesis_core::capability_learning::{
    admit_verified_capability, CapabilitySlot, GraphId, SchemaVersion, SemanticGraph,
    VerifiedCapabilityAdmissionRequest, CAPABILITY_ADMISSION_SCHEMA_VERSION,
};
use linguigenesis_core::entity::{Entity, EntityId, EntityType, RelationType};
use linguigenesis_core::registry::Registry;
use std::fs;
use std::path::PathBuf;

const FS_READ: EntityId = 1;
const FS_WRITE: EntityId = 2;
const FS_LIST: EntityId = 3;
const PATH_KIND: EntityId = 10;
const TEXT_KIND: EntityId = 11;
const UTF8_TEXT_KIND: EntityId = 12;
const BYTE_COUNT_KIND: EntityId = 13;
const ACK_KIND: EntityId = 14;

fn temp_root(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "nsynth-induction-{label}-{}-{}",
        std::process::id(),
        AgentRunId::new().0
    ))
}

fn canonical_registry() -> Registry {
    let registry = Registry::new();
    for (id, lemma, entity_type) in [
        (FS_READ, "fs.read", EntityType::Function),
        (FS_WRITE, "fs.write", EntityType::Function),
        (FS_LIST, "fs.list", EntityType::Function),
        (PATH_KIND, "sandbox_path", EntityType::Type),
        (TEXT_KIND, "text", EntityType::Type),
        (UTF8_TEXT_KIND, "utf8_text", EntityType::Type),
        (BYTE_COUNT_KIND, "byte_count", EntityType::Type),
        (ACK_KIND, "write_ack", EntityType::Type),
    ] {
        registry
            .add_entity(Entity::new(id, lemma.into(), entity_type))
            .expect("canonical entity");
    }
    registry
        .link_lemma_relation("utf8_text", RelationType::Hypernym, "text")
        .expect("utf8_text is-a text");
    registry
}

fn admit(
    registry: &Registry,
    graph: &mut SemanticGraph,
    capability_entity: EntityId,
    capability_name: &str,
    digest_seed: char,
    slots: Vec<CapabilitySlot>,
) {
    let mut evidence = vec![capability_entity];
    for slot in &slots {
        evidence.push(slot.data_kind_entity_id);
    }
    evidence.sort_unstable();
    evidence.dedup();
    let request = VerifiedCapabilityAdmissionRequest {
        schema_version: CAPABILITY_ADMISSION_SCHEMA_VERSION,
        canonical_entity_id: capability_entity,
        run_id: format!("run-{capability_name}"),
        verification_trace_digest: digest_seed.to_string().repeat(64),
        artifact_digest: "b".repeat(64),
        evidence_entity_ids: evidence,
        capability_name: capability_name.into(),
        conformance_test: format!("{capability_name}_conformance"),
        proposal_digest: digest_seed.to_string().repeat(63) + "e",
        slots,
    };
    admit_verified_capability(registry, graph, &request).expect("canonical admission");
}

/// fs.read and fs.write with real contracts; fs.list admitted without one.
fn canonical_graph(registry: &Registry) -> SemanticGraph {
    let mut graph = SemanticGraph::new(
        GraphId::new("canonical:induction-test"),
        SchemaVersion::new("usg-0.1.0"),
    );
    admit(
        registry,
        &mut graph,
        FS_READ,
        "fs.read",
        'a',
        vec![
            CapabilitySlot::consumed("path", PATH_KIND),
            CapabilitySlot::produced_content(UTF8_TEXT_KIND),
        ],
    );
    admit(
        registry,
        &mut graph,
        FS_WRITE,
        "fs.write",
        'c',
        vec![
            CapabilitySlot::consumed("content", TEXT_KIND),
            CapabilitySlot::consumed("path", PATH_KIND),
            CapabilitySlot::produced_content(ACK_KIND),
            CapabilitySlot::produced_metadata("bytes", BYTE_COUNT_KIND),
        ],
    );
    admit(registry, &mut graph, FS_LIST, "fs.list", 'd', Vec::new());
    graph
}

fn request(runtime: &SecureToolRuntime, goal: &str) -> StepGraphInductionRequest {
    StepGraphInductionRequest {
        run_id: AgentRunId::new(),
        evidence_entity_ids: vec![FS_READ, FS_WRITE],
        policy: ToolExecutionPolicy::from_secure_runtime(runtime, 2_000, 4096),
        goal_capability: goal.into(),
        seeds: vec![
            InductionSeed::new("seed.txt", PATH_KIND),
            InductionSeed::new("copy.txt", PATH_KIND),
        ],
        max_steps: 3,
        max_plans: 32,
        max_explored_chains: 512,
    }
}

#[test]
fn canonical_contracts_induce_a_causal_chain_that_really_executes() {
    let root = temp_root("execute");
    fs::create_dir_all(&root).expect("root");
    fs::write(root.join("seed.txt"), "induced λ\n").expect("seed");
    let runtime = SecureToolRuntime::for_repo_repair(&root, GuardrailPolicy::default());
    let registry = canonical_registry();
    let graph = canonical_graph(&registry);

    let induced = ToolStepGraphInducer::induce(&graph, &registry, &request(&runtime, "fs.write"))
        .expect("induction");

    // fs.write needs text it cannot get from a path seed, so every induced plan
    // must read first: the data flow, not a phrase, forces the chain.
    assert!(!induced.plans.is_empty());
    assert!(induced.plans.iter().all(|plan| plan.steps.len() == 2));
    assert!(induced.plans.iter().all(|plan| {
        plan.steps[0].capability() == "fs.read"
            && plan.steps[1].capability() == "fs.write"
            && plan.steps[1].arguments.get("content")
                == Some(&ToolArgumentBinding::prior_content("step-1-fs-read"))
    }));
    assert_eq!(
        induced.composable_capabilities,
        vec!["fs.read".to_string(), "fs.write".to_string()]
    );
    assert_eq!(
        induced.uncomposable_capabilities,
        vec![UncomposableCapability {
            capability_name: "fs.list".into(),
            reason: UncomposableReason::NoDeclaredContract,
        }]
    );

    let chosen = induced
        .plans
        .iter()
        .find(|plan| {
            plan.steps[0].arguments.get("path") == Some(&ToolArgumentBinding::literal("seed.txt"))
                && plan.steps[1].arguments.get("path")
                    == Some(&ToolArgumentBinding::literal("copy.txt"))
        })
        .expect("read seed then write copy was induced");

    let chain =
        ToolChainExecutor::execute(chosen, &runtime, AgentRunBudget::default()).expect("chain");
    assert_eq!(chain.outcome, ToolChainOutcome::Succeeded);
    assert_eq!(
        fs::read_to_string(root.join("copy.txt")).expect("copy"),
        "induced λ\n"
    );
    chain.validate_against_plan(chosen).expect("integrity");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn induction_refuses_goals_outside_policy_or_without_a_contract() {
    let root = temp_root("refuse");
    fs::create_dir_all(&root).expect("root");
    let registry = canonical_registry();
    let graph = canonical_graph(&registry);

    let read_only = SecureToolRuntime::deny_by_default(&root);
    let mut restricted = request(&read_only, "fs.write");
    restricted.policy = ToolExecutionPolicy::new(vec!["fs.read".into()], 1_000, 1024);
    assert_eq!(
        ToolStepGraphInducer::induce(&graph, &registry, &restricted),
        Err(InductionError::GoalNotComposable("fs.write".into()))
    );

    let runtime = SecureToolRuntime::for_repo_repair(&root, GuardrailPolicy::default());
    let no_contract = request(&runtime, "fs.list");
    assert_eq!(
        ToolStepGraphInducer::induce(&graph, &registry, &no_contract),
        Err(InductionError::GoalNotComposable("fs.list".into()))
    );

    let unknown = request(&runtime, "http.get");
    assert_eq!(
        ToolStepGraphInducer::induce(&graph, &registry, &unknown),
        Err(InductionError::GoalNotComposable("http.get".into()))
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn induction_needs_a_canonical_relation_and_respects_its_limits() {
    let root = temp_root("relations");
    fs::create_dir_all(&root).expect("root");
    let runtime = SecureToolRuntime::for_repo_repair(&root, GuardrailPolicy::default());

    // Same contracts, but without the utf8_text is-a text relation the read
    // output no longer satisfies the write input, so nothing composes.
    let unrelated = Registry::new();
    for (id, lemma, entity_type) in [
        (FS_READ, "fs.read", EntityType::Function),
        (FS_WRITE, "fs.write", EntityType::Function),
        (FS_LIST, "fs.list", EntityType::Function),
        (PATH_KIND, "sandbox_path", EntityType::Type),
        (TEXT_KIND, "text", EntityType::Type),
        (UTF8_TEXT_KIND, "utf8_text", EntityType::Type),
        (BYTE_COUNT_KIND, "byte_count", EntityType::Type),
        (ACK_KIND, "write_ack", EntityType::Type),
    ] {
        unrelated
            .add_entity(Entity::new(id, lemma.into(), entity_type))
            .expect("entity");
    }
    let graph = canonical_graph(&unrelated);
    let induced = ToolStepGraphInducer::induce(&graph, &unrelated, &request(&runtime, "fs.write"))
        .expect("induction");
    assert!(
        induced.plans.is_empty(),
        "no canonical relation, no induced data flow: {:?}",
        induced.plans
    );

    // Limits are honest: a one-plan ceiling reports truncation.
    let registry = canonical_registry();
    let related = canonical_graph(&registry);
    let mut capped = request(&runtime, "fs.write");
    capped.max_plans = 1;
    let capped_result =
        ToolStepGraphInducer::induce(&related, &registry, &capped).expect("capped induction");
    assert_eq!(capped_result.plans.len(), 1);
    assert!(capped_result.search_truncated);

    let mut invalid = request(&runtime, "fs.write");
    invalid.max_steps = 0;
    assert_eq!(
        ToolStepGraphInducer::induce(&related, &registry, &invalid),
        Err(InductionError::InvalidLimit("max_steps"))
    );
    let _ = fs::remove_dir_all(root);
}
