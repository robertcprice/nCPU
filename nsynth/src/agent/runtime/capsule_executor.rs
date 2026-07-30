//! Execute one content-bound capsule through the real nCPU process sandbox.
//!
//! This is the narrow production seam between canonical LinguaGenesis
//! comprehension lineage, nSynth artifacts, and nCPU execution. It does not own
//! knowledge or promote capabilities. It emits a verifier trace; a separate
//! admission policy decides whether that trace is strong enough to propose.

use super::{
    AgentRunBudget, ExecutionCapsule, ExecutionFailure, ExecutionFailureKind, ExecutionTrace,
    TraceError,
};
use crate::agent::provenance::{certify, Verdict};
use crate::benchmark::{
    generated_holdouts_with_source, Example as BenchmarkExample, Problem, Value,
};
use crate::execution::{
    Example, InputValue, Language, Sandbox, SandboxConfig, SandboxError, SandboxFailureKind,
    VerificationReport,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const SANDBOX_EXECUTION_CAPABILITY: &str = "sandbox.execute";

pub struct CapsuleExecutor;

impl CapsuleExecutor {
    pub fn execute(
        capsule: &ExecutionCapsule,
        evaluator: &Problem,
    ) -> Result<ExecutionTrace, TraceError> {
        capsule.validate_integrity().map_err(TraceError::Capsule)?;
        capsule
            .validate_evaluator(evaluator)
            .map_err(TraceError::Capsule)?;
        if evaluator.function_name() != capsule.artifact.function_name {
            return Err(TraceError::EvaluatorMismatch(
                "artifact function name differs from evaluator".into(),
            ));
        }

        let visible_examples = sandbox_examples(&evaluator.examples)?;
        if visible_examples != capsule.examples {
            return Err(TraceError::EvaluatorMismatch(
                "capsule examples differ from evaluator examples".into(),
            ));
        }

        let mut budget = capsule.task.budget.clone();
        if let Err(exhausted) = budget.record_attempt() {
            return ExecutionTrace::from_failure(
                capsule,
                ExecutionFailure::new(ExecutionFailureKind::BudgetExhausted, exhausted.reason),
                budget,
                vec![],
            );
        }
        if !capsule.policy.allows(SANDBOX_EXECUTION_CAPABILITY) {
            return ExecutionTrace::from_failure(
                capsule,
                ExecutionFailure::new(
                    ExecutionFailureKind::PolicyDenied,
                    "capsule policy does not allow sandbox execution",
                ),
                budget,
                vec![],
            );
        }
        if capsule.artifact.language != Language::Rust {
            return ExecutionTrace::from_failure(
                capsule,
                ExecutionFailure::new(
                    ExecutionFailureKind::UnsupportedInterface,
                    "typed function execution currently supports Rust artifacts only",
                ),
                budget,
                vec![],
            );
        }

        let started = Instant::now();
        let initial_wall_ms = budget.wall_ms_used;
        let sandbox = match Sandbox::with_config(SandboxConfig {
            timeout: Duration::from_millis(capsule.policy.timeout_ms),
            memory_limit: capsule.policy.memory_limit_bytes,
            max_output_size: capsule.policy.max_output_bytes,
            enable_isolation: true,
            working_directory: None,
            env_vars: HashMap::new(),
        }) {
            Ok(sandbox) => sandbox,
            Err(error) => {
                record_wall(&mut budget, initial_wall_ms, started);
                return ExecutionTrace::from_failure(
                    capsule,
                    ExecutionFailure::new(
                        ExecutionFailureKind::Sandbox,
                        format!("sandbox initialization failed: {error}"),
                    ),
                    budget,
                    vec![SANDBOX_EXECUTION_CAPABILITY.into()],
                );
            }
        };

        let visible_report = match sandbox.verify_rust_function(
            &capsule.artifact.source,
            &capsule.artifact.function_name,
            &capsule.examples,
        ) {
            Ok(report) => report,
            Err(error) => {
                record_wall(&mut budget, initial_wall_ms, started);
                return failure_trace(capsule, error, budget);
            }
        };
        record_wall(&mut budget, initial_wall_ms, started);
        if budget.wall_ms_used > budget.max_wall_ms {
            return ExecutionTrace::from_failure(
                capsule,
                ExecutionFailure::new(
                    ExecutionFailureKind::BudgetExhausted,
                    format!(
                        "wall clock budget exhausted ({}ms / {}ms)",
                        budget.wall_ms_used, budget.max_wall_ms
                    ),
                ),
                budget,
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
            );
        }
        if let Some(failure) = report_failure(&visible_report) {
            return ExecutionTrace::from_failure(
                capsule,
                failure,
                budget,
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
            );
        }
        if !visible_report.all_passed() {
            return ExecutionTrace::from_report(
                capsule,
                visible_report,
                None,
                None,
                budget,
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
            );
        }

        let (holdouts, _) = generated_holdouts_with_source(evaluator);
        if holdouts.is_empty() {
            return ExecutionTrace::from_failure(
                capsule,
                ExecutionFailure::new(
                    ExecutionFailureKind::Verification,
                    "evaluator supplied no non-visible holdout evidence",
                ),
                budget,
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
            );
        }
        let holdout_examples = sandbox_examples(&holdouts)?;
        let holdout_report = match sandbox.verify_rust_function(
            &capsule.artifact.source,
            &capsule.artifact.function_name,
            &holdout_examples,
        ) {
            Ok(report) => report,
            Err(error) => {
                record_wall(&mut budget, initial_wall_ms, started);
                return failure_trace(capsule, error, budget);
            }
        };
        record_wall(&mut budget, initial_wall_ms, started);
        if budget.wall_ms_used > budget.max_wall_ms {
            return ExecutionTrace::from_failure(
                capsule,
                ExecutionFailure::new(
                    ExecutionFailureKind::BudgetExhausted,
                    format!(
                        "wall clock budget exhausted ({}ms / {}ms)",
                        budget.wall_ms_used, budget.max_wall_ms
                    ),
                ),
                budget,
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
            );
        }
        if let Some(failure) = report_failure(&holdout_report) {
            return ExecutionTrace::from_failure(
                capsule,
                failure,
                budget,
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
            );
        }
        if !holdout_report.all_passed() {
            return ExecutionTrace::from_report(
                capsule,
                visible_report,
                Some(holdout_report),
                None,
                budget,
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
            );
        }

        let provenance = match certify(
            evaluator,
            &capsule.artifact.source,
            &capsule.artifact.synthesis_method,
        ) {
            Verdict::Verified(certificate) => certificate,
            Verdict::Refuted { reason } => {
                record_wall(&mut budget, initial_wall_ms, started);
                return ExecutionTrace::from_failure(
                    capsule,
                    ExecutionFailure::new(ExecutionFailureKind::Verification, reason),
                    budget,
                    vec![SANDBOX_EXECUTION_CAPABILITY.into()],
                );
            }
        };
        record_wall(&mut budget, initial_wall_ms, started);
        if budget.wall_ms_used > budget.max_wall_ms {
            return ExecutionTrace::from_failure(
                capsule,
                ExecutionFailure::new(
                    ExecutionFailureKind::BudgetExhausted,
                    format!(
                        "wall clock budget exhausted ({}ms / {}ms)",
                        budget.wall_ms_used, budget.max_wall_ms
                    ),
                ),
                budget,
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
            );
        }
        ExecutionTrace::from_report(
            capsule,
            visible_report,
            Some(holdout_report),
            Some(provenance),
            budget,
            vec![SANDBOX_EXECUTION_CAPABILITY.into()],
        )
    }
}

fn sandbox_examples(examples: &[BenchmarkExample]) -> Result<Vec<Example>, TraceError> {
    examples
        .iter()
        .map(|example| {
            Ok(Example {
                inputs: example
                    .inputs
                    .iter()
                    .map(sandbox_value)
                    .collect::<Result<Vec<_>, _>>()?,
                expected: sandbox_value(&example.expected)?,
            })
        })
        .collect()
}

fn sandbox_value(value: &Value) -> Result<InputValue, TraceError> {
    match value {
        Value::Int(value) => Ok(InputValue::Int(*value)),
        _ => Err(TraceError::UnsupportedEvaluatorValue),
    }
}

fn failure_trace(
    capsule: &ExecutionCapsule,
    error: SandboxError,
    budget: AgentRunBudget,
) -> Result<ExecutionTrace, TraceError> {
    ExecutionTrace::from_failure(
        capsule,
        ExecutionFailure::new(failure_kind(error.kind()), error.to_string()),
        budget,
        vec![SANDBOX_EXECUTION_CAPABILITY.into()],
    )
}

fn report_failure(report: &VerificationReport) -> Option<ExecutionFailure> {
    report.results.iter().find_map(|result| {
        let kind = result.failure_kind?;
        Some(ExecutionFailure::new(
            failure_kind(kind),
            result
                .error
                .clone()
                .unwrap_or_else(|| "sandbox execution failed".into()),
        ))
    })
}

fn failure_kind(kind: SandboxFailureKind) -> ExecutionFailureKind {
    match kind {
        SandboxFailureKind::Compilation => ExecutionFailureKind::Compilation,
        SandboxFailureKind::Timeout => ExecutionFailureKind::Timeout,
        SandboxFailureKind::OutputLimit => ExecutionFailureKind::OutputLimit,
        SandboxFailureKind::Security => ExecutionFailureKind::PolicyDenied,
        SandboxFailureKind::UnsupportedInterface => ExecutionFailureKind::UnsupportedInterface,
        SandboxFailureKind::MemoryLimit
        | SandboxFailureKind::Runtime
        | SandboxFailureKind::Signal
        | SandboxFailureKind::Io
        | SandboxFailureKind::RuntimeUnavailable => ExecutionFailureKind::Sandbox,
    }
}

fn record_wall(budget: &mut AgentRunBudget, initial_wall_ms: u64, started: Instant) {
    budget.wall_ms_used = initial_wall_ms.saturating_add(started.elapsed().as_millis() as u64);
}

#[cfg(test)]
mod tests {
    use super::super::{
        CapabilityAdmission, CodeTaskSpec, ExecutableArtifact, ExecutionPolicy,
        ExecutionTraceOutcome,
    };
    use super::*;
    use crate::agent::coding_intent::CodingIntent;
    use crate::benchmark::HoldoutSource;
    use crate::benchmark::{Example as BenchmarkExample, Value};
    use crate::linguigenesis_bridge::LinguigenesisBridge;
    use linguigenesis_core::capability_learning::{GraphId, SchemaVersion, SemanticGraph};
    use linguigenesis_core::entity::{Entity, EntityId, EntityType};
    use linguigenesis_core::registry::Registry;

    fn benchmark_example(input: i64, expected: i64) -> BenchmarkExample {
        BenchmarkExample {
            inputs: vec![Value::Int(input)],
            expected: Value::Int(expected),
        }
    }

    fn square_problem() -> Problem {
        Problem {
            name: "capsule_square_v0".into(),
            category: "arithmetic",
            description: "square a number",
            signature: "fn capsule_square_v0(a: i64) -> i64",
            examples: vec![
                benchmark_example(2, 4),
                benchmark_example(3, 9),
                benchmark_example(4, 16),
                benchmark_example(12, 144),
            ],
            reference_code: "fn capsule_square_v0(a: i64) -> i64 { return a * a; }",
            ..Default::default()
        }
    }

    fn capsule(problem: &Problem, allowed: bool) -> ExecutionCapsule {
        let task = CodeTaskSpec::from_nl(
            "/tmp/repo",
            "square a number",
            CodingIntent::from_nl("square a number").expect("intent"),
            "cargo test square",
            vec!["src/lib.rs".into()],
            2,
        );
        let policy = ExecutionPolicy::new(
            if allowed {
                vec![SANDBOX_EXECUTION_CAPABILITY.into()]
            } else {
                vec![]
            },
            2_000,
            128 * 1024 * 1024,
            4096,
        );
        ExecutionCapsule::new(
            task,
            ExecutableArtifact::new(
                "capsule_square_v0",
                "fn capsule_square_v0(a: i64) -> i64 { return a * a; }",
                Language::Rust,
                "test-synthesizer",
                vec![17 as EntityId],
            ),
            problem,
            sandbox_examples(&problem.examples).expect("examples"),
            policy,
        )
        .expect("capsule")
    }

    #[test]
    fn executes_exact_artifact_on_visible_and_generated_holdouts() {
        let problem = square_problem();
        let trace = CapsuleExecutor::execute(&capsule(&problem, true), &problem).expect("trace");
        assert_eq!(trace.outcome, ExecutionTraceOutcome::Verified);
        assert!(trace.verification.as_ref().is_some_and(|r| r.all_passed()));
        assert!(trace
            .generalization_verification
            .as_ref()
            .is_some_and(|r| r.all_passed() && r.total > 0));
        assert!(trace.admission_eligible(), "{trace:#?}");
    }

    #[test]
    fn verified_trace_admits_into_the_caller_owned_canonical_graph() {
        let problem = square_problem();
        let trace = CapsuleExecutor::execute(&capsule(&problem, true), &problem).expect("trace");
        let admission =
            CapabilityAdmission::from_verified_trace(&trace, 17, "square", "square_conformance")
                .expect("proposal");

        let registry = Registry::new();
        registry
            .add_entity(Entity::new(17, "square".into(), EntityType::Function))
            .expect("canonical entity");
        let mut graph = SemanticGraph::new(
            GraphId::new("canonical:ncpu-capability-test"),
            SchemaVersion::new("usg-0.1.0"),
        );
        let receipt = admission
            .admit_into_canonical_graph(&trace, &registry, &mut graph)
            .expect("canonical admission");
        assert!(receipt.graph_changed);
        assert!(graph
            .nodes
            .keys()
            .any(|node_id| node_id.as_str() == receipt.capability_node_id));

        let replay = admission
            .admit_into_canonical_graph(&trace, &registry, &mut graph)
            .expect("idempotent canonical admission");
        assert!(!replay.graph_changed);
    }

    #[test]
    fn deny_by_default_policy_prevents_execution() {
        let problem = square_problem();
        let trace = CapsuleExecutor::execute(&capsule(&problem, false), &problem).expect("trace");
        assert_eq!(trace.outcome, ExecutionTraceOutcome::PolicyDenied);
        assert!(!trace.admission_eligible());
    }

    #[test]
    fn substituted_evaluator_is_rejected_before_execution() {
        let original = square_problem();
        let capsule = capsule(&original, true);
        let mut substituted = original.clone();
        substituted.examples[0].expected = Value::Int(5);
        assert!(matches!(
            CapsuleExecutor::execute(&capsule, &substituted),
            Err(TraceError::Capsule(_))
        ));
    }

    fn refresh_capsule_digest(capsule: &mut ExecutionCapsule) {
        capsule.artifact.source_digest =
            super::super::ContentDigest::sha256(capsule.artifact.source.as_bytes());
        capsule.capsule_digest = capsule.recompute_digest().expect("capsule digest");
        capsule.validate_integrity().expect("valid capsule");
    }

    #[test]
    fn compilation_timeout_output_and_budget_fail_closed() {
        let problem = square_problem();

        let mut compilation = capsule(&problem, true);
        compilation.artifact.source =
            "fn capsule_square_v0(a: i64) -> i64 { return missing + a; }".into();
        refresh_capsule_digest(&mut compilation);
        let compilation_trace =
            CapsuleExecutor::execute(&compilation, &problem).expect("failure trace");
        assert_eq!(
            compilation_trace.failure.as_ref().expect("failure").kind,
            ExecutionFailureKind::Compilation
        );
        assert!(!compilation_trace.admission_eligible());

        let mut timeout = capsule(&problem, true);
        timeout.artifact.source = "fn capsule_square_v0(_a: i64) -> i64 { loop {} }".into();
        timeout.policy.timeout_ms = 200;
        refresh_capsule_digest(&mut timeout);
        let timeout_trace = CapsuleExecutor::execute(&timeout, &problem).expect("failure trace");
        assert_eq!(
            timeout_trace.failure.as_ref().expect("failure").kind,
            ExecutionFailureKind::Timeout
        );
        assert!(!timeout_trace.admission_eligible());

        let mut output = capsule(&problem, true);
        output.artifact.source = r#"fn capsule_square_v0(a: i64) -> i64 {
            print!("12345678901234567890");
            a * a
        }"#
        .into();
        output.policy.max_output_bytes = 8;
        refresh_capsule_digest(&mut output);
        let output_trace = CapsuleExecutor::execute(&output, &problem).expect("failure trace");
        assert_eq!(
            output_trace.failure.as_ref().expect("failure").kind,
            ExecutionFailureKind::OutputLimit
        );
        assert!(!output_trace.admission_eligible());

        let mut exhausted = capsule(&problem, true);
        exhausted.task.budget.max_attempts = 0;
        refresh_capsule_digest(&mut exhausted);
        let exhausted_trace =
            CapsuleExecutor::execute(&exhausted, &problem).expect("failure trace");
        assert_eq!(
            exhausted_trace.outcome,
            ExecutionTraceOutcome::BudgetExhausted
        );
        assert_eq!(
            exhausted_trace.failure.as_ref().expect("failure").kind,
            ExecutionFailureKind::BudgetExhausted
        );
        assert!(!exhausted_trace.admission_eligible());
    }

    #[test]
    fn canonical_requirement_synthesizes_executes_and_retains_lineage() {
        let request = "add two numbers";
        let bridge = LinguigenesisBridge::new();
        let requirement = bridge.nl_to_requirement(request).expect("requirement");
        let problem = bridge
            .problem_from_requirement(&requirement, None)
            .expect("evaluator");
        let synthesized = bridge
            .synthesize_from_requirement(&requirement, None)
            .expect("synthesis");
        assert!(synthesized.success, "{synthesized:#?}");

        let task = CodeTaskSpec::from_nl(
            "/tmp/repo",
            request,
            CodingIntent::from_nl(request).expect("intent"),
            "cargo test add",
            vec!["src/lib.rs".into()],
            2,
        );
        let capsule = ExecutionCapsule::new(
            task,
            ExecutableArtifact::new(
                &requirement.function_name,
                synthesized.code,
                Language::Rust,
                synthesized.method,
                requirement.evidence_entity_ids.clone(),
            ),
            &problem,
            sandbox_examples(&problem.examples).expect("sandbox examples"),
            ExecutionPolicy::new(
                vec![SANDBOX_EXECUTION_CAPABILITY.into()],
                5_000,
                128 * 1024 * 1024,
                4096,
            ),
        )
        .expect("capsule");
        let trace = CapsuleExecutor::execute(&capsule, &problem).expect("execution trace");

        assert_eq!(trace.outcome, ExecutionTraceOutcome::Verified);
        assert_eq!(
            trace.evidence_entity_ids,
            capsule.artifact.evidence_entity_ids
        );
        assert_eq!(
            trace
                .provenance
                .as_ref()
                .expect("provenance")
                .holdout_source,
            HoldoutSource::HandFallback
        );
        assert!(
            !trace.admission_eligible(),
            "registry-reserved examples are honest hand-fallback evidence, not generated oracle holdouts"
        );
    }
}
