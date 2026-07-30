//! Typed, content-addressed input to bounded capability execution.
//!
//! A capsule binds the already-grounded [`CodeTaskSpec`], synthesized artifact,
//! canonical LinguaGenesis evidence lineage, executable examples, and the exact
//! policy envelope. It is a transport contract, not another knowledge store.

use super::{AgentRunId, CodeTaskSpec, ContentDigest, SCHEMA_VERSION};
use crate::agent::tools::SecureToolRuntime;
use crate::benchmark::{FunctionDef, Problem};
use crate::execution::{Example, Language};
use linguigenesis_core::entity::EntityId;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A synthesized artifact plus its canonical comprehension lineage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableArtifact {
    pub schema_version: u32,
    pub function_name: String,
    pub source: String,
    pub language: Language,
    pub synthesis_method: String,
    pub evidence_entity_ids: Vec<EntityId>,
    pub source_digest: ContentDigest,
}

impl ExecutableArtifact {
    pub fn new(
        function_name: impl Into<String>,
        source: impl Into<String>,
        language: Language,
        synthesis_method: impl Into<String>,
        mut evidence_entity_ids: Vec<EntityId>,
    ) -> Self {
        evidence_entity_ids.sort_unstable();
        evidence_entity_ids.dedup();
        let source = source.into();
        let source_digest = ContentDigest::sha256(source.as_bytes());
        Self {
            schema_version: SCHEMA_VERSION,
            function_name: function_name.into(),
            source,
            language,
            synthesis_method: synthesis_method.into(),
            evidence_entity_ids,
            source_digest,
        }
    }

    pub fn validate_integrity(&self) -> Result<(), CapsuleError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(CapsuleError::SchemaVersion {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.function_name.trim().is_empty() {
            return Err(CapsuleError::EmptyField("artifact.function_name"));
        }
        if self.source.trim().is_empty() {
            return Err(CapsuleError::EmptyField("artifact.source"));
        }
        if self.synthesis_method.trim().is_empty() {
            return Err(CapsuleError::EmptyField("artifact.synthesis_method"));
        }
        if !strictly_sorted_unique(&self.evidence_entity_ids) {
            return Err(CapsuleError::NonCanonical("artifact.evidence_entity_ids"));
        }
        if !self.source_digest.verifies(self.source.as_bytes()) {
            return Err(CapsuleError::DigestMismatch("artifact.source"));
        }
        Ok(())
    }
}

/// Resource and tool boundary captured before execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    pub allowed_capabilities: Vec<String>,
    pub timeout_ms: u64,
    pub memory_limit_bytes: usize,
    pub max_output_bytes: usize,
}

impl ExecutionPolicy {
    pub fn new(
        mut allowed_capabilities: Vec<String>,
        timeout_ms: u64,
        memory_limit_bytes: usize,
        max_output_bytes: usize,
    ) -> Self {
        allowed_capabilities.sort();
        allowed_capabilities.dedup();
        Self {
            allowed_capabilities,
            timeout_ms,
            memory_limit_bytes,
            max_output_bytes,
        }
    }

    /// Snapshot the actual deny-by-default tool allowlist into the capsule.
    pub fn from_secure_runtime(
        runtime: &SecureToolRuntime,
        timeout_ms: u64,
        memory_limit_bytes: usize,
        max_output_bytes: usize,
    ) -> Self {
        Self::new(
            runtime.allowed_capabilities(),
            timeout_ms,
            memory_limit_bytes,
            max_output_bytes,
        )
    }

    pub fn allows(&self, capability: &str) -> bool {
        self.allowed_capabilities
            .binary_search_by(|candidate| candidate.as_str().cmp(capability))
            .is_ok()
    }

    fn validate(&self) -> Result<(), CapsuleError> {
        if self.timeout_ms == 0 {
            return Err(CapsuleError::InvalidLimit("policy.timeout_ms"));
        }
        if self.memory_limit_bytes == 0 {
            return Err(CapsuleError::InvalidLimit("policy.memory_limit_bytes"));
        }
        if self.max_output_bytes == 0 {
            return Err(CapsuleError::InvalidLimit("policy.max_output_bytes"));
        }
        if !strictly_sorted_unique(&self.allowed_capabilities)
            || self
                .allowed_capabilities
                .iter()
                .any(|capability| capability.trim().is_empty())
        {
            return Err(CapsuleError::NonCanonical("policy.allowed_capabilities"));
        }
        Ok(())
    }
}

/// Immutable execution request whose digest binds every decision-bearing field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCapsule {
    pub schema_version: u32,
    pub run_id: AgentRunId,
    pub task: CodeTaskSpec,
    pub artifact: ExecutableArtifact,
    pub evaluator_digest: ContentDigest,
    pub examples: Vec<Example>,
    pub policy: ExecutionPolicy,
    pub capsule_digest: ContentDigest,
}

#[derive(Serialize)]
struct CapsulePayload<'a> {
    schema_version: u32,
    run_id: &'a AgentRunId,
    task: &'a CodeTaskSpec,
    artifact: &'a ExecutableArtifact,
    evaluator_digest: &'a ContentDigest,
    examples: &'a [Example],
    policy: &'a ExecutionPolicy,
}

#[derive(Serialize)]
struct EvaluatorPayload<'a> {
    name: &'a str,
    category: &'a str,
    description: &'a str,
    signature: &'a str,
    examples: &'a [crate::benchmark::Example],
    holdouts: &'a [crate::benchmark::Example],
    reference_code: &'a str,
    synthetic_args: &'a [String],
    synthetic_values: &'a [Vec<i64>],
    recursive_allowed: bool,
    tree_input: bool,
    explicit_stack: bool,
    functions: &'a [FunctionDef],
}

impl ExecutionCapsule {
    pub fn new(
        task: CodeTaskSpec,
        artifact: ExecutableArtifact,
        evaluator: &Problem,
        examples: Vec<Example>,
        policy: ExecutionPolicy,
    ) -> Result<Self, CapsuleError> {
        let run_id = task.run_id.clone();
        let evaluator_digest = evaluator_digest(evaluator)?;
        let mut capsule = Self {
            schema_version: SCHEMA_VERSION,
            run_id,
            task,
            artifact,
            evaluator_digest,
            examples,
            policy,
            capsule_digest: ContentDigest::sha256(&[]),
        };
        capsule.capsule_digest = capsule.recompute_digest()?;
        capsule.validate_integrity()?;
        Ok(capsule)
    }

    pub fn validate_integrity(&self) -> Result<(), CapsuleError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(CapsuleError::SchemaVersion {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.task.schema_version != SCHEMA_VERSION {
            return Err(CapsuleError::SchemaVersion {
                found: self.task.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.run_id.0.trim().is_empty() || self.run_id != self.task.run_id {
            return Err(CapsuleError::RunIdMismatch);
        }
        if self.examples.is_empty() {
            return Err(CapsuleError::EmptyExamples);
        }
        if !self.evaluator_digest.is_well_formed() {
            return Err(CapsuleError::DigestMismatch("evaluator"));
        }
        self.artifact.validate_integrity()?;
        self.policy.validate()?;
        let expected = self.recompute_digest()?;
        if !self.capsule_digest.is_well_formed() || self.capsule_digest != expected {
            return Err(CapsuleError::DigestMismatch("capsule"));
        }
        Ok(())
    }

    /// Verify that the evaluator presented at execution is exactly the one
    /// content-bound into this capsule.
    pub fn validate_evaluator(&self, evaluator: &Problem) -> Result<(), CapsuleError> {
        if self.evaluator_digest != evaluator_digest(evaluator)? {
            return Err(CapsuleError::DigestMismatch("evaluator"));
        }
        Ok(())
    }

    pub fn recompute_digest(&self) -> Result<ContentDigest, CapsuleError> {
        let payload = CapsulePayload {
            schema_version: self.schema_version,
            run_id: &self.run_id,
            task: &self.task,
            artifact: &self.artifact,
            evaluator_digest: &self.evaluator_digest,
            examples: &self.examples,
            policy: &self.policy,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| CapsuleError::Encoding(error.to_string()))?;
        Ok(ContentDigest::sha256(&bytes))
    }
}

fn evaluator_digest(evaluator: &Problem) -> Result<ContentDigest, CapsuleError> {
    let payload = EvaluatorPayload {
        name: &evaluator.name,
        category: evaluator.category,
        description: evaluator.description,
        signature: evaluator.signature,
        examples: &evaluator.examples,
        holdouts: &evaluator.holdouts,
        reference_code: evaluator.reference_code,
        synthetic_args: &evaluator.synthetic_args,
        synthetic_values: &evaluator.synthetic_values,
        recursive_allowed: evaluator.recursive_allowed,
        tree_input: evaluator.tree_input,
        explicit_stack: evaluator.explicit_stack,
        functions: &evaluator.functions,
    };
    let bytes =
        serde_json::to_vec(&payload).map_err(|error| CapsuleError::Encoding(error.to_string()))?;
    Ok(ContentDigest::sha256(&bytes))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapsuleError {
    SchemaVersion { found: u32, expected: u32 },
    EmptyField(&'static str),
    EmptyExamples,
    InvalidLimit(&'static str),
    NonCanonical(&'static str),
    RunIdMismatch,
    DigestMismatch(&'static str),
    Encoding(String),
}

impl fmt::Display for CapsuleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaVersion { found, expected } => {
                write!(
                    formatter,
                    "schema version {found} does not match {expected}"
                )
            }
            Self::EmptyField(field) => write!(formatter, "{field} is empty"),
            Self::EmptyExamples => formatter.write_str("execution examples are empty"),
            Self::InvalidLimit(field) => write!(formatter, "{field} must be greater than zero"),
            Self::NonCanonical(field) => write!(formatter, "{field} is not sorted and unique"),
            Self::RunIdMismatch => formatter.write_str("capsule and task run IDs differ"),
            Self::DigestMismatch(field) => write!(formatter, "{field} digest mismatch"),
            Self::Encoding(error) => write!(formatter, "capsule encoding failed: {error}"),
        }
    }
}

impl std::error::Error for CapsuleError {}

fn strictly_sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::coding_intent::CodingIntent;
    use crate::execution::InputValue;

    fn evaluator() -> Problem {
        Problem {
            name: "add".into(),
            category: "arithmetic",
            description: "add two numbers",
            signature: "fn add(a: i64, b: i64) -> i64",
            examples: vec![crate::benchmark::Example {
                inputs: vec![
                    crate::benchmark::Value::Int(2),
                    crate::benchmark::Value::Int(3),
                ],
                expected: crate::benchmark::Value::Int(5),
            }],
            ..Default::default()
        }
    }

    fn capsule() -> ExecutionCapsule {
        let intent = CodingIntent::from_nl("add two numbers").expect("intent");
        let task = CodeTaskSpec::from_nl(
            "/tmp/repo",
            "add two numbers",
            intent,
            "cargo test add",
            vec!["src/lib.rs".into()],
            2,
        );
        let artifact = ExecutableArtifact::new(
            "add",
            "fn add(a: i64, b: i64) -> i64 { a + b }",
            Language::Rust,
            "unit-test",
            vec![9, 7, 9],
        );
        let examples = vec![Example {
            inputs: vec![InputValue::Int(2), InputValue::Int(3)],
            expected: InputValue::Int(5),
        }];
        ExecutionCapsule::new(
            task,
            artifact,
            &evaluator(),
            examples,
            ExecutionPolicy::new(vec!["fs.read".into()], 1_000, 64 * 1024 * 1024, 64 * 1024),
        )
        .expect("capsule")
    }

    #[test]
    fn capsule_roundtrip_preserves_content_address() {
        let original = capsule();
        assert_eq!(original.artifact.evidence_entity_ids, vec![7, 9]);
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ExecutionCapsule = serde_json::from_str(&json).expect("deserialize");
        restored.validate_integrity().expect("valid roundtrip");
        assert_eq!(restored.capsule_digest, original.capsule_digest);
        restored
            .validate_evaluator(&evaluator())
            .expect("bound evaluator");
    }

    #[test]
    fn source_or_policy_tampering_is_detected() {
        let mut source_tampered = capsule();
        source_tampered.artifact.source.push_str(" // changed");
        assert!(matches!(
            source_tampered.validate_integrity(),
            Err(CapsuleError::DigestMismatch("artifact.source"))
        ));

        let mut policy_tampered = capsule();
        policy_tampered
            .policy
            .allowed_capabilities
            .push("shell.run".into());
        assert!(policy_tampered.validate_integrity().is_err());

        let mut different_evaluator = evaluator();
        different_evaluator.reference_code = "fn add(a:i64,b:i64)->i64{return a-b;}";
        assert!(matches!(
            capsule().validate_evaluator(&different_evaluator),
            Err(CapsuleError::DigestMismatch("evaluator"))
        ));
    }

    #[test]
    fn policy_snapshots_real_secure_runtime_allowlist() {
        let mut runtime = SecureToolRuntime::deny_by_default("/tmp/repo");
        runtime.allow("fs", "read").allow("git", "status");
        let policy = ExecutionPolicy::from_secure_runtime(&runtime, 100, 1024, 512);
        assert_eq!(
            policy.allowed_capabilities,
            vec!["fs.read".to_string(), "git.status".to_string()]
        );
        assert!(policy.allows("fs.read"));
        assert!(!policy.allows("shell.run"));
    }
}
