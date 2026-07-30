//! Atomic persistence for proof-carrying execution evidence.
//!
//! This is an append-only audit artifact, not a knowledge or capability store.
//! A capsule and its trace are persisted together under the trace content digest
//! so neither can be silently paired with a different evaluator or artifact.

use super::{ExecutionCapsule, ExecutionTrace, TraceError, SCHEMA_VERSION};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvidenceBundle {
    pub schema_version: u32,
    pub capsule: ExecutionCapsule,
    pub trace: ExecutionTrace,
}

impl ExecutionEvidenceBundle {
    pub fn new(
        capsule: ExecutionCapsule,
        trace: ExecutionTrace,
    ) -> Result<Self, EvidencePersistenceError> {
        let bundle = Self {
            schema_version: SCHEMA_VERSION,
            capsule,
            trace,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<(), EvidencePersistenceError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(EvidencePersistenceError::SchemaVersion {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        self.trace
            .validate_against_capsule(&self.capsule)
            .map_err(EvidencePersistenceError::Trace)
    }
}

#[derive(Debug)]
pub enum EvidencePersistenceError {
    SchemaVersion { found: u32, expected: u32 },
    Trace(TraceError),
    Io(String),
    Encoding(String),
    InvalidEvidenceFile(PathBuf),
    ContentAddressPathMismatch(PathBuf),
    ContentAddressCollision(PathBuf),
}

impl std::fmt::Display for EvidencePersistenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaVersion { found, expected } => {
                write!(
                    formatter,
                    "evidence schema {found} does not match {expected}"
                )
            }
            Self::Trace(error) => write!(formatter, "invalid execution evidence: {error}"),
            Self::Io(error) => write!(formatter, "execution evidence I/O failed: {error}"),
            Self::Encoding(error) => {
                write!(formatter, "execution evidence encoding failed: {error}")
            }
            Self::InvalidEvidenceFile(path) => write!(
                formatter,
                "execution evidence path is not a regular file: {}",
                path.display()
            ),
            Self::ContentAddressPathMismatch(path) => write!(
                formatter,
                "execution evidence filename does not match its trace digest: {}",
                path.display()
            ),
            Self::ContentAddressCollision(path) => write!(
                formatter,
                "different execution evidence already exists at {}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for EvidencePersistenceError {}

pub fn execution_evidence_dir(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(".nsynth").join("execution-evidence")
}

pub fn execution_evidence_path(root: impl AsRef<Path>, trace: &ExecutionTrace) -> PathBuf {
    execution_evidence_dir(root).join(format!("{}.json", trace.trace_digest))
}

pub fn save_execution_evidence(
    root: impl AsRef<Path>,
    capsule: &ExecutionCapsule,
    trace: &ExecutionTrace,
) -> Result<PathBuf, EvidencePersistenceError> {
    let bundle = ExecutionEvidenceBundle::new(capsule.clone(), trace.clone())?;
    let bytes = serde_json::to_vec_pretty(&bundle)
        .map_err(|error| EvidencePersistenceError::Encoding(error.to_string()))?;
    let directory = execution_evidence_dir(&root);
    fs::create_dir_all(&directory)
        .map_err(|error| EvidencePersistenceError::Io(error.to_string()))?;
    let target = execution_evidence_path(&root, trace);
    match fs::symlink_metadata(&target) {
        Ok(_) => {
            ensure_same_content(&target, &bytes)?;
            return Ok(target);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(EvidencePersistenceError::Io(error.to_string())),
    }

    let temp_id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let temp = directory.join(format!(
        ".{}.{}.{}.tmp",
        trace.trace_digest,
        std::process::id(),
        temp_id
    ));
    let write_result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temp)
            .map_err(|error| EvidencePersistenceError::Io(error.to_string()))?;
        file.write_all(&bytes)
            .map_err(|error| EvidencePersistenceError::Io(error.to_string()))?;
        file.sync_all()
            .map_err(|error| EvidencePersistenceError::Io(error.to_string()))?;

        match fs::hard_link(&temp, &target) {
            Ok(()) => {}
            Err(link_error) => match fs::symlink_metadata(&target) {
                Ok(_) => ensure_same_content(&target, &bytes)?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Err(EvidencePersistenceError::Io(link_error.to_string()));
                }
                Err(error) => return Err(EvidencePersistenceError::Io(error.to_string())),
            },
        }
        sync_directory(&directory)?;
        Ok(())
    })();
    let _ = fs::remove_file(&temp);
    write_result?;
    Ok(target)
}

pub fn load_execution_evidence(
    path: impl AsRef<Path>,
) -> Result<ExecutionEvidenceBundle, EvidencePersistenceError> {
    let path = path.as_ref();
    ensure_regular_file(path)?;
    let bytes = fs::read(path).map_err(|error| EvidencePersistenceError::Io(error.to_string()))?;
    let bundle: ExecutionEvidenceBundle = serde_json::from_slice(&bytes)
        .map_err(|error| EvidencePersistenceError::Encoding(error.to_string()))?;
    bundle.validate()?;
    let expected_name = format!("{}.json", bundle.trace.trace_digest);
    if path.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str()) {
        return Err(EvidencePersistenceError::ContentAddressPathMismatch(
            path.to_path_buf(),
        ));
    }
    Ok(bundle)
}

fn ensure_same_content(path: &Path, expected: &[u8]) -> Result<(), EvidencePersistenceError> {
    ensure_regular_file(path)?;
    let existing =
        fs::read(path).map_err(|error| EvidencePersistenceError::Io(error.to_string()))?;
    if existing == expected {
        Ok(())
    } else {
        Err(EvidencePersistenceError::ContentAddressCollision(
            path.to_path_buf(),
        ))
    }
}

fn ensure_regular_file(path: &Path) -> Result<(), EvidencePersistenceError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| EvidencePersistenceError::Io(error.to_string()))?;
    if metadata.file_type().is_file() {
        Ok(())
    } else {
        Err(EvidencePersistenceError::InvalidEvidenceFile(
            path.to_path_buf(),
        ))
    }
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> Result<(), EvidencePersistenceError> {
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|error| EvidencePersistenceError::Io(error.to_string()))
}

#[cfg(not(unix))]
fn sync_directory(_directory: &Path) -> Result<(), EvidencePersistenceError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{
        CodeTaskSpec, ExecutableArtifact, ExecutionPolicy, SANDBOX_EXECUTION_CAPABILITY,
    };
    use super::*;
    use crate::agent::coding_intent::CodingIntent;
    use crate::agent::runtime::CapsuleExecutor;
    use crate::benchmark::{Example as BenchmarkExample, Problem, Value};
    use crate::execution::{Example, InputValue, Language};

    fn problem() -> Problem {
        Problem {
            name: "persist_inc_v0".into(),
            category: "arithmetic",
            description: "increment",
            signature: "fn persist_inc_v0(a: i64) -> i64",
            examples: vec![BenchmarkExample {
                inputs: vec![Value::Int(1)],
                expected: Value::Int(2),
            }],
            holdouts: vec![BenchmarkExample {
                inputs: vec![Value::Int(8)],
                expected: Value::Int(9),
            }],
            ..Default::default()
        }
    }

    fn denied_bundle() -> (ExecutionCapsule, ExecutionTrace) {
        let evaluator = problem();
        let task = CodeTaskSpec::from_nl(
            "/tmp/repo",
            "increment",
            CodingIntent::from_nl("increment").expect("intent"),
            "cargo test increment",
            vec!["src/lib.rs".into()],
            1,
        );
        let capsule = ExecutionCapsule::new(
            task,
            ExecutableArtifact::new(
                "persist_inc_v0",
                "fn persist_inc_v0(a: i64) -> i64 { a + 1 }",
                Language::Rust,
                "persistence-test",
                vec![41],
            ),
            &evaluator,
            vec![Example {
                inputs: vec![InputValue::Int(1)],
                expected: InputValue::Int(2),
            }],
            ExecutionPolicy::new(vec![], 1_000, 1024 * 1024, 1024),
        )
        .expect("capsule");
        assert!(!capsule.policy.allows(SANDBOX_EXECUTION_CAPABILITY));
        let trace = CapsuleExecutor::execute(&capsule, &evaluator).expect("denial trace");
        (capsule, trace)
    }

    #[test]
    fn atomic_content_addressed_roundtrip_is_idempotent() {
        let root =
            std::env::temp_dir().join(format!("nsynth_execution_evidence_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let (capsule, trace) = denied_bundle();
        let first = save_execution_evidence(&root, &capsule, &trace).expect("save");
        let second = save_execution_evidence(&root, &capsule, &trace).expect("idempotent save");
        assert_eq!(first, second);
        let restored = load_execution_evidence(&first).expect("load");
        assert_eq!(restored.trace.trace_digest, trace.trace_digest);
        assert_eq!(restored.capsule.capsule_digest, capsule.capsule_digest);
        let temporary_files = fs::read_dir(execution_evidence_dir(&root))
            .expect("evidence directory")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(temporary_files, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn mismatched_or_corrupted_evidence_is_refused() {
        let (capsule, trace) = denied_bundle();
        let mut mismatched = capsule.clone();
        mismatched.artifact.source.push_str(" // different");
        mismatched.artifact.source_digest =
            super::super::ContentDigest::sha256(mismatched.artifact.source.as_bytes());
        mismatched.capsule_digest = mismatched.recompute_digest().expect("digest");
        assert!(matches!(
            ExecutionEvidenceBundle::new(mismatched, trace.clone()),
            Err(EvidencePersistenceError::Trace(
                TraceError::TraceCapsuleMismatch
            ))
        ));

        let root = std::env::temp_dir().join(format!(
            "nsynth_execution_evidence_corrupt_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let path = save_execution_evidence(&root, &capsule, &trace).expect("save");
        fs::write(&path, b"{\"corrupted\":true}").expect("corrupt");
        assert!(load_execution_evidence(&path).is_err());
        assert!(matches!(
            save_execution_evidence(&root, &capsule, &trace),
            Err(EvidencePersistenceError::ContentAddressCollision(_))
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn address_path_and_non_file_targets_are_refused() {
        let (capsule, trace) = denied_bundle();
        let root = std::env::temp_dir().join(format!(
            "nsynth_execution_evidence_address_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let path = save_execution_evidence(&root, &capsule, &trace).expect("save");
        let renamed = path.with_file_name("wrong-digest.json");
        fs::rename(&path, &renamed).expect("rename");
        assert!(matches!(
            load_execution_evidence(&renamed),
            Err(EvidencePersistenceError::ContentAddressPathMismatch(_))
        ));

        fs::create_dir(&path).expect("occupy content address with directory");
        assert!(matches!(
            save_execution_evidence(&root, &capsule, &trace),
            Err(EvidencePersistenceError::InvalidEvidenceFile(_))
        ));
        let _ = fs::remove_dir_all(root);
    }
}
