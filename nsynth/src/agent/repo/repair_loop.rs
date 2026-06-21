use crate::agent::repo::{
    FailureAnalysis, GuardrailDecision, GuardrailPolicy, PatchGate, RepoTaskSpec,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairEdit {
    pub path: String,
    pub old_text: String,
    pub new_text: String,
    pub reason: String,
}

impl RepairEdit {
    pub fn new(
        path: impl Into<String>,
        old_text: impl Into<String>,
        new_text: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            old_text: old_text.into(),
            new_text: new_text.into(),
            reason: reason.into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairPatch {
    pub edits: Vec<RepairEdit>,
    pub metadata: Vec<(String, String)>,
}

impl RepairPatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_edit(mut self, edit: RepairEdit) -> Self {
        self.edits.push(edit);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    pub fn diff_summary(&self) -> String {
        self.edits
            .iter()
            .map(|edit| {
                let path = edit.path.replace('\n', "\\n");
                let reason = edit.reason.replace('\n', " ");
                format!("--- a/{path}\n+++ b/{path}\n@@ edited @@\n+{reason}")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairContext {
    pub root: String,
    pub files: Vec<RepairFile>,
}

impl RepairContext {
    pub fn build(root: impl AsRef<Path>, policy: &GuardrailPolicy) -> Result<Self, String> {
        let root = root.as_ref();
        let mut files = Vec::new();
        Self::scan_dir(root, root, policy, &mut files)?;
        Ok(Self {
            root: root.to_string_lossy().to_string(),
            files,
        })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn scan_dir(
        root: &Path,
        dir: &Path,
        policy: &GuardrailPolicy,
        files: &mut Vec<RepairFile>,
    ) -> Result<(), String> {
        let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                Self::scan_dir(root, &path, policy, files)?;
            } else if path.is_file() {
                if let Some(file) = Self::read_file(root, &path, policy) {
                    files.push(file);
                }
            }
        }
        Ok(())
    }

    fn read_file(root: &Path, path: &Path, policy: &GuardrailPolicy) -> Option<RepairFile> {
        let relative = path.strip_prefix(root).ok()?;
        let normalized = relative.to_string_lossy().replace('\\', "/");
        if !matches!(policy.check_path(relative, false), GuardrailDecision::Allow) {
            return None;
        }
        let bytes = fs::metadata(path).ok()?.len() as usize;
        let content = fs::read_to_string(path).ok()?;
        Some(RepairFile {
            path: normalized,
            bytes,
            lines: content.lines().count(),
            text: Some(content),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairFile {
    pub path: String,
    pub bytes: usize,
    pub lines: usize,
    pub text: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairVerification {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub command: String,
}

impl RepairVerification {
    pub fn failure_output(&self) -> String {
        if self.stderr.is_empty() {
            self.stdout.clone()
        } else if self.stdout.is_empty() {
            self.stderr.clone()
        } else {
            format!("{}\n{}", self.stdout, self.stderr)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepairLoopResult {
    pub success: bool,
    pub iterations: usize,
    pub last_verification: Option<RepairVerification>,
    pub last_failure: Option<FailureAnalysis>,
    pub rejected: Vec<String>,
    pub trace_len: usize,
}

pub struct RepairVerifier {
    root: PathBuf,
    policy: GuardrailPolicy,
}

impl RepairVerifier {
    pub fn new(root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        Self {
            root: root.into(),
            policy,
        }
    }

    pub fn verify(&self, command: &str) -> Result<RepairVerification, String> {
        let runtime = crate::agent::tools::SecureToolRuntime::deny_by_default(&self.root)
            .with_guardrails(self.policy.clone());
        runtime.run_verification_command(command)
    }
}

pub struct RepairLoop {
    agent: super::RepairAgent,
    verifier: RepairVerifier,
    experience_path: Option<PathBuf>,
}

impl RepairLoop {
    pub fn new(root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        let root = root.into();
        Self {
            agent: super::RepairAgent::with_policy(&root, policy.clone()),
            verifier: RepairVerifier::new(root, policy),
            experience_path: None,
        }
    }

    pub fn with_experience_path(mut self, experience_path: impl Into<PathBuf>) -> Self {
        self.experience_path = Some(experience_path.into());
        self
    }

    pub fn verifier(&self) -> &RepairVerifier {
        &self.verifier
    }

    pub fn trace(&self) -> &crate::agent::repo::AgentTrace {
        self.agent.trace()
    }

    pub fn context(&self) -> Result<RepairContext, String> {
        RepairContext::build(&self.verifier.root, &self.verifier.policy)
    }

    pub fn run<F>(&mut self, task: &RepoTaskSpec, proposer: &F) -> Result<RepairLoopResult, String>
    where
        F: Fn(&RepoTaskSpec, usize, Option<&FailureAnalysis>) -> Result<RepairPatch, String>,
    {
        let context = self.context()?;
        self.run_inner(task, &context, &|_, _, iteration, analysis| {
            proposer(task, iteration, analysis)
        })
    }

    pub fn run_with_context<F>(
        &mut self,
        task: &RepoTaskSpec,
        context: &RepairContext,
        proposer: &F,
    ) -> Result<RepairLoopResult, String>
    where
        F: Fn(
            &RepoTaskSpec,
            &RepairContext,
            usize,
            Option<&FailureAnalysis>,
        ) -> Result<RepairPatch, String>,
    {
        let started = Instant::now();
        let result = self.run_inner(task, context, proposer)?;
        let _ = self.record_experience(task, &result, started.elapsed().as_millis() as u64);
        Ok(result)
    }

    fn run_inner<F>(
        &mut self,
        task: &RepoTaskSpec,
        context: &RepairContext,
        proposer: &F,
    ) -> Result<RepairLoopResult, String>
    where
        F: Fn(
            &RepoTaskSpec,
            &RepairContext,
            usize,
            Option<&FailureAnalysis>,
        ) -> Result<RepairPatch, String>,
    {
        let mut last_failure = None;
        let mut last_verification = None;
        let mut rejected = Vec::new();

        for iteration in 0..task.max_iterations {
            let verification = self.verifier.verify(&task.test_command)?;
            last_verification = Some(verification.clone());
            self.agent.trace_mut().push(
                "verify",
                task.id.clone(),
                format!("exit_code={:?}", verification.exit_code),
                if verification.success { "ok" } else { "failed" }.to_string(),
            );

            if verification.success {
                let result = RepairLoopResult {
                    success: true,
                    iterations: iteration,
                    last_verification: Some(verification),
                    last_failure,
                    rejected,
                    trace_len: self.agent.trace().len(),
                };
                self.agent.trace_mut().push(
                    "loop_result",
                    task.id.clone(),
                    format!("success=true iterations={}", result.iterations),
                    "ok",
                );
                return Ok(result);
            }

            let analysis = self.agent.parse_failure(&verification.failure_output());
            last_failure = Some(analysis.clone());

            let patch = proposer(task, context, iteration, Some(&analysis)).map_err(|e| {
                self.agent.trace_mut().push(
                    "propose_patch",
                    task.id.clone(),
                    format!("iteration={iteration}"),
                    format!("rejected: {e}"),
                );
                e
            })?;

            if patch.edits.is_empty() {
                return Err("proposer returned an empty patch".to_string());
            }

            let gate = self
                .agent
                .validate_patch(&patch.diff_summary(), &task.allowed_files);
            if !gate.allowed {
                rejected.extend(gate.rejected);
                return Err(format!("patch rejected: {}", rejected.join("; ")));
            }

            self.apply_patch(task, &patch).map_err(|e| {
                self.agent.trace_mut().push(
                    "apply_patch",
                    task.id.clone(),
                    format!("iteration={iteration}"),
                    format!("failed: {e}"),
                );
                e
            })?;
            self.agent.trace_mut().push(
                "apply_patch",
                task.id.clone(),
                format!("iteration={iteration} edits={}", patch.edits.len()),
                "ok",
            );

            let verification = self.verifier.verify(&task.test_command)?;
            last_verification = Some(verification.clone());
            self.agent.trace_mut().push(
                "verify_after_patch",
                task.id.clone(),
                format!("exit_code={:?}", verification.exit_code),
                if verification.success { "ok" } else { "failed" }.to_string(),
            );

            if verification.success {
                let result = RepairLoopResult {
                    success: true,
                    iterations: iteration + 1,
                    last_verification: Some(verification),
                    last_failure,
                    rejected,
                    trace_len: self.agent.trace().len(),
                };
                self.agent.trace_mut().push(
                    "loop_result",
                    task.id.clone(),
                    format!("success=true iterations={}", result.iterations),
                    "ok",
                );
                return Ok(result);
            }

            last_failure = Some(self.agent.parse_failure(&verification.failure_output()));
        }

        let result = RepairLoopResult {
            success: false,
            iterations: task.max_iterations,
            last_verification,
            last_failure,
            rejected,
            trace_len: self.agent.trace().len(),
        };
        self.agent.trace_mut().push(
            "loop_result",
            task.id.clone(),
            format!("success=false iterations={}", result.iterations),
            "exhausted",
        );
        Ok(result)
    }

    pub fn record_experience(
        &self,
        task: &RepoTaskSpec,
        result: &RepairLoopResult,
        elapsed_ms: u64,
    ) -> Result<(), String> {
        let Some(path) = &self.experience_path else {
            return Ok(());
        };
        let verification = result
            .last_verification
            .as_ref()
            .map(|verification| verification.command.clone())
            .unwrap_or_default();
        let error = if result.success {
            None
        } else {
            result
                .last_failure
                .as_ref()
                .map(|failure| failure.message.clone())
                .or_else(|| {
                    result
                        .last_verification
                        .as_ref()
                        .map(|verification| verification.failure_output())
                })
        };
        let problem = crate::benchmark::Problem {
            name: task.id.clone(),
            category: "repair",
            description: "",
            signature: "fn repair() -> bool",
            examples: vec![crate::benchmark::Example {
                inputs: Vec::new(),
                expected: crate::benchmark::Value::Bool(result.success),
            }],
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: Vec::new(),
        };
        let solve_result = crate::solver::SolveResult {
            success: result.success,
            code: verification,
            method: "repair_loop".to_string(),
            error,
            metadata: crate::differentiable::DifferentiableMetadata::default(),
        };
        let mut db = crate::learning::experience::ExperienceDB::new(path.clone())?;
        db.record_experience(&problem, &solve_result, elapsed_ms)
    }

    fn apply_patch(&self, task: &RepoTaskSpec, patch: &RepairPatch) -> Result<(), String> {
        for edit in &patch.edits {
            let path = self.resolve_path(&edit.path)?;
            self.check_edit_allowed(task, &path)?;
        }
        let mut tx = crate::agent::edit::EditTransaction::begin(&self.verifier.root);
        if let Err(error) = tx.apply_repair_patch(patch) {
            let _ = tx.rollback();
            return Err(error);
        }
        tx.commit()
    }

    fn resolve_path(&self, rel: &str) -> Result<PathBuf, String> {
        let path = Path::new(rel);
        if path.is_absolute() {
            return Err(format!("absolute paths are not allowed: {rel}"));
        }
        for component in path.components() {
            match component {
                Component::ParentDir => {
                    return Err(format!("parent-directory traversal is not allowed: {rel}"));
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(format!("rooted paths are not allowed: {rel}"));
                }
                _ => {}
            }
        }
        Ok(self.verifier.root.join(path))
    }

    fn check_edit_allowed(&self, task: &RepoTaskSpec, path: &Path) -> Result<(), String> {
        let relative = path
            .strip_prefix(&self.verifier.root)
            .map_err(|e| e.to_string())?;
        let normalized = relative.to_string_lossy().replace('\\', "/");

        if !task
            .allowed_files
            .iter()
            .any(|pattern| pattern_matches(pattern, &normalized))
        {
            return Err(format!(
                "path is outside allowed file policy: {}",
                normalized
            ));
        }

        match self.verifier.policy.check_path(relative, true) {
            GuardrailDecision::Allow => Ok(()),
            GuardrailDecision::Deny(reason) => Err(reason),
            GuardrailDecision::Ask(reason) => Err(reason),
        }
    }
}

fn pattern_matches(pattern: &str, path: &str) -> bool {
    let pattern = pattern.trim_start_matches("./").trim_end_matches('/');
    let path = path.trim_start_matches("./");
    if pattern.ends_with("/**") {
        let prefix = pattern.trim_end_matches("/**");
        return path == prefix || path.starts_with(&format!("{prefix}/"));
    }
    if pattern.ends_with('*') {
        let prefix = pattern.trim_end_matches('*');
        return path.starts_with(prefix);
    }
    path == pattern || path.starts_with(&format!("{pattern}/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::repo::{HardnessProfile, HardnessTier, RepoTaskKind};

    const LOOP_CARGO_TOML: &str = r#"[package]
name = "loop-fixture"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
"#;

    const LOOP_LIB_FAILING: &str = r#"pub fn value() -> i64 { 0 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_oracle() {
        assert_eq!(value(), 1);
    }
}
"#;

    const LOOP_LIB_PASSING: &str = r#"pub fn value() -> i64 { 1 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_oracle() {
        assert_eq!(value(), 1);
    }
}
"#;

    const LOOP_CARGO_TEST: &str = "cargo test loop_oracle --lib";

    fn temp_repo(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("nsynth_repo_loop_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("Cargo.toml"), LOOP_CARGO_TOML).unwrap();
        fs::write(root.join("src/lib.rs"), LOOP_LIB_FAILING).unwrap();
        root
    }

    fn temp_repo_passing(tag: &str) -> PathBuf {
        let root = temp_repo(tag);
        fs::write(root.join("src/lib.rs"), LOOP_LIB_PASSING).unwrap();
        root
    }

    fn task(root: &Path, command: impl Into<String>) -> RepoTaskSpec {
        RepoTaskSpec {
            id: "loop-test".to_string(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "replace 0 with 1".to_string(),
            test_command: command.into(),
            allowed_files: vec!["src/**".to_string()],
            max_iterations: 3,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        }
    }

    #[test]
    fn loop_succeeds_after_verifier_gate_passes() {
        let root = temp_repo("success");
        let task = task(&root, LOOP_CARGO_TEST);
        let old = fs::read_to_string(root.join("src/lib.rs")).unwrap();
        let new = old.replace("pub fn value() -> i64 { 0 }", "pub fn value() -> i64 { 1 }");
        let mut loop_runner = RepairLoop::new(&root, GuardrailPolicy::default());
        let result = loop_runner
            .run(&task, &|_, _, _| {
                Ok(RepairPatch::new().with_edit(RepairEdit::new(
                    "src/lib.rs",
                    old.clone(),
                    new.clone(),
                    "make verifier pass",
                )))
            })
            .unwrap();

        assert!(result.success);
        assert_eq!(result.iterations, 1);
        assert!(fs::read_to_string(root.join("src/lib.rs"))
            .unwrap()
            .contains("pub fn value() -> i64 { 1 }"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn context_includes_readable_files_and_skips_ignored_paths() {
        let root = temp_repo("context");
        fs::write(root.join(".env"), "secret=value\n").unwrap();
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).unwrap();
        assert!(context.files.iter().any(|file| file.path == "src/lib.rs"));
        assert!(!context.files.iter().any(|file| file.path == ".env"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn run_with_context_records_verified_experience() {
        let root = temp_repo("experience");
        let db_path = std::env::temp_dir().join(format!(
            "nsynth_repair_experience_{}_{}.json",
            "experience",
            std::process::id()
        ));
        let _ = fs::remove_file(&db_path);
        let task = task(&root, LOOP_CARGO_TEST);
        let old = fs::read_to_string(root.join("src/lib.rs")).unwrap();
        let new = old.replace("pub fn value() -> i64 { 0 }", "pub fn value() -> i64 { 1 }");
        let mut loop_runner =
            RepairLoop::new(&root, GuardrailPolicy::default()).with_experience_path(&db_path);
        let context = loop_runner.context().unwrap();
        let result = loop_runner
            .run_with_context(&task, &context, &|_, _, _, _| {
                Ok(RepairPatch::new().with_edit(RepairEdit::new(
                    "src/lib.rs",
                    old.clone(),
                    new.clone(),
                    "make verifier pass",
                )))
            })
            .unwrap();
        assert!(result.success);
        let db = crate::learning::experience::ExperienceDB::new(db_path.clone()).unwrap();
        assert_eq!(db.len(), 1);
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn loop_reports_initial_success_without_proposer() {
        let root = temp_repo_passing("initial");
        let task = task(&root, LOOP_CARGO_TEST);
        let mut loop_runner = RepairLoop::new(&root, GuardrailPolicy::default());
        let result = loop_runner
            .run(&task, &|_, _, _| Err("should not be called".to_string()))
            .unwrap();

        assert!(result.success);
        assert_eq!(result.iterations, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verifier_blocks_unsafe_command() {
        let root = temp_repo("unsafe");
        let verifier = RepairVerifier::new(&root, GuardrailPolicy::default());
        let err = verifier.verify("rm -rf .").unwrap_err();
        assert!(err.contains("allowlist"));
        let verification = verifier.verify("cargo test loop_oracle --lib").unwrap();
        assert!(!verification.success);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn patch_rejects_paths_outside_allowed_policy() {
        let root = temp_repo("outside");
        let task = task(&root, LOOP_CARGO_TEST);
        let mut loop_runner = RepairLoop::new(&root, GuardrailPolicy::default());
        let err = loop_runner
            .run(&task, &|_, _, _| {
                Ok(RepairPatch::new().with_edit(RepairEdit::new(
                    "../secret.txt",
                    "x",
                    "y",
                    "escape",
                )))
            })
            .unwrap_err();

        assert!(err.contains("outside allowed file policy"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn patch_does_not_write_when_any_edit_is_invalid() {
        let root = temp_repo("rollback");
        let task = task(&root, LOOP_CARGO_TEST);
        let loop_runner = RepairLoop::new(&root, GuardrailPolicy::default());
        let old = fs::read_to_string(root.join("src/lib.rs")).unwrap();
        let patch = RepairPatch::new()
            .with_edit(RepairEdit::new(
                "src/lib.rs",
                old.clone(),
                old.replace("pub fn value() -> i64 { 0 }", "pub fn value() -> i64 { 1 }"),
                "valid",
            ))
            .with_edit(RepairEdit::new("src/lib.rs", "missing", "nope", "invalid"));

        let err = loop_runner.apply_patch(&task, &patch).unwrap_err();
        assert!(err.contains("found 0"));
        assert!(fs::read_to_string(root.join("src/lib.rs"))
            .unwrap()
            .contains("pub fn value() -> i64 { 0 }"));
        let _ = fs::remove_dir_all(root);
    }
}
