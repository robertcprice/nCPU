use crate::agent::repo::{HardnessProfile, HardnessTier, RepoTaskKind, RepoTaskSpec};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalBenchmarkTask {
    pub id: String,
    pub kind: RepoTaskKind,
    pub issue: String,
    pub test_command: String,
    pub allowed_files: Vec<String>,
    pub expected_tier_min: HardnessTier,
    pub max_iterations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepoBenchmark {
    root: PathBuf,
    tasks: Vec<LocalBenchmarkTask>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkValidation {
    pub task_id: String,
    pub errors: Vec<String>,
}

impl RepoBenchmark {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            tasks: Vec::new(),
        }
    }

    pub fn from_standard_suite(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            tasks: standard_n_cpu_suite(),
        }
    }

    pub fn add_task(&mut self, task: LocalBenchmarkTask) {
        self.tasks.push(task);
    }

    pub fn tasks(&self) -> &[LocalBenchmarkTask] {
        &self.tasks
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn validate_all(&self) -> Vec<BenchmarkValidation> {
        self.tasks
            .iter()
            .map(|task| self.validate_task(task))
            .collect()
    }

    pub fn validate_task(&self, task: &LocalBenchmarkTask) -> BenchmarkValidation {
        let mut errors = Vec::new();
        if task.id.trim().is_empty() {
            errors.push("task id is empty".to_string());
        }
        if task.issue.trim().is_empty() {
            errors.push("issue is empty".to_string());
        }
        if task.test_command.trim().is_empty() {
            errors.push("test command is empty".to_string());
        }
        if task.allowed_files.is_empty() {
            errors.push("allowed files are empty".to_string());
        }
        if task.max_iterations == 0 {
            errors.push("max iterations must be greater than zero".to_string());
        }
        BenchmarkValidation {
            task_id: task.id.clone(),
            errors,
        }
    }

    pub fn to_task_specs(&self) -> Vec<RepoTaskSpec> {
        self.tasks
            .iter()
            .map(|task| RepoTaskSpec {
                id: task.id.clone(),
                repo: self.root.to_string_lossy().to_string(),
                kind: task.kind,
                issue: task.issue.clone(),
                test_command: task.test_command.clone(),
                allowed_files: task.allowed_files.clone(),
                max_iterations: task.max_iterations,
                hardness: HardnessProfile::for_expected_tier(task.expected_tier_min),
                signals: Vec::new(),
            })
            .collect()
    }
}

impl LocalBenchmarkTask {
    pub fn to_task_spec(&self, root: impl Into<PathBuf>) -> RepoTaskSpec {
        RepoTaskSpec {
            id: self.id.clone(),
            repo: root.into().to_string_lossy().to_string(),
            kind: self.kind,
            issue: self.issue.clone(),
            test_command: self.test_command.clone(),
            allowed_files: self.allowed_files.clone(),
            max_iterations: self.max_iterations,
            hardness: HardnessProfile::for_expected_tier(self.expected_tier_min),
            signals: Vec::new(),
        }
    }
}

impl HardnessProfile {
    pub fn for_expected_tier(tier: HardnessTier) -> Self {
        let score = match tier {
            HardnessTier::Trivial => 0.1,
            HardnessTier::SingleFileBug => 0.25,
            HardnessTier::MultiFileBug => 0.38,
            HardnessTier::RegressionRepair => 0.48,
            HardnessTier::FeatureWithTests => 0.58,
            HardnessTier::CrossModuleRefactor => 0.68,
            HardnessTier::SweBenchIssue => 0.82,
            HardnessTier::LongHorizonMigration => 0.94,
        };
        Self {
            localization: score,
            ambiguity: score,
            edit_surface: score,
            test_complexity: score,
            regression_risk: score,
            reasoning_depth: score,
            tool_dependency: score,
            verification_cost: score,
        }
    }
}

pub fn standard_n_cpu_suite() -> Vec<LocalBenchmarkTask> {
    vec![
        LocalBenchmarkTask {
            id: "bug_score_wraparound".to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "Fix score wraparound boundary behavior".to_string(),
            test_command: "cargo test score".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::SingleFileBug,
            max_iterations: 4,
        },
        LocalBenchmarkTask {
            id: "bug_array_count".to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "Fix array count off-by-one behavior".to_string(),
            test_command: "cargo test array_count".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::SingleFileBug,
            max_iterations: 4,
        },
        LocalBenchmarkTask {
            id: "bug_string_lower".to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "Fix string-input routing for lowercase tasks".to_string(),
            test_command: "cargo test string_lower".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::MultiFileBug,
            max_iterations: 5,
        },
        LocalBenchmarkTask {
            id: "bug_probabilistic_false_pass".to_string(),
            kind: RepoTaskKind::RegressionRepair,
            issue: "Prevent probabilistic synthesizer from false-passing array tasks".to_string(),
            test_command: "cargo test probabilistic".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::RegressionRepair,
            max_iterations: 6,
        },
        LocalBenchmarkTask {
            id: "bug_cache_load_utf8".to_string(),
            kind: RepoTaskKind::RegressionRepair,
            issue: "Fix solved-cache load hang caused by corrupt UTF-8 entries".to_string(),
            test_command: "cargo test solved_cache".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::RegressionRepair,
            max_iterations: 6,
        },
        LocalBenchmarkTask {
            id: "test_array_transform".to_string(),
            kind: RepoTaskKind::MissingTest,
            issue: "Add coverage for array-input scalar-output transforms".to_string(),
            test_command: "cargo test array_transform".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 5,
        },
        LocalBenchmarkTask {
            id: "test_string_synthesis".to_string(),
            kind: RepoTaskKind::MissingTest,
            issue: "Add tests for string-to-string synthesis routing".to_string(),
            test_command: "cargo test string_synthesis".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 5,
        },
        LocalBenchmarkTask {
            id: "test_dp_bottom_up".to_string(),
            kind: RepoTaskKind::MissingTest,
            issue: "Add bottom-up DP coverage probes".to_string(),
            test_command: "cargo test dp".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 5,
        },
        LocalBenchmarkTask {
            id: "test_probabilistic_verification".to_string(),
            kind: RepoTaskKind::MissingTest,
            issue: "Add verifier gate around probabilistic synthesis".to_string(),
            test_command: "cargo test probabilistic_verification".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 6,
        },
        LocalBenchmarkTask {
            id: "test_repo_agent".to_string(),
            kind: RepoTaskKind::MissingTest,
            issue: "Add repo-agent guardrail and patch-gate tests".to_string(),
            test_command: "cargo test repo_agent".to_string(),
            allowed_files: vec!["src/agent/repo/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 4,
        },
        LocalBenchmarkTask {
            id: "feature_repo_agent_loop".to_string(),
            kind: RepoTaskKind::Feature,
            issue: "Implement deterministic repo-agent loop skeleton".to_string(),
            test_command: "cargo test repo_agent".to_string(),
            allowed_files: vec![
                "src/agent/repo/**".to_string(),
                "src/agent/mod.rs".to_string(),
            ],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 6,
        },
        LocalBenchmarkTask {
            id: "feature_failure_parser".to_string(),
            kind: RepoTaskKind::Feature,
            issue: "Implement structured failure parser".to_string(),
            test_command: "cargo test failure_parser".to_string(),
            allowed_files: vec![
                "src/agent/repo/failure_parser.rs".to_string(),
                "tests/**".to_string(),
            ],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 4,
        },
        LocalBenchmarkTask {
            id: "feature_patch_gate".to_string(),
            kind: RepoTaskKind::Feature,
            issue: "Implement patch gate for allowed paths and merge markers".to_string(),
            test_command: "cargo test patch_gate".to_string(),
            allowed_files: vec![
                "src/agent/repo/patch_gate.rs".to_string(),
                "tests/**".to_string(),
            ],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 4,
        },
        LocalBenchmarkTask {
            id: "feature_credit_assignment".to_string(),
            kind: RepoTaskKind::Feature,
            issue: "Implement credit assignment ledger".to_string(),
            test_command: "cargo test credit".to_string(),
            allowed_files: vec![
                "src/agent/repo/credit.rs".to_string(),
                "tests/**".to_string(),
            ],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 4,
        },
        LocalBenchmarkTask {
            id: "feature_hardness_miner".to_string(),
            kind: RepoTaskKind::Feature,
            issue: "Implement hardness miner for repo signals".to_string(),
            test_command: "cargo test hardness".to_string(),
            allowed_files: vec![
                "src/agent/repo/hardness.rs".to_string(),
                "tests/**".to_string(),
            ],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 5,
        },
        LocalBenchmarkTask {
            id: "refactor_solver_pipeline".to_string(),
            kind: RepoTaskKind::Refactor,
            issue: "Refactor solver pipeline to support search-only mode".to_string(),
            test_command: "cargo test solver".to_string(),
            allowed_files: vec![
                "src/solver/**".to_string(),
                "src/agent/**".to_string(),
                "tests/**".to_string(),
            ],
            expected_tier_min: HardnessTier::CrossModuleRefactor,
            max_iterations: 8,
        },
        LocalBenchmarkTask {
            id: "refactor_memory_store".to_string(),
            kind: RepoTaskKind::Refactor,
            issue: "Refactor repo memory store around failure and strategy memories".to_string(),
            test_command: "cargo test memory".to_string(),
            allowed_files: vec![
                "src/learning/**".to_string(),
                "src/memory/**".to_string(),
                "tests/**".to_string(),
            ],
            expected_tier_min: HardnessTier::CrossModuleRefactor,
            max_iterations: 8,
        },
        LocalBenchmarkTask {
            id: "regression_probabilistic_cache".to_string(),
            kind: RepoTaskKind::RegressionRepair,
            issue: "Regression repair for probabilistic cache false-pass".to_string(),
            test_command: "cargo test probabilistic_cache".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::RegressionRepair,
            max_iterations: 6,
        },
        LocalBenchmarkTask {
            id: "regression_array_transform_hang".to_string(),
            kind: RepoTaskKind::RegressionRepair,
            issue: "Regression repair for array transform pre-stage hang".to_string(),
            test_command: "cargo test array_transform".to_string(),
            allowed_files: vec!["src/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::RegressionRepair,
            max_iterations: 6,
        },
        LocalBenchmarkTask {
            id: "regression_orchestrator_stack".to_string(),
            kind: RepoTaskKind::RegressionRepair,
            issue: "Regression repair for orchestrator stack overflow in search-only mode"
                .to_string(),
            test_command: "cargo test orchestrator".to_string(),
            allowed_files: vec!["src/agent/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::RegressionRepair,
            max_iterations: 6,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_suite_has_twenty_tasks() {
        let suite = standard_n_cpu_suite();
        assert_eq!(suite.len(), 20);
        assert!(suite.iter().all(|task| !task.allowed_files.is_empty()));
    }

    #[test]
    fn benchmark_validates_and_exports_task_specs() {
        let benchmark = RepoBenchmark::from_standard_suite("/tmp/ncpu");
        let validations = benchmark.validate_all();
        assert!(validations
            .iter()
            .all(|validation| validation.errors.is_empty()));
        let specs = benchmark.to_task_specs();
        assert_eq!(specs.len(), 20);
        assert!(specs.iter().all(|spec| spec.max_iterations > 0));
    }
}
