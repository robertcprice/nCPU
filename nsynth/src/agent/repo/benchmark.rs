use super::nl_fixture_cargo_test_command;
use crate::agent::coding_intent::CodingIntent;
use crate::agent::repo::{HardnessProfile, HardnessTier, RepoTaskKind, RepoTaskSpec};
use crate::agent::runtime::CodeTaskSpec;
use crate::linguigenesis_bridge::{BridgeError, LinguigenesisBridge};
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
            id: "test_repair_agent".to_string(),
            kind: RepoTaskKind::MissingTest,
            issue: "Add repair-agent guardrail and patch-gate tests".to_string(),
            test_command: "cargo test repair_agent".to_string(),
            allowed_files: vec!["src/agent/repo/**".to_string(), "tests/**".to_string()],
            expected_tier_min: HardnessTier::FeatureWithTests,
            max_iterations: 4,
        },
        LocalBenchmarkTask {
            id: "feature_repair_agent_loop".to_string(),
            kind: RepoTaskKind::Feature,
            issue: "Implement deterministic repair-agent loop skeleton".to_string(),
            test_command: "cargo test repair_agent".to_string(),
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

/// Wrong `src/lib.rs` body for isolated NL fixture harnesses (Package H).
pub fn nl_fixture_wrong_stub(fixture_id: &str) -> Option<&'static str> {
    match fixture_id {
        "nl_fixture_add" => Some("pub fn add_two(a: i64, b: i64) -> i64 { a - b }\n"),
        "nl_fixture_subtract" => Some("pub fn subtract(a: i64, b: i64) -> i64 { a + b }\n"),
        "nl_fixture_multiply" => Some("pub fn multiply(a: i64, b: i64) -> i64 { a / b }\n"),
        "nl_fixture_divide" => Some("pub fn divide(a: i64, b: i64) -> i64 { a * b }\n"),
        "nl_fixture_max" => {
            Some("pub fn max_of(a: i64, b: i64) -> i64 { if a < b { a } else { b } }\n")
        }
        "nl_fixture_reverse" => Some("pub fn reverse(xs: &[i64]) -> Vec<i64> { xs.to_vec() }\n"),
        // Unseen op (no keyword-table entry): only real synthesis from the
        // inline examples in the issue can repair this. Proves the closed loop
        // generalizes beyond the canned scalar vocabulary.
        "nl_fixture_triple" => Some("pub fn triple(a: i64) -> i64 { a + 1 }\n"),
        // Broadened unseen-NL corpus (G5 sign-off): each repairs ONLY via real
        // example-driven synthesis (no keyword-table entry exists), spanning
        // distinct solver families — nonlinear scalar, multi-arg/single-arg
        // affine, branch, and array fold reductions.
        "nl_fixture_square" => Some("pub fn square(a: i64) -> i64 { a + a }\n"),
        "nl_fixture_negate" => Some("pub fn negate(a: i64) -> i64 { a }\n"),
        "nl_fixture_abs" => Some("pub fn absval(a: i64) -> i64 { a }\n"),
        "nl_fixture_sum3" => Some("pub fn add3(a: i64, b: i64, c: i64) -> i64 { a + b }\n"),
        "nl_fixture_arrsum" => Some("pub fn total(xs: Vec<i64>) -> i64 { 0 }\n"),
        "nl_fixture_arrmax" => Some("pub fn biggest(xs: Vec<i64>) -> i64 { 0 }\n"),
        "nl_fixture_arrlen" => Some("pub fn howmany(xs: Vec<i64>) -> i64 { 0 }\n"),
        "nl_fixture_min3" => Some("pub fn smallest(a: i64, b: i64, c: i64) -> i64 { a }\n"),
        _ => None,
    }
}

// gcd uses write_gcd_fixture in harness (no single-file wrong stub).

/// Isolated NL synthesis fixtures for repair-loop verification (Package H/M).
pub fn nl_synthesis_fixture_suite() -> Vec<LocalBenchmarkTask> {
    vec![
        nl_fixture_task("nl_fixture_add", "synthesize: add two numbers"),
        nl_fixture_task("nl_fixture_subtract", "synthesize: subtract two numbers"),
        nl_fixture_task("nl_fixture_multiply", "synthesize: multiply two numbers"),
        nl_fixture_task("nl_fixture_divide", "synthesize: divide two numbers"),
        nl_fixture_task(
            "nl_fixture_max",
            "synthesize: return the larger of two numbers",
        ),
        nl_fixture_task("nl_fixture_reverse", "synthesize: reverse array"),
        nl_fixture_task(
            "nl_fixture_multifile_multiply",
            "synthesize: multiply two numbers",
        ),
        nl_fixture_task("nl_fixture_gcd", "synthesize: greatest common divisor"),
        nl_fixture_task(
            "nl_fixture_triple",
            "synthesize: a function where triple(2)=6 and triple(5)=15 and triple(3)=9",
        ),
        nl_fixture_task(
            "nl_fixture_square",
            "synthesize: a function where square(2)=4 and square(3)=9 and square(4)=16 and square(5)=25 and square(6)=36 and square(0)=0",
        ),
        nl_fixture_task(
            "nl_fixture_negate",
            "synthesize: a function where negate(5)=-5 and negate(-3)=3 and negate(0)=0 and negate(7)=-7 and negate(-12)=12",
        ),
        nl_fixture_task(
            "nl_fixture_abs",
            "synthesize: a function where absval(-3)=3 and absval(4)=4 and absval(-10)=10 and absval(0)=0 and absval(-1)=1 and absval(8)=8",
        ),
        nl_fixture_task(
            "nl_fixture_sum3",
            "synthesize: a function where add3(1,2,3)=6 and add3(0,0,5)=5 and add3(2,2,2)=6 and add3(10,20,30)=60 and add3(-1,1,0)=0",
        ),
        nl_fixture_task(
            "nl_fixture_arrsum",
            "synthesize: a function where total([1,2,3])=6 and total([4,5])=9 and total([10])=10 and total([2,2,2,2])=8 and total([7,3])=10",
        ),
        nl_fixture_task(
            "nl_fixture_arrmax",
            "synthesize: a function where biggest([3,1,2])=3 and biggest([5,9,1])=9 and biggest([7])=7 and biggest([-1,-5,-2])=-1 and biggest([2,2,8,4])=8",
        ),
        nl_fixture_task(
            "nl_fixture_arrlen",
            "synthesize: a function where howmany([3,1,2])=3 and howmany([5,9])=2 and howmany([7])=1 and howmany([1,2,3,4,5])=5 and howmany([6,6])=2",
        ),
        nl_fixture_task(
            "nl_fixture_min3",
            "synthesize: a function where smallest(3,7,5)=3 and smallest(9,2,8)=2 and smallest(1,4,1)=1 and smallest(5,5,2)=2 and smallest(-1,0,3)=-1 and smallest(8,8,8)=8 and smallest(4,1,9)=1",
        ),
    ]
}

/// Fast CI subset: holdout fixtures with inline I/O (no registry-vocab dependency).
pub fn nl_synthesis_fixture_ci_subset() -> Vec<LocalBenchmarkTask> {
    let ids = [
        "nl_fixture_triple",
        "nl_fixture_square",
        "nl_fixture_negate",
    ];
    nl_synthesis_fixture_suite()
        .into_iter()
        .filter(|task| ids.contains(&task.id.as_str()))
        .collect()
}

fn fixture_intent_from_nl(nl: &str) -> CodingIntent {
    let bridge = LinguigenesisBridge::new();
    let req = match bridge.nl_to_requirement(nl) {
        Ok(req) => req,
        Err(BridgeError::ClarificationNeeded { partial, .. }) => partial,
        Err(err) => panic!("fixture intent for '{nl}': {err}"),
    };
    CodingIntent::from_requirement(&req)
}

fn nl_fixture_task(id: &str, issue: &str) -> LocalBenchmarkTask {
    LocalBenchmarkTask {
        id: id.to_string(),
        kind: RepoTaskKind::Feature,
        issue: issue.to_string(),
        test_command: nl_fixture_cargo_test_command(id).expect("fixture cargo test command"),
        allowed_files: vec!["src/**".to_string()],
        expected_tier_min: HardnessTier::SingleFileBug,
        max_iterations: 3,
    }
}

/// Build canonical `CodeTaskSpec` list for NL fixture harnesses.
pub fn nl_fixture_code_specs(root: impl AsRef<Path>) -> Vec<CodeTaskSpec> {
    let root = root.as_ref();
    nl_synthesis_fixture_suite()
        .into_iter()
        .map(|task| {
            let nl = task
                .issue
                .strip_prefix("synthesize:")
                .map(str::trim)
                .unwrap_or(&task.issue);
            let intent = fixture_intent_from_nl(nl);
            CodeTaskSpec::from_nl(
                root.to_string_lossy(),
                nl,
                intent,
                task.test_command,
                task.allowed_files,
                task.max_iterations,
            )
        })
        .collect()
}

impl RepoBenchmark {
    pub fn from_nl_fixture_suite(root: impl Into<PathBuf>) -> Self {
        let mut benchmark = Self::new(root);
        for task in nl_synthesis_fixture_suite() {
            benchmark.add_task(task);
        }
        benchmark
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_synthesis_fixture_ci_subset_is_holdout_only() {
        let subset = nl_synthesis_fixture_ci_subset();
        assert_eq!(subset.len(), 3);
        let ids: Vec<_> = subset.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&"nl_fixture_triple"));
        assert!(ids.contains(&"nl_fixture_square"));
        assert!(ids.contains(&"nl_fixture_negate"));
    }

    #[test]
    fn nl_synthesis_fixture_suite_covers_unseen_holdout_corpus() {
        let suite = nl_synthesis_fixture_suite();
        assert_eq!(suite.len(), 17);
        let ids: Vec<_> = suite.iter().map(|t| t.id.as_str()).collect();
        for id in [
            "nl_fixture_add",
            "nl_fixture_triple",
            "nl_fixture_square",
            "nl_fixture_negate",
            "nl_fixture_abs",
            "nl_fixture_sum3",
            "nl_fixture_arrsum",
            "nl_fixture_arrmax",
            "nl_fixture_arrlen",
            "nl_fixture_min3",
        ] {
            assert!(ids.contains(&id), "missing fixture {id}");
        }
    }

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
