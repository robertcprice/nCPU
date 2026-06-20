use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepoTaskKind {
    BugFix,
    MissingTest,
    Feature,
    RegressionRepair,
    Refactor,
    Documentation,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoSignal {
    pub source: String,
    pub detail: String,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardnessProfile {
    pub localization: f64,
    pub ambiguity: f64,
    pub edit_surface: f64,
    pub test_complexity: f64,
    pub regression_risk: f64,
    pub reasoning_depth: f64,
    pub tool_dependency: f64,
    pub verification_cost: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HardnessTier {
    Trivial,
    SingleFileBug,
    MultiFileBug,
    RegressionRepair,
    FeatureWithTests,
    CrossModuleRefactor,
    SweBenchIssue,
    LongHorizonMigration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoTaskSpec {
    pub id: String,
    pub repo: String,
    pub kind: RepoTaskKind,
    pub issue: String,
    pub test_command: String,
    pub allowed_files: Vec<String>,
    pub max_iterations: usize,
    pub hardness: HardnessProfile,
    pub signals: Vec<RepoSignal>,
}

pub struct HardnessMiner {
    root: PathBuf,
}

impl HardnessMiner {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn scan_repo(&self) -> std::io::Result<Vec<RepoSignal>> {
        let mut signals = Vec::new();
        self.scan_dir(&self.root, &self.root, &mut signals)?;
        Ok(signals)
    }

    pub fn propose_task(
        &self,
        id: impl Into<String>,
        kind: RepoTaskKind,
        issue: impl Into<String>,
        test_command: impl Into<String>,
        allowed_files: Vec<String>,
        max_iterations: usize,
        signals: Vec<RepoSignal>,
    ) -> RepoTaskSpec {
        RepoTaskSpec {
            id: id.into(),
            repo: self.root.to_string_lossy().to_string(),
            kind,
            issue: issue.into(),
            test_command: test_command.into(),
            allowed_files,
            max_iterations,
            hardness: HardnessProfile::from_signals(kind, &signals),
            signals,
        }
    }

    fn scan_dir(
        &self,
        dir: &Path,
        root: &Path,
        signals: &mut Vec<RepoSignal>,
    ) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                self.scan_dir(&path, root, signals)?;
            } else if path.is_file() {
                self.inspect_file(&path, root, signals);
            }
        }
        Ok(())
    }

    fn inspect_file(&self, path: &Path, root: &Path, signals: &mut Vec<RepoSignal>) {
        let Ok(relative) = path.strip_prefix(root) else {
            return;
        };
        let relative = relative.to_string_lossy().replace('\\', "/");
        let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
            return;
        };

        if extension == "rs" && relative.starts_with("src/") {
            signals.push(RepoSignal {
                source: "rust_file".to_string(),
                detail: relative.clone(),
                weight: 0.05,
            });
            if !self.has_rust_test_for(path) {
                signals.push(RepoSignal {
                    source: "untested_rust_file".to_string(),
                    detail: relative.clone(),
                    weight: 0.25,
                });
            }
        }

        let Ok(content) = fs::read_to_string(path) else {
            return;
        };
        let todo_count = content.matches("TODO").count();
        let fixme_count = content.matches("FIXME").count();
        if todo_count > 0 {
            signals.push(RepoSignal {
                source: "todo_marker".to_string(),
                detail: format!("{relative}:{todo_count} TODO markers"),
                weight: (todo_count as f64 * 0.2).min(1.0),
            });
        }
        if fixme_count > 0 {
            signals.push(RepoSignal {
                source: "fixme_marker".to_string(),
                detail: format!("{relative}:{fixme_count} FIXME markers"),
                weight: (fixme_count as f64 * 0.35).min(1.0),
            });
        }
    }

    fn has_rust_test_for(&self, path: &Path) -> bool {
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            return false;
        };
        let tests = self.root.join("tests");
        tests.join(format!("{stem}.rs")).exists()
            || tests.join(format!("{stem}_test.rs")).exists()
            || tests.join(format!("test_{stem}.rs")).exists()
    }
}

impl HardnessProfile {
    pub fn from_signals(kind: RepoTaskKind, signals: &[RepoSignal]) -> Self {
        let weight_sum: f64 = signals.iter().map(|signal| signal.weight).sum();
        let untested = signals
            .iter()
            .filter(|signal| signal.source == "untested_rust_file")
            .count();
        let todo = signals
            .iter()
            .filter(|signal| signal.source == "todo_marker")
            .count();
        let fixme = signals
            .iter()
            .filter(|signal| signal.source == "fixme_marker")
            .count();
        let rust_files = signals
            .iter()
            .filter(|signal| signal.source == "rust_file")
            .count();

        let mut profile = Self {
            localization: (0.15 + (untested as f64 * 0.08) + (todo as f64 * 0.05)).min(1.0),
            ambiguity: 0.35,
            edit_surface: (0.2 + (rust_files as f64 * 0.01) + (untested as f64 * 0.05)).min(1.0),
            test_complexity: (0.25 + (untested as f64 * 0.1)).min(1.0),
            regression_risk: (0.2 + (fixme as f64 * 0.08)).min(1.0),
            reasoning_depth: 0.4,
            tool_dependency: 0.35,
            verification_cost: 0.45,
        };

        match kind {
            RepoTaskKind::BugFix => {
                profile.localization = (profile.localization + 0.2).min(1.0);
                profile.reasoning_depth = (profile.reasoning_depth + 0.2).min(1.0);
            }
            RepoTaskKind::MissingTest => {
                profile.test_complexity = (profile.test_complexity + 0.35).min(1.0);
                profile.verification_cost = (profile.verification_cost + 0.2).min(1.0);
            }
            RepoTaskKind::Feature => {
                profile.edit_surface = (profile.edit_surface + 0.35).min(1.0);
                profile.reasoning_depth = (profile.reasoning_depth + 0.25).min(1.0);
                profile.tool_dependency = (profile.tool_dependency + 0.2).min(1.0);
            }
            RepoTaskKind::RegressionRepair => {
                profile.regression_risk = (profile.regression_risk + 0.45).min(1.0);
                profile.verification_cost = (profile.verification_cost + 0.35).min(1.0);
            }
            RepoTaskKind::Refactor => {
                profile.edit_surface = (profile.edit_surface + 0.45).min(1.0);
                profile.regression_risk = (profile.regression_risk + 0.3).min(1.0);
            }
            RepoTaskKind::Documentation => {
                profile.reasoning_depth = (profile.reasoning_depth - 0.2).max(0.0);
                profile.verification_cost = (profile.verification_cost - 0.2).max(0.0);
            }
            RepoTaskKind::Unknown => {}
        }

        if weight_sum > 3.0 {
            profile.localization = (profile.localization + 0.15).min(1.0);
            profile.verification_cost = (profile.verification_cost + 0.1).min(1.0);
        }
        profile
    }

    pub fn score(&self) -> f64 {
        let total = self.localization
            + self.ambiguity
            + self.edit_surface
            + self.test_complexity
            + self.regression_risk
            + self.reasoning_depth
            + self.tool_dependency
            + self.verification_cost;
        (total / 8.0).clamp(0.0, 1.0)
    }

    pub fn tier(&self) -> HardnessTier {
        match self.score() {
            score if score < 0.2 => HardnessTier::Trivial,
            score if score < 0.35 => HardnessTier::SingleFileBug,
            score if score < 0.42 => HardnessTier::MultiFileBug,
            score if score < 0.54 => HardnessTier::RegressionRepair,
            score if score < 0.65 => HardnessTier::FeatureWithTests,
            score if score < 0.78 => HardnessTier::CrossModuleRefactor,
            score if score < 0.9 => HardnessTier::SweBenchIssue,
            _ => HardnessTier::LongHorizonMigration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardness_tier_increases_with_signal_weight() {
        let signals = vec![
            RepoSignal {
                source: "rust_file".to_string(),
                detail: "src/a.rs".to_string(),
                weight: 0.05,
            },
            RepoSignal {
                source: "untested_rust_file".to_string(),
                detail: "src/a.rs".to_string(),
                weight: 0.25,
            },
            RepoSignal {
                source: "todo_marker".to_string(),
                detail: "src/a.rs".to_string(),
                weight: 0.2,
            },
            RepoSignal {
                source: "fixme_marker".to_string(),
                detail: "src/a.rs".to_string(),
                weight: 0.35,
            },
        ];
        let profile = HardnessProfile::from_signals(RepoTaskKind::RegressionRepair, &signals);
        assert!(
            profile.score()
                > HardnessProfile::from_signals(RepoTaskKind::Documentation, &[]).score()
        );
        assert!(profile.tier() >= HardnessTier::RegressionRepair);
    }

    #[test]
    fn propose_task_enriches_hardness() {
        let tmp = std::env::temp_dir().join(format!("nsynth_hardness_{}", std::process::id()));
        let _ = fs::create_dir_all(tmp.join("src"));
        let _ = fs::create_dir_all(tmp.join("tests"));
        let _ = fs::write(tmp.join("src/lib.rs"), "pub fn a() {}");
        let miner = HardnessMiner::new(&tmp);
        let task = miner.propose_task(
            "task-1",
            RepoTaskKind::MissingTest,
            "add tests",
            "cargo test",
            vec!["src/lib.rs".to_string(), "tests/lib.rs".to_string()],
            3,
            miner.scan_repo().unwrap(),
        );
        assert_eq!(task.id, "task-1");
        assert!(task.hardness.score() > 0.0);
        let _ = fs::remove_dir_all(tmp);
    }
}
