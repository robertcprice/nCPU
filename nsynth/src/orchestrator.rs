use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::benchmark::Problem;
use crate::differentiable::DifferentiableMetadata;
use crate::interactive::{
    lift_problem_to_interactive, solve_interactive_problem, verify_interactive_program,
};
use crate::runtime::{verify_problem_code_strict, verify_problem_code_via_main};
use crate::solver::{
    solve_problem_differentiable_only, solve_problem_legacy_only,
    solve_problem_prefer_differentiable, solve_problem_search_only,
    solve_problem_with_legacy_fallback,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathwayRecord {
    pub problem_name: String,
    pub family: String,
    pub signature: String,
    pub description: String,
    pub code: String,
    #[serde(default)]
    pub metadata: DifferentiableMetadata,
}

#[derive(Clone, Debug)]
pub struct PathwayMemory {
    root: PathBuf,
    records: Vec<PathwayRecord>,
}

impl PathwayMemory {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)
            .map_err(|err| format!("failed to create memory root {}: {err}", root.display()))?;
        let path = root.join("pathways.json");
        let records = if path.exists() {
            let raw = fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            if raw.trim().is_empty() {
                Vec::new()
            } else {
                serde_json::from_str::<Vec<PathwayRecord>>(&raw)
                    .map_err(|err| format!("failed to parse {}: {err}", path.display()))?
            }
        } else {
            Vec::new()
        };
        Ok(Self { root, records })
    }

    pub fn save(&self) -> Result<(), String> {
        let path = self.root.join("pathways.json");
        let raw = serde_json::to_string_pretty(&self.records)
            .map_err(|err| format!("failed to serialize pathways: {err}"))?;
        fs::write(&path, raw).map_err(|err| format!("failed to write {}: {err}", path.display()))
    }

    pub fn total_successes(&self) -> usize {
        self.records.len()
    }

    pub fn record_success(&mut self, record: PathwayRecord) {
        if let Some(existing) = self
            .records
            .iter_mut()
            .find(|item| item.problem_name == record.problem_name)
        {
            *existing = record;
        } else {
            self.records.push(record);
        }
    }

    pub fn retrieve_similar(
        &self,
        description: &str,
        signature: &str,
        top_k: usize,
    ) -> Vec<PathwayRecord> {
        let query_tokens = tokenize(description);
        let mut scored = self
            .records
            .iter()
            .cloned()
            .map(|record| {
                let mut score = 0.0f64;
                if record.signature == signature {
                    score += 10.0;
                }
                let record_tokens = tokenize(&record.description);
                let overlap = query_tokens.intersection(&record_tokens).count() as f64;
                let union = query_tokens.union(&record_tokens).count() as f64;
                if union > 0.0 {
                    score += overlap / union;
                }
                (score, record)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect::<Vec<_>>();
        scored.sort_by(|(left_score, left_record), (right_score, right_record)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| {
                    pathway_ambiguity_rank(left_record).cmp(&pathway_ambiguity_rank(right_record))
                })
                .then_with(|| left_record.family.cmp(&right_record.family))
        });
        scored
            .into_iter()
            .take(top_k)
            .map(|(_, record)| record)
            .collect()
    }
}

fn pathway_ambiguity_rank(record: &PathwayRecord) -> usize {
    if record.family.contains("diff_gradient_") {
        if record.metadata.recursive_refinement_resolved {
            return 0;
        }
        record.metadata.ambiguity_count
    } else {
        0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrchestratorResult {
    pub success: bool,
    pub method: String,
    pub family: String,
    pub code: String,
    pub error: Option<String>,
    pub metadata: DifferentiableMetadata,
}

pub struct Orchestrator {
    pub memory: PathwayMemory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SolveMode {
    SearchOnly,
    DifferentiableOnly,
    PreferDifferentiable,
    InteractiveDifferentiableOnly,
    SearchThenLegacyFallback,
    LegacyOnly,
}

fn memory_problem_name(problem: &Problem, mode: SolveMode) -> String {
    match mode {
        SolveMode::InteractiveDifferentiableOnly => format!("interactive_{}", problem.name),
        _ => problem.name.clone(),
    }
}

fn memory_signature(problem: &Problem, mode: SolveMode) -> String {
    match mode {
        SolveMode::InteractiveDifferentiableOnly => format!("interactive::{}", problem.signature),
        _ => problem.signature.to_string(),
    }
}

fn memory_description(problem: &Problem, mode: SolveMode) -> String {
    match mode {
        SolveMode::InteractiveDifferentiableOnly => {
            format!("interactive stream version of {}", problem.description)
        }
        _ => problem.description.to_string(),
    }
}

impl Orchestrator {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, String> {
        Ok(Self {
            memory: PathwayMemory::load(root)?,
        })
    }

    pub fn solve(&mut self, problem: &Problem) -> OrchestratorResult {
        self.solve_with_mode(problem, SolveMode::PreferDifferentiable)
    }

    pub fn solve_differentiable_only(&mut self, problem: &Problem) -> OrchestratorResult {
        self.solve_with_mode(problem, SolveMode::DifferentiableOnly)
    }

    pub fn solve_prefer_differentiable(&mut self, problem: &Problem) -> OrchestratorResult {
        self.solve_with_mode(problem, SolveMode::PreferDifferentiable)
    }

    pub fn solve_interactive(&mut self, problem: &Problem) -> OrchestratorResult {
        self.solve_with_mode(problem, SolveMode::InteractiveDifferentiableOnly)
    }

    pub fn solve_search_only(&mut self, problem: &Problem) -> OrchestratorResult {
        self.solve_with_mode(problem, SolveMode::SearchOnly)
    }

    pub fn solve_with_legacy_fallback(&mut self, problem: &Problem) -> OrchestratorResult {
        self.solve_with_mode(problem, SolveMode::SearchThenLegacyFallback)
    }

    pub fn solve_legacy_only(&mut self, problem: &Problem) -> OrchestratorResult {
        self.solve_with_mode(problem, SolveMode::LegacyOnly)
    }

    fn solve_with_mode(&mut self, problem: &Problem, mode: SolveMode) -> OrchestratorResult {
        let query_description = memory_description(problem, mode);
        let query_signature = memory_signature(problem, mode);
        for record in self
            .memory
            .retrieve_similar(&query_description, &query_signature, 8)
            .into_iter()
        {
            let allowed_by_mode = match mode {
                SolveMode::SearchOnly => record.family.starts_with("search_"),
                SolveMode::DifferentiableOnly => record.family.starts_with("diff_gradient_"),
                SolveMode::PreferDifferentiable => record.family.starts_with("diff_gradient_"),
                SolveMode::InteractiveDifferentiableOnly => {
                    record.family.starts_with("interactive_diff_gradient_")
                }
                SolveMode::SearchThenLegacyFallback => true,
                SolveMode::LegacyOnly => record.family.starts_with("legacy_"),
            };
            let verification = if record.family.starts_with("interactive_diff_gradient_") {
                lift_problem_to_interactive(problem)
                    .and_then(|interactive| verify_interactive_program(&interactive, &record.code))
            } else if record.family.starts_with("search_")
                || record.family.starts_with("diff_gradient_")
            {
                verify_problem_code_strict(problem, &record.code)
            } else {
                verify_problem_code_via_main(problem, &record.code)
            };
            if allowed_by_mode && verification.is_ok() {
                return OrchestratorResult {
                    success: true,
                    method: "retrieval".to_string(),
                    family: record.family,
                    code: record.code,
                    error: None,
                    metadata: record.metadata,
                };
            }
        }

        let solved = match mode {
            SolveMode::SearchOnly => solve_problem_search_only(problem),
            SolveMode::DifferentiableOnly => solve_problem_differentiable_only(problem),
            SolveMode::PreferDifferentiable => solve_problem_prefer_differentiable(problem),
            SolveMode::InteractiveDifferentiableOnly => {
                let solved = solve_interactive_problem(problem);
                if solved.success {
                    self.memory.record_success(PathwayRecord {
                        problem_name: memory_problem_name(problem, mode),
                        family: solved.method.clone(),
                        signature: memory_signature(problem, mode),
                        description: memory_description(problem, mode),
                        code: solved.code.clone(),
                        metadata: solved.metadata.clone(),
                    });
                    let _ = self.memory.save();
                    return OrchestratorResult {
                        success: true,
                        method: "search".to_string(),
                        family: solved.method,
                        code: solved.code,
                        error: None,
                        metadata: solved.metadata,
                    };
                }
                return OrchestratorResult {
                    success: false,
                    method: "failed".to_string(),
                    family: solved.method,
                    code: solved.code,
                    error: solved.error,
                    metadata: solved.metadata,
                };
            }
            SolveMode::SearchThenLegacyFallback => solve_problem_with_legacy_fallback(problem),
            SolveMode::LegacyOnly => solve_problem_legacy_only(problem),
        };
        if solved.success {
            self.memory.record_success(PathwayRecord {
                problem_name: memory_problem_name(problem, mode),
                family: solved.method.clone(),
                signature: memory_signature(problem, mode),
                description: memory_description(problem, mode),
                code: solved.code.clone(),
                metadata: solved.metadata.clone(),
            });
            let _ = self.memory.save();
            return OrchestratorResult {
                success: true,
                method: "search".to_string(),
                family: solved.method,
                code: solved.code,
                error: None,
                metadata: solved.metadata,
            };
        }

        OrchestratorResult {
            success: false,
            method: "failed".to_string(),
            family: solved.method,
            code: solved.code,
            error: solved.error,
            metadata: solved.metadata,
        }
    }

    pub fn solve_batch(&mut self, problems: &[Problem]) -> Vec<OrchestratorResult> {
        problems.iter().map(|problem| self.solve(problem)).collect()
    }

    pub fn solve_batch_differentiable_only(
        &mut self,
        problems: &[Problem],
    ) -> Vec<OrchestratorResult> {
        problems
            .iter()
            .map(|problem| self.solve_differentiable_only(problem))
            .collect()
    }

    pub fn solve_batch_prefer_differentiable(
        &mut self,
        problems: &[Problem],
    ) -> Vec<OrchestratorResult> {
        problems
            .iter()
            .map(|problem| self.solve_prefer_differentiable(problem))
            .collect()
    }

    pub fn solve_batch_interactive(&mut self, problems: &[Problem]) -> Vec<OrchestratorResult> {
        problems
            .iter()
            .map(|problem| self.solve_interactive(problem))
            .collect()
    }

    pub fn solve_batch_search_only(&mut self, problems: &[Problem]) -> Vec<OrchestratorResult> {
        problems
            .iter()
            .map(|problem| self.solve_search_only(problem))
            .collect()
    }

    pub fn solve_batch_with_legacy_fallback(
        &mut self,
        problems: &[Problem],
    ) -> Vec<OrchestratorResult> {
        problems
            .iter()
            .map(|problem| self.solve_with_legacy_fallback(problem))
            .collect()
    }

    pub fn solve_batch_legacy_only(&mut self, problems: &[Problem]) -> Vec<OrchestratorResult> {
        problems
            .iter()
            .map(|problem| self.solve_legacy_only(problem))
            .collect()
    }
}

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|part| !part.is_empty())
        .map(|part| part.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::benchmark::get_benchmark;

    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mog_synth_{name}_{nanos}"))
    }

    #[test]
    fn orchestrator_solves_batch() {
        let root = temp_root("solve_batch");
        let problems = get_benchmark(1);
        let mut orchestrator = Orchestrator::new(&root).unwrap();
        let results = orchestrator.solve_batch(&problems[..5]);
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|result| result.success));
    }

    #[test]
    fn orchestrator_default_prefers_differentiable_when_supported() {
        // Opt into the python3 bridge route (Rust-only default keeps it off);
        // the guard serializes against the default-off gate test.
        let _bridge = crate::differentiable::enable_diff_bridge_for_tests();
        let root = temp_root("prefer_differentiable_default");
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();
        let mut orchestrator = Orchestrator::new(&root).unwrap();
        let result = orchestrator.solve(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(result.family.starts_with("diff_gradient_"));
    }

    #[test]
    fn orchestrator_solves_interactive_problem() {
        let root = temp_root("interactive_problem");
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();
        let mut orchestrator = Orchestrator::new(&root).unwrap();
        let result = orchestrator.solve_interactive(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(result.family.starts_with("interactive_diff_gradient_"));
        assert_eq!(orchestrator.memory.total_successes(), 1);
    }

    #[test]
    fn orchestrator_persists_and_retrieves_interactive() {
        let root = temp_root("interactive_persist");
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();

        let mut first = Orchestrator::new(&root).unwrap();
        let solved = first.solve_interactive(&problem);
        assert!(solved.success, "{:?}", solved.error);
        first.memory.save().unwrap();

        let mut second = Orchestrator::new(&root).unwrap();
        let replay = second.solve_interactive(&problem);
        assert!(replay.success, "{:?}", replay.error);
        assert_eq!(replay.method, "retrieval");
        assert!(replay.family.starts_with("interactive_diff_gradient_"));
    }

    #[test]
    fn orchestrator_retrieves_interactive_among_mixed_records() {
        let root = temp_root("interactive_mixed");
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();

        let mut first = Orchestrator::new(&root).unwrap();
        let base = first.solve(&problem);
        assert!(base.success, "{:?}", base.error);
        let interactive = first.solve_interactive(&problem);
        assert!(interactive.success, "{:?}", interactive.error);
        first.memory.save().unwrap();

        let mut second = Orchestrator::new(&root).unwrap();
        let replay = second.solve_interactive(&problem);
        assert!(replay.success, "{:?}", replay.error);
        assert_eq!(replay.method, "retrieval");
        assert!(replay.family.starts_with("interactive_diff_gradient_"));
    }

    #[test]
    fn orchestrator_solves_batch_interactive() {
        let root = temp_root("interactive_batch");
        let problems = get_benchmark(1);
        let mut orchestrator = Orchestrator::new(&root).unwrap();
        let results = orchestrator.solve_batch_interactive(&problems[..2]);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.success));
        assert!(results
            .iter()
            .all(|result| result.family.starts_with("interactive_diff_gradient_")));
    }

    #[test]
    fn orchestrator_persists_and_retrieves() {
        let root = temp_root("persist");
        let problems = get_benchmark(1);

        let mut orchestrator = Orchestrator::new(&root).unwrap();
        let first = orchestrator.solve(&problems[0]);
        assert!(first.success);
        assert_eq!(orchestrator.memory.total_successes(), 1);
        orchestrator.memory.save().unwrap();

        let mut second = Orchestrator::new(&root).unwrap();
        let replay = second.solve(&problems[0]);
        assert!(replay.success);
        assert_eq!(replay.method, "retrieval");
    }

    #[test]
    fn orchestrator_retrieval_prefers_lower_ambiguity_differentiable_record() {
        let root = temp_root("ambiguity_preference");
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();
        let signature = memory_signature(&problem, SolveMode::PreferDifferentiable);
        let description = memory_description(&problem, SolveMode::PreferDifferentiable);
        let code = "fn add_two(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n".to_string();

        let mut first = Orchestrator::new(&root).unwrap();
        first.memory.record_success(PathwayRecord {
            problem_name: "add_two_ambiguous_record".to_string(),
            family: "diff_gradient_ambiguous".to_string(),
            signature: signature.clone(),
            description: description.clone(),
            code: code.clone(),
            metadata: DifferentiableMetadata {
                ambiguity_count: 5,
                exact_alternatives: vec![
                    "diff_gradient_alt_1".to_string(),
                    "diff_gradient_alt_2".to_string(),
                ],
                ..DifferentiableMetadata::default()
            },
        });
        first.memory.record_success(PathwayRecord {
            problem_name: "add_two_unique_record".to_string(),
            family: "diff_gradient_unique".to_string(),
            signature,
            description,
            code,
            metadata: DifferentiableMetadata {
                ambiguity_count: 0,
                exact_alternatives: vec!["diff_gradient_unique".to_string()],
                ..DifferentiableMetadata::default()
            },
        });
        first.memory.save().unwrap();

        let mut second = Orchestrator::new(&root).unwrap();
        let replay = second.solve(&problem);
        assert!(replay.success, "{:?}", replay.error);
        assert_eq!(replay.method, "retrieval");
        assert_eq!(replay.family, "diff_gradient_unique");
        assert_eq!(replay.metadata.ambiguity_count, 0);
        assert_eq!(
            replay.metadata.exact_alternatives,
            vec!["diff_gradient_unique".to_string()]
        );
    }

    #[test]
    fn orchestrator_solves_batch_search_only() {
        let root = temp_root("solve_batch_search_only");
        let problems = get_benchmark(1);
        let mut orchestrator = Orchestrator::new(&root).unwrap();
        let results = orchestrator.solve_batch_search_only(&problems);
        assert_eq!(results.len(), problems.len());
        let failures = problems
            .iter()
            .zip(&results)
            .filter(|(_, result)| !result.success)
            .map(|(problem, result)| {
                format!(
                    "{}: {}",
                    problem.name,
                    result.error.as_deref().unwrap_or("unknown")
                )
            })
            .collect::<Vec<_>>();
        assert!(failures.is_empty(), "search-only failures: {failures:#?}");
        assert!(results
            .iter()
            .all(|result| result.family.starts_with("search_")));
    }

    #[test]
    fn orchestrator_solves_batch_with_legacy_fallback() {
        let root = temp_root("solve_batch_legacy");
        let problems = get_benchmark(1);
        let mut orchestrator = Orchestrator::new(&root).unwrap();
        let results = orchestrator.solve_batch_with_legacy_fallback(&problems);
        assert_eq!(results.len(), problems.len());
        assert!(results.iter().all(|result| result.success));
    }

    #[test]
    #[ignore = "legacy-only cannot solve full search benchmark portfolio — see legacy_only_entrypoint_rejects_reference_oracles"]
    fn orchestrator_solves_batch_legacy_only() {
        let root = temp_root("solve_batch_legacy_only");
        let problems = get_benchmark(1);
        let mut orchestrator = Orchestrator::new(&root).unwrap();
        let results = orchestrator.solve_batch_legacy_only(&problems);
        assert_eq!(results.len(), problems.len());
        assert!(results.iter().all(|result| result.success));
        assert!(results
            .iter()
            .all(|result| result.family.starts_with("legacy_")));
    }
}
