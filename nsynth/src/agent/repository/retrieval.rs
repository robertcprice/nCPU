//! Deterministic path retrieval over `RepoIndex` (Package E).

use super::index::RepoIndex;

/// Score and rank indexed paths against NL issue tokens (path-only, no embedding theater).
pub fn retrieve_paths(index: &RepoIndex, query: &str, limit: usize) -> Vec<String> {
    let tokens = query_tokens(query);
    if tokens.is_empty() {
        return index.rust_sources().into_iter().take(limit).collect();
    }

    let mut scored: Vec<(i32, String)> = index
        .files
        .iter()
        .map(|path| (score_path(path, &tokens), path.clone()))
        .filter(|(score, _)| *score > 0)
        .collect();

    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    scored
        .into_iter()
        .take(limit)
        .map(|(_, path)| path)
        .collect()
}

fn query_tokens(query: &str) -> Vec<String> {
    query
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 3)
        .map(str::to_string)
        .collect()
}

fn score_path(path: &str, tokens: &[String]) -> i32 {
    let lower = path.to_ascii_lowercase();
    let mut score = 0;
    for token in tokens {
        if lower.contains(token) {
            score += 2;
        }
    }
    if lower.ends_with(".rs") {
        score += 1;
    }
    if lower.contains("/src/") {
        score += 1;
    }
    score
}

/// Higher when retrieval narrows the repo to a small ranked file set.
pub fn localization_confidence(retrieved: &[String], index_file_count: usize) -> f64 {
    if index_file_count == 0 {
        return 0.0;
    }
    if retrieved.is_empty() {
        return 0.25;
    }
    let focus = retrieved.len() as f64 / index_file_count as f64;
    (1.0 - focus).clamp(0.35, 0.95)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalCase {
    pub query: String,
    pub expect_any: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalBenchmarkReport {
    pub total: usize,
    pub passed: usize,
    pub failures: Vec<String>,
}

/// Conformance harness for deterministic retrieval (Package E).
pub fn run_retrieval_benchmark(index: &RepoIndex, cases: &[RetrievalCase]) -> RetrievalBenchmarkReport {
    let mut failures = Vec::new();
    let mut passed = 0usize;
    for case in cases {
        let hits = retrieve_paths(index, &case.query, 8);
        let ok = case.expect_any.iter().any(|expected| hits.iter().any(|hit| hit == expected));
        if ok {
            passed += 1;
        } else {
            failures.push(format!(
                "query={} expected one of {:?} got {:?}",
                case.query, case.expect_any, hits
            ));
        }
    }
    RetrievalBenchmarkReport {
        total: cases.len(),
        passed,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::repo::GuardrailPolicy;
    use std::fs;

    fn sample_index() -> RepoIndex {
        let root = std::env::temp_dir().join(format!("nsynth_retrieval_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/multiply.rs"), "multiply").unwrap();
        fs::write(root.join("src/divide.rs"), "divide").unwrap();
        fs::write(root.join("src/reverse.rs"), "reverse").unwrap();
        fs::write(root.join("README.md"), "readme").unwrap();
        let index = RepoIndex::build(&root, &GuardrailPolicy::default()).expect("index");
        index
    }

    #[test]
    fn retrieval_benchmark_hits_expected_paths() {
        let index = sample_index();
        let report = run_retrieval_benchmark(
            &index,
            &[
                RetrievalCase {
                    query: "multiply two numbers".into(),
                    expect_any: vec!["src/multiply.rs".into()],
                },
                RetrievalCase {
                    query: "divide array reverse".into(),
                    expect_any: vec!["src/reverse.rs".into(), "src/divide.rs".into()],
                },
            ],
        );
        assert_eq!(report.passed, report.total);
        assert!(report.failures.is_empty());
        let _ = fs::remove_dir_all(index.root());
    }
}
