//! `CodeKnowledgeGraph` — Phase 3.1 knowledge index for analogy-driven synthesis.
//!
//! This is a *thin index over existing data*, never a new store. Its donors are
//! the programs already in [`crate::solved_cache`]; its similarity metric is the
//! exact ranker the trusted `CachedTeachers` path uses
//! ([`crate::meta_learner::rank_teachers_with_meta_topk`]). The only thing the
//! graph adds over re-querying the cache directly is a structured, reusable
//! handle plus build-time caching of each donor's code feature vector (a
//! measured fast-path for a future ranker that consumes pre-extracted vectors).
//!
//! Design constraints (from the adversarially-reviewed Phase 3 spec):
//! - **Never persisted.** Feature vectors are rebuilt every solve, because
//!   `extract_code_features`' slot layout (`FEATURE_DIM`) can change and a
//!   persisted vector would silently mismatch.
//! - **Scoring reuses `meta_learner` only.** `nearest_donors` delegates to
//!   `rank_teachers_with_meta_topk` verbatim, so the graph can never diverge
//!   from the metric `CachedTeachers` already trusts.
//! - **The graph never stores or emits a *solution*.** It only ranks donor
//!   programs; acceptance is decided downstream by the verifier. So it can never
//!   be a source of unverified ("fabricated") code.

use crate::benchmark::Problem;
use crate::meta_learner::{extract_code_features, rank_teachers_with_meta_topk, FEATURE_DIM};

/// A single donor: a program already proven on some prior problem, indexed by
/// its cached feature vector and transfer-credit metadata.
#[derive(Clone, Debug)]
pub struct DonorNode {
    /// The method/route string that produced this program (cache key part).
    pub method: String,
    /// The donor program source.
    pub code: String,
    /// How many times this donor has been credited with a transfer win.
    pub success_count: u32,
    /// Unix seconds of last use (recency, from the cache).
    pub last_used_at: u64,
    /// Feature vector of `code`, extracted once at build time.
    pub code_feat: [f64; FEATURE_DIM],
}

/// A thin, in-memory index over the solved-program cache used as the donor pool
/// for analogy transfer. Rebuilt per solve; never persisted.
#[derive(Clone, Debug, Default)]
pub struct CodeKnowledgeGraph {
    donors: Vec<DonorNode>,
}

impl CodeKnowledgeGraph {
    /// Build the graph from the current solved-program cache. Each donor's
    /// feature vector is extracted once here.
    pub fn build_from_cache() -> Self {
        let donors = crate::solved_cache::snapshot_solutions_with_meta()
            .into_iter()
            .map(|(method, code, success_count, last_used_at)| {
                let code_feat = extract_code_features(&code);
                DonorNode {
                    method,
                    code,
                    success_count,
                    last_used_at,
                    code_feat,
                }
            })
            .collect();
        Self { donors }
    }

    /// Number of indexed donors.
    pub fn len(&self) -> usize {
        self.donors.len()
    }

    /// Whether the graph has no donors (nothing to transfer from).
    pub fn is_empty(&self) -> bool {
        self.donors.is_empty()
    }

    /// Borrow the raw donor nodes (for tests / inspection).
    pub fn donors(&self) -> &[DonorNode] {
        &self.donors
    }

    /// Rank donors by analogical similarity to `problem`, returning the top `k`
    /// (or all when `k == 0`). Delegates byte-for-byte to the trusted
    /// `rank_teachers_with_meta_topk`, so the ordering is identical to what
    /// `CachedTeachers` consumes — the graph is a *view*, not a competing metric.
    ///
    /// `rank_teachers_with_meta_topk` takes its candidates by value and
    /// re-extracts code features internally, so this reconstructs an owned
    /// candidate tuple Vec per call (cheap: hundreds of small rows).
    pub fn nearest_donors(&self, problem: &Problem, k: usize) -> Vec<DonorNode> {
        if self.donors.is_empty() {
            return Vec::new();
        }
        let candidates: Vec<(String, String, u32, u64)> = self
            .donors
            .iter()
            .map(|d| {
                (
                    d.method.clone(),
                    d.code.clone(),
                    d.success_count,
                    d.last_used_at,
                )
            })
            .collect();
        rank_teachers_with_meta_topk(problem, candidates, k)
            .into_iter()
            .map(|(_dist, method, code, success_count, last_used_at)| {
                let code_feat = extract_code_features(&code);
                DonorNode {
                    method,
                    code,
                    success_count,
                    last_used_at,
                    code_feat,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn square_problem() -> Problem {
        Problem {
            name: "square".to_string(),
            category: "arithmetic",
            description: "",
            signature: "fn square(n: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: Value::Int(4),
                },
                Example {
                    inputs: vec![Value::Int(3)],
                    expected: Value::Int(9),
                },
            ],
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    /// Run `f` with a scratch on-disk solved-cache (the cache no-ops under test
    /// unless `NSYNTH_CACHE_PATH` is set). Mirrors the solver test idiom.
    fn with_scratch_cache<R>(f: impl FnOnce() -> R) -> R {
        crate::solved_cache::with_test_lock(|| {
            let cache = std::env::temp_dir().join(format!(
                "nsynth_kg_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::env::set_var("NSYNTH_CACHE_PATH", &cache);
            crate::solved_cache::reset_for_tests();
            let _ = std::fs::remove_file(&cache);
            let result = f();
            std::env::remove_var("NSYNTH_CACHE_PATH");
            crate::solved_cache::reset_for_tests();
            let _ = std::fs::remove_file(&cache);
            result
        })
    }

    #[test]
    fn build_caches_code_features() {
        with_scratch_cache(|| {
            let p = square_problem();
            crate::solved_cache::record(
                &p,
                "search_scalar_expr",
                "fn square(n: i64) -> i64 { return (n * n); }",
            );
            let kg = CodeKnowledgeGraph::build_from_cache();
            assert_eq!(kg.len(), 1, "one donor expected");
            let d = &kg.donors()[0];
            // The cached vector matches a fresh extraction of the same code.
            assert_eq!(d.code_feat, extract_code_features(&d.code));
        });
    }

    #[test]
    fn nearest_donors_matches_trusted_ranker_ordering() {
        with_scratch_cache(|| {
            let p = square_problem();
            crate::solved_cache::record(&p, "m_a", "fn square(n: i64) -> i64 { return (n * n); }");
            // Distinct fingerprints so all three coexist in the cache.
            let mut p2 = square_problem();
            p2.examples[0].inputs = vec![Value::Int(7)];
            crate::solved_cache::record(
                &p2,
                "m_b",
                "fn f(a: i64, b: i64) -> i64 { return (a + b); }",
            );
            let mut p3 = square_problem();
            p3.examples[0].inputs = vec![Value::Int(11)];
            crate::solved_cache::record(&p3, "m_c", "fn g(n: i64) -> i64 { return (n - 1); }");

            let snapshot = crate::solved_cache::snapshot_solutions_with_meta();
            assert_eq!(snapshot.len(), 3, "three donors expected");
            let k = 0;
            let trusted = rank_teachers_with_meta_topk(&p, snapshot, k);
            let kg = CodeKnowledgeGraph::build_from_cache();
            let got = kg.nearest_donors(&p, k);

            // Regression-lock: identical (method, code) sequence — the KG must
            // not introduce a divergent metric.
            let trusted_seq: Vec<(String, String)> =
                trusted.into_iter().map(|(_, m, c, _, _)| (m, c)).collect();
            let got_seq: Vec<(String, String)> =
                got.into_iter().map(|d| (d.method, d.code)).collect();
            assert_eq!(got_seq, trusted_seq);
            assert_eq!(got_seq.len(), 3);
        });
    }
}
