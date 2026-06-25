//! Emergent pipeline ordering via learned method routing.
//!
//! The solver tries methods in a fixed hand-coded order —
//! `enumerative → synthesize_gradient_only → synthesize_array → expr_only →
//! search_teachers → …`. Most problems match *one* of these; the others
//! burn their budget missing before the right one runs.
//!
//! This module watches which method actually wins for each *shape* of
//! problem (category + arg count + array-input flag) and persists those
//! associations. On the next solve, we extract the shape features, consult
//! the router, and return a ranked list of methods most likely to succeed.
//!
//! The `solve_problem` caller can then try the top-ranked method FIRST,
//! before the hardcoded pipeline, and short-circuit on success. When the
//! router is cold (empty bank) or reports low confidence, behavior is
//! unchanged — the hardcoded pipeline still runs as-is.
//!
//! Nothing about this module is specific to any one synthesis stage; it's
//! a thin (features → method) learning layer sitting atop the whole
//! solver. Every success, no matter which stage found it, feeds back.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::benchmark::{Problem, Value};

/// Maximum entries retained on disk. Each row is (feature_key, per-method
/// win counts) — kept bounded so the file doesn't grow without end on
/// long-lived checkouts.
const ROUTER_CAPACITY: usize = 1024;

/// Minimum observations before the router's recommendation is trusted. If
/// a feature bucket has fewer than this many total wins, we skip the
/// short-circuit and fall through to the hardcoded pipeline — prevents the
/// router from overfitting on a single lucky solve.
pub const MIN_CONFIDENCE: u32 = 2;

/// Compact problem-shape feature. Deliberately coarse: we want to
/// *generalize* the routing, not memorize exact problems (the `solved_cache`
/// already handles exact memoization). A new problem in the same "bucket"
/// as many past wins can safely reuse the winning method.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProblemFeatures {
    /// Factory category label ("arrays", "arithmetic", "loops", "strings",
    /// "structs", ...). Hand-authored once per factory; same category
    /// across variants means router hits are near-instant for future
    /// variants of a known factory.
    pub category: String,
    /// Number of formal arguments (not counting the array).
    pub n_args: usize,
    /// True if the first input is a `Value::Array`. Separates scalar-only
    /// problems from array-input problems — these use different pipeline
    /// branches entirely.
    pub has_array: bool,
    /// True if any example input has a `Value::Str`. Pure string problems
    /// always fall to the search-teacher path.
    pub has_string: bool,
    /// True if any example input is a `Value::Pair` / struct.
    pub has_struct: bool,
}

impl ProblemFeatures {
    pub fn of(problem: &Problem) -> Self {
        let (has_array, has_string, has_struct) = match problem.examples.first() {
            Some(first) => {
                let mut arr = false;
                let mut st = false;
                let mut pr = false;
                for ex in &problem.examples {
                    for v in &ex.inputs {
                        match v {
                            Value::Array(_) => arr = true,
                            Value::Str(_) => st = true,
                            Value::Pair(_, _) => pr = true,
                            Value::Quad(_, _, _, _) => pr = true,
                            Value::Tree(_) => pr = true,
                            Value::Tuple(_) => pr = true,
                            Value::Struct(_) => pr = true,
                            Value::Tensor { .. } => {}
                            Value::Int(_) | Value::Float(_) | Value::Bool(_) => {}
                        }
                    }
                }
                // Also check first example's first input specifically to
                // capture "array-at-position-0" which downstream stages
                // care about.
                if matches!(first.inputs.first(), Some(Value::Array(_))) {
                    arr = true;
                }
                (arr, st, pr)
            }
            None => (false, false, false),
        };
        let n_args = problem
            .examples
            .first()
            .map(|e| e.inputs.len())
            .unwrap_or(0);
        Self {
            category: problem.category.to_string(),
            n_args,
            has_array,
            has_string,
            has_struct,
        }
    }

    /// Compact string form for JSON key use and log messages.
    pub fn key(&self) -> String {
        format!(
            "{}|n={}|a={}|s={}|p={}",
            self.category,
            self.n_args,
            self.has_array as u8,
            self.has_string as u8,
            self.has_struct as u8,
        )
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct Bucket {
    /// method_name → win_count
    #[serde(default)]
    wins: BTreeMap<String, u32>,
    /// method_name → miss_count
    #[serde(default)]
    misses: BTreeMap<String, u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MethodRecommendation {
    pub method: String,
    pub wins: u32,
    pub misses: u32,
}

impl MethodRecommendation {
    pub fn attempts(&self) -> u32 {
        self.wins + self.misses
    }
}

/// Persistent feature→method association table.
#[derive(Default, Serialize, Deserialize)]
pub struct MethodRouter {
    buckets: BTreeMap<String, Bucket>,
    #[serde(skip, default)]
    dirty: bool,
}

fn router_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_METHOD_ROUTER_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_method_router.json"))
}

impl MethodRouter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load() -> Self {
        let Some(path) = router_path() else {
            return Self::default();
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    /// Record that `method` solved a problem with `features`. If the same
    /// (features, method) pair already exists, bump its count; else create
    /// the entry. Buckets are evicted lazily once the table exceeds
    /// [`ROUTER_CAPACITY`] — lowest total-win buckets drop first.
    pub fn record_win(&mut self, features: &ProblemFeatures, method: &str) {
        let key = features.key();
        let bucket = self.buckets.entry(key).or_default();
        *bucket.wins.entry(method.to_string()).or_insert(0) += 1;
        self.dirty = true;
        // Lazy cap — don't evict every call, only when we've grown past
        // capacity. Drop the coldest (lowest total-wins) bucket.
        while self.buckets.len() > ROUTER_CAPACITY {
            if let Some(coldest) = self
                .buckets
                .iter()
                .min_by_key(|(_, b)| {
                    b.wins.values().copied().sum::<u32>() + b.misses.values().copied().sum::<u32>()
                })
                .map(|(k, _)| k.clone())
            {
                self.buckets.remove(&coldest);
            } else {
                break;
            }
        }
    }

    pub fn record_miss(&mut self, features: &ProblemFeatures, method: &str) {
        let key = features.key();
        let bucket = self.buckets.entry(key).or_default();
        *bucket.misses.entry(method.to_string()).or_insert(0) += 1;
        self.dirty = true;
        while self.buckets.len() > ROUTER_CAPACITY {
            if let Some(coldest) = self
                .buckets
                .iter()
                .min_by_key(|(_, b)| {
                    b.wins.values().copied().sum::<u32>() + b.misses.values().copied().sum::<u32>()
                })
                .map(|(k, _)| k.clone())
            {
                self.buckets.remove(&coldest);
            } else {
                break;
            }
        }
    }

    /// Return methods sorted by win-count (descending) for this feature
    /// bucket. Empty vec if the bucket is unknown or total wins are below
    /// [`MIN_CONFIDENCE`].
    pub fn recommend_detailed(&self, features: &ProblemFeatures) -> Vec<MethodRecommendation> {
        let Some(bucket) = self.buckets.get(&features.key()) else {
            return Vec::new();
        };
        let total: u32 = bucket.wins.values().copied().sum::<u32>()
            + bucket.misses.values().copied().sum::<u32>();
        if total < MIN_CONFIDENCE {
            return Vec::new();
        }
        let mut ranked: Vec<MethodRecommendation> = bucket
            .wins
            .iter()
            .filter(|(_, wins)| **wins > 0)
            .map(|(method, wins)| MethodRecommendation {
                method: method.clone(),
                wins: *wins,
                misses: bucket.misses.get(method).copied().unwrap_or(0),
            })
            .collect();
        ranked.sort_by(|a, b| {
            let a_attempts = a.attempts();
            let b_attempts = b.attempts();
            let lhs = (a.wins as u64) * (b_attempts as u64);
            let rhs = (b.wins as u64) * (a_attempts as u64);
            rhs.cmp(&lhs)
                .then_with(|| b.wins.cmp(&a.wins))
                .then_with(|| a.misses.cmp(&b.misses))
                .then_with(|| a.method.cmp(&b.method))
        });
        ranked
    }

    pub fn recommend(&self, features: &ProblemFeatures) -> Vec<(String, u32)> {
        self.recommend_detailed(features)
            .into_iter()
            .map(|rec| (rec.method, rec.wins))
            .collect()
    }

    pub fn save(&mut self) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }
        let Some(path) = router_path() else {
            return Ok(());
        };
        let raw = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&path, raw).map_err(|e| format!("write {}: {e}", path.display()))?;
        self.dirty = false;
        Ok(())
    }

    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }
}

// Process-wide singleton.
static ROUTER: Mutex<Option<MethodRouter>> = Mutex::new(None);
#[cfg(test)]
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn with_router<R>(f: impl FnOnce(&mut MethodRouter) -> R) -> R {
    let mut guard = ROUTER.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(MethodRouter::load());
    }
    f(guard.as_mut().expect("router initialized"))
}

/// Record a successful solve keyed by the problem's features.
pub fn record_win(problem: &Problem, method: &str) {
    let features = ProblemFeatures::of(problem);
    with_router(|r| {
        r.record_win(&features, method);
        if let Err(e) = r.save() {
            eprintln!("[method-router] save failed: {e}");
        }
    });
}

pub fn record_miss(problem: &Problem, method: &str) {
    let features = ProblemFeatures::of(problem);
    with_router(|r| {
        r.record_miss(&features, method);
        if let Err(e) = r.save() {
            eprintln!("[method-router] save failed: {e}");
        }
    });
}

/// Return the router's ranked recommendation for this problem, or empty if
/// the bucket is cold / below confidence threshold.
pub fn recommend(problem: &Problem) -> Vec<(String, u32)> {
    let features = ProblemFeatures::of(problem);
    with_router(|r| r.recommend(&features))
}

pub fn recommend_detailed(problem: &Problem) -> Vec<MethodRecommendation> {
    let features = ProblemFeatures::of(problem);
    with_router(|r| r.recommend_detailed(&features))
}

pub fn bucket_count() -> usize {
    with_router(|r| r.bucket_count())
}

pub fn flush() -> (usize, bool) {
    with_router(|r| {
        let n = r.bucket_count();
        let was_dirty = r.dirty;
        if let Err(e) = r.save() {
            eprintln!("[method-router] save failed: {e}");
        }
        (n, was_dirty)
    })
}

#[cfg(test)]
pub fn reset_for_tests() {
    let mut guard = ROUTER.lock().unwrap_or_else(|p| p.into_inner());
    *guard = None;
}

#[cfg(test)]
pub fn with_test_lock<R>(f: impl FnOnce() -> R) -> R {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Example;

    fn with_scratch_router<R>(f: impl FnOnce() -> R) -> R {
        with_test_lock(|| {
            let scratch = std::env::temp_dir().join(format!(
                "nsynth_router_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::env::set_var("NSYNTH_METHOD_ROUTER_PATH", &scratch);
            reset_for_tests();
            let _ = std::fs::remove_file(&scratch);
            let r = f();
            std::env::remove_var("NSYNTH_METHOD_ROUTER_PATH");
            reset_for_tests();
            let _ = std::fs::remove_file(&scratch);
            r
        })
    }

    fn scalar_problem(category: &'static str, n_args: usize) -> Problem {
        let inputs = (0..n_args).map(|i| Value::Int(i as i64)).collect();
        Problem {
            name: "test".to_string(),
            category,
            description: "",
            signature: "fn test()",
            examples: vec![Example {
                inputs,
                expected: Value::Int(0),
            }],
            holdouts: vec![],
            reference_code: "fn test() -> i64 { return 0; }",

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn features_separate_scalar_and_array_shapes() {
        let scalar = scalar_problem("arithmetic", 2);
        let mut array = scalar_problem("arrays", 1);
        array.examples[0].inputs = vec![Value::int_array(&[1, 2, 3])];
        let a = ProblemFeatures::of(&scalar);
        let b = ProblemFeatures::of(&array);
        assert!(!a.has_array);
        assert!(b.has_array);
        assert_ne!(a.key(), b.key());
    }

    #[test]
    fn router_returns_empty_below_min_confidence() {
        with_scratch_router(|| {
            let p = scalar_problem("arithmetic", 1);
            record_win(&p, "enumerative");
            // Only 1 win, below MIN_CONFIDENCE = 2 — no recommendation yet.
            let rec = recommend(&p);
            assert!(rec.is_empty());
        });
    }

    #[test]
    fn router_ranks_by_win_count_once_confident() {
        with_scratch_router(|| {
            let p = scalar_problem("arithmetic", 1);
            // enumerative wins 3 times, search_teacher wins 1 time.
            for _ in 0..3 {
                record_win(&p, "enumerative");
            }
            record_win(&p, "search_teacher");
            let rec = recommend(&p);
            assert_eq!(rec.len(), 2);
            assert_eq!(rec[0].0, "enumerative", "got {:?}", rec);
            assert_eq!(rec[0].1, 3);
            assert_eq!(rec[1].0, "search_teacher");
        });
    }

    #[test]
    fn router_uses_misses_to_demote_weaker_routes() {
        with_scratch_router(|| {
            let p = scalar_problem("arithmetic", 1);
            record_win(&p, "enumerative");
            record_win(&p, "enumerative");
            record_miss(&p, "enumerative");
            record_miss(&p, "enumerative");

            record_win(&p, "search_teacher");
            record_win(&p, "search_teacher");

            let rec = recommend_detailed(&p);
            assert_eq!(rec[0].method, "search_teacher", "got {:?}", rec);
            assert_eq!(rec[0].wins, 2);
            assert_eq!(rec[0].misses, 0);
            assert_eq!(rec[1].method, "enumerative");
            assert_eq!(rec[1].misses, 2);
        });
    }

    #[test]
    fn separate_buckets_do_not_cross_contaminate() {
        with_scratch_router(|| {
            let scalar = scalar_problem("arithmetic", 1);
            let mut array = scalar_problem("arrays", 1);
            array.examples[0].inputs = vec![Value::int_array(&[1])];
            record_win(&scalar, "enumerative");
            record_win(&scalar, "enumerative");
            record_win(&array, "univ_arr_gradient");
            record_win(&array, "univ_arr_gradient");
            let scalar_rec = recommend(&scalar);
            let array_rec = recommend(&array);
            assert_eq!(scalar_rec[0].0, "enumerative");
            assert_eq!(array_rec[0].0, "univ_arr_gradient");
        });
    }
}
