//! Learned ranking inside exact symbolic search.
//!
//! `solve_by_search` historically tries exact search families in a fixed
//! hand-written order. That keeps behavior deterministic, but it also means
//! every problem re-pays the cost of all earlier recognizers even after we
//! have enough evidence that one specific search family usually wins for this
//! bucket of problems.
//!
//! This router is intentionally narrow: it only ranks exact search families.
//! It does not change verification, only the order in which families are
//! attempted.

#[cfg(test)]
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::benchmark::Problem;
use crate::method_router::{MethodRecommendation, ProblemFeatures};

const ROUTER_CAPACITY: usize = 2048;
const MIN_CONFIDENCE: u32 = 2;

#[derive(Default, Clone, Serialize, Deserialize)]
struct Bucket {
    #[serde(default)]
    wins: BTreeMap<String, u32>,
    #[serde(default)]
    misses: BTreeMap<String, u32>,
}

#[derive(Default, Serialize, Deserialize)]
struct SearchFamilyRouter {
    buckets: BTreeMap<String, Bucket>,
    #[serde(skip, default)]
    dirty: bool,
}

fn router_path() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(path) = TEST_ROUTER_PATH.with(|slot| slot.borrow().clone()) {
        return Some(path);
    }
    if let Ok(val) = std::env::var("NSYNTH_SEARCH_FAMILY_ROUTER_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_search_family_router.json"))
}

impl SearchFamilyRouter {
    fn load() -> Self {
        let Some(path) = router_path() else {
            return Self::default();
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    fn save(&mut self) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }
        let Some(path) = router_path() else {
            self.dirty = false;
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create search-family-router dir: {e}"))?;
        }
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize router: {e}"))?;
        std::fs::write(&path, json).map_err(|e| format!("write router {}: {e}", path.display()))?;
        self.dirty = false;
        Ok(())
    }

    fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    fn record_attempt(
        &mut self,
        features: &ProblemFeatures,
        tried: &[&'static str],
        winner: Option<&'static str>,
    ) {
        let key = features.key();
        let bucket = self.buckets.entry(key).or_default();
        let mut saw_winner = false;

        for method in tried {
            if Some(*method) == winner {
                *bucket.wins.entry((*method).to_string()).or_insert(0) += 1;
                saw_winner = true;
            } else {
                *bucket.misses.entry((*method).to_string()).or_insert(0) += 1;
            }
        }

        if let Some(method) = winner.filter(|_| !saw_winner) {
            *bucket.wins.entry(method.to_string()).or_insert(0) += 1;
        }

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

    fn recommend_detailed(&self, features: &ProblemFeatures) -> Vec<MethodRecommendation> {
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
            let lhs = (a.wins as u64) * (b.attempts() as u64);
            let rhs = (b.wins as u64) * (a.attempts() as u64);
            rhs.cmp(&lhs)
                .then_with(|| b.wins.cmp(&a.wins))
                .then_with(|| a.misses.cmp(&b.misses))
                .then_with(|| a.method.cmp(&b.method))
        });
        ranked
    }
}

static ROUTER: Mutex<Option<SearchFamilyRouter>> = Mutex::new(None);

#[cfg(test)]
static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
thread_local! {
    static TEST_ROUTER_PATH: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

fn with_router<R>(f: impl FnOnce(&mut SearchFamilyRouter) -> R) -> Option<R> {
    router_path()?;
    let mut guard = ROUTER.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(SearchFamilyRouter::load());
    }
    Some(f(guard.as_mut().expect("search family router initialized")))
}

pub fn recommend_detailed(problem: &Problem) -> Vec<MethodRecommendation> {
    let features = ProblemFeatures::of(problem);
    with_router(|r| r.recommend_detailed(&features)).unwrap_or_default()
}

pub fn record_attempt(problem: &Problem, tried: &[&'static str], winner: Option<&'static str>) {
    let features = ProblemFeatures::of(problem);
    let _ = with_router(|r| {
        r.record_attempt(&features, tried, winner);
        if let Err(e) = r.save() {
            eprintln!("[search-family-router] save failed: {e}");
        }
    });
}

pub fn bucket_count() -> usize {
    with_router(|r| r.bucket_count()).unwrap_or(0)
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
pub fn with_test_router_path<R>(path: Option<PathBuf>, f: impl FnOnce() -> R) -> R {
    with_test_lock(|| {
        let previous = TEST_ROUTER_PATH.with(|slot| slot.replace(path));
        reset_for_tests();
        let result = f();
        TEST_ROUTER_PATH.with(|slot| {
            slot.replace(previous);
        });
        reset_for_tests();
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn with_scratch_router<R>(f: impl FnOnce() -> R) -> R {
        let scratch = std::env::temp_dir().join(format!(
            "nsynth_search_family_router_test_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id(),
        ));
        with_test_router_path(Some(scratch.clone()), || {
            let _ = std::fs::remove_file(&scratch);
            let r = f();
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
            reference_code: "",

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn router_returns_empty_below_min_confidence() {
        with_scratch_router(|| {
            let p = scalar_problem("arithmetic", 2);
            record_attempt(&p, &["search_scalar_expr"], Some("search_scalar_expr"));
            assert!(recommend_detailed(&p).is_empty());
        });
    }

    #[test]
    fn router_ranks_by_win_rate_once_confident() {
        with_scratch_router(|| {
            let p = scalar_problem("arithmetic", 2);
            record_attempt(&p, &["search_scalar_expr"], Some("search_scalar_expr"));
            record_attempt(&p, &["search_scalar_expr"], Some("search_scalar_expr"));
            record_attempt(&p, &["search_max2_formula"], Some("search_max2_formula"));
            record_attempt(&p, &["search_max2_formula"], None);

            let ranked = recommend_detailed(&p);
            assert_eq!(
                ranked.first().map(|r| r.method.as_str()),
                Some("search_scalar_expr")
            );
        });
    }
}
