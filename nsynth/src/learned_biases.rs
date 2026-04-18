//! Self-growing library of initial parameter biases that led to successful
//! gradient synthesis.
//!
//! Today the 26 restart attempts in `synthesize_universal_array_fallback`
//! seed each attempt with one of 20+ hand-designed parameter biases — each
//! `if restart == N { bias_body_slot(..., 0, s0_idx, 0, 1, ...) }` is a
//! pattern the author guessed would be useful (identity accumulator,
//! branched accumulator, two-register loops, …). That's not learning; it's
//! an expert-system prior baked into the source.
//!
//! This module turns those priors into a **runtime-grown library**. Every
//! time a gradient restart succeeds, we snapshot:
//!
//!   * the initial parameter vector (the bias config the restart started
//!     with, before Adam moved anything)
//!   * the `n_scalar` context so we only replay biases on compatible shapes
//!   * a short tag describing where it came from (hand-bias-index N, or
//!     a previous learned-bias hash)
//!
//! The bank is persisted as one JSON line per bias. On the next solve, the
//! gradient loop prefetches the K most-recent matching biases and tries
//! them *before* the hand-coded restarts. Because the initial vector alone
//! determines the entire gradient trajectory (Adam is deterministic given
//! the same examples + init), a learned bias that solved a prior
//! `longest_plateau` problem will likely solve a variant `longest_plateau_v1`
//! in zero gradient steps — the discretization from the verbatim replayed
//! init already fits.
//!
//! The hand-coded biases stay in place as seeds: on a fresh install the
//! bank is empty, so the system bootstraps from expert priors and then
//! replaces/augments them with what actually works in practice.

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Maximum number of biases cached in memory (and written to disk). Anything
/// older than this is dropped — a crude LRU so the bank doesn't grow
/// unbounded on long-lived machines. Bump if you want deeper history.
const BANK_CAPACITY: usize = 256;

/// How many learned biases to replay on each new solve, oldest-first. The
/// remaining `N_UNIV_ARR_RESTARTS - REPLAY_WINDOW` slots fall back to the
/// hand-coded biases.
pub const REPLAY_WINDOW: usize = 8;

#[derive(Clone, Serialize, Deserialize)]
pub struct LearnedBias {
    /// Number of scalar function arguments the original problem had.
    /// Biases are keyed by this so we don't replay a 1-arg init into a
    /// 2-arg parameter shape.
    pub n_scalar: usize,
    /// Byte length of the serialized params vec — cheap sanity check.
    pub n_params: usize,
    /// Raw initial parameter vector, verbatim.
    pub params: Vec<f32>,
    /// Short free-form tag for telemetry (`hand:3`, `random:9f1c`, …).
    pub origin: String,
    /// Unix-timestamp seconds of the originating solve. Oldest entries are
    /// evicted first once the bank reaches [`BANK_CAPACITY`] *and* no
    /// higher-success entries are competing for the same slot.
    pub discovered_at: u64,
    /// Number of subsequent replays that discretized+verified (possibly
    /// after a few warm-refine Adam steps) on *different* examples. Acts as
    /// the "this bias keeps paying off" signal — the LRU is actually a
    /// score-weighted LRU.
    #[serde(default)]
    pub success_count: u32,
    /// Unix-timestamp of the most recent replay hit. Used together with
    /// `success_count` to rank eviction candidates.
    #[serde(default)]
    pub last_used_at: u64,
}

/// On-disk format: one [`LearnedBias`] per line, serialized as JSON.
/// Append-only during a run; a single `save()` at shutdown rewrites the
/// whole file (capped at [`BANK_CAPACITY`]). If the file doesn't exist
/// the bank simply starts empty.
pub struct LearnedBiasBank {
    entries: VecDeque<LearnedBias>,
    dirty: bool,
}

fn bank_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_BIAS_BANK_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_learned_biases.jsonl"))
}

/// Compute a monotone eviction score. Higher = stronger claim to stay in
/// the bank. The formula weights actual replay success heavily (each hit
/// adds 1 << 8 to the score) while giving a small freshness boost so
/// brand-new entries aren't immediately evicted before they have a chance
/// to be tried.
fn bias_score(b: &LearnedBias, now: u64) -> u64 {
    let success = (b.success_count as u64) << 8;
    let age_secs = now.saturating_sub(b.last_used_at);
    // Freshness is a log-scale bonus that decays to 0 over ~24 hours.
    let freshness = if age_secs < 60 {
        32
    } else if age_secs < 600 {
        16
    } else if age_secs < 3600 {
        8
    } else if age_secs < 86_400 {
        4
    } else {
        0
    };
    success + freshness
}

fn now_epoch_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl LearnedBiasBank {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            dirty: false,
        }
    }

    /// Load from disk. Malformed lines are skipped silently so a corrupted
    /// bank never takes down a solve.
    pub fn load() -> Self {
        let Some(path) = bank_path() else {
            return Self::new();
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::new();
        };
        let mut entries = VecDeque::new();
        for line in raw.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_str::<LearnedBias>(line) {
                if entry.params.len() == entry.n_params {
                    entries.push_back(entry);
                }
            }
        }
        while entries.len() > BANK_CAPACITY {
            entries.pop_front();
        }
        Self {
            entries,
            dirty: false,
        }
    }

    pub fn record(&mut self, n_scalar: usize, params: Vec<f32>, origin: impl Into<String>) {
        let now = now_epoch_secs();
        let entry = LearnedBias {
            n_scalar,
            n_params: params.len(),
            params,
            origin: origin.into(),
            discovered_at: now,
            success_count: 0,
            last_used_at: now,
        };
        self.entries.push_back(entry);
        self.evict_to_capacity();
        self.dirty = true;
    }

    /// Called by the replay path when a stored bias discretized+verified on
    /// a new problem. Increments its success_count and bumps its timestamp
    /// so the score-weighted LRU keeps biases that keep working.
    pub fn note_replay_hit(&mut self, origin: &str) {
        let now = now_epoch_secs();
        for entry in self.entries.iter_mut() {
            if entry.origin == origin {
                entry.success_count = entry.success_count.saturating_add(1);
                entry.last_used_at = now;
                self.dirty = true;
                return;
            }
        }
    }

    /// Evict the lowest-scoring entries first when over capacity. Score is
    /// `success_count + freshness`, where freshness compresses age into a
    /// monotone 0..=31 band. Net effect: a bias that's been replayed many
    /// times stays even if older than recent zero-success entries.
    fn evict_to_capacity(&mut self) {
        while self.entries.len() > BANK_CAPACITY {
            let now = now_epoch_secs();
            let (weakest_idx, _) = self
                .entries
                .iter()
                .enumerate()
                .map(|(i, b)| (i, bias_score(b, now)))
                .min_by(|a, b| a.1.cmp(&b.1))
                .expect("non-empty by loop guard");
            self.entries.remove(weakest_idx);
        }
    }

    /// Return up to `limit` biases that match `n_scalar`, highest-scoring
    /// first. Score combines success_count (dominant) with a freshness
    /// bonus, so biases that keep paying off float to the top of the
    /// replay order — the explicit feedback loop "learn what works".
    /// Ties break on insertion order (newest first).
    pub fn recent_matching(&self, n_scalar: usize, limit: usize) -> Vec<&LearnedBias> {
        let now = now_epoch_secs();
        // Iterate in reverse (newest-first) so a stable sort breaks score
        // ties in favor of the more recent entry.
        let mut candidates: Vec<&LearnedBias> = self
            .entries
            .iter()
            .rev()
            .filter(|b| b.n_scalar == n_scalar)
            .collect();
        candidates.sort_by(|a, b| bias_score(b, now).cmp(&bias_score(a, now)));
        candidates.truncate(limit);
        candidates
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn save(&mut self) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }
        let Some(path) = bank_path() else {
            return Ok(());
        };
        let mut handle =
            std::fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
        for entry in &self.entries {
            let line = serde_json::to_string(entry).map_err(|e| format!("serialize: {e}"))?;
            writeln!(handle, "{}", line).map_err(|e| format!("write: {e}"))?;
        }
        self.dirty = false;
        Ok(())
    }
}

// Process-wide singleton. First access lazy-loads from disk; subsequent
// `record()` calls are cheap (in-memory push + set dirty). `flush()` writes
// the whole file once on shutdown.
static BANK: Mutex<Option<LearnedBiasBank>> = Mutex::new(None);

fn with_bank<R>(f: impl FnOnce(&mut LearnedBiasBank) -> R) -> R {
    let mut guard = BANK.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(LearnedBiasBank::load());
    }
    f(guard.as_mut().expect("bank initialized"))
}

/// Snapshot an initial parameter vector that led to a successful gradient
/// solve. Safe to call from any synthesis stage; persisted immediately via
/// [`flush`] after each record so abrupt `std::process::exit` doesn't lose
/// learned state.
pub fn record_success(n_scalar: usize, params: Vec<f32>, origin: impl Into<String>) {
    with_bank(|b| {
        b.record(n_scalar, params, origin);
        if let Err(e) = b.save() {
            eprintln!("[learned-biases] save failed: {e}");
        }
    });
}

/// Record that a previously-stored bias was replayed successfully against a
/// new problem. Bumps its `success_count` and `last_used_at` so the
/// score-weighted LRU keeps biases that keep paying off.
pub fn note_replay_hit(origin: &str) {
    with_bank(|b| {
        b.note_replay_hit(origin);
        if let Err(e) = b.save() {
            eprintln!("[learned-biases] save failed: {e}");
        }
    });
}

/// Return up to `limit` biases matching `n_scalar`, newest first. Each is
/// an owned clone so the caller can mutate the returned `Vec<f32>` without
/// affecting the bank.
pub fn recent_biases(n_scalar: usize, limit: usize) -> Vec<LearnedBias> {
    with_bank(|b| {
        b.recent_matching(n_scalar, limit)
            .into_iter()
            .cloned()
            .collect()
    })
}

/// Number of biases currently in the bank (loaded + in-memory additions).
pub fn len() -> usize {
    with_bank(|b| b.len())
}

/// Flush pending writes. Most callers don't need to invoke this directly —
/// [`record_success`] already persists each addition — but the bench runner
/// and `main` call it defensively on exit.
pub fn flush() -> (usize, bool) {
    with_bank(|b| {
        let n = b.len();
        let was_dirty = b.dirty;
        if let Err(e) = b.save() {
            eprintln!("[learned-biases] save failed: {e}");
        }
        (n, was_dirty)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // All tests share one process-wide singleton (`BANK`) and one process-wide
    // env var (`NSYNTH_BIAS_BANK_PATH`). Parallel test execution would race
    // on both, so we serialize via this test-local lock. Each test that
    // wants a clean bank uses `with_scratch_bank` below.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn with_scratch_bank<R>(f: impl FnOnce() -> R) -> R {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let scratch = std::env::temp_dir().join(format!(
            "nsynth_bias_test_{}_{:?}.jsonl",
            std::process::id(),
            std::thread::current().id(),
        ));
        std::env::set_var("NSYNTH_BIAS_BANK_PATH", &scratch);
        // Reset the singleton so the new path takes effect.
        {
            let mut guard = BANK.lock().unwrap_or_else(|p| p.into_inner());
            *guard = None;
        }
        let _ = std::fs::remove_file(&scratch);
        let r = f();
        let _ = std::fs::remove_file(&scratch);
        r
    }

    #[test]
    fn recent_matching_respects_n_scalar() {
        with_scratch_bank(|| {
            record_success(0, vec![1.0, 2.0, 3.0], "hand:0");
            record_success(1, vec![4.0, 5.0, 6.0], "hand:1");
            let one = recent_biases(0, 10);
            assert_eq!(one.len(), 1);
            assert_eq!(one[0].origin, "hand:0");
            let other = recent_biases(1, 10);
            assert_eq!(other.len(), 1);
            assert_eq!(other[0].origin, "hand:1");
        });
    }

    #[test]
    fn recent_matching_returns_newest_first() {
        with_scratch_bank(|| {
            record_success(0, vec![1.0], "oldest");
            record_success(0, vec![2.0], "middle");
            record_success(0, vec![3.0], "newest");
            let all = recent_biases(0, 3);
            assert_eq!(all.len(), 3);
            assert_eq!(all[0].origin, "newest");
            assert_eq!(all[1].origin, "middle");
            assert_eq!(all[2].origin, "oldest");
        });
    }

    #[test]
    fn persisted_biases_survive_a_reload() {
        with_scratch_bank(|| {
            record_success(2, vec![0.1, 0.2], "pre-reload");
            flush();
            // Drop the singleton to force the next access to re-load from disk.
            {
                let mut guard = BANK.lock().unwrap();
                *guard = None;
            }
            let after = recent_biases(2, 10);
            assert_eq!(after.len(), 1);
            assert_eq!(after[0].origin, "pre-reload");
            assert_eq!(after[0].params, vec![0.1, 0.2]);
        });
    }

    #[test]
    fn replay_hits_promote_bias_above_fresh_zero_success_entries() {
        with_scratch_bank(|| {
            record_success(0, vec![1.0], "veteran");
            // Give the veteran some field wins.
            for _ in 0..4 {
                note_replay_hit("veteran");
            }
            // Now add two brand-new, never-replayed biases.
            record_success(0, vec![2.0], "rookie_a");
            record_success(0, vec![3.0], "rookie_b");
            let top = recent_biases(0, 3);
            assert_eq!(
                top[0].origin, "veteran",
                "a bias with 4 replay hits must beat zero-success newcomers"
            );
            // Tie-break between the two rookies goes to the newer one.
            assert!(
                top[1].origin == "rookie_b",
                "newer rookie should rank above older rookie on freshness tie-break, got {}",
                top[1].origin
            );
        });
    }

    #[test]
    fn eviction_keeps_high_success_entries_over_fresh_zero_success_ones() {
        with_scratch_bank(|| {
            // Seed a single veteran with many wins.
            record_success(0, vec![99.0], "keeper");
            for _ in 0..16 {
                note_replay_hit("keeper");
            }
            // Flood the bank with zero-success newcomers — enough to trigger
            // eviction. The keeper should survive even as it ages.
            for i in 0..(BANK_CAPACITY + 4) {
                record_success(0, vec![i as f32], format!("flood_{i}"));
            }
            assert_eq!(len(), BANK_CAPACITY);
            let top = recent_biases(0, BANK_CAPACITY);
            assert!(
                top.iter().any(|b| b.origin == "keeper"),
                "high-success bias must survive eviction by zero-success newcomers"
            );
        });
    }

    #[test]
    fn bank_caps_at_capacity() {
        with_scratch_bank(|| {
            for i in 0..(BANK_CAPACITY + 32) {
                record_success(0, vec![i as f32], format!("i{i}"));
            }
            assert_eq!(len(), BANK_CAPACITY);
            // Newest entry should still be retained; oldest should have been
            // evicted.
            let newest = recent_biases(0, 1);
            assert_eq!(newest[0].origin, format!("i{}", BANK_CAPACITY + 31));
        });
    }
}
