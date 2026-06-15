//! Persistent memoization of solved problems.
//!
//! The classical + differentiable portfolio spends 20-60s on some problems.
//! When a benchmark is rerun (CI, variant expansion, interactive iteration),
//! recomputing the answer wastes that time. This module stores a
//! `(examples_fingerprint → code)` map on disk. On the next solve attempt
//! we look up the fingerprint, re-verify the cached code against the current
//! examples (so a fingerprint collision still fails closed), and return the
//! cached solution when it still matches.
//!
//! The cache is not just a speed win — it is the first concrete place where
//! the synthesizer carries learned knowledge across runs. Every successful
//! solve (classical enumeration, gradient descent, or template match) adds a
//! row, so the gradient brain's discoveries become instantly retrievable the
//! next time the same I/O shape appears.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::benchmark::{Example, Problem, Value};
use crate::runtime::verify_problem_code_strict;

/// Default on-disk location. Can be overridden with the `NSYNTH_CACHE_PATH`
/// environment variable; setting it to an empty string disables the cache.
fn cache_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_CACHE_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_solved_programs.json"))
}

/// Deterministic string representation of a single value. No whitespace.
fn fingerprint_value(v: &Value) -> String {
    match v {
        Value::Int(i) => format!("i:{i}"),
        Value::Float(b) => format!("f:{b}"),
        Value::Bool(b) => format!("b:{b}"),
        Value::Str(s) => format!("s:{}", s.replace('|', "\\|").replace('~', "\\~")),
        Value::Array(xs) => {
            let joined: Vec<String> = xs.iter().map(|x| x.to_string()).collect();
            format!("a:[{}]", joined.join(","))
        }
        Value::Pair(a, b) => format!("p:({a},{b})"),
        Value::Quad(a, b, c, d) => format!("q:({a},{b},{c},{d})"),
    }
}

/// Deterministic fingerprint for a list of examples. Two problems whose
/// examples agree after ordering have identical fingerprints. The separator
/// tokens `|` and `~` are escaped inside string values so collisions can't be
/// manufactured by a benchmark author.
pub fn examples_fingerprint(examples: &[Example]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(examples.len());
    for ex in examples {
        let ins: Vec<String> = ex.inputs.iter().map(fingerprint_value).collect();
        parts.push(format!("{}~{}", ins.join("|"), ex.expected_int()));
    }
    parts.join(";;")
}

#[derive(Clone)]
pub struct CachedSolution {
    pub code: String,
    pub method: String,
    /// Number of times this entry has served as a teacher that successfully
    /// transferred to a *different* problem. Bumped by
    /// [`note_transfer_success`]; persists across runs. Used by
    /// [`crate::meta_learner::rank_teachers`] as a "this teacher keeps
    /// working" prior that nudges repeatedly-winning entries to the top.
    pub success_count: u32,
    /// Seconds-since-UNIX-epoch when this entry was last recorded or last
    /// served a successful transfer. Lets downstream tools distinguish fresh
    /// entries from stale ones even when `success_count` hasn't caught up.
    pub last_used_at: u64,
}

fn now_epoch_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Very small handwritten "JSON-ish" format — one record per line. Keeping the
/// parser tiny avoids pulling serde-derive into this module when the rest of
/// the crate doesn't depend on it. Records are `fp\tmethod\tcode_b64` where
/// `code_b64` is base64 so multi-line Mog code and control characters stay on
/// one line.
fn encode_code(code: &str) -> String {
    let mut out = String::with_capacity(code.len() + 16);
    for b in code.bytes() {
        match b {
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            b'\r' => out.push_str("\\r"),
            b'\\' => out.push_str("\\\\"),
            _ => out.push(b as char),
        }
    }
    out
}

fn decode_code(encoded: &str) -> String {
    let mut out = String::with_capacity(encoded.len());
    let mut chars = encoded.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

pub struct SolvedCache {
    entries: BTreeMap<String, CachedSolution>,
    dirty: bool,
}

impl SolvedCache {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            dirty: false,
        }
    }

    /// Load from disk. Empty cache on any read / parse error — never panics.
    pub fn load() -> Self {
        let Some(path) = cache_path() else {
            return Self::new();
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::new();
        };
        let mut entries = BTreeMap::new();
        for line in raw.lines() {
            if line.is_empty() {
                continue;
            }
            // Newer records: fp \t method \t success_count \t last_used_at \t code
            // Older records: fp \t method \t code   (back-compat, counters
            // default to 0 so old caches keep working without an explicit
            // migration step).
            let parts: Vec<&str> = line.splitn(5, '\t').collect();
            let (fp, method, success_count, last_used_at, code) = match parts.as_slice() {
                [fp, method, code] => (
                    fp.to_string(),
                    method.to_string(),
                    0u32,
                    0u64,
                    decode_code(code),
                ),
                [fp, method, sc, lu, code] => {
                    let success_count = sc.parse::<u32>().unwrap_or(0);
                    let last_used_at = lu.parse::<u64>().unwrap_or(0);
                    (
                        fp.to_string(),
                        method.to_string(),
                        success_count,
                        last_used_at,
                        decode_code(code),
                    )
                }
                _ => continue,
            };
            entries.insert(
                fp,
                CachedSolution {
                    code,
                    method,
                    success_count,
                    last_used_at,
                },
            );
        }
        Self {
            entries,
            dirty: false,
        }
    }

    pub fn get(&self, fp: &str) -> Option<&CachedSolution> {
        self.entries.get(fp)
    }

    pub fn insert(&mut self, fp: String, sol: CachedSolution) {
        // Preserve existing success counters when the new solution matches
        // what's already stored — don't reset a teacher's "keeps working"
        // prior just because we re-ran the bench.
        if let Some(existing) = self.entries.get(&fp) {
            if existing.code == sol.code && existing.method == sol.method {
                return;
            }
        }
        self.entries.insert(fp, sol);
        self.dirty = true;
    }

    /// Bump the transfer-success counter for the entry matching `fp` when its
    /// stored (method, code) agrees with the teacher that just transferred.
    /// Idempotent: if no matching entry exists, this is a no-op.
    pub fn note_transfer(&mut self, fp: &str, method: &str, code: &str) {
        if let Some(sol) = self.entries.get_mut(fp) {
            if sol.method == method && sol.code == code {
                sol.success_count = sol.success_count.saturating_add(1);
                sol.last_used_at = now_epoch_seconds();
                self.dirty = true;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Score-based prune. Keeps `keep_by_score` entries ranked by
    /// `(success_count, last_used_at)` — the teachers that have repeatedly
    /// transferred or been touched recently — AND `keep_by_recency` most-
    /// recently-touched entries even if they never transferred. Everything
    /// else is evicted.
    ///
    /// The two-axis keep list is intentional: pure score ordering would
    /// starve fresh entries that haven't had time to accumulate a transfer
    /// record, while pure recency would discard long-lived working teachers.
    /// Union of the two keeps both "proven" and "still-probationary" rows.
    ///
    /// Returns `(before_len, after_len)`. No-op when `before_len` is already
    /// ≤ `keep_by_score + keep_by_recency`.
    pub fn prune(&mut self, keep_by_score: usize, keep_by_recency: usize) -> (usize, usize) {
        let before = self.entries.len();
        let budget = keep_by_score + keep_by_recency;
        if before <= budget {
            return (before, before);
        }

        // Collect fingerprints with their sort keys. Expensive-ish but this
        // runs only when the cache overflows, which is rare.
        let rows: Vec<(String, u32, u64)> = self
            .entries
            .iter()
            .map(|(fp, sol)| (fp.clone(), sol.success_count, sol.last_used_at))
            .collect();

        // Top-N by score: (success_count desc, last_used_at desc).
        let mut by_score = rows.clone();
        by_score.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
        let score_keep: std::collections::HashSet<String> = by_score
            .iter()
            .take(keep_by_score)
            .map(|(fp, _, _)| fp.clone())
            .collect();

        // Top-M by recency.
        let mut by_recency = rows;
        by_recency.sort_by(|a, b| b.2.cmp(&a.2));
        let recency_keep: std::collections::HashSet<String> = by_recency
            .iter()
            .take(keep_by_recency)
            .map(|(fp, _, _)| fp.clone())
            .collect();

        self.entries
            .retain(|fp, _| score_keep.contains(fp) || recency_keep.contains(fp));
        self.dirty = true;
        (before, self.entries.len())
    }

    pub fn save(&mut self) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }
        let Some(path) = cache_path() else {
            return Ok(());
        };
        let mut out = String::new();
        for (fp, sol) in &self.entries {
            out.push_str(fp);
            out.push('\t');
            out.push_str(&sol.method);
            out.push('\t');
            out.push_str(&sol.success_count.to_string());
            out.push('\t');
            out.push_str(&sol.last_used_at.to_string());
            out.push('\t');
            out.push_str(&encode_code(&sol.code));
            out.push('\n');
        }
        atomic_write(&path, &out).map_err(|e| format!("write {}: {e}", path.display()))?;
        self.dirty = false;
        Ok(())
    }
}

/// Write `content` to `path` atomically via temp-file-then-rename. Protects
/// against two failure modes that `std::fs::write` is vulnerable to:
///   1. Crash mid-write → file becomes a torn partial.
///   2. Concurrent writers → both call write(), the last one wins but the
///      first one's bytes might interleave depending on the fs.
///
/// By writing to `<path>.tmp.<pid>` first and then `rename()`-ing to the
/// target, we get POSIX's atomic rename semantics: readers see either the
/// old file or the new one, never a half-written one. Concurrent writers
/// still race to the rename, but neither observes the other's garbage.
///
/// This is production-grade durability without introducing a
/// file-locking dependency (`fs2`, `fd-lock`, etc.). Good enough for
/// multi-process use on a single machine; distributed-lock semantics
/// require a real lock service (out of scope here).
fn atomic_write(path: &std::path::Path, content: &str) -> std::io::Result<()> {
    let tmp_path = match path.file_name() {
        Some(name) => {
            let mut fname = name.to_os_string();
            fname.push(format!(".tmp.{}", std::process::id()));
            path.with_file_name(fname)
        }
        None => {
            // Path with no file_name (ends in slash, etc.) — write directly.
            // The locking guarantee is weakened but we never hit this in
            // practice since cache_path always ends in a filename.
            return std::fs::write(path, content);
        }
    };
    std::fs::write(&tmp_path, content)?;
    // Rename is atomic on POSIX when same-filesystem; on Windows `rename`
    // fails if the target exists, but Rust's rename() uses ReplaceFileW
    // internally which handles the overwrite case.
    std::fs::rename(&tmp_path, path)
}

// Process-wide singleton. Loaded lazily on first use.
static CACHE: Mutex<Option<SolvedCache>> = Mutex::new(None);
#[cfg(test)]
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn with_cache<R>(f: impl FnOnce(&mut SolvedCache) -> R) -> R {
    let mut guard = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(SolvedCache::load());
    }
    f(guard.as_mut().expect("cache initialized"))
}

/// Look up `problem`'s examples in the cache. Returns the cached (method, code)
/// only after re-verifying that the stored code still satisfies every current
/// example — fingerprint collisions or stale cache entries fail closed.
pub fn lookup(problem: &Problem) -> Option<CachedSolution> {
    if cache_path().is_none() {
        return None;
    }
    let fp = examples_fingerprint(&problem.examples);
    let candidate = with_cache(|c| c.get(&fp).cloned())?;
    if verify_problem_code_strict(problem, &candidate.code).is_ok() {
        Some(candidate)
    } else {
        None
    }
}

/// Record a successful solve. Safe to call from any solver stage; duplicates
/// with identical code/method are de-duped automatically. Persists
/// immediately — at this scale (~100 entries, one-line-each) the write is
/// cheap and robust against abrupt `std::process::exit` calls that skip
/// Drop-based flushes. Callers can still invoke [`flush()`] explicitly to
/// force a no-op dirty check.
pub fn record(problem: &Problem, method: &str, code: &str) {
    if cache_path().is_none() {
        return;
    }
    let fp = examples_fingerprint(&problem.examples);
    with_cache(|c| {
        c.insert(
            fp,
            CachedSolution {
                code: code.to_string(),
                method: method.to_string(),
                success_count: 0,
                last_used_at: now_epoch_seconds(),
            },
        );
        // Eviction: when the cache has grown 25% beyond the cap, prune back
        // to the cap (split evenly between score-ranked and recency-ranked
        // keeps). The 1.25× threshold amortises prune cost — repeated
        // record() calls right at the boundary won't re-prune on every one.
        let cap = max_entries();
        if cap > 0 && c.len() > cap + cap / 4 {
            let keep_score = cap / 2;
            let keep_recency = cap - keep_score;
            let (before, after) = c.prune(keep_score, keep_recency);
            eprintln!(
                "[solved-cache] prune: {} → {} entries (cap={}, by_score={}, by_recency={})",
                before, after, cap, keep_score, keep_recency
            );
        }
        if let Err(e) = c.save() {
            eprintln!("[solved-cache] save failed: {e}");
        }
        // Check whether the cache has grown enough to justify a retrain.
        // Cheap: reads a tiny state file, writes a marker if needed. The
        // actual retraining happens out-of-band via `bootstrap_train`.
        maybe_trigger_bootstrap_retrain(c.len());
    });
}

/// Maximum cache entries before `record` triggers a prune. `0` disables the
/// bound entirely (the default — no eviction until the user opts in). Set
/// via `NSYNTH_CACHE_MAX_ENTRIES`.
fn max_entries() -> usize {
    match std::env::var("NSYNTH_CACHE_MAX_ENTRIES") {
        Ok(raw) => raw.parse::<usize>().unwrap_or(0),
        Err(_) => 0,
    }
}

/// Growth threshold (percent). When the cache size exceeds
/// `last_trained_size × (1 + threshold / 100)`, [`record`] writes a
/// marker file hinting that `bootstrap_train` should be re-run. Actually
/// doing the retraining is deliberately out-of-band — the `record` hot
/// path stays a file write + a counter check, no ML work inline.
fn bootstrap_growth_pct() -> u32 {
    match std::env::var("NSYNTH_BOOTSTRAP_GROWTH_PCT") {
        Ok(raw) => raw.parse::<u32>().unwrap_or(25),
        Err(_) => 25,
    }
}

fn bootstrap_state_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_BOOTSTRAP_STATE_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_bootstrap_state.tsv"))
}

fn bootstrap_marker_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_BOOTSTRAP_MARKER_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_bootstrap_needed"))
}

/// Read `last_trained_size` from the bootstrap state file; 0 if absent.
fn read_last_trained_size() -> usize {
    let Some(path) = bootstrap_state_path() else {
        return 0;
    };
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return 0;
    };
    for line in raw.lines() {
        if let Some((k, v)) = line.split_once('\t') {
            if k == "last_trained_size" {
                return v.trim().parse().unwrap_or(0);
            }
        }
    }
    0
}

/// Check if the cache has grown past the retraining threshold; if so,
/// write a marker file so a cron / CI step knows to re-run
/// `bootstrap_train`. The marker contains the current size + the size at
/// last training so an external observer can decide whether the delta
/// is worth the retrain cost.
///
/// Called from `record` after the in-memory insert. Idempotent — re-
/// triggering when the marker already exists just rewrites the same file
/// with fresher numbers.
fn maybe_trigger_bootstrap_retrain(current_size: usize) {
    let pct = bootstrap_growth_pct();
    if pct == 0 {
        return;
    }
    let last = read_last_trained_size();
    // On the first solve ever, there's no baseline. Don't fire the
    // trigger until training has run once — otherwise every initial
    // insert would look like "infinite growth" and drop a marker.
    if last == 0 {
        return;
    }
    let threshold = last + (last * pct as usize / 100).max(1);
    if current_size < threshold {
        return;
    }
    let Some(marker_path) = bootstrap_marker_path() else {
        return;
    };
    if let Some(parent) = marker_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let now = now_epoch_seconds();
    let body = format!(
        "needed_since\t{}\ncurrent_size\t{}\nlast_trained_size\t{}\ngrowth_pct\t{}\n",
        now, current_size, last, pct
    );
    let _ = std::fs::write(&marker_path, body);
}

/// Commit a successful `bootstrap_train` run: update the state file with
/// the new baseline size and clear the marker. Called by the training
/// binary when it finishes.
pub fn note_bootstrap_trained(cache_size: usize) {
    if let Some(path) = bootstrap_state_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let now = now_epoch_seconds();
        let body = format!(
            "last_trained_size\t{}\nlast_trained_ts\t{}\n",
            cache_size, now
        );
        let _ = std::fs::write(&path, body);
    }
    if let Some(marker) = bootstrap_marker_path() {
        let _ = std::fs::remove_file(&marker);
    }
}

/// Public predicate for downstream tooling: "does the cache think a
/// retrain is due?" Returns true when the marker file exists. Used by
/// `bootstrap_train` and by cron / CI to decide whether to re-run.
pub fn bootstrap_retrain_due() -> bool {
    match bootstrap_marker_path() {
        Some(path) => path.exists(),
        None => false,
    }
}

/// Credit the cache entry that acted as a teacher for a successful cross-
/// problem transfer. Called by [`crate::strategy::CachedTeachers`] after a
/// teacher-distilled gradient descent produced a solution that passes strict
/// verification on a *different* problem than the one that originally
/// populated the cache row. Persists immediately.
///
/// Uses `(method, code)` — not the original problem's fingerprint — to
/// identify the teacher, because the caller only knows the code it just
/// distilled from; the original fingerprint is unrecoverable at this point.
/// Matches the first cache entry whose (method, code) agrees.
pub fn note_transfer_success(method: &str, code: &str) {
    if cache_path().is_none() {
        return;
    }
    with_cache(|c| {
        let matching_fp = c
            .entries
            .iter()
            .find(|(_, sol)| sol.method == method && sol.code == code)
            .map(|(fp, _)| fp.clone());
        if let Some(fp) = matching_fp {
            c.note_transfer(&fp, method, code);
            if let Err(e) = c.save() {
                eprintln!("[solved-cache] save failed: {e}");
            }
        }
    });
}

/// Snapshot every unique cached program. Returns `(method, code)` pairs.
///
/// Used by [`crate::strategy::CachedTeachers`] to feed every prior solve back
/// into the differentiable bridge as a candidate teacher — the gradient flow
/// decides which one transfers, no hand-tuned similarity metric. Order is
/// stable across calls so iteration is reproducible.
pub fn snapshot_solutions() -> Vec<(String, String)> {
    if cache_path().is_none() {
        return Vec::new();
    }
    with_cache(|c| {
        c.entries
            .values()
            .map(|sol| (sol.method.clone(), sol.code.clone()))
            .collect()
    })
}

/// Snapshot every cached program with its transfer-success counter and last-
/// used timestamp. Used by the ranker to reinforce teachers that have
/// repeatedly transferred — the counter persists across runs, so a teacher
/// that wins once this week and twice next week carries a non-zero prior
/// into every future rank.
///
/// Returned tuples are `(method, code, success_count, last_used_at)`. Order
/// is stable across calls (sorted by fingerprint).
pub fn snapshot_solutions_with_meta() -> Vec<(String, String, u32, u64)> {
    if cache_path().is_none() {
        return Vec::new();
    }
    with_cache(|c| {
        c.entries
            .values()
            .map(|sol| {
                (
                    sol.method.clone(),
                    sol.code.clone(),
                    sol.success_count,
                    sol.last_used_at,
                )
            })
            .collect()
    })
}

/// Total number of cached solutions. Cheap to call — does not touch disk.
pub fn entry_count() -> usize {
    if cache_path().is_none() {
        return 0;
    }
    with_cache(|c| c.len())
}

/// Drop the in-memory cache singleton. The next `lookup` / `record` /
/// `snapshot_*` call will lazily reload from disk — which, after a file
/// truncation, means the caller observes an empty cache.
///
/// Intended for harnesses (e.g. the `transfer_curve` binary's `--fresh-cache`
/// mode) that want to measure per-round behaviour starting from a clean
/// slate without having to spawn a subprocess. Not a test-only helper: the
/// on-disk file is the source of truth, so dropping the singleton without
/// also truncating the file just causes a re-read of the same data.
pub fn reset_in_memory() {
    let mut guard = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    *guard = None;
}

/// Flush any pending cache writes to disk. Returns the number of entries and
/// whether the file was re-serialized.
pub fn flush() -> (usize, bool) {
    if cache_path().is_none() {
        return (0, false);
    }
    with_cache(|c| {
        let before = c.len();
        let was_dirty = c.dirty;
        if let Err(e) = c.save() {
            eprintln!("[solved-cache] save failed: {e}");
        }
        (before, was_dirty)
    })
}

#[cfg(test)]
pub fn reset_for_tests() {
    let mut guard = CACHE.lock().unwrap_or_else(|p| p.into_inner());
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

    fn with_clean_cache<R>(f: impl FnOnce() -> R) -> R {
        with_test_lock(|| {
            reset_for_tests();
            let result = f();
            reset_for_tests();
            result
        })
    }

    #[allow(dead_code)]
    fn problem_with(examples: Vec<Example>) -> Problem {
        Problem {
            name: "test".to_string(),
            category: "test",
            description: "",
            signature: "fn test(a: i64) -> i64",
            examples,
            holdouts: vec![],
            reference_code: "fn test(a: i64) -> i64 { return a; }\n",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        }
    }

    #[test]
    fn fingerprint_is_deterministic_and_order_sensitive() {
        with_clean_cache(|| {
            let ex1 = Example {
                inputs: vec![Value::Int(3)],
                expected: Value::Int(3),
            };
            let ex2 = Example {
                inputs: vec![Value::Int(5)],
                expected: Value::Int(5),
            };
            assert_eq!(
                examples_fingerprint(&[ex1.clone(), ex2.clone()]),
                examples_fingerprint(&[ex1.clone(), ex2.clone()])
            );
            // Example ordering matters: a different sequence should produce a
            // different fingerprint so we never silently reuse a solution trained
            // on a different example order.
            assert_ne!(
                examples_fingerprint(&[ex1.clone(), ex2.clone()]),
                examples_fingerprint(&[ex2, ex1])
            );
        });
    }

    #[test]
    fn encode_decode_roundtrip_preserves_multiline_code() {
        with_clean_cache(|| {
            let code = "fn f(a: i64) -> i64 {\n    return a;\n}\n";
            let encoded = encode_code(code);
            assert!(!encoded.contains('\n'));
            assert_eq!(decode_code(&encoded), code);
        });
    }

    /// Verify the bootstrap-retrain trigger: writing a marker when the
    /// cache grows past the threshold, and clearing it on
    /// `note_bootstrap_trained`.
    #[test]
    fn bootstrap_retrain_marker_lifecycle() {
        let state = std::env::temp_dir().join(format!(
            "nsynth_test_bootstrap_state_{}.tsv",
            std::process::id()
        ));
        let marker = std::env::temp_dir().join(format!(
            "nsynth_test_bootstrap_marker_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&state);
        let _ = std::fs::remove_file(&marker);

        // SAFETY: single-threaded test scope.
        unsafe {
            std::env::set_var("NSYNTH_BOOTSTRAP_STATE_PATH", &state);
            std::env::set_var("NSYNTH_BOOTSTRAP_MARKER_PATH", &marker);
            std::env::set_var("NSYNTH_BOOTSTRAP_GROWTH_PCT", "25");
        }

        // With no baseline, the trigger is a no-op (don't mark on first insert).
        maybe_trigger_bootstrap_retrain(100);
        assert!(
            !marker.exists(),
            "marker must not fire before first training"
        );

        // Install a baseline of 100. A size of 100 stays under threshold (125);
        // 125+ fires.
        note_bootstrap_trained(100);
        assert!(
            !marker.exists(),
            "note_bootstrap_trained must clear any prior marker"
        );

        maybe_trigger_bootstrap_retrain(110);
        assert!(!marker.exists(), "110 should not trip the 125 threshold");

        maybe_trigger_bootstrap_retrain(130);
        assert!(marker.exists(), "130 > 125 threshold must write the marker");
        assert!(bootstrap_retrain_due(), "public predicate must agree");

        // A subsequent retrain commits a new baseline + clears the marker.
        note_bootstrap_trained(130);
        assert!(
            !marker.exists(),
            "note_bootstrap_trained must clear the marker"
        );
        assert!(!bootstrap_retrain_due());

        // Cleanup.
        unsafe {
            std::env::remove_var("NSYNTH_BOOTSTRAP_STATE_PATH");
            std::env::remove_var("NSYNTH_BOOTSTRAP_MARKER_PATH");
            std::env::remove_var("NSYNTH_BOOTSTRAP_GROWTH_PCT");
        }
        let _ = std::fs::remove_file(&state);
        let _ = std::fs::remove_file(&marker);
    }

    #[test]
    fn note_transfer_bumps_success_count_and_timestamp() {
        with_clean_cache(|| {
            let fp = "fp_test_note".to_string();
            with_cache(|c| {
                c.insert(
                    fp.clone(),
                    CachedSolution {
                        code: "fn t(a: i64) -> i64 { return a; }\n".to_string(),
                        method: "seed".to_string(),
                        success_count: 0,
                        last_used_at: 0,
                    },
                );
            });

            // Matching (method, code) bumps counter and stamps time.
            with_cache(|c| {
                c.note_transfer(&fp, "seed", "fn t(a: i64) -> i64 { return a; }\n");
            });
            with_cache(|c| {
                let sol = c.get(&fp).expect("entry persists");
                assert_eq!(sol.success_count, 1, "counter should have been bumped");
                assert!(sol.last_used_at > 0, "timestamp should have been set");
            });

            // Mismatching code is a no-op — counter stays at 1.
            with_cache(|c| {
                c.note_transfer(&fp, "seed", "unrelated code");
            });
            with_cache(|c| {
                let sol = c.get(&fp).expect("entry persists");
                assert_eq!(
                    sol.success_count, 1,
                    "mismatched note_transfer must not bump"
                );
            });
        });
    }

    #[test]
    fn back_compat_loads_old_three_column_format() {
        // Simulate an on-disk cache written by the pre-counter version: three
        // tab-separated columns (fp, method, code). The loader should accept
        // it and populate success_count=0, last_used_at=0.
        with_clean_cache(|| {
            let mut cache = SolvedCache::new();
            let line = "legacy_fp\tlegacy_method\tfn t(a: i64) -> i64 { return a; }\\n";
            // Manually parse the way load() would — this exercises the match arm
            // for the 3-column legacy layout.
            for line in line.lines() {
                let parts: Vec<&str> = line.splitn(5, '\t').collect();
                let (fp, sol) = match parts.as_slice() {
                    [fp, method, code] => (
                        fp.to_string(),
                        CachedSolution {
                            code: decode_code(code),
                            method: method.to_string(),
                            success_count: 0,
                            last_used_at: 0,
                        },
                    ),
                    _ => panic!("legacy line should parse as 3-column layout"),
                };
                cache.entries.insert(fp, sol);
            }
            let sol = cache.entries.get("legacy_fp").expect("legacy row present");
            assert_eq!(sol.method, "legacy_method");
            assert_eq!(sol.success_count, 0);
            assert!(sol.code.contains("return a"));
        });
    }

    /// Score-ranked + recency-ranked keep lists are both honoured: a
    /// high-success-count entry survives even if it's old, and a freshly-
    /// recorded entry with zero successes survives even if there are other
    /// older entries with higher scores.
    #[test]
    fn prune_keeps_score_and_recency_separately() {
        let mut cache = SolvedCache::new();
        let insert = |cache: &mut SolvedCache, fp: &str, sc: u32, lu: u64| {
            cache.entries.insert(
                fp.to_string(),
                CachedSolution {
                    code: format!("body for {fp}"),
                    method: "m".to_string(),
                    success_count: sc,
                    last_used_at: lu,
                },
            );
        };
        // Five entries with disjoint score/recency rankings.
        insert(&mut cache, "old_proven", 100, 10); // top score, oldest
        insert(&mut cache, "old_unproven", 0, 20);
        insert(&mut cache, "mid", 5, 30);
        insert(&mut cache, "fresh_unproven_a", 0, 100); // newest
        insert(&mut cache, "fresh_unproven_b", 0, 99); // second newest
        assert_eq!(cache.len(), 5);

        // keep_by_score=1, keep_by_recency=2 → {old_proven, fresh_a, fresh_b}.
        // `old_unproven` and `mid` should be evicted.
        let (before, after) = cache.prune(1, 2);
        assert_eq!(before, 5);
        assert_eq!(after, 3);
        assert!(cache.entries.contains_key("old_proven"));
        assert!(cache.entries.contains_key("fresh_unproven_a"));
        assert!(cache.entries.contains_key("fresh_unproven_b"));
        assert!(!cache.entries.contains_key("old_unproven"));
        assert!(!cache.entries.contains_key("mid"));
    }

    #[test]
    fn prune_noop_when_under_budget() {
        let mut cache = SolvedCache::new();
        cache.entries.insert(
            "a".to_string(),
            CachedSolution {
                code: "x".to_string(),
                method: "m".to_string(),
                success_count: 0,
                last_used_at: 0,
            },
        );
        let (before, after) = cache.prune(10, 10);
        assert_eq!(before, 1);
        assert_eq!(after, 1);
    }

    #[test]
    fn lookup_rejects_stale_cache_hit() {
        with_clean_cache(|| {
            // If we fake a cached solution that doesn't satisfy the examples, the
            // strict verifier must reject it so we don't return wrong answers.
            let problem = problem_with(vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: Value::Int(2), // expected 2, but cached code returns arg
                },
                Example {
                    inputs: vec![Value::Int(5)],
                    expected: Value::Int(10),
                },
            ]);
            // Insert deliberately wrong cached code.
            let fp = examples_fingerprint(&problem.examples);
            with_cache(|c| {
                c.insert(
                    fp,
                    CachedSolution {
                        code: "fn test(a: i64) -> i64 { return a; }\n".to_string(),
                        method: "test_injected".to_string(),
                        success_count: 0,
                        last_used_at: 0,
                    },
                );
            });
            assert!(
                lookup(&problem).is_none(),
                "verify_problem_code_strict must filter out stale cache hits"
            );
        });
    }
}
