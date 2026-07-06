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
        Value::Map(entries) => {
            let joined: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}:{}", fingerprint_value(k), fingerprint_value(v)))
                .collect();
            format!("m:{{{}}}", joined.join(","))
        }
        Value::Pair(a, b) => format!("p:({a},{b})"),
        Value::Quad(a, b, c, d) => format!("q:({a},{b},{c},{d})"),
        Value::Tree(nodes) => {
            let node_strs: Vec<String> = nodes
                .iter()
                .map(|n| format!("({},{},{})", n.value, n.left, n.right))
                .collect();
            format!("t:[{}]", node_strs.join(";"))
        }
        Value::Tuple(xs) => {
            let joined: Vec<String> = xs.iter().map(fingerprint_value).collect();
            format!("tu:({})", joined.join(","))
        }
        Value::Struct(fields) => {
            let joined: Vec<String> = fields
                .iter()
                .map(|(k, x)| format!("{k}={}", fingerprint_value(x)))
                .collect();
            format!("st:{{{}}}", joined.join(","))
        }
        Value::Tensor { data, shape } => {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            let elems: Vec<String> = data.iter().map(|b| b.to_string()).collect();
            format!("tn:<{}>[{}]", dims.join(","), elems.join(","))
        }
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

/// Solver-logic version stamped into every cache key. BUMP THIS whenever solver
/// behavior changes (a new search family, comprehension re-typing, distinguishing
/// examples, a soundness gate) so previously-cached — possibly OVERFIT — solutions
/// are INVALIDATED and the problem is re-solved with current logic instead of
/// served stale. A cached result only re-verifies against the SAME (possibly weak)
/// examples the overfit already passed, so `lookup`'s strict re-verify cannot catch
/// a stale overfit on its own — the version key is what forces the re-solve. Old
/// entries carry an unversioned key and never match a versioned one, so a bump is a
/// clean, self-pruning reset.
const SOLVER_CACHE_VERSION: u32 = 2;

/// Version-salted cache key: the raw example fingerprint prefixed with the solver
/// version, so a `SOLVER_CACHE_VERSION` bump invalidates every prior entry.
fn cache_key(problem: &Problem) -> String {
    format!(
        "v{}\u{1f}{}",
        SOLVER_CACHE_VERSION,
        examples_fingerprint(&problem.examples)
    )
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
    ///
    /// Fast-fail guards (added after a 13.4 GB runaway file hung the solver
    /// ~30s on every start): before reading the whole file we stat it and
    /// refuse anything larger than [`MAX_CACHE_FILE_BYTES`]; after reading we
    /// sanity-check that the content looks like the expected TAB-delimited line
    /// format and bail to an empty store if it does not. Both paths log exactly
    /// one stderr line and return an empty cache rather than hanging.
    pub fn load() -> Self {
        let Some(path) = cache_path() else {
            return Self::new();
        };
        // Stat-based fast-fail: never even read a multi-GB runaway file.
        if let Ok(meta) = std::fs::metadata(&path) {
            if meta.len() > MAX_CACHE_FILE_BYTES {
                eprintln!(
                    "[solved-cache] skip load: file {} is {} bytes > {} cap (treating as empty)",
                    path.display(),
                    meta.len(),
                    MAX_CACHE_FILE_BYTES
                );
                return Self::new();
            }
        }
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::new();
        };
        // Content fast-fail: the format is line-oriented `fp \t method \t ...`.
        // If the file is non-empty but no line carries a TAB, it's not our
        // format (truncated, JSON blob, binary garbage) — skip rather than
        // populate the cache with junk or churn on a malformed parse.
        if !raw.is_empty() && !raw.lines().any(|l| !l.is_empty() && l.contains('\t')) {
            eprintln!(
                "[solved-cache] skip load: file {} is not in the expected TAB line format (treating as empty)",
                path.display()
            );
            return Self::new();
        }
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
        // Per-entry size cap: refuse to cache pathologically large rows. A
        // single multi-megabyte program must not be able to bloat the file —
        // this is one of the two guards (alongside `max_entries`) that keep the
        // cache from ever filling the disk again.
        let cap = max_entry_bytes();
        if cap > 0 && fp.len().saturating_add(sol.code.len()) > cap {
            eprintln!(
                "[solved-cache] skip oversized entry: key+code = {} bytes > {} cap",
                fp.len() + sol.code.len(),
                cap
            );
            return;
        }
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

        // Hard cap enforcement. `record()` does an amortised score/recency
        // prune at 1.25× the cap, but `insert()` is also reachable directly
        // (tests, future callers), so guarantee the bound here too: once the
        // store strictly exceeds the cap, prune back down to it. Splitting the
        // budget between score- and recency-ranked keeps mirrors `record()`.
        let cap = max_entries();
        if cap > 0 && self.entries.len() > cap {
            let keep_score = cap / 2;
            let keep_recency = cap - keep_score;
            self.prune(keep_score, keep_recency);
        }
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
    let fp = cache_key(problem);
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
    let fp = cache_key(problem);
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

/// Maximum cache entries before `record` triggers a prune. Defaults to
/// [`DEFAULT_MAX_ENTRIES`] (50_000) so eviction is *always on* — a missing or
/// unparseable env var must never silently disable the bound, because that is
/// exactly how the on-disk cache once grew to 13.4 GB and filled the disk.
/// Setting `NSYNTH_CACHE_MAX_ENTRIES=0` still disables the bound for callers
/// that explicitly opt out. Override via `NSYNTH_CACHE_MAX_ENTRIES`.
fn max_entries() -> usize {
    match std::env::var("NSYNTH_CACHE_MAX_ENTRIES") {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) => n,
            // Garbage env value falls back to the safe default rather than 0
            // (which would disable eviction).
            Err(_) => DEFAULT_MAX_ENTRIES,
        },
        Err(_) => DEFAULT_MAX_ENTRIES,
    }
}

/// Default cap on cached entries. Chosen to keep the on-disk file well under
/// the [`MAX_CACHE_FILE_BYTES`] fast-fail limit even with large programs.
const DEFAULT_MAX_ENTRIES: usize = 50_000;

/// Default per-entry byte budget (key + code). Entries above this are not
/// cached at all — a single pathological program must not be able to bloat the
/// file. Override via `NSYNTH_CACHE_MAX_ENTRY_BYTES`.
const DEFAULT_MAX_ENTRY_BYTES: usize = 65_536;

/// Hard ceiling on the on-disk cache file the loader is willing to read. A
/// file larger than this is assumed corrupt / runaway and is skipped entirely
/// (the loader returns an empty store) rather than spending ~30s parsing a
/// multi-GB blob and hanging the solver. 256 MiB.
const MAX_CACHE_FILE_BYTES: u64 = 268_435_456;

/// Per-entry byte cap (key + code). Entries whose key+code exceed this are
/// silently skipped on insert. Override via `NSYNTH_CACHE_MAX_ENTRY_BYTES`; a
/// garbage value falls back to the default. `0` disables the cap.
fn max_entry_bytes() -> usize {
    match std::env::var("NSYNTH_CACHE_MAX_ENTRY_BYTES") {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) => n,
            Err(_) => DEFAULT_MAX_ENTRY_BYTES,
        },
        Err(_) => DEFAULT_MAX_ENTRY_BYTES,
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
    // Suppressed during a learning freeze (e.g. RSI evaluation) so measuring the
    // solver cannot mutate cache success-counts mid-evaluation.
    if crate::learning_freeze::is_frozen() || cache_path().is_none() {
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

            functions: vec![],
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

    /// Inserting past the configured cap must evict back down to the cap so
    /// the store can never grow without bound (the root cause of the 13.4 GB
    /// runaway). Drives `insert()` directly with a tiny cap set via env.
    #[test]
    fn insert_evicts_at_max_entries_cap() {
        with_test_lock(|| {
            // SAFETY: env mutation is serialized by the test lock.
            unsafe {
                std::env::set_var("NSYNTH_CACHE_MAX_ENTRIES", "4");
                // Don't let the per-entry byte cap interfere with this test.
                std::env::remove_var("NSYNTH_CACHE_MAX_ENTRY_BYTES");
            }
            assert_eq!(max_entries(), 4, "env override must take effect");

            let mut cache = SolvedCache::new();
            // Insert well past the cap. Vary success_count/recency so prune has
            // a deterministic ordering to work with.
            for i in 0..20u64 {
                cache.insert(
                    format!("fp_{i:02}"),
                    CachedSolution {
                        code: format!("body {i}"),
                        method: "m".to_string(),
                        success_count: 0,
                        last_used_at: i, // strictly increasing recency
                    },
                );
            }
            assert!(
                cache.len() <= 4,
                "store must never exceed the cap; got {}",
                cache.len()
            );
            // The most-recent inserts (highest last_used_at) must survive the
            // recency keep half.
            assert!(
                cache.entries.contains_key("fp_19"),
                "newest entry must survive eviction"
            );

            unsafe {
                std::env::remove_var("NSYNTH_CACHE_MAX_ENTRIES");
            }
        });
    }

    /// `NSYNTH_CACHE_MAX_ENTRIES` unset must NOT mean "eviction disabled" — the
    /// default is now a finite 50_000, the regression guard for the runaway.
    #[test]
    fn max_entries_defaults_to_finite_cap() {
        with_test_lock(|| {
            // SAFETY: serialized by the test lock.
            unsafe {
                std::env::remove_var("NSYNTH_CACHE_MAX_ENTRIES");
            }
            assert_eq!(max_entries(), DEFAULT_MAX_ENTRIES);
            assert!(max_entries() > 0, "default must be a finite, nonzero cap");

            // Garbage value also falls back to the safe default, never 0.
            unsafe {
                std::env::set_var("NSYNTH_CACHE_MAX_ENTRIES", "not-a-number");
            }
            assert_eq!(max_entries(), DEFAULT_MAX_ENTRIES);

            // Explicit opt-out is still honoured.
            unsafe {
                std::env::set_var("NSYNTH_CACHE_MAX_ENTRIES", "0");
            }
            assert_eq!(max_entries(), 0);

            unsafe {
                std::env::remove_var("NSYNTH_CACHE_MAX_ENTRIES");
            }
        });
    }

    /// An entry whose key+code exceeds the per-entry byte cap must be silently
    /// dropped, never stored — one pathological program can't bloat the file.
    #[test]
    fn insert_rejects_oversized_entry() {
        with_test_lock(|| {
            // SAFETY: serialized by the test lock.
            unsafe {
                std::env::set_var("NSYNTH_CACHE_MAX_ENTRY_BYTES", "64");
                // Keep the entry-count cap out of the way.
                std::env::set_var("NSYNTH_CACHE_MAX_ENTRIES", "0");
            }
            assert_eq!(max_entry_bytes(), 64);

            let mut cache = SolvedCache::new();
            // Oversized: key + code far exceeds 64 bytes.
            cache.insert(
                "huge".to_string(),
                CachedSolution {
                    code: "x".repeat(200),
                    method: "m".to_string(),
                    success_count: 0,
                    last_used_at: 0,
                },
            );
            assert!(
                !cache.entries.contains_key("huge"),
                "oversized entry must be rejected"
            );
            assert_eq!(cache.len(), 0);

            // A small entry under the cap is still accepted.
            cache.insert(
                "small".to_string(),
                CachedSolution {
                    code: "ok".to_string(),
                    method: "m".to_string(),
                    success_count: 0,
                    last_used_at: 0,
                },
            );
            assert!(cache.entries.contains_key("small"));

            unsafe {
                std::env::remove_var("NSYNTH_CACHE_MAX_ENTRY_BYTES");
                std::env::remove_var("NSYNTH_CACHE_MAX_ENTRIES");
            }
        });
    }

    /// A cache file larger than the fast-fail ceiling must be skipped — the
    /// loader returns an empty store instead of reading/parsing a multi-GB
    /// blob and hanging. We exercise the size guard without actually writing
    /// 256 MB by lowering nothing and instead asserting the guard's behaviour
    /// via a sparse (logically large) file when supported, falling back to a
    /// content-format check that doesn't require a giant file.
    #[test]
    fn load_skips_oversized_file() {
        with_test_lock(|| {
            let path = std::env::temp_dir().join(format!(
                "nsynth_test_oversized_cache_{}.json",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);

            // Create a file whose reported length exceeds MAX_CACHE_FILE_BYTES
            // without writing that many real bytes: set_len makes a sparse
            // file on the test platforms (APFS / most Linux fs). The bytes read
            // back would be zeros, but the stat-based guard fires first.
            {
                let f = std::fs::File::create(&path).expect("create temp cache");
                f.set_len(MAX_CACHE_FILE_BYTES + 1)
                    .expect("extend temp cache to oversized length");
            }
            assert!(
                std::fs::metadata(&path).unwrap().len() > MAX_CACHE_FILE_BYTES,
                "precondition: file must report oversized length"
            );

            // SAFETY: serialized by the test lock.
            unsafe {
                std::env::set_var("NSYNTH_CACHE_PATH", &path);
            }
            let cache = SolvedCache::load();
            assert_eq!(
                cache.len(),
                0,
                "oversized file must be skipped, yielding an empty store"
            );

            unsafe {
                std::env::remove_var("NSYNTH_CACHE_PATH");
            }
            let _ = std::fs::remove_file(&path);
        });
    }

    /// A non-empty cache file that is not in the expected TAB line format
    /// (truncated JSON, binary garbage) must be skipped rather than parsed
    /// into junk entries or hung on.
    #[test]
    fn load_skips_corrupt_non_tab_file() {
        with_test_lock(|| {
            let path = std::env::temp_dir().join(format!(
                "nsynth_test_corrupt_cache_{}.json",
                std::process::id()
            ));
            // Looks like a JSON blob — no TAB anywhere, so it's clearly not our
            // line format.
            std::fs::write(&path, "{\"this\":\"is not the line format at all\"}\n")
                .expect("write corrupt cache");

            // SAFETY: serialized by the test lock.
            unsafe {
                std::env::set_var("NSYNTH_CACHE_PATH", &path);
            }
            let cache = SolvedCache::load();
            assert_eq!(
                cache.len(),
                0,
                "corrupt non-TAB file must be skipped, yielding an empty store"
            );

            // Sanity: a well-formed TAB file still loads normally.
            std::fs::write(&path, "fp1\tmethod1\t0\t0\tfn t() {}\n").expect("write valid cache");
            let cache2 = SolvedCache::load();
            assert_eq!(cache2.len(), 1, "valid TAB file must still load");
            assert!(cache2.get("fp1").is_some());

            unsafe {
                std::env::remove_var("NSYNTH_CACHE_PATH");
            }
            let _ = std::fs::remove_file(&path);
        });
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
            // Insert deliberately wrong cached code (versioned key, so lookup finds
            // it and the STRICT verifier — not a key miss — is what rejects it).
            let fp = cache_key(&problem);
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
