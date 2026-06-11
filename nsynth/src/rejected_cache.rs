//! Persistent negative memoization — remember programs that FAILED.
//!
//! Counterpart to [`crate::solved_cache`] (which remembers successes). During
//! gradient synthesis the discretizer emits thousands of candidate programs;
//! each one that fails strict verification is deterministic dead weight for
//! that exact example set — verification is a pure function of (code,
//! examples), so a rejected candidate can never become correct on a rerun.
//! Today those rejections live in a per-call `HashSet` and are forgotten when
//! the process exits; every rerun re-parses and re-executes the same dead
//! programs.
//!
//! This module persists rejection fingerprints across runs, keyed by the same
//! `examples_fingerprint` the solved-cache uses. The search literally cannot
//! make the same mistake twice: a candidate whose code hash is in the bank is
//! skipped before the Mog parser ever sees it.
//!
//! Codes are stored as 128-bit double-FNV hashes rather than full source —
//! thousands of rejections per problem stay cheap on disk, and a hash
//! collision merely skips one candidate (the restart cascade explores
//! neighbors, so coverage risk at 2^-128 per pair is ignorable).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Default on-disk location. Override with `NSYNTH_REJECTED_PATH`; setting it
/// to an empty string disables persistence entirely.
fn cache_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_REJECTED_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".nsynth_rejected_programs.tsv"))
}

/// Per-fingerprint cap. A single gradient run can reject a few thousand
/// candidates; we keep the most recent insertions up to this bound.
pub const PER_FP_CAP: usize = 4096;
/// Total fingerprints retained; least-recently-used rows are evicted first.
pub const FP_ROWS_CAP: usize = 512;

/// 128-bit content hash: two independent FNV-1a passes with distinct seeds.
pub fn code_hash(code: &str) -> u128 {
    const SEED_A: u64 = 0xcbf29ce484222325;
    const SEED_B: u64 = 0x9e3779b97f4a7c15;
    const PRIME: u64 = 0x100000001b3;
    let mut a = SEED_A;
    let mut b = SEED_B;
    for byte in code.bytes() {
        a ^= byte as u64;
        a = a.wrapping_mul(PRIME);
        b ^= (byte as u64).rotate_left(17);
        b = b.wrapping_mul(PRIME).rotate_left(5);
    }
    ((a as u128) << 64) | b as u128
}

#[derive(Default, Clone)]
struct Row {
    last_used: u64,
    hashes: BTreeSet<u128>,
}

#[derive(Default)]
struct Store {
    rows: BTreeMap<String, Row>,
}

fn now_epoch_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn escape_fp(fp: &str) -> String {
    let mut out = String::with_capacity(fp.len() + 8);
    for c in fp.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out
}

fn unescape_fp(encoded: &str) -> String {
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

fn encode_store(store: &Store) -> String {
    let mut out = String::new();
    for (fp, row) in &store.rows {
        let hashes: Vec<String> = row.hashes.iter().map(|h| format!("{h:032x}")).collect();
        out.push_str(&format!(
            "{}\t{}\t{}\n",
            row.last_used,
            hashes.join(","),
            escape_fp(fp)
        ));
    }
    out
}

fn decode_store(text: &str) -> Store {
    let mut store = Store::default();
    for line in text.lines() {
        let mut parts = line.splitn(3, '\t');
        let (Some(ts), Some(hashes), Some(fp_enc)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let Ok(last_used) = ts.parse::<u64>() else {
            continue;
        };
        let mut set = BTreeSet::new();
        for h in hashes.split(',') {
            if h.is_empty() {
                continue;
            }
            if let Ok(v) = u128::from_str_radix(h, 16) {
                set.insert(v);
            }
        }
        if set.is_empty() {
            continue;
        }
        store.rows.insert(unescape_fp(fp_enc), Row { last_used, hashes: set });
    }
    store
}

fn load_from(path: &Path) -> Store {
    match std::fs::read_to_string(path) {
        Ok(text) => decode_store(&text),
        Err(_) => Store::default(),
    }
}

fn persist_to(path: &Path, store: &Store) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, encode_store(store));
}

/// Merge new rejections into a store row, enforcing both caps. Pure so the
/// policy is unit-testable without touching disk or globals.
fn merge_into(store: &mut Store, fp: &str, new_hashes: &[u128], now: u64) {
    let row = store.rows.entry(fp.to_string()).or_default();
    row.last_used = now;
    for h in new_hashes {
        row.hashes.insert(*h);
    }
    while row.hashes.len() > PER_FP_CAP {
        let first = *row.hashes.iter().next().expect("non-empty");
        row.hashes.remove(&first);
    }
    while store.rows.len() > FP_ROWS_CAP {
        let oldest = store
            .rows
            .iter()
            .min_by_key(|(_, r)| r.last_used)
            .map(|(k, _)| k.clone())
            .expect("non-empty");
        store.rows.remove(&oldest);
    }
}

static STORE: Mutex<Option<Store>> = Mutex::new(None);

fn with_store<R>(f: impl FnOnce(&mut Store, Option<&Path>) -> R) -> R {
    let path = cache_path();
    let mut guard = STORE.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(match &path {
            Some(p) => load_from(p),
            None => Store::default(),
        });
    }
    f(guard.as_mut().expect("initialized"), path.as_deref())
}

/// Snapshot of every known-bad code hash for this fingerprint.
pub fn rejected_for(fp: &str) -> BTreeSet<u128> {
    with_store(|store, _| store.rows.get(fp).map(|r| r.hashes.clone()).unwrap_or_default())
}

/// Record rejections discovered this run and persist the bank.
pub fn record_rejections(fp: &str, new_hashes: &[u128]) {
    if new_hashes.is_empty() {
        return;
    }
    with_store(|store, path| {
        merge_into(store, fp, new_hashes, now_epoch_seconds());
        if let Some(p) = path {
            persist_to(p, store);
        }
    });
}

/// RAII flush guard: collect rejection hashes during a solve and persist them
/// on every exit path (success, failure, or early return) without threading
/// flush calls through the search loops.
pub struct RejectionRecorder {
    fp: String,
    pub known: BTreeSet<u128>,
    pending: Vec<u128>,
}

impl RejectionRecorder {
    pub fn new(fp: String) -> Self {
        let known = rejected_for(&fp);
        Self { fp, known, pending: Vec::new() }
    }

    /// True when this exact code already failed verification for this
    /// fingerprint in a previous run.
    pub fn known_bad(&self, code: &str) -> bool {
        self.known.contains(&code_hash(code))
    }

    pub fn note_rejection(&mut self, code: &str) {
        let h = code_hash(code);
        if self.known.insert(h) {
            self.pending.push(h);
        }
    }
}

impl Drop for RejectionRecorder {
    fn drop(&mut self) {
        if !self.pending.is_empty() {
            record_rejections(&self.fp, &self.pending);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_hash_is_stable_and_content_sensitive() {
        let a = code_hash("fn f(xs) { return sum(xs); }");
        let b = code_hash("fn f(xs) { return sum(xs); }");
        let c = code_hash("fn f(xs) { return max(xs); }");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn store_round_trips_through_text() {
        let mut store = Store::default();
        merge_into(&mut store, "fp|with~tokens\tand\ttabs", &[1, 2, 3], 100);
        merge_into(&mut store, "other", &[code_hash("bad prog")], 200);
        let text = encode_store(&store);
        let back = decode_store(&text);
        assert_eq!(back.rows.len(), 2);
        let row = back.rows.get("fp|with~tokens\tand\ttabs").expect("escaped fp survives");
        assert_eq!(row.hashes.len(), 3);
        assert_eq!(row.last_used, 100);
        assert!(back.rows.get("other").unwrap().hashes.contains(&code_hash("bad prog")));
    }

    #[test]
    fn per_fingerprint_cap_enforced() {
        let mut store = Store::default();
        let hashes: Vec<u128> = (0..(PER_FP_CAP as u128 + 100)).collect();
        merge_into(&mut store, "fp", &hashes, 1);
        assert_eq!(store.rows.get("fp").unwrap().hashes.len(), PER_FP_CAP);
    }

    #[test]
    fn lru_row_eviction() {
        let mut store = Store::default();
        for i in 0..(FP_ROWS_CAP + 10) {
            merge_into(&mut store, &format!("fp{i}"), &[i as u128], i as u64);
        }
        assert_eq!(store.rows.len(), FP_ROWS_CAP);
        assert!(!store.rows.contains_key("fp0"), "oldest evicted");
        assert!(store.rows.contains_key(&format!("fp{}", FP_ROWS_CAP + 9)));
    }

    #[test]
    fn recorder_skips_known_and_flushes_new() {
        // Pure-struct behavior (persistence disabled under cfg!(test)).
        let mut rec = RejectionRecorder::new("test-fp".to_string());
        assert!(!rec.known_bad("prog A"));
        rec.note_rejection("prog A");
        assert!(rec.known_bad("prog A"));
        assert!(!rec.known_bad("prog B"));
        // Duplicate notes don't double-pend.
        rec.note_rejection("prog A");
        assert_eq!(rec.pending.len(), 1);
    }

    #[test]
    fn disk_round_trip_via_explicit_path() {
        let dir = std::env::temp_dir().join(format!("nsynth_rejected_test_{}", std::process::id()));
        let path = dir.join("bank.tsv");
        let mut store = Store::default();
        merge_into(&mut store, "fp", &[code_hash("x"), code_hash("y")], 42);
        persist_to(&path, &store);
        let back = load_from(&path);
        assert_eq!(back.rows.get("fp").unwrap().hashes.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
