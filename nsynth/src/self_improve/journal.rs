//! The reflection journal: the durable record of every self-modification
//! attempt.
//!
//! Each [`JournalEntry`] captures one attempt to close a gap — what gap, what
//! action was tried, by which method, and the three verdicts that decide its
//! fate: whether the synthesized program `verified`, whether it passed the
//! `regression_passed` gate, and whether it was ultimately `accepted`. The
//! journal is append-only and inspectable after the fact, so the system's
//! self-modification history can be audited and so later phases remember what
//! has already been tried.
//!
//! Persistence mirrors `crate::learned_biases`: one
//! `serde_json::to_string(entry)` per line (JSONL), `#[serde(default)]` on every
//! field for forward/back-compat, and malformed lines skipped silently on read.
//! The journal is append-only, so instead of a load-mutate-save singleton it
//! appends each line directly via `OpenOptions::append` — POSIX append writes
//! interleave whole lines under concurrency, and a single serialized entry is
//! small enough to land atomically. The on-disk path comes from
//! `NCPU_JOURNAL_PATH` (empty value disables, so tests / CI never write to `~`),
//! defaulting to `$HOME/.ncpu_reflection_journal.jsonl`.

use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// One recorded self-modification attempt.
///
/// Every field is a plain owned value so the entry serializes to a single JSONL
/// line with no references into the engine or world model. The booleans form the
/// accept/reject decision trail: an attempt that `verified` and whose
/// `regression_passed` is the only kind that should ever be `accepted`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Unix timestamp (seconds) when the attempt was recorded.
    #[serde(default)]
    pub when_unix: u64,
    /// The gap being closed (a short description of the missing capability).
    #[serde(default)]
    pub gap: String,
    /// The action tried to close the gap.
    #[serde(default)]
    pub action: String,
    /// Which method/teacher produced the candidate.
    #[serde(default)]
    pub method: String,
    /// Whether the synthesized program verified against its examples.
    #[serde(default)]
    pub verified: bool,
    /// Whether the candidate passed the regression gate.
    #[serde(default)]
    pub regression_passed: bool,
    /// Whether the change was ultimately accepted.
    #[serde(default)]
    pub accepted: bool,
    /// Free-form note for diagnostics.
    #[serde(default)]
    pub note: String,
}

/// On-disk location of the reflection journal.
///
/// Resolution mirrors `crate::solved_cache::cache_path` and
/// `crate::learned_biases::bank_path`:
///   * `NCPU_JOURNAL_PATH` set to a non-empty value → use it verbatim.
///   * `NCPU_JOURNAL_PATH` set to an *empty* value → `None`, which disables the
///     journal entirely (a no-op). This is what tests / CI set so a run never
///     pollutes the developer's home directory.
///   * Unset → `$HOME/.ncpu_reflection_journal.jsonl` (falling back to the
///     current directory when `HOME` is unavailable).
fn journal_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NCPU_JOURNAL_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".ncpu_reflection_journal.jsonl"))
}

/// Seconds since the UNIX epoch, or 0 if the clock is before it.
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Append one attempt to the durable journal.
///
/// Serializes `entry` to a single JSONL line via `serde_json::to_string` —
/// exactly the encoding `crate::learned_biases::save` uses — and appends it to
/// the journal file. The `when_unix` field is stamped from the system clock at
/// record time, overwriting whatever the caller left in it, so every line
/// carries an authoritative timestamp regardless of how the entry was built.
///
/// The append is done with `OpenOptions::append`, which on POSIX issues each
/// `write` at the current end-of-file: concurrent recorders interleave whole
/// lines rather than corrupting each other, and a single `to_string` line stays
/// well under `PIPE_BUF`, so the line is written atomically in practice. The
/// file is created on first record. All I/O errors are swallowed — the journal
/// is an audit aid, never on a hot correctness path, so a failed write must not
/// take down a self-modification attempt.
///
/// When [`journal_path`] returns `None` (empty `NCPU_JOURNAL_PATH`) this is a
/// no-op.
pub fn record(entry: &JournalEntry) {
    let Some(path) = journal_path() else {
        return;
    };

    let mut stamped = entry.clone();
    stamped.when_unix = now_unix();

    let Ok(mut line) = serde_json::to_string(&stamped) else {
        return;
    };
    line.push('\n');

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Read back all recorded attempts, oldest-first.
///
/// Parses the JSONL file line-by-line with `serde_json::from_str`, skipping
/// blank and malformed lines silently (matching `learned_biases::load`). A
/// missing file — the common case before the first `record` — yields an empty
/// vector, never an error. When [`journal_path`] returns `None` this is also
/// empty.
pub fn entries() -> Vec<JournalEntry> {
    let Some(path) = journal_path() else {
        return Vec::new();
    };
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<JournalEntry>(line) {
            out.push(entry);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes journal tests against each other. They mutate the shared
    /// `NCPU_JOURNAL_PATH` process environment variable, so they must not run
    /// concurrently even though Rust runs `#[test]`s on multiple threads.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Run `f` with `NCPU_JOURNAL_PATH` pointed at a fresh, unique temp file
    /// that is removed before and after. Restores the prior env value on exit.
    fn with_temp_journal<R>(f: impl FnOnce(&std::path::Path) -> R) -> R {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("NCPU_JOURNAL_PATH").ok();
        let path = std::env::temp_dir().join(format!(
            "ncpu_journal_test_{}_{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        // SAFETY: ENV_LOCK guarantees single-threaded access for the duration.
        unsafe {
            std::env::set_var("NCPU_JOURNAL_PATH", &path);
        }
        let result = f(&path);
        match prev {
            Some(v) => unsafe { std::env::set_var("NCPU_JOURNAL_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_JOURNAL_PATH") },
        }
        let _ = std::fs::remove_file(&path);
        result
    }

    fn sample(gap: &str, accepted: bool) -> JournalEntry {
        JournalEntry {
            when_unix: 0, // record() overwrites this with the real clock.
            gap: gap.to_string(),
            action: format!("try {gap}"),
            method: "search_string_equality_map".to_string(),
            verified: true,
            regression_passed: true,
            accepted,
            note: "unit-test entry".to_string(),
        }
    }

    #[test]
    fn missing_file_reads_as_empty() {
        with_temp_journal(|path| {
            // No record() yet → file absent → entries() is empty, not an error.
            assert!(!path.exists());
            assert!(entries().is_empty());
        });
    }

    #[test]
    fn record_two_entries_round_trips_in_order() {
        with_temp_journal(|_path| {
            let a = sample("pluralize irregular nouns", false);
            let b = sample("3sg sibilant verbs", true);
            record(&a);
            record(&b);

            let got = entries();
            assert_eq!(got.len(), 2, "both records must be read back");

            // Oldest-first: a was recorded before b.
            assert_eq!(got[0].gap, a.gap);
            assert_eq!(got[0].action, a.action);
            assert_eq!(got[0].method, a.method);
            assert_eq!(got[0].verified, a.verified);
            assert_eq!(got[0].regression_passed, a.regression_passed);
            assert_eq!(got[0].accepted, a.accepted);
            assert_eq!(got[0].note, a.note);

            assert_eq!(got[1].gap, b.gap);
            assert_eq!(got[1].accepted, b.accepted);

            // when_unix is stamped by record(), so it must be non-zero even
            // though the caller passed 0.
            assert!(got[0].when_unix > 0, "record() must stamp when_unix");
            assert!(got[1].when_unix >= got[0].when_unix);
        });
    }

    #[test]
    fn malformed_lines_are_skipped() {
        with_temp_journal(|path| {
            record(&sample("good entry", true));
            // Inject a blank line and a garbage line directly.
            {
                let mut f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(path)
                    .expect("open journal for corruption");
                f.write_all(b"\nnot valid json at all\n{ partial\n")
                    .unwrap();
            }
            record(&sample("another good entry", false));

            let got = entries();
            assert_eq!(
                got.len(),
                2,
                "only the two well-formed records survive parsing"
            );
            assert_eq!(got[0].gap, "good entry");
            assert_eq!(got[1].gap, "another good entry");
        });
    }

    #[test]
    fn empty_env_disables_record_and_entries() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("NCPU_JOURNAL_PATH").ok();
        // SAFETY: ENV_LOCK held for the duration of this test.
        unsafe {
            std::env::set_var("NCPU_JOURNAL_PATH", "");
        }
        // record() is a no-op and entries() is empty when the env is empty.
        record(&sample("should be dropped", true));
        assert!(
            entries().is_empty(),
            "empty NCPU_JOURNAL_PATH must disable the journal"
        );
        match prev {
            Some(v) => unsafe { std::env::set_var("NCPU_JOURNAL_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_JOURNAL_PATH") },
        }
    }
}
