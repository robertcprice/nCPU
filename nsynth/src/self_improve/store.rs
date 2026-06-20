//! The learned-component store: the durable, cross-run memory of every
//! component the understanding engine has taught itself.
//!
//! `Engine::new()` rebuilds the base lexicons/rules from scratch every process
//! start, with NO cumulative memory of anything `self_extend` ever grafted on.
//! This store is the spine that closes that gap: every accepted self-extension
//! is persisted as a [`StoredComponent`] (its Mog `name`, `signature`, synthesized
//! `code`, recovering `method`, and the `examples_fingerprint` that characterizes
//! it), so a later phase can re-graft each one back into a fresh engine — gated by
//! the regression gate, exactly like `self_extend` — giving the engine a memory
//! that compounds across runs.
//!
//! ## Persistence idiom (mirrors the rest of the crate)
//!
//! This file follows the SAME on-disk contract as
//! [`crate::solved_cache`], [`crate::self_improve::journal`], and
//! [`crate::learned_biases`]: a JSONL file (one `serde_json::to_string` per line),
//! `#[serde(default)]` on every field for forward/back-compat, malformed lines
//! skipped silently on read, and a path resolved from an environment variable with
//! the **empty-value-disables** convention so tests / CI never read or write a
//! real store:
//!
//!   * `NCPU_COMPONENTS_PATH` set to a non-empty value → use it verbatim.
//!   * `NCPU_COMPONENTS_PATH` set to an *empty* value → `None`, which disables the
//!     store entirely ([`load`] returns empty, [`save_one`] is a no-op). This is
//!     what tests set so a run never pollutes the developer's home directory.
//!   * Unset → `$HOME/.ncpu_learned_components.jsonl` (falling back to the current
//!     directory when `HOME` is unavailable).

use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One learned component, durably recorded so it can be re-grafted into a fresh
/// engine on a later run.
///
/// Every field is a plain owned value so the record serializes to a single JSONL
/// line with no references into the engine. Together they carry everything a
/// reload step needs to reconstruct and re-verify the component:
///
/// * `name` — the Mog function name of the component (e.g. `"creature_class"`).
/// * `signature` — the component's Mog signature (e.g.
///   `"fn creature_class(s: string) -> i64"`).
/// * `code` — the synthesized Mog source for the component.
/// * `method` — the teacher / method that recovered it (provenance).
/// * `examples_fingerprint` — a deterministic fingerprint of the I/O examples
///   that characterize the component, so a reload can detect a stale / collided
///   row and fail closed (mirroring how `solved_cache::lookup` re-verifies before
///   trusting a cached row).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredComponent {
    /// The Mog function name of the component.
    #[serde(default)]
    pub name: String,
    /// The component's Mog signature.
    #[serde(default)]
    pub signature: String,
    /// The synthesized Mog source for the component.
    #[serde(default)]
    pub code: String,
    /// The teacher / method that recovered the component.
    #[serde(default)]
    pub method: String,
    /// Deterministic fingerprint of the characterizing examples.
    #[serde(default)]
    pub examples_fingerprint: String,
    /// The VERIFIED example domain (word, is_member) for an `<x>_class` component,
    /// so a reloaded classifier answers within exactly the evidence it was proven
    /// on — its generalization to unseen words stays UNKNOWN, soundly, across runs.
    /// `#[serde(default)]` keeps older store rows (without this field) loadable.
    #[serde(default)]
    pub members: Vec<(String, bool)>,
}

/// One learned GRAMMAR CONSTRUCTION, durably recorded so it can be re-registered
/// onto a fresh engine on a later run — the word-order analogue of
/// [`StoredComponent`].
///
/// A [`StoredComponent`] persists a synthesized *lexical* program (a function the
/// engine grafts into its Mog source); a `StoredConstruction` persists a
/// *grammatical* role assignment — the [`LearnedConstruction`] the parser's
/// object-fronting fallback consults. Every field is a plain owned value so the
/// record serializes to a single JSONL line with no references into the engine,
/// and every field is `#[serde(default)]` for forward/back-compat (an older store
/// row missing a field still loads, defaulting it).
///
/// Together the fields carry everything a reload step needs to reconstruct the
/// construction and re-gate it:
///
/// * `name` — human-readable tag, e.g. `"object_fronting"`.
/// * `skeletons` — the exact class-code arrays (from
///   [`token_classes`](crate::understanding::semantics::token_classes)) this
///   construction was VERIFIED on; it fires ONLY on a byte-for-byte match.
/// * `agent_idx` / `patient_idx` / `predicate_idx` — the token indices (constant
///   within any recorded skeleton) that fill the agent / patient / predicate slot.
///
/// SOUNDNESS across runs mirrors `StoredComponent`: the reload step re-gates the
/// construction against the *current* golden battery, so a stale / poisoned /
/// now-incompatible row (e.g. a skeleton that COLLIDES with a base-parseable
/// pattern and would change a golden answer) is REJECTED on load and never
/// poisons a fresh boot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredConstruction {
    /// Human-readable tag of the construction (e.g. `"object_fronting"`).
    #[serde(default)]
    pub name: String,
    /// The class skeletons this role assignment is VERIFIED for.
    #[serde(default)]
    pub skeletons: Vec<Vec<i64>>,
    /// Token index filling the AGENT slot.
    #[serde(default)]
    pub agent_idx: usize,
    /// Token index filling the PATIENT slot.
    #[serde(default)]
    pub patient_idx: usize,
    /// Token index of the PREDICATE (lexical verb).
    #[serde(default)]
    pub predicate_idx: usize,
}

/// On-disk location of the learned-CONSTRUCTION store.
///
/// Resolution mirrors [`store_path`] but keys off `NCPU_CONSTRUCTIONS_PATH` so
/// constructions live in their OWN file, distinct from the component store:
///   * `NCPU_CONSTRUCTIONS_PATH` set to a non-empty value → use it verbatim.
///   * `NCPU_CONSTRUCTIONS_PATH` set to an *empty* value → `None`, which disables
///     the construction store entirely (a no-op). This is what tests / CI set so a
///     run never reads or writes a real store.
///   * Unset → `$HOME/.ncpu_learned_constructions.jsonl` (falling back to the
///     current directory when `HOME` is unavailable).
fn construction_store_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NCPU_CONSTRUCTIONS_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".ncpu_learned_constructions.jsonl"))
}

/// Read back every stored construction, oldest-first.
///
/// Same JSONL contract as [`load`]: blank and malformed lines are skipped
/// silently, a missing file yields an empty vector, and a disabled store
/// (empty `NCPU_CONSTRUCTIONS_PATH`) is also empty.
pub fn load_constructions() -> Vec<StoredConstruction> {
    let Some(path) = construction_store_path() else {
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
        if let Ok(construction) = serde_json::from_str::<StoredConstruction>(line) {
            out.push(construction);
        }
    }
    out
}

/// Persist one learned construction, merging by `name`.
///
/// Mirrors [`save_one`]: a re-learned construction with the same `name` REPLACES
/// its prior row (no duplicates), the fresh row lands last (load is oldest-first),
/// the file is rewritten atomically, and all I/O errors are swallowed. A disabled
/// store (empty `NCPU_CONSTRUCTIONS_PATH`) is a no-op.
pub fn save_one_construction(c: &StoredConstruction) {
    let Some(path) = construction_store_path() else {
        return;
    };

    let mut rows: Vec<StoredConstruction> = load_constructions();
    rows.retain(|existing| existing.name != c.name);
    rows.push(c.clone());

    let mut out = String::new();
    for row in &rows {
        let Ok(line) = serde_json::to_string(row) else {
            continue;
        };
        out.push_str(&line);
        out.push('\n');
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let _ = atomic_write(&path, &out);
}

/// Clear the construction store — remove the on-disk file so a later
/// [`load_constructions`] reads empty. Test support / clean-slate primitive; a
/// missing file or a disabled store is a no-op.
pub fn clear_constructions() {
    let Some(path) = construction_store_path() else {
        return;
    };
    let _ = std::fs::remove_file(&path);
}

/// On-disk location of the learned-component store.
///
/// Resolution mirrors `crate::solved_cache::cache_path`,
/// `crate::self_improve::journal::journal_path`, and
/// `crate::learned_biases::bank_path`:
///   * `NCPU_COMPONENTS_PATH` set to a non-empty value → use it verbatim.
///   * `NCPU_COMPONENTS_PATH` set to an *empty* value → `None`, which disables the
///     store entirely (a no-op). This is what tests / CI set so a run never reads
///     or writes a real store.
///   * Unset → `$HOME/.ncpu_learned_components.jsonl` (falling back to the current
///     directory when `HOME` is unavailable).
fn store_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NCPU_COMPONENTS_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Some(PathBuf::from(home).join(".ncpu_learned_components.jsonl"))
}

/// Read back every stored component, oldest-first.
///
/// Parses the JSONL file line-by-line with `serde_json::from_str`, skipping blank
/// and malformed lines silently (matching
/// [`crate::self_improve::journal::entries`] and `crate::learned_biases::load`). A
/// missing file — the common case before the first [`save_one`] — yields an empty
/// vector, never an error. When [`store_path`] returns `None` (empty
/// `NCPU_COMPONENTS_PATH`) this is also empty.
pub fn load() -> Vec<StoredComponent> {
    let Some(path) = store_path() else {
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
        if let Ok(component) = serde_json::from_str::<StoredComponent>(line) {
            out.push(component);
        }
    }
    out
}

/// Persist one learned component, merging by `name`.
///
/// A re-learned component with the same `name` REPLACES its prior row rather than
/// accumulating duplicates, so the store carries one row per distinct component
/// name and `load()` never re-grafts the same name twice. We read the existing
/// rows, drop any with a matching name, append the fresh one (so it lands last —
/// `load()` is oldest-first and a refreshed component should reload after the
/// untouched ones), and rewrite the whole file atomically (temp-file-then-rename,
/// mirroring `crate::solved_cache::atomic_write`).
///
/// All I/O errors are swallowed — the store is durable memory, never on a hot
/// correctness path: a failed write must not take down a self-modification
/// attempt (the component is still live on the in-process engine that just
/// accepted it). When [`store_path`] returns `None` (empty `NCPU_COMPONENTS_PATH`)
/// this is a no-op, so tests / CI never touch a real store.
pub fn save_one(c: &StoredComponent) {
    let Some(path) = store_path() else {
        return;
    };

    // Read the current rows, drop any with the same name (merge-by-name), then
    // append the fresh component last.
    let mut rows: Vec<StoredComponent> = load();
    rows.retain(|existing| existing.name != c.name);
    rows.push(c.clone());

    let mut out = String::new();
    for row in &rows {
        let Ok(line) = serde_json::to_string(row) else {
            continue;
        };
        out.push_str(&line);
        out.push('\n');
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let _ = atomic_write(&path, &out);
}

/// Clear the store — remove the on-disk file so a later [`load`] reads empty.
///
/// Test support, and a clean-slate primitive for harnesses that want to start
/// from no learned memory. A missing file or a disabled store ([`store_path`] →
/// `None`) is a no-op — there is nothing to clear and removing an absent file is
/// not an error.
pub fn clear() {
    let Some(path) = store_path() else {
        return;
    };
    let _ = std::fs::remove_file(&path);
}

/// Write `content` to `path` atomically via temp-file-then-rename. Mirrors
/// `crate::solved_cache::atomic_write`: writing to `<path>.tmp.<pid>` first and
/// then `rename()`-ing to the target gives POSIX's atomic-rename semantics, so a
/// concurrent reader sees either the old file or the new one, never a half-written
/// store.
fn atomic_write(path: &std::path::Path, content: &str) -> std::io::Result<()> {
    let tmp_path = match path.file_name() {
        Some(name) => {
            let mut fname = name.to_os_string();
            fname.push(format!(".tmp.{}", std::process::id()));
            path.with_file_name(fname)
        }
        None => return std::fs::write(path, content),
    };
    {
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(content.as_bytes())?;
        f.flush()?;
    }
    std::fs::rename(&tmp_path, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comprehension::{creature_class_examples, Engine};
    use crate::self_improve::gate::regression_gate;
    use crate::solved_cache::examples_fingerprint;

    /// Run `f` with `NCPU_COMPONENTS_PATH` pointed at a fresh, unique temp file
    /// removed before and after, holding the crate-wide journal env lock so the
    /// process-global env mutation never races another env-mutating test. The
    /// journal lock is reused (not a store-specific one) because the
    /// `self_improve`/`comprehension` reload path also touches `NCPU_JOURNAL_PATH`
    /// indirectly via the gate's journaling, and serializing all of them on ONE
    /// lock is the crate's established convention (see `journal::test_support`).
    fn with_temp_store<R>(f: impl FnOnce(&std::path::Path) -> R) -> R {
        use crate::self_improve::journal::test_support::ENV_LOCK;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("NCPU_COMPONENTS_PATH").ok();
        // Also disable the journal so the reload-through-gate path never writes
        // to $HOME during the test.
        let prev_journal = std::env::var("NCPU_JOURNAL_PATH").ok();
        let path = std::env::temp_dir().join(format!(
            "ncpu_components_test_{}_{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        // SAFETY: ENV_LOCK guarantees single-threaded access for the duration.
        unsafe {
            std::env::set_var("NCPU_COMPONENTS_PATH", &path);
            std::env::set_var("NCPU_JOURNAL_PATH", "");
        }
        let result = f(&path);
        match prev {
            Some(v) => unsafe { std::env::set_var("NCPU_COMPONENTS_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_COMPONENTS_PATH") },
        }
        match prev_journal {
            Some(v) => unsafe { std::env::set_var("NCPU_JOURNAL_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_JOURNAL_PATH") },
        }
        let _ = std::fs::remove_file(&path);
        result
    }

    fn sample(name: &str, code: &str) -> StoredComponent {
        StoredComponent {
            name: name.to_string(),
            signature: format!("fn {name}(s: string) -> i64"),
            code: code.to_string(),
            method: "search_string_equality_map".to_string(),
            examples_fingerprint: "fp-test".to_string(),
            members: Vec::new(),
        }
    }

    #[test]
    fn missing_file_loads_as_empty() {
        with_temp_store(|path| {
            assert!(!path.exists());
            assert!(load().is_empty(), "no file → empty load, not an error");
        });
    }

    #[test]
    fn empty_env_disables_store() {
        use crate::self_improve::journal::test_support::ENV_LOCK;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("NCPU_COMPONENTS_PATH").ok();
        // SAFETY: ENV_LOCK held for the duration.
        unsafe {
            std::env::set_var("NCPU_COMPONENTS_PATH", "");
        }
        save_one(&sample(
            "x_class",
            "fn x_class(s: string) -> i64 { return 0; }\n",
        ));
        assert!(
            load().is_empty(),
            "empty NCPU_COMPONENTS_PATH must disable the store"
        );
        match prev {
            Some(v) => unsafe { std::env::set_var("NCPU_COMPONENTS_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_COMPONENTS_PATH") },
        }
    }

    #[test]
    fn save_one_round_trips_and_merges_by_name() {
        with_temp_store(|_path| {
            let a = sample("a_class", "fn a_class(s: string) -> i64 { return 1; }\n");
            let b = sample("b_class", "fn b_class(s: string) -> i64 { return 2; }\n");
            save_one(&a);
            save_one(&b);
            let got = load();
            assert_eq!(got.len(), 2, "two distinct names → two rows");
            assert_eq!(got[0].name, "a_class");
            assert_eq!(got[1].name, "b_class");

            // Re-learning `a_class` with new code REPLACES, doesn't duplicate.
            let a2 = sample("a_class", "fn a_class(s: string) -> i64 { return 99; }\n");
            save_one(&a2);
            let got = load();
            assert_eq!(got.len(), 2, "merge-by-name must not duplicate a_class");
            // a_class moved to last (refreshed), b_class is now first.
            let a_row = got
                .iter()
                .find(|c| c.name == "a_class")
                .expect("a_class present");
            assert!(a_row.code.contains("99"), "refreshed code must win");
        });
    }

    #[test]
    fn malformed_lines_are_skipped() {
        with_temp_store(|path| {
            save_one(&sample("good", "fn good(s: string) -> i64 { return 1; }\n"));
            {
                let mut f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(path)
                    .expect("open store for corruption");
                f.write_all(b"\nnot valid json\n{ partial json\n").unwrap();
            }
            // A trailing good line (written by hand to avoid save_one's read-merge
            // dropping the garbage we just injected).
            {
                let mut f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(path)
                    .expect("open store to append");
                let line = serde_json::to_string(&sample(
                    "good2",
                    "fn good2(s: string) -> i64 { return 2; }\n",
                ))
                .unwrap();
                f.write_all(line.as_bytes()).unwrap();
                f.write_all(b"\n").unwrap();
            }
            let got = load();
            assert_eq!(got.len(), 2, "only the two well-formed rows survive");
            assert_eq!(got[0].name, "good");
            assert_eq!(got[1].name, "good2");
        });
    }

    #[test]
    fn clear_removes_the_store() {
        with_temp_store(|path| {
            save_one(&sample(
                "c_class",
                "fn c_class(s: string) -> i64 { return 0; }\n",
            ));
            assert!(path.exists());
            assert_eq!(load().len(), 1);
            clear();
            assert!(!path.exists(), "clear must remove the on-disk file");
            assert!(load().is_empty(), "cleared store loads empty");
        });
    }

    /// END-TO-END (good component): persist a genuinely-good `creature_class`
    /// component, then build a FRESH `Engine::new()` and prove the reload-and-
    /// re-gate step in `comprehension` grafted it back in — `has_component` is
    /// true, the component evaluates correctly, AND the engine is still sound
    /// (the regression gate stays green after the reload).
    #[test]
    fn fresh_engine_reloads_a_good_component() {
        with_temp_store(|_path| {
            // Synthesize creature_class once via the real solver to get a verified
            // body + its recovering method, then persist it. The grafted source is
            // exactly the suffix `try_extend` appended to the base program
            // (`"\n" + result.code`), recovered by slicing past the base length —
            // the same suffix-recovery `self_extend` uses when it persists.
            let examples = creature_class_examples();
            let fp = examples_fingerprint(&examples);
            let base = Engine::new_base();
            let base_len = base.program().len();
            let candidate = base
                .try_extend(
                    "creature_class",
                    "fn creature_class(s: string) -> i64",
                    examples,
                )
                .expect("creature_class must synthesize");
            let code = candidate
                .program()
                .get(base_len..)
                .map(|s| s.trim_start_matches('\n').to_string())
                .expect("grafted suffix must be recoverable");
            assert!(
                code.contains("fn creature_class("),
                "the recovered suffix must contain the synthesized component"
            );
            let method = candidate
                .method_for("creature_class")
                .unwrap_or("")
                .to_string();
            save_one(&StoredComponent {
                name: "creature_class".to_string(),
                signature: "fn creature_class(s: string) -> i64".to_string(),
                code,
                method,
                examples_fingerprint: fp,
                members: Vec::new(),
            });

            // A FRESH engine must reload it (the comprehension reload step runs
            // inside Engine::new()).
            let reloaded = Engine::new();
            assert!(
                reloaded.has_component("creature_class"),
                "the persisted good component must be re-grafted onto a fresh engine"
            );
            assert_eq!(
                reloaded.eval_int("creature_class(\"dragon\")"),
                1,
                "the reloaded component must evaluate correctly (dragon → 1)"
            );
            assert_eq!(
                reloaded.eval_int("creature_class(\"report\")"),
                0,
                "report → 0 on the reloaded engine"
            );
            // The reloaded engine is still sound.
            assert!(
                regression_gate(&reloaded).ok(),
                "a fresh engine with a good reloaded component must stay green"
            );
        });
    }

    /// END-TO-END (poisoned component): persist a stored entry whose code, once
    /// grafted, REGRESSES a golden case — a `noun_animacy` override that
    /// misclassifies `teacher` as inanimate. The reload step must RE-GATE it,
    /// REJECT it, and the fresh engine must stay sound with the base behavior
    /// intact (teacher classified animate, gate green). This is the load-time
    /// mirror of `extend::regressing_extension_is_rejected_by_the_gate`: a stale /
    /// poisoned / incompatible store entry cannot poison a fresh boot.
    #[test]
    fn fresh_engine_rejects_a_poisoned_stored_component() {
        with_temp_store(|_path| {
            // A `noun_animacy` override that misclassifies the taxonomy agents.
            // Grafted onto the base it shadows the real lexicon (later def wins),
            // breaking "Is the teacher a person?".
            let poison_code = "\
fn noun_animacy(s: string) -> i64 {\n\
    if s == \"teacher\" { return 2; }\n\
    if s == \"editor\" { return 2; }\n\
    if s == \"author\" { return 2; }\n\
    if s == \"student\" { return 2; }\n\
    return 0;\n\
}\n";
            save_one(&StoredComponent {
                name: "noun_animacy".to_string(),
                signature: "fn noun_animacy(s: string) -> i64".to_string(),
                code: poison_code.to_string(),
                method: "poisoned".to_string(),
                examples_fingerprint: "fp-poison".to_string(),
                members: Vec::new(),
            });

            // A fresh engine must REJECT the poisoned entry on reload and stay
            // sound. `has_component` is true for the BASE noun_animacy regardless
            // (the base always defines it), so we assert on BEHAVIOR, not presence.
            let reloaded = Engine::new();
            assert_eq!(
                reloaded.noun_class("teacher"),
                1,
                "the base noun_animacy must be intact: teacher is animate (1), not poisoned to 2"
            );
            assert!(
                reloaded.is_person("teacher"),
                "the reloaded engine must still answer 'teacher is a person' = true"
            );
            assert!(
                regression_gate(&reloaded).ok(),
                "a fresh engine must stay sound after rejecting a poisoned stored component"
            );
        });
    }

    // -----------------------------------------------------------------------
    // CONSTRUCTION store: persist + gated reload of a word-order construction.
    // -----------------------------------------------------------------------

    /// Run `f` with `NCPU_CONSTRUCTIONS_PATH` (and the component store + journal)
    /// pointed at fresh temp state, holding the crate-wide env lock. Mirrors
    /// [`with_temp_store`] but for the construction store; the component store and
    /// journal are DISABLED (empty) so a reload-through-gate never writes $HOME and
    /// no leftover component row perturbs the fresh engine.
    fn with_temp_construction_store<R>(f: impl FnOnce(&std::path::Path) -> R) -> R {
        use crate::self_improve::journal::test_support::ENV_LOCK;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_c = std::env::var("NCPU_CONSTRUCTIONS_PATH").ok();
        let prev_comp = std::env::var("NCPU_COMPONENTS_PATH").ok();
        let prev_journal = std::env::var("NCPU_JOURNAL_PATH").ok();
        let path = std::env::temp_dir().join(format!(
            "ncpu_constructions_test_{}_{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        // SAFETY: ENV_LOCK guarantees single-threaded access for the duration.
        unsafe {
            std::env::set_var("NCPU_CONSTRUCTIONS_PATH", &path);
            std::env::set_var("NCPU_COMPONENTS_PATH", "");
            std::env::set_var("NCPU_JOURNAL_PATH", "");
        }
        let result = f(&path);
        match prev_c {
            Some(v) => unsafe { std::env::set_var("NCPU_CONSTRUCTIONS_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_CONSTRUCTIONS_PATH") },
        }
        match prev_comp {
            Some(v) => unsafe { std::env::set_var("NCPU_COMPONENTS_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_COMPONENTS_PATH") },
        }
        match prev_journal {
            Some(v) => unsafe { std::env::set_var("NCPU_JOURNAL_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_JOURNAL_PATH") },
        }
        let _ = std::fs::remove_file(&path);
        result
    }

    /// The verified object-fronting construction: skeleton `[0,1,0,1,2]`
    /// (det noun det noun verb) with agent at index 3, patient at index 1, verb at
    /// index 4 — the role assignment proven in `understanding::grammar`'s tests.
    fn osv_construction() -> StoredConstruction {
        StoredConstruction {
            name: "object_fronting".to_string(),
            skeletons: vec![vec![0, 1, 0, 1, 2]],
            agent_idx: 3,
            patient_idx: 1,
            predicate_idx: 4,
        }
    }

    #[test]
    fn construction_store_round_trips_and_merges_by_name() {
        with_temp_construction_store(|_path| {
            assert!(load_constructions().is_empty(), "no file → empty");
            save_one_construction(&osv_construction());
            let got = load_constructions();
            assert_eq!(got.len(), 1);
            assert_eq!(got[0], osv_construction());

            // Re-learning the same name REPLACES (merge-by-name), never duplicates.
            let mut updated = osv_construction();
            updated.predicate_idx = 4;
            updated.skeletons = vec![vec![0, 1, 0, 1, 2], vec![0, 1, 0, 1, 2, 0, 1]];
            save_one_construction(&updated);
            let got = load_constructions();
            assert_eq!(got.len(), 1, "merge-by-name must not duplicate");
            assert_eq!(got[0].skeletons.len(), 2, "refreshed row must win");
        });
    }

    /// END-TO-END (good construction): persist the object-fronting construction,
    /// then build a FRESH `Engine::new()` and prove the gated reload registered it —
    /// the engine now PARSES the OSV sentence (which the base parser left Unknown)
    /// to the correct Event, and the regression gate stays green.
    #[test]
    fn fresh_engine_reloads_a_good_construction() {
        with_temp_construction_store(|_path| {
            // A bare base engine MIS-handles object fronting (returns Unknown).
            let base = Engine::new();
            assert!(
                base.learned_grammar().is_empty(),
                "no construction persisted yet → empty grammar on the base engine"
            );
            let sentence = "The report the teacher writes.";
            assert!(
                matches!(
                    crate::understanding::semantics::understand(&base, sentence),
                    crate::understanding::meaning::Meaning::Unknown(_)
                ),
                "the base parser must leave the object-fronted clause Unknown"
            );

            // Persist the verified construction, then reload into a fresh engine.
            save_one_construction(&osv_construction());
            let reloaded = Engine::new();
            assert_eq!(
                reloaded.learned_grammar().len(),
                1,
                "the persisted construction must be re-registered onto a fresh engine"
            );

            // The reloaded engine PARSES the OSV sentence correctly via the fallback.
            let m = crate::understanding::semantics::understand(&reloaded, sentence);
            let crate::understanding::meaning::Meaning::Event(e) = m else {
                panic!("expected an Event from the reloaded construction, got {m:?}");
            };
            assert_eq!(e.predicate, "write");
            assert_eq!(
                e.agent,
                Some(crate::understanding::meaning::Term::Entity(
                    "teacher".to_string()
                ))
            );
            assert_eq!(
                e.patient,
                Some(crate::understanding::meaning::Term::Entity(
                    "report".to_string()
                ))
            );

            // The reloaded engine is still SOUND (a sound OSV rule leaves the gate
            // green: its skeleton appears in no base-parseable golden case).
            assert!(
                regression_gate(&reloaded).ok(),
                "a fresh engine with a good reloaded construction must stay green"
            );
        });
    }

    /// A construction whose skeleton COLLIDES with a base-parseable pattern: the
    /// SVO declarative skeleton `[0,1,2,0,1]` of "The teacher writes the report."
    /// (a golden SETUP) with agent/patient SWAPPED (agent at index 4 = "report",
    /// patient at index 1 = "teacher"). The base parser handles SVO correctly, so
    /// registering this rule would, if the fallback were ever reached for that
    /// shape, assert the REVERSE event — a collision the gate's
    /// collision-soundness invariant must catch.
    fn colliding_svo_construction() -> StoredConstruction {
        StoredConstruction {
            name: "collide_svo".to_string(),
            skeletons: vec![vec![0, 1, 2, 0, 1]],
            agent_idx: 4,   // SWAP: "report" as agent (base says "teacher")
            patient_idx: 1, // SWAP: "teacher" as patient
            predicate_idx: 2,
        }
    }

    /// END-TO-END (regressing construction): persist a construction whose skeleton
    /// COLLIDES with a base-parseable golden pattern (SVO) and assigns swapped
    /// roles. The gated reload must RE-GATE it, find the collision-soundness
    /// invariant violated, REJECT it, and leave the fresh engine sound with the
    /// construction NOT registered. This is the load-time mirror of the accept-time
    /// gate: a stale / poisoned / now-incompatible construction row cannot poison a
    /// fresh boot.
    #[test]
    fn fresh_engine_rejects_a_regressing_construction() {
        with_temp_construction_store(|_path| {
            // Sanity: a hand-built engine with the colliding rule registered fails the
            // gate (the collision-soundness invariant fires).
            let mut hand = Engine::new_base();
            hand.register_construction(crate::understanding::grammar::LearnedConstruction {
                name: "collide_svo".to_string(),
                skeletons: vec![vec![0, 1, 2, 0, 1]],
                agent_idx: 4,
                patient_idx: 1,
                predicate_idx: 2,
            });
            assert!(
                !regression_gate(&hand).ok(),
                "a colliding construction must redden the gate (collision-soundness)"
            );

            // Now persist it and confirm the RELOAD path makes the same decision: the
            // fresh engine REJECTS it (never registers it) and stays sound.
            save_one_construction(&colliding_svo_construction());
            let reloaded = Engine::new();
            assert!(
                reloaded.learned_grammar().is_empty(),
                "the regressing construction must be rejected on load (not registered)"
            );
            assert!(
                regression_gate(&reloaded).ok(),
                "a fresh engine must stay sound after rejecting a regressing construction"
            );

            // The base SVO parse is unchanged — the colliding rule never registered,
            // and the fallback is Unknown-only so an SVO sentence is base-parsed.
            let svo = crate::understanding::semantics::understand(
                &reloaded,
                "The teacher writes the report.",
            );
            let crate::understanding::meaning::Meaning::Event(e) = svo else {
                panic!("SVO must parse to an Event");
            };
            assert_eq!(e.predicate, "write");
            assert_eq!(
                e.agent,
                Some(crate::understanding::meaning::Term::Entity(
                    "teacher".to_string()
                )),
                "the base SVO agent must remain `teacher` — the colliding rule was rejected"
            );
        });
    }
}
