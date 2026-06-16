//! ADVERSARIAL: prove the autonomous self-extension loop journals EVERY attempt.
//!
//! The substrate contract (`self_improve`) claims every self-modification attempt
//! — accepted OR rejected — is appended to a durable reflection journal. A SILENT
//! self-modification (one that takes effect, or is even tried, without leaving a
//! journal entry) would make the audit trail a lie. This test runs the REAL
//! `self_extend` loop twice against a real on-disk journal (a temp file pointed to
//! by `NCPU_JOURNAL_PATH`) — once for an extension that is ACCEPTED and once for
//! one that is REJECTED *after verifying* (a genuine gate-red regression, not a
//! synthesis failure) — and asserts BOTH attempts are present in
//! `journal::entries()` with the correct `accepted` / `regression_passed` /
//! `verified` flags and the correct gap text.
//!
//! Each `self_extend` is verified empirically (its `LearnReport` is asserted), the
//! gate is shown to REALLY reject the regressing extension (candidate discarded,
//! base engine left untouched), and a control proves the entries we read are the
//! ones THIS test wrote (not a stale global journal).

use std::sync::Mutex;

use mog_synth::benchmark::{Example, Value};
use mog_synth::comprehension::{creature_class_examples, Engine};
use mog_synth::self_improve::extend::{self_extend, LearnRequest};
use mog_synth::self_improve::journal::{self, JournalEntry};

/// `NCPU_JOURNAL_PATH` is process-GLOBAL. `cargo test` runs the tests in this
/// binary CONCURRENTLY by default, so every test here that mutates the env var
/// MUST serialize on this one lock or they race — one test clearing/replacing the
/// path while another is mid-write to it (the exact hazard the journal module
/// documents). Holding this lock for the whole body of each test makes the file
/// robust under default parallelism, not just under `--test-threads=1`.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn str_ex(s: &str, v: i64) -> Example {
    Example {
        inputs: vec![Value::Str(s.to_string())],
        expected: Value::Int(v),
    }
}

/// Hold [`ENV_LOCK`], point `NCPU_JOURNAL_PATH` at a fresh, unique temp file for
/// the duration of `f`, then restore the env and remove the file. The lock is
/// held for the WHOLE closure so no sibling test can interleave its own env
/// mutation while this one is recording/reading.
///
/// It ALSO disables the sibling learned-component store (empty
/// `NCPU_COMPONENTS_PATH`) for the closure's duration: `self_extend` now PERSISTS
/// every ACCEPTED component, so without this an accepted `creature_class` here
/// would leak into the developer's real `$HOME` store — and a later
/// `Engine::new()` (in this binary OR another adversarial binary cargo runs after
/// it) would reload it, breaking unrelated "component is genuinely absent" /
/// "explain_self refuses for an unknown topic" preconditions. The prior
/// `NCPU_COMPONENTS_PATH` value is restored on exit. This test only exercises the
/// JOURNAL, so disabling the store entirely is exactly the right fence.
fn with_temp_journal<R>(f: impl FnOnce(&std::path::Path) -> R) -> R {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let prev = std::env::var("NCPU_JOURNAL_PATH").ok();
    let prev_components = std::env::var("NCPU_COMPONENTS_PATH").ok();
    let path = std::env::temp_dir().join(format!(
        "ncpu_journal_adv_{}_{:?}.jsonl",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&path);
    std::env::set_var("NCPU_JOURNAL_PATH", &path);
    // Disable the learned-component store so an accepted extension never writes
    // to the real $HOME store (this test asserts only on the journal).
    std::env::set_var("NCPU_COMPONENTS_PATH", "");
    let out = f(&path);
    let _ = std::fs::remove_file(&path);
    match prev {
        Some(v) => std::env::set_var("NCPU_JOURNAL_PATH", v),
        None => std::env::remove_var("NCPU_JOURNAL_PATH"),
    }
    match prev_components {
        Some(v) => std::env::set_var("NCPU_COMPONENTS_PATH", v),
        None => std::env::remove_var("NCPU_COMPONENTS_PATH"),
    }
    out
}

fn find_by_gap<'a>(entries: &'a [JournalEntry], gap: &str) -> &'a JournalEntry {
    entries
        .iter()
        .find(|e| e.gap == gap)
        .unwrap_or_else(|| panic!("journal is missing an entry for gap {gap:?}; SILENT self-modification"))
}

#[test]
fn every_attempt_accepted_and_rejected_is_journaled() {
    with_temp_journal(|path| {
        // GUARD (do-not-pollute-the-real-store): the components store MUST be
        // disabled for this test, because the accepted extension below persists.
        // If a future edit drops the `NCPU_COMPONENTS_PATH=""` fence in
        // `with_temp_journal`, this fails loudly instead of silently writing
        // `creature_class` into the developer's real `$HOME` store (which used to
        // cause a flaky cross-binary failure in `adversarial_explain_self`).
        assert_eq!(
            std::env::var("NCPU_COMPONENTS_PATH").as_deref(),
            Ok(""),
            "the learned-component store MUST be disabled (empty NCPU_COMPONENTS_PATH) \
             so this journal test never persists a component to the real $HOME store"
        );
        // Precondition: nothing journaled yet — the file does not exist.
        assert!(!path.exists(), "temp journal must start absent");
        assert!(
            journal::entries().is_empty(),
            "journal must start empty for this temp path; a non-empty start means \
             we are reading a stale/global journal and the test proves nothing"
        );

        let engine = Engine::new();

        // ---- ATTEMPT 1: ACCEPTED ---------------------------------------
        // A fresh, vocabulary-disjoint lexicon. Synthesis succeeds, the gate
        // stays green (it perturbs no golden case), and the extension is accepted.
        let accepted_gap = "cannot classify mythical creatures (dragon, griffin, phoenix)";
        let accept_req = LearnRequest {
            gap: accepted_gap.to_string(),
            name: "creature_class".to_string(),
            signature: "fn creature_class(s: string) -> i64",
            examples: creature_class_examples(),
        };
        let (accept_cand, accept_report) = self_extend(&engine, &accept_req);
        assert!(
            accept_report.synthesized,
            "accepted attempt must synthesize: {}",
            accept_report.message
        );
        assert!(
            accept_report.regression_passed,
            "accepted attempt's gate must be green: {}",
            accept_report.message
        );
        assert!(
            accept_report.accepted,
            "a synthesized + gated extension must be accepted: {}",
            accept_report.message
        );
        let accept_engine =
            accept_cand.expect("an accepted extension must return Some(engine)");
        assert_eq!(
            accept_engine.eval_int("creature_class(\"dragon\")"),
            1,
            "the accepted engine must actually answer the new query"
        );

        // ---- ATTEMPT 2: REJECTED (verified, then gate-red regression) ---
        // Redefine `noun_animacy` so "teacher" is no longer animate. Synthesis
        // produces a VERIFIED program for these examples, but grafting it onto a
        // clone makes the qa path break dozens of golden cases (teacher is no
        // longer a person/agent), so the gate goes RED and the candidate is
        // discarded. This is the strong rejection arm: verified=true,
        // regression_passed=false, accepted=false — NOT a synthesis failure.
        let rejected_gap = "redefine noun_animacy to drop teacher animacy (regressing)";
        let reject_req = LearnRequest {
            gap: rejected_gap.to_string(),
            name: "noun_animacy".to_string(),
            signature: "fn noun_animacy(s: string) -> i64",
            examples: vec![str_ex("teacher", 0), str_ex("book", 0), str_ex("dragon", 0)],
        };
        let (reject_cand, reject_report) = self_extend(&engine, &reject_req);
        assert!(
            reject_report.synthesized,
            "the regressing attempt must still SYNTHESIZE a verified program \
             (this is the verified-but-regressing arm, not a synthesis failure): {}",
            reject_report.message
        );
        assert!(
            !reject_report.regression_passed,
            "the regressing attempt MUST fail the gate; if the gate passed it here \
             the gate does not actually guard anything: {}",
            reject_report.message
        );
        assert!(
            !reject_report.accepted,
            "a gate-red extension must never be accepted: {}",
            reject_report.message
        );
        assert!(
            reject_cand.is_none(),
            "the gate REALLY rejects: a regressing candidate is discarded (None)"
        );
        // Base engine is left exactly as it was: it still has the original
        // behavior (teacher is a person) and never grew the bad redefinition's
        // effect. (`has_component` is true on the base because noun_animacy is a
        // core component — the point is the BASE engine object is unchanged.)
        assert_eq!(
            engine.noun_class("teacher"),
            1,
            "the base engine must be UNCHANGED after a rejected self-modification \
             — teacher must still classify as animate"
        );

        // ---- JOURNAL: BOTH attempts must be recorded -------------------
        let entries = journal::entries();
        assert_eq!(
            entries.len(),
            2,
            "EXACTLY two attempts were made (one accepted, one rejected); the \
             journal must hold both — no attempt may be silent. got {} entries: {:?}",
            entries.len(),
            entries.iter().map(|e| &e.gap).collect::<Vec<_>>()
        );

        // The ACCEPTED entry: verified + gated + accepted, with the right gap.
        let acc = find_by_gap(&entries, accepted_gap);
        assert!(acc.verified, "accepted entry: verified must be true");
        assert!(
            acc.regression_passed,
            "accepted entry: regression_passed must be true"
        );
        assert!(acc.accepted, "accepted entry: accepted must be true");
        assert!(
            !acc.method.is_empty(),
            "accepted entry must record the recovering teacher/method"
        );
        assert_eq!(
            acc.action, "synthesize creature_class",
            "accepted entry must record the action attempted"
        );

        // The REJECTED entry: verified but gate-red, NOT accepted, right gap.
        let rej = find_by_gap(&entries, rejected_gap);
        assert!(
            rej.verified,
            "rejected entry: verified must be true (it synthesized a program)"
        );
        assert!(
            !rej.regression_passed,
            "rejected entry: regression_passed must be FALSE (the gate went red)"
        );
        assert!(
            !rej.accepted,
            "rejected entry: accepted must be FALSE (a regressing change is discarded)"
        );
        assert_eq!(
            rej.action, "synthesize noun_animacy",
            "rejected entry must record the action attempted"
        );
        assert!(
            rej.note.to_lowercase().contains("regression gate red"),
            "rejected entry's note must explain WHY it was rejected (gate red); got: {}",
            rej.note
        );

        // Both entries carry an authoritative timestamp stamped by record().
        assert!(
            acc.when_unix > 0 && rej.when_unix > 0,
            "every journaled attempt must carry a record-time timestamp"
        );
    });
}

/// CONTROL: with the journal DISABLED (empty `NCPU_JOURNAL_PATH`), a real
/// self_extend leaves NO entry — proving the entries the main test reads come
/// from the temp file it set, and that the empty-env disable path is honored on
/// the live loop (CI must never write to `$HOME`).
#[test]
fn empty_env_disables_the_loop_journal() {
    // Hold the same lock so this never races the main test's env mutation.
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let prev = std::env::var("NCPU_JOURNAL_PATH").ok();
    let prev_components = std::env::var("NCPU_COMPONENTS_PATH").ok();
    std::env::set_var("NCPU_JOURNAL_PATH", "");
    // ALSO disable the learned-component store: the control extension below is
    // ACCEPTED (it synthesizes + gates), and `self_extend` persists accepted
    // components — without this fence the control would leak `creature_class` into
    // the real $HOME store, the very thing CI must never do.
    std::env::set_var("NCPU_COMPONENTS_PATH", "");
    let engine = Engine::new();
    let req = LearnRequest {
        gap: "control: should never be journaled".to_string(),
        name: "creature_class".to_string(),
        signature: "fn creature_class(s: string) -> i64",
        examples: creature_class_examples(),
    };
    let (_cand, report) = self_extend(&engine, &req);
    assert!(report.accepted, "the control extension still synthesizes + gates");
    assert!(
        journal::entries().is_empty(),
        "an empty NCPU_JOURNAL_PATH must disable journaling even on the live loop"
    );
    match prev {
        Some(v) => std::env::set_var("NCPU_JOURNAL_PATH", v),
        None => std::env::remove_var("NCPU_JOURNAL_PATH"),
    }
    match prev_components {
        Some(v) => std::env::set_var("NCPU_COMPONENTS_PATH", v),
        None => std::env::remove_var("NCPU_COMPONENTS_PATH"),
    }
}
