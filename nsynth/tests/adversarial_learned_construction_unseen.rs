//! ADVERSARIAL verification (Thrust E, grammar induction):
//! "learned-construction-parses-unseen-soundly".
//!
//! Skeptic's claim under test, exactly as posed:
//!   1. A FRESH Mind returns Unknown for the object-fronting clause
//!      "the report the teacher writes" (and cannot answer a question about it).
//!   2. After learning the OSV construction from labeled examples, THAT SAME
//!      sentence parses to Event{write, agent:teacher, patient:report}.
//!   3. An UNSEEN-word OSV sentence "the book the author reads" parses to
//!      Event{read, agent:author, patient:book} — lexical generalization WITHIN
//!      the verified skeleton (the noun "author" never appeared in training).
//!   4. A fresh engine WITHOUT the construction STILL returns Unknown — no change
//!      absent learning.
//!   5. WRONG role assignment (agent/patient swapped) = FALSE.
//!   6. The learned rule must NOT corrupt a base-parseable sentence (the SVO
//!      "the teacher writes the report" must parse identically to a mind that
//!      never learned anything) and must introduce NO false entailment (a Q&A
//!      about the parsed OSV event answers Yes, and its converse does NOT).
//!
//! Written by an external adversary using ONLY the public crate API. Env-fenced:
//! runs only when NCPU_VERIFY_LEARNED_CONSTRUCTION=1 so it never touches the
//! developer's real cross-run stores by accident. The construction stores are
//! also redirected to empty (disabled) inside the test so nothing persists.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::grammar::{learn_construction_from_examples, ConstructionExample};
use mog_synth::understanding::meaning::{Meaning, Term};
use mog_synth::understanding::mind::Mind;
use mog_synth::understanding::semantics;

/// The labeled OSV training set: three sentences with the SAME word-order shape
/// (det noun det noun verb) but DIFFERENT words. Each is tagged with which
/// surface word is the agent / patient and the predicate's lemma — the learner is
/// told the ROLES, never the positions. Crucially, "author" never appears here.
fn osv_training() -> Vec<ConstructionExample<'static>> {
    vec![
        ("the report the teacher writes", "teacher", "report", "write"),
        ("the book the student reads", "student", "book", "read"),
        ("the memo the doctor fixes", "doctor", "memo", "fix"),
    ]
}

fn ent(s: &str) -> Option<Term> {
    Some(Term::Entity(s.to_string()))
}

/// Disable the durable construction + journal stores so this test never reads or
/// writes the developer's real $HOME stores, then restore the prior values.
fn with_stores_disabled<R>(f: impl FnOnce() -> R) -> R {
    let keys = ["NCPU_CONSTRUCTIONS_PATH", "NCPU_COMPONENTS_PATH", "NCPU_JOURNAL_PATH"];
    let prev: Vec<Option<String>> = keys.iter().map(|k| std::env::var(k).ok()).collect();
    for k in &keys {
        // SAFETY: this test is single-threaded for env access; it is the only one
        // gated behind NCPU_VERIFY_LEARNED_CONSTRUCTION and is run in isolation.
        unsafe { std::env::set_var(k, "") };
    }
    let out = f();
    for (k, v) in keys.iter().zip(prev) {
        match v {
            Some(val) => unsafe { std::env::set_var(k, val) },
            None => unsafe { std::env::remove_var(k) },
        }
    }
    out
}

fn fenced() -> bool {
    std::env::var("NCPU_VERIFY_LEARNED_CONSTRUCTION").as_deref() == Ok("1")
}

/// THE END-TO-END ADVERSARIAL CLAIM, via the highest-level `Mind` path (the gated
/// `learn_construction` → regression gate → adopt pipeline).
#[test]
fn learned_construction_parses_unseen_soundly() {
    if !fenced() {
        eprintln!("skipped: set NCPU_VERIFY_LEARNED_CONSTRUCTION=1 to run");
        return;
    }
    with_stores_disabled(|| {
        // ---- (1) FRESH mind: object-fronting is Unknown, and unanswerable. ------
        let trained_sentence = "the report the teacher writes";
        let unseen_sentence = "the book the author reads";

        let fresh = Mind::new();
        assert!(
            fresh.learned_constructions().is_empty(),
            "a fresh mind must carry no constructions"
        );
        assert!(
            matches!(fresh.understand(trained_sentence), Meaning::Unknown(_)),
            "PRECONDITION: a fresh mind must return Unknown for the trained OSV clause, got {:?}",
            fresh.understand(trained_sentence)
        );
        assert!(
            matches!(fresh.understand(unseen_sentence), Meaning::Unknown(_)),
            "PRECONDITION: a fresh mind must return Unknown for the unseen OSV clause, got {:?}",
            fresh.understand(unseen_sentence)
        );
        // It cannot answer a question about the fronted clause either: reading it is
        // a no-op (Unknown is not asserted), so the question is unknown.
        {
            let mut fresh_qa = Mind::new();
            fresh_qa.read(trained_sentence);
            let ans = fresh_qa.ask("does the teacher write the report");
            assert!(
                !ans.to_lowercase().starts_with("yes"),
                "a fresh mind must NOT be able to affirm the OSV event, got: {ans}"
            );
        }

        // ---- (2)+(3)+(5)+(6) LEARN, then verify trained + unseen + soundness. ---
        let mut mind = Mind::new();
        let accepted = mind.learn_construction("object_fronting", &osv_training());
        assert!(accepted, "a verified, gate-green OSV construction must be ACCEPTED");
        assert_eq!(mind.learned_constructions().len(), 1, "exactly one construction adopted");

        // (2) The TRAINED sentence now flips Unknown -> the correct Event.
        let Meaning::Event(e) = mind.understand(trained_sentence) else {
            panic!(
                "after learning, the TRAINED OSV clause must parse to an Event, got {:?}",
                mind.understand(trained_sentence)
            );
        };
        assert_eq!(e.predicate, "write", "predicate must be `write`");
        assert_eq!(e.agent, ent("teacher"), "AGENT must be teacher");
        assert_eq!(e.patient, ent("report"), "PATIENT must be report");
        // (5) WRONG-ROLE guard: roles must NOT be swapped.
        assert_ne!(e.agent, ent("report"), "agent must NOT be the fronted object");
        assert_ne!(e.patient, ent("teacher"), "patient must NOT be the embedded subject");
        assert_eq!(e.recipient, None, "a 2-place OSV clause has no recipient");

        // (3) The UNSEEN-WORD sentence parses to the correct Event. "author" is a
        // novel noun; only the SHAPE [det noun det noun verb] was learned.
        let Meaning::Event(u) = mind.understand(unseen_sentence) else {
            panic!(
                "after learning, the UNSEEN OSV clause must parse to an Event, got {:?}",
                mind.understand(unseen_sentence)
            );
        };
        assert_eq!(u.predicate, "read", "unseen predicate must be `read`");
        assert_eq!(u.agent, ent("author"), "unseen AGENT must be author");
        assert_eq!(u.patient, ent("book"), "unseen PATIENT must be book");
        // (5) WRONG-ROLE guard on the unseen sentence too.
        assert_ne!(u.agent, ent("book"), "unseen agent must NOT be the fronted object");
        assert_ne!(u.patient, ent("author"), "unseen patient must NOT be the embedded subject");
        assert_eq!(u.recipient, None);

        // (6a) SOUNDNESS — no corruption of a base-parseable SVO clause: the learned
        // mind parses ordinary SVO EXACTLY as a mind that never learned anything.
        let svo = "the teacher writes the report";
        let plain = Mind::new();
        assert!(plain.learned_constructions().is_empty());
        assert_eq!(
            mind.understand(svo),
            plain.understand(svo),
            "the OSV fallback must NOT perturb a base-parseable SVO clause"
        );

        // (6b) SOUNDNESS — gate still green after acquisition (monotone growth).
        assert!(mind.self_check().ok(), "the mind must stay green after acquiring grammar");

        // (6c) NO FALSE ENTAILMENT — reading the OSV clause affirms its event and
        // its CONVERSE is not affirmed.
        let mut qa = Mind::new();
        assert!(qa.learn_construction("object_fronting", &osv_training()));
        qa.read(trained_sentence);
        let yes = qa.ask("does the teacher write the report");
        assert!(
            yes.to_lowercase().starts_with("yes"),
            "the parsed OSV event must answer Yes to its own question, got: {yes}"
        );
        let converse = qa.ask("does the report write the teacher");
        assert!(
            !converse.to_lowercase().starts_with("yes"),
            "the CONVERSE (report writes teacher) must NOT be affirmed — no false entailment, \
             got: {converse}"
        );
    });
}

/// (4) NO CHANGE ABSENT LEARNING — a fresh engine without the construction still
/// returns Unknown, proven at the `Engine`/`semantics::understand` layer too. This
/// is the open-world discipline: the learned fallback is the ONLY reason the
/// trained engine parses OSV.
#[test]
fn fresh_engine_unchanged_without_construction() {
    if !fenced() {
        eprintln!("skipped: set NCPU_VERIFY_LEARNED_CONSTRUCTION=1 to run");
        return;
    }
    with_stores_disabled(|| {
        let engine = Engine::new();
        assert!(engine.learned_grammar().is_empty());
        for s in ["the report the teacher writes", "the book the author reads"] {
            assert!(
                matches!(semantics::understand(&engine, s), Meaning::Unknown(_)),
                "without the construction, `{s}` must be Unknown, got {:?}",
                semantics::understand(&engine, s)
            );
        }
    });
}

/// EXTRA adversarial probe: register the SYNTHESIZED-and-VERIFIED construction
/// (via `learn_construction_from_examples`) on one engine, register it on a fresh
/// engine, and confirm the unseen-word clause parses correctly there — proving the
/// generalization is carried by the construction object, not by any side effect of
/// the learner's engine state.
#[test]
fn transplanted_construction_generalizes() {
    if !fenced() {
        eprintln!("skipped: set NCPU_VERIFY_LEARNED_CONSTRUCTION=1 to run");
        return;
    }
    with_stores_disabled(|| {
        let learner = Engine::new();
        let c = learn_construction_from_examples(&learner, "osv", &osv_training())
            .expect("OSV construction should synthesize + verify");
        // recovered indices must be the OSV ones.
        assert_eq!(c.skeletons, vec![vec![0, 1, 0, 1, 2]]);
        assert_eq!(c.patient_idx, 1);
        assert_eq!(c.agent_idx, 3);
        assert_eq!(c.predicate_idx, 4);

        let mut engine = Engine::new();
        engine.register_construction(c);
        let Meaning::Event(u) = semantics::understand(&engine, "the book the author reads") else {
            panic!("transplanted construction must parse unseen OSV to an Event");
        };
        assert_eq!(u.predicate, "read");
        assert_eq!(u.agent, ent("author"));
        assert_eq!(u.patient, ent("book"));
    });
}
