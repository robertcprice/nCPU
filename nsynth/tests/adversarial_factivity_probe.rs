//! ADVERSARIAL factivity probe — external skeptic, not the module author.
//!
//! The KEY soundness test for the epistemic-frontier layer:
//!   * "know that P" is FACTIVE  -> P must become derivable TRUE.
//!   * "believe that P" is NON-factive -> P must remain UNKNOWN (never entailed).
//!
//! Crucially we use the SAME proposition ("the report is long") in BOTH worlds,
//! and we drive everything through the SURFACE-ENGLISH pipeline (parse -> world
//! -> qa::answer / world.holds), not hand-built Meanings. Over-derivation (a
//! false entailment from `believe`) is an automatic FAIL.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::{qa, semantics};

/// (1) FACTIVE world: "The teacher knows that the report is long."
///   - "Is the report long?"            -> must be derivable TRUE.
///   - "Does the teacher know that ...?" -> Yes.
#[test]
fn know_is_factive_entails_content() {
    let engine = Engine::new();
    let mut world = Discourse::new();
    world.read(&engine, "The teacher knows that the report is long.");

    // The factive entailment: the embedded content is now TRUE in the world.
    let content = semantics::understand(&engine, "the report is long");
    assert_eq!(
        world.world.holds(&content),
        Some(true),
        "know is FACTIVE: 'the report is long' must be entailed TRUE"
    );

    // Surface QA: "Is the report long?" must answer affirmatively.
    let ans = qa::answer(&engine, &world, "Is the report long?").to_lowercase();
    assert!(
        ans.starts_with("yes"),
        "factive 'Is the report long?' must be Yes, got: {ans:?}"
    );

    // The attitude itself: "Does the teacher know that the report is long?" -> Yes.
    let att = qa::answer(&engine, &world, "Does the teacher know that the report is long?")
        .to_lowercase();
    assert!(
        att.starts_with("yes"),
        "attitude question must be Yes, got: {att:?}"
    );
}

/// (2) NON-FACTIVE world (FRESH): "The teacher believes that the report is long."
///   - "Is the report long?" must be UNKNOWN — believe must NOT entail P.
///   - The attitude itself is still TRUE (the teacher does believe it).
#[test]
fn believe_is_non_factive_does_not_entail_content() {
    let engine = Engine::new();
    let mut world = Discourse::new();
    world.read(&engine, "The teacher believes that the report is long.");

    // The SAME proposition as the factive world — must stay UNKNOWN.
    let content = semantics::understand(&engine, "the report is long");
    assert_eq!(
        world.world.holds(&content),
        None,
        "believe is NON-FACTIVE: 'the report is long' must be UNKNOWN, never entailed (over-derivation = FAIL)"
    );

    // Surface QA: "Is the report long?" must NOT answer Yes.
    let ans = qa::answer(&engine, &world, "Is the report long?").to_lowercase();
    assert!(
        !ans.starts_with("yes"),
        "non-factive 'Is the report long?' must NOT be Yes (no false entailment), got: {ans:?}"
    );

    // Sanity: the attitude proposition itself is still true — the teacher DOES believe it.
    let att = qa::answer(&engine, &world, "Does the teacher believe that the report is long?")
        .to_lowercase();
    assert!(
        att.starts_with("yes"),
        "the belief itself must hold, got: {att:?}"
    );
}

/// (3) CROSS-CHECK: a single program run that establishes the believe-world does
/// NOT leak factivity even when the know-verb appears earlier in a DIFFERENT
/// attitude. Guards against a global "any attitude entails P" shortcut.
#[test]
fn believe_world_isolated_from_know_factivity() {
    let engine = Engine::new();

    // FRESH believe-only world: content must be UNKNOWN.
    let mut believe_world = Discourse::new();
    believe_world.read(&engine, "The editor believes that the letter is short.");
    let letter_short = semantics::understand(&engine, "the letter is short");
    assert_eq!(
        believe_world.world.holds(&letter_short),
        None,
        "fresh believe world must not entail its content"
    );

    // The teacher believing P, AND separately the teacher KNOWING a DIFFERENT
    // proposition Q, must entail Q but NOT P.
    let mut mixed = Discourse::new();
    mixed.read(&engine, "The teacher believes that the report is long.");
    mixed.read(&engine, "The teacher knows that the book is heavy.");

    let report_long = semantics::understand(&engine, "the report is long");
    let book_heavy = semantics::understand(&engine, "the book is heavy");

    assert_eq!(
        mixed.world.holds(&book_heavy),
        Some(true),
        "known proposition Q must be entailed TRUE"
    );
    assert_eq!(
        mixed.world.holds(&report_long),
        None,
        "believed proposition P must remain UNKNOWN even when a sibling 'know' fires"
    );
}
