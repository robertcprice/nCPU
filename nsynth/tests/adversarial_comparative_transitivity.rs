//! Adversarial probe: comparative transitivity SOUNDNESS through the full
//! natural-language pipeline (read sentences -> world model -> answer questions).
//!
//! Unlike the world_model unit tests, every assertion here goes through the
//! ACTUAL parser/semantics (`Discourse::read`) and the ACTUAL answerer
//! (`qa::answer`) — no hand-built `Meaning::Comparison`. This is what "genuine
//! understanding" must survive.
//!
//! The contract under test:
//!   (1) TRANSITIVITY:  "A longer than B" + "B longer than C" ⊢ "A longer than C" = Yes.
//!   (2) ASYMMETRY:     with the above, "C longer than A?" must NOT be Yes.
//!   (3) NO SYMMETRY:   with ONLY "A longer than B" known, "B longer than A?"
//!                      must NOT be Yes (asking the reverse of a single fact).
//! Over-derivation (a spurious "Yes" to a reverse comparison) is an automatic fail.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::qa;

/// True iff the answer is an affirmative ("Yes, ...").
fn is_yes(ans: &str) -> bool {
    ans.trim_start().to_lowercase().starts_with("yes")
}

/// True iff the answer is a denial ("No, ...").
fn is_no(ans: &str) -> bool {
    ans.trim_start().to_lowercase().starts_with("no")
}

/// The exact scenario from the verification task.
#[test]
fn task_scenario_transitive_yes_and_reverse_not_yes() {
    let engine = Engine::new();
    let mut d = Discourse::new();

    d.read(&engine, "The report is longer than the book.");
    d.read(&engine, "The book is longer than the letter.");

    // (1) Transitive derivation: report > book > letter ⊢ report > letter.
    let q_trans = "Is the report longer than the letter?";
    let a_trans = qa::answer(&engine, &d, q_trans);
    assert!(
        is_yes(&a_trans),
        "transitivity FAILED: {q_trans:?} -> {a_trans:?} (expected Yes)"
    );

    // (2) CRITICAL ASYMMETRY: the reverse must be NOT-Yes. With report>letter
    //     proven, the reverse letter>report should be an explicit No (proven
    //     false by asymmetry of the strict order). It must never be Yes.
    let q_rev = "Is the letter longer than the report?";
    let a_rev = qa::answer(&engine, &d, q_rev);
    assert!(
        !is_yes(&a_rev),
        "ASYMMETRY VIOLATED (over-derivation): {q_rev:?} -> {a_rev:?} (must NOT be Yes)"
    );
    // Stronger: because report>letter is provable, the reverse is provably false.
    assert!(
        is_no(&a_rev),
        "asymmetry should be a definite No here: {q_rev:?} -> {a_rev:?}"
    );

    // Adjacent reverse pairs must also respect asymmetry.
    let a_book_report = qa::answer(&engine, &d, "Is the book longer than the report?");
    assert!(
        is_no(&a_book_report),
        "reverse of asserted edge must be No: {a_book_report:?}"
    );
    let a_letter_book = qa::answer(&engine, &d, "Is the letter longer than the book?");
    assert!(
        is_no(&a_letter_book),
        "reverse of asserted edge must be No: {a_letter_book:?}"
    );
}

/// With ONLY a single comparison known, the REVERSE must not be Yes.
/// This is the purest asymmetry trap: nothing else can mask a symmetry bug.
#[test]
fn single_fact_reverse_is_not_yes() {
    let engine = Engine::new();
    let mut d = Discourse::new();

    d.read(&engine, "The report is longer than the book.");

    // Forward holds.
    let a_fwd = qa::answer(&engine, &d, "Is the report longer than the book?");
    assert!(
        is_yes(&a_fwd),
        "forward single fact should be Yes: {a_fwd:?}"
    );

    // Reverse must NOT be Yes. (Here it is provably false by asymmetry -> No.)
    let a_rev = qa::answer(&engine, &d, "Is the book longer than the report?");
    assert!(
        !is_yes(&a_rev),
        "SYMMETRY BUG: single fact reverse answered Yes: {a_rev:?}"
    );
}

/// The "shorter" converse of a known "longer" fact is SOUND and must be Yes;
/// but the reverse of that ("report shorter than book") must NOT be Yes.
#[test]
fn converse_phrasing_is_sound_but_not_symmetric() {
    let engine = Engine::new();
    let mut d = Discourse::new();
    d.read(&engine, "The report is longer than the book.");

    // "the book is shorter than the report" is the SAME ordering -> Yes.
    let a_conv = qa::answer(&engine, &d, "Is the book shorter than the report?");
    assert!(
        is_yes(&a_conv),
        "converse phrasing (book shorter than report) should be Yes: {a_conv:?}"
    );
    // "the report is shorter than the book" is the OPPOSITE ordering -> NOT Yes.
    let a_bad = qa::answer(&engine, &d, "Is the report shorter than the book?");
    assert!(
        !is_yes(&a_bad),
        "OVER-DERIVATION: report shorter than book should NOT be Yes: {a_bad:?}"
    );
}

/// An unrelated pair (no path on the scale, neither direction asserted) must be
/// UNKNOWN ("I don't know"), never a fabricated Yes.
#[test]
fn unrelated_pair_is_unknown_not_yes() {
    let engine = Engine::new();
    let mut d = Discourse::new();
    d.read(&engine, "The report is longer than the book.");
    d.read(&engine, "The book is longer than the letter.");

    // "memo" and "note" were never mentioned in any ordering.
    let a = qa::answer(&engine, &d, "Is the memo longer than the note?");
    assert!(
        !is_yes(&a),
        "fabricated Yes for an unrelated pair (over-derivation): {a:?}"
    );
    assert!(
        a.to_lowercase().contains("don't know") || a.to_lowercase().contains("do not know"),
        "unrelated pair should be 'I don't know': {a:?}"
    );
}

/// A longer transitive chain (4 nodes) must still derive the far end AND keep
/// every reverse edge/path not-Yes. Guards against a closure that accidentally
/// symmetrizes once depth > 2.
#[test]
fn deeper_chain_transitive_and_asymmetric() {
    let engine = Engine::new();
    let mut d = Discourse::new();
    d.read(&engine, "The report is longer than the essay.");
    d.read(&engine, "The essay is longer than the book.");
    d.read(&engine, "The book is longer than the letter.");

    // Far-end transitivity: report > letter (3 hops).
    let a_far = qa::answer(&engine, &d, "Is the report longer than the letter?");
    assert!(is_yes(&a_far), "3-hop transitivity failed: {a_far:?}");

    // Every reverse must be not-Yes.
    for (q, label) in [
        ("Is the letter longer than the report?", "far reverse"),
        ("Is the book longer than the essay?", "mid reverse"),
        ("Is the letter longer than the essay?", "2-hop reverse"),
        ("Is the essay longer than the report?", "edge reverse"),
    ] {
        let a = qa::answer(&engine, &d, q);
        assert!(
            !is_yes(&a),
            "ASYMMETRY VIOLATED ({label}): {q:?} -> {a:?} (must NOT be Yes)"
        );
    }
}
