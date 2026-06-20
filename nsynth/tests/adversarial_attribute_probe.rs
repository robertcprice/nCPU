//! ADVERSARIAL verification of ATTRIBUTE (adjectival) semantics.
//!
//! Written by an external skeptic, NOT the module author. Goal: prove the
//! deep-semantics layer genuinely does attribute-semantics — it represents
//! "the teacher is careful" as an adjectival PROPERTY (HasProperty), stores it,
//! and answers attribute questions from the stored model under the open-world
//! assumption — and crucially that it does NOT over-derive: an attribute never
//! asserted (especially about a DIFFERENT entity of the same category) must be
//! "unknown", never a false "Yes".
//!
//! Exact task scenario:
//!   read  "The teacher is careful."
//!   ask   "Is the teacher careful?"  -> Yes (asserted)
//!   ask   "Is the editor careful?"   -> unknown/No (NEVER stated)

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::meaning::{Meaning, Term};
use mog_synth::understanding::{qa, semantics};

use std::sync::OnceLock;
fn engine() -> &'static Engine {
    static E: OnceLock<Engine> = OnceLock::new();
    E.get_or_init(Engine::new)
}

/// (1) PARSE SHAPE: "the teacher is careful" must be HasProperty, NOT IsA.
/// A shallow system would shove "careful" into a category slot. The deep layer
/// must distinguish an adjectival MODIFIER from a nominal category.
#[test]
fn careful_parses_to_hasproperty_not_isa() {
    let m = semantics::understand(engine(), "The teacher is careful.");
    match &m {
        Meaning::HasProperty {
            subject,
            property,
            negated,
        } => {
            assert_eq!(*subject, Term::Entity("teacher".to_string()));
            assert_eq!(property, "careful");
            assert!(!negated, "affirmative attribute");
        }
        Meaning::IsA { .. } => panic!(
            "UNSOUND: 'careful' was parsed as a category (IsA), not an adjectival property: {m:?}"
        ),
        other => panic!("expected HasProperty, got {other:?}"),
    }

    // The question form must also be a HasProperty query, not a category query.
    let q = semantics::understand(engine(), "Is the teacher careful?");
    let Meaning::YesNoQuestion(inner) = q else {
        panic!("expected YesNoQuestion, got {q:?}");
    };
    assert!(
        matches!(*inner, Meaning::HasProperty { .. }),
        "attribute question must wrap a HasProperty, got {inner:?}"
    );
}

/// (2) END-TO-END, EXACT TASK PROMPTS.
/// Read the one fact, then ask both questions through the real English pipeline.
#[test]
fn attribute_queryable_and_not_overderived() {
    let mut d = Discourse::new();
    d.read(engine(), "The teacher is careful.");

    // (a) The asserted attribute is queryable -> Yes.
    let a = qa::answer(engine(), &d, "Is the teacher careful?");
    assert!(
        a.to_lowercase().starts_with("yes"),
        "asserted attribute must be answered Yes; got: {a:?}"
    );

    // (b) THE TRAP: editor is the SAME category as teacher (both AGENTS) and a
    // fully-known noun, but "editor is careful" was NEVER stated. A system that
    // confused category with attribute, or that defaulted unknown->true, would
    // wrongly say Yes. Sound behavior is "I don't know" (open world) or "No".
    let b = qa::answer(engine(), &d, "Is the editor careful?");
    let bl = b.to_lowercase();
    assert!(
        !bl.starts_with("yes"),
        "UNSOUND OVER-DERIVATION: editor's carefulness was never stated, must not be Yes; got: {b:?}"
    );
    assert!(
        bl.contains("don't know") || bl.starts_with("no"),
        "unstated attribute must be unknown/No; got: {b:?}"
    );
}

/// (3) ATTRIBUTE IS PER-PROPERTY, not a blanket "teacher is nice about everything".
/// A different unstated adjective on the SAME (correctly-known-careful) entity
/// must still be unknown. Catches a system that stores "has some property" rather
/// than the specific property.
#[test]
fn different_unstated_property_on_same_entity_is_unknown() {
    let mut d = Discourse::new();
    d.read(engine(), "The teacher is careful.");

    let c = qa::answer(engine(), &d, "Is the teacher brave?");
    let cl = c.to_lowercase();
    assert!(
        !cl.starts_with("yes"),
        "UNSOUND: 'brave' was never stated about the teacher; got: {c:?}"
    );
    assert!(
        cl.contains("don't know") || cl.starts_with("no"),
        "unstated property must be unknown/No; got: {c:?}"
    );
}

/// (4) NEGATION POLARITY: "the teacher is not careful" must make
/// "Is the teacher careful?" answer No (Some(false)), not unknown, and not a
/// silently-dropped negation that flips to Yes.
#[test]
fn negated_attribute_answers_no() {
    let mut d = Discourse::new();
    d.read(engine(), "The teacher is not careful.");

    let a = qa::answer(engine(), &d, "Is the teacher careful?");
    assert!(
        a.to_lowercase().starts_with("no"),
        "a negated attribute must answer No to the positive query; got: {a:?}"
    );
}
