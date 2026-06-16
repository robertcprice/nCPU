//! ADVERSARIAL verification of 3-place (ditransitive) predicate understanding.
//!
//! Skeptic's goal: prove the recipient is a GENUINELY DISTINCT third thematic
//! role, not conflated with the patient (direct object) nor with the agent.
//!
//! CRITICAL: every fact and every question goes through the natural-language
//! pipeline — `Discourse::read` (parser -> world) and `qa::answer` (parser ->
//! world query). Nothing is hand-asserted into the world. If the parser failed
//! to fill the recipient slot from "to the student", or conflated it with the
//! patient, these tests fail.
//!
//! Over-derivation is an automatic fail: a recipient query must NOT return the
//! patient/agent; an agent query must NOT return the recipient/patient; a
//! patient query must NOT return the agent/recipient. We also probe that a
//! recipient query against a NON-ditransitive 2-place fact stays open-world
//! ("I don't know.") rather than fabricating a filler.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::qa::answer;

/// Lowercase, trimmed answer for substring assertions.
fn norm(s: &str) -> String {
    s.trim().to_lowercase()
}

/// The canonical probe from the task: read one ditransitive sentence, then ask
/// the three role-distinguishing questions. The recipient MUST be a third role
/// distinct from agent and patient.
#[test]
fn three_roles_are_distinct_end_to_end() {
    let engine = Engine::new();
    let mut d = Discourse::new();

    // Read the 3-place predication. Parser must fill agent=teacher,
    // patient=book, recipient=student.
    d.read(&engine, "The teacher gives the book to the student.");

    // Q1: recipient query. Expect "student" — NOT teacher (agent), NOT book
    // (patient).
    let recip = answer(&engine, &d, "Who does the teacher give the book to?");
    let r = norm(&recip);
    assert!(
        r.contains("student"),
        "recipient query must answer 'student'; got: {recip}"
    );
    assert!(
        !r.contains("teacher") && !r.contains("book"),
        "recipient must NOT be conflated with agent/patient; got: {recip}"
    );

    // Q2: agent query. Expect "teacher" — NOT student (recipient), NOT book.
    let agent = answer(&engine, &d, "Who gives the book to the student?");
    let a = norm(&agent);
    assert!(
        a.contains("teacher"),
        "agent query must answer 'teacher'; got: {agent}"
    );
    assert!(
        !a.contains("student") && !a.contains("book"),
        "agent must NOT be conflated with recipient/patient; got: {agent}"
    );

    // Q3: patient query. Expect "book" — NOT teacher (agent), NOT student
    // (recipient). This is the crux of "recipient is distinct from patient":
    // the patient slot still holds the book, not the student.
    let patient = answer(&engine, &d, "What does the teacher give to the student?");
    let p = norm(&patient);
    assert!(
        p.contains("book"),
        "patient query must answer 'book'; got: {patient}"
    );
    assert!(
        !p.contains("student") && !p.contains("teacher"),
        "patient must NOT be conflated with recipient/agent; got: {patient}"
    );
}

/// Over-derivation trap #1: a recipient query against a TWO-place fact (no "to"
/// phrase, no recipient) must stay open-world, never fabricate a recipient by
/// borrowing the patient.
#[test]
fn recipient_query_on_two_place_fact_is_open_world() {
    let engine = Engine::new();
    let mut d = Discourse::new();

    // A plain transitive (2-place) fact: no recipient exists.
    d.read(&engine, "The teacher writes the report.");

    // Ask a recipient-shaped question about a verb that has no recipient. The
    // parser only mints a Recipient slot for ditransitive verbs, and the world
    // has no recipient fact, so this must be "I don't know." — NOT "the report".
    let ans = answer(&engine, &d, "Who does the teacher give the report to?");
    let a = norm(&ans);
    assert!(
        a.contains("don't know"),
        "recipient query with no ditransitive fact must be open-world; got: {ans}"
    );
    assert!(
        !a.contains("report"),
        "must NOT borrow the patient as a fake recipient; got: {ans}"
    );
}

/// Over-derivation trap #2: after reading the ditransitive fact, a question
/// whose AGENT or PATIENT constraint does NOT match the stored fact must be
/// "I don't know." — the wh-matcher must respect ALL three slots, not just the
/// queried one. If recipient and patient were conflated, a mismatched-patient
/// recipient query could spuriously match.
#[test]
fn recipient_query_respects_other_slot_constraints() {
    let engine = Engine::new();
    let mut d = Discourse::new();
    d.read(&engine, "The teacher gives the book to the student.");

    // Recipient query but with a WRONG patient (the report, not the book).
    // The fact's patient is "book", so this constrained query must NOT match.
    let wrong_patient = answer(&engine, &d, "Who does the teacher give the report to?");
    let wp = norm(&wrong_patient);
    assert!(
        wp.contains("don't know"),
        "recipient query with non-matching patient must be open-world; got: {wrong_patient}"
    );
    assert!(
        !wp.contains("student"),
        "must NOT return the recipient when the patient constraint fails; got: {wrong_patient}"
    );

    // Recipient query with a WRONG agent (the editor, not the teacher).
    let wrong_agent = answer(&engine, &d, "Who does the editor give the book to?");
    let wa = norm(&wrong_agent);
    assert!(
        wa.contains("don't know"),
        "recipient query with non-matching agent must be open-world; got: {wrong_agent}"
    );
}

/// Double-object dative ("gives the student the book") must assign the SAME
/// three roles as the prepositional dative — recipient=student, patient=book —
/// proving the role assignment is structural, not a positional artifact of the
/// surface "to" word.
#[test]
fn double_object_dative_assigns_same_roles() {
    let engine = Engine::new();
    let mut d = Discourse::new();
    d.read(&engine, "The teacher gives the student the book.");

    // Recipient (the indirect object, FIRST after the verb) = student.
    let recip = answer(&engine, &d, "Who does the teacher give the book to?");
    let r = norm(&recip);
    assert!(
        r.contains("student") && !r.contains("book"),
        "double-object recipient must be 'student', not 'book'; got: {recip}"
    );

    // Patient (the direct object, SECOND) = book.
    let patient = answer(&engine, &d, "What does the teacher give to the student?");
    let p = norm(&patient);
    assert!(
        p.contains("book") && !p.contains("student"),
        "double-object patient must be 'book', not 'student'; got: {patient}"
    );
}
