//! ADVERSARIAL probe of quantifier-truth scope correctness.
//!
//! Independent of the authors' own unit tests. We construct a world where
//! SOME but NOT ALL members of a category perform an action, then demand that
//! the universal / existential / negative readings get DIFFERENT, scope-correct
//! verdicts. A system that ignores quantifier force (e.g. just checks "does any
//! matching fact exist") would collapse these into the same answer and fail.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::meaning::{Event, Meaning, Quantifier, Tense, Term};
use mog_synth::understanding::semantics;
use mog_synth::understanding::world_model::World;

/// present, affirmative (or negated) write(agent, patient)
fn write_event(agent: &str, patient: &str, negated: bool) -> Event {
    Event {
        predicate: "write".to_string(),
        agent: Some(Term::Entity(agent.to_string())),
        patient: Some(Term::Entity(patient.to_string())),
        recipient: None,
        tense: Tense::Present,
        negated,
    }
}

/// body of a quantified meaning: agent open (bound by quantifier), patient given
fn quant_body(predicate: &str, patient: &str) -> Event {
    Event {
        predicate: predicate.to_string(),
        agent: None,
        patient: Some(Term::Indefinite(patient.to_string())),
        recipient: None,
        tense: Tense::Present,
        negated: false,
    }
}

fn quantified(q: Quantifier, cat: &str, body: Event) -> Meaning {
    Meaning::Quantified { quant: q, var_category: cat.to_string(), body }
}

/// THE CORE PROBE.
///
/// World: teacher and editor are both `person`/`agent`.
///   - teacher writes the report   (affirmative)
///   - editor does NOT write the report (explicitly negated -> determined-false)
///
/// This is "SOME but not ALL agents write a report". Scope-correct truth:
///   every agent writes a report  -> FALSE  (editor is a counterexample)
///   some  agent writes a report  -> TRUE   (teacher does)
///   no    agent writes a report  -> FALSE  (teacher does)
///
/// All three are over the SAME predicate, SAME patient, SAME category. The ONLY
/// thing distinguishing the verdicts is the quantifier. If the system ignored
/// quantifier force it could not produce three different answers.
#[test]
fn some_but_not_all_three_quantifiers_disagree() {
    let mut w = World::new();
    w.assert(&Meaning::Event(write_event("teacher", "report", false)));
    w.assert(&Meaning::Event(write_event("editor", "report", true))); // negated

    let every = w.holds(&quantified(Quantifier::Every, "agent", quant_body("write", "report")));
    let some = w.holds(&quantified(Quantifier::Some, "agent", quant_body("write", "report")));
    let no = w.holds(&quantified(Quantifier::No, "agent", quant_body("write", "report")));

    assert_eq!(every, Some(false), "EVERY must be false: editor is a counterexample");
    assert_eq!(some, Some(true), "SOME must be true: teacher writes a report");
    assert_eq!(no, Some(false), "NO must be false: teacher writes a report");

    // The decisive scope check: the three verdicts are NOT all equal.
    assert!(
        !(every == some && some == no),
        "quantifier force ignored: all three readings collapsed to the same verdict {every:?}"
    );
}

/// SOUNDNESS trap: an `Every` must NOT over-derive `true` when a member's body
/// truth is UNKNOWN (open-world), as opposed to determined-false. Here editor
/// has NO write fact at all (neither affirmative nor negated). teacher writes.
///   every agent writes a report -> must be None (editor undetermined), NOT true
///   some  agent writes a report -> TRUE (teacher does, witnessed)
///   no    agent writes a report -> FALSE (teacher is a positive witness)
/// A system that conflates "no counterexample asserted" with "universally true"
/// would unsoundly report Some(true) for the universal.
#[test]
fn universal_with_unknown_member_is_none_not_overderived_true() {
    let mut w = World::new();
    w.assert(&Meaning::Event(write_event("teacher", "report", false)));
    // Register editor as a known agent WITHOUT a write fact, via a category stmt.
    w.assert(&Meaning::IsA {
        subject: Term::Entity("editor".to_string()),
        category: "person".to_string(),
        negated: false,
    });

    // Sanity: editor really is a known member of the category.
    assert!(
        w.entities().contains(&"editor".to_string()),
        "editor should be a known entity"
    );

    let every = w.holds(&quantified(Quantifier::Every, "agent", quant_body("write", "report")));
    let some = w.holds(&quantified(Quantifier::Some, "agent", quant_body("write", "report")));

    assert_eq!(
        every,
        None,
        "UNSOUND OVER-DERIVATION: universal reported a verdict while editor's body-truth is unknown"
    );
    assert_eq!(some, Some(true), "existential should be witnessed-true by teacher");
}

/// EXISTENTIAL must be FALSE (not None, not True) when EVERY member is
/// determined-false. Both teacher and editor explicitly do NOT write.
///   some agent writes a report -> FALSE
///   no   agent writes a report -> TRUE
///   every agent writes a report -> FALSE
#[test]
fn existential_false_when_all_members_determined_false() {
    let mut w = World::new();
    w.assert(&Meaning::Event(write_event("teacher", "report", true))); // negated
    w.assert(&Meaning::Event(write_event("editor", "report", true))); // negated

    let some = w.holds(&quantified(Quantifier::Some, "agent", quant_body("write", "report")));
    let no = w.holds(&quantified(Quantifier::No, "agent", quant_body("write", "report")));
    let every = w.holds(&quantified(Quantifier::Every, "agent", quant_body("write", "report")));

    assert_eq!(some, Some(false), "SOME must be false: nobody writes");
    assert_eq!(no, Some(true), "NO must be true: nobody writes");
    assert_eq!(every, Some(false), "EVERY must be false: every member is a counterexample");
}

/// Quantifier force is NOT a fixed mapping from the predicate: with the SAME
/// world, two DIFFERENT bodies must yield different universal verdicts. This
/// guards against "the universal answer is hardwired per category".
///   world: teacher + editor both write a report; neither reads a book.
///   every agent writes a report -> TRUE
///   every agent reads  a book   -> FALSE  (none read)
#[test]
fn universal_verdict_tracks_the_body_not_just_the_category() {
    let mut w = World::new();
    w.assert(&Meaning::Event(write_event("teacher", "report", false)));
    w.assert(&Meaning::Event(write_event("editor", "report", false)));
    // Make "reads a book" determined-false for both so the universal is FALSE,
    // not merely undetermined.
    w.assert(&Meaning::Event(Event {
        predicate: "read".to_string(),
        agent: Some(Term::Entity("teacher".to_string())),
        patient: Some(Term::Entity("book".to_string())),
        recipient: None,
        tense: Tense::Present,
        negated: true,
    }));
    w.assert(&Meaning::Event(Event {
        predicate: "read".to_string(),
        agent: Some(Term::Entity("editor".to_string())),
        patient: Some(Term::Entity("book".to_string())),
        recipient: None,
        tense: Tense::Present,
        negated: true,
    }));

    let every_write = w.holds(&quantified(Quantifier::Every, "agent", quant_body("write", "report")));
    let every_read = w.holds(&quantified(Quantifier::Every, "agent", quant_body("read", "book")));

    assert_eq!(every_write, Some(true), "every agent writes a report");
    assert_eq!(every_read, Some(false), "every agent reads a book must be FALSE (none read)");
    assert_ne!(
        every_write, every_read,
        "universal verdict failed to track the body predicate"
    );
}

/// The patient must scope too: with teacher writing a REPORT (not a MEMO),
///   some agent writes a report -> TRUE
///   some agent writes a memo   -> None (open-world; nobody asserted re: memo)
/// A system that ignored the patient argument would wrongly call the memo query
/// true off the report fact.
#[test]
fn existential_respects_the_patient_argument() {
    let mut w = World::new();
    w.assert(&Meaning::Event(write_event("teacher", "report", false)));

    let some_report = w.holds(&quantified(Quantifier::Some, "agent", quant_body("write", "report")));
    let some_memo = w.holds(&quantified(Quantifier::Some, "agent", quant_body("write", "memo")));

    assert_eq!(some_report, Some(true), "teacher writes a report");
    assert_eq!(
        some_memo, None,
        "UNSOUND: existential over 'writes a memo' should be undetermined, not true off the report fact"
    );
}

/// FULL-PIPELINE probe through the natural-language parser. The parser binds the
/// quantified category to a concrete lexicon noun ("teacher"), so to get a
/// multi-member SOME-but-not-ALL domain we evaluate the parsed force against a
/// world whose `person` category contains multiple members. We confirm two
/// things end-to-end:
///   (1) the three quantifier FORCES survive English parsing (Every/Some/No),
///   (2) the evaluator gives scope-correct, DISTINCT verdicts on a real
///       SOME-but-not-ALL world.
#[test]
fn parser_captures_force_and_evaluator_is_scope_correct() {
    let engine = Engine::new();

    // (1) The three forces parse out of English distinctly. "teacher" is a
    // recognized lexicon noun, so the parser builds Quantified meanings.
    let every = semantics::understand(&engine, "Every teacher writes a report.");
    let some = semantics::understand(&engine, "Some teacher writes a report.");
    let no = semantics::understand(&engine, "No teacher writes a report.");

    let force = |m: &Meaning| match m {
        Meaning::Quantified { quant, .. } => Some(*quant),
        _ => None,
    };
    assert_eq!(force(&every), Some(Quantifier::Every), "parsed universal force: {every:?}");
    assert_eq!(force(&some), Some(Quantifier::Some), "parsed existential force: {some:?}");
    assert_eq!(force(&no), Some(Quantifier::No), "parsed negative force: {no:?}");
    // The three parses are genuinely different meanings, not one collapsed form.
    assert_ne!(every, some);
    assert_ne!(some, no);
    assert_ne!(every, no);

    // (2) Build a real SOME-but-not-ALL world over the `agent`/`person` category:
    // teacher writes a report; editor explicitly does NOT. Then evaluate the
    // three forces (over the multi-member category) and demand distinct verdicts.
    let mut w = World::new();
    w.assert(&Meaning::Event(write_event("teacher", "report", false)));
    w.assert(&Meaning::Event(write_event("editor", "report", true))); // negated

    let v_every = w.holds(&quantified(Quantifier::Every, "agent", quant_body("write", "report")));
    let v_some = w.holds(&quantified(Quantifier::Some, "agent", quant_body("write", "report")));
    let v_no = w.holds(&quantified(Quantifier::No, "agent", quant_body("write", "report")));

    assert_eq!(v_every, Some(false), "EVERY agent writes a report must be FALSE (editor counterexample)");
    assert_eq!(v_some, Some(true), "SOME agent writes a report must be TRUE (teacher does)");
    assert_eq!(v_no, Some(false), "NO agent writes a report must be FALSE (teacher does)");
    assert!(
        !(v_every == v_some && v_some == v_no),
        "evaluator collapsed quantifier force: every={v_every:?} some={v_some:?} no={v_no:?}"
    );
}

/// END-TO-END via the demo's exact pattern: read English to build the world, then
/// parse a quantified question from English and evaluate it. This mirrors the
/// shipped `comprehend` demo path ("Does every teacher write a report?") but adds
/// the contrasting FALSE universal to prove the parser+evaluator track the body.
#[test]
fn parser_world_build_then_quantified_query_tracks_body() {
    let engine = Engine::new();
    let mut d = Discourse::new();
    // teacher writes a report (true body); teacher does NOT read a book (false body).
    d.read(&engine, "The teacher writes the report.");
    d.read(&engine, "The teacher does not read the book.");

    // Parse two universals over the SAME category noun but DIFFERENT bodies.
    let writes = semantics::understand(&engine, "Every teacher writes a report.");
    let reads = semantics::understand(&engine, "Every teacher reads a book.");

    // Both parse as universals (force preserved).
    assert!(matches!(writes, Meaning::Quantified { quant: Quantifier::Every, .. }));
    assert!(matches!(reads, Meaning::Quantified { quant: Quantifier::Every, .. }));

    let v_writes = d.world.holds(&writes);
    let v_reads = d.world.holds(&reads);

    assert_eq!(v_writes, Some(true), "every teacher writes a report -> TRUE");
    assert_eq!(v_reads, Some(false), "every teacher reads a book -> FALSE (teacher does not)");
    assert_ne!(v_writes, v_reads, "universal verdict failed to track the body through the parser");
}

/// QA-LAYER end-to-end: ask the three quantified questions IN ENGLISH and check
/// the natural-language ANSWER is scope-correct. The world: the teacher writes a
/// report but does NOT read a book. Over the single known teacher:
///   "Does every teacher write a report?" -> Yes
///   "Does every teacher read  a book?"   -> No   (teacher is a counterexample)
///   "Does no    teacher read  a book?"   -> Yes  (none read)
///   "Does some  teacher read  a book?"   -> No
/// If the answerer ignored quantifier force, the read-a-book questions could not
/// split into No/Yes/No across every/no/some.
#[test]
fn qa_layer_quantified_answers_are_scope_correct() {
    let engine = Engine::new();
    let mut d = Discourse::new();
    d.read(&engine, "The teacher writes the report.");
    d.read(&engine, "The teacher does not read the book.");

    let ask = |q: &str| {
        let a = mog_synth::understanding::qa::answer(&engine, &d, q);
        a.to_lowercase()
    };

    let every_write = ask("Does every teacher write a report?");
    let every_read = ask("Does every teacher read a book?");
    let no_read = ask("Does no teacher read a book?");
    let some_read = ask("Does some teacher read a book?");

    assert!(every_write.starts_with("yes"), "every teacher writes a report -> Yes, got: {every_write}");
    assert!(every_read.starts_with("no"), "every teacher reads a book -> No, got: {every_read}");
    assert!(no_read.starts_with("yes"), "no teacher reads a book -> Yes, got: {no_read}");
    assert!(some_read.starts_with("no"), "some teacher reads a book -> No, got: {some_read}");

    // Scope-sensitivity proof: the SAME body ("read a book") yields DIFFERENT
    // answers under different quantifier force.
    assert_ne!(every_read, no_read, "every vs no over same body must differ");
}
