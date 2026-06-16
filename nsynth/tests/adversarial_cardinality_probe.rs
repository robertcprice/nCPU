//! ADVERSARIAL probe of the semantic-frontier CARDINALITY / counting layer.
//!
//! Goal: prove the system GENUINELY counts distinct satisfiers with sound,
//! monotone at-least semantics — not a hardcoded "Two" or a shallow string
//! trick. We build worlds the authors' demo never used, exercise the real
//! end-to-end English path (Discourse::read -> qa::answer / semantics::understand),
//! and demand:
//!   * count over a fresh 2-writer world = 2 (a number)
//!   * "two ... write a report" TRUE, "three ... write a report" FALSE
//!   * counting is over DISTINCT entities (a writer counted once, not per mention)
//!   * NO over-derivation: at-least-3 is never TRUE with only 2 witnesses;
//!     a non-writer must not be counted; unknown category must not fabricate.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::meaning::{Event, Meaning, Tense, Term};
use mog_synth::understanding::world_model::World;
use mog_synth::understanding::{qa, semantics};

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

/// body of a count/cardinal: agent open (bound), patient an indefinite report.
fn report_body() -> Event {
    Event {
        predicate: "write".to_string(),
        agent: None,
        patient: Some(Term::Indefinite("report".to_string())),
        recipient: None,
        tense: Tense::Present,
        negated: false,
    }
}

// ---------------------------------------------------------------------------
// 1. END-TO-END ENGLISH: a fresh world with exactly two DISTINCT-noun agents
//    (editor + author) over the "agent" taxonomy class. This is NOT the demo's
//    teacher+author; we pick a noun PAIR the demo never used for cardinality,
//    so a hardcoded "Two" tied to specific words cannot pass. Both nouns must
//    be in the recognized AGENT lexicon to be counted as agents.
// ---------------------------------------------------------------------------
#[test]
fn english_count_two_distinct_agents_writing_a_report() {
    let engine = Engine::new();
    let mut disc = Discourse::new();
    disc.read(&engine, "The editor writes the report.");
    disc.read(&engine, "The author writes the report.");

    // "How many agents write a report?" must answer a NUMBER == two.
    let ans = qa::answer(&engine, &disc, "How many agents write a report?");
    let low = ans.to_lowercase();
    assert!(
        low.contains("two") || low.contains('2'),
        "expected the count answer to be two, got: {ans:?}"
    );

    // at-least-2 TRUE, at-least-3 FALSE/unknown (monotone, sound).
    let card2 = semantics::understand(&engine, "two agents write a report");
    let card3 = semantics::understand(&engine, "three agents write a report");
    assert_eq!(
        disc.world.holds(&card2),
        Some(true),
        "two agents write a report should be TRUE with 2 witnesses"
    );
    assert_ne!(
        disc.world.holds(&card3),
        Some(true),
        "OVER-DERIVATION: three agents write a report must NOT be TRUE (only 2 witnesses)"
    );
    // Stronger: with a closed-enough 2-member ceiling it should be determinately FALSE.
    assert_eq!(
        disc.world.holds(&card3),
        Some(false),
        "three agents write a report should be FALSE (ceiling 2 < 3)"
    );
}

// ---------------------------------------------------------------------------
// 2. DISTINCTNESS: counting is over distinct ENTITIES, not mentions. Reading
//    the SAME writer twice must not inflate the count. Reading a writer who
//    writes a DIFFERENT patient must not count toward "report".
// ---------------------------------------------------------------------------
#[test]
fn count_is_over_distinct_entities_not_mentions() {
    let mut w = World::new();
    // editor writes a report — asserted twice (repeat mention).
    w.assert(&Meaning::Event(write_event("editor", "report", false)));
    w.assert(&Meaning::Event(write_event("editor", "report", false)));
    // author writes a report.
    w.assert(&Meaning::Event(write_event("author", "report", false)));
    // teacher writes a LETTER, not a report — must NOT count toward report.
    w.assert(&Meaning::Event(write_event("teacher", "letter", false)));

    let n = w.count_satisfying("agent", &report_body());
    assert_eq!(
        n, 2,
        "distinct report-writers are editor+author = 2; the repeated editor mention and \
         the letter-writer must not change the count, got {n}"
    );
}

// ---------------------------------------------------------------------------
// 3. SOUNDNESS / NO OVER-DERIVATION at the boundary: a determined-FALSE member
//    can never count, and the at-least-3 query is genuinely false, not vacuous.
// ---------------------------------------------------------------------------
#[test]
fn negated_writer_never_counts_and_no_false_entailment() {
    let mut w = World::new();
    w.assert(&Meaning::Event(write_event("editor", "report", false))); // writes
    w.assert(&Meaning::Event(write_event("author", "report", false))); // writes
    w.assert(&Meaning::Event(write_event("teacher", "report", true))); // does NOT write

    // Only 2 genuine writers.
    assert_eq!(w.count_satisfying("agent", &report_body()), 2);

    let card = |n: usize| Meaning::Cardinal {
        at_least: n,
        var_category: "agent".to_string(),
        body: report_body(),
    };
    // Monotone: >=1 and >=2 true.
    assert_eq!(w.holds(&card(1)), Some(true));
    assert_eq!(w.holds(&card(2)), Some(true));
    // >=3 FALSE: editor is determined-false so ceiling = 2 < 3. No over-derivation.
    assert_eq!(
        w.holds(&card(3)),
        Some(false),
        "OVER-DERIVATION: >=3 must be FALSE (editor determined-false, ceiling 2)"
    );
}

// ---------------------------------------------------------------------------
// 4. OPEN-WORLD HONESTY: an unrecognized category must not fabricate a count
//    or a truth value. Counting an empty/unknown category is 0, and the
//    at-least claim over it is undetermined (None), never a confident yes/no.
// ---------------------------------------------------------------------------
#[test]
fn unknown_category_does_not_fabricate() {
    let mut w = World::new();
    w.assert(&Meaning::Event(write_event("editor", "report", false)));
    // "dragon" is not a known category.
    assert_eq!(w.count_satisfying("dragon", &report_body()), 0);
    let card = Meaning::Cardinal {
        at_least: 1,
        var_category: "dragon".to_string(),
        body: report_body(),
    };
    assert_eq!(
        w.holds(&card),
        None,
        "an unknown category must yield None, not a fabricated truth value"
    );
}

// ---------------------------------------------------------------------------
// 5. THE TASK'S LITERAL REQUEST, probed honestly: "exactly two distinct
//    TEACHERS write a report." Entities are keyed by noun HEAD, so two
//    "teacher" mentions collapse to one entity. We record the ACTUAL behavior
//    so the report can state the representational limit precisely. This test
//    asserts the observed truth (collapse to 1) — it documents, it does not
//    pretend the system can do per-instance counting it cannot.
// ---------------------------------------------------------------------------
#[test]
fn two_same_noun_teachers_collapse_to_one_entity() {
    let engine = Engine::new();
    let mut disc = Discourse::new();
    // Two teachers, same surface noun.
    disc.read(&engine, "The teacher writes the report.");
    disc.read(&engine, "The teacher writes the report.");

    let n = disc.world.count_satisfying("teacher", &report_body());
    // Observed: the two same-noun teachers are ONE entity "teacher".
    assert_eq!(
        n, 1,
        "two same-noun teachers collapse to a single entity keyed by head 'teacher'; \
         the model counts distinct heads, so this is 1, not 2 (representational limit)"
    );

    // The English count question therefore answers ONE, not TWO. This is SOUND
    // for distinct-entity counting (there is one known teacher), but it is NOT
    // the per-instance "two teachers" reading a human would intend.
    let ans = qa::answer(&engine, &disc, "How many teachers write a report?");
    let low = ans.to_lowercase();
    assert!(
        low.contains("one") || low.contains('1'),
        "with one distinct teacher entity the count answer is one, got: {ans:?}"
    );
}

// ---------------------------------------------------------------------------
// 6. THE TASK SPEC, satisfiable form: two DISTINCT agent entities require
//    distinct heads (editor, author are both recognized agents/persons). Verify
//    the count question itself routes to a NUMBER (not a yes/no), exactly.
// ---------------------------------------------------------------------------
#[test]
fn count_question_routes_to_a_number_not_a_yesno() {
    let engine = Engine::new();
    let mut disc = Discourse::new();
    disc.read(&engine, "The editor writes the report.");
    disc.read(&engine, "The author writes the report.");
    let m = semantics::understand(&engine, "how many agents write a report");
    assert!(
        matches!(m, Meaning::CountQuestion { .. }),
        "a 'how many' question must parse to a CountQuestion, got {m:?}"
    );
    // CountQuestion is a query: world.holds must REFUSE to truth-value it.
    assert_eq!(
        disc.world.holds(&m),
        None,
        "a CountQuestion is a numeric query, never truth-evaluated"
    );
}

// ---------------------------------------------------------------------------
// 7. THE COUNT IS COMPUTED, NOT HARDCODED: the same English count question over
//    a ONE-writer world answers "One", over a THREE-writer world answers
//    "Three". A demo that memorized "Two" would fail both. And in the ONE-writer
//    world, "two agents write a report" must NOT be TRUE (no over-derivation).
// ---------------------------------------------------------------------------
#[test]
fn count_tracks_the_actual_number_one_and_three() {
    let engine = Engine::new();

    // ONE writer.
    let mut one = Discourse::new();
    one.read(&engine, "The editor writes the report.");
    let a1 = qa::answer(&engine, &one, "How many agents write a report?").to_lowercase();
    assert!(
        a1.contains("one") || a1.contains('1'),
        "one writer -> count one, got: {a1:?}"
    );
    // SOUNDNESS: with a single witness, "two agents write a report" is NOT true.
    let card2 = semantics::understand(&engine, "two agents write a report");
    assert_ne!(
        one.world.holds(&card2),
        Some(true),
        "OVER-DERIVATION: two agents write a report must NOT be TRUE with 1 writer"
    );

    // THREE writers (three distinct recognized agent nouns).
    let mut three = Discourse::new();
    three.read(&engine, "The editor writes the report.");
    three.read(&engine, "The author writes the report.");
    three.read(&engine, "The teacher writes the report.");
    let a3 = qa::answer(&engine, &three, "How many agents write a report?").to_lowercase();
    assert!(
        a3.contains("three") || a3.contains('3'),
        "three writers -> count three, got: {a3:?}"
    );
    // at-least-3 now genuinely TRUE; at-least-4 not TRUE.
    let card3 = semantics::understand(&engine, "three agents write a report");
    assert_eq!(three.world.holds(&card3), Some(true));
}
