//! ADVERSARIAL verification of the deep-semantics layer.
//!
//! Goal: prove the disjunction + inference additions are SOUND — they must
//! report a true disjunction when one disjunct holds, a false one when all
//! fail, and must NEVER over-derive (no false entailment). These tests are
//! written by an external skeptic, not the module author, and target the
//! specific soundness traps named in the task.

use mog_synth::understanding::inference::{relation, Relation};
use mog_synth::understanding::meaning::{Event, Meaning, Quantifier, Tense, Term};
use mog_synth::understanding::world_model::World;

fn ent(s: &str) -> Term {
    Term::Entity(s.to_string())
}
fn indef(s: &str) -> Term {
    Term::Indefinite(s.to_string())
}

fn event(pred: &str, agent: Option<Term>, patient: Option<Term>, negated: bool) -> Meaning {
    Meaning::Event(Event {
        predicate: pred.to_string(),
        agent,
        patient,
        recipient: None,
        tense: Tense::Present,
        negated,
    })
}

fn prop(subj: Term, property: &str, negated: bool) -> Meaning {
    Meaning::HasProperty {
        subject: subj,
        property: property.to_string(),
        negated,
    }
}

fn quantified(q: Quantifier, cat: &str) -> Meaning {
    Meaning::Quantified {
        quant: q,
        var_category: cat.to_string(),
        body: Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(indef("report")),
            recipient: None,
            tense: Tense::Present,
            negated: false,
        },
    }
}

fn entails(p: &Meaning, h: &Meaning) -> bool {
    matches!(relation(p, h), Relation::Entails)
}

// ===========================================================================
// PART 1 — DISJUNCTION TRUTH (world model `holds`)
// ===========================================================================

#[test]
fn disjunction_true_when_exactly_one_disjunct_holds() {
    // World knows: "the teacher writes the report" (and nothing about author).
    let mut w = World::new();
    let fact = event("write", Some(ent("teacher")), Some(ent("report")), false);
    w.assert(&fact);

    // "teacher writes report" OR "author reads book"
    let disj = Meaning::Or(vec![
        event("write", Some(ent("teacher")), Some(ent("report")), false),
        event("read", Some(ent("author")), Some(ent("book")), false),
    ]);
    // One disjunct (the first) holds => the disjunction is TRUE.
    assert_eq!(
        w.holds(&disj),
        Some(true),
        "X-or-Y must be true when one disjunct holds"
    );
}

#[test]
fn disjunction_true_when_only_second_disjunct_holds() {
    // Order independence: truth of the SECOND disjunct must also surface.
    let mut w = World::new();
    w.assert(&event("read", Some(ent("author")), Some(ent("book")), false));

    let disj = Meaning::Or(vec![
        event("write", Some(ent("teacher")), Some(ent("report")), false),
        event("read", Some(ent("author")), Some(ent("book")), false),
    ]);
    assert_eq!(w.holds(&disj), Some(true));
}

#[test]
fn disjunction_false_when_both_disjuncts_determined_false() {
    // World explicitly DENIES both disjuncts (negated facts) => disjunction false.
    let mut w = World::new();
    w.assert(&event("write", Some(ent("teacher")), Some(ent("report")), true)); // teacher does NOT write report
    w.assert(&event("read", Some(ent("author")), Some(ent("book")), true)); // author does NOT read book

    let disj = Meaning::Or(vec![
        event("write", Some(ent("teacher")), Some(ent("report")), false),
        event("read", Some(ent("author")), Some(ent("book")), false),
    ]);
    assert_eq!(
        w.holds(&disj),
        Some(false),
        "X-or-Y must be FALSE when both disjuncts are determined false"
    );
}

#[test]
fn disjunction_unknown_when_no_disjunct_true_and_some_undetermined() {
    // One disjunct denied, the other simply unknown => open-world None,
    // NOT a spurious true or false. (Guards against over-claiming.)
    let mut w = World::new();
    w.assert(&event("write", Some(ent("teacher")), Some(ent("report")), true)); // denied

    let disj = Meaning::Or(vec![
        event("write", Some(ent("teacher")), Some(ent("report")), false),
        event("read", Some(ent("author")), Some(ent("book")), false), // unknown
    ]);
    assert_eq!(
        w.holds(&disj),
        None,
        "disjunction must be undetermined, not spuriously true/false"
    );
}

// ===========================================================================
// PART 2 — DISJUNCTION ENTAILMENT (inference `relation`)
// ===========================================================================

#[test]
fn disjunct_entails_disjunction_but_disjunction_does_not_pick_a_disjunct() {
    let a = event("write", Some(ent("teacher")), Some(ent("report")), false);
    let b = event("read", Some(ent("author")), Some(ent("book")), false);
    let disj = Meaning::Or(vec![a.clone(), b.clone()]);

    // SOUND: asserting A entails "A or B".
    assert!(entails(&a, &disj), "a disjunct must entail the disjunction");

    // ADVERSARIAL/SOUNDNESS: "A or B" must NOT entail A (we cannot pick a
    // disjunct). This is the classic disjunction-elimination trap.
    assert!(
        !entails(&disj, &a),
        "UNSOUND: disjunction must not entail one of its disjuncts"
    );
    assert!(
        !entails(&disj, &b),
        "UNSOUND: disjunction must not entail one of its disjuncts"
    );

    // An unrelated premise must not entail the disjunction.
    let unrelated = event("read", Some(ent("doctor")), Some(ent("letter")), false);
    assert!(
        !entails(&unrelated, &disj),
        "UNSOUND: unrelated premise must not entail the disjunction"
    );
}

// ===========================================================================
// PART 3 — SOUNDNESS TRAPS (no false entailment / over-derivation)
// ===========================================================================

#[test]
fn some_does_not_entail_every() {
    // "some teacher writes a report" must NOT entail "every teacher writes a report".
    let some = quantified(Quantifier::Some, "teacher");
    let every = quantified(Quantifier::Every, "teacher");
    assert!(
        !entails(&some, &every),
        "UNSOUND OVER-DERIVATION: some must not entail every"
    );
    // And it must not silently be reported as a contradiction either.
    assert!(
        matches!(relation(&some, &every), Relation::Neutral),
        "some vs every should be Neutral, got non-neutral"
    );
}

#[test]
fn some_does_not_entail_every_at_world_truth_level() {
    // World level: one teacher writes a report, a second teacher does NOT.
    // "some teacher writes a report" is TRUE; "every teacher writes a report"
    // must be FALSE — the layer must not let the existential leak into the
    // universal.
    let mut w = World::new();
    // Register two teachers via events so both are known members of "teacher".
    w.assert(&event("write", Some(ent("teacher")), Some(ent("report")), false));
    // Second teacher entity: give it a distinct head but same category. We use
    // a second agent noun present in the lexicon to be a teacher-category member
    // only if it shares the head; instead we assert a negated fact for a second
    // explicit teacher-like entity through IsA + event.
    // Simpler: assert teacher2 is a teacher, and that teacher2 does NOT write.
    w.assert(&Meaning::IsA {
        subject: ent("teacher2"),
        category: "teacher".to_string(),
        negated: false,
    });
    w.assert(&event("write", Some(ent("teacher2")), Some(ent("report")), true));

    let some = quantified(Quantifier::Some, "teacher");
    let every = quantified(Quantifier::Every, "teacher");
    // some is true (teacher writes), every is false (teacher2 counterexample).
    assert_eq!(w.holds(&some), Some(true), "some teacher writes => true");
    assert_eq!(
        w.holds(&every),
        Some(false),
        "UNSOUND if every is true: teacher2 is a counterexample"
    );
}

#[test]
fn property_does_not_leak_across_entities() {
    // "the teacher is careful" must NOT make "the editor is careful" true.
    let mut w = World::new();
    w.assert(&prop(ent("teacher"), "careful", false));

    // The asserted entity holds the property.
    assert_eq!(
        w.holds(&prop(ent("teacher"), "careful", false)),
        Some(true),
        "the teacher should be careful"
    );
    // A DIFFERENT entity must NOT inherit it — open-world unknown, not true.
    assert_eq!(
        w.holds(&prop(ent("editor"), "careful", false)),
        None,
        "UNSOUND OVER-DERIVATION: editor's carefulness must not be derived from teacher's"
    );
    assert_ne!(
        w.holds(&prop(ent("editor"), "careful", false)),
        Some(true),
        "UNSOUND: editor must not be reported careful"
    );
}

#[test]
fn property_does_not_leak_across_entities_at_inference_level() {
    // Inference level: "teacher is careful" must not entail "editor is careful".
    let teacher_careful = prop(ent("teacher"), "careful", false);
    let editor_careful = prop(ent("editor"), "careful", false);
    assert!(
        !entails(&teacher_careful, &editor_careful),
        "UNSOUND: teacher's property must not entail editor's"
    );
    assert!(
        matches!(relation(&teacher_careful, &editor_careful), Relation::Neutral),
        "different-entity properties should be Neutral"
    );
}

#[test]
fn different_property_of_same_entity_not_derived() {
    // "the teacher is careful" must NOT make "the teacher is calm" true.
    let mut w = World::new();
    w.assert(&prop(ent("teacher"), "careful", false));
    assert_eq!(
        w.holds(&prop(ent("teacher"), "calm", false)),
        None,
        "UNSOUND: a different property must not be derived from carefulness"
    );
}

#[test]
fn existential_event_does_not_specialize() {
    // "a teacher writes the report" (existential) must NOT entail
    // "the teacher writes the report" (specific entity). Generalization is
    // sound only definite -> indefinite, never the reverse.
    let some_teacher = event("write", Some(indef("teacher")), Some(ent("report")), false);
    let the_teacher = event("write", Some(ent("teacher")), Some(ent("report")), false);
    assert!(
        !entails(&some_teacher, &the_teacher),
        "UNSOUND: indefinite must not entail a specific definite"
    );
}

#[test]
fn dropping_patient_does_not_run_backwards() {
    // "the teacher writes [something]" must NOT entail "the teacher writes the report".
    let writes_something = event("write", Some(ent("teacher")), None, false);
    let writes_report = event("write", Some(ent("teacher")), Some(ent("report")), false);
    assert!(
        !entails(&writes_something, &writes_report),
        "UNSOUND: existential patient must not entail a specific patient"
    );
}

#[test]
fn taxonomy_does_not_run_downward() {
    // "x is an agent" must NOT entail "x is a teacher" (agents need not be teachers).
    let agent = Meaning::IsA {
        subject: ent("x"),
        category: "agent".to_string(),
        negated: false,
    };
    let teacher = Meaning::IsA {
        subject: ent("x"),
        category: "teacher".to_string(),
        negated: false,
    };
    assert!(
        !entails(&agent, &teacher),
        "UNSOUND: hypernym must not entail hyponym"
    );
}

#[test]
fn negated_event_does_not_generalize() {
    // "the teacher does NOT write the report" must NOT entail "the teacher does
    // not write anything" (dropping the patient under negation is unsound).
    let not_report = event("write", Some(ent("teacher")), Some(ent("report")), true);
    let not_anything = event("write", Some(ent("teacher")), None, true);
    assert!(
        !entails(&not_report, &not_anything),
        "UNSOUND: negated event must not drop its patient"
    );
}

#[test]
fn affirmative_does_not_entail_its_own_negation() {
    let yes = event("write", Some(ent("teacher")), Some(ent("report")), false);
    let no = event("write", Some(ent("teacher")), Some(ent("report")), true);
    // It should CONTRADICT, never ENTAIL.
    assert!(
        !entails(&yes, &no),
        "UNSOUND: an affirmative event must not entail its negation"
    );
    assert!(matches!(relation(&yes, &no), Relation::Contradicts));
}
