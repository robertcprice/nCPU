//! ADVERSARIAL HYPOTHETICAL-ISOLATION PROBE — external skeptic, not the author.
//!
//! CLAIM UNDER TEST: nCPU's reflection / deeper-reasoning layer evaluates
//! `suppose(..)` ("suppose X; then would Y?") and `what_if_not(..)` ("what if X
//! had NOT been true?") in a HYPOTHETICAL world that MUST NOT leak into the real
//! world.
//!
//! We attack the claim from two angles, both via the PUBLIC API only:
//!
//!   (A) MIND LEVEL: drive the real `Mind`. Record `mind.ask(Q0)` for several
//!       baseline questions, run BOTH `suppose(..)` and `what_if_not(..)`, then
//!       assert each `mind.ask(Q0)` is BYTE-IDENTICAL to before. The questions
//!       are chosen to be the ones most at risk of a leak: one about the *editor*
//!       (subject of the supposition) and one about the *teacher* (subject of the
//!       retracted fact).
//!
//!   (B) WORLD LEVEL: reproduce the EXACT internal mechanism on a public
//!       `Discourse` (this is what `Mind` holds and clones). Snapshot the real
//!       world's `facts()` / `entities()` / `contradictions()`, perform the same
//!       clone-and-mutate that `suppose`/`what_if_not` do, then assert the
//!       ORIGINAL discourse's world is unchanged — proving the clone is a deep,
//!       isolating copy and the hypothetical mutation cannot alias back.
//!
//! NON-VACUITY: we additionally confirm the supposition GENUINELY affected the
//! hypothetical answer (the editor "writes something" only UNDER the supposition,
//! never in the real world) so the isolation guarantee is not satisfied trivially
//! by the methods being no-ops.
//!
//! Any leak — a changed `ask` answer, a changed `facts()`, a new entity, or a new
//! contradiction in the real world — is an automatic FAIL.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::inference::polarity_flip;
use mog_synth::understanding::meaning::Event;
use mog_synth::understanding::mind::Mind;
use mog_synth::understanding::{qa, semantics};

/// (A) MIND LEVEL: `suppose` + `what_if_not` must leave every `mind.ask(..)`
/// answer byte-identical, and the supposition must be NON-VACUOUS.
#[test]
fn mind_level_suppose_and_what_if_not_never_leak_into_ask() {
    let mut mind = Mind::new();
    mind.read("The teacher writes the report.");
    mind.read("The author reads the book.");

    // ---- BASELINE: record ask(Q0) for the questions most at risk of leaking. --
    let q0 = "What does the author read?";
    let q_editor = "Does the editor write something?";
    let q_teacher = "Does the teacher write the report?";

    let baseline_q0 = mind.ask(q0);
    let baseline_editor = mind.ask(q_editor);
    let baseline_teacher = mind.ask(q_teacher);

    // Guard the NON-VACUITY precondition: in the REAL world the editor is unknown
    // (never read), so the supposition has real work to do.
    assert!(
        !baseline_editor.to_lowercase().starts_with("yes"),
        "precondition: the editor must NOT already write in the real world, else \
         the supposition is vacuous. Got: {baseline_editor:?}"
    );

    // ---- RUN BOTH HYPOTHETICAL METHODS ON THE REAL MIND ----------------------
    let supposed = mind.suppose(
        "the editor writes the report",
        "does the editor write something?",
    );
    let counterfactual = mind.what_if_not(
        "the teacher writes the report",
        "does the teacher write the report?",
    );

    // ---- NON-VACUITY: the hypotheticals genuinely changed the hypothetical ----
    // answer. `suppose` must answer affirmatively UNDER the supposition; the real
    // world (asserted above) does not.
    assert!(
        supposed.to_lowercase().contains("then yes"),
        "NON-VACUOUS: supposition must make 'the editor writes something' true in \
         the hypothetical. Got: {supposed:?}"
    );
    // `what_if_not` must report a genuine FLIP ("... rather than ..."), not a
    // no-op ("the answer would be the same").
    assert!(
        counterfactual.to_lowercase().contains("rather than"),
        "NON-VACUOUS: the counterfactual retraction must flip the verdict. \
         Got: {counterfactual:?}"
    );

    // ---- NO LEAK: every baseline ask is BYTE-IDENTICAL -----------------------
    assert_eq!(
        mind.ask(q0).as_bytes(),
        baseline_q0.as_bytes(),
        "LEAK: Q0 answer changed after hypotheticals"
    );
    assert_eq!(
        mind.ask(q_editor).as_bytes(),
        baseline_editor.as_bytes(),
        "LEAK: `suppose` leaked the editor assumption into the real world"
    );
    assert_eq!(
        mind.ask(q_teacher).as_bytes(),
        baseline_teacher.as_bytes(),
        "LEAK: `what_if_not` leaked the teacher negation into the real world"
    );
    // The real teacher fact still holds affirmatively (the negation did not stick).
    assert!(
        mind.ask(q_teacher).to_lowercase().starts_with("yes"),
        "the real teacher fact must survive the counterfactual: {:?}",
        mind.ask(q_teacher)
    );
    // And the real world still has NO contradiction recorded.
    assert!(
        mind.contradictions().is_empty(),
        "LEAK: a counterfactual recorded a contradiction in the real world"
    );
}

/// (B) WORLD LEVEL: reproduce the EXACT clone-and-mutate that `suppose` and
/// `what_if_not` perform internally, on a public `Discourse`, and prove the
/// ORIGINAL world's `facts()` (and entities/contradictions) are untouched.
///
/// `Mind::suppose`  does: `let mut h = self.discourse.clone(); h.read(engine, assumption);`
/// `Mind::what_if_not` does: `let mut c = self.discourse.clone(); c.world.assert(&negation);`
/// We perform both, then assert the source `Discourse`'s world is byte-for-byte
/// (value-for-value) what it was before.
#[test]
fn world_level_clone_isolation_leaves_facts_untouched() {
    let engine = Engine::new();
    let mut discourse = Discourse::new();
    discourse.read(&engine, "The teacher writes the report.");
    discourse.read(&engine, "The author reads the book.");

    // ---- SNAPSHOT the real world's observable state --------------------------
    let facts_before: Vec<Event> = discourse.world.facts().to_vec();
    let entities_before: Vec<String> = discourse.world.entities();
    let contradictions_before: usize = discourse.world.contradictions().len();
    assert_eq!(
        facts_before.len(),
        2,
        "sanity: two facts were read into the real world"
    );

    // ---- MIRROR `suppose`: clone, read the assumption into the CLONE ---------
    let supposed_answer = {
        let mut hypothetical = discourse.clone();
        hypothetical.read(&engine, "the editor writes the report");
        qa::answer(&engine, &hypothetical, "does the editor write something?")
    };
    // NON-VACUITY: under the supposition the clone answers affirmatively...
    assert!(
        supposed_answer.to_lowercase().starts_with("yes"),
        "NON-VACUOUS: the supposition makes the editor write something in the \
         clone: {supposed_answer:?}"
    );
    // ...but the REAL world still does not (the assumption stayed in the clone).
    let real_editor = qa::answer(&engine, &discourse, "does the editor write something?");
    assert!(
        !real_editor.to_lowercase().starts_with("yes"),
        "LEAK: the supposed editor fact escaped into the real world: {real_editor:?}"
    );

    // ---- MIRROR `what_if_not`: clone, assert the contradictory into the CLONE -
    let fact_meaning = semantics::understand(&engine, "the teacher writes the report");
    let negation =
        polarity_flip(&fact_meaning).expect("an asserted event must have a sound contradictory");
    let cf_answer = {
        let mut counterfactual = discourse.clone();
        counterfactual.world.assert(&negation);
        qa::answer(
            &engine,
            &counterfactual,
            "does the teacher write the report?",
        )
    };
    // NON-VACUITY: in the counterfactual clone the verdict flipped away from "yes".
    assert!(
        !cf_answer.to_lowercase().starts_with("yes"),
        "NON-VACUOUS: asserting the negation in the clone flips the verdict: {cf_answer:?}"
    );

    // ---- NO LEAK: the ORIGINAL world is byte/value-identical -----------------
    let facts_after: Vec<Event> = discourse.world.facts().to_vec();
    assert_eq!(
        facts_after, facts_before,
        "LEAK: cloning + mutating the clone changed the original world's facts()"
    );
    assert_eq!(
        discourse.world.entities(),
        entities_before,
        "LEAK: the original world's entities changed"
    );
    assert_eq!(
        discourse.world.contradictions().len(),
        contradictions_before,
        "LEAK: the counterfactual negation recorded a contradiction in the real world"
    );
    // The real teacher fact still answers "yes" — the negation never aliased back.
    let real_teacher = qa::answer(&engine, &discourse, "does the teacher write the report?");
    assert!(
        real_teacher.to_lowercase().starts_with("yes"),
        "LEAK: the real teacher fact was disturbed by the counterfactual: {real_teacher:?}"
    );
}
