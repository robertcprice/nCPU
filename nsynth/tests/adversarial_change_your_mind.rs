//! ADVERSARIAL verification: "change-your-mind-is-real".
//!
//! Skeptic's claim under test: `Mind::what_would_change_your_mind` names a flip
//! that GENUINELY flips the verdict — not a plausible-sounding counterfactual that
//! would leave the answer unchanged.
//!
//! These tests are written by an external adversary using ONLY the public crate
//! API (no private `Mind` internals: no `query_atoms`, no `proof_leaves`, no
//! `counterfactual_changes_verdict`). The genuineness check is reconstructed from
//! scratch:
//!   1. Build a Discourse and read facts so a yes/no question answers Yes BY A PROOF.
//!   2. Call `what_would_change_your_mind`.
//!   3. Independently extract the proof's asserted leaf, form its sound
//!      contradictory (`inference::polarity_flip`), assert it into a CLONE of the
//!      discourse, and confirm the PUBLIC `qa::answer` verdict actually changes.
//! A named flip that does NOT change the verdict in this independent re-derivation
//! means the property is FALSE.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::inference::{polarity_flip, prove, Proof};
use mog_synth::understanding::meaning::Meaning;
use mog_synth::understanding::mind::Mind;
use mog_synth::understanding::{qa, semantics};

/// The three values an answer can carry, derived from its surface string the same
/// way the system distinguishes them ("Yes," / "No," / "I don't know.").
/// Reconstructed independently of the module's private `Verdict`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum V {
    Yes,
    No,
    Idk,
    Other,
}

fn verdict_of(answer: &str) -> V {
    let a = answer.trim().to_lowercase();
    if a.starts_with("yes") {
        V::Yes
    } else if a.starts_with("no") {
        V::No
    } else if a.contains("don't know") {
        V::Idk
    } else {
        V::Other
    }
}

/// Walk a public `Proof` and collect its asserted leaves (rule == "asserted",
/// no premises). Independent reimplementation of mind.rs's private `proof_leaves`.
fn proof_leaves(p: &Proof) -> Vec<Meaning> {
    let mut out = Vec::new();
    fn go(p: &Proof, out: &mut Vec<Meaning>) {
        if p.rule == "asserted" && p.premises.is_empty() {
            if !out.contains(&p.conclusion) {
                out.push(p.conclusion.clone());
            }
            return;
        }
        for prem in &p.premises {
            go(prem, out);
        }
    }
    go(p, &mut out);
    out
}

/// Re-derive a goal's verdict over a fact set using ONLY the public `prove`
/// + `polarity_flip` — the model-theoretic counterpart, built independently.
fn verdict_against(facts: &[Meaning], goal: &Meaning) -> V {
    if prove(facts, goal).is_some() {
        return V::Yes;
    }
    if let Some(flip) = polarity_flip(goal) {
        if prove(facts, &flip).is_some() {
            return V::No;
        }
    }
    V::Idk
}

/// THE headline property, on the PROOF-BACKED branch — under BELIEF REVISION.
///
/// We read one concrete fact, then ask a TRANSITIVELY entailed yes/no question
/// ("Does a teacher write something?") which is true by chaining drop-patient +
/// generalize-agent — so the answer is Yes AND `answer_explained` returns a PROOF
/// (the `Verdict::Yes => Some(p)` branch, NOT the opaque directly-asserted branch).
///
/// The world model now performs BELIEF REVISION: a directly-asserted fact whose
/// negation is later directly asserted is RETRACTED (most-recent-wins resolves the
/// F-vs-NOT-F clash to one coherent belief; see `world_model::Revision`). So
/// TELLING the mind "the teacher does NOT write the report" — the only way
/// information enters the world — genuinely RETRACTS the supporting leaf, and the
/// existential generalization that rested on it loses its only support: the verdict
/// moves OFF Yes. The named flip is therefore REAL, not a false promise. This test
/// pins both, against the actual revised semantics:
///   (i)  `what_would_change_your_mind` honestly NAMES the leaf's negation as the
///        flip (it is genuinely tellable now that revision is in force);
///   (ii) telling the leaf's negation through the PUBLIC path independently moves
///        the public verdict away from Yes — the genuineness check that proves the
///        named flip actually flips, reconstructed from scratch by the adversary.
/// A flip the impl names that did NOT actually move the verdict would falsify (i)+(ii).
#[test]
fn proof_backed_yes_names_a_real_flip_that_revision_makes_genuine() {
    let engine = Engine::new();
    let mut discourse = Discourse::new();
    discourse.read(&engine, "The teacher writes the report.");

    let q = "Does a teacher write something?";

    // (a) Baseline: a determined Yes that REALLY rests on a proof.
    let baseline = qa::answer(&engine, &discourse, q);
    assert_eq!(
        verdict_of(&baseline),
        V::Yes,
        "baseline must be Yes: {baseline}"
    );

    let parsed = semantics::understand(&engine, q);
    let (_ans, proof) = qa::answer_explained(&engine, &discourse, &parsed);
    let proof = proof.expect("a transitively-entailed Yes must carry a PROOF (the Some branch)");
    let leaves = proof_leaves(&proof);
    assert_eq!(
        leaves.len(),
        1,
        "exactly one asserted leaf backs this derivation: {leaves:?}"
    );
    let leaf = leaves[0].clone();

    // (b) Build a Mind over the SAME facts and ask what would change its mind.
    let mut mind = Mind::new();
    mind.read("The teacher writes the report.");
    let baseline_mind = mind.ask(q);
    assert_eq!(
        verdict_of(&baseline_mind),
        V::Yes,
        "Mind baseline Yes: {baseline_mind}"
    );
    let wwcym = mind.what_would_change_your_mind(q);
    let lw = wwcym.to_lowercase();
    // SOUND under revision: telling the leaf's negation IS actionable (it retracts
    // the support), so the mind honestly names it as a flip.
    assert!(
        lw.contains("i would change my mind if you told me"),
        "names the genuinely-tellable flip: {wwcym}"
    );
    assert!(
        lw.contains("the teacher does not write the report"),
        "names the contradictory of the supporting leaf: {wwcym}"
    );

    // (c) THE GROUNDING CHECK, through the PUBLIC path: asserting the leaf's
    // negation into a CLONE is the real way a user "tells" the mind something.
    // Belief revision RETRACTS the positive leaf, so the existential generalization
    // loses its support and the verdict moves OFF Yes. This genuine flip is exactly
    // WHY naming it (not a dependency) is the correct, sound report.
    let flip = polarity_flip(&leaf).expect("an event leaf has a contradictory");
    let mut clone = discourse.clone();
    clone.world.assert(&flip);
    let cloned_answer = qa::answer(&engine, &clone, q);
    assert_ne!(
        verdict_of(&cloned_answer),
        V::Yes,
        "telling the mind the leaf's negation retracts the support under belief \
         revision — the named flip genuinely moves the verdict off Yes. Got: {cloned_answer}"
    );

    // (e) NO LEAK: the original discourse is unchanged — same baseline verdict.
    let after = qa::answer(&engine, &discourse, q);
    assert_eq!(
        after, baseline,
        "verification must not mutate the source discourse"
    );

    eprintln!("--- wwcym: {wwcym}\n--- cloned answer: {cloned_answer} (was {baseline}) ---");
}

/// THE headline property, on the OPAQUE directly-asserted branch (proof == None):
/// a direct truth query answered Yes by the world model owning the fact. The flip
/// is the negation of the queried proposition itself.
#[test]
fn named_flip_on_an_opaque_yes_genuinely_flips_the_public_verdict() {
    let engine = Engine::new();
    let mut discourse = Discourse::new();
    discourse.read(&engine, "The teacher writes the report.");

    let q = "Does the teacher write the report?";
    let baseline = qa::answer(&engine, &discourse, q);
    assert_eq!(verdict_of(&baseline), V::Yes, "baseline Yes: {baseline}");

    let mut mind = Mind::new();
    mind.read("The teacher writes the report.");
    let wwcym = mind.what_would_change_your_mind(q).to_lowercase();
    assert!(
        wwcym.contains("i would change my mind if you told me"),
        "names a flip: {wwcym}"
    );
    assert!(
        wwcym.contains("the teacher does not write the report"),
        "names the contradictory of the queried fact: {wwcym}"
    );

    // Independent: assert the contradictory of the queried proposition into a clone.
    let proposition = semantics::understand(&engine, "the teacher writes the report");
    let flip = polarity_flip(&proposition).expect("an event has a contradictory");
    let mut clone = discourse.clone();
    clone.world.assert(&flip);
    let cloned_answer = qa::answer(&engine, &clone, q);
    assert_ne!(
        verdict_of(&cloned_answer),
        verdict_of(&baseline),
        "the named flip genuinely flips the verdict: {cloned_answer} (was {baseline})"
    );
}

/// FALSIFIABILITY / NEGATIVE CONTROL: the system must NOT name a flip that fails to
/// change the verdict. Two INDEPENDENT facts entail the same generalized goal, so
/// negating ONE leaf still leaves the other proving the goal — every single-leaf
/// flip is a verified no-op and MUST be dropped. If the impl named such a flip,
/// "named flip GENUINELY flips" would be FALSE.
#[test]
fn a_flip_that_does_not_change_the_verdict_is_dropped() {
    let engine = Engine::new();
    let mut discourse = Discourse::new();
    discourse.read(&engine, "The teacher writes the report.");
    discourse.read(&engine, "The teacher writes the letter.");

    let q = "Does a teacher write something?";
    let baseline = qa::answer(&engine, &discourse, q);
    assert_eq!(verdict_of(&baseline), V::Yes, "baseline Yes: {baseline}");

    // Independently confirm: negate ONE leaf (report) but keep the OTHER (letter).
    // The surviving letter fact still entails the existential goal => verdict UNCHANGED.
    let report = semantics::understand(&engine, "the teacher writes the report");
    let letter = semantics::understand(&engine, "the teacher writes the letter");
    let flip_report = polarity_flip(&report).unwrap();
    // The queried goal: "a teacher writes something" — recover it from the proof.
    let parsed = semantics::understand(&engine, q);
    let (_a, proof) = qa::answer_explained(&engine, &discourse, &parsed);
    let goal = proof.expect("proof-backed Yes").conclusion;
    let facts = vec![flip_report, letter.clone()];
    assert_eq!(
        verdict_against(&facts, &goal),
        V::Yes,
        "negating one of two independent supports must NOT flip the verdict"
    );

    // The system must therefore NOT name either single-leaf negation as a flip.
    let mut mind = Mind::new();
    mind.read("The teacher writes the report.");
    mind.read("The teacher writes the letter.");
    let wwcym = mind.what_would_change_your_mind(q).to_lowercase();
    assert!(
        !wwcym.contains("does not write the report"),
        "must NOT name a single-leaf flip that does not change the verdict: {wwcym}"
    );
    assert!(
        !wwcym.contains("does not write the letter"),
        "must NOT name the other no-op single-leaf flip either: {wwcym}"
    );
    assert!(
        wwcym.contains("nothing you could tell me would change my answer"),
        "honest 'nothing changes it' when every candidate flip is a no-op: {wwcym}"
    );
    let _ = report;
}

/// CROSS-CHECK: an undetermined query. The mind says "I don't know"; the flips it
/// names are the proposition (→ Yes) and its negation (→ No). Each must genuinely
/// move the verdict away from Idk when asserted into a clone.
#[test]
fn named_flips_on_an_open_question_each_genuinely_decide_it() {
    let engine = Engine::new();
    let mut discourse = Discourse::new();
    discourse.read(&engine, "The teacher writes the report.");

    let q = "Does the author read the book?";
    let baseline = qa::answer(&engine, &discourse, q);
    assert_eq!(
        verdict_of(&baseline),
        V::Idk,
        "baseline undetermined: {baseline}"
    );

    let mut mind = Mind::new();
    mind.read("The teacher writes the report.");
    let wwcym = mind.what_would_change_your_mind(q).to_lowercase();
    assert!(
        wwcym.contains("i would change my mind if you told me"),
        "names a decider: {wwcym}"
    );

    // Asserting the proposition decides it Yes.
    let prop = semantics::understand(&engine, "the author reads the book");
    let mut yes_clone = discourse.clone();
    yes_clone.world.assert(&prop);
    let yes_ans = qa::answer(&engine, &yes_clone, q);
    assert_ne!(
        verdict_of(&yes_ans),
        V::Idk,
        "asserting the proposition decides it: {yes_ans}"
    );

    // Asserting its negation decides it No.
    let neg = polarity_flip(&prop).unwrap();
    let mut no_clone = discourse.clone();
    no_clone.world.assert(&neg);
    let no_ans = qa::answer(&engine, &no_clone, q);
    assert_ne!(
        verdict_of(&no_ans),
        V::Idk,
        "asserting the negation decides it: {no_ans}"
    );
}
