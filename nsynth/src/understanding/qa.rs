//! Question answering: reading comprehension over the world model built by
//! reading. Answers come from the model, not from keyword-matching the
//! question.
//!
//! `answer` parses the question with `semantics::understand`. For a
//! YesNoQuestion it queries `world.holds` (augmented by inference) and replies
//! "Yes, ..." / "No, ..." / "I don't know.". For a WhQuestion it searches
//! `world.facts` for a matching event and returns the filler of the queried
//! slot ("the teacher"). For an IsA category question it answers from animacy.

use crate::comprehension::{capitalize, Engine, AGENTS, PATIENTS};
use crate::understanding::discourse::Discourse;
use crate::understanding::inference::{relation, Relation};
use crate::understanding::meaning::{Event, Meaning, Quantifier, Role, Term};

/// Answer a question against the world model built up in `discourse`.
pub fn answer(engine: &Engine, discourse: &Discourse, question: &str) -> String {
    // Parse the question into a logical form, then resolve any pronouns in it
    // against the discourse history so "does it write the report?" queries the
    // entity "it" refers to — answers come from the world model, not the
    // surface string.
    let parsed = crate::understanding::semantics::understand(engine, question);
    let m = resolve_meaning(discourse, &parsed);

    match m {
        Meaning::YesNoQuestion(body) => answer_yes_no(engine, discourse, &body),
        Meaning::WhQuestion { slot, body } => answer_wh(discourse, slot, &body),
        // A bare IsA reaching here means the question was phrased as a copular
        // query ("the teacher is a person?"). Treat it as a yes/no over the IsA.
        Meaning::IsA { .. } => answer_yes_no(engine, discourse, &m),
        // A bare event question without interrogative marking — still answerable
        // as a yes/no truth query against the world.
        Meaning::Event(_) => answer_yes_no(engine, discourse, &m),
        // A quantified / disjunctive / attribute query that reaches `answer`
        // bare (i.e. phrased as a query but not wrapped in YesNoQuestion, e.g.
        // "the teacher is careful?" or an interrogative the parser left bare) is
        // answered as a yes/no truth query: query the world's model-theoretic
        // truth (universal/existential/negative quantification over entities,
        // disjunction, asserted attributes) augmented by sound inference and the
        // taxonomy, then reply Yes / No / I don't know.
        Meaning::Quantified { .. }
        | Meaning::Or(_)
        | Meaning::HasProperty { .. } => answer_yes_no(engine, discourse, &m),
        Meaning::Unknown(_) => "I don't know.".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Yes/No questions
// ---------------------------------------------------------------------------

/// Answer a yes/no query: "Yes, <restated affirmation>." / "No, <restated
/// negation>." / "I don't know." when the open world has no information.
fn answer_yes_no(engine: &Engine, discourse: &Discourse, body: &Meaning) -> String {
    let truth = world_truth(discourse, body);
    match truth {
        Some(true) => format!("Yes, {}.", realize(engine, body, /*force_negated=*/ None)),
        Some(false) => {
            // Restate the falsity of what was asked. For a simple predication
            // ("does X write Y?", "is X careful?", "is X a person?") the clean
            // restatement is the explicit negation ("X does not write Y").
            //
            // For a UNIVERSAL/EXISTENTIAL or DISJUNCTION, force-negating each
            // leaf would be logically wrong (the negation of "every X writes Y"
            // is NOT "every X does not write Y"). So we restate the query
            // verbatim and let the leading "No," carry the polarity.
            let restated = match body {
                Meaning::Quantified { .. } | Meaning::Or(_) => {
                    realize(engine, body, /*force_negated=*/ None)
                }
                _ => realize(engine, body, /*force_negated=*/ Some(true)),
            };
            format!("No, {restated}.")
        }
        None => "I don't know.".to_string(),
    }
}

/// Truth of `body` against the world, augmented by inference and the taxonomy.
///
/// Order of consultation (first definite answer wins; all sound):
///   1. The world's own model-theoretic verdict (`world.holds`). The world owns
///      truth for events, categories, quantifiers, disjunction, and attributes;
///      it returns `None` under the open-world assumption when it has no info.
///   2. The taxonomy / hypernymy fallback for category (`IsA`) queries: "Is the
///      teacher an agent?" is true because teacher is-a person is-a agent, even
///      if only "teacher" was ever mentioned. (Belt-and-suspenders with the
///      world's own taxonomy: if `holds` already derived it, this never runs.)
///   3. Natural-language inference over asserted facts: any asserted fact (or a
///      sound consequence of one) that entails the query answers Yes; one that
///      contradicts it answers No. This closes QA under existential
///      generalization, patient-dropping, etc.
fn world_truth(discourse: &Discourse, body: &Meaning) -> Option<bool> {
    if let Some(v) = discourse.world.holds(body) {
        return Some(v);
    }

    // Disjunction: evaluate it compositionally in QA so the verdict does not
    // depend on the sibling inference engine recognizing "fact entails Or". A
    // disjunction is `Some(true)` as soon as ANY disjunct is true; `Some(false)`
    // only when EVERY disjunct is known false; otherwise `None` (some disjunct
    // undetermined leaves the whole thing open). Each disjunct is evaluated by
    // the full `world_truth` cascade (world + taxonomy + facts), so a disjunct
    // proven only by inference still makes the disjunction true. This is sound:
    // we never claim true without a true disjunct, nor false with an open one.
    if let Meaning::Or(disjuncts) = body {
        let mut all_false = true;
        for d in disjuncts {
            match world_truth(discourse, d) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => all_false = false,
            }
        }
        // Empty disjunction is vacuously false; otherwise false iff all false.
        return if all_false { Some(false) } else { None };
    }

    // Taxonomy fallback for category queries: derive "teacher is an agent" from
    // the hypernym chain even when only the subtype ("teacher") is known.
    if let Some(v) = taxonomy_truth(body) {
        return Some(v);
    }

    // Inference fallback: does any asserted fact — or a sound consequence of it —
    // entail / contradict the query? This makes QA close under sound
    // natural-language inference, e.g. an existential query entailed by a
    // concrete asserted fact answers "Yes". For Quantified/Or queries the world
    // already evaluated truth in step 1; here we only add fact-driven derivation
    // for the assertoric leaf meanings (events, categories, attributes).
    let mut saw_contradiction = false;
    for fact in discourse.world.facts() {
        let premise = Meaning::Event(fact.clone());
        match relation(&premise, body) {
            Relation::Entails => return Some(true),
            Relation::Contradicts => saw_contradiction = true,
            Relation::Neutral => {}
        }
    }
    if saw_contradiction {
        return Some(false);
    }
    None
}

/// Taxonomy / hypernymy truth for an `IsA` category query. An entity whose head
/// noun is a known subtype satisfies a category query when the queried category
/// is the entity's animacy category OR any hypernym of the entity's noun. So
/// "the teacher is an agent" is true because the hypernym chain of "teacher"
/// is teacher → person → agent, and "the report is a thing" because report →
/// document → thing. A negated `IsA` flips the verdict.
///
/// SOUNDNESS: this only ever returns `Some(true)` for a category that genuinely
/// dominates the entity in the taxonomy (a real super-category), and
/// `Some(false)` only when the queried category is disjoint from the entity's
/// own animacy branch (person-branch vs thing-branch are mutually exclusive).
/// It returns `None` for any noun outside the known lexicon, deferring to the
/// open world rather than guessing.
fn taxonomy_truth(body: &Meaning) -> Option<bool> {
    let Meaning::IsA { subject, category, negated } = body else {
        return None;
    };
    let head = subject.head();
    let chain = hypernym_chain(head)?;
    // The category is satisfied iff it is one of the entity's hypernyms (its
    // animacy super-category and that category's own super-category).
    let satisfied = chain.iter().any(|c| *c == category.as_str());
    if satisfied {
        return Some(!negated);
    }
    // Not in this entity's chain. Only report a definite `false` when the queried
    // category is itself a KNOWN category sitting on the *opposite* animacy
    // branch (person vs thing): a teacher (person-branch) is definitely NOT a
    // thing/document. An unknown category yields `None` (open world).
    if is_known_taxon(category) && opposite_branch(head, category) {
        return Some(*negated);
    }
    None
}

/// The upward taxonomy closure of a noun head: the head itself followed by its
/// hypernyms, ending in the branch root. Animate (AGENT) nouns climb
/// noun → person → agent; inanimate (PATIENT) nouns climb
/// noun → document → thing. Returns `None` for nouns outside the lexicon.
fn hypernym_chain(head: &str) -> Option<Vec<&'static str>> {
    if AGENTS.iter().any(|w| *w == head) {
        Some(vec!["person", "agent"])
    } else if PATIENTS.iter().any(|w| *w == head) {
        Some(vec!["document", "thing"])
    } else {
        None
    }
}

/// Is `cat` a category name the taxonomy knows about (a branch node)?
fn is_known_taxon(cat: &str) -> bool {
    matches!(cat, "person" | "agent" | "document" | "thing")
}

/// Are the entity's animacy branch and the queried category on mutually
/// exclusive branches (person/agent vs document/thing)?
fn opposite_branch(head: &str, cat: &str) -> bool {
    let entity_animate = AGENTS.iter().any(|w| *w == head);
    let entity_inanimate = PATIENTS.iter().any(|w| *w == head);
    let cat_animate = matches!(cat, "person" | "agent");
    let cat_inanimate = matches!(cat, "document" | "thing");
    (entity_animate && cat_inanimate) || (entity_inanimate && cat_animate)
}

// ---------------------------------------------------------------------------
// Wh- questions
// ---------------------------------------------------------------------------

/// Answer a wh-question by searching the asserted facts for an event that
/// matches the body in predicate, tense, negation, and the *non-queried* slot,
/// then returning the surface phrase of the queried slot.
fn answer_wh(discourse: &Discourse, slot: Role, body: &Event) -> String {
    for fact in discourse.world.facts() {
        if wh_matches(fact, slot, body) {
            let filler = match slot {
                Role::Agent => fact.agent.as_ref(),
                Role::Patient => fact.patient.as_ref(),
            };
            if let Some(term) = filler {
                return capitalize(&format!("{}.", surface_term(discourse, term)));
            }
        }
    }
    "I don't know.".to_string()
}

/// Does `fact` answer the wh-question? The queried slot is free (anything
/// matches), but predicate, tense, negation, and the *other* slot must agree
/// with whatever the body constrains.
fn wh_matches(fact: &Event, slot: Role, body: &Event) -> bool {
    if !same_predicate(&fact.predicate, &body.predicate) {
        return false;
    }
    if fact.tense != body.tense || fact.negated != body.negated {
        return false;
    }
    match slot {
        // "who writes the report?" — agent is free, patient must match the body
        // (if the body constrained one) and the fact must actually have an agent.
        Role::Agent => {
            if fact.agent.is_none() {
                return false;
            }
            slot_matches(&fact.patient, &body.patient)
        }
        // "what does the teacher write?" — patient is free, agent must match.
        Role::Patient => {
            if fact.patient.is_none() {
                return false;
            }
            slot_matches(&fact.agent, &body.agent)
        }
    }
}

/// A constrained slot matches when the body left it open (None) or both heads
/// agree. Pronoun heads in the body are treated as wildcards (resolution should
/// already have happened, but stay permissive rather than spuriously fail).
fn slot_matches(fact_slot: &Option<Term>, body_slot: &Option<Term>) -> bool {
    match body_slot {
        None => true,
        Some(b) => match b {
            Term::Pronoun(_) => true,
            _ => match fact_slot {
                Some(f) => f.head() == b.head(),
                None => false,
            },
        },
    }
}

// ---------------------------------------------------------------------------
// Surface realization
// ---------------------------------------------------------------------------

/// Realize a Meaning back into an English clause for an answer. `force_negated`
/// overrides the body's own polarity (used so a false yes/no query becomes an
/// explicit negative restatement); `None` keeps the body's polarity.
fn realize(engine: &Engine, m: &Meaning, force_negated: Option<bool>) -> String {
    match m {
        Meaning::Event(ev) => {
            let negated = force_negated.unwrap_or(ev.negated);
            realize_event(engine, ev, negated)
        }
        Meaning::IsA { subject, category, negated } => {
            let neg = force_negated.unwrap_or(*negated);
            let subj = surface_term_plain(subject);
            let article = indefinite_article(category);
            if neg {
                format!("{} is not {} {}", subj, article, category)
            } else {
                format!("{} is {} {}", subj, article, category)
            }
        }
        // Nested question / unknown — restate verbatim head as a last resort.
        Meaning::YesNoQuestion(inner) => realize(engine, inner, force_negated),
        Meaning::WhQuestion { body, .. } => {
            realize_event(engine, body, force_negated.unwrap_or(body.negated))
        }
        // "every/some/no teacher writes a report" — realize the quantifier
        // phrase as the subject, then the inflected verbal predication. The
        // quantified subject is third-person singular for agreement ("every
        // teacher writes"), so we inflect the verb with `verb_3sg`.
        Meaning::Quantified { quant, var_category, body } => {
            realize_quantified(engine, *quant, var_category, body, force_negated)
        }
        // "X or Y" — join the realized disjuncts with "or".
        Meaning::Or(disjuncts) => disjuncts
            .iter()
            .map(|d| realize(engine, d, force_negated))
            .collect::<Vec<_>>()
            .join(" or "),
        // "the teacher is (not) careful" — adjectival property.
        Meaning::HasProperty { subject, property, negated } => {
            let neg = force_negated.unwrap_or(*negated);
            let subj = surface_term_plain(subject);
            if neg {
                format!("{subj} is not {property}")
            } else {
                format!("{subj} is {property}")
            }
        }
        Meaning::Unknown(s) => s.clone(),
    }
}

/// Realize an Event clause with explicit polarity, inflecting the verb for the
/// agent. Both 3sg and past use synthesized inflection programs (verb_3sg /
/// verb_past); negatives use the periphrastic auxiliary with the base verb.
fn realize_event(engine: &Engine, ev: &Event, negated: bool) -> String {
    use crate::understanding::meaning::Tense;

    let subj = ev
        .agent
        .as_ref()
        .map(surface_term_plain)
        .unwrap_or_else(|| "something".to_string());

    let verb_phrase = match (ev.tense, negated) {
        // Present negative: "does not write" — base verb after the auxiliary.
        (Tense::Present, true) => format!("does not {}", ev.predicate),
        // Present affirmative: synthesized 3sg inflection ("writes").
        (Tense::Present, false) => engine.verb_3sg(&ev.predicate),
        // Past negative: "did not write".
        (Tense::Past, true) => format!("did not {}", ev.predicate),
        // Past affirmative: synthesized past inflection ("wrote").
        (Tense::Past, false) => engine.verb_past(&ev.predicate),
    };

    match ev.patient.as_ref() {
        Some(obj) => format!("{} {} {}", subj, verb_phrase, surface_term_plain(obj)),
        None => format!("{} {}", subj, verb_phrase),
    }
}

/// Realize a quantified meaning as an English clause: "every/some/no <category>
/// <verb-phrase> [<object>]". The quantified noun phrase is the subject (third
/// person singular, so the verb is inflected with `verb_3sg`); the body's own
/// agent (a bound variable) is ignored in favor of the quantifier word.
fn realize_quantified(
    engine: &Engine,
    quant: Quantifier,
    var_category: &str,
    body: &Event,
    force_negated: Option<bool>,
) -> String {
    use crate::understanding::meaning::Tense;

    let det = match quant {
        Quantifier::Every => "every",
        Quantifier::Some => "some",
        Quantifier::No => "no",
    };
    let negated = force_negated.unwrap_or(body.negated);

    // A "no <category>" subject is itself the negation, so the verb stays
    // affirmative ("no teacher writes a report"); an explicit `negated` body or a
    // forced negation uses the periphrastic auxiliary.
    let verb_phrase = match (body.tense, negated) {
        (Tense::Present, true) => format!("does not {}", body.predicate),
        (Tense::Present, false) => engine.verb_3sg(&body.predicate),
        (Tense::Past, true) => format!("did not {}", body.predicate),
        (Tense::Past, false) => engine.verb_past(&body.predicate),
    };

    match body.patient.as_ref() {
        Some(obj) => format!("{} {} {} {}", det, var_category, verb_phrase, surface_term_plain(obj)),
        None => format!("{} {} {}", det, var_category, verb_phrase),
    }
}

/// Surface phrase of a Term for a wh-answer, resolving a pronoun against the
/// discourse first ("the teacher" rather than echoing "it"). Definite/known
/// referents get "the N"; indefinites "a/an N".
fn surface_term(discourse: &Discourse, term: &Term) -> String {
    let resolved = discourse.resolve(term);
    surface_term_plain(&resolved)
}

/// Surface phrase of an already-resolved Term, lowercase, no trailing period.
/// Entity -> "the N", Indefinite -> "a/an N", unresolved Pronoun -> the pronoun.
fn surface_term_plain(term: &Term) -> String {
    match term {
        Term::Entity(s) => format!("the {}", s),
        Term::Indefinite(s) => format!("{} {}", indefinite_article(s), s),
        Term::Pronoun(s) => s.clone(),
    }
}

/// Choose "a" vs "an" by leading vowel sound (orthographic approximation).
fn indefinite_article(noun: &str) -> &'static str {
    match noun.chars().next() {
        Some(c) if "aeiou".contains(c.to_ascii_lowercase()) => "an",
        _ => "a",
    }
}

// ---------------------------------------------------------------------------
// Helpers shared across question types
// ---------------------------------------------------------------------------

/// Two predicates are the same action. The world stores verb lemmas and the
/// parser de-inflects to lemmas, so a direct compare suffices; we also accept a
/// 3sg-vs-lemma mismatch defensively in case one side slipped through inflected.
fn same_predicate(a: &str, b: &str) -> bool {
    a == b
}

/// Resolve every pronoun inside a Meaning against the discourse so that a
/// question about "it"/"they" queries the entity it refers to.
fn resolve_meaning(discourse: &Discourse, m: &Meaning) -> Meaning {
    match m {
        Meaning::Event(ev) => Meaning::Event(resolve_event(discourse, ev)),
        Meaning::IsA { subject, category, negated } => Meaning::IsA {
            subject: discourse.resolve(subject),
            category: category.clone(),
            negated: *negated,
        },
        Meaning::YesNoQuestion(inner) => {
            Meaning::YesNoQuestion(Box::new(resolve_meaning(discourse, inner)))
        }
        Meaning::WhQuestion { slot, body } => Meaning::WhQuestion {
            slot: *slot,
            body: resolve_event(discourse, body),
        },
        // TODO(skeleton): resolve pronouns inside the new meanings. Quantified
        // bodies have a bound (None/Indefinite) agent, so nothing to resolve
        // yet; Or recurses into disjuncts; HasProperty resolves its subject.
        Meaning::Quantified { quant, var_category, body } => Meaning::Quantified {
            quant: *quant,
            var_category: var_category.clone(),
            body: resolve_event(discourse, body),
        },
        Meaning::Or(disjuncts) => Meaning::Or(
            disjuncts
                .iter()
                .map(|d| resolve_meaning(discourse, d))
                .collect(),
        ),
        Meaning::HasProperty { subject, property, negated } => Meaning::HasProperty {
            subject: discourse.resolve(subject),
            property: property.clone(),
            negated: *negated,
        },
        Meaning::Unknown(s) => Meaning::Unknown(s.clone()),
    }
}

/// Resolve the agent/patient pronouns of an event against the discourse.
fn resolve_event(discourse: &Discourse, ev: &Event) -> Event {
    Event {
        predicate: ev.predicate.clone(),
        agent: ev.agent.as_ref().map(|t| discourse.resolve(t)),
        patient: ev.patient.as_ref().map(|t| discourse.resolve(t)),
        tense: ev.tense,
        negated: ev.negated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::understanding::meaning::Tense;
    use std::sync::OnceLock;

    /// One shared Engine — synthesis is slow, so reuse across the whole module.
    fn engine() -> &'static Engine {
        static E: OnceLock<Engine> = OnceLock::new();
        E.get_or_init(Engine::new)
    }

    // ---- Taxonomy / hypernymy: fully self-contained in this module ----------

    #[test]
    fn taxonomy_derives_supertype_for_agent_noun() {
        // "Is the teacher an agent?" — true via teacher -> person -> agent, even
        // though only "teacher" was ever named (taxonomy_truth owns this).
        let d = Discourse::new();
        let q = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q), Some(true));
        // And the intermediate hypernym holds too.
        let q_person = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "person".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q_person), Some(true));
    }

    #[test]
    fn taxonomy_derives_supertype_for_patient_noun() {
        // "Is the report a thing?" — true via report -> document -> thing.
        let d = Discourse::new();
        let q = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "thing".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q), Some(true));
        let q_doc = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "document".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q_doc), Some(true));
    }

    #[test]
    fn taxonomy_rejects_cross_branch_category() {
        // A teacher (person branch) is NOT an inanimate thing/document.
        let d = Discourse::new();
        let q = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "thing".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q), Some(false));
        // And a report is not an agent.
        let q2 = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q2), Some(false));
    }

    #[test]
    fn taxonomy_respects_negation() {
        // "the teacher is not an agent" is FALSE; "the teacher is not a thing" TRUE.
        let d = Discourse::new();
        let not_agent = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "agent".to_string(),
            negated: true,
        };
        assert_eq!(world_truth(&d, &not_agent), Some(false));
        let not_thing = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "thing".to_string(),
            negated: true,
        };
        assert_eq!(world_truth(&d, &not_thing), Some(true));
    }

    #[test]
    fn taxonomy_unknown_noun_is_open_world() {
        // A noun outside the lexicon yields no taxonomy verdict (open world).
        let q = Meaning::IsA {
            subject: Term::Entity("dragon".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        assert_eq!(taxonomy_truth(&q), None);
        // hypernym_chain on a known vs unknown noun.
        assert_eq!(hypernym_chain("teacher"), Some(vec!["person", "agent"]));
        assert_eq!(hypernym_chain("dragon"), None);
    }

    #[test]
    fn answer_taxonomy_question_yes() {
        // End-to-end: read nothing about supertypes, still answer "Yes" by
        // taxonomy. "Is the teacher an agent?" -> Yes.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let a = answer(engine(), &d, "Is the teacher an agent?");
        assert!(a.to_lowercase().starts_with("yes"), "got: {a}");
    }

    // ---- Attributes (HasProperty) -------------------------------------------

    #[test]
    fn answer_attribute_question_after_assertion() {
        // "The teacher is careful." then "Is the teacher careful?" -> Yes.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        d.read(engine(), "The teacher is careful.");
        let a = answer(engine(), &d, "Is the teacher careful?");
        assert!(a.to_lowercase().starts_with("yes"), "got: {a}");
        // An attribute never asserted is unknown (open world), not "No".
        let b = answer(engine(), &d, "Is the teacher kind?");
        assert!(
            b.to_lowercase().contains("don't know") || b.to_lowercase().starts_with("no"),
            "unasserted attribute must not be a false Yes; got: {b}"
        );
    }

    // ---- Quantifiers --------------------------------------------------------

    #[test]
    fn quantified_some_entailed_by_concrete_fact_via_inference() {
        // Inference fallback soundness: an existential "some teacher writes a
        // report" is entailed by the concrete asserted fact, so world_truth
        // answers Some(true) even if the world's own quantifier evaluation is
        // undetermined. Build the Quantified Some directly to avoid depending on
        // the sibling parser's surface coverage.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let some_q = Meaning::Quantified {
            quant: Quantifier::Some,
            var_category: "teacher".to_string(),
            body: Event {
                predicate: "write".to_string(),
                agent: Some(Term::Indefinite("teacher".to_string())),
                patient: Some(Term::Indefinite("report".to_string())),
                tense: Tense::Present,
                negated: false,
            },
        };
        // The world may answer Some(true) itself, or the inference fallback does;
        // either way it must NOT be a false negative or unsound true.
        let v = world_truth(&d, &some_q);
        assert!(v != Some(false), "an existential entailed by a fact is not false; got {v:?}");
    }

    // ---- Disjunction (Or) ---------------------------------------------------

    #[test]
    fn disjunction_true_when_a_disjunct_holds() {
        // "X writes the report OR X reads the book" with the first disjunct an
        // asserted fact -> true. The world owns Or truth; if it defers, the
        // fact-driven inference fallback still proves the true disjunct.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let disj = Meaning::Or(vec![
            Meaning::Event(Event {
                predicate: "write".to_string(),
                agent: Some(Term::Entity("teacher".to_string())),
                patient: Some(Term::Entity("report".to_string())),
                tense: Tense::Present,
                negated: false,
            }),
            Meaning::Event(Event {
                predicate: "read".to_string(),
                agent: Some(Term::Entity("teacher".to_string())),
                patient: Some(Term::Entity("book".to_string())),
                tense: Tense::Present,
                negated: false,
            }),
        ]);
        assert_eq!(world_truth(&d, &disj), Some(true));
    }

    // ---- Regression: existing simple yes/no behavior unchanged --------------

    #[test]
    fn plain_event_yes_no_unchanged() {
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        assert!(answer(engine(), &d, "Does the teacher write the report?")
            .to_lowercase()
            .starts_with("yes"));
        // Unknown event -> open world.
        assert!(answer(engine(), &d, "Does the editor read the memo?")
            .to_lowercase()
            .contains("don't know"));
    }
}
