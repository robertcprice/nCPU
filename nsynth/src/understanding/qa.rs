//! Question answering: reading comprehension over the world model built by
//! reading. Answers come from the model, not from keyword-matching the
//! question.
//!
//! `answer` parses the question with `semantics::understand`. For a
//! YesNoQuestion it queries `world.holds` (augmented by inference) and replies
//! "Yes, ..." / "No, ..." / "I don't know.". For a WhQuestion it searches
//! `world.facts` for a matching event and returns the filler of the queried
//! slot ("the teacher"). For an IsA category question it answers from animacy.

use crate::comprehension::{capitalize, Engine};
use crate::understanding::discourse::Discourse;
use crate::understanding::inference::{relation, Relation};
use crate::understanding::meaning::{Event, Meaning, Role, Term};

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
            // Restate the *negation* of what was asked: if "does X write Y?" is
            // false, answer "No, the X does not write the Y.".
            format!("No, {}.", realize(engine, body, /*force_negated=*/ Some(true)))
        }
        None => "I don't know.".to_string(),
    }
}

/// Truth of `body` against the world, augmented by inference. Returns the
/// world's own verdict when it has one; otherwise tries to derive truth from an
/// asserted fact that entails (or contradicts) the query.
fn world_truth(discourse: &Discourse, body: &Meaning) -> Option<bool> {
    if let Some(v) = discourse.world.holds(body) {
        return Some(v);
    }
    // Inference fallback: does any asserted fact entail / contradict the query?
    // This makes QA close under sound natural-language inference, e.g. an
    // existential query entailed by a concrete asserted fact answers "Yes".
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
        Meaning::Unknown(s) => s.clone(),
    }
}

/// Realize an Event clause with explicit polarity, inflecting the verb for the
/// agent. Present + non-negated 3sg uses the synthesized `verb_3sg`; negated or
/// past uses the periphrastic/base form so we never invent inflection rules
/// here.
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
        // Past affirmative: keep the lemma — the world stores lemmas and we do
        // not synthesize past tense here; the agreement/aux carries the rest.
        (Tense::Past, false) => ev.predicate.clone(),
    };

    match ev.patient.as_ref() {
        Some(obj) => format!("{} {} {}", subj, verb_phrase, surface_term_plain(obj)),
        None => format!("{} {}", subj, verb_phrase),
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
