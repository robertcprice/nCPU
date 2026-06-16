//! The world model: a model-theoretic store of asserted facts and categories
//! against which a `Meaning`'s truth is evaluated.
//!
//! `assert` adds declaratives as facts/categories (and auto-derives a category
//! from noun animacy: an Entity that is an animate noun is a "person",
//! inanimate is a "thing"). `holds` answers whether the world makes a Meaning
//! true: an Event holds iff a matching asserted fact exists (respecting
//! negation); an IsA holds iff the recorded category matches; `None` when the
//! world has no information (open-world assumption).

use std::collections::{BTreeMap, BTreeSet};

use crate::comprehension::{AGENTS, PATIENTS};
use crate::understanding::meaning::{Event, Meaning, Term};

/// A small model: asserted event facts, entity categories, and known entities.
pub struct World {
    /// asserted event facts. Each carries its own `negated` polarity, so a
    /// negated assertion ("the teacher does not write the report") is stored as
    /// a fact with `negated = true` and consulted by `holds` to report
    /// `Some(false)` for the corresponding positive query.
    facts: Vec<Event>,
    /// entity head -> category ("person" / "thing" / ...)
    category: BTreeMap<String, String>,
    /// every entity head mentioned/asserted so far
    entities: BTreeSet<String>,
}

impl World {
    pub fn new() -> Self {
        World {
            facts: Vec::new(),
            category: BTreeMap::new(),
            entities: BTreeSet::new(),
        }
    }

    /// Add a declarative's content to the world. Questions are ignored.
    pub fn assert(&mut self, m: &Meaning) {
        match m {
            Meaning::Event(ev) => self.assert_event(ev),
            Meaning::IsA {
                subject,
                category,
                negated,
            } => self.assert_isa(subject, category, *negated),
            // Questions and unparseable meanings carry no assertable content.
            Meaning::YesNoQuestion(_)
            | Meaning::WhQuestion { .. }
            | Meaning::Unknown(_) => {}
        }
    }

    /// Truth evaluation: does the world make this Meaning true?
    /// `Some(true)`/`Some(false)` when known, `None` under the open-world
    /// assumption (no information).
    pub fn holds(&self, m: &Meaning) -> Option<bool> {
        match m {
            Meaning::Event(query) => self.holds_event(query),
            Meaning::IsA {
                subject,
                category,
                negated,
            } => self.holds_isa(subject, category, *negated),
            // A question's truth is the truth of the meaning it wraps/queries.
            Meaning::YesNoQuestion(inner) => self.holds(inner),
            Meaning::WhQuestion { body, .. } => self.holds(&Meaning::Event(body.clone())),
            Meaning::Unknown(_) => None,
        }
    }

    /// All entity heads known to the world.
    pub fn entities(&self) -> Vec<String> {
        self.entities.iter().cloned().collect()
    }

    /// All asserted event facts.
    pub fn facts(&self) -> &[Event] {
        &self.facts
    }

    // ----------------------------------------------------------------------
    // assertion helpers
    // ----------------------------------------------------------------------

    /// Record an event predication. Registers its arguments as entities,
    /// derives their animacy categories, and stores the fact (deduplicated).
    fn assert_event(&mut self, ev: &Event) {
        if let Some(t) = &ev.agent {
            self.register_term(t);
        }
        if let Some(t) = &ev.patient {
            self.register_term(t);
        }
        // Deduplicate exact-equal facts so repeated reads don't bloat the model.
        if !self.facts.iter().any(|f| f == ev) {
            self.facts.push(ev.clone());
        }
    }

    /// Record a category statement ("X is a person"). A negated category
    /// statement removes/suppresses the asserted-positive category rather than
    /// installing it, so a later `holds` of the positive returns the recorded
    /// negative evidence.
    fn assert_isa(&mut self, subject: &Term, cat: &str, negated: bool) {
        let head = subject.head().to_string();
        self.entities.insert(head.clone());
        // Seed the animacy-derived category first so the entity always has a
        // baseline classification, then let an explicit positive assertion win.
        self.derive_category(subject);
        if !negated {
            self.category.insert(head, cat.to_string());
        }
        // For a negated category we keep the animacy-derived baseline and do not
        // overwrite it with the (false) asserted category; `holds_isa` reasons
        // about the mismatch.
    }

    /// Register a term as a known entity and assign its animacy-derived
    /// category if it does not already have one.
    fn register_term(&mut self, t: &Term) {
        // Pronouns are not concrete referents; only record resolved entities and
        // indefinites, which name an actual (if generic) head noun.
        match t {
            Term::Entity(_) | Term::Indefinite(_) => {
                let head = t.head().to_string();
                self.entities.insert(head);
                self.derive_category(t);
            }
            Term::Pronoun(_) => {}
        }
    }

    /// Auto-derive a category from noun animacy: an animate noun head is a
    /// "person", an inanimate noun head is a "thing". An explicit prior
    /// assertion is never overwritten.
    fn derive_category(&mut self, t: &Term) {
        let head = t.head().to_string();
        if self.category.contains_key(&head) {
            return;
        }
        if let Some(cat) = animacy_category(&head) {
            self.category.insert(head, cat.to_string());
        }
    }

    // ----------------------------------------------------------------------
    // truth-evaluation helpers
    // ----------------------------------------------------------------------

    /// Evaluate the truth of an event query against asserted facts.
    ///
    /// We compare on the *content* of the predication — predicate, agent,
    /// patient, tense — ignoring the `negated` flag during matching, then fold
    /// each matching fact's polarity against the query's polarity:
    ///   - a positive fact makes its positive query true and its negative query false;
    ///   - a negative fact makes its positive query false and its negative query true.
    /// If matching facts disagree we have a contradiction in the model; we report
    /// the most recent assertion (facts are appended in reading order).
    fn holds_event(&self, query: &Event) -> Option<bool> {
        let mut verdict: Option<bool> = None;
        for fact in &self.facts {
            if !same_event_content(fact, query) {
                continue;
            }
            // Does this fact make the query true?
            // query asks for polarity `!query.negated` (true if the query is a
            // plain positive event). The fact asserts polarity `!fact.negated`.
            let fact_polarity = !fact.negated;
            let query_polarity = !query.negated;
            verdict = Some(fact_polarity == query_polarity);
        }
        verdict
    }

    /// Evaluate the truth of a category statement against recorded categories
    /// and animacy.
    fn holds_isa(&self, subject: &Term, cat: &str, negated: bool) -> Option<bool> {
        let head = subject.head();
        // Prefer an explicitly recorded category; fall back to animacy so that
        // "the teacher is a person" is judged true even without a prior IsA.
        let known = self
            .category
            .get(head)
            .map(|s| s.as_str())
            .or_else(|| animacy_category(head));
        let matches = match known {
            Some(known_cat) => known_cat == cat,
            None => return None, // open-world: no information about this entity
        };
        // The query is satisfied when the recorded category matches the queried
        // category XOR the query's negation.
        Some(matches != negated)
    }
}

// --------------------------------------------------------------------------
// free helpers (lexical / animacy facts via the synthesized lexicon constants)
// --------------------------------------------------------------------------

/// Map a noun head to its animacy-derived category: animate -> "person",
/// inanimate -> "thing", unknown -> None. Uses the synthesized lexicon's
/// AGENTS (animate) / PATIENTS (inanimate) constant lists rather than a
/// hardcoded word list.
fn animacy_category(head: &str) -> Option<&'static str> {
    if AGENTS.iter().any(|w| *w == head) {
        Some("person")
    } else if PATIENTS.iter().any(|w| *w == head) {
        Some("thing")
    } else {
        None
    }
}

/// Two events share content (for fact-matching) when predicate, agent, patient,
/// and tense all match — independent of the `negated` polarity flag.
fn same_event_content(a: &Event, b: &Event) -> bool {
    a.predicate == b.predicate
        && a.tense == b.tense
        && terms_match(&a.agent, &b.agent)
        && terms_match(&a.patient, &b.patient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::understanding::meaning::Tense;

    /// Build a present, affirmative (or negated) write(agent, patient) event.
    fn write_event(agent: &str, patient: &str, negated: bool) -> Event {
        Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity(agent.to_string())),
            patient: Some(Term::Entity(patient.to_string())),
            tense: Tense::Present,
            negated,
        }
    }

    #[test]
    fn holds_returns_some_true_for_asserted_fact() {
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        let q = Meaning::Event(write_event("teacher", "report", false));
        assert_eq!(w.holds(&q), Some(true));
    }

    #[test]
    fn holds_returns_some_false_for_negated_fact() {
        // Asserting "teacher does NOT write report" must make the positive query false.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", true)));
        let q = Meaning::Event(write_event("teacher", "report", false));
        assert_eq!(w.holds(&q), Some(false));
    }

    #[test]
    fn holds_returns_none_for_unknown() {
        // The world knows nothing about this event -> open-world None.
        let w = World::new();
        let q = Meaning::Event(write_event("editor", "memo", false));
        assert_eq!(w.holds(&q), None);
    }

    #[test]
    fn holds_indefinite_patient_matches_definite_fact() {
        // "the teacher writes the report" should satisfy "the teacher writes a report".
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        let q = Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Indefinite("report".to_string())),
            tense: Tense::Present,
            negated: false,
        });
        assert_eq!(w.holds(&q), Some(true));
    }

    #[test]
    fn holds_isa_from_animacy() {
        // Without any explicit IsA assertion, animacy of "teacher" (animate)
        // makes "the teacher is a person" true and "is a thing" false.
        let w = World::new();
        let is_person = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "person".to_string(),
            negated: false,
        };
        let is_thing = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "thing".to_string(),
            negated: false,
        };
        assert_eq!(w.holds(&is_person), Some(true));
        assert_eq!(w.holds(&is_thing), Some(false));
    }

    #[test]
    fn assert_registers_entities_and_facts() {
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        assert_eq!(w.facts().len(), 1);
        let ents = w.entities();
        assert!(ents.contains(&"teacher".to_string()));
        assert!(ents.contains(&"report".to_string()));
    }
}

/// Compare two optional argument terms by their head noun, so that an
/// indefinite "a report" and a definite "the report" naming the same head
/// count as the same participant for truth evaluation. A missing argument on
/// either side matches only a missing argument on the other.
fn terms_match(a: &Option<Term>, b: &Option<Term>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => x.head() == y.head(),
        _ => false,
    }
}
