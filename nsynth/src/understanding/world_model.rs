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
use crate::understanding::meaning::{Event, Meaning, Quantifier, Term};

/// One asserted adjectival attribute of an entity: "the teacher is careful" is
/// stored as `Attribute { entity: "teacher", property: "careful", negated:false }`.
/// A negated assertion ("the teacher is not careful") is stored with
/// `negated = true` and consulted by `holds` to report `Some(false)` for the
/// positive query.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Attribute {
    entity: String,
    property: String,
    negated: bool,
}

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
    /// asserted adjectival attributes ("the teacher is careful").
    attributes: Vec<Attribute>,
}

impl World {
    pub fn new() -> Self {
        World {
            facts: Vec::new(),
            category: BTreeMap::new(),
            entities: BTreeSet::new(),
            attributes: Vec::new(),
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
            // "the teacher is careful" -> record an attribute fact (and register
            // the entity so it participates in categories / quantification).
            Meaning::HasProperty {
                subject,
                property,
                negated,
            } => self.assert_property(subject, property, *negated),
            // A universal "every teacher writes a report" asserts that every
            // KNOWN entity of the category satisfies the body. We materialize it
            // over the entities the world already knows about, so later queries
            // about those specific entities are answered. (Entities introduced
            // afterwards are not retroactively bound — that would require storing
            // the rule and is intentionally out of scope to stay sound and
            // terminating.) Existential/negative quantifiers are claims to be
            // *checked*, not facts to store, so they no-op on assert.
            Meaning::Quantified {
                quant,
                var_category,
                body,
            } => {
                if *quant == Quantifier::Every {
                    self.assert_universal(var_category, body);
                }
            }
            // Questions and unparseable meanings carry no assertable content.
            // Disjunctions are queries, not assertions, so they no-op too.
            Meaning::Or(_)
            | Meaning::YesNoQuestion(_)
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
            // Quantified truth over the world's known entities of the category.
            Meaning::Quantified {
                quant,
                var_category,
                body,
            } => self.holds_quantified(*quant, var_category, body),
            // Disjunction: true if any disjunct is true, false if all are false,
            // None while any disjunct is undetermined (and none is yet true).
            Meaning::Or(disjuncts) => self.holds_or(disjuncts),
            // Attribute truth from asserted attributes (respecting negation).
            Meaning::HasProperty {
                subject,
                property,
                negated,
            } => self.holds_property(subject, property, *negated),
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

    /// FORWARD-CHAINING CLOSURE: the extra `Meaning`s soundly derivable from the
    /// world's asserted facts + the taxonomy, that are NOT themselves directly
    /// asserted. Currently this materializes, for every known entity, the
    /// `IsA{ entity, hypernym }` facts up its taxonomy chain — e.g. an asserted
    /// "the teacher writes the report" makes `teacher` a known entity, from which
    /// we derive "teacher is a person" and "teacher is an agent".
    ///
    /// Termination: the entity set is finite and `hypernyms` returns a fixed,
    /// acyclic chain, so the loop runs in O(entities x chain-depth). Soundness:
    /// every emitted IsA is true by the taxonomy definition (teacher really is a
    /// person/agent); we never emit a category an entity is not in.
    pub fn closure(&self) -> Vec<Meaning> {
        let mut derived: Vec<Meaning> = Vec::new();
        for entity in &self.entities {
            // Build the full upward chain: the entity's own classification plus
            // the hypernyms of its noun. Each becomes a derived IsA.
            let mut cats: Vec<&str> = Vec::new();
            // Recorded / animacy category and its chain.
            if let Some(kc) = self
                .category
                .get(entity)
                .map(|s| s.as_str())
                .or_else(|| animacy_category(entity))
            {
                if kc != entity.as_str() {
                    cats.push(kc);
                }
                for h in hypernyms(kc) {
                    cats.push(h);
                }
            }
            // Hypernyms of the entity's noun directly (teacher -> person, agent).
            for h in hypernyms(entity) {
                cats.push(h);
            }
            // Deduplicate while preserving order, then emit one IsA per category.
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            for cat in cats {
                if seen.insert(cat) {
                    derived.push(Meaning::IsA {
                        subject: Term::Entity(entity.clone()),
                        category: cat.to_string(),
                        negated: false,
                    });
                }
            }
        }
        derived
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

    /// Record an adjectival attribute ("the teacher is careful"). Registers the
    /// subject as an entity, derives its animacy category, and stores the
    /// attribute fact (deduplicated). A repeated assertion with the opposite
    /// polarity is allowed to coexist; `holds_property` reports the most recent.
    fn assert_property(&mut self, subject: &Term, property: &str, negated: bool) {
        self.register_term(subject);
        let attr = Attribute {
            entity: subject.head().to_string(),
            property: property.to_string(),
            negated,
        };
        if !self.attributes.iter().any(|a| *a == attr) {
            self.attributes.push(attr);
        }
    }

    /// Materialize a universal ("every <category> <body>") over the entities the
    /// world ALREADY knows to belong to `var_category`: each such entity gets the
    /// body event asserted with itself as the (definite) agent. This is sound —
    /// "every teacher writes a report" does entail "the teacher writes a report"
    /// for each known teacher — and terminating (a finite entity set). We do not
    /// invent agents the world has never seen.
    fn assert_universal(&mut self, var_category: &str, body: &Event) {
        let members: Vec<String> = self
            .entities
            .iter()
            .filter(|e| self.entity_in_category(e, var_category))
            .cloned()
            .collect();
        for member in members {
            let mut ev = body.clone();
            ev.agent = Some(Term::Entity(member));
            self.assert_event(&ev);
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

    /// Evaluate the truth of a category statement against recorded categories,
    /// animacy, and the hypernym TAXONOMY.
    ///
    /// "The teacher is a teacher" is trivially true (the entity's own noun).
    /// "The teacher is a person" / "is an agent" hold because `person` and
    /// `agent` are hypernyms of `teacher`. "The teacher is a thing" is false —
    /// `thing` is not in `teacher`'s upward chain, and the entity has a known
    /// (animate) classification, so we can soundly report `Some(false)` rather
    /// than `None`.
    fn holds_isa(&self, subject: &Term, cat: &str, negated: bool) -> Option<bool> {
        let head = subject.head();
        match self.entity_in_category_known(head, cat) {
            Some(in_cat) => Some(in_cat != negated),
            None => None, // open-world: no information about this entity
        }
    }

    // ----------------------------------------------------------------------
    // truth-evaluation helpers for quantifiers / disjunction / attributes
    // ----------------------------------------------------------------------

    /// Truth of "every/some/no <var_category> <body>" over the world's known
    /// entities that belong to `var_category`.
    ///
    /// Soundness:
    ///   - `Every`: true iff EVERY known member of the category satisfies the
    ///     body, and the body's truth is determined for all of them. If any
    ///     member's body-truth is unknown, the universal is undetermined
    ///     (`None`) rather than spuriously true/false. With no known members but
    ///     a known category, the universal is vacuously true.
    ///   - `Some`: true iff at least one known member satisfies the body; false
    ///     iff every known member's body is determined-false; else `None`.
    ///   - `No`: the negation of `Some` (true iff none satisfy, false iff at
    ///     least one does, else `None`).
    fn holds_quantified(
        &self,
        quant: Quantifier,
        var_category: &str,
        body: &Event,
    ) -> Option<bool> {
        let members = self.category_members(var_category);

        // Vacuous case: no known members. A universal over an empty (but
        // recognized) domain is vacuously true; an existential is false; a
        // negative existential ("no X ...") is vacuously true. If the category
        // itself is unrecognized, we have no information at all.
        if members.is_empty() {
            if !self.category_is_known(var_category) {
                return None;
            }
            return match quant {
                Quantifier::Every => Some(true),
                Quantifier::Some => Some(false),
                Quantifier::No => Some(true),
            };
        }

        // Evaluate the body once for each member (member bound to the agent
        // slot) and tally the three-valued outcomes.
        let mut any_true = false;
        let mut any_false = false;
        let mut any_unknown = false;
        for member in &members {
            let mut ev = body.clone();
            ev.agent = Some(Term::Entity(member.clone()));
            match self.holds_event(&ev) {
                Some(true) => any_true = true,
                Some(false) => any_false = true,
                None => any_unknown = true,
            }
        }
        let all_true = !any_false && !any_unknown; // members is non-empty here

        match quant {
            Quantifier::Every => {
                if all_true {
                    Some(true)
                } else if any_false {
                    // A determined-false member is a counterexample, even if
                    // others are unknown.
                    Some(false)
                } else {
                    // No counterexample yet, but some members undetermined.
                    None
                }
            }
            Quantifier::Some => {
                if any_true {
                    Some(true)
                } else if any_unknown {
                    None
                } else {
                    // Every member determined, none true => existential false.
                    Some(false)
                }
            }
            Quantifier::No => {
                if any_true {
                    Some(false)
                } else if any_unknown {
                    None
                } else {
                    Some(true)
                }
            }
        }
    }

    /// Disjunction truth: `Some(true)` if any disjunct holds, `Some(false)` if
    /// every disjunct is determined-false, `None` if no disjunct is true but at
    /// least one is undetermined. An empty `Or` is vacuously false.
    fn holds_or(&self, disjuncts: &[Meaning]) -> Option<bool> {
        let mut any_unknown = false;
        for d in disjuncts {
            match self.holds(d) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => any_unknown = true,
            }
        }
        // No disjunct is true: undetermined if any was unknown, else all-false.
        if any_unknown {
            None
        } else {
            Some(false)
        }
    }

    /// Attribute truth from asserted attributes. Matching is by entity head and
    /// property; the most recent matching assertion's polarity decides. `None`
    /// if no attribute fact mentions this (entity, property).
    fn holds_property(&self, subject: &Term, property: &str, negated: bool) -> Option<bool> {
        let head = subject.head();
        let mut verdict: Option<bool> = None;
        for attr in &self.attributes {
            if attr.entity == head && attr.property == property {
                // The stored attribute asserts polarity `!attr.negated`; the
                // query asks for polarity `!negated`.
                verdict = Some(attr.negated == negated);
            }
        }
        verdict
    }

    // ----------------------------------------------------------------------
    // taxonomy / category membership
    // ----------------------------------------------------------------------

    /// All known entities that belong to `category` (by noun identity, hypernym
    /// chain, recorded category, or animacy category).
    fn category_members(&self, category: &str) -> Vec<String> {
        self.entities
            .iter()
            .filter(|e| self.entity_in_category(e, category))
            .cloned()
            .collect()
    }

    /// Does the world recognize `category` as a category at all? True if it is a
    /// known noun (AGENT/PATIENT), one of the animacy/taxonomy class names, or
    /// the head of some known entity. Prevents spuriously concluding vacuous
    /// truth over a category the world has never heard of.
    fn category_is_known(&self, category: &str) -> bool {
        is_known_noun(category)
            || is_taxonomy_class(category)
            || self.entities.iter().any(|e| e == category)
    }

    /// Does entity `head` belong to `category`, treating unknown membership as
    /// "no" (used where a boolean is needed, e.g. collecting members). For a
    /// three-valued answer use [`Self::entity_in_category_known`].
    fn entity_in_category(&self, head: &str, category: &str) -> bool {
        self.entity_in_category_known(head, category).unwrap_or(false)
    }

    /// Three-valued category membership: `Some(true)` if `head` provably belongs
    /// to `category`, `Some(false)` if it provably does NOT (the entity has a
    /// known classification that excludes the category), `None` if unknown.
    ///
    /// Membership holds when `category` is:
    ///   - the entity's own noun ("teacher" is a "teacher"),
    ///   - a hypernym of the entity's noun ("teacher" -> person, agent),
    ///   - the entity's recorded or animacy category, or a hypernym thereof
    ///     ("person" -> "person"; and "person" up-chains too if extended).
    fn entity_in_category_known(&self, head: &str, category: &str) -> Option<bool> {
        // 1) Reflexive: an entity is always in its own noun category.
        if head == category {
            return Some(true);
        }
        // 2) Hypernym chain of the entity's noun.
        if hypernyms(head).iter().any(|h| *h == category) {
            return Some(true);
        }
        // 3) Recorded / animacy category of the entity, plus its hypernym chain.
        let known_cat = self
            .category
            .get(head)
            .map(|s| s.as_str())
            .or_else(|| animacy_category(head));
        if let Some(kc) = known_cat {
            if kc == category || hypernyms(kc).iter().any(|h| *h == category) {
                return Some(true);
            }
            // The entity has a definite classification that does NOT include the
            // queried category. We can soundly report a NEGATIVE only when the
            // queried category is itself a recognized class that is mutually
            // exclusive with the entity's known one (the two animacy roots, plus
            // their disjoint hypernym chains). Otherwise stay open (`None`): the
            // category might be an orthogonal property we simply have not linked.
            if categories_are_disjoint(kc, category) {
                return Some(false);
            }
        }
        // 4) No information linking this entity to the category.
        None
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

/// Is `word` a known curriculum noun (animate AGENT or inanimate PATIENT)?
fn is_known_noun(word: &str) -> bool {
    AGENTS.iter().any(|w| *w == word) || PATIENTS.iter().any(|w| *w == word)
}

/// Is `word` one of the taxonomy/animacy class names (the non-leaf nodes of the
/// hierarchy)?
fn is_taxonomy_class(word: &str) -> bool {
    matches!(word, "person" | "agent" | "document" | "thing")
}

/// The upward hypernym chain of a noun or class name (most specific first,
/// excluding the word itself). The hierarchy is two disjoint chains:
///
///   <any AGENT noun>   -> person   -> agent
///   "person"           -> agent
///   <any PATIENT noun> -> document -> thing
///   "document"         -> thing
///
/// This is the SOUND, terminating taxonomy the contract requires: every animate
/// noun is-a person is-a agent; every inanimate noun is-a document is-a thing.
/// Class names also up-chain so "person" entails "agent". Returns a fixed,
/// acyclic list, so closure over it always terminates.
pub fn hypernyms(noun: &str) -> Vec<&'static str> {
    if AGENTS.iter().any(|w| *w == noun) {
        return vec!["person", "agent"];
    }
    if PATIENTS.iter().any(|w| *w == noun) {
        return vec!["document", "thing"];
    }
    match noun {
        "person" => vec!["agent"],
        "document" => vec!["thing"],
        // "agent" and "thing" are roots; unknown words have no known chain.
        _ => vec![],
    }
}

/// Are two category names provably disjoint — i.e. nothing can be in both?
/// We model the two animacy super-trees (person/agent vs. document/thing) as
/// mutually exclusive. A class in the animate tree is disjoint from any class in
/// the inanimate tree (and vice versa). Classes in the SAME tree, or any pairing
/// involving an unrecognized class, are NOT declared disjoint (stay open).
fn categories_are_disjoint(a: &str, b: &str) -> bool {
    match (taxonomy_root(a), taxonomy_root(b)) {
        (Some(ra), Some(rb)) => ra != rb,
        _ => false,
    }
}

/// The root of the taxonomy tree a class/noun belongs to: "agent" for the
/// animate tree, "thing" for the inanimate tree, `None` if unrecognized.
fn taxonomy_root(word: &str) -> Option<&'static str> {
    if AGENTS.iter().any(|w| *w == word) || matches!(word, "person" | "agent") {
        Some("agent")
    } else if PATIENTS.iter().any(|w| *w == word) || matches!(word, "document" | "thing") {
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

    // -------------------------------------------------------------------
    // Taxonomy / hypernymy
    // -------------------------------------------------------------------

    #[test]
    fn taxonomy_derives_agent_from_teacher() {
        // Only "teacher" was ever mentioned, yet "the teacher is an agent" must
        // be derivable from the hypernym chain teacher -> person -> agent.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        let is_agent = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        let is_person = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "person".to_string(),
            negated: false,
        };
        let is_teacher = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "teacher".to_string(),
            negated: false,
        };
        assert_eq!(w.holds(&is_agent), Some(true));
        assert_eq!(w.holds(&is_person), Some(true));
        assert_eq!(w.holds(&is_teacher), Some(true)); // reflexive
    }

    #[test]
    fn taxonomy_disjoint_trees_report_false_not_none() {
        // A teacher (animate tree) is provably NOT a thing/document (inanimate
        // tree): mutually exclusive roots let us answer Some(false), not None.
        let w = World::new();
        let teacher_is_thing = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "thing".to_string(),
            negated: false,
        };
        let report_is_agent = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        assert_eq!(w.holds(&teacher_is_thing), Some(false));
        assert_eq!(w.holds(&report_is_agent), Some(false));
        // A report IS a document and a thing.
        let report_is_document = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "document".to_string(),
            negated: false,
        };
        assert_eq!(w.holds(&report_is_document), Some(true));
    }

    #[test]
    fn taxonomy_unknown_entity_is_none() {
        // Open-world: nothing known about "widget" -> None, not a false positive.
        let w = World::new();
        let q = Meaning::IsA {
            subject: Term::Entity("widget".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        assert_eq!(w.holds(&q), None);
    }

    // -------------------------------------------------------------------
    // Quantifiers
    // -------------------------------------------------------------------

    /// Build a present, affirmative body event with the agent left open (bound
    /// by the quantifier) and the given patient.
    fn quant_body(patient: &str) -> Event {
        Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(Term::Indefinite(patient.to_string())),
            tense: Tense::Present,
            negated: false,
        }
    }

    #[test]
    fn universal_true_when_all_members_satisfy() {
        // Two teachers, both write a report. "Every teacher writes a report" = true.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        w.assert(&Meaning::Event(write_event("editor", "report", false)));
        // teacher and editor are both AGENTs -> category "person"/"agent".
        let q = Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: quant_body("report"),
        };
        assert_eq!(w.holds(&q), Some(true));
        // Existential also true.
        let some = Meaning::Quantified {
            quant: Quantifier::Some,
            var_category: "person".to_string(),
            body: quant_body("report"),
        };
        assert_eq!(w.holds(&some), Some(true));
    }

    #[test]
    fn universal_false_with_a_counterexample() {
        // Two teachers; one writes a report, one explicitly does NOT.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        w.assert(&Meaning::Event(write_event("editor", "report", true))); // negated
        let every = Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: quant_body("report"),
        };
        assert_eq!(w.holds(&every), Some(false));
        // "No person writes a report" is also false (teacher does).
        let no = Meaning::Quantified {
            quant: Quantifier::No,
            var_category: "person".to_string(),
            body: quant_body("report"),
        };
        assert_eq!(w.holds(&no), Some(false));
        // "Some person writes a report" is true.
        let some = Meaning::Quantified {
            quant: Quantifier::Some,
            var_category: "person".to_string(),
            body: quant_body("report"),
        };
        assert_eq!(w.holds(&some), Some(true));
    }

    #[test]
    fn quantifier_over_unknown_category_is_none() {
        // No entities, and "dragon" is not a recognized category.
        let w = World::new();
        let q = Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "dragon".to_string(),
            body: quant_body("report"),
        };
        assert_eq!(w.holds(&q), None);
    }

    #[test]
    fn universal_assert_materializes_per_entity() {
        // Assert two teachers exist, then assert the universal. The universal
        // should make each specific teacher-writes-report fact hold.
        let mut w = World::new();
        // Introduce entities via category statements (no write fact yet).
        w.assert(&Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "person".to_string(),
            negated: false,
        });
        w.assert(&Meaning::IsA {
            subject: Term::Entity("editor".to_string()),
            category: "person".to_string(),
            negated: false,
        });
        w.assert(&Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: quant_body("report"),
        });
        // Now "the teacher writes a report" is a derived fact.
        let q = Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Indefinite("report".to_string())),
            tense: Tense::Present,
            negated: false,
        });
        assert_eq!(w.holds(&q), Some(true));
    }

    // -------------------------------------------------------------------
    // Attributes (HasProperty)
    // -------------------------------------------------------------------

    #[test]
    fn attribute_assert_and_query() {
        let mut w = World::new();
        w.assert(&Meaning::HasProperty {
            subject: Term::Entity("teacher".to_string()),
            property: "careful".to_string(),
            negated: false,
        });
        let q = Meaning::HasProperty {
            subject: Term::Entity("teacher".to_string()),
            property: "careful".to_string(),
            negated: false,
        };
        assert_eq!(w.holds(&q), Some(true));
        // Querying a property never asserted -> None (open world).
        let q2 = Meaning::HasProperty {
            subject: Term::Entity("teacher".to_string()),
            property: "kind".to_string(),
            negated: false,
        };
        assert_eq!(w.holds(&q2), None);
    }

    #[test]
    fn attribute_negation_makes_positive_false() {
        let mut w = World::new();
        w.assert(&Meaning::HasProperty {
            subject: Term::Entity("teacher".to_string()),
            property: "careful".to_string(),
            negated: true,
        });
        let positive = Meaning::HasProperty {
            subject: Term::Entity("teacher".to_string()),
            property: "careful".to_string(),
            negated: false,
        };
        assert_eq!(w.holds(&positive), Some(false));
    }

    // -------------------------------------------------------------------
    // Disjunction (Or)
    // -------------------------------------------------------------------

    #[test]
    fn disjunction_true_when_any_disjunct_holds() {
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        // "teacher writes report OR author writes book": first holds (true).
        let or = Meaning::Or(vec![
            Meaning::Event(write_event("teacher", "report", false)),
            Meaning::Event(write_event("author", "book", false)),
        ]);
        assert_eq!(w.holds(&or), Some(true));
    }

    #[test]
    fn disjunction_false_when_all_determined_false() {
        let mut w = World::new();
        // Both disjuncts explicitly negated -> each Some(false) -> Or Some(false).
        w.assert(&Meaning::Event(write_event("teacher", "report", true)));
        w.assert(&Meaning::Event(write_event("editor", "memo", true)));
        let or = Meaning::Or(vec![
            Meaning::Event(write_event("teacher", "report", false)),
            Meaning::Event(write_event("editor", "memo", false)),
        ]);
        assert_eq!(w.holds(&or), Some(false));
    }

    #[test]
    fn disjunction_none_when_undetermined_and_no_truth() {
        let w = World::new();
        // Nothing asserted: one disjunct unknown -> Or None.
        let or = Meaning::Or(vec![
            Meaning::Event(write_event("teacher", "report", false)),
            Meaning::Event(write_event("author", "book", false)),
        ]);
        assert_eq!(w.holds(&or), None);
    }

    // -------------------------------------------------------------------
    // Forward-chaining closure
    // -------------------------------------------------------------------

    #[test]
    fn closure_derives_taxonomy_isa_facts() {
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        let derived = w.closure();
        // teacher -> person, agent ; report -> document, thing.
        let want = |head: &str, cat: &str| Meaning::IsA {
            subject: Term::Entity(head.to_string()),
            category: cat.to_string(),
            negated: false,
        };
        assert!(derived.contains(&want("teacher", "person")));
        assert!(derived.contains(&want("teacher", "agent")));
        assert!(derived.contains(&want("report", "document")));
        assert!(derived.contains(&want("report", "thing")));
        // Soundness: never derives a cross-tree (false) IsA.
        assert!(!derived.contains(&want("teacher", "thing")));
        assert!(!derived.contains(&want("report", "agent")));
        // Every derived fact actually holds in the world (closure is sound).
        for m in &derived {
            assert_eq!(w.holds(m), Some(true), "closure emitted a non-holding fact: {m:?}");
        }
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
