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

/// The set of FACTIVE attitude verbs: "know that P" entails P. Everything else
/// (believe/think/say/...) is non-factive — asserting the attitude says nothing
/// about the truth of its content. Kept as a tiny, explicit allow-list so the
/// factive/non-factive split is auditable and never accidentally widens.
const FACTIVE_VERBS: &[&str] = &["know"];

/// Is `verb` a factive attitude verb (its complement is entailed)?
fn is_factive(verb: &str) -> bool {
    FACTIVE_VERBS.iter().any(|v| *v == verb)
}

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

/// One asserted comparative ordering on a gradable scale, stored CANONICALLY as
/// `greater > lesser` so transitive reasoning is uniform regardless of the
/// surface polarity. "the report is longer than the book" and "the book is
/// shorter than the report" both store `Order { scale:"length", greater:"report",
/// lesser:"book" }`. A NEGATED comparison ("the report is NOT longer than the
/// book") is stored with `negated = true` and consulted by `holds` to report the
/// directed pair is explicitly denied.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Order {
    scale: String,
    greater: String,
    lesser: String,
    negated: bool,
}

/// One asserted propositional attitude: "the teacher knows/believes/... that
/// <content>". Stored verbatim (holder head, attitude verb, embedded meaning,
/// polarity). FACTIVITY is NOT baked into this record — it is decided by the
/// verb at assert/query time (a factive `know` additionally asserts its content
/// as a fact in its own right; this record only attests the attitude itself).
#[derive(Clone, Debug, PartialEq, Eq)]
struct AttitudeFact {
    holder: String,
    verb: String,
    content: Meaning,
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
    /// asserted comparative orderings ("the report is longer than the book").
    orderings: Vec<Order>,
    /// asserted propositional attitudes ("the teacher knows that ...").
    attitudes: Vec<AttitudeFact>,
}

impl World {
    pub fn new() -> Self {
        World {
            facts: Vec::new(),
            category: BTreeMap::new(),
            entities: BTreeSet::new(),
            attributes: Vec::new(),
            orderings: Vec::new(),
            attitudes: Vec::new(),
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
            // A comparison records a directed ordering on a gradable scale. We
            // store it canonically (greater > lesser) so transitive closure is
            // uniform; the surface polarity (`more`) only decides which argument
            // is the greater one.
            Meaning::Comparison {
                subject,
                scale,
                more,
                than,
                negated,
            } => self.assert_comparison(subject, scale, *more, than, *negated),
            // A propositional attitude records the attitude itself; a FACTIVE,
            // non-negated "know that P" ALSO asserts its content P (factivity:
            // knowing entails truth). Non-factive believe/think/say record ONLY
            // the attitude — never the content — so "the teacher believes that
            // the report is long" leaves the report's length open.
            Meaning::Attitude {
                holder,
                verb,
                content,
                negated,
            } => self.assert_attitude(holder, verb, content, *negated),
            // A cardinal "two teachers write a report" is a CLAIM about how many
            // entities satisfy the body — like an existential, it is checked, not
            // stored, so it no-ops on assert (materializing fresh entities would
            // be unsound: we never invent agents the world has not seen). Its
            // truth is evaluated against known members by `holds`.
            Meaning::Cardinal { .. } => {}
            // CountQuestion is a query (answered numerically by `count_satisfying`).
            Meaning::CountQuestion { .. } => {}
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
            // Comparative truth: does the queried ordering follow from the known
            // orderings on that scale (directly or by transitive closure)?
            Meaning::Comparison {
                subject,
                scale,
                more,
                than,
                negated,
            } => self.holds_comparison(subject, scale, *more, than, *negated),
            // Attitude truth: was this exact attitude (holder verb content
            // polarity) asserted? (A factive `know` ALSO asserted its content as
            // a fact, so the content itself is queryable separately.)
            Meaning::Attitude {
                holder,
                verb,
                content,
                negated,
            } => self.holds_attitude(holder, verb, content, *negated),
            // Cardinal at-least-N truth over the world's known members.
            Meaning::Cardinal {
                at_least,
                var_category,
                body,
            } => self.holds_cardinal(*at_least, var_category, body),
            // A counting question is a query whose answer is a NUMBER, not a
            // truth value — it is answered via `count_satisfying`, never here.
            Meaning::CountQuestion { .. } => None,
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

    /// How many KNOWN entities of `category` provably satisfy `body` (the body
    /// evaluated with the member bound to its agent slot, `holds_event` =
    /// `Some(true)`). This is the model-theoretic count behind a CountQuestion
    /// ("how many teachers write a report?") and the at-least check behind a
    /// Cardinal. It counts only entities whose body-truth is DETERMINED TRUE, so
    /// the number is a sound lower bound under the open-world assumption: more
    /// entities might satisfy the body once asserted, but never fewer than this.
    ///
    /// Termination: `category_members` is a finite set and each member is
    /// evaluated once against the finite fact base.
    pub fn count_satisfying(&self, category: &str, body: &Event) -> usize {
        self.category_members(category)
            .iter()
            .filter(|member| {
                let mut ev = body.clone();
                ev.agent = Some(Term::Entity((*member).clone()));
                self.holds_event(&ev) == Some(true)
            })
            .count()
    }

    /// Total number of KNOWN entities of `category`, whether or not they satisfy
    /// any particular body. Lets a CountQuestion answer distinguish "I know of N
    /// members and 0 satisfy" from "I know of no members at all".
    pub fn category_member_count(&self, category: &str) -> usize {
        self.category_members(category).len()
    }

    /// The embedded contents of every POSITIVELY-asserted attitude held by
    /// `holder` under attitude verb `verb`, in assertion order. This backs the
    /// wh-attitude question "What does the teacher know?" -> realize each known
    /// content. Negated attitudes ("does not know that P") are excluded — the
    /// holder does not hold that content. `verb` matching is exact on the lemma,
    /// so "know" returns only knowledge, not beliefs.
    pub fn known_attitude_contents(&self, holder: &str, verb: &str) -> Vec<Meaning> {
        self.attitudes
            .iter()
            .filter(|a| a.holder == holder && a.verb == verb && !a.negated)
            .map(|a| a.content.clone())
            .collect()
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

    /// Record a comparative ordering on a gradable scale. We canonicalize to
    /// `greater > lesser`: with `more` (longer/bigger/...) the subject is the
    /// greater one; with `!more` (shorter/smaller/...) the subject is the lesser
    /// one (equivalently `than > subject`). Both arguments are registered as
    /// entities. The stored `negated` flag carries an explicit denial ("X is NOT
    /// longer than Y") so `holds` can report `Some(false)` for the positive query
    /// without inventing the reverse ordering (X<Y is NOT implied by ¬(X>Y)).
    fn assert_comparison(
        &mut self,
        subject: &Term,
        scale: &str,
        more: bool,
        than: &Term,
        negated: bool,
    ) {
        self.register_term(subject);
        self.register_term(than);
        let (greater, lesser) = if more {
            (subject.head().to_string(), than.head().to_string())
        } else {
            (than.head().to_string(), subject.head().to_string())
        };
        // A self-comparison ("X is longer than X") is degenerate; never store an
        // ordering of an entity with itself (it would create a length-1 cycle and
        // is semantically false anyway).
        if greater == lesser {
            return;
        }
        let order = Order {
            scale: scale.to_string(),
            greater,
            lesser,
            negated,
        };
        if !self.orderings.iter().any(|o| *o == order) {
            self.orderings.push(order);
        }
    }

    /// Record a propositional attitude. Always stores the attitude fact itself.
    /// FACTIVITY: a non-negated, factive attitude ("the teacher KNOWS that P")
    /// additionally asserts its content P as a fact in its own right, so a later
    /// query of P answers Yes. Non-factive verbs (believe/think/say) and any
    /// negated attitude ("does NOT know that P") assert ONLY the attitude — never
    /// the content — because believing/doubting P says nothing about P's truth.
    fn assert_attitude(&mut self, holder: &Term, verb: &str, content: &Meaning, negated: bool) {
        self.register_term(holder);
        let fact = AttitudeFact {
            holder: holder.head().to_string(),
            verb: verb.to_string(),
            content: content.clone(),
            negated,
        };
        if !self.attitudes.iter().any(|a| *a == fact) {
            self.attitudes.push(fact);
        }
        // Factive entailment: knowing P (positively) makes P true in the world.
        if !negated && is_factive(verb) {
            self.assert(content);
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
    // truth-evaluation helpers for comparisons / attitudes / cardinals
    // ----------------------------------------------------------------------

    /// Comparative truth on a gradable scale. The query asks whether `greater >
    /// lesser` holds, where the (greater, lesser) pair is read off `more` exactly
    /// as in assertion. We answer:
    ///   - `Some(true)`  if the directed ordering is reachable (directly or by
    ///     TRANSITIVE closure) over the positively-asserted orderings on the scale;
    ///   - `Some(false)` if the EXACT directed ordering was explicitly denied
    ///     (a stored `negated` ordering), OR if the REVERSE ordering is known to
    ///     hold (an asymmetric scale: `Y > X` ⊢ `¬(X > Y)`);
    ///   - `None` otherwise (open world: no path proves it and nothing denies it).
    ///
    /// The query's own `negated` flag flips a definite verdict at the end.
    ///
    /// SOUNDNESS: we never infer symmetry (`X>Y` does not yield `Y>X`); the only
    /// negative we derive is from the proven reverse ordering (genuine asymmetry)
    /// or an explicit denial. Transitive closure is computed over a finite graph
    /// with a visited-set, so it always terminates even on a (malformed) cycle.
    fn holds_comparison(
        &self,
        subject: &Term,
        scale: &str,
        more: bool,
        than: &Term,
        negated: bool,
    ) -> Option<bool> {
        let (greater, lesser) = if more {
            (subject.head().to_string(), than.head().to_string())
        } else {
            (than.head().to_string(), subject.head().to_string())
        };

        // A self-comparison is never true (nothing exceeds itself).
        if greater == lesser {
            return Some(negated);
        }

        let base = self.ordering_truth(scale, &greater, &lesser);
        // Apply the query's outer negation to a determined verdict.
        base.map(|v| v != negated)
    }

    /// Three-valued truth of the bare directed ordering `greater > lesser` on
    /// `scale` (ignoring any outer query negation). Factored out so both
    /// `holds_comparison` and the inference-facing tests share one sound engine.
    fn ordering_truth(&self, scale: &str, greater: &str, lesser: &str) -> Option<bool> {
        // 1) Explicit denial of this exact directed pair -> false.
        if self
            .orderings
            .iter()
            .any(|o| o.negated && o.scale == scale && o.greater == greater && o.lesser == lesser)
        {
            return Some(false);
        }
        // 2) Provable by (transitive) closure of the positive orderings -> true.
        if self.ordering_reachable(scale, greater, lesser) {
            return Some(true);
        }
        // 3) The REVERSE ordering provably holds -> the forward one is false
        //    (a gradable scale is a strict order: asymmetric).
        if self.ordering_reachable(scale, lesser, greater) {
            return Some(false);
        }
        // 4) Nothing proves or denies it.
        None
    }

    /// Is `greater > lesser` reachable on `scale` from the POSITIVE asserted
    /// orderings via transitive closure? A depth-first reachability search over
    /// the directed graph whose edges are `g -> l` for each positive `Order`.
    ///
    /// Termination: a `visited` set bounds each node to one expansion, so even a
    /// (semantically impossible but defensively handled) cycle cannot loop.
    fn ordering_reachable(&self, scale: &str, greater: &str, lesser: &str) -> bool {
        let mut stack: Vec<&str> = vec![greater];
        let mut visited: BTreeSet<&str> = BTreeSet::new();
        visited.insert(greater);
        while let Some(node) = stack.pop() {
            for o in &self.orderings {
                if o.negated || o.scale != scale || o.greater != node {
                    continue;
                }
                if o.lesser == lesser {
                    return true;
                }
                if visited.insert(o.lesser.as_str()) {
                    stack.push(o.lesser.as_str());
                }
            }
        }
        false
    }

    /// Attitude truth: was an attitude with this (holder, verb, content)
    /// asserted, and with what polarity? Matching is by holder head, verb lemma,
    /// and STRUCTURAL equality of the embedded content meaning. The stored
    /// attitude asserts polarity `!fact.negated`; the query asks for polarity
    /// `!negated`. The most recent matching assertion decides. `None` if no
    /// attitude fact mentions this (holder, verb, content) triple — the open
    /// world says nothing about whether the holder holds that attitude.
    ///
    /// NOTE: this reports truth of the ATTITUDE ("does the teacher know that P?"),
    /// not of the content P. Factive entailment of P is handled at assert time
    /// (a factive `know` records P as its own fact), so a query of P proper is an
    /// ordinary content query, evaluated by `holds` on that content meaning.
    fn holds_attitude(
        &self,
        holder: &Term,
        verb: &str,
        content: &Meaning,
        negated: bool,
    ) -> Option<bool> {
        let head = holder.head();
        let mut verdict: Option<bool> = None;
        for a in &self.attitudes {
            if a.holder == head && a.verb == verb && &a.content == content {
                verdict = Some(a.negated == negated);
            }
        }
        verdict
    }

    /// Cardinal at-least-N truth over the world's known members of the category.
    ///
    /// Let `sat` = number of known members whose body is DETERMINED TRUE and
    /// `total` = number of known members of the category. Three-valued, sound,
    /// open-world:
    ///   - `Some(true)`  iff `sat >= at_least` — we already have witnesses enough.
    ///   - `Some(false)` iff even in the most generous case fewer than `at_least`
    ///     can satisfy: that is, the count of members NOT determined-false is
    ///     below `at_least` AND the category is a known, CLOSED-enough domain.
    ///     To stay sound under the open world we only report `false` when the
    ///     optimistic ceiling (`total` minus members that are determined-FALSE)
    ///     is below `at_least`; otherwise unknowns might still push us over N.
    ///   - `None` otherwise (undetermined: not enough proven yet, but the
    ///     optimistic ceiling still admits reaching N).
    ///
    /// `at_least == 0` is vacuously true. An unknown category yields `None`.
    fn holds_cardinal(&self, at_least: usize, var_category: &str, body: &Event) -> Option<bool> {
        if at_least == 0 {
            return Some(true);
        }
        if !self.category_is_known(var_category) {
            return None;
        }
        let members = self.category_members(var_category);

        let mut sat = 0usize; // determined-true members
        let mut det_false = 0usize; // determined-false members
        for member in &members {
            let mut ev = body.clone();
            ev.agent = Some(Term::Entity(member.clone()));
            match self.holds_event(&ev) {
                Some(true) => sat += 1,
                Some(false) => det_false += 1,
                None => {}
            }
        }

        // Enough witnesses already -> true.
        if sat >= at_least {
            return Some(true);
        }
        // Optimistic ceiling: every member that is NOT determined-false could
        // (under the open world) end up satisfying the body. If even that ceiling
        // falls short, the cardinal is determinately false.
        let ceiling = members.len().saturating_sub(det_false);
        if ceiling < at_least {
            return Some(false);
        }
        // Otherwise unknowns leave room to reach N — undetermined.
        None
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
            recipient: None,
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
            recipient: None,
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
            recipient: None,
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
            recipient: None,
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

    // -------------------------------------------------------------------
    // Comparatives (Comparison) + transitivity
    // -------------------------------------------------------------------

    /// Build "subject is <more?longer:shorter> (scale) than `than`".
    fn cmp(subject: &str, scale: &str, more: bool, than: &str, negated: bool) -> Meaning {
        Meaning::Comparison {
            subject: Term::Entity(subject.to_string()),
            scale: scale.to_string(),
            more,
            than: Term::Entity(than.to_string()),
            negated,
        }
    }

    #[test]
    fn comparison_direct_truth_and_asymmetry() {
        let mut w = World::new();
        // "the report is longer than the book".
        w.assert(&cmp("report", "length", true, "book", false));
        // The exact query holds.
        assert_eq!(w.holds(&cmp("report", "length", true, "book", false)), Some(true));
        // ASYMMETRY: the reverse "the book is longer than the report" is FALSE,
        // never silently true (we must not infer symmetry).
        assert_eq!(w.holds(&cmp("book", "length", true, "report", false)), Some(false));
        // "the book is SHORTER than the report" is the same ordering -> true.
        assert_eq!(w.holds(&cmp("book", "length", false, "report", false)), Some(true));
        // An unrelated scale is open-world unknown.
        assert_eq!(w.holds(&cmp("report", "weight", true, "book", false)), None);
    }

    #[test]
    fn comparison_transitive_closure() {
        let mut w = World::new();
        // A > B and B > C on length.
        w.assert(&cmp("report", "length", true, "essay", false));
        w.assert(&cmp("essay", "length", true, "book", false));
        // Transitivity: A > C.
        assert_eq!(w.holds(&cmp("report", "length", true, "book", false)), Some(true));
        // ... and the reverse C > A is false by asymmetry of the proven order.
        assert_eq!(w.holds(&cmp("book", "length", true, "report", false)), Some(false));
        // A pair with no path on the scale stays unknown.
        assert_eq!(w.holds(&cmp("book", "length", true, "essay", false)), Some(false)); // essay>book known, so book>essay false
        assert_eq!(w.holds(&cmp("memo", "length", true, "note", false)), None);
    }

    #[test]
    fn comparison_negation_and_explicit_denial() {
        let mut w = World::new();
        w.assert(&cmp("report", "length", true, "book", false));
        // Query "is the report NOT longer than the book?" -> No (it IS longer).
        assert_eq!(w.holds(&cmp("report", "length", true, "book", true)), Some(false));
        // An explicit denial makes the positive query false without inventing the
        // reverse ordering.
        let mut w2 = World::new();
        w2.assert(&cmp("memo", "length", true, "note", true)); // "memo is NOT longer than note"
        assert_eq!(w2.holds(&cmp("memo", "length", true, "note", false)), Some(false));
        // But the reverse is NOT thereby asserted true (¬(X>Y) does not give Y>X).
        assert_eq!(w2.holds(&cmp("note", "length", true, "memo", false)), None);
    }

    #[test]
    fn comparison_cycle_does_not_loop() {
        // Defensive: a (semantically impossible) cycle must still terminate.
        let mut w = World::new();
        w.assert(&cmp("report", "length", true, "book", false));
        w.assert(&cmp("book", "length", true, "report", false));
        // Both directions are "reachable", so each query is provable-true; the
        // search must not hang. We only assert it RETURNS (terminates).
        let _ = w.holds(&cmp("report", "length", true, "book", false));
        let _ = w.holds(&cmp("book", "length", true, "report", false));
    }

    // -------------------------------------------------------------------
    // Epistemic attitudes (Attitude) + factivity
    // -------------------------------------------------------------------

    fn long_report() -> Meaning {
        Meaning::HasProperty {
            subject: Term::Entity("report".to_string()),
            property: "long".to_string(),
            negated: false,
        }
    }

    fn attitude(verb: &str, negated: bool) -> Meaning {
        Meaning::Attitude {
            holder: Term::Entity("teacher".to_string()),
            verb: verb.to_string(),
            content: Box::new(long_report()),
            negated,
        }
    }

    #[test]
    fn factive_know_entails_content() {
        let mut w = World::new();
        // "the teacher knows that the report is long".
        w.assert(&attitude("know", false));
        // The attitude itself holds.
        assert_eq!(w.holds(&attitude("know", false)), Some(true));
        // FACTIVITY: the content is now true in the world.
        assert_eq!(w.holds(&long_report()), Some(true));
    }

    #[test]
    fn nonfactive_believe_does_not_entail_content() {
        let mut w = World::new();
        // "the teacher believes that the report is long".
        w.assert(&attitude("believe", false));
        // The attitude holds...
        assert_eq!(w.holds(&attitude("believe", false)), Some(true));
        // ... but the content is NOT asserted: believing P says nothing about P.
        assert_eq!(w.holds(&long_report()), None);
        // think/say are likewise non-factive.
        let mut w2 = World::new();
        w2.assert(&attitude("think", false));
        assert_eq!(w2.holds(&long_report()), None);
    }

    #[test]
    fn negated_know_does_not_entail_content() {
        // "the teacher does NOT know that the report is long" must not assert P.
        let mut w = World::new();
        w.assert(&attitude("know", true));
        assert_eq!(w.holds(&attitude("know", true)), Some(true)); // the negated attitude holds
        assert_eq!(w.holds(&attitude("know", false)), Some(false)); // positive attitude is false
        assert_eq!(w.holds(&long_report()), None); // content stays open
    }

    #[test]
    fn attitude_unknown_is_open_world() {
        let w = World::new();
        assert_eq!(w.holds(&attitude("know", false)), None);
    }

    #[test]
    fn known_attitude_contents_backs_wh_question() {
        // "What does the teacher know?" -> the contents of positive knowledge.
        let mut w = World::new();
        w.assert(&attitude("know", false)); // knows that the report is long
        w.assert(&attitude("believe", false)); // believes that the report is long
        // Only knowledge is returned for verb "know" (beliefs excluded).
        let known = w.known_attitude_contents("teacher", "know");
        assert_eq!(known.len(), 1);
        assert_eq!(known[0], long_report());
        // A negated "know" is excluded (the holder does NOT know that content).
        let mut w2 = World::new();
        w2.assert(&attitude("know", true));
        assert!(w2.known_attitude_contents("teacher", "know").is_empty());
    }

    // -------------------------------------------------------------------
    // Cardinality (Cardinal) + at-least monotonicity
    // -------------------------------------------------------------------

    /// "at least N <category> write a report", body agent bound by the cardinal.
    fn cardinal(at_least: usize, category: &str) -> Meaning {
        Meaning::Cardinal {
            at_least,
            var_category: category.to_string(),
            body: quant_body("report"),
        }
    }

    #[test]
    fn cardinal_at_least_counts_known_satisfiers() {
        let mut w = World::new();
        // Two distinct agents write a report.
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        w.assert(&Meaning::Event(write_event("editor", "report", false)));
        // count_satisfying over the "person" category sees both.
        assert_eq!(w.count_satisfying("person", &quant_body("report")), 2);
        // "at least 2 persons write a report" -> true.
        assert_eq!(w.holds(&cardinal(2, "person")), Some(true));
        // MONOTONICITY: at-least-1 also true (witnesses ≥ 1).
        assert_eq!(w.holds(&cardinal(1, "person")), Some(true));
        // at-least-0 vacuously true.
        assert_eq!(w.holds(&cardinal(0, "person")), Some(true));
    }

    #[test]
    fn cardinal_false_when_ceiling_below_n() {
        let mut w = World::new();
        // Exactly one known person writes; the other known person explicitly does
        // NOT (a determined-false member, so it can never count).
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        w.assert(&Meaning::Event(write_event("editor", "report", true))); // negated
        // sat = 1, det_false = 1, total = 2, ceiling = 1.
        // "at least 2 persons write a report" -> false (ceiling 1 < 2).
        assert_eq!(w.holds(&cardinal(2, "person")), Some(false));
        // "at least 1" is already witnessed true.
        assert_eq!(w.holds(&cardinal(1, "person")), Some(true));
    }

    #[test]
    fn cardinal_undetermined_when_unknowns_could_reach_n() {
        let mut w = World::new();
        // One person writes (witness); a second known person's body is UNKNOWN.
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        // Introduce "editor" without saying whether they write a report.
        w.assert(&Meaning::IsA {
            subject: Term::Entity("editor".to_string()),
            category: "person".to_string(),
            negated: false,
        });
        // sat = 1, det_false = 0, total = 2, ceiling = 2 >= 2 -> undetermined.
        assert_eq!(w.holds(&cardinal(2, "person")), None);
    }

    #[test]
    fn cardinal_unknown_category_is_none() {
        let w = World::new();
        assert_eq!(w.holds(&cardinal(2, "dragon")), None);
        // count over an unknown/empty category is 0.
        assert_eq!(w.count_satisfying("dragon", &quant_body("report")), 0);
    }

    #[test]
    fn count_satisfying_backs_count_question() {
        // The CountQuestion answer derives from count_satisfying.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        w.assert(&Meaning::Event(write_event("editor", "report", false)));
        w.assert(&Meaning::Event(write_event("author", "book", false))); // wrong patient
        // Only teacher and editor write a *report*.
        assert_eq!(w.count_satisfying("person", &quant_body("report")), 2);
        // A CountQuestion is never truth-evaluated.
        let cq = Meaning::CountQuestion {
            var_category: "person".to_string(),
            body: quant_body("report"),
        };
        assert_eq!(w.holds(&cq), None);
        // We do know of 3 persons total (teacher, editor, author).
        assert_eq!(w.category_member_count("person"), 3);
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
