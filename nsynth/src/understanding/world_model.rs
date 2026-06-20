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
use crate::understanding::meaning::{
    Aspect, Event, Meaning, Modality, Quantifier, TemporalRel, Term,
};

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

/// One asserted modal fact: "the teacher can/must/might/should [not] write the
/// report". Stored verbatim (modality, the event body, polarity). MONOTONICITY
/// (`Must` entails `Can`) is NOT baked into the record — it is decided by
/// `holds` at query time, so the stored facts stay minimal and auditable. A
/// negated modal ("cannot write") is stored with `negated = true`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ModalFact {
    modality: Modality,
    body: Event,
    negated: bool,
}

/// One asserted temporal ordering of two events, canonicalized to `Before`:
/// "X writes before Y reads" and "Y reads after X writes" both store
/// `TemporalFact { earlier: write-event, later: read-event }`. Storing only the
/// `Before` direction makes transitive closure and asymmetry uniform regardless
/// of the surface relation word.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TemporalFact {
    earlier: Event,
    later: Event,
}

/// One asserted causal link: "the street floods BECAUSE the rain falls" stores
/// `CausalFact { cause: rain-falls, effect: street-floods }`. Asserting the link
/// presupposes both happened, so `assert` also records `cause` and `effect` as
/// facts in their own right. The link is directed: storing C->E never records
/// E->C (causation is not commutative).
#[derive(Clone, Debug, PartialEq, Eq)]
struct CausalFact {
    cause: Meaning,
    effect: Meaning,
}

/// One asserted conditional rule: "if the rain falls THEN the street floods"
/// stores `ConditionalFact { antecedent: rain-falls, consequent: street-floods }`.
/// STRICTLY WEAKER than a `CausalFact`: asserting the rule does NOT presuppose
/// either side happened (no `assert(antecedent)`/`assert(consequent)`), so unlike
/// `assert_causal` we only record the implication. Modus-ponens forward chaining
/// reads these rules; the link is directed (P->Q is not Q->P). `negated` records
/// an asserted denial of the conditional ("it is not the case that if P then Q").
#[derive(Clone, Debug, PartialEq, Eq)]
struct ConditionalFact {
    antecedent: Meaning,
    consequent: Meaning,
    negated: bool,
}

/// PROVENANCE of a stored fact: was it directly `Asserted` by an input sentence,
/// or `Derived` by an inference rule (e.g. modus ponens forward chaining)?
/// Default is `Asserted`, matching the pre-existing assert path so behavior is
/// unchanged. Tracked in a parallel ledger (`fact_provenance`) keyed by insertion
/// order, so adding it does not alter the `Event` type or any existing query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Provenance {
    /// stated directly by an input sentence (the only kind produced today)
    #[default]
    Asserted,
    /// produced by a sound inference rule over existing facts (reserved for
    /// modus-ponens forward chaining; no fact carries this tag yet)
    Derived,
}

/// A logged INCONSISTENCY: an incoming assertoric meaning whose claim the world
/// already entails the opposite of. We RECORD the conflict observationally and
/// — for Event facts — RESOLVE it by [`World::revise_event`] (see [`Revision`]),
/// so the fact store never simultaneously holds F and NOT F. `incoming` is the
/// meaning that conflicts; `note` is a short human-readable description of the
/// clash (and, when a resolution was applied, of how it was resolved).
#[derive(Clone, Debug)]
pub struct Contradiction {
    pub incoming: Meaning,
    pub note: String,
}

/// A logged BELIEF REVISION: when an incoming Event assertion conflicts (same
/// content, opposite polarity) with a belief the world already holds, we resolve
/// the clash to ONE coherent belief and record what changed and why.
///
/// `superseded` is the Event fact that LOST and was retracted from the live
/// store; `surviving` is the Event fact that REMAINS true afterward. `reason`
/// is the principled justification (provenance-weighting or most-recent-wins).
/// The two events always share content and differ ONLY in polarity, so the world
/// is coherent after revision: exactly one of {F, NOT F} survives.
///
/// SOUNDNESS: revision only ever resolves an inconsistency to one of the two
/// directly-involved beliefs — it never fabricates a third belief and never
/// changes any UNRELATED fact, so it can only move the world from incoherent
/// (both F and NOT F stored) to coherent; it can never turn a correct answer
/// into a wrong one.
#[derive(Clone, Debug)]
pub struct Revision {
    /// the belief that lost and was retracted from the live store
    pub superseded: Event,
    /// the belief that survives (the live, coherent belief afterward)
    pub surviving: Event,
    /// principled justification for the resolution
    pub reason: String,
}

/// A small model: asserted event facts, entity categories, and known entities.
#[derive(Clone)]
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
    /// asserted modal facts ("the teacher can write the report").
    modals: Vec<ModalFact>,
    /// asserted temporal orderings of events ("X writes before Y reads"),
    /// canonicalized to the `Before` direction.
    temporals: Vec<TemporalFact>,
    /// asserted causal links ("the street floods because the rain falls").
    causals: Vec<CausalFact>,
    /// asserted conditional rules ("if the rain falls then the street floods").
    /// Stored WITHOUT presupposing either side; read by modus-ponens chaining.
    conditionals: Vec<ConditionalFact>,
    /// PARALLEL provenance ledger for `facts`: `fact_provenance[i]` records how
    /// `facts[i]` entered the world (`Asserted` by default). Additive and
    /// append-only — kept in lockstep with `facts` in `assert_event`, never
    /// consulted by existing queries, so behavior is unchanged.
    fact_provenance: Vec<Provenance>,
    /// observed inconsistencies: an incoming assertion the world already entails
    /// the opposite of. FLAGGED, and (for Event facts) RESOLVED by `revise_event`.
    contradictions: Vec<Contradiction>,
    /// belief revisions applied while asserting Event facts: each records the
    /// superseded belief, the surviving belief, and the principled reason. Kept
    /// in detection order; surfaced via [`World::revisions`].
    revisions: Vec<Revision>,
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
            modals: Vec::new(),
            temporals: Vec::new(),
            causals: Vec::new(),
            conditionals: Vec::new(),
            fact_provenance: Vec::new(),
            contradictions: Vec::new(),
            revisions: Vec::new(),
        }
    }

    /// Add a declarative's content to the world. Questions are ignored.
    pub fn assert(&mut self, m: &Meaning) {
        // CONTRADICTION DETECTION (observational, non-destructive): before
        // recording an assertoric fact, check whether the world ALREADY entails
        // the opposite of what this assertion claims. If so, log the conflict.
        // We still record the fact below exactly as before — the log is purely
        // observational and belief revision is intentionally out of scope.
        self.detect_contradiction(m);
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
            // A modal ("the teacher can/must/might/should write the report")
            // records the modal force over the (entity-resolved) event. We do NOT
            // assert the bare event as a fact: "can write" / "might write" do not
            // mean the writing actually happened (possibility is not actuality).
            Meaning::Modal {
                modality,
                body,
                negated,
            } => self.assert_modal(*modality, body, *negated),
            // A temporal ordering ("X writes before/after Y reads") records the
            // ordered event pair (canonicalized to Before) AND asserts both events
            // as facts — saying "X happens before Y" presupposes both happen.
            Meaning::Temporal { rel, first, second } => self.assert_temporal(*rel, first, second),
            // A causal link ("E because C") records the directed cause->effect link
            // and ALSO asserts both the cause and the effect: asserting the link
            // presupposes both happened.
            Meaning::Causal { cause, effect } => self.assert_causal(cause, effect),
            // A conditional rule ("if P then Q") is recorded WITHOUT asserting
            // either side — unlike Causal, it presupposes nothing. We only store
            // the implication so modus-ponens chaining can fire later. (Forward
            // chaining over already-known antecedents will hook into `revise` /
            // the assert path; for now this is a pure, sound store.)
            Meaning::Conditional {
                antecedent,
                consequent,
                negated,
            } => self.assert_conditional(antecedent, consequent, *negated),
            // Outer negation: assert the three-valued negation of the inner
            // meaning. "X does not write the report" wrapped as Not(Event) records
            // the event with flipped polarity; a Not over a quantifier records
            // nothing storable (its truth is checked, not stored) — see
            // `assert_not` for the case analysis.
            Meaning::Not(inner) => self.assert_not(inner),
            // A degree question ("how long is the report?") is a query, not an
            // assertion — it carries no storable content.
            Meaning::DegreeQuestion { .. } => {}
        }
        // FORWARD-CHAINING RE-FIRE: a newly-asserted fact may make a previously
        // stored conditional's antecedent true (modus ponens) or its consequent
        // false (modus tollens). Re-run the bounded fixpoint so rules fire
        // regardless of whether the rule or the triggering fact was read first.
        // A `Conditional` already fired chaining inside `assert_conditional`, so
        // skip the redundant pass for it; everything else re-fires. Cheap no-op
        // when no rules are stored.
        if !self.conditionals.is_empty() && !matches!(m, Meaning::Conditional { .. }) {
            self.derive_modus_ponens();
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
            // Modal truth with monotonicity (`Must` |- `Can`, possibility does
            // NOT entail actuality).
            Meaning::Modal {
                modality,
                body,
                negated,
            } => self.holds_modal(*modality, body, *negated),
            // Temporal-order truth by transitive closure of asserted `Before`
            // pairs (asymmetric: a known reverse order makes the query false).
            Meaning::Temporal { rel, first, second } => self.holds_temporal(*rel, first, second),
            // Causal-link truth: is the directed cause->effect link known?
            Meaning::Causal { cause, effect } => self.holds_causal(cause, effect),
            // Conditional truth (sound, three-valued material implication):
            //   - vacuously TRUE when the antecedent is `Some(false)`;
            //   - TRUE when the consequent is `Some(true)`;
            //   - FALSE when antecedent `Some(true)` AND consequent `Some(false)`;
            //   - otherwise `None` (open world). `negated` flips the verdict.
            Meaning::Conditional {
                antecedent,
                consequent,
                negated,
            } => self.holds_conditional(antecedent, consequent, *negated),
            // Three-valued negation of the inner meaning's truth.
            Meaning::Not(inner) => negate3(self.holds(inner)),
            // A degree question is a query whose answer is a comparison phrase, not
            // a truth value — answered via `degree_position`, never here.
            Meaning::DegreeQuestion { .. } => None,
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

    /// All inconsistencies the world has flagged so far, in detection order. Each
    /// is an incoming assertion the world ALREADY entailed the opposite of at the
    /// moment it was asserted. For Event facts the clash is also RESOLVED (see
    /// [`World::revisions`]); the observational record of the clash is kept here.
    pub fn contradictions(&self) -> &[Contradiction] {
        &self.contradictions
    }

    /// All belief revisions applied so far, in the order they were resolved. Each
    /// records the SUPERSEDED belief, the SURVIVING belief, and the principled
    /// reason it was resolved that way (provenance-weighting, or most-recent-wins
    /// when both were directly asserted). After every revision the world holds
    /// exactly one of {F, NOT F} as true — never both — so this is the auditable
    /// history of how the world stayed coherent.
    pub fn revisions(&self) -> &[Revision] {
        &self.revisions
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

    /// The positively-asserted comparison orderings, each reconstructed as a
    /// `Meaning::Comparison{subject: Entity(greater), more: true, than: Entity(lesser)}`.
    /// This exposes the world's comparison fact-base so the proof layer can
    /// reconstruct a TRANSITIVE derivation ("report > book", "book > letter" =>
    /// "report > letter") with the intermediate named — the world model owns the
    /// transitive-closure VERDICT but no derivation, so `inference::prove` rebuilds
    /// the chain over these edges. Negated orderings are excluded (only positive
    /// "greater than" edges chain transitively).
    pub fn comparison_facts(&self) -> Vec<Meaning> {
        self.orderings
            .iter()
            .filter(|o| !o.negated)
            .map(|o| Meaning::Comparison {
                subject: Term::Entity(o.greater.clone()),
                scale: o.scale.clone(),
                more: true,
                than: Term::Entity(o.lesser.clone()),
                negated: false,
            })
            .collect()
    }

    /// The world's DIRECTLY-ASSERTED event facts (provenance `Asserted`), the
    /// genuine PREMISES a derivation reasons from — DERIVED facts are excluded
    /// because they are conclusions, not premises. Used to reconstruct a
    /// modus-ponens / modus-tollens proof: feeding `prove` the materialized derived
    /// consequent would let it short-circuit to that fact as an `"asserted"` leaf
    /// instead of rebuilding the inference chain. Returns each as an `Event`.
    pub fn asserted_event_facts(&self) -> Vec<Event> {
        self.facts
            .iter()
            .enumerate()
            .filter(|(i, _)| self.fact_provenance.get(*i).copied() == Some(Provenance::Asserted))
            .map(|(_, f)| f.clone())
            .collect()
    }

    /// The world's asserted CONDITIONAL rules as `Meaning::Conditional`s, so the
    /// proof layer can build a modus-ponens / modus-tollens DERIVATION ("the guard
    /// wakes because the alarm rings and if the alarm rings then the guard wakes").
    /// The world model owns the forward-chaining VERDICT (it materializes the
    /// derived consequent as a fact), but exposes no proof; `inference::prove`
    /// rebuilds the certificate over these rule edges + the asserted event facts.
    /// Both polarities are emitted (the rule's own `negated` flag is preserved) so
    /// `prove` can apply the correct inference and never fire a denied rule.
    pub fn conditional_facts(&self) -> Vec<Meaning> {
        self.conditionals
            .iter()
            .map(|c| Meaning::Conditional {
                antecedent: Box::new(c.antecedent.clone()),
                consequent: Box::new(c.consequent.clone()),
                negated: c.negated,
            })
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

    /// CONTRADICTION DETECTION. If `m` is an assertoric meaning (Event / IsA /
    /// HasProperty / Comparison) and the world ALREADY entails the OPPOSITE of
    /// what `m` claims, log a `Contradiction`. We compare on the POSITIVE form:
    /// `m` claims that positive form is true (when `m` is affirmative) or false
    /// (when `m` is negated); a contradiction is a definite prior verdict equal
    /// to the opposite of that claim.
    ///
    /// SOUNDNESS / SCOPE: only DEFINITE prior verdicts (`Some(true)`/`Some(false)`)
    /// can clash — an open-world `None` is never a contradiction. We never
    /// retract: the conflicting fact is still recorded by `assert`. Non-assertoric
    /// meanings (quantifiers, modals, attitudes, questions, ...) are skipped — the
    /// contract limits flagging to the four ground assertoric kinds.
    fn detect_contradiction(&mut self, m: &Meaning) {
        let Some((positive_form, claims_true)) = assertoric_positive_form(m) else {
            return;
        };
        // What does the world ALREADY entail about the positive proposition?
        let prior = self.holds(&positive_form);
        // A contradiction is a DEFINITE prior verdict opposite to the claim.
        if prior == Some(!claims_true) {
            let note = format!(
                "incoming assertion claims the proposition is {}, but the world already entails it is {}",
                if claims_true { "true" } else { "false" },
                if claims_true { "false" } else { "true" },
            );
            self.contradictions.push(Contradiction {
                incoming: m.clone(),
                note,
            });
        }
    }

    /// Record an event predication. Registers its arguments as entities,
    /// derives their animacy categories, and stores the fact (deduplicated).
    ///
    /// RELATIVE CLAUSES: any argument that is a `Restricted{head, clause}` term is
    /// resolved to the concrete entity of category `head` that satisfies `clause`
    /// (when one is known) BEFORE the fact is stored, so the asserted fact carries
    /// a plain entity and matches ordinary queries. The clause's own facts are
    /// registered too (the relative clause presupposes its content), which is how
    /// "the teacher who writes the report reads the book" makes both the read and
    /// (already-known) write available.
    fn assert_event(&mut self, ev: &Event) {
        self.assert_event_with_provenance(ev, Provenance::Asserted);
    }

    /// Record an event predication with an explicit PROVENANCE, applying
    /// ACTIONABLE BELIEF REVISION when the incoming fact conflicts (same content,
    /// opposite polarity) with one the world already holds.
    ///
    /// Resolution policy (sound, provenance-weighted, most-recent-wins fallback):
    ///   * A directly ASSERTED fact OUTRANKS a DERIVED one. If the incoming fact
    ///     is `Asserted` and the conflicting stored belief is `Derived`, the
    ///     incoming wins and the derived belief is RETRACTED.
    ///   * If the incoming fact is `Derived` and the conflicting stored belief is
    ///     `Asserted`, the asserted belief WINS — the incoming derived fact is
    ///     NOT installed (a derivation never overturns a direct assertion).
    ///   * If BOTH are directly asserted (or both derived), the MOST RECENT wins
    ///     (the incoming) and the prior belief is RETRACTED and recorded as
    ///     superseded.
    /// In every branch the world ends holding exactly one of {F, NOT F}, never
    /// both, and a [`Revision`] is logged. Non-conflicting facts are appended
    /// exactly as before (deduplicating exact repeats).
    fn assert_event_with_provenance(&mut self, ev: &Event, prov: Provenance) {
        // Register and resolve the arguments. We register the relative clause's
        // participants so the head entity is known, then resolve the Restricted
        // term to its referent for storage.
        let mut stored = ev.clone();
        if let Some(t) = &ev.agent {
            self.register_term(t);
            stored.agent = Some(self.resolve_restricted(t));
        }
        if let Some(t) = &ev.patient {
            self.register_term(t);
            stored.patient = Some(self.resolve_restricted(t));
        }
        if let Some(t) = &ev.recipient {
            self.register_term(t);
            stored.recipient = Some(self.resolve_restricted(t));
        }
        // Deduplicate exact-equal facts so repeated reads don't bloat the model.
        if self.facts.iter().any(|f| f == &stored) {
            return;
        }
        // BELIEF REVISION: find a stored fact of the SAME content but OPPOSITE
        // polarity. Such a fact is a genuine F vs NOT-F clash that must be
        // resolved so the world never holds both at once.
        if let Some(idx) = self.conflicting_fact_index(&stored) {
            self.revise_event(idx, stored, prov);
            return;
        }
        // No conflict: append normally.
        self.push_fact(stored, prov);
    }

    /// Index of a stored fact that shares CONTENT + aspect with `incoming` but has
    /// the OPPOSITE polarity (a direct F-vs-NOT-F clash) — if any. We scan in
    /// reverse so the most-recently asserted conflicting belief is the one we
    /// resolve against, matching the "most-recent-wins" reading the rest of the
    /// model already uses. Same-polarity facts are NOT conflicts (and exact
    /// duplicates were already filtered by the caller).
    fn conflicting_fact_index(&self, incoming: &Event) -> Option<usize> {
        self.facts.iter().enumerate().rev().find_map(|(i, f)| {
            let same = same_event_content(f, incoming)
                && f.aspect == incoming.aspect
                && f.negated != incoming.negated;
            if same {
                Some(i)
            } else {
                None
            }
        })
    }

    /// Resolve a detected conflict between the `incoming` fact (provenance
    /// `incoming_prov`) and the stored fact at `idx` (opposite polarity, same
    /// content). Applies the provenance-weighted, most-recent-wins policy and logs
    /// a [`Revision`]. INVARIANT on return: the world holds exactly one of the two
    /// beliefs — never both.
    fn revise_event(&mut self, idx: usize, incoming: Event, incoming_prov: Provenance) {
        let existing = self.facts[idx].clone();
        let existing_prov = self.fact_provenance[idx];

        // Decide who wins. A directly ASSERTED fact outranks a DERIVED one;
        // otherwise the most recent (the incoming) wins.
        let incoming_wins = match (incoming_prov, existing_prov) {
            // Direct assertion beats a derived belief.
            (Provenance::Asserted, Provenance::Derived) => true,
            // A derivation never overturns a direct assertion.
            (Provenance::Derived, Provenance::Asserted) => false,
            // Same provenance class on both sides: most-recent (incoming) wins.
            _ => true,
        };

        let reason = revision_reason(incoming_prov, existing_prov, incoming_wins);

        if incoming_wins {
            // Retract the stored (now superseded) belief and install the incoming.
            self.facts.remove(idx);
            self.fact_provenance.remove(idx);
            self.revisions.push(Revision {
                superseded: existing,
                surviving: incoming.clone(),
                reason,
            });
            self.push_fact(incoming, incoming_prov);
        } else {
            // The stored belief wins; the incoming (derived) fact is NOT installed.
            self.revisions.push(Revision {
                superseded: incoming,
                surviving: existing,
                reason,
            });
        }
    }

    /// Append an event fact AND its provenance tag, keeping the parallel
    /// `fact_provenance` ledger in lockstep with `facts`. The only caller today
    /// passes `Provenance::Asserted` (preserving existing behavior); a future
    /// modus-ponens forward chainer will pass `Provenance::Derived`.
    fn push_fact(&mut self, ev: Event, prov: Provenance) {
        self.facts.push(ev);
        self.fact_provenance.push(prov);
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
            // A restricted term "the <head> who <clause>" names a head-category
            // entity constrained by a relative clause. Register the head so it
            // participates like a definite, AND register the clause's own
            // participants (its patient/recipient) so they become known entities
            // too — the relative clause is a presupposed predication, not a fresh
            // assertion, so we register but do NOT add it as a standalone fact here
            // (it is asserted in its own right elsewhere in the discourse). Note:
            // we do not recurse into the clause's agent, which is the head itself.
            Term::Restricted { head, clause } => {
                self.entities.insert(head.clone());
                self.derive_category(t);
                if let Some(p) = &clause.patient {
                    self.register_term(p);
                }
                if let Some(r) = &clause.recipient {
                    self.register_term(r);
                }
            }
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
    ///
    /// ASPECT ENTAILMENT: a stored Progressive ("is writing") or Perfect ("has
    /// written") fact entails the SIMPLE event holds, so it answers a Simple query.
    /// The converse is NOT sound (a habitual "writes" need not be ongoing right
    /// now, nor completed), so a Progressive/Perfect *query* is satisfied only by a
    /// fact of the SAME aspect. `aspect_satisfies` encodes this one-directional
    /// subsumption.
    ///
    /// RELATIVE CLAUSES: a `Restricted{head, clause}` argument in the query is
    /// resolved to its concrete referent before matching, so "does the teacher who
    /// writes the report read the book?" is matched as "does the teacher read the
    /// book?" once the teacher is identified.
    fn holds_event(&self, query: &Event) -> Option<bool> {
        // Resolve any relative-clause arguments to their concrete referents.
        let mut q = query.clone();
        if let Some(a) = &query.agent {
            q.agent = Some(self.resolve_restricted(a));
        }
        if let Some(p) = &query.patient {
            q.patient = Some(self.resolve_restricted(p));
        }
        if let Some(r) = &query.recipient {
            q.recipient = Some(self.resolve_restricted(r));
        }
        let query = &q;

        let mut verdict: Option<bool> = None;
        for fact in &self.facts {
            if !same_event_content(fact, query) {
                continue;
            }
            // Aspect gate: the fact must be able to witness the query's aspect.
            if !aspect_satisfies(fact.aspect, query.aspect) {
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

    /// Resolve a `Restricted{head, clause}` term to the concrete entity it
    /// denotes: the known entity of category `head` for which `clause` holds (with
    /// that entity bound as the clause's agent). When a unique such entity is
    /// known we return it; otherwise (none found, or not a Restricted term) we
    /// return the term unchanged except that a Restricted term collapses to its
    /// head `Entity` (the curriculum's single-entity-per-noun default), which
    /// keeps matching sound and total. Non-Restricted terms pass through.
    fn resolve_restricted(&self, t: &Term) -> Term {
        let Term::Restricted { head, clause } = t else {
            return t.clone();
        };
        // Candidate entities: those of the head's category whose clause holds.
        let mut matches: Vec<String> = Vec::new();
        for member in self.category_members(head) {
            let mut ev = (**clause).clone();
            ev.agent = Some(Term::Entity(member.clone()));
            if self.holds_event(&ev) == Some(true) {
                matches.push(member);
            }
        }
        match matches.as_slice() {
            // Exactly one entity satisfies the restriction: that is the referent.
            [only] => Term::Entity(only.clone()),
            // Ambiguous or none proven: fall back to the head entity itself, which
            // is the lexicon's canonical single referent for the noun. (If the head
            // is itself a satisfier it will already be the [only] case.)
            _ => Term::Entity(head.clone()),
        }
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
    // assertion + truth helpers for the grammatical-core domains
    // (modal / temporal / causal / outer-negation / degree)
    // ----------------------------------------------------------------------

    /// Record a modal fact ("the teacher can/must/might/should [not] write the
    /// report"). We register the event's participants but DO NOT assert the event
    /// itself: modal force is not actuality ("can write" / "might write" leave the
    /// actual writing open). Monotonicity (`Must` |- `Can`) is applied at query
    /// time, not stored, so the record set stays minimal. A modal whose body is
    /// actually KNOWN to hold is consistent with the modal but is recorded
    /// separately as an ordinary event by whatever asserted it.
    fn assert_modal(&mut self, modality: Modality, body: &Event, negated: bool) {
        let resolved = self.resolve_event_terms(body);
        let fact = ModalFact {
            modality,
            body: resolved,
            negated,
        };
        if !self.modals.iter().any(|m| *m == fact) {
            self.modals.push(fact);
        }
    }

    /// Record a temporal ordering of two events, canonicalized to the `Before`
    /// direction ("A after B" is stored as "B before A"). Asserting an ordering
    /// PRESUPPOSES both events happen, so we also assert each as an ordinary fact
    /// (this is what lets "X writes before Y reads" answer "does X write?" -> Yes).
    /// Self-orderings (an event before itself) are degenerate and dropped.
    fn assert_temporal(&mut self, rel: TemporalRel, first: &Event, second: &Event) {
        let first = self.resolve_event_terms(first);
        let second = self.resolve_event_terms(second);
        // Both events occur (presupposition of the ordering).
        self.assert_event(&first);
        self.assert_event(&second);
        // Canonicalize to Before: for `Before`, first precedes second; for
        // `After`, the surface "first after second" means second precedes first.
        let (earlier, later) = match rel {
            TemporalRel::Before => (first, second),
            TemporalRel::After => (second, first),
        };
        if self.temporal_same(&earlier, &later) {
            // An event ordered before itself is degenerate; never store it.
            return;
        }
        let fact = TemporalFact { earlier, later };
        if !self.temporals.iter().any(|t| *t == fact) {
            self.temporals.push(fact);
        }
    }

    /// Record a causal link "EFFECT because CAUSE". Asserting it presupposes both
    /// the cause and the effect occurred, so we assert BOTH as facts in their own
    /// right, then store the directed cause->effect link. Direction is preserved
    /// exactly — we never store the reverse link (causation is not commutative).
    fn assert_causal(&mut self, cause: &Meaning, effect: &Meaning) {
        // Presupposition: both happened.
        self.assert(cause);
        self.assert(effect);
        let fact = CausalFact {
            cause: cause.clone(),
            effect: effect.clone(),
        };
        if !self.causals.iter().any(|c| *c == fact) {
            self.causals.push(fact);
        }
    }

    /// Record a conditional rule ("if P then Q"). UNLIKE `assert_causal`, this
    /// presupposes NOTHING: we do NOT assert the antecedent or the consequent —
    /// only the implication is stored (deduplicated). This keeps a conditional a
    /// strictly weaker statement than a causal link.
    ///
    /// MODUS-PONENS HOOK (where forward chaining will live): after storing the
    /// rule, a sound bounded forward chainer would scan whether the antecedent
    /// `holds() == Some(true)` and, if so, `assert` the consequent tagged
    /// `Provenance::Derived`. That step is intentionally a NO-OP for now (see
    /// `derive_modus_ponens` / `revise`) so this scaffold changes no behavior.
    fn assert_conditional(&mut self, antecedent: &Meaning, consequent: &Meaning, negated: bool) {
        let fact = ConditionalFact {
            antecedent: antecedent.clone(),
            consequent: consequent.clone(),
            negated,
        };
        if !self.conditionals.iter().any(|c| *c == fact) {
            self.conditionals.push(fact);
        }
        // Fire sound forward chaining: a freshly-stored rule whose antecedent is
        // already known true derives its consequent (modus ponens), and one whose
        // consequent is already known false derives the negated antecedent (modus
        // tollens). Runs to a bounded fixpoint so chained rules cascade.
        self.derive_modus_ponens();
    }

    /// Sound three-valued truth of a conditional "if P then Q" (material/
    /// defeasible implication), with `negated` flipping the final verdict:
    ///   - `Some(true)`  if the consequent holds, OR the antecedent is `Some(false)`
    ///                   (vacuously true);
    ///   - `Some(false)` if the antecedent is `Some(true)` AND the consequent is
    ///                   `Some(false)`;
    ///   - `None`        otherwise (open world — not enough information).
    fn holds_conditional(
        &self,
        antecedent: &Meaning,
        consequent: &Meaning,
        negated: bool,
    ) -> Option<bool> {
        let ant = self.holds(antecedent);
        let cons = self.holds(consequent);
        let base = if cons == Some(true) || ant == Some(false) {
            Some(true)
        } else if ant == Some(true) && cons == Some(false) {
            Some(false)
        } else {
            None
        };
        if negated {
            negate3(base)
        } else {
            base
        }
    }

    /// SOUND FORWARD CHAINING over the stored conditional rules — modus ponens AND
    /// modus tollens, run to a bounded fixpoint.
    ///
    /// For every non-negated `ConditionalFact { antecedent: P, consequent: Q }`:
    ///   * MODUS PONENS — if `holds(P) == Some(true)`, the consequent must hold,
    ///     so we assert `Q` as a `Provenance::Derived` fact. (P, P->Q ⊢ Q.)
    ///   * MODUS TOLLENS — if `holds(Q) == Some(false)`, the antecedent must NOT
    ///     hold, so we assert `NOT P` (the polarity-flip of P) as `Derived`.
    ///     (¬Q, P->Q ⊢ ¬P.)
    ///
    /// SOUNDNESS — the two fallacies are STRUCTURALLY impossible here because we
    /// only ever read the two valid premise shapes:
    ///   * We NEVER affirm the consequent: a true `Q` (consequent) is not used to
    ///     derive `P` — only a true ANTECEDENT fires ponens.
    ///   * We NEVER deny the antecedent: a false `P` (antecedent) is not used to
    ///     derive `¬Q` — only a false CONSEQUENT fires tollens.
    /// A NEGATED conditional ("it is not the case that if P then Q") licenses no
    /// chaining at all and is skipped.
    ///
    /// TERMINATION: each pass derives only NEW facts (the typed assert helpers
    /// deduplicate), and the derivable universe — events/categories/attributes over
    /// a finite entity/predicate vocabulary — is finite, so the fixpoint loop is
    /// bounded. A small hard iteration cap is a belt-and-suspenders guard against
    /// any unexpected non-monotonicity from belief revision.
    fn derive_modus_ponens(&mut self) {
        const MAX_PASSES: usize = 64;
        for _ in 0..MAX_PASSES {
            // Snapshot the rules: chaining mutates `facts`, not `conditionals`, so a
            // clone keeps the borrow checker happy without changing the rule set.
            let rules = self.conditionals.clone();
            let mut derived_any = false;
            for rule in &rules {
                if rule.negated {
                    continue; // a denied rule licenses no inference
                }
                // MODUS PONENS: antecedent true => derive the consequent.
                if self.holds(&rule.antecedent) == Some(true) {
                    derived_any |= self.assert_derived(&rule.consequent);
                }
                // MODUS TOLLENS: consequent false => derive the negated antecedent.
                if self.holds(&rule.consequent) == Some(false) {
                    if let Some(not_ant) = meaning_polarity_flip(&rule.antecedent) {
                        derived_any |= self.assert_derived(&not_ant);
                    }
                }
            }
            if !derived_any {
                break; // fixpoint reached
            }
        }
    }

    /// Assert `m` as a `Provenance::Derived` fact (the conclusion of a sound
    /// inference step), returning `true` iff this introduced genuinely NEW
    /// information (so the forward-chaining fixpoint can detect quiescence).
    ///
    /// Only the leaf assertoric shapes the curriculum's conditional clauses
    /// actually take are materialized — Events (and their negations), categories,
    /// and adjectival properties. Each is routed through the existing typed
    /// `assert_*` paths with `Derived` provenance where the path supports it, so
    /// belief revision still applies (a derived fact never overturns a direct
    /// assertion). Shapes we do not materialize (quantifiers, nested conditionals,
    /// questions, …) return `false` — soundly deriving nothing rather than guessing.
    fn assert_derived(&mut self, m: &Meaning) -> bool {
        // `holds` short-circuit: if the world already determines `m` with the
        // same verdict, there is nothing new to add (keeps the fixpoint finite and
        // avoids re-logging revisions for an already-settled belief).
        match m {
            Meaning::Event(ev) => {
                if self.holds_event(ev) == Some(!ev.negated) {
                    return false;
                }
                // `assert_derived_event` runs the same contradiction-detection +
                // provenance-weighted revision as a direct assert, but tags the new
                // belief `Derived` so a later direct assertion can outrank it.
                self.assert_derived_event(ev);
                true
            }
            Meaning::IsA { .. } | Meaning::HasProperty { .. } | Meaning::Comparison { .. } => {
                if self.holds(m) == Some(true) {
                    return false;
                }
                // These typed paths do not yet carry provenance; asserting through
                // the ordinary path is sound (the conclusion is genuinely true) and
                // keeps the verdict available to `holds`.
                self.assert(m);
                true
            }
            // The polarity-flip of a leaf (used by modus tollens) is itself one of
            // the shapes above wrapped in their `negated` flag, so it is handled by
            // the arms above. Anything else: derive nothing (stay sound).
            _ => false,
        }
    }

    /// Provenance of `facts[i]`, or `None` if out of range. Additive accessor;
    /// not consulted by any existing query path.
    #[allow(dead_code)]
    pub fn provenance_of(&self, i: usize) -> Option<Provenance> {
        self.fact_provenance.get(i).copied()
    }

    /// Assert an event meaning as a DERIVED belief (provenance `Derived`) rather
    /// than a direct assertion. This is the entry point a sound modus-ponens
    /// forward chainer uses for facts it INFERS, and is the counterpart of the
    /// public `assert` for the derived case. Goes through the same revision
    /// policy, so a derived belief that contradicts a directly asserted one is
    /// rejected (the assertion wins) — see [`World::revise_event`].
    pub fn assert_derived_event(&mut self, ev: &Event) {
        self.detect_contradiction(&Meaning::Event(ev.clone()));
        self.assert_event_with_provenance(ev, Provenance::Derived);
    }

    /// BELIEF-REVISION RESOLUTION for the REMAINING fact kinds (no-op for now).
    ///
    /// EVENT facts now resolve eagerly and soundly at assert time (see
    /// [`World::assert_event_with_provenance`] / [`World::revise_event`] /
    /// [`World::revisions`]): a directly asserted fact supersedes a derived one,
    /// most-recent wins between two assertions, the superseded belief is
    /// retracted, and a [`Revision`] is logged — so the world never simultaneously
    /// holds F and NOT F as Event facts.
    ///
    /// This hook is reserved for extending the SAME provenance-weighted policy to
    /// the accumulating Comparison `orderings` (removing a stale edge rather than
    /// appending its negation) and to attributes/modals. Until then it changes
    /// nothing: those kinds keep their soft most-recent-wins reading in `holds`.
    #[allow(dead_code)]
    pub fn revise(&mut self) {
        // no-op: Event revision is wired at assert time; the remaining kinds
        // (orderings/attributes/modals) keep their most-recent-wins reading.
    }

    /// Assert an outer-negated meaning `Not(inner)`. Where `inner` has a storable
    /// polarity we record its NEGATION as a fact so a later positive query of
    /// `inner` returns `Some(false)`:
    ///   - `Not(Event)` stores the event with flipped `negated`.
    ///   - `Not(IsA/HasProperty/Comparison/Attitude)` re-asserts the polarity-
    ///     flipped meaning through the normal path.
    /// A `Not` over a QUANTIFIER ("not every teacher writes a report") is a CLAIM
    /// to be checked, not a fact to materialize (its truth is computed in `holds`
    /// from the inner quantifier's three-valued truth), so it no-ops on assert —
    /// exactly mirroring how a plain existential/cardinal no-ops. This is what
    /// keeps the two negation-scope readings distinct WITHOUT polluting the model.
    fn assert_not(&mut self, inner: &Meaning) {
        match inner {
            Meaning::Event(ev) => {
                let mut e = ev.clone();
                e.negated = !e.negated;
                self.assert_event(&e);
            }
            Meaning::IsA {
                subject,
                category,
                negated,
            } => self.assert_isa(subject, category, !*negated),
            Meaning::HasProperty {
                subject,
                property,
                negated,
            } => self.assert_property(subject, property, !*negated),
            Meaning::Comparison {
                subject,
                scale,
                more,
                than,
                negated,
            } => self.assert_comparison(subject, scale, *more, than, !*negated),
            Meaning::Attitude {
                holder,
                verb,
                content,
                negated,
            } => self.assert_attitude(holder, verb, content, !*negated),
            // Quantified / Or / Cardinal / nested Not / questions: nothing storable
            // — their negation is a checked claim, not a ground fact. (A nested
            // double negation Not(Not(m)) is asserted as m, restoring the inner
            // assertion.)
            Meaning::Not(double) => self.assert(double),
            // `Not(Conditional{..})` — a denial of the rule. We re-assert the
            // conditional with its `negated` flag flipped through the normal
            // path, mirroring how Not(IsA/HasProperty/...) flips polarity.
            Meaning::Conditional {
                antecedent,
                consequent,
                negated,
            } => self.assert_conditional(antecedent, consequent, !*negated),
            Meaning::Quantified { .. }
            | Meaning::Or(_)
            | Meaning::Cardinal { .. }
            | Meaning::Modal { .. }
            | Meaning::Temporal { .. }
            | Meaning::Causal { .. }
            | Meaning::YesNoQuestion(_)
            | Meaning::WhQuestion { .. }
            | Meaning::CountQuestion { .. }
            | Meaning::DegreeQuestion { .. }
            | Meaning::Unknown(_) => {}
        }
    }

    /// Modal truth with monotonicity and the possibility/actuality firewall.
    ///
    /// SOUNDNESS (the load-bearing modal logic):
    ///   - `Must P` |- `Can P`: a stored *necessity* makes a *possibility* query
    ///     true. (Necessity entails possibility.)
    ///   - actuality |- possibility: if the bare event is KNOWN to hold in the
    ///     world (`holds_event == Some(true)`), then `Can P` / `Might P` are true
    ///     (what is actual is possible). Actuality does NOT make `Must`/`Should`
    ///     true (deontic/alethic necessity is not entailed by a single occurrence).
    ///   - possibility does NOT |- actuality: a stored `Can P`/`Might P` says
    ///     NOTHING about whether the event holds, and never about a `Must` query.
    ///   - a stored exact (modality, body, polarity) match is reported directly.
    ///   - a stored NEGATED necessity-or-possibility is consulted for explicit
    ///     denials ("cannot write" makes "can write" false).
    /// Anything not provable stays `None` (open world).
    fn holds_modal(&self, modality: Modality, body: &Event, negated: bool) -> Option<bool> {
        let body = self.resolve_event_terms(body);

        // 1) Exact stored match (same modality, body, ignoring stored polarity):
        //    fold the stored polarity against the query polarity, most-recent wins.
        let mut verdict: Option<bool> = None;
        for m in &self.modals {
            if m.modality == modality && self.modal_body_matches(&m.body, &body) {
                verdict = Some(m.negated == negated);
            }
        }

        // 2) MONOTONICITY: a stored, non-negated `Must` proves a `Can`/`Might`
        //    query. (Necessity entails possibility.) Only strengthens a positive
        //    possibility query; never fabricates a `Must`.
        if !negated && is_possibility(modality) {
            for m in &self.modals {
                if !m.negated
                    && m.modality == Modality::Must
                    && self.modal_body_matches(&m.body, &body)
                {
                    return Some(true);
                }
            }
            // 3) actuality |- possibility: a known occurrence makes "can/might"
            //    true. We consult the bare event's truth.
            if self.holds_event(&body) == Some(true) {
                return Some(true);
            }
        }

        verdict
    }

    /// Do two modal bodies denote the same event for modal matching? Reuses the
    /// content comparison (predicate/args/tense) and the aspect subsumption, so a
    /// modal over a Simple event is matched by a Simple-bodied stored modal. We do
    /// NOT compare the body's `negated` flag here — body polarity is carried by
    /// the surrounding modal's own `negated`, and the events constructed for modal
    /// bodies are affirmative.
    fn modal_body_matches(&self, a: &Event, b: &Event) -> bool {
        same_event_content(a, b) && aspect_satisfies(a.aspect, b.aspect)
    }

    /// Temporal-order truth. The query asks "does `first` happen `rel` `second`?".
    /// We canonicalize to a `Before` reachability question over the stored ordered
    /// pairs and answer three-valued, SOUNDLY:
    ///   - `Some(true)`  if `earlier -> later` is reachable by TRANSITIVE closure
    ///     of the stored `Before` pairs;
    ///   - `Some(false)` if the REVERSE ordering is reachable (Before is
    ///     ASYMMETRIC: `B before A` |- ¬`A before B`);
    ///   - `None` otherwise (open world: nothing orders this pair).
    /// Termination: closure runs over a finite pair-set with a visited set.
    fn holds_temporal(&self, rel: TemporalRel, first: &Event, second: &Event) -> Option<bool> {
        let first = self.resolve_event_terms(first);
        let second = self.resolve_event_terms(second);
        // Canonicalize the QUERY to the Before direction.
        let (earlier, later) = match rel {
            TemporalRel::Before => (&first, &second),
            TemporalRel::After => (&second, &first),
        };
        // A degenerate "X before X" is false (nothing strictly precedes itself).
        if self.temporal_same(earlier, later) {
            return Some(false);
        }
        if self.temporal_reachable(earlier, later) {
            return Some(true);
        }
        // ASYMMETRY: the proven reverse ordering makes the forward one false.
        if self.temporal_reachable(later, earlier) {
            return Some(false);
        }
        None
    }

    /// Is `later` reachable from `earlier` through the stored `Before` pairs by
    /// transitive closure? DFS over the event-ordering graph with a visited set on
    /// event indices, so it always terminates (even on a malformed cycle).
    fn temporal_reachable(&self, earlier: &Event, later: &Event) -> bool {
        // Frontier of events known to come at-or-after `earlier`.
        let mut stack: Vec<Event> = vec![earlier.clone()];
        let mut visited: Vec<Event> = vec![earlier.clone()];
        while let Some(node) = stack.pop() {
            for t in &self.temporals {
                if !self.temporal_same(&t.earlier, &node) {
                    continue;
                }
                if self.temporal_same(&t.later, later) {
                    return true;
                }
                if !visited.iter().any(|v| self.temporal_same(v, &t.later)) {
                    visited.push(t.later.clone());
                    stack.push(t.later.clone());
                }
            }
        }
        false
    }

    /// Two events are "the same" for temporal matching when their content +
    /// (subsuming) aspect + polarity align — the same notion used to store pairs.
    fn temporal_same(&self, a: &Event, b: &Event) -> bool {
        same_event_content(a, b)
            && (aspect_satisfies(a.aspect, b.aspect) || aspect_satisfies(b.aspect, a.aspect))
            && a.negated == b.negated
    }

    /// Causal-link truth: is the directed `cause -> effect` link known? Matching
    /// is structural on both meanings. `Some(true)` when the exact directed link
    /// was asserted; `None` otherwise. We DELIBERATELY never report `true` for the
    /// reverse link (causation is not commutative) and never derive a causal link
    /// from a mere material conditional — only an explicitly asserted "because".
    fn holds_causal(&self, cause: &Meaning, effect: &Meaning) -> Option<bool> {
        for c in &self.causals {
            if &c.cause == cause && &c.effect == effect {
                return Some(true);
            }
        }
        None
    }

    /// The cause of `effect`, if a causal link was asserted with this effect —
    /// backs the "why does the street flood?" answer. Two `Event` effects match by
    /// CONTENT (predicate + tense + agent/patient) so the question's surface aspect
    /// need not equal the asserting sentence's; other meanings match exactly.
    /// Returns the MOST-RECENTLY asserted cause for the effect, `None` if no link
    /// records it. The reverse (effect->cause) direction is never consulted, so
    /// causation's non-commutativity is preserved.
    pub fn cause_of(&self, effect: &Meaning) -> Option<Meaning> {
        self.causals
            .iter()
            .rev()
            .find(|c| match (&c.effect, effect) {
                (Meaning::Event(stored), Meaning::Event(asked)) => {
                    same_event_content(stored, asked)
                }
                (stored, asked) => stored == asked,
            })
            .map(|c| c.cause.clone())
    }

    /// DEGREE-QUESTION SUPPORT: the known comparative position of `entity` on
    /// `scale`, as a `(more, other)` pair meaning "`entity` is more/less than
    /// `other`". We answer from asserted comparison facts: if the entity is the
    /// greater end of some ordering on the scale we report `(true, lesser)` ("it
    /// is longer than <lesser>"); if it is the lesser end we report
    /// `(false, greater)` ("it is shorter than <greater>"). `None` when no
    /// comparison on the scale mentions the entity — honestly "I don't know",
    /// since the world has no numeric measures, only relative comparisons.
    ///
    /// SOUNDNESS: we only surface a DIRECTLY-asserted, non-negated ordering edge
    /// incident to the entity; we do not invent magnitudes. Preference is given to
    /// a "greater-than" framing (it reads more naturally as an answer) but either
    /// is correct.
    pub fn degree_position(&self, entity: &str, scale: &str) -> Option<(bool, String)> {
        // Prefer "entity is greater than X".
        if let Some(o) = self
            .orderings
            .iter()
            .find(|o| !o.negated && o.scale == scale && o.greater == entity)
        {
            return Some((true, o.lesser.clone()));
        }
        // Else "entity is less than X".
        if let Some(o) = self
            .orderings
            .iter()
            .find(|o| !o.negated && o.scale == scale && o.lesser == entity)
        {
            return Some((false, o.greater.clone()));
        }
        None
    }

    /// Resolve every relative-clause argument of an event to its concrete
    /// referent, leaving plain terms untouched. Used by the modal/temporal/causal
    /// asserters and queries so a `Restricted` subject is matched consistently.
    fn resolve_event_terms(&self, ev: &Event) -> Event {
        let mut out = ev.clone();
        if let Some(a) = &ev.agent {
            out.agent = Some(self.resolve_restricted(a));
        }
        if let Some(p) = &ev.patient {
            out.patient = Some(self.resolve_restricted(p));
        }
        if let Some(r) = &ev.recipient {
            out.recipient = Some(self.resolve_restricted(r));
        }
        out
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
        self.entity_in_category_known(head, category)
            .unwrap_or(false)
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

/// Aspect subsumption for event matching: can a stored fact of aspect
/// `fact_aspect` WITNESS a query of aspect `query_aspect`?
///
/// ENTAILMENT DIRECTION (sound): a Progressive ("is writing") or Perfect ("has
/// written") event entails the SIMPLE event holds, so a fact of ANY aspect
/// satisfies a Simple query. The converse is unsound — a habitual/simple "writes"
/// need not be ongoing right now (Progressive) nor completed (Perfect) — so a
/// Progressive/Perfect query is satisfied ONLY by a fact of the exact same
/// aspect. (Progressive and Perfect are not interderivable from each other
/// either, so they only match like-for-like.)
fn aspect_satisfies(fact_aspect: Aspect, query_aspect: Aspect) -> bool {
    match query_aspect {
        // Any aspect entails the simple event holds.
        Aspect::Simple => true,
        // A marked aspect requires a like-marked witness.
        Aspect::Progressive => fact_aspect == Aspect::Progressive,
        Aspect::Perfect => fact_aspect == Aspect::Perfect,
    }
}

/// Is `modality` a POSSIBILITY operator (`Can`/`Might`)? Possibility is entailed
/// by necessity (`Must`) and by actuality; necessity operators (`Must`/`Should`)
/// are not. Used to gate the monotonicity rule in `holds_modal`.
fn is_possibility(modality: Modality) -> bool {
    matches!(modality, Modality::Can | Modality::Might)
}

/// Human-readable justification for a belief revision, given the provenance of
/// the incoming and existing beliefs and which one won. Keeps the policy
/// auditable: the reason text NAMES the principle that decided the resolution.
fn revision_reason(
    incoming_prov: Provenance,
    existing_prov: Provenance,
    incoming_wins: bool,
) -> String {
    match (incoming_prov, existing_prov) {
        (Provenance::Asserted, Provenance::Derived) => {
            "directly asserted fact supersedes a derived belief it contradicts".to_string()
        }
        (Provenance::Derived, Provenance::Asserted) => {
            "derived belief is rejected: it contradicts a directly asserted fact".to_string()
        }
        _ if incoming_wins => {
            "most-recent assertion supersedes the prior contradictory assertion".to_string()
        }
        _ => "prior belief retained over the contradictory incoming belief".to_string(),
    }
}

/// Three-valued (Kleene) negation of a truth value: `Some(true)` <-> `Some(false)`,
/// and `None` (undetermined) negates to `None`. This is the truth function behind
/// the outer-scope `Not(m)` meaning — distinct from a leaf's own `negated` flag.
fn negate3(v: Option<bool>) -> Option<bool> {
    v.map(|b| !b)
}

/// The sound CONTRADICTORY of a leaf assertoric meaning: its same content with
/// the `negated` flag flipped. Used by MODUS TOLLENS to materialize "NOT P" once
/// the consequent of "if P then Q" is known false. Defined only for the leaf
/// shapes a conditional clause actually takes in this curriculum (Event, IsA,
/// HasProperty, Comparison); a wide-scope `Not(inner)` flips by unwrapping to its
/// inner meaning. Returns `None` for everything else, so modus tollens derives a
/// negation only when it can name a genuine single-meaning contradictory — never
/// fabricating one.
fn meaning_polarity_flip(m: &Meaning) -> Option<Meaning> {
    match m {
        Meaning::Event(ev) => {
            let mut e = ev.clone();
            e.negated = !e.negated;
            Some(Meaning::Event(e))
        }
        Meaning::IsA {
            subject,
            category,
            negated,
        } => Some(Meaning::IsA {
            subject: subject.clone(),
            category: category.clone(),
            negated: !negated,
        }),
        Meaning::HasProperty {
            subject,
            property,
            negated,
        } => Some(Meaning::HasProperty {
            subject: subject.clone(),
            property: property.clone(),
            negated: !negated,
        }),
        Meaning::Comparison {
            subject,
            scale,
            more,
            than,
            negated,
        } => Some(Meaning::Comparison {
            subject: subject.clone(),
            scale: scale.clone(),
            more: *more,
            than: than.clone(),
            negated: !negated,
        }),
        // A wide-scope negation flips to its inner meaning (¬¬P ≡ P).
        Meaning::Not(inner) => Some((**inner).clone()),
        // No single-meaning contradictory for the rest: derive nothing.
        _ => None,
    }
}

/// For an assertoric meaning (Event / IsA / HasProperty / Comparison), return
/// its POSITIVE (un-negated) form together with whether the original meaning
/// CLAIMS that positive form is true. An affirmative assertion claims `true`; a
/// negated one ("does not write" / "is not careful" / "is not longer than")
/// claims `false`. A wide-scope `Not(...)` wrapper over one of these four kinds
/// is unwrapped and flips the claimed polarity (so `Not(Event{neg:false})` and
/// `Event{neg:true}` agree). Returns `None` for every NON-assertoric meaning
/// (quantifiers, disjunctions, modals, attitudes, temporals, causals,
/// questions, unknowns) — contradiction flagging is limited to ground
/// assertoric facts by contract.
fn assertoric_positive_form(m: &Meaning) -> Option<(Meaning, bool)> {
    match m {
        Meaning::Event(ev) => {
            let mut positive = ev.clone();
            let claims_true = !positive.negated;
            positive.negated = false;
            Some((Meaning::Event(positive), claims_true))
        }
        Meaning::IsA {
            subject,
            category,
            negated,
        } => Some((
            Meaning::IsA {
                subject: subject.clone(),
                category: category.clone(),
                negated: false,
            },
            !*negated,
        )),
        Meaning::HasProperty {
            subject,
            property,
            negated,
        } => Some((
            Meaning::HasProperty {
                subject: subject.clone(),
                property: property.clone(),
                negated: false,
            },
            !*negated,
        )),
        Meaning::Comparison {
            subject,
            scale,
            more,
            than,
            negated,
        } => Some((
            Meaning::Comparison {
                subject: subject.clone(),
                scale: scale.clone(),
                more: *more,
                than: than.clone(),
                negated: false,
            },
            !*negated,
        )),
        // A wide-scope negation over an assertoric kind: unwrap and FLIP the
        // claimed polarity. We descend at most one level into the four assertoric
        // kinds; a `Not` over anything else (quantifier/modal/...) is not an
        // assertoric claim we flag.
        Meaning::Not(inner) => assertoric_positive_form(inner).map(|(p, claims)| (p, !claims)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::understanding::meaning::{Aspect, Modality, TemporalRel, Tense};

    /// Build a present, affirmative (or negated) write(agent, patient) event.
    fn write_event(agent: &str, patient: &str, negated: bool) -> Event {
        Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity(agent.to_string())),
            patient: Some(Term::Entity(patient.to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
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
            aspect: Aspect::Simple,
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
            aspect: Aspect::Simple,
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
            aspect: Aspect::Simple,
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
            assert_eq!(
                w.holds(m),
                Some(true),
                "closure emitted a non-holding fact: {m:?}"
            );
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
        assert_eq!(
            w.holds(&cmp("report", "length", true, "book", false)),
            Some(true)
        );
        // ASYMMETRY: the reverse "the book is longer than the report" is FALSE,
        // never silently true (we must not infer symmetry).
        assert_eq!(
            w.holds(&cmp("book", "length", true, "report", false)),
            Some(false)
        );
        // "the book is SHORTER than the report" is the same ordering -> true.
        assert_eq!(
            w.holds(&cmp("book", "length", false, "report", false)),
            Some(true)
        );
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
        assert_eq!(
            w.holds(&cmp("report", "length", true, "book", false)),
            Some(true)
        );
        // ... and the reverse C > A is false by asymmetry of the proven order.
        assert_eq!(
            w.holds(&cmp("book", "length", true, "report", false)),
            Some(false)
        );
        // A pair with no path on the scale stays unknown.
        assert_eq!(
            w.holds(&cmp("book", "length", true, "essay", false)),
            Some(false)
        ); // essay>book known, so book>essay false
        assert_eq!(w.holds(&cmp("memo", "length", true, "note", false)), None);
    }

    #[test]
    fn comparison_negation_and_explicit_denial() {
        let mut w = World::new();
        w.assert(&cmp("report", "length", true, "book", false));
        // Query "is the report NOT longer than the book?" -> No (it IS longer).
        assert_eq!(
            w.holds(&cmp("report", "length", true, "book", true)),
            Some(false)
        );
        // An explicit denial makes the positive query false without inventing the
        // reverse ordering.
        let mut w2 = World::new();
        w2.assert(&cmp("memo", "length", true, "note", true)); // "memo is NOT longer than note"
        assert_eq!(
            w2.holds(&cmp("memo", "length", true, "note", false)),
            Some(false)
        );
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

    // ===================================================================
    // GRAMMATICAL-CORE DOMAINS (the nine new forms)
    // ===================================================================

    /// A write(agent, patient) event with explicit tense + aspect.
    fn write_ta(agent: &str, patient: &str, tense: Tense, aspect: Aspect) -> Event {
        Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity(agent.to_string())),
            patient: Some(Term::Entity(patient.to_string())),
            recipient: None,
            tense,
            aspect,
            negated: false,
        }
    }

    // ---- (1) ASPECT ---------------------------------------------------

    #[test]
    fn aspect_progressive_and_perfect_entail_simple() {
        // "the teacher is writing the report" (Progressive) makes the SIMPLE event
        // hold, and likewise a Perfect fact.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Progressive,
        )));
        let simple = Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Simple,
        ));
        assert_eq!(w.holds(&simple), Some(true));

        let mut w2 = World::new();
        w2.assert(&Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Perfect,
        )));
        assert_eq!(w2.holds(&simple), Some(true));
    }

    #[test]
    fn aspect_simple_does_not_entail_progressive_or_perfect() {
        // SOUNDNESS: a habitual/simple "writes" must NOT make "is writing" or "has
        // written" true — the converse aspect entailment is invalid.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Simple,
        )));
        let prog = Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Progressive,
        ));
        let perf = Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Perfect,
        ));
        assert_eq!(w.holds(&prog), None);
        assert_eq!(w.holds(&perf), None);
    }

    #[test]
    fn aspect_progressive_query_matches_progressive_fact() {
        // A like-for-like aspect query is answered.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Progressive,
        )));
        let prog = Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Progressive,
        ));
        assert_eq!(w.holds(&prog), Some(true));
    }

    #[test]
    fn future_tense_is_distinct_from_present() {
        // "will write" (Future) is a different fact from "writes" (Present): one
        // does not answer the other.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Future,
            Aspect::Simple,
        )));
        let future = Meaning::Event(write_ta("teacher", "report", Tense::Future, Aspect::Simple));
        let present = Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Simple,
        ));
        assert_eq!(w.holds(&future), Some(true));
        assert_eq!(w.holds(&present), None);
    }

    // ---- (2) MODALITY -------------------------------------------------

    fn modal(modality: Modality, negated: bool) -> Meaning {
        Meaning::Modal {
            modality,
            body: Box::new(write_ta(
                "teacher",
                "report",
                Tense::Present,
                Aspect::Simple,
            )),
            negated,
        }
    }

    #[test]
    fn modal_must_entails_can_not_converse() {
        // "the teacher MUST write the report" makes "CAN write" true (necessity
        // entails possibility)...
        let mut w = World::new();
        w.assert(&modal(Modality::Must, false));
        assert_eq!(w.holds(&modal(Modality::Must, false)), Some(true));
        assert_eq!(w.holds(&modal(Modality::Can, false)), Some(true));
        assert_eq!(w.holds(&modal(Modality::Might, false)), Some(true));

        // ...but the CONVERSE fails: "can write" does NOT make "must write" true.
        let mut w2 = World::new();
        w2.assert(&modal(Modality::Can, false));
        assert_eq!(w2.holds(&modal(Modality::Can, false)), Some(true));
        assert_eq!(w2.holds(&modal(Modality::Must, false)), None);
    }

    #[test]
    fn modal_possibility_does_not_entail_actuality() {
        // SOUNDNESS: "the teacher can write the report" says NOTHING about whether
        // the writing actually happens.
        let mut w = World::new();
        w.assert(&modal(Modality::Can, false));
        let actual = Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Simple,
        ));
        assert_eq!(w.holds(&actual), None);
        // ...and "might" likewise.
        let mut w2 = World::new();
        w2.assert(&modal(Modality::Might, false));
        assert_eq!(w2.holds(&actual), None);
    }

    #[test]
    fn modal_actuality_entails_possibility() {
        // A KNOWN occurrence makes "can/might" true (what is actual is possible)
        // but never fabricates a necessity.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Simple,
        )));
        assert_eq!(w.holds(&modal(Modality::Can, false)), Some(true));
        assert_eq!(w.holds(&modal(Modality::Might, false)), Some(true));
        assert_eq!(w.holds(&modal(Modality::Must, false)), None);
        assert_eq!(w.holds(&modal(Modality::Should, false)), None);
    }

    #[test]
    fn modal_negation_and_open_world() {
        // "the teacher cannot write the report": the positive "can" is false.
        let mut w = World::new();
        w.assert(&modal(Modality::Can, true));
        assert_eq!(w.holds(&modal(Modality::Can, false)), Some(false));
        // Nothing asserted -> open world.
        let w2 = World::new();
        assert_eq!(w2.holds(&modal(Modality::Can, false)), None);
    }

    // ---- (3) RELATIVE CLAUSES -----------------------------------------

    #[test]
    fn relative_clause_subject_resolves_to_matching_entity() {
        // "the teacher who writes the report reads the book": the subject is the
        // teacher that satisfies the clause; the read fact is about that teacher.
        let mut w = World::new();
        // The clause content is already known.
        w.assert(&Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Present,
            Aspect::Simple,
        )));
        let restricted = Term::Restricted {
            head: "teacher".to_string(),
            clause: Box::new(write_ta(
                "teacher",
                "report",
                Tense::Present,
                Aspect::Simple,
            )),
        };
        let read = Event {
            predicate: "read".to_string(),
            agent: Some(restricted.clone()),
            patient: Some(Term::Entity("book".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        w.assert(&Meaning::Event(read));
        // The read fact is stored as the plain teacher reading the book.
        let q = Event {
            predicate: "read".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Entity("book".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        assert_eq!(w.holds(&Meaning::Event(q)), Some(true));
    }

    // ---- (4) PASSIVE --------------------------------------------------

    #[test]
    fn passive_maps_to_active_predicate_argument_structure() {
        // "the report was written by the teacher" is the SAME fact as the active
        // "the teacher wrote the report" (agent=teacher, patient=report, past).
        let mut w = World::new();
        // Active assertion.
        w.assert(&Meaning::Event(write_ta(
            "teacher",
            "report",
            Tense::Past,
            Aspect::Simple,
        )));
        // The passive surface produces the same Event shape, so the same query
        // answers Yes.
        let passive_as_active =
            Meaning::Event(write_ta("teacher", "report", Tense::Past, Aspect::Simple));
        assert_eq!(w.holds(&passive_as_active), Some(true));
    }

    // ---- (5) PLURALS / distributive -----------------------------------

    #[test]
    fn plural_distributive_universal_over_known_members() {
        // "teachers write reports" reads as Every-over-known-persons; asserting it
        // makes each known person write, and the universal query holds.
        let mut w = World::new();
        // Two known persons.
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
        // Distributive plural represented as a universal (the chosen rep).
        w.assert(&Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: quant_body("report"),
        });
        let teacher_writes = Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        });
        assert_eq!(w.holds(&teacher_writes), Some(true));
    }

    // ---- (6) TEMPORAL -------------------------------------------------

    fn read_event(agent: &str, patient: &str) -> Event {
        Event {
            predicate: "read".to_string(),
            agent: Some(Term::Entity(agent.to_string())),
            patient: Some(Term::Entity(patient.to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        }
    }

    fn temporal(rel: TemporalRel, first: Event, second: Event) -> Meaning {
        Meaning::Temporal {
            rel,
            first: Box::new(first),
            second: Box::new(second),
        }
    }

    #[test]
    fn temporal_before_truth_and_presupposition() {
        // "the teacher writes the report before the editor reads the book".
        let mut w = World::new();
        let write = write_ta("teacher", "report", Tense::Present, Aspect::Simple);
        let read = read_event("editor", "book");
        w.assert(&temporal(TemporalRel::Before, write.clone(), read.clone()));
        // The ordering holds.
        assert_eq!(
            w.holds(&temporal(TemporalRel::Before, write.clone(), read.clone())),
            Some(true)
        );
        // PRESUPPOSITION: both events are now facts.
        assert_eq!(w.holds(&Meaning::Event(write.clone())), Some(true));
        assert_eq!(w.holds(&Meaning::Event(read.clone())), Some(true));
        // "after" is the converse and is the same stored ordering.
        assert_eq!(
            w.holds(&temporal(TemporalRel::After, read.clone(), write.clone())),
            Some(true)
        );
    }

    #[test]
    fn temporal_asymmetry() {
        // SOUNDNESS: "A before B" makes "B before A" FALSE (asymmetric), never
        // silently true.
        let mut w = World::new();
        let a = write_ta("teacher", "report", Tense::Present, Aspect::Simple);
        let b = read_event("editor", "book");
        w.assert(&temporal(TemporalRel::Before, a.clone(), b.clone()));
        assert_eq!(
            w.holds(&temporal(TemporalRel::Before, b.clone(), a.clone())),
            Some(false)
        );
    }

    #[test]
    fn temporal_transitivity() {
        // A before B and B before C entail A before C (and not C before A).
        let mut w = World::new();
        let a = write_ta("teacher", "report", Tense::Present, Aspect::Simple);
        let b = read_event("editor", "book");
        let c = Event {
            predicate: "send".to_string(),
            agent: Some(Term::Entity("clerk".to_string())),
            patient: Some(Term::Entity("memo".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        w.assert(&temporal(TemporalRel::Before, a.clone(), b.clone()));
        w.assert(&temporal(TemporalRel::Before, b.clone(), c.clone()));
        assert_eq!(
            w.holds(&temporal(TemporalRel::Before, a.clone(), c.clone())),
            Some(true)
        );
        assert_eq!(
            w.holds(&temporal(TemporalRel::Before, c.clone(), a.clone())),
            Some(false)
        );
        // An unordered pair stays open.
        let d = read_event("author", "letter");
        assert_eq!(w.holds(&temporal(TemporalRel::Before, a, d)), None);
    }

    #[test]
    fn temporal_cycle_terminates() {
        // Defensive: a (degenerate) cycle must not hang.
        let mut w = World::new();
        let a = write_ta("teacher", "report", Tense::Present, Aspect::Simple);
        let b = read_event("editor", "book");
        w.assert(&temporal(TemporalRel::Before, a.clone(), b.clone()));
        w.assert(&temporal(TemporalRel::Before, b.clone(), a.clone()));
        let _ = w.holds(&temporal(TemporalRel::Before, a.clone(), b.clone()));
        let _ = w.holds(&temporal(TemporalRel::Before, b, a));
    }

    // ---- (7) CAUSAL ---------------------------------------------------

    fn flood() -> Meaning {
        Meaning::Event(Event {
            predicate: "flood".to_string(),
            agent: Some(Term::Entity("street".to_string())),
            patient: None,
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        })
    }
    fn rain_falls() -> Meaning {
        Meaning::Event(Event {
            predicate: "fall".to_string(),
            agent: Some(Term::Entity("rain".to_string())),
            patient: None,
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        })
    }

    #[test]
    fn causal_link_truth_presupposition_and_noncommutativity() {
        // "the street floods because the rain falls".
        let mut w = World::new();
        let causal = Meaning::Causal {
            cause: Box::new(rain_falls()),
            effect: Box::new(flood()),
        };
        w.assert(&causal);
        // The link holds...
        assert_eq!(w.holds(&causal), Some(true));
        // PRESUPPOSITION: both cause and effect are facts.
        assert_eq!(w.holds(&rain_falls()), Some(true));
        assert_eq!(w.holds(&flood()), Some(true));
        // NON-COMMUTATIVITY: the reverse link is NOT known.
        let reverse = Meaning::Causal {
            cause: Box::new(flood()),
            effect: Box::new(rain_falls()),
        };
        assert_eq!(w.holds(&reverse), None);
        // cause_of backs the "why" answer.
        assert_eq!(w.cause_of(&flood()), Some(rain_falls()));
        assert_eq!(w.cause_of(&rain_falls()), None);
    }

    // ---- (8) DEGREE QUESTIONS -----------------------------------------

    #[test]
    fn degree_position_from_comparison_facts() {
        // "how long is the report?" answerable from "the report is longer than the
        // book": the report's known position on length is greater-than book.
        let mut w = World::new();
        w.assert(&Meaning::Comparison {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
            more: true,
            than: Term::Entity("book".to_string()),
            negated: false,
        });
        assert_eq!(
            w.degree_position("report", "length"),
            Some((true, "book".to_string()))
        );
        // The book's position is the lesser end.
        assert_eq!(
            w.degree_position("book", "length"),
            Some((false, "report".to_string()))
        );
        // Unknown scale / entity -> honest None.
        assert_eq!(w.degree_position("report", "weight"), None);
        assert_eq!(w.degree_position("memo", "length"), None);
        // The DegreeQuestion meaning itself is never truth-evaluated.
        let dq = Meaning::DegreeQuestion {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
        };
        assert_eq!(w.holds(&dq), None);
    }

    // ---- (9) NEGATION SCOPE -------------------------------------------

    #[test]
    fn negation_scope_two_readings_differ() {
        // World: teacher writes a report, editor does NOT.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        w.assert(&Meaning::Event(write_event("editor", "report", true)));

        // Reading A: "NOT every teacher writes a report" = Not(Every ...).
        // "every person writes" is false (editor counterexample), so Not(...) is
        // TRUE ("some person does not write").
        let not_every = Meaning::Not(Box::new(Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: quant_body("report"),
        }));
        assert_eq!(w.holds(&not_every), Some(true));

        // Reading B: "every person does NOT write a report" = Every with negated
        // body = "no person writes". teacher DOES write, so this is FALSE.
        let mut negated_body = quant_body("report");
        negated_body.negated = true;
        let every_not = Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: negated_body,
        };
        assert_eq!(w.holds(&every_not), Some(false));

        // The two readings get DIFFERENT truth values from the same world.
        assert_ne!(w.holds(&not_every), w.holds(&every_not));
    }

    #[test]
    fn outer_negation_is_three_valued() {
        // Not(m) negates m's three-valued truth: true<->false, None stays None.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        let p = Meaning::Event(write_event("teacher", "report", false));
        assert_eq!(w.holds(&Meaning::Not(Box::new(p.clone()))), Some(false));
        // Double negation restores the value.
        assert_eq!(
            w.holds(&Meaning::Not(Box::new(Meaning::Not(Box::new(p))))),
            Some(true)
        );
        // Unknown inner -> Not is also None (honest).
        let unknown = Meaning::Event(write_event("author", "book", false));
        assert_eq!(w.holds(&Meaning::Not(Box::new(unknown))), None);
    }

    #[test]
    fn assert_not_event_records_negation() {
        // Asserting Not(Event) records the event as negated, so the positive query
        // is Some(false).
        let mut w = World::new();
        let p = Meaning::Event(write_event("teacher", "report", false));
        w.assert(&Meaning::Not(Box::new(p.clone())));
        assert_eq!(w.holds(&p), Some(false));
    }

    // -------------------------------------------------------------------
    // Contradiction detection
    // -------------------------------------------------------------------

    #[test]
    fn contradiction_flagged_on_event_negation_clash() {
        // "the teacher writes the report." then "the teacher does not write the
        // report." — the negation contradicts the prior positive fact.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        // No conflict yet (first assertion is over an empty world).
        assert_eq!(w.contradictions().len(), 0);
        // The negation conflicts with the entailed-true positive fact.
        w.assert(&Meaning::Event(write_event("teacher", "report", true)));
        assert_eq!(w.contradictions().len(), 1);
        // The logged conflict carries the incoming (negated) meaning.
        let c = &w.contradictions()[0];
        assert_eq!(
            c.incoming,
            Meaning::Event(write_event("teacher", "report", true))
        );
        assert!(!c.note.is_empty());
        // We FLAG but never RETRACT: the negated fact is still recorded, so the
        // most-recent assertion now makes the positive query Some(false).
        let positive = Meaning::Event(write_event("teacher", "report", false));
        assert_eq!(w.holds(&positive), Some(false));
    }

    #[test]
    fn consistent_pair_adds_no_contradiction() {
        // Re-asserting the SAME positive fact is not a contradiction.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        assert_eq!(w.contradictions().len(), 0);
        // An unrelated, independent fact is also no contradiction.
        w.assert(&Meaning::Event(write_event("editor", "memo", false)));
        assert_eq!(w.contradictions().len(), 0);
    }

    #[test]
    fn contradiction_flagged_when_positive_clashes_with_prior_negation() {
        // Reverse order: assert the negation first, then the positive. The world
        // already entails the positive is FALSE, so the positive assertion clashes
        // (this is the literal `holds(positive_form) == Some(false)` trigger).
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", true)));
        assert_eq!(w.contradictions().len(), 0);
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        assert_eq!(w.contradictions().len(), 1);
    }

    #[test]
    fn contradiction_flagged_on_isa_clash() {
        // "the report is a thing" is entailed (animacy/taxonomy); asserting "the
        // report is not a thing" contradicts it.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        let report_not_thing = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "thing".to_string(),
            negated: true,
        };
        w.assert(&report_not_thing);
        assert_eq!(w.contradictions().len(), 1);
    }

    #[test]
    fn contradiction_flagged_on_property_clash() {
        // "the teacher is careful." then "the teacher is not careful." clash.
        let mut w = World::new();
        let careful = Meaning::HasProperty {
            subject: Term::Entity("teacher".to_string()),
            property: "careful".to_string(),
            negated: false,
        };
        let not_careful = Meaning::HasProperty {
            subject: Term::Entity("teacher".to_string()),
            property: "careful".to_string(),
            negated: true,
        };
        w.assert(&careful);
        assert_eq!(w.contradictions().len(), 0);
        w.assert(&not_careful);
        assert_eq!(w.contradictions().len(), 1);
    }

    #[test]
    fn contradiction_flagged_via_not_wrapper() {
        // A wide-scope Not(Event) over an entailed-true positive is also flagged
        // (the Not wrapper is unwrapped to the assertoric positive form).
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        let p = Meaning::Event(write_event("teacher", "report", false));
        w.assert(&Meaning::Not(Box::new(p)));
        assert_eq!(w.contradictions().len(), 1);
    }

    #[test]
    fn open_world_unknown_is_not_a_contradiction() {
        // With no prior information, a negated assertion is just a new fact, not a
        // clash (holds(positive) is None, not a definite opposite).
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", true)));
        assert_eq!(w.contradictions().len(), 0);
    }

    #[test]
    fn non_assertoric_meanings_never_flagged() {
        // A quantifier whose body is even determined-false is a CHECKED claim, not
        // an assertoric ground fact, so it is never flagged as a contradiction.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", true)));
        w.assert(&Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: quant_body("report"),
        });
        assert_eq!(w.contradictions().len(), 0);
    }

    #[test]
    fn discrimination_probe_detector_is_not_trivial() {
        // ANTI-TRIVIALITY: prove detect_contradiction actually DISCRIMINATES and
        // is neither an always-0 stub (would miss the clash) nor an always-1 stub
        // (would false-positive on the consistent case).

        // (1) Consistent second fact after the first => still 0 (not always-1).
        let mut consistent = World::new();
        consistent.assert(&Meaning::Event(write_event("teacher", "report", false)));
        consistent.assert(&Meaning::Event(write_event("teacher", "report", false)));
        assert_eq!(consistent.contradictions().len(), 0);

        // (2) Genuine clash => >=1 (not always-0). Independent injected clash.
        let mut clash = World::new();
        clash.assert(&Meaning::HasProperty {
            subject: Term::Entity("editor".to_string()),
            property: "tired".to_string(),
            negated: false,
        });
        clash.assert(&Meaning::HasProperty {
            subject: Term::Entity("editor".to_string()),
            property: "tired".to_string(),
            negated: true,
        });
        assert_eq!(clash.contradictions().len(), 1);

        // (3) Asserting the SAME negation twice over a held-true fact flags TWICE
        // (each assertion is checked against the then-current world). Documents the
        // append-only, per-assertion semantics — it is not deduplicated.
        let mut twice = World::new();
        twice.assert(&Meaning::Event(write_event("teacher", "report", false)));
        twice.assert(&Meaning::Event(write_event("teacher", "report", true)));
        twice.assert(&Meaning::Event(write_event("teacher", "report", true)));
        // After the first negation the world holds Some(false) for the positive;
        // the SECOND negation claims the positive is false too -> NO new clash with
        // the (now-negative) state, so the count stays at 1. This pins the exact
        // semantics under repetition.
        assert_eq!(twice.contradictions().len(), 1);
    }

    #[test]
    fn adversarial_contradiction_substrate_exact_counts() {
        // ADVERSARIAL VERIFICATION of the contradiction-detection substrate.
        // Contract: asserting a fact then its negation must report EXACTLY ONE
        // inconsistency; a wholly consistent world reports ZERO; and a large
        // battery of independent consistent assertions must NOT produce a single
        // false positive.

        // ---- Part A: fact then its negation -> EXACTLY 1 inconsistency. ----
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        assert_eq!(
            w.contradictions().len(),
            0,
            "a lone fact over an empty world is consistent"
        );
        // The direct negation of the just-asserted fact.
        w.assert(&Meaning::Event(write_event("teacher", "report", true)));
        assert_eq!(
            w.contradictions().len(),
            1,
            "a fact and its negation must report EXACTLY one inconsistency"
        );
        // The single logged conflict names the incoming (negated) meaning and a
        // non-empty human-readable note.
        let c = &w.contradictions()[0];
        assert_eq!(
            c.incoming,
            Meaning::Event(write_event("teacher", "report", true)),
            "the logged contradiction carries the conflicting incoming meaning"
        );
        assert!(
            !c.note.is_empty(),
            "the contradiction carries a description"
        );

        // ---- Part B: a wholly consistent world reports ZERO. ----
        // A SEPARATE world built only from mutually compatible assertions: distinct
        // events, an IsA on its animacy-consistent category, a property, and a
        // benign re-assertion of an identical fact (idempotent, not a clash).
        let mut consistent = World::new();
        // Distinct, non-overlapping events (different agents/patients/predicates).
        consistent.assert(&Meaning::Event(write_event("teacher", "report", false)));
        consistent.assert(&Meaning::Event(read_event("editor", "book")));
        consistent.assert(&Meaning::Event(write_event("author", "memo", false)));
        // An IsA consistent with animacy ("report" is inanimate => a thing).
        consistent.assert(&Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "thing".to_string(),
            negated: false,
        });
        // A property with no opposing assertion.
        consistent.assert(&Meaning::HasProperty {
            subject: Term::Entity("teacher".to_string()),
            property: "careful".to_string(),
            negated: false,
        });
        // Idempotent re-assertion of an already-held fact (NOT a contradiction).
        consistent.assert(&Meaning::Event(write_event("teacher", "report", false)));
        // A negated assertion about a SEPARATE, previously-unmentioned fact: the
        // world holds None for it, so under the open-world assumption this is new
        // information, not a clash.
        consistent.assert(&Meaning::Event(write_event("clerk", "ledger", true)));
        assert_eq!(
            consistent.contradictions().len(),
            0,
            "no false positives: a fully consistent world flags nothing"
        );

        // ---- Part C: the substrate stays sound under repeated probing. ----
        // The flag is observational and append-only: re-running holds on the
        // first world's positive fact still reflects the (non-retracted) negation,
        // and the contradiction count never spontaneously grows.
        let positive = Meaning::Event(write_event("teacher", "report", false));
        assert_eq!(
            w.holds(&positive),
            Some(false),
            "flagged-but-not-retracted: the negation is still recorded"
        );
        assert_eq!(
            w.contradictions().len(),
            1,
            "querying does not mutate the contradiction ledger"
        );
    }

    // -------------------------------------------------------------------
    // Actionable belief revision (sound, provenance-weighted)
    // -------------------------------------------------------------------

    #[test]
    fn revision_two_direct_assertions_most_recent_wins() {
        // F asserted, then NOT F asserted (both DIRECT). Most-recent-wins: the
        // world ends holding NOT F, exactly ONE revision is recorded, and the
        // world NEVER simultaneously holds F and NOT F.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false))); // F
        assert_eq!(w.revisions().len(), 0, "no revision before any conflict");
        w.assert(&Meaning::Event(write_event("teacher", "report", true))); // NOT F

        let positive = Meaning::Event(write_event("teacher", "report", false));
        let negative = Meaning::Event(write_event("teacher", "report", true));
        // The SURVIVING live belief is NOT F (the most recent direct assertion).
        assert_eq!(
            w.holds(&positive),
            Some(false),
            "world holds NOT F after revision"
        );
        assert_eq!(
            w.holds(&negative),
            Some(true),
            "the negation is the live belief"
        );

        // Exactly one revision, and it supersedes F in favor of NOT F.
        assert_eq!(w.revisions().len(), 1, "exactly one revision recorded");
        let r = &w.revisions()[0];
        assert_eq!(r.superseded.negated, false, "the superseded belief was F");
        assert_eq!(r.surviving.negated, true, "the surviving belief is NOT F");
        assert!(
            r.reason.contains("most-recent"),
            "most-recent-wins reason: {}",
            r.reason
        );

        // COHERENCE INVARIANT: the live store holds exactly ONE polarity for this
        // content — never both F and NOT F at once.
        let pos_facts = w
            .facts()
            .iter()
            .filter(|f| {
                same_event_content(f, &write_event("teacher", "report", false)) && !f.negated
            })
            .count();
        let neg_facts = w
            .facts()
            .iter()
            .filter(|f| {
                same_event_content(f, &write_event("teacher", "report", false)) && f.negated
            })
            .count();
        assert_eq!(pos_facts, 0, "F was retracted: no positive fact remains");
        assert_eq!(neg_facts, 1, "exactly the surviving NOT F remains");
    }

    #[test]
    fn revision_derived_belief_yields_to_later_direct_assertion() {
        // A DERIVED belief F, then a DIRECT assertion of NOT F. Provenance-weighted
        // policy: the direct assertion outranks the derived belief, so the world
        // ends holding NOT F (the direct one), with exactly one revision.
        let mut w = World::new();
        w.assert_derived_event(&write_event("teacher", "report", false)); // derived F
        let positive = Meaning::Event(write_event("teacher", "report", false));
        assert_eq!(w.holds(&positive), Some(true), "derived F holds initially");
        assert_eq!(w.revisions().len(), 0);

        // Direct assertion of NOT F supersedes the derived F.
        w.assert(&Meaning::Event(write_event("teacher", "report", true)));
        assert_eq!(
            w.holds(&positive),
            Some(false),
            "direct NOT F supersedes derived F"
        );
        assert_eq!(w.revisions().len(), 1, "exactly one revision");
        let r = &w.revisions()[0];
        assert_eq!(r.superseded.negated, false, "the derived F was superseded");
        assert_eq!(r.surviving.negated, true, "the direct NOT F survives");
        assert!(
            r.reason.contains("directly asserted"),
            "provenance-weighted reason names the asserted-beats-derived rule: {}",
            r.reason
        );
    }

    #[test]
    fn revision_direct_assertion_beats_later_derived_belief() {
        // The DUAL case: a DIRECT assertion of F, then a DERIVED belief of NOT F.
        // A derivation must NEVER overturn a direct assertion, so the world keeps
        // F; the derived NOT F is rejected (not installed), recorded as superseded.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false))); // direct F
        let positive = Meaning::Event(write_event("teacher", "report", false));
        assert_eq!(w.holds(&positive), Some(true));

        // A derived NOT F arrives — it must NOT overturn the direct F.
        w.assert_derived_event(&write_event("teacher", "report", true));
        assert_eq!(
            w.holds(&positive),
            Some(true),
            "the direct assertion is retained over a contradicting derived belief"
        );
        assert_eq!(
            w.revisions().len(),
            1,
            "the rejection is recorded as a revision"
        );
        let r = &w.revisions()[0];
        assert_eq!(r.surviving.negated, false, "the direct F survives");
        assert_eq!(
            r.superseded.negated, true,
            "the derived NOT F is superseded"
        );
        assert!(
            r.reason.contains("rejected"),
            "reason names the derived-belief-rejected rule: {}",
            r.reason
        );

        // COHERENCE: only the single direct F remains in the live store.
        let neg_facts = w.facts().iter().filter(|f| f.negated).count();
        assert_eq!(
            neg_facts, 0,
            "the rejected derived NOT F was never installed"
        );
    }

    #[test]
    fn revision_consistent_assertions_record_no_revision() {
        // Consistent assertions (re-assert F, then an UNRELATED fact) must record
        // NO revision and NO contradiction — revision only ever fires on a genuine
        // F vs NOT F clash.
        let mut w = World::new();
        w.assert(&Meaning::Event(write_event("teacher", "report", false)));
        w.assert(&Meaning::Event(write_event("teacher", "report", false))); // idempotent
        w.assert(&Meaning::Event(write_event("editor", "memo", false))); // unrelated
        assert_eq!(
            w.revisions().len(),
            0,
            "no revision for consistent assertions"
        );
        assert_eq!(w.contradictions().len(), 0, "no contradiction either");
        // Both live facts are still answerable and TRUE.
        assert_eq!(
            w.holds(&Meaning::Event(write_event("teacher", "report", false))),
            Some(true)
        );
        assert_eq!(
            w.holds(&Meaning::Event(write_event("editor", "memo", false))),
            Some(true)
        );
    }

    #[test]
    fn revision_never_holds_both_polarities_after_oscillation() {
        // SOUNDNESS under oscillation: F, NOT F, F again. After each step the world
        // holds exactly ONE coherent belief, and the FINAL live belief is the most
        // recent (F). Revisions are logged on each genuine flip.
        let mut w = World::new();
        let positive = Meaning::Event(write_event("teacher", "report", false));
        let negative = Meaning::Event(write_event("teacher", "report", true));

        w.assert(&positive); // F
        w.assert(&negative); // NOT F  (revision #1: F -> NOT F)
        assert_eq!(w.holds(&positive), Some(false));
        w.assert(&positive); // F again (revision #2: NOT F -> F)
        assert_eq!(
            w.holds(&positive),
            Some(true),
            "final live belief is the most recent F"
        );

        assert_eq!(
            w.revisions().len(),
            2,
            "one revision per genuine polarity flip"
        );
        // The live store never holds both polarities: exactly one fact of this
        // content remains, and it is positive.
        let matching: Vec<bool> = w
            .facts()
            .iter()
            .filter(|f| same_event_content(f, &write_event("teacher", "report", false)))
            .map(|f| f.negated)
            .collect();
        assert_eq!(
            matching,
            vec![false],
            "exactly one coherent belief (F) survives"
        );
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
