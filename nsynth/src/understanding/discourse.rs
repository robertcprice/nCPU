//! Discourse processing: reads sentences in sequence, resolves pronouns to
//! entities via discourse history (real coreference, not last-subject string
//! substitution), and maintains the world model as it reads.
//!
//! `read` understands a sentence, resolves its pronouns against the mentions
//! seen so far, asserts the resolved meaning to the world, and returns it.
//! `resolve` maps a Pronoun to the most salient compatible Entity: "it" picks
//! the last INANIMATE entity, "they"/"he"/"she" an animate one, preferring
//! recency. Mentions are tracked across turns.

use std::collections::BTreeMap;

use crate::comprehension::Engine;
use crate::understanding::meaning::{Event, Meaning, Term};
use crate::understanding::world_model::World;

/// A running discourse: the world built so far plus coreference state.
#[derive(Clone)]
pub struct Discourse {
    /// the world model accumulated by reading
    pub world: World,
    /// every concrete entity term mentioned so far, in order (oldest -> newest)
    mentions: Vec<Term>,
    /// most recent grammatical subject (for salience / continuity heuristics)
    last_subject: Option<Term>,
    /// animacy memory for every entity head we have resolved/observed:
    /// head -> true if animate (a "person"), false if inanimate (a "thing").
    /// Populated at `read` time (when the Engine's synthesized lexicon is
    /// available) so that `resolve` — which has no Engine — can still honor
    /// number/animacy agreement when picking an antecedent.
    animacy: BTreeMap<String, bool>,
}

impl Discourse {
    pub fn new() -> Self {
        Discourse {
            world: World::new(),
            mentions: Vec::new(),
            last_subject: None,
            animacy: BTreeMap::new(),
        }
    }

    /// Read a sentence into the world model. Clause/VP conjunction ("X and Y") is
    /// split into independent conjuncts (a verb-initial conjunct inherits the
    /// preceding subject), each understood, resolved, and asserted in turn, so
    /// both facts enter the world. Returns the last conjunct's resolved meaning.
    pub fn read(&mut self, engine: &Engine, sentence: &str) -> Meaning {
        let conjuncts = self.split_conjuncts(engine, sentence);
        let mut result = Meaning::Unknown(sentence.to_string());
        for part in &conjuncts {
            result = self.read_one(engine, part);
        }
        result
    }

    /// Split "X and Y" into independent clauses. A conjunct that begins with a
    /// verb (VP-ellipsis: "...writes the report and reads the book") inherits the
    /// subject of the preceding conjunct. No clause-level "and" -> the sentence
    /// unchanged.
    fn split_conjuncts(&self, engine: &Engine, sentence: &str) -> Vec<String> {
        let toks = crate::comprehension::words_of(sentence);
        let Some(pos) = toks.iter().position(|w| w == "and") else {
            return vec![sentence.to_string()];
        };
        if pos == 0 || pos + 1 >= toks.len() {
            return vec![sentence.to_string()];
        }
        let left = &toks[..pos];
        let right = &toks[pos + 1..];
        let right_has_subject = matches!(
            right.first().map(|s| s.as_str()),
            Some("the") | Some("a") | Some("an")
        ) || right
            .first()
            .map(|w| engine.noun_class(w) > 0)
            .unwrap_or(false);

        let mut conjuncts = vec![left.join(" ")];
        if right_has_subject {
            conjuncts.extend(self.split_conjuncts(engine, &right.join(" ")));
        } else if let Some(si) = left.iter().position(|w| engine.noun_class(w) > 0) {
            // VP-ellipsis: re-attach the left subject to the verb-initial conjunct.
            let elided = format!("the {} {}", left[si], right.join(" "));
            conjuncts.extend(self.split_conjuncts(engine, &elided));
        } else {
            conjuncts.push(right.join(" "));
        }
        conjuncts
    }

    /// Understand one (already conjunction-free) clause, resolve its pronouns,
    /// assert it, and return the resolved meaning.
    fn read_one(&mut self, engine: &Engine, sentence: &str) -> Meaning {
        let raw = crate::understanding::semantics::understand(engine, sentence);

        // Record the animacy of every concrete entity this sentence introduces,
        // BEFORE resolution, so antecedents are available to later pronouns and
        // so the new entities themselves are candidates for any pronoun later in
        // the same sentence's processing order.
        self.observe_entities(engine, &raw);

        // Resolve pronouns in the meaning against the discourse history.
        let resolved = self.resolve_meaning(&raw);

        // Register the resolved entities as mentions (recency order) and update
        // the salient subject. Then assert declarative content to the world.
        self.register_mentions(engine, &resolved);
        self.world.assert(&resolved);

        resolved
    }

    /// Resolve a term: a Pronoun maps to the most salient compatible Entity
    /// seen so far; non-pronoun terms are returned unchanged.
    pub fn resolve(&self, term: &Term) -> Term {
        match term {
            Term::Pronoun(p) => self.resolve_pronoun(p).unwrap_or_else(|| term.clone()),
            other => other.clone(),
        }
    }

    // ----------------------------------------------------------------------
    // Internal helpers
    // ----------------------------------------------------------------------

    /// Find the most salient antecedent Entity for a pronoun string. "it" wants
    /// the most recent INANIMATE entity; "they"/"he"/"she" the most recent
    /// ANIMATE entity. Returns `None` if no compatible antecedent has been seen.
    fn resolve_pronoun(&self, pronoun: &str) -> Option<Term> {
        let want_animate = match pronoun.to_lowercase().as_str() {
            "it" => false,
            "they" | "them" | "he" | "him" | "she" | "her" => true,
            // Unknown pronoun: prefer any most-recent entity (no animacy filter).
            _ => return self.most_recent_entity(None),
        };
        self.most_recent_entity(Some(want_animate))
    }

    /// Most recent mentioned Entity matching the desired animacy (if any).
    /// When `want_animate` is `None`, animacy is not constrained. Entities whose
    /// animacy is unknown are treated as compatible (we cannot rule them out).
    fn most_recent_entity(&self, want_animate: Option<bool>) -> Option<Term> {
        for term in self.mentions.iter().rev() {
            if let Term::Entity(head) = term {
                if self.compatible(head, want_animate) {
                    return Some(term.clone());
                }
            }
        }
        // Fallback: the most recent grammatical subject, when compatible. This
        // captures subject-continuity salience even if recency in the flat
        // mention list would otherwise prefer an object antecedent that is not
        // animacy-compatible.
        if let Some(Term::Entity(head)) = self.last_subject.as_ref() {
            if self.compatible(head, want_animate) {
                return self.last_subject.clone();
            }
        }
        None
    }

    /// Is the entity `head` compatible with the desired animacy? Unknown
    /// animacy is treated as compatible (we cannot rule it out).
    fn compatible(&self, head: &str, want_animate: Option<bool>) -> bool {
        match want_animate {
            None => true,
            Some(want) => match self.animacy.get(head) {
                Some(&is_animate) => is_animate == want,
                None => true,
            },
        }
    }

    /// Resolve every pronoun inside a Meaning, leaving non-pronoun terms intact.
    fn resolve_meaning(&self, m: &Meaning) -> Meaning {
        match m {
            Meaning::Event(ev) => Meaning::Event(self.resolve_event(ev)),
            Meaning::IsA { subject, category, negated } => Meaning::IsA {
                subject: self.resolve(subject),
                category: category.clone(),
                negated: *negated,
            },
            Meaning::YesNoQuestion(inner) => {
                Meaning::YesNoQuestion(Box::new(self.resolve_meaning(inner)))
            }
            Meaning::WhQuestion { slot, body } => Meaning::WhQuestion {
                slot: *slot,
                body: self.resolve_event(body),
            },
            // TODO(skeleton): resolution for the new meanings. Quantified bodies
            // have a bound agent (nothing to resolve yet); Or recurses; the
            // HasProperty subject is resolved against discourse history.
            Meaning::Quantified { quant, var_category, body } => Meaning::Quantified {
                quant: *quant,
                var_category: var_category.clone(),
                body: self.resolve_event(body),
            },
            Meaning::Or(disjuncts) => Meaning::Or(
                disjuncts.iter().map(|d| self.resolve_meaning(d)).collect(),
            ),
            Meaning::HasProperty { subject, property, negated } => Meaning::HasProperty {
                subject: self.resolve(subject),
                property: property.clone(),
                negated: *negated,
            },
            // Resolve pronouns inside the new meanings against discourse history.
            Meaning::Comparison { subject, scale, more, than, negated } => Meaning::Comparison {
                subject: self.resolve(subject),
                scale: scale.clone(),
                more: *more,
                than: self.resolve(than),
                negated: *negated,
            },
            Meaning::Attitude { holder, verb, content, negated } => Meaning::Attitude {
                holder: self.resolve(holder),
                verb: verb.clone(),
                content: Box::new(self.resolve_meaning(content)),
                negated: *negated,
            },
            Meaning::Cardinal { at_least, var_category, body } => Meaning::Cardinal {
                at_least: *at_least,
                var_category: var_category.clone(),
                body: self.resolve_event(body),
            },
            Meaning::CountQuestion { var_category, body } => Meaning::CountQuestion {
                var_category: var_category.clone(),
                body: self.resolve_event(body),
            },
            // PLACEHOLDER (skeleton): pass the new grammatical-core meanings
            // through unchanged. The discourse owner adds pronoun resolution for
            // their embedded terms/events (modal/temporal bodies, causal
            // sub-meanings, restricted-term heads, degree subjects, inner Not).
            Meaning::Modal { .. }
            | Meaning::Temporal { .. }
            | Meaning::Causal { .. }
            // PLACEHOLDER (skeleton): a Conditional passes through unchanged, like
            // Causal. Pronoun resolution for its antecedent/consequent sub-meanings
            // lands when conditional anaphora is wired (mirror the Causal owner).
            | Meaning::Conditional { .. }
            | Meaning::DegreeQuestion { .. }
            | Meaning::Not(_) => m.clone(),
            Meaning::Unknown(s) => Meaning::Unknown(s.clone()),
        }
    }

    /// Resolve the agent and patient terms of an event.
    fn resolve_event(&self, ev: &Event) -> Event {
        Event {
            predicate: ev.predicate.clone(),
            agent: ev.agent.as_ref().map(|t| self.resolve(t)),
            patient: ev.patient.as_ref().map(|t| self.resolve(t)),
            recipient: ev.recipient.as_ref().map(|t| self.resolve(t)),
            tense: ev.tense,
            aspect: ev.aspect,
            negated: ev.negated,
        }
    }

    /// Record the animacy of the concrete (non-pronoun) entity heads occurring
    /// in a meaning, using the synthesized lexicon. Pronouns are skipped (their
    /// referent's animacy is recorded when the antecedent itself was read).
    fn observe_entities(&mut self, engine: &Engine, m: &Meaning) {
        match m {
            Meaning::Event(ev) => {
                self.note_term_animacy(engine, ev.agent.as_ref());
                self.note_term_animacy(engine, ev.patient.as_ref());
            }
            Meaning::IsA { subject, .. } => {
                self.note_term_animacy(engine, Some(subject));
            }
            Meaning::YesNoQuestion(inner) => self.observe_entities(engine, inner),
            Meaning::WhQuestion { body, .. } => {
                self.note_term_animacy(engine, body.agent.as_ref());
                self.note_term_animacy(engine, body.patient.as_ref());
            }
            // TODO(skeleton): observe entities in the new meanings. Quantified
            // bodies carry a bound (None/Indefinite) agent and a concrete
            // patient; Or recurses into disjuncts; HasProperty notes its subject.
            Meaning::Quantified { body, .. } => {
                self.note_term_animacy(engine, body.agent.as_ref());
                self.note_term_animacy(engine, body.patient.as_ref());
            }
            Meaning::Or(disjuncts) => {
                for d in disjuncts {
                    self.observe_entities(engine, d);
                }
            }
            Meaning::HasProperty { subject, .. } => {
                self.note_term_animacy(engine, Some(subject));
            }
            // Observe the concrete entities mentioned in the new meanings so they
            // are available as antecedents and get animacy on record.
            Meaning::Comparison { subject, than, .. } => {
                self.note_term_animacy(engine, Some(subject));
                self.note_term_animacy(engine, Some(than));
            }
            Meaning::Attitude { holder, content, .. } => {
                self.note_term_animacy(engine, Some(holder));
                self.observe_entities(engine, content);
            }
            Meaning::Cardinal { body, .. } | Meaning::CountQuestion { body, .. } => {
                self.note_term_animacy(engine, body.agent.as_ref());
                self.note_term_animacy(engine, body.patient.as_ref());
                self.note_term_animacy(engine, body.recipient.as_ref());
            }
            // PLACEHOLDER (skeleton): observe no entities yet for the new forms.
            // The discourse owner notes animacy of the entities inside their
            // bodies/sub-meanings (modal/temporal events, causal clauses, degree
            // subject) when that logic lands.
            Meaning::Modal { .. }
            | Meaning::Temporal { .. }
            | Meaning::Causal { .. }
            // PLACEHOLDER (skeleton): observe no entities yet for a Conditional,
            // like Causal. The owner notes animacy inside antecedent/consequent
            // when conditional entity tracking lands.
            | Meaning::Conditional { .. }
            | Meaning::DegreeQuestion { .. }
            | Meaning::Not(_) => {}
            Meaning::Unknown(_) => {}
        }
    }

    /// Record animacy for a single term if it is a concrete entity with a known
    /// noun class. `noun_class` > 0 means it is a noun; class 1 is animate.
    fn note_term_animacy(&mut self, engine: &Engine, term: Option<&Term>) {
        if let Some(Term::Entity(head)) | Some(Term::Indefinite(head)) = term {
            let class = engine.noun_class(head);
            if class > 0 {
                // class == 1 => animate ("person"); class == 2 => inanimate.
                self.animacy.insert(head.clone(), class == 1);
            }
        }
    }

    /// Register the resolved entity terms of a meaning as mentions (in recency
    /// order: agent before patient, mirroring surface order) and update the
    /// most-salient grammatical subject.
    fn register_mentions(&mut self, engine: &Engine, m: &Meaning) {
        match m {
            Meaning::Event(ev) => {
                if let Some(subj) = ev.agent.as_ref() {
                    self.set_subject(subj);
                }
                self.push_mention(engine, ev.agent.as_ref());
                self.push_mention(engine, ev.patient.as_ref());
            }
            Meaning::IsA { subject, .. } => {
                self.set_subject(subject);
                self.push_mention(engine, Some(subject));
            }
            // Questions do not introduce new discourse referents that later
            // sentences should anaphorically pick up, but resolving a pronoun
            // inside a question still uses the existing history. We do not push
            // their (typically already-known) entities again to avoid skewing
            // recency for subsequent declaratives.
            // TODO(skeleton): register discourse referents from the new meanings.
            // A HasProperty subject IS a concrete referent later anaphora can
            // pick up, so register it; Quantified introduces a bound variable
            // (no stable referent) and Or is a query, so neither pushes mentions.
            Meaning::HasProperty { subject, .. } => {
                self.set_subject(subject);
                self.push_mention(engine, Some(subject));
            }
            // A Comparison's subject (and the standard it is compared against) and
            // an Attitude's holder are concrete referents later anaphora can pick
            // up. Cardinal/CountQuestion introduce a bound variable / are queries,
            // so they push no stable mention.
            Meaning::Comparison { subject, than, .. } => {
                self.set_subject(subject);
                self.push_mention(engine, Some(subject));
                self.push_mention(engine, Some(than));
            }
            Meaning::Attitude { holder, .. } => {
                self.set_subject(holder);
                self.push_mention(engine, Some(holder));
            }
            Meaning::Quantified { .. }
            | Meaning::Or(_)
            | Meaning::Cardinal { .. }
            | Meaning::CountQuestion { .. } => {}
            // PLACEHOLDER (skeleton): register no new discourse referents for the
            // new forms yet. The discourse owner registers concrete referents
            // inside their bodies (modal/temporal/causal/degree subjects) so
            // later anaphora can pick them up.
            Meaning::Modal { .. }
            | Meaning::Temporal { .. }
            | Meaning::Causal { .. }
            // PLACEHOLDER (skeleton): register no new referents for a Conditional
            // yet, like Causal. The owner registers concrete referents inside its
            // antecedent/consequent when conditional anaphora is wired.
            | Meaning::Conditional { .. }
            | Meaning::DegreeQuestion { .. }
            | Meaning::Not(_) => {}
            Meaning::YesNoQuestion(_) | Meaning::WhQuestion { .. } | Meaning::Unknown(_) => {}
        }
    }

    /// Push a concrete Entity term onto the mention history (most recent last).
    /// Indefinite and Pronoun terms are not stable referents and are skipped;
    /// after resolution, surviving Pronouns had no antecedent and resolved
    /// Indefinites remain indefinite, so neither is a reliable antecedent.
    fn push_mention(&mut self, engine: &Engine, term: Option<&Term>) {
        if let Some(Term::Entity(head)) = term {
            // Ensure animacy is on record for this entity (idempotent).
            if !self.animacy.contains_key(head) {
                let class = engine.noun_class(head);
                if class > 0 {
                    self.animacy.insert(head.clone(), class == 1);
                }
            }
            self.mentions.push(Term::Entity(head.clone()));
        }
    }

    /// Update the most-salient grammatical subject if `term` is a concrete
    /// Entity.
    fn set_subject(&mut self, term: &Term) {
        if let Term::Entity(_) = term {
            self.last_subject = Some(term.clone());
        }
    }
}

impl Default for Discourse {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn engine() -> &'static Engine {
        static E: OnceLock<Engine> = OnceLock::new();
        E.get_or_init(Engine::new)
    }

    #[test]
    fn resolve_it_to_inanimate_entity() {
        // After reading about a teacher (animate) and a report (inanimate),
        // "it" must resolve to the report, not the teacher.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let m = d.read(engine(), "It is a thing.");
        let Meaning::IsA { subject, .. } = m else {
            panic!("expected IsA, got {m:?}");
        };
        assert_eq!(subject, Term::Entity("report".to_string()));
    }

    #[test]
    fn resolve_they_to_animate_entity() {
        // "they" must resolve to an animate entity (the author), not the book.
        let mut d = Discourse::new();
        d.read(engine(), "The author reads the book.");
        let m = d.read(engine(), "They write the report.");
        let Meaning::Event(ev) = m else {
            panic!("expected Event, got {m:?}");
        };
        assert_eq!(ev.agent, Some(Term::Entity("author".to_string())));
    }

    #[test]
    fn resolve_pronoun_method_directly() {
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        // "it" -> the inanimate report; "they" -> the animate teacher.
        assert_eq!(
            d.resolve(&Term::Pronoun("it".to_string())),
            Term::Entity("report".to_string())
        );
        assert_eq!(
            d.resolve(&Term::Pronoun("they".to_string())),
            Term::Entity("teacher".to_string())
        );
    }

    #[test]
    fn read_asserts_resolved_meaning_to_world() {
        // After coreference, the world should hold the resolved fact.
        let mut d = Discourse::new();
        d.read(engine(), "The author reads the book.");
        d.read(engine(), "They read the letter.");
        // "the author reads the letter" should now be a fact (they -> author).
        let q =
            crate::understanding::semantics::understand(engine(), "the author reads the letter");
        assert_eq!(d.world.holds(&q), Some(true));
    }
}
