//! Natural-language inference over `Meaning`s: how one meaning relates to
//! another, and what a meaning entails.
//!
//! `relation` classifies a (premise, hypothesis) pair as Entails / Contradicts
//! / Neutral: an event entails itself and its existential generalizations;
//! the negation of the same event contradicts it; unrelated meanings are
//! neutral. `consequences` derives the sound entailments of a meaning (e.g. an
//! Event entails that its agent "did something"; dropping the patient gives
//! "teacher writes [something]"; an IsA person entails not-a-thing). Only valid
//! entailments — keep it sound.
//!
//! Deepened semantics (quantifiers / disjunction / attributes / taxonomy /
//! forward-chaining):
//!   * Taxonomy: a noun subsumes its hypernym chain (teacher ⊑ person ⊑ agent,
//!     report ⊑ document ⊑ thing). `IsA{x, teacher}` therefore entails
//!     `IsA{x, person}` and `IsA{x, agent}`. The two branches (animate vs
//!     inanimate) are mutually exclusive, so a person/agent is NOT a
//!     thing/document and vice versa.
//!   * Quantified `Every` entails its existential `Some` (existential-import
//!     reading the curriculum uses — "every teacher" presupposes a teacher) and
//!     entails `No` is its negation. `Some`/`No` are negations of each other.
//!   * `Or` is entailed BY any of its disjuncts; the disjunction itself entails
//!     nothing in general (we cannot pick a true disjunct).
//!   * `HasProperty` entails itself (reflexive) and flips polarity under
//!     negation.
//!   * `closure` is sound, terminating forward-chaining: given asserted facts it
//!     derives each entity's hypernym `IsA`s so QA can answer derived knowledge.

use crate::comprehension::{AGENTS, PATIENTS};
use crate::understanding::meaning::{Event, Meaning, Quantifier, Term};

/// The set of FACTIVE attitude verbs: "know that P" entails P. Non-factive
/// attitudes (believe/think/say) carry no commitment to the truth of their
/// complement, so they are deliberately absent here. This is the single source
/// of truth for factivity in inference — keeping it sound means NEVER deriving
/// the content of a non-factive attitude.
fn is_factive(verb: &str) -> bool {
    matches!(verb, "know" | "knows" | "knew")
}

/// How a hypothesis relates to a premise under natural-language inference.
pub enum Relation {
    Entails,
    Contradicts,
    Neutral,
}

/// Classify the inferential relation from `premise` to `hypothesis`.
///
/// Soundness contract:
/// - `Entails`: the truth of `premise` guarantees the truth of `hypothesis`.
///   This holds when the hypothesis is the premise itself, or any meaning in
///   its sound consequence set (existential generalizations: drop the patient,
///   generalize a definite argument to an indefinite "something").
/// - `Contradicts`: the truth of `premise` guarantees the FALSITY of
///   `hypothesis`. This holds when the hypothesis is the polarity-flip of the
///   premise (or of one of its consequences) — e.g. "X writes Y" contradicts
///   "X does not write Y".
/// - `Neutral`: neither holds.
pub fn relation(premise: &Meaning, hypothesis: &Meaning) -> Relation {
    // Questions and Unknowns carry no assertoric content, so they neither
    // entail nor contradict anything.
    if is_non_assertoric(premise) || is_non_assertoric(hypothesis) {
        return Relation::Neutral;
    }

    // Entailment: hypothesis is the premise itself or a sound consequence of it.
    if entails(premise, hypothesis) {
        return Relation::Entails;
    }

    // Disjunction in the hypothesis: a premise entails "A or B" iff it entails
    // some disjunct. This is the ONLY sound direction for `Or` on the
    // hypothesis side (the premise need not pin down which disjunct, only that
    // at least one is guaranteed).
    if let Meaning::Or(disjuncts) = hypothesis {
        if disjuncts
            .iter()
            .any(|d| matches!(relation(premise, d), Relation::Entails))
        {
            return Relation::Entails;
        }
    }

    // Contradiction: the premise (or one of its consequences) entails the
    // polarity-flip of the hypothesis. Equivalently, flip the hypothesis'
    // polarity and check whether the premise entails *that*. If "X writes Y"
    // entails "X writes Y", then it contradicts "X does NOT write Y".
    if let Some(flipped) = polarity_flip(hypothesis) {
        if entails(premise, &flipped) {
            return Relation::Contradicts;
        }
    }

    Relation::Neutral
}

/// Derive the sound entailed meanings of `m` (excluding `m` itself, though the
/// reflexive case is handled by `entails`). Every returned meaning is true
/// whenever `m` is true.
pub fn consequences(m: &Meaning) -> Vec<Meaning> {
    let mut out: Vec<Meaning> = Vec::new();

    match m {
        Meaning::Event(ev) => {
            // Generalizations are only sound for AFFIRMATIVE (non-negated)
            // events. "The teacher writes the report" entails "the teacher
            // writes something", but "the teacher does NOT write the report"
            // does NOT entail "the teacher does not write anything". Under
            // negation, dropping/weakening an argument is unsound, so we emit
            // no generalizations there.
            if !ev.negated {
                // 1) Drop the patient: "teacher writes the report"
                //    entails "teacher writes [something]".
                if ev.patient.is_some() {
                    let mut e = ev.clone();
                    e.patient = None;
                    push_unique(&mut out, Meaning::Event(e));
                }

                // 2) Generalize the agent to an existential indefinite:
                //    "the teacher writes the report" entails
                //    "a teacher writes the report" (someone of that kind).
                if let Some(g) = generalize_term(&ev.agent) {
                    let mut e = ev.clone();
                    e.agent = Some(g);
                    push_unique(&mut out, Meaning::Event(e));
                }

                // 3) Generalize the patient to an existential indefinite.
                if let Some(g) = generalize_term(&ev.patient) {
                    let mut e = ev.clone();
                    e.patient = Some(g);
                    push_unique(&mut out, Meaning::Event(e));
                }

                // 4) Combined: drop the patient AND generalize the agent —
                //    "a teacher writes [something]".
                if ev.patient.is_some() {
                    if let Some(g) = generalize_term(&ev.agent) {
                        let mut e = ev.clone();
                        e.patient = None;
                        e.agent = Some(g);
                        push_unique(&mut out, Meaning::Event(e));
                    }
                }
            }
        }
        Meaning::IsA {
            subject,
            category,
            negated,
        } => {
            if !negated {
                // TAXONOMY: a category subsumes its hypernym chain, so
                // "X is a teacher" entails "X is a person" and "X is an agent"
                // (person ⊑ agent). Emit every strict hypernym of the asserted
                // category as a positive IsA. This is the engine that lets QA
                // answer "is the teacher an agent?" -> Yes from "teacher".
                for hyper in hypernyms(category) {
                    push_unique(
                        &mut out,
                        Meaning::IsA {
                            subject: subject.clone(),
                            category: hyper.to_string(),
                            negated: false,
                        },
                    );
                }
                // MUTUAL EXCLUSION across the animate/inanimate branches: an
                // entity in the {person, agent, <animate noun>} branch is NOT in
                // the {thing, document, <inanimate noun>} branch and vice versa.
                // Emit the negative IsA for each category in the opposite branch
                // (only the stable hypernym labels — we do not enumerate every
                // noun, just the branch roots that QA actually queries).
                for opp in opposite_branch_labels(category) {
                    push_unique(
                        &mut out,
                        Meaning::IsA {
                            subject: subject.clone(),
                            category: opp.to_string(),
                            negated: true,
                        },
                    );
                }
            }
        }
        Meaning::Quantified {
            quant,
            var_category,
            body,
        } => {
            match quant {
                // UNIVERSAL: "every teacher writes a report" entails the
                // existential "some teacher writes a report". Sound under the
                // existential-import reading the curriculum uses (a universal
                // claim about a named kind presupposes the kind is non-empty;
                // empty-domain truth is handled separately by the world model's
                // vacuous-truth rule, which does not flow through here).
                Quantifier::Every => {
                    push_unique(
                        &mut out,
                        Meaning::Quantified {
                            quant: Quantifier::Some,
                            var_category: var_category.clone(),
                            body: body.clone(),
                        },
                    );
                }
                // EXISTENTIAL / NEGATIVE carry no further generalization that is
                // sound without the entity domain; their truth is evaluated by
                // the world model. (We deliberately do NOT derive `Some` from
                // `No`, nor any specific-entity body, here.)
                Quantifier::Some | Quantifier::No => {}
            }
        }
        // `Or` entails nothing in general: knowing "A or B" does not let us pick
        // a true disjunct, so there is no sound consequence to emit. (The
        // entailed-BY-a-disjunct direction is handled in `relation`.)
        Meaning::Or(_) => {}
        // `HasProperty` entails itself; reflexivity is covered by `entails`'s
        // equality check, so there is no extra consequence to derive.
        Meaning::HasProperty { .. } => {}
        // COMPARISON. A single comparison meaning does NOT, on its own, license
        // transitive inference — that needs a SECOND fact (A>B together with
        // B>C), which is a multi-fact derivation handled by the world model's
        // transitive closure, not by the consequences of one meaning. What IS
        // sound from one affirmative comparison is the CONVERSE PHRASING under the
        // antisymmetry of a strict order on a single scale: "A is longer than B"
        // (more on `length`) is logically equivalent to "B is shorter than A"
        // (less on `length`). So we emit the polarity-swapped, argument-swapped
        // restatement. We do NOT emit `B more A` (that is the FALSE symmetric
        // claim) and we emit nothing under negation (the negation of a strict
        // comparison is the non-strict converse-or-equal, which we cannot pin
        // down as a single comparison meaning).
        Meaning::Comparison {
            subject,
            scale,
            more,
            than,
            negated,
        } => {
            if !negated {
                // (1) CONVERSE restatement: "A is longer than B" <=> "B is
                //     shorter than A" (swap args, flip `more`). Equivalent, sound.
                push_unique(
                    &mut out,
                    Meaning::Comparison {
                        subject: than.clone(),
                        scale: scale.clone(),
                        more: !more,
                        than: subject.clone(),
                        negated: false,
                    },
                );
                // (2) INCOMPATIBILITY 1: a strict order is asymmetric, so
                //     "A more B" entails "NOT (B more A)" — the reverse-direction
                //     comparison is false. Emitting it as a negated consequence
                //     lets `relation` report Contradicts against "B more A".
                push_unique(
                    &mut out,
                    Meaning::Comparison {
                        subject: than.clone(),
                        scale: scale.clone(),
                        more: *more,
                        than: subject.clone(),
                        negated: true,
                    },
                );
                // (3) INCOMPATIBILITY 2: the two directions on the same pair are
                //     mutually exclusive, so "A more B" entails "NOT (A less B)"
                //     (same args, opposite `more`, negated). This yields the
                //     "longer-than" vs "shorter-than" contradiction.
                push_unique(
                    &mut out,
                    Meaning::Comparison {
                        subject: subject.clone(),
                        scale: scale.clone(),
                        more: !more,
                        than: than.clone(),
                        negated: true,
                    },
                );
            }
        }
        // ATTITUDE. FACTIVITY is the only sound single-meaning entailment here:
        // an affirmative "X knows that P" entails its content P (and, by
        // composition, every sound consequence OF P). Non-factive
        // believe/think/say entail NOTHING about P — that is the load-bearing
        // soundness constraint. A NEGATED factive ("X does not know that P")
        // entails nothing about P either: failing to know P leaves P's truth
        // open. (Reflexivity of the attitude itself is covered by `entails`.)
        Meaning::Attitude {
            verb,
            content,
            negated,
            ..
        } => {
            if !negated && is_factive(verb) {
                // The content is true...
                push_unique(&mut out, (**content).clone());
                // ...and so is everything the content soundly entails. This lets
                // "X knows that the teacher writes the report" answer
                // "does the teacher write something?" -> Yes.
                for c in consequences(content) {
                    push_unique(&mut out, c);
                }
            }
        }
        // CARDINAL at-least monotonicity. "At least N teachers write a report"
        // entails "at least M teachers write a report" for every 1 <= M < N
        // (a stronger count entails every weaker count), and in particular
        // entails the existential "some teacher writes a report" (the M=1 floor,
        // which we also surface as a `Some` Quantified so existential queries
        // phrased that way are answered). We emit the weaker cardinals and the
        // existential generalization; we never emit a STRONGER count (that would
        // be unsound) nor a universal (at-least says nothing about all members).
        Meaning::Cardinal {
            at_least,
            var_category,
            body,
        } => {
            // Weaker at-least counts: M = at_least-1 down to 1.
            let mut m = at_least.saturating_sub(1);
            while m >= 1 {
                push_unique(
                    &mut out,
                    Meaning::Cardinal {
                        at_least: m,
                        var_category: var_category.clone(),
                        body: body.clone(),
                    },
                );
                m -= 1;
            }
            // Existential generalization: at-least-1 (implied by any N>=1) means
            // "some <category> <body>".
            if *at_least >= 1 {
                push_unique(
                    &mut out,
                    Meaning::Quantified {
                        quant: Quantifier::Some,
                        var_category: var_category.clone(),
                        body: body.clone(),
                    },
                );
            }
        }
        // A counting question asserts nothing, so it entails nothing.
        Meaning::CountQuestion { .. } => {}
        // Questions / Unknowns assert nothing, so they entail nothing.
        Meaning::YesNoQuestion(_) | Meaning::WhQuestion { .. } | Meaning::Unknown(_) => {}
    }

    out
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

/// True for meanings that carry no truth-evaluable assertion.
fn is_non_assertoric(m: &Meaning) -> bool {
    matches!(
        m,
        Meaning::YesNoQuestion(_)
            | Meaning::WhQuestion { .. }
            // A counting question is interrogative, not assertoric.
            | Meaning::CountQuestion { .. }
            | Meaning::Unknown(_)
    )
}

/// Does `premise` entail `hypothesis`? Reflexive (a meaning entails itself)
/// plus the sound consequence set.
fn entails(premise: &Meaning, hypothesis: &Meaning) -> bool {
    if meaning_eq(premise, hypothesis) {
        return true;
    }
    consequences(premise)
        .iter()
        .any(|c| meaning_eq(c, hypothesis))
}

/// Structural equality of two meanings, treating an `Entity(x)` and an
/// `Indefinite(x)` argument as compatible ONLY in the directions that are
/// sound; here we use exact `Term` equality so that consequences (which carry
/// the generalized `Indefinite` terms explicitly) drive all matching. This
/// keeps `entails` sound: the hypothesis must match a literally-derived
/// consequence, not be loosely "close".
fn meaning_eq(a: &Meaning, b: &Meaning) -> bool {
    a == b
}

/// Flip the polarity of an assertoric meaning, if it has one. Returns the
/// negation of `m` (same content, opposite `negated`). `None` for
/// non-assertoric meanings.
fn polarity_flip(m: &Meaning) -> Option<Meaning> {
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
        // `polarity_flip` here returns a sound CONTRADICTORY (a meaning that the
        // input guarantees to be false), which is exactly what the `relation`
        // contradiction check consumes. For quantifiers over the SAME category
        // and body, `Every` and `No` are mutually exclusive under the curriculum's
        // existential-import reading: "every teacher writes" guarantees "no
        // teacher writes" is FALSE, and vice versa. So we map `Every` -> `No` and
        // `No` -> `Every`. (These are contradictory, not full logical negations —
        // the strict negation of "every X P" is "some X not-P", which we do not
        // model as a single Meaning.) `Some` has no single-statement
        // contradictory in our vocabulary, so it does not flip.
        Meaning::Quantified {
            quant,
            var_category,
            body,
        } => match quant {
            Quantifier::Every => Some(Meaning::Quantified {
                quant: Quantifier::No,
                var_category: var_category.clone(),
                body: body.clone(),
            }),
            Quantifier::No => Some(Meaning::Quantified {
                quant: Quantifier::Every,
                var_category: var_category.clone(),
                body: body.clone(),
            }),
            Quantifier::Some => None,
        },
        Meaning::HasProperty {
            subject,
            property,
            negated,
        } => Some(Meaning::HasProperty {
            subject: subject.clone(),
            property: property.clone(),
            negated: !negated,
        }),
        // A disjunction has no single-meaning polarity flip (its negation is a
        // conjunction of negated disjuncts, not representable as one `Meaning`),
        // so we report no flip and let it read as Neutral under contradiction.
        Meaning::Or(_) => None,
        // COMPARISON. The contradictory of an affirmative strict comparison is
        // its own negation (same subject/scale/than/more, flipped `negated`):
        // "A is longer than B" guarantees "A is NOT longer than B" is false, and
        // vice versa. We flip ONLY the `negated` flag and keep `more`/arguments
        // fixed. Crucially we do NOT map `A more B` to `B more A` (that symmetric
        // claim is also false under a strict order, but treating it as the
        // polarity flip would let `relation` over-fire on unrelated phrasings; the
        // genuine A-vs-B-on-B-vs-A contradiction is reached instead through the
        // CONVERSE consequence `B less A` plus this same-shape flip, all soundly).
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
        // ATTITUDE. The contradictory is the same attitude with flipped polarity:
        // "X knows that P" contradicts "X does not know that P". This is sound for
        // BOTH factive and non-factive verbs because it is a claim about the
        // ATTITUDE itself (whether the holder holds it), independent of P's truth.
        // We keep `verb` and `content` fixed — we do NOT flip the embedded P, and
        // we do NOT cross factivity (knowing P does not contradict believing P).
        Meaning::Attitude {
            holder,
            verb,
            content,
            negated,
        } => Some(Meaning::Attitude {
            holder: holder.clone(),
            verb: verb.clone(),
            content: content.clone(),
            negated: !negated,
        }),
        // CARDINAL has no single-meaning contradictory in our vocabulary. The
        // negation of "at least N <cat> <body>" is "at most N-1 <cat> <body>",
        // which we cannot express as another `Cardinal` (which is always
        // at-LEAST). Reporting a same-shape flip would be UNSOUND (two at-least
        // claims never contradict — the larger entails the smaller). So we report
        // no flip and let cardinal pairs read as Neutral under contradiction.
        Meaning::Cardinal { .. } => None,
        // A counting question is non-assertoric — no polarity to flip.
        Meaning::CountQuestion { .. } => None,
        Meaning::YesNoQuestion(_) | Meaning::WhQuestion { .. } | Meaning::Unknown(_) => None,
    }
}

/// Existentially generalize a definite argument: `Entity(x)` -> `Indefinite(x)`
/// ("the teacher" -> "a teacher" = "some teacher"), which is a sound weakening.
/// An already-indefinite term, a pronoun, or an absent argument yields `None`
/// (nothing new to derive).
fn generalize_term(t: &Option<Term>) -> Option<Term> {
    match t {
        Some(Term::Entity(s)) => Some(Term::Indefinite(s.clone())),
        // Already existential, or an unresolved pronoun (no sound
        // generalization), or no argument at all.
        Some(Term::Indefinite(_)) | Some(Term::Pronoun(_)) | None => None,
    }
}

// ----------------------------------------------------------------------------
// Taxonomy / hypernymy
// ----------------------------------------------------------------------------
//
// Two disjoint subsumption chains, rooted in the synthesized lexicon:
//   * every AGENT noun (animate)   ⊑ person   ⊑ agent
//   * every PATIENT noun (inanimate) ⊑ document ⊑ thing
// Plus the intermediate labels subsume upward: person ⊑ agent, document ⊑ thing.
// The two branches are mutually exclusive: nothing is both an agent and a thing.
//
// NOTE FOR THE INTEGRATOR: the parallel `world_model.rs` is specified to provide
// its own `hypernyms(noun)` table for `holds`. This module keeps a SELF-CONTAINED
// copy (so `inference.rs` is sound standalone and does not depend on the sibling
// module's exact signature). The two tables MUST agree; if you unify them, expose
// one canonical `hypernyms`/branch helper and have both call it. The labels here
// — person, agent, document, thing — are the stable query targets QA uses.

/// The branch a category/noun belongs to: `true` = animate (person/agent
/// branch), `false` = inanimate (thing/document branch). `None` if the label is
/// not part of either chain.
fn branch_is_animate(label: &str) -> Option<bool> {
    match label {
        "person" | "agent" => Some(true),
        "thing" | "document" => Some(false),
        _ => {
            if AGENTS.iter().any(|w| *w == label) {
                Some(true)
            } else if PATIENTS.iter().any(|w| *w == label) {
                Some(false)
            } else {
                None
            }
        }
    }
}

/// The strict hypernyms (proper super-categories) of a category or noun, in
/// ascending generality. A bottom noun yields its full chain; an intermediate
/// label yields only what is strictly above it; a top label yields nothing.
///
///   teacher  -> ["person", "agent"]
///   report   -> ["document", "thing"]
///   person   -> ["agent"]
///   document -> ["thing"]
///   agent / thing -> []
pub fn hypernyms(category: &str) -> Vec<&'static str> {
    match category {
        // Top-of-chain labels: nothing strictly above them.
        "agent" | "thing" => Vec::new(),
        // Intermediate labels.
        "person" => vec!["agent"],
        "document" => vec!["thing"],
        // Bottom nouns: walk the whole branch above them.
        other => match branch_is_animate(other) {
            Some(true) => vec!["person", "agent"],
            Some(false) => vec!["document", "thing"],
            None => Vec::new(),
        },
    }
}

/// The stable category labels of the OPPOSITE branch from `category`, used to
/// emit mutual-exclusion negatives ("X is a person" ⊨ "X is NOT a thing/
/// document"). Only the branch-root labels are returned — we do not enumerate
/// every individual noun, since QA queries the labels.
fn opposite_branch_labels(category: &str) -> Vec<&'static str> {
    match branch_is_animate(category) {
        // An animate thing is NOT in the inanimate branch.
        Some(true) => vec!["thing", "document"],
        // An inanimate thing is NOT in the animate branch.
        Some(false) => vec!["person", "agent"],
        None => Vec::new(),
    }
}

// ----------------------------------------------------------------------------
// Forward-chaining closure
// ----------------------------------------------------------------------------

/// Sound, terminating forward-chaining over a set of asserted event facts: for
/// every concrete entity that appears as an argument, derive the `IsA` facts
/// implied by the taxonomy (its hypernym chain) so QA can consult DERIVED
/// knowledge ("is the teacher an agent?" -> Yes, even though only the verb fact
/// was asserted).
///
/// Termination: the derivation set is bounded — each entity contributes at most
/// (1 base IsA + |hypernyms| ≤ 2) `IsA` meanings, and `hypernyms` is acyclic by
/// construction, so a single pass reaches fixpoint. We do not chain over derived
/// IsAs again because `hypernyms(label)` already returns the FULL upward chain
/// from any node, making one pass complete.
///
/// Soundness: every emitted meaning is a positive `IsA{entity, C}` where `C` is
/// the entity's own noun or a genuine super-category of it — true whenever the
/// entity exists in the world. We emit no negatives and no event facts here, so
/// nothing unsound can leak in.
pub fn closure(facts: &[Event]) -> Vec<Meaning> {
    let mut out: Vec<Meaning> = Vec::new();
    let mut seen_entities: Vec<String> = Vec::new();

    for ev in facts {
        for term in [ev.agent.as_ref(), ev.patient.as_ref()].into_iter().flatten() {
            // Only concrete entities (definite/indefinite) name a real referent
            // we can type. Unresolved pronouns carry no noun to look up.
            let head = match term {
                Term::Entity(s) | Term::Indefinite(s) => s.clone(),
                Term::Pronoun(_) => continue,
            };
            if seen_entities.iter().any(|e| e == &head) {
                continue;
            }
            seen_entities.push(head.clone());

            // Only nouns that sit in a known taxonomy branch get typed; an
            // unknown head yields no sound IsA.
            if branch_is_animate(&head).is_none() {
                continue;
            }
            let subject = Term::Entity(head.clone());
            // The entity is-a its own noun ("the teacher is a teacher").
            push_unique(
                &mut out,
                Meaning::IsA {
                    subject: subject.clone(),
                    category: head.clone(),
                    negated: false,
                },
            );
            // ... and is-a each of its hypernyms.
            for hyper in hypernyms(&head) {
                push_unique(
                    &mut out,
                    Meaning::IsA {
                        subject: subject.clone(),
                        category: hyper.to_string(),
                        negated: false,
                    },
                );
            }
        }
    }

    out
}

/// Push `m` into `out` only if structurally new.
fn push_unique(out: &mut Vec<Meaning>, m: Meaning) {
    if !out.iter().any(|x| x == &m) {
        out.push(m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::understanding::meaning::{Event, Tense};

    fn ev(agent: Term, patient: Option<Term>, negated: bool) -> Meaning {
        Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(agent),
            patient,
            recipient: None,
            tense: Tense::Present,
            negated,
        })
    }

    fn entity(s: &str) -> Term {
        Term::Entity(s.to_string())
    }

    #[test]
    fn reflexive_entailment() {
        let p = ev(entity("teacher"), Some(entity("report")), false);
        assert!(matches!(relation(&p, &p), Relation::Entails));
    }

    #[test]
    fn existential_generalization_entails() {
        // "the teacher writes the report" entails "a teacher writes the report".
        let p = ev(entity("teacher"), Some(entity("report")), false);
        let h = ev(Term::Indefinite("teacher".to_string()), Some(entity("report")), false);
        assert!(matches!(relation(&p, &h), Relation::Entails));
    }

    #[test]
    fn drop_patient_entails() {
        // "the teacher writes the report" entails "the teacher writes [something]".
        let p = ev(entity("teacher"), Some(entity("report")), false);
        let h = ev(entity("teacher"), None, false);
        assert!(matches!(relation(&p, &h), Relation::Entails));
    }

    #[test]
    fn negation_contradicts() {
        let p = ev(entity("teacher"), Some(entity("report")), false);
        let h = ev(entity("teacher"), Some(entity("report")), true);
        assert!(matches!(relation(&p, &h), Relation::Contradicts));
    }

    #[test]
    fn unrelated_is_neutral() {
        let p = ev(entity("teacher"), Some(entity("report")), false);
        let h = ev(entity("author"), Some(entity("book")), false);
        assert!(matches!(relation(&p, &h), Relation::Neutral));
    }

    #[test]
    fn isa_person_contradicts_thing() {
        let p = Meaning::IsA {
            subject: entity("teacher"),
            category: "person".to_string(),
            negated: false,
        };
        let h = Meaning::IsA {
            subject: entity("teacher"),
            category: "thing".to_string(),
            negated: false,
        };
        assert!(matches!(relation(&p, &h), Relation::Contradicts));
    }

    #[test]
    fn consequences_of_negated_event_are_empty() {
        // Generalizing under negation is unsound; no consequences should be emitted.
        let p = Event {
            predicate: "write".to_string(),
            agent: Some(entity("teacher")),
            patient: Some(entity("report")),
            recipient: None,
            tense: Tense::Present,
            negated: true,
        };
        assert!(consequences(&Meaning::Event(p)).is_empty());
    }

    // ------------------------------------------------------------------
    // Deepened semantics: taxonomy / quantifiers / disjunction / attributes
    // ------------------------------------------------------------------
    // (`Quantifier` is already in scope via `use super::*`.)

    fn isa(subject: Term, category: &str, negated: bool) -> Meaning {
        Meaning::IsA {
            subject,
            category: category.to_string(),
            negated,
        }
    }

    fn quant(q: Quantifier) -> Meaning {
        Meaning::Quantified {
            quant: q,
            var_category: "teacher".to_string(),
            body: Event {
                predicate: "write".to_string(),
                agent: None,
                patient: Some(Term::Indefinite("report".to_string())),
                recipient: None,
                tense: Tense::Present,
                negated: false,
            },
        }
    }

    #[test]
    fn taxonomy_hypernym_chain() {
        // teacher ⊑ person ⊑ agent ; report ⊑ document ⊑ thing.
        assert_eq!(hypernyms("teacher"), vec!["person", "agent"]);
        assert_eq!(hypernyms("report"), vec!["document", "thing"]);
        assert_eq!(hypernyms("person"), vec!["agent"]);
        assert_eq!(hypernyms("document"), vec!["thing"]);
        assert!(hypernyms("agent").is_empty());
        assert!(hypernyms("thing").is_empty());
        assert!(hypernyms("banana").is_empty()); // unknown noun: no chain
    }

    #[test]
    fn isa_teacher_entails_agent_via_taxonomy() {
        // "the teacher is a teacher" entails "the teacher is an agent".
        let p = isa(entity("teacher"), "teacher", false);
        let h = isa(entity("teacher"), "agent", false);
        assert!(matches!(relation(&p, &h), Relation::Entails));
        // ... and "the teacher is a person".
        let h2 = isa(entity("teacher"), "person", false);
        assert!(matches!(relation(&p, &h2), Relation::Entails));
    }

    #[test]
    fn isa_teacher_contradicts_thing_branch() {
        // An agent (animate branch) is NOT a thing/document (inanimate branch).
        let p = isa(entity("teacher"), "teacher", false);
        assert!(matches!(
            relation(&p, &isa(entity("teacher"), "thing", false)),
            Relation::Contradicts
        ));
        assert!(matches!(
            relation(&p, &isa(entity("teacher"), "document", false)),
            Relation::Contradicts
        ));
    }

    #[test]
    fn isa_taxonomy_is_directional_not_symmetric() {
        // "teacher" entails "agent", but a bare "agent" does NOT entail "teacher"
        // (an agent need not be a teacher). Soundness: no false specialization.
        let agent_claim = isa(entity("x"), "agent", false);
        let teacher_claim = isa(entity("x"), "teacher", false);
        assert!(matches!(
            relation(&agent_claim, &teacher_claim),
            Relation::Neutral
        ));
    }

    #[test]
    fn every_entails_some_not_no() {
        // "every teacher writes a report" entails "some teacher writes a report".
        assert!(matches!(
            relation(&quant(Quantifier::Every), &quant(Quantifier::Some)),
            Relation::Entails
        ));
        // ... and contradicts "no teacher writes a report".
        assert!(matches!(
            relation(&quant(Quantifier::Every), &quant(Quantifier::No)),
            Relation::Contradicts
        ));
        // Soundness guard: "some" does NOT entail "every".
        assert!(matches!(
            relation(&quant(Quantifier::Some), &quant(Quantifier::Every)),
            Relation::Neutral
        ));
    }

    #[test]
    fn disjunct_entails_disjunction() {
        // A premise entails "A or B" when it entails a disjunct.
        let a = ev(entity("teacher"), Some(entity("report")), false);
        let b = ev(entity("author"), Some(entity("book")), false);
        let disjunction = Meaning::Or(vec![a.clone(), b.clone()]);
        // Asserting A entails "A or B".
        assert!(matches!(relation(&a, &disjunction), Relation::Entails));
        // An unrelated premise does not entail the disjunction.
        let unrelated = ev(entity("doctor"), Some(entity("letter")), false);
        assert!(matches!(
            relation(&unrelated, &disjunction),
            Relation::Neutral
        ));
    }

    #[test]
    fn has_property_reflexive_and_polarity() {
        let careful = Meaning::HasProperty {
            subject: entity("teacher"),
            property: "careful".to_string(),
            negated: false,
        };
        // Reflexive entailment.
        assert!(matches!(relation(&careful, &careful), Relation::Entails));
        // "careful" contradicts "not careful".
        let not_careful = Meaning::HasProperty {
            subject: entity("teacher"),
            property: "careful".to_string(),
            negated: true,
        };
        assert!(matches!(
            relation(&careful, &not_careful),
            Relation::Contradicts
        ));
    }

    #[test]
    fn closure_derives_hypernym_isas() {
        // From "the teacher writes the report", forward-chaining derives that the
        // teacher is a person/agent and the report is a document/thing.
        let fact = Event {
            predicate: "write".to_string(),
            agent: Some(entity("teacher")),
            patient: Some(entity("report")),
            recipient: None,
            tense: Tense::Present,
            negated: false,
        };
        let derived = closure(&[fact]);
        let has = |s: Term, c: &str| derived.contains(&isa(s, c, false));
        assert!(has(entity("teacher"), "teacher"));
        assert!(has(entity("teacher"), "person"));
        assert!(has(entity("teacher"), "agent"));
        assert!(has(entity("report"), "document"));
        assert!(has(entity("report"), "thing"));
        // Soundness: closure emits no negatives and no cross-branch claims.
        assert!(!derived.iter().any(|m| matches!(
            m,
            Meaning::IsA { negated: true, .. }
        )));
        // Termination/idempotence: a duplicated fact yields the same derived set.
        let fact2 = Event {
            predicate: "write".to_string(),
            agent: Some(entity("teacher")),
            patient: Some(entity("report")),
            recipient: None,
            tense: Tense::Present,
            negated: false,
        };
        let derived2 = closure(&[fact2.clone(), fact2]);
        assert_eq!(derived.len(), derived2.len());
    }

    // ------------------------------------------------------------------
    // New domains: comparison / attitude / cardinal
    // ------------------------------------------------------------------

    fn comp(subject: &str, more: bool, than: &str, negated: bool) -> Meaning {
        Meaning::Comparison {
            subject: entity(subject),
            scale: "length".to_string(),
            more,
            than: entity(than),
            negated,
        }
    }

    fn cardinal(n: usize, cat: &str) -> Meaning {
        Meaning::Cardinal {
            at_least: n,
            var_category: cat.to_string(),
            body: Event {
                predicate: "write".to_string(),
                agent: None,
                patient: Some(Term::Indefinite("report".to_string())),
                recipient: None,
                tense: Tense::Present,
                negated: false,
            },
        }
    }

    #[test]
    fn comparison_reflexive_and_converse() {
        // "report longer than book" entails itself (reflexive)...
        let p = comp("report", true, "book", false);
        assert!(matches!(relation(&p, &p), Relation::Entails));
        // ...and entails the converse "book shorter than report".
        let converse = comp("book", false, "report", false);
        assert!(matches!(relation(&p, &converse), Relation::Entails));
    }

    #[test]
    fn comparison_asymmetry_is_not_entailment() {
        // SOUNDNESS: "A longer than B" must NOT entail "B longer than A".
        let p = comp("report", true, "book", false);
        let symmetric = comp("book", true, "report", false);
        assert!(!matches!(relation(&p, &symmetric), Relation::Entails));
    }

    #[test]
    fn comparison_contradictions() {
        let p = comp("report", true, "book", false);
        // "A longer than B" contradicts "B longer than A" (asymmetry).
        assert!(matches!(
            relation(&p, &comp("book", true, "report", false)),
            Relation::Contradicts
        ));
        // ...contradicts "A shorter than B" (opposite direction, same pair).
        assert!(matches!(
            relation(&p, &comp("report", false, "book", false)),
            Relation::Contradicts
        ));
        // ...contradicts its own negation "A NOT longer than B".
        assert!(matches!(
            relation(&p, &comp("report", true, "book", true)),
            Relation::Contradicts
        ));
    }

    #[test]
    fn comparison_unrelated_pair_is_neutral() {
        // A comparison on an unrelated pair neither entails nor contradicts.
        let p = comp("report", true, "book", false);
        assert!(matches!(
            relation(&p, &comp("letter", true, "memo", false)),
            Relation::Neutral
        ));
    }

    #[test]
    fn factive_know_entails_content_nonfactive_does_not() {
        let content = Meaning::HasProperty {
            subject: entity("report"),
            property: "long".to_string(),
            negated: false,
        };
        // FACTIVE: "the teacher knows that the report is long" entails "the
        // report is long".
        let know = Meaning::Attitude {
            holder: entity("teacher"),
            verb: "know".to_string(),
            content: Box::new(content.clone()),
            negated: false,
        };
        assert!(matches!(relation(&know, &content), Relation::Entails));

        // NON-FACTIVE: "the teacher believes that the report is long" does NOT
        // entail the report is long.
        let believe = Meaning::Attitude {
            holder: entity("teacher"),
            verb: "believe".to_string(),
            content: Box::new(content.clone()),
            negated: false,
        };
        assert!(!matches!(relation(&believe, &content), Relation::Entails));
        // ...and likewise think/say.
        for v in ["think", "say"] {
            let att = Meaning::Attitude {
                holder: entity("teacher"),
                verb: v.to_string(),
                content: Box::new(content.clone()),
                negated: false,
            };
            assert!(!matches!(relation(&att, &content), Relation::Entails));
        }
    }

    #[test]
    fn negated_factive_does_not_entail_content() {
        // "the teacher does NOT know that the report is long" leaves the report's
        // length open — it must NOT entail the content.
        let content = Meaning::HasProperty {
            subject: entity("report"),
            property: "long".to_string(),
            negated: false,
        };
        let not_know = Meaning::Attitude {
            holder: entity("teacher"),
            verb: "know".to_string(),
            content: Box::new(content.clone()),
            negated: true,
        };
        assert!(!matches!(relation(&not_know, &content), Relation::Entails));
    }

    #[test]
    fn factive_know_entails_content_consequences() {
        // "X knows that the teacher writes the report" entails (via factivity +
        // event generalization) "the teacher writes [something]".
        let event = ev(entity("teacher"), Some(entity("report")), false);
        let know = Meaning::Attitude {
            holder: entity("student"),
            verb: "knows".to_string(),
            content: Box::new(event),
            negated: false,
        };
        let dropped_patient = ev(entity("teacher"), None, false);
        assert!(matches!(
            relation(&know, &dropped_patient),
            Relation::Entails
        ));
    }

    #[test]
    fn attitude_contradiction_is_about_the_attitude() {
        let content = Meaning::HasProperty {
            subject: entity("report"),
            property: "long".to_string(),
            negated: false,
        };
        let know = Meaning::Attitude {
            holder: entity("teacher"),
            verb: "know".to_string(),
            content: Box::new(content.clone()),
            negated: false,
        };
        let not_know = Meaning::Attitude {
            holder: entity("teacher"),
            verb: "know".to_string(),
            content: Box::new(content.clone()),
            negated: true,
        };
        // "knows that P" contradicts "does not know that P".
        assert!(matches!(relation(&know, &not_know), Relation::Contradicts));
        // SOUNDNESS: knowing P does NOT contradict BELIEVING P (different verbs,
        // both can hold).
        let believe = Meaning::Attitude {
            holder: entity("teacher"),
            verb: "believe".to_string(),
            content: Box::new(content),
            negated: false,
        };
        assert!(matches!(relation(&know, &believe), Relation::Neutral));
    }

    #[test]
    fn cardinal_at_least_monotonicity() {
        // "at least 3 teachers write a report" entails "at least 2" and "at least 1".
        let three = cardinal(3, "teacher");
        assert!(matches!(
            relation(&three, &cardinal(2, "teacher")),
            Relation::Entails
        ));
        assert!(matches!(
            relation(&three, &cardinal(1, "teacher")),
            Relation::Entails
        ));
        // ...and entails the existential "some teacher writes a report".
        let some = Meaning::Quantified {
            quant: Quantifier::Some,
            var_category: "teacher".to_string(),
            body: Event {
                predicate: "write".to_string(),
                agent: None,
                patient: Some(Term::Indefinite("report".to_string())),
                recipient: None,
                tense: Tense::Present,
                negated: false,
            },
        };
        assert!(matches!(relation(&three, &some), Relation::Entails));
    }

    #[test]
    fn cardinal_does_not_entail_stronger_count() {
        // SOUNDNESS: "at least 2" must NOT entail "at least 3".
        let two = cardinal(2, "teacher");
        assert!(matches!(
            relation(&two, &cardinal(3, "teacher")),
            Relation::Neutral
        ));
    }

    #[test]
    fn cardinal_pairs_do_not_contradict() {
        // SOUNDNESS: two at-least claims never contradict (the larger entails the
        // smaller); a weaker count is Entailed, not contradicted, and an
        // unrelated count is Neutral — never Contradicts.
        let three = cardinal(3, "teacher");
        assert!(!matches!(
            relation(&three, &cardinal(5, "teacher")),
            Relation::Contradicts
        ));
        assert!(!matches!(
            relation(&three, &cardinal(2, "teacher")),
            Relation::Contradicts
        ));
    }

    #[test]
    fn count_question_is_non_assertoric() {
        // A counting question carries no assertion: it neither entails nor
        // contradicts, and yields no consequences.
        let cq = Meaning::CountQuestion {
            var_category: "teacher".to_string(),
            body: Event {
                predicate: "write".to_string(),
                agent: None,
                patient: Some(Term::Indefinite("report".to_string())),
                recipient: None,
                tense: Tense::Present,
                negated: false,
            },
        };
        assert!(consequences(&cq).is_empty());
        let p = ev(entity("teacher"), Some(entity("report")), false);
        assert!(matches!(relation(&p, &cq), Relation::Neutral));
        assert!(matches!(relation(&cq, &p), Relation::Neutral));
    }
}
