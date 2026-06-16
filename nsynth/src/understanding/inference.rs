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

use crate::understanding::meaning::{Meaning, Term};

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
                // "X is a person" entails "X is NOT a thing", and vice versa,
                // for the two mutually-exclusive animacy categories we model.
                if let Some(opp) = opposite_category(category) {
                    push_unique(
                        &mut out,
                        Meaning::IsA {
                            subject: subject.clone(),
                            category: opp,
                            negated: true,
                        },
                    );
                }
            }
        }
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
        Meaning::YesNoQuestion(_) | Meaning::WhQuestion { .. } | Meaning::Unknown(_)
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

/// The mutually-exclusive opposite of an animacy category, if one exists.
/// "person" and "thing" are the two categories the world model derives from
/// noun animacy, and they are mutually exclusive.
fn opposite_category(category: &str) -> Option<String> {
    match category {
        "person" => Some("thing".to_string()),
        "thing" => Some("person".to_string()),
        _ => None,
    }
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
            tense: Tense::Present,
            negated: true,
        };
        assert!(consequences(&Meaning::Event(p)).is_empty());
    }
}
