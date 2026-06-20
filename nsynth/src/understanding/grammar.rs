//! Grammar induction scaffold: a VERIFIED, class-skeleton-bounded role
//! assignment that the parser consults as a learned fallback.
//!
//! THE FRONTIER. Every construction in `semantics.rs` is a hand-written Rust
//! branch ("first noun = subject, last noun = object"). This module is the
//! skeleton of the replacement: a `LearnedConstruction` records, for one or more
//! CLASS SKELETONS (the part-of-speech code arrays produced by
//! [`token_classes`](super::semantics::token_classes)), which token INDEX fills
//! the agent / patient / predicate slot. Given a sentence whose skeleton matches,
//! it assembles exactly the `Event` a hand rule would — but the slot indices come
//! from an acquired program, not a hard-coded assumption.
//!
//! ## Why bounding to recorded skeletons keeps it SOUND
//!
//! A `LearnedConstruction` only fires when the incoming sentence's class skeleton
//! is byte-for-byte one of the `skeletons` it was VERIFIED on. WITHIN a fixed
//! skeleton the role indices are constant — the same positions are agent /
//! patient / predicate regardless of WHICH words fill the slots, because the
//! skeleton abstracts the lexicon away and leaves only the grammatical shape. So
//! verifying the assignment on the recorded skeletons proves it for EVERY
//! sentence sharing that shape. A sentence with a skeleton the construction never
//! saw returns `None` (open-world discipline) and the parser keeps its existing
//! behavior — the learned rule can only ADD coverage on proven shapes, never
//! override or mis-fire on an unproven one.
//!
//! This file is the SCAFFOLD: the type, its `apply`, and a stub holder on the
//! engine side. Synthesizing the index assignment, gating it, grafting it, and
//! persisting it is the next phase.

use crate::benchmark::{Example, Problem, Value};
use crate::comprehension::{words_of, Engine};
use crate::solver::solve_problem;
use crate::understanding::meaning::{Aspect, Event, Meaning};
use crate::understanding::semantics::{lemma_and_tense, term_from, token_classes};

/// A verified role assignment for one family of CLASS SKELETONS.
///
/// `skeletons` are the exact token-class arrays (from [`token_classes`]) this
/// construction was proven on. `agent_idx` / `patient_idx` / `predicate_idx` are
/// the token indices — constant within any of those skeletons — that fill the
/// agent, patient, and predicate slots respectively.
///
/// Example (object-fronting, "The report the teacher writes."):
///   skeleton `[0,1,0,1,2]`  (det noun det noun verb)
///   agent_idx 3 ("teacher"), patient_idx 1 ("report"), predicate_idx 4 ("writes").
/// The default hand rule would instead take index 1 as the agent and never read
/// index 4 as the verb — this construction overrides that, but ONLY for sentences
/// whose skeleton is exactly `[0,1,0,1,2]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LearnedConstruction {
    /// human-readable tag, e.g. "object_fronting".
    pub name: String,
    /// the class skeletons this assignment is VERIFIED for.
    pub skeletons: Vec<Vec<i64>>,
    /// token index filling the AGENT slot.
    pub agent_idx: usize,
    /// token index filling the PATIENT slot.
    pub patient_idx: usize,
    /// token index of the PREDICATE (lexical verb).
    pub predicate_idx: usize,
}

impl LearnedConstruction {
    /// Apply this construction to a sentence's tokens, IF its class skeleton is
    /// one of the recorded (verified) `skeletons`.
    ///
    /// Returns `Some(Meaning::Event{..})` with agent / patient built by
    /// [`term_from`] at the recorded indices and the predicate de-inflected by
    /// [`lemma_and_tense`] — the SAME constructors the hand rules use. Returns
    /// `None` when the skeleton does not match (so the caller falls through to
    /// the existing parser) or when any recorded index is out of range for these
    /// tokens (a defensive guard; a well-formed construction never records an
    /// index past its own skeleton length).
    pub fn apply(&self, engine: &Engine, toks: &[String], classes: &[i64]) -> Option<Meaning> {
        // SOUNDNESS GATE: only fire on a skeleton we were VERIFIED on.
        if !self.skeletons.iter().any(|s| s.as_slice() == classes) {
            return None;
        }
        // Defensive bounds check: the skeleton length equals `toks.len()` by
        // construction (both come from the same `words_of`), but never index past
        // the end if a malformed construction slips through.
        let max_idx = self.agent_idx.max(self.patient_idx).max(self.predicate_idx);
        if max_idx >= toks.len() {
            return None;
        }

        let agent = term_from(toks, self.agent_idx);
        let patient = term_from(toks, self.patient_idx);
        let (predicate, tense) = lemma_and_tense(engine, &toks[self.predicate_idx]);

        Some(Meaning::Event(Event {
            predicate,
            agent: Some(agent),
            patient: Some(patient),
            recipient: None,
            tense,
            aspect: Aspect::Simple,
            negated: false,
        }))
    }
}

/// One labeled example of a construction: a sentence, plus the surface words that
/// fill its agent and patient slots and the LEMMA of its predicate verb.
///
/// `(sentence, agent_word, patient_word, predicate_lemma)`. The words are matched
/// (case-insensitively, after the same alphabetic tokenization the parser uses)
/// against the sentence's tokens to RECOVER the slot INDICES — which is the whole
/// point: the learner is told the *roles* (who is the agent), not the *positions*,
/// and induces the position-to-role mapping itself.
pub type ConstructionExample<'a> = (&'a str, &'a str, &'a str, &'a str);

/// Locate the token index of `agent`/`patient` (exact, case-folded surface match)
/// and the predicate (the token whose [`lemma_and_tense`] lemma equals
/// `predicate_lemma`). Returns `(agent_idx, patient_idx, predicate_idx)` or an
/// `Err` describing the first slot that could not be located in `toks`.
fn locate_slots(
    engine: &Engine,
    toks: &[String],
    agent: &str,
    patient: &str,
    predicate_lemma: &str,
) -> Result<(usize, usize, usize), String> {
    let agent = agent.to_lowercase();
    let patient = patient.to_lowercase();
    let predicate_lemma = predicate_lemma.to_lowercase();

    let agent_idx = toks
        .iter()
        .position(|t| *t == agent)
        .ok_or_else(|| format!("agent word `{agent}` not found in tokens {toks:?}"))?;
    let patient_idx = toks
        .iter()
        .position(|t| *t == patient)
        .ok_or_else(|| format!("patient word `{patient}` not found in tokens {toks:?}"))?;
    let predicate_idx = toks
        .iter()
        .position(|t| lemma_and_tense(engine, t).0 == predicate_lemma)
        .ok_or_else(|| {
            format!("no token in {toks:?} de-inflects to predicate lemma `{predicate_lemma}`")
        })?;

    Ok((agent_idx, patient_idx, predicate_idx))
}

/// Synthesize + VERIFY one array->int slot program: skeleton (class codes) ->
/// `target_idx`. Returns the solver's verdict; `Ok(())` means a program that
/// reproduces every (skeleton -> index) pair was found and PROVEN against the
/// examples by [`solve_problem`]. `Err` carries the solver's explanation.
///
/// This is what makes the construction a SYNTHESIZED rule rather than mere
/// bookkeeping: the index assignment is the output of an actual program search,
/// certified on the labeled skeletons before we ever record it.
fn verify_slot_program(
    name: &str,
    signature: &'static str,
    skeletons: &[Vec<i64>],
    targets: &[usize],
) -> Result<(), String> {
    let examples: Vec<Example> = skeletons
        .iter()
        .zip(targets.iter())
        .map(|(sk, idx)| Example {
            inputs: vec![Value::Array(sk.clone())],
            expected: Value::Int(*idx as i64),
        })
        .collect();
    let problem = Problem {
        name: name.to_string(),
        category: "grammar-induction",
        description: "",
        signature,
        examples,
        ..Problem::default()
    };
    let result = solve_problem(&problem);
    if result.success {
        Ok(())
    } else {
        Err(result
            .error
            .unwrap_or_else(|| format!("synthesis failed for slot program {name}")))
    }
}

/// LEARN a [`LearnedConstruction`] from labeled examples — grammar induction.
///
/// Each example is `(sentence, agent_word, patient_word, predicate_lemma)`. For
/// every example we:
///   1. tokenize + compute its class skeleton ([`token_classes`]);
///   2. locate the agent / patient / predicate TOKEN INDICES (the agent/patient by
///      surface word, the predicate by de-inflecting to the given lemma).
///
/// We then REQUIRE that every example sharing a class skeleton agree on
/// `(agent_idx, patient_idx, predicate_idx)`. If two examples have the same shape
/// but disagree on where a role lives, the construction is NOT well-formed (the
/// role-to-position mapping is not a function of the skeleton) and we return
/// `Err`. Soundness depends on this: `apply` keys purely on the skeleton, so a
/// skeleton may carry exactly one role assignment.
///
/// Finally — and this is what makes the result a genuinely SYNTHESIZED + VERIFIED
/// rule rather than a lookup table — we build two `[i64] -> i64` problems
/// (skeleton -> agent_idx and skeleton -> patient_idx), hand each to
/// [`solve_problem`], and ASSERT success. The recovered programs prove the index
/// assignment is reproducible by an actual program over the class arrays; only
/// then do we record the verified skeletons + indices in the returned
/// `LearnedConstruction`.
pub fn learn_construction_from_examples(
    engine: &Engine,
    name: &str,
    examples: &[ConstructionExample],
) -> Result<LearnedConstruction, String> {
    if examples.is_empty() {
        return Err("no examples provided".to_string());
    }

    // skeleton -> the agreed (agent_idx, patient_idx, predicate_idx), in
    // first-seen order so the synthesized examples are deterministic.
    let mut order: Vec<Vec<i64>> = Vec::new();
    let mut assigned: std::collections::HashMap<Vec<i64>, (usize, usize, usize)> =
        std::collections::HashMap::new();

    for (sentence, agent, patient, predicate_lemma) in examples {
        let toks = words_of(sentence);
        let classes = token_classes(engine, sentence);
        let (a, p, v) = locate_slots(engine, &toks, agent, patient, predicate_lemma)?;

        match assigned.get(&classes) {
            Some(&(ea, ep, ev)) => {
                // SOUNDNESS: a single skeleton must carry a single role assignment.
                if (ea, ep, ev) != (a, p, v) {
                    return Err(format!(
                        "ill-formed construction `{name}`: skeleton {classes:?} maps to both \
                         (agent {ea}, patient {ep}, pred {ev}) and (agent {a}, patient {p}, \
                         pred {v}) — role-to-position mapping is not a function of the skeleton"
                    ));
                }
            }
            None => {
                order.push(classes.clone());
                assigned.insert(classes, (a, p, v));
            }
        }
    }

    // The distinct skeletons + their agreed indices, in first-seen order.
    let skeletons: Vec<Vec<i64>> = order.clone();
    let agent_targets: Vec<usize> = order.iter().map(|s| assigned[s].0).collect();
    let patient_targets: Vec<usize> = order.iter().map(|s| assigned[s].1).collect();
    let (agent_idx, patient_idx, predicate_idx) = assigned[&order[0]];

    // SYNTHESIZE + VERIFY the index assignment as real programs over the class
    // arrays. Both must solve, else the assignment is not a recoverable rule.
    verify_slot_program(
        &format!("{name}_agent_idx"),
        "fn agent_idx(arr: [i64]) -> i64",
        &skeletons,
        &agent_targets,
    )?;
    verify_slot_program(
        &format!("{name}_patient_idx"),
        "fn patient_idx(arr: [i64]) -> i64",
        &skeletons,
        &patient_targets,
    )?;

    Ok(LearnedConstruction {
        name: name.to_string(),
        skeletons,
        agent_idx,
        patient_idx,
        predicate_idx,
    })
}

/// Side-table holder for the constructions an engine has ACQUIRED.
///
/// STUB for this phase: a freshly-built holder is empty, so consulting it is
/// inert and the parser's behavior is byte-for-byte unchanged. The real wiring —
/// synthesizing an index assignment from examples, gating it through the
/// regression gate, grafting it here, and persisting it across runs — lands next
/// phase. `apply_first` is the single entry point the parser fallback will call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LearnedGrammar {
    constructions: Vec<LearnedConstruction>,
}

impl LearnedGrammar {
    /// An empty grammar (no acquired constructions). On a fresh engine this is
    /// what the parser sees, so every learned-grammar consultation returns `None`
    /// and the existing hand-coded parser is fully in charge.
    pub fn new() -> Self {
        Self::default()
    }

    /// Graft a verified construction onto the grammar. (Real callers will gate
    /// before reaching here; this is the raw insertion point.)
    pub fn add(&mut self, construction: LearnedConstruction) {
        self.constructions.push(construction);
    }

    /// Number of acquired constructions.
    pub fn len(&self) -> usize {
        self.constructions.len()
    }

    /// Is the grammar empty (no acquired constructions)?
    pub fn is_empty(&self) -> bool {
        self.constructions.is_empty()
    }

    /// The acquired constructions, in adoption order. Read by the regression gate's
    /// collision-soundness invariant, which checks each registered construction
    /// against the golden battery's base-parseable sentences.
    pub fn constructions(&self) -> &[LearnedConstruction] {
        &self.constructions
    }

    /// Consult every acquired construction in order and return the FIRST that
    /// fires on `sentence`'s class skeleton, or `None` if none match. This is the
    /// learned fallback the parser calls on an Unknown / object-fronted clause.
    ///
    /// Encodes the skeleton ONCE here ([`token_classes`]) and hands it to each
    /// construction's `apply`, so the per-token classification cost is paid a
    /// single time per consultation.
    pub fn apply_first(&self, engine: &Engine, sentence: &str) -> Option<Meaning> {
        if self.constructions.is_empty() {
            return None;
        }
        let toks = crate::comprehension::words_of(sentence);
        let classes = token_classes(engine, sentence);
        self.constructions
            .iter()
            .find_map(|c| c.apply(engine, &toks, &classes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::understanding::meaning::Term;

    /// An empty grammar (the fresh-engine STUB state) is inert: any sentence
    /// returns `None`, so the parser's existing behavior is unchanged.
    #[test]
    fn empty_grammar_is_inert() {
        let engine = Engine::new();
        let g = LearnedGrammar::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert_eq!(
            g.apply_first(&engine, "The report the teacher writes."),
            None
        );
    }

    /// A construction whose skeleton does NOT match the sentence returns `None`
    /// (open-world: never fire on an unproven shape).
    #[test]
    fn non_matching_skeleton_returns_none() {
        let engine = Engine::new();
        let toks = crate::comprehension::words_of("The teacher writes the report.");
        let classes = token_classes(&engine, "The teacher writes the report.");
        let c = LearnedConstruction {
            name: "object_fronting".to_string(),
            // a DIFFERENT skeleton than the SVO sentence above.
            skeletons: vec![vec![0, 1, 0, 1, 2]],
            agent_idx: 3,
            patient_idx: 1,
            predicate_idx: 4,
        };
        assert_eq!(c.apply(&engine, &toks, &classes), None);
    }

    /// The CORE acquisition claim: a construction with the object-fronting
    /// skeleton assigns report=patient, teacher=agent, writes=predicate — the
    /// SAME `Event` an SVO sentence "The teacher writes the report." produces,
    /// which the base parser MIS-PARSES for the fronted order.
    #[test]
    fn object_fronting_skeleton_recovers_correct_roles() {
        let engine = Engine::new();
        let sentence = "The report the teacher writes.";
        let toks = crate::comprehension::words_of(sentence);
        let classes = token_classes(&engine, sentence);
        // Skeleton sanity: det noun det noun verb.
        assert_eq!(classes, vec![0, 1, 0, 1, 2]);

        let c = LearnedConstruction {
            name: "object_fronting".to_string(),
            skeletons: vec![vec![0, 1, 0, 1, 2]],
            agent_idx: 3,
            patient_idx: 1,
            predicate_idx: 4,
        };
        let m = c.apply(&engine, &toks, &classes).expect("should fire");
        let Meaning::Event(e) = m else {
            panic!("expected Event, got {m:?}");
        };
        assert_eq!(e.predicate, "write");
        assert_eq!(e.agent, Some(Term::Entity("teacher".to_string())));
        assert_eq!(e.patient, Some(Term::Entity("report".to_string())));
        assert_eq!(e.recipient, None);
    }

    /// `apply_first` dispatches through a grafted construction.
    #[test]
    fn apply_first_dispatches_grafted_construction() {
        let engine = Engine::new();
        let mut g = LearnedGrammar::new();
        g.add(LearnedConstruction {
            name: "object_fronting".to_string(),
            skeletons: vec![vec![0, 1, 0, 1, 2]],
            agent_idx: 3,
            patient_idx: 1,
            predicate_idx: 4,
        });
        let m = g
            .apply_first(&engine, "The report the teacher writes.")
            .expect("grafted construction should fire");
        assert!(matches!(m, Meaning::Event(_)));
    }

    // =======================================================================
    // GRAMMAR INDUCTION: learning a construction from labeled examples, and the
    // end-to-end parser fallback that consults it ON UNSEEN words.
    // =======================================================================

    /// The labeled OSV (object-subject-verb, "the OBJECT the SUBJECT VERBs")
    /// training set: three sentences with DIFFERENT words but the same word-order
    /// shape, each tagged with which surface word is the agent / patient and the
    /// predicate's lemma. The learner is told the ROLES, never the positions.
    fn osv_examples() -> Vec<ConstructionExample<'static>> {
        vec![
            // "the report the teacher writes" — teacher writes the report.
            (
                "the report the teacher writes",
                "teacher",
                "report",
                "write",
            ),
            // "the book the student reads" — student reads the book.
            ("the book the student reads", "student", "book", "read"),
            // "the memo the doctor fixes" — doctor fixes the memo.
            ("the memo the doctor fixes", "doctor", "memo", "fix"),
        ]
    }

    /// CORE acquisition claim: learning OSV from labeled examples yields a
    /// VERIFIED construction whose synthesized skeleton->index programs both
    /// solved, and whose recovered slot indices are the OSV ones
    /// (patient first, agent third, verb fourth).
    #[test]
    fn learn_osv_construction_verifies() {
        let engine = Engine::new();
        let examples = osv_examples();
        let c = learn_construction_from_examples(&engine, "osv", &examples)
            .expect("OSV construction should synthesize + verify");

        // Every training sentence shares the OSV skeleton det noun det noun verb.
        assert_eq!(c.skeletons, vec![vec![0, 1, 0, 1, 2]]);
        // RECOVERED slot indices: patient at 1, agent at 3, predicate at 4.
        assert_eq!(c.patient_idx, 1, "patient is the FRONTED object (index 1)");
        assert_eq!(c.agent_idx, 3, "agent is the embedded subject (index 3)");
        assert_eq!(c.predicate_idx, 4, "predicate is the final verb (index 4)");

        // Applying it to a TRAINING sentence reproduces the labeled roles.
        let toks = crate::comprehension::words_of("the report the teacher writes");
        let classes = token_classes(&engine, "the report the teacher writes");
        let m = c
            .apply(&engine, &toks, &classes)
            .expect("should fire on training shape");
        let Meaning::Event(e) = m else {
            panic!("expected Event, got {m:?}")
        };
        assert_eq!(e.predicate, "write");
        assert_eq!(e.agent, Some(Term::Entity("teacher".to_string())));
        assert_eq!(e.patient, Some(Term::Entity("report".to_string())));
    }

    /// END-TO-END: a Mind with the learned OSV construction REGISTERED parses an
    /// UNSEEN-word OSV sentence ("the letter the editor reads") to the correct
    /// Event — generalization across the lexicon, because the construction keys on
    /// the class skeleton, not the specific words.
    #[test]
    fn registered_construction_parses_unseen_osv() {
        // Learn the construction on one engine...
        let learner = Engine::new();
        let c = learn_construction_from_examples(&learner, "osv", &osv_examples())
            .expect("OSV construction should synthesize + verify");

        // ...register it on a fresh engine and parse an UNSEEN-word OSV sentence.
        let mut engine = Engine::new();
        engine.register_construction(c);

        // "letter" / "editor" / "read" never appeared together in the OSV training
        // set; only the SHAPE [0,1,0,1,2] was learned.
        let sentence = "the letter the editor reads";
        let m = crate::understanding::semantics::understand(&engine, sentence);
        let Meaning::Event(e) = m else {
            panic!("learned fallback should parse unseen OSV to an Event, got {m:?}");
        };
        assert_eq!(e.predicate, "read");
        assert_eq!(e.agent, Some(Term::Entity("editor".to_string())));
        assert_eq!(e.patient, Some(Term::Entity("letter".to_string())));
        assert_eq!(e.recipient, None);
    }

    /// SOUNDNESS / open-world: a FRESH engine WITHOUT the construction returns
    /// `Unknown` for the same OSV sentence — the learned fallback is the ONLY
    /// reason the registered engine parses it. (The hand parser cannot read an
    /// object-fronted clause; it refuses rather than fabricating an Event.)
    #[test]
    fn fresh_engine_returns_unknown_for_osv() {
        let engine = Engine::new();
        assert!(engine.learned_grammar().is_empty());
        let m = crate::understanding::semantics::understand(&engine, "the letter the editor reads");
        assert!(
            matches!(m, Meaning::Unknown(_)),
            "without the construction, OSV must be Unknown, got {m:?}"
        );
    }

    /// An ILL-FORMED construction — two examples with the SAME skeleton but
    /// DIFFERENT role positions — is rejected: the role-to-position mapping is not
    /// a function of the skeleton, so it cannot be a sound construction.
    #[test]
    fn ill_formed_construction_is_rejected() {
        let engine = Engine::new();
        // Both sentences have skeleton [0,1,0,1,2], but the FIRST labels the agent
        // at index 3 (OSV) and the SECOND labels the agent at index 1 (SVO-ish) —
        // a contradiction on the same shape.
        let bad: Vec<ConstructionExample> = vec![
            (
                "the report the teacher writes",
                "teacher",
                "report",
                "write",
            ),
            ("the editor the letter reads", "editor", "letter", "read"),
        ];
        let err = learn_construction_from_examples(&engine, "bad", &bad)
            .expect_err("contradictory role positions must be rejected");
        assert!(err.contains("not a function of the skeleton"), "got: {err}");
    }
}
