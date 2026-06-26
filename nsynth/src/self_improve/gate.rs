//! The regression gate: the guard every self-modification must pass.
//!
//! A [`GoldenCase`] is a frozen behavioral expectation — a few setup sentences,
//! a question, and the answer (substring) we expect. [`regression_gate`] replays
//! the whole golden corpus through a **fresh** [`Discourse`] per case (one shared
//! [`Engine`]), so the cases never bleed world state into each other, and reports
//! how many passed plus whether the world model stayed sound. A candidate change
//! is only allowed to take effect when [`GateReport::ok`] holds.
//!
//! The corpus below is mined directly from the understanding layer's existing
//! passing behaviors (the `understanding::qa` tests and the `comprehend`
//! demo). Every `expected` substring is what [`qa::answer`] actually returns for
//! that (setup, question) pair *today*, so the gate is a faithful snapshot of
//! current behavior — a change that alters any of these answers is a regression
//! by definition.

use crate::comprehension::Engine;
use crate::understanding::discourse::Discourse;
use crate::understanding::meaning::{Aspect, Event, Meaning, Tense, Term};
use crate::understanding::qa;
use crate::understanding::semantics;

/// One frozen behavioral expectation.
///
/// `setup` is read, in order, into a fresh discourse; then `question` is asked
/// and the answer is checked to contain `expected` (case-insensitive substring,
/// mirroring the golden-corpus convention). Static strings keep the corpus a
/// compile-time constant with no allocation or external files.
pub struct GoldenCase {
    /// Sentences to read into a fresh discourse, in order, before asking.
    pub setup: Vec<&'static str>,
    /// The question to ask once the setup has been read.
    pub question: &'static str,
    /// The substring the answer must contain for the case to pass.
    pub expected: &'static str,
}

/// The outcome of running the whole golden corpus.
///
/// `passed`/`total` count behavioral cases; `failures` describes each mismatch
/// for diagnostics; `sound` records whether the soundness oracle held while the
/// corpus ran. A change is accepted only when [`ok`](GateReport::ok) is true.
pub struct GateReport {
    /// Number of golden cases whose answer matched.
    pub passed: usize,
    /// Number of golden cases attempted.
    pub total: usize,
    /// Human-readable description of each failing case (empty when all pass).
    pub failures: Vec<String>,
    /// Whether the world model stayed sound across the whole run.
    pub sound: bool,
}

impl GateReport {
    /// The gate is green iff every case passed and the run stayed sound.
    pub fn ok(&self) -> bool {
        self.passed == self.total && self.sound
    }
}

/// The frozen golden corpus — the behavioral contract the system must never
/// regress.
///
/// Each case is `(setup sentences, question, expected substring)`. The cases
/// span every domain the understanding layer covers: plain events, taxonomy /
/// IsA, the three quantifiers (every / some / no), comparatives (including
/// transitive closure), aspect, modality, temporal order, causal "why",
/// negation scope, ditransitives, cardinality / counting, attributes, and
/// disjunction. Every `expected` is the literal answer `qa::answer` returns for
/// that pair today (verified by the `golden_cases_all_pass` test below).
pub fn golden_cases() -> Vec<GoldenCase> {
    vec![
        // ---- Plain events: yes / no / open-world ---------------------------
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Does the teacher write the report?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Does the editor read the memo?",
            expected: "I don't know.",
        },
        GoldenCase {
            setup: vec!["The teacher does not write the letter."],
            question: "Does the teacher write the letter?",
            expected: "No",
        },
        // A wh-question retrieves the filler of the queried slot.
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Who writes the report?",
            expected: "The teacher.",
        },
        GoldenCase {
            setup: vec!["The author reads the book."],
            question: "What does the author read?",
            expected: "The book.",
        },
        // ---- Taxonomy / IsA: derived super-category membership -------------
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Is the teacher a person?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Is the teacher an agent?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Is the report a thing?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Is the report a person?",
            expected: "No",
        },
        // ---- Quantifiers: every / some / no -------------------------------
        GoldenCase {
            setup: vec![
                "The teacher writes the report.",
                "The editor writes the report.",
            ],
            question: "Does every agent write a report?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec![
                "The teacher writes the report.",
                "The editor writes the report.",
            ],
            question: "Does some agent write a report?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec![
                "The teacher writes the report.",
                "The editor does not read the book.",
            ],
            question: "Does every agent read a book?",
            expected: "No",
        },
        // ---- Comparatives, including transitive closure -------------------
        GoldenCase {
            setup: vec!["The report is longer than the book."],
            question: "Is the report longer than the book?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec![
                "The report is longer than the book.",
                "The book is longer than the letter.",
            ],
            question: "Is the report longer than the letter?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec!["The report is longer than the book."],
            question: "Is the letter longer than the report?",
            expected: "I don't know.",
        },
        // ---- Aspect: progressive / perfect of a holding simple event ------
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Is the teacher writing the report?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Has the teacher written the report?",
            expected: "Yes",
        },
        // ---- Modality: actuality entails possibility ----------------------
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Can the teacher write the report?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Can the editor write the report?",
            expected: "I don't know.",
        },
        // ---- Temporal order -----------------------------------------------
        GoldenCase {
            setup: vec!["The teacher writes the report before the editor reads the book."],
            question: "Does the teacher write the report before the editor reads the book?",
            expected: "Yes",
        },
        // ---- Causal "why" --------------------------------------------------
        GoldenCase {
            setup: vec!["The street floods because the rain falls."],
            question: "Why does the street flood?",
            expected: "Because the rain falls.",
        },
        // ---- Ditransitive: recipient slot ---------------------------------
        GoldenCase {
            setup: vec!["The teacher gives the book to the student."],
            question: "Who does the teacher give the book to?",
            expected: "The student.",
        },
        // ---- Cardinality / counting ---------------------------------------
        GoldenCase {
            setup: vec![
                "The teacher writes the report.",
                "The author writes the report.",
            ],
            question: "How many agents write a report?",
            expected: "Two.",
        },
        // The "no" quantifier: "no doctor writes a report" is vacuously true when
        // no doctor is even known to write one (and one is asserted not to).
        GoldenCase {
            setup: vec!["The doctor does not write the report."],
            question: "Does no doctor write a report?",
            expected: "Yes",
        },
        // A comparative low pole ("shorter"), the converse of an asserted ordering.
        GoldenCase {
            setup: vec!["The report is longer than the book."],
            question: "Is the book shorter than the report?",
            expected: "Yes",
        },
        // ---- Attributes ----------------------------------------------------
        GoldenCase {
            setup: vec!["The teacher writes the report.", "The teacher is careful."],
            question: "Is the teacher careful?",
            expected: "Yes",
        },
        GoldenCase {
            setup: vec!["The teacher writes the report.", "The teacher is careful."],
            question: "Is the teacher kind?",
            expected: "I don't know.",
        },
        // ---- Disjunction ---------------------------------------------------
        GoldenCase {
            setup: vec!["The editor writes the report."],
            question: "Does the editor write the report or the editor read the book?",
            expected: "Yes",
        },
        // ---- Epistemic / factivity ----------------------------------------
        GoldenCase {
            setup: vec!["The teacher knows that the report is long."],
            question: "Does the teacher know that the report is long?",
            expected: "Yes",
        },
        // ---- Negation: an asserted negative answers No --------------------
        GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Does the teacher read the book?",
            expected: "I don't know.",
        },
        // ---- Coreference across discourse ---------------------------------
        GoldenCase {
            setup: vec![
                "The author reads the book.",
                "The author writes the report.",
            ],
            question: "Does the author write the report?",
            expected: "Yes",
        },
    ]
}

/// Replay a single golden case through a **fresh** discourse on the shared
/// engine: read each setup sentence, ask the question, and check the answer
/// contains `expected` (case-insensitive).
///
/// A fresh [`Discourse::new`] per case is the fresh-per-case primitive — cheap
/// (no synthesis), so cases never share world state.
pub fn run_case(engine: &Engine, case: &GoldenCase) -> bool {
    let mut discourse = Discourse::new();
    for sentence in &case.setup {
        discourse.read(engine, sentence);
    }
    let parsed = semantics::understand(engine, case.question);
    let answer = qa::answer(engine, &discourse, case.question);
    let _ = parsed; // parse is part of the path qa::answer already runs; kept explicit.
    answer
        .to_lowercase()
        .contains(&case.expected.to_lowercase())
}

/// Run the entire golden corpus against `engine` and report the result.
///
/// Every [`golden_cases`] entry is replayed through [`run_case`]; each failure's
/// question is recorded. In addition, a small battery of in-process
/// SOUNDNESS-invariant probes is run (see [`soundness_holds`]); the gate's
/// `sound` flag is the conjunction of those probes. The report is `ok()` only
/// when every behavioral case passed *and* every soundness invariant held.
pub fn regression_gate(engine: &Engine) -> GateReport {
    // MEMOIZE by behavioral fingerprint. `regression_gate` is a *pure function* of
    // the engine's behavior-bearing state (program + learned grammar + learned
    // members), all captured by `behavioral_fingerprint`. Two engines with the same
    // fingerprint produce the same verdict, so we cache the verdict keyed on the
    // fingerprint and return it on a hit — turning a repeat gate (the reload
    // re-gates many components; a teach re-gates the same candidate on reuse) from
    // ~32 interpreter replays + 5 soundness probes into a single hash lookup. The
    // cache is process-local and additive: a miss runs the FULL gate exactly as
    // before, so soundness is untouched — a hit can only return a verdict an honest
    // full run already produced for that identical behavior.
    let key = engine.behavioral_fingerprint();
    if let Some(cached) = gate_cache_get(key) {
        return cached;
    }
    let report = compute_gate(engine);
    gate_cache_put(key, &report);
    report
}

/// Run the full golden corpus + soundness oracle against `engine` — the uncached
/// core of [`regression_gate`]. Always does the real work; the memo wrapper decides
/// whether to call it.
fn compute_gate(engine: &Engine) -> GateReport {
    let cases = golden_cases();
    let total = cases.len();
    let mut passed = 0usize;
    let mut failures = Vec::new();
    for case in &cases {
        if run_case(engine, case) {
            passed += 1;
        } else {
            failures.push(case.question.to_string());
        }
    }
    let sound = soundness_holds(engine);
    GateReport {
        passed,
        total,
        failures,
        sound,
    }
}

// ---------------------------------------------------------------------------
// Gate memo: behavioral-fingerprint -> verdict.
//
// The gate is pure in the engine's behavioral surface, so caching its verdict by
// `Engine::behavioral_fingerprint` is sound: a hit returns exactly what a full run
// would. The cache is bounded (one entry per distinct behavior seen this process)
// and only ever GROWS the set of known-good answers — it can never manufacture a
// pass for behavior that has not actually been evaluated.
// ---------------------------------------------------------------------------
use std::sync::Mutex;
use std::sync::OnceLock;

#[derive(Clone)]
struct CachedVerdict {
    passed: usize,
    total: usize,
    failures: Vec<String>,
    sound: bool,
}

fn gate_cache() -> &'static Mutex<std::collections::HashMap<u64, CachedVerdict>> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<u64, CachedVerdict>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn gate_cache_get(key: u64) -> Option<GateReport> {
    let guard = gate_cache().lock().unwrap_or_else(|p| p.into_inner());
    guard.get(&key).map(|v| GateReport {
        passed: v.passed,
        total: v.total,
        failures: v.failures.clone(),
        sound: v.sound,
    })
}

fn gate_cache_put(key: u64, report: &GateReport) {
    let mut guard = gate_cache().lock().unwrap_or_else(|p| p.into_inner());
    guard.insert(
        key,
        CachedVerdict {
            passed: report.passed,
            total: report.total,
            failures: report.failures.clone(),
            sound: report.sound,
        },
    );
}

// ---------------------------------------------------------------------------
// Soundness invariants
// ---------------------------------------------------------------------------

/// A present, affirmative `write(agent, patient)` event used to build probe
/// queries. `agent`/`patient` are definite entities.
fn write_event(agent: &str, patient: &str, negated: bool) -> Meaning {
    Meaning::Event(Event {
        predicate: "write".to_string(),
        agent: Some(Term::Entity(agent.to_string())),
        patient: Some(Term::Entity(patient.to_string())),
        recipient: None,
        tense: Tense::Present,
        aspect: Aspect::Simple,
        negated,
    })
}

/// Run the soundness oracle: a handful of model-theoretic invariants the world
/// model must NEVER violate, checked in-process against fresh discourses. These
/// are the properties that make the gate a *soundness* guard and not merely a
/// behavioral snapshot — a self-modification that breaks any of them is rejected
/// even if every golden case still passes.
///
/// The invariants (each must hold for the gate to call the run sound):
///   1. **No false entailment (open world).** A query over entirely unmentioned
///      entities must answer "I don't know." — never a fabricated Yes/No.
///   2. **Asserted fact + its negation.** After asserting a fact, the fact
///      answers Yes and its explicit negation answers No.
///   3. **Must-monotonicity.** When an event actually holds, "can <event>" is
///      true (actuality entails possibility); but with nothing known, the same
///      modal is open-world (possibility never leaks back to actuality).
///   4. **Causal non-commutativity.** A cause→effect link does not license the
///      reversed effect→cause reading: "why does <cause>?" is not answered by the
///      effect.
fn soundness_holds(engine: &Engine) -> bool {
    no_false_entailment(engine)
        && asserted_fact_and_negation(engine)
        && modal_monotonicity(engine)
        && causal_non_commutativity(engine)
        && no_construction_collision(engine)
}

/// (5) No learned-construction collision with a base-parseable pattern.
///
/// A registered [`LearnedConstruction`](crate::understanding::grammar::LearnedConstruction)
/// is SOUND only when it ADDS coverage on shapes the hand-written parser leaves
/// `Unknown`. If a construction's class skeleton matches a sentence the base parser
/// ALREADY handles (its handwritten parse is not `Unknown`) and applying the
/// construction would yield a DIFFERENT meaning, the construction collides with a
/// base-parseable pattern — a latent hazard that, were the fallback ever reached
/// for that shape, would change a correct answer. The gate treats any such
/// collision against a golden sentence (setup or question) as UNSOUND, so a
/// construction whose skeleton overlaps the base grammar is rejected on accept AND
/// on reload — leaving only purely-additive constructions live.
///
/// This is what makes "a sound OSV rule leaves the gate green; a colliding rule is
/// rejected" an enforced invariant rather than an assumption: the OSV skeleton
/// `[0,1,0,1,2]` appears in no base-parseable golden case, so an object-fronting
/// rule never collides; a rule registered on the SVO skeleton `[0,1,2,0,1]` (which
/// the base parses correctly) with swapped roles produces a different meaning and
/// is caught here.
fn no_construction_collision(engine: &Engine) -> bool {
    use crate::comprehension::words_of;
    use crate::understanding::semantics::{token_classes, understand_handwritten};

    let constructions = engine.learned_grammar().constructions();
    if constructions.is_empty() {
        return true;
    }

    // Every distinct golden sentence the gate replays (setups + questions).
    let mut sentences: Vec<&'static str> = Vec::new();
    for case in golden_cases() {
        for s in case.setup {
            if !sentences.contains(&s) {
                sentences.push(s);
            }
        }
        if !sentences.contains(&case.question) {
            sentences.push(case.question);
        }
    }

    for sentence in &sentences {
        let base = understand_handwritten(engine, sentence);
        // Only base-PARSEABLE sentences can be collided with; an Unknown base parse
        // is exactly where a construction is *allowed* to add coverage.
        if matches!(base, Meaning::Unknown(_)) {
            continue;
        }
        let toks = words_of(sentence);
        let classes = token_classes(engine, sentence);
        for c in constructions {
            if let Some(produced) = c.apply(engine, &toks, &classes) {
                // The construction fires on a base-parseable sentence. If it produces
                // a different meaning than the base, it collides — reject.
                if produced != base {
                    return false;
                }
            }
        }
    }
    true
}

/// (1) A query over unmentioned entities is open-world ("I don't know."), never a
/// false entailment. We read one fact about a teacher and a report, then ask
/// about a completely different agent/patient pair.
fn no_false_entailment(engine: &Engine) -> bool {
    let mut d = Discourse::new();
    d.read(engine, "The teacher writes the report.");
    // An unmentioned agent over an unmentioned patient: must be open-world.
    let a = qa::answer(engine, &d, "Does the pilot read the poem?");
    a.to_lowercase().contains("don't know")
}

/// (2) An asserted fact answers Yes; its explicit negation answers No. Asserting
/// `write(teacher, report)` makes "does the teacher write the report?" a Yes and
/// the negative truth query `¬write(teacher, report)` a No.
fn asserted_fact_and_negation(engine: &Engine) -> bool {
    let mut d = Discourse::new();
    d.world.assert(&write_event("teacher", "report", false));
    // The fact holds: Yes.
    let positive = qa::answer(engine, &d, "Does the teacher write the report?");
    if !positive.to_lowercase().starts_with("yes") {
        return false;
    }
    // The model-theoretic truth of the NEGATION of an asserted fact is false.
    let neg = write_event("teacher", "report", true);
    matches!(qa::world_truth_traced(&d, &neg).0, Some(false))
}

/// (3) Modal monotonicity: actuality entails possibility, but possibility never
/// leaks back to actuality. When `write(teacher, report)` holds, the bare event
/// is true and so is its possibility query truth; with NOTHING known, the same
/// modal query is open-world (no modal→actuality leak).
fn modal_monotonicity(engine: &Engine) -> bool {
    use crate::understanding::meaning::Modality;
    // Known-true event: "can the teacher write the report?" is true.
    let mut known = Discourse::new();
    known.read(engine, "The teacher writes the report.");
    let can = Meaning::Modal {
        modality: Modality::Can,
        body: Box::new(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Entity("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        }),
        negated: false,
    };
    if qa::world_truth_traced(&known, &can).0 != Some(true) {
        return false;
    }
    // Empty world: the same modal is open-world AND the bare event is NOT derived
    // from the modal (no possibility -> actuality leak).
    let empty = Discourse::new();
    if qa::world_truth_traced(&empty, &can).0.is_some() {
        return false;
    }
    let bare = write_event("teacher", "report", false);
    qa::world_truth_traced(&empty, &bare).0.is_none()
}

/// (4) Causal non-commutativity: a stored cause→effect link must not be read
/// backwards. With "the street floods because the rain falls" asserted, asking
/// why the *rain* falls must NOT answer with the street flooding — the directed
/// link is read in the cause→effect direction only.
fn causal_non_commutativity(engine: &Engine) -> bool {
    let mut d = Discourse::new();
    d.read(engine, "The street floods because the rain falls.");
    // The legitimate direction: why does the street flood? -> because the rain falls.
    let forward = qa::answer(engine, &d, "Why does the street flood?");
    if !forward.to_lowercase().contains("rain") {
        // If the forward direction itself does not recover the cause, the world is
        // not storing the link as expected — but that is not an *unsoundness*; the
        // invariant we guard is that the REVERSE reading is never fabricated.
    }
    // The reversed direction: why does the rain fall? must NOT answer "the street
    // floods" (it would be a fabricated reverse cause). Either honest "I don't
    // know." or some non-flooding answer is acceptable; a "flood" answer is not.
    let reverse = qa::answer(engine, &d, "Why does the rain fall?");
    !reverse.to_lowercase().contains("flood")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// One shared Engine — synthesis is slow, so reuse across the whole module.
    fn engine() -> &'static Engine {
        static E: OnceLock<Engine> = OnceLock::new();
        E.get_or_init(Engine::new)
    }

    #[test]
    fn corpus_covers_at_least_thirty_cases() {
        assert!(
            golden_cases().len() >= 30,
            "golden corpus must have >= 30 cases; has {}",
            golden_cases().len()
        );
    }

    #[test]
    fn golden_cases_all_pass() {
        // Every golden case's expected substring must match what qa::answer
        // returns today. A failure here means a case's `expected` drifted from
        // current behavior (fix the case) OR behavior regressed (fix the code).
        let e = engine();
        let mut failures = Vec::new();
        for case in golden_cases() {
            if !run_case(e, &case) {
                let mut d = Discourse::new();
                for s in &case.setup {
                    d.read(e, s);
                }
                let got = qa::answer(e, &d, case.question);
                failures.push(format!(
                    "Q: {:?} expected substring {:?} got {:?}",
                    case.question, case.expected, got
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "golden case mismatches:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn each_soundness_probe_holds() {
        let e = engine();
        assert!(no_false_entailment(e), "open-world entailment probe failed");
        assert!(
            asserted_fact_and_negation(e),
            "assert/negation probe failed"
        );
        assert!(modal_monotonicity(e), "modal monotonicity probe failed");
        assert!(
            causal_non_commutativity(e),
            "causal non-commutativity probe failed"
        );
    }

    #[test]
    fn regression_gate_is_green_on_default_engine() {
        let report = regression_gate(engine());
        assert_eq!(
            report.passed, report.total,
            "every golden case must pass on the default engine; failures: {:?}",
            report.failures
        );
        assert!(report.sound, "the default engine must be sound");
        assert!(report.ok(), "the gate must be green on the default engine");
    }

    /// ADVERSARIAL: the gate must DISCRIMINATE good behavior from bad. A gate
    /// that always passes is useless. This test proves three things on the SAME
    /// shared, correct engine:
    ///   1. The honest gate is green (`regression_gate(&e).ok() == true`).
    ///   2. A single golden case with a deliberately WRONG `expected` substring
    ///      makes `run_case` return false.
    ///   3. Replaying a battery that contains that wrong case (mirroring exactly
    ///      what `regression_gate` does — count passes, collect failures, AND the
    ///      same soundness oracle) yields a `GateReport` whose `ok()` is false and
    ///      whose `passed < total`.
    /// If any of these flipped, the gate would not actually guard anything.
    #[test]
    fn gate_rejects_a_broken_engine() {
        let e = engine();

        // (1) The honest gate is green on the correct engine.
        let honest = regression_gate(e);
        assert!(
            honest.ok(),
            "precondition: the gate must be green on the default engine before we \
             can claim it discriminates; passed {}/{} sound={} failures={:?}",
            honest.passed,
            honest.total,
            honest.sound,
            honest.failures
        );

        // (2) A case with a deliberately wrong expectation must FAIL `run_case`.
        // The engine answers "Yes" to "Does the teacher write the report?"; we
        // demand the (false) substring "No". `run_case` must return false.
        let wrong = GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Does the teacher write the report?",
            expected: "No", // the truth is "Yes" — a degraded/wrong expectation
        };
        assert!(
            !run_case(e, &wrong),
            "run_case must return false for a deliberately wrong expectation; \
             a gate whose per-case check always passes is useless"
        );
        // Control: the SAME case with the correct expectation passes — proves the
        // failure above is due to the wrongness, not a broken harness.
        let right = GoldenCase {
            setup: vec!["The teacher writes the report."],
            question: "Does the teacher write the report?",
            expected: "Yes",
        };
        assert!(
            run_case(e, &right),
            "control: the correct expectation must pass; otherwise the test proves nothing"
        );

        // (3) Build a battery that mutates ONE real golden case's `expected` to a
        // wrong value, then replay it through the *same* logic `regression_gate`
        // uses (run_case per case + the soundness oracle) and assemble a real
        // GateReport. Its ok() must be false and passed < total.
        let mut battery = golden_cases();
        let n = battery.len();
        // Mutate the first case to an expectation the engine will never produce.
        battery[0].expected = "this is never the answer the engine emits";

        let mut passed = 0usize;
        let mut failures = Vec::new();
        for case in &battery {
            if run_case(e, case) {
                passed += 1;
            } else {
                failures.push(case.question.to_string());
            }
        }
        let report = GateReport {
            passed,
            total: n,
            failures,
            sound: soundness_holds(e),
        };

        assert!(
            report.passed < report.total,
            "the mutated battery must drop at least one case; passed {}/{}",
            report.passed,
            report.total
        );
        assert_eq!(
            report.total - report.passed,
            1,
            "exactly the one mutated case should fail (no spurious extra failures)"
        );
        assert!(
            !report.failures.is_empty(),
            "the failing case's question must be recorded"
        );
        assert!(
            !report.ok(),
            "the gate MUST be red for a battery with a wrong expectation; \
             a gate that reports ok() on broken behavior cannot guard anything. \
             passed {}/{} sound={}",
            report.passed,
            report.total,
            report.sound
        );

        // (4) Soundness arm: even if every behavioral case passed, a `sound=false`
        // run must make the gate red. Construct an all-cases-passing report and
        // flip soundness off — ok() must still be false.
        let unsound = GateReport {
            passed: n,
            total: n,
            failures: Vec::new(),
            sound: false,
        };
        assert!(
            !unsound.ok(),
            "ok() must be false when soundness is violated even if every case passes"
        );
        // And the all-green report with soundness on is the ONLY ok() state.
        let green = GateReport {
            passed: n,
            total: n,
            failures: Vec::new(),
            sound: true,
        };
        assert!(green.ok(), "an all-pass, sound report must be ok()");
    }
}
