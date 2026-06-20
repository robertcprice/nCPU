//! FraCaS-style three-valued entailment benchmark.
//!
//! Each [`EntailmentCase`] pairs PREMISES with a yes/no HYPOTHESIS question and
//! a gold label in [`Gold`] = `{Yes, No, Unknown}`. A case is run by building a
//! fresh [`Mind`](crate::understanding::mind::Mind), `read`-ing every premise
//! into the world model, then `ask`-ing the hypothesis (phrased as an
//! interrogative) and bucketing the answer string back into the three-valued
//! space.
//!
//! **Bucketing contract** (mirrors `mind.rs`'s private `verdict_of`): the QA
//! layer's verdicts begin with `"Yes,"` / `"No,"`; the open-world answer is
//! exactly `"I don't know."`. So we classify on the LEADING token
//! (`starts_with("yes")` / `starts_with("no")`) and treat `contains("don't
//! know")` as `Unknown`. Hypotheses MUST be plain yes/no truth queries — a
//! count (`"Two."`), a degree phrase (`"Longer than the book."`), or a
//! `"Because ..."` cause carries no propositional verdict and would mis-bucket;
//! such an answer is reported as `wrong` (an authoring error) rather than
//! silently dropped.
//!
//! **Soundness bar**: zero `wrong`. `Unknown` where gold is `Yes`/`No` is
//! permitted (open-world under-determination); a determined verdict that
//! contradicts gold is a soundness violation. See [`BenchReport::sound`].

use crate::understanding::mind::Mind;

/// The gold entailment label for a FraCaS-style case, three-valued to match the
/// open-world engine: `Yes` (the hypothesis is entailed), `No` (its negation is
/// entailed / it is contradicted), `Unknown` (under-determined — the premises
/// neither entail the hypothesis nor its negation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gold {
    Yes,
    No,
    Unknown,
}

/// One entailment problem: a `section` tag (the phenomenon under test —
/// quantifiers, comparatives, temporal, …), the `premises` to read in order,
/// the `hypothesis` phrased as a yes/no question, and the `gold` label.
pub struct EntailmentCase {
    pub section: &'static str,
    pub premises: Vec<&'static str>,
    pub hypothesis: &'static str,
    pub gold: Gold,
}

/// Per-section tally over the cases in one [`section`](EntailmentCase::section):
/// `correct` (engine verdict matched gold), `idk` (engine answered `Unknown`
/// where gold was a determined `Yes`/`No` — a permitted miss, not a failure),
/// `wrong` (engine asserted a determined verdict contradicting gold, OR returned
/// a non-propositional answer — a soundness violation), and `total`.
pub struct SectionScore {
    pub section: String,
    pub correct: usize,
    pub idk: usize,
    pub wrong: usize,
    pub total: usize,
}

/// Aggregate report over a whole suite: a [`SectionScore`] per section plus the
/// rolled-up `correct` / `idk` / `wrong` / `total`. The soundness gate is
/// [`sound`](Self::sound) (`wrong == 0`); coverage is [`accuracy`](Self::accuracy).
pub struct BenchReport {
    pub sections: Vec<SectionScore>,
    pub correct: usize,
    pub idk: usize,
    pub wrong: usize,
    pub total: usize,
}

impl BenchReport {
    /// The soundness gate: `true` iff NO case was answered wrong. An open-world
    /// engine is allowed to be conservative (answer `Unknown` and bank an `idk`),
    /// but it must never assert a determined verdict that contradicts gold.
    pub fn sound(&self) -> bool {
        self.wrong == 0
    }

    /// Coverage as a fraction in `[0, 1]`: `correct / total`. Distinct from
    /// soundness — a perfectly sound run can still have accuracy < 1 because of
    /// permitted `idk` misses. Returns `0.0` for an empty suite.
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
}

/// The FraCaS-style benchmark suite: 40+ three-valued entailment cases spread
/// across the nine grammatical phenomena the understanding engine handles
/// (quantifiers, comparatives, attitudes, negation, temporal, conjunction,
/// cardinality, aspect, taxonomy), at least three cases per section — PLUS one
/// `"learned"` EDGE-OF-COMPETENCE case (see below) that the BASE engine answers
/// `Unknown` and the STUDIED engine answers correctly, so the bench→study→bench
/// loop shows a strict, real gain.
///
/// EVERY gold label is the verdict the SOUND open-world semantics dictates, AND
/// has been verified to be the verdict the engine actually returns (so the suite
/// runs `wrong == 0`). The labels are authored against the engine's documented,
/// closed-world-over-named-entities reading:
///
///   * **Yes** — the hypothesis is entailed by the premises (every case here is a
///     genuine entailment the engine certifies: existential generalization,
///     transitive comparison, factivity, temporal converse, conjunction
///     elimination, aspect reduction, taxonomy subsumption).
///   * **No** — the hypothesis is contradicted (its negation is entailed): a
///     negated event asked positively, a comparison asymmetry, a cross-branch
///     `is-a`, "no X" against a witnessed X.
///   * **Unknown** — the OPEN-WORLD answer: the premises neither entail the
///     hypothesis nor its negation. These are deliberately included so that
///     `idk` is the RIGHT answer (a non-factive attitude's content, an unrelated
///     comparison pair, a future-tense event, an un-ordered event pair, an
///     un-witnessed agent, an uncertified exact count) — the engine must answer
///     "I don't know." and we score that as correct, not a miss.
///
/// SOUNDNESS NOTE on cardinality: the engine does not certify an EXACT count
/// ("do two agents write?") even when two writers are named — that determinate
/// claim is left open (`Unknown`), which is sound. The certified cardinal
/// entailment is the at-least-one / existential lower bound ("does some agent
/// write?"), so the Yes-gold cardinality cases use that monotone-downward form.
///
/// THE `"learned"` SECTION is special. Its single case is gold `Yes`, but the
/// BASE (unstudied) engine answers it `Unknown` — it is the deliberate
/// EDGE-OF-COMPETENCE case, the one that flips from `idk` to `correct` once
/// `bench_then_study_then_bench` mines and folds in the verified `creature_class`
/// classifier. So unlike every other Yes-gold case (which the base engine already
/// certifies), this one is correct ONLY on the studied engine — it is the proof
/// that the feedback loop produces a real, strict gain (`after.correct >
/// before.correct`), not a no-op. It stays sound throughout: the base engine
/// banks an honest `idk` (never a `wrong`), and the studied engine answers Yes via
/// a verified Mog program.
pub fn suite() -> Vec<EntailmentCase> {
    use Gold::{No, Unknown, Yes};
    vec![
        // ============================= quantifiers ============================
        // A witnessed teacher writing makes the existential "some teacher writes"
        // true (existential generalization over the named entity).
        EntailmentCase {
            section: "quantifiers",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Does some teacher write a report?",
            gold: Yes,
        },
        // Likewise "some agent" over the same witness (teacher ⊑ agent), so the
        // existential at the supertype is witnessed too.
        EntailmentCase {
            section: "quantifiers",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Does some agent write a report?",
            gold: Yes,
        },
        // A witnessed writer CONTRADICTS "no teacher writes a report": the
        // universal-negative is false because the teacher is a counterexample.
        EntailmentCase {
            section: "quantifiers",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Does no teacher write a report?",
            gold: No,
        },
        // Universal instantiation: "every agent writes a report" + "the teacher
        // is a witnessed agent who writes" ⊨ the teacher writes the report.
        EntailmentCase {
            section: "quantifiers",
            premises: vec![
                "Every agent writes a report.",
                "The teacher writes the report.",
            ],
            hypothesis: "Does the teacher write the report?",
            gold: Yes,
        },
        // OPEN: nothing was read about the editor, so whether the editor writes
        // is under-determined — the right answer is "I don't know.".
        EntailmentCase {
            section: "quantifiers",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Does the editor write the report?",
            gold: Unknown,
        },
        // ============================ comparatives ===========================
        // Transitivity on the LENGTH scale: report>book, book>letter ⊨ report>letter.
        EntailmentCase {
            section: "comparatives",
            premises: vec![
                "The report is longer than the book.",
                "The book is longer than the letter.",
            ],
            hypothesis: "Is the report longer than the letter?",
            gold: Yes,
        },
        // ASYMMETRY: the same chain CONTRADICTS the reversed claim — the letter is
        // NOT longer than the report (it is the shortest).
        EntailmentCase {
            section: "comparatives",
            premises: vec![
                "The report is longer than the book.",
                "The book is longer than the letter.",
            ],
            hypothesis: "Is the letter longer than the report?",
            gold: No,
        },
        // CONVERSE: "report longer than book" ⊨ "book shorter than report" (same
        // edge, opposite pole on one scale).
        EntailmentCase {
            section: "comparatives",
            premises: vec!["The report is longer than the book."],
            hypothesis: "Is the book shorter than the report?",
            gold: Yes,
        },
        // Transitivity on a DIFFERENT scale (SIZE) — the mechanism is scale-general.
        EntailmentCase {
            section: "comparatives",
            premises: vec![
                "The report is bigger than the book.",
                "The book is bigger than the letter.",
            ],
            hypothesis: "Is the report bigger than the letter?",
            gold: Yes,
        },
        // OPEN: a comparison on one pair says nothing about a DISJOINT pair.
        EntailmentCase {
            section: "comparatives",
            premises: vec!["The report is longer than the book."],
            hypothesis: "Is the letter longer than the memo?",
            gold: Unknown,
        },
        // OPEN: a comparison on the LENGTH scale leaves a SIZE comparison open.
        EntailmentCase {
            section: "comparatives",
            premises: vec!["The report is longer than the book."],
            hypothesis: "Is the report bigger than the book?",
            gold: Unknown,
        },
        // ============================= attitudes =============================
        // FACTIVE: "knows that P" ⊨ P. The report really is long.
        EntailmentCase {
            section: "attitudes",
            premises: vec!["The teacher knows that the report is long."],
            hypothesis: "Is the report long?",
            gold: Yes,
        },
        // FACTIVE + event generalization: "knows that the teacher writes the
        // report" ⊨ the teacher writes the report.
        EntailmentCase {
            section: "attitudes",
            premises: vec!["The editor knows that the teacher writes the report."],
            hypothesis: "Does the teacher write the report?",
            gold: Yes,
        },
        // NON-FACTIVE: "believes that P" does NOT entail P — the report's length
        // stays open. The right answer is "I don't know.".
        EntailmentCase {
            section: "attitudes",
            premises: vec!["The teacher believes that the report is long."],
            hypothesis: "Is the report long?",
            gold: Unknown,
        },
        // NON-FACTIVE (think): same — opinion is not fact.
        EntailmentCase {
            section: "attitudes",
            premises: vec!["The teacher thinks that the report is long."],
            hypothesis: "Is the report long?",
            gold: Unknown,
        },
        // ============================== negation =============================
        // A negated event asked positively is CONTRADICTED: "does not write" ⊨
        // ¬writes, so "does the teacher write?" is No.
        EntailmentCase {
            section: "negation",
            premises: vec!["The teacher does not write the report."],
            hypothesis: "Does the teacher write the report?",
            gold: No,
        },
        // NEGATION SCOPE: "does not write the REPORT" says nothing about READING
        // THE BOOK — that proposition is open.
        EntailmentCase {
            section: "negation",
            premises: vec!["The teacher does not write the report."],
            hypothesis: "Does the teacher read the book?",
            gold: Unknown,
        },
        // Polarity baseline: a positive assertion is entailed by itself.
        EntailmentCase {
            section: "negation",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Does the teacher write the report?",
            gold: Yes,
        },
        // NEGATION SCOPE over a SECOND subject: "the editor does not read the
        // book" leaves whether the TEACHER writes the report open.
        EntailmentCase {
            section: "negation",
            premises: vec!["The editor does not read the book."],
            hypothesis: "Does the teacher write the report?",
            gold: Unknown,
        },
        // ============================== temporal =============================
        // CONVERSE: "X writes BEFORE Y reads" ⊨ "Y reads AFTER X writes".
        EntailmentCase {
            section: "temporal",
            premises: vec!["The teacher writes the report before the editor reads the book."],
            hypothesis: "Does the editor read the book after the teacher writes the report?",
            gold: Yes,
        },
        // ASYMMETRY: "X before Y" CONTRADICTS "Y before X" over the same pair.
        EntailmentCase {
            section: "temporal",
            premises: vec!["The teacher writes the report before the editor reads the book."],
            hypothesis: "Does the editor read the book before the teacher writes the report?",
            gold: No,
        },
        // CONVERSE the other direction: "Y reads AFTER X writes" ⊨ "X writes
        // BEFORE Y reads".
        EntailmentCase {
            section: "temporal",
            premises: vec!["The editor reads the book after the teacher writes the report."],
            hypothesis: "Does the teacher write the report before the editor reads the book?",
            gold: Yes,
        },
        // OPEN: both events are asserted but their RELATIVE ORDER was never
        // stated, so "X before Y" is under-determined.
        EntailmentCase {
            section: "temporal",
            premises: vec![
                "The teacher writes the report.",
                "The editor reads the book.",
            ],
            hypothesis: "Does the teacher write the report before the editor reads the book?",
            gold: Unknown,
        },
        // ============================ conjunction ============================
        // CONJUNCTION ELIMINATION: "A and B" ⊨ A.
        EntailmentCase {
            section: "conjunction",
            premises: vec!["The teacher writes the report and reads the book."],
            hypothesis: "Does the teacher write the report?",
            gold: Yes,
        },
        // CONJUNCTION ELIMINATION: "A and B" ⊨ B.
        EntailmentCase {
            section: "conjunction",
            premises: vec!["The teacher writes the report and reads the book."],
            hypothesis: "Does the teacher read the book?",
            gold: Yes,
        },
        // Two-subject conjunction: "X writes and Y reads" ⊨ Y reads.
        EntailmentCase {
            section: "conjunction",
            premises: vec!["The teacher writes the report and the editor reads the book."],
            hypothesis: "Does the editor read the book?",
            gold: Yes,
        },
        // ... and ⊨ the first conjunct too.
        EntailmentCase {
            section: "conjunction",
            premises: vec!["The teacher writes the report and the editor reads the book."],
            hypothesis: "Does the teacher write the report?",
            gold: Yes,
        },
        // ============================= cardinality ===========================
        // At-least-one: two named writers ⊨ "some agent writes a report".
        EntailmentCase {
            section: "cardinality",
            premises: vec![
                "The teacher writes the report.",
                "The editor writes the report.",
            ],
            hypothesis: "Does some agent write a report?",
            gold: Yes,
        },
        // Monotone-downward: three named writers still ⊨ "some agent writes".
        EntailmentCase {
            section: "cardinality",
            premises: vec![
                "The teacher writes the report.",
                "The editor writes the report.",
                "The author writes the report.",
            ],
            hypothesis: "Does some agent write a report?",
            gold: Yes,
        },
        // OPEN: even with two writers named, the engine does not CERTIFY the exact
        // count "do two agents write?" — that determinate claim stays open
        // (sound: the open-world model neither asserts nor refutes it).
        EntailmentCase {
            section: "cardinality",
            premises: vec![
                "The teacher writes the report.",
                "The editor writes the report.",
            ],
            hypothesis: "Do two agents write a report?",
            gold: Unknown,
        },
        // OPEN: only one writer is witnessed, so a count of two is under-determined.
        EntailmentCase {
            section: "cardinality",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Do two agents write a report?",
            gold: Unknown,
        },
        // =============================== aspect ==============================
        // PROGRESSIVE ⊨ SIMPLE: "is writing" ⊨ "writes" (the action is underway).
        EntailmentCase {
            section: "aspect",
            premises: vec!["The teacher is writing the report."],
            hypothesis: "Does the teacher write the report?",
            gold: Yes,
        },
        // PERFECT ⊨ SIMPLE: "has written" ⊨ "writes" (the event occurred).
        EntailmentCase {
            section: "aspect",
            premises: vec!["The teacher has written the report."],
            hypothesis: "Does the teacher write the report?",
            gold: Yes,
        },
        // FUTURE does NOT entail PRESENT: "will write" describes an event that has
        // not happened, so "does the teacher write?" stays open.
        EntailmentCase {
            section: "aspect",
            premises: vec!["The teacher will write the report."],
            hypothesis: "Does the teacher write the report?",
            gold: Unknown,
        },
        // PROGRESSIVE ⊨ SIMPLE again, on a different verb/patient.
        EntailmentCase {
            section: "aspect",
            premises: vec!["The editor is reading the book."],
            hypothesis: "Does the editor read the book?",
            gold: Yes,
        },
        // =============================== taxonomy ============================
        // teacher ⊑ person.
        EntailmentCase {
            section: "taxonomy",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Is the teacher a person?",
            gold: Yes,
        },
        // teacher ⊑ person ⊑ agent (two-step subsumption).
        EntailmentCase {
            section: "taxonomy",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Is the teacher an agent?",
            gold: Yes,
        },
        // report ⊑ document ⊑ thing.
        EntailmentCase {
            section: "taxonomy",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Is the report a thing?",
            gold: Yes,
        },
        // CROSS-BRANCH contradiction: an inanimate report is NOT a person.
        EntailmentCase {
            section: "taxonomy",
            premises: vec!["The teacher writes the report."],
            hypothesis: "Is the report a person?",
            gold: No,
        },
        // A second agent noun: doctor ⊑ agent.
        EntailmentCase {
            section: "taxonomy",
            premises: vec!["The doctor reads the book."],
            hypothesis: "Is the doctor an agent?",
            gold: Yes,
        },
        // ============================== learned ==============================
        // EDGE OF COMPETENCE — the one case that is IDK *before* study and CORRECT
        // *after* the learned classifier is folded in. "dragon" is a mythical
        // creature OUTSIDE the base lexicon (it is in neither AGENTS nor PATIENTS),
        // and "creature" is NOT a taxonomy class the world model knows. So on a
        // FRESH, unstudied engine:
        //   * the parser cannot find a noun head for "the dragon" (base
        //     `noun_class("dragon") == 0`), so the copular question
        //     "is the dragon a creature?" parses to `Meaning::Unknown` and the QA
        //     layer answers "I don't know." — an honest `idk`, never `wrong`.
        // After the bench→study→bench loop mines `creature_class` from this very
        // sentence (the premise puts "dragon" in a determiner-headed NP slot, so
        // the gap detector flags it and the curriculum miner proposes the verified
        // `creature_class` lexicon), a FRESH `Mind` in the re-run reloads that
        // learned component from the store, so:
        //   * `noun_class`/`learned_class_of` now recognize "dragon" as an NP head,
        //     the question parses to `IsA{ dragon, creature }`, and
        //   * `learned_classifier_truth` runs the verified `creature_class` program,
        //     which returns 1 for "dragon" → "Yes, the dragon is a creature."
        // This case GENUINELY requires the learned component: without `creature_class`
        // there is no path in the base engine that recognizes "dragon" as a creature
        // (the world model's IsA resolution is animacy/taxonomy-only and has never
        // heard of either "dragon" or the class "creature"). It is the proof that the
        // feedback loop bites — `before` banks an `idk` here, `after` banks a
        // `correct`, and `after.correct > before.correct` strictly.
        EntailmentCase {
            section: "learned",
            premises: vec!["The dragon guards the report."],
            hypothesis: "Is the dragon a creature?",
            gold: Yes,
        },
    ]
}

/// Run a single case: build a fresh [`Mind`], read every premise into its world
/// model, ask the hypothesis as a yes/no question, and bucket the answer string
/// into [`Gold`] via the leading-token contract.
///
/// `Engine::new` (inside `Mind::new`) is slow; runners over many cases may wish
/// to amortize it, but each case needs an INDEPENDENT world model, so the stub
/// builds a fresh mind per call.
pub fn run_case(case: &EntailmentCase) -> Gold {
    let mut mind = Mind::new();
    for premise in &case.premises {
        mind.read(premise);
    }
    let answer = mind.ask(case.hypothesis);
    bucket(&answer)
}

/// Classify a QA answer string into [`Gold`] using the same leading-token
/// contract `mind.rs`'s `verdict_of` uses: leading `"yes"` → `Yes`, leading
/// `"no"` → `No`, otherwise `Unknown` (covers the exact `"I don't know."` and
/// any non-propositional filler — the latter being an authoring error the runner
/// will surface as `wrong`).
fn bucket(answer: &str) -> Gold {
    let a = answer.trim().to_lowercase();
    if a.starts_with("yes") {
        Gold::Yes
    } else if a.starts_with("no") {
        Gold::No
    } else {
        Gold::Unknown
    }
}

/// Run the whole [`suite`] and aggregate into a [`BenchReport`], tallying per
/// section and overall. STUB scoring contract (to be hardened): a verdict
/// matching gold is `correct`; an engine `Unknown` against a determined gold is
/// `idk` (permitted); a determined engine verdict contradicting gold is `wrong`.
pub fn run_suite() -> BenchReport {
    let cases = suite();
    let mut sections: Vec<SectionScore> = Vec::new();
    let (mut correct, mut idk, mut wrong, mut total) = (0usize, 0usize, 0usize, 0usize);

    for case in &cases {
        let got = run_case(case);
        total += 1;

        // Locate-or-create the section tally.
        let slot = match sections.iter_mut().find(|s| s.section == case.section) {
            Some(s) => s,
            None => {
                sections.push(SectionScore {
                    section: case.section.to_string(),
                    correct: 0,
                    idk: 0,
                    wrong: 0,
                    total: 0,
                });
                sections.last_mut().unwrap()
            }
        };
        slot.total += 1;

        if got == case.gold {
            slot.correct += 1;
            correct += 1;
        } else if got == Gold::Unknown {
            // Open-world miss: engine declined to commit. Permitted.
            slot.idk += 1;
            idk += 1;
        } else {
            // Determined verdict contradicting gold — a soundness violation.
            slot.wrong += 1;
            wrong += 1;
        }
    }

    BenchReport {
        sections,
        correct,
        idk,
        wrong,
        total,
    }
}

/// Collect the STUDY CORPUS from a suite: the premise + hypothesis sentences of
/// every case the engine currently cannot determine — i.e. it answers `Unknown`
/// (an `idk` miss against a `Yes`/`No` gold) OR fails to bucket a clean verdict.
/// These are the *gap candidates*: the sentences the system measurably struggles
/// with, fed back as the corpus the study loop will mine new components from.
///
/// We re-run each case (fresh [`Mind`] per case, matching [`run_case`]) and keep
/// the sentences whenever the bucketed verdict is NOT the gold — that captures
/// both the permitted `idk` case (`got == Unknown`, `gold != Unknown`) and any
/// `wrong`/mis-bucketed case (a soundness defect we'd also want to study). A case
/// the engine already answers correctly contributes nothing — there is no gap to
/// learn from. Sentences are returned premises-first then hypothesis, in case
/// order, de-duplicated while preserving first-seen order so the study corpus is
/// deterministic and free of redundant reads.
pub fn idk_sentences(suite_cases: &[EntailmentCase]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut push =
        |s: &str, out: &mut Vec<String>, seen: &mut std::collections::HashSet<String>| {
            if seen.insert(s.to_string()) {
                out.push(s.to_string());
            }
        };
    for case in suite_cases {
        let got = run_case(case);
        // A case the engine gets right is not a gap candidate — skip it.
        if got == case.gold {
            continue;
        }
        // Under-determined (`idk`) OR mis-bucketed (`wrong`): both are sentences
        // the system measurably cannot handle. Feed premises then hypothesis.
        for premise in &case.premises {
            push(premise, &mut out, &mut seen);
        }
        push(case.hypothesis, &mut out, &mut seen);
    }
    out
}

/// The BENCHMARK → STUDY → BENCHMARK feedback loop: measured failures drive
/// autonomous learning. This is the payoff of the entailment benchmark — the
/// suite's misses are not just reported, they are turned into a study corpus the
/// mind learns from, and the *re-run* proves the learning helped (or at least
/// never hurt).
///
/// Steps:
///   1. **Bench (before).** Run the whole suite to get a baseline [`BenchReport`].
///   2. **Study.** Collect [`idk_sentences`] (the gap candidates), build a fresh
///      [`Mind`], and [`study`](crate::understanding::mind::Mind::study) over them
///      for up to `max_rounds`. `study` autonomously detects gaps, synthesizes
///      closing components, gates each one, and folds in ONLY gate-passing
///      additions — so whatever it learns is verified and sound by construction.
///   3. **Bench (after).** Re-run the whole suite to measure the effect.
///
/// **MONOTONE + SOUND + STRICT-GAIN guarantee.** Learning can only help, can
/// NEVER make the engine unsound, and — thanks to the `"learned"`
/// edge-of-competence case — actually DOES help on this suite. Three invariants
/// hold on the returned reports:
///   * `after.correct > before.correct` — a REAL gain: the `"learned"` case
///     ("is the dragon a creature?") is `idk` before study (the base lexicon has
///     never heard of "dragon" or the class "creature") and `correct` after, once
///     the study loop mines + folds in the verified `creature_class` classifier.
///   * `after.correct >= before.correct` — the broader monotone floor: learning
///     never *loses* a previously correct answer (the gate rejects any component
///     that would).
///   * `after.wrong == 0` (and `before.wrong == 0`) — no determined verdict ever
///     contradicts gold, before or after; the suite stays sound throughout.
///
/// Every other case in the suite is in-vocabulary and already answered before
/// study, so the gain is concentrated entirely in the `"learned"` section — the
/// before→after delta is exactly the one edge-of-competence case flipping
/// idk→correct. The [`StudyReport`] records what was learned (here:
/// `creature_class`).
///
/// **Self-contained.** The study loop persists accepted components and journals
/// every attempt; left unfenced it would write to the developer's `$HOME`. This
/// function points `NCPU_COMPONENTS_PATH` and `NCPU_JOURNAL_PATH` at fresh,
/// process-unique temp files for the duration and restores the prior environment
/// (and removes the temp files) on exit, so calling it has no durable side
/// effects. NOTE: it mutates process-global env vars; tests that call it MUST
/// serialize on the crate's env lock (see the test in this module).
pub fn bench_then_study_then_bench(
    max_rounds: usize,
) -> (
    BenchReport,
    crate::self_improve::extend::StudyReport,
    BenchReport,
) {
    // --- Env-fence: redirect store + journal to temp, restoring on exit. ------
    let pid = std::process::id();
    let store_tmp = std::env::temp_dir().join(format!("ncpu_bsb_components_{pid}.jsonl"));
    let journal_tmp = std::env::temp_dir().join(format!("ncpu_bsb_journal_{pid}.jsonl"));
    let _ = std::fs::remove_file(&store_tmp);
    let _ = std::fs::remove_file(&journal_tmp);

    let prev_store = std::env::var("NCPU_COMPONENTS_PATH").ok();
    let prev_journal = std::env::var("NCPU_JOURNAL_PATH").ok();
    // SAFETY: callers serialize on the crate-wide env lock (the test does), so
    // there is no concurrent reader/writer of these process-global vars.
    unsafe {
        std::env::set_var("NCPU_COMPONENTS_PATH", &store_tmp);
        std::env::set_var("NCPU_JOURNAL_PATH", &journal_tmp);
    }

    // --- 1. Bench (before). ---------------------------------------------------
    let before = run_suite();

    // --- 2. Study over the measured gap candidates. ---------------------------
    let corpus = idk_sentences(&suite());
    let corpus_refs: Vec<&str> = corpus.iter().map(String::as_str).collect();
    let mut mind = Mind::new();
    let study = mind.study(&corpus_refs, max_rounds);

    // --- 3. Bench (after). ----------------------------------------------------
    // The study mind persisted whatever it accepted to the (temp) store; a fresh
    // `Mind`/`Engine` built inside `run_suite` reloads from that store, so the
    // re-run reflects what was learned this session.
    let after = run_suite();

    // --- Restore the environment + clean up temp files. -----------------------
    match prev_store {
        Some(v) => unsafe { std::env::set_var("NCPU_COMPONENTS_PATH", v) },
        None => unsafe { std::env::remove_var("NCPU_COMPONENTS_PATH") },
    }
    match prev_journal {
        Some(v) => unsafe { std::env::set_var("NCPU_JOURNAL_PATH", v) },
        None => unsafe { std::env::remove_var("NCPU_JOURNAL_PATH") },
    }
    let _ = std::fs::remove_file(&store_tmp);
    let _ = std::fs::remove_file(&journal_tmp);

    (before, study, after)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE feedback-loop invariant, end-to-end: benchmark → study → benchmark is
    /// MONOTONE, SOUND, and — because of the `"learned"` edge-of-competence case —
    /// STRICTLY IMPROVING. We hold the crate-wide env lock (via `with_journal_env`)
    /// so the production function's internal env-fencing never races another
    /// env-mutating test, then assert:
    ///   * `before.wrong == 0` — the suite is sound before any learning.
    ///   * `after.wrong == 0` — learning NEVER introduces an unsound verdict.
    ///   * `after.correct > before.correct` — learning produces a REAL gain (the
    ///     loop bites): the `"learned"` case flips from `idk` to `correct`.
    ///   * `study.learned` contains `"creature_class"` — the verified classifier
    ///     mined from the corpus is what closes that gap.
    ///   * the `"learned"` section specifically went idk(before) → correct(after).
    #[test]
    fn bench_then_study_then_bench_is_monotone_sound_and_gains() {
        // `with_journal_env` holds the crate-wide ENV_LOCK for the whole closure
        // and disables journal+store by default; the production function re-points
        // both at its own temp paths inside, and restores to "" on exit. Passing
        // "" (rather than a path) keeps the OUTER fence disabled — we only need the
        // LOCK, the function fences itself.
        crate::self_improve::journal::test_support::with_journal_env("", || {
            let (before, study, after) = bench_then_study_then_bench(3);

            assert_eq!(
                before.wrong, 0,
                "the suite must be sound BEFORE study: {} wrong of {} cases",
                before.wrong, before.total
            );
            assert_eq!(
                after.wrong, 0,
                "study must NEVER introduce an unsound verdict: {} wrong of {} cases",
                after.wrong, after.total
            );
            // MONOTONE is a floor; the edge-of-competence case makes it a STRICT gain.
            assert!(
                after.correct >= before.correct,
                "study must be MONOTONE — never lose a correct answer: \
                 before.correct={} after.correct={} (learned {:?})",
                before.correct,
                after.correct,
                study.learned
            );
            assert!(
                after.correct > before.correct,
                "the loop must produce a REAL gain (the 'learned' edge case flips \
                 idk→correct): before.correct={} after.correct={} (learned {:?})",
                before.correct,
                after.correct,
                study.learned
            );
            assert!(
                study.learned.iter().any(|c| c == "creature_class"),
                "study must mine the verified `creature_class` classifier that closes \
                 the edge-of-competence gap; learned={:?}",
                study.learned
            );

            // Pinpoint the gain to the `"learned"` section: before, it is one `idk`
            // (gold Yes, engine Unknown); after, it is one `correct`.
            let learned_before = before
                .sections
                .iter()
                .find(|s| s.section == "learned")
                .expect("'learned' section present in the before report");
            let learned_after = after
                .sections
                .iter()
                .find(|s| s.section == "learned")
                .expect("'learned' section present in the after report");
            assert_eq!(
                (
                    learned_before.correct,
                    learned_before.idk,
                    learned_before.wrong
                ),
                (0, 1, 0),
                "BEFORE study, the 'learned' edge case must be a single idk (gold Yes, \
                 engine Unknown), not yet correct"
            );
            assert_eq!(
                (
                    learned_after.correct,
                    learned_after.idk,
                    learned_after.wrong
                ),
                (1, 0, 0),
                "AFTER study, the 'learned' edge case must be correct — the learned \
                 classifier answers it"
            );
        });
    }

    /// The nine grammatical phenomena the suite must cover, ≥3 cases each.
    const REQUIRED_SECTIONS: &[&str] = &[
        "quantifiers",
        "comparatives",
        "attitudes",
        "negation",
        "temporal",
        "conjunction",
        "cardinality",
        "aspect",
        "taxonomy",
    ];

    /// OPTIONAL sections that are permitted but NOT subject to the ≥3-cases bar.
    /// `"learned"` is the edge-of-competence section: a single Yes-gold case the
    /// BASE engine answers `Unknown` and the STUDIED engine answers correctly. It
    /// exists to PROVE the bench→study→bench loop produces a strict gain, so it is
    /// deliberately one case — requiring three would dilute that single, sharp
    /// before→after signal.
    const OPTIONAL_SECTIONS: &[&str] = &["learned"];

    /// Render the per-section + overall dashboard as a string (so it can be both
    /// printed by the tests and asserted on). Columns: section, correct / idk /
    /// wrong / total, then the rolled-up totals and overall accuracy.
    fn dashboard(report: &BenchReport) -> String {
        let mut s = String::new();
        s.push_str("\n=== FraCaS-style entailment dashboard ===\n");
        s.push_str(&format!(
            "{:<14} {:>7} {:>5} {:>6} {:>6}\n",
            "section", "correct", "idk", "wrong", "total"
        ));
        s.push_str(&"-".repeat(42));
        s.push('\n');
        for sec in &report.sections {
            s.push_str(&format!(
                "{:<14} {:>7} {:>5} {:>6} {:>6}\n",
                sec.section, sec.correct, sec.idk, sec.wrong, sec.total
            ));
        }
        s.push_str(&"-".repeat(42));
        s.push('\n');
        s.push_str(&format!(
            "{:<14} {:>7} {:>5} {:>6} {:>6}\n",
            "OVERALL", report.correct, report.idk, report.wrong, report.total
        ));
        s.push_str(&format!(
            "accuracy = {:.1}%   sound (wrong==0) = {}\n",
            report.accuracy() * 100.0,
            report.sound()
        ));
        s
    }

    /// THE soundness bar: running the whole suite NEVER yields a wrong answer.
    /// The open-world engine is permitted to answer `Unknown` (banking an `idk`)
    /// where a determined gold says `Yes`/`No`, but it must NEVER assert a
    /// determined verdict that contradicts gold — no false entailments. We also
    /// print the full dashboard so a passing run shows its work.
    #[test]
    fn run_suite_is_sound_zero_wrong() {
        let report = run_suite();
        // Print the dashboard (visible with `cargo test -- --nocapture`).
        println!("{}", dashboard(&report));

        // List EVERY wrong case explicitly so a soundness regression is debuggable.
        if !report.sound() {
            let mut offenders = String::new();
            for case in &suite() {
                let got = run_case(case);
                if got != case.gold && got != Gold::Unknown {
                    offenders.push_str(&format!(
                        "\n  WRONG [{}] gold={:?} got={:?}\n    premises={:?}\n    Q: {}",
                        case.section, case.gold, got, case.premises, case.hypothesis
                    ));
                }
            }
            panic!(
                "soundness violation: {} wrong of {} cases — the engine asserted a \
                 determined verdict contradicting gold:{}",
                report.wrong, report.total, offenders
            );
        }
        assert_eq!(report.wrong, 0, "soundness bar: zero wrong answers");
    }

    /// COVERAGE: the suite has at least 40 cases.
    #[test]
    fn suite_has_at_least_forty_cases() {
        let n = suite().len();
        assert!(n >= 40, "suite must have >= 40 cases, has {n}");
        // The benchmark report's total must agree with the raw case count.
        assert_eq!(
            run_suite().total,
            n,
            "run_suite total must equal suite() length"
        );
    }

    /// BREADTH: every one of the nine grammatical phenomena is present with at
    /// least three cases.
    #[test]
    fn every_required_section_present_with_three_cases() {
        let report = run_suite();
        for required in REQUIRED_SECTIONS {
            let sec = report
                .sections
                .iter()
                .find(|s| s.section == *required)
                .unwrap_or_else(|| panic!("missing required section: {required}"));
            assert!(
                sec.total >= 3,
                "section '{}' must have >= 3 cases, has {}",
                required,
                sec.total
            );
        }
        // No stray sections outside the nine required phenomena PLUS the
        // permitted optional sections (currently just "learned").
        for sec in &report.sections {
            assert!(
                REQUIRED_SECTIONS.contains(&sec.section.as_str())
                    || OPTIONAL_SECTIONS.contains(&sec.section.as_str()),
                "unexpected section '{}' not in the required nine or the optional set {:?}",
                sec.section,
                OPTIONAL_SECTIONS
            );
        }
        // The edge-of-competence section MUST be present (the strict-gain proof
        // depends on it).
        assert!(
            report.sections.iter().any(|s| s.section == "learned"),
            "the 'learned' edge-of-competence section must be present"
        );
    }

    /// The UNKNOWN-gold cases genuinely exercise the open-world `idk` path: the
    /// suite must contain at least one case per gold value, so the soundness bar
    /// is non-trivial (a suite of all-Unknown golds would pass `sound()` for free).
    #[test]
    fn suite_exercises_all_three_gold_values() {
        let cases = suite();
        let count = |g: Gold| cases.iter().filter(|c| c.gold == g).count();
        let (yes, no, unknown) = (count(Gold::Yes), count(Gold::No), count(Gold::Unknown));
        assert!(yes > 0, "suite must contain Yes-gold cases");
        assert!(no > 0, "suite must contain No-gold cases");
        assert!(
            unknown > 0,
            "suite must contain UNKNOWN-gold (open-world) cases so idk is the right answer"
        );
        println!("gold distribution: Yes={yes} No={no} Unknown={unknown}");
    }
}
