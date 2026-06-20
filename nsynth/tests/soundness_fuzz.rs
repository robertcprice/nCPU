//! DETERMINISTIC SOUNDNESS FUZZER for the comprehension engine.
//!
//! Goal: hammer the read -> ask pipeline (`Discourse::read` + `qa::answer`,
//! the same end-to-end English path the `Mind` exposes) with many randomly
//! generated small worlds + queries and assert it NEVER makes a false claim.
//!
//! This is a property-based test driven by a SEEDED linear-congruential
//! generator (no external `rand` crate — fully reproducible). One `Engine` is
//! built ONCE and reused across every iteration (synthesis is slow); only the
//! `Discourse` (the world) is fresh per iteration.
//!
//! Three soundness properties are asserted across >= 200 seeded iterations:
//!
//!   (a) ASSERTED-FACT RECALL — every fact read into the world answers "Yes" to
//!       its own question (or "No" if the fact was read NEGATED); its
//!       polarity-flipped question answers the opposite. The engine must
//!       remember and correctly report exactly what it was told.
//!
//!   (b) NEGATION CONSISTENCY — for ANY generated query Q (asserted or not), the
//!       system never answers BOTH "Q?" = Yes AND "not Q?" = Yes. It must not
//!       contradict itself.
//!
//!   (c) NO-SPURIOUS-ENTAILMENT — a query whose entities were NEVER mentioned
//!       answers "I don't know." (never a confident Yes/No). The open world
//!       must not fabricate a verdict about things it was never told.
//!
//! On ANY violation we panic with the SEED, the offending world, and the
//! offending query, so every failure is exactly reproducible by re-running with
//! that seed.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::qa;

// ---------------------------------------------------------------------------
// Deterministic PRNG: a 64-bit linear-congruential generator (no external rand).
// Constants are the well-known PCG/Knuth LCG multiplier + increment; we consume
// the high bits, which have the best statistical quality for an LCG.
// ---------------------------------------------------------------------------
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Mix the seed once so seed 0 is not a degenerate fixed point.
        Lcg {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
        }
    }

    /// Advance and return a fresh 32-bit value from the high bits of the state.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use the top 32 bits.
        (self.state >> 32) as u32
    }

    /// A uniform value in `0..n` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }

    /// A fair coin.
    fn coin(&mut self) -> bool {
        self.next_u32() & 1 == 1
    }
}

// ---------------------------------------------------------------------------
// Fixed in-lexicon vocabulary.
//
// The ASSERTED grid is the words we ever read into a world. The UNSEEN words are
// also valid in-lexicon nouns but are GUARANTEED never asserted, so a query built
// from them is genuinely "never mentioned" — the substrate for property (c).
// ---------------------------------------------------------------------------
const ASSERT_AGENTS: &[&str] = &["teacher", "editor", "author"];
const ASSERT_PATIENTS: &[&str] = &["report", "book", "letter"];
/// (base, third-singular) — the base is used in questions ("does X write Y?"),
/// the 3sg in declaratives ("X writes Y.").
const VERBS: &[(&str, &str)] = &[("write", "writes"), ("read", "reads")];

/// In-lexicon nouns that are DISJOINT from the asserted grid above, so a query
/// over them references entities the world was never told about.
const UNSEEN_AGENTS: &[&str] = &["doctor", "nurse", "pilot"];
const UNSEEN_PATIENTS: &[&str] = &["essay", "memo", "poem"];

/// A single predication: `<agent> <verb> <patient>`, with a polarity.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Fact {
    agent: usize,   // index into ASSERT_AGENTS
    verb: usize,    // index into VERBS
    patient: usize, // index into ASSERT_PATIENTS
    negated: bool,
}

impl Fact {
    fn agent(&self) -> &'static str {
        ASSERT_AGENTS[self.agent]
    }
    fn patient(&self) -> &'static str {
        ASSERT_PATIENTS[self.patient]
    }
    fn verb_base(&self) -> &'static str {
        VERBS[self.verb].0
    }
    fn verb_3sg(&self) -> &'static str {
        VERBS[self.verb].1
    }

    /// The English declarative that reads this fact into the world.
    /// Affirmative: "The teacher writes the report."
    /// Negated:     "The teacher does not write the report."
    fn declarative(&self) -> String {
        if self.negated {
            format!(
                "The {} does not {} the {}.",
                self.agent(),
                self.verb_base(),
                self.patient()
            )
        } else {
            format!(
                "The {} {} the {}.",
                self.agent(),
                self.verb_3sg(),
                self.patient()
            )
        }
    }

    /// The English yes/no question for this exact predication, optionally
    /// polarity-flipped. The base form is "Does the teacher write the report?";
    /// the flipped form inserts "not": "Does the teacher not write the report?".
    fn question(&self, flip: bool) -> String {
        let neg = self.negated ^ flip;
        if neg {
            format!(
                "Does the {} not {} the {}?",
                self.agent(),
                self.verb_base(),
                self.patient()
            )
        } else {
            format!(
                "Does the {} {} the {}?",
                self.agent(),
                self.verb_base(),
                self.patient()
            )
        }
    }
}

/// A query that does NOT touch the asserted grid: a verb over UNSEEN agent +
/// UNSEEN patient. Used for property (c) and as one of the (b) query sources.
struct UnseenQuery {
    agent: &'static str,
    patient: &'static str,
    verb_base: &'static str,
}

impl UnseenQuery {
    fn question(&self, negated: bool) -> String {
        if negated {
            format!(
                "Does the {} not {} the {}?",
                self.agent, self.verb_base, self.patient
            )
        } else {
            format!(
                "Does the {} {} the {}?",
                self.agent, self.verb_base, self.patient
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Answer classification.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Yes,
    No,
    DontKnow,
    /// Anything we did not expect for a yes/no query (would itself be a bug).
    Other,
}

fn classify(answer: &str) -> Verdict {
    let a = answer.trim().to_lowercase();
    if a.contains("don't know") || a.contains("dont know") {
        Verdict::DontKnow
    } else if a.starts_with("yes") {
        Verdict::Yes
    } else if a.starts_with("no") {
        Verdict::No
    } else {
        Verdict::Other
    }
}

/// Render a world (the read declaratives) for a reproducible panic message.
fn world_repr(facts: &[Fact]) -> String {
    if facts.is_empty() {
        return "<empty world>".to_string();
    }
    facts
        .iter()
        .map(|f| f.declarative())
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// World generation.
// ---------------------------------------------------------------------------

/// Generate a deduped set of facts. Dedup is BY PREDICATION (agent,verb,patient)
/// keeping exactly one polarity per triple — otherwise reading both "X writes Y"
/// and "X does not write Y" would put a genuine contradiction in the world and
/// recall (property a) would be testing the engine against an inconsistent input
/// rather than a real soundness bug.
fn generate_world(rng: &mut Lcg) -> Vec<Fact> {
    // 1..=5 facts per world (small worlds, as specified).
    let n = 1 + rng.below(5);
    let mut chosen: Vec<Fact> = Vec::new();
    let mut seen_triples: Vec<(usize, usize, usize)> = Vec::new();

    let mut attempts = 0;
    while chosen.len() < n && attempts < 64 {
        attempts += 1;
        let f = Fact {
            agent: rng.below(ASSERT_AGENTS.len()),
            verb: rng.below(VERBS.len()),
            patient: rng.below(ASSERT_PATIENTS.len()),
            negated: rng.coin(),
        };
        let triple = (f.agent, f.verb, f.patient);
        if seen_triples.contains(&triple) {
            continue; // already have a polarity for this predication
        }
        seen_triples.push(triple);
        chosen.push(f);
    }
    chosen
}

// ---------------------------------------------------------------------------
// The fuzzer.
// ---------------------------------------------------------------------------

#[test]
fn soundness_fuzz_never_makes_a_false_claim() {
    // Build the synthesized engine exactly ONCE — this is the slow part.
    let engine = Engine::new();

    // Fixed seed for reproducibility. Every iteration derives its own sub-seed so
    // the panic message can pinpoint the exact failing world.
    const MASTER_SEED: u64 = 0xC0FFEE_1234_5678;
    const ITERATIONS: usize = 240; // >= 200 as required

    // Every property below panics on the FIRST violation (with the reproducing
    // seed), so reaching the end of the loop is itself the "0 violations" proof.
    for iter in 0..ITERATIONS {
        // A per-iteration seed, deterministically derived from the master seed.
        let seed = MASTER_SEED ^ (iter as u64).wrapping_mul(0x100000001B3);
        let mut rng = Lcg::new(seed);

        // --- Build a fresh world and read it ---------------------------------
        let facts = generate_world(&mut rng);
        let mut disc = Discourse::new();
        for f in &facts {
            disc.read(&engine, &f.declarative());
        }

        let ask = |q: &str| -> Verdict { classify(&qa::answer(&engine, &disc, q)) };

        // =====================================================================
        // (a) ASSERTED-FACT RECALL
        //   Every read fact answers Yes/No matching its own polarity; the
        //   polarity-flipped question answers the opposite.
        // =====================================================================
        for f in &facts {
            // Expected verdict for the fact's own question: a negated fact
            // ("does not write") is true, so "Does X not write Y?" => Yes; an
            // affirmative fact => "Does X write Y?" => Yes. In both cases the
            // question() mirrors the stored polarity, so the answer is Yes.
            let own_q = f.question(false);
            let own = ask(&own_q);
            if own != Verdict::Yes {
                panic!(
                    "SOUNDNESS VIOLATION (a) RECALL: seed={seed:#x} iter={iter}\n\
                     world: {}\n\
                     read fact: {:?} -> {}\n\
                     its own question {:?} expected Yes, got {:?}\n\
                     full answer: {:?}",
                    world_repr(&facts),
                    f.declarative(),
                    "(should recall as true)",
                    own_q,
                    own,
                    qa::answer(&engine, &disc, &own_q),
                );
            }

            // The polarity-flip must answer No (the world determined it false).
            let flip_q = f.question(true);
            let flip = ask(&flip_q);
            if flip != Verdict::No {
                panic!(
                    "SOUNDNESS VIOLATION (a) RECALL-FLIP: seed={seed:#x} iter={iter}\n\
                     world: {}\n\
                     read fact: {}\n\
                     polarity-flipped question {:?} expected No, got {:?}\n\
                     full answer: {:?}",
                    world_repr(&facts),
                    f.declarative(),
                    flip_q,
                    flip,
                    qa::answer(&engine, &disc, &flip_q),
                );
            }
        }

        // =====================================================================
        // (b) NEGATION CONSISTENCY
        //   For a batch of generated queries (some over the asserted grid, some
        //   over unseen entities), "Q?"=Yes and "not Q?"=Yes must never both
        //   hold. This is the no-self-contradiction invariant and must hold for
        //   EVERY query regardless of whether it was asserted.
        // =====================================================================
        let n_queries = 3 + rng.below(4); // 3..=6 probes
        for _ in 0..n_queries {
            // Either an on-grid predication or a fully unseen one.
            let (q_pos, q_neg, descr) = if rng.coin() {
                // On-grid: any agent/verb/patient over the asserted vocabulary.
                let f = Fact {
                    agent: rng.below(ASSERT_AGENTS.len()),
                    verb: rng.below(VERBS.len()),
                    patient: rng.below(ASSERT_PATIENTS.len()),
                    negated: false,
                };
                (f.question(false), f.question(true), f.declarative())
            } else {
                // Off-grid (unseen entities).
                let uq = UnseenQuery {
                    agent: UNSEEN_AGENTS[rng.below(UNSEEN_AGENTS.len())],
                    patient: UNSEEN_PATIENTS[rng.below(UNSEEN_PATIENTS.len())],
                    verb_base: VERBS[rng.below(VERBS.len())].0,
                };
                let pos = uq.question(false);
                let neg = uq.question(true);
                let descr = format!("(unseen) {}", pos);
                (pos, neg, descr)
            };

            let a_pos = ask(&q_pos);
            let a_neg = ask(&q_neg);
            if a_pos == Verdict::Yes && a_neg == Verdict::Yes {
                panic!(
                    "SOUNDNESS VIOLATION (b) NEGATION CONSISTENCY: seed={seed:#x} iter={iter}\n\
                     world: {}\n\
                     query: {descr}\n\
                     \"{}\" => Yes  AND  \"{}\" => Yes  (a self-contradiction)\n\
                     pos answer: {:?}\n\
                     neg answer: {:?}",
                    world_repr(&facts),
                    q_pos,
                    q_neg,
                    qa::answer(&engine, &disc, &q_pos),
                    qa::answer(&engine, &disc, &q_neg),
                );
            }
        }

        // =====================================================================
        // (c) NO-SPURIOUS-ENTAILMENT
        //   A query whose AGENT and PATIENT were never mentioned (drawn from the
        //   UNSEEN pools, disjoint from everything read) must answer
        //   "I don't know." — never a confident Yes or No.
        // =====================================================================
        let n_unseen = 2 + rng.below(3); // 2..=4 unseen probes
        for _ in 0..n_unseen {
            let uq = UnseenQuery {
                agent: UNSEEN_AGENTS[rng.below(UNSEEN_AGENTS.len())],
                patient: UNSEEN_PATIENTS[rng.below(UNSEEN_PATIENTS.len())],
                verb_base: VERBS[rng.below(VERBS.len())].0,
            };
            let q = uq.question(rng.coin());
            let v = ask(&q);
            if v != Verdict::DontKnow {
                panic!(
                    "SOUNDNESS VIOLATION (c) NO-SPURIOUS-ENTAILMENT: seed={seed:#x} iter={iter}\n\
                     world: {}\n\
                     unseen query {:?} (agent '{}', patient '{}' never mentioned)\n\
                     expected \"I don't know.\", got {:?}\n\
                     full answer: {:?}",
                    world_repr(&facts),
                    q,
                    uq.agent,
                    uq.patient,
                    v,
                    qa::answer(&engine, &disc, &q),
                );
            }
        }
    }

    // If we got here, no panic fired. Assert the headline invariant explicitly so
    // the test's success is a positive claim, and print the iteration count.
    eprintln!(
        "soundness_fuzz: {ITERATIONS} seeded iterations, 0 violations \
         (master seed {MASTER_SEED:#x})"
    );
}
