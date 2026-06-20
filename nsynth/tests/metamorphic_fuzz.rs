//! METAMORPHIC PARAPHRASE-INVARIANCE FUZZER for the comprehension engine.
//!
//! Soundness fuzzers (`soundness_fuzz.rs`) check that the engine never makes a
//! FALSE claim. This fuzzer checks the orthogonal METAMORPHIC property: two
//! surface strings that mean the SAME thing must get the SAME answer. Re-phrasing
//! a question must not change its truth — that is the essence of *understanding*
//! a sentence rather than pattern-matching its tokens.
//!
//! Like the soundness fuzzer this is property-based and driven by a SEEDED
//! linear-congruential generator (NO external `rand` crate — fully
//! reproducible). One `Engine` is built ONCE and reused across every iteration
//! (synthesis is slow); only the `Discourse` (the world) is fresh per iteration.
//!
//! Two metamorphic relations are probed across >= 150 seeded random in-lexicon
//! worlds:
//!
//!   (1) ACTIVE <-> PASSIVE.  After asserting "the <agent> writes the <patient>":
//!         - "does the <agent> write the <patient>?"           (active)
//!         - "is the <patient> written by the <agent>?"        (passive)
//!       must AGREE (both Yes).  The role-swapped passive
//!         - "is the <agent> written by the <patient>?"        (converse mismatch)
//!       must NOT be Yes (swapping agent/patient is a different proposition).
//!
//!   (2) COMPARATIVE CONVERSE.  After asserting "the <a> is longer than the <b>":
//!         - "is the <a> longer than the <b>?"                 (forward)
//!         - "is the <b> shorter than the <a>?"                (converse)
//!       must AGREE (both Yes).  The reversed forward
//!         - "is the <b> longer than the <a>?"                 (reverse)
//!       must be No (asymmetry of a strict order).
//!
//! AGREEMENT POLICY.  For each paraphrase pair we record honestly whether it is
//! FULL-AGREE (both sides resolve to an identical Yes/No verdict — the strong
//! property) or merely NO-CONTRADICT (never one Yes and the other No — the weaker
//! sound property a parser gap still must satisfy). The test PANICS on any
//! contradiction (a real soundness bug) and additionally panics if a pair that
//! was expected to fully agree does not. The empirical agree-vs-no-contradict
//! tally is printed at the end so the support story is reported truthfully.
//!
//! On ANY disagreement we panic with the SEED, the world, the two phrasings, and
//! the two answers, so every failure is exactly reproducible by re-running with
//! that seed.

use mog_synth::comprehension::Engine;
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::qa;

// ---------------------------------------------------------------------------
// Deterministic PRNG: a 64-bit linear-congruential generator (no external rand).
// Same construction as `soundness_fuzz.rs` so the two fuzzers share a vocabulary
// of reproducibility — consume the high 32 bits (best statistical quality).
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

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    /// A uniform value in `0..n` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }

    /// An index in `0..n` GUARANTEED distinct from `avoid` (requires n >= 2).
    /// Used to pick a second entity that is never equal to the first, so a
    /// generated proposition is never the degenerate "X verbs X" / "X longer
    /// than X" (which the engine correctly refuses, and which would otherwise be
    /// a spurious TEST failure rather than a real bug).
    fn below_other(&mut self, n: usize, avoid: usize) -> usize {
        // Offset in 1..n keeps us off `avoid` after the modular add.
        (avoid + 1 + self.below(n - 1)) % n
    }
}

// ---------------------------------------------------------------------------
// Fixed in-lexicon vocabulary.
//
// For the ACTIVE<->PASSIVE relation we need verbs whose passive participle the
// parser resolves back to the base lemma. "write"/"read" are both in the
// synthesized PAST_PARTICIPLE lexicon ("written", and "read" whose participle is
// itself), so both voices land on the SAME Event{predicate, agent, patient}.
//
// For the COMPARATIVE relation we use the gradable LENGTH scale (longer/shorter),
// the dimension the curriculum's GRADABLE table covers most fully.
// ---------------------------------------------------------------------------
// Every word here is drawn from the engine's baked-in lexicons
// (`comprehension::AGENTS` / `comprehension::PATIENTS`) so each generated world
// is genuinely IN-LEXICON — using an out-of-lexicon noun would make even the
// active question under-derive to "I don't know" (the parser cannot resolve an
// unknown agent), which is a TEST bug, not an engine soundness bug.
const AGENTS: &[&str] = &["teacher", "editor", "author", "doctor"];
const PATIENTS: &[&str] = &["report", "book", "letter", "note"];
/// (base, third-singular, past-participle) for the two transitive verbs we read
/// in the active voice and question in either voice.
const VERBS: &[(&str, &str, &str)] = &[("write", "writes", "written"), ("read", "reads", "read")];

/// Distinct nouns for the two ends of a comparative ordering. Kept apart from the
/// active/passive grid only for readability; any in-lexicon noun works.
const CMP_NOUNS: &[&str] = &["report", "book", "letter", "note", "essay"];

// ---------------------------------------------------------------------------
// Answer classification (mirrors soundness_fuzz.rs).
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Yes,
    No,
    DontKnow,
    /// Anything we did not expect for a yes/no query (itself a bug).
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

/// A direct contradiction between two paraphrases: one says Yes and the other
/// says No. This is ALWAYS a soundness bug regardless of parser coverage.
fn contradict(a: Verdict, b: Verdict) -> bool {
    matches!(
        (a, b),
        (Verdict::Yes, Verdict::No) | (Verdict::No, Verdict::Yes)
    )
}

// ---------------------------------------------------------------------------
// Per-pair agreement bookkeeping. We tally, per metamorphic relation, how many
// random worlds produced FULL agreement (identical Yes/No verdicts) vs merely
// NO-CONTRADICTION (e.g. one side under-derived to "I don't know"). This lets the
// final report be honest about which paraphrase pairs are fully supported today.
// ---------------------------------------------------------------------------
#[derive(Default)]
struct Tally {
    full_agree: usize,
    no_contradict_only: usize,
}

impl Tally {
    /// Record one observation of a paraphrase pair that must AGREE. Returns the
    /// classification so callers can also enforce the strong property when the
    /// pair is one we expect to fully agree.
    fn record_agree(&mut self, a: Verdict, b: Verdict) {
        if a == b && matches!(a, Verdict::Yes | Verdict::No) {
            self.full_agree += 1;
        } else {
            self.no_contradict_only += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// The fuzzer.
// ---------------------------------------------------------------------------
#[test]
fn metamorphic_paraphrase_invariance() {
    // Build the synthesized engine exactly ONCE — this is the slow part.
    let engine = Engine::new();

    // "DEAD_BEEF_FACE" — a valid hex constant, distinct from the soundness
    // fuzzer's seed so the two explore different random worlds.
    const MASTER_SEED: u64 = 0xDEAD_BEEF_FACE;
    const ITERATIONS: usize = 180; // >= 150 as required

    // Empirical agreement bookkeeping, reported at the end.
    let mut active_passive = Tally::default();
    let mut cmp_converse = Tally::default();

    for iter in 0..ITERATIONS {
        // Per-iteration seed derived from the master seed (FNV-style spread).
        let seed = MASTER_SEED ^ (iter as u64).wrapping_mul(0x100000001B3);
        let mut rng = Lcg::new(seed);

        // =====================================================================
        // (1) ACTIVE <-> PASSIVE
        // =====================================================================
        {
            // A fresh world holding exactly one transitive fact, read in the
            // ACTIVE voice ("the teacher writes the report.").
            let agent = AGENTS[rng.below(AGENTS.len())];
            // AGENTS and PATIENTS are disjoint pools, so the patient is always a
            // different word from the agent; the role-swap probe ("is the AGENT
            // written by the PATIENT?") is therefore always a genuinely distinct
            // proposition.
            let patient = PATIENTS[rng.below(PATIENTS.len())];
            let (base, third, _pp) = VERBS[rng.below(VERBS.len())];

            let declarative = format!("The {agent} {third} the {patient}.");
            let mut disc = Discourse::new();
            disc.read(&engine, &declarative);

            let ask = |q: &str| -> Verdict { classify(&qa::answer(&engine, &disc, q)) };

            // The active and passive paraphrases of the SAME proposition.
            let active_q = format!("Does the {agent} {base} the {patient}?");
            let passive_q = format!("Is the {patient} {pp} by the {agent}?", pp = _pp);
            // Role-swapped passive: "is the AGENT written by the PATIENT?" — a
            // DIFFERENT proposition (subject and by-phrase exchanged).
            let swapped_q = format!("Is the {agent} {pp} by the {patient}?", pp = _pp);

            let a_active = ask(&active_q);
            let a_passive = ask(&passive_q);
            let a_swapped = ask(&swapped_q);

            // --- HARD INVARIANT: active and passive must never CONTRADICT. -----
            if contradict(a_active, a_passive) {
                panic!(
                    "METAMORPHIC VIOLATION (1) ACTIVE/PASSIVE CONTRADICTION: \
                     seed={seed:#x} iter={iter}\n\
                     world: {declarative}\n\
                     active  {active_q:?} => {a_active:?}\n\
                     passive {passive_q:?} => {a_passive:?}\n\
                     (a paraphrase pair must never resolve Yes vs No)\n\
                     active  answer: {:?}\n\
                     passive answer: {:?}",
                    qa::answer(&engine, &disc, &active_q),
                    qa::answer(&engine, &disc, &passive_q),
                );
            }

            // --- STRONG INVARIANT: this pair is FULLY SUPPORTED, so both must --
            // resolve to Yes (we just asserted the fact in the active voice). If
            // either under-derives, that is a real regression for a relation we
            // have verified works, so fail loudly.
            if !(a_active == Verdict::Yes && a_passive == Verdict::Yes) {
                panic!(
                    "METAMORPHIC VIOLATION (1) ACTIVE/PASSIVE NON-AGREEMENT: \
                     seed={seed:#x} iter={iter}\n\
                     world: {declarative}\n\
                     active  {active_q:?} expected Yes, got {a_active:?}\n\
                     passive {passive_q:?} expected Yes, got {a_passive:?}\n\
                     active  answer: {:?}\n\
                     passive answer: {:?}",
                    qa::answer(&engine, &disc, &active_q),
                    qa::answer(&engine, &disc, &passive_q),
                );
            }
            active_passive.record_agree(a_active, a_passive);

            // --- CONVERSE MISMATCH must NOT be Yes (different proposition). -----
            if a_swapped == Verdict::Yes {
                panic!(
                    "METAMORPHIC VIOLATION (1) ROLE-SWAP OVER-DERIVATION: \
                     seed={seed:#x} iter={iter}\n\
                     world: {declarative}\n\
                     role-swapped passive {swapped_q:?} answered Yes\n\
                     (swapping agent/patient is a DIFFERENT proposition; \
                     must not be Yes)\n\
                     answer: {:?}",
                    qa::answer(&engine, &disc, &swapped_q),
                );
            }
        }

        // =====================================================================
        // (2) COMPARATIVE CONVERSE
        // =====================================================================
        {
            // Two GUARANTEED-distinct comparison ends a, b — a self-comparison
            // ("X longer than X") is degenerate (the world model never stores it
            // and the engine answers No), so we never generate one.
            let a_idx = rng.below(CMP_NOUNS.len());
            let b_idx = rng.below_other(CMP_NOUNS.len(), a_idx);
            let a = CMP_NOUNS[a_idx];
            let b = CMP_NOUNS[b_idx];

            // Assert "the <a> is longer than the <b>".
            let declarative = format!("The {a} is longer than the {b}.");
            let mut disc = Discourse::new();
            disc.read(&engine, &declarative);

            let ask = |q: &str| -> Verdict { classify(&qa::answer(&engine, &disc, q)) };

            // Forward and its converse phrasing — the SAME ordering on length.
            let forward_q = format!("Is the {a} longer than the {b}?");
            let converse_q = format!("Is the {b} shorter than the {a}?");
            // Reversed forward — the OPPOSITE ordering, false by asymmetry.
            let reverse_q = format!("Is the {b} longer than the {a}?");

            let a_fwd = ask(&forward_q);
            let a_conv = ask(&converse_q);
            let a_rev = ask(&reverse_q);

            // --- HARD INVARIANT: forward and converse must never CONTRADICT. ---
            if contradict(a_fwd, a_conv) {
                panic!(
                    "METAMORPHIC VIOLATION (2) COMPARATIVE CONTRADICTION: \
                     seed={seed:#x} iter={iter}\n\
                     world: {declarative}\n\
                     forward  {forward_q:?} => {a_fwd:?}\n\
                     converse {converse_q:?} => {a_conv:?}\n\
                     (longer-than and its shorter-than converse must never \
                     resolve Yes vs No)\n\
                     forward  answer: {:?}\n\
                     converse answer: {:?}",
                    qa::answer(&engine, &disc, &forward_q),
                    qa::answer(&engine, &disc, &converse_q),
                );
            }

            // --- STRONG INVARIANT: this pair is FULLY SUPPORTED -> both Yes. ----
            if !(a_fwd == Verdict::Yes && a_conv == Verdict::Yes) {
                panic!(
                    "METAMORPHIC VIOLATION (2) COMPARATIVE NON-AGREEMENT: \
                     seed={seed:#x} iter={iter}\n\
                     world: {declarative}\n\
                     forward  {forward_q:?} expected Yes, got {a_fwd:?}\n\
                     converse {converse_q:?} expected Yes, got {a_conv:?}\n\
                     forward  answer: {:?}\n\
                     converse answer: {:?}",
                    qa::answer(&engine, &disc, &forward_q),
                    qa::answer(&engine, &disc, &converse_q),
                );
            }
            cmp_converse.record_agree(a_fwd, a_conv);

            // --- REVERSE must be No (asymmetry of the strict order). -----------
            if a_rev != Verdict::No {
                panic!(
                    "METAMORPHIC VIOLATION (2) COMPARATIVE ASYMMETRY: \
                     seed={seed:#x} iter={iter}\n\
                     world: {declarative}\n\
                     reverse {reverse_q:?} expected No, got {a_rev:?}\n\
                     (with <a> longer than <b> proven, <b> longer than <a> is \
                     provably false)\n\
                     answer: {:?}",
                    qa::answer(&engine, &disc, &reverse_q),
                );
            }
        }
    }

    // No panic fired -> 0 disagreements. Report the empirical support story
    // honestly: how many of the >=150 iterations achieved FULL agreement for each
    // relation vs merely no-contradiction.
    let total = ITERATIONS;
    eprintln!(
        "metamorphic_fuzz: {total} seeded iterations, 0 disagreements \
         (master seed {MASTER_SEED:#x})"
    );
    eprintln!(
        "  ACTIVE<->PASSIVE:        full-agree {}/{}, no-contradict-only {}/{}",
        active_passive.full_agree, total, active_passive.no_contradict_only, total
    );
    eprintln!(
        "  COMPARATIVE converse:    full-agree {}/{}, no-contradict-only {}/{}",
        cmp_converse.full_agree, total, cmp_converse.no_contradict_only, total
    );

    // Both relations were verified FULL-AGREE on every iteration above (the
    // strong invariants would have panicked otherwise); assert that the tally
    // agrees so the headline claim is a positive, machine-checked statement.
    assert_eq!(
        active_passive.full_agree, total,
        "active<->passive should be full-agree on every iteration"
    );
    assert_eq!(
        cmp_converse.full_agree, total,
        "comparative converse should be full-agree on every iteration"
    );
}
