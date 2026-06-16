//! Self-extension: the request/report types and entry point for *closing a gap*
//! by synthesizing and grafting a new component onto the understanding engine.
//!
//! The intended flow (filled in by a later phase) is the substrate contract from
//! [`super`]: **propose → journal the attempt → run the gate → accept only on a
//! green gate → journal the outcome.** A [`LearnRequest`] names a gap and the
//! examples that characterize the missing component; [`self_extend`] will try to
//! synthesize it (via [`Engine::try_extend`](crate::comprehension::Engine::try_extend)),
//! run the resulting candidate engine through the regression gate, and accept the
//! extension only if the gate stays green — emitting a [`LearnReport`] describing
//! what happened either way.
//!
//! The synthesize → gate → accept logic is implemented in [`self_extend`]: it
//! routes a [`LearnRequest`] through `Engine::try_extend` (synthesize + verify),
//! runs the candidate through the regression gate, journals the attempt, and
//! accepts the extension only on a green gate — guaranteeing **monotone growth**
//! (a new component that breaks anything is rejected, leaving the engine
//! untouched).

use crate::benchmark::Example;
use crate::comprehension::Engine;
use crate::self_improve::gate::regression_gate;
use crate::self_improve::journal::{self, JournalEntry};
use crate::self_improve::store::{self, StoredComponent};
use crate::solved_cache::examples_fingerprint;

/// A request to close an observed gap by learning a new component.
///
/// * `gap` — a human-readable description of what the engine could not do.
/// * `name` — the Mog function name to synthesize for the new component.
/// * `signature` — the component's `&'static str` Mog signature (e.g.
///   `"fn creature_class(s: string) -> i64"`); must be `'static` because
///   [`crate::benchmark::Problem::signature`] is `&'static str`.
/// * `examples` — the I/O examples characterizing the component; synthesis must
///   reproduce all of them before a candidate is considered.
pub struct LearnRequest {
    pub gap: String,
    pub name: String,
    pub signature: &'static str,
    pub examples: Vec<Example>,
}

/// The outcome of a [`self_extend`] attempt — the audit-trail payload.
///
/// * `gap` — echoes the gap from the request.
/// * `synthesized` — did synthesis produce a verified candidate component?
/// * `method` — the teacher/method that recovered the component (empty if not
///   synthesized).
/// * `regression_passed` — did the candidate engine pass the regression gate?
/// * `accepted` — was the extension ultimately adopted (synthesized AND gated)?
/// * `message` — a human-readable explanation of the outcome.
pub struct LearnReport {
    pub gap: String,
    pub synthesized: bool,
    pub method: String,
    pub regression_passed: bool,
    pub accepted: bool,
    pub message: String,
}

/// What KIND of gap the mind detected — what sort of thing it could not handle.
///
/// This classification drives which curriculum a later phase will propose to
/// close the gap (e.g. a [`Lexical`](GapKind::Lexical) gap mines a string→class
/// lexicon; a [`Structural`](GapKind::Structural) gap mines a parsing/transduction
/// rule; an [`Inferential`](GapKind::Inferential) gap mines a reasoning rule). The
/// scaffold defines the taxonomy; the detection + routing logic lands next phase.
#[derive(Clone, Debug, PartialEq)]
pub enum GapKind {
    /// An unknown WORD: the lexicons carry no class/animacy/etc. for a surface
    /// token (e.g. a noun or verb the engine has never been taught).
    Lexical,
    /// An unparseable STRUCTURE: the surface form is understood word-by-word but
    /// the construction (a clause shape, an inflection pattern) is not recovered.
    Structural,
    /// A missing INFERENCE: every word/structure parses, but the mind cannot
    /// derive the answer because a reasoning rule it would need is absent.
    Inferential,
}

/// One observed gap — something the mind read but could not fully handle.
///
/// * `kind` — the [`GapKind`] classifying what sort of capability is missing.
/// * `surface` — the specific surface fragment that triggered the gap (the
///   unknown word, the unparsed clause, the underivable proposition).
/// * `context` — the surrounding input the gap was observed in, kept so a later
///   phase can mine characterizing examples from real usage.
#[derive(Clone, Debug, PartialEq)]
pub struct Gap {
    pub kind: GapKind,
    pub surface: String,
    pub context: String,
}

/// The outcome of a [`study`](crate::understanding::mind::Mind::study) session —
/// a self-directed loop that reads a corpus, detects gaps, proposes curricula,
/// and folds in every verified + gated component.
///
/// * `rounds` — how many study rounds were run.
/// * `learned` — the names of the components actually adopted (synthesized AND
///   gated AND accepted), in adoption order.
/// * `attempted` — how many self-extension attempts were made across all rounds.
/// * `rejected` — how many attempts were rejected (synthesis failed, or the gate
///   went red). `attempted == learned.len() + rejected` holds by construction.
#[derive(Clone, Debug)]
pub struct StudyReport {
    pub rounds: usize,
    pub learned: Vec<String>,
    pub attempted: usize,
    pub rejected: usize,
}

/// Attempt to extend `engine` to close the gap described by `req`.
///
/// This is the self-extension loop — the one place the understanding layer adds
/// a *verified* component to itself, fully gated. It enforces the substrate
/// contract end-to-end:
///
/// 1. **Synthesize + verify.** Call
///    [`Engine::try_extend`](crate::comprehension::Engine::try_extend) with the
///    request's `name` / `signature` / `examples`. The solver only reports
///    success once it has a candidate program that reproduces *every* example,
///    so a returned candidate is verified by construction. On synthesis failure
///    the gap stays open: we journal the attempt (`verified=false`,
///    `accepted=false`), return `(None, report)`, and never touch `engine`.
/// 2. **Gate.** Run the candidate engine through
///    [`regression_gate`](crate::self_improve::gate::regression_gate). The
///    extension is accepted **only** if [`GateReport::ok`] holds — every golden
///    behavioral case still passes *and* every soundness invariant holds. This is
///    what makes growth monotone: an addition that regresses any existing
///    behavior, or makes the world model unsound, is rejected.
/// 3. **Journal.** Record a [`JournalEntry`] for the attempt (accepted or not):
///    the gap, the action (`"synthesize <name>"`), the recovering `method`,
///    `verified=true`, and the gate's verdict mirrored into both
///    `regression_passed` and `accepted`.
/// 4. **Return.** `(Some(candidate), report)` with `accepted=true` on a green
///    gate; otherwise `(None, report)` whose `message` names the failing golden
///    cases (and flags an unsound run), so the rejection is auditable.
///
/// `engine` is never mutated. The returned candidate (when `Some`) is a fresh,
/// already-gated `Engine` the caller may adopt as the new live engine.
pub fn self_extend(engine: &Engine, req: &LearnRequest) -> (Option<Engine>, LearnReport) {
    // --- 1. SYNTHESIZE + VERIFY ------------------------------------------
    let candidate = match engine.try_extend(&req.name, req.signature, req.examples.clone()) {
        Ok(candidate) => candidate,
        Err(err) => {
            // No verified program closes this gap. The engine is untouched.
            let message = format!(
                "no verified program for gap {:?}: synthesis of `{}` failed ({})",
                req.gap, req.name, err
            );
            let report = LearnReport {
                gap: req.gap.clone(),
                synthesized: false,
                method: String::new(),
                regression_passed: false,
                accepted: false,
                message: message.clone(),
            };
            journal::record(&JournalEntry {
                when_unix: 0, // stamped by record()
                gap: req.gap.clone(),
                action: format!("synthesize {}", req.name),
                method: String::new(),
                verified: false,
                regression_passed: false,
                accepted: false,
                note: message,
            });
            return (None, report);
        }
    };

    // The recovering teacher for the freshly grafted component (last method).
    let method = candidate
        .methods
        .last()
        .map(|(_, m)| m.clone())
        .unwrap_or_default();

    // --- 2. GATE ----------------------------------------------------------
    // Re-running the whole golden corpus + soundness oracle against the
    // candidate proves the addition broke nothing. Additive grafting cannot
    // regress prior behavior in principle, but the gate is the *enforced* proof
    // of that invariant, not a trusted assumption.
    let gate = regression_gate(&candidate);
    let passed_gate = gate.ok();

    let message = if passed_gate {
        format!(
            "accepted `{}` via {} (gate green: {}/{} golden cases passed, sound)",
            req.name, method, gate.passed, gate.total
        )
    } else {
        // Name the failing cases (and the soundness verdict) so the rejection
        // is fully auditable from the report alone.
        let failures = if gate.failures.is_empty() {
            "(no behavioral failures)".to_string()
        } else {
            gate.failures.join("; ")
        };
        format!(
            "rejected `{}` via {}: regression gate red ({}/{} golden cases passed, sound={}); \
             failing cases: {}",
            req.name, method, gate.passed, gate.total, gate.sound, failures
        )
    };

    let report = LearnReport {
        gap: req.gap.clone(),
        synthesized: true,
        method: method.clone(),
        regression_passed: passed_gate,
        accepted: passed_gate,
        message: message.clone(),
    };

    // --- 3. JOURNAL -------------------------------------------------------
    journal::record(&JournalEntry {
        when_unix: 0, // stamped by record()
        gap: req.gap.clone(),
        action: format!("synthesize {}", req.name),
        method,
        verified: true,
        regression_passed: passed_gate,
        accepted: passed_gate,
        note: message,
    });

    // --- 4. PERSIST (accepted only) + RETURN ------------------------------
    if passed_gate {
        // Durably record the accepted component so a later run can re-graft it
        // (gated again) into a fresh engine — cross-run compounding memory. The
        // grafted source is exactly the suffix `try_extend` appended to the base
        // program (`"\n" + result.code`), recovered by slicing the candidate's
        // program after the original engine's program. Slicing the suffix (rather
        // than brace-matching the function out) reproduces the synthesized bytes
        // verbatim. Persistence is best-effort: `store::save_one` swallows I/O
        // errors and is a no-op when the store is disabled, so a store failure
        // never blocks adoption (the component is already live on `candidate`).
        let base_len = engine.program().len();
        let code = candidate
            .program()
            .get(base_len..)
            .map(|s| s.trim_start_matches('\n').to_string())
            .unwrap_or_default();
        if !code.is_empty() {
            // For an `<x>_class` component, persist its VERIFIED example domain
            // (word, is_member) so a reload answers within exactly the proven
            // evidence — its unverified generalization stays UNKNOWN across runs.
            let members: Vec<(String, bool)> = if req.name.ends_with("_class") {
                req.examples
                    .iter()
                    .filter_map(|ex| match (ex.inputs.first(), &ex.expected) {
                        (Some(crate::benchmark::Value::Str(w)), crate::benchmark::Value::Int(l)) => {
                            Some((w.clone(), *l == 1))
                        }
                        _ => None,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            store::save_one(&StoredComponent {
                name: req.name.clone(),
                signature: req.signature.to_string(),
                code,
                // `method` was moved into the JournalEntry above; the report holds
                // an equivalent clone, so read the provenance back from there.
                method: report.method.clone(),
                examples_fingerprint: examples_fingerprint(&req.examples),
                members,
            });
        }
        (Some(candidate), report)
    } else {
        // The candidate is discarded; `engine` stays the live engine.
        (None, report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comprehension::creature_class_examples;

    /// A GOOD extension: synthesize a fresh lexicon component (`creature_class`)
    /// over a vocabulary disjoint from every existing component, run it through
    /// the full substrate, and prove it is synthesized, gated, accepted, and
    /// actually live on the returned engine.
    ///
    /// `creature_class` maps mythical creatures (dragon/griffin/phoenix/…) to 1
    /// and known non-creatures (report/book/…) to 0. The vocabulary never
    /// collides with the engine's existing lexicons, so adding it cannot perturb
    /// any golden behavioral case — the gate stays green and the extension is
    /// accepted, giving monotone growth.
    #[test]
    fn good_extension_is_synthesized_gated_and_accepted() {
        // Disable journal persistence so the test never writes to $HOME, holding
        // the crate-wide journal-env lock so we never race another env-mutating
        // test on the process-global `NCPU_JOURNAL_PATH`.
        crate::self_improve::journal::test_support::with_journal_env("", || {
        let engine = Engine::new();
        // Precondition: the component is not already present.
        assert!(
            !engine.has_component("creature_class"),
            "creature_class must be a genuinely new component for this test to mean anything"
        );

        let req = LearnRequest {
            gap: "cannot classify mythical creatures (dragon, griffin, phoenix)".to_string(),
            name: "creature_class".to_string(),
            signature: "fn creature_class(s: string) -> i64",
            examples: creature_class_examples(),
        };

        let (candidate, report) = self_extend(&engine, &req);

        // The extension was synthesized, gated green, and accepted.
        assert!(report.synthesized, "creature_class must synthesize: {}", report.message);
        assert!(
            report.regression_passed,
            "the gate must stay green for an additive disjoint component: {}",
            report.message
        );
        assert!(report.accepted, "a synthesized + gated extension must be accepted: {}", report.message);
        assert!(!report.method.is_empty(), "the recovering teacher must be recorded");

        // The returned engine actually answers the new query.
        let new_engine = candidate.expect("an accepted extension must return Some(engine)");
        assert!(
            new_engine.has_component("creature_class"),
            "the accepted engine must contain the grafted component"
        );
        // The test words contain no characters needing escaping, so a plain
        // double-quoted literal is a faithful Mog string arg.
        assert_eq!(
            new_engine.eval_int("creature_class(\"dragon\")"),
            1,
            "dragon must classify as a creature on the extended engine"
        );
        assert_eq!(
            new_engine.eval_int("creature_class(\"report\")"),
            0,
            "report must classify as a non-creature on the extended engine"
        );

        // The original engine is untouched: it still lacks the component.
        assert!(
            !engine.has_component("creature_class"),
            "self_extend must not mutate the input engine"
        );
        });
    }

    /// THE safety property, end-to-end through the real `self_extend` entry
    /// point: a candidate that **synthesizes successfully** (passes the verify
    /// step) but **regresses a golden behavioral case** when grafted must be
    /// REJECTED BY THE GATE — not silently adopted.
    ///
    /// This is strictly stronger than `unsatisfiable_extension_is_rejected_...`
    /// (which is rejected at *synthesis*, before the gate ever runs) and stronger
    /// than `gate::gate_rejects_a_broken_engine` (which mutates the golden corpus
    /// rather than grafting a real component). Here a REAL component flows through
    /// the genuine synthesize → graft → gate → accept pipeline and the gate is the
    /// only thing standing between it and adoption.
    ///
    /// THE ATTACK. The engine composes one shared Mog module; `try_extend`
    /// appends the synthesized component's source to that module. The Mog runtime
    /// parses functions into a `HashMap` keyed by name, so a LATER definition of a
    /// name OVERRIDES an earlier one (verified empirically: a 2nd `fn f` wins).
    /// We therefore request a component literally named `noun_animacy` — the
    /// lexicon every taxonomy golden case depends on — whose examples
    /// deliberately MISCLASSIFY `teacher` as inanimate (class 2) instead of
    /// animate (class 1). The examples are mutually consistent (each input maps
    /// to one output), so synthesis SUCCEEDS and the candidate verifies. But once
    /// grafted it shadows the real `noun_animacy`, so "Is the teacher a person?"
    /// — a golden case that expects "Yes" — now answers "No". The candidate
    /// regresses behavior.
    ///
    /// We assert the gate is LOAD-BEARING:
    ///   * `synthesized == true`  — the component really did synthesize+verify, so
    ///     the rejection is NOT a synthesis failure; the gate is what catches it.
    ///   * `regression_passed == false` and `accepted == false` — the gate went
    ///     red and the extension was declined.
    ///   * the returned candidate is `None`.
    ///   * the input `engine` is UNCHANGED: it still answers the taxonomy query
    ///     correctly (real `noun_animacy` intact) and the would-be grafted
    ///     override is absent — the engine never carried two `noun_animacy` defs.
    ///   * the report's message names the failing golden case and `sound`, so the
    ///     rejection is auditable.
    /// If a regressing self-modification like this were ACCEPTED, soundness would
    /// be false. The whole point of this test is that it is rejected.
    #[test]
    fn regressing_extension_is_rejected_by_the_gate() {
        use crate::benchmark::{Example, Value};
        // Point the journal at a temp file (not "" which disables it) so we can
        // also prove the REJECTED attempt is journaled — "every attempt is
        // recorded" is half the substrate contract.
        let tmp = std::env::temp_dir().join(format!(
            "ncpu_gate_reject_journal_{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let tmp_path = tmp.to_string_lossy().to_string();
        crate::self_improve::journal::test_support::with_journal_env(&tmp_path, || {

        let engine = Engine::new();

        // PRECONDITION 1: the honest engine's gate is green, and the taxonomy
        // golden behavior we're about to attack is currently correct.
        let before = regression_gate(&engine);
        assert!(
            before.ok(),
            "precondition: the gate must be green on the default engine before we \
             can claim the attack regresses it; passed {}/{} sound={} failures={:?}",
            before.passed, before.total, before.sound, before.failures
        );
        assert_eq!(
            engine.noun_class("teacher"),
            1,
            "precondition: the honest engine classifies 'teacher' as animate (1)"
        );
        assert!(
            engine.is_person("teacher"),
            "precondition: the honest engine answers 'teacher is a person' = true"
        );

        // The POISON spec: a `noun_animacy` that misclassifies the agents the
        // taxonomy golden cases depend on (teacher/editor/author/student -> 2,
        // inanimate) while still being a well-posed, synthesizable string->int
        // map (every input has exactly one output, so synthesis succeeds).
        let poison = |w: &str, c: i64| Example {
            inputs: vec![Value::Str(w.to_string())],
            expected: Value::Int(c),
        };
        let examples = vec![
            poison("teacher", 2), // WRONG: teacher is animate; this breaks taxonomy
            poison("editor", 2),
            poison("author", 2),
            poison("student", 2),
            poison("report", 2),
            poison("book", 2),
            poison("dog", 1), // a couple of correct anchors so the map isn't constant
            poison("cat", 1),
        ];

        let req = LearnRequest {
            gap: "attempt to 'refine' the animacy lexicon (adversarial: regresses taxonomy)"
                .to_string(),
            name: "noun_animacy".to_string(), // COLLIDES with the engine's lexicon
            signature: "fn noun_animacy(s: string) -> i64",
            examples,
        };

        let (candidate, report) = self_extend(&engine, &req);

        // (1) Synthesis SUCCEEDED — the rejection is the GATE's doing, not a
        // synthesis failure. This is the crux: the gate is load-bearing.
        assert!(
            report.synthesized,
            "the poison component must actually synthesize (well-posed map), so the \
             rejection is attributable to the GATE, not to synthesis failure: {}",
            report.message
        );

        // (2) The gate went red and the extension was declined.
        assert!(
            !report.regression_passed,
            "the gate MUST go red for a component that regresses a golden case: {}",
            report.message
        );
        assert!(
            !report.accepted,
            "a regressing extension must NEVER be accepted: {}",
            report.message
        );

        // (3) No candidate engine is returned.
        assert!(
            candidate.is_none(),
            "a gate-rejected extension must return None, not a usable engine"
        );

        // (4) The input engine is UNCHANGED: the real noun_animacy is intact, the
        // taxonomy query is still correct, and the engine never carried a grafted
        // override (self_extend grafts onto a CLONE; it never mutates `engine`).
        assert_eq!(
            engine.noun_class("teacher"),
            1,
            "the live engine's noun_animacy must be untouched after a rejection"
        );
        assert!(
            engine.is_person("teacher"),
            "the live engine must still answer 'teacher is a person' = true"
        );
        let after = regression_gate(&engine);
        assert!(
            after.ok(),
            "the live engine's gate must still be green after a rejected attempt; \
             passed {}/{} sound={} failures={:?}",
            after.passed, after.total, after.sound, after.failures
        );

        // (5) The rejection is auditable: the message records the gate verdict and
        // names at least one failing golden case (a taxonomy/agreement question).
        assert!(
            report.message.contains("rejected") && report.message.contains("regression gate red"),
            "the report must explain the gate rejected the extension: {}",
            report.message
        );
        assert!(
            report.message.to_lowercase().contains("person")
                || report.message.to_lowercase().contains("agent")
                || report.message.to_lowercase().contains("how many")
                || report.message.contains("failing cases:"),
            "the report must name the regressed golden case(s): {}",
            report.message
        );

        // (6) The rejected attempt is JOURNALED: synthesize was attempted and
        // verified, but it was NOT accepted and did NOT pass the gate. The audit
        // trail records the rejection, not just successes.
        let entries = crate::self_improve::journal::entries();
        let mine = entries
            .iter()
            .find(|e| e.action == "synthesize noun_animacy")
            .expect("the rejected attempt must be journaled");
        assert!(mine.verified, "the journaled attempt synthesized+verified");
        assert!(!mine.accepted, "the journaled attempt was rejected, not accepted");
        assert!(!mine.regression_passed, "the journaled attempt failed the gate");
        });
        let _ = std::fs::remove_file(&tmp);
    }

    /// A BAD request that synthesis cannot satisfy must close the gap cleanly:
    /// report `synthesized=false`, `accepted=false`, return `None`, and leave the
    /// engine untouched. We force a synthesis failure with contradictory examples
    /// (the same input mapped to two different outputs), which no program can
    /// reproduce.
    #[test]
    fn unsatisfiable_extension_is_rejected_without_synthesis() {
        crate::self_improve::journal::test_support::with_journal_env("", || {
        let engine = Engine::new();
        // Contradictory spec: "dragon" -> 1 AND "dragon" -> 0. No deterministic
        // function reproduces both, so synthesis must fail.
        let examples = vec![
            Example {
                inputs: vec![crate::benchmark::Value::Str("dragon".to_string())],
                expected: crate::benchmark::Value::Int(1),
            },
            Example {
                inputs: vec![crate::benchmark::Value::Str("dragon".to_string())],
                expected: crate::benchmark::Value::Int(0),
            },
        ];
        let req = LearnRequest {
            gap: "impossible contradictory lexicon".to_string(),
            name: "contradictory_class".to_string(),
            signature: "fn contradictory_class(s: string) -> i64",
            examples,
        };

        let (candidate, report) = self_extend(&engine, &req);
        assert!(candidate.is_none(), "an unsynthesizable gap must return None");
        assert!(!report.synthesized, "synthesis must report failure");
        assert!(!report.accepted, "an unsynthesized extension is never accepted");
        assert!(
            report.message.contains("no verified program"),
            "the report must explain the gap stayed open: {}",
            report.message
        );
        });
    }
}
