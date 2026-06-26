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
use crate::self_improve::gate::{regression_gate, GateReport};
use crate::self_improve::journal::{self, JournalEntry};
use crate::self_improve::store::{self, StoredComponent, StoredConstruction};
use crate::solved_cache::examples_fingerprint;
use crate::understanding::grammar::{
    learn_construction_from_examples, ConstructionExample, LearnedConstruction,
};

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

/// The per-teach time budget, in seconds. A single self-extension (synthesize +
/// gate) must complete within this wall-clock bound or it is honestly refused
/// rather than allowed to run unbounded — the cure for the "uninterruptible
/// minutes" failure mode. Overridable via `NCPU_TEACH_BUDGET_SECS` (e.g. CI may
/// widen it on slow hardware) or, for a precise sub-second bound in tests, via
/// `NCPU_TEACH_BUDGET_MS` (which takes precedence). An unset / unparseable / zero
/// value falls back to the default. The budget bounds the *caller's* wall-clock:
/// synthesis runs on a worker thread and the teach returns within the budget on
/// timeout (the worker is detached — the solver ignores SIGTERM internally — but
/// the teach no longer hangs and is interruptible from the caller's side).
// 30s default: the worker's synthesis is bounded to 70% of this (= 21s), which
// comfortably covers the enumerative solver's own default per-call budget (18s for
// unary ops) so an HONEST teach is never truncated — the teach budget bounds a
// RUNAWAY teach, it does not starve a legitimate one. CI/slow hosts can widen it
// via NCPU_TEACH_BUDGET_SECS.
const DEFAULT_TEACH_BUDGET_SECS: u64 = 30;

fn teach_budget() -> std::time::Duration {
    // Millisecond override wins when present (lets a test force a deterministic
    // sub-second timeout without a flaky 1s floor).
    if let Some(ms) = std::env::var("NCPU_TEACH_BUDGET_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&m| m > 0)
    {
        return std::time::Duration::from_millis(ms);
    }
    let secs = std::env::var("NCPU_TEACH_BUDGET_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_TEACH_BUDGET_SECS);
    std::time::Duration::from_secs(secs)
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
/// 3. **Journal.** Record a [`JournalEntry`] for the attempt (accepted or not).
/// 4. **Return.** `(Some(candidate), report)` with `accepted=true` on a green
///    gate; otherwise `(None, report)` whose `message` names the failing golden
///    cases, so the rejection is auditable.
///
/// `engine` is never mutated. The returned candidate (when `Some`) is a fresh,
/// already-gated `Engine` the caller may adopt as the new live engine.
///
/// BUDGET + CANCELLATION (UNWALL-4-OPT). The synthesize+gate work runs on a worker
/// thread and the caller waits only up to the per-teach budget ([`teach_budget`]);
/// on timeout it returns an honest "could not learn within budget" refusal instead
/// of hanging — a teach is bounded and interruptible from the caller's side.
pub fn self_extend(engine: &Engine, req: &LearnRequest) -> (Option<Engine>, LearnReport) {
    // --- 0. BUDGET + CANCEL ----------------------------------------------
    // Run the heavy, potentially-unbounded part of the teach — synthesis (the
    // solver search) followed by the regression gate — on a WORKER THREAD, and
    // wait for it only up to the per-teach budget. On timeout the teach returns an
    // honest "could not learn within budget" refusal instead of blocking the caller
    // for the minutes the solver might take. This is the cooperative-cancellation
    // boundary the substrate needs: a teach is now BOUNDED and INTERRUPTIBLE from
    // the caller's perspective. The detached worker may finish its solve in the
    // background, but its result is dropped (the channel receiver is gone) and it
    // never mutates `engine` — soundness is unaffected, because a green gate is
    // still required for any acceptance.
    let budget = teach_budget();

    // Bound the WORKER's own synthesis so it self-terminates near the deadline
    // instead of pinning a core for the solver's full default stage budgets after
    // the caller has already refused. The pipeline's two slow stages each honor a
    // COOPERATIVE deadline read from the environment — the enumerative search
    // (`NSYNTH_ENUM_BUDGET_MS`, default 18s) and teacher distillation
    // (`NSYNTH_TEACHER_BUDGET_SEC`, default 15s). Shrinking BOTH to fit the teach
    // budget makes the detached worker give up promptly, closing the "runaway
    // worker lingers for tens of seconds" gap. Both are restored after the recv
    // resolves (by then the worker has read them — recv only returns after `budget`
    // elapses).
    //
    // The worker budget is 70% of the teach budget (leaving room for the gate), but
    // FLOORED so the worker can always resolve-or-give-up in a bounded short time
    // and exit even when the teach budget is TINY — the caller's `recv_timeout`
    // still uses the full (possibly sub-second) teach budget to decide the refusal,
    // so a tiny budget still refuses promptly; the floor only governs how long the
    // already-refused worker keeps running before it stops. The 70% value is never
    // RAISED above an existing tighter env value, and for the 30s default it yields
    // ~21s of enumeration — above the solver's own 18s default — so an HONEST teach
    // is never truncated; only a runaway one is bounded.
    const WORKER_SOLVER_FLOOR_MS: u64 = 200;
    let worker_ms = (budget.as_millis() as u64 * 7 / 10).max(WORKER_SOLVER_FLOOR_MS);
    let prev_enum_budget = std::env::var("NSYNTH_ENUM_BUDGET_MS").ok();
    let effective_enum_ms = prev_enum_budget
        .as_deref()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|existing| existing.min(worker_ms))
        .unwrap_or(worker_ms);
    let prev_teacher_budget = std::env::var("NSYNTH_TEACHER_BUDGET_SEC").ok();
    // Teacher budget is in SECONDS (float). Convert the worker ms budget, flooring
    // at 0 so a sub-second teach budget disables the (otherwise 15s) teacher stage.
    let worker_teacher_sec = worker_ms as f64 / 1000.0;
    let effective_teacher_sec = prev_teacher_budget
        .as_deref()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|existing| existing.min(worker_teacher_sec))
        .unwrap_or(worker_teacher_sec);
    // SAFETY: in tests this runs under ENV_LOCK (held by the test helpers); in
    // production a teach is the only thing running. The restore below returns the
    // env to its prior state once the worker's result (or the timeout) is in.
    unsafe {
        std::env::set_var("NSYNTH_ENUM_BUDGET_MS", effective_enum_ms.to_string());
        std::env::set_var("NSYNTH_TEACHER_BUDGET_SEC", format!("{effective_teacher_sec}"));
    }

    let (tx, rx) = std::sync::mpsc::channel::<(Option<Engine>, bool, GateOutcome)>();
    {
        // Move owned copies onto the worker so the closure is 'static + Send.
        let name = req.name.clone();
        let signature = req.signature;
        let examples = req.examples.clone();
        let base = engine.clone();
        std::thread::spawn(move || {
            // 1. SYNTHESIZE + VERIFY on the worker.
            let result = match base.try_extend(&name, signature, examples) {
                Ok(candidate) => {
                    // 2. GATE on the worker.
                    let gate = regression_gate(&candidate);
                    let outcome = GateOutcome {
                        passed: gate.passed,
                        total: gate.total,
                        sound: gate.sound,
                        failures: gate.failures,
                        synthesis_error: None,
                    };
                    let ok = outcome.passed == outcome.total && outcome.sound;
                    (Some(candidate), ok, outcome)
                }
                Err(err) => (None, false, GateOutcome::synthesis_failure(err)),
            };
            // The receiver may already be gone (budget exceeded); ignore send error.
            let _ = tx.send(result);
        });
    }

    let recv = rx.recv_timeout(budget);
    // Restore the budgets now that the worker has read them (or the deadline hit).
    match &prev_enum_budget {
        Some(v) => unsafe { std::env::set_var("NSYNTH_ENUM_BUDGET_MS", v) },
        None => unsafe { std::env::remove_var("NSYNTH_ENUM_BUDGET_MS") },
    }
    match &prev_teacher_budget {
        Some(v) => unsafe { std::env::set_var("NSYNTH_TEACHER_BUDGET_SEC", v) },
        None => unsafe { std::env::remove_var("NSYNTH_TEACHER_BUDGET_SEC") },
    }

    let (candidate, passed_gate, outcome) = match recv {
        Ok(triple) => triple,
        Err(_) => {
            // BUDGET EXCEEDED: honest refusal, not a hang. The engine is untouched
            // and nothing is persisted; the gap stays open.
            let message = format!(
                "could not learn `{}` within budget ({}s): synthesis did not complete in time \
                 (honest refusal — the engine is unchanged and nothing was persisted)",
                req.name,
                budget.as_secs(),
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
                when_unix: 0,
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

    // SYNTHESIS FAILURE (well-formed but unsatisfiable / no candidate) — the worker
    // reported it with the synthesis error. The engine stays untouched, exactly as
    // the original synchronous path did.
    let Some(candidate) = candidate else {
        let err = outcome.synthesis_error.unwrap_or_default();
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
            when_unix: 0,
            gap: req.gap.clone(),
            action: format!("synthesize {}", req.name),
            method: String::new(),
            verified: false,
            regression_passed: false,
            accepted: false,
            note: message,
        });
        return (None, report);
    };

    // The recovering teacher for the freshly grafted component (last method).
    let method = candidate
        .methods
        .last()
        .map(|(_, m)| m.clone())
        .unwrap_or_default();

    // The gate already ran on the worker; reuse its verdict (the gate is memoized
    // by behavioral fingerprint, so this is consistent and cheap).
    let gate = GateReport {
        passed: outcome.passed,
        total: outcome.total,
        failures: outcome.failures,
        sound: outcome.sound,
    };
    finish_self_extend(engine, req, candidate, method, gate, passed_gate)
}

/// The carrier the synthesize+gate worker sends back: either a synthesis failure
/// (with the solver error) or a gate verdict for a synthesized candidate.
struct GateOutcome {
    passed: usize,
    total: usize,
    sound: bool,
    failures: Vec<String>,
    synthesis_error: Option<String>,
}

impl GateOutcome {
    fn synthesis_failure(err: String) -> Self {
        GateOutcome {
            passed: 0,
            total: 0,
            sound: false,
            failures: Vec::new(),
            synthesis_error: Some(err),
        }
    }
}

/// JOURNAL + PERSIST + RETURN for a synthesized candidate whose gate already ran —
/// the unchanged tail of [`self_extend`] (steps 3 + 4). Factored out so the
/// budget-bounded entry can share it. `passed_gate` is the gate's `ok()` verdict
/// already computed on the worker.
fn finish_self_extend(
    engine: &Engine,
    req: &LearnRequest,
    candidate: Engine,
    method: String,
    gate: GateReport,
    passed_gate: bool,
) -> (Option<Engine>, LearnReport) {
    let message = if passed_gate {
        format!(
            "accepted `{}` via {} (gate green: {}/{} golden cases passed, sound)",
            req.name, method, gate.passed, gate.total
        )
    } else {
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

    journal::record(&JournalEntry {
        when_unix: 0,
        gap: req.gap.clone(),
        action: format!("synthesize {}", req.name),
        method,
        verified: true,
        regression_passed: passed_gate,
        accepted: passed_gate,
        note: message,
    });

    if passed_gate {
        let base_len = engine.program().len();
        let code = candidate
            .program()
            .get(base_len..)
            .map(|s| s.trim_start_matches('\n').to_string())
            .unwrap_or_default();
        if !code.is_empty() {
            let members: Vec<(String, bool)> = if req.name.ends_with("_class") {
                req.examples
                    .iter()
                    .filter_map(|ex| match (ex.inputs.first(), &ex.expected) {
                        (
                            Some(crate::benchmark::Value::Str(w)),
                            crate::benchmark::Value::Int(l),
                        ) => Some((w.clone(), *l == 1)),
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
                method: report.method.clone(),
                examples_fingerprint: examples_fingerprint(&req.examples),
                members,
            });
        }
        (Some(candidate), report)
    } else {
        (None, report)
    }
}

/// A request to acquire a word-order CONSTRUCTION (grammar induction) by learning
/// the role-to-position mapping for one family of class skeletons.
///
/// * `gap` — a human-readable description of what the parser could not handle
///   (e.g. "object-fronted declaratives parse to Unknown").
/// * `name` — the construction's tag (e.g. `"object_fronting"`).
/// * `examples` — labeled `(sentence, agent_word, patient_word, predicate_lemma)`
///   tuples (see [`ConstructionExample`]); the learner is told the ROLES and
///   induces the position mapping, then SYNTHESIZES + VERIFIES it as a program over
///   the class skeletons.
pub struct ConstructionRequest<'a> {
    pub gap: String,
    pub name: String,
    pub examples: Vec<ConstructionExample<'a>>,
}

/// Acquire a word-order construction, fully gated — the grammar-induction analogue
/// of [`self_extend`].
///
/// Enforces the SAME substrate contract end-to-end:
///
/// 1. **Synthesize + verify.** Call
///    [`learn_construction_from_examples`] to induce the role-to-position mapping
///    and PROVE it as `[i64] -> i64` slot programs over the class skeletons. A
///    failure (no recoverable rule, or an ill-formed skeleton→role mapping) is
///    journaled and the engine is untouched.
/// 2. **Gate.** Register the construction onto a CLONE of `engine` and run the
///    whole golden corpus + soundness oracle against it. A sound construction
///    (whose skeleton appears in no base-parseable golden case) leaves the gate
///    green; one whose skeleton COLLIDES with a base-parseable pattern would change
///    a golden answer and redden the gate. The parser fallback consults the
///    construction ONLY on an otherwise-Unknown parse, so the gate is the enforced
///    proof that the addition broke nothing.
/// 3. **Journal.** Record a [`JournalEntry`] for the attempt (accepted or not).
/// 4. **Persist (accepted only).** Durably record the construction as a
///    [`StoredConstruction`] so a later `Engine::new` can re-register it (gated
///    again) — cross-run compounding grammar.
///
/// Returns `(Some(engine_with_construction), report)` on accept, else
/// `(None, report)` with `engine` untouched. Guarantees monotone growth: a
/// construction that breaks anything is rejected.
pub fn self_learn_construction(
    engine: &Engine,
    req: &ConstructionRequest,
) -> (Option<Engine>, LearnReport) {
    // --- 1. SYNTHESIZE + VERIFY ------------------------------------------
    let construction = match learn_construction_from_examples(engine, &req.name, &req.examples) {
        Ok(c) => c,
        Err(err) => {
            let message = format!(
                "no verified construction for gap {:?}: induction of `{}` failed ({})",
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
                when_unix: 0,
                gap: req.gap.clone(),
                action: format!("induce construction {}", req.name),
                method: "grammar_induction".to_string(),
                verified: false,
                regression_passed: false,
                accepted: false,
                note: message,
            });
            return (None, report);
        }
    };

    // --- 2. GATE ----------------------------------------------------------
    let mut candidate = engine.clone();
    candidate.register_construction(construction.clone());
    let gate = regression_gate(&candidate);
    let passed_gate = gate.ok();

    let message = if passed_gate {
        format!(
            "accepted construction `{}` (gate green: {}/{} golden cases passed, sound)",
            req.name, gate.passed, gate.total
        )
    } else {
        let failures = if gate.failures.is_empty() {
            "(no behavioral failures)".to_string()
        } else {
            gate.failures.join("; ")
        };
        format!(
            "rejected construction `{}`: regression gate red ({}/{} golden cases passed, \
             sound={}); failing cases: {}",
            req.name, gate.passed, gate.total, gate.sound, failures
        )
    };

    let report = LearnReport {
        gap: req.gap.clone(),
        synthesized: true,
        method: "grammar_induction".to_string(),
        regression_passed: passed_gate,
        accepted: passed_gate,
        message: message.clone(),
    };

    // --- 3. JOURNAL -------------------------------------------------------
    journal::record(&JournalEntry {
        when_unix: 0,
        gap: req.gap.clone(),
        action: format!("induce construction {}", req.name),
        method: "grammar_induction".to_string(),
        verified: true,
        regression_passed: passed_gate,
        accepted: passed_gate,
        note: message,
    });

    // --- 4. PERSIST (accepted only) + RETURN ------------------------------
    if passed_gate {
        persist_construction(&construction);
        (Some(candidate), report)
    } else {
        (None, report)
    }
}

/// Persist an accepted construction to the durable construction store (best-effort;
/// `save_one_construction` swallows I/O errors and is a no-op when the store is
/// disabled, so a store failure never blocks adoption — the construction is already
/// live on the returned engine).
fn persist_construction(c: &LearnedConstruction) {
    store::save_one_construction(&StoredConstruction {
        name: c.name.clone(),
        skeletons: c.skeletons.clone(),
        agent_idx: c.agent_idx,
        patient_idx: c.patient_idx,
        predicate_idx: c.predicate_idx,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comprehension::creature_class_examples;

    /// FAST-GATE LEVER (the "minutes -> seconds" proof): the base curriculum is a
    /// pure constant, but the solved-program cache does NOT cover it (it holds
    /// array/numeric problems, not the comprehension lexicons), so EVERY
    /// `Engine::new_base()` used to re-synthesize all 11 components through the
    /// solver — *minutes* on a cold/contended host. A single teach-then-reuse flow
    /// calls `Engine::new()` several times (teach engine + each fresh reload), and
    /// the reload re-gates every stored component, so that per-call cost dominated
    /// the whole flow. The UNWALL-4-OPT memoization makes the synthesis run AT MOST
    /// ONCE per process; every later `Engine::new_base()` is a `String`+`Vec` clone.
    ///
    /// This test proves the lever directly and un-gameably: it times the FIRST
    /// `new_base()` (which pays the one-time synthesis) and a SECOND one (which must
    /// hit the memo), and asserts the second is DRAMATICALLY faster — at least 100x,
    /// and under 50ms in absolute terms — AND that the two engines are behaviorally
    /// identical (same program), so the memo returns the real base, not a stub.
    /// Without the memo the second call would re-pay the full synthesis and this
    /// ratio could never hold.
    #[test]
    fn base_engine_construction_is_memoized_after_first_build() {
        use std::time::Instant;

        // First build pays the one-time synthesis (cold). We don't assert on its
        // duration (it is the cost we are AMORTIZING), only that it produces a real
        // engine.
        let t0 = Instant::now();
        let first = Engine::new_base();
        let first_dur = t0.elapsed();
        assert!(
            first.has_component("noun_animacy"),
            "the base engine must carry its synthesized components"
        );

        // Second build MUST hit the process-global memo: no solver calls, just a
        // clone of the cached (program, methods).
        let t1 = Instant::now();
        let second = Engine::new_base();
        let second_dur = t1.elapsed();

        // The memoized build is a clone — it must be far faster than the first AND
        // fast in absolute terms. (On a quiet host the first build is seconds-to-
        // minutes; the clone is microseconds. We use conservative thresholds so the
        // assertion is robust under heavy CI contention while still proving the
        // synthesis did not re-run.)
        assert!(
            second_dur < std::time::Duration::from_millis(50),
            "the memoized base build must be a fast clone (<50ms); took {second_dur:?} \
             (first build was {first_dur:?})"
        );
        assert!(
            second_dur * 100 < first_dur,
            "the memoized build must be >=100x faster than the first (synthesis ran once); \
             first={first_dur:?} second={second_dur:?}"
        );

        // The memo returns the REAL base, byte-for-byte — not a stub or partial.
        assert_eq!(
            first.program(),
            second.program(),
            "the memoized base must be identical to the first-built base"
        );
        assert!(
            second.has_component("valid_argument") && second.has_component("verb_3sg"),
            "the memoized base must carry every synthesized component + wrapper"
        );
    }

    /// BUDGET + CANCELLATION: a teach whose budget is exceeded must return an
    /// HONEST REFUSAL — not hang. We force the budget to 1ms via
    /// `NCPU_TEACH_BUDGET_MS` so the synthesis worker cannot possibly complete in
    /// time, then prove:
    ///   * the call RETURNS (the test itself would time out / hang if it didn't);
    ///   * `accepted == false` and `synthesized == false` (nothing was learned);
    ///   * the report message says it could not learn within budget (auditable
    ///     refusal, not a fabricated success);
    ///   * the input engine is UNCHANGED (no component was grafted, no store write);
    ///   * the whole call completes well under a generous wall-clock ceiling, proving
    ///     boundedness (it does not wait for the minutes-long solve to finish).
    /// This is the cure for the "uninterruptible minutes" failure mode: a teach is
    /// now bounded and interruptible from the caller's side.
    #[test]
    fn teach_exceeding_budget_is_refused_not_hung() {
        crate::self_improve::journal::test_support::with_journal_env("", || {
            // ENV_LOCK is held by with_journal_env for this whole closure, so
            // setting the budget env here is race-free.
            let prev = std::env::var("NCPU_TEACH_BUDGET_MS").ok();
            // SAFETY: ENV_LOCK guarantees single-threaded access for the duration.
            unsafe { std::env::set_var("NCPU_TEACH_BUDGET_MS", "1") }

            let engine = Engine::new_base();
            assert!(
                !engine.has_component("budget_probe_class"),
                "precondition: the op must not pre-exist"
            );

            // The spec is a STRING-input classifier with a CONTRADICTORY label
            // ("a" -> 1 AND "a" -> 0). Two properties make the detached worker
            // short-lived: (1) a string (non-scalar) input routes AWAY from the
            // numeric gradient stage — the pipeline's only stage without an
            // env-readable deadline — so the worker stays in the budget-bounded
            // enumerative path; (2) the contradiction is unsatisfiable, so that
            // bounded search exhausts and the worker FAILS FAST, then exits. With
            // the 1ms teach budget the CALLER's `recv_timeout` fires long before the
            // worker can even spawn + clone + enter the solver, so the refusal is
            // deterministically the BUDGET path — proving a teach is bounded and
            // interruptible from the caller's side regardless of what the worker is
            // doing. (Genuine synthesis success/failure paths are covered by the
            // good_extension / unsatisfiable tests.)
            let req = LearnRequest {
                gap: "budget probe".to_string(),
                name: "budget_probe_class".to_string(),
                signature: "fn budget_probe_class(s: string) -> i64",
                examples: vec![
                    crate::benchmark::Example {
                        inputs: vec![crate::benchmark::Value::Str("a".to_string())],
                        expected: crate::benchmark::Value::Int(1),
                    },
                    crate::benchmark::Example {
                        inputs: vec![crate::benchmark::Value::Str("a".to_string())],
                        expected: crate::benchmark::Value::Int(0),
                    },
                ],
            };

            let started = std::time::Instant::now();
            let (candidate, report) = self_extend(&engine, &req);
            let elapsed = started.elapsed();

            // BOUNDED: returned far under a generous ceiling (the un-budgeted solve
            // would take many seconds-to-minutes). 5s is a wide margin for thread
            // spawn + the 1ms recv timeout under a contended host.
            assert!(
                elapsed < std::time::Duration::from_secs(5),
                "a budget-exceeded teach must return promptly, not hang; took {elapsed:?}"
            );

            // HONEST REFUSAL: nothing learned, nothing accepted, message explains.
            assert!(!report.accepted, "a budget-exceeded teach must NOT be accepted");
            assert!(
                !report.synthesized,
                "a budget-exceeded teach reports no synthesized component"
            );
            assert!(
                candidate.is_none(),
                "a budget-exceeded teach returns no candidate engine"
            );
            assert!(
                report.message.contains("within budget"),
                "the refusal must explain the budget was exceeded: {}",
                report.message
            );

            // ENGINE UNTOUCHED.
            assert!(
                !engine.has_component("budget_probe_class"),
                "a budget-exceeded teach must not graft anything onto the input engine"
            );

            match prev {
                Some(v) => unsafe { std::env::set_var("NCPU_TEACH_BUDGET_MS", v) },
                None => unsafe { std::env::remove_var("NCPU_TEACH_BUDGET_MS") },
            }
        });
    }

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
            assert!(
                report.synthesized,
                "creature_class must synthesize: {}",
                report.message
            );
            assert!(
                report.regression_passed,
                "the gate must stay green for an additive disjoint component: {}",
                report.message
            );
            assert!(
                report.accepted,
                "a synthesized + gated extension must be accepted: {}",
                report.message
            );
            assert!(
                !report.method.is_empty(),
                "the recovering teacher must be recorded"
            );

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
                before.passed,
                before.total,
                before.sound,
                before.failures
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
                after.passed,
                after.total,
                after.sound,
                after.failures
            );

            // (5) The rejection is auditable: the message records the gate verdict and
            // names at least one failing golden case (a taxonomy/agreement question).
            assert!(
                report.message.contains("rejected")
                    && report.message.contains("regression gate red"),
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
            assert!(
                !mine.accepted,
                "the journaled attempt was rejected, not accepted"
            );
            assert!(
                !mine.regression_passed,
                "the journaled attempt failed the gate"
            );
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
            assert!(
                candidate.is_none(),
                "an unsynthesizable gap must return None"
            );
            assert!(!report.synthesized, "synthesis must report failure");
            assert!(
                !report.accepted,
                "an unsynthesized extension is never accepted"
            );
            assert!(
                report.message.contains("no verified program"),
                "the report must explain the gap stayed open: {}",
                report.message
            );
        });
    }

    /// The labeled OSV training set (mirrors `understanding::grammar`'s
    /// `osv_examples`): three sentences, same word-order shape, different words,
    /// each tagged with its agent / patient surface word and predicate lemma.
    fn osv_examples() -> Vec<ConstructionExample<'static>> {
        vec![
            (
                "the report the teacher writes",
                "teacher",
                "report",
                "write",
            ),
            ("the book the student reads", "student", "book", "read"),
            ("the memo the doctor fixes", "doctor", "memo", "fix"),
        ]
    }

    /// END-TO-END learn-accept for a CONSTRUCTION: `self_learn_construction` induces
    /// the OSV role mapping from labeled examples, SYNTHESIZES + VERIFIES the slot
    /// programs, runs the candidate through the regression gate (green — the OSV
    /// skeleton appears in no base-parseable golden case), accepts it, REGISTERS it
    /// on the returned engine (which now parses OSV), and PERSISTS it to the
    /// construction store so a later boot can re-register it.
    #[test]
    fn good_construction_is_induced_gated_accepted_and_persisted() {
        // Fence the journal + component store (disabled) and point the CONSTRUCTION
        // store at a fresh temp file so the accept-time persist is observable but
        // never touches $HOME. The journal-env helper holds the crate-wide ENV_LOCK,
        // so setting NCPU_CONSTRUCTIONS_PATH inside the closure is race-free.
        crate::self_improve::journal::test_support::with_journal_env("", || {
            let path = std::env::temp_dir().join(format!(
                "ncpu_construction_accept_{}_{:?}.jsonl",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_file(&path);
            let prev = std::env::var("NCPU_CONSTRUCTIONS_PATH").ok();
            // SAFETY: with_journal_env holds ENV_LOCK for this whole closure.
            unsafe { std::env::set_var("NCPU_CONSTRUCTIONS_PATH", &path) }

            let engine = Engine::new();
            assert!(
                engine.learned_grammar().is_empty(),
                "fresh engine has no acquired constructions"
            );

            let req = ConstructionRequest {
                gap: "object-fronted declaratives parse to Unknown".to_string(),
                name: "object_fronting".to_string(),
                examples: osv_examples(),
            };
            let (candidate, report) = self_learn_construction(&engine, &req);

            // Induced, gated green, accepted.
            assert!(
                report.synthesized,
                "OSV must induce + verify: {}",
                report.message
            );
            assert!(
                report.regression_passed,
                "an additive OSV construction must pass the gate (its skeleton is not \
                 base-parseable): {}",
                report.message
            );
            assert!(
                report.accepted,
                "a verified + gated construction must be accepted"
            );

            // The returned engine registered it AND now parses OSV correctly.
            let learned = candidate.expect("an accepted construction returns Some(engine)");
            assert_eq!(learned.learned_grammar().len(), 1);
            let m = crate::understanding::semantics::understand(
                &learned,
                "the letter the editor reads",
            );
            let crate::understanding::meaning::Meaning::Event(e) = m else {
                panic!("the learned construction must parse unseen OSV to an Event, got {m:?}");
            };
            assert_eq!(e.predicate, "read");
            assert_eq!(
                e.agent,
                Some(crate::understanding::meaning::Term::Entity(
                    "editor".to_string()
                ))
            );
            assert_eq!(
                e.patient,
                Some(crate::understanding::meaning::Term::Entity(
                    "letter".to_string()
                ))
            );

            // PERSISTED: the accepted construction is durably in the store.
            let stored = store::load_constructions();
            assert_eq!(
                stored.len(),
                1,
                "the accepted construction must be persisted"
            );
            assert_eq!(stored[0].name, "object_fronting");
            assert_eq!(stored[0].skeletons, vec![vec![0, 1, 0, 1, 2]]);
            assert_eq!(stored[0].agent_idx, 3);
            assert_eq!(stored[0].patient_idx, 1);
            assert_eq!(stored[0].predicate_idx, 4);

            // The INPUT engine is untouched (self_learn_construction grafts onto a clone).
            assert!(
                engine.learned_grammar().is_empty(),
                "self_learn_construction must not mutate the input engine"
            );

            let _ = std::fs::remove_file(&path);
            match prev {
                Some(v) => unsafe { std::env::set_var("NCPU_CONSTRUCTIONS_PATH", v) },
                None => unsafe { std::env::remove_var("NCPU_CONSTRUCTIONS_PATH") },
            }
        });
    }
}
