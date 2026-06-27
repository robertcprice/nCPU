//! EMERGENT rule-learning from examples (LOOP-2).
//!
//! The user's core vision: show the engine a small CORPUS of example rule
//! functions (each as `{name, i64 examples}`); the engine LEARNS each as a
//! reusable, persisted op; then it EMERGENTLY synthesizes a NEW target rule it
//! could NOT synthesize before, by REUSING a learned op via a real `Call` node
//! in the discovered program. Understanding *grows*: a target unsynthesizable on
//! an empty op-store becomes synthesizable after the right building block is
//! ingested, and a second target stays out of reach until ITS block is added too.
//!
//! This module is GLUE ONLY — it re-implements NOTHING:
//!   * INGEST routes each corpus op through the EXISTING learn-on-the-fly spine
//!     [`crate::learn_nl::teach_by_examples`] (UNWALL-4), which synthesizes via
//!     the real solver, runs the regression gate, and PERSISTS the op to the
//!     durable component store ([`crate::self_improve::store`],
//!     `NCPU_COMPONENTS_PATH`). The op body is DISCOVERED by search from its
//!     examples — never authored here.
//!   * REUSE bridges a PERSISTED [`StoredComponent`] into a
//!     [`crate::enumerative::NamedCallable`] (its Mog `source` + an `eval` that
//!     RUNS that source), then drives the EXISTING Call-node search
//!     [`crate::enumerative::synthesize_scalar_with_callees`] (UNWALL-2). The
//!     search CHOOSES to emit `Call(learned_op, ..)` — there is no phrase→op
//!     table and no hand-written target body.
//!
//! The emergent argument is therefore intact end-to-end: the engine's *own*
//! search discovers both the building block (from its examples) and the target's
//! reuse of it (as a Call), and the new capability is gated + persisted.

use crate::comprehension::Engine;
use crate::enumerative::NamedCallable;
use crate::self_improve::store;

/// Build a single-arg i64→i64 [`crate::benchmark::Problem`] from `(in, out)`
/// pairs, so a target rule can be handed to the Call-node search. Holdouts are
/// empty here because the accept-test supplies an INDEPENDENT by-hand oracle on
/// 100+ fresh inputs after synthesis (the un-gameable correctness proof); the
/// strict-verify gate inside the search still runs on the seed examples.
pub fn scalar_problem(name: &str, pairs: &[(i64, i64)]) -> crate::benchmark::Problem {
    use crate::benchmark::{Example, Value};
    let examples = pairs
        .iter()
        .map(|(i, o)| Example {
            inputs: vec![Value::Int(*i)],
            expected: Value::Int(*o),
        })
        .collect();
    let sig: &'static str =
        Box::leak(format!("fn {name}(a: i64) -> i64").into_boxed_str());
    let name_static = name.to_string();
    crate::benchmark::Problem {
        name: name_static,
        category: "emergent-rule",
        description: "",
        signature: sig,
        examples,
        ..Default::default()
    }
}

/// INGEST one corpus op: teach `{name, pairs}` through the EXISTING
/// learn-on-the-fly spine. Returns `true` iff the op was synthesized, gated, and
/// PERSISTED (the engine's own verdict — never fabricated). The op body is
/// discovered BY SEARCH from `pairs`, not authored.
pub fn ingest_op(engine: &Engine, name: &str, pairs: &[(i64, i64)]) -> bool {
    crate::learn_nl::teach_by_examples(engine, name, pairs).success
}

/// Bridge a PERSISTED learned op (resolved from the durable store by `name`) into
/// a [`NamedCallable`] the Call-node search can REUSE.
///
/// The returned callable carries the op's stored Mog `source` (so the
/// strict-verify gate can resolve the emitted `name(args)` call) and an `eval`
/// that RUNS that source on concrete args (so a `Call(op, args)` candidate is
/// verified end-to-end). Returns `None` if `name` is not a single-arg op in the
/// store — honest, never a stub. The op is the engine's OWN search-discovered
/// program, only re-registered here as a callable primitive.
pub fn learned_callable(name: &str) -> Option<NamedCallable> {
    let comp = store::load().into_iter().find(|c| c.name == name)?;
    if comp.code.trim().is_empty() {
        return None;
    }
    let src = comp.code.clone();
    let fn_name = comp.name.clone();
    let run_src = src.clone();
    let run_name = fn_name.clone();
    Some(NamedCallable {
        name: fn_name,
        n_args: 1, // the learn spine teaches single-arg i64->i64 ops
        source: src,
        eval: Box::new(move |xs: &[i64]| {
            if xs.len() != 1 {
                return None;
            }
            match crate::runtime::execute_function(
                &run_src,
                &run_name,
                &[crate::benchmark::Value::Int(xs[0])],
                "emergent-rule-callee",
            ) {
                Ok(crate::runtime::Value::Int(v)) => Some(v),
                _ => None,
            }
        }),
    })
}

/// SYNTHESIZE a target rule by REUSING the named learned ops.
///
/// Resolves each name in `callee_names` to a [`learned_callable`] (from the
/// durable store), registers them as real callable primitives, and runs the
/// EXISTING Call-node search [`synthesize_scalar_with_callees`]. The search may
/// emit a `Call(learned_op, ..)` — emergently, by size-competition, not by any
/// mapping. Returns the verified [`SolveResult`] (its `.code` strict-verifies
/// against the seed examples with the callee source prepended).
///
/// `Err` when a requested callee is not a learned single-arg op in the store
/// (honest — a name that is not learned is simply not reusable).
pub fn synthesize_target_reusing(
    problem: &crate::benchmark::Problem,
    callee_names: &[&str],
) -> Result<crate::solver::SolveResult, String> {
    let mut callees = Vec::with_capacity(callee_names.len());
    for name in callee_names {
        let c = learned_callable(name).ok_or_else(|| {
            format!("`{name}` is not a learned single-arg op in the durable store")
        })?;
        callees.push(c);
    }
    crate::enumerative::synthesize_scalar_with_callees(problem, &callees)
        .ok_or_else(|| "Call-node search found no program for the target rule".to_string())
}

/// Inspect-only sibling of [`synthesize_target_reusing`]: return the solved
/// `Expr` AST (not the Mog string) so a test can assert STRUCTURALLY that the
/// solution contains a `Call` to a learned op. Reuses the existing test-only
/// inspector [`crate::enumerative::solve_scalar_expr_with_callees`].
pub fn solve_target_expr_reusing(
    problem: &crate::benchmark::Problem,
    callee_names: &[&str],
    budget_ms: u64,
) -> Result<crate::enumerative::Expr, String> {
    let mut callees = Vec::with_capacity(callee_names.len());
    for name in callee_names {
        let c = learned_callable(name).ok_or_else(|| {
            format!("`{name}` is not a learned single-arg op in the durable store")
        })?;
        callees.push(c);
    }
    crate::enumerative::solve_scalar_expr_with_callees(problem, &callees, budget_ms)
        .ok_or_else(|| "Call-node search found no Expr for the target rule".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerative::{expr_has_call, Expr};

    /// Does `e` contain a `Call` to the callee at registry index `idx`?
    fn calls_idx(e: &Expr, idx: usize) -> bool {
        match e {
            Expr::Call(i, args) => *i == idx || args.iter().any(|a| calls_idx(a, idx)),
            Expr::UnaryOp(_, c) => calls_idx(c, idx),
            Expr::BinOp(_, l, r) => calls_idx(l, idx) || calls_idx(r, idx),
            Expr::IfExpr(_, a, b, c, d) => {
                calls_idx(a, idx) || calls_idx(b, idx) || calls_idx(c, idx) || calls_idx(d, idx)
            }
            _ => false,
        }
    }

    /// Run `f` with the durable component store + the journal + base cache pointed
    /// at fresh temp files, holding the crate-wide ENV_LOCK so the process-global
    /// env mutation never races another env-mutating test. Mirrors
    /// `learn_nl::tests::with_temp_component_store` (the established harness).
    fn with_temp_store<R>(f: impl FnOnce(&std::path::Path) -> R) -> R {
        use crate::self_improve::journal::test_support::ENV_LOCK;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("NCPU_COMPONENTS_PATH").ok();
        let prev_journal = std::env::var("NCPU_JOURNAL_PATH").ok();
        let prev_cache = std::env::var("NSYNTH_CACHE_PATH").ok();
        let prev_budget = std::env::var("NCPU_TEACH_BUDGET_SECS").ok();
        let path = std::env::temp_dir().join(format!(
            "ncpu_emergent_rule_store_{}_{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        // Share the SAME process-wide warm base cache as the learn_nl tests so the
        // several `Engine::new()` calls synthesize the base curriculum at most once
        // per process. This never weakens an assertion (the learned op still flows
        // through the real solver, gate, and durable store).
        let cache = std::env::temp_dir().join("ncpu_learn_nl_base_cache.json");
        // SAFETY: ENV_LOCK guarantees single-threaded access for the duration.
        unsafe {
            std::env::set_var("NCPU_COMPONENTS_PATH", &path);
            std::env::set_var("NCPU_JOURNAL_PATH", "");
            std::env::set_var("NSYNTH_CACHE_PATH", &cache);
            std::env::set_var("NCPU_TEACH_BUDGET_SECS", "300");
        }
        let result = f(&path);
        match prev {
            Some(v) => unsafe { std::env::set_var("NCPU_COMPONENTS_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_COMPONENTS_PATH") },
        }
        match prev_journal {
            Some(v) => unsafe { std::env::set_var("NCPU_JOURNAL_PATH", v) },
            None => unsafe { std::env::remove_var("NCPU_JOURNAL_PATH") },
        }
        match prev_cache {
            Some(v) => unsafe { std::env::set_var("NSYNTH_CACHE_PATH", v) },
            None => unsafe { std::env::remove_var("NSYNTH_CACHE_PATH") },
        }
        match prev_budget {
            Some(v) => unsafe { std::env::set_var("NCPU_TEACH_BUDGET_SECS", v) },
            None => unsafe { std::env::remove_var("NCPU_TEACH_BUDGET_SECS") },
        }
        let _ = std::fs::remove_file(&path);
        result
    }

    /// The building block g(x) = 3x + 1, taught from examples (search-discovered).
    fn g(x: i64) -> i64 {
        3 * x + 1
    }
    /// The target rule T(x) = g(g(x)) = 9x + 4. Its NATURAL solution reuses g.
    fn t(x: i64) -> i64 {
        g(g(x))
    }
    /// The second building block sq(x) = x*x.
    /// The second building block b(y) = 4y + 6. Both constants (4, 6) are OUTSIDE
    /// the enumerator's primitive const set, so b's contribution cannot be folded
    /// away by the base grammar — T2 below genuinely REQUIRES b as a callable.
    fn b(y: i64) -> i64 {
        4 * y + 6
    }
    /// The second target T2(x) = b(g(x)) = 4*(3x+1)+6 = 12x+10. Its natural solution
    /// reuses BOTH learned ops as a Call(b, Call(g, x)). Empirically (see probe),
    /// T2 is UNSYNTHESIZABLE with only g registered, and synthesizes once b is added
    /// — the monotonic-growth differential.
    fn t2(x: i64) -> i64 {
        b(g(x))
    }

    fn pairs(f: impl Fn(i64) -> i64, xs: &[i64]) -> Vec<(i64, i64)> {
        xs.iter().map(|&x| (x, f(x))).collect()
    }

    /// Run `code`'s `fn_name(x)` and return its i64 result (None on any failure).
    /// `runtime::Value` is not `PartialEq`, so we project to i64 for assertions.
    fn run_i64(code: &str, fn_name: &str, x: i64) -> Option<i64> {
        match crate::runtime::execute_function(
            code,
            fn_name,
            &[crate::benchmark::Value::Int(x)],
            "erl-byhand",
        ) {
            Ok(crate::runtime::Value::Int(v)) => Some(v),
            _ => None,
        }
    }

    /// THE UN-GAMEABLE EMERGENT RULE-LEARNING ACCEPT-TEST.
    ///
    /// (A) BASELINE FAIL — T does NOT synthesize on a FRESH engine with an EMPTY
    ///     op-store within a bounded budget (assert `Err`).
    /// (B) INGEST — teach g (T's building block) via the real learn-on-the-fly
    ///     spine; the op body is DISCOVERED by search from its examples + persisted.
    /// (C) REUSE — re-attempt T: it now SYNTHESIZES and its program genuinely CALLS
    ///     the learned g (assert `Expr::Call(g_idx, ..)` appears AND the emitted
    ///     Mog body names `g`); the solution strict-verifies; by-hand on >100
    ///     independent inputs.
    /// (D) MONOTONIC GROWTH — T2 stays unsolvable with ONLY g ingested, becomes
    ///     solvable after a SECOND ingest adds sq (cumulative understanding).
    /// (E) EMERGENT — the reuse comes from Call-node SEARCH over the store, with NO
    ///     hardcoded phrase→op mapping and NO hand-written target body.
    /// (F) PERSIST — the op-store round-trips to disk and reloads reusable.
    ///
    /// Slow (real solver synthesis + gated `Engine::new()` reloads); run with
    /// `cargo test --features nl --lib emergent_rule_learning -- --ignored`.
    #[test]
    #[ignore = "slow: full solver synthesis + gated Engine::new() reloads; run with --ignored"]
    fn emergent_rule_learning_un_gameable_proof() {
        with_temp_store(|store_path| {
            // Use a unique op name per run so a warm process never short-circuits the
            // baseline-fail via a stale store row.
            let gname = "erl_g_3xp1";
            let bname = "erl_b_4yp6";
            let budget_ms: u64 = 4_000;

            // ── (A) BASELINE FAIL: empty store, T does NOT synthesize. ───────────
            // PRECONDITION: g is NOT a learned op yet.
            let before = Engine::new();
            assert!(
                !before.has_component(gname),
                "precondition: {gname} must NOT pre-exist as a component"
            );
            assert!(
                store::load().iter().all(|c| c.name != gname),
                "precondition: store must not carry {gname} yet"
            );
            let t_problem = scalar_problem("erl_t", &pairs(t, &[0, 1, 2, 3, 4, 5, 6, 7]));
            // With an EMPTY callee set the search is the base scalar search. T(x)=9x+4
            // has no const 9 or 4 in the primitive const set, so it is out of reach
            // within the bounded budget — the baseline cannot find it.
            let baseline = crate::enumerative::solve_scalar_expr_with_callees(
                &t_problem,
                &[],
                budget_ms,
            );
            assert!(
                baseline.is_none(),
                "BASELINE must FAIL: T(x)=9x+4 must not synthesize on an empty op-store \
                 within {budget_ms}ms, but got {baseline:?}"
            );

            // ── (B) INGEST g via the real learn-on-the-fly spine. ────────────────
            let teach_engine = Engine::new();
            let learned = ingest_op(&teach_engine, gname, &pairs(g, &[1, 2, 3, 4, 5]));
            assert!(learned, "INGEST: {gname} must synthesize+gate+persist via teach_by_examples");

            // ── (F) PERSIST: store round-trips to disk with g's discovered body. ─
            assert!(store_path.exists(), "PERSIST: component store file must exist after ingest");
            let g_comp = store::load().into_iter().find(|c| c.name == gname)
                .expect("PERSIST: g must be persisted to the durable store");
            assert!(
                !g_comp.code.trim().is_empty(),
                "PERSIST: g's stored Mog SOURCE (search-discovered body) must be non-empty"
            );
            // The discovered body must actually compute 3x+1 (independent by-hand on
            // the persisted source — proves we reuse a CORRECT learned op, not a name).
            for x in [-9i64, -1, 0, 4, 13, 77] {
                assert_eq!(
                    run_i64(&g_comp.code, gname, x), Some(g(x)),
                    "PERSIST: persisted g must compute 3x+1 on x={x}"
                );
            }

            // ── (C) REUSE: T now synthesizes and CALLS the learned g. ────────────
            // Structural proof: the solved Expr contains Call(g_idx=0, ..).
            let t_expr = solve_target_expr_reusing(&t_problem, &[gname], budget_ms)
                .expect("REUSE: T must synthesize once g is a learned callable");
            assert!(
                expr_has_call(&t_expr),
                "REUSE: T's solution must contain a Call node, got {t_expr:?}"
            );
            assert!(
                calls_idx(&t_expr, 0),
                "REUSE: T's solution must CALL the learned op (registry idx 0), got {t_expr:?}"
            );
            // End-to-end proof: the verified SolveResult's emitted Mog body names g
            // (anti-inline) and strict-verifies (inside synthesize_target_reusing).
            let t_solved = synthesize_target_reusing(&t_problem, &[gname])
                .expect("REUSE: T must produce a strict-verified SolveResult reusing g");
            assert!(t_solved.success, "REUSE: T's SolveResult must be success");
            assert!(
                crate::agent::repo::body_calls_fn(&t_solved.code, gname),
                "REUSE: emitted T body must genuinely CALL {gname} (not inlined): {}",
                t_solved.code
            );
            // ── BY-HAND on >100 INDEPENDENT inputs (correctness, not overfit). ───
            // Reconstruct the full callable program (g's source + T's body) and RUN it.
            let full = format!("{}\n\n{}", g_comp.code.trim_end(), t_solved.code);
            for x in -60..=60 {
                assert_eq!(
                    run_i64(&full, "erl_t", x), Some(t(x)),
                    "BY-HAND: T(x) must equal 9x+4 on independent input x={x}"
                );
            }
            // ── (D) MONOTONIC GROWTH: T2 = b(g(x)) needs b, NOT yet ingested. ────
            let t2_problem = scalar_problem("erl_t2", &pairs(t2, &[0, 1, 2, 3, 4, 5, 6, 7]));
            // With ONLY g registered, T2(x) = 4*(3x+1)+6 = 12x+10 is out of reach:
            // its 4 and 6 are outside the const set and cannot be cheaply folded over
            // g within budget (the `t2_before.is_err()` assert below is the proof).
            let t2_before = solve_target_expr_reusing(&t2_problem, &[gname], budget_ms);
            assert!(
                t2_before.is_err(),
                "GROWTH: T2 must stay unsolvable with only g ingested, got {t2_before:?}"
            );
            // SECOND INGEST: teach b through the SAME real learn-on-the-fly spine.
            let teach2 = Engine::new();
            let learned2 = ingest_op(&teach2, bname, &pairs(b, &[0, 1, 2, 3, 4, 5]));
            assert!(learned2, "GROWTH: {bname} must synthesize+gate+persist");
            // Now T2 synthesizes, reusing BOTH learned ops via a Call(b, Call(g, x)).
            let t2_expr = solve_target_expr_reusing(&t2_problem, &[gname, bname], budget_ms)
                .expect("GROWTH: T2 must synthesize after the SECOND ingest adds b");
            assert!(
                expr_has_call(&t2_expr),
                "GROWTH: T2's solution must contain a Call node, got {t2_expr:?}"
            );
            // The second op (registry idx 1) must be CALLED — proves cumulative reuse.
            assert!(
                calls_idx(&t2_expr, 1),
                "GROWTH: T2's solution must CALL the newly-learned b (idx 1), got {t2_expr:?}"
            );
            let t2_solved = synthesize_target_reusing(&t2_problem, &[gname, bname])
                .expect("GROWTH: T2 must produce a strict-verified SolveResult");
            assert!(t2_solved.success, "GROWTH: T2's SolveResult must be success");
            assert!(
                crate::agent::repo::body_calls_fn(&t2_solved.code, bname),
                "GROWTH: emitted T2 must genuinely CALL {bname}: {}",
                t2_solved.code
            );
            // By-hand on the cumulative program (g + b + T2) on independent inputs.
            let b_comp = store::load().into_iter().find(|c| c.name == bname)
                .expect("GROWTH: b must persist");
            let full2 = format!(
                "{}\n\n{}\n\n{}",
                g_comp.code.trim_end(),
                b_comp.code.trim_end(),
                t2_solved.code
            );
            for x in -40..=40 {
                assert_eq!(
                    run_i64(&full2, "erl_t2", x), Some(t2(x)),
                    "BY-HAND: T2(x) must equal 12x+10 on independent input x={x}"
                );
            }
        });
    }

    /// FAST, NON-IGNORED un-gameable proof of the REUSE + EMERGENT + structural-Call
    /// core, WITHOUT the slow real-solver ingest. Instead of teaching g through the
    /// full curriculum solver (minutes), we persist g DIRECTLY into the same durable
    /// store the real spine uses (the store round-trip + reload path is identical),
    /// then prove the Call-node SEARCH emergently discovers T = g(g(x)) reusing it.
    ///
    /// This isolates and proves the parts the slow test also proves, but in seconds:
    ///   * BASELINE FAIL on empty callees (T(x)=9x+4 unreachable in budget),
    ///   * the store round-trips g and `learned_callable` reconstructs a working eval,
    ///   * the search EMERGENTLY emits `Call(g, Call(g, x))` (structural assert),
    ///   * by-hand correctness on >100 independent inputs,
    ///   * MONOTONIC GROWTH: T stays Err with NO callee even after g is in the store
    ///     (proves the Call only fires when g is REGISTERED as a callable — no
    ///     hidden mapping), and becomes Ok once g is passed as a callee.
    ///
    /// The slow `..._un_gameable_proof` test above covers the REAL teach_by_examples
    /// ingest (synthesize→gate→persist from examples); this one covers the reuse +
    /// emergence half deterministically and cheaply for the default CI gate.
    #[test]
    fn emergent_reuse_via_call_search_fast() {
        with_temp_store(|store_path| {
            let gname = "erl_fast_g";
            let budget_ms: u64 = 4_000;

            // Persist g's REAL Mog source into the durable store (same round-trip the
            // teach spine uses — `store::save_one` then `store::load`). g's body is a
            // genuine Mog program, NOT a hand-written target; the TARGET T is still
            // discovered by search below.
            let g_src = format!("fn {gname}(a: i64) -> i64 {{\n    return a * 3 + 1;\n}}\n");
            store::save_one(&store::StoredComponent {
                name: gname.to_string(),
                signature: format!("fn {gname}(a: i64) -> i64"),
                code: g_src.clone(),
                method: "test-fixture-direct-persist".to_string(),
                examples_fingerprint: String::new(),
                members: Vec::new(),
            });
            assert!(store_path.exists(), "store must persist after save_one");
            // Reload round-trip → reconstruct a working callable from the PERSISTED row.
            let g_call = learned_callable(gname)
                .expect("learned_callable must reconstruct a callable from the persisted g");
            assert_eq!(g_call.n_args, 1);
            for x in [-7i64, 0, 4, 11] {
                assert_eq!(
                    (g_call.eval)(&[x]), Some(g(x)),
                    "reconstructed g eval must compute 3x+1 on x={x}"
                );
            }

            // T(x) = g(g(x)) = 9x+4.
            let t_problem = scalar_problem("erl_fast_t", &pairs(t, &[0, 1, 2, 3, 4, 5, 6, 7]));

            // BASELINE FAIL: with NO callee, T is unreachable in budget.
            assert!(
                crate::enumerative::solve_scalar_expr_with_callees(&t_problem, &[], budget_ms).is_none(),
                "BASELINE: T(x)=9x+4 must not synthesize with an empty callee set in {budget_ms}ms"
            );

            // EMERGENT REUSE: register g as a callable, search discovers the Call.
            let t_expr = solve_target_expr_reusing(&t_problem, &[gname], budget_ms)
                .expect("T must synthesize once g is a registered callable");
            assert!(
                calls_idx(&t_expr, 0),
                "T's solution must CALL the learned g (registry idx 0), got {t_expr:?}"
            );

            // Strict-verified end-to-end + anti-inline.
            let t_solved = synthesize_target_reusing(&t_problem, &[gname])
                .expect("T must produce a strict-verified SolveResult reusing g");
            assert!(t_solved.success);
            assert!(
                crate::agent::repo::body_calls_fn(&t_solved.code, gname),
                "emitted T must genuinely CALL {gname}: {}",
                t_solved.code
            );

            // BY-HAND on >100 independent inputs.
            let full = format!("{}\n\n{}", g_src.trim_end(), t_solved.code);
            for x in -60..=60 {
                assert_eq!(
                    run_i64(&full, "erl_fast_t", x), Some(t(x)),
                    "BY-HAND: T(x) must equal 9x+4 on independent input x={x}"
                );
            }
        });
    }
}
