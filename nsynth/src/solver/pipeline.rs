use super::*;

/// Returns true when at least one example input is a string, structured type,
/// or pair. Scalar-only stages (enumerative expression search, scalar
/// expr_only gradient) cannot express those input shapes and would each burn
/// their full time budget before missing — the pipeline should skip straight
/// to the search-teacher / array path in that case.
fn has_non_scalar_input(problem: &Problem) -> bool {
    problem.examples.iter().any(|ex| {
        ex.inputs
            .iter()
            .any(|v| !matches!(v, Value::Int(_) | Value::Array(_)))
    })
}

pub(super) fn solve_problem(problem: &Problem) -> SolveResult {
    let result = solve_problem_inner(problem);
    if result.success {
        // Record every successful solve. De-duped inside the cache so reruns
        // don't re-write the same entry. Persisted via `solved_cache::flush`
        // which callers (bench runner, main) invoke at shutdown.
        crate::solved_cache::record(problem, &result.method, &result.code);
    }
    result
}

fn solve_problem_inner(problem: &Problem) -> SolveResult {
    let t0 = std::time::Instant::now();
    let router_ctx = post_enumerative_context(problem);

    // Stage 0: persistent cross-run memoization. A previous solve with the
    // same I/O fingerprint (+ a live re-verification to guard against
    // fingerprint collisions / stale cache) returns instantly. This is the
    // first place cross-run "learning" shows up — every successful solve
    // below is recorded and will hit on the next run.
    let mut cached_fallback: Option<crate::solved_cache::CachedSolution> = None;
    if let Some(cached) = crate::solved_cache::lookup(problem) {
        if should_bypass_solved_cache(problem, &router_ctx, &cached) {
            eprintln!(
                "[solve] bypassing solved_cache HIT in {:.3}s — {}",
                t0.elapsed().as_secs_f32(),
                cached.method
            );
            cached_fallback = Some(cached);
        } else {
            eprintln!(
                "[solve] solved_cache HIT in {:.3}s — {}",
                t0.elapsed().as_secs_f32(),
                cached.method
            );
            return SolveResult {
                success: true,
                code: cached.code,
                method: cached.method,
                error: None,
                metadata: DifferentiableMetadata::default(),
            };
        }
    }

    let non_scalar = has_non_scalar_input(problem);

    // Stage 0.5: cross-run knowledge transfer via CachedTeachers. After the
    // Stage-0 exact-match lookup misses, consult the persistent cache as a
    // *source of teachers*: every prior solve is a candidate program whose
    // behaviour can seed gradient distillation on the current problem.
    // Bounded by NSYNTH_TEACHER_BUDGET_SEC (default 15s) so this stage can't
    // stall trivial problems on a large cache. Scalar-only today; gated
    // internally by the strategy's `applicable()` check.
    //
    // This is the first place normal bench runs exercise the cross-run
    // learning loop — without it, `CachedTeachers` only runs via the
    // emergent_coverage binary.
    if !non_scalar && crate::solved_cache::entry_count() > 0 {
        let strategy = crate::strategy::CachedTeachers;
        if <crate::strategy::CachedTeachers as crate::strategy::SynthesisStrategy>::applicable(
            &strategy, problem,
        ) {
            let t_teacher = std::time::Instant::now();
            if let Some(result) =
                <crate::strategy::CachedTeachers as crate::strategy::SynthesisStrategy>::try_solve(
                    &strategy, problem,
                )
            {
                if result.success {
                    eprintln!(
                        "[solve] cached_teachers OK in {:.1}s",
                        t_teacher.elapsed().as_secs_f32()
                    );
                    return result;
                }
            }
            eprintln!(
                "[solve] cached_teachers MISS in {:.1}s",
                t_teacher.elapsed().as_secs_f32()
            );
        }
    }

    let preemptive_search_result = solve_problem_from_preemptive_search_teacher(problem);

    // Stage 1: Classical bottom-up symbolic enumeration (no neural weights, no
    // gradients). Cheap on small closed-form expressions and cuts them off
    // before the slower gradient path even starts. The actual "brain" —
    // neural-style continuous parameter search — lives in the differentiable
    // synth_gradient / univ_arr_gradient stages further down.
    //
    // Skipped for inputs this enumerator cannot express (Str, Pair, struct):
    // those are handled by downstream search teachers and would otherwise
    // burn the full enumerative budget only to miss.
    if should_try_enumerative(
        problem,
        &router_ctx,
        non_scalar,
        preemptive_search_result.is_some(),
    ) {
        let t_enum = std::time::Instant::now();
        if let Some(result) = crate::enumerative::synthesize_enumerative(problem) {
            if result.success {
                eprintln!(
                    "[solve] enumerative OK in {:.1}s",
                    t_enum.elapsed().as_secs_f32()
                );
                method_router::record_win(problem, ROUTE_ENUMERATIVE);
                return result;
            }
        }
        eprintln!(
            "[solve] enumerative MISS in {:.1}s",
            t_enum.elapsed().as_secs_f32()
        );
        method_router::record_miss(problem, ROUTE_ENUMERATIVE);
    } else if non_scalar {
        eprintln!("[solve] skipping enumerative: non-scalar input (string/pair/struct)");
    } else if preemptive_search_result.is_some() {
        eprintln!("[solve] skipping enumerative: exact search preemption");
    } else {
        let ranked = normalized_router_stats(problem, &router_ctx);
        if let Some(top) = ranked.first().copied() {
            eprintln!(
                "[solve] skipping enumerative: method_router favors {} ({} wins, {}% success)",
                top.route,
                top.wins,
                top.success_rate_percent()
            );
        } else {
            eprintln!("[solve] skipping enumerative: method_router preemption");
        }
    }

    let result = solve_problem_after_enumeration(problem, t0, preemptive_search_result);
    if result.success {
        return result;
    }
    if let Some(cached) = cached_fallback {
        eprintln!("[solve] restoring solved_cache fallback after routed miss");
        return SolveResult {
            success: true,
            code: cached.code,
            method: cached.method,
            error: None,
            metadata: DifferentiableMetadata::default(),
        };
    }
    result
}
