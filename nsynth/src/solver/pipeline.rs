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

/// Route string-output problems (signature `-> string`) to the string-program
/// path: the fast morphology specialist, then the general enumerative string
/// synthesizer. Returns None for non-string problems so the numeric pipeline runs.
fn solve_string_output(problem: &Problem) -> Option<SolveResult> {
    if !problem.signature.replace(' ', "").contains("->string") {
        return None;
    }
    // Examples must be all-string-input, string-output.
    let to_rows = |exs: &[crate::benchmark::Example]| -> Option<Vec<(Vec<String>, String)>> {
        exs.iter()
            .map(|e| {
                let ins: Option<Vec<String>> = e
                    .inputs
                    .iter()
                    .map(|v| match v {
                        Value::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                match (&ins, &e.expected) {
                    (Some(i), Value::Str(o)) => Some((i.clone(), o.clone())),
                    _ => None,
                }
            })
            .collect()
    };
    let train = to_rows(&problem.examples)?;
    let holds = to_rows(&problem.holdouts).unwrap_or_default();
    let fn_name = problem.function_name();
    let single = train.iter().all(|(i, _)| i.len() == 1);

    // Fast morphology specialist (single-arg suffix transduction).
    if single {
        use crate::morph_transduce::{solve_morph_transduction, StrExample};
        let mk = |rs: &[(Vec<String>, String)]| {
            rs.iter()
                .map(|(i, o)| StrExample { input: i[0].clone(), expected: o.clone() })
                .collect::<Vec<_>>()
        };
        let m = solve_morph_transduction(fn_name, &mk(&train), &mk(&holds));
        if m.success {
            return Some(SolveResult {
                success: true,
                code: m.code,
                method: m.method,
                error: None,
                metadata: Default::default(),
            });
        }
    }

    // General enumerative string synthesizer.
    use crate::string_synth::{synthesize_string_program, StrSynthExample};
    let params: Vec<String> = problem
        .signature
        .split_once('(')
        .and_then(|(_, r)| r.split_once(')'))
        .map(|(p, _)| p)
        .unwrap_or("")
        .split(',')
        .filter_map(|p| p.split(':').next().map(|n| n.trim().to_string()))
        .filter(|n| !n.is_empty())
        .collect();
    let params = if params.is_empty() {
        vec!["s".to_string()]
    } else {
        params
    };
    let all: Vec<StrSynthExample> = train
        .iter()
        .chain(holds.iter())
        .map(|(i, o)| StrSynthExample { inputs: i.clone(), expected: o.clone() })
        .collect();
    let r = synthesize_string_program(&params, &all);
    if r.success {
        let code = r.code.replacen("fn transform(", &format!("fn {fn_name}("), 1);
        return Some(SolveResult {
            success: true,
            code,
            method: r.method,
            error: None,
            metadata: Default::default(),
        });
    }
    None
}

pub(super) fn solve_problem(problem: &Problem) -> SolveResult {
    // String-output problems take the additive string-program path (the i64
    // gradient/search pipeline cannot express string outputs).
    if let Some(result) = solve_string_output(problem) {
        if result.success {
            crate::solved_cache::record(problem, &result.method, &result.code);
        }
        return result;
    }
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

    // Float (continuous) lane first: a `-> f64` problem is least-squares affine
    // regression, a different regime from the exact-integer machinery below
    // (which would choke on f64 inputs). Self-gates to f64 signatures and returns
    // None for everything else, so integer problems are untouched.
    if let Some(result) = super::search_float::search_float_affine(problem, &problem.function_name())
    {
        if result.success {
            return result;
        }
    }

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

    // Exact multi-argument linear family first: a 2-3 arg affine or
    // single-threshold-affine rule is solved by a direct integer linear solve in
    // microseconds and verified against every example, so it must short-circuit
    // ahead of the search/gradient stages rather than risk being starved by an
    // earlier candidate. No-op (instant None) for 1-arg or non-linear data.
    if let Some(result) = super::search::solve_multi_arg_affine(problem) {
        if result.success {
            eprintln!(
                "[solve] multi-arg affine OK in {:.3}s — {}",
                t0.elapsed().as_secs_f32(),
                result.method
            );
            return result;
        }
    }

    let non_scalar = has_non_scalar_input(problem);

    // Run the cheap preemptive search teacher first — it's ms-scale and
    // covers a large fraction of the benchmark. This is the correct pre-
    // gradient stage ordering; running CachedTeachers before this was a
    // measured regression (curve_analysis flagged 22-24s cumulative wall-
    // clock on problems the preemptive stage solved in 5-10ms).
    let preemptive_search_result = solve_problem_from_preemptive_search_teacher(problem);

    // Stage 1.5: cross-run knowledge transfer via CachedTeachers. Runs only
    // when cheaper stages (Stage 0 exact-match + preemptive search) have
    // already missed. Bounded by NSYNTH_TEACHER_BUDGET_SEC (default 15s).
    //
    // This is the "gradient distillation from prior solves" path — real work,
    // not a lookup. Placing it after preemptive search means fast problems
    // stay fast; placing it before enumerative + synth_gradient means a
    // transfer win short-circuits the slowest stages of the pipeline.
    if !non_scalar && preemptive_search_result.is_none() && crate::solved_cache::entry_count() > 0 {
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
