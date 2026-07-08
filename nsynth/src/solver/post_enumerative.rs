use std::collections::HashSet;
use std::time::Instant;

use crate::benchmark::Problem;
use crate::differentiable::{
    solve_problem_differentiable_fast_probe as solve_problem_differentiable_probe,
    solve_problem_differentiable_from_teacher as solve_problem_differentiable_teacher,
    solve_problem_differentiable_only as solve_problem_differentiable_bridge,
    DifferentiableMetadata,
};
use crate::method_router;
use crate::runtime::verify_problem_code_strict;
use crate::synthesis;

use super::routing::{
    default_post_enumerative_routes, post_enumerative_context, recommended_post_enumerative_routes,
    route_is_applicable, PostEnumerativeContext, ROUTE_ARRAY_GRADIENT,
    ROUTE_ARRAY_REFERENCE_DISTILLATION, ROUTE_BRIDGE_GRADIENT, ROUTE_EXPR_ONLY,
    ROUTE_EXPR_TEMPLATES, ROUTE_NATIVE_REFERENCE_DISTILLATION, ROUTE_REFERENCE_DISTILLATION,
    ROUTE_REGISTER_MACHINE, ROUTE_SCALAR_GRADIENT, ROUTE_SCALAR_TEMPLATES, ROUTE_SEARCH,
    ROUTE_SEARCH_TEACHER, ROUTE_TEMPLATE_REFERENCE, ROUTE_UNIVERSAL,
};
use super::{solve_by_search, SolveResult};

fn search_result_supports_differentiable_probe(result: &SolveResult) -> bool {
    match result.method.as_str() {
        "search_scalar_expr"
        | "search_abs_diff_formula"
        | "search_clamp_formula"
        | "search_sign_branch"
        | "search_is_even_formula"
        | "search_digit_sum_loop"
        | "search_reverse_digits_loop"
        | "search_digit_count_loop"
        | "search_count_even_digits_loop" => true,
        "search_unary_range_loop" => {
            result.code.contains("acc = acc + i;") || result.code.contains("acc = acc * i;")
        }
        _ => false,
    }
}

pub(super) fn search_result_preempts_native_gradient(result: &SolveResult) -> bool {
    match result.method.as_str() {
        "search_lcm_formula"
        | "search_gcd_loop"
        | "search_abs_diff_formula"
        | "search_max2_formula"
        | "search_safe_div_or_neg1_branch"
        | "search_positive_or_default_branch"
        | "search_clamp_formula"
        | "search_sign_branch"
        | "search_is_even_formula"
        | "search_euler_totient_loop"
        | "search_next_power_of_2"
        | "search_triangular_check_loop"
        | "search_collatz_loop"
        | "search_lucas_loop"
        | "search_celsius_to_fahrenheit"
        | "search_is_perfect_square"
        | "search_leading_digit"
        | "search_popcount"
        | "search_polynomial_quadratic"
        | "search_polynomial_multi"
        | "search_clamp_affine"
        | "search_composed_features"
        // Exact bitwise-structured rule (mask/set/toggle/pairwise-combine, plus
        // optional affine): recovered by an over-determined integer solve and
        // verified on every example, so it generalizes by construction and
        // preempts the slow native-gradient distillation.
        | "search_bitwise"
        // Exact array-feature composition: an affine mix of array reductions
        // (and scalar args), recovered by an over-determined integer solve and
        // fully verified, so it generalizes by construction and preempts the slow
        // native-gradient distillation rather than feeding it.
        | "array_affine_features"
        // Conditional logic (`if x%m==r { affine } else { affine }`): both
        // branches are exact integer affines on over-determined partitions and
        // the whole if/else is verified on every example, so it generalizes by
        // construction and is returned directly instead of feeding gradient.
        | "search_predicate_branch"
        // Single-argument closed-interval branch (`if lo <= x && x <= hi`):
        // recovered by deterministic 3-run segmentation, both bodies exact, the
        // whole program verified — generalizes by construction, preempts gradient.
        | "search_interval_branch"
        // Rational floor-division (`(a*x + b) / d`, affine inside the divide):
        // recovered by a bounded divisor/slope search, b forced by the floor
        // inequalities, gated to a,b,x >= 0 so trunc == floor, and verified — so
        // it generalizes by construction and preempts gradient.
        | "search_rational_floor"
        // Full modular case analysis (`match x%m { r => affine }`): every residue
        // class is an exact affine on an over-determined bucket and the whole
        // chain is verified, so it generalizes by construction and is returned
        // directly instead of feeding gradient.
        | "search_modular_cases"
        // Value-based envelope (`max(A,B)` / `min(A,B)` of two affines): both
        // pieces are exact affines and the reconstructed envelope is verified on
        // every example, so it generalizes by construction and is returned
        // directly instead of feeding gradient.
        | "search_minmax_affine"
        | "search_min3_branch"
        | "search_scalar_expr"
        | "search_digit_sum_loop"
        | "search_reverse_digits_loop"
        | "search_digit_count_loop"
        | "search_count_even_digits_loop"
        | "search_digit_product_loop"
        | "search_max_digit_loop"
        | "search_digital_root"
        | "search_fib_iter_loop"
        // Newer DP fibonacci teacher: a fully-verified exact iterative solution
        // (same preemption rationale as search_fib_iter_loop — it generalizes by
        // construction and preempts the slow native-gradient distillation). It
        // now ranks ahead of the loop variant for fibonacci_v0.
        | "search_fibonacci_dp"
        | "search_count_divisors_loop"
        | "search_sum_of_divisors_loop"
        | "search_sum_odd_digits_loop"
        | "search_digits_filter_map_reduce"
        | "search_power_loop"
        | "search_modpow_loop"
        | "search_is_prime_loop"
        | "search_harmonic_sum_loop"
        | "search_run_length_decode_sum"
        | "search_count_adjacent_diff"
        | "search_prefix_sum_k"
        | "search_second_max"
        | "search_array_range"
        | "search_arr_sum_squares"
        | "search_min_element"
        | "search_sum_absolute"
        | "search_count_evens"
        | "search_sum_positives"
        | "search_max_consecutive_sum"
        | "search_min_consecutive_sum"
        | "search_max_stock_profit"
        | "search_is_sorted"
        | "search_longest_increasing_run"
        | "search_longest_plateau"
        | "search_prefix_max_sum"
        | "search_sum_at_even_indices"
        | "search_kth_from_end"
        | "search_max_abs"
        | "search_min_positive"
        | "search_count_peaks"
        | "search_alternating_sum"
        | "search_is_palindrome_arr"
        | "search_sum_odd_indexed"
        | "search_max_pair_diff"
        | "search_combat_resolve_branch"
        | "search_grid_bounds_check_branch"
        | "search_score_tracker_branch"
        | "search_simulate_gravity_clamp"
        | "search_turn_order_rotate"
        | "search_vending_change_branch"
        // Generic scalar branch search: a verified single/two-branch program
        // is already an exact solution, so it preempts the (slow) gradient
        // distillation rather than being fed to it as a teacher. This is what
        // makes novel piecewise rules (thresholds + affine pieces) solve in
        // milliseconds instead of timing out.
        | "search_single_branch"
        | "search_two_branch"
        // Exact piecewise-affine recovery (any number of tiers): the program is
        // verified on the examples and generalizes by construction (breakpoints
        // read from the data, placed at piece intersections), so likewise it is
        // returned directly instead of being distilled by gradient.
        | "search_piecewise_affine"
        // Exact multi-argument linear solves (global affine, single-threshold
        // affine, and multi-tier piecewise affine): verified against every
        // example, so returned directly.
        | "search_affine"
        | "search_affine_threshold"
        | "search_affine_piecewise"
        // Disjunctive class learners (string suffix / array membership): the
        // emitted program is verified on examples + holdouts and generalizes by
        // construction (admissible features never fire on a negative), so it is
        // returned directly instead of feeding the slow native-gradient path.
        // This is what makes morpheme-tokenized grammaticality solve in
        // milliseconds instead of timing out in the array-gradient distillation.
        | "search_suffix_class"
        | "search_array_member_class"
        | "search_array_conjunction"
        | "search_array_dnf"
        | "search_array_sequence"
        | "search_array_feature_dnf"
        | "search_string_subsequence_class"
        | "search_stateful_reducer"
        | "search_stateful_reducer_dual"
        | "search_stateful_reducer_event"
        | "search_stateful_reducer_temporal"
        | "search_stateful_replace"
        | "search_strictly_increasing"
        | "search_has_strictly_increasing_run"
        | "search_first_index_of"
        | "search_last_index_of"
        | "search_is_anagram"
        | "search_longest_run"
        | "search_intersects"
        | "search_count_distinct"
        | "search_kth_smallest"
        // Stage 2: broadcast/dot-product/matmul templates
        // These are structured numeric computation patterns that are recovered via
        // over-determined linear algebra and fully verified on every example, so
        // they generalize by construction and preempt gradient distillation.
        | "search_broadcast_pattern"
        | "search_dot_product_search"
        | "search_matmul_template"
        // Stage 3: struct-based field manipulation and conditional logic
        // Field reduction, coupled field transformations, and conditional struct
        // assembly are all verified exact solutions, so they generalize by
        // construction and preempt the slow gradient path.
        | "search_struct_field_reduction"
        | "search_struct_coupled_fields"
        | "search_struct_conditional_fields"
        // U5c: SEARCHED linear recursion `f(n)=(n<=k)?base:combine(n,f(n-1))`.
        // The base threshold, base value, and combine op are enumerated over a
        // small grammar and the resulting REAL recursive program is verified on
        // every example AND on fresh reference-derived holdouts (strict verify),
        // so it generalizes by construction and is returned directly instead of
        // being distilled by the slow native-gradient path.
        | "search_linear_recursion" => true,
        "search_unary_range_loop" => {
            result.code.contains("acc = acc + i;") || result.code.contains("acc = acc * i;")
        }
        _ => false,
    }
}

fn solve_problem_from_search_result_after_probe(
    problem: &Problem,
    search_result: SolveResult,
) -> SolveResult {
    if let Some(result) = synthesis::synthesize_scalar_from_teacher(problem, &search_result.code) {
        if result.success {
            return result;
        }
    }

    if let Some(result) = synthesis::synthesize_array_from_teacher(problem, &search_result.code) {
        if result.success {
            return result;
        }
    }

    let teacher_result = solve_problem_differentiable_teacher(problem, &search_result.code);
    if teacher_result.success {
        return SolveResult {
            success: teacher_result.success,
            code: teacher_result.code,
            method: teacher_result.method,
            error: teacher_result.error,
            metadata: teacher_result.metadata,
        };
    }

    search_result
}

fn solve_problem_from_search_teacher(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let search_result = solve_by_search(problem, fn_name)?;
    if search_result_supports_differentiable_probe(&search_result) {
        let probe = solve_problem_differentiable_probe(problem);
        let probe = SolveResult {
            success: probe.success,
            code: probe.code,
            method: probe.method,
            error: probe.error,
            metadata: probe.metadata,
        };
        if probe.success {
            return Some(probe);
        }
    }
    if search_result_preempts_native_gradient(&search_result) {
        Some(search_result)
    } else {
        Some(solve_problem_from_search_result_after_probe(
            problem,
            search_result,
        ))
    }
}

pub(super) fn solve_problem_from_preemptive_search_teacher(
    problem: &Problem,
) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let search_result = solve_by_search(problem, fn_name)?;
    if search_result_preempts_native_gradient(&search_result) {
        Some(search_result)
    } else {
        None
    }
}

pub(super) fn solve_problem_prefer_differentiable(problem: &Problem) -> SolveResult {
    if let Some(result) = solve_problem_from_search_teacher(problem) {
        return result;
    }

    let result = solve_problem_differentiable_probe(problem);
    SolveResult {
        success: result.success,
        code: result.code,
        method: result.method,
        error: result.error,
        metadata: result.metadata,
    }
}

/// True when a global `NSYNTH_SOLVE_BUDGET_MS` is set and the elapsed solve time
/// (from `t0`) has already exceeded it. Unset ⇒ always false (no behavior change).
/// Mirrors `pipeline::solve_budget_ms` but is read here to gate the unbounded
/// gradient/register-machine routes without a cross-module coupling.
fn global_solve_budget_exhausted(t0: Instant) -> bool {
    std::env::var("NSYNTH_SOLVE_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse::<u128>().ok())
        .is_some_and(|budget| t0.elapsed().as_millis() > budget)
}

fn try_post_enumerative_route(
    problem: &Problem,
    ctx: &PostEnumerativeContext,
    t0: Instant,
    route: &'static str,
) -> Option<SolveResult> {
    if !route_is_applicable(route, problem, ctx) {
        return None;
    }

    // Analogy universal re-fit guards. A re-fit re-solves a teacher-augmented
    // (possibly contradictory) problem, so two bounds apply — both are no-ops
    // outside a re-fit, leaving top-level solves untouched:
    //   1. Once the re-fit's wall-clock budget is spent, stop trying routes.
    //      A re-fit overshoots by at most the one route already in flight.
    //   2. Skip the gradient stages outright: they have no global time budget
    //      and spin for minutes on data that never converges. The cheap routes
    //      (enumerative already ran; the budgeted teachers/templates below)
    //      carry any genuine transfer.
    if super::analogy::refit_budget_exhausted() {
        return None;
    }
    if super::analogy::in_refit() && matches!(route, ROUTE_SCALAR_GRADIENT | ROUTE_ARRAY_GRADIENT) {
        return None;
    }
    // Global solve-budget guard. The gradient + register-machine routes have NO
    // internal wall-clock bound — their training runs on worker threads that do not
    // inherit the thread-local train deadline, so one route can spin ~60s on data
    // that never converges (measured: synthesize_gradient_only 61s, register_machine
    // 24s on nth_composite), blowing far past a set budget. When a global
    // NSYNTH_SOLVE_BUDGET_MS is set and ALREADY exhausted (an earlier route overran),
    // skip these expensive routes so the solve degrades gracefully instead of
    // cascading minutes of doomed search. Opt-in: with no budget set this is a no-op,
    // so default behaviour (and the benchmark) is unchanged; it never skips a route
    // reached while still within budget, so a legitimate gradient solve is preserved.
    if matches!(
        route,
        ROUTE_SCALAR_GRADIENT
            | ROUTE_ARRAY_GRADIENT
            | ROUTE_REGISTER_MACHINE
            | ROUTE_UNIVERSAL
            | ROUTE_BRIDGE_GRADIENT
    ) && global_solve_budget_exhausted(t0)
    {
        return None;
    }

    match route {
        ROUTE_SCALAR_GRADIENT => {
            let t_grad = Instant::now();
            if let Some(result) = synthesis::synthesize_gradient_only(problem) {
                if result.success {
                    eprintln!(
                        "[solve] synthesize_gradient_only OK in {:.1}s — {}",
                        t_grad.elapsed().as_secs_f32(),
                        result.method
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] synthesize_gradient_only MISS in {:.1}s",
                t_grad.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_ARRAY_GRADIENT => {
            let t_arr = Instant::now();
            if let Some(result) = synthesis::synthesize_array(problem) {
                if result.success {
                    eprintln!(
                        "[solve] synthesize_array OK in {:.1}s",
                        t_arr.elapsed().as_secs_f32()
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] synthesize_array MISS in {:.1}s",
                t_arr.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_EXPR_ONLY => {
            let t_expr = Instant::now();
            if let Some(result) = synthesis::synthesize_scalar_expr_only(problem) {
                if result.success {
                    eprintln!(
                        "[solve] expr_only OK in {:.1}s — {}",
                        t0.elapsed().as_secs_f32(),
                        result.method
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] expr_only MISS in {:.1}s",
                t_expr.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_SEARCH_TEACHER => {
            let t_search = Instant::now();
            if let Some(result) = solve_problem_from_search_teacher(problem) {
                eprintln!(
                    "[solve] search_teacher OK in {:.1}s — {}",
                    t_search.elapsed().as_secs_f32(),
                    result.method
                );
                return Some(result);
            }
            eprintln!(
                "[solve] search_teacher MISS in {:.1}s",
                t_search.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_REGISTER_MACHINE => {
            let t_rm = Instant::now();
            if let Some(result) = synthesis::synthesize_register_machine(problem) {
                if result.success {
                    eprintln!(
                        "[solve] register_machine OK in {:.1}s",
                        t_rm.elapsed().as_secs_f32()
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] register_machine MISS in {:.1}s",
                t_rm.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_BRIDGE_GRADIENT => {
            let t_bridge = Instant::now();
            let result = solve_problem_differentiable_bridge(problem);
            if result.success {
                eprintln!(
                    "[solve] differentiable_bridge OK in {:.1}s — {}",
                    t_bridge.elapsed().as_secs_f32(),
                    result.method
                );
                return Some(SolveResult {
                    success: result.success,
                    code: result.code,
                    method: result.method,
                    error: result.error,
                    metadata: result.metadata,
                });
            }
            eprintln!(
                "[solve] differentiable_bridge MISS in {:.1}s — {}",
                t_bridge.elapsed().as_secs_f32(),
                result.method
            );
            None
        }
        ROUTE_REFERENCE_DISTILLATION => {
            // Historical telemetry may still recommend this route, but benchmark
            // reference implementations are never visible to production solving.
            None
        }
        ROUTE_NATIVE_REFERENCE_DISTILLATION => None,
        ROUTE_ARRAY_REFERENCE_DISTILLATION => None,
        ROUTE_EXPR_TEMPLATES => {
            let t_expr_tpl = Instant::now();
            if let Some(result) = synthesis::synthesize_scalar_expr_templates_only(problem) {
                if result.success {
                    eprintln!(
                        "[solve] expr_templates OK in {:.1}s — {}",
                        t_expr_tpl.elapsed().as_secs_f32(),
                        result.method
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] expr_templates MISS in {:.1}s",
                t_expr_tpl.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_SCALAR_TEMPLATES => {
            let t_templates = Instant::now();
            if let Some(result) = synthesis::synthesize_scalar_templates_only(problem) {
                if result.success {
                    eprintln!(
                        "[solve] synthesize_scalar_templates_only OK in {:.1}s — {}",
                        t_templates.elapsed().as_secs_f32(),
                        result.method
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] synthesize_scalar_templates_only MISS in {:.1}s",
                t_templates.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_TEMPLATE_REFERENCE => None,
        ROUTE_SEARCH => {
            if ctx.is_external && ctx.n_args > 3 {
                eprintln!(
                    "[solve] skipping search fallback for external {}-arg problem",
                    ctx.n_args
                );
                return Some(SolveResult {
                    success: false,
                    code: String::new(),
                    method: "none".to_string(),
                    error: Some(format!(
                        "search budget exhausted for this {}-arg problem; \
                         enumerative frontier persisted and resumable (not a \
                         proof of impossibility)",
                        ctx.n_args
                    )),
                    metadata: DifferentiableMetadata::default(),
                });
            }
            eprintln!("[solve] falling back to search...");
            let t_search = Instant::now();
            let result = solve_problem_prefer_differentiable(problem);
            eprintln!(
                "[solve] search done in {:.1}s — success={}",
                t_search.elapsed().as_secs_f32(),
                result.success
            );
            Some(result)
        }
        ROUTE_UNIVERSAL => {
            let t_u = Instant::now();
            // SoftUniversalProgram: 5 gradient restarts at a bounded step budget;
            // strict-verified internally. Discovers general learned loops the fixed
            // diff-zoo architectures miss. Late fallback → bounded latency impact.
            if let Some((result, _)) = synthesis::synthesize_universal_and_collect(problem, 2_000) {
                if result.success {
                    eprintln!(
                        "[solve] synthesize_universal OK in {:.1}s — {}",
                        t_u.elapsed().as_secs_f32(),
                        result.method
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] synthesize_universal MISS in {:.1}s",
                t_u.elapsed().as_secs_f32()
            );
            None
        }
        _ => None,
    }
}

pub(super) fn solve_problem_after_enumeration(
    problem: &Problem,
    t0: Instant,
    preemptive_search_result: Option<SolveResult>,
) -> SolveResult {
    let ctx = post_enumerative_context(problem);

    // Exact search-backed teachers can skip the native gradient stack for a
    // small set of known hard misses, keeping full-pipeline coverage bounded.
    let t_preempt_search = Instant::now();
    if let Some(result) = preemptive_search_result {
        eprintln!(
            "[solve] preemptive_search_teacher OK in {:.1}s — {}",
            t_preempt_search.elapsed().as_secs_f32(),
            result.method
        );
        method_router::record_win(problem, ROUTE_SEARCH_TEACHER);
        return result;
    }

    let recommended_routes = recommended_post_enumerative_routes(problem, &ctx);
    let mut attempted_routes: HashSet<&'static str> = HashSet::new();

    if !recommended_routes.is_empty() {
        eprintln!(
            "[solve] method_router rec: {}",
            recommended_routes.join(" -> ")
        );
    }

    for route in recommended_routes {
        attempted_routes.insert(route);
        if let Some(result) = try_post_enumerative_route(problem, &ctx, t0, route) {
            if result.success {
                method_router::record_win(problem, route);
            } else {
                method_router::record_miss(problem, route);
            }
            return result;
        }
        method_router::record_miss(problem, route);
    }

    for route in default_post_enumerative_routes(problem, &ctx) {
        if !attempted_routes.insert(route) {
            continue;
        }
        if let Some(result) = try_post_enumerative_route(problem, &ctx, t0, route) {
            if result.success {
                method_router::record_win(problem, route);
            } else {
                method_router::record_miss(problem, route);
            }
            return result;
        }
        method_router::record_miss(problem, route);
    }

    SolveResult {
        success: false,
        code: String::new(),
        // The portfolio did not solve this problem WITHIN THIS CALL's budget.
        // The enumerative search frontier is persisted (keyed by the examples
        // fingerprint), so a re-invocation resumes deeper rather than restarting
        // from size 1 — this is NOT a proof that no program exists.
        method: "none".to_string(),
        error: Some(
            "search budget exhausted; enumerative frontier persisted and \
             resumable (not a proof of impossibility)"
                .to_string(),
        ),
        metadata: DifferentiableMetadata::default(),
    }
}
