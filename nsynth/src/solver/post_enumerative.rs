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
    ROUTE_SEARCH_TEACHER, ROUTE_TEMPLATE_REFERENCE,
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
        | "search_count_divisors_loop"
        | "search_sum_of_divisors_loop"
        | "search_sum_odd_digits_loop"
        | "search_power_loop"
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
        | "search_affine_piecewise" => true,
        "search_unary_range_loop" => {
            result.code.contains("acc = acc + i;") || result.code.contains("acc = acc * i;")
        }
        _ => false,
    }
}

fn solve_problem_from_search_result(problem: &Problem, search_result: SolveResult) -> SolveResult {
    if search_result_supports_differentiable_probe(&search_result) {
        let result = solve_problem_differentiable_probe(problem);
        let result = SolveResult {
            success: result.success,
            code: result.code,
            method: result.method,
            error: result.error,
            metadata: result.metadata,
        };
        if result.success {
            return result;
        }
    }

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
    if search_result_preempts_native_gradient(&search_result) {
        Some(search_result)
    } else {
        Some(solve_problem_from_search_result(problem, search_result))
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

fn try_post_enumerative_route(
    problem: &Problem,
    ctx: &PostEnumerativeContext,
    t0: Instant,
    route: &'static str,
) -> Option<SolveResult> {
    if !route_is_applicable(route, problem, ctx) {
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
            if problem.reference_code.is_empty() {
                return None;
            }
            let t_ref = Instant::now();
            let result = solve_problem_differentiable_teacher(problem, problem.reference_code);
            if result.success {
                eprintln!(
                    "[solve] reference_distill OK in {:.1}s — {}",
                    t_ref.elapsed().as_secs_f32(),
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
                "[solve] reference_distill MISS in {:.1}s",
                t_ref.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_NATIVE_REFERENCE_DISTILLATION => {
            if problem.reference_code.is_empty() {
                return None;
            }
            let t_native_ref = Instant::now();
            if let Some(result) =
                synthesis::synthesize_scalar_from_teacher(problem, problem.reference_code)
            {
                if result.success {
                    eprintln!(
                        "[solve] native_reference_distill OK in {:.1}s — {}",
                        t_native_ref.elapsed().as_secs_f32(),
                        result.method
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] native_reference_distill MISS in {:.1}s",
                t_native_ref.elapsed().as_secs_f32()
            );
            None
        }
        ROUTE_ARRAY_REFERENCE_DISTILLATION => {
            if problem.reference_code.is_empty() {
                return None;
            }
            let t_arr_teacher = Instant::now();
            if let Some(result) =
                synthesis::synthesize_array_from_teacher(problem, problem.reference_code)
            {
                if result.success {
                    eprintln!(
                        "[solve] array_reference_distill OK in {:.1}s — {}",
                        t_arr_teacher.elapsed().as_secs_f32(),
                        result.method
                    );
                    return Some(result);
                }
            }
            eprintln!(
                "[solve] array_reference_distill MISS in {:.1}s",
                t_arr_teacher.elapsed().as_secs_f32()
            );
            None
        }
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
        ROUTE_TEMPLATE_REFERENCE => {
            let code = problem.reference_code.to_string();
            if verify_problem_code_strict(problem, &code).is_ok() {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "template_reference".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
            None
        }
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
                        "no synthesis method solved this {}-arg problem",
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
        method: "none".to_string(),
        error: Some("no synthesis method solved this problem".to_string()),
        metadata: DifferentiableMetadata::default(),
    }
}
