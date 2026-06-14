use crate::search_family_router;

use super::search_affine::*;
use super::search_array_compose::*;
use super::search_catalog::*;
use super::search_families::*;
use super::search_numeric_families::*;
use super::search_scalar_families::*;
use super::search_text_families::*;
use super::*;

type SearchFn = fn(&Problem, &str) -> Option<SolveResult>;

#[derive(Clone, Copy)]
struct SearchCandidate {
    key: &'static str,
    func: SearchFn,
}

const SEARCH_CANDIDATES: &[SearchCandidate] = &[
    SearchCandidate {
        key: "search_array_item_loop",
        func: search_array_item_loop,
    },
    SearchCandidate {
        key: "search_run_length_decode_sum",
        func: search_run_length_decode_sum,
    },
    SearchCandidate {
        key: "search_count_adjacent_diff",
        func: search_count_adjacent_diff,
    },
    SearchCandidate {
        key: "search_second_max",
        func: search_second_max,
    },
    SearchCandidate {
        key: "search_array_range",
        func: search_array_range,
    },
    // General membership/DNF array classifiers. Each is guarded to require >= 12
    // examples (every structural benchmark problem has <= 10), so they own the
    // large curriculum language-classification tasks without shadowing the exact
    // structural solvers (palindrome, sorted, ...) on small problems.
    SearchCandidate {
        key: "search_array_member_class",
        func: search_array_member_class,
    },
    SearchCandidate {
        key: "search_array_conjunction",
        func: search_array_conjunction,
    },
    SearchCandidate {
        key: "search_array_dnf",
        func: search_array_dnf,
    },
    SearchCandidate {
        key: "search_min_element",
        func: search_min_element,
    },
    SearchCandidate {
        key: "search_max_consecutive_sum",
        func: search_max_consecutive_sum,
    },
    SearchCandidate {
        key: "search_min_consecutive_sum",
        func: search_min_consecutive_sum,
    },
    SearchCandidate {
        key: "search_kth_smallest",
        func: search_kth_smallest,
    },
    SearchCandidate {
        key: "search_max_stock_profit",
        func: search_max_stock_profit,
    },
    SearchCandidate {
        key: "search_is_sorted",
        func: search_is_sorted,
    },
    SearchCandidate {
        key: "search_longest_increasing_run",
        func: search_longest_increasing_run,
    },
    SearchCandidate {
        key: "search_digital_root",
        func: search_digital_root,
    },
    SearchCandidate {
        key: "search_two_sum_exists",
        func: search_two_sum_exists,
    },
    SearchCandidate {
        key: "search_count_distinct",
        func: search_count_distinct,
    },
    SearchCandidate {
        key: "search_binary_search",
        func: search_binary_search,
    },
    SearchCandidate {
        key: "search_longest_plateau",
        func: search_longest_plateau,
    },
    SearchCandidate {
        key: "search_prefix_max_sum",
        func: search_prefix_max_sum,
    },
    SearchCandidate {
        key: "search_arr_sum_squares",
        func: search_arr_sum_squares,
    },
    SearchCandidate {
        key: "search_sum_absolute",
        func: search_sum_absolute,
    },
    SearchCandidate {
        key: "search_count_evens",
        func: search_count_evens,
    },
    SearchCandidate {
        key: "search_sum_positives",
        func: search_sum_positives,
    },
    SearchCandidate {
        key: "search_sum_at_even_indices",
        func: search_sum_at_even_indices,
    },
    SearchCandidate {
        key: "search_kth_from_end",
        func: search_kth_from_end,
    },
    SearchCandidate {
        key: "search_max_abs",
        func: search_max_abs,
    },
    SearchCandidate {
        key: "search_lucas_loop",
        func: search_lucas_loop,
    },
    SearchCandidate {
        key: "search_celsius_to_fahrenheit",
        func: search_celsius_to_fahrenheit,
    },
    SearchCandidate {
        key: "search_is_perfect_square",
        func: search_is_perfect_square,
    },
    SearchCandidate {
        key: "search_next_power_of_2",
        func: search_next_power_of_2,
    },
    SearchCandidate {
        key: "search_min_positive",
        func: search_min_positive,
    },
    SearchCandidate {
        key: "search_count_peaks",
        func: search_count_peaks,
    },
    SearchCandidate {
        key: "search_alternating_sum",
        func: search_alternating_sum,
    },
    SearchCandidate {
        key: "search_dot_product",
        func: search_dot_product,
    },
    SearchCandidate {
        key: "search_leading_digit",
        func: search_leading_digit,
    },
    SearchCandidate {
        key: "search_popcount",
        func: search_popcount,
    },
    SearchCandidate {
        key: "search_is_palindrome_arr",
        func: search_is_palindrome_arr,
    },
    SearchCandidate {
        key: "search_sum_odd_indexed",
        func: search_sum_odd_indexed,
    },
    SearchCandidate {
        key: "search_count_zeros",
        func: search_count_zeros,
    },
    SearchCandidate {
        key: "search_closure_map_sum",
        func: search_closure_map_sum,
    },
    SearchCandidate {
        key: "search_max_pair_diff",
        func: search_max_pair_diff,
    },
    SearchCandidate {
        key: "search_struct_pair_patterns",
        func: search_struct_pair_patterns,
    },
    SearchCandidate {
        key: "search_trimmed_len",
        func: search_trimmed_len,
    },
    SearchCandidate {
        key: "search_starts_with_literal",
        func: search_starts_with_literal,
    },
    SearchCandidate {
        key: "search_contains_literal",
        func: search_contains_literal,
    },
    SearchCandidate {
        key: "search_vowel_count",
        func: search_vowel_count,
    },
    SearchCandidate {
        key: "search_count_words",
        func: search_count_words,
    },
    SearchCandidate {
        key: "search_palindrome",
        func: search_palindrome,
    },
    SearchCandidate {
        key: "search_suffix_class",
        func: search_suffix_class,
    },
    SearchCandidate {
        key: "search_gcd_loop",
        func: search_gcd_loop,
    },
    SearchCandidate {
        key: "search_abs_diff_formula",
        func: search_abs_diff_formula,
    },
    SearchCandidate {
        key: "search_max2_formula",
        func: search_max2_formula,
    },
    SearchCandidate {
        key: "search_safe_div_or_neg1_branch",
        func: search_safe_div_or_neg1_branch,
    },
    SearchCandidate {
        key: "search_positive_or_default_branch",
        func: search_positive_or_default_branch,
    },
    SearchCandidate {
        key: "search_clamp_formula",
        func: search_clamp_formula,
    },
    SearchCandidate {
        key: "search_sign_branch",
        func: search_sign_branch,
    },
    SearchCandidate {
        key: "search_is_even_formula",
        func: search_is_even_formula,
    },
    SearchCandidate {
        key: "search_lcm_formula",
        func: search_lcm_formula,
    },
    SearchCandidate {
        key: "search_unary_range_loop",
        func: search_unary_range_loop,
    },
    SearchCandidate {
        key: "search_power_loop",
        func: search_power_loop,
    },
    SearchCandidate {
        key: "search_collatz_loop",
        func: search_collatz_loop,
    },
    SearchCandidate {
        key: "search_is_prime_loop",
        func: search_is_prime_loop,
    },
    SearchCandidate {
        key: "search_digit_loop",
        func: search_digit_loop,
    },
    SearchCandidate {
        key: "search_fib_iter_loop",
        func: search_fib_iter_loop,
    },
    SearchCandidate {
        key: "search_count_divisors_loop",
        func: search_count_divisors_loop,
    },
    SearchCandidate {
        key: "search_sum_of_divisors_loop",
        func: search_sum_of_divisors_loop,
    },
    SearchCandidate {
        key: "search_sum_odd_digits_loop",
        func: search_sum_odd_digits_loop,
    },
    SearchCandidate {
        key: "search_harmonic_sum_loop",
        func: search_harmonic_sum_loop,
    },
    SearchCandidate {
        key: "search_triangular_check_loop",
        func: search_triangular_check_loop,
    },
    SearchCandidate {
        key: "search_euler_totient_loop",
        func: search_euler_totient_loop,
    },
    SearchCandidate {
        key: "search_polynomial_quadratic",
        func: search_polynomial_quadratic,
    },
    SearchCandidate {
        key: "search_min3_branch",
        func: search_min3_branch,
    },
    SearchCandidate {
        key: "search_combat_resolve",
        func: search_combat_resolve,
    },
    SearchCandidate {
        key: "search_score_tracker",
        func: search_score_tracker,
    },
    SearchCandidate {
        key: "search_vending_change",
        func: search_vending_change,
    },
    SearchCandidate {
        key: "search_turn_order_rotate",
        func: search_turn_order_rotate,
    },
    SearchCandidate {
        key: "search_grid_bounds_check",
        func: search_grid_bounds_check,
    },
    SearchCandidate {
        key: "search_simulate_gravity",
        func: search_simulate_gravity,
    },
    SearchCandidate {
        key: "search_piecewise_affine",
        func: search_piecewise_affine,
    },
    SearchCandidate {
        key: "search_scalar_expr",
        func: search_scalar_expr,
    },
    SearchCandidate {
        key: "search_single_branch",
        func: search_single_branch,
    },
    SearchCandidate {
        key: "search_two_branch",
        func: search_two_branch,
    },
    // Array-feature composition — an exact affine mix of array reductions (sum,
    // max, len, count_positive, sum_of_squares, …) plus the scalar args, e.g.
    // `5 + 2*len + sum`, `sum_sq - 3*sum`. Placed at the tail of the array block,
    // AFTER every dedicated single-reduction solver (which own and run first on
    // their bare-reduction problems) and BEFORE the scalar/affine block. Refuses
    // the single-reduction restatement and constant outputs, so it only claims
    // the genuinely compositional array rules the dedicated solvers cannot express.
    SearchCandidate {
        key: "array_affine_features",
        func: search_array_affine_features,
    },
    // Separable degree-2 family — tried ahead of the linear solvers so curved
    // (non-zero curvature) data is recovered exactly, but after the single-purpose
    // scalar/array solvers. Returns None on pure-linear multi-arg data (early-out),
    // so it never steals a problem search_affine owns.
    SearchCandidate {
        key: "search_polynomial_multi",
        func: search_polynomial_multi,
    },
    // Clamped-affine family — a floor/cap/band saturating an affine rule. Fires
    // only when the data is NOT one straight line (verified_result rejects any
    // coincidental bound), so it never steals a problem the pure solvers own.
    SearchCandidate {
        key: "search_clamp_affine",
        func: search_clamp_affine,
    },
    // Compositional ("think-in-code") family — an exact affine rule over derived
    // features (squares, cross-terms, modulo, floor-div). Sparse-first + fully
    // verified, so it only claims genuinely nonlinear rules the pure linear and
    // separable-polynomial solvers cannot express, and refuses otherwise.
    SearchCandidate {
        key: "search_composed_features",
        func: search_composed_features,
    },
    // Multi-argument linear family — placed last so single-input solvers keep
    // their problems; these only catch the 2-3 arg rules the others cannot
    // express at all.
    SearchCandidate {
        key: "search_affine",
        func: search_affine,
    },
    SearchCandidate {
        key: "search_affine_threshold",
        func: search_affine_threshold,
    },
    SearchCandidate {
        key: "search_affine_piecewise",
        func: search_affine_piecewise,
    },
    // Conditional logic — `if x % m == r { affine } else { affine }`, a modular/
    // parity class split with an exact affine on each side. Last in the affine
    // block so the non-branching and argument-threshold solvers keep their
    // problems; this catches the genuine modular branches they cannot express
    // (argument thresholds stay with the threshold/scalar-branch solvers, which
    // own breakpoint placement). Fully verified, so a coincidental split is
    // rejected.
    SearchCandidate {
        key: "search_predicate_branch",
        func: search_predicate_branch,
    },
    // Full modular case analysis — `match x % m { 0 => affine, 1 => affine, … }`,
    // a distinct affine per residue class. Generalises the 2-way predicate
    // branch to the full cyclic split (day % 7, phase % 3); runs after it so the
    // simpler 2-way form is preferred. Fully verified.
    SearchCandidate {
        key: "search_modular_cases",
        func: search_modular_cases,
    },
    // Value-based branching — `max(A(x), B(x))` / `min(A(x), B(x))`, the upper/
    // lower envelope of two non-constant affines ("take the better of two
    // formulas"). The winning region is a half-space carved from the data, not an
    // axis threshold, so no threshold/branch solver expresses it. Recovered by
    // iterative partition refinement, fully verified.
    SearchCandidate {
        key: "search_minmax_affine",
        func: search_minmax_affine,
    },
];

fn ranked_search_candidates(problem: &Problem) -> Vec<SearchCandidate> {
    let mut ranked: Vec<(usize, SearchCandidate)> =
        SEARCH_CANDIDATES.iter().copied().enumerate().collect();
    let recommended = search_family_router::recommend_detailed(problem);
    if recommended.is_empty() {
        return ranked.into_iter().map(|(_, candidate)| candidate).collect();
    }

    let positions: std::collections::HashMap<&str, usize> = recommended
        .iter()
        .enumerate()
        .map(|(idx, rec)| (rec.method.as_str(), idx))
        .collect();
    ranked.sort_by_key(|(default_idx, candidate)| {
        (
            positions.get(candidate.key).copied().unwrap_or(usize::MAX),
            *default_idx,
        )
    });
    ranked.into_iter().map(|(_, candidate)| candidate).collect()
}

#[cfg(test)]
pub(super) fn ranked_search_candidate_keys(problem: &Problem) -> Vec<&'static str> {
    ranked_search_candidates(problem)
        .into_iter()
        .map(|candidate| candidate.key)
        .collect()
}

/// Exact multi-argument linear solvers, run before any other stage. A 2-3 arg
/// affine or single-threshold-affine rule is recovered in microseconds by a
/// direct integer linear solve, so it should never fall through to the slower
/// search/gradient stages (which environment-specific initialisation can
/// derail). 1-arg and non-linear data return None instantly, so this is a cheap
/// no-op for everything else.
pub(super) fn solve_multi_arg_affine(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    search_affine(problem, fn_name)
        .or_else(|| search_polynomial_multi(problem, fn_name))
        .or_else(|| search_clamp_affine(problem, fn_name))
        .or_else(|| search_composed_features(problem, fn_name))
        .or_else(|| search_affine_piecewise(problem, fn_name))
        .or_else(|| search_affine_threshold(problem, fn_name))
        .or_else(|| search_predicate_branch(problem, fn_name))
        .or_else(|| search_modular_cases(problem, fn_name))
        .or_else(|| search_minmax_affine(problem, fn_name))
}

pub(super) fn solve_by_search(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
    let ranked = ranked_search_candidates(problem);
    let mut tried = Vec::with_capacity(ranked.len());

    for candidate in ranked {
        tried.push(candidate.key);
        if let Some(result) = (candidate.func)(problem, fn_name) {
            search_family_router::record_attempt(problem, &tried, Some(candidate.key));
            return Some(result);
        }
    }

    search_family_router::record_attempt(problem, &tried, None);
    None
}

pub(super) fn solve_problem_search_only(problem: &Problem) -> SolveResult {
    let fn_name = problem.function_name();
    if let Some(result) = solve_by_search(problem, fn_name) {
        return result;
    }
    SolveResult {
        success: false,
        code: String::new(),
        method: family_name(problem),
        error: Some("search-only mode could not synthesize this problem".to_string()),
        metadata: DifferentiableMetadata::default(),
    }
}
