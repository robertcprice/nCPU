use std::collections::HashMap;

use crate::benchmark::{Problem, Value};
use crate::method_router;
use crate::solved_cache::CachedSolution;

pub(super) const ROUTE_SCALAR_GRADIENT: &str = "scalar_gradient";
pub(super) const ROUTE_ARRAY_GRADIENT: &str = "array_gradient";
pub(super) const ROUTE_ENUMERATIVE: &str = "enumerative";
pub(super) const ROUTE_EXPR_ONLY: &str = "expr_only";
pub(super) const ROUTE_SEARCH_TEACHER: &str = "search_teacher";
pub(super) const ROUTE_REGISTER_MACHINE: &str = "register_machine";
pub(super) const ROUTE_BRIDGE_GRADIENT: &str = "bridge_gradient";
pub(super) const ROUTE_REFERENCE_DISTILLATION: &str = "reference_distillation";
pub(super) const ROUTE_NATIVE_REFERENCE_DISTILLATION: &str = "native_reference_distillation";
pub(super) const ROUTE_ARRAY_REFERENCE_DISTILLATION: &str = "array_reference_distillation";
pub(super) const ROUTE_EXPR_TEMPLATES: &str = "expr_templates";
pub(super) const ROUTE_SCALAR_TEMPLATES: &str = "scalar_templates";
pub(super) const ROUTE_TEMPLATE_REFERENCE: &str = "template_reference";
pub(super) const ROUTE_SEARCH: &str = "search";

const ENUMERATION_SKIP_MIN_WINS: u32 = 3;
const ENUMERATION_SKIP_MIN_SUCCESS_RATE_PERCENT: u32 = 70;
const CACHE_BYPASS_MIN_WINS: u32 = 4;
const CACHE_BYPASS_MIN_SUCCESS_RATE_PERCENT: u32 = 80;
const CACHE_BYPASS_SUCCESS_MARGIN_PERCENT: u32 = 20;

#[derive(Clone, Copy, Debug)]
pub(super) struct PostEnumerativeContext {
    pub(super) n_args: usize,
    pub(super) is_external: bool,
    pub(super) has_array_input: bool,
    pub(super) scalar_only_inputs: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NormalizedRouteStats {
    pub(super) route: &'static str,
    pub(super) wins: u32,
    pub(super) misses: u32,
}

impl NormalizedRouteStats {
    pub(super) fn attempts(self) -> u32 {
        self.wins + self.misses
    }

    pub(super) fn success_rate_percent(self) -> u32 {
        let attempts = self.attempts();
        if attempts == 0 {
            return 0;
        }
        self.wins * 100 / attempts
    }
}

pub(super) fn post_enumerative_context(problem: &Problem) -> PostEnumerativeContext {
    PostEnumerativeContext {
        n_args: problem
            .examples
            .first()
            .map(|e| e.inputs.len())
            .unwrap_or(0),
        is_external: problem.category == "external",
        has_array_input: problem
            .examples
            .first()
            .map(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Array(_))))
            .unwrap_or(false),
        scalar_only_inputs: problem
            .examples
            .iter()
            .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_)))),
    }
}

pub(super) fn route_is_applicable(
    route: &'static str,
    problem: &Problem,
    ctx: &PostEnumerativeContext,
) -> bool {
    match route {
        ROUTE_SCALAR_GRADIENT
        | ROUTE_BRIDGE_GRADIENT
        | ROUTE_REFERENCE_DISTILLATION
        | ROUTE_SCALAR_TEMPLATES => ctx.scalar_only_inputs && (!ctx.is_external || ctx.n_args <= 3),
        ROUTE_ARRAY_GRADIENT
        | ROUTE_SEARCH_TEACHER
        | ROUTE_REGISTER_MACHINE
        | ROUTE_EXPR_TEMPLATES
        | ROUTE_SEARCH => true,
        ROUTE_EXPR_ONLY | ROUTE_NATIVE_REFERENCE_DISTILLATION => ctx.scalar_only_inputs,
        ROUTE_ARRAY_REFERENCE_DISTILLATION => ctx.has_array_input,
        ROUTE_TEMPLATE_REFERENCE => !problem.reference_code.is_empty(),
        _ => !problem.examples.is_empty(),
    }
}

pub(super) fn default_post_enumerative_routes(
    problem: &Problem,
    ctx: &PostEnumerativeContext,
) -> Vec<&'static str> {
    let mut routes = Vec::new();

    if route_is_applicable(ROUTE_SCALAR_GRADIENT, problem, ctx) {
        routes.push(ROUTE_SCALAR_GRADIENT);
    }
    routes.push(ROUTE_ARRAY_GRADIENT);
    if route_is_applicable(ROUTE_EXPR_ONLY, problem, ctx) {
        routes.push(ROUTE_EXPR_ONLY);
    }
    routes.push(ROUTE_SEARCH_TEACHER);
    routes.push(ROUTE_REGISTER_MACHINE);
    if route_is_applicable(ROUTE_BRIDGE_GRADIENT, problem, ctx) {
        routes.push(ROUTE_BRIDGE_GRADIENT);
    }
    if route_is_applicable(ROUTE_REFERENCE_DISTILLATION, problem, ctx)
        && !problem.reference_code.is_empty()
    {
        routes.push(ROUTE_REFERENCE_DISTILLATION);
    }
    if route_is_applicable(ROUTE_NATIVE_REFERENCE_DISTILLATION, problem, ctx)
        && !problem.reference_code.is_empty()
    {
        routes.push(ROUTE_NATIVE_REFERENCE_DISTILLATION);
    }
    if route_is_applicable(ROUTE_ARRAY_REFERENCE_DISTILLATION, problem, ctx)
        && !problem.reference_code.is_empty()
    {
        routes.push(ROUTE_ARRAY_REFERENCE_DISTILLATION);
    }
    routes.push(ROUTE_EXPR_TEMPLATES);
    if route_is_applicable(ROUTE_SCALAR_TEMPLATES, problem, ctx) {
        routes.push(ROUTE_SCALAR_TEMPLATES);
    }
    if route_is_applicable(ROUTE_TEMPLATE_REFERENCE, problem, ctx) {
        routes.push(ROUTE_TEMPLATE_REFERENCE);
    }
    routes.push(ROUTE_SEARCH);

    routes
}

pub(super) fn normalize_router_route(route: &str) -> Option<&'static str> {
    match route {
        ROUTE_ENUMERATIVE => Some(ROUTE_ENUMERATIVE),
        route
            if route == "enumerative-array"
                || route == "enumerative-nested"
                || route == "enumerative-while-cond" =>
        {
            Some(ROUTE_ENUMERATIVE)
        }
        ROUTE_SCALAR_GRADIENT | "synth_gradient" => Some(ROUTE_SCALAR_GRADIENT),
        ROUTE_ARRAY_GRADIENT => Some(ROUTE_ARRAY_GRADIENT),
        route
            if route == "arr_gradient"
                || route == "univ_arr_gradient"
                || route.starts_with("arr_gradient_") =>
        {
            Some(ROUTE_ARRAY_GRADIENT)
        }
        ROUTE_EXPR_ONLY => Some(ROUTE_EXPR_ONLY),
        ROUTE_SEARCH_TEACHER => Some(ROUTE_SEARCH_TEACHER),
        ROUTE_REGISTER_MACHINE => Some(ROUTE_REGISTER_MACHINE),
        ROUTE_BRIDGE_GRADIENT => Some(ROUTE_BRIDGE_GRADIENT),
        ROUTE_REFERENCE_DISTILLATION => Some(ROUTE_REFERENCE_DISTILLATION),
        ROUTE_NATIVE_REFERENCE_DISTILLATION => Some(ROUTE_NATIVE_REFERENCE_DISTILLATION),
        ROUTE_ARRAY_REFERENCE_DISTILLATION => Some(ROUTE_ARRAY_REFERENCE_DISTILLATION),
        ROUTE_EXPR_TEMPLATES => Some(ROUTE_EXPR_TEMPLATES),
        "expr_template" | "loop_template" | "arr_template" => Some(ROUTE_EXPR_TEMPLATES),
        ROUTE_SCALAR_TEMPLATES | "template" => Some(ROUTE_SCALAR_TEMPLATES),
        ROUTE_TEMPLATE_REFERENCE => Some(ROUTE_TEMPLATE_REFERENCE),
        ROUTE_SEARCH => Some(ROUTE_SEARCH),
        route if route.starts_with("search_") => Some(ROUTE_SEARCH_TEACHER),
        route if route.starts_with("diff_gradient_") => Some(ROUTE_BRIDGE_GRADIENT),
        _ => None,
    }
}

pub(super) fn normalized_router_stats(
    problem: &Problem,
    ctx: &PostEnumerativeContext,
) -> Vec<NormalizedRouteStats> {
    let mut totals: HashMap<&'static str, (u32, u32)> = HashMap::new();

    for rec in method_router::recommend_detailed(problem) {
        let Some(route) = normalize_router_route(&rec.method) else {
            continue;
        };
        if route != ROUTE_ENUMERATIVE && !route_is_applicable(route, problem, ctx) {
            continue;
        }
        let entry = totals.entry(route).or_insert((0, 0));
        entry.0 += rec.wins;
        entry.1 += rec.misses;
    }

    let mut ranked: Vec<NormalizedRouteStats> = totals
        .into_iter()
        .map(|(route, (wins, misses))| NormalizedRouteStats {
            route,
            wins,
            misses,
        })
        .collect();
    ranked.sort_by(|a, b| {
        let lhs = (a.wins as u64) * (b.attempts() as u64);
        let rhs = (b.wins as u64) * (a.attempts() as u64);
        rhs.cmp(&lhs)
            .then_with(|| b.wins.cmp(&a.wins))
            .then_with(|| a.misses.cmp(&b.misses))
            .then_with(|| a.route.cmp(&b.route))
    });
    ranked
}

pub(super) fn recommended_post_enumerative_routes(
    problem: &Problem,
    ctx: &PostEnumerativeContext,
) -> Vec<&'static str> {
    let mut routes = Vec::new();
    for stats in normalized_router_stats(problem, ctx) {
        if stats.route == ROUTE_ENUMERATIVE {
            continue;
        }
        routes.push(stats.route);
    }
    routes
}

/// Routes whose per-problem cost dwarfs the enumerative guard stage.
///
/// The enumerative pass is a cheap *guard*: on array/scalar problems it
/// either matches a closed-form fold/expression in well under 0.1s or bails
/// immediately (measured: enumerative-array wins ≤0.05s, misses ~0.0s). The
/// gradient routes, by contrast, run thousands of Adam steps and routinely
/// take 20–60s on the array-reduction family.
///
/// The router's "skip enumerative" short-circuit exists to preempt the
/// enumerative grind for problems a cheaper downstream stage will solve
/// anyway (e.g. the ms-scale `search_teacher`). Letting it skip the guard in
/// favour of a *slower* route is strictly backwards: when enumerative would
/// have solved the problem in 0.02s, skipping it forces the full gradient
/// descent first. This caused array_sum/interactive_sum/reverse_sum variants
/// to regress from 0.02s (enumerative-array) to 20–60s (arr_gradient) once
/// the in-run router had accumulated enough array_gradient wins to favour it.
fn route_dwarfs_enumerative_guard(route: &'static str) -> bool {
    matches!(
        route,
        ROUTE_SCALAR_GRADIENT | ROUTE_ARRAY_GRADIENT | ROUTE_BRIDGE_GRADIENT
    )
}

pub(super) fn should_try_enumerative(
    problem: &Problem,
    ctx: &PostEnumerativeContext,
    non_scalar: bool,
    has_preemptive_search_hit: bool,
) -> bool {
    if non_scalar {
        return false;
    }
    if has_preemptive_search_hit {
        return false;
    }

    let ranked = normalized_router_stats(problem, ctx);
    let Some(top) = ranked.first().copied() else {
        return true;
    };
    if top.route == ROUTE_ENUMERATIVE {
        return true;
    }
    // Never trade away the cheap enumerative guard for a route that is far
    // more expensive than running enumerative itself. Enumerative is a
    // sub-0.1s pass; if it solves, we skip the 20–60s gradient grind entirely.
    if route_dwarfs_enumerative_guard(top.route) {
        return true;
    }
    if top.wins < ENUMERATION_SKIP_MIN_WINS {
        return true;
    }
    if top.success_rate_percent() < ENUMERATION_SKIP_MIN_SUCCESS_RATE_PERCENT {
        return true;
    }

    let enum_wins = ranked
        .iter()
        .find_map(|stats| (stats.route == ROUTE_ENUMERATIVE).then_some(stats.wins))
        .unwrap_or(0);
    enum_wins >= top.wins
}

fn route_supports_cache_bypass(route: &'static str) -> bool {
    matches!(
        route,
        ROUTE_SCALAR_GRADIENT
            | ROUTE_ARRAY_GRADIENT
            | ROUTE_EXPR_ONLY
            | ROUTE_SEARCH_TEACHER
            | ROUTE_REGISTER_MACHINE
            | ROUTE_BRIDGE_GRADIENT
    )
}

pub(super) fn should_bypass_solved_cache(
    problem: &Problem,
    ctx: &PostEnumerativeContext,
    cached: &CachedSolution,
) -> bool {
    let ranked = normalized_router_stats(problem, ctx);
    let Some(top) = ranked.first().copied() else {
        return false;
    };
    if !route_supports_cache_bypass(top.route) {
        return false;
    }
    if top.wins < CACHE_BYPASS_MIN_WINS {
        return false;
    }
    if top.success_rate_percent() < CACHE_BYPASS_MIN_SUCCESS_RATE_PERCENT {
        return false;
    }

    let cached_route = normalize_router_route(&cached.method);
    if cached_route == Some(top.route) {
        return false;
    }

    let cached_success_rate = cached_route
        .and_then(|route| ranked.iter().find(|stats| stats.route == route).copied())
        .map(|stats| stats.success_rate_percent())
        .unwrap_or(0);

    top.success_rate_percent() >= cached_success_rate + CACHE_BYPASS_SUCCESS_MARGIN_PERCENT
}

#[cfg(test)]
pub(super) fn planned_post_enumerative_routes(
    problem: &Problem,
    ctx: &PostEnumerativeContext,
) -> Vec<&'static str> {
    let mut routes = Vec::new();
    let mut seen = std::collections::HashSet::new();

    if super::post_enumerative::solve_problem_from_preemptive_search_teacher(problem).is_some()
        && seen.insert(ROUTE_SEARCH_TEACHER)
    {
        routes.push(ROUTE_SEARCH_TEACHER);
    }

    for route in recommended_post_enumerative_routes(problem, ctx) {
        if seen.insert(route) {
            routes.push(route);
        }
    }
    for route in default_post_enumerative_routes(problem, ctx) {
        if seen.insert(route) {
            routes.push(route);
        }
    }

    routes
}
