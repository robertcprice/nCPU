use crate::benchmark::{Problem, Value};
use crate::differentiable::{
    solve_problem_differentiable_only as solve_problem_differentiable_bridge,
    DifferentiableMetadata,
};
use crate::method_router;

mod benchmarking;
mod helpers;
mod legacy_fallback;
mod pipeline;
mod post_enumerative;
mod routing;
mod scalar_search;
mod search;
mod search_catalog;
mod search_catalog_advanced;
mod search_catalog_codegen;
mod search_catalog_runtime;
mod search_catalog_simple;
mod search_codegen;
mod search_families;
mod search_numeric_families;
mod search_runtime;
mod search_scalar_families;
mod search_text_families;
mod signature;

use self::helpers::{
    family_name, int_value, templ, validate_array_and_int, validate_binary_int,
    validate_quaternary_int, validate_ternary_int, validate_two_arrays, validate_unary_array,
    validate_unary_int, validate_unary_pair, validate_unary_str,
};
use self::signature::{
    parse_param_types, scalar_param_names, scalar_params_decl, unary_pair_examples,
    unary_string_examples, ParamType,
};

#[cfg(test)]
use self::benchmarking::find_python_warmstart_model;
#[cfg(test)]
use self::post_enumerative::search_result_preempts_native_gradient;
use self::post_enumerative::{
    solve_problem_after_enumeration, solve_problem_from_preemptive_search_teacher,
};
#[cfg(test)]
use self::routing::planned_post_enumerative_routes;
use self::routing::{
    normalized_router_stats, post_enumerative_context, should_bypass_solved_cache,
    should_try_enumerative, ROUTE_ENUMERATIVE,
};
#[cfg(test)]
use self::routing::{ROUTE_ARRAY_GRADIENT, ROUTE_SCALAR_GRADIENT, ROUTE_SEARCH_TEACHER};
use self::search::solve_by_search;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveResult {
    pub success: bool,
    pub code: String,
    pub method: String,
    pub error: Option<String>,
    pub metadata: DifferentiableMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkSummary {
    pub total: usize,
    pub solved: usize,
    pub failures: Vec<String>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostEnumerativeStage {
    ScalarGradientOnly,
    ArrayGradient,
    ExprOnly,
    SearchTeacher,
    ExprTemplates,
    ScalarTemplates,
    RegisterMachine,
    BridgeGradient,
    ReferenceDistillation,
    NativeScalarTeacherDistillation,
    ArrayTeacherDistillation,
    TemplateReference,
    Search,
}

#[cfg(test)]
fn post_enumerative_stage_order(problem: &Problem) -> Vec<PostEnumerativeStage> {
    let fn_name = problem.function_name();
    let search_teacher_preempts = solve_by_search(problem, fn_name)
        .map(|result| search_result_preempts_native_gradient(&result))
        .unwrap_or(false);
    let n_args = problem
        .examples
        .first()
        .map(|e| e.inputs.len())
        .unwrap_or(0);
    let is_external = problem.category == "external";
    let has_array_input = problem
        .examples
        .first()
        .map(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Array(_))))
        .unwrap_or(false);
    let scalar_only_inputs = problem
        .examples
        .iter()
        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))));
    let mut stages = Vec::new();
    if search_teacher_preempts {
        stages.push(PostEnumerativeStage::SearchTeacher);
    }
    if scalar_only_inputs && (!is_external || n_args <= 3) {
        stages.push(PostEnumerativeStage::ScalarGradientOnly);
    } else if !has_array_input {
        // scalar gradient stage is skipped for non-scalar, non-array problems
    }
    stages.push(PostEnumerativeStage::ArrayGradient);
    stages.push(PostEnumerativeStage::ExprOnly);
    if !search_teacher_preempts {
        stages.push(PostEnumerativeStage::SearchTeacher);
    }
    stages.push(PostEnumerativeStage::RegisterMachine);
    if scalar_only_inputs && (!is_external || n_args <= 3) {
        stages.push(PostEnumerativeStage::BridgeGradient);
    }
    if scalar_only_inputs && (!is_external || n_args <= 3) && !problem.reference_code.is_empty() {
        stages.push(PostEnumerativeStage::ReferenceDistillation);
    }
    if scalar_only_inputs && !problem.reference_code.is_empty() {
        stages.push(PostEnumerativeStage::NativeScalarTeacherDistillation);
    }
    if has_array_input && !problem.reference_code.is_empty() {
        stages.push(PostEnumerativeStage::ArrayTeacherDistillation);
    }
    stages.push(PostEnumerativeStage::ExprTemplates);
    if scalar_only_inputs && (!is_external || n_args <= 3) {
        stages.push(PostEnumerativeStage::ScalarTemplates);
    }
    stages.push(PostEnumerativeStage::TemplateReference);
    if !(is_external && n_args > 3) {
        stages.push(PostEnumerativeStage::Search);
    }
    stages
}

pub fn solve_problem_with_legacy_fallback(problem: &Problem) -> SolveResult {
    let result = solve_problem_search_only(problem);
    if result.success {
        return result;
    }
    legacy_fallback::solve(problem)
}

pub fn solve_problem_legacy_only(problem: &Problem) -> SolveResult {
    legacy_fallback::solve(problem)
}

pub fn solve_problem_differentiable_only(problem: &Problem) -> SolveResult {
    let result = solve_problem_differentiable_bridge(problem);
    SolveResult {
        success: result.success,
        code: result.code,
        method: result.method,
        error: result.error,
        metadata: result.metadata,
    }
}

pub fn solve_problem_prefer_differentiable(problem: &Problem) -> SolveResult {
    post_enumerative::solve_problem_prefer_differentiable(problem)
}

pub fn solve_problem(problem: &Problem) -> SolveResult {
    pipeline::solve_problem(problem)
}

pub fn solve_problem_search_only(problem: &Problem) -> SolveResult {
    search::solve_problem_search_only(problem)
}

pub fn solve_benchmark_with_legacy_fallback(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::run_benchmark(problems, solve_problem_with_legacy_fallback)
}

pub fn solve_benchmark_legacy_only(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::run_benchmark(problems, solve_problem_legacy_only)
}

pub fn solve_benchmark_differentiable_only(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::run_benchmark(problems, solve_problem_differentiable_only)
}

pub fn solve_benchmark_prefer_differentiable(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::solve_benchmark_prefer_differentiable(problems)
}

pub fn solve_benchmark(problems: &[Problem]) -> BenchmarkSummary {
    solve_benchmark_prefer_differentiable(problems)
}

pub fn solve_benchmark_search_only(problems: &[Problem]) -> BenchmarkSummary {
    benchmarking::run_benchmark(problems, solve_problem_search_only)
}

#[cfg(test)]
mod tests;
