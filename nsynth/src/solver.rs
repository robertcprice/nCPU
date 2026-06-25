use crate::benchmark::{Problem, Value};
use crate::differentiable::{
    solve_problem_differentiable_only as solve_problem_differentiable_bridge,
    DifferentiableMetadata,
};
use crate::method_router;

mod analogy;
mod benchmarking;
mod experience_advisor;
mod generalization;
mod helpers;
mod hierarchical;
mod legacy_fallback;
mod pipeline;
mod post_enumerative;
mod probabilistic;
mod recovery;
mod routing;
mod scalar_search;
mod search;
// Salvaged from stale fork: difficulty-aware routing, curriculum/sequencing,
// emergent method stats, allocation/portfolio, and the enhanced solver layer.
pub mod allocation_strategy;
mod curriculum;
mod difficulty;
pub mod enhanced_integration;
pub mod method_stats;
pub mod metrics;
pub mod parallel_executor;
pub mod portfolio_router;
mod sequencing;
mod search_affine;
mod search_array_compose;
mod search_bitwise;
mod search_catalog;
mod search_catalog_advanced;
mod search_catalog_codegen;
mod search_catalog_runtime;
mod search_catalog_simple;
mod search_codegen;
mod search_families;
mod search_float;
mod search_numeric_families;
mod search_runtime;
mod search_scalar_families;
mod search_text_families;
mod search_time_families;
mod search_tree_families;
mod signature;

use self::helpers::{
    family_name, int_value, templ, validate_array_and_int, validate_binary_int,
    validate_quaternary_int, validate_ternary_int, validate_two_arrays, validate_unary_array,
    validate_unary_int, validate_unary_pair, validate_unary_str,
};
use self::signature::{
    parse_param_types, scalar_param_names, scalar_params_decl, unary_array_examples,
    unary_pair_examples, unary_string_examples, ParamType,
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

/// Re-export the engine's stateful-reducer op surface so the reflective
/// capability descriptor (`synthesis::stateful_reducer_surface`) can BIND its
/// mined NL vocabulary to the SAME slice the solver enumerates over — keeping the
/// mined surface emergent (a new reducer/op grows NL reach with no hand edit) and
/// fail-closed (the surface guard drifts if the descriptor diverges).
pub(crate) use self::search_families::{
    stateful_reducer_apply, STATEFUL_REDUCER_NAMES, STATEFUL_REDUCER_OPS,
};

/// Error categories for better error handling and user feedback.
///
/// Ported from the stale fork's richer `SolveResult` model. The canonical
/// `SolveResult` does not carry an `error_category` field, but several ported
/// learning/observability modules (`metrics`, `method_stats`, `transfer_learning`)
/// classify outcomes through this enum, so it is exposed here as a standalone
/// public type they can share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Transient errors that may succeed on retry (network, timeout, resource contention)
    Transient { retry_after_ms: Option<u64> },
    /// Permanent errors that won't change on retry (syntax, type errors, invalid input)
    Permanent,
    /// Partial success - some parts succeeded but not all
    Partial { succeeded: usize, total: usize },
    /// Resource exhaustion (out of memory, compute budget exceeded)
    ResourceExhaustion,
    /// Configuration errors (missing files, invalid settings)
    Configuration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveResult {
    pub success: bool,
    pub code: String,
    pub method: String,
    pub error: Option<String>,
    pub metadata: DifferentiableMetadata,
}

// Type aliases for compatibility with orchestrator
pub type SynthesisResult = SolveResult;

#[derive(Debug, Clone, thiserror::Error)]
pub enum SolverError {
    #[error("Communication error: {0}")]
    CommunicationError(String),
    #[error("No solution found: {0}")]
    NoSolutionFound(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Other: {0}")]
    Other(String),
    #[error("Task join error: {0}")]
    JoinError(String),
}

impl From<std::io::Error> for SolverError {
    fn from(err: std::io::Error) -> Self {
        SolverError::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for SolverError {
    fn from(err: serde_json::Error) -> Self {
        SolverError::ParseError(err.to_string())
    }
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
    stages.push(PostEnumerativeStage::ExprTemplates);
    if scalar_only_inputs && (!is_external || n_args <= 3) {
        stages.push(PostEnumerativeStage::ScalarTemplates);
    }
    if !(is_external && n_args > 3) {
        stages.push(PostEnumerativeStage::Search);
    }
    stages
}

pub fn solve_problem_with_legacy_fallback(problem: &Problem) -> SolveResult {
    let synthesis_problem = problem.synthesis_view();
    let result = search::solve_problem_search_only(problem);
    if result.success {
        return result;
    }
    legacy_fallback::solve(&synthesis_problem)
}

pub fn solve_problem_legacy_only(problem: &Problem) -> SolveResult {
    legacy_fallback::solve(&problem.synthesis_view())
}

pub fn solve_problem_differentiable_only(problem: &Problem) -> SolveResult {
    let synthesis_problem = problem.synthesis_view();
    let result = solve_problem_differentiable_bridge(&synthesis_problem);
    SolveResult {
        success: result.success,
        code: result.code,
        method: result.method,
        error: result.error,
        metadata: result.metadata,
    }
}

pub fn solve_problem_prefer_differentiable(problem: &Problem) -> SolveResult {
    post_enumerative::solve_problem_prefer_differentiable(&problem.synthesis_view())
}

pub fn solve_problem(problem: &Problem) -> SolveResult {
    pipeline::solve_problem(&problem.synthesis_view())
}

/// Build a whole-word string->string lookup-table program from single-arg
/// examples (irregular inflection and similar arbitrary lexicons), or None if
/// the mapping is not such a lexicon. Exposed so the `--problem-json` CLI shares
/// the same lexicon recovery the in-process solver uses.
pub fn string_lexicon_map_code(train: &[(Vec<String>, String)], fn_name: &str) -> Option<String> {
    pipeline::string_lexicon_map_code(train, fn_name)
}

/// Search-only portfolio solve. Holdouts remain on the `Problem` so
/// `verified_result` can reject training-only overfits; candidate generators
/// only read `examples`, never holdouts.
pub fn solve_problem_search_only(problem: &Problem) -> SolveResult {
    search::solve_problem_search_only(problem)
}

/// Solve using the Phase 2 agentic orchestration layer.
///
/// The problem is decomposed into a dependency graph of synthesis subtasks by the
/// adaptive `TaskDecomposer`, the standard pipeline solver is invoked exactly once
/// to produce the actual program, and the planning graph is then walked in
/// dependency order to record completion. The decomposer is an orchestration layer
/// around the real solver — it does not fabricate solutions. On a planning or
/// synthesis failure the error is surfaced through the normal `SolveResult` shape.
pub fn solve_problem_agentic(problem: &Problem) -> SolveResult {
    use crate::agent::{DecompositionStrategy, TaskDecomposer};
    use std::time::Instant;

    let started = Instant::now();

    // Multi-function problems are genuine compositions: synthesize each declared
    // function independently via the real solver, driven by the async executor,
    // and compose the results.
    let result = if !problem.functions.is_empty() {
        crate::agent::solve_compositional(problem)
    } else {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::Adaptive);
        match decomposer.integrate_solver(problem, solve_problem) {
            Ok(result) => result,
            Err(e) => SolveResult {
                success: false,
                code: String::new(),
                method: "agentic".to_string(),
                error: Some(e.to_string()),
                metadata: Default::default(),
            },
        }
    };

    // Record the run as an experience for the learning loop (best-effort: a
    // recording failure must never fail a solve). Skipped under test.
    record_agentic_experience(problem, &result, started.elapsed().as_millis() as u64);

    result
}

/// Observability for the Phase 4.1 learning loop: the experience-derived route
/// win-boosts that would be folded into routing for `problem`, sorted strongest
/// first. Empty when no experience DB is loaded. Exposed for the CLI
/// `--experience-boosts` flag and for inspecting cold-vs-warm behavior.
pub fn experience_route_boosts(problem: &Problem) -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = experience_advisor::route_boosts(problem)
        .into_iter()
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

/// Best-effort recording of an agentic solve into the experience DB. All errors
/// are swallowed — learning is a side-channel, never a failure mode for solving.
/// Shares its DB path with the routing advisor so writes and reads agree.
fn record_agentic_experience(problem: &Problem, result: &SolveResult, solve_time_ms: u64) {
    let Some(path) = experience_advisor::db_path() else {
        return;
    };
    if let Ok(mut db) = crate::learning::experience::ExperienceDB::new(path) {
        let _ = db.record_experience(problem, result, solve_time_ms);
    }
}

/// Universal synthesis entry: NL → registry requirement → Problem → solver portfolio.
pub fn solve_from_description(
    description: &str,
    fn_name: Option<&str>,
) -> Result<SolveResult, String> {
    let bridge = crate::linguigenesis_bridge::LinguigenesisBridge::new();
    bridge.synthesize_from_description(description, fn_name)
}

/// Solve from natural language description (`nl` feature alias).
#[cfg(feature = "nl")]
pub fn solve_from_nl(description: &str, fn_name: Option<&str>) -> Result<SolveResult, String> {
    solve_from_description(description, fn_name)
}

/// Get belief state from NL (for debugging/analysis)
#[cfg(feature = "nl")]
pub fn analyze_nl(
    description: &str,
) -> Result<crate::linguigenesis_bridge::BridgeBeliefState, String> {
    use crate::linguigenesis_bridge::LinguigenesisBridge;

    let bridge = LinguigenesisBridge::new();
    let belief = bridge
        .get_belief_state(description)
        .map_err(|e| format!("Failed to parse NL: {}", e))?;

    Ok(crate::linguigenesis_bridge::BridgeBeliefState {
        intent_type: format!("{:?}", belief.intent.intent_type),
        entities: belief.comprehension.entities,
        confidence: belief.reflection.clarity_score as f64,
    })
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
