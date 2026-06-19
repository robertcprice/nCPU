use crate::benchmark::{Problem, Value};
use crate::differentiable::{
    solve_problem_differentiable_only as solve_problem_differentiable_bridge,
    DifferentiableMetadata,
};
use crate::method_router;

mod analogy;
mod benchmarking;
mod experience_advisor;
mod helpers;
mod legacy_fallback;
mod probabilistic;
mod pipeline;
mod post_enumerative;
mod routing;
mod scalar_search;
mod search;
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
mod hierarchical;
mod recovery;
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

/// Solve from natural language description
/// Uses Linguigenesis bridge to parse NL → examples → code
#[cfg(feature = "nl")]
pub fn solve_from_nl(description: &str, fn_name: Option<&str>) -> Result<SolveResult, String> {
    use crate::linguigenesis_bridge::LinguigenesisBridge;

    let bridge = LinguigenesisBridge::new();

    // Parse NL into examples
    let examples = bridge.nl_to_examples(description)
        .map_err(|e| format!("Failed to parse NL: {}", e))?;

    if examples.is_empty() {
        return Err("No examples generated from description".to_string());
    }

    // Build Problem from examples
    let name = fn_name.unwrap_or("synthesized").to_string();
    let signature = crate::linguigenesis_bridge::infer_signature(&name, &examples);
    let signature = Box::leak(signature.into_boxed_str()); // Leak to get &'static str
    let problem = Problem {
        name,
        category: "nl",
        description: "", // Can't store non-static str, use name field
        signature, // Inferred from examples
        examples,
        holdouts: Vec::new(),
        reference_code: "",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: Vec::new(),
    };

    // Solve using standard pipeline
    Ok(solve_problem(&problem))
}

/// Get belief state from NL (for debugging/analysis)
#[cfg(feature = "nl")]
pub fn analyze_nl(description: &str) -> Result<crate::linguigenesis_bridge::BridgeBeliefState, String> {
    use crate::linguigenesis_bridge::LinguigenesisBridge;

    let bridge = LinguigenesisBridge::new();
    let belief = bridge.get_belief_state(description)
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
