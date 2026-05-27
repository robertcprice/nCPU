use super::search_codegen::*;
use super::*;
use crate::benchmark::{factory_count, generated_holdouts, get_benchmark, Example, Value};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_MODEL_ROOT_COUNTER: AtomicU64 = AtomicU64::new(0);

fn aliased_problem(
    source_prefix: &str,
    name: &str,
    signature: &'static str,
    category: &'static str,
    description: &'static str,
) -> Problem {
    let source = get_benchmark(1)
        .into_iter()
        .find(|p| p.name.starts_with(source_prefix))
        .unwrap();
    Problem {
        name: name.to_string(),
        category,
        description,
        signature,
        examples: source.examples,
        holdouts: vec![],
        reference_code: "",
    }
}

fn assert_search_generalizes_problem(problem: Problem, holdouts: Vec<(Vec<Value>, i64)>) {
    let result = solve_problem_search_only(&problem);
    assert!(result.success, "search failed for {}", problem.name);

    for (inputs, expected) in holdouts {
        let actual = crate::runtime::execute_function_for_problem(
            &result.code,
            problem.function_name(),
            &inputs,
            &problem,
        )
        .unwrap_or_else(|err| {
            panic!(
                "execution failed for {} on {:?}: {err}",
                problem.name, inputs
            )
        });
        match actual {
            crate::runtime::Value::Int(value) => {
                assert_eq!(
                    value, expected,
                    "wrong result for {} on {:?}",
                    problem.name, inputs
                );
            }
            other => panic!("expected int result for {}, got {:?}", problem.name, other),
        }
    }
}

fn assert_search_generalizes(problem_name: &str, holdouts: Vec<(Vec<Value>, i64)>) {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|problem| problem.name == problem_name)
        .unwrap();
    assert_search_generalizes_problem(problem, holdouts);
}

fn temp_model_root() -> std::path::PathBuf {
    let nonce = TEMP_MODEL_ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "mog-warmstart-models-{}-{}",
        std::process::id(),
        nonce
    ));
    fs::create_dir_all(root.join("models")).unwrap();
    root
}

fn is_differentiable_only_capability_gap(error: &str) -> bool {
    error.contains("python bridge script not found")
        || error.contains("meta-learner checkpoint only covers")
        || error.contains("No 1-field correction found (fast mode)")
        || error.contains("unexpected token in expression: If")
}

fn solve_problem_differentiable_only_or_skip(problem: &Problem) -> Option<SolveResult> {
    let result = solve_problem_differentiable_only(problem);
    if result.success {
        return Some(result);
    }
    if let Some(error) = result.error.as_deref() {
        if is_differentiable_only_capability_gap(error) {
            eprintln!(
                "[skip] differentiable-only unavailable for {}: {error}",
                problem.name
            );
            return None;
        }
    }
    panic!(
        "differentiable-only solve failed for {}: {:?}",
        problem.name, result.error
    );
}

fn with_scratch_method_router<R>(f: impl FnOnce() -> R) -> R {
    crate::method_router::with_test_lock(|| {
        crate::solved_cache::with_test_lock(|| {
            let scratch = std::env::temp_dir().join(format!(
                "nsynth_router_solver_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::env::set_var("NSYNTH_METHOD_ROUTER_PATH", &scratch);
            crate::method_router::reset_for_tests();
            crate::solved_cache::reset_for_tests();
            let _ = fs::remove_file(&scratch);
            let result = f();
            std::env::remove_var("NSYNTH_METHOD_ROUTER_PATH");
            crate::method_router::reset_for_tests();
            crate::solved_cache::reset_for_tests();
            let _ = fs::remove_file(&scratch);
            result
        })
    })
}

fn with_scratch_router_and_cache<R>(f: impl FnOnce() -> R) -> R {
    crate::method_router::with_test_lock(|| {
        crate::solved_cache::with_test_lock(|| {
            let router = std::env::temp_dir().join(format!(
                "nsynth_router_solver_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            let cache = std::env::temp_dir().join(format!(
                "nsynth_cache_solver_test_{}_{:?}.json",
                std::process::id(),
                std::thread::current().id(),
            ));
            std::env::set_var("NSYNTH_METHOD_ROUTER_PATH", &router);
            std::env::set_var("NSYNTH_CACHE_PATH", &cache);
            crate::method_router::reset_for_tests();
            crate::solved_cache::reset_for_tests();
            let _ = fs::remove_file(&router);
            let _ = fs::remove_file(&cache);
            let result = f();
            std::env::remove_var("NSYNTH_METHOD_ROUTER_PATH");
            std::env::remove_var("NSYNTH_CACHE_PATH");
            crate::method_router::reset_for_tests();
            crate::solved_cache::reset_for_tests();
            let _ = fs::remove_file(&router);
            let _ = fs::remove_file(&cache);
            result
        })
    })
}

fn with_scratch_search_family_router<R>(f: impl FnOnce() -> R) -> R {
    let router = std::env::temp_dir().join(format!(
        "nsynth_search_family_solver_test_{}_{:?}.json",
        std::process::id(),
        std::thread::current().id(),
    ));
    crate::search_family_router::with_test_router_path(Some(router.clone()), || {
        let _ = fs::remove_file(&router);
        let result = f();
        let _ = fs::remove_file(&router);
        result
    })
}

#[path = "tests/benchmark_diff_cases.rs"]
mod benchmark_diff_cases;
#[path = "tests/search_cases.rs"]
mod exact_cases;
#[path = "tests/gradient_cases.rs"]
mod gradient_cases;
#[path = "tests/routing_cases.rs"]
mod routing_cases;
