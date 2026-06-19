//! Executor-driven compositional synthesis.
//!
//! A [`Problem`] may declare multiple helper [`FunctionDef`]s, each carrying its
//! own input/output examples. This module synthesizes each declared function
//! *independently* via the real solver, orchestrated through the asynchronous
//! [`Executor`] (which provides dependency ordering, bounded parallelism, retry,
//! and progress tracking), then composes the synthesized functions into one
//! program.
//!
//! This is genuine divide-and-conquer and it is non-fabricating: every function
//! body originates from `solve_problem`. The composition layer only orders and
//! concatenates results — it never invents code. When a `Problem` declares no
//! functions, callers should use the single-shot path instead; this module
//! requires at least one declared function.

use crate::agent::executor::{Executor, Plan, Task};
use crate::benchmark::{FunctionDef, Problem};
use crate::solver::{solve_problem, SolveResult, SolverError};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Build a standalone sub-[`Problem`] for a single declared function. The
/// function's own examples become the synthesis target; problem-level recursion
/// and tree flags are inherited so structurally recursive helpers remain
/// solvable.
fn subproblem_for(func: &FunctionDef, parent: &Problem) -> Problem {
    // Solver requires a `&'static str` signature; leak the per-function
    // signature. The leak is bounded by the number of declared functions.
    let signature: &'static str = Box::leak(func.signature.clone().into_boxed_str());
    Problem {
        name: func.name.clone(),
        category: "agentic_subtask",
        description: "",
        signature,
        examples: func.examples.clone(),
        holdouts: Vec::new(),
        reference_code: "",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: parent.recursive_allowed,
        tree_input: parent.tree_input,
        explicit_stack: parent.explicit_stack,
        functions: Vec::new(),
    }
}

fn failure(method: &str, msg: impl Into<String>) -> SolveResult {
    SolveResult {
        success: false,
        code: String::new(),
        method: method.to_string(),
        error: Some(msg.into()),
        metadata: Default::default(),
    }
}

/// Synthesize a multi-function problem by solving each declared function via the
/// real solver, driven by the asynchronous [`Executor`], then composing the
/// results. Safe to call from any context (sync CLI or async): the async work
/// runs on a dedicated thread with its own runtime, avoiding nested-runtime
/// panics.
pub fn solve_compositional(problem: &Problem) -> SolveResult {
    if problem.functions.is_empty() {
        return failure(
            "agentic_compositional",
            "problem declares no functions; use the single-shot path",
        );
    }

    // Clone the data the async workload needs so the dedicated thread owns it.
    let functions = problem.functions.clone();
    let parent = problem.clone();

    let handle = std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => return failure("agentic_compositional", format!("runtime build: {e}")),
        };
        runtime.block_on(run_compositional(functions, parent))
    });

    handle
        .join()
        .unwrap_or_else(|_| failure("agentic_compositional", "compositional worker panicked"))
}

async fn run_compositional(functions: Vec<FunctionDef>, parent: Problem) -> SolveResult {
    // Pre-build each function's sub-problem, keyed by function name.
    let mut subproblems: HashMap<String, Problem> = HashMap::new();
    for func in &functions {
        subproblems.insert(func.name.clone(), subproblem_for(func, &parent));
    }
    let subproblems = Arc::new(subproblems);

    // Collected synthesis results, keyed by function name.
    let results: Arc<Mutex<HashMap<String, SolveResult>>> = Arc::new(Mutex::new(HashMap::new()));

    // One executor task per declared function. Functions are independent (no
    // declared inter-function dependencies), so all are parallel-safe. Synthesis
    // is deterministic, so retries would not help: max_retries = 0.
    let mut plan = Plan::new("compositional", parent.name.clone());
    for func in &functions {
        plan = plan.with_task(
            Task::new(func.name.clone(), func.name.clone())
                .with_parallel_safe(true)
                .with_max_retries(0),
        );
    }

    let executor = Executor::new();
    let subs = subproblems.clone();
    let res = results.clone();
    executor
        .with_executor(move |task: &Task| {
            let id = task.id.as_str().to_string();
            let subs = subs.clone();
            let res = res.clone();
            async move {
                let sub = subs.get(&id).ok_or_else(|| {
                    SolverError::ConfigurationError(format!("no sub-problem for '{id}'"))
                })?;
                // Real synthesis happens here, once per function.
                let solved = solve_problem(sub);
                let ok = solved.success;
                let code = solved.code.clone();
                res.lock().await.insert(id.clone(), solved);
                if ok {
                    Ok(code)
                } else {
                    Err(SolverError::NoSolutionFound(format!(
                        "no solution for function '{id}'"
                    )))
                }
            }
        })
        .await;

    if let Err(e) = executor.execute(&plan).await {
        return failure(
            "agentic_compositional",
            format!("compositional execution failed: {e}"),
        );
    }

    // Compose: helpers (entry_point == false) first, entry point(s) last, so the
    // generated source reads top-down with dependencies declared before use.
    let map = results.lock().await;
    let mut ordered: Vec<&FunctionDef> = functions.iter().collect();
    ordered.sort_by_key(|f| f.entry_point);

    let mut bodies: Vec<String> = Vec::new();
    let mut methods: Vec<String> = Vec::new();
    for func in ordered {
        match map.get(&func.name) {
            Some(result) if result.success => {
                bodies.push(result.code.clone());
                methods.push(result.method.clone());
            }
            _ => {
                return failure(
                    "agentic_compositional",
                    format!("function '{}' was not synthesized", func.name),
                );
            }
        }
    }

    SolveResult {
        success: true,
        code: bodies.join("\n\n"),
        method: format!("agentic_compositional[{}]", methods.join("+")),
        error: None,
        metadata: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn func(name: &str, sig: &str, examples: Vec<Example>, entry: bool) -> FunctionDef {
        FunctionDef {
            name: name.to_string(),
            signature: sig.to_string(),
            examples,
            entry_point: entry,
        }
    }

    fn ex(a: i64, b: i64, out: i64) -> Example {
        Example {
            inputs: vec![Value::Int(a), Value::Int(b)],
            expected: Value::Int(out),
        }
    }

    fn multi_function_problem() -> Problem {
        let add = func(
            "add_two",
            "fn add_two(a: i64, b: i64) -> i64",
            vec![ex(1, 2, 3), ex(4, 5, 9), ex(10, 20, 30), ex(0, 0, 0)],
            false,
        );
        let mul = func(
            "mul_two",
            "fn mul_two(a: i64, b: i64) -> i64",
            vec![ex(2, 3, 6), ex(4, 5, 20), ex(6, 7, 42), ex(1, 1, 1)],
            true,
        );
        Problem {
            name: "compose_add_mul".to_string(),
            category: "test",
            description: "",
            signature: "fn compose_add_mul(a: i64, b: i64) -> i64",
            examples: Vec::new(),
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![add, mul],
        }
    }

    #[test]
    fn test_empty_functions_rejected() {
        let mut p = multi_function_problem();
        p.functions.clear();
        let result = solve_compositional(&p);
        assert!(!result.success);
        assert!(result.error.unwrap().contains("no functions"));
    }

    #[test]
    fn test_compositional_synthesizes_all_functions() {
        let problem = multi_function_problem();
        let result = solve_compositional(&problem);
        assert!(
            result.success,
            "compositional synthesis failed: {:?}",
            result.error
        );
        // Both declared functions must appear in the composed output.
        assert!(result.code.contains("add_two"), "missing add_two: {}", result.code);
        assert!(result.code.contains("mul_two"), "missing mul_two: {}", result.code);
        // Method tag records the compositional path.
        assert!(result.method.starts_with("agentic_compositional"));
    }

    #[test]
    fn test_helper_ordered_before_entry() {
        let problem = multi_function_problem();
        let result = solve_compositional(&problem);
        assert!(result.success);
        // add_two is a helper (entry_point=false), mul_two is the entry point;
        // the helper must be emitted first.
        let add_pos = result.code.find("add_two").unwrap();
        let mul_pos = result.code.find("mul_two").unwrap();
        assert!(add_pos < mul_pos, "helper should precede entry point");
    }
}
