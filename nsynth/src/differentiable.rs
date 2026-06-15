use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::benchmark::{generated_holdouts, Example, Problem, Value};
use crate::interactive::InteractiveTrace;
use crate::runtime::{
    execute_function_for_problem, verify_problem_code_strict, Value as RuntimeValue,
};
use crate::solver::solve_problem_search_only;

#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DifferentiableMetadata {
    #[serde(default)]
    pub ambiguity_count: usize,
    #[serde(default)]
    pub exact_alternatives: Vec<String>,
    #[serde(default)]
    pub recursive_refinement_applied: bool,
    #[serde(default)]
    pub recursive_refinement_resolved: bool,
    #[serde(default)]
    pub recursive_refinement_winner: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DifferentiableSolveResult {
    pub success: bool,
    pub code: String,
    pub method: String,
    pub error: Option<String>,
    pub metadata: DifferentiableMetadata,
}

#[derive(Clone, Serialize)]
struct BridgeExample {
    inputs: Vec<i64>,
    expected: i64,
}

#[derive(Serialize)]
struct BridgeRequest<'a> {
    signature: &'a str,
    function_name: &'a str,
    examples: Vec<BridgeExample>,
    steps: usize,
    num_restarts: usize,
    seed: u64,
}

#[derive(Clone, Serialize)]
struct BridgeInteractiveTrace {
    input_stream: Vec<i64>,
    expected_output: Vec<i64>,
}

#[derive(Serialize)]
struct InteractiveBridgeRequest<'a> {
    mode: &'a str,
    signature: &'a str,
    function_name: &'a str,
    interactive_traces: Vec<BridgeInteractiveTrace>,
    steps: usize,
    num_restarts: usize,
    seed: u64,
}

#[derive(Deserialize)]
struct BridgeResponse {
    supported: bool,
    success: bool,
    code: Option<String>,
    loss: Option<f64>,
    structure: Option<String>,
    error: Option<String>,
    #[serde(default)]
    metadata: Option<DifferentiableMetadata>,
}

fn repo_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path
}

fn bridge_script() -> PathBuf {
    repo_root().join("egdc").join("mog_gradient_bridge.py")
}

fn scalar_bridge_examples(examples: &[Example]) -> Option<Vec<BridgeExample>> {
    let mut out = Vec::with_capacity(examples.len());
    for example in examples {
        let mut inputs = Vec::with_capacity(example.inputs.len());
        for value in &example.inputs {
            match value {
                Value::Int(v) => inputs.push(*v),
                _ => return None,
            }
        }
        out.push(BridgeExample {
            inputs,
            expected: example.expected_int(),
        });
    }
    Some(out)
}

fn scalar_examples(problem: &Problem) -> Option<Vec<BridgeExample>> {
    scalar_bridge_examples(&problem.examples)
}

fn scalar_counterexamples(problem: &Problem) -> Option<Vec<BridgeExample>> {
    scalar_bridge_examples(&generated_holdouts(problem))
}

fn is_small_output_scalar_problem(problem: &Problem) -> bool {
    let Some(_) = scalar_examples(problem) else {
        return false;
    };
    let mut outputs = BTreeSet::new();
    for example in &problem.examples {
        outputs.insert(example.expected_int());
    }
    outputs.len() <= 3 && problem.examples.len() >= 4
}

fn dedupe_bridge_examples(examples: Vec<BridgeExample>) -> Vec<BridgeExample> {
    let mut seen = BTreeSet::new();
    let mut deduped = Vec::new();
    for example in examples {
        if seen.insert(example.inputs.clone()) {
            deduped.push(example);
        }
    }
    deduped
}

fn scalar_seed_inputs(problem: &Problem) -> Option<Vec<Vec<i64>>> {
    let mut seeds = Vec::new();
    for example in &problem.examples {
        let mut row = Vec::with_capacity(example.inputs.len());
        for value in &example.inputs {
            let Value::Int(v) = value else {
                return None;
            };
            row.push(*v);
        }
        seeds.push(row);
    }
    for example in generated_holdouts(problem) {
        let mut row = Vec::with_capacity(example.inputs.len());
        for value in &example.inputs {
            let Value::Int(v) = value else {
                return None;
            };
            row.push(*v);
        }
        seeds.push(row);
    }
    Some(seeds)
}

fn candidate_values(value: i64) -> Vec<i64> {
    let mut values = vec![value, value - 1, value + 1, -value, 0, 1, -1];
    values.sort_unstable();
    values.dedup();
    values
}

fn failure_focus_inputs(error: &str) -> Option<Vec<i64>> {
    let start = error.find("inputs [")?;
    let rest = &error[start + "inputs [".len()..];
    let end = rest.find(']')?;
    let segment = &rest[..end];
    let mut values = Vec::new();
    let mut cursor = segment;
    while let Some(pos) = cursor.find("Int(") {
        let tail = &cursor[pos + 4..];
        let close = tail.find(')')?;
        let value = tail[..close].trim().parse::<i64>().ok()?;
        values.push(value);
        cursor = &tail[close + 1..];
    }
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn mismatched_scalar_inputs(problem: &Problem, code: &str) -> Option<Vec<Vec<i64>>> {
    let mut failing = Vec::new();
    for example in problem
        .examples
        .iter()
        .chain(generated_holdouts(problem).iter())
    {
        let mut inputs = Vec::with_capacity(example.inputs.len());
        for value in &example.inputs {
            let Value::Int(v) = value else {
                return None;
            };
            inputs.push(*v);
        }
        let actual =
            execute_function_for_problem(code, problem.function_name(), &example.inputs, problem)
                .ok()?;
        let RuntimeValue::Int(actual) = actual else {
            return None;
        };
        if actual != example.expected_int() {
            failing.push(inputs);
        }
    }

    if failing.is_empty() {
        None
    } else {
        Some(failing)
    }
}

fn teacher_examples_from_code(
    problem: &Problem,
    teacher_code: &str,
    focus_inputs: Option<&[Vec<i64>]>,
    last_error: Option<&str>,
) -> Option<Vec<BridgeExample>> {
    if teacher_code.trim().is_empty() {
        return None;
    }
    if verify_problem_code_strict(problem, teacher_code).is_err() {
        return None;
    }

    let mut seed_inputs = scalar_seed_inputs(problem)?;
    if let Some(focus_rows) = focus_inputs {
        for focus in focus_rows.iter().rev() {
            seed_inputs.insert(0, focus.clone());
        }
    } else if let Some(focus) = last_error.and_then(failure_focus_inputs) {
        seed_inputs.insert(0, focus);
    }

    let mut seen = BTreeSet::new();
    let mut candidate_inputs = Vec::new();
    for seed in seed_inputs {
        if seed.is_empty() || seed.len() > 3 {
            return None;
        }
        if seen.insert(seed.clone()) {
            candidate_inputs.push(seed.clone());
        }
        for idx in 0..seed.len() {
            for cand in candidate_values(seed[idx]) {
                let mut row = seed.clone();
                row[idx] = cand;
                if seen.insert(row.clone()) {
                    candidate_inputs.push(row);
                }
            }
        }
        if candidate_inputs.len() >= 20 {
            break;
        }
    }

    let mut out = Vec::new();
    for inputs in candidate_inputs.into_iter().take(20) {
        let values = inputs.iter().copied().map(Value::Int).collect::<Vec<_>>();
        let actual =
            execute_function_for_problem(teacher_code, problem.function_name(), &values, problem)
                .ok()?;
        let RuntimeValue::Int(expected) = actual else {
            return None;
        };
        out.push(BridgeExample { inputs, expected });
    }

    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn teacher_augmented_examples(
    problem: &Problem,
    focus_inputs: Option<&[Vec<i64>]>,
    last_error: Option<&str>,
) -> Option<Vec<BridgeExample>> {
    let teacher = solve_problem_search_only(problem);
    if !teacher.success {
        return None;
    }
    teacher_examples_from_code(problem, &teacher.code, focus_inputs, last_error)
}

fn unsupported_scalar_result() -> DifferentiableSolveResult {
    DifferentiableSolveResult {
        success: false,
        code: String::new(),
        method: "diff_gradient_unsupported".to_string(),
        error: Some(
            "differentiable solver currently supports scalar numeric problems only".to_string(),
        ),
        metadata: DifferentiableMetadata::default(),
    }
}

fn execute_bridge_payload(raw: &[u8]) -> Result<BridgeResponse, DifferentiableSolveResult> {
    let script = bridge_script();
    if !script.exists() {
        return Err(DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: "diff_gradient_unsupported".to_string(),
            error: Some(format!(
                "python bridge script not found at {}",
                script.display()
            )),
            metadata: DifferentiableMetadata::default(),
        });
    }

    let mut child = match Command::new("python3")
        .arg(&script)
        .current_dir(repo_root())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return Err(DifferentiableSolveResult {
                success: false,
                code: String::new(),
                method: "diff_gradient_error".to_string(),
                error: Some(format!("failed to launch python bridge: {err}")),
                metadata: DifferentiableMetadata::default(),
            });
        }
    };

    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(raw) {
            return Err(DifferentiableSolveResult {
                success: false,
                code: String::new(),
                method: "diff_gradient_error".to_string(),
                error: Some(format!("failed to write bridge request: {err}")),
                metadata: DifferentiableMetadata::default(),
            });
        }
    }

    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return Err(DifferentiableSolveResult {
                success: false,
                code: String::new(),
                method: "diff_gradient_error".to_string(),
                error: Some(format!("failed to read bridge response: {err}")),
                metadata: DifferentiableMetadata::default(),
            });
        }
    };

    if !output.status.success() {
        return Err(DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: "diff_gradient_error".to_string(),
            error: Some(format!(
                "python bridge exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )),
            metadata: DifferentiableMetadata::default(),
        });
    }

    match serde_json::from_slice(&output.stdout) {
        Ok(response) => Ok(response),
        Err(err) => Err(DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: "diff_gradient_error".to_string(),
            error: Some(format!(
                "failed to parse bridge response: {err}; stdout={}",
                String::from_utf8_lossy(&output.stdout)
            )),
            metadata: DifferentiableMetadata::default(),
        }),
    }
}

fn run_bridge_request(
    problem: &Problem,
    examples: &[BridgeExample],
    steps: usize,
    num_restarts: usize,
    seed: u64,
) -> DifferentiableSolveResult {
    let request = BridgeRequest {
        signature: problem.signature,
        function_name: problem.function_name(),
        examples: examples.to_vec(),
        steps,
        num_restarts,
        seed,
    };

    let raw = match serde_json::to_vec(&request) {
        Ok(raw) => raw,
        Err(err) => {
            return DifferentiableSolveResult {
                success: false,
                code: String::new(),
                method: "diff_gradient_error".to_string(),
                error: Some(format!("failed to encode bridge request: {err}")),
                metadata: DifferentiableMetadata::default(),
            };
        }
    };

    let response = match execute_bridge_payload(&raw) {
        Ok(response) => response,
        Err(result) => return result,
    };
    let metadata = response.metadata.clone().unwrap_or_default();

    if !response.supported {
        return DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: "diff_gradient_unsupported".to_string(),
            error: response.error,
            metadata,
        };
    }

    if !response.success {
        return DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: format!(
                "diff_gradient_{}",
                response.structure.unwrap_or_else(|| "failed".to_string())
            ),
            error: response.error,
            metadata,
        };
    }

    let code = response.code.unwrap_or_default();
    if let Err(err) = verify_problem_code_strict(problem, &code) {
        return DifferentiableSolveResult {
            success: false,
            code,
            method: format!(
                "diff_gradient_{}",
                response
                    .structure
                    .unwrap_or_else(|| "unverified".to_string())
            ),
            error: Some(format!(
                "differentiable synthesis failed strict verification{}: {err}",
                response
                    .loss
                    .map(|loss| format!(" (loss={loss})"))
                    .unwrap_or_default()
            )),
            metadata,
        };
    }

    DifferentiableSolveResult {
        success: true,
        code,
        method: format!(
            "diff_gradient_{}",
            response.structure.unwrap_or_else(|| "unknown".to_string())
        ),
        error: None,
        metadata,
    }
}

fn run_interactive_bridge_request(
    name: &str,
    traces: &[InteractiveTrace],
    steps: usize,
    num_restarts: usize,
    seed: u64,
) -> DifferentiableSolveResult {
    let request = InteractiveBridgeRequest {
        mode: "interactive",
        signature: "fn main() -> i64",
        function_name: name,
        interactive_traces: traces
            .iter()
            .map(|trace| BridgeInteractiveTrace {
                input_stream: trace.input_stream.clone(),
                expected_output: trace.expected_output.clone(),
            })
            .collect(),
        steps,
        num_restarts,
        seed,
    };

    let raw = match serde_json::to_vec(&request) {
        Ok(raw) => raw,
        Err(err) => {
            return DifferentiableSolveResult {
                success: false,
                code: String::new(),
                method: "diff_gradient_error".to_string(),
                error: Some(format!(
                    "failed to encode interactive bridge request: {err}"
                )),
                metadata: DifferentiableMetadata::default(),
            };
        }
    };

    let response = match execute_bridge_payload(&raw) {
        Ok(response) => response,
        Err(result) => return result,
    };
    let metadata = response.metadata.clone().unwrap_or_default();

    if !response.supported {
        return DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: "diff_gradient_unsupported".to_string(),
            error: response.error,
            metadata,
        };
    }

    if !response.success {
        return DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: format!(
                "diff_gradient_{}",
                response.structure.unwrap_or_else(|| "failed".to_string())
            ),
            error: response.error,
            metadata,
        };
    }

    DifferentiableSolveResult {
        success: true,
        code: response.code.unwrap_or_default(),
        method: format!(
            "diff_gradient_{}",
            response.structure.unwrap_or_else(|| "unknown".to_string())
        ),
        error: None,
        metadata,
    }
}

fn solve_problem_differentiable_with_schedule(
    problem: &Problem,
    schedule: &[(usize, usize, u64)],
    refinement_schedule: &[(usize, usize, u64)],
) -> DifferentiableSolveResult {
    let Some(examples) = scalar_examples(problem) else {
        return unsupported_scalar_result();
    };
    let counterexamples = scalar_counterexamples(problem);
    let prefers_direct_distillation = is_small_output_scalar_problem(problem);

    let mut last_result = DifferentiableSolveResult {
        success: false,
        code: String::new(),
        method: "diff_gradient_failed".to_string(),
        error: Some("no differentiable schedule was attempted".to_string()),
        metadata: DifferentiableMetadata::default(),
    };

    for &(steps, num_restarts, seed) in schedule {
        let result = run_bridge_request(problem, &examples, steps, num_restarts, seed);
        if result.success || result.method == "diff_gradient_unsupported" {
            return result;
        }
        last_result = result;
    }

    if !prefers_direct_distillation {
        if let Some(counterexamples) = counterexamples.filter(|items| !items.is_empty()) {
            let should_refine = last_result
                .error
                .as_deref()
                .map(|err| err.contains("holdout failed") || err.contains("strict verification"))
                .unwrap_or(false);

            if should_refine {
                let mut refined_examples = examples.clone();
                refined_examples.extend(counterexamples);
                let refined_examples = dedupe_bridge_examples(refined_examples);
                for &(steps, num_restarts, seed) in refinement_schedule {
                    let result =
                        run_bridge_request(problem, &refined_examples, steps, num_restarts, seed);
                    if result.success || result.method == "diff_gradient_unsupported" {
                        return result;
                    }
                    last_result = result;
                }
            }
        }
    }

    let focus_inputs = if last_result.code.is_empty() {
        None
    } else {
        mismatched_scalar_inputs(problem, &last_result.code)
    };

    if let Some(teacher_examples) = teacher_augmented_examples(
        problem,
        focus_inputs.as_deref(),
        last_result.error.as_deref(),
    ) {
        let should_distill = last_result
            .error
            .as_deref()
            .map(|err| {
                err.contains("stdout mismatch")
                    || err.contains("holdout failed")
                    || err.contains("strict verification")
            })
            .unwrap_or(false);

        if should_distill {
            let mut distilled_examples = examples.clone();
            distilled_examples.extend(teacher_examples);
            let distilled_examples = dedupe_bridge_examples(distilled_examples);
            for &(steps, num_restarts, seed) in &[(240, 1, 62_u64), (420, 1, 63_u64)] {
                let result =
                    run_bridge_request(problem, &distilled_examples, steps, num_restarts, seed);
                if result.success || result.method == "diff_gradient_unsupported" {
                    return result;
                }
                last_result = result;
            }
        }
    }

    if last_result.error.is_none() {
        last_result.error = Some(
            "differentiable schedules did not converge to a strictly verified program".to_string(),
        );
    }
    last_result
}

pub fn solve_problem_differentiable_fast_probe(problem: &Problem) -> DifferentiableSolveResult {
    solve_problem_differentiable_with_schedule(problem, &[(120, 1, 42)], &[(180, 1, 52)])
}

pub fn solve_problem_differentiable_from_teacher(
    problem: &Problem,
    teacher_code: &str,
) -> DifferentiableSolveResult {
    let Some(examples) = scalar_examples(problem) else {
        return unsupported_scalar_result();
    };
    let Some(teacher_examples) = teacher_examples_from_code(problem, teacher_code, None, None)
    else {
        return DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: "diff_gradient_teacher_failed".to_string(),
            error: Some(
                "teacher code could not be verified into scalar distillation examples".to_string(),
            ),
            metadata: DifferentiableMetadata::default(),
        };
    };

    let mut distilled_examples = examples.clone();
    distilled_examples.extend(teacher_examples);
    let mut distilled_examples = dedupe_bridge_examples(distilled_examples);

    let mut last_result = DifferentiableSolveResult {
        success: false,
        code: String::new(),
        method: "diff_gradient_teacher_failed".to_string(),
        error: Some(
            "reference distillation schedule did not converge to a verified program".to_string(),
        ),
        metadata: DifferentiableMetadata::default(),
    };

    for &(steps, num_restarts, seed) in &[(240, 1, 72_u64), (420, 2, 73_u64)] {
        let result = run_bridge_request(problem, &distilled_examples, steps, num_restarts, seed);
        if result.success || result.method == "diff_gradient_unsupported" {
            return result;
        }
        last_result = result;
    }

    let focus_inputs = if last_result.code.is_empty() {
        None
    } else {
        mismatched_scalar_inputs(problem, &last_result.code)
    };

    if let Some(extra_teacher) = teacher_examples_from_code(
        problem,
        teacher_code,
        focus_inputs.as_deref(),
        last_result.error.as_deref(),
    ) {
        distilled_examples.extend(extra_teacher);
        distilled_examples = dedupe_bridge_examples(distilled_examples);
        for &(steps, num_restarts, seed) in &[(480, 2, 74_u64), (900, 3, 75_u64)] {
            let result =
                run_bridge_request(problem, &distilled_examples, steps, num_restarts, seed);
            if result.success || result.method == "diff_gradient_unsupported" {
                return result;
            }
            last_result = result;
        }
    }

    if last_result.error.is_none() {
        last_result.error = Some(
            "reference distillation schedules did not converge to a strictly verified program"
                .to_string(),
        );
    }
    last_result
}

pub fn solve_problem_differentiable_only(problem: &Problem) -> DifferentiableSolveResult {
    if is_small_output_scalar_problem(problem) {
        return solve_problem_differentiable_with_schedule(problem, &[(240, 1, 42)], &[]);
    }

    solve_problem_differentiable_with_schedule(
        problem,
        &[(300, 1, 42), (600, 2, 43)],
        &[(450, 2, 52), (900, 3, 53)],
    )
}

pub fn solve_interactive_traces_differentiable_only(
    name: &str,
    traces: &[InteractiveTrace],
) -> DifferentiableSolveResult {
    if traces.is_empty() {
        return DifferentiableSolveResult {
            success: false,
            code: String::new(),
            method: "diff_gradient_unsupported".to_string(),
            error: Some(
                "interactive differentiable synthesis requires at least one trace".to_string(),
            ),
            metadata: DifferentiableMetadata::default(),
        };
    }

    let mut last_result = DifferentiableSolveResult {
        success: false,
        code: String::new(),
        method: "diff_gradient_failed".to_string(),
        error: Some("no interactive differentiable schedule was attempted".to_string()),
        metadata: DifferentiableMetadata::default(),
    };

    for &(steps, num_restarts, seed) in &[(64_usize, 1_usize, 42_u64), (160, 2, 43), (280, 4, 44)] {
        let result = run_interactive_bridge_request(name, traces, steps, num_restarts, seed);
        if result.success || result.method == "diff_gradient_unsupported" {
            return result;
        }
        last_result = result;
    }

    last_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_examples_from_code_expand_reference_behavior() {
        let problem = Problem {
            name: "abs_diff_custom_v0".to_string(),
            category: "test",
            description: "Return the absolute difference between a and b.",
            signature: "fn abs_diff_custom(a: i64, b: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(5), Value::Int(3)],
                    expected: Value::Int(2),
                },
                Example {
                    inputs: vec![Value::Int(3), Value::Int(5)],
                    expected: Value::Int(2),
                },
                Example {
                    inputs: vec![Value::Int(7), Value::Int(7)],
                    expected: Value::Int(0),
                },
                Example {
                    inputs: vec![Value::Int(-2), Value::Int(5)],
                    expected: Value::Int(7),
                },
            ],
            holdouts: vec![],
            reference_code: "",

        synthetic_args: Vec::new(),

        synthetic_values: Vec::new(),

        recursive_allowed: false,

        tree_input: false,

        explicit_stack: false,

        };
        let teacher_code = "fn abs_diff_custom(a: i64, b: i64) -> i64 {\n    if a >= b { return a - b; }\n    return b - a;\n}\n";
        let examples = teacher_examples_from_code(&problem, teacher_code, None, None)
            .expect("teacher code should produce distillation examples");
        assert!(examples
            .iter()
            .any(|example| example.inputs == vec![5, 4] && example.expected == 1));
        assert!(examples
            .iter()
            .any(|example| example.inputs == vec![4, 3] && example.expected == 1));
    }
}
