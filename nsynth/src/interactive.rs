use crate::benchmark::{Example, Problem, Value};
use crate::differentiable::{
    solve_interactive_traces_differentiable_only as solve_native_interactive_traces_differentiable_only,
    DifferentiableMetadata,
};
use crate::runtime::execute_program_with_input;
use crate::solver::{solve_problem_differentiable_only, SolveResult};

const INTERACTIVE_UNARY_SIGNATURE: &str = "fn interactive_unary_step(x: i64) -> i64";
const INTERACTIVE_BINARY_SIGNATURE: &str = "fn interactive_binary_step(a: i64, b: i64) -> i64";
const INTERACTIVE_TERNARY_SIGNATURE: &str =
    "fn interactive_ternary_step(a: i64, b: i64, c: i64) -> i64";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InteractiveTrace {
    pub input_stream: Vec<i64>,
    pub expected_output: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InteractiveProblem {
    pub name: String,
    pub base_problem: Problem,
    pub traces: Vec<InteractiveTrace>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InteractiveSolveResult {
    pub success: bool,
    pub code: String,
    pub method: String,
    pub error: Option<String>,
    pub metadata: DifferentiableMetadata,
}

fn int_inputs(problem: &Problem) -> Option<Vec<(Vec<i64>, i64)>> {
    let mut cases = Vec::with_capacity(problem.examples.len());
    for example in &problem.examples {
        let mut inputs = Vec::with_capacity(example.inputs.len());
        for value in &example.inputs {
            let Value::Int(v) = value else {
                return None;
            };
            inputs.push(*v);
        }
        cases.push((inputs, example.expected));
    }
    Some(cases)
}

fn parse_param_names(signature: &str) -> Vec<String> {
    let params = signature
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(params, _)| params)
        .unwrap_or("")
        .trim();

    if params.is_empty() {
        return Vec::new();
    }

    params
        .split(',')
        .map(|param| {
            param
                .split_once(':')
                .map(|(name, _)| name.trim().to_string())
                .unwrap_or_default()
        })
        .collect()
}

fn build_trace_from_cases(cases: &[(Vec<i64>, i64)], indices: &[usize]) -> InteractiveTrace {
    let mut input_stream = Vec::new();
    let mut expected_output = Vec::new();
    for &index in indices {
        input_stream.extend_from_slice(&cases[index].0);
        expected_output.push(cases[index].1);
    }
    InteractiveTrace {
        input_stream,
        expected_output,
    }
}

fn infer_group_arity(traces: &[InteractiveTrace]) -> Option<usize> {
    let mut arity = None;
    for trace in traces {
        if trace.expected_output.is_empty() || trace.input_stream.is_empty() {
            return None;
        }
        if trace.input_stream.len() % trace.expected_output.len() != 0 {
            return None;
        }
        let this_arity = trace.input_stream.len() / trace.expected_output.len();
        if this_arity == 0 || this_arity > 3 {
            return None;
        }
        match arity {
            Some(existing) if existing != this_arity => return None,
            None => arity = Some(this_arity),
            _ => {}
        }
    }
    arity
}

fn grouped_signature(arity: usize) -> Option<&'static str> {
    match arity {
        1 => Some(INTERACTIVE_UNARY_SIGNATURE),
        2 => Some(INTERACTIVE_BINARY_SIGNATURE),
        3 => Some(INTERACTIVE_TERNARY_SIGNATURE),
        _ => None,
    }
}

fn grouped_examples_from_traces(traces: &[InteractiveTrace], arity: usize) -> Option<Vec<Example>> {
    let mut examples = Vec::new();
    for trace in traces {
        if trace.expected_output.is_empty()
            || trace.input_stream.len() != trace.expected_output.len() * arity
        {
            return None;
        }
        for (idx, expected) in trace.expected_output.iter().enumerate() {
            let start = idx * arity;
            let inputs = trace.input_stream[start..start + arity]
                .iter()
                .copied()
                .map(Value::Int)
                .collect();
            examples.push(Example {
                inputs,
                expected: *expected,
            });
        }
    }
    Some(examples)
}

fn grouped_interactive_problem(
    name: &str,
    traces: &[InteractiveTrace],
) -> Result<InteractiveProblem, String> {
    let arity = infer_group_arity(traces).ok_or_else(|| {
        format!(
            "interactive grouped lifting requires a consistent 1:1, 2:1, or 3:1 input/output ratio for {}",
            name
        )
    })?;
    let signature = grouped_signature(arity)
        .ok_or_else(|| format!("interactive grouped lifting does not support arity {arity}"))?;
    let examples = grouped_examples_from_traces(traces, arity).ok_or_else(|| {
        format!(
            "interactive grouped lifting could not form scalar examples for {}",
            name
        )
    })?;
    Ok(InteractiveProblem {
        name: name.to_string(),
        base_problem: Problem {
            name: format!("{name}_scalarized"),
            category: "interactive",
            description: "Scalar step problem distilled from interactive traces",
            signature,
            examples,
            holdouts: vec![],
            reference_code: "",
        },
        traces: traces.to_vec(),
    })
}

pub fn lift_problem_to_interactive(problem: &Problem) -> Result<InteractiveProblem, String> {
    let cases = int_inputs(problem).ok_or_else(|| {
        format!(
            "interactive lifting currently supports only scalar i64 problems, got {}",
            problem.signature
        )
    })?;
    if cases.is_empty() {
        return Err(format!("problem {} has no examples", problem.name));
    }

    let arity = cases[0].0.len();
    if arity == 0 {
        return Err(format!(
            "interactive lifting requires at least one parameter for {}",
            problem.name
        ));
    }
    if cases.iter().any(|(inputs, _)| inputs.len() != arity) {
        return Err(format!(
            "interactive lifting requires a fixed scalar arity for {}",
            problem.name
        ));
    }

    let mut traces = Vec::new();
    let original = (0..cases.len()).collect::<Vec<_>>();
    traces.push(build_trace_from_cases(&cases, &original));

    if cases.len() > 1 {
        let mut reversed = original.clone();
        reversed.reverse();
        traces.push(build_trace_from_cases(&cases, &reversed));

        let mut rotated = original.clone();
        rotated.rotate_left(1);
        if rotated != original && rotated != reversed {
            traces.push(build_trace_from_cases(&cases, &rotated));
        }

        let mut repeated = original.clone();
        repeated.extend_from_slice(&original);
        traces.push(build_trace_from_cases(&cases, &repeated));
    }

    Ok(InteractiveProblem {
        name: format!("interactive_{}", problem.name),
        base_problem: problem.clone(),
        traces,
    })
}

pub fn build_interactive_wrapper(problem: &Problem) -> Result<String, String> {
    let fn_name = problem.function_name();
    let params = parse_param_names(problem.signature);
    if params.is_empty() {
        return Err(format!(
            "interactive wrapper requires at least one parameter for {}",
            problem.name
        ));
    }

    let mut lines = vec![
        "fn main() -> i64 {".to_string(),
        "    while has_input() == 1 {".to_string(),
    ];
    for (idx, param) in params.iter().enumerate() {
        if idx > 0 {
            lines.push("        if has_input() != 1 {".to_string());
            lines.push("            break;".to_string());
            lines.push("        }".to_string());
        }
        lines.push(format!("        {param} := read_i64();"));
    }
    lines.push(format!(
        "        println_i64({fn_name}({}));",
        params.join(", ")
    ));
    lines.push("    }".to_string());
    lines.push("    return 0;".to_string());
    lines.push("}".to_string());
    Ok(lines.join("\n"))
}

pub fn verify_interactive_traces(
    name: &str,
    traces: &[InteractiveTrace],
    code: &str,
) -> Result<(), String> {
    for trace in traces {
        let input = trace
            .input_stream
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        let actual = execute_program_with_input(code, input)?;
        let expected = trace
            .expected_output
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        if actual.output != expected {
            return Err(format!(
                "interactive trace mismatch for {}: expected {:?}, got {:?}",
                name, expected, actual.output
            ));
        }
    }
    Ok(())
}

pub fn verify_interactive_program(problem: &InteractiveProblem, code: &str) -> Result<(), String> {
    verify_interactive_traces(&problem.name, &problem.traces, code)
}

fn wrap_solver_result(
    interactive_problem: &InteractiveProblem,
    base_result: SolveResult,
) -> InteractiveSolveResult {
    if !base_result.success {
        return InteractiveSolveResult {
            success: false,
            code: String::new(),
            method: format!("interactive_{}", base_result.method),
            error: base_result.error,
            metadata: DifferentiableMetadata::default(),
        };
    }

    let wrapper = match build_interactive_wrapper(&interactive_problem.base_problem) {
        Ok(wrapper) => wrapper,
        Err(err) => {
            return InteractiveSolveResult {
                success: false,
                code: String::new(),
                method: format!("interactive_{}", base_result.method),
                error: Some(err),
                metadata: DifferentiableMetadata::default(),
            };
        }
    };

    let code = format!("{}\n\n{}\n", base_result.code.trim_end(), wrapper);
    if let Err(err) = verify_interactive_program(interactive_problem, &code) {
        return InteractiveSolveResult {
            success: false,
            code,
            method: format!("interactive_{}", base_result.method),
            error: Some(err),
            metadata: DifferentiableMetadata::default(),
        };
    }

    InteractiveSolveResult {
        success: true,
        code,
        method: format!("interactive_{}", base_result.method),
        error: None,
        metadata: DifferentiableMetadata::default(),
    }
}

pub fn solve_interactive_problem(problem: &Problem) -> InteractiveSolveResult {
    solve_interactive_problem_differentiable_only(problem)
}

pub fn solve_interactive_traces_differentiable_only(
    name: &str,
    traces: &[InteractiveTrace],
) -> InteractiveSolveResult {
    let mut errors = Vec::new();

    let native = solve_native_interactive_traces_differentiable_only(name, traces);
    let native_metadata = native.metadata.clone();
    if native.success {
        if let Err(err) = verify_interactive_traces(name, traces, &native.code) {
            errors.push(format!("native interactive verification failed: {err}"));
        } else {
            return InteractiveSolveResult {
                success: true,
                code: native.code,
                method: format!("interactive_{}", native.method),
                error: None,
                metadata: native_metadata,
            };
        }
    } else if let Some(err) = native.error {
        errors.push(format!("native interactive failed: {err}"));
    }

    if let Ok(grouped_problem) = grouped_interactive_problem(name, traces) {
        let grouped = wrap_solver_result(
            &grouped_problem,
            solve_problem_differentiable_only(&grouped_problem.base_problem),
        );
        if grouped.success {
            return grouped;
        }
        if let Some(err) = grouped.error {
            errors.push(format!("grouped scalar interactive failed: {err}"));
        }
    }

    InteractiveSolveResult {
        success: false,
        code: String::new(),
        method: "interactive_diff_gradient_failed".to_string(),
        error: Some(if errors.is_empty() {
            "interactive differentiable synthesis failed".to_string()
        } else {
            errors.join("; ")
        }),
        metadata: native_metadata,
    }
}

pub fn solve_interactive_problem_differentiable_only(problem: &Problem) -> InteractiveSolveResult {
    let interactive_problem = match lift_problem_to_interactive(problem) {
        Ok(problem) => problem,
        Err(err) => {
            return InteractiveSolveResult {
                success: false,
                code: String::new(),
                method: "interactive_unsupported".to_string(),
                error: Some(err),
                metadata: DifferentiableMetadata::default(),
            };
        }
    };

    let native = solve_interactive_traces_differentiable_only(
        &interactive_problem.name,
        &interactive_problem.traces,
    );
    if native.success {
        return native;
    }

    let mut wrapped = wrap_solver_result(
        &interactive_problem,
        solve_problem_differentiable_only(problem),
    );
    if wrapped.success {
        return wrapped;
    }

    wrapped.error = Some(match (native.error, wrapped.error.take()) {
        (Some(native_err), Some(wrapped_err)) => {
            format!("native interactive failed: {native_err}; scalar wrapper failed: {wrapped_err}")
        }
        (Some(native_err), None) => format!("native interactive failed: {native_err}"),
        (None, Some(wrapped_err)) => wrapped_err,
        (None, None) => "interactive differentiable synthesis failed".to_string(),
    });
    wrapped.metadata = native.metadata;
    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::get_benchmark;

    #[test]
    fn lifts_add_two_to_interactive_traces() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();
        let interactive = lift_problem_to_interactive(&problem).unwrap();
        assert_eq!(interactive.traces.len(), 4);
        assert!(interactive
            .traces
            .iter()
            .take(3)
            .all(|trace| trace.expected_output.len() == problem.examples.len()));
        assert!(interactive
            .traces
            .iter()
            .any(|trace| trace.expected_output.len() == problem.examples.len() * 2));
    }

    #[test]
    fn solves_native_interactive_running_sum_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![1, 2, 3],
                expected_output: vec![1, 3, 6],
            },
            InteractiveTrace {
                input_stream: vec![5, -2, 4],
                expected_output: vec![5, 3, 7],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_running_sum_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_state_add_input"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec!["4".to_string(), "1".to_string(), "-3".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "4\n5\n2");
    }

    #[test]
    fn solves_native_interactive_counter_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![9, 8, 7],
                expected_output: vec![1, 2, 3],
            },
            InteractiveTrace {
                input_stream: vec![5, 4],
                expected_output: vec![1, 2],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_counter_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_state_add_const"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "11".to_string(),
                "22".to_string(),
                "33".to_string(),
                "44".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "1\n2\n3\n4");
    }

    #[test]
    fn solves_native_interactive_passthrough_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![3, -2, 7],
                expected_output: vec![3, -2, 7],
            },
            InteractiveTrace {
                input_stream: vec![0, 5],
                expected_output: vec![0, 5],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_passthrough_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_passthrough"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec!["8".to_string(), "-1".to_string(), "0".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "8\n-1\n0");
    }

    #[test]
    fn solves_native_interactive_running_max_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![-5, -2, -7, 4],
                expected_output: vec![-5, -2, -2, 4],
            },
            InteractiveTrace {
                input_stream: vec![3, 1, 8],
                expected_output: vec![3, 3, 8],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_running_max_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_running_max"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec!["-9".to_string(), "-1".to_string(), "-4".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "-9\n-1\n-1");
    }

    #[test]
    fn solves_native_interactive_running_min_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![5, 7, 2, 9],
                expected_output: vec![5, 5, 2, 2],
            },
            InteractiveTrace {
                input_stream: vec![10, 9, 12],
                expected_output: vec![10, 9, 9],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_running_min_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_running_min"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec!["8".to_string(), "11".to_string(), "6".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "8\n8\n6");
    }

    #[test]
    fn solves_native_positive_running_sum_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![-2, 5, -1, 4, -10, 3],
                expected_output: vec![3, 2, 6],
            },
            InteractiveTrace {
                input_stream: vec![1, 1, -5, 10],
                expected_output: vec![1, 2, 7],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_positive_running_sum_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_state_emit_state_add_input_state_positive"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "-1".to_string(),
                "3".to_string(),
                "-2".to_string(),
                "5".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "2\n5");
    }

    #[test]
    fn solves_native_positive_input_counter_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![-1, 5, 0, 2],
                expected_output: vec![2, 4],
            },
            InteractiveTrace {
                input_stream: vec![7, -2, 9],
                expected_output: vec![1, 3],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_positive_input_counter_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_state_emit_state_add_const_input_positive"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "-3".to_string(),
                "4".to_string(),
                "6".to_string(),
                "-1".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "2\n3");
    }

    #[test]
    fn solves_native_running_average_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![2, 4, 6],
                expected_output: vec![2, 3, 4],
            },
            InteractiveTrace {
                input_stream: vec![5, 1, 3],
                expected_output: vec![5, 3, 3],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_running_average_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_two_register_accum_input_add_const_out_div"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec!["8".to_string(), "4".to_string(), "2".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "8\n6\n4");
    }

    #[test]
    fn solves_native_positive_running_average_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![2, -2, 4, 6, -4],
                expected_output: vec![2, 1, 2],
            },
            InteractiveTrace {
                input_stream: vec![5, 1, -2, 8],
                expected_output: vec![5, 3, 3],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_positive_running_average_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert!(
            result.method == "interactive_diff_gradient_interactive_two_register_accum_input_add_const_out_div_input_positive"
            || result.method == "interactive_diff_gradient_interactive_two_register_kadane_step_add_const_out_div_input_positive",
            "unexpected method: {}",
            result.method
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "3".to_string(),
                "3".to_string(),
                "-3".to_string(),
                "9".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "3\n3\n3");
    }

    #[test]
    fn solves_native_running_average_change_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![2, 2, 2, -3, -3],
                expected_output: vec![2, 0],
            },
            InteractiveTrace {
                input_stream: vec![5, 1, 1, 1, -8],
                expected_output: vec![5, 3, 2, 0],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_running_average_change_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert!(result
            .method
            .starts_with("interactive_diff_gradient_interactive_two_register_"));
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "6".to_string(),
                "0".to_string(),
                "0".to_string(),
                "-6".to_string(),
                "0".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "6\n3\n2\n0");
    }

    #[test]
    fn solves_native_running_average_crosses_threshold_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![2, 4, 8, 8],
                expected_output: vec![4],
            },
            InteractiveTrace {
                input_stream: vec![0, 8, 0, 8],
                expected_output: vec![4, 4],
            },
            InteractiveTrace {
                input_stream: vec![10, -4, 4, -2],
                expected_output: vec![],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_running_average_crosses_threshold_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_two_register_accum_input_add_const_out_div_output_crosses_above_threshold"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "1".to_string(),
                "7".to_string(),
                "1".to_string(),
                "7".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "4\n4");
    }

    #[test]
    fn solves_native_running_sum_exceeds_count_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![2, 0, 0, 0, 2],
                expected_output: vec![2],
            },
            InteractiveTrace {
                input_stream: vec![0, 0, 5, 0, 0, 5],
                expected_output: vec![5, 5, 10],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_running_sum_exceeds_count_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert!(result
            .method
            .starts_with("interactive_diff_gradient_interactive_two_register_"));
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "3".to_string(),
                "0".to_string(),
                "0".to_string(),
                "4".to_string(),
                "0".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "3\n3\n7\n7");
    }

    #[test]
    fn reports_ambiguity_metadata_for_ambiguous_native_streams() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![2, 0, 0, 0, 2],
                expected_output: vec![2],
            },
            InteractiveTrace {
                input_stream: vec![0, 0, 5, 0, 0, 5],
                expected_output: vec![5, 5, 10],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_running_sum_exceeds_count_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert!(result.metadata.ambiguity_count >= 2);
        assert!(result.metadata.recursive_refinement_applied);
        assert!(!result.metadata.recursive_refinement_resolved);
        assert!(!result.metadata.recursive_refinement_winner.is_empty());
        assert!(result
            .metadata
            .exact_alternatives
            .iter()
            .any(|name| name.contains("reg_a_gt_reg_b")));
    }

    #[test]
    fn solves_native_running_sum_crosses_count_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![5, 0],
                expected_output: vec![],
            },
            InteractiveTrace {
                input_stream: vec![0, 5],
                expected_output: vec![5],
            },
            InteractiveTrace {
                input_stream: vec![0, 0, 5, 0, 0, 5],
                expected_output: vec![5, 10],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_running_sum_crosses_count_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert!(result
            .method
            .starts_with("interactive_diff_gradient_interactive_two_register_"));
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "0".to_string(),
                "0".to_string(),
                "6".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "6".to_string(),
                "0".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "6\n12");
    }

    #[test]
    fn solves_native_running_sum_minus_count_threshold_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![0, 5],
                expected_output: vec![5],
            },
            InteractiveTrace {
                input_stream: vec![0, 0, 5, 0, 0],
                expected_output: vec![],
            },
            InteractiveTrace {
                input_stream: vec![0, 0, 5, 0, 5, 0],
                expected_output: vec![10, 10],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_running_sum_minus_count_threshold_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert!(result
            .method
            .starts_with("interactive_diff_gradient_interactive_two_register_"));
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "0".to_string(),
                "6".to_string(),
                "0".to_string(),
                "0".to_string(),
                "6".to_string(),
                "0".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "6\n6\n12\n12");
    }

    #[test]
    fn solves_native_output_crosses_count_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![0, 5, 0, 0, 5, 0],
                expected_output: vec![3],
            },
            InteractiveTrace {
                input_stream: vec![5, 0, 0, 5, 0, 0],
                expected_output: vec![6],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_output_crosses_count_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert!(result
            .method
            .starts_with("interactive_diff_gradient_interactive_two_register_"));
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "0".to_string(),
                "6".to_string(),
                "0".to_string(),
                "0".to_string(),
                "6".to_string(),
                "0".to_string(),
                "0".to_string(),
                "6".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "4\n7\n10");
    }

    #[test]
    fn solves_native_output_minus_count_threshold_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![0, 5, 0, 0, 5, 0],
                expected_output: vec![],
            },
            InteractiveTrace {
                input_stream: vec![5, 0, 0, 5, 0, 0],
                expected_output: vec![4, 6],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_output_minus_count_threshold_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_two_register_accum_input_add_const_out_sub_output_minus_reg_b_above_threshold"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "6".to_string(),
                "0".to_string(),
                "0".to_string(),
                "6".to_string(),
                "0".to_string(),
                "0".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "5\n4\n8\n7");
    }

    #[test]
    fn solves_native_pairwise_add_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![3, 4, -2, 9],
                expected_output: vec![7, 7],
            },
            InteractiveTrace {
                input_stream: vec![10, 3, 7, 7],
                expected_output: vec![13, 14],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_pair_add_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_pair_add"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "1".to_string(),
                "2".to_string(),
                "8".to_string(),
                "-1".to_string(),
                "99".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "3\n7");
    }

    #[test]
    fn solves_native_pairwise_abs_diff_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![10, 3, -4, 9],
                expected_output: vec![7, 13],
            },
            InteractiveTrace {
                input_stream: vec![5, 5, 2, 8],
                expected_output: vec![0, 6],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_pair_abs_diff_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_pair_abs_diff"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "12".to_string(),
                "4".to_string(),
                "-1".to_string(),
                "-9".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "8\n8");
    }

    #[test]
    fn solves_native_pairwise_max_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![3, 9, -2, -5],
                expected_output: vec![9, -2],
            },
            InteractiveTrace {
                input_stream: vec![7, 1, 4, 4],
                expected_output: vec![7, 4],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_pair_max_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_pair_max"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "-8".to_string(),
                "-3".to_string(),
                "10".to_string(),
                "2".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "-3\n10");
    }

    #[test]
    fn solves_native_grouped_digit_sum_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![123, 1002, 999],
                expected_output: vec![6, 3, 27],
            },
            InteractiveTrace {
                input_stream: vec![0, 2468],
                expected_output: vec![0, 20],
            },
        ];
        let result = solve_interactive_traces_differentiable_only(
            "interactive_grouped_digit_sum_v0",
            &traces,
        );
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "interactive_diff_gradient_digit_loop");
        let exec =
            execute_program_with_input(&result.code, vec!["555".to_string(), "100".to_string()])
                .unwrap();
        assert_eq!(exec.output, "15\n1");
    }

    #[test]
    fn solves_native_filter_positive_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![3, -1, 0, 5, 8],
                expected_output: vec![3, 5, 8],
            },
            InteractiveTrace {
                input_stream: vec![-4, 7, 8, -1],
                expected_output: vec![7, 8],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_filter_positive_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_filter_positive_passthrough"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "0".to_string(),
                "9".to_string(),
                "-3".to_string(),
                "4".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "9\n4");
    }

    #[test]
    fn solves_native_filter_even_stream_with_differentiable_only() {
        let traces = vec![
            InteractiveTrace {
                input_stream: vec![1, 2, 3, 4, 6],
                expected_output: vec![2, 4, 6],
            },
            InteractiveTrace {
                input_stream: vec![8, 7, 0],
                expected_output: vec![8, 0],
            },
        ];
        let result =
            solve_interactive_traces_differentiable_only("interactive_filter_even_v0", &traces);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            result.method,
            "interactive_diff_gradient_interactive_filter_even_passthrough"
        );
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "5".to_string(),
                "12".to_string(),
                "13".to_string(),
                "14".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "12\n14");
    }

    #[test]
    fn solves_interactive_add_two_with_differentiable_only() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();
        let result = solve_interactive_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(result.method.starts_with("interactive_diff_gradient_"));
        assert!(result.code.contains("while has_input() == 1"));
        assert!(result.code.contains("read_i64()"));
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "3".to_string(),
                "4".to_string(),
                "-2".to_string(),
                "9".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "7\n7");
    }

    #[test]
    fn solves_interactive_abs_diff_with_differentiable_only() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "abs_diff_v0")
            .unwrap();
        let result = solve_interactive_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(result.method.starts_with("interactive_diff_gradient_"));
        let exec = execute_program_with_input(
            &result.code,
            vec![
                "10".to_string(),
                "3".to_string(),
                "-4".to_string(),
                "9".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(exec.output, "7\n13");
    }

    #[test]
    fn solves_interactive_digit_sum_with_differentiable_only() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "digit_sum_v0")
            .unwrap();
        let result = solve_interactive_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "interactive_diff_gradient_digit_loop");
        let exec = execute_program_with_input(
            &result.code,
            vec!["123".to_string(), "1002".to_string(), "999".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "6\n3\n27");
    }

    #[test]
    fn solves_interactive_reverse_digits_with_differentiable_only() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "reverse_digits_v0")
            .unwrap();
        let result = solve_interactive_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "interactive_diff_gradient_digit_loop");
        let exec = execute_program_with_input(
            &result.code,
            vec!["120".to_string(), "907".to_string(), "4005".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "21\n709\n5004");
    }

    #[test]
    fn solves_interactive_count_even_digits_with_differentiable_only() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "count_even_digits_v0")
            .unwrap();
        let result = solve_interactive_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.method, "interactive_diff_gradient_digit_loop");
        let exec = execute_program_with_input(
            &result.code,
            vec!["0".to_string(), "12030".to_string(), "24680".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "1\n3\n5");
    }

    #[test]
    fn interactive_wrapper_ignores_incomplete_final_argument_group() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();
        let result = solve_interactive_problem_differentiable_only(&problem);
        assert!(result.success, "{:?}", result.error);
        let exec = execute_program_with_input(
            &result.code,
            vec!["3".to_string(), "4".to_string(), "9".to_string()],
        )
        .unwrap();
        assert_eq!(exec.output, "7");
    }

    #[test]
    fn interactive_verifier_rejects_wrong_program() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();
        let interactive = lift_problem_to_interactive(&problem).unwrap();
        let wrapper = build_interactive_wrapper(&problem).unwrap();
        let bad_code = format!(
            "fn add_two(a: i64, b: i64) -> i64 {{\n    return a - b;\n}}\n\n{}\n",
            wrapper
        );
        let err = verify_interactive_program(&interactive, &bad_code).unwrap_err();
        assert!(err.contains("interactive trace mismatch"));
    }

    #[test]
    fn rejects_array_problem_for_interactive_lift() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "array_sum_v0")
            .unwrap();
        let result = solve_interactive_problem_differentiable_only(&problem);
        assert!(!result.success);
        assert_eq!(result.method, "interactive_unsupported");
    }

    #[test]
    fn default_interactive_solver_is_always_differentiable() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|problem| problem.name == "add_two_v0")
            .unwrap();
        let result = solve_interactive_problem(&problem);
        assert!(result.success, "{:?}", result.error);
        assert!(result.method.starts_with("interactive_diff_gradient_"));
    }
}
