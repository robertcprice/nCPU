use super::*;

#[derive(Clone, Copy, Debug)]
enum StructuredArrayKernel {
    KthSmallest,
    TwoSumExists,
    CountDistinct,
    BinarySearch,
}

impl StructuredArrayKernel {
    fn method(self) -> &'static str {
        match self {
            StructuredArrayKernel::KthSmallest => "arr_gradient_kth_smallest",
            StructuredArrayKernel::TwoSumExists => "arr_gradient_two_sum_exists",
            StructuredArrayKernel::CountDistinct => "arr_gradient_count_distinct",
            StructuredArrayKernel::BinarySearch => "arr_gradient_binary_search",
        }
    }

    fn predict(self, example: &Example) -> Option<i64> {
        match self {
            StructuredArrayKernel::KthSmallest => match example.inputs.as_slice() {
                [Value::Array(arr), Value::Int(k)] => {
                    if *k < 1 || *k as usize > arr.len() {
                        None
                    } else {
                        let mut values = arr.clone();
                        values.sort_unstable();
                        Some(values[(*k - 1) as usize])
                    }
                }
                _ => None,
            },
            StructuredArrayKernel::TwoSumExists => match example.inputs.as_slice() {
                [Value::Array(arr), Value::Int(target)] => {
                    for i in 0..arr.len() {
                        for j in (i + 1)..arr.len() {
                            if arr[i] + arr[j] == *target {
                                return Some(1);
                            }
                        }
                    }
                    Some(0)
                }
                _ => None,
            },
            StructuredArrayKernel::CountDistinct => match example.inputs.as_slice() {
                [Value::Array(arr)] => {
                    let mut values = arr.clone();
                    values.sort_unstable();
                    values.dedup();
                    Some(values.len() as i64)
                }
                _ => None,
            },
            StructuredArrayKernel::BinarySearch => match example.inputs.as_slice() {
                [Value::Array(arr), Value::Int(target)] => {
                    let mut lo = 0i64;
                    let mut hi = arr.len() as i64 - 1;
                    while lo <= hi {
                        let mid = (lo + hi) / 2;
                        let value = arr[mid as usize];
                        if value == *target {
                            return Some(mid);
                        }
                        if value < *target {
                            lo = mid + 1;
                        } else {
                            hi = mid - 1;
                        }
                    }
                    Some(-1)
                }
                _ => None,
            },
        }
    }

    fn emit(self, fn_name: &str) -> String {
        match self {
            StructuredArrayKernel::KthSmallest => format!(
                "fn {fn_name}(arr: [i64], k: i64) -> i64 {{\n    arr.sort();\n    return arr[k - 1];\n}}\n"
            ),
            StructuredArrayKernel::TwoSumExists => format!(
                "fn {fn_name}(arr: [i64], target: i64) -> i64 {{\n    i: i64 = 0;\n    while i < arr.len {{\n        j: i64 = i + 1;\n        while j < arr.len {{\n            if arr[i] + arr[j] == target {{ return 1; }}\n            j = j + 1;\n        }}\n        i = i + 1;\n    }}\n    return 0;\n}}\n"
            ),
            StructuredArrayKernel::CountDistinct => format!(
                "fn {fn_name}(arr: [i64]) -> i64 {{\n    if arr.len == 0 {{ return 0; }}\n    arr.sort();\n    count: i64 = 1;\n    i: i64 = 1;\n    while i < arr.len {{\n        if arr[i] != arr[i - 1] {{\n            count = count + 1;\n        }}\n        i = i + 1;\n    }}\n    return count;\n}}\n"
            ),
            StructuredArrayKernel::BinarySearch => format!(
                "fn {fn_name}(arr: [i64], target: i64) -> i64 {{\n    lo: i64 = 0;\n    hi: i64 = arr.len - 1;\n    while lo <= hi {{\n        mid: i64 = (lo + hi) / 2;\n        if arr[mid] == target {{ return mid; }}\n        if arr[mid] < target {{ lo = mid + 1; }}\n        if arr[mid] > target {{ hi = mid - 1; }}\n    }}\n    return -1;\n}}\n"
            ),
        }
    }
}

fn structured_candidates(problem: &Problem) -> Vec<StructuredArrayKernel> {
    let Some(first) = problem.examples.first() else {
        return Vec::new();
    };
    match first.inputs.as_slice() {
        [Value::Array(_)] => vec![StructuredArrayKernel::CountDistinct],
        [Value::Array(_), Value::Int(_)] => vec![
            StructuredArrayKernel::KthSmallest,
            StructuredArrayKernel::TwoSumExists,
            StructuredArrayKernel::BinarySearch,
        ],
        _ => Vec::new(),
    }
}

fn structured_kernel_loss(problem: &Problem, kernel: StructuredArrayKernel) -> Option<f32> {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for example in &problem.examples {
        let predicted = kernel.predict(example)? as f32;
        let diff = predicted - example.expected as f32;
        total += diff * diff;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(total / count as f32)
    }
}

fn structured_selector_loss(
    params: &[f32],
    kernels: &[StructuredArrayKernel],
    problem: &Problem,
    temp: f32,
) -> Option<f32> {
    let weights = softmax_temp(params, temp);
    let mut total = 0.0f32;
    for example in &problem.examples {
        let mut predicted = 0.0f32;
        for (weight, kernel) in weights.iter().zip(kernels) {
            predicted += *weight * kernel.predict(example)? as f32;
        }
        let diff = predicted - example.expected as f32;
        total += diff * diff;
    }
    Some(total / problem.examples.len().max(1) as f32)
}

fn try_emit_structured(
    problem: &Problem,
    kernels: &[StructuredArrayKernel],
    params: &[f32],
) -> Option<SolveResult> {
    let kernel = *kernels.get(argmax(params))?;
    let code = kernel.emit(problem.function_name());
    if verify_problem_code_strict(problem, &code).is_err() {
        return None;
    }
    Some(SolveResult {
        success: true,
        code,
        method: kernel.method().to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

pub(super) fn synthesize_structured_array(problem: &Problem) -> Option<SolveResult> {
    let mut kernels = Vec::new();
    let mut params = Vec::new();
    for kernel in structured_candidates(problem) {
        let Some(loss) = structured_kernel_loss(problem, kernel) else {
            continue;
        };
        if !loss.is_finite() {
            continue;
        }
        kernels.push(kernel);
        params.push(-loss.min(1_000_000.0));
    }
    if kernels.is_empty() {
        return None;
    }

    if let Some(result) = try_emit_structured(problem, &kernels, &params) {
        return Some(result);
    }

    let mut opt = Adam::new(params.len(), 0.1);
    let mut best_params = params.clone();
    let mut best_loss = structured_selector_loss(&params, &kernels, problem, 1.0)?;
    for step in 0..80 {
        let temp = (2.0f32 * (1.0 - step as f32 / 80.0)).max(0.1);
        let loss = structured_selector_loss(&params, &kernels, problem, temp)?;
        if loss < best_loss {
            best_loss = loss;
            best_params = params.clone();
        }
        if let Some(result) = try_emit_structured(problem, &kernels, &params) {
            return Some(result);
        }
        let grads = fd_grad(
            &params,
            |trial, trial_temp| {
                structured_selector_loss(trial, &kernels, problem, trial_temp).unwrap_or(f32::MAX)
            },
            temp,
        );
        opt.step(&mut params, &grads);
    }

    try_emit_structured(problem, &kernels, &best_params)
}
