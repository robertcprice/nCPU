use super::*;
use std::collections::BTreeSet;

fn array_seed_inputs(problem: &Problem) -> Option<Vec<(Vec<i64>, Vec<i64>)>> {
    let mut seeds = Vec::new();
    for example in problem.examples.iter().chain(problem.holdouts.iter()) {
        let arr = match example.inputs.first()? {
            Value::Array(values) => values.clone(),
            _ => return None,
        };
        let mut scalar_args = Vec::with_capacity(example.inputs.len().saturating_sub(1));
        for value in &example.inputs[1..] {
            match value {
                Value::Int(v) => scalar_args.push(*v),
                _ => return None,
            }
        }
        seeds.push((arr, scalar_args));
    }
    Some(seeds)
}

fn array_teacher_scalar_values(value: i64) -> Vec<i64> {
    let mut values = vec![value, value - 1, value + 1, -value, 0, 1, -1];
    values.sort_unstable();
    values.dedup();
    values
}

fn push_array_teacher_candidate(
    seen: &mut BTreeSet<(Vec<i64>, Vec<i64>)>,
    candidates: &mut Vec<(Vec<i64>, Vec<i64>)>,
    arr: Vec<i64>,
    scalar_args: Vec<i64>,
) {
    if arr.len() > MAX_ARR {
        return;
    }
    let key = (arr.clone(), scalar_args.clone());
    if seen.insert(key) {
        candidates.push((arr, scalar_args));
    }
}

fn array_variants(arr: &[i64]) -> Vec<Vec<i64>> {
    let mut seen = BTreeSet::new();
    let mut variants = Vec::new();
    let push_variant =
        |candidate: Vec<i64>, seen: &mut BTreeSet<Vec<i64>>, variants: &mut Vec<Vec<i64>>| {
            if candidate.len() <= MAX_ARR && seen.insert(candidate.clone()) {
                variants.push(candidate);
            }
        };

    if arr.is_empty() {
        push_variant(Vec::new(), &mut seen, &mut variants);
        return variants;
    }

    push_variant(arr.to_vec(), &mut seen, &mut variants);
    if arr.len() > 1 {
        let mut reversed = arr.to_vec();
        reversed.reverse();
        push_variant(reversed, &mut seen, &mut variants);

        let mut asc = arr.to_vec();
        asc.sort_unstable();
        push_variant(asc.clone(), &mut seen, &mut variants);
        asc.reverse();
        push_variant(asc, &mut seen, &mut variants);

        let mut rot_left = arr.to_vec();
        rot_left.rotate_left(1);
        push_variant(rot_left, &mut seen, &mut variants);

        let mut rot_right = arr.to_vec();
        rot_right.rotate_right(1);
        push_variant(rot_right, &mut seen, &mut variants);
    }

    let mut interesting_idx = vec![0];
    if arr.len() > 2 {
        interesting_idx.push(arr.len() / 2);
    }
    if arr.len() > 1 {
        interesting_idx.push(arr.len() - 1);
    }
    interesting_idx.sort_unstable();
    interesting_idx.dedup();

    for idx in interesting_idx {
        for replacement in [arr[idx] - 1, arr[idx] + 1, -arr[idx], 0] {
            let mut mutated = arr.to_vec();
            mutated[idx] = replacement;
            push_variant(mutated, &mut seen, &mut variants);
        }
    }

    variants
}

fn scalar_arg_variants(values: &[i64]) -> Vec<Vec<i64>> {
    let mut seen = BTreeSet::new();
    let mut variants = Vec::new();
    if seen.insert(values.to_vec()) {
        variants.push(values.to_vec());
    }
    for idx in 0..values.len() {
        for candidate in array_teacher_scalar_values(values[idx]) {
            let mut variant = values.to_vec();
            variant[idx] = candidate;
            if seen.insert(variant.clone()) {
                variants.push(variant);
            }
        }
    }
    variants
}

fn array_teacher_examples_from_code(problem: &Problem, teacher_code: &str) -> Option<Vec<Example>> {
    if teacher_code.trim().is_empty() {
        return None;
    }
    if verify_problem_code_strict(problem, teacher_code).is_err() {
        return None;
    }

    let seed_inputs = array_seed_inputs(problem)?;
    let mut seen = BTreeSet::new();
    let mut candidates = Vec::new();
    let mut produced = 0usize;

    for (arr, scalar_args) in seed_inputs {
        let arr_variants = array_variants(&arr);
        let scalar_variants = scalar_arg_variants(&scalar_args);
        for arr_variant in arr_variants.iter().take(6) {
            for scalar_variant in scalar_variants.iter().take(4) {
                push_array_teacher_candidate(
                    &mut seen,
                    &mut candidates,
                    arr_variant.clone(),
                    scalar_variant.clone(),
                );
                produced += 1;
                if produced >= 18 {
                    break;
                }
            }
            if produced >= 18 {
                break;
            }
        }
        if produced >= 18 {
            break;
        }
    }

    let mut distilled = Vec::new();
    for (arr, scalar_args) in candidates.into_iter().take(18) {
        let mut inputs = Vec::with_capacity(1 + scalar_args.len());
        inputs.push(Value::Array(arr));
        inputs.extend(scalar_args.into_iter().map(Value::Int));
        let actual = match execute_function_for_problem(
            teacher_code,
            problem.function_name(),
            &inputs,
            problem,
        ) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let RuntimeValue::Int(expected) = actual else {
            return None;
        };
        distilled.push(Example { inputs, expected: Value::Int(expected) });
        if distilled.len() >= 12 {
            break;
        }
    }

    if distilled.is_empty() {
        None
    } else {
        Some(distilled)
    }
}

pub(super) fn synthesize_array_gradient_only(problem: &Problem) -> Option<SolveResult> {
    if let Some(result) = super::two_array::synthesize_two_array(problem) {
        return Some(result);
    }
    if let Some(result) = super::structured_array::synthesize_structured_array(problem) {
        return Some(result);
    }
    super::native_array::synthesize_array_gradient_core(problem)
}

pub fn synthesize_array_from_teacher(problem: &Problem, teacher_code: &str) -> Option<SolveResult> {
    let mut augmented = problem.clone();
    let teacher_examples = array_teacher_examples_from_code(problem, teacher_code)?;
    augmented.examples.extend(teacher_examples);
    let result = synthesize_array_gradient_only(&augmented)?;
    if !result.success || verify_problem_code_strict(problem, &result.code).is_err() {
        return None;
    }
    Some(result)
}

pub fn synthesize_array(problem: &Problem) -> Option<SolveResult> {
    synthesize_array_gradient_only(problem)
}
