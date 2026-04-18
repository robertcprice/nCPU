use crate::benchmark::{Problem, Value};

pub(super) fn templ(template: &str, fn_name: &str) -> String {
    template.replace("__FN__", fn_name)
}

pub(super) fn int_value(value: &Value) -> Option<i64> {
    match value {
        Value::Int(v) => Some(*v),
        _ => None,
    }
}

pub(super) fn str_value(value: &Value) -> Option<&str> {
    match value {
        Value::Str(v) => Some(v.as_str()),
        _ => None,
    }
}

pub(super) fn array_value(value: &Value) -> Option<&[i64]> {
    match value {
        Value::Array(v) => Some(v.as_slice()),
        _ => None,
    }
}

pub(super) fn pair_value(value: &Value) -> Option<(i64, i64)> {
    match value {
        Value::Pair(a, b) => Some((*a, *b)),
        _ => None,
    }
}

pub(super) fn family_name(problem: &Problem) -> String {
    problem
        .name
        .rsplit_once("_v")
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| problem.name.clone())
}

pub(super) fn validate_unary_int<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(i64) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 1
            && int_value(&ex.inputs[0])
                .map(|x| func(x) == ex.expected)
                .unwrap_or(false)
    })
}

pub(super) fn validate_binary_int<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(i64, i64) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 2
            && int_value(&ex.inputs[0])
                .zip(int_value(&ex.inputs[1]))
                .map(|(a, b)| func(a, b) == ex.expected)
                .unwrap_or(false)
    })
}

pub(super) fn validate_ternary_int<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(i64, i64, i64) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 3
            && int_value(&ex.inputs[0])
                .zip(int_value(&ex.inputs[1]))
                .zip(int_value(&ex.inputs[2]))
                .map(|((a, b), c)| func(a, b, c) == ex.expected)
                .unwrap_or(false)
    })
}

pub(super) fn validate_quaternary_int<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(i64, i64, i64, i64) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 4
            && int_value(&ex.inputs[0])
                .zip(int_value(&ex.inputs[1]))
                .zip(int_value(&ex.inputs[2]))
                .zip(int_value(&ex.inputs[3]))
                .map(|(((a, b), c), d)| func(a, b, c, d) == ex.expected)
                .unwrap_or(false)
    })
}

pub(super) fn validate_unary_array<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(&[i64]) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 1
            && array_value(&ex.inputs[0])
                .map(|arr| func(arr) == ex.expected)
                .unwrap_or(false)
    })
}

pub(super) fn validate_array_and_int<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(&[i64], i64) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 2
            && array_value(&ex.inputs[0])
                .zip(int_value(&ex.inputs[1]))
                .map(|(arr, target)| func(arr, target) == ex.expected)
                .unwrap_or(false)
    })
}

pub(super) fn validate_unary_str<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(&str) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 1
            && str_value(&ex.inputs[0])
                .map(|s| func(s) == ex.expected)
                .unwrap_or(false)
    })
}

pub(super) fn validate_unary_pair<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(i64, i64) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 1
            && pair_value(&ex.inputs[0])
                .map(|(a, b)| func(a, b) == ex.expected)
                .unwrap_or(false)
    })
}

pub(super) fn validate_two_arrays<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(&[i64], &[i64]) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 2
            && array_value(&ex.inputs[0])
                .zip(array_value(&ex.inputs[1]))
                .map(|(a, b)| func(a, b) == ex.expected)
                .unwrap_or(false)
    })
}
