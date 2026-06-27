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

/// Extract the integer payload of an array value for the numeric solvers.
///
/// Now that `Value::Array` carries `Vec<Value>`, this returns an *owned*
/// `Vec<i64>` (it cannot borrow `&[i64]` out of the element vector) and yields
/// `None` unless every element is a `Value::Int`. Numeric solvers therefore
/// keep seeing exactly the integer arrays they did before; typed/nested arrays
/// produce `None` and are routed to the typed-array path instead.
pub(super) fn array_value(value: &Value) -> Option<Vec<i64>> {
    value.as_i64_slice()
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
                .map(|x| func(x) == ex.expected_int())
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
                .map(|(a, b)| func(a, b) == ex.expected_int())
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
                .map(|((a, b), c)| func(a, b, c) == ex.expected_int())
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
                .map(|(((a, b), c), d)| func(a, b, c, d) == ex.expected_int())
                .unwrap_or(false)
    })
}

/// Invoke an array-consuming validator primitive TOTALLY: any panic it raises on
/// a probe example (e.g. `arr[0]` / `min().unwrap()` on an EMPTY array — which the
/// reference-driven example sampler now generates, since array lengths are sampled
/// from `0..=MAX`) is caught and converted to a clean validation MISS rather than
/// aborting the whole synthesizer mid-validation. This is the single chokepoint
/// that makes the entire `validate_*_array` primitive family total, mirroring the
/// `run_isolated` catch_unwind on the runtime verify path. Soundness is unchanged:
/// the candidate is still proven against the real reference holdouts by the strict
/// verifier, so a primitive that panics on an input simply fails to match here and
/// the family is rejected — never fabricated.
fn array_probe<R>(f: impl FnOnce() -> R) -> Option<R> {
    crate::runtime::install_silent_panic_hook_once();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok()
}

pub(super) fn validate_unary_array<F>(problem: &Problem, func: F) -> bool
where
    F: Fn(&[i64]) -> i64,
{
    problem.examples.iter().all(|ex| {
        ex.inputs.len() == 1
            && array_value(&ex.inputs[0])
                .and_then(|arr| array_probe(|| func(&arr) == ex.expected_int()))
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
                .and_then(|(arr, target)| array_probe(|| func(&arr, target) == ex.expected_int()))
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
                .map(|s| func(s) == ex.expected_int())
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
                .map(|(a, b)| func(a, b) == ex.expected_int())
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
                .and_then(|(a, b)| array_probe(|| func(&a, &b) == ex.expected_int()))
                .unwrap_or(false)
    })
}
