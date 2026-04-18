use crate::benchmark::Problem;

use super::helpers::{pair_value, str_value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ParamType {
    I64,
    ArrayI64,
    String,
    Other(String),
}

pub(super) fn parse_param_types(signature: &str) -> Vec<ParamType> {
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
            let ty = param
                .split_once(':')
                .map(|(_, ty)| ty.trim())
                .unwrap_or_default();
            match ty {
                "i64" => ParamType::I64,
                "[i64]" => ParamType::ArrayI64,
                "string" => ParamType::String,
                other => ParamType::Other(other.to_string()),
            }
        })
        .collect()
}

pub(super) fn scalar_param_names(arity: usize) -> Vec<String> {
    match arity {
        0 => Vec::new(),
        1 => vec!["x".to_string()],
        2 => vec!["a".to_string(), "b".to_string()],
        3 => vec!["a".to_string(), "b".to_string(), "c".to_string()],
        n => (0..n).map(|idx| format!("x{idx}")).collect(),
    }
}

pub(super) fn unary_string_examples(problem: &Problem) -> Option<Vec<String>> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::String] {
        return None;
    }
    problem
        .examples
        .iter()
        .map(|example| {
            if example.inputs.len() != 1 {
                return None;
            }
            str_value(&example.inputs[0]).map(|value| value.to_string())
        })
        .collect()
}

pub(super) fn unary_pair_examples(problem: &Problem) -> Option<Vec<(i64, i64)>> {
    let param_types = parse_param_types(problem.signature);
    if param_types.len() != 1 {
        return None;
    }
    match &param_types[0] {
        ParamType::Other(_) => problem
            .examples
            .iter()
            .map(|example| {
                if example.inputs.len() != 1 {
                    return None;
                }
                pair_value(&example.inputs[0])
            })
            .collect(),
        _ => None,
    }
}

pub(super) fn scalar_params_decl(param_names: &[String]) -> String {
    param_names
        .iter()
        .map(|name| format!("{name}: i64"))
        .collect::<Vec<_>>()
        .join(", ")
}
