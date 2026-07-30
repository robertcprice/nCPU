//! Generate the stdin/stdout executable harness for a synthesized Rust function.
//!
//! The nSynth verifier accepts a function as its artifact. The OS sandbox
//! executes complete programs. This adapter keeps those two representations
//! honest by embedding the exact function source unchanged in a typed runner.
//! The first production slice intentionally supports only scalar `i64`
//! interfaces; unsupported shapes are refused rather than coerced.

use super::sandbox::{Example, InputValue, SandboxError};

pub(super) fn wrap_rust_function(
    function_source: &str,
    function_name: &str,
    examples: &[Example],
) -> Result<String, SandboxError> {
    if function_source.trim().is_empty() {
        return Err(unsupported("function source is empty"));
    }
    if !is_rust_identifier(function_name) {
        return Err(unsupported("function name is not a Rust identifier"));
    }
    let arity = examples
        .first()
        .map(|example| example.inputs.len())
        .ok_or_else(|| unsupported("function examples are empty"))?;
    if examples.iter().any(|example| {
        example.inputs.len() != arity
            || example
                .inputs
                .iter()
                .any(|value| !matches!(value, InputValue::Int(_)))
            || !matches!(example.expected, InputValue::Int(_))
    }) {
        return Err(unsupported(
            "Rust function harness currently supports one fixed scalar i64 signature",
        ));
    }

    let arguments = (0..arity)
        .map(|index| format!("values[{index}]"))
        .collect::<Vec<_>>()
        .join(", ");
    let parse_inputs = if arity == 0 {
        String::new()
    } else {
        format!(
            r#"
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).unwrap();
    let values: Vec<i64> = input.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|value| value.trim().parse::<i64>().unwrap())
        .collect();
    if values.len() != {arity} {{
        std::process::exit(64);
    }}
"#
        )
    };

    Ok(format!(
        r#"{function_source}

fn main() {{{parse_inputs}
    let output = {function_name}({arguments});
    println!("{{}}", output);
}}
"#
    ))
}

fn unsupported(reason: &str) -> SandboxError {
    SandboxError::UnsupportedFunctionInterface {
        reason: reason.to_string(),
    }
}

fn is_rust_identifier(candidate: &str) -> bool {
    let mut characters = candidate.chars();
    characters
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_example(inputs: &[i64], expected: i64) -> Example {
        Example {
            inputs: inputs.iter().copied().map(InputValue::Int).collect(),
            expected: InputValue::Int(expected),
        }
    }

    #[test]
    fn embeds_exact_function_in_typed_runner() {
        let source = "fn add(a: i64, b: i64) -> i64 { a + b }";
        let wrapped =
            wrap_rust_function(source, "add", &[int_example(&[2, 3], 5)]).expect("wrapper");
        assert!(wrapped.starts_with(source));
        assert!(wrapped.contains("let output = add(values[0], values[1]);"));
    }

    #[test]
    fn refuses_unsupported_or_injectable_interfaces() {
        let string_example = Example {
            inputs: vec![InputValue::String("x".into())],
            expected: InputValue::String("x".into()),
        };
        assert!(wrap_rust_function("fn f() {}", "f", &[string_example]).is_err());
        assert!(wrap_rust_function(
            "fn f(a: i64) -> i64 { a }",
            "f); std::process::exit(0); fn injected(",
            &[int_example(&[1], 1)]
        )
        .is_err());
    }
}
