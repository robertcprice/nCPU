//! Typed binary-wire harness for executing an exact synthesized Rust function.
//!
//! nSynth verifies function artifacts while the OS sandbox executes complete
//! programs. This adapter embeds the exact function source unchanged and adds a
//! generated runner selected only from the examples' value shapes. It does not
//! inspect source text, infer intent, or coerce unsupported values.

use super::sandbox::{Example, InputValue, SandboxError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RustValueShape {
    I64,
    F64,
    Bool,
    String,
    I64Array,
}

impl RustValueShape {
    fn of(value: &InputValue) -> Self {
        match value {
            InputValue::Int(_) => Self::I64,
            InputValue::Float(_) => Self::F64,
            InputValue::Bool(_) => Self::Bool,
            InputValue::String(_) => Self::String,
            InputValue::IntArray(_) => Self::I64Array,
        }
    }

    fn read_expression(self) -> &'static str {
        match self {
            Self::I64 => "__nsynth_wire::read_i64(&input, &mut cursor)",
            Self::F64 => "__nsynth_wire::read_f64(&input, &mut cursor)",
            Self::Bool => "__nsynth_wire::read_bool(&input, &mut cursor)",
            Self::String => "__nsynth_wire::read_string(&input, &mut cursor)",
            Self::I64Array => "__nsynth_wire::read_i64_array(&input, &mut cursor)",
        }
    }
}

/// Exact function source plus the value-shape contract of its generated runner.
pub(super) struct RustFunctionHarness {
    pub source: String,
    input_shapes: Vec<RustValueShape>,
    output_shape: RustValueShape,
}

impl RustFunctionHarness {
    pub fn build(
        function_source: &str,
        function_name: &str,
        examples: &[Example],
    ) -> Result<Self, SandboxError> {
        if function_source.trim().is_empty() {
            return Err(unsupported("function source is empty"));
        }
        if !is_rust_identifier(function_name) {
            return Err(unsupported("function name is not a Rust identifier"));
        }
        let first = examples
            .first()
            .ok_or_else(|| unsupported("function examples are empty"))?;
        let input_shapes = first
            .inputs
            .iter()
            .map(RustValueShape::of)
            .collect::<Vec<_>>();
        let output_shape = RustValueShape::of(&first.expected);
        if examples.iter().any(|example| {
            example.inputs.len() != input_shapes.len()
                || example
                    .inputs
                    .iter()
                    .zip(&input_shapes)
                    .any(|(value, shape)| RustValueShape::of(value) != *shape)
                || RustValueShape::of(&example.expected) != output_shape
        }) {
            return Err(unsupported(
                "Rust function examples do not share one fixed typed interface",
            ));
        }

        let read_arguments = input_shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| format!("    let arg{index} = {};\n", shape.read_expression()))
            .collect::<String>();
        let arguments = (0..input_shapes.len())
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let render_output = render_output(output_shape);
        let source = format!(
            r#"{function_source}

{BINARY_WIRE_READERS}

fn main() {{
    let mut input = Vec::new();
    std::io::Read::read_to_end(&mut std::io::stdin(), &mut input)
        .unwrap_or_else(|_| __nsynth_wire::fail_wire());
    let mut cursor = 0usize;
{read_arguments}    if cursor != input.len() {{
        __nsynth_wire::fail_wire();
    }}
    let output = {function_name}({arguments});
{render_output}
}}
"#
        );
        Ok(Self {
            source,
            input_shapes,
            output_shape,
        })
    }

    pub fn encode_inputs(&self, inputs: &[InputValue]) -> Result<Vec<u8>, SandboxError> {
        if inputs.len() != self.input_shapes.len() {
            return Err(unsupported(
                "Rust function input arity differs from its typed interface",
            ));
        }
        let mut wire = Vec::new();
        for (value, shape) in inputs.iter().zip(&self.input_shapes) {
            if RustValueShape::of(value) != *shape {
                return Err(unsupported(
                    "Rust function input shape differs from its typed interface",
                ));
            }
            match value {
                InputValue::Int(value) => wire.extend_from_slice(&value.to_be_bytes()),
                InputValue::Float(value) => wire.extend_from_slice(&value.to_bits().to_be_bytes()),
                InputValue::Bool(value) => wire.push(u8::from(*value)),
                InputValue::String(value) => {
                    encode_len(&mut wire, value.len())?;
                    wire.extend_from_slice(value.as_bytes());
                }
                InputValue::IntArray(values) => {
                    encode_len(&mut wire, values.len())?;
                    for value in values {
                        wire.extend_from_slice(&value.to_be_bytes());
                    }
                }
            }
        }
        Ok(wire)
    }

    pub fn parse_output(&self, output: &str) -> Result<InputValue, SandboxError> {
        let output = output.strip_suffix('\n').unwrap_or(output);
        let output = output.strip_suffix('\r').unwrap_or(output);
        match self.output_shape {
            RustValueShape::I64 => output
                .strip_prefix("i:")
                .and_then(|value| value.parse::<i64>().ok())
                .map(InputValue::Int)
                .ok_or_else(|| malformed_output(output)),
            RustValueShape::F64 => output
                .strip_prefix("f:")
                .and_then(|value| u64::from_str_radix(value, 16).ok())
                .map(|bits| InputValue::Float(f64::from_bits(bits)))
                .ok_or_else(|| malformed_output(output)),
            RustValueShape::Bool => match output {
                "b:0" => Ok(InputValue::Bool(false)),
                "b:1" => Ok(InputValue::Bool(true)),
                _ => Err(malformed_output(output)),
            },
            RustValueShape::String => {
                let encoded = output
                    .strip_prefix("s:")
                    .ok_or_else(|| malformed_output(output))?;
                let bytes = decode_hex(encoded).ok_or_else(|| malformed_output(output))?;
                String::from_utf8(bytes)
                    .map(InputValue::String)
                    .map_err(|_| malformed_output(output))
            }
            RustValueShape::I64Array => {
                let encoded = output
                    .strip_prefix("a:")
                    .ok_or_else(|| malformed_output(output))?;
                if encoded.is_empty() {
                    return Ok(InputValue::IntArray(Vec::new()));
                }
                encoded
                    .split(',')
                    .map(|value| value.parse::<i64>())
                    .collect::<Result<Vec<_>, _>>()
                    .map(InputValue::IntArray)
                    .map_err(|_| malformed_output(output))
            }
        }
    }
}

fn encode_len(wire: &mut Vec<u8>, length: usize) -> Result<(), SandboxError> {
    let length = u64::try_from(length)
        .map_err(|_| unsupported("Rust function value exceeds the binary wire length limit"))?;
    wire.extend_from_slice(&length.to_be_bytes());
    Ok(())
}

fn decode_hex(encoded: &str) -> Option<Vec<u8>> {
    if encoded.len() % 2 != 0 || !encoded.is_ascii() {
        return None;
    }
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_digit(pair[0])?;
            let low = hex_digit(pair[1])?;
            Some((high << 4) | low)
        })
        .collect()
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn render_output(shape: RustValueShape) -> &'static str {
    match shape {
        RustValueShape::I64 => r#"    println!("i:{}", output);"#,
        RustValueShape::F64 => r#"    println!("f:{:016x}", output.to_bits());"#,
        RustValueShape::Bool => r#"    println!("b:{}", u8::from(output));"#,
        RustValueShape::String => {
            r#"    print!("s:");
    for byte in output.as_bytes() {
        print!("{byte:02x}");
    }
    println!();"#
        }
        RustValueShape::I64Array => {
            r#"    print!("a:");
    for (index, value) in output.iter().enumerate() {
        if index > 0 {
            print!(",");
        }
        print!("{}", value);
    }
    println!();"#
        }
    }
}

fn malformed_output(output: &str) -> SandboxError {
    SandboxError::RuntimePanic {
        message: format!("typed Rust runner produced malformed output: {output:?}"),
        backtrace: None,
    }
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

const BINARY_WIRE_READERS: &str = r#"mod __nsynth_wire {
use std::convert::TryFrom;

pub fn fail_wire() -> ! {
    std::process::exit(64);
}

fn read_exact<const N: usize>(input: &[u8], cursor: &mut usize) -> [u8; N] {
    let end = cursor.checked_add(N).unwrap_or_else(|| fail_wire());
    let bytes = input.get(*cursor..end).unwrap_or_else(|| fail_wire());
    let mut output = [0u8; N];
    output.copy_from_slice(bytes);
    *cursor = end;
    output
}

fn read_u64(input: &[u8], cursor: &mut usize) -> u64 {
    u64::from_be_bytes(read_exact(input, cursor))
}

pub fn read_i64(input: &[u8], cursor: &mut usize) -> i64 {
    i64::from_be_bytes(read_exact(input, cursor))
}

pub fn read_f64(input: &[u8], cursor: &mut usize) -> f64 {
    f64::from_bits(read_u64(input, cursor))
}

pub fn read_bool(input: &[u8], cursor: &mut usize) -> bool {
    match read_exact::<1>(input, cursor)[0] {
        0 => false,
        1 => true,
        _ => fail_wire(),
    }
}

fn read_len(input: &[u8], cursor: &mut usize) -> usize {
    usize::try_from(read_u64(input, cursor)).unwrap_or_else(|_| fail_wire())
}

pub fn read_string(input: &[u8], cursor: &mut usize) -> String {
    let length = read_len(input, cursor);
    let end = cursor.checked_add(length).unwrap_or_else(|| fail_wire());
    let bytes = input.get(*cursor..end).unwrap_or_else(|| fail_wire());
    *cursor = end;
    String::from_utf8(bytes.to_vec()).unwrap_or_else(|_| fail_wire())
}

pub fn read_i64_array(input: &[u8], cursor: &mut usize) -> Vec<i64> {
    let length = read_len(input, cursor);
    (0..length).map(|_| read_i64(input, cursor)).collect()
}
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn example(inputs: Vec<InputValue>, expected: InputValue) -> Example {
        Example { inputs, expected }
    }

    #[test]
    fn embeds_exact_function_and_roundtrips_every_supported_shape() {
        let source = "fn typed(a: i64, b: f64, c: bool, d: String, e: Vec<i64>) -> String { d }";
        let fixture = example(
            vec![
                InputValue::Int(-7),
                InputValue::Float(3.25),
                InputValue::Bool(true),
                InputValue::String("42\nλ\0".into()),
                InputValue::IntArray(vec![-2, 0, 8]),
            ],
            InputValue::String("42\nλ\0".into()),
        );
        let harness =
            RustFunctionHarness::build(source, "typed", &[fixture.clone()]).expect("typed harness");
        assert!(harness.source.starts_with(source));
        assert!(harness
            .source
            .contains("let output = typed(arg0, arg1, arg2, arg3, arg4);"));
        assert!(!harness.encode_inputs(&fixture.inputs).unwrap().is_empty());
        assert_eq!(
            harness.parse_output("s:34320acebb00\n").unwrap(),
            fixture.expected
        );

        for (expected, wire) in [
            (InputValue::Int(-9), "i:-9\n"),
            (InputValue::Float(-0.0), "f:8000000000000000\n"),
            (InputValue::Bool(true), "b:1\n"),
            (InputValue::IntArray(vec![-1, 0, 2]), "a:-1,0,2\n"),
        ] {
            let output_harness = RustFunctionHarness::build(
                "fn f() -> i64 { 0 }",
                "f",
                &[example(vec![], expected.clone())],
            )
            .expect("output harness");
            assert_eq!(output_harness.parse_output(wire).unwrap(), expected);
        }
    }

    #[test]
    fn refuses_mixed_interfaces_and_injectable_names() {
        let int = example(vec![InputValue::Int(1)], InputValue::Int(1));
        let mixed = example(vec![InputValue::Bool(true)], InputValue::Int(1));
        assert!(
            RustFunctionHarness::build("fn f(a: i64) -> i64 { a }", "f", &[int, mixed]).is_err()
        );
        assert!(RustFunctionHarness::build(
            "fn f(a: i64) -> i64 { a }",
            "f); std::process::exit(0); fn injected(",
            &[example(vec![InputValue::Int(1)], InputValue::Int(1))]
        )
        .is_err());
    }
}
