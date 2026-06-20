//! Interface discovery for modular synthesis
//!
//! Infers function signatures, data structures, and constraints
//! from examples and specifications.

use crate::benchmark::Example;

pub use super::decomposition::Interface;

/// Discover interface from examples
pub fn discover_interface(examples: &[Example]) -> Interface {
    if examples.is_empty() {
        return Interface {
            imports: Vec::new(),
            exports: Vec::new(),
            types: Vec::new(),
        };
    }

    let mut exports = Vec::new();
    let mut types = Vec::new();

    // Infer function signature from examples
    let signature = infer_signature_from_examples(examples);
    exports.push(super::decomposition::Export {
        name: "synthesized".to_string(),
        signature,
        visibility: super::decomposition::Visibility::Public,
    });

    // Infer types from examples
    let inferred_types = infer_types_from_examples(examples);
    types.extend(inferred_types);

    Interface {
        imports: Vec::new(),
        exports,
        types,
    }
}

/// Infer function signature from examples
fn infer_signature_from_examples(examples: &[Example]) -> String {
    if examples.is_empty() {
        return "fn synthesized() -> i64".to_string();
    }

    let first = &examples[0];
    let mut param_types = Vec::new();
    let mut param_idx = 0;

    for input in &first.inputs {
        let type_str = match input {
            crate::benchmark::Value::Int(_) => "i64",
            crate::benchmark::Value::Float(_) => "f64",
            crate::benchmark::Value::Str(_) => "String",
            crate::benchmark::Value::Bool(_) => "bool",
            crate::benchmark::Value::Array(_) => "Vec<i64>",
            crate::benchmark::Value::Pair(_, _) => "(i64, i64)",
            crate::benchmark::Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}",
            crate::benchmark::Value::Tree(_) => "Tree",
        };
        param_idx += 1;
        param_types.push(format!("{}: {}", param_name(param_idx), type_str));
    }

    let return_type = match &first.expected {
        crate::benchmark::Value::Int(_) => "i64",
        crate::benchmark::Value::Float(_) => "f64",
        crate::benchmark::Value::Str(_) => "String",
        crate::benchmark::Value::Bool(_) => "bool",
        crate::benchmark::Value::Array(_) => "Vec<i64>",
        crate::benchmark::Value::Pair(_, _) => "(i64, i64)",
        crate::benchmark::Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}",
        crate::benchmark::Value::Tree(_) => "Tree",
    };

    format!(
        "fn synthesized({}) -> {}",
        param_types.join(", "),
        return_type
    )
}

/// Parameter names
fn param_name(idx: usize) -> &'static str {
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    if idx <= names.len() {
        names[idx - 1]
    } else {
        "arg"
    }
}

/// Infer types from examples
fn infer_types_from_examples(examples: &[Example]) -> Vec<super::decomposition::TypeDecl> {
    let mut types = Vec::new();
    let mut seen_types = std::collections::HashSet::new();

    for example in examples {
        // Check input types
        for input in &example.inputs {
            if let Some(type_name) = value_to_type(input) {
                if !seen_types.contains(&type_name) {
                    seen_types.insert(type_name.clone());
                    types.push(super::decomposition::TypeDecl {
                        name: type_name,
                        kind: super::decomposition::TypeKind::Alias("inferred".to_string()),
                        generics: Vec::new(),
                    });
                }
            }
        }

        // Check output type
        if let Some(type_name) = value_to_type(&example.expected) {
            if !seen_types.contains(&type_name) {
                seen_types.insert(type_name.clone());
                types.push(super::decomposition::TypeDecl {
                    name: type_name,
                    kind: super::decomposition::TypeKind::Alias("inferred".to_string()),
                    generics: Vec::new(),
                });
            }
        }
    }

    types
}

/// Convert value to type name
fn value_to_type(value: &crate::benchmark::Value) -> Option<String> {
    Some(match value {
        crate::benchmark::Value::Int(_) => "i64".to_string(),
        crate::benchmark::Value::Float(_) => "f64".to_string(),
        crate::benchmark::Value::Str(_) => "String".to_string(),
        crate::benchmark::Value::Bool(_) => "bool".to_string(),
        crate::benchmark::Value::Array(_) => "Vec<i64>".to_string(),
        crate::benchmark::Value::Pair(_, _) => "(i64, i64)".to_string(),
        crate::benchmark::Value::Quad(_, _, _, _) => "Quad".to_string(),
        crate::benchmark::Value::Tree(_) => "Tree".to_string(),
    })
}

/// Extract constraints from examples
pub fn extract_constraints(examples: &[Example]) -> Vec<super::decomposition::Constraint> {
    let mut constraints = Vec::new();

    // Analyze input patterns
    if examples
        .iter()
        .all(|ex| ex.inputs.len() == examples[0].inputs.len())
    {
        constraints.push(super::decomposition::Constraint {
            kind: super::decomposition::ConstraintKind::Correctness,
            description: "Fixed arity function".to_string(),
        });
    }

    // Check for array operations
    if examples.iter().any(|ex| {
        ex.inputs
            .iter()
            .any(|v| matches!(v, crate::benchmark::Value::Array(_)))
    }) {
        constraints.push(super::decomposition::Constraint {
            kind: super::decomposition::ConstraintKind::Performance,
            description: "Array operations may have O(n) complexity".to_string(),
        });
    }

    constraints
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Value;

    #[test]
    fn test_discover_interface_empty() {
        let interface = discover_interface(&[]);
        assert_eq!(interface.exports.len(), 0);
    }

    #[test]
    fn test_discover_interface_simple() {
        let examples = vec![Example {
            inputs: vec![Value::Int(2), Value::Int(3)],
            expected: Value::Int(5),
        }];

        let interface = discover_interface(&examples);
        assert_eq!(interface.exports.len(), 1);
        assert!(interface.exports[0].signature.contains("i64"));
    }
}
