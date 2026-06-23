//! Type-driven refinement for synthesis
//!
//! Uses type constraints to guide search and prune invalid branches.

use crate::benchmark::Problem;

/// Type context for refinement
#[derive(Debug, Clone)]
pub struct TypeContext {
    /// Type constraints for each variable
    pub variable_types: std::collections::HashMap<String, Type>,
    /// Function signatures
    pub signatures: std::collections::HashMap<String, Signature>,
    /// Global type constraints
    pub constraints: Vec<TypeConstraint>,
}

/// Type representation
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    String,
    Bool,
    Array(Box<Type>),
    Vec(Box<Type>),
    Tuple(Vec<Type>),
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    Unknown,
    Named(String),
}

/// Function signature
#[derive(Debug, Clone)]
pub struct Signature {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
}

/// Type constraint
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    pub variable: String,
    pub constraint: ConstraintKind,
}

/// Constraint kind
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    Equals(Type),
    SupertypeOf(Type),
    SubtypeOf(Type),
    ImplementsTrait(String),
}

/// Refine program with type constraints
pub fn refine_with_types(_partial: &Problem, _types: &TypeContext) -> Result<Problem, String> {
    // In production, this would:
    // 1. Analyze partial program for type holes
    // 2. Use type context to fill holes
    // 3. Propagate type information
    // 4. Prune invalid search branches
    // 5. Guide synthesis toward type-consistent solutions

    Ok(_partial.clone())
}

/// Infer types from examples
pub fn infer_types_from_examples(examples: &[crate::benchmark::Example]) -> TypeContext {
    let mut variable_types = std::collections::HashMap::new();
    let signatures = std::collections::HashMap::new();
    let mut constraints = Vec::new();

    if let Some(first) = examples.first() {
        // Infer parameter types
        for (i, input) in first.inputs.iter().enumerate() {
            let param_name = format!("param{}", i);
            let type_ = value_to_type(input);
            variable_types.insert(param_name.clone(), type_.clone());
            constraints.push(TypeConstraint {
                variable: param_name,
                constraint: ConstraintKind::Equals(type_),
            });
        }

        // Infer return type
        let return_type = value_to_type(&first.expected);
        constraints.push(TypeConstraint {
            variable: "return".to_string(),
            constraint: ConstraintKind::Equals(return_type),
        });
    }

    TypeContext {
        variable_types,
        signatures,
        constraints,
    }
}

/// Convert benchmark value to type
fn value_to_type(value: &crate::benchmark::Value) -> Type {
    match value {
        crate::benchmark::Value::Int(_) => Type::Int,
        crate::benchmark::Value::Float(_) => Type::Float,
        crate::benchmark::Value::Str(_) => Type::String,
        crate::benchmark::Value::Bool(_) => Type::Bool,
        crate::benchmark::Value::Array(_) => Type::Array(Box::new(Type::Int)),
        crate::benchmark::Value::Pair(_, _) => Type::Tuple(vec![Type::Int, Type::Int]),
        crate::benchmark::Value::Quad(_, _, _, _) => {
            Type::Tuple(vec![Type::Int, Type::Int, Type::Int, Type::Int])
        }
        crate::benchmark::Value::Tree(_) => Type::Named("Tree".to_string()),
        crate::benchmark::Value::Tuple(elems) => {
            Type::Tuple(elems.iter().map(value_to_type).collect())
        }
        crate::benchmark::Value::Struct(_) => Type::Named("Struct".to_string()),
    }
}

/// Validate program against type constraints
pub fn validate_types(_program: &str, _context: &TypeContext) -> Result<(), Vec<String>> {
    // In production, this would:
    // 1. Parse program into AST
    // 2. Type-check each expression
    // 3. Validate all constraints
    // 4. Return list of type errors

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    #[test]
    fn test_infer_types_from_examples() {
        let examples = vec![Example {
            inputs: vec![Value::Int(5)],
            expected: Value::Int(10),
        }];

        let context = infer_types_from_examples(&examples);
        assert_eq!(context.constraints.len(), 2);
    }

    #[test]
    fn test_value_to_type() {
        assert_eq!(value_to_type(&Value::Int(42)), Type::Int);
        assert_eq!(value_to_type(&Value::Bool(true)), Type::Bool);
    }
}
