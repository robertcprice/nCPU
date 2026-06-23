//! Canonical coding intent contract (Package B slice).
//!
//! Bridges Linguigenesis `SynthesisRequirement` to nsynth `Problem` without
//! keyword routing or category string dispatch.

use crate::benchmark::{Example, Problem, Value};
use crate::linguigenesis_bridge::{BridgeError, LinguigenesisBridge};
use linguigenesis_core::coding_requirements::{LiteralValue, SynthesisRequirement};
use serde::{Deserialize, Serialize};

/// Agent-facing coding intent derived from NL or structured input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodingIntent {
    pub function_name: String,
    pub signature: String,
    pub category: String,
    pub description: String,
    pub examples: Vec<CodingExample>,
    pub constraints: Vec<String>,
    pub confidence: f32,
    pub unresolved: Vec<String>,
    pub evidence_entity_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodingExample {
    pub inputs: Vec<CodingValue>,
    pub expected: CodingValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum CodingValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Array(Vec<i64>),
    Pair(i64, i64),
}

impl CodingIntent {
    /// Build from a registry-derived synthesis requirement.
    pub fn from_requirement(req: &SynthesisRequirement) -> Self {
        Self {
            function_name: req.function_name.clone(),
            signature: req.signature.clone(),
            category: req.category.clone(),
            description: req.description.clone(),
            examples: req
                .examples
                .iter()
                .map(|ex| CodingExample {
                    inputs: ex.inputs.iter().map(literal_to_coding_value).collect(),
                    expected: literal_to_coding_value(&ex.expected),
                })
                .collect(),
            constraints: req.constraints.clone(),
            confidence: req.confidence,
            unresolved: req.unresolved.clone(),
            evidence_entity_ids: req.evidence_entity_ids.clone(),
        }
    }

    /// Parse NL via Linguigenesis bridge (production path).
    pub fn from_nl(description: &str) -> Result<Self, BridgeError> {
        let bridge = LinguigenesisBridge::new();
        let req = bridge.nl_to_requirement(description)?;
        Ok(Self::from_requirement(&req))
    }

    /// Convert to solver `Problem` when examples are present.
    pub fn to_problem(&self) -> Result<Problem, String> {
        if self.examples.is_empty() {
            return Err("CodingIntent has no examples".to_string());
        }
        let examples: Vec<Example> = self
            .examples
            .iter()
            .map(|ex| {
                let inputs: Result<Vec<Value>, String> =
                    ex.inputs.iter().map(coding_value_to_benchmark).collect();
                let inputs = inputs?;
                let expected = coding_value_to_benchmark(&ex.expected)?;
                Ok(Example { inputs, expected })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let signature = Box::leak(self.signature.clone().into_boxed_str());
        let category = Box::leak(self.category.clone().into_boxed_str());
        let description = Box::leak(self.description.clone().into_boxed_str());

        Ok(Problem {
            name: self.function_name.clone(),
            category,
            description,
            signature,
            examples,
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: Vec::new(),
        })
    }
}

fn literal_to_coding_value(lit: &LiteralValue) -> CodingValue {
    match lit {
        LiteralValue::Int(v) => CodingValue::Int(*v),
        LiteralValue::Float(v) => CodingValue::Float(*v),
        LiteralValue::Str(s) => CodingValue::Str(s.clone()),
        LiteralValue::Bool(b) => CodingValue::Bool(*b),
        LiteralValue::Array(a) => CodingValue::Array(a.clone()),
        LiteralValue::Pair(a, b) => CodingValue::Pair(*a, *b),
    }
}

fn coding_value_to_benchmark(v: &CodingValue) -> Result<Value, String> {
    Ok(match v {
        CodingValue::Int(n) => Value::Int(*n),
        CodingValue::Float(f) => Value::Float(f.to_bits()),
        CodingValue::Str(s) => Value::Str(s.clone()),
        CodingValue::Bool(b) => Value::Bool(*b),
        CodingValue::Array(a) => Value::int_array(a),
        CodingValue::Pair(a, b) => Value::Pair(*a, *b),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coding_intent_from_add_nl() {
        let intent = CodingIntent::from_nl("add two numbers").expect("add");
        assert!(!intent.examples.is_empty());
        assert_eq!(intent.examples[0].expected, CodingValue::Int(5));
    }
}
