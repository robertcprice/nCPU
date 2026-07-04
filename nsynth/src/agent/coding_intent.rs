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

    /// Like [`Self::from_nl`], but accepts comprehend partials when clarification
    /// is needed (registry ops with sparse examples). Used by G5 repair fixtures.
    pub fn from_nl_lenient(description: &str) -> Result<Self, BridgeError> {
        let bridge = LinguigenesisBridge::new();
        let req = match bridge.nl_to_requirement(description) {
            Ok(req) => req,
            Err(BridgeError::ClarificationNeeded { partial, .. }) => partial,
            Err(err) => return Err(err),
        };
        Ok(Self::from_requirement(&req))
    }

    /// REFERENCE-IMPLEMENTATION front door: build a solver `Problem` from a
    /// runnable reference alone — no hand-authored examples required.
    ///
    /// This is the agent-facing sibling of [`Self::to_problem`]: where
    /// `to_problem` hard-fails on empty examples, this manufactures the seed
    /// I/O examples by running the reference (via
    /// [`crate::benchmark::problem_from_reference`], which owns the holdout
    /// sampling machinery) and keeps `reference_code` set so the strict verifier
    /// does differential testing against the reference. A spec given ONLY a
    /// reference now synthesizes + verifies.
    ///
    /// `to_problem` is intentionally left unchanged — this is additive.
    pub fn problem_from_reference(
        name: &str,
        signature: &str,
        reference_code: &str,
    ) -> Result<Problem, String> {
        // Problem fields are `&'static str`; leak the user input at the front
        // door (the existing pattern in `to_problem` for signature/category/
        // description). Bounded, per-spec — acceptable for a front-door ctor.
        let signature: &'static str = Box::leak(signature.to_string().into_boxed_str());
        let reference_code: &'static str =
            Box::leak(reference_code.to_string().into_boxed_str());
        crate::benchmark::problem_from_reference(name, signature, reference_code)
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
        // Struct literals postdate this intake path and have no CodingValue
        // representation; they never occur in the PBE benchmarks. Map to 0 to
        // keep the conversion total (pre-Struct behavior: unreachable).
        LiteralValue::Struct(_) => CodingValue::Int(0),
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

/// Unified spec front door — the four ways a synthesis task can be specified.
///
/// Collapses the previously-scattered intake paths into one sum type so a caller
/// holds "a spec" without committing to how it was expressed:
///   - [`Spec::Examples`] — classic PBE: input/output pairs (the existing path).
///   - [`Spec::Reference`] — a runnable reference whose behavior IS the spec;
///     seed examples are manufactured by running it
///     ([`crate::benchmark::problem_from_reference`]).
///   - [`Spec::Property`] — a predicate the output must satisfy, used as the
///     verify oracle ([`crate::benchmark::verify_code_against_property`]).
///   - [`Spec::Nl`] — a natural-language description, resolved via the bridge.
///
/// `Examples`/`Reference`/`Nl` reduce to a solver [`Problem`] via
/// [`Spec::to_problem`]; `Property` is a *verification* spec (no fixed outputs to
/// search toward) so it exposes [`Spec::verify`] instead.
#[derive(Debug, Clone)]
pub enum Spec {
    Examples(CodingIntent),
    Reference {
        name: String,
        signature: String,
        code: String,
    },
    Property {
        candidate_name: String,
        candidate_signature: String,
        predicate_name: String,
        predicate_signature: String,
        predicate_code: String,
    },
    Nl(String),
}

impl Spec {
    /// Reduce a spec to a solver `Problem`, for the arms that pin down outputs
    /// (Examples / Reference / NL). `Property` returns `Err` — it is verified,
    /// not solved-toward (use [`Spec::verify`]).
    pub fn to_problem(&self) -> Result<Problem, String> {
        match self {
            Spec::Examples(intent) => intent.to_problem(),
            Spec::Reference {
                name,
                signature,
                code,
            } => CodingIntent::problem_from_reference(name, signature, code),
            Spec::Nl(text) => {
                let intent = CodingIntent::from_nl(text).map_err(|e| e.to_string())?;
                intent.to_problem()
            }
            Spec::Property { .. } => Err(
                "a property spec has no fixed outputs to synthesize toward; use Spec::verify"
                    .to_string(),
            ),
        }
    }

    /// Verify a candidate against a `Property` spec's predicate oracle. Only
    /// meaningful for [`Spec::Property`]; the other arms verify through
    /// example/reference matching inside the solver pipeline and return `Err`.
    pub fn verify(&self, candidate_code: &str) -> Result<(), String> {
        match self {
            Spec::Property {
                candidate_name,
                candidate_signature,
                predicate_name,
                predicate_signature,
                predicate_code,
            } => {
                // `verify_code_against_property` stores the signatures on
                // temporary problems for arg coercion, so they must be 'static.
                // Leak at the front door (bounded, per-spec) — the existing
                // pattern in `problem_from_reference`.
                let candidate_signature: &'static str =
                    Box::leak(candidate_signature.clone().into_boxed_str());
                let predicate_signature: &'static str =
                    Box::leak(predicate_signature.clone().into_boxed_str());
                crate::benchmark::verify_code_against_property(
                    candidate_name,
                    candidate_signature,
                    candidate_code,
                    predicate_name,
                    predicate_signature,
                    predicate_code,
                )
            }
            _ => Err("Spec::verify is only meaningful for a Property spec".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coding_intent_from_nl_lenient_accepts_sparse_registry_partial() {
        let intent =
            CodingIntent::from_nl_lenient("subtract two numbers").expect("partial subtract");
        assert!(!intent.examples.is_empty());
        assert!(!intent.unresolved.is_empty() || intent.confidence < 1.0);
    }

    #[test]
    fn coding_intent_from_add_nl() {
        let intent = CodingIntent::from_nl("add two numbers").expect("add");
        assert!(!intent.examples.is_empty());
        assert_eq!(intent.examples[0].expected, CodingValue::Int(5));
    }

    #[test]
    fn problem_from_reference_no_hand_examples_synthesizes_and_verifies() {
        // Reference-only spec: NO hand examples supplied.
        let problem = CodingIntent::problem_from_reference(
            "double",
            "fn double(x: i64) -> i64",
            "fn double(x: i64) -> i64 { return x * 2; }",
        )
        .expect("reference intake should build a problem");

        // Seeds were manufactured by running the reference.
        assert!(!problem.examples.is_empty());
        for example in &problem.examples {
            let Value::Int(input) = example.inputs[0] else {
                panic!("expected int input");
            };
            assert_eq!(example.expected, Value::Int(input * 2));
        }
        assert!(!problem.reference_code.is_empty());

        // Equivalent candidate verifies; non-equivalent is rejected.
        assert!(crate::runtime::verify_problem_code_strict(
            &problem,
            "fn double(x: i64) -> i64 { return x + x; }",
        )
        .is_ok());
        assert!(crate::runtime::verify_problem_code_strict(
            &problem,
            "fn double(x: i64) -> i64 { return x + 1; }",
        )
        .is_err());
    }

    #[test]
    fn spec_reference_arm_reduces_to_problem() {
        let spec = Spec::Reference {
            name: "double".into(),
            signature: "fn double(x: i64) -> i64".into(),
            code: "fn double(x: i64) -> i64 { return x * 2; }".into(),
        };
        let problem = spec
            .to_problem()
            .expect("a reference spec must reduce to a problem");
        assert!(!problem.examples.is_empty());
        // A property spec has no fixed outputs, so to_problem must refuse.
        let property = Spec::Property {
            candidate_name: "inc".into(),
            candidate_signature: "fn inc(x: i64) -> i64".into(),
            predicate_name: "gt".into(),
            predicate_signature: "fn gt(x: i64, out: i64) -> i64".into(),
            predicate_code: "fn gt(x: i64, out: i64) -> i64 { if out > x { return 1; } return 0; }"
                .into(),
        };
        assert!(property.to_problem().is_err());
    }

    #[test]
    fn spec_property_arm_verifies_via_predicate_oracle() {
        let spec = Spec::Property {
            candidate_name: "inc".into(),
            candidate_signature: "fn inc(x: i64) -> i64".into(),
            predicate_name: "gt".into(),
            predicate_signature: "fn gt(x: i64, out: i64) -> i64".into(),
            predicate_code: "fn gt(x: i64, out: i64) -> i64 { if out > x { return 1; } return 0; }"
                .into(),
        };
        // Satisfying candidate verifies; violating candidate is rejected.
        spec.verify("fn inc(x: i64) -> i64 { return x + 1; }")
            .expect("a satisfying candidate must verify against the property");
        assert!(spec
            .verify("fn inc(x: i64) -> i64 { return x - 1; }")
            .is_err());
    }
}
