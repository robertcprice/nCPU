//! Probabilistic Synthesis Teacher
//!
//! Detects probabilistic problems and synthesizes programs with uncertainty
//! and randomness using MCMC inference.

use crate::benchmark::{Example, Problem, Value};
use crate::differentiable::DifferentiableMetadata;
use crate::prob::ProbDistribution;
use crate::solver::SolveResult;

/// Detect if a problem requires probabilistic synthesis
///
/// Uses heuristics to identify problems with uncertainty:
/// - Conflicting examples (same inputs, different outputs)
/// - Float outputs without clear deterministic pattern
/// - Sampling-like patterns (many examples with similar structure)
pub fn is_probabilistic_problem(problem: &Problem) -> bool {
    if has_conflicting_scalar_examples(&problem.examples) {
        return true;
    }

    if suggests_uncertainty(&problem.examples) {
        return true;
    }

    if suggests_sampling_process(&problem.examples) {
        return true;
    }

    false
}

fn has_conflicting_scalar_examples(examples: &[Example]) -> bool {
    use std::collections::HashMap;

    let mut input_map: HashMap<Vec<i64>, Vec<i64>> = HashMap::new();

    for ex in examples {
        if !ex
            .inputs
            .iter()
            .all(|input| matches!(input, Value::Int(_) | Value::Float(_) | Value::Bool(_)))
        {
            continue;
        }

        let key = inputs_to_key(&ex.inputs);
        let output = output_to_int(&ex.expected);

        if let Some(existing) = input_map.get(&key) {
            if existing.iter().any(|&o| o != output) {
                return true;
            }
        }

        input_map.entry(key).or_default().push(output);
    }

    false
}

/// Convert inputs to comparable key
fn inputs_to_key(inputs: &[Value]) -> Vec<i64> {
    inputs
        .iter()
        .map(|v| match v {
            Value::Int(i) => *i,
            Value::Float(b) => f64::from_bits(*b) as i64,
            Value::Bool(b) => *b as i64,
            _ => 0,
        })
        .collect()
}

/// Convert output to int for comparison
fn output_to_int(v: &Value) -> i64 {
    match v {
        Value::Int(i) => *i,
        Value::Float(b) => f64::from_bits(*b) as i64,
        Value::Bool(b) => *b as i64,
        _ => 0,
    }
}

/// Check if examples suggest uncertainty (float outputs, non-exact patterns)
fn suggests_uncertainty(examples: &[Example]) -> bool {
    if examples.is_empty() {
        return false;
    }

    // Check if outputs are floats and show variance
    let float_outputs: Vec<f64> = examples
        .iter()
        .filter_map(|ex| match &ex.expected {
            Value::Float(b) => Some(f64::from_bits(*b)),
            _ => None,
        })
        .collect();

    if float_outputs.len() < 3 {
        return false;
    }

    // Check variance
    let mean: f64 = float_outputs.iter().sum::<f64>() / float_outputs.len() as f64;
    let variance: f64 = float_outputs
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / float_outputs.len() as f64;

    // High variance suggests probabilistic behavior
    variance > 0.01
}

/// Check if examples suggest a sampling process
fn suggests_sampling_process(examples: &[Example]) -> bool {
    if examples.len() < 5 {
        return false;
    }

    // Check if we have similar input structures with varied outputs
    // Common in sampling: empty or constant inputs, varied outputs
    let empty_inputs = examples.iter().filter(|ex| ex.inputs.is_empty()).count();

    if empty_inputs > examples.len() / 2 {
        // Many examples with no inputs → likely sampling/Random
        return true;
    }

    // NOTE: bool OUTPUT alone is NOT evidence of a stochastic process. A
    // deterministic predicate (is_even, is_positive: distinct inputs → fixed
    // outputs) has bool output but is NOT a coin flip. Classifying it as Bernoulli
    // made the probabilistic teacher emit a random sampler that FALSE-ACCEPTED
    // (e.g. is_even "solved" by a bias-0.625 rand::Rng). Genuine randomness is
    // caught by has_conflicting_scalar_examples (same input → different output)
    // and the empty-input check above. A deterministic bool predicate must fall
    // through to real synthesis (and honestly fail if no path covers it) rather
    // than be faked. (Removed the `bool_count > len/2` heuristic.)
    false
}

/// Inferred distribution types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionType {
    Bernoulli,
    Normal,
    Categorical,
    Poisson,
    Uniform,
    Unknown,
}

/// Infer distribution type from examples
pub fn infer_distribution_type(examples: &[Example]) -> DistributionType {
    if examples.is_empty() {
        return DistributionType::Unknown;
    }

    let first_output = &examples[0].expected;

    match first_output {
        Value::Bool(_) => DistributionType::Bernoulli,
        Value::Float(_) => infer_float_distribution(examples),
        Value::Int(i) => infer_int_distribution(examples, *i),
        _ => DistributionType::Unknown,
    }
}

/// Infer which float distribution fits best
fn infer_float_distribution(examples: &[Example]) -> DistributionType {
    let float_values: Vec<f64> = examples
        .iter()
        .filter_map(|ex| match &ex.expected {
            Value::Float(b) => Some(f64::from_bits(*b)),
            _ => None,
        })
        .collect();

    if float_values.is_empty() {
        return DistributionType::Unknown;
    }

    // Check if values are roughly uniform in range
    let min = float_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = float_values
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;

    // If values cluster around a mean with variance → Normal
    let mean: f64 = float_values.iter().sum::<f64>() / float_values.len() as f64;
    let variance: f64 = float_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / float_values.len() as f64;

    // Normal if variance is moderate relative to range
    if variance > 0.01 && variance < range.powi(2) / 10.0 {
        DistributionType::Normal
    } else {
        DistributionType::Uniform
    }
}

/// Infer which integer distribution fits best
fn infer_int_distribution(examples: &[Example], _first: i64) -> DistributionType {
    let int_values: Vec<i64> = examples
        .iter()
        .filter_map(|ex| match &ex.expected {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .collect();

    if int_values.is_empty() {
        return DistributionType::Unknown;
    }

    // Check if values are non-negative counts → Poisson
    let all_non_negative = int_values.iter().all(|&x| x >= 0);
    let mean: f64 = int_values.iter().map(|&x| x as f64).sum::<f64>() / int_values.len() as f64;
    let variance: f64 = int_values
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / int_values.len() as f64;

    // Poisson: variance ≈ mean
    if all_non_negative && (variance - mean).abs() < mean * 0.5 {
        return DistributionType::Poisson;
    }

    // Check if values are in a small range → Categorical or Uniform
    let min = int_values.iter().min().copied().unwrap_or(0);
    let max = int_values.iter().max().copied().unwrap_or(0);
    let range = max - min + 1;

    if range <= 10 {
        DistributionType::Categorical
    } else {
        DistributionType::Unknown
    }
}

/// Simple probabilistic model for synthesis
#[derive(Debug, Clone)]
struct SimpleProbModel {
    distribution: ProbDistribution,
}

/// Solve a probabilistic problem
pub fn solve_probabilistic_problem(problem: &Problem) -> SolveResult {
    eprintln!("[probabilistic] analyzing problem: {}", problem.name);

    // Infer distribution type
    let dist_type = infer_distribution_type(&problem.examples);
    eprintln!("[probabilistic] inferred distribution: {:?}", dist_type);

    // Create distribution
    let dist = match create_distribution(dist_type, &problem.examples) {
        Ok(d) => d,
        Err(e) => {
            return SolveResult {
                success: false,
                code: String::new(),
                method: "probabilistic".to_string(),
                error: Some(format!("Failed to create distribution: {}", e)),
                metadata: DifferentiableMetadata::default(),
            };
        }
    };

    // Extract parameters
    let params = extract_params(&dist);
    eprintln!("[probabilistic] learned parameters: {:?}", params);

    // Generate code
    let code =
        crate::prob::codegen::generate_probabilistic_code(&dist, &params, problem.function_name());

    SolveResult {
        success: true,
        code,
        method: "probabilistic".to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    }
}

/// Create a distribution from type and examples
fn create_distribution(
    dist_type: DistributionType,
    examples: &[Example],
) -> Result<ProbDistribution, String> {
    match dist_type {
        DistributionType::Bernoulli => {
            let true_count = examples
                .iter()
                .filter(|ex| matches!(&ex.expected, Value::Bool(true)))
                .count();
            let p = (true_count as f64) / (examples.len() as f64);
            Ok(ProbDistribution::Bernoulli {
                p: p.max(0.01).min(0.99),
            })
        }
        DistributionType::Normal => {
            let values: Vec<f64> = examples
                .iter()
                .filter_map(|ex| match &ex.expected {
                    Value::Float(b) => Some(f64::from_bits(*b)),
                    _ => None,
                })
                .collect();
            if values.is_empty() {
                return Err("No float values for Normal distribution".to_string());
            }
            let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
            let variance: f64 =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std = variance.sqrt().max(0.01);
            Ok(ProbDistribution::Normal { mean, std })
        }
        DistributionType::Categorical => {
            let values: Vec<i64> = examples
                .iter()
                .filter_map(|ex| match &ex.expected {
                    Value::Int(i) => Some(*i),
                    _ => None,
                })
                .collect();
            if values.is_empty() {
                return Err("No int values for Categorical distribution".to_string());
            }
            let max_val = values.iter().max().copied().unwrap_or(0);
            let mut counts = vec![0usize; (max_val + 1) as usize];
            for &v in &values {
                counts[v as usize] += 1;
            }
            let total = values.len() as f64;
            let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();
            Ok(ProbDistribution::Categorical(probs))
        }
        DistributionType::Poisson => {
            let values: Vec<i64> = examples
                .iter()
                .filter_map(|ex| match &ex.expected {
                    Value::Int(i) => Some(*i),
                    _ => None,
                })
                .collect();
            if values.is_empty() {
                return Err("No int values for Poisson distribution".to_string());
            }
            let lambda: f64 = values.iter().map(|&x| x as f64).sum::<f64>() / values.len() as f64;
            Ok(ProbDistribution::Poisson {
                lambda: lambda.max(0.1),
            })
        }
        DistributionType::Uniform => {
            let values: Vec<f64> = examples
                .iter()
                .filter_map(|ex| match &ex.expected {
                    Value::Float(b) => Some(f64::from_bits(*b)),
                    Value::Int(i) => Some(*i as f64),
                    _ => None,
                })
                .collect();
            if values.is_empty() {
                return Err("No values for Uniform distribution".to_string());
            }
            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            Ok(ProbDistribution::Uniform {
                low: min,
                high: max,
            })
        }
        DistributionType::Unknown => Err("Unknown distribution type".to_string()),
    }
}

/// Extract parameters from distribution
fn extract_params(dist: &ProbDistribution) -> Vec<f64> {
    match dist {
        ProbDistribution::Bernoulli { p } => vec![*p],
        ProbDistribution::Normal { mean, std } => vec![*mean, *std],
        ProbDistribution::Categorical(probs) => probs.clone(),
        ProbDistribution::Poisson { lambda } => vec![*lambda],
        ProbDistribution::Uniform { low, high } => vec![*low, *high],
        _ => vec![0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_conflicting_examples() {
        let examples = vec![
            Example {
                inputs: vec![],
                expected: Value::Bool(true),
            },
            Example {
                inputs: vec![],
                expected: Value::Bool(false),
            },
        ];
        assert!(has_conflicting_scalar_examples(&examples));
    }

    #[test]
    fn test_array_variance_does_not_count_as_conflict() {
        let examples = vec![
            Example {
                inputs: vec![Value::int_array(&[1, 2, 3])],
                expected: Value::Int(3),
            },
            Example {
                inputs: vec![Value::int_array(&[-5])],
                expected: Value::Int(0),
            },
        ];
        assert!(!has_conflicting_scalar_examples(&examples));
    }

    #[test]
    fn test_bernoulli_detection() {
        let examples = vec![
            Example {
                inputs: vec![],
                expected: Value::Bool(true),
            },
            Example {
                inputs: vec![],
                expected: Value::Bool(false),
            },
            Example {
                inputs: vec![],
                expected: Value::Bool(true),
            },
        ];
        assert!(is_probabilistic_problem(&create_test_problem(examples)));
    }

    #[test]
    fn test_infer_bernoulli() {
        let examples = vec![
            Example {
                inputs: vec![],
                expected: Value::Bool(true),
            },
            Example {
                inputs: vec![],
                expected: Value::Bool(false),
            },
            Example {
                inputs: vec![],
                expected: Value::Bool(true),
            },
        ];
        let dist_type = infer_distribution_type(&examples);
        assert_eq!(dist_type, DistributionType::Bernoulli);
    }

    fn create_test_problem(examples: Vec<Example>) -> Problem {
        Problem {
            name: "test".to_string(),
            category: "test",
            description: "",
            signature: "fn test() -> bool",
            examples,
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: Vec::new(),
        }
    }
}
