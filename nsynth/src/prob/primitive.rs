//! Probabilistic Primitives for nCPU/nSynth
//!
//! Sample and observe operations for probabilistic program synthesis.

use crate::prob::distribution::{ProbDistribution, Value};

/// Sample a value from a distribution
pub fn sample(dist: ProbDistribution) -> Value {
    dist.sample()
}

/// Observe a value under a distribution (compute log likelihood)
pub fn observe(dist: ProbDistribution, x: Value) -> f64 {
    dist.log_prob(&x)
}

/// Conditioning on observed evidence
#[derive(Debug, Clone)]
pub struct Observation {
    /// Variable name
    pub variable: String,
    /// Observed value
    pub value: Value,
    /// Distribution used for observation
    pub distribution: ProbDistribution,
}

impl Observation {
    /// Create a new observation
    pub fn new(variable: impl Into<String>, value: Value, distribution: ProbDistribution) -> Self {
        Self {
            variable: variable.into(),
            value,
            distribution,
        }
    }

    /// Get log likelihood of this observation
    pub fn log_likelihood(&self) -> f64 {
        self.distribution.log_prob(&self.value)
    }

    /// Check if this observation is consistent (log prob > -inf)
    pub fn is_consistent(&self) -> bool {
        self.log_likelihood().is_finite()
    }
}

/// Probabilistic context for inference
#[derive(Debug, Clone)]
pub struct ProbContext {
    /// Observed evidence
    pub observations: Vec<Observation>,
    /// Sampled variables
    pub sampled: Vec<(String, Value)>,
}

impl ProbContext {
    /// Create a new probabilistic context
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            sampled: Vec::new(),
        }
    }

    /// Add an observation
    pub fn observe(&mut self, obs: Observation) {
        self.observations.push(obs);
    }

    /// Add a sampled variable
    pub fn sample(&mut self, name: impl Into<String>, value: Value) {
        self.sampled.push((name.into(), value));
    }

    /// Get total log likelihood of all observations
    pub fn log_likelihood(&self) -> f64 {
        self.observations.iter()
            .map(|obs| obs.log_likelihood())
            .sum()
    }

    /// Check if all observations are consistent
    pub fn is_consistent(&self) -> bool {
        self.observations.iter().all(|obs| obs.is_consistent())
    }

    /// Get number of observations
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Get number of sampled variables
    pub fn sampled_count(&self) -> usize {
        self.sampled.len()
    }
}

impl Default for ProbContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Query type for probabilistic inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Maximum a posteriori (MAP) estimate
    MAP,
    /// Posterior mean
    Mean,
    /// Posterior samples
    Samples,
    /// Full posterior distribution
    Full,
}

/// Query result from probabilistic inference
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// Single value result
    Single(Value),
    /// Multiple samples
    Samples(Vec<Value>),
    /// Distribution
    Distribution(ProbDistribution),
    /// Mean with variance
    MeanVariance { mean: f64, variance: f64 },
}

impl QueryResult {
    /// Get single value from result
    pub fn as_single(&self) -> Option<&Value> {
        match self {
            QueryResult::Single(v) => Some(v),
            QueryResult::Samples(samples) => samples.first(),
            _ => None,
        }
    }

    /// Get samples from result
    pub fn as_samples(&self) -> Option<&[Value]> {
        match self {
            QueryResult::Samples(samples) => Some(samples),
            _ => None,
        }
    }
}

/// Probabilistic model for synthesis
pub trait ProbabilisticModel {
    /// Generate samples from the model
    fn sample(&self, context: &mut ProbContext) -> Value;

    /// Compute log likelihood of observations
    fn log_likelihood(&self, observations: &[Observation]) -> f64;

    /// Get model parameters (for inference)
    fn parameters(&self) -> Vec<f64>;

    /// Set model parameters (from inference)
    fn set_parameters(&mut self, params: &[f64]) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample() {
        let dist = ProbDistribution::Uniform { low: 0.0, high: 1.0 };
        let val = sample(dist);
        if let Value::Float(f) = val {
            assert!(f >= 0.0 && f <= 1.0);
        } else {
            panic!("Expected Float value");
        }
    }

    #[test]
    fn test_observe() {
        let dist = ProbDistribution::Bernoulli { p: 0.5 };
        let ll = observe(dist.clone(), Value::Bool(true));
        assert!((ll - 0.5_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_observation() {
        let obs = Observation::new(
            "coin",
            Value::Bool(true),
            ProbDistribution::Bernoulli { p: 0.7 }
        );

        assert_eq!(obs.variable, "coin");
        assert!(obs.is_consistent());
        assert!((obs.log_likelihood() - 0.7_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_prob_context() {
        let mut ctx = ProbContext::new();
        ctx.observe(Observation::new(
            "x",
            Value::Int(5),
            ProbDistribution::Poisson { lambda: 5.0 }
        ));

        assert_eq!(ctx.observation_count(), 1);
        assert!(ctx.is_consistent());
    }

    #[test]
    fn test_query_result() {
        let result = QueryResult::Single(Value::Int(42));
        assert_eq!(result.as_single(), Some(&Value::Int(42)));
        assert!(result.as_samples().is_none());

        let samples = QueryResult::Samples(vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(samples.as_samples(), Some(&vec![Value::Int(1), Value::Int(2)][..]));
    }
}
