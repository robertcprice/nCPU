//! Probabilistic Synthesis for nCPU/nSynth
//!
//! Bayesian program synthesis with probabilistic modeling, MCMC inference,
//! and uncertainty quantification.

pub mod distribution;
pub mod primitive;
pub mod mcmc;
pub mod codegen;

pub use distribution::{ProbDistribution, Value as DistValue};
pub use codegen::generate_probabilistic_code;
pub use primitive::{sample, observe, Observation, ProbContext, QueryType, QueryResult, ProbabilisticModel};
pub use mcmc::{McmcSampler, McmcConfig, McmcState, McmcResult, VariationalInference, VariationalResult};

/// Probabilistic synthesis configuration
#[derive(Debug, Clone)]
pub struct ProbConfig {
    /// Maximum MCMC iterations
    pub max_iterations: usize,
    /// Burn-in period
    pub burn_in: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for ProbConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            burn_in: 100,
            convergence_threshold: 0.01,
            seed: None,
        }
    }
}

impl ProbConfig {
    /// Create new probabilistic config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max iterations
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set burn-in period
    pub fn with_burn_in(mut self, n: usize) -> Self {
        self.burn_in = n;
        self
    }

    /// Set convergence threshold
    pub fn with_convergence_threshold(mut self, t: f64) -> Self {
        self.convergence_threshold = t;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Run Bayesian inference on a probabilistic model
pub fn infer<F>(
    log_target: F,
    init: Vec<f64>,
    config: ProbConfig,
) -> McmcResult
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let sampler = McmcSampler::new(
        McmcConfig::default()
            .with_iterations(config.max_iterations)
            .with_burn_in(config.burn_in)
    );

    sampler.metropolis_hastings(log_target, init)
}

/// Synthesize a probabilistic program from observations
pub fn synthesize_probabilistic(
    observations: &[Observation],
    _examples: &[crate::benchmark::Example],
) -> Result<String, String> {
    // Analyze observations to infer model structure
    // Use examples to guide synthesis
    // Generate program with sample/observe primitives

    if observations.is_empty() {
        return Err("No observations provided".to_string());
    }

    // In production, would:
    // 1. Detect patterns in observations (coin flips, measurements, etc.)
    // 2. Select appropriate distribution types
    // 3. Synthesize program structure
    // 4. Run MCMC to learn parameters
    // 5. Generate executable code

    Ok("// Probabilistic program synthesis placeholder".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prob_config() {
        let config = ProbConfig::new()
            .with_max_iterations(500)
            .with_burn_in(50);

        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.burn_in, 50);
    }

    #[test]
    fn test_infer() {
        let result = infer(
            |params| -(params[0] - 5.0).powi(2), // Peak at x=5
            vec![0.0],
            ProbConfig::default(),
        );

        assert!(!result.samples.is_empty());
        let mean = result.posterior_mean();
        assert!((mean[0] - 5.0) < 1.0); // Should be near 5
    }

    #[test]
    fn test_synthesize_probabilistic() {
        let obs = vec![
            Observation::new("coin", DistValue::Bool(true), ProbDistribution::Bernoulli { p: 0.5 }),
        ];

        let examples = vec![
            crate::benchmark::Example {
                inputs: vec![crate::benchmark::Value::Int(1)],
                expected: crate::benchmark::Value::Int(1),
            },
        ];

        let result = synthesize_probabilistic(&obs, &examples);
        assert!(result.is_ok());
    }
}
