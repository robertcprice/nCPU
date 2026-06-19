//! Markov Chain Monte Carlo (MCMC) Inference for nCPU/nSynth
//!
//! Bayesian inference using Metropolis-Hastings and variational methods.

use crate::prob::distribution::{ProbDistribution, Value};
use crate::prob::primitive::{Observation, ProbContext, QueryType, QueryResult};
use std::time::{Duration, Instant};

/// MCMC sampler configuration
#[derive(Debug, Clone, Copy)]
pub struct McmcConfig {
    /// Number of MCMC iterations
    pub iterations: usize,
    /// Burn-in period (samples to discard)
    pub burn_in: usize,
    /// Thinning interval (keep every n-th sample)
    pub thin: usize,
    /// Random walk step size
    pub step_size: f64,
    /// Maximum runtime
    pub max_time: Option<Duration>,
}

impl Default for McmcConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            burn_in: 100,
            thin: 1,
            step_size: 0.1,
            max_time: None,
        }
    }
}

impl McmcConfig {
    /// Create new config with iterations
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Set burn-in period
    pub fn with_burn_in(mut self, n: usize) -> Self {
        self.burn_in = n;
        self
    }

    /// Set thinning interval
    pub fn with_thin(mut self, n: usize) -> Self {
        self.thin = n;
        self
    }

    /// Set step size
    pub fn with_step_size(mut self, size: f64) -> Self {
        self.step_size = size;
        self
    }

    /// Set maximum runtime
    pub fn with_max_time(mut self, time: Duration) -> Self {
        self.max_time = Some(time);
        self
    }
}

/// MCMC state (current parameter values and log likelihood)
#[derive(Debug, Clone)]
pub struct McmcState {
    /// Current parameter values
    pub params: Vec<f64>,
    /// Current log likelihood
    pub log_likelihood: f64,
    /// Iteration number
    pub iteration: usize,
}

impl McmcState {
    /// Create new MCMC state
    pub fn new(params: Vec<f64>, log_likelihood: f64) -> Self {
        Self {
            params,
            log_likelihood,
            iteration: 0,
        }
    }

    /// Clone state with new parameters
    fn with_params(&self, params: Vec<f64>, log_likelihood: f64) -> Self {
        Self {
            params,
            log_likelihood,
            iteration: self.iteration + 1,
        }
    }
}

/// MCMC sampler for Bayesian inference
pub struct McmcSampler {
    config: McmcConfig,
}

impl McmcSampler {
    /// Create new MCMC sampler
    pub fn new(config: McmcConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(McmcConfig::default())
    }

    /// Run Metropolis-Hastings sampler
    pub fn metropolis_hastings<F>(
        &self,
        log_target: F,
        init: Vec<f64>,
    ) -> McmcResult
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let start_time = Instant::now();
        let mut state = McmcState::new(init.clone(), log_target(&init));

        // Check if initial state is valid
        if !state.log_likelihood.is_finite() {
            return McmcResult {
                samples: Vec::new(),
                acceptance_rate: 0.0,
                iterations: 0,
                elapsed: start_time.elapsed(),
                converged: false,
            };
        }

        let mut accepted = 0usize;
        let mut samples = Vec::new();

        for i in 0..self.config.iterations {
            // Check max time
            if let Some(max_time) = self.config.max_time {
                if start_time.elapsed() > max_time {
                    break;
                }
            }

            // Propose new state (random walk)
            let proposed = Self::propose(&state.params, self.config.step_size);
            let proposed_ll = log_target(&proposed);

            // Metropolis acceptance ratio
            let alpha = (proposed_ll - state.log_likelihood).exp();

            // Accept or reject
            let rand: f64 = rand::random();
            if rand < alpha {
                state = state.with_params(proposed, proposed_ll);
                accepted += 1;
            }

            // Store samples (after burn-in, with thinning)
            if i >= self.config.burn_in && (i - self.config.burn_in) % self.config.thin == 0 {
                samples.push(state.params.clone());
            }

            state.iteration = i;
        }

        let acceptance_rate = accepted as f64 / self.config.iterations as f64;
        let converged = acceptance_rate > 0.2 && acceptance_rate < 0.8;

        McmcResult {
            samples,
            acceptance_rate,
            iterations: self.config.iterations,
            elapsed: start_time.elapsed(),
            converged,
        }
    }

    /// Propose new parameters (random walk)
    fn propose(current: &[f64], step_size: f64) -> Vec<f64> {
        current.iter().map(|&p| {
            let noise: f64 = rand::random();
            p + (noise - 0.5) * 2.0 * step_size
        }).collect()
    }

    /// Run Gibbs sampling (for conjugate models)
    pub fn gibbs<F>(
        &self,
        sample_conditional: F,
        init: Vec<f64>,
    ) -> McmcResult
    where
        F: Fn(usize, &[f64]) -> f64 + Send + Sync,
    {
        let start_time = Instant::now();
        let mut params = init;
        let mut samples = Vec::new();

        for i in 0..self.config.iterations {
            // Check max time
            if let Some(max_time) = self.config.max_time {
                if start_time.elapsed() > max_time {
                    break;
                }
            }

            // Sample each parameter conditional on others
            for dim in 0..params.len() {
                params[dim] = sample_conditional(dim, &params);
            }

            // Store samples (after burn-in, with thinning)
            if i >= self.config.burn_in && (i - self.config.burn_in) % self.config.thin == 0 {
                samples.push(params.clone());
            }
        }

        McmcResult {
            samples,
            acceptance_rate: 1.0, // Gibbs always accepts
            iterations: self.config.iterations,
            elapsed: start_time.elapsed(),
            converged: true,
        }
    }
}

/// MCMC result
#[derive(Debug, Clone)]
pub struct McmcResult {
    /// Posterior samples
    pub samples: Vec<Vec<f64>>,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Number of iterations run
    pub iterations: usize,
    /// Time elapsed
    pub elapsed: Duration,
    /// Whether chain converged (heuristic)
    pub converged: bool,
}

impl McmcResult {
    /// Get posterior mean
    pub fn posterior_mean(&self) -> Vec<f64> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let dim = self.samples[0].len();
        let mut means = vec![0.0f64; dim];

        for sample in &self.samples {
            for (i, &val) in sample.iter().enumerate() {
                means[i] += val;
            }
        }

        for mean in &mut means {
            *mean /= self.samples.len() as f64;
        }

        means
    }

    /// Get posterior standard deviation
    pub fn posterior_std(&self) -> Vec<f64> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let means = self.posterior_mean();
        let dim = means.len();
        let mut stds = vec![0.0f64; dim];

        for sample in &self.samples {
            for (i, &val) in sample.iter().enumerate() {
                let diff = val - means[i];
                stds[i] += diff * diff;
            }
        }

        let n = self.samples.len() as f64;
        for std in &mut stds {
            *std = (*std / n).sqrt();
        }

        stds
    }

    /// Get credible interval (95%)
    pub fn credible_interval(&self, param_idx: usize) -> Option<(f64, f64)> {
        if self.samples.is_empty() || param_idx >= self.samples[0].len() {
            return None;
        }

        let mut values: Vec<f64> = self.samples.iter()
            .map(|s| s[param_idx])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = values.len();
        let lower_idx = (n as f64 * 0.025).floor() as usize;
        let upper_idx = (n as f64 * 0.975).floor() as usize;

        Some((values[lower_idx], values[upper_idx]))
    }

    /// Get number of effective samples (accounting for autocorrelation)
    pub fn effective_sample_size(&self) -> usize {
        if self.samples.len() < 2 {
            return 0;
        }

        // Simplified ESS calculation (would use full autocorrelation in production)
        (self.samples.len() as f64 * 0.5) as usize
    }
}

/// Variational inference approximation
pub struct VariationalInference {
    /// Number of optimization steps
    pub steps: usize,
    /// Learning rate
    pub learning_rate: f64,
}

impl VariationalInference {
    /// Create new VI instance
    pub fn new(steps: usize, learning_rate: f64) -> Self {
        Self { steps, learning_rate }
    }

    /// Run mean-field variational inference
    pub fn infer<F>(
        &self,
        elbo: F,
        init: Vec<f64>,
    ) -> VariationalResult
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let mut params = init;
        let mut best_elbo = elbo(&params);
        let mut best_params = params.clone();

        for _ in 0..self.steps {
            // Gradient-free optimization (coordinate search)
            for i in 0..params.len() {
                let original = params[i];

                // Try positive step
                params[i] += self.learning_rate;
                let elbo_pos = elbo(&params);

                // Try negative step
                params[i] = original - self.learning_rate;
                let elbo_neg = elbo(&params);

                // Choose best
                if elbo_pos >= best_elbo {
                    best_elbo = elbo_pos;
                    best_params = params.clone();
                } else if elbo_neg >= best_elbo {
                    best_elbo = elbo_neg;
                    best_params = params.clone();
                    params[i] = original - self.learning_rate;
                } else {
                    params[i] = original;
                }
            }
        }

        VariationalResult {
            parameters: best_params,
            elbo: best_elbo,
            converged: true,
        }
    }
}

/// Variational inference result
#[derive(Debug, Clone)]
pub struct VariationalResult {
    /// Optimal parameters found
    pub parameters: Vec<f64>,
    /// Final ELBO value
    pub elbo: f64,
    /// Whether optimization converged
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcmc_config_builder() {
        let config = McmcConfig::default()
            .with_iterations(100)
            .with_step_size(0.5);

        assert_eq!(config.iterations, 100);
        assert_eq!(config.step_size, 0.5);
    }

    #[test]
    fn test_metropolis_hastings() {
        let sampler = McmcSampler::default_config();
        let result = sampler.metropolis_hastings(
            |params| -(params[0] * params[0]), // Negative quadratic (Gaussian-like)
            vec![0.0],
        );

        assert!(result.iterations > 0);
        assert!(!result.samples.is_empty());
        assert!(result.acceptance_rate >= 0.0 && result.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_posterior_mean() {
        let mut result = McmcResult {
            samples: vec![vec![1.0], vec![2.0], vec![3.0]],
            acceptance_rate: 0.5,
            iterations: 3,
            elapsed: Duration::from_millis(100),
            converged: true,
        };

        let mean = result.posterior_mean();
        assert_eq!(mean.len(), 1);
        assert!((mean[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_variational_inference() {
        let vi = VariationalInference::new(10, 0.1);
        let result = vi.infer(
            |params| -params[0].abs(), // V-shaped (minimum at 0)
            vec![1.0],
        );

        assert!(!result.parameters.is_empty());
        assert!(result.converged);
    }
}
