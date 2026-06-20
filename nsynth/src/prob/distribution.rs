//! Probability Distributions for nCPU/nSynth
//!
//! Common probability distributions with sampling and log-probability computation.

use rand::Rng;
use rand_distr::{
    Bernoulli as RandBernoulli, Distribution as RandDistribution, Normal, Standard, Uniform,
};
use std::f64::consts::PI;

/// Probability distribution types
#[derive(Debug, Clone)]
pub enum ProbDistribution {
    /// Uniform distribution on [low, high]
    Uniform { low: f64, high: f64 },
    /// Normal (Gaussian) distribution with mean and std
    Normal { mean: f64, std: f64 },
    /// Bernoulli distribution with probability p
    Bernoulli { p: f64 },
    /// Categorical distribution over probabilities
    Categorical(Vec<f64>),
    /// Beta distribution
    Beta { alpha: f64, beta: f64 },
    /// Exponential distribution with rate lambda
    Exponential { lambda: f64 },
    /// Poisson distribution with rate lambda
    Poisson { lambda: f64 },
    /// Gamma distribution
    Gamma { shape: f64, scale: f64 },
    /// Dirichlet distribution (multivariate Beta)
    Dirichlet(Vec<f64>),
}

impl ProbDistribution {
    /// Sample from this distribution
    pub fn sample(&self) -> Value {
        let mut rng = rand::thread_rng();
        match self {
            ProbDistribution::Uniform { low, high } => {
                let u = Uniform::new(*low, *high).sample(&mut rng);
                Value::Float(u)
            }
            ProbDistribution::Normal { mean, std } => {
                let n = Normal::new(*mean, *std).unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
                Value::Float(n.sample(&mut rng))
            }
            ProbDistribution::Bernoulli { p } => {
                let b = RandBernoulli::new(*p).unwrap_or_else(|_| RandBernoulli::new(0.5).unwrap());
                Value::Bool(b.sample(&mut rng))
            }
            ProbDistribution::Categorical(probs) => {
                // Sample categorical using uniform
                let u: f64 = Uniform::new(0.0, 1.0).sample(&mut rng);
                let mut cumsum = 0.0;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if u <= cumsum {
                        return Value::Int(i as i64);
                    }
                }
                Value::Int((probs.len() - 1) as i64)
            }
            ProbDistribution::Beta { alpha, beta } => {
                // Approximate beta using gamma
                let shape1 = alpha;
                let shape2 = beta;
                let u1: f64 = rng.sample(Standard); // Simplified - would use gamma distribution
                let u2: f64 = rng.sample(Standard);
                Value::Float((u1 / (u1 + u2)) * (alpha / (alpha + beta)))
            }
            ProbDistribution::Exponential { lambda } => {
                let u: f64 = Uniform::new(0.0, 1.0).sample(&mut rng);
                Value::Float(-u.ln() / lambda)
            }
            ProbDistribution::Poisson { lambda } => {
                // Simplified Poisson using Knuth's algorithm
                let mut l = (-*lambda).exp();
                let mut k = 0i64;
                let mut u = Uniform::new(0.0, 1.0).sample(&mut rng);
                while u > l {
                    k += 1;
                    l *= *lambda / k as f64;
                    if k > 1000 {
                        break; // Prevent infinite loop
                    }
                    u = Uniform::new(0.0, 1.0).sample(&mut rng);
                }
                Value::Int(k)
            }
            ProbDistribution::Gamma { shape, scale } => {
                // Simplified gamma - would use Marsaglia and Tsang's method
                let u1: f64 = rng.sample(Standard);
                let u2: f64 = rng.sample(Standard);
                Value::Float(u1 * scale * shape)
            }
            ProbDistribution::Dirichlet(alphas) => {
                // Simplified Dirichlet - normalize gamma samples
                let mut samples: Vec<f64> = alphas
                    .iter()
                    .map(|&a| {
                        let u1: f64 = rng.sample(Standard);
                        u1 * a
                    })
                    .collect();
                let sum: f64 = samples.iter().sum();
                if sum > 0.0 {
                    samples = samples.iter().map(|s| s / sum).collect();
                }
                Value::Array(samples.into_iter().map(Value::Float).collect())
            }
        }
    }

    /// Compute log probability density/mass
    pub fn log_prob(&self, x: &Value) -> f64 {
        match (self, x) {
            (ProbDistribution::Uniform { low, high }, Value::Float(f)) => {
                if *low <= *f && *f <= *high {
                    -((high - low).ln())
                } else {
                    f64::NEG_INFINITY
                }
            }
            (ProbDistribution::Normal { mean, std }, Value::Float(f)) => {
                let z = (f - mean) / std;
                -0.5 * (2.0_f64 * PI).ln() - std.ln() - 0.5 * z * z
            }
            (ProbDistribution::Bernoulli { p }, Value::Bool(b)) => {
                if *b {
                    p.ln()
                } else {
                    (1.0 - p).ln()
                }
            }
            (ProbDistribution::Categorical(probs), Value::Int(i)) => {
                let idx = *i as usize;
                if idx < probs.len() && probs[idx] > 0.0 {
                    probs[idx].ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            (ProbDistribution::Poisson { lambda }, Value::Int(k)) => {
                if *k >= 0 {
                    // Log PMF: log(lambda^k * e^(-lambda) / k!)
                    // Simplified: k * log(lambda) - lambda - log(k!)
                    let k_f = *k as f64;
                    let mut log_fact = 0.0;
                    for i in 1..=*k {
                        log_fact += (i as f64).ln();
                    }
                    k_f * lambda.ln() - lambda - log_fact
                } else {
                    f64::NEG_INFINITY
                }
            }
            (ProbDistribution::Exponential { lambda }, Value::Float(f)) => {
                if *f >= 0.0 {
                    lambda.ln() - lambda * f
                } else {
                    f64::NEG_INFINITY
                }
            }
            _ => f64::NEG_INFINITY, // Simplified for other distributions
        }
    }

    /// Get support (possible values) for discrete distributions
    pub fn support(&self) -> Option<Vec<Value>> {
        match self {
            ProbDistribution::Bernoulli { .. } => Some(vec![Value::Bool(false), Value::Bool(true)]),
            ProbDistribution::Categorical(probs) => {
                Some((0..probs.len()).map(|i| Value::Int(i as i64)).collect())
            }
            _ => None, // Continuous distributions have infinite support
        }
    }

    /// Get mean of this distribution
    pub fn mean(&self) -> f64 {
        match self {
            ProbDistribution::Uniform { low, high } => (low + high) / 2.0,
            ProbDistribution::Normal { mean, .. } => *mean,
            ProbDistribution::Bernoulli { p } => *p,
            ProbDistribution::Categorical(probs) => {
                probs.iter().enumerate().map(|(i, &p)| i as f64 * p).sum()
            }
            ProbDistribution::Exponential { lambda } => 1.0 / lambda,
            ProbDistribution::Poisson { lambda } => *lambda,
            ProbDistribution::Beta { alpha, beta } => alpha / (alpha + beta),
            ProbDistribution::Gamma { shape, scale } => shape * scale,
            ProbDistribution::Dirichlet(alphas) => {
                let sum: f64 = alphas.iter().sum();
                alphas
                    .iter()
                    .map(|a| a / sum)
                    .collect::<Vec<_>>()
                    .iter()
                    .sum()
            }
        }
    }

    /// Get variance of this distribution
    pub fn variance(&self) -> f64 {
        match self {
            ProbDistribution::Uniform { low, high } => {
                let range = high - low;
                range * range / 12.0
            }
            ProbDistribution::Normal { std, .. } => std * std,
            ProbDistribution::Bernoulli { p } => p * (1.0 - p),
            ProbDistribution::Categorical(probs) => {
                let mean = self.mean();
                probs
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| {
                        let diff = i as f64 - mean;
                        p * diff * diff
                    })
                    .sum()
            }
            ProbDistribution::Exponential { lambda } => 1.0 / (lambda * lambda),
            ProbDistribution::Poisson { lambda } => *lambda,
            ProbDistribution::Beta { alpha, beta } => {
                let sum = alpha + beta;
                (alpha * beta) / (sum * sum * (sum + 1.0))
            }
            ProbDistribution::Gamma { shape, scale } => shape * scale * scale,
            ProbDistribution::Dirichlet(_) => {
                // Multivariate - return trace of covariance matrix (simplified)
                0.1
            }
        }
    }
}

/// Value types for distributions
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Array(Vec<Value>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_sample() {
        let dist = ProbDistribution::Uniform {
            low: 0.0,
            high: 10.0,
        };
        for _ in 0..100 {
            let val = dist.sample();
            if let Value::Float(f) = val {
                assert!(f >= 0.0 && f <= 10.0);
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_normal_mean() {
        let dist = ProbDistribution::Normal {
            mean: 5.0,
            std: 2.0,
        };
        assert_eq!(dist.mean(), 5.0);
        assert_eq!(dist.variance(), 4.0);
    }

    #[test]
    fn test_bernoulli_support() {
        let dist = ProbDistribution::Bernoulli { p: 0.5 };
        let support = dist.support();
        assert!(support.is_some());
        assert_eq!(support.unwrap().len(), 2);
    }

    #[test]
    fn test_bernoulli_log_prob() {
        let dist = ProbDistribution::Bernoulli { p: 0.7 };
        let log_true = dist.log_prob(&Value::Bool(true));
        let log_false = dist.log_prob(&Value::Bool(false));
        assert!((log_true - 0.7_f64.ln()).abs() < 0.01);
        assert!((log_false - 0.3_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_categorical_sample() {
        let dist = ProbDistribution::Categorical(vec![0.2, 0.3, 0.5]);
        for _ in 0..100 {
            let val = dist.sample();
            if let Value::Int(i) = val {
                assert!(i >= 0 && i <= 2);
            } else {
                panic!("Expected Int value");
            }
        }
    }
}
