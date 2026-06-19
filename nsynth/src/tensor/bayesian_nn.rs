//! Bayesian Neural Network Primitives
//!
//! Implementation of Bayesian neural network components including:
//! - Bayesian Linear layers with weight uncertainty
//! - MC Dropout for uncertainty estimation
//! - Variational inference layers
//! - Bayes by backprop algorithm

use crate::tensor::ops::{Shape, Tensor};
use std::boxed::Box;

/// Bayesian Linear Layer with weight uncertainty
///
/// Uses variational inference to learn distributions over weights rather than point estimates.
/// Weights are parameterized by mu (mean) and rho (log std), enabling reparameterization.
#[derive(Debug, Clone)]
pub struct BayesianLinear {
    /// Input feature dimension
    pub in_features: usize,
    /// Output feature dimension
    pub out_features: usize,
    /// Weight mean parameters
    pub weight_mu: Tensor,
    /// Weight log-std parameters (rho)
    pub weight_rho: Tensor,
    /// Bias mean parameters
    pub bias_mu: Tensor,
    /// Bias log-std parameters (rho)
    pub bias_rho: Tensor,
    /// Prior standard deviation for KL divergence
    prior_sigma: f64,
}

impl BayesianLinear {
    /// Create a new Bayesian linear layer
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight_shape = Shape::new(vec![out_features, in_features]);
        let bias_shape = Shape::new(vec![out_features]);

        // Initialize weight parameters
        let weight_mu = Tensor::randn_scaled(weight_shape.clone(), 0.0, 0.1);
        let weight_rho = Tensor::full(weight_shape.dims, -3.0); // log(0.05) ≈ -3

        // Initialize bias parameters
        let bias_mu = Tensor::randn_scaled(bias_shape.clone(), 0.0, 0.1);
        let bias_rho = Tensor::full(bias_shape.dims, -3.0);

        Self {
            in_features,
            out_features,
            weight_mu,
            weight_rho,
            bias_mu,
            bias_rho,
            prior_sigma: 1.0,
        }
    }

    /// Create layer with custom prior
    pub fn with_prior(in_features: usize, out_features: usize, prior_sigma: f64) -> Self {
        let mut layer = Self::new(in_features, out_features);
        layer.prior_sigma = prior_sigma;
        layer
    }

    /// Sample weights from the variational distribution (reparameterization)
    ///
    /// Returns (weight_samples, bias_samples) using the reparameterization trick:
    /// w = mu + sigma * epsilon, where epsilon ~ N(0, I)
    pub fn sample_weights(&self) -> (Tensor, Tensor) {
        // Sample epsilon for weights
        let eps_weight = Tensor::randn(self.weight_mu.shape.clone());
        let sigma_weight = self.softplus(&self.weight_rho);
        let weight_samples = self.weight_mu.add(&sigma_weight.mul(&eps_weight).unwrap()).unwrap();

        // Sample epsilon for bias
        let eps_bias = Tensor::randn(self.bias_mu.shape.clone());
        let sigma_bias = self.softplus(&self.bias_rho);
        let bias_samples = self.bias_mu.add(&sigma_bias.mul(&eps_bias).unwrap()).unwrap();

        (weight_samples, bias_samples)
    }

    /// Forward pass with sampled weights
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch_size, in_features)
    ///
    /// # Returns
    /// Output tensor of shape (batch_size, out_features)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (weight_samples, bias_samples) = self.sample_weights();
        self.linear_forward(x, &weight_samples, &bias_samples)
    }

    /// Perform linear transformation with given weights and bias
    fn linear_forward(&self, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
        // x: (batch_size, in_features)
        // weight: (out_features, in_features)
        // bias: (out_features,)

        let batch_size = x.shape.dims[0];
        let mut output_data = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            for out_idx in 0..self.out_features {
                let mut sum = 0.0;
                for in_idx in 0..self.in_features {
                    let x_val = x.data[b * self.in_features + in_idx];
                    let w_val = weight.data[out_idx * self.in_features + in_idx];
                    sum += x_val * w_val;
                }
                output_data[b * self.out_features + out_idx] = sum + bias.data[out_idx];
            }
        }

        Tensor::new(output_data, Shape::new(vec![batch_size, self.out_features]))
    }

    /// Compute KL divergence between variational posterior and prior
    ///
    /// KL(q(w)||p(w)) where q(w) = N(mu, sigma²) and p(w) = N(0, prior_sigma²)
    pub fn kl_divergence(&self) -> Tensor {
        let sigma_weight = self.softplus(&self.weight_rho);
        let sigma_bias = self.softplus(&self.bias_rho);

        // KL for weights: 0.5 * sum(log(prior²/sigma²) + (sigma² + mu²)/prior² - 1)
        let kl_weight = self.compute_kl_term(&self.weight_mu, &sigma_weight, self.prior_sigma);

        // KL for bias
        let kl_bias = self.compute_kl_term(&self.bias_mu, &sigma_bias, self.prior_sigma);

        // Total KL
        let total_kl = kl_weight + kl_bias;
        Tensor::scalar(total_kl)
    }

    /// Compute KL term for a single parameter group
    fn compute_kl_term(&self, mu: &Tensor, sigma: &Tensor, prior_sigma: f64) -> f64 {
        let prior_var = prior_sigma * prior_sigma;
        let mut kl_sum = 0.0;

        for i in 0..mu.data.len() {
            let sigma_val = sigma.data[i];
            let sigma_var = sigma_val * sigma_val;
            let mu_val = mu.data[i];

            let term = 0.5 * ((prior_var / sigma_var).ln() + (sigma_var + mu_val * mu_val) / prior_var - 1.0);
            kl_sum += term;
        }

        kl_sum
    }

    /// Softplus activation for converting rho to sigma: log(1 + exp(rho))
    fn softplus(&self, rho: &Tensor) -> Tensor {
        let data: Vec<f64> = rho.data.iter().map(|&x| (1.0 + x.exp()).ln()).collect();
        Tensor::new(data, rho.shape.clone())
    }
}

/// MC Dropout Layer for uncertainty estimation
///
/// Uses Monte Carlo Dropout at inference time to estimate predictive uncertainty.
/// Multiple forward passes with different dropout masks provide uncertainty estimates.
///
/// Note: This struct cannot derive Clone due to the boxed function pointer.
/// For cloning behavior, create a new instance with the same parameters.
pub struct MCDropout {
    /// Base layer function (boxed to support any layer type)
    pub base_layer: Box<dyn Fn(&Tensor) -> Tensor>,
    /// Dropout probability
    pub dropout_rate: f64,
    /// Number of MC samples for uncertainty estimation
    pub num_samples: usize,
}

impl MCDropout {
    /// Create a new MC Dropout layer
    ///
    /// # Arguments
    /// * `base_layer` - Function representing the base layer transformation
    /// * `dropout_rate` - Probability of dropping units (0.0 to 1.0)
    /// * `num_samples` - Number of forward passes for uncertainty estimation
    pub fn new<F>(base_layer: F, dropout_rate: f64, num_samples: usize) -> Self
    where
        F: Fn(&Tensor) -> Tensor + 'static,
    {
        Self {
            base_layer: Box::new(base_layer),
            dropout_rate,
            num_samples,
        }
    }

    /// Forward pass with dropout enabled (training mode)
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// Output tensor with dropout applied
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let dropped = self.apply_dropout(x);
        (self.base_layer)(&dropped)
    }

    /// Forward pass with uncertainty estimation (inference mode)
    ///
    /// Performs multiple forward passes with different dropout masks and returns
    /// the mean prediction and predictive variance.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// Tuple of (mean_prediction, variance) tensors
    pub fn forward_with_uncertainty(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut outputs = Vec::with_capacity(self.num_samples);

        // Perform multiple forward passes
        for _ in 0..self.num_samples {
            let dropped = self.apply_dropout(x);
            outputs.push((self.base_layer)(&dropped));
        }

        // Compute mean and variance across samples
        self.compute_mean_variance(&outputs)
    }

    /// Apply dropout mask to input tensor
    fn apply_dropout(&self, x: &Tensor) -> Tensor {
        let scale = 1.0 / (1.0 - self.dropout_rate);
        let mut data = Vec::with_capacity(x.data.len());

        for &val in &x.data {
            let keep = if pseudo_random() < self.dropout_rate { 0.0 } else { 1.0 };
            data.push(val * keep * scale);
        }

        Tensor::new(data, x.shape.clone())
    }

    /// Compute mean and variance across MC samples
    fn compute_mean_variance(&self, samples: &[Tensor]) -> (Tensor, Tensor) {
        let n = samples.len();
        let output_shape = samples[0].shape.clone();
        let output_size = output_shape.size();

        // Compute mean
        let mut mean_data = vec![0.0; output_size];
        for i in 0..output_size {
            let sum: f64 = samples.iter().map(|s| s.data[i]).sum();
            mean_data[i] = sum / n as f64;
        }

        // Compute variance
        let mut var_data = vec![0.0; output_size];
        for i in 0..output_size {
            let mean_val = mean_data[i];
            let var_sum: f64 = samples.iter().map(|s| (s.data[i] - mean_val).powi(2)).sum();
            var_data[i] = var_sum / n as f64;
        }

        (
            Tensor::new(mean_data, output_shape.clone()),
            Tensor::new(var_data, output_shape),
        )
    }
}

/// Variational Inference Layer
///
/// A general variational layer that learns distributions over parameters.
/// Similar to BayesianLinear but more flexible for different layer types.
#[derive(Debug, Clone)]
pub struct VariationalLayer {
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
    /// Weight mean
    pub weight_mu: Tensor,
    /// Weight log-std
    pub weight_rho: Tensor,
    /// Prior mean for weights
    pub prior_mu: f64,
    /// Prior standard deviation
    pub prior_sigma: f64,
}

impl VariationalLayer {
    /// Create a new variational layer
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight_shape = Shape::new(vec![out_features, in_features]);
        let weight_mu = Tensor::randn_scaled(weight_shape.clone(), 0.0, 0.1);
        let weight_rho = Tensor::full(weight_shape.dims, -3.0);

        Self {
            in_features,
            out_features,
            weight_mu,
            weight_rho,
            prior_mu: 0.0,
            prior_sigma: 1.0,
        }
    }

    /// Forward pass with reparameterization
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let weight_samples = self.reparameterize(&self.weight_mu, &self.weight_rho);
        self.matmul(x, &weight_samples)
    }

    /// Matrix multiplication for forward pass
    fn matmul(&self, x: &Tensor, weight: &Tensor) -> Tensor {
        let batch_size = x.shape.dims[0];
        let mut output_data = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            for out_idx in 0..self.out_features {
                let mut sum = 0.0;
                for in_idx in 0..self.in_features {
                    let x_val = x.data[b * self.in_features + in_idx];
                    let w_val = weight.data[out_idx * self.in_features + in_idx];
                    sum += x_val * w_val;
                }
                output_data[b * self.out_features + out_idx] = sum;
            }
        }

        Tensor::new(output_data, Shape::new(vec![batch_size, self.out_features]))
    }

    /// Compute KL loss for variational inference
    ///
    /// Returns the KL divergence between learned distribution and prior
    pub fn kl_loss(&self) -> Tensor {
        let sigma = self.softplus(&self.weight_rho);
        let mut kl_sum = 0.0;

        for i in 0..self.weight_mu.data.len() {
            let mu_val = self.weight_mu.data[i];
            let sigma_val = sigma.data[i];
            let sigma_sq = sigma_val * sigma_val;

            // KL(q||p) where q ~ N(mu, sigma²), p ~ N(prior_mu, prior_sigma²)
            let kl_term = 0.5 * (
                (self.prior_sigma.powi(2) / sigma_sq).ln() +
                (sigma_sq + (mu_val - self.prior_mu).powi(2)) / self.prior_sigma.powi(2) -
                1.0
            );
            kl_sum += kl_term;
        }

        Tensor::scalar(kl_sum)
    }

    /// Reparameterization trick: sample from N(mu, sigma²)
    fn reparameterize(&self, mu: &Tensor, rho: &Tensor) -> Tensor {
        let eps = Tensor::randn(mu.shape.clone());
        let sigma = self.softplus(rho);
        mu.add(&sigma.mul(&eps).unwrap()).unwrap()
    }

    /// Softplus activation
    fn softplus(&self, rho: &Tensor) -> Tensor {
        let data: Vec<f64> = rho.data.iter().map(|&x| (1.0 + x.exp()).ln()).collect();
        Tensor::new(data, rho.shape.clone())
    }
}

/// Bayes by Backprop loss computation
///
/// Implements the loss function from the "Bayes by Backprop" paper (Blundell et al., 2015).
/// Combines negative log-likelihood with KL divergence weighted by batch size.
///
/// # Arguments
/// * `neg_log_likelihood` - Negative log-likelihood tensor
/// * `kl` - KL divergence tensor
/// * `num_batches` - Total number of batches in the dataset (for KL annealing)
///
/// # Returns
/// Total loss: L = NLL + (KL / num_batches) * num_samples
pub fn bbb_loss(neg_log_likelihood: &Tensor, kl: &Tensor, num_batches: usize) -> Tensor {
    let nll_val = neg_log_likelihood.data[0];
    let kl_val = kl.data[0];

    // Scale KL by number of batches (annealing)
    let scaled_kl = kl_val / num_batches as f64;

    // Total loss
    let total_loss = nll_val + scaled_kl;
    Tensor::scalar(total_loss)
}

/// Reparameterization trick for sampling from Gaussian
///
/// # Arguments
/// * `mu` - Mean tensor
/// * `rho` - Log-std tensor
///
/// # Returns
/// Sampled tensor: mu + softplus(rho) * epsilon, epsilon ~ N(0, I)
pub fn reparameterize(mu: &Tensor, rho: &Tensor) -> Tensor {
    let eps = Tensor::randn(mu.shape.clone());
    let sigma = softplus(rho);
    mu.add(&sigma.mul(&eps).unwrap()).unwrap()
}

/// Softplus helper function
fn softplus(rho: &Tensor) -> Tensor {
    let data: Vec<f64> = rho.data.iter().map(|&x| (1.0 + x.exp()).ln()).collect();
    Tensor::new(data, rho.shape.clone())
}

// Simple pseudo-random number generator for deterministic testing
fn pseudo_random() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(42);
    }
    SEED.with(|seed| {
        let s = seed.get();
        seed.set(s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
        (s >> 11) as f64 / ((1u64 << 53) as f64)
    })
}

// Box-Muller transform for normal distribution
fn box_muller() -> f64 {
    let u1 = pseudo_random();
    let u2 = pseudo_random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_linear_creation() {
        let layer = BayesianLinear::new(10, 5);
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 5);
        assert_eq!(layer.weight_mu.shape.dims, vec![5, 10]);
        assert_eq!(layer.bias_mu.shape.dims, vec![5]);
    }

    #[test]
    fn test_bayesian_linear_forward() {
        let layer = BayesianLinear::new(3, 2);
        let x = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let output = layer.forward(&x);

        assert_eq!(output.shape.dims, vec![2, 2]);
        // Output should have some variation due to sampling
        assert!(!output.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_bayesian_linear_sample_weights() {
        let layer = BayesianLinear::new(4, 3);
        let (weights, bias) = layer.sample_weights();

        assert_eq!(weights.shape.dims, vec![3, 4]);
        assert_eq!(bias.shape.dims, vec![3]);

        // Samples should have non-zero values due to randomness
        assert!(weights.data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_bayesian_linear_kl_divergence() {
        let layer = BayesianLinear::new(5, 3);
        let kl = layer.kl_divergence();

        // KL should be a scalar
        assert_eq!(kl.shape.dims, vec![1]);
        // KL should be non-negative
        assert!(kl.data[0] >= 0.0);
    }

    #[test]
    fn test_mc_dropout_creation() {
        let base_layer = |x: &Tensor| x.clone();
        let mc_dropout = MCDropout::new(base_layer, 0.5, 10);

        assert_eq!(mc_dropout.dropout_rate, 0.5);
        assert_eq!(mc_dropout.num_samples, 10);
    }

    #[test]
    fn test_mc_dropout_forward() {
        let base_layer = |x: &Tensor| x.clone();
        let mc_dropout = MCDropout::new(base_layer, 0.5, 5);

        let x = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let output = mc_dropout.forward(&x);

        assert_eq!(output.shape.dims, vec![2, 2]);
    }

    #[test]
    fn test_mc_dropout_uncertainty() {
        let base_layer = |x: &Tensor| x.clone();
        let mc_dropout = MCDropout::new(base_layer, 0.3, 100);

        let x = Tensor::ones(Shape::new(vec![10, 5]));
        let (mean, variance) = mc_dropout.forward_with_uncertainty(&x);

        assert_eq!(mean.shape.dims, vec![10, 5]);
        assert_eq!(variance.shape.dims, vec![10, 5]);

        // Variance should be positive due to dropout randomness
        assert!(variance.data.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_variational_layer_creation() {
        let layer = VariationalLayer::new(8, 4);
        assert_eq!(layer.in_features, 8);
        assert_eq!(layer.out_features, 4);
        assert_eq!(layer.weight_mu.shape.dims, vec![4, 8]);
    }

    #[test]
    fn test_variational_layer_forward() {
        let layer = VariationalLayer::new(3, 2);
        let x = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let output = layer.forward(&x);

        assert_eq!(output.shape.dims, vec![2, 2]);
    }

    #[test]
    fn test_variational_layer_kl_loss() {
        let layer = VariationalLayer::new(5, 3);
        let kl_loss = layer.kl_loss();

        assert_eq!(kl_loss.shape.dims, vec![1]);
        assert!(kl_loss.data[0] >= 0.0);
    }

    #[test]
    fn test_bbb_loss_computation() {
        let nll = Tensor::scalar(1.5);
        let kl = Tensor::scalar(2.5);
        let num_batches = 100;

        let loss = bbb_loss(&nll, &kl, num_batches);

        assert_eq!(loss.shape.dims, vec![1]);
        let expected = 1.5 + 2.5 / 100.0;
        assert!((loss.data[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reparameterize_function() {
        let mu = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let rho = Tensor::matrix(vec![-1.0, -1.0, -1.0, -1.0], 2, 2);

        let sample = reparameterize(&mu, &rho);

        assert_eq!(sample.shape.dims, vec![2, 2]);
        // Samples should vary from mu due to randomness
        let mu_different = sample.data.iter().zip(mu.data.iter()).any(|(s, &m)| (*s - m).abs() > 0.01);
        assert!(mu_different);
    }

    #[test]
    fn test_softplus() {
        let rho = Tensor::vector(vec![-3.0, -1.0, 0.0, 1.0, 3.0]);
        let sigma = softplus(&rho);

        // Softplus should be positive
        assert!(sigma.data.iter().all(|&x| x > 0.0));

        // softplus(0) ≈ 0.693 (ln(2))
        assert!((sigma.data[2] - 0.693147).abs() < 0.01);

        // softplus should be roughly exp(rho) for large positive rho
        assert!((sigma.data[4] - 3.0_f64.exp().ln()).abs() < 0.1);
    }
}
