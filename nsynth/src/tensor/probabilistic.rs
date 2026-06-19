//! Probabilistic Primitives and Variational Inference
//!
//! Implementation of:
//! - KL Divergence for categorical and continuous distributions
//! - Shannon entropy, conditional entropy, mutual information
//! - Variational inference with ELBO optimization
//! - Normal distribution operations

use super::ops::{Shape, Tensor};
use std::f64::consts::PI;

// ============================================================================
// KL Divergence Primitives
// ============================================================================

/// Compute KL divergence for categorical distributions
///
/// KL(p||q) = sum(p * log(p/q))
///
/// # Arguments
/// * `p` - Probability distribution (categorical)
/// * `q` - Probability distribution (categorical)
///
/// # Returns
/// Scalar tensor containing KL divergence value
pub fn kl_divergence(p: &Tensor, q: &Tensor) -> Tensor {
    // Ensure both distributions have same shape
    assert_eq!(p.shape.dims, q.shape.dims, "Distributions must have same shape");

    // Add small epsilon for numerical stability
    let epsilon = 1e-8;
    let n = p.data.len();

    let mut kl = 0.0;
    for i in 0..n {
        let p_i = p.data[i].max(epsilon);
        let q_i = q.data[i].max(epsilon);

        // Ensure p is a valid probability distribution (sums to 1)
        // And q is valid (non-negative, sums to 1)
        if p_i > 0.0 && q_i > 0.0 {
            kl += p_i * (p_i / q_i).ln();
        }
    }

    Tensor::scalar(kl)
}

/// Compute KL divergence for continuous distributions via Monte Carlo
///
/// KL(p||q) ≈ E_p[log(p(x)/q(x))]
///
/// # Arguments
/// * `p_mean` - Mean of distribution p
/// * `p_var` - Variance of distribution p
/// * `q_mean` - Mean of distribution q
/// * `q_var` - Variance of distribution q
///
/// # Returns
/// Scalar tensor containing KL divergence value
pub fn kl_divergence_continuous(
    p_mean: &Tensor,
    p_var: &Tensor,
    q_mean: &Tensor,
    q_var: &Tensor,
) -> Tensor {
    // Assume diagonal Gaussian distributions
    kl_divergence_diag_gaussian(p_mean, p_var, q_mean, q_var)
}

/// Compute KL divergence between diagonal Gaussian distributions
///
/// KL(N(μ1, Σ1) || N(μ2, Σ2)) where Σ1, Σ2 are diagonal
///
/// Formula:
/// KL = 0.5 * (tr(Σ2^-1 * Σ1) + (μ2 - μ1)^T * Σ2^-1 * (μ2 - μ1) - k + log(det(Σ2) / det(Σ1)))
///
/// For diagonal covariance, simplifies to sum over dimensions:
/// KL = 0.5 * sum((σ1²/σ2²) + ((μ2 - μ1)²/σ2²) - 1 + log(σ2²/σ1²))
///
/// # Arguments
/// * `mu1` - Mean of first distribution
/// * `var1` - Variance of first distribution (diagonal)
/// * `mu2` - Mean of second distribution
/// * `var2` - Variance of second distribution (diagonal)
///
/// # Returns
/// Scalar tensor containing KL divergence value
pub fn kl_divergence_diag_gaussian(
    mu1: &Tensor,
    var1: &Tensor,
    mu2: &Tensor,
    var2: &Tensor,
) -> Tensor {
    let dim = mu1.data.len();
    assert_eq!(dim, mu2.data.len(), "Means must have same dimension");
    assert_eq!(dim, var1.data.len(), "Variance must match mean dimension");
    assert_eq!(dim, var2.data.len(), "Variance must match mean dimension");

    let mut kl = 0.0;

    for i in 0..dim {
        let mu1_i = mu1.data[i];
        let mu2_i = mu2.data[i];
        let var1_i = var1.data[i].max(1e-8); // Prevent division by zero
        let var2_i = var2.data[i].max(1e-8);

        // Component-wise KL for diagonal Gaussian
        let component_kl = 0.5 * (
            var1_i / var2_i +
            ((mu2_i - mu1_i).powi(2) / var2_i) -
            1.0 +
            (var2_i / var1_i).ln()
        );

        kl += component_kl;
    }

    Tensor::scalar(kl)
}

// ============================================================================
// Entropy Primitives
// ============================================================================

/// Compute Shannon entropy of a categorical distribution
///
/// H(p) = -sum(p * log(p))
///
/// # Arguments
/// * `distribution` - Probability distribution (categorical)
///
/// # Returns
/// Scalar tensor containing entropy value (in nats)
pub fn entropy(distribution: &Tensor) -> Tensor {
    let epsilon = 1e-8;
    let mut ent = 0.0;

    for &p in &distribution.data {
        if p > epsilon {
            ent -= p * p.max(epsilon).ln();
        }
    }

    Tensor::scalar(ent)
}

/// Compute conditional entropy H(X|Y)
///
/// H(X|Y) = H(X,Y) - H(Y)
///
/// # Arguments
/// * `x` - First variable (joint distribution with y)
/// * `y` - Second variable (joint distribution with x)
///
/// # Note
/// This assumes x represents the joint distribution P(X,Y) and
/// we can marginalize to get P(Y). For simplicity, this assumes
/// x is already P(X|Y) conditioned on Y.
///
/// # Returns
/// Scalar tensor containing conditional entropy value
pub fn conditional_entropy(x: &Tensor, y: &Tensor) -> Tensor {
    // For simplicity, assume x is P(X,Y) joint distribution
    // H(X|Y) = H(X,Y) - H(Y)

    // First, marginalize to get P(Y) by summing over X dimensions
    // This is a simplified version - in practice, you'd need proper
    // joint distribution handling

    // Compute H(X,Y) - treating x as joint
    let h_joint = entropy(x);

    // Compute H(Y)
    let h_y = entropy(y);

    // H(X|Y) = H(X,Y) - H(Y)
    let h_cond = h_joint.data[0] - h_y.data[0];

    Tensor::scalar(h_cond.max(0.0)) // Ensure non-negative
}

/// Compute mutual information I(X;Y)
///
/// I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y)
///
/// # Arguments
/// * `x` - First variable distribution
/// * `y` - Second variable distribution
///
/// # Note
/// This is a simplified version assuming x and y are independent marginals.
/// For proper MI, you need the joint distribution P(X,Y).
///
/// # Returns
/// Scalar tensor containing mutual information value
pub fn mutual_information(x: &Tensor, y: &Tensor) -> Tensor {
    // I(X;Y) = H(X) + H(Y) - H(X,Y)
    // For simplified case where we don't have joint distribution,
    // this would be 0 for independent variables

    // Compute marginal entropies
    let h_x = entropy(x);
    let h_y = entropy(y);

    // In a full implementation, we'd compute H(X,Y) from joint distribution
    // For now, return a placeholder that would need joint distribution
    let _h_joint = h_x.data[0] + h_y.data[0]; // Upper bound

    // MI = H(X) + H(Y) - H(X,Y)
    // For independent variables, MI = 0
    Tensor::scalar(0.0) // Placeholder - requires joint distribution
}

// ============================================================================
// Normal Distribution Primitives
// ============================================================================

/// Normal (Gaussian) distribution with learnable parameters
#[derive(Debug, Clone)]
pub struct NormalDistribution {
    /// Mean parameter μ
    pub mu: Tensor,
    /// Standard deviation parameter σ
    pub sigma: Tensor,
}

impl NormalDistribution {
    /// Create a new normal distribution with given parameters
    ///
    /// # Arguments
    /// * `mu` - Mean tensor
    /// * `sigma` - Standard deviation tensor (must be positive)
    pub fn new(mu: Tensor, sigma: Tensor) -> Self {
        // Validate sigma is positive
        for &s in &sigma.data {
            assert!(s > 0.0, "Standard deviation must be positive, got {}", s);
        }

        Self { mu, sigma }
    }

    /// Create standard normal distribution (μ=0, σ=1)
    pub fn standard(dim: usize) -> Self {
        let mu = Tensor::zeros(Shape::new(vec![dim]));
        let sigma_data = vec![1.0; dim];
        let sigma = Tensor::new(sigma_data, Shape::new(vec![dim]));
        Self::new(mu, sigma)
    }

    /// Sample from the distribution using reparameterization trick
    ///
    /// Uses: z = μ + σ * ε, where ε ~ N(0,1)
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to draw
    ///
    /// # Returns
    /// Tensor of shape [num_samples, dim]
    pub fn sample(&self, num_samples: usize) -> Tensor {
        let dim = self.mu.data.len();
        let mut samples = Vec::with_capacity(num_samples * dim);

        for _ in 0..num_samples {
            for i in 0..dim {
                // Sample from standard normal using Box-Muller transform
                let u1: f64 = pseudo_random();
                let u2: f64 = pseudo_random();

                // Box-Muller transform to get standard normal
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

                // Reparameterization trick: μ + σ * z
                let sample = self.mu.data[i] + self.sigma.data[i] * z0;
                samples.push(sample);
            }
        }

        Tensor::new(samples, Shape::new(vec![num_samples, dim]))
    }

    /// Compute log probability density function
    ///
    /// log p(x) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [dim] or [num_samples, dim]
    ///
    /// # Returns
    /// Scalar tensor containing log probability (summed over dimensions)
    pub fn log_prob(&self, x: &Tensor) -> Tensor {
        let x_flat = if x.shape.dims.len() == 2 {
            // Flatten [num_samples, dim] -> [num_samples * dim]
            x.data.clone()
        } else {
            x.data.clone()
        };

        let dim = self.mu.data.len();
        let mut log_prob = 0.0;

        for i in 0..dim {
            let x_i = x_flat[i];
            let mu_i = self.mu.data[i];
            let sigma_i = self.sigma.data[i];

            // Log probability of univariate normal
            let log_p_i = -0.5 * (2.0 * PI).ln()
                - sigma_i.ln()
                - 0.5 * ((x_i - mu_i) / sigma_i).powi(2);

            log_prob += log_p_i;
        }

        Tensor::scalar(log_prob)
    }

    /// Compute entropy of the normal distribution
    ///
    /// H(N(μ, σ²)) = 0.5 * (1 + log(2πσ²)) = 0.5 + log(σ) + 0.5 * log(2π)
    ///
    /// # Returns
    /// Scalar tensor containing entropy value (in nats)
    pub fn entropy(&self) -> Tensor {
        let mut ent = 0.0;

        for &sigma_i in &self.sigma.data {
            let var = sigma_i.powi(2);
            // Differential entropy for Gaussian
            ent += 0.5 * (1.0 + (2.0 * PI * var).ln());
        }

        Tensor::scalar(ent)
    }
}

// ============================================================================
// Variational Inference Primitives
// ============================================================================

/// Variational inference optimizer for approximating posterior distributions
///
/// Uses the Evidence Lower BOund (ELBO) objective:
/// ELBO = E_q(z|x)[log p(x,z)] - E_q(z|x)[log q(z|x)]
///      = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
pub struct VariationalInference {
    /// Approximate posterior distribution q(z|x)
    approx_posterior: Box<dyn Fn(&Tensor) -> NormalDistribution>,
    /// Likelihood function p(x|z)
    likelihood: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
    /// Prior distribution p(z) (typically standard normal)
    prior: NormalDistribution,
}

impl VariationalInference {
    /// Create a new variational inference optimizer
    ///
    /// # Arguments
    /// * `approx_posterior` - Function that returns approximate posterior q(z|x)
    /// * `likelihood` - Function computing log-likelihood log p(x|z)
    pub fn new(
        approx_posterior: Box<dyn Fn(&Tensor) -> NormalDistribution>,
        likelihood: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
    ) -> Self {
        // Default prior is standard normal
        let prior_dim = 1; // Will be updated on first use
        let prior = NormalDistribution::standard(prior_dim);

        Self {
            approx_posterior,
            likelihood,
            prior,
        }
    }

    /// Create with custom prior distribution
    pub fn with_prior(
        approx_posterior: Box<dyn Fn(&Tensor) -> NormalDistribution>,
        likelihood: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
        prior: NormalDistribution,
    ) -> Self {
        Self {
            approx_posterior,
            likelihood,
            prior,
        }
    }

    /// Compute the Evidence Lower BOund (ELBO)
    ///
    /// ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
    ///
    /// # Arguments
    /// * `x` - Observed data
    ///
    /// # Returns
    /// Scalar tensor containing ELBO value
    pub fn elbo(&self, x: &Tensor) -> Tensor {
        // Get approximate posterior q(z|x)
        let q = (self.approx_posterior)(x);

        // Sample from q(z|x) using single sample (Monte Carlo estimate)
        let z = q.sample(1);

        // Reshape z to [dim] for likelihood computation
        let z_flat = Tensor::new(
            z.data[0..q.mu.data.len()].to_vec(),
            q.mu.shape.clone()
        );

        // Expected log-likelihood: E_q[log p(x|z)]
        let log_likelihood = (self.likelihood)(x, &z_flat);

        // KL divergence: KL(q(z|x) || p(z))
        let kl = kl_divergence_diag_gaussian(
            &q.mu,
            &q.sigma.clone().mul(&q.sigma).unwrap(), // Convert σ to σ²
            &self.prior.mu,
            &self.prior.sigma.clone().mul(&self.prior.sigma).unwrap(),
        );

        // ELBO = E_q[log p(x|z)] - KL(q || p)
        let elbo_value = log_likelihood.data[0] - kl.data[0];

        Tensor::scalar(elbo_value)
    }

    /// Compute variational inference loss (negative ELBO)
    ///
    /// Loss = -ELBO = KL(q||p) - E_q[log p(x|z)]
    ///
    /// # Arguments
    /// * `x` - Observed data
    /// * `num_samples` - Number of samples for Monte Carlo estimation
    ///
    /// # Returns
    /// Scalar tensor containing loss value (to be minimized)
    pub fn vi_loss(&self, x: &Tensor, num_samples: usize) -> Tensor {
        let mut total_elbo = 0.0;

        // Monte Carlo estimation with multiple samples
        for _ in 0..num_samples {
            let elbo_single = self.elbo(x);
            total_elbo += elbo_single.data[0];
        }

        // Average ELBO
        let avg_elbo = total_elbo / num_samples as f64;

        // Return negative ELBO as loss (to minimize)
        Tensor::scalar(-avg_elbo)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple pseudo-random number generator (for reproducibility in tests)
/// In production, use a proper PRNG like rand::rngs::StdRng
fn pseudo_random() -> f64 {
    use std::cell::RefCell;
    use std::rc::Rc;

    thread_local! {
        static COUNTER: Rc<RefCell<u64>> = Rc::new(RefCell::new(123456789u64));
    }

    COUNTER.with(|counter| {
        let mut c = counter.borrow_mut();
        *c = c.wrapping_mul(6364136223846793005);
        *c = c.wrapping_add(1442695040888963407);

        // Convert to f64 in [0, 1)
        let bits = (*c >> 11) as u64;
        (bits as f64) / ((1u64 << 53) as f64)
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_divergence_categorical() {
        // KL divergence between identical distributions should be 0
        let p_data = vec![0.25, 0.25, 0.25, 0.25];
        let p = Tensor::new(p_data.clone(), Shape::new(vec![4]));
        let q = Tensor::new(p_data, Shape::new(vec![4]));

        let kl = kl_divergence(&p, &q);
        assert!(kl.data[0].abs() < 1e-6, "KL(p||p) should be ~0");

        // KL divergence should be non-negative
        let p_data = vec![0.5, 0.3, 0.1, 0.1];
        let q_data = vec![0.25, 0.25, 0.25, 0.25];
        let p = Tensor::new(p_data, Shape::new(vec![4]));
        let q = Tensor::new(q_data, Shape::new(vec![4]));

        let kl = kl_divergence(&p, &q);
        assert!(kl.data[0] > 0.0, "KL divergence should be positive");
    }

    #[test]
    fn test_kl_divergence_diag_gaussian() {
        // KL(N(μ,σ²) || N(μ,σ²)) = 0
        let mu1 = Tensor::new(vec![0.0, 0.0], Shape::new(vec![2]));
        let var1 = Tensor::new(vec![1.0, 1.0], Shape::new(vec![2]));
        let mu2 = Tensor::new(vec![0.0, 0.0], Shape::new(vec![2]));
        let var2 = Tensor::new(vec![1.0, 1.0], Shape::new(vec![2]));

        let kl = kl_divergence_diag_gaussian(&mu1, &var1, &mu2, &var2);
        assert!(kl.data[0].abs() < 1e-6, "KL(N||N) with same params should be 0");

        // KL(N(0,1) || N(0,2)) should be positive
        let mu1 = Tensor::new(vec![0.0], Shape::new(vec![1]));
        let var1 = Tensor::new(vec![1.0], Shape::new(vec![1]));
        let mu2 = Tensor::new(vec![0.0], Shape::new(vec![1]));
        let var2 = Tensor::new(vec![2.0], Shape::new(vec![1]));

        let kl = kl_divergence_diag_gaussian(&mu1, &var1, &mu2, &var2);
        assert!(kl.data[0] > 0.0, "KL should be positive for different distributions");
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = Tensor::new(vec![0.25, 0.25, 0.25, 0.25], Shape::new(vec![4]));
        let h_uniform = entropy(&uniform);

        // Delta function has zero entropy
        let delta = Tensor::new(vec![1.0, 0.0, 0.0, 0.0], Shape::new(vec![4]));
        let h_delta = entropy(&delta);

        assert!(h_uniform.data[0] > h_delta.data[0], "Uniform has higher entropy than delta");
        assert!(h_delta.data[0].abs() < 1e-6, "Delta distribution has ~0 entropy");

        // Entropy should be non-negative
        assert!(h_uniform.data[0] > 0.0, "Entropy should be non-negative");
    }

    #[test]
    fn test_normal_distribution() {
        // Create standard normal
        let normal = NormalDistribution::standard(2);

        assert_eq!(normal.mu.data.len(), 2);
        assert_eq!(normal.sigma.data.len(), 2);

        // Sample from distribution
        let samples = normal.sample(100);
        assert_eq!(samples.shape.dims, vec![100, 2]);

        // Log probability should be finite
        let x = Tensor::new(vec![0.0, 0.0], Shape::new(vec![2]));
        let log_prob = normal.log_prob(&x);
        assert!(log_prob.data[0].is_finite());

        // Entropy should be positive
        let ent = normal.entropy();
        assert!(ent.data[0] > 0.0, "Gaussian entropy should be positive");
    }

    #[test]
    fn test_normal_reparameterization() {
        // Create normal with non-zero mean
        let mu = Tensor::new(vec![5.0, -3.0], Shape::new(vec![2]));
        let sigma = Tensor::new(vec![0.1, 2.0], Shape::new(vec![2]));
        let normal = NormalDistribution::new(mu.clone(), sigma.clone());

        // Samples should be centered around mu
        let samples = normal.sample(1000);

        // Compute sample means (approximately)
        let mut mean_0 = 0.0;
        let mut mean_1 = 0.0;
        for i in 0..1000 {
            mean_0 += samples.data[i * 2];
            mean_1 += samples.data[i * 2 + 1];
        }
        mean_0 /= 1000.0;
        mean_1 /= 1000.0;

        // Sample means should be close to true means
        assert!((mean_0 - 5.0).abs() < 0.2, "Sample mean should be close to true mean");
        assert!((mean_1 - (-3.0)).abs() < 0.2, "Sample mean should be close to true mean");
    }

    #[test]
    fn test_variational_inference() {
        // Create simple variational inference setup
        let approx_posterior = Box::new(|_x: &Tensor| {
            NormalDistribution::new(
                Tensor::new(vec![0.0], Shape::new(vec![1])),
                Tensor::new(vec![1.0], Shape::new(vec![1])),
            )
        });

        let likelihood = Box::new(|_x: &Tensor, z: &Tensor| {
            // Simple Gaussian likelihood
            let mu = Tensor::new(vec![0.0], Shape::new(vec![1]));
            let sigma = Tensor::new(vec![1.0], Shape::new(vec![1]));
            let normal = NormalDistribution::new(mu, sigma);
            normal.log_prob(z)
        });

        let vi = VariationalInference::new(approx_posterior, likelihood);

        let x = Tensor::new(vec![1.0], Shape::new(vec![1]));

        // ELBO should be finite
        let elbo = vi.elbo(&x);
        assert!(elbo.data[0].is_finite(), "ELBO should be finite");

        // VI loss should be positive (negative ELBO)
        let loss = vi.vi_loss(&x, 10);
        assert!(loss.data[0].is_finite(), "VI loss should be finite");
    }

    #[test]
    fn test_kl_symmetry() {
        // KL divergence is NOT symmetric: KL(p||q) ≠ KL(q||p)
        let p = Tensor::new(vec![0.9, 0.1], Shape::new(vec![2]));
        let q = Tensor::new(vec![0.5, 0.5], Shape::new(vec![2]));

        let kl_pq = kl_divergence(&p, &q);
        let kl_qp = kl_divergence(&q, &p);

        // They should be different
        assert!((kl_pq.data[0] - kl_qp.data[0]).abs() > 1e-6);

        // But both should be positive
        assert!(kl_pq.data[0] > 0.0);
        assert!(kl_qp.data[0] > 0.0);
    }
}
