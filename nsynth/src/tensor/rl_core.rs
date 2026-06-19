//! Reinforcement Learning Primitives for nCPU/nSynth
//!
//! Core RL algorithms including:
//! - Policy Gradient (REINFORCE)
//! - Proximal Policy Optimization (PPO)
//! - Value Functions
//! - Generalized Advantage Estimation (GAE)

use super::ops::Tensor;
use std::boxed::Box;

/// Policy Gradient (REINFORCE) Algorithm
///
/// Monte Carlo policy gradient method that optimizes the policy
/// using full episode returns.
pub struct PolicyGradient<F>
where
    F: Fn(&Tensor) -> Tensor + 'static,
{
    /// Policy network that maps states to action probabilities
    pub policy_network: Box<F>,
    /// Discount factor for future rewards
    pub gamma: f64,
}

impl<F> PolicyGradient<F>
where
    F: Fn(&Tensor) -> Tensor + 'static,
{
    /// Create new REINFORCE algorithm
    pub fn new(policy_network: F, gamma: f64) -> Self {
        Self {
            policy_network: Box::new(policy_network),
            gamma,
        }
    }

    /// Forward pass - get action probabilities from states
    pub fn forward(&self, states: &Tensor) -> Tensor {
        (self.policy_network)(states)
    }

    /// Compute REINFORCE loss
    ///
    /// Uses the policy gradient theorem: ∇J(θ) = E[∇log π(a|s) * G]
    /// where G is the return (discounted sum of rewards)
    pub fn compute_loss(&self, rewards: &Tensor, log_probs: &Tensor) -> Tensor {
        // Compute discounted returns
        let returns = Self::discount_rewards(rewards, self.gamma);

        // Normalize returns for stability
        let returns_normalized = self.normalize_returns(&returns);

        // Policy loss: -log_prob * return (negative because we maximize)
        let neg_log_probs = log_probs.clone();
        let weighted = Self::elementwise_mul(&neg_log_probs, &returns_normalized);

        // Mean loss over the episode
        Self::mean(&weighted)
    }

    /// Update policy using the REINFORCE rule
    pub fn update(&self, states: &Tensor, actions: &Tensor, rewards: &Tensor) -> Tensor {
        // Forward pass to get action probabilities
        let action_probs = self.forward(states);

        // Compute log probabilities for taken actions
        let log_probs = Self::log_prob(&action_probs, actions);

        // Compute and return loss
        self.compute_loss(rewards, &log_probs)
    }

    /// Discount rewards over time (public for testing)
    pub fn discount_rewards(rewards: &Tensor, gamma: f64) -> Tensor {
        let data = &rewards.data;
        let mut discounted = Vec::with_capacity(data.len());
        let mut running_sum = 0.0;

        // Iterate backwards for efficient discounting
        for &reward in data.iter().rev() {
            running_sum = reward + gamma * running_sum;
            discounted.push(running_sum);
        }

        // Reverse to get correct order
        discounted.reverse();
        Tensor::vector(discounted)
    }

    /// Normalize returns to zero mean and unit variance (public for testing)
    pub fn normalize_returns(&self, returns: &Tensor) -> Tensor {
        let mean = Self::mean(returns).data[0];
        let variance = Self::variance(returns, mean);
        let std = variance.sqrt();

        if std < 1e-8 {
            // Avoid division by zero
            return returns.clone();
        }

        let normalized: Vec<f64> = returns.data.iter().map(|&x| (x - mean) / std).collect();

        Tensor::vector(normalized)
    }

    /// Compute log probability of actions under policy
    fn log_prob(action_probs: &Tensor, actions: &Tensor) -> Tensor {
        // For each action, get its log probability
        // actions are assumed to be indices into the probability distribution
        let mut log_probs = Vec::with_capacity(actions.data.len());

        for (i, &action_idx) in actions.data.iter().enumerate() {
            let idx = action_idx as usize;
            // Small epsilon for numerical stability
            let prob = action_probs
                .data
                .get(idx)
                .copied()
                .unwrap_or(1e-8)
                .max(1e-8);
            log_probs.push(prob.ln());
        }

        Tensor::vector(log_probs)
    }

    /// Element-wise multiplication of two vectors
    fn elementwise_mul(a: &Tensor, b: &Tensor) -> Tensor {
        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x * y)
            .collect();

        Tensor::vector(result)
    }

    /// Mean of tensor values
    fn mean(tensor: &Tensor) -> Tensor {
        let sum: f64 = tensor.data.iter().sum();
        Tensor::scalar(sum / tensor.data.len() as f64)
    }

    /// Variance of tensor values
    fn variance(tensor: &Tensor, mean: f64) -> f64 {
        let sum_sq_diff: f64 = tensor
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum();
        sum_sq_diff / tensor.data.len() as f64
    }
}

/// Proximal Policy Optimization (PPO) with Clipped Objective
///
/// On-policy algorithm that uses a clipped surrogate objective
/// to prevent too large policy updates.
pub struct PPOClip<P, V>
where
    P: Fn(&Tensor) -> Tensor + 'static,
    V: Fn(&Tensor) -> Tensor + 'static,
{
    /// Policy network (old policy for ratio computation)
    pub policy: Box<P>,
    /// Value function network
    pub value_fn: Box<V>,
    /// Clipping parameter (typically 0.2)
    pub clip_eps: f64,
    /// Entropy coefficient for exploration
    pub entropy_coef: f64,
    /// Discount factor
    pub gamma: f64,
    /// GAE lambda parameter
    pub gae_lambda: f64,
}

impl<P, V> PPOClip<P, V>
where
    P: Fn(&Tensor) -> Tensor + 'static,
    V: Fn(&Tensor) -> Tensor + 'static,
{
    /// Create new PPO optimizer
    pub fn new(policy: P, value_fn: V, clip_eps: f64, entropy_coef: f64) -> Self {
        Self {
            policy: Box::new(policy),
            value_fn: Box::new(value_fn),
            clip_eps,
            entropy_coef,
            gamma: 0.99,
            gae_lambda: 0.95,
        }
    }

    /// Set discount factor
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set GAE lambda
    pub fn with_gae_lambda(mut self, gae_lambda: f64) -> Self {
        self.gae_lambda = gae_lambda;
        self
    }

    /// Forward pass - get action logits and values
    pub fn forward(&self, states: &Tensor) -> (Tensor, Tensor) {
        let actions = (self.policy)(states);
        let values = (self.value_fn)(states);
        (actions, values)
    }

    /// Compute PPO clipped loss
    ///
    /// L_clip = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
    /// where ratio = π_new(a|s) / π_old(a|s)
    pub fn ppo_loss(
        &self,
        old_log_probs: &Tensor,
        new_log_probs: &Tensor,
        advantages: &Tensor,
    ) -> Tensor {
        // Compute probability ratio
        let log_ratio = Self::sub(new_log_probs, old_log_probs);
        let ratio = Self::exp(&log_ratio);

        // Clipped surrogate objective
        let clipped_ratio = Self::clip(&ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps);

        // Policy loss: -min(ratio * A, clipped_ratio * A)
        let surr1 = Self::elementwise_mul(&ratio, advantages);
        let surr2 = Self::elementwise_mul(&clipped_ratio, advantages);

        let policy_loss = Self::neg(&Self::elementwise_min(&surr1, &surr2));

        // Mean loss
        Self::mean(&policy_loss)
    }

    /// Compute advantages using GAE
    pub fn compute_advantages(&self, rewards: &Tensor, values: &Tensor) -> Tensor {
        Self::compute_gae(rewards, values, self.gamma, self.gae_lambda)
    }

    /// Compute total PPO loss (policy + value + entropy)
    pub fn total_loss(
        &self,
        old_log_probs: &Tensor,
        new_log_probs: &Tensor,
        advantages: &Tensor,
        value_preds: &Tensor,
        value_targets: &Tensor,
        entropy: &Tensor,
    ) -> Tensor {
        let policy_loss = self.ppo_loss(old_log_probs, new_log_probs, advantages);
        let value_loss = Self::value_mse_loss(value_preds, value_targets);
        let entropy_penalty = Self::scalar_mul(&entropy, -self.entropy_coef);

        // Total loss = policy_loss + value_loss - entropy_coef * entropy
        let total = Self::add(&policy_loss, &value_loss);
        Self::add(&total, &entropy_penalty)
    }

    /// GAE computation
    fn compute_gae(rewards: &Tensor, values: &Tensor, gamma: f64, lambda: f64) -> Tensor {
        let mut advantages = Vec::with_capacity(rewards.data.len());
        let mut gae = 0.0;

        // Iterate backwards for TD(lambda) computation
        for i in (0..rewards.data.len()).rev() {
            let reward = rewards.data[i];
            let value = values.data[i];
            let next_value = if i + 1 < values.data.len() {
                values.data[i + 1]
            } else {
                0.0 // Terminal state
            };

            // TD residual
            let td_error = reward + gamma * next_value - value;

            // GAE accumulation
            gae = td_error + gamma * lambda * gae;
            advantages.push(gae);
        }

        advantages.reverse();
        Tensor::vector(advantages)
    }

    /// Clip values to range
    fn clip(tensor: &Tensor, min: f64, max: f64) -> Tensor {
        let clipped: Vec<f64> = tensor.data.iter().map(|&x| x.max(min).min(max)).collect();
        Tensor::vector(clipped)
    }

    /// Element-wise subtraction
    fn sub(a: &Tensor, b: &Tensor) -> Tensor {
        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x - y)
            .collect();
        Tensor::vector(result)
    }

    /// Element-wise addition
    fn add(a: &Tensor, b: &Tensor) -> Tensor {
        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::vector(result)
    }

    /// Element-wise minimum
    fn elementwise_min(a: &Tensor, b: &Tensor) -> Tensor {
        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x.min(y))
            .collect();
        Tensor::vector(result)
    }

    /// Element-wise multiplication
    fn elementwise_mul(a: &Tensor, b: &Tensor) -> Tensor {
        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x * y)
            .collect();
        Tensor::vector(result)
    }

    /// Element-wise exp
    fn exp(tensor: &Tensor) -> Tensor {
        let result: Vec<f64> = tensor.data.iter().map(|&x| x.exp()).collect();
        Tensor::vector(result)
    }

    /// Element-wise negation
    fn neg(tensor: &Tensor) -> Tensor {
        let result: Vec<f64> = tensor.data.iter().map(|&x| -x).collect();
        Tensor::vector(result)
    }

    /// Scalar multiplication
    fn scalar_mul(tensor: &Tensor, scalar: f64) -> Tensor {
        let result: Vec<f64> = tensor.data.iter().map(|&x| x * scalar).collect();
        Tensor::vector(result)
    }

    /// Mean of tensor
    fn mean(tensor: &Tensor) -> Tensor {
        let sum: f64 = tensor.data.iter().sum();
        Tensor::scalar(sum / tensor.data.len() as f64)
    }

    /// MSE loss for value function
    fn value_mse_loss(preds: &Tensor, targets: &Tensor) -> Tensor {
        let mse: Vec<f64> = preds
            .data
            .iter()
            .zip(targets.data.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                diff * diff
            })
            .collect();

        let sum: f64 = mse.iter().sum();
        Tensor::scalar(sum / mse.len() as f64)
    }
}

/// Value Function (State Value V(s))
///
/// Estimates the expected return from a given state.
pub struct ValueFunction<F>
where
    F: Fn(&Tensor) -> Tensor + 'static,
{
    /// Value network
    pub network: Box<F>,
}

impl<F> ValueFunction<F>
where
    F: Fn(&Tensor) -> Tensor + 'static,
{
    /// Create new value function
    pub fn new(network: F) -> Self {
        Self {
            network: Box::new(network),
        }
    }

    /// Forward pass - compute state values
    pub fn forward(&self, states: &Tensor) -> Tensor {
        (self.network)(states)
    }

    /// Compute value loss (MSE between predicted and target values)
    pub fn value_loss(&self, predicted: &Tensor, targets: &Tensor) -> Tensor {
        let mse: Vec<f64> = predicted
            .data
            .iter()
            .zip(targets.data.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                diff * diff
            })
            .collect();

        let sum: f64 = mse.iter().sum();
        Tensor::scalar(sum / mse.len() as f64)
    }
}

/// Advantage Estimation Functions
///
/// Standalone functions for computing advantages and returns.
pub struct AdvantageEstimation;

impl AdvantageEstimation {
    /// Compute Generalized Advantage Estimation (GAE)
    ///
    /// GAE balances bias and variance in advantage estimates.
    /// λ=0 gives TD(0) (high bias, low variance)
    /// λ=1 gives Monte Carlo (low bias, high variance)
    ///
    /// A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    /// where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    pub fn compute_gae(rewards: &Tensor, values: &Tensor, gamma: f64, lambda: f64) -> Tensor {
        let mut advantages = Vec::with_capacity(rewards.data.len());
        let mut gae = 0.0;

        // Iterate backwards
        for i in (0..rewards.data.len()).rev() {
            let reward = rewards.data[i];
            let value = values.data[i];
            let next_value = if i + 1 < values.data.len() {
                values.data[i + 1]
            } else {
                0.0 // Terminal state
            };

            // TD error
            let td_error = reward + gamma * next_value - value;

            // GAE
            gae = td_error + gamma * lambda * gae;
            advantages.push(gae);
        }

        advantages.reverse();
        Tensor::vector(advantages)
    }

    /// Discount rewards to compute returns
    ///
    /// G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^{T-t-1}r_{T-1}
    pub fn discount_rewards(rewards: &Tensor, gamma: f64) -> Tensor {
        let mut discounted = Vec::with_capacity(rewards.data.len());
        let mut running_sum = 0.0;

        // Iterate backwards
        for &reward in rewards.data.iter().rev() {
            running_sum = reward + gamma * running_sum;
            discounted.push(running_sum);
        }

        discounted.reverse();
        Tensor::vector(discounted)
    }

    /// Compute both advantages and returns using GAE
    ///
    /// Returns (advantages, returns) where returns = advantages + values
    pub fn compute_gae_with_returns(
        rewards: &Tensor,
        values: &Tensor,
        gamma: f64,
        lambda: f64,
    ) -> (Tensor, Tensor) {
        let advantages = Self::compute_gae(rewards, values, gamma, lambda);

        // Returns = advantages + values
        let returns: Vec<f64> = advantages
            .data
            .iter()
            .zip(values.data.iter())
            .map(|(&a, &v)| a + v)
            .collect();

        (advantages, Tensor::vector(returns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_policy(states: &Tensor) -> Tensor {
        // Simple linear policy for testing
        Tensor::vector(vec![0.7, 0.3])
    }

    fn simple_value_fn(states: &Tensor) -> Tensor {
        Tensor::scalar(5.0)
    }

    #[test]
    fn test_policy_gradient_discount_rewards() {
        let rewards = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let gamma = 0.9;

        // Manual computation:
        // t=2: 3.0
        // t=1: 2.0 + 0.9*3.0 = 4.7
        // t=0: 1.0 + 0.9*4.7 = 5.23

        let discounted = PolicyGradient::<fn(&Tensor) -> Tensor>::discount_rewards(&rewards, gamma);

        assert!((discounted.data[0] - 5.23).abs() < 0.01);
        assert!((discounted.data[1] - 4.7).abs() < 0.01);
        assert!((discounted.data[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_gae_computation() {
        let rewards = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let values = Tensor::vector(vec![5.0, 6.0, 7.0]);
        let gamma = 0.9;
        let lambda = 0.95;

        let advantages = AdvantageEstimation::compute_gae(&rewards, &values, gamma, lambda);

        // Advantages should be computed
        assert_eq!(advantages.data.len(), 3);
        // Last step: r + gamma*0 - V = 3 + 0 - 7 = -4
        assert!((advantages.data[2] - (-4.0)).abs() < 0.1);
    }

    #[test]
    fn test_discount_rewards() {
        let rewards = Tensor::vector(vec![1.0, 1.0, 1.0]);
        let gamma = 0.5;

        let discounted = AdvantageEstimation::discount_rewards(&rewards, gamma);

        // 1 + 0.5 + 0.25 = 1.75
        // 1 + 0.5 = 1.5
        // 1
        assert!((discounted.data[0] - 1.75).abs() < 0.01);
        assert!((discounted.data[1] - 1.5).abs() < 0.01);
        assert!((discounted.data[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ppo_clip_ratio() {
        let ppo = PPOClip::new(simple_policy, simple_value_fn, 0.2, 0.01);

        let old_log_probs = Tensor::vector(vec![-0.3, -0.5]);
        let new_log_probs = Tensor::vector(vec![-0.2, -0.6]);
        let advantages = Tensor::vector(vec![1.0, 1.0]);

        let loss = ppo.ppo_loss(&old_log_probs, &new_log_probs, &advantages);

        // Loss should be computed
        assert!(loss.data.len() > 0);
        // Loss is negative because PPO maximizes the objective (we minimize negative)
        // The loss should be negative since we have positive advantages
        assert!(loss.data[0] < 0.0);
    }

    #[test]
    fn test_value_function_loss() {
        let vf = ValueFunction::new(simple_value_fn);

        let predicted = Tensor::vector(vec![5.0, 6.0, 7.0]);
        let targets = Tensor::vector(vec![5.5, 6.0, 6.5]);

        let loss = vf.value_loss(&predicted, &targets);

        // MSE: (0.25 + 0 + 0.25) / 3 = 0.166...
        assert!((loss.data[0] - 0.166).abs() < 0.01);
    }

    #[test]
    fn test_gae_with_returns() {
        let rewards = Tensor::vector(vec![1.0, 2.0]);
        let values = Tensor::vector(vec![5.0, 6.0]);

        let (advantages, returns) =
            AdvantageEstimation::compute_gae_with_returns(&rewards, &values, 0.99, 0.95);

        assert_eq!(advantages.data.len(), 2);
        assert_eq!(returns.data.len(), 2);

        // Returns should be greater than or equal to advantages (since values are positive)
        for (adv, ret) in advantages.data.iter().zip(returns.data.iter()) {
            assert!(ret >= adv);
        }
    }

    #[test]
    fn test_normalize_returns() {
        let returns = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let gamma = 0.99;

        let pg = PolicyGradient::new(simple_policy, gamma);
        let normalized = pg.normalize_returns(&returns);

        // Mean should be close to 0
        let mean: f64 = normalized.data.iter().sum::<f64>() / normalized.data.len() as f64;
        assert!(mean.abs() < 0.1);

        // Std should be close to 1
        let variance: f64 = normalized
            .data
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / normalized.data.len() as f64;
        assert!((variance.sqrt() - 1.0).abs() < 0.1);
    }
}
