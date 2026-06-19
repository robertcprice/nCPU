//! Reinforcement Learning Experience Replay and Actor-Critic Methods
//!
//! Comprehensive implementations of:
//! - Experience replay buffers (uniform and prioritized)
//! - Actor-Critic architectures (A3C, A2C-style)
//! - Rollout buffers for on-policy algorithms (PPO, TRPO)
//! - GAE (Generalized Advantage Estimation)

use crate::tensor::ops::{Shape, Tensor};
use std::collections::VecDeque;

/// Experience tuple: (state, action, reward, next_state, done)
pub type Experience = (Tensor, Tensor, Tensor, Tensor, bool);

/// Priority-based experience tuple with index
#[derive(Debug, Clone)]
pub struct PrioritizedExperience {
    pub state: Tensor,
    pub action: Tensor,
    pub reward: Tensor,
    pub next_state: Tensor,
    pub done: bool,
    pub priority: f64,
    pub index: usize,
}

// ============================================================================
// Replay Buffer - Uniform Experience Replay
// ============================================================================

/// Standard experience replay buffer with uniform sampling
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    /// Maximum capacity of the buffer
    capacity: usize,
    /// Ring buffer of states
    states: VecDeque<Tensor>,
    /// Ring buffer of actions
    actions: VecDeque<Tensor>,
    /// Ring buffer of rewards
    rewards: VecDeque<Tensor>,
    /// Ring buffer of next states
    next_states: VecDeque<Tensor>,
    /// Ring buffer of done flags
    dones: VecDeque<bool>,
    /// Current size of the buffer
    size: usize,
    /// State dimension for validation
    state_dim: usize,
    /// Action dimension for validation
    action_dim: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer with specified capacity
    pub fn new(capacity: usize, state_dim: usize, action_dim: usize) -> Self {
        Self {
            capacity,
            states: VecDeque::with_capacity(capacity),
            actions: VecDeque::with_capacity(capacity),
            rewards: VecDeque::with_capacity(capacity),
            next_states: VecDeque::with_capacity(capacity),
            dones: VecDeque::with_capacity(capacity),
            size: 0,
            state_dim,
            action_dim,
        }
    }

    /// Add a new experience to the buffer
    pub fn push(&mut self, experience: Experience) {
        let (state, action, reward, next_state, done) = experience;

        // Validate dimensions
        assert_eq!(
            state.shape.dims.iter().product::<usize>(),
            self.state_dim,
            "State dimension mismatch"
        );
        assert_eq!(
            action.shape.dims.iter().product::<usize>(),
            self.action_dim,
            "Action dimension mismatch"
        );

        // If at capacity, remove oldest
        if self.states.len() >= self.capacity {
            self.states.pop_front();
            self.actions.pop_front();
            self.rewards.pop_front();
            self.next_states.pop_front();
            self.dones.pop_front();
        } else {
            self.size = self.states.len();
        }

        self.states.push_back(state);
        self.actions.push_back(action);
        self.rewards.push_back(reward);
        self.next_states.push_back(next_state);
        self.dones.push_back(done);
        self.size = self.states.len();
    }

    /// Sample a batch of experiences uniformly at random
    pub fn sample(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        if self.size < batch_size {
            return None;
        }

        let indices: Vec<usize> = (0..self.size)
            .collect::<Vec<_>>()
            .iter()
            .cloned()
            .collect::<Vec<_>>();

        // Sample batch_size indices
        let mut batch_indices = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = indices[rand::random::<usize>() % indices.len()];
            batch_indices.push(idx);
        }

        // Gather batch data
        let batch_states = self.gather_states(&batch_indices);
        let batch_actions = self.gather_actions(&batch_indices);
        let batch_rewards = self.gather_rewards(&batch_indices);
        let batch_next_states = self.gather_next_states(&batch_indices);
        let batch_dones = self.gather_dones(&batch_indices);

        Some((
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
        ))
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.next_states.clear();
        self.dones.clear();
        self.size = 0;
    }

    // --- Helper methods for gathering batches ---

    fn gather_states(&self, indices: &[usize]) -> Tensor {
        let batch_data: Vec<f64> = indices
            .iter()
            .flat_map(|&idx| self.states[idx].data.clone())
            .collect();
        let shape = Shape::new(vec![indices.len(), self.state_dim]);
        Tensor::new(batch_data, shape)
    }

    fn gather_actions(&self, indices: &[usize]) -> Tensor {
        let batch_data: Vec<f64> = indices
            .iter()
            .flat_map(|&idx| self.actions[idx].data.clone())
            .collect();
        let shape = Shape::new(vec![indices.len(), self.action_dim]);
        Tensor::new(batch_data, shape)
    }

    fn gather_rewards(&self, indices: &[usize]) -> Tensor {
        let batch_data: Vec<f64> = indices
            .iter()
            .map(|&idx| self.rewards[idx].data[0])
            .collect();
        Tensor::new(batch_data, Shape::new(vec![indices.len(), 1]))
    }

    fn gather_next_states(&self, indices: &[usize]) -> Tensor {
        let batch_data: Vec<f64> = indices
            .iter()
            .flat_map(|&idx| self.next_states[idx].data.clone())
            .collect();
        let shape = Shape::new(vec![indices.len(), self.state_dim]);
        Tensor::new(batch_data, shape)
    }

    fn gather_dones(&self, indices: &[usize]) -> Tensor {
        let batch_data: Vec<f64> = indices
            .iter()
            .map(|&idx| if self.dones[idx] { 1.0 } else { 0.0 })
            .collect();
        Tensor::new(batch_data, Shape::new(vec![indices.len(), 1]))
    }
}

// ============================================================================
// Prioritized Experience Replay (PER)
// ============================================================================

/// Prioritized Experience Replay buffer using proportional prioritization
/// Based on "Prioritized Experience Replay" (Schaul et al., 2015)
#[derive(Debug, Clone)]
pub struct PrioritizedReplay {
    /// Base replay buffer
    buffer: ReplayBuffer,
    /// Priorities for each experience
    priorities: VecDeque<f64>,
    /// Priority exponent (alpha): how much prioritization to use
    /// alpha=0 -> uniform sampling, alpha=1 -> full prioritization
    alpha: f64,
    /// Importance sampling exponent (beta): compensates for bias
    /// Starts at beta_start and anneals to 1.0
    beta: f64,
    /// Initial priority for new experiences
    default_priority: f64,
    /// Maximum priority (for normalization)
    max_priority: f64,
}

impl PrioritizedReplay {
    /// Create a new prioritized replay buffer
    pub fn new(capacity: usize, state_dim: usize, action_dim: usize, alpha: f64) -> Self {
        Self {
            buffer: ReplayBuffer::new(capacity, state_dim, action_dim),
            priorities: VecDeque::with_capacity(capacity),
            alpha,
            beta: 0.4, // Default beta
            default_priority: 1.0,
            max_priority: 1.0,
        }
    }

    /// Set the beta parameter for importance sampling
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
    }

    /// Get current beta value
    pub fn get_beta(&self) -> f64 {
        self.beta
    }

    /// Add a new experience to the buffer
    pub fn push(&mut self, experience: Experience) {
        // Add to base buffer
        self.buffer.push(experience.clone());

        // Add priority (use max priority for new experiences)
        if self.priorities.len() >= self.buffer.capacity {
            self.priorities.pop_front();
        }
        self.priorities.push_back(self.max_priority);
    }

    /// Sample a batch with prioritization
    /// Returns: (states, actions, rewards, next_states, dones, indices, weights)
    pub fn sample(
        &self,
        batch_size: usize,
    ) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Vec<usize>, Tensor)> {
        if self.buffer.is_empty() || self.buffer.len() < batch_size {
            return None;
        }

        let buffer_size = self.buffer.len();

        // Calculate sampling probabilities: p_i^alpha / sum(p^alpha)
        let priorities_alpha: Vec<f64> = self
            .priorities
            .iter()
            .map(|&p| p.powf(self.alpha))
            .collect();
        let sum_priorities: f64 = priorities_alpha.iter().sum();
        let probs: Vec<f64> = priorities_alpha
            .iter()
            .map(|&p| p / sum_priorities)
            .collect();

        // Sample indices based on priorities
        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Sample using proportional prioritization
            let r = rand::random::<f64>();
            let mut cumsum = 0.0;
            let mut sampled_idx = 0;

            for (idx, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if r <= cumsum {
                    sampled_idx = idx;
                    break;
                }
            }

            indices.push(sampled_idx);

            // Calculate importance sampling weight
            // w = (N * p(i)) ^ -beta
            let p_i = probs[sampled_idx];
            let weight = (buffer_size as f64 * p_i).powf(-self.beta);
            weights.push(weight);
        }

        // Normalize weights
        let max_weight = weights.iter().cloned().fold(f64::NAN, f64::max);
        let weights: Vec<f64> = weights.iter().map(|&w| w / max_weight).collect();

        // Gather batch data
        let batch_states = self.gather_states(&indices);
        let batch_actions = self.gather_actions(&indices);
        let batch_rewards = self.gather_rewards(&indices);
        let batch_next_states = self.gather_next_states(&indices);
        let batch_dones = self.gather_dones(&indices);
        let weights_tensor = Tensor::new(weights, Shape::new(vec![batch_size, 1]));

        Some((
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            indices,
            weights_tensor,
        ))
    }

    /// Update priorities for sampled experiences
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &Tensor) {
        for (idx, &priority) in indices.iter().zip(priorities.data.iter()) {
            if let Some(p) = self.priorities.get_mut(*idx) {
                *p = priority;
                self.max_priority = self.max_priority.max(priority);
            }
        }
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.priorities.clear();
        self.max_priority = self.default_priority;
    }

    // --- Helper methods for gathering batches ---

    fn gather_states(&self, indices: &[usize]) -> Tensor {
        let state_dim = self.buffer.state_dim;
        let batch_data: Vec<f64> = indices
            .iter()
            .flat_map(|&idx| self.buffer.states[idx].data.clone())
            .collect();
        let shape = Shape::new(vec![indices.len(), state_dim]);
        Tensor::new(batch_data, shape)
    }

    fn gather_actions(&self, indices: &[usize]) -> Tensor {
        let action_dim = self.buffer.action_dim;
        let batch_data: Vec<f64> = indices
            .iter()
            .flat_map(|&idx| self.buffer.actions[idx].data.clone())
            .collect();
        let shape = Shape::new(vec![indices.len(), action_dim]);
        Tensor::new(batch_data, shape)
    }

    fn gather_rewards(&self, indices: &[usize]) -> Tensor {
        let batch_data: Vec<f64> = indices
            .iter()
            .map(|&idx| self.buffer.rewards[idx].data[0])
            .collect();
        Tensor::new(batch_data, Shape::new(vec![indices.len(), 1]))
    }

    fn gather_next_states(&self, indices: &[usize]) -> Tensor {
        let state_dim = self.buffer.state_dim;
        let batch_data: Vec<f64> = indices
            .iter()
            .flat_map(|&idx| self.buffer.next_states[idx].data.clone())
            .collect();
        let shape = Shape::new(vec![indices.len(), state_dim]);
        Tensor::new(batch_data, shape)
    }

    fn gather_dones(&self, indices: &[usize]) -> Tensor {
        let batch_data: Vec<f64> = indices
            .iter()
            .map(|&idx| if self.buffer.dones[idx] { 1.0 } else { 0.0 })
            .collect();
        Tensor::new(batch_data, Shape::new(vec![indices.len(), 1]))
    }
}

// ============================================================================
// A3C - Asynchronous Actor-Critic
// ============================================================================

/// Asynchronous Actor-Critic (A3C) loss functions
/// Based on "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
#[derive(Debug, Clone)]
pub struct A3C {
    /// Discount factor for future rewards
    pub gamma: f64,
    /// Entropy coefficient for exploration
    pub entropy_coef: f64,
    /// Value function coefficient
    pub value_loss_coef: f64,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
}

impl A3C {
    /// Create a new A3C instance with default parameters
    pub fn new() -> Self {
        Self {
            gamma: 0.99,
            entropy_coef: 0.01,
            value_loss_coef: 0.5,
            max_grad_norm: 40.0,
        }
    }

    /// Create A3C with custom parameters
    pub fn with_params(
        gamma: f64,
        entropy_coef: f64,
        value_loss_coef: f64,
        max_grad_norm: f64,
    ) -> Self {
        Self {
            gamma,
            entropy_coef,
            value_loss_coef,
            max_grad_norm,
        }
    }

    /// Calculate actor (policy) loss
    /// Uses policy gradient: -log_prob * advantage
    pub fn actor_loss(&self, log_probs: &Tensor, advantages: &Tensor) -> Tensor {
        // policy_loss = -mean(log_prob * advantage)
        // Shape: [batch_size] for both

        let mut policy_loss = 0.0;
        let batch_size = log_probs.data.len();

        for i in 0..batch_size {
            policy_loss += log_probs.data[i] * advantages.data[i];
        }

        let mean_policy_loss = policy_loss / batch_size as f64;
        Tensor::new(vec![-mean_policy_loss], Shape::new(vec![1, 1]))
    }

    /// Calculate critic (value) loss
    /// Uses MSE between predicted values and returns
    pub fn critic_loss(&self, values: &Tensor, returns: &Tensor) -> Tensor {
        // value_loss = mean((value - return)^2)

        let mut value_loss = 0.0;
        let batch_size = values.data.len();

        for i in 0..batch_size {
            let diff = values.data[i] - returns.data[i];
            value_loss += diff * diff;
        }

        let mean_value_loss = value_loss / batch_size as f64;
        Tensor::new(vec![mean_value_loss], Shape::new(vec![1, 1]))
    }

    /// Calculate entropy bonus for exploration
    /// entropy_bonus = -mean(policy * log(policy))
    pub fn entropy_bonus(&self, action_probs: &Tensor) -> Tensor {
        let mut entropy = 0.0;
        let num_actions = action_probs.data.len();

        for &prob in action_probs.data.iter() {
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }

        let mean_entropy = entropy / num_actions as f64;
        Tensor::new(vec![mean_entropy], Shape::new(vec![1, 1]))
    }

    /// Calculate combined A3C loss
    /// total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    pub fn total_loss(
        &self,
        log_probs: &Tensor,
        advantages: &Tensor,
        values: &Tensor,
        returns: &Tensor,
        action_probs: &Tensor,
    ) -> Tensor {
        let policy_loss_tensor = self.actor_loss(log_probs, advantages);
        let value_loss_tensor = self.critic_loss(values, returns);
        let entropy_tensor = self.entropy_bonus(action_probs);

        let total_loss = policy_loss_tensor.data[0]
            + self.value_loss_coef * value_loss_tensor.data[0]
            - self.entropy_coef * entropy_tensor.data[0];

        Tensor::new(vec![total_loss], Shape::new(vec![1, 1]))
    }

    /// Compute n-step returns
    pub fn compute_n_step_returns(
        &self,
        rewards: &[f64],
        values: &[f64],
        dones: &[bool],
        bootstrap_value: f64,
    ) -> Vec<f64> {
        let mut returns = Vec::with_capacity(rewards.len());

        for i in 0..rewards.len() {
            let mut ret = 0.0;
            let mut gamma_pow = 1.0;
            let mut done = false;

            for j in i..rewards.len() {
                if dones[j] {
                    done = true;
                    break;
                }
                ret += gamma_pow * rewards[j];
                gamma_pow *= self.gamma;
            }

            if !done {
                ret += gamma_pow * bootstrap_value;
            }

            returns.push(ret);
        }

        returns
    }

    /// Compute advantages using GAE (Generalized Advantage Estimation)
    pub fn compute_gae_advantages(
        &self,
        rewards: &[f64],
        values: &[f64],
        dones: &[bool],
        bootstrap_value: f64,
        gae_lambda: f64,
    ) -> Vec<f64> {
        let mut advantages = vec![0.0; rewards.len()];
        let mut last_advantage = 0.0;
        let mut last_value = bootstrap_value;

        for i in (0..rewards.len()).rev() {
            if dones[i] {
                last_value = 0.0;
                last_advantage = 0.0;
            }

            let delta = rewards[i] + self.gamma * last_value - values[i];
            advantages[i] = delta + self.gamma * gae_lambda * last_advantage;

            last_value = values[i];
            last_advantage = advantages[i];
        }

        advantages
    }
}

impl Default for A3C {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Rollout Buffer - On-Policy Experience Storage
// ============================================================================

/// Rollout buffer for on-policy algorithms (PPO, TRPO, A2C)
/// Stores complete trajectories before updating
#[derive(Debug, Clone)]
pub struct RolloutBuffer {
    /// States encountered
    pub states: Vec<Tensor>,
    /// Actions taken
    pub actions: Vec<Tensor>,
    /// Log probabilities of actions
    pub log_probs: Vec<Tensor>,
    /// Rewards received
    pub rewards: Vec<f64>,
    /// Value function estimates
    pub values: Vec<f64>,
    /// Episode termination flags
    pub dones: Vec<bool>,
    /// Buffer capacity
    capacity: usize,
}

impl RolloutBuffer {
    /// Create a new rollout buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            log_probs: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a step to the rollout buffer
    pub fn push(
        &mut self,
        state: Tensor,
        action: Tensor,
        log_prob: Tensor,
        reward: f64,
        value: f64,
        done: bool,
    ) {
        if self.states.len() >= self.capacity {
            // Clear oldest half if full (simple FIFO)
            let half = self.capacity / 2;
            self.states.drain(0..half);
            self.actions.drain(0..half);
            self.log_probs.drain(0..half);
            self.rewards.drain(0..half);
            self.values.drain(0..half);
            self.dones.drain(0..half);
        }

        self.states.push(state);
        self.actions.push(action);
        self.log_probs.push(log_prob);
        self.rewards.push(reward);
        self.values.push(value);
        self.dones.push(done);
    }

    /// Compute returns using discounted cumulative sum
    pub fn compute_returns(&mut self, gamma: f64, last_value: f64) -> Vec<f64> {
        let mut returns = vec![0.0; self.rewards.len()];
        let mut running_return = last_value;

        for i in (0..self.rewards.len()).rev() {
            if self.dones[i] {
                running_return = 0.0;
            }
            running_return = self.rewards[i] + gamma * running_return;
            returns[i] = running_return;
        }

        returns
    }

    /// Compute GAE (Generalized Advantage Estimation)
    pub fn compute_gae(
        &mut self,
        gamma: f64,
        gae_lambda: f64,
        last_value: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut advantages = vec![0.0; self.rewards.len()];
        let mut last_advantage = 0.0;
        let mut last_value_iter = last_value;

        for i in (0..self.rewards.len()).rev() {
            if self.dones[i] {
                last_value_iter = 0.0;
                last_advantage = 0.0;
            }

            let delta = self.rewards[i] + gamma * last_value_iter - self.values[i];
            advantages[i] = delta + gamma * gae_lambda * last_advantage;

            last_value_iter = self.values[i];
            last_advantage = advantages[i];
        }

        // Returns = advantages + values
        let returns: Vec<f64> = advantages
            .iter()
            .zip(self.values.iter())
            .map(|(&adv, &val)| adv + val)
            .collect();

        (returns, advantages)
    }

    /// Get the current size of the buffer
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Clear all data from the buffer
    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.values.clear();
        self.dones.clear();
    }

    /// Get a batch of data as tensors
    pub fn get_batch(&self) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        if self.is_empty() {
            return None;
        }

        let batch_size = self.len();
        let state_dim = self.states[0].shape.dims.iter().product::<usize>();
        let action_dim = self.actions[0].shape.dims.iter().product::<usize>();

        // Stack states
        let states_data: Vec<f64> = self.states.iter().flat_map(|s| s.data.clone()).collect();
        let states_tensor = Tensor::new(states_data, Shape::new(vec![batch_size, state_dim]));

        // Stack actions
        let actions_data: Vec<f64> = self.actions.iter().flat_map(|a| a.data.clone()).collect();
        let actions_tensor = Tensor::new(actions_data, Shape::new(vec![batch_size, action_dim]));

        // Stack log_probs
        let log_probs_data: Vec<f64> = self
            .log_probs
            .iter()
            .flat_map(|lp| lp.data.clone())
            .collect();
        let log_probs_tensor = Tensor::new(log_probs_data, Shape::new(vec![batch_size, 1]));

        // Rewards as tensor
        let rewards_tensor = Tensor::new(self.rewards.clone(), Shape::new(vec![batch_size, 1]));

        // Values as tensor
        let values_tensor = Tensor::new(self.values.clone(), Shape::new(vec![batch_size, 1]));

        // Dones as tensor
        let dones_data: Vec<f64> = self
            .dones
            .iter()
            .map(|&d| if d { 1.0 } else { 0.0 })
            .collect();
        let dones_tensor = Tensor::new(dones_data, Shape::new(vec![batch_size, 1]));

        Some((
            states_tensor,
            actions_tensor,
            log_probs_tensor,
            rewards_tensor,
            values_tensor,
            dones_tensor,
        ))
    }

    /// Normalize advantages (for PPO)
    pub fn normalize_advantages(advantages: &mut Vec<f64>) {
        if advantages.is_empty() {
            return;
        }

        let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let variance: f64 =
            advantages.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / advantages.len() as f64;
        let std = (variance + 1e-8).sqrt();

        for adv in advantages.iter_mut() {
            *adv = (*adv - mean) / std;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_tensor(shape: Vec<usize>) -> Tensor {
        let size = shape.iter().product::<usize>();
        let data: Vec<f64> = (0..size).map(|_| rand::random::<f64>()).collect();
        Tensor::new(data, Shape::new(shape))
    }

    #[test]
    fn test_replay_buffer_push_and_len() {
        let mut buffer = ReplayBuffer::new(10, 4, 2);
        assert_eq!(buffer.len(), 0);

        let state = random_tensor(vec![4]);
        let action = random_tensor(vec![2]);
        let reward = Tensor::scalar(1.0);
        let next_state = random_tensor(vec![4]);

        buffer.push((
            state.clone(),
            action.clone(),
            reward.clone(),
            next_state.clone(),
            false,
        ));
        assert_eq!(buffer.len(), 1);

        buffer.push((state, action, reward, next_state, false));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let mut buffer = ReplayBuffer::new(5, 4, 2);
        let state = random_tensor(vec![4]);
        let action = random_tensor(vec![2]);
        let reward = Tensor::scalar(1.0);
        let next_state = random_tensor(vec![4]);

        // Fill buffer beyond capacity
        for _ in 0..7 {
            buffer.push((
                state.clone(),
                action.clone(),
                reward.clone(),
                next_state.clone(),
                false,
            ));
        }

        assert_eq!(buffer.len(), 5); // Should be at capacity
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buffer = ReplayBuffer::new(10, 4, 2);

        // Add some experiences
        for _ in 0..5 {
            let state = random_tensor(vec![4]);
            let action = random_tensor(vec![2]);
            let reward = Tensor::scalar(rand::random::<f64>());
            let next_state = random_tensor(vec![4]);
            buffer.push((state, action, reward, next_state, false));
        }

        let batch = buffer.sample(3);
        assert!(batch.is_some());

        let (states, actions, rewards, next_states, dones) = batch.unwrap();
        assert_eq!(states.shape.dims[0], 3); // Batch size
        assert_eq!(states.shape.dims[1], 4); // State dim
        assert_eq!(actions.shape.dims[1], 2); // Action dim
        assert_eq!(rewards.shape.dims[0], 3);
        assert_eq!(dones.shape.dims[0], 3);
    }

    #[test]
    fn test_replay_buffer_clear() {
        let mut buffer = ReplayBuffer::new(10, 4, 2);

        let state = random_tensor(vec![4]);
        let action = random_tensor(vec![2]);
        let reward = Tensor::scalar(1.0);
        let next_state = random_tensor(vec![4]);

        buffer.push((state, action, reward, next_state, false));
        assert_eq!(buffer.len(), 1);

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_prioritized_replay_basic() {
        let mut per_buffer = PrioritizedReplay::new(10, 4, 2, 0.6);

        let state = random_tensor(vec![4]);
        let action = random_tensor(vec![2]);
        let reward = Tensor::scalar(1.0);
        let next_state = random_tensor(vec![4]);

        per_buffer.push((state, action, reward, next_state, false));
        assert_eq!(per_buffer.len(), 1);

        // Need more samples
        for _ in 0..5 {
            let state = random_tensor(vec![4]);
            let action = random_tensor(vec![2]);
            let reward = Tensor::scalar(rand::random::<f64>());
            let next_state = random_tensor(vec![4]);
            per_buffer.push((state, action, reward, next_state, false));
        }

        let batch = per_buffer.sample(3);
        assert!(batch.is_some());

        let (states, actions, rewards, next_states, dones, indices, weights) = batch.unwrap();
        assert_eq!(states.shape.dims[0], 3);
        assert_eq!(indices.len(), 3);
        assert_eq!(weights.shape.dims[0], 3);
    }

    #[test]
    fn test_prioritized_replay_update() {
        let mut per_buffer = PrioritizedReplay::new(10, 4, 2, 0.6);

        // Add experiences
        for _ in 0..5 {
            let state = random_tensor(vec![4]);
            let action = random_tensor(vec![2]);
            let reward = Tensor::scalar(1.0);
            let next_state = random_tensor(vec![4]);
            per_buffer.push((state, action, reward, next_state, false));
        }

        // Sample and update
        let batch = per_buffer.sample(3);
        if let Some((_, _, _, _, _, indices, _)) = batch {
            let new_priorities = Tensor::new(vec![1.5, 2.0, 1.8], Shape::new(vec![3, 1]));
            per_buffer.update_priorities(&indices, &new_priorities);
            assert_eq!(per_buffer.max_priority, 2.0);
        }
    }

    #[test]
    fn test_a3c_actor_loss() {
        let a3c = A3C::new();

        // Create dummy log_probs and advantages
        let log_probs = Tensor::new(vec![-0.5, -1.0, -0.3], Shape::new(vec![3, 1]));
        let advantages = Tensor::new(vec![0.8, -0.5, 1.2], Shape::new(vec![3, 1]));

        let loss = a3c.actor_loss(&log_probs, &advantages);
        // Expected: -((-0.5*0.8 -1.0*-0.5 -0.3*1.2)/3)
        // = -((-0.4 + 0.5 - 0.36)/3) = -((-0.26)/3) = 0.087
        assert!(loss.data[0] > 0.0);
    }

    #[test]
    fn test_a3c_critic_loss() {
        let a3c = A3C::new();

        let values = Tensor::new(vec![1.0, 2.0, 1.5], Shape::new(vec![3, 1]));
        let returns = Tensor::new(vec![1.2, 1.8, 1.3], Shape::new(vec![3, 1]));

        let loss = a3c.critic_loss(&values, &returns);
        assert!(loss.data[0] > 0.0);
    }

    #[test]
    fn test_a3c_entropy_bonus() {
        let a3c = A3C::new();

        // Uniform distribution (max entropy)
        let uniform_probs = Tensor::new(vec![0.25, 0.25, 0.25, 0.25], Shape::new(vec![4, 1]));
        let entropy = a3c.entropy_bonus(&uniform_probs);
        // Entropy formula: -sum(p * ln(p)) / n
        // For uniform [0.25, 0.25, 0.25, 0.25]:
        // Sum: -4 * (0.25 * ln(0.25)) = 1.386
        // Mean (divided by 4): 0.3466 = -ln(0.25) / 4
        let expected = -(0.25_f64).ln() / 4.0;
        assert!((entropy.data[0] - expected).abs() < 0.01);
    }

    #[test]
    fn test_a3c_total_loss() {
        let a3c = A3C::new();

        let log_probs = Tensor::new(vec![-0.5, -1.0], Shape::new(vec![2, 1]));
        let advantages = Tensor::new(vec![0.8, -0.5], Shape::new(vec![2, 1]));
        let values = Tensor::new(vec![1.0, 2.0], Shape::new(vec![2, 1]));
        let returns = Tensor::new(vec![1.2, 1.8], Shape::new(vec![2, 1]));
        // Create action probs with positive entropy
        let action_probs = Tensor::new(vec![0.5, 0.5], Shape::new(vec![2, 1]));

        let total_loss = a3c.total_loss(&log_probs, &advantages, &values, &returns, &action_probs);
        // Loss = policy_loss + value_coef*value_loss - entropy_coef*entropy
        // Just check it's finite (can be positive or negative depending on values)
        assert!(total_loss.data[0].is_finite());
    }

    #[test]
    fn test_a3c_n_step_returns() {
        let a3c = A3C::new();

        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 1.0, 1.5];
        let dones = vec![false, false, false];
        let bootstrap_value = 2.0;

        let returns = a3c.compute_n_step_returns(&rewards, &values, &dones, bootstrap_value);

        // R0 = 1 + 0.99*2 + 0.99^2*3 + 0.99^3*2
        assert!((returns[0] - 7.88).abs() < 0.1);
    }

    #[test]
    fn test_rollout_buffer_basic() {
        let mut buffer = RolloutBuffer::new(10);

        let state = random_tensor(vec![4]);
        let action = random_tensor(vec![2]);
        let log_prob = Tensor::scalar(-0.5);

        buffer.push(
            state.clone(),
            action.clone(),
            log_prob.clone(),
            1.0,
            0.5,
            false,
        );
        assert_eq!(buffer.len(), 1);

        buffer.push(state, action, log_prob, 2.0, 1.0, true);
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_rollout_compute_returns() {
        let mut buffer = RolloutBuffer::new(10);

        buffer.push(
            random_tensor(vec![4]),
            random_tensor(vec![2]),
            Tensor::scalar(-0.5),
            1.0,
            0.5,
            false,
        );
        buffer.push(
            random_tensor(vec![4]),
            random_tensor(vec![2]),
            Tensor::scalar(-0.3),
            2.0,
            1.0,
            false,
        );

        let returns = buffer.compute_returns(0.99, 1.5);
        assert_eq!(returns.len(), 2);
        // R[1] = 2.0 + 0.99*1.5 = 3.485
        // R[0] = 1.0 + 0.99*3.485 = 4.45
        assert!((returns[1] - 3.485).abs() < 0.01);
        assert!((returns[0] - 4.45).abs() < 0.01);
    }

    #[test]
    fn test_rollout_compute_gae() {
        let mut buffer = RolloutBuffer::new(10);

        buffer.push(
            random_tensor(vec![4]),
            random_tensor(vec![2]),
            Tensor::scalar(-0.5),
            1.0,
            0.5,
            false,
        );
        buffer.push(
            random_tensor(vec![4]),
            random_tensor(vec![2]),
            Tensor::scalar(-0.3),
            2.0,
            1.0,
            false,
        );

        let (returns, advantages) = buffer.compute_gae(0.99, 0.95, 1.5);
        assert_eq!(returns.len(), 2);
        assert_eq!(advantages.len(), 2);
    }

    #[test]
    fn test_rollout_get_batch() {
        let mut buffer = RolloutBuffer::new(10);

        for _ in 0..3 {
            buffer.push(
                random_tensor(vec![4]),
                random_tensor(vec![2]),
                Tensor::scalar(rand::random::<f64>() - 1.0),
                rand::random::<f64>(),
                rand::random::<f64>(),
                rand::random::<bool>(),
            );
        }

        let batch = buffer.get_batch();
        assert!(batch.is_some());

        let (states, actions, log_probs, rewards, values, dones) = batch.unwrap();
        assert_eq!(states.shape.dims[0], 3);
        assert_eq!(states.shape.dims[1], 4);
        assert_eq!(actions.shape.dims[1], 2);
        assert_eq!(log_probs.shape.dims[0], 3);
        assert_eq!(rewards.shape.dims[0], 3);
        assert_eq!(values.shape.dims[0], 3);
        assert_eq!(dones.shape.dims[0], 3);
    }

    #[test]
    fn test_normalize_advantages() {
        let mut advantages = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        RolloutBuffer::normalize_advantages(&mut advantages);

        // After normalization: mean=0, std=1
        let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let variance: f64 =
            advantages.iter().map(|&x| x.powi(2)).sum::<f64>() / advantages.len() as f64;

        assert!((mean - 0.0).abs() < 0.001);
        assert!((variance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rollout_buffer_clear() {
        let mut buffer = RolloutBuffer::new(10);

        buffer.push(
            random_tensor(vec![4]),
            random_tensor(vec![2]),
            Tensor::scalar(-0.5),
            1.0,
            0.5,
            false,
        );
        assert_eq!(buffer.len(), 1);

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }
}
