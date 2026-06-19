//! Neural Architecture Search (NAS) Primitives for nCPU/nSynth
//!
//! Differentiable and efficient neural architecture search including:
//! - DARTS: Differentiable Architecture Search
//! - ENAS: Efficient Neural Architecture Search
//! - Search Space definition
//! - Bilevel optimization for architecture and weights

use std::boxed::Box;
use super::ops::{Tensor, Shape};

/// DARTS Cell - Differentiable Architecture Search Cell
///
/// Implements the DARTS approach where architecture parameters are
/// learned jointly with network weights using gradient descent.
/// The cell maintains continuous relaxation over discrete architectures.
pub struct DARTSCell {
    /// Number of computation nodes in the cell
    pub num_nodes: usize,
    /// Candidate operations (each maps Tensor -> Tensor)
    pub op_candidates: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    /// Architecture parameters (alpha weights for each operation)
    pub alphas: Tensor,
    /// Number of operations per edge
    num_ops: usize,
}

impl DARTSCell {
    /// Create a new DARTS cell with specified operations
    ///
    /// # Arguments
    /// * `num_nodes` - Number of computation nodes (typically 4-7)
    /// * `ops` - Vector of candidate operations for each edge
    pub fn new(num_nodes: usize, ops: Vec<Box<dyn Fn(&Tensor) -> Tensor>>) -> Self {
        let num_ops = ops.len();
        let num_edges = num_nodes * (num_nodes + 1) / 2; // Fully connected DAG

        // Initialize architecture parameters (zeros, will be learned)
        let alphas = Tensor::zeros(Shape::new(vec![num_edges, num_ops]));

        Self {
            num_nodes,
            op_candidates: ops,
            alphas,
            num_ops,
        }
    }

    /// Forward pass through the DARTS cell
    ///
    /// Uses softmax-weighted sum of all operations (continuous relaxation)
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// Output tensor after mixed operation computation
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut current_nodes = vec![x.clone()];

        // Each node computes from all previous nodes
        for i in 1..self.num_nodes {
            let mut node_outputs = Vec::new();

            // Connect to all previous nodes
            for j in 0..i {
                let edge_idx = self.edge_index(j, i);

                // Get architecture weights for this edge
                let edge_alphas = self.get_edge_alphas(edge_idx);

                // Compute softmax over operations
                let softmax_weights = self.compute_softmax(&edge_alphas);

                // Apply weighted sum of operations
                let mixed_op = self.apply_mixed_operation(&current_nodes[j], &softmax_weights);
                node_outputs.push(mixed_op);
            }

            // Aggregate all incoming edges (sum)
            let node_output = self.aggregate_edges(&node_outputs);
            current_nodes.push(node_output);
        }

        // Return output of last node
        current_nodes.last().unwrap().clone()
    }

    /// Get architecture parameters for optimization
    ///
    /// # Returns
    /// Flattened tensor of all architecture parameters
    pub fn arch_parameters(&self) -> Tensor {
        self.alphas.clone()
    }

    /// Sample a discrete architecture from the continuous distribution
    ///
    /// Uses Gumbel-Softmax trick for differentiable sampling
    ///
    /// # Returns
    /// Vector of operation indices (one per edge)
    pub fn sample_architecture(&self) -> Vec<usize> {
        let mut selected_ops = Vec::new();
        let num_edges = self.num_nodes * (self.num_nodes + 1) / 2;

        for edge_idx in 0..num_edges {
            let edge_alphas = self.get_edge_alphas(edge_idx);

            // Add Gumbel noise and argmax for sampling
            let sampled_idx = self.gumbel_softmax_sample(&edge_alphas);
            selected_ops.push(sampled_idx);
        }

        selected_ops
    }

    /// Forward pass with a specific discrete architecture
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `arch` - Architecture specification (operation indices per edge)
    ///
    /// # Returns
    /// Output tensor using only the specified operations
    pub fn forward_with_arch(&self, x: &Tensor, arch: &[usize]) -> Tensor {
        let mut current_nodes = vec![x.clone()];
        let mut arch_idx = 0;

        for i in 1..self.num_nodes {
            let mut node_outputs = Vec::new();

            for j in 0..i {
                let op_idx = arch[arch_idx];
                arch_idx += 1;

                // Apply only the selected operation
                let op_output = self.apply_single_operation(&current_nodes[j], op_idx);
                node_outputs.push(op_output);
            }

            let node_output = self.aggregate_edges(&node_outputs);
            current_nodes.push(node_output);
        }

        current_nodes.last().unwrap().clone()
    }

    // Helper: compute linear edge index from (i, j) pair
    fn edge_index(&self, i: usize, j: usize) -> usize {
        // Upper triangular indexing
        let mut idx = 0;
        for row in 0..i {
            idx += self.num_nodes - row - 1;
        }
        idx + (j - i - 1)
    }

    // Helper: get architecture parameters for a specific edge
    fn get_edge_alphas(&self, edge_idx: usize) -> Tensor {
        let start = edge_idx * self.num_ops;
        let end = start + self.num_ops;
        Tensor::vector(self.alphas.data[start..end.min(self.alphas.data.len())].to_vec())
    }

    // Helper: compute softmax for operation weights
    fn compute_softmax(&self, logits: &Tensor) -> Tensor {
        let max_val = logits.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f64 = logits.data.iter()
            .map(|&x| (x - max_val).exp())
            .sum();

        let probs: Vec<f64> = logits.data.iter()
            .map(|&x| ((x - max_val).exp()) / exp_sum)
            .collect();

        Tensor::vector(probs)
    }

    // Helper: apply weighted sum of operations
    fn apply_mixed_operation(&self, input: &Tensor, weights: &Tensor) -> Tensor {
        let mut weighted_data = vec![0.0; input.data.len()];

        for (idx, &weight) in weights.data.iter().enumerate() {
            let op_output = (self.op_candidates[idx])(input);
            for (i, &val) in op_output.data.iter().enumerate() {
                if i < weighted_data.len() {
                    weighted_data[i] += val * weight;
                }
            }
        }

        Tensor::new(weighted_data, input.shape.clone())
    }

    // Helper: apply single operation
    fn apply_single_operation(&self, input: &Tensor, op_idx: usize) -> Tensor {
        let idx = op_idx.min(self.op_candidates.len() - 1);
        (self.op_candidates[idx])(input)
    }

    // Helper: aggregate incoming edges
    fn aggregate_edges(&self, inputs: &[Tensor]) -> Tensor {
        if inputs.is_empty() {
            return Tensor::scalar(0.0);
        }

        let mut sum_data = inputs[0].data.clone();
        for input in &inputs[1..] {
            for (i, &val) in input.data.iter().enumerate() {
                if i < sum_data.len() {
                    sum_data[i] += val;
                }
            }
        }

        Tensor::new(sum_data, inputs[0].shape.clone())
    }

    // Helper: Gumbel-Softmax sampling
    fn gumbel_softmax_sample(&self, logits: &Tensor) -> usize {
        let mut sampled_values = Vec::new();

        for &logit in &logits.data {
            let gumbel_noise = (-(-rand::random::<f64>().ln()).ln()).max(-1e10).min(1e10);
            sampled_values.push(logit + gumbel_noise);
        }

        // Argmax
        sampled_values.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

/// ENAS - Efficient Neural Architecture Search
///
/// Uses a controller network to generate architectures and weight sharing
/// to efficiently explore the search space.
pub struct ENAS {
    /// Controller network (RNN or Transformer) that generates architectures
    pub controller: Box<dyn Fn(&Tensor) -> Tensor>,
    /// Shared weights across all child models
    pub shared_weights: Tensor,
    /// Entropy coefficient for exploration bonus
    pub entropy_coef: f64,
    /// Temperature for softmax (controls exploration)
    pub temperature: f64,
}

impl ENAS {
    /// Create new ENAS optimizer
    ///
    /// # Arguments
    /// * `controller` - Network that generates architecture embeddings
    /// * `shared_weights` - Initial shared weights
    /// * `entropy_coef` - Coefficient for entropy regularization
    pub fn new(
        controller: Box<dyn Fn(&Tensor) -> Tensor>,
        shared_weights: Tensor,
        entropy_coef: f64,
    ) -> Self {
        Self {
            controller,
            shared_weights,
            entropy_coef,
            temperature: 1.0,
        }
    }

    /// Sample a child model architecture from the controller
    ///
    /// # Returns
    /// Tuple of (model_embedding, log_probability)
    pub fn sample_child_model(&self) -> (Tensor, Tensor) {
        // Generate architecture from controller
        let controller_input = Tensor::scalar(0.0); // Dummy input
        let logits = (self.controller)(&controller_input);

        // Apply temperature scaling
        let scaled_logits = Tensor::vector(
            logits.data.iter().map(|&x| x / self.temperature).collect()
        );

        // Sample with log probability computation
        let (arch_embedding, log_prob) = self.sample_with_log_prob(&scaled_logits);

        (arch_embedding, log_prob)
    }

    /// Update controller using policy gradient (REINFORCE)
    ///
    /// # Arguments
    /// * `rewards` - Rewards for sampled architectures
    ///
    /// # Returns
    /// Policy gradient loss for controller update
    pub fn update_controller(&self, rewards: &Tensor) -> Tensor {
        // Normalize rewards for stability
        let normalized_rewards = self.normalize_rewards(rewards);

        // REINFORCE loss: -log_prob * reward
        let controller_input = Tensor::scalar(0.0);
        let logits = (self.controller)(&controller_input);

        // Compute policy loss
        let log_probs = self.compute_log_probs(&logits);
        let loss = self.compute_reinforce_loss(&log_probs, &normalized_rewards);

        loss
    }

    /// Compute REINFORCE loss with baseline subtraction
    ///
    /// # Arguments
    /// * `log_prob` - Log probability of sampled architecture
    /// * `reward` - Reward signal (e.g., validation accuracy)
    ///
    /// # Returns
    /// Scalar loss tensor
    pub fn reinforce_loss(&self, log_prob: &Tensor, reward: f64) -> Tensor {
        // Negative because we maximize reward (minimize negative reward)
        let neg_log_prob = Tensor::vector(
            log_prob.data.iter().map(|&x| -x * reward).collect()
        );
        neg_log_prob
    }

    /// Evaluate a child model with shared weights
    ///
    /// # Arguments
    /// * `arch_embedding` - Architecture specification
    /// * `inputs` - Input data
    ///
    /// # Returns
    /// Model output and entropy bonus
    pub fn evaluate_child_model(&self, arch_embedding: &Tensor, inputs: &Tensor) -> (Tensor, Tensor) {
        // Forward pass with shared weights
        let output = self.forward_with_shared_weights(arch_embedding, inputs);

        // Compute entropy for exploration bonus
        let entropy = self.compute_entropy(arch_embedding);

        (output, entropy)
    }

    // Helper: normalize rewards
    fn normalize_rewards(&self, rewards: &Tensor) -> Tensor {
        if rewards.data.is_empty() {
            return rewards.clone();
        }

        let mean: f64 = rewards.data.iter().sum::<f64>() / rewards.data.len() as f64;
        let variance: f64 = rewards.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / rewards.data.len() as f64;
        let std = variance.sqrt().max(1e-8);

        Tensor::vector(rewards.data.iter()
            .map(|&x| (x - mean) / std)
            .collect())
    }

    // Helper: sample with log probability
    fn sample_with_log_prob(&self, logits: &Tensor) -> (Tensor, Tensor) {
        let max_val = logits.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f64> = logits.data.iter()
            .map(|&x| (x - max_val).exp())
            .collect();
        let exp_sum: f64 = exp_vals.iter().sum();

        // Sample index
        let mut rand_val = rand::random::<f64>();
        let mut selected_idx = 0;
        let mut cum_prob = 0.0;

        for (idx, &exp_val) in exp_vals.iter().enumerate() {
            cum_prob += exp_val / exp_sum;
            if rand_val <= cum_prob {
                selected_idx = idx;
                break;
            }
        }

        // Log probability
        let log_prob = (exp_vals[selected_idx] / exp_sum).ln();

        (Tensor::scalar(selected_idx as f64), Tensor::scalar(log_prob))
    }

    // Helper: compute log probabilities
    fn compute_log_probs(&self, logits: &Tensor) -> Tensor {
        let softmax_probs = self.compute_softmax(logits);
        Tensor::vector(
            softmax_probs.data.iter().map(|&p| p.ln()).collect()
        )
    }

    // Helper: compute softmax
    fn compute_softmax(&self, logits: &Tensor) -> Tensor {
        let max_val = logits.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f64 = logits.data.iter()
            .map(|&x| (x - max_val).exp())
            .sum();

        Tensor::vector(logits.data.iter()
            .map(|&x| (x - max_val).exp() / exp_sum)
            .collect())
    }

    // Helper: forward with shared weights
    fn forward_with_shared_weights(&self, _arch: &Tensor, inputs: &Tensor) -> Tensor {
        // Simplified: return input scaled by shared weights
        let scale_factor = self.shared_weights.data.get(0).copied().unwrap_or(1.0);
        Tensor::vector(
            inputs.data.iter().map(|&x| x * scale_factor).collect()
        )
    }

    // Helper: compute entropy
    fn compute_entropy(&self, probs: &Tensor) -> Tensor {
        let entropy: f64 = probs.data.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        Tensor::scalar(entropy * self.entropy_coef)
    }

    // Helper: compute reinforce loss
    fn compute_reinforce_loss(&self, log_probs: &Tensor, rewards: &Tensor) -> Tensor {
        let loss: f64 = log_probs.data.iter()
            .zip(rewards.data.iter())
            .map(|(&lp, &r)| -lp * r)
            .sum::<f64>() / log_probs.data.len() as f64;

        Tensor::scalar(loss)
    }
}

/// Search Space Definition for Neural Architecture Search
///
/// Defines the valid architectures and provides encoding/decoding.
pub struct SearchSpace {
    /// Layer options for each layer in the network
    pub layers: Vec<LayerOptions>,
    /// Adjacency matrix for skip connections
    pub connections: AdjacencyMatrix,
    /// Encoding dimension
    encoding_dim: usize,
}

/// Options available for a single layer
#[derive(Clone, Debug)]
pub struct LayerOptions {
    /// Number of layer choices (e.g., conv3x3, conv5x5, maxpool)
    pub num_choices: usize,
    /// Valid filter sizes
    pub filter_sizes: Vec<usize>,
    /// Valid channel counts
    pub channels: Vec<usize>,
}

impl LayerOptions {
    /// Create new layer options
    pub fn new(num_choices: usize, filter_sizes: Vec<usize>, channels: Vec<usize>) -> Self {
        Self {
            num_choices,
            filter_sizes,
            channels,
        }
    }

    /// Total number of configurations for this layer
    pub fn total_configurations(&self) -> usize {
        self.num_choices * self.filter_sizes.len() * self.channels.len()
    }
}

/// Adjacency matrix for network connectivity
#[derive(Clone, Debug)]
pub struct AdjacencyMatrix {
    /// Matrix dimensions (num_layers x num_layers)
    pub size: usize,
    /// Valid connections (flattened upper triangular)
    pub valid_connections: Vec<bool>,
}

impl AdjacencyMatrix {
    /// Create new adjacency matrix for DAG
    pub fn new(num_layers: usize) -> Self {
        let total_connections = num_layers * (num_layers - 1) / 2;
        Self {
            size: num_layers,
            valid_connections: vec![true; total_connections],
        }
    }

    /// Check if connection is valid (respects DAG property)
    pub fn is_valid(&self, from: usize, to: usize) -> bool {
        if from >= to {
            return false; // No backward or self connections
        }

        let idx = self.flatten_index(from, to);
        self.valid_connections.get(idx).copied().unwrap_or(false)
    }

    /// Set connection validity
    pub fn set_connection(&mut self, from: usize, to: usize, valid: bool) {
        if from < to {
            let idx = self.flatten_index(from, to);
            if idx < self.valid_connections.len() {
                self.valid_connections[idx] = valid;
            }
        }
    }

    /// Flatten (from, to) to index
    fn flatten_index(&self, from: usize, to: usize) -> usize {
        let mut idx = 0;
        for i in 0..from {
            idx += self.size - i - 1;
        }
        idx + (to - from - 1)
    }
}

impl SearchSpace {
    /// Create new search space
    ///
    /// # Arguments
    /// * `layers` - Layer options for each layer
    /// * `connections` - Valid connectivity patterns
    pub fn new(layers: Vec<LayerOptions>, connections: AdjacencyMatrix) -> Self {
        let encoding_dim = layers.iter()
            .map(|l| l.total_configurations())
            .sum::<usize>() + connections.valid_connections.len();

        Self {
            layers,
            connections,
            encoding_dim,
        }
    }

    /// Sample a random architecture from the search space
    ///
    /// # Returns
    /// Vector of configuration indices
    pub fn random_sample(&self) -> Vec<usize> {
        let mut architecture = Vec::new();

        // Sample layer configurations
        for layer in &self.layers {
            let choice = rand::random::<usize>() % layer.num_choices;
            let filter_idx = rand::random::<usize>() % layer.filter_sizes.len();
            let channel_idx = rand::random::<usize>() % layer.channels.len();

            architecture.push(choice);
            architecture.push(filter_idx);
            architecture.push(channel_idx);
        }

        // Sample skip connections
        for &valid in &self.connections.valid_connections {
            if valid {
                architecture.push(rand::random::<usize>() % 2); // 0 or 1
            } else {
                architecture.push(0);
            }
        }

        architecture
    }

    /// Encode architecture to tensor representation
    ///
    /// # Arguments
    /// * `arch` - Architecture specification
    ///
    /// # Returns
    /// One-hot encoded tensor
    pub fn encode_architecture(&self, arch: &[usize]) -> Tensor {
        let mut encoding = vec![0.0; self.encoding_dim];

        let mut idx = 0;
        let mut arch_idx = 0;

        // Encode layer configurations
        for layer in &self.layers {
            let total = layer.total_configurations();

            // Compute flat index
            let choice = arch[arch_idx];
            let filter_idx = arch[arch_idx + 1];
            let channel_idx = arch[arch_idx + 2];
            arch_idx += 3;

            let flat_idx = choice * layer.filter_sizes.len() * layer.channels.len()
                + filter_idx * layer.channels.len()
                + channel_idx;

            encoding[idx + flat_idx] = 1.0;
            idx += total;
        }

        // Encode connections
        for (j, &valid) in self.connections.valid_connections.iter().enumerate() {
            if valid && arch_idx < arch.len() {
                encoding[idx + j] = arch[arch_idx] as f64;
                arch_idx += 1;
            }
        }

        Tensor::vector(encoding)
    }

    /// Decode tensor to architecture
    ///
    /// # Arguments
    /// * `encoding` - One-hot or probability tensor
    ///
    /// # Returns
    /// Architecture configuration vector
    pub fn decode_architecture(&self, encoding: &Tensor) -> Vec<usize> {
        let mut arch = Vec::new();
        let mut idx = 0;

        // Decode layer configurations
        for layer in &self.layers {
            let total = layer.total_configurations();

            // Find argmax
            let max_idx = encoding.data[idx..idx + total]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            // Decode flat index
            let n_filters = layer.filter_sizes.len();
            let n_channels = layer.channels.len();

            let choice = max_idx / (n_filters * n_channels);
            let remainder = max_idx % (n_filters * n_channels);
            let filter_idx = remainder / n_channels;
            let channel_idx = remainder % n_channels;

            arch.push(choice);
            arch.push(filter_idx);
            arch.push(channel_idx);

            idx += total;
        }

        // Decode connections
        for &valid in &self.connections.valid_connections {
            if idx < encoding.data.len() {
                arch.push(if encoding.data[idx] > 0.5 { 1 } else { 0 });
                idx += 1;
            }
        }

        arch
    }

    /// Validate architecture
    ///
    /// # Arguments
    /// * `arch` - Architecture to validate
    ///
    /// # Returns
    /// True if architecture is valid
    pub fn is_valid(&self, arch: &[usize]) -> bool {
        // Check layer indices are in range
        let mut arch_idx = 0;

        for layer in &self.layers {
            if arch_idx + 2 >= arch.len() {
                return false;
            }

            if arch[arch_idx] >= layer.num_choices {
                return false;
            }
            if arch[arch_idx + 1] >= layer.filter_sizes.len() {
                return false;
            }
            if arch[arch_idx + 2] >= layer.channels.len() {
                return false;
            }

            arch_idx += 3;
        }

        // Check connections
        for &valid in &self.connections.valid_connections {
            if valid && arch_idx < arch.len() && arch[arch_idx] > 1 {
                return false;
            }
            if valid {
                arch_idx += 1;
            }
        }

        true
    }
}

/// NAS Optimizer - Joint optimization of architecture and weights
///
/// Implements bilevel optimization where architecture is optimized
/// on validation set while weights are optimized on training set.
pub struct NasOptimizer {
    /// Learning rate for architecture parameters
    pub arch_lr: f64,
    /// Learning rate for network weights
    pub weight_lr: f64,
    /// Momentum for architecture updates
    pub arch_momentum: f64,
    /// Weight decay for regularization
    pub weight_decay: f64,
}

impl NasOptimizer {
    /// Create new NAS optimizer
    ///
    /// # Arguments
    /// * `arch_lr` - Learning rate for architecture
    /// * `weight_lr` - Learning rate for weights
    pub fn new(arch_lr: f64, weight_lr: f64) -> Self {
        Self {
            arch_lr,
            weight_lr,
            arch_momentum: 0.9,
            weight_decay: 0.0001,
        }
    }

    /// Single optimization step for both architecture and weights
    ///
    /// # Arguments
    /// * `arch_grads` - Gradients for architecture parameters
    /// * `weight_grads` - Gradients for network weights
    ///
    /// # Returns
    /// Tuple of (arch_updates, weight_updates)
    pub fn step(
        &self,
        arch_grads: &Tensor,
        weight_grads: &Tensor,
    ) -> (Tensor, Tensor) {
        // Apply momentum and learning rate to architecture
        let arch_updates = self.compute_arch_updates(arch_grads);

        // Apply weight decay and learning rate to weights
        let weight_updates = self.compute_weight_updates(weight_grads);

        (arch_updates, weight_updates)
    }

    /// Bilevel optimization: optimize arch on val set, weights on train set
    ///
    /// This implements the DARTS validation approximation where:
    /// 1. Weights w* are optimized on training data
    /// 2. Architecture α is optimized on validation data using w*
    ///
    /// # Arguments
    /// * `val_data` - (inputs, targets) for validation
    /// * `train_data` - (inputs, targets) for training
    ///
    /// # Returns
    /// Validation loss for architecture gradient
    pub fn bilevel_optimization(
        &self,
        val_data: &(Tensor, Tensor),
        train_data: &(Tensor, Tensor),
    ) -> Tensor {
        let (val_inputs, val_targets) = val_data;
        let (_train_inputs, _train_targets) = train_data;

        // Simplified: compute validation loss
        // In practice, this would involve:
        // 1. Forward pass on training data
        // 2. Compute weight gradients
        // 3. Update weights (virtual step)
        // 4. Forward pass on validation data with updated weights
        // 5. Compute validation loss
        // 6. Backprop through the entire process

        // Placeholder: return MSE on validation data
        self.compute_mse(val_inputs, val_targets)
    }

    /// First-order approximation (DARTS paper)
    ///
    /// Approximates bilevel optimization without second-order derivatives.
    ///
    /// # Arguments
    /// * `arch_params` - Current architecture parameters
    /// * `weights` - Current network weights
    /// * `val_loss_fn` - Function computing validation loss
    ///
    /// # Returns
    /// Approximate architecture gradient
    pub fn first_order_approximation(
        &self,
        arch_params: &Tensor,
        weights: &Tensor,
        val_loss_fn: &dyn Fn(&Tensor, &Tensor) -> Tensor,
    ) -> Tensor {
        // Compute validation loss gradient wrt architecture
        let val_loss = val_loss_fn(arch_params, weights);

        // Return as architecture gradient (simplified)
        val_loss
    }

    // Helper: compute architecture updates with momentum
    fn compute_arch_updates(&self, grads: &Tensor) -> Tensor {
        let mut updates = Vec::with_capacity(grads.data.len());

        for &grad in &grads.data {
            // Apply learning rate and momentum (simplified)
            let update = -self.arch_lr * grad * self.arch_momentum;
            updates.push(update);
        }

        Tensor::vector(updates)
    }

    // Helper: compute weight updates with decay
    fn compute_weight_updates(&self, grads: &Tensor) -> Tensor {
        let mut updates = Vec::with_capacity(grads.data.len());

        for &grad in &grads.data {
            // Apply learning rate and weight decay
            let decay_term = self.weight_decay * grad;
            let update = -self.weight_lr * (grad + decay_term);
            updates.push(update);
        }

        Tensor::vector(updates)
    }

    // Helper: compute MSE
    fn compute_mse(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mse: f64 = predictions.data.iter()
            .zip(targets.data.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                diff * diff
            })
            .sum::<f64>() / predictions.data.len() as f64;

        Tensor::scalar(mse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper operations for testing
    fn identity_op(x: &Tensor) -> Tensor {
        x.clone()
    }

    fn scale_op(x: &Tensor) -> Tensor {
        Tensor::vector(x.data.iter().map(|&v| v * 2.0).collect())
    }

    fn negate_op(x: &Tensor) -> Tensor {
        Tensor::vector(x.data.iter().map(|&v| -v).collect())
    }

    #[test]
    fn test_darts_cell_creation() {
        let ops: Vec<Box<dyn Fn(&Tensor) -> Tensor>> = vec![
            Box::new(identity_op),
            Box::new(scale_op),
            Box::new(negate_op),
        ];

        let cell = DARTSCell::new(4, ops);

        assert_eq!(cell.num_nodes, 4);
        assert_eq!(cell.num_ops, 3);
        // 4 nodes -> 4*5/2 = 10 edges, each with 3 ops
        assert_eq!(cell.alphas.data.len(), 30);
    }

    #[test]
    fn test_darts_forward() {
        let ops: Vec<Box<dyn Fn(&Tensor) -> Tensor>> = vec![
            Box::new(identity_op),
            Box::new(scale_op),
        ];

        let cell = DARTSCell::new(3, ops);
        let input = Tensor::vector(vec![1.0, 2.0, 3.0]);

        let output = cell.forward(&input);

        // Output should be computed (non-zero due to mixed operations)
        assert!(!output.data.is_empty());
    }

    #[test]
    fn test_darts_sample_architecture() {
        let ops: Vec<Box<dyn Fn(&Tensor) -> Tensor>> = vec![
            Box::new(identity_op),
            Box::new(scale_op),
        ];

        let cell = DARTSCell::new(3, ops);

        let arch = cell.sample_architecture();

        // Should have one operation per edge
        let num_edges = 3 * 4 / 2; // 6 edges for 3 nodes
        assert_eq!(arch.len(), num_edges);
    }

    #[test]
    fn test_darts_arch_parameters() {
        let ops: Vec<Box<dyn Fn(&Tensor) -> Tensor>> = vec![
            Box::new(identity_op),
            Box::new(scale_op),
        ];

        let cell = DARTSCell::new(2, ops);

        let params = cell.arch_parameters();

        // 2 nodes -> 1 edge, 2 operations
        assert_eq!(params.data.len(), 2);
    }

    #[test]
    fn test_enas_creation() {
        let controller = Box::new(|_: &Tensor| Tensor::vector(vec![0.5, 0.3, 0.2]));
        let shared_weights = Tensor::vector(vec![1.0, 2.0, 3.0]);

        let enas = ENAS::new(controller, shared_weights, 0.01);

        assert_eq!(enas.entropy_coef, 0.01);
        assert_eq!(enas.temperature, 1.0);
    }

    #[test]
    fn test_enas_sample_child_model() {
        let controller = Box::new(|_: &Tensor| Tensor::vector(vec![1.0, 2.0, 3.0]));
        let shared_weights = Tensor::scalar(1.0);

        let enas = ENAS::new(controller, shared_weights, 0.01);

        let (_arch, log_prob) = enas.sample_child_model();

        // Log probability should be computed
        assert!(!log_prob.data.is_empty());
    }

    #[test]
    fn test_enas_reinforce_loss() {
        let controller = Box::new(|_: &Tensor| Tensor::vector(vec![0.5, 0.5]));
        let shared_weights = Tensor::scalar(1.0);

        let enas = ENAS::new(controller, shared_weights, 0.01);

        let log_prob = Tensor::scalar(-0.693); // ln(0.5)
        let reward = 0.8;

        let loss = enas.reinforce_loss(&log_prob, reward);

        // Loss should be negative log_prob * reward
        assert!((loss.data[0] - (-(-0.693) * 0.8)).abs() < 0.01);
    }

    #[test]
    fn test_search_space_creation() {
        let layers = vec![
            LayerOptions::new(3, vec![3, 5, 7], vec![32, 64, 128]),
            LayerOptions::new(2, vec![1, 3], vec![64, 128]),
        ];
        let connections = AdjacencyMatrix::new(2);

        let search_space = SearchSpace::new(layers, connections);

        assert_eq!(search_space.layers.len(), 2);
        assert_eq!(search_space.connections.size, 2);
    }

    #[test]
    fn test_search_space_random_sample() {
        let layers = vec![
            LayerOptions::new(2, vec![3, 5], vec![32, 64]),
        ];
        let connections = AdjacencyMatrix::new(1);

        let search_space = SearchSpace::new(layers, connections);

        let arch = search_space.random_sample();

        // Should have layer config (3 values) + connection (1 value)
        assert_eq!(arch.len(), 4);
    }

    #[test]
    fn test_search_space_encode_decode() {
        let layers = vec![
            LayerOptions::new(2, vec![3, 5], vec![32, 64]),
        ];
        let connections = AdjacencyMatrix::new(1);

        let search_space = SearchSpace::new(layers, connections);

        let arch = vec![0, 1, 1, 1]; // choice=0, filter=1, channel=1, connection=1
        let encoded = search_space.encode_architecture(&arch);

        assert!(!encoded.data.is_empty());

        let decoded = search_space.decode_architecture(&encoded);

        // Should decode to same architecture
        assert_eq!(decoded, arch);
    }

    #[test]
    fn test_search_space_is_valid() {
        let layers = vec![
            LayerOptions::new(2, vec![3, 5], vec![32, 64]),
        ];
        let connections = AdjacencyMatrix::new(1);

        let search_space = SearchSpace::new(layers, connections);

        // Valid architecture
        let valid_arch = vec![0, 1, 1, 1];
        assert!(search_space.is_valid(&valid_arch));

        // Invalid: choice out of range
        let invalid_arch = vec![5, 1, 1, 1];
        assert!(!search_space.is_valid(&invalid_arch));
    }

    #[test]
    fn test_nas_optimizer_step() {
        let optimizer = NasOptimizer::new(0.01, 0.1);

        let arch_grads = Tensor::vector(vec![0.1, 0.2, 0.3]);
        let weight_grads = Tensor::vector(vec![0.5, 0.6, 0.7]);

        let (arch_updates, weight_updates) = optimizer.step(&arch_grads, &weight_grads);

        assert!(!arch_updates.data.is_empty());
        assert!(!weight_updates.data.is_empty());

        // Updates should have opposite sign to gradients
        for (i, grad) in arch_grads.data.iter().enumerate() {
            assert!(arch_updates.data[i] * grad < 0.0);
        }
    }

    #[test]
    fn test_nas_optimizer_bilevel() {
        let optimizer = NasOptimizer::new(0.01, 0.1);

        let val_inputs = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let val_targets = Tensor::vector(vec![1.5, 2.5, 3.5]);
        let train_inputs = Tensor::vector(vec![2.0, 3.0, 4.0]);
        let train_targets = Tensor::vector(vec![2.5, 3.5, 4.5]);

        let val_data = (val_inputs, val_targets);
        let train_data = (train_inputs, train_targets);

        let loss = optimizer.bilevel_optimization(&val_data, &train_data);

        // Loss should be computed
        assert!(loss.data.len() > 0);
    }

    #[test]
    fn test_adjacency_matrix_validity() {
        let mut matrix = AdjacencyMatrix::new(4);

        // Valid: forward connection
        assert!(matrix.is_valid(0, 2));

        // Invalid: backward connection
        assert!(!matrix.is_valid(2, 0));

        // Invalid: self connection
        assert!(!matrix.is_valid(1, 1));

        // Disable specific connection
        matrix.set_connection(0, 1, false);
        assert!(!matrix.is_valid(0, 1));
    }

    #[test]
    fn test_layer_options_total_configurations() {
        let layer = LayerOptions::new(3, vec![3, 5, 7], vec![32, 64, 128]);

        // 3 choices * 3 filter sizes * 3 channel counts = 27
        assert_eq!(layer.total_configurations(), 27);
    }
}
