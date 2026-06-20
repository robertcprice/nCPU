//! Meta-Learning Primitives for Learning to Learn
//!
//! Implements MAML (Model-Agnostic Meta-Learning), Reptile, and related
//! few-shot learning algorithms for fast adaptation to new tasks.

use crate::tensor::{Shape, Tensor};

/// MAML: Model-Agnostic Meta-Learning
///
/// Learns an initialization that can be quickly adapted to new tasks
/// with few gradient steps. Meta-learner optimizes for parameters that
/// generalize across tasks, not performance on any single task.
///
/// # References
/// - Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation"
///   ICML 2017
pub struct MAML {
    /// Base model architecture (function from parameters to forward pass)
    base_model: Box<dyn Fn(&Tensor) -> Tensor>,
    /// Inner loop learning rate for task-specific adaptation
    inner_lr: f64,
    /// Meta learning rate for updating initialization
    meta_lr: f64,
    /// Number of gradient steps in inner loop
    num_inner_steps: usize,
}

impl MAML {
    /// Create a new MAML meta-learner
    ///
    /// # Arguments
    /// * `base_model` - Model architecture function
    /// * `inner_lr` - Learning rate for task-specific updates
    /// * `meta_lr` - Learning rate for meta-update
    /// * `num_inner_steps` - Number of inner-loop adaptation steps
    pub fn new<F>(base_model: F, inner_lr: f64, meta_lr: f64, num_inner_steps: usize) -> Self
    where
        F: Fn(&Tensor) -> Tensor + 'static,
    {
        Self {
            base_model: Box::new(base_model),
            inner_lr,
            meta_lr,
            num_inner_steps,
        }
    }

    /// Inner loop: Adapt model to a single support task
    ///
    /// Performs gradient descent on the support set to obtain
    /// task-specific parameters.
    ///
    /// # Arguments
    /// * `model` - Current model parameters
    /// * `support_data` - (inputs, targets) for the support set
    ///
    /// # Returns
    /// Adapted model parameters after inner-loop updates
    pub fn inner_update(&self, model: &Tensor, support_data: &(Tensor, Tensor)) -> Tensor {
        let (_support_x, support_y) = support_data;
        let mut adapted = model.clone();

        for _ in 0..self.num_inner_steps {
            // Forward pass with current adapted parameters
            let output = (self.base_model)(&adapted);
            let loss = inner_loss(&output, support_y);

            // Compute gradient (simplified - in real impl use autodiff)
            let grad = compute_gradient(&loss, &adapted);

            // Gradient descent step: adapted = adapted - lr * grad
            let update_data: Vec<f64> = grad.data.iter().map(|&g| g * self.inner_lr).collect();
            let update = Tensor::new(update_data, grad.shape.clone());
            adapted = adapted.sub(&update).unwrap_or_else(|_| adapted.clone());
        }

        adapted
    }

    /// Meta update: Optimize initialization across multiple tasks
    ///
    /// Performs meta-gradient descent to find initialization that
    /// minimizes loss on query sets after adaptation.
    ///
    /// # Arguments
    /// * `support_tasks` - Support sets for meta-training tasks
    /// * `query_tasks` - Query sets for meta-validation tasks
    ///
    /// # Returns
    /// Updated model parameters after meta-update
    pub fn meta_update(
        &self,
        support_tasks: &[(Tensor, Tensor)],
        query_tasks: &[(Tensor, Tensor)],
    ) -> Tensor {
        // Start with current initialization (simplified)
        let initial_model = self.initialize_parameters();
        let mut meta_params = initial_model.clone();

        // Collect adapted models and query losses for meta-gradient
        let mut adapted_models = Vec::new();

        for (support, _query) in support_tasks.iter().zip(query_tasks.iter()) {
            let adapted = self.inner_update(&meta_params, support);
            adapted_models.push(adapted);
        }

        // Compute meta-loss across all tasks
        let meta_loss = self.meta_loss(&adapted_models, query_tasks);

        // Meta-gradient step (simplified)
        let meta_grad = compute_gradient(&meta_loss, &meta_params);
        let update_data: Vec<f64> = meta_grad.data.iter().map(|&g| g * self.meta_lr).collect();
        let update = Tensor::new(update_data, meta_grad.shape.clone());
        meta_params = meta_params.sub(&update).unwrap_or(meta_params);

        meta_params
    }

    /// Compute meta-loss across adapted models
    ///
    /// Measures how well adapted models perform on query sets.
    ///
    /// # Arguments
    /// * `adapted_models` - Task-adapted models
    /// * `query_data` - Query sets for evaluating adaptation
    ///
    /// # Returns
    /// Average query loss across all tasks
    pub fn meta_loss(&self, adapted_models: &[Tensor], query_data: &[(Tensor, Tensor)]) -> Tensor {
        let mut total_loss = Tensor::zeros(Shape::new(vec![1]));

        for (model, (_query_x, query_y)) in adapted_models.iter().zip(query_data.iter()) {
            let output = (self.base_model)(model);
            let task_loss = inner_loss(&output, query_y);
            total_loss = total_loss.add(&task_loss).unwrap_or(total_loss);
        }

        // Average over tasks
        let num_tasks = adapted_models.len() as f64;
        Tensor::vector(total_loss.data.iter().map(|&v| v / num_tasks).collect())
    }

    /// Initialize model parameters (placeholder)
    fn initialize_parameters(&self) -> Tensor {
        // In real impl, initialize from base model architecture
        Tensor::zeros(Shape::new(vec![100]))
    }
}

/// Reptile: First-Order MAML Approximation
///
/// Simplified meta-learning that directly interpolates between
/// initialization and task-specific parameters, avoiding second-order
/// derivatives. More computationally efficient than full MAML.
///
/// # References
/// - Nichol et al., "On First-Order Meta-Learning Algorithms"
///   2018
pub struct Reptile {
    /// Current model parameters
    model: Tensor,
    /// Meta learning rate for interpolation
    meta_lr: f64,
    /// Number of inner-loop adaptation steps
    num_inner_steps: usize,
}

impl Reptile {
    /// Create a new Reptile meta-learner
    ///
    /// # Arguments
    /// * `model` - Initial model parameters
    /// * `meta_lr` - Learning rate for meta-update (interpolation)
    /// * `num_inner_steps` - Gradient steps per task
    pub fn new(model: Tensor, meta_lr: f64, num_inner_steps: usize) -> Self {
        Self {
            model,
            meta_lr,
            num_inner_steps,
        }
    }

    /// Inner loop: Adapt to a single task
    ///
    /// Performs standard gradient descent on the task.
    ///
    /// # Arguments
    /// * `model` - Current model parameters
    /// * `task` - (inputs, targets) for the task
    ///
    /// # Returns
    /// Task-adapted parameters
    pub fn inner_update(&self, model: &Tensor, task: &(Tensor, Tensor)) -> Tensor {
        let (_task_x, task_y) = task;
        let mut adapted = model.clone();

        for _ in 0..self.num_inner_steps {
            let output = forward_simple(&adapted);
            let loss = inner_loss(&output, task_y);
            let grad = compute_gradient(&loss, &adapted);
            let update_data: Vec<f64> = grad.data.iter().map(|&g| g * 0.01).collect();
            let update = Tensor::new(update_data, grad.shape.clone());
            adapted = adapted.sub(&update).unwrap_or_else(|_| adapted.clone());
        }

        adapted
    }

    /// Reptile meta-update
    ///
    /// Moves initialization toward task-specific parameters:
    /// θ ← θ - α(θ - θ_task)
    ///
    /// # Arguments
    /// * `tasks` - Batch of meta-training tasks
    ///
    /// # Returns
    /// Updated model parameters
    pub fn reptile_update(&self, tasks: &[(Tensor, Tensor)]) -> Tensor {
        let mut adapted_params = Vec::new();

        for task in tasks {
            let adapted = self.inner_update(&self.model, task);
            adapted_params.push(adapted);
        }

        // Average adapted parameters
        let avg_adapted = average_tensors(&adapted_params);

        // Reptile update: interpolate toward adapted
        // θ ← θ + α * (θ_task - θ) = θ + α * diff
        let diff = avg_adapted
            .sub(&self.model)
            .unwrap_or_else(|_| avg_adapted.clone());
        let scaled_diff = Tensor::vector(diff.data.iter().map(|&d| d * self.meta_lr).collect());
        self.model.add(&scaled_diff).unwrap_or(self.model.clone())
    }
}

/// Task sampling and batching for meta-learning
///
/// Manages task distributions for few-shot learning, handling
/// support/query splits and task sampling.
pub struct TaskBatch {
    /// Available tasks for meta-training/evaluation
    tasks: Vec<(Tensor, Tensor)>,
    /// Number of tasks per meta-batch
    batch_size: usize,
}

impl TaskBatch {
    /// Create a new task batcher
    ///
    /// # Arguments
    /// * `tasks` - Collection of (inputs, targets) tasks
    /// * `batch_size` - Tasks to sample per meta-batch
    pub fn new(tasks: Vec<(Tensor, Tensor)>, batch_size: usize) -> Self {
        Self { tasks, batch_size }
    }

    /// Sample random tasks for meta-batch
    ///
    /// # Arguments
    /// * `num_tasks` - Number of tasks to sample
    ///
    /// # Returns
    /// Random subset of tasks
    pub fn sample_tasks(&self, num_tasks: usize) -> Vec<(Tensor, Tensor)> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.tasks
            .choose_multiple(&mut rng, num_tasks)
            .cloned()
            .collect()
    }

    /// Split task data into support and query sets
    ///
    /// For N-shot K-way classification, splits each task into
    /// support (adaptation) and query (evaluation) portions.
    ///
    /// # Arguments
    /// * `task` - Full task data (inputs, targets)
    /// * `support_size` - Fraction (0-1) of data for support set
    ///
    /// # Returns
    /// ((support_x, support_y), (query_x, query_y))
    pub fn split_support_query(
        &self,
        task: &(Tensor, Tensor),
        support_size: f64,
    ) -> ((Tensor, Tensor), (Tensor, Tensor)) {
        let (all_x, all_y) = task;
        let total = all_x.shape.dims[0];

        if total == 0 {
            // Handle edge case of empty tensors
            let empty_shape = Shape::new(vec![0]);
            return (
                (
                    Tensor::new(vec![], empty_shape.clone()),
                    Tensor::new(vec![], empty_shape.clone()),
                ),
                (
                    Tensor::new(vec![], empty_shape.clone()),
                    Tensor::new(vec![], empty_shape),
                ),
            );
        }

        let support_count = std::cmp::max(1, (total as f64 * support_size) as usize);
        let support_end = support_count.saturating_sub(1);
        let query_start = support_count;
        let query_end = total.saturating_sub(1);

        // Split using proper RangeInclusive syntax
        let support_x = all_x.index(&[0..=support_end]);
        let support_y = all_y.index(&[0..=support_end]);
        let query_x = if query_start < total {
            all_x.index(&[query_start..=query_end])
        } else {
            // Empty query set
            Tensor::new(vec![], Shape::new(vec![0]))
        };
        let query_y = if query_start < total {
            all_y.index(&[query_start..=query_end])
        } else {
            Tensor::new(vec![], Shape::new(vec![0]))
        };

        ((support_x, support_y), (query_x, query_y))
    }
}

/// Meta-learning loss functions
///
/// Computes losses for meta-optimization and inner-loop adaptation.
pub struct MetaLoss;

impl MetaLoss {
    /// Compute meta-loss across adapted models
    ///
    /// Measures adaptation quality on query sets.
    ///
    /// # Arguments
    /// * `adapted_outputs` - Model outputs after adaptation
    /// * `query_targets` - Ground truth for query sets
    ///
    /// # Returns
    /// Average meta-loss (MSE or cross-entropy)
    pub fn meta_loss(adapted_outputs: &[Tensor], query_targets: &[Tensor]) -> Tensor {
        let mut total = Tensor::zeros(Shape::new(vec![1]));

        for (output, target) in adapted_outputs.iter().zip(query_targets.iter()) {
            total = total.add(&inner_loss(output, target)).unwrap_or(total);
        }

        let n = adapted_outputs.len() as f64;
        total.mul_scalar(1.0 / n)
    }

    /// Inner-loop loss (task-specific)
    ///
    /// # Arguments
    /// * `model_output` - Predictions
    /// * `target` - Ground truth
    ///
    /// # Returns
    /// MSE loss (for regression) or cross-entropy (classification)
    pub fn inner_loss(model_output: &Tensor, target: &Tensor) -> Tensor {
        // MSE for simplicity
        let diff = model_output
            .sub(target)
            .unwrap_or_else(|_| model_output.clone());
        let squared = diff.mul(&diff).unwrap_or(diff);
        // Compute mean
        let sum = squared.sum();
        let count = squared.shape.size() as f64;
        Tensor::new(vec![sum / count], Shape::new(vec![1]))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Inner-loop loss (alias for MetaLoss::inner_loss)
fn inner_loss(model_output: &Tensor, target: &Tensor) -> Tensor {
    MetaLoss::inner_loss(model_output, target)
}

/// Compute gradient (simplified placeholder)
fn compute_gradient(_loss: &Tensor, params: &Tensor) -> Tensor {
    // In real impl, use autodiff graph
    // Returns dummy gradient of same shape as params
    Tensor::vector(params.data.iter().map(|_| 0.01).collect())
}

/// Simple forward pass (placeholder)
fn forward_simple(params: &Tensor) -> Tensor {
    // Dummy linear transformation
    Tensor::vector(vec![1.0; params.data.len()])
}

/// Average multiple tensors element-wise
fn average_tensors(tensors: &[Tensor]) -> Tensor {
    if tensors.is_empty() {
        return Tensor::zeros(Shape::new(vec![1]));
    }

    let mut sum = tensors[0].clone();
    for tensor in &tensors[1..] {
        sum = sum.add(tensor).unwrap_or(sum);
    }

    let n = tensors.len() as f64;
    Tensor::vector(sum.data.iter().map(|&v| v / n).collect())
}

// ============================================================================
// Extension methods for Tensor
// ============================================================================

/// Extension trait for meta-learning tensor operations
trait TensorExt {
    fn mul_scalar(&self, scalar: f64) -> Tensor;
    fn add(&self, other: &Tensor) -> Tensor;
    fn sub(&self, other: &Tensor) -> Tensor;
    fn mul(&self, other: &Tensor) -> Tensor;
    fn index(&self, range: &std::ops::Range<usize>) -> Tensor;
}

impl TensorExt for Tensor {
    fn mul_scalar(&self, scalar: f64) -> Tensor {
        let scalar_tensor = Tensor::scalar(scalar);
        self.mul(&scalar_tensor).unwrap_or_else(|_| self.clone())
    }

    fn add(&self, other: &Tensor) -> Tensor {
        Tensor::add(self, other).unwrap_or_else(|_| self.clone())
    }

    fn sub(&self, other: &Tensor) -> Tensor {
        Tensor::sub(self, other).unwrap_or_else(|_| self.clone())
    }

    fn mul(&self, other: &Tensor) -> Tensor {
        Tensor::mul(self, other).unwrap_or_else(|_| self.clone())
    }

    fn index(&self, range: &std::ops::Range<usize>) -> Tensor {
        // Simplified indexing - in real impl use proper tensor slicing
        let start = range.start;
        let end = std::cmp::min(range.end, self.shape.size());
        let new_data = self.data[start..end].to_vec();
        let new_shape = if self.shape.rank() > 0 {
            let mut dims = self.shape.dims.clone();
            dims[0] = end - start;
            Shape::new(dims)
        } else {
            Shape::new(vec![end - start])
        };
        Tensor::new(new_data, new_shape)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_tensor(dims: &[usize], value: f64) -> Tensor {
        let size = dims.iter().product();
        let data = vec![value; size];
        Tensor::new(data, Shape::new(dims.to_vec()))
    }

    #[test]
    fn test_maml_inner_update() {
        let model = create_dummy_tensor(&[10], 1.0);
        let support = (
            create_dummy_tensor(&[5, 10], 0.5),
            create_dummy_tensor(&[5], 0.3),
        );

        let maml = MAML::new(
            |params: &Tensor| params.clone(), // Identity model
            0.01,
            0.001,
            2,
        );

        let adapted = maml.inner_update(&model, &support);

        // Adapted should differ from original
        let diff = adapted.sub(&model).unwrap_or(adapted.clone());
        let abs_diff_sum: f64 = diff.data.iter().map(|x| x.abs()).sum();
        assert!(abs_diff_sum > 0.0);
    }

    #[test]
    fn test_reptile_inner_update() {
        let model = create_dummy_tensor(&[10], 1.0);
        let task = (
            create_dummy_tensor(&[5, 10], 0.5),
            create_dummy_tensor(&[5], 0.3),
        );

        let reptile = Reptile::new(model.clone(), 0.1, 3);

        let adapted = reptile.inner_update(&model, &task);

        // Adapted parameters should exist and have correct shape
        assert_eq!(adapted.shape.dims, model.shape.dims);
    }

    #[test]
    fn test_reptile_meta_update() {
        let model = create_dummy_tensor(&[10], 1.0);
        let tasks = vec![
            (
                create_dummy_tensor(&[5, 10], 0.5),
                create_dummy_tensor(&[5], 0.3),
            ),
            (
                create_dummy_tensor(&[5, 10], 0.7),
                create_dummy_tensor(&[5], 0.2),
            ),
        ];

        let reptile = Reptile::new(model.clone(), 0.1, 2);
        let updated = reptile.reptile_update(&tasks);

        // Updated model should exist with same shape
        assert_eq!(updated.shape.dims, model.shape.dims);
    }

    #[test]
    fn test_task_batch_sampling() {
        let tasks = vec![
            (
                create_dummy_tensor(&[5, 10], 0.1),
                create_dummy_tensor(&[5], 0.2),
            ),
            (
                create_dummy_tensor(&[5, 10], 0.3),
                create_dummy_tensor(&[5], 0.4),
            ),
            (
                create_dummy_tensor(&[5, 10], 0.5),
                create_dummy_tensor(&[5], 0.6),
            ),
        ];

        let batcher = TaskBatch::new(tasks, 2);
        let sampled = batcher.sample_tasks(2);

        assert_eq!(sampled.len(), 2);
    }

    #[test]
    fn test_meta_loss_computation() {
        let outputs = vec![
            create_dummy_tensor(&[5], 0.5),
            create_dummy_tensor(&[5], 0.7),
        ];
        let targets = vec![
            create_dummy_tensor(&[5], 0.4),
            create_dummy_tensor(&[5], 0.6),
        ];

        let loss = MetaLoss::meta_loss(&outputs, &targets);

        // Loss should be non-negative
        let loss_val = loss.data[0];
        assert!(loss_val >= 0.0);
    }

    #[test]
    fn test_inner_loss_mse() {
        let output = create_dummy_tensor(&[5], 1.0);
        let target = create_dummy_tensor(&[5], 0.5);

        let loss = MetaLoss::inner_loss(&output, &target);

        // MSE of [0.5, 0.5, 0.5, 0.5, 0.5] = 0.25
        let loss_val = loss.data[0];
        assert!((loss_val - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_tensor_mul_scalar() {
        let tensor = create_dummy_tensor(&[5], 2.0);
        let scaled = tensor.mul_scalar(3.0);

        // Each element should be 6.0
        for val in &scaled.data {
            assert!((val - 6.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_average_tensors() {
        let t1 = create_dummy_tensor(&[3], 1.0);
        let t2 = create_dummy_tensor(&[3], 3.0);
        let t3 = create_dummy_tensor(&[3], 5.0);

        let avg = average_tensors(&[t1, t2, t3]);

        // Average should be 3.0
        for val in &avg.data {
            assert!((val - 3.0).abs() < 0.01);
        }
    }
}
