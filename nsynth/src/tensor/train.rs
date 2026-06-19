//! Training Primitives for nCPU/nSynth
//!
//! Optimizers, loss functions, training loops.

use super::model::Model;
use super::ops::Tensor;
use std::collections::HashMap;

/// Optimizer for parameter updates
#[derive(Debug, Clone)]
pub enum Optimizer {
    SGD {
        learning_rate: f64,
        momentum: Option<f64>,
    },
    Adam {
        beta1: f64,
        beta2: f64,
        learning_rate: f64,
    },
    RMSprop {
        learning_rate: f64,
        decay: f64,
    },
}

impl Optimizer {
    /// Create SGD optimizer
    pub fn sgd(learning_rate: f64) -> Self {
        Optimizer::SGD {
            learning_rate,
            momentum: None,
        }
    }

    /// Create SGD with momentum
    pub fn sgd_momentum(learning_rate: f64, momentum: f64) -> Self {
        Optimizer::SGD {
            learning_rate,
            momentum: Some(momentum),
        }
    }

    /// Create Adam optimizer
    pub fn adam(learning_rate: f64) -> Self {
        Optimizer::Adam {
            beta1: 0.9,
            beta2: 0.999,
            learning_rate,
        }
    }

    /// Update parameters with gradients
    pub fn step(
        &self,
        params: &mut [Tensor],
        grads: &[Option<Tensor>],
        state: &mut OptimizerState,
    ) {
        for (i, param) in params.iter_mut().enumerate() {
            if let Some(ref grad) = grads[i] {
                match self {
                    Optimizer::SGD {
                        learning_rate,
                        momentum,
                    } => {
                        if let Some(mom) = momentum {
                            // Update momentum
                            let m = state.get_momentum(i, param.data.len());
                            for j in 0..param.data.len() {
                                m[j] = *mom * m[j] + (1.0 - *mom) * grad.data[j];
                                param.data[j] -= *learning_rate * m[j];
                            }
                        } else {
                            // Standard SGD
                            for j in 0..param.data.len() {
                                param.data[j] -= *learning_rate * grad.data[j];
                            }
                        }
                    }
                    Optimizer::Adam {
                        beta1,
                        beta2,
                        learning_rate,
                    } => {
                        let (t, m, v) = state.get_adam_state(i, param.data.len());
                        *t += 1.0;

                        let lr =
                            *learning_rate * (1.0 - beta2.powf(*t)).sqrt() / (1.0 - beta1.powf(*t));

                        for j in 0..param.data.len() {
                            m[j] = *beta1 * m[j] + (1.0 - *beta1) * grad.data[j];
                            v[j] = *beta2 * v[j] + (1.0 - *beta2) * grad.data[j].powf(2.0);

                            let m_hat = m[j] / (1.0 - beta1.powf(*t));
                            let v_hat = v[j] / (1.0 - beta2.powf(*t));

                            param.data[j] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                        }
                    }
                    Optimizer::RMSprop {
                        learning_rate,
                        decay,
                    } => {
                        let v = state.get_rmsprop_state(i, param.data.len());

                        for j in 0..param.data.len() {
                            v[j] = *decay * v[j] + (1.0 - *decay) * grad.data[j].powf(2.0);
                            param.data[j] -= *learning_rate * grad.data[j] / (v[j].sqrt() + 1e-8);
                        }
                    }
                }
            }
        }
    }
}

/// Optimizer state (for momentum, Adam, etc.)
#[derive(Debug)]
pub struct OptimizerState {
    /// Momentum buffers (indexed by parameter index)
    momentums: HashMap<usize, Vec<f64>>,
    /// Adam state: (t, m, v)
    adam_state: HashMap<usize, (f64, Vec<f64>, Vec<f64>)>,
    /// RMSprop state
    rmsprop_state: HashMap<usize, Vec<f64>>,
}

impl OptimizerState {
    pub fn new() -> Self {
        Self {
            momentums: HashMap::new(),
            adam_state: HashMap::new(),
            rmsprop_state: HashMap::new(),
        }
    }

    fn get_momentum(&mut self, index: usize, size: usize) -> &mut Vec<f64> {
        self.momentums
            .entry(index)
            .or_insert_with(|| vec![0.0; size])
    }

    fn get_adam_state(
        &mut self,
        index: usize,
        size: usize,
    ) -> (&mut f64, &mut Vec<f64>, &mut Vec<f64>) {
        let state = self
            .adam_state
            .entry(index)
            .or_insert_with(|| (0.0, vec![0.0; size], vec![0.0; size]));
        (&mut state.0, &mut state.1, &mut state.2)
    }

    fn get_rmsprop_state(&mut self, index: usize, size: usize) -> &mut Vec<f64> {
        self.rmsprop_state
            .entry(index)
            .or_insert_with(|| vec![0.0; size])
    }
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Loss function
#[derive(Debug, Clone, Copy)]
pub enum LossFn {
    MSE,          // Mean Squared Error
    MAE,          // Mean Absolute Error
    CrossEntropy, // Cross Entropy
    Hinge,        // Hinge Loss
    BinaryCrossEntropy,
}

impl LossFn {
    /// Compute loss
    pub fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor, String> {
        match self {
            LossFn::MSE => Ok(self.mse(predictions, targets)),
            LossFn::MAE => Ok(self.mae(predictions, targets)),
            LossFn::CrossEntropy => Ok(self.cross_entropy(predictions, targets)?),
            LossFn::Hinge => Ok(self.hinge(predictions, targets)),
            LossFn::BinaryCrossEntropy => Ok(self.binary_cross_entropy(predictions, targets)),
        }
    }

    /// Compute gradient of loss w.r.t predictions
    pub fn grad(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor, String> {
        match self {
            LossFn::MSE => Ok(self.mse_grad(predictions, targets)),
            LossFn::MAE => Ok(self.mae_grad(predictions, targets)),
            LossFn::CrossEntropy => Ok(self.cross_entropy_grad(predictions, targets)?),
            LossFn::Hinge => Ok(self.hinge_grad(predictions, targets)),
            LossFn::BinaryCrossEntropy => Ok(self.binary_cross_entropy_grad(predictions, targets)),
        }
    }

    fn mse(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let diff = predictions.sub(targets).unwrap();
        let squared = diff.mul(&diff).unwrap();
        Tensor::scalar(squared.sum() / predictions.data.len() as f64)
    }

    fn mse_grad(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let n = predictions.data.len() as f64;
        predictions
            .sub(targets)
            .unwrap()
            .mul(&Tensor::scalar(2.0 / n))
            .unwrap()
    }

    fn mae(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let diff = predictions.sub(targets).unwrap();
        let abs_diff: Vec<f64> = diff.data.iter().map(|&x| x.abs()).collect();
        Tensor::scalar(abs_diff.iter().sum::<f64>() / predictions.data.len() as f64)
    }

    fn mae_grad(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let n = predictions.data.len() as f64;
        let diff = predictions.sub(targets).unwrap();
        let signs: Vec<f64> = diff
            .data
            .iter()
            .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        Tensor::vector(signs).mul(&Tensor::scalar(1.0 / n)).unwrap()
    }

    fn cross_entropy(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor, String> {
        // Predictions should be probabilities (after softmax)
        // Targets should be one-hot encoded
        let mut loss = 0.0;

        for i in 0..predictions.data.len() {
            let p = predictions.data[i].max(1e-10); // Avoid log(0)
            let t = targets.data[i];
            loss -= t * p.ln();
        }

        Ok(Tensor::scalar(loss / predictions.data.len() as f64))
    }

    fn cross_entropy_grad(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor, String> {
        // d/dx CE = predictions - targets
        predictions.sub(targets)
    }

    fn hinge(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Hinge loss: max(0, 1 - t * y)
        let mut loss = 0.0;

        for i in 0..predictions.data.len() {
            let val = 1.0 - targets.data[i] * predictions.data[i];
            loss += val.max(0.0);
        }

        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn hinge_grad(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];

        for i in 0..predictions.data.len() {
            let val = 1.0 - targets.data[i] * predictions.data[i];
            if val > 0.0 {
                grad[i] = -targets.data[i];
            }
        }

        Tensor::vector(grad)
    }

    fn binary_cross_entropy(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;

        for i in 0..predictions.data.len() {
            let p = predictions.data[i].clamp(1e-10, 1.0 - 1e-10);
            let t = targets.data[i];
            loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
        }

        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn binary_cross_entropy_grad(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];

        for i in 0..predictions.data.len() {
            let p = predictions.data[i].clamp(1e-10, 1.0 - 1e-10);
            let t = targets.data[i];
            grad[i] = -(t / p - (1.0 - t) / (1.0 - p));
        }

        Tensor::vector(grad)
    }
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub shuffle: bool,
    pub verbose: bool,
}

impl TrainingConfig {
    pub fn new() -> Self {
        Self {
            batch_size: 32,
            epochs: 100,
            learning_rate: 0.001,
            shuffle: true,
            verbose: true,
        }
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Training loop
#[derive(Debug)]
pub struct Trainer {
    pub optimizer: Optimizer,
    pub loss_fn: LossFn,
    pub config: TrainingConfig,
    pub state: OptimizerState,
}

impl Trainer {
    /// Create new trainer
    pub fn new(optimizer: Optimizer, loss_fn: LossFn, config: TrainingConfig) -> Self {
        Self {
            optimizer,
            loss_fn,
            config,
            state: OptimizerState::new(),
        }
    }

    /// Train a model
    pub fn train<M: Model>(
        &mut self,
        model: &mut M,
        train_data: &[(Tensor, Tensor)],
    ) -> Result<Vec<f64>, String> {
        let mut history = Vec::new();

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut batches = 0;

            // Process in batches
            for batch in train_data.chunks(self.config.batch_size) {
                // Forward pass
                let mut batch_loss = 0.0;
                let mut batch_params: Vec<Tensor> = Vec::new();
                let mut batch_grads: Vec<Option<Tensor>> = Vec::new();

                // Collect params and grads for batch
                for param in model.parameters() {
                    batch_params.push(param);
                }
                for grad in model.gradients() {
                    batch_grads.push(grad);
                }

                for (inputs, targets) in batch {
                    let predictions = model.forward(inputs);
                    let loss = self.loss_fn.compute(&predictions, targets)?;
                    batch_loss += loss.data[0];
                }

                epoch_loss += batch_loss / batch.len() as f64;
                batches += 1;

                // TODO: Implement actual gradient computation and backprop
                // This would integrate with the autodiff module
            }

            let avg_loss = epoch_loss / batches as f64;
            history.push(avg_loss);

            if self.config.verbose {
                println!("Epoch {}: loss = {:.6}", epoch + 1, avg_loss);
            }
        }

        Ok(history)
    }

    /// Evaluate model
    pub fn evaluate<M: Model>(
        &self,
        model: &M,
        test_data: &[(Tensor, Tensor)],
    ) -> Result<f64, String> {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (inputs, targets) in test_data {
            let predictions = model.forward(inputs);
            let loss = self.loss_fn.compute(&predictions, targets)?;
            total_loss += loss.data[0];
            count += 1;
        }

        Ok(total_loss / count as f64)
    }

    /// Train for one epoch
    pub fn train_epoch<M: Model>(
        &mut self,
        model: &mut M,
        train_data: &[(Tensor, Tensor)],
    ) -> Result<f64, String> {
        let mut epoch_loss = 0.0;
        let mut batches = 0;

        for batch in train_data.chunks(self.config.batch_size) {
            let mut batch_loss = 0.0;

            for (inputs, targets) in batch {
                let predictions = model.forward(inputs);
                let loss = self.loss_fn.compute(&predictions, targets)?;
                batch_loss += loss.data[0];
            }

            epoch_loss += batch_loss / batch.len() as f64;
            batches += 1;
        }

        Ok(epoch_loss / batches as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_sgd() {
        let opt = Optimizer::sgd(0.01);
        let mut params = vec![Tensor::vector(vec![1.0, 2.0])];
        let grads = vec![Some(Tensor::vector(vec![0.1, 0.2]))];
        let mut state = OptimizerState::new();

        opt.step(&mut params, &grads, &mut state);

        // param = param - lr * grad
        assert!((params[0].data[0] - 0.999).abs() < 0.01); // 1 - 0.01 * 0.1
        assert!((params[0].data[1] - 1.998).abs() < 0.01); // 2 - 0.01 * 0.2
    }

    #[test]
    fn test_loss_mse() {
        let predictions = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let targets = Tensor::vector(vec![1.5, 2.5, 3.5]);

        let loss_fn = LossFn::MSE;
        let loss = loss_fn.compute(&predictions, &targets).unwrap();

        // MSE = ((-0.5)^2 + (-0.5)^2 + (-0.5)^2) / 3 = 0.75 / 3 = 0.25
        assert!((loss.data[0] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_loss_grad_mse() {
        let predictions = Tensor::vector(vec![1.0, 2.0]);
        let targets = Tensor::vector(vec![1.5, 2.5]);

        let loss_fn = LossFn::MSE;
        let grad = loss_fn.grad(&predictions, &targets).unwrap();

        // grad = 2 * (pred - target) / n = 2 * [-0.5, -0.5] / 2 = [-0.5, -0.5]
        assert!((grad.data[0] - (-0.5)).abs() < 0.001);
        assert!((grad.data[1] - (-0.5)).abs() < 0.001);
    }

    #[test]
    fn test_cross_entropy() {
        let predictions = Tensor::vector(vec![0.7, 0.3]);
        let targets = Tensor::vector(vec![1.0, 0.0]); // one-hot

        let loss_fn = LossFn::CrossEntropy;
        let loss = loss_fn.compute(&predictions, &targets).unwrap();

        // CE = -(1 * log(0.7) + 0 * log(0.3)) / 2 = -log(0.7) / 2 ≈ 0.178
        assert!((loss.data[0] - 0.178).abs() < 0.01);
    }

    #[test]
    fn test_trainer_config() {
        let config = TrainingConfig::new()
            .with_batch_size(64)
            .with_epochs(50)
            .with_learning_rate(0.0001);

        assert_eq!(config.batch_size, 64);
        assert_eq!(config.epochs, 50);
        assert_eq!(config.learning_rate, 0.0001);
    }

    #[test]
    fn test_trainer_create() {
        let optimizer = Optimizer::adam(0.001);
        let loss_fn = LossFn::MSE;
        let config = TrainingConfig::new();

        let trainer = Trainer::new(optimizer.clone(), loss_fn, config);

        match trainer.optimizer {
            Optimizer::Adam { .. } => {}
            _ => panic!("Expected Adam optimizer"),
        }
    }
}
