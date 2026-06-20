//! Training Utilities for nCPU/nSynth
//!
//! Checkpointing, early stopping, gradient clipping, etc.

use super::ops::{Shape, Tensor};
use std::collections::HashMap;
use std::fs;
use std::io;

// ============================================================================
// CHECKPOINTING
// ============================================================================

/// Checkpoint manager for saving/loading model state
#[derive(Debug)]
pub struct CheckpointManager {
    pub save_dir: String,
    pub max_keep: usize,
}

impl CheckpointManager {
    pub fn new(save_dir: &str, max_keep: usize) -> Self {
        Self {
            save_dir: save_dir.to_string(),
            max_keep,
        }
    }

    /// Save model parameters to disk
    pub fn save(
        &self,
        epoch: usize,
        params: &[Tensor],
        optimizer_state: &HashMap<String, Tensor>,
    ) -> io::Result<()> {
        let filename = format!("{}/checkpoint_epoch_{}.json", self.save_dir, epoch);

        // Create directory if it doesn't exist
        fs::create_dir_all(&self.save_dir)?;

        // Serialize parameters (simplified JSON)
        let mut data = HashMap::new();
        data.insert("epoch".to_string(), epoch.to_string());

        for (i, param) in params.iter().enumerate() {
            data.insert(format!("param_{}", i), format!("{:?}", param.data));
        }

        let json = serde_json::to_string_pretty(&data)?;
        fs::write(&filename, json)?;

        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(())
    }

    /// Load model parameters from disk
    pub fn load(&self, epoch: usize) -> io::Result<Vec<Vec<f64>>> {
        let filename = format!("{}/checkpoint_epoch_{}.json", self.save_dir, epoch);
        let json = fs::read_to_string(filename)?;
        let data: HashMap<String, String> = serde_json::from_str(&json)?;

        let mut params = Vec::new();
        let mut i = 0;

        while let Some(param_str) = data.get(&format!("param_{}", i)) {
            // Parse parameter data (simplified)
            let data_vec: Vec<f64> = param_str
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            params.push(data_vec);
            i += 1;
        }

        Ok(params)
    }

    /// Get latest checkpoint epoch
    pub fn get_latest_epoch(&self) -> Option<usize> {
        let dir = fs::read_dir(&self.save_dir).ok()?;
        let mut latest = None;

        for entry in dir.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("checkpoint_epoch_") && name.ends_with(".json") {
                let epoch_str = name
                    .strip_prefix("checkpoint_epoch_")?
                    .strip_suffix(".json")?;
                let epoch = epoch_str.parse::<usize>().ok()?;
                latest = Some(latest.as_ref().map_or(epoch, |l: &usize| *l.max(&epoch)));
            }
        }

        latest
    }

    fn cleanup_old_checkpoints(&self) -> io::Result<()> {
        let dir = fs::read_dir(&self.save_dir)?;
        let mut checkpoints: Vec<(usize, String)> = Vec::new();

        for entry in dir.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("checkpoint_epoch_") && name.ends_with(".json") {
                let epoch_str = name
                    .strip_prefix("checkpoint_epoch_")
                    .unwrap()
                    .strip_suffix(".json")
                    .unwrap();
                if let Ok(epoch) = epoch_str.parse::<usize>() {
                    checkpoints.push((epoch, entry.path().to_string_lossy().to_string()));
                }
            }
        }

        checkpoints.sort_by_key(|(e, _)| *e);

        while checkpoints.len() > self.max_keep {
            if let Some((_, path)) = checkpoints.first() {
                fs::remove_file(path)?;
                checkpoints.remove(0);
            }
        }

        Ok(())
    }
}

// ============================================================================
// EARLY STOPPING
// ============================================================================

/// Early stopping callback
#[derive(Debug)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    pub mode: EarlyStopMode,
    pub counter: usize,
    pub best_score: Option<f64>,
    pub stopped: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum EarlyStopMode {
    Min, // Lower is better (loss)
    Max, // Higher is better (accuracy)
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64, mode: EarlyStopMode) -> Self {
        Self {
            patience,
            min_delta,
            mode,
            counter: 0,
            best_score: None,
            stopped: false,
        }
    }

    pub fn update(&mut self, score: f64) -> bool {
        let improved = if let Some(best) = self.best_score {
            match self.mode {
                EarlyStopMode::Min => score < best - self.min_delta,
                EarlyStopMode::Max => score > best + self.min_delta,
            }
        } else {
            true
        };

        if improved {
            self.best_score = Some(score);
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.stopped = true;
                true
            } else {
                false
            }
        }
    }

    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_score = None;
        self.stopped = false;
    }
}

// ============================================================================
// GRADIENT CLIPPING
// ============================================================================

/// Clip gradients by value
pub fn clip_grad_value(grads: &mut [Tensor], clip_value: f64) {
    for grad in grads.iter_mut() {
        for v in grad.data.iter_mut() {
            *v = v.clamp(-clip_value, clip_value);
        }
    }
}

/// Clip gradients by norm
pub fn clip_grad_norm(grads: &mut [Tensor], max_norm: f64) -> f64 {
    let mut total_norm = 0.0;

    for grad in grads.iter() {
        let grad_norm: f64 = grad.data.iter().map(|&x| x * x).sum::<f64>().sqrt();
        total_norm += grad_norm * grad_norm;
    }

    total_norm = total_norm.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for grad in grads.iter_mut() {
            for v in grad.data.iter_mut() {
                *v *= scale;
            }
        }
    }

    total_norm
}

/// Clip gradients by norm per layer
pub fn clip_grad_norm_per_layer(grads: &mut [Tensor], max_norm: f64) {
    for grad in grads.iter_mut() {
        let grad_norm: f64 = grad.data.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if grad_norm > max_norm {
            let scale = max_norm / grad_norm;
            for v in grad.data.iter_mut() {
                *v *= scale;
            }
        }
    }
}

// ============================================================================
// LEARNING RATE UTILITIES
// ============================================================================

/// Learning rate finder (LR range test)
#[derive(Debug)]
pub struct LRFinder {
    pub start_lr: f64,
    pub end_lr: f64,
    pub num_iter: usize,
    pub current_iter: usize,
    pub best_loss: Option<f64>,
    pub lr_history: Vec<(f64, f64)>,
}

impl LRFinder {
    pub fn new(start_lr: f64, end_lr: f64, num_iter: usize) -> Self {
        Self {
            start_lr,
            end_lr,
            num_iter,
            current_iter: 0,
            best_loss: None,
            lr_history: Vec::new(),
        }
    }

    pub fn get_lr(&self) -> f64 {
        if self.num_iter <= 1 {
            return self.start_lr;
        }

        let gamma = (self.end_lr / self.start_lr).powf(1.0 / (self.num_iter - 1) as f64);
        self.start_lr * gamma.powf(self.current_iter as f64)
    }

    pub fn step(&mut self, loss: f64) {
        let lr = self.get_lr();
        self.lr_history.push((lr, loss));

        if self.best_loss.is_none() || Some(loss) < self.best_loss {
            self.best_loss = Some(loss);
        }

        self.current_iter += 1;
    }

    pub fn suggest_lr(&self) -> f64 {
        // Find the steepest descent point
        if self.lr_history.len() < 2 {
            return self.start_lr;
        }

        let mut max_gradient = 0.0;
        let mut suggested_lr = self.start_lr;

        for i in 1..self.lr_history.len() {
            let (lr1, loss1) = self.lr_history[i - 1];
            let (lr2, loss2) = self.lr_history[i];
            let gradient = ((loss2 - loss1) / (lr2 - lr1)).abs();

            if gradient > max_gradient {
                max_gradient = gradient;
                suggested_lr = lr2;
            }
        }

        suggested_lr
    }
}

// ============================================================================
// WARMUP
// ============================================================================

/// Learning rate warmup scheduler
#[derive(Debug, Clone)]
pub struct WarmupScheduler {
    pub warmup_epochs: usize,
    pub base_lr: f64,
    pub warmup_lr: f64,
    pub current_epoch: usize,
}

impl WarmupScheduler {
    pub fn new(warmup_epochs: usize, base_lr: f64, warmup_lr: f64) -> Self {
        Self {
            warmup_epochs,
            base_lr,
            warmup_lr,
            current_epoch: 0,
        }
    }

    pub fn get_lr(&self) -> f64 {
        if self.current_epoch < self.warmup_epochs {
            // Linear warmup
            let progress = self.current_epoch as f64 / self.warmup_epochs as f64;
            self.warmup_lr + (self.base_lr - self.warmup_lr) * progress
        } else {
            self.base_lr
        }
    }

    pub fn step(&mut self) {
        self.current_epoch += 1;
    }
}

// ============================================================================
// WEIGHT DECAY
// ============================================================================

/// Apply L2 weight decay
pub fn apply_weight_decay(params: &mut [Tensor], decay_rate: f64) {
    for param in params.iter_mut() {
        for v in param.data.iter_mut() {
            *v *= 1.0 - decay_rate;
        }
    }
}

/// Apply L1 weight decay (lasso)
pub fn apply_l1_decay(params: &mut [Tensor], decay_rate: f64) {
    for param in params.iter_mut() {
        for v in param.data.iter_mut() {
            *v -= decay_rate * v.signum();
        }
    }
}

/// Apply elastic net decay (L1 + L2)
pub fn apply_elastic_net_decay(params: &mut [Tensor], l1_rate: f64, l2_rate: f64) {
    for param in params.iter_mut() {
        for v in param.data.iter_mut() {
            *v = *v * (1.0 - l2_rate) - l1_rate * v.signum();
        }
    }
}

// ============================================================================
// GRADIENT ACCUMULATION
// ============================================================================

/// Gradient accumulation buffer
#[derive(Debug)]
pub struct GradientAccumulator {
    pub accumulated_gradients: Vec<Vec<f64>>,
    pub accumulation_steps: usize,
    pub current_step: usize,
}

impl GradientAccumulator {
    pub fn new(num_params: usize, accumulation_steps: usize) -> Self {
        Self {
            accumulated_gradients: vec![Vec::new(); num_params],
            accumulation_steps,
            current_step: 0,
        }
    }

    pub fn accumulate(&mut self, gradients: &[Tensor]) -> bool {
        if self.accumulated_gradients.is_empty() {
            self.accumulated_gradients = gradients.iter().map(|g| g.data.clone()).collect();
        } else {
            for (acc, grad) in self.accumulated_gradients.iter_mut().zip(gradients.iter()) {
                for (a, &g) in acc.iter_mut().zip(grad.data.iter()) {
                    *a += g;
                }
            }
        }

        self.current_step += 1;
        let should_step = self.current_step >= self.accumulation_steps;

        if should_step {
            self.reset();
        }

        should_step
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
        for acc in &mut self.accumulated_gradients {
            acc.fill(0.0);
        }
    }

    pub fn get_accumulated(&self) -> Vec<Tensor> {
        self.accumulated_gradients
            .iter()
            .map(|data| {
                Tensor::new(
                    data.iter()
                        .map(|&v| v / self.accumulation_steps as f64)
                        .collect(),
                    Shape::new(vec![data.len()]),
                )
            })
            .collect()
    }
}

// ============================================================================
// MIXED PRECISION SIMULATION
// ============================================================================

/// Simulate FP16 training by scaling gradients
#[derive(Debug, Clone)]
pub struct MixedPrecisionScaler {
    pub scale: f64,
    pub growth_factor: f64,
    pub backoff_factor: f64,
    pub growth_interval: usize,
    pub growth_step: usize,
    pub num_skipped: usize,
}

impl MixedPrecisionScaler {
    pub fn new(scale: f64) -> Self {
        Self {
            scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_step: 0,
            num_skipped: 0,
        }
    }

    pub fn scale_grads(&self, grads: &mut [Tensor]) {
        for grad in grads.iter_mut() {
            for v in grad.data.iter_mut() {
                *v *= self.scale;
            }
        }
    }

    pub fn unscale_grads(&self, grads: &mut [Tensor]) {
        for grad in grads.iter_mut() {
            for v in grad.data.iter_mut() {
                *v /= self.scale;
            }
        }
    }

    pub fn update(&mut self, has_inf_or_nan: bool) {
        if has_inf_or_nan {
            self.scale *= self.backoff_factor;
            self.num_skipped += 1;
            self.growth_step = 0;
        } else {
            self.growth_step += 1;
            if self.growth_step >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_step = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping() {
        let mut es = EarlyStopping::new(3, 0.01, EarlyStopMode::Min);

        assert!(!es.update(1.0)); // First - becomes best
        assert!(!es.update(0.95)); // Better
        assert!(!es.update(0.94)); // Better (within delta)
        assert!(!es.update(0.93)); // Better
        assert!(!es.update(0.93)); // Same - not better
        assert!(!es.update(0.93)); // Still not better
        assert!(es.update(0.93)); // Patience exceeded - should stop
    }

    #[test]
    fn test_clip_grad_value() {
        let mut grads = vec![Tensor::vector(vec![10.0, -10.0, 5.0])];
        clip_grad_value(&mut grads, 5.0);
        assert_eq!(grads[0].data, vec![5.0, -5.0, 5.0]);
    }

    #[test]
    fn test_warmup_scheduler() {
        let mut sched = WarmupScheduler::new(5, 0.01, 0.001);
        assert_eq!(sched.get_lr(), 0.001); // Start with warmup LR

        sched.step();
        sched.step();
        let lr = sched.get_lr();
        assert!(lr > 0.001 && lr < 0.01); // In warmup

        for _ in 0..10 {
            sched.step();
        }
        assert_eq!(sched.get_lr(), 0.01); // After warmup, base LR
    }
}
