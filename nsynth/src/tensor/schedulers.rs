//! Learning Rate Schedulers for nCPU/nSynth
//!
//! Learning rate scheduling strategies.

/// Learning rate scheduler trait
pub trait Scheduler {
    fn get_lr(&self, epoch: usize) -> f64;
    fn step(&mut self);
    fn get_current(&self) -> f64;
}

/// Constant learning rate (no scheduling)
#[derive(Debug, Clone)]
pub struct ConstantLR {
    pub lr: f64,
}

impl ConstantLR {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Scheduler for ConstantLR {
    fn get_lr(&self, _epoch: usize) -> f64 {
        self.lr
    }

    fn step(&mut self) {
        // No-op for constant LR
    }

    fn get_current(&self) -> f64 {
        self.lr
    }
}

/// Step LR: decay by gamma every step_size epochs
#[derive(Debug, Clone)]
pub struct StepLR {
    pub base_lr: f64,
    pub step_size: usize,
    pub gamma: f64,
    pub epoch: usize,
}

impl StepLR {
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            base_lr,
            step_size,
            gamma,
            epoch: 0,
        }
    }
}

impl Scheduler for StepLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        let steps = (epoch as f64 / self.step_size as f64).floor() as i32;
        self.base_lr * self.gamma.powi(steps)
    }

    fn step(&mut self) {
        self.epoch += 1;
    }

    fn get_current(&self) -> f64 {
        self.get_lr(self.epoch)
    }
}

/// Exponential LR: decay by gamma every epoch
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    pub base_lr: f64,
    pub gamma: f64,
    pub epoch: usize,
}

impl ExponentialLR {
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self {
            base_lr,
            gamma,
            epoch: 0,
        }
    }
}

impl Scheduler for ExponentialLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        self.base_lr * self.gamma.powi(epoch as i32)
    }

    fn step(&mut self) {
        self.epoch += 1;
    }

    fn get_current(&self) -> f64 {
        self.get_lr(self.epoch)
    }
}

/// Cosine annealing
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    pub base_lr: f64,
    pub min_lr: f64,
    pub t_max: usize,
    pub epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f64, t_max: usize) -> Self {
        Self {
            base_lr,
            min_lr: 0.0,
            t_max,
            epoch: 0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl Scheduler for CosineAnnealingLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        let t = epoch as f64 % self.t_max as f64;
        let cosine = (std::f64::consts::PI * t / self.t_max as f64).cos();
        self.min_lr + (self.base_lr - self.min_lr) * (1.0 + cosine) / 2.0
    }

    fn step(&mut self) {
        self.epoch += 1;
    }

    fn get_current(&self) -> f64 {
        self.get_lr(self.epoch)
    }
}

/// ReduceLROnPlateau: reduce LR when metric stops improving
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    pub base_lr: f64,
    pub factor: f64,
    pub patience: usize,
    pub min_lr: f64,
    pub epoch: usize,
    pub best_metric: Option<f64>,
    pub wait: usize,
}

impl ReduceLROnPlateau {
    pub fn new(base_lr: f64, factor: f64, patience: usize) -> Self {
        Self {
            base_lr,
            factor,
            patience,
            min_lr: 1e-10,
            epoch: 0,
            best_metric: None,
            wait: 0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Update scheduler with new metric value
    pub fn step_metric(&mut self, metric: f64) -> bool {
        // Returns true if LR was reduced
        let improved = if let Some(best) = self.best_metric {
            metric < best
        } else {
            true
        };

        if improved {
            self.best_metric = Some(metric);
            self.wait = 0;
            false
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.base_lr = (self.base_lr * self.factor).max(self.min_lr);
                self.wait = 0;
                true
            } else {
                false
            }
        }
    }
}

impl Scheduler for ReduceLROnPlateau {
    fn get_lr(&self, _epoch: usize) -> f64 {
        self.base_lr
    }

    fn step(&mut self) {
        self.epoch += 1;
    }

    fn get_current(&self) -> f64 {
        self.base_lr
    }
}

/// Linear warmup with cosine decay
#[derive(Debug, Clone)]
pub struct WarmupCosineScheduler {
    pub warmup_epochs: usize,
    pub max_epochs: usize,
    pub base_lr: f64,
    pub min_lr: f64,
    pub epoch: usize,
}

impl WarmupCosineScheduler {
    pub fn new(warmup_epochs: usize, max_epochs: usize, base_lr: f64) -> Self {
        Self {
            warmup_epochs,
            max_epochs,
            base_lr,
            min_lr: 0.0,
            epoch: 0,
        }
    }
}

impl Scheduler for WarmupCosineScheduler {
    fn get_lr(&self, epoch: usize) -> f64 {
        if epoch < self.warmup_epochs {
            // Linear warmup
            self.base_lr * (epoch as f64 / self.warmup_epochs as f64)
        } else if epoch < self.max_epochs {
            // Cosine decay
            let progress =
                (epoch - self.warmup_epochs) as f64 / (self.max_epochs - self.warmup_epochs) as f64;
            let cosine = (std::f64::consts::PI * progress).cos();
            self.min_lr + (self.base_lr - self.min_lr) * (1.0 + cosine) / 2.0
        } else {
            self.min_lr
        }
    }

    fn step(&mut self) {
        self.epoch += 1;
    }

    fn get_current(&self) -> f64 {
        self.get_lr(self.epoch)
    }
}

/// Polynomial decay
#[derive(Debug, Clone)]
pub struct PolynomialLR {
    pub base_lr: f64,
    pub end_lr: f64,
    pub total_epochs: usize,
    pub power: f64,
    pub epoch: usize,
}

impl PolynomialLR {
    pub fn new(base_lr: f64, total_epochs: usize) -> Self {
        Self {
            base_lr,
            end_lr: 0.0,
            total_epochs,
            power: 1.0,
            epoch: 0,
        }
    }

    pub fn with_end_lr(mut self, end_lr: f64) -> Self {
        self.end_lr = end_lr;
        self
    }

    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }
}

impl Scheduler for PolynomialLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        let t = (epoch as f64 / self.total_epochs as f64).min(1.0);
        (self.base_lr - self.end_lr) * (1.0 - t).powf(self.power) + self.end_lr
    }

    fn step(&mut self) {
        self.epoch += 1;
    }

    fn get_current(&self) -> f64 {
        self.get_lr(self.epoch)
    }
}

/// Cyclic LR: triangular learning rate policy
#[derive(Debug, Clone)]
pub struct CyclicLR {
    pub base_lr: f64,
    pub max_lr: f64,
    pub step_size_up: usize,
    pub step_size_down: usize,
    pub epoch: usize,
    pub mode: CycleMode,
}

#[derive(Debug, Clone, Copy)]
pub enum CycleMode {
    Triangular,
    Triangular2,
    ExpRange,
}

impl CyclicLR {
    pub fn new(base_lr: f64, max_lr: f64, step_size_up: usize) -> Self {
        let step_size_down = step_size_up;
        Self {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            epoch: 0,
            mode: CycleMode::Triangular,
        }
    }

    pub fn with_step_size_down(mut self, step_size_down: usize) -> Self {
        self.step_size_down = step_size_down;
        self
    }

    pub fn with_mode(mut self, mode: CycleMode) -> Self {
        self.mode = mode;
        self
    }
}

impl Scheduler for CyclicLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        let cycle = epoch as f64 / (self.step_size_up + self.step_size_down) as f64;
        let x = epoch as f64 / (self.step_size_up as f64);

        let base_lr = match self.mode {
            CycleMode::Triangular2 => self.base_lr / (2.0_f64).powf(cycle.floor()),
            CycleMode::ExpRange => self.base_lr,
            CycleMode::Triangular => self.base_lr,
        };

        let max_lr = match self.mode {
            CycleMode::Triangular2 => self.max_lr / (2.0_f64).powf(cycle.floor()),
            CycleMode::ExpRange => {
                let gamma = (self.max_lr / self.base_lr).powf(1.0 / (self.step_size_up as f64));
                self.base_lr * gamma.powf(x)
            }
            CycleMode::Triangular => self.max_lr,
        };

        let cycle_progress = x - (self.step_size_up as f64) * cycle.floor();
        let lr = if cycle_progress <= self.step_size_up as f64 {
            base_lr + (max_lr - base_lr) * (cycle_progress / self.step_size_up as f64)
        } else {
            max_lr
                - (max_lr - base_lr)
                    * ((cycle_progress - self.step_size_up as f64) / self.step_size_up as f64)
        };

        lr.max(base_lr)
    }

    fn step(&mut self) {
        self.epoch += 1;
    }

    fn get_current(&self) -> f64 {
        self.get_lr(self.epoch)
    }
}

/// OneCycleLR: single cycle learning rate policy
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    pub max_lr: f64,
    pub total_epochs: usize,
    pub pct_start: f64,
    pub base_lr: f64,
    pub epoch: usize,
}

impl OneCycleLR {
    pub fn new(max_lr: f64, total_epochs: usize) -> Self {
        Self {
            max_lr,
            total_epochs,
            pct_start: 0.3,
            base_lr: max_lr / 10.0,
            epoch: 0,
        }
    }

    pub fn with_pct_start(mut self, pct_start: f64) -> Self {
        self.pct_start = pct_start;
        self
    }

    pub fn with_base_lr(mut self, base_lr: f64) -> Self {
        self.base_lr = base_lr;
        self
    }
}

impl Scheduler for OneCycleLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        let epoch_f = epoch as f64;
        let total_f = self.total_epochs as f64;
        let step_start = (total_f * self.pct_start) as usize;
        let step_end = self.total_epochs - step_start;

        if epoch <= step_start {
            // Increasing phase
            let pct = epoch_f / step_start as f64;
            self.base_lr + (self.max_lr - self.base_lr) * pct
        } else if epoch <= step_start + step_end {
            // Decreasing phase
            let pct = (epoch_f - step_start as f64) / step_end as f64;
            self.max_lr - (self.max_lr - self.base_lr) * pct
        } else {
            // Final decay to base_lr / 1e4
            let pct = (epoch_f - (step_start + step_end) as f64)
                / (self.total_epochs - step_start - step_end) as f64;
            let final_lr = self.base_lr / 1e4;
            self.base_lr - (self.base_lr - final_lr) * pct
        }
    }

    fn step(&mut self) {
        self.epoch += 1;
    }

    fn get_current(&self) -> f64 {
        self.get_lr(self.epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let sched = ConstantLR::new(0.01);
        assert_eq!(sched.get_lr(0), 0.01);
        assert_eq!(sched.get_lr(100), 0.01);
    }

    #[test]
    fn test_step_lr() {
        let sched = StepLR::new(0.1, 10, 0.5);
        assert_eq!(sched.get_lr(0), 0.1);
        assert_eq!(sched.get_lr(10), 0.05);
        assert_eq!(sched.get_lr(20), 0.025);
    }

    #[test]
    fn test_cosine_annealing() {
        let sched = CosineAnnealingLR::new(0.1, 100);
        let lr_0 = sched.get_lr(0);
        let lr_50 = sched.get_lr(50);

        assert!((lr_0 - 0.1).abs() < 0.01);
        assert!(lr_50 < lr_0 && lr_50 > 0.0);
        // At half cycle (50/100), cos(π/2) = 0, so lr = 0.5 * base_lr = 0.05
        assert!((lr_50 - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_exponential_lr() {
        let sched = ExponentialLR::new(0.1, 0.95);
        assert_eq!(sched.get_lr(0), 0.1);
        assert!((sched.get_lr(1) - 0.095).abs() < 0.001);
    }
}
