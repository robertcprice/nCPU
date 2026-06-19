//! Advanced Optimizers for nCPU/nSynth
//!
//! State-of-the-art optimization algorithms.

use super::ops::Tensor;
use std::collections::HashMap;

/// Optimizer trait
pub trait Optimizer {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]);
    fn zero_grad(&mut self);
}

/// AdamW - Adam with decoupled weight decay
#[derive(Debug)]
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<Vec<f64>>,
    pub v: Vec<Vec<f64>>,
}

impl AdamW {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;
        let lr = self.lr;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Ensure state vectors exist
            if self.m.len() <= i {
                self.m.push(vec![0.0; param.data.len()]);
                self.v.push(vec![0.0; param.data.len()]);
            }

            // Apply weight decay (decoupled from gradient)
            for p in param.data.iter_mut() {
                *p -= self.lr * self.weight_decay * *p;
            }

            // Update biased moments
            for (j, (&g, m)) in grad.data.iter().zip(self.m[i].iter_mut()).enumerate() {
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            }
            for (j, (&g, v)) in grad.data.iter().zip(self.v[i].iter_mut()).enumerate() {
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            }

            // Update parameters
            for (j, p) in param.data.iter_mut().enumerate() {
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;
                *p -= lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    fn zero_grad(&mut self) {
        // Gradients are zeroed externally
    }
}

/// Adagrad - adaptive gradients
#[derive(Debug)]
pub struct Adagrad {
    pub lr: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub state: Vec<Vec<f64>>,
}

impl Adagrad {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            eps: 1e-10,
            weight_decay: 0.0,
            state: Vec::new(),
        }
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.state.len() <= i {
                self.state.push(vec![0.0; param.data.len()]);
            }

            // Accumulate squared gradients
            for (j, &g) in grad.data.iter().enumerate() {
                self.state[i][j] += g * g;
            }

            // Update with adaptive learning rate
            for (j, p) in param.data.iter_mut().enumerate() {
                let g = grad.data[j];
                let denom = self.state[i][j].sqrt() + self.eps;
                *p -= self.lr * g / denom;
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// Adadelta - adaptive learning rates
#[derive(Debug)]
pub struct Adadelta {
    pub rho: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub acc_grad: Vec<Vec<f64>>,
    pub acc_update: Vec<Vec<f64>>,
}

impl Adadelta {
    pub fn new(rho: f64) -> Self {
        Self {
            rho,
            eps: 1e-8,
            weight_decay: 0.0,
            acc_grad: Vec::new(),
            acc_update: Vec::new(),
        }
    }
}

impl Optimizer for Adadelta {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.acc_grad.len() <= i {
                self.acc_grad.push(vec![0.0; param.data.len()]);
                self.acc_update.push(vec![0.0; param.data.len()]);
            }

            for j in 0..param.data.len() {
                let g = grad.data[j];
                // Accumulate gradient
                self.acc_grad[i][j] = self.rho * self.acc_grad[i][j] + (1.0 - self.rho) * g * g;

                // Compute update
                let old_update = self.acc_update[i][j];
                let update =
                    (old_update.sqrt() + self.eps) / (self.acc_grad[i][j].sqrt() + self.eps) * g;

                // Accumulate update
                self.acc_update[i][j] = self.rho * old_update + (1.0 - self.rho) * update * update;

                // Apply update
                param.data[j] -= update;
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// RMSprop centered
#[derive(Debug)]
pub struct RMSpropCentered {
    pub lr: f64,
    pub alpha: f64,
    pub eps: f64,
    pub centered: bool,
    pub momentum: f64,
    pub s: Vec<Vec<f64>>,
    pub g: Vec<Vec<f64>>,
    pub buf: Vec<Vec<f64>>,
}

impl RMSpropCentered {
    pub fn new(lr: f64, centered: bool) -> Self {
        Self {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            centered,
            momentum: 0.0,
            s: Vec::new(),
            g: Vec::new(),
            buf: Vec::new(),
        }
    }
}

impl Optimizer for RMSpropCentered {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.s.len() <= i {
                self.s.push(vec![0.0; param.data.len()]);
                self.g.push(vec![0.0; param.data.len()]);
                if self.momentum > 0.0 {
                    self.buf.push(vec![0.0; param.data.len()]);
                }
            }

            for j in 0..param.data.len() {
                let g = grad.data[j];

                // Update moving average of gradient
                self.g[i][j] = self.alpha * self.g[i][j] + (1.0 - self.alpha) * g;

                // Update squared gradient average
                let g_sq = g * g;
                let g_mean_sq = if self.centered {
                    let g_diff = g - self.g[i][j];
                    self.alpha * self.s[i][j] + (1.0 - self.alpha) * g_diff * g_diff
                } else {
                    self.alpha * self.s[i][j] + (1.0 - self.alpha) * g_sq
                };
                self.s[i][j] = g_mean_sq;

                // Compute update
                let denom = if self.centered {
                    (self.s[i][j] - self.g[i][j] * self.g[i][j]).abs()
                } else {
                    self.s[i][j]
                };

                if self.momentum > 0.0 {
                    self.buf[i][j] =
                        self.momentum * self.buf[i][j] - (self.lr / (denom.sqrt() + self.eps)) * g;
                    param.data[j] += self.buf[i][j];
                } else {
                    param.data[j] -= (self.lr / (denom.sqrt() + self.eps)) * g;
                }
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// SGD with Nesterov momentum
#[derive(Debug)]
pub struct NesterovSGD {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub buf: Vec<Vec<f64>>,
}

impl NesterovSGD {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: true,
            buf: Vec::new(),
        }
    }
}

impl Optimizer for NesterovSGD {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.buf.len() <= i {
                self.buf.push(vec![0.0; param.data.len()]);
            }

            for j in 0..param.data.len() {
                let g = grad.data[j];

                // Apply weight decay
                let g = g + self.weight_decay * param.data[j];

                if self.momentum > 0.0 {
                    let buf = self.buf[i][j];
                    let new_buf = self.momentum * buf + (1.0 - self.dampening) * g;
                    self.buf[i][j] = new_buf;

                    if self.nesterov {
                        // Nesterov update
                        param.data[j] -= self.lr * (g + self.momentum * new_buf);
                    } else {
                        param.data[j] -= self.lr * new_buf;
                    }
                } else {
                    param.data[j] -= self.lr * g;
                }
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// LAMB - Layer-wise Adaptive Moments (for large batch training)
#[derive(Debug)]
pub struct LAMB {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<Vec<f64>>,
    pub v: Vec<Vec<f64>>,
}

impl LAMB {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    fn layer_adaptation(&self, param: &Tensor, grad: &Tensor, m: &[f64], v: &[f64]) -> f64 {
        // Compute trust ratio
        let param_norm: f64 = param.data.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let grad_norm: f64 = grad.data.iter().map(|&x| x * x).sum::<f64>().sqrt();

        let update_norm: f64 = m
            .iter()
            .zip(v.iter())
            .map(|(&mi, &vi)| (mi / (vi.sqrt() + self.eps)).powi(2))
            .sum::<f64>()
            .sqrt();

        if grad_norm > 0.0 && update_norm > 0.0 {
            (param_norm / update_norm).min(10.0)
        } else {
            1.0
        }
    }
}

impl Optimizer for LAMB {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.m.len() <= i {
                self.m.push(vec![0.0; param.data.len()]);
                self.v.push(vec![0.0; param.data.len()]);
            }

            // Update moments
            for (j, &g) in grad.data.iter().enumerate() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
            }

            // Apply weight decay
            for p in param.data.iter_mut() {
                *p -= self.lr * self.weight_decay * *p;
            }

            // Compute adaptive step
            let trust_ratio = self.layer_adaptation(param, grad, &self.m[i], &self.v[i]);

            for (j, p) in param.data.iter_mut().enumerate() {
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;
                *p -= self.lr * trust_ratio * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// LARS - Layer-wise Adaptive Rate Scaling
#[derive(Debug)]
pub struct LARS {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub trust_coefficient: f64,
    pub eps: f64,
    pub buf: Vec<Vec<f64>>,
}

impl LARS {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            weight_decay: 1e-4,
            trust_coefficient: 0.001,
            eps: 1e-8,
            buf: Vec::new(),
        }
    }

    fn compute_lr(&self, param: &Tensor, grad: &Tensor) -> f64 {
        let param_norm: f64 = param.data.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let grad_norm: f64 = grad.data.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if param_norm > 0.0 && grad_norm > 0.0 {
            let local_lr = self.trust_coefficient * param_norm / grad_norm;
            local_lr.min(self.lr)
        } else {
            self.lr
        }
    }
}

impl Optimizer for LARS {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.buf.len() <= i {
                self.buf.push(vec![0.0; param.data.len()]);
            }

            // Apply weight decay
            for j in 0..param.data.len() {
                param.data[j] -= self.weight_decay * param.data[j];
            }

            let local_lr = self.compute_lr(param, grad);

            for j in 0..param.data.len() {
                let g = grad.data[j];
                self.buf[i][j] = self.momentum * self.buf[i][j] - local_lr * g;
                param.data[j] += self.buf[i][j];
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// RAdam - Rectified Adam (variance rectification)
#[derive(Debug)]
pub struct RAdam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<Vec<f64>>,
    pub v: Vec<Vec<f64>>,
}

impl RAdam {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    fn compute_rectification(&self) -> (f64, f64) {
        let beta2_t = self.beta2.powi(self.t as i32);
        let n_sma_max = 5.0 / (2.0 - self.beta2);
        let n_sma = n_sma_max - 2.0 * self.t as f64 * beta2_t / (1.0 - beta2_t);
        (n_sma, n_sma_max)
    }
}

impl Optimizer for RAdam {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let (n_sma, n_sma_max) = self.compute_rectification();

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.m.len() <= i {
                self.m.push(vec![0.0; param.data.len()]);
                self.v.push(vec![0.0; param.data.len()]);
            }

            // Update moments
            for (j, &g) in grad.data.iter().enumerate() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
            }

            for (j, p) in param.data.iter_mut().enumerate() {
                if n_sma >= 5.0 {
                    // Adam update
                    let m_hat = self.m[i][j] / bias_correction1;
                    let v_hat = self.v[i][j];
                    let r = (n_sma / (n_sma + 4.0) * (n_sma + 2.0) / n_sma * n_sma_max
                        / (n_sma_max + 4.0)
                        * (n_sma_max + 2.0)
                        / n_sma_max)
                        .sqrt();
                    *p -= self.lr * r * m_hat / (v_hat.sqrt() + self.eps);
                } else {
                    // SGD update
                    *p -= self.lr * self.m[i][j];
                }

                // Weight decay
                *p -= self.lr * self.weight_decay * *p;
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// NovoGrad
#[derive(Debug)]
pub struct NovoGrad {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub grad_ema: f64,
    pub v: Vec<Vec<f64>>,
}

impl NovoGrad {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.95,
            beta2: 0.98,
            eps: 1e-8,
            weight_decay,
            grad_ema: 0.0,
            v: Vec::new(),
        }
    }
}

impl Optimizer for NovoGrad {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Compute gradient norm
            let grad_norm: f64 = grad.data.iter().map(|&x| x * x).sum::<f64>().sqrt();
            self.grad_ema = self.beta1 * self.grad_ema + (1.0 - self.beta1) * grad_norm;

            if self.v.len() <= i {
                self.v.push(vec![0.0; param.data.len()]);
            }

            for j in 0..param.data.len() {
                let g = grad.data[j];
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;

                let adapted_lr = self.lr / (self.grad_ema + self.eps);
                let denom = self.v[i][j].sqrt() + self.eps;
                param.data[j] -= adapted_lr * g / denom;
                param.data[j] -= self.lr * self.weight_decay * param.data[j];
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// Lookahead wrapper (improves any optimizer)
#[derive(Debug)]
pub struct Lookahead<O: Optimizer> {
    pub base_optimizer: O,
    pub k: usize,
    pub alpha: f64,
    pub slow_params: Vec<Vec<f64>>,
    pub step_count: usize,
}

impl<O: Optimizer> Lookahead<O> {
    pub fn new(base_optimizer: O, k: usize, alpha: f64) -> Self {
        Self {
            base_optimizer,
            k,
            alpha,
            slow_params: Vec::new(),
            step_count: 0,
        }
    }
}

impl<O: Optimizer> Optimizer for Lookahead<O> {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.step_count += 1;

        // Initialize slow params on first step
        if self.slow_params.is_empty() {
            for param in params.iter() {
                self.slow_params.push(param.data.clone());
            }
        }

        // Step with base optimizer
        self.base_optimizer.step(params, grads);

        // Sync slow params every k steps
        if self.step_count % self.k == 0 {
            for (i, param) in params.iter_mut().enumerate() {
                let param_len = param.data.len();
                for j in 0..param_len {
                    let p = param.data[j];
                    let slow_p = self.slow_params[i][j];
                    self.slow_params[i][j] = slow_p + self.alpha * (p - slow_p);
                    param.data[j] = self.slow_params[i][j];
                }
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// AdaBelief
#[derive(Debug)]
pub struct AdaBelief {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<Vec<f64>>,
    pub s: Vec<Vec<f64>>,
}

impl AdaBelief {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m: Vec::new(),
            s: Vec::new(),
        }
    }
}

impl Optimizer for AdaBelief {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.m.len() <= i {
                self.m.push(vec![0.0; param.data.len()]);
                self.s.push(vec![0.0; param.data.len()]);
            }

            for j in 0..param.data.len() {
                let g = grad.data[j];

                // Update first moment
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;

                // Update second moment (belief about gradient)
                let m_diff = g - self.m[i][j];
                self.s[i][j] = self.beta2 * self.s[i][j] + (1.0 - self.beta2) * m_diff * m_diff;

                // Update parameters
                let m_hat = self.m[i][j] / bias_correction1;
                let s_hat = self.s[i][j] / bias_correction2;
                param.data[j] -= self.lr * m_hat / (s_hat.sqrt() + self.eps);

                // Weight decay
                param.data[j] -= self.lr * self.weight_decay * param.data[j];
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// Shampoo - second-order optimizer
#[derive(Debug)]
pub struct Shampoo {
    pub lr: f64,
    pub weight_decay: f64,
    pub eps: f64,
    pub update_freq: usize,
    pub t: usize,
    pub grad_g: Vec<Vec<f64>>,
}

impl Shampoo {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            weight_decay: 0.0,
            eps: 1e-8,
            update_freq: 1,
            t: 0,
            grad_g: Vec::new(),
        }
    }
}

impl Optimizer for Shampoo {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.grad_g.len() <= i {
                let n = param.data.len();
                self.grad_g.push(vec![0.0; n * n]);
            }

            // Accumulate gradient outer product (simplified)
            if self.t % self.update_freq == 0 {
                let n = param.data.len();
                for row in 0..n {
                    for col in 0..n {
                        let idx = row * n + col;
                        self.grad_g[i][idx] += grad.data[row] * grad.data[col];
                    }
                }
            }

            // Simplified Shampoo update (full version requires matrix roots)
            let param_len = param.data.len();
            for (j, p) in param.data.iter_mut().enumerate() {
                let g = grad.data[j];
                let preconditioner =
                    (self.grad_g[i][j * (1 + param_len.min(1))].abs() + self.eps).sqrt();
                *p -= self.lr * g / preconditioner;
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// Nadam - Nesterov-accelerated Adam
#[derive(Debug)]
pub struct Nadam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<Vec<f64>>,
    pub v: Vec<Vec<f64>>,
    pub m_buffer: Vec<Vec<f64>>,
}

impl Nadam {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
            m_buffer: Vec::new(),
        }
    }
}

impl Optimizer for Nadam {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.m.len() <= i {
                self.m.push(vec![0.0; param.data.len()]);
                self.v.push(vec![0.0; param.data.len()]);
                self.m_buffer.push(vec![0.0; param.data.len()]);
            }

            // Update moments
            for (j, &g) in grad.data.iter().enumerate() {
                self.m_buffer[i][j] = self.m[i][j];
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
            }

            for (j, p) in param.data.iter_mut().enumerate() {
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;
                let g = grad.data[j];

                // Nesterov lookahead
                let m_nesterov =
                    (1.0 + self.beta1) * self.m[i][j] - self.beta1 * self.m_buffer[i][j];
                let m_nesterov_hat = m_nesterov / bias_correction1;

                *p -= self.lr * (self.beta1 * m_nesterov_hat + (1.0 - self.beta1) * g)
                    / (v_hat.sqrt() + self.eps);

                // Weight decay
                *p -= self.lr * self.weight_decay * *p;
            }
        }
    }

    fn zero_grad(&mut self) {}
}

/// AdaMax - infinite norm variant of Adam
#[derive(Debug)]
pub struct AdaMax {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<Vec<f64>>,
    pub u: Vec<Vec<f64>>,
}

impl AdaMax {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m: Vec::new(),
            u: Vec::new(),
        }
    }
}

impl Optimizer for AdaMax {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.m.len() <= i {
                self.m.push(vec![0.0; param.data.len()]);
                self.u.push(vec![0.0; param.data.len()]);
            }

            for (j, &g) in grad.data.iter().enumerate() {
                // Update first moment
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;

                // Update infinity norm (exp. moving max of |grad|)
                self.u[i][j] = (self.beta2 * self.u[i][j]).max(g.abs());

                // Update parameter
                let m_hat = self.m[i][j] / bias_correction1;
                param.data[j] -= self.lr * m_hat / (self.u[i][j] + self.eps);

                // Weight decay
                param.data[j] -= self.lr * self.weight_decay * param.data[j];
            }
        }
    }

    fn zero_grad(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_step() {
        let mut opt = AdamW::new(0.01, 0.01);
        let mut params = vec![Tensor::vector(vec![1.0, 2.0, 3.0])];
        let grads = vec![Tensor::vector(vec![0.1, 0.1, 0.1])];
        opt.step(&mut params, &grads);
        // Parameters should change
        assert!(params[0].data[0] != 1.0);
    }

    #[test]
    fn test_adagrad_step() {
        let mut opt = Adagrad::new(0.01);
        let mut params = vec![Tensor::vector(vec![1.0, 2.0, 3.0])];
        let grads = vec![Tensor::vector(vec![0.1, 0.1, 0.1])];
        opt.step(&mut params, &grads);
        assert!(params[0].data[0] != 1.0);
    }

    #[test]
    fn test_nesterov_sgd() {
        let mut opt = NesterovSGD::new(0.01, 0.9);
        let mut params = vec![Tensor::vector(vec![1.0, 2.0, 3.0])];
        let grads = vec![Tensor::vector(vec![0.1, 0.1, 0.1])];
        opt.step(&mut params, &grads);
        assert!(params[0].data[0] != 1.0);
    }
}
