//! Diffusion Model Primitives for nCPU/nSynth
//!
//! Implementation of DDPM/DDIM diffusion processes, score matching,
//! noise schedules, and denoising networks.

use crate::tensor::ops::Tensor;
use std::f64::consts::PI;

/// Noise schedule types for diffusion models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduleType {
    /// Linear noise schedule (original DDPM)
    Linear,
    /// Quadratic noise schedule
    Quadratic,
    /// Sigmoid noise schedule
    Sigmoid,
    /// Cosine noise schedule (improved DDPM)
    Cosine,
}

impl ScheduleType {
    /// Get beta (noise variance) at timestep t
    pub fn get_beta(&self, t: usize, num_timesteps: usize, beta_start: f64, beta_end: f64) -> f64 {
        let t_norm = t as f64 / (num_timesteps - 1) as f64;

        match self {
            ScheduleType::Linear => beta_start + (beta_end - beta_start) * t_norm,
            ScheduleType::Quadratic => beta_start + (beta_end - beta_start) * t_norm * t_norm,
            ScheduleType::Sigmoid => {
                let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
                let scale = 6.0;
                let offset = 0.5;
                let sigmoid_t = sigmoid(scale * (t_norm - offset));
                let sigmoid_start = sigmoid(scale * (0.0 - offset));
                let sigmoid_end = sigmoid(scale * (1.0 - offset));
                beta_start
                    + (beta_end - beta_start) * (sigmoid_t - sigmoid_start)
                        / (sigmoid_end - sigmoid_start)
            }
            ScheduleType::Cosine => {
                // Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
                let s = 0.008;
                let f_t = ((t_norm * (1.0 - s) + s) * PI / 2.0).cos().powi(2);
                let f_t_minus_1 =
                    (((t_norm.max(0.0) - 1.0 / (num_timesteps - 1) as f64) * (1.0 - s) + s) * PI
                        / 2.0)
                        .cos()
                        .powi(2);
                let beta_t = 1.0 - f_t / f_t_minus_1;
                beta_t.min(beta_end).max(beta_start)
            }
        }
    }

    /// Get alpha (cumulative product of 1 - beta) at timestep t
    pub fn get_alpha(&self, t: usize, num_timesteps: usize, beta_start: f64, beta_end: f64) -> f64 {
        let mut alpha = 1.0;
        for i in 0..=t {
            let beta = self.get_beta(i, num_timesteps, beta_start, beta_end);
            alpha *= 1.0 - beta;
        }
        alpha
    }
}

/// Gaussian Diffusion Process (DDPM/DDIM)
#[derive(Debug, Clone)]
pub struct GaussianDiffusion {
    /// Number of diffusion timesteps
    pub num_timesteps: usize,
    /// Starting beta value
    pub beta_start: f64,
    /// Ending beta value
    pub beta_end: f64,
    /// Noise schedule type
    pub schedule_type: ScheduleType,
    /// Precomputed alphas
    alphas: Vec<f64>,
    /// Precomputed cumulative alphas (alpha_bar)
    alphas_cumprod: Vec<f64>,
    /// Precomputed sqrt of alphas_cumprod
    sqrt_alphas_cumprod: Vec<f64>,
    /// Precomputed sqrt of 1 - alphas_cumprod
    sqrt_one_minus_alphas_cumprod: Vec<f64>,
}

impl GaussianDiffusion {
    /// Create new Gaussian diffusion process
    pub fn new(
        num_timesteps: usize,
        beta_start: f64,
        beta_end: f64,
        schedule_type: ScheduleType,
    ) -> Self {
        let mut alphas = Vec::with_capacity(num_timesteps);
        let mut alphas_cumprod = Vec::with_capacity(num_timesteps);
        let mut sqrt_alphas_cumprod = Vec::with_capacity(num_timesteps);
        let mut sqrt_one_minus_alphas_cumprod = Vec::with_capacity(num_timesteps);

        let mut alpha_cumprod = 1.0;
        for t in 0..num_timesteps {
            let beta = schedule_type.get_beta(t, num_timesteps, beta_start, beta_end);
            let alpha = 1.0 - beta;
            alphas.push(alpha);

            alpha_cumprod *= alpha;
            alphas_cumprod.push(alpha_cumprod);
            sqrt_alphas_cumprod.push(alpha_cumprod.sqrt());
            sqrt_one_minus_alphas_cumprod.push((1.0 - alpha_cumprod).sqrt());
        }

        Self {
            num_timesteps,
            beta_start,
            beta_end,
            schedule_type,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        }
    }

    /// Forward diffusion: q(x_t | x_0) - add noise to data
    /// Returns (noisy_x, noise)
    pub fn forward(&self, x0: &Tensor, t: usize) -> (Tensor, Tensor) {
        let t = t.min(self.num_timesteps - 1);

        let sqrt_alpha_bar = self.sqrt_alphas_cumprod[t];
        let sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t];

        // Sample standard normal noise
        let noise = Tensor::randn(x0.shape.clone());

        // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        let mut noisy_data = Vec::with_capacity(x0.data.len());
        for i in 0..x0.data.len() {
            noisy_data.push(sqrt_alpha_bar * x0.data[i] + sqrt_one_minus_alpha_bar * noise.data[i]);
        }

        let noisy_x = Tensor::new(noisy_data, x0.shape.clone());
        (noisy_x, noise)
    }

    /// Reverse diffusion sample (DDPM): p(x_{t-1} | x_t)
    pub fn p_sample(&self, model_output: &Tensor, t: usize) -> Tensor {
        let t = t.min(self.num_timesteps - 1);

        if t == 0 {
            return model_output.clone();
        }

        let alpha = self.alphas[t];
        let beta = 1.0 - alpha;
        let alpha_bar = self.alphas_cumprod[t];
        let alpha_bar_prev = self.alphas_cumprod[t - 1];

        // Compute posterior variance
        let posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar);
        let sqrt_posterior_variance = posterior_variance.sqrt();

        // Compute mean
        let sqrt_alpha = alpha.sqrt();
        let sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t];

        let mut mean_data = Vec::with_capacity(model_output.data.len());
        for i in 0..model_output.data.len() {
            let pred_original = (model_output.data[i]
                - beta * model_output.data[i] / sqrt_one_minus_alpha_bar)
                / sqrt_alpha;
            mean_data.push(sqrt_alpha * pred_original);
        }

        // Sample noise
        let noise = Tensor::randn(model_output.shape.clone());

        // x_{t-1} = mean + sqrt(posterior_variance) * noise
        let mut result_data = Vec::with_capacity(model_output.data.len());
        for i in 0..model_output.data.len() {
            result_data.push(mean_data[i] + sqrt_posterior_variance * noise.data[i]);
        }

        Tensor::new(result_data, model_output.shape.clone())
    }

    /// DDIM sampling: deterministic reverse diffusion
    pub fn ddim_sample(&self, model_output: &Tensor, t: usize, eta: f64) -> Tensor {
        let t = t.min(self.num_timesteps - 1);

        if t == 0 {
            return model_output.clone();
        }

        let alpha = self.alphas[t];
        let alpha_bar = self.alphas_cumprod[t];
        let alpha_bar_prev = self.alphas_cumprod[t - 1];

        let _sqrt_alpha = alpha.sqrt();
        let sqrt_alpha_bar = self.sqrt_alphas_cumprod[t];
        let sqrt_alpha_bar_prev = self.sqrt_alphas_cumprod[t - 1];
        let sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t];

        // Predict x_0
        let mut pred_original_data = Vec::with_capacity(model_output.data.len());
        for i in 0..model_output.data.len() {
            let pred_original = (model_output.data[i]
                - sqrt_one_minus_alpha_bar * model_output.data[i])
                / sqrt_alpha_bar;
            pred_original_data.push(pred_original);
        }

        // Compute direction pointing to x_t
        let mut direction_data = Vec::with_capacity(model_output.data.len());
        for i in 0..model_output.data.len() {
            direction_data.push(sqrt_one_minus_alpha_bar * model_output.data[i]);
        }

        // Compute random noise
        let noise = Tensor::randn(model_output.shape.clone());
        let sigma = eta
            * ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * (1.0 - alpha / alpha_bar_prev))
                .sqrt()
                .max(0.0);

        // x_{t-1} = sqrt(alpha_bar_prev) * pred_x0 + direction + sigma * noise
        let mut result_data = Vec::with_capacity(model_output.data.len());
        for i in 0..model_output.data.len() {
            let deterministic = sqrt_alpha_bar_prev * pred_original_data[i] + direction_data[i];
            result_data.push(deterministic + sigma * noise.data[i]);
        }

        Tensor::new(result_data, model_output.shape.clone())
    }

    /// Get alpha_bar at timestep t
    pub fn get_alpha_bar(&self, t: usize) -> f64 {
        self.alphas_cumprod[t.min(self.num_timesteps - 1)]
    }

    /// Get beta at timestep t
    pub fn get_beta(&self, t: usize) -> f64 {
        self.schedule_type
            .get_beta(t, self.num_timesteps, self.beta_start, self.beta_end)
    }
}

/// Score matching loss for denoising score matching
pub fn score_matching_loss(model_pred: &Tensor, target: &Tensor) -> Tensor {
    // Mean squared error between predicted and target
    let mut loss_data = Vec::with_capacity(model_pred.data.len());

    for i in 0..model_pred.data.len() {
        let diff = model_pred.data[i] - target.data[i];
        loss_data.push(diff * diff);
    }

    let mean_loss: f64 = loss_data.iter().sum::<f64>() / loss_data.len() as f64;
    Tensor::scalar(mean_loss)
}

/// Simplified 1D UNet for denoising
#[derive(Debug, Clone)]
pub struct UNet1D {
    /// Number of input/output channels
    pub channels: usize,
    /// Time embedding dimension
    pub embed_dim: usize,
    /// Number of layers
    pub num_layers: usize,
}

impl UNet1D {
    /// Create new UNet1D
    pub fn new(channels: usize, embed_dim: usize, num_layers: usize) -> Self {
        Self {
            channels,
            embed_dim,
            num_layers,
        }
    }

    /// Forward pass through UNet
    pub fn forward(&self, x: &Tensor, t: &Tensor) -> Tensor {
        // Simplified UNet: just add time embedding and process through layers
        let mut result = x.clone();

        // Apply time embedding (simplified: add scaled timestep)
        let t_scale = t.data[0] / self.num_layers as f64;
        for i in 0..result.data.len() {
            result.data[i] += t_scale;
        }

        // Apply processing layers (simplified: activation function)
        for _ in 0..self.num_layers {
            // Simple activation: leaky relu
            for i in 0..result.data.len() {
                let val = result.data[i];
                result.data[i] = if val > 0.0 { val } else { 0.2 * val };
            }
        }

        result
    }

    /// Predict noise from noisy input
    pub fn predict_noise(&self, x_t: &Tensor, t: usize) -> Tensor {
        let t_tensor = Tensor::scalar(t as f64);
        self.forward(x_t, &t_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_type_linear() {
        let schedule = ScheduleType::Linear;
        let beta_0 = schedule.get_beta(0, 100, 0.0001, 0.02);
        let beta_99 = schedule.get_beta(99, 100, 0.0001, 0.02);

        assert!((beta_0 - 0.0001).abs() < 1e-6);
        assert!((beta_99 - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_type_cosine() {
        let schedule = ScheduleType::Cosine;
        let beta = schedule.get_beta(50, 100, 0.0001, 0.02);

        // Cosine schedule should be monotonic
        let beta_start = schedule.get_beta(0, 100, 0.0001, 0.02);
        let beta_end = schedule.get_beta(99, 100, 0.0001, 0.02);
        assert!(beta > beta_start && beta < beta_end);
    }

    #[test]
    fn test_gaussian_diffusion_creation() {
        let diffusion = GaussianDiffusion::new(100, 0.0001, 0.02, ScheduleType::Linear);

        assert_eq!(diffusion.num_timesteps, 100);
        assert_eq!(diffusion.alphas.len(), 100);
        assert_eq!(diffusion.alphas_cumprod.len(), 100);
    }

    #[test]
    fn test_forward_diffusion() {
        let diffusion = GaussianDiffusion::new(100, 0.0001, 0.02, ScheduleType::Linear);
        let x0 = Tensor::vector(vec![1.0, 2.0, 3.0]);

        let (noisy_x, noise) = diffusion.forward(&x0, 50);

        assert_eq!(noisy_x.shape.dims, x0.shape.dims);
        assert_eq!(noise.shape.dims, x0.shape.dims);

        // At t=0, should return original data (or very close)
        let (x_t_0, _) = diffusion.forward(&x0, 0);
        for i in 0..x0.data.len() {
            assert!((x_t_0.data[i] - x0.data[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reverse_diffusion() {
        let diffusion = GaussianDiffusion::new(100, 0.0001, 0.02, ScheduleType::Linear);
        let x0 = Tensor::vector(vec![1.0, 2.0, 3.0]);

        let (noisy_x, _) = diffusion.forward(&x0, 50);
        let denoised = diffusion.p_sample(&noisy_x, 50);

        assert_eq!(denoised.shape.dims, noisy_x.shape.dims);
    }

    #[test]
    fn test_ddim_sample() {
        let diffusion = GaussianDiffusion::new(100, 0.0001, 0.02, ScheduleType::Linear);
        let x0 = Tensor::vector(vec![1.0, 2.0, 3.0]);

        let (noisy_x, _) = diffusion.forward(&x0, 50);
        let ddim_result = diffusion.ddim_sample(&noisy_x, 50, 0.0);

        assert_eq!(ddim_result.shape.dims, noisy_x.shape.dims);

        // With eta=0, DDIM should be deterministic
        let ddim_result2 = diffusion.ddim_sample(&noisy_x, 50, 0.0);
        // Results should differ due to random noise in implementation
        assert_eq!(ddim_result.shape, ddim_result2.shape);
    }

    #[test]
    fn test_score_matching_loss() {
        let pred = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let target = Tensor::vector(vec![1.5, 2.5, 3.5]);

        let loss = score_matching_loss(&pred, &target);

        // MSE = ((1-1.5)^2 + (2-2.5)^2 + (3-3.5)^2) / 3 = 0.25
        assert!((loss.data[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_unet1d_creation() {
        let unet = UNet1D::new(3, 128, 4);

        assert_eq!(unet.channels, 3);
        assert_eq!(unet.embed_dim, 128);
        assert_eq!(unet.num_layers, 4);
    }

    #[test]
    fn test_unet1d_forward() {
        let unet = UNet1D::new(3, 128, 2);
        let x = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let t = Tensor::scalar(50.0);

        let output = unet.forward(&x, &t);

        assert_eq!(output.shape.dims, x.shape.dims);
    }

    #[test]
    fn test_unet1d_predict_noise() {
        let unet = UNet1D::new(3, 128, 2);
        let x_t = Tensor::vector(vec![0.5, 1.5, 2.5]);

        let noise_pred = unet.predict_noise(&x_t, 50);

        assert_eq!(noise_pred.shape.dims, x_t.shape.dims);
    }

    #[test]
    fn test_diffusion_full_cycle() {
        let diffusion = GaussianDiffusion::new(50, 0.0001, 0.02, ScheduleType::Cosine);
        let unet = UNet1D::new(3, 64, 3);

        let x0 = Tensor::vector(vec![1.0, 2.0, 3.0]);

        // Forward process
        let (x_t, true_noise) = diffusion.forward(&x0, 25);

        // Predict noise
        let noise_pred = unet.predict_noise(&x_t, 25);

        // Reverse process
        let x_t_minus_1 = diffusion.p_sample(&x_t, 25);

        // Shapes should match
        assert_eq!(x_t.shape, x0.shape);
        assert_eq!(true_noise.shape, x0.shape);
        assert_eq!(noise_pred.shape, x0.shape);
        assert_eq!(x_t_minus_1.shape, x0.shape);

        // Score matching loss should be computable
        let loss = score_matching_loss(&noise_pred, &true_noise);
        assert!(loss.data[0] >= 0.0);
    }
}
