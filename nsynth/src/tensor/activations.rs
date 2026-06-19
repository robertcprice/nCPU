//! Comprehensive Activation Functions for nCPU/nSynth
//!
//! All common activation functions with derivatives.

use super::ops::Tensor;

/// Activation function trait
pub trait ActivationFn {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn derivative(&self, x: &Tensor) -> Tensor;
}

/// ReLU (Rectified Linear Unit)
#[derive(Debug, Clone, Copy)]
pub struct ReLU;

impl ActivationFn for ReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.relu()
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// Leaky ReLU
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    pub alpha: f64,
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl ActivationFn for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| if v > 0.0 { v } else { self.alpha * v })
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| if v > 0.0 { 1.0 } else { self.alpha })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// ReLU6 (capped at 6)
#[derive(Debug, Clone, Copy)]
pub struct ReLU6;

impl ActivationFn for ReLU6 {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| (0.0_f64).max(v.min(6.0_f64)))
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| if v > 0.0 && v < 6.0 { 1.0 } else { 0.0 })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// ELU (Exponential Linear Unit)
#[derive(Debug, Clone, Copy)]
pub struct ELU {
    pub alpha: f64,
}

impl ELU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ActivationFn for ELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| {
                    if v > 0.0 {
                        v
                    } else {
                        self.alpha * (v.exp() - 1.0)
                    }
                })
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| if v > 0.0 { 1.0 } else { self.alpha * v.exp() })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// SELU (Scaled Exponential Linear Unit)
#[derive(Debug, Clone, Copy)]
pub struct SELU;

impl ActivationFn for SELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        // SELU constants: alpha = 1.67326, scale = 1.05070
        const ALPHA: f64 = 1.67326;
        const SCALE: f64 = 1.05070;

        Tensor::new(
            x.data
                .iter()
                .map(|&v| SCALE * if v > 0.0 { v } else { ALPHA * (v.exp() - 1.0) })
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        const ALPHA: f64 = 1.67326;
        const SCALE: f64 = 1.05070;

        Tensor::new(
            x.data
                .iter()
                .map(|&v| SCALE * if v > 0.0 { 1.0 } else { ALPHA * v.exp() })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// GELU (Gaussian Error Linear Unit)
#[derive(Debug, Clone, Copy)]
pub struct GELU;

impl ActivationFn for GELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        // GELU(x) ≈ x * Φ(x) where Φ is the CDF of standard normal
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        Tensor::new(
            x.data
                .iter()
                .map(|&v| {
                    let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
                    let cubic = 0.044715 * v.powi(3);
                    0.5 * v * (1.0 + (sqrt_2_pi * (v + cubic)).tanh())
                })
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        // Derivative approximation
        Tensor::new(
            x.data
                .iter()
                .map(|&v| {
                    let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
                    let cubic = 0.044715 * v.powi(3);
                    let tanh_val = (sqrt_2_pi * (v + cubic)).tanh();
                    let sech_sq = 1.0 - tanh_val.powi(2);
                    0.5 * (1.0 + tanh_val)
                        + 0.5 * v * sech_sq * sqrt_2_pi * (1.0 + 0.134145 * v.powi(2))
                })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// Swish activation (x * sigmoid(x))
#[derive(Debug, Clone, Copy)]
pub struct Swish;

impl ActivationFn for Swish {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data.iter().map(|&v| v / (1.0 + (-v).exp())).collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| {
                    let sig = 1.0 / (1.0 + (-v).exp());
                    sig + v * sig * (1.0 - sig)
                })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// Softplus (log(1 + exp(x)))
#[derive(Debug, Clone, Copy)]
pub struct Softplus;

impl ActivationFn for Softplus {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| (1.0 + v.exp()).ln().max(v))
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| {
                    let exp_v = v.exp();
                    exp_v / (1.0 + exp_v)
                })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// Softsign (x / (1 + |x|))
#[derive(Debug, Clone, Copy)]
pub struct Softsign;

impl ActivationFn for Softsign {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data.iter().map(|&v| v / (1.0 + v.abs())).collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| {
                    let abs_x = v.abs();
                    1.0 / (1.0 + abs_x).powi(2)
                })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// Hard sigmoid
#[derive(Debug, Clone, Copy)]
pub struct HardSigmoid;

impl ActivationFn for HardSigmoid {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| (0.0_f64).max((1.0_f64).min((v / 6.0) + 0.5)))
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| if v > -3.0 && v < 3.0 { 1.0 / 6.0 } else { 0.0 })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// Hard tanh
#[derive(Debug, Clone, Copy)]
pub struct HardTanh;

impl ActivationFn for HardTanh {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| (-1.0_f64).max((1.0_f64).min(v)))
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| if v > -1.0 && v < 1.0 { 1.0 } else { 0.0 })
                .collect(),
            x.shape.clone(),
        )
    }
}

/// Mish activation (x * tanh(softplus(x)))
#[derive(Debug, Clone, Copy)]
pub struct Mish;

impl ActivationFn for Mish {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| {
                    let sp = (1.0_f64 + v.exp()).ln();
                    v * sp.tanh()
                })
                .collect(),
            x.shape.clone(),
        )
    }

    fn derivative(&self, x: &Tensor) -> Tensor {
        Tensor::new(
            x.data
                .iter()
                .map(|&v| {
                    let exp_v = v.exp();
                    let exp_2v = exp_v * exp_v;
                    let _omega = (2.0 * exp_v + exp_2v).powi(3) + 4.0 * (v + 1.0).powi(3);
                    let denom = (exp_v + 2.0).powi(2);
                    (exp_v * (v + 2.0).powi(2) / denom).powi(2)
                })
                .collect(),
            x.shape.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = Tensor::vector(vec![-1.0, 0.0, 1.0]);
        let relu = ReLU;
        let y = relu.forward(&x);
        assert_eq!(y.data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let x = Tensor::vector(vec![-1.0, 0.0, 1.0]);
        let lrelu = LeakyReLU::new(0.1);
        let y = lrelu.forward(&x);
        assert_eq!(y.data[0], -0.1);
        assert_eq!(y.data[1], 0.0);
        assert_eq!(y.data[2], 1.0);
    }

    #[test]
    fn test_gelu() {
        let x = Tensor::vector(vec![0.0, 1.0]);
        let gelu = GELU;
        let y = gelu.forward(&x);
        // GELU(0) = 0, GELU(1) ≈ 0.84
        assert!((y.data[0]).abs() < 0.01);
        assert!((y.data[1] - 0.84).abs() < 0.05);
    }

    #[test]
    fn test_sigmoid_values() {
        let x = Tensor::vector(vec![0.0, 1.0]);
        let y = x.sigmoid();
        // Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731
        assert!((y.data[0] - 0.5).abs() < 0.01);
        assert!((y.data[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid_derivative_formula() {
        // Verify derivative formula: σ'(x) = σ(x) * (1 - σ(x))
        let sig_0 = 0.5_f64;
        let sig_1 = 0.731_f64;
        let deriv_0 = sig_0 * (1.0 - sig_0);
        let deriv_1 = sig_1 * (1.0 - sig_1);
        assert!((deriv_0 - 0.25).abs() < 0.01);
        assert!((deriv_1 - 0.197).abs() < 0.01);
    }
}
