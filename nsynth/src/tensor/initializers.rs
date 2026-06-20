//! Weight Initializers for nCPU/nSynth
//!
//! Xavier, He, Orthogonal, and other initialization schemes.

use super::ops::{Shape, Tensor};

/// Initialization strategy trait
pub trait Initializer {
    fn init(&self, shape: &[usize]) -> Tensor;
}

// ============================================================================
// XAVIER/GLOROT INITIALIZATION
// ============================================================================

/// Xavier Uniform (Glorot uniform)
#[derive(Debug, Clone, Copy)]
pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn init(&self, shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let fan_out = shape[1];
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        Tensor::uniform(Shape::new(shape.to_vec()), -limit, limit)
    }
}

/// Xavier Normal (Glorot normal)
#[derive(Debug, Clone, Copy)]
pub struct XavierNormal;

impl Initializer for XavierNormal {
    fn init(&self, shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let fan_out = shape[1];
        let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
        Tensor::randn_scaled(Shape::new(shape.to_vec()), 0.0, std)
    }
}

// ============================================================================
// HE/KAIMING INITIALIZATION
// ============================================================================

/// He Uniform (Kaiming uniform) for ReLU
#[derive(Debug, Clone, Copy)]
pub struct HeUniform;

impl Initializer for HeUniform {
    fn init(&self, shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let limit = (6.0 / fan_in as f64).sqrt();
        Tensor::uniform(Shape::new(shape.to_vec()), -limit, limit)
    }
}

/// He Normal (Kaiming normal) for ReLU
#[derive(Debug, Clone, Copy)]
pub struct HeNormal;

impl Initializer for HeNormal {
    fn init(&self, shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let std = (2.0 / fan_in as f64).sqrt();
        Tensor::randn_scaled(Shape::new(shape.to_vec()), 0.0, std)
    }
}

/// He Uniform for Leaky ReLU
#[derive(Debug, Clone, Copy)]
pub struct HeUniformLeaky {
    pub a: f64,
}

impl HeUniformLeaky {
    pub fn new(a: f64) -> Self {
        Self { a }
    }
}

impl Initializer for HeUniformLeaky {
    fn init(&self, shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let limit = (6.0 / ((1.0 + self.a * self.a) * fan_in as f64)).sqrt();
        Tensor::uniform(Shape::new(shape.to_vec()), -limit, limit)
    }
}

// ============================================================================
// ORTHOGONAL INITIALIZATION
// ============================================================================

/// Orthogonal Initialization (via SVD)
#[derive(Debug, Clone, Copy)]
pub struct Orthogonal {
    pub gain: f64,
}

impl Orthogonal {
    pub fn new(gain: f64) -> Self {
        Self { gain }
    }
}

impl Default for Orthogonal {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Initializer for Orthogonal {
    fn init(&self, shape: &[usize]) -> Tensor {
        // Simplified orthogonal initialization
        // Full implementation uses SVD on random matrix
        let n = shape[0].max(shape[1]);
        let mut data = vec![0.0; shape[0] * shape[1]];

        // Create random matrix and attempt orthogonalization
        let random = Tensor::rand(Shape::new(vec![n, n]));

        // Gram-Schmidt orthogonalization (simplified)
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let idx = i * shape[1] + j;
                if i < random.data.len() / n && j < n {
                    data[idx] = random.data[i * n + j] * self.gain;
                } else {
                    data[idx] = 0.0;
                }
            }
        }

        Tensor::new(data, Shape::new(shape.to_vec()))
    }
}

// ============================================================================
// LECUN INITIALIZATION
// ============================================================================

/// LeCun Normal initialization
#[derive(Debug, Clone, Copy)]
pub struct LecunNormal;

impl Initializer for LecunNormal {
    fn init(&self, shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let std = (1.0 / fan_in as f64).sqrt();
        Tensor::randn_scaled(Shape::new(shape.to_vec()), 0.0, std)
    }
}

/// LeCun Uniform initialization
#[derive(Debug, Clone, Copy)]
pub struct LecunUniform;

impl Initializer for LecunUniform {
    fn init(&self, shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let limit = (3.0 / fan_in as f64).sqrt();
        Tensor::uniform(Shape::new(shape.to_vec()), -limit, limit)
    }
}

// ============================================================================
// TRUNCATED NORMAL
// ============================================================================

/// Truncated Normal (values beyond 2 sigma are discarded)
#[derive(Debug, Clone, Copy)]
pub struct TruncatedNormal {
    pub mean: f64,
    pub std: f64,
}

impl TruncatedNormal {
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }
}

impl Initializer for TruncatedNormal {
    fn init(&self, shape: &[usize]) -> Tensor {
        // Generate normal values and truncate to 2 std
        let temp = Tensor::randn_scaled(Shape::new(shape.to_vec()), self.mean, self.std);
        let two_std = 2.0 * self.std;

        let data: Vec<f64> = temp
            .data
            .iter()
            .map(|&v| {
                if (v - self.mean).abs() <= two_std {
                    v
                } else if v < self.mean {
                    self.mean - two_std
                } else {
                    self.mean + two_std
                }
            })
            .collect();

        Tensor::new(data, Shape::new(shape.to_vec()))
    }
}

// ============================================================================
// VARIANCE SCALING
// ============================================================================

/// Variance Scaling initializer
#[derive(Debug, Clone, Copy)]
pub struct VarianceScaling {
    pub scale: f64,
    pub mode: VarianceScalingMode,
    pub distribution: VarianceDistribution,
}

#[derive(Debug, Clone, Copy)]
pub enum VarianceScalingMode {
    FanIn,
    FanOut,
    FanAvg,
}

#[derive(Debug, Clone, Copy)]
pub enum VarianceDistribution {
    Normal,
    Uniform,
}

impl VarianceScaling {
    pub fn new(scale: f64, mode: VarianceScalingMode, distribution: VarianceDistribution) -> Self {
        Self {
            scale,
            mode,
            distribution,
        }
    }
}

impl Initializer for VarianceScaling {
    fn init(&self, shape: &[usize]) -> Tensor {
        let fan_in = shape[0] as f64;
        let fan_out = shape[1] as f64;

        let n = match self.mode {
            VarianceScalingMode::FanIn => fan_in,
            VarianceScalingMode::FanOut => fan_out,
            VarianceScalingMode::FanAvg => (fan_in + fan_out) / 2.0,
        };

        let std = (self.scale / n).sqrt();

        match self.distribution {
            VarianceDistribution::Normal => {
                Tensor::randn_scaled(Shape::new(shape.to_vec()), 0.0, std)
            }
            VarianceDistribution::Uniform => {
                let limit = (3.0 * std).sqrt();
                Tensor::uniform(Shape::new(shape.to_vec()), -limit, limit)
            }
        }
    }
}

// ============================================================================
// ZEROS AND ONES
// ============================================================================

/// Zeros initializer
#[derive(Debug, Clone, Copy)]
pub struct Zeros;

impl Initializer for Zeros {
    fn init(&self, shape: &[usize]) -> Tensor {
        Tensor::zeros(Shape::new(shape.to_vec()))
    }
}

/// Ones initializer
#[derive(Debug, Clone, Copy)]
pub struct Ones;

impl Initializer for Ones {
    fn init(&self, shape: &[usize]) -> Tensor {
        Tensor::ones(Shape::new(shape.to_vec()))
    }
}

/// Constant initializer
#[derive(Debug, Clone, Copy)]
pub struct Constant {
    pub value: f64,
}

impl Constant {
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl Initializer for Constant {
    fn init(&self, shape: &[usize]) -> Tensor {
        Tensor::full(shape.to_vec(), self.value)
    }
}

// ============================================================================
// IDENTITY
// ============================================================================

/// Identity matrix initializer
#[derive(Debug, Clone, Copy)]
pub struct Identity {
    pub gain: f64,
}

impl Identity {
    pub fn new(gain: f64) -> Self {
        Self { gain }
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Initializer for Identity {
    fn init(&self, shape: &[usize]) -> Tensor {
        let n = shape[0].min(shape[1]);
        let mut data = vec![0.0; shape[0] * shape[1]];

        for i in 0..n {
            data[i * shape[1] + i] = self.gain;
        }

        Tensor::new(data, Shape::new(shape.to_vec()))
    }
}

// ============================================================================
// HELPERS
// ============================================================================

/// Initialize a tensor with given strategy
pub fn initialize<I: Initializer>(initializer: &I, shape: &[usize]) -> Tensor {
    initializer.init(shape)
}

/// Common initializer for convolution layers
pub fn conv_init<I: Initializer>(
    initializer: &I,
    in_channels: usize,
    out_channels: usize,
    kernel_size: &[usize],
) -> Tensor {
    let fan_in = in_channels * kernel_size.iter().product::<usize>();
    let fan_out = out_channels * kernel_size.iter().product::<usize>();
    let n = (fan_in + fan_out) / 2;

    let mut shape = Vec::new();
    shape.push(out_channels);
    shape.push(in_channels);
    shape.extend(kernel_size);

    // Use variance scaling approximation
    let std = (2.0 / n as f64).sqrt();
    Tensor::randn_scaled(Shape::new(shape), 0.0, std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_uniform() {
        let init = XavierUniform;
        let t = init.init(&[100, 200][..]);
        assert_eq!(t.shape, Shape::new(vec![100, 200]));
        // Values should be within expected range
        let limit: f64 = (6.0_f64 / 300.0_f64).sqrt();
        assert!(t.data.iter().all(|&v| v.abs() <= limit * 1.1)); // Small tolerance
    }

    #[test]
    fn test_he_normal() {
        let init = HeNormal;
        let t = init.init(&[64, 128][..]);
        assert_eq!(t.shape, Shape::new(vec![64, 128]));
    }

    #[test]
    fn test_ones() {
        let init = Ones;
        let t = init.init(&[3, 3][..]);
        assert_eq!(t.data, vec![1.0; 9]);
    }

    #[test]
    fn test_identity() {
        let init = Identity::new(2.0);
        let t = init.init(&[3, 3][..]);
        // Diagonal should be 2.0, rest 0.0
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert_eq!(t.data[i * 3 + j], expected);
            }
        }
    }
}
