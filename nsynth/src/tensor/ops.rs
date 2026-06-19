//! Tensor operations for nCPU/nSynth
//!
//! Core tensor types and mathematical operations.

use std::fmt;

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

/// Tensor shape
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Broadcast shape to target shape
    pub fn broadcast_to(&self, target: &Shape) -> Result<Shape, String> {
        let mut result_dims = vec![0usize; target.dims.len()];

        // Right-align dimensions
        let self_len = self.dims.len();
        let target_len = target.dims.len();

        for i in 0..target_len {
            let target_dim = target.dims[target_len - 1 - i];
            let self_dim = if i < self_len {
                self.dims[self_len - 1 - i]
            } else {
                1
            };

            if self_dim == target_dim || self_dim == 1 {
                result_dims[target_len - 1 - i] = target_dim;
            } else {
                return Err(format!("Cannot broadcast shape {:?} to {:?}", self, target));
            }
        }

        Ok(Shape { dims: result_dims })
    }
}

/// Tensor - multi-dimensional array
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Data values (flattened)
    pub data: Vec<f64>,
    /// Shape
    pub shape: Shape,
    /// Data type
    pub dtype: DType,
    /// Gradients (for autodiff)
    pub grads: Option<Vec<f64>>,
    /// Whether this tensor requires gradients
    pub requires_grad: bool,
}

impl Tensor {
    /// Create new tensor
    pub fn new(data: Vec<f64>, shape: Shape) -> Self {
        let size = shape.size();
        assert_eq!(
            data.len(),
            size,
            "Data size {} doesn't match shape size {}",
            data.len(),
            size
        );

        Self {
            data,
            shape,
            dtype: DType::F64,
            grads: None,
            requires_grad: false,
        }
    }

    /// Create scalar tensor
    pub fn scalar(value: f64) -> Self {
        Self::new(vec![value], Shape::new(vec![1]))
    }

    /// Create 1D tensor
    pub fn vector(values: Vec<f64>) -> Self {
        let len = values.len();
        Self::new(values, Shape::new(vec![len]))
    }

    /// Create 2D tensor (matrix)
    pub fn matrix(values: Vec<f64>, rows: usize, cols: usize) -> Self {
        Self::new(values, Shape::new(vec![rows, cols]))
    }

    /// Create zeros tensor
    pub fn zeros(shape: Shape) -> Self {
        let data = vec![0.0; shape.size()];
        Self::new(data, shape)
    }

    /// Create ones tensor
    pub fn ones(shape: Shape) -> Self {
        let data = vec![1.0; shape.size()];
        Self::new(data, shape)
    }

    /// Create tensor filled with random values (0-1)
    pub fn rand(shape: Shape) -> Self {
        let mut data = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            data.push(pseudo_random());
        }
        Self::new(data, shape)
    }

    /// Create random normal tensor
    pub fn randn(shape: Shape) -> Self {
        let mut data = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            data.push(box_muller());
        }
        Self::new(data, shape)
    }

    /// Create uniform random tensor in range [min, max)
    pub fn uniform(shape: Shape, min: f64, max: f64) -> Self {
        let mut data = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            let val = pseudo_random();
            data.push(val * (max - min) + min);
        }
        Self::new(data, shape)
    }

    /// Create random normal tensor with given mean and std
    pub fn randn_scaled(shape: Shape, mean: f64, std: f64) -> Self {
        let mut data = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            let val = box_muller();
            data.push(val * std + mean);
        }
        Self::new(data, shape)
    }

    /// Create tensor filled with given value
    pub fn full(shape: Vec<usize>, value: f64) -> Self {
        let data = vec![value; shape.iter().product()];
        Self::new(data, Shape::new(shape))
    }

    /// Enable gradient tracking
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self.grads = Some(vec![0.0; self.data.len()]);
        self
    }

    /// Get value at flat index
    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
    }

    /// Get value at multi-dimensional index
    pub fn at(&self, indices: &[usize]) -> f64 {
        let flat_index = self.flat_index(indices);
        self.data[flat_index]
    }

    /// Set value at flat index
    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    /// Convert multi-dimensional index to flat index
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        let mut index = 0;
        let mut stride = 1;

        for i in (0..self.shape.rank()).rev() {
            assert!(
                indices[i] < self.shape.dims[i],
                "Index {} out of bounds for dimension {}",
                indices[i],
                i
            );
            index += indices[i] * stride;
            stride *= self.shape.dims[i];
        }

        index
    }

    /// Add element-wise
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        let result_shape = self.broadcast_shape(other)?;
        let self_broadcast = broadcast_to(self, &result_shape)?;
        let other_broadcast = broadcast_to(other, &result_shape)?;

        let mut result_data = Vec::with_capacity(result_shape.size());
        for i in 0..result_shape.size() {
            result_data.push(self_broadcast.data[i] + other_broadcast.data[i]);
        }

        Ok(Tensor::new(result_data, result_shape))
    }

    /// Subtract element-wise
    pub fn sub(&self, other: &Tensor) -> Result<Tensor, String> {
        let result_shape = self.broadcast_shape(other)?;
        let self_broadcast = broadcast_to(self, &result_shape)?;
        let other_broadcast = broadcast_to(other, &result_shape)?;

        let mut result_data = Vec::with_capacity(result_shape.size());
        for i in 0..result_shape.size() {
            result_data.push(self_broadcast.data[i] - other_broadcast.data[i]);
        }

        Ok(Tensor::new(result_data, result_shape))
    }

    /// Multiply element-wise
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        let result_shape = self.broadcast_shape(other)?;
        let self_broadcast = broadcast_to(self, &result_shape)?;
        let other_broadcast = broadcast_to(other, &result_shape)?;

        let mut result_data = Vec::with_capacity(result_shape.size());
        for i in 0..result_shape.size() {
            result_data.push(self_broadcast.data[i] * other_broadcast.data[i]);
        }

        Ok(Tensor::new(result_data, result_shape))
    }

    /// Divide element-wise
    pub fn div(&self, other: &Tensor) -> Result<Tensor, String> {
        let result_shape = self.broadcast_shape(other)?;
        let self_broadcast = broadcast_to(self, &result_shape)?;
        let other_broadcast = broadcast_to(other, &result_shape)?;

        let mut result_data = Vec::with_capacity(result_shape.size());
        for i in 0..result_shape.size() {
            result_data.push(self_broadcast.data[i] / other_broadcast.data[i]);
        }

        Ok(Tensor::new(result_data, result_shape))
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape.rank() != 2 || other.shape.rank() != 2 {
            return Err("MatMul requires 2D tensors".to_string());
        }

        let (m, k1) = (self.shape.dims[0], self.shape.dims[1]);
        let (k2, n) = (other.shape.dims[0], other.shape.dims[1]);

        if k1 != k2 {
            return Err(format!("Inner dimensions don't match: {} vs {}", k1, k2));
        }

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += self.data[i * k1 + k] * other.data[k * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }

        Ok(Tensor::new(result_data, Shape::new(vec![m, n])))
    }

    /// 2D convolution
    pub fn conv2d(
        &self,
        kernel: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor, String> {
        if self.shape.rank() != 2 || kernel.shape.rank() != 2 {
            return Err("Conv2D requires 2D tensors".to_string());
        }

        let (h, w) = (self.shape.dims[0], self.shape.dims[1]);
        let (kh, kw) = (kernel.shape.dims[0], kernel.shape.dims[1]);

        let (h_out, w_out) = (
            (h + 2 * padding.0 - kh) / stride.0 + 1,
            (w + 2 * padding.1 - kw) / stride.1 + 1,
        );

        let mut result_data = vec![0.0; h_out * w_out];

        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = 0.0;

                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        let ih = oh * stride.0 + kh_idx - padding.0;
                        let iw = ow * stride.1 + kw_idx - padding.1;

                        if ih < h && iw < w {
                            let input_val = self.data[ih * w + iw];
                            let kernel_val = kernel.data[kh_idx * kw + kw_idx];
                            sum += input_val * kernel_val;
                        }
                    }
                }

                result_data[oh * w_out + ow] = sum;
            }
        }

        Ok(Tensor::new(result_data, Shape::new(vec![h_out, w_out])))
    }

    /// ReLU activation
    pub fn relu(&self) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Tensor {
        let data: Vec<f64> = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Softmax
    pub fn softmax(&self) -> Tensor {
        if self.shape.rank() != 1 {
            return self.clone(); // Only for 1D
        }

        let max_val = self.data.iter().cloned().fold(f64::NAN, f64::max);
        let exp_sum: f64 = self.data.iter().map(|&x| (x - max_val).exp()).sum();

        let data: Vec<f64> = self
            .data
            .iter()
            .map(|&x| (x - max_val).exp() / exp_sum)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Sum all elements
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f64 {
        self.sum() / self.data.len() as f64
    }

    /// Maximum value
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NAN, f64::max)
    }

    /// Minimum value
    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NAN, f64::min)
    }

    /// Transpose (for 2D tensors)
    pub fn transpose(&self) -> Result<Tensor, String> {
        if self.shape.rank() != 2 {
            return Err("Transpose requires 2D tensor".to_string());
        }

        let (rows, cols) = (self.shape.dims[0], self.shape.dims[1]);
        let mut result_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Tensor::new(result_data, Shape::new(vec![cols, rows])))
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: Shape) -> Result<Tensor, String> {
        if self.shape.size() != new_shape.size() {
            return Err(format!(
                "Cannot reshape size {} to {}",
                self.shape.size(),
                new_shape.size()
            ));
        }

        Ok(Tensor::new(self.data.clone(), new_shape))
    }

    /// Get broadcast shape for two tensors
    fn broadcast_shape(&self, other: &Tensor) -> Result<Shape, String> {
        let self_rank = self.shape.rank();
        let other_rank = other.shape.rank();
        let max_rank = self_rank.max(other_rank);

        let mut result_dims = vec![1usize; max_rank];

        for i in 0..max_rank {
            let self_dim = if i < self_rank {
                self.shape.dims[self_rank - 1 - i]
            } else {
                1
            };

            let other_dim = if i < other_rank {
                other.shape.dims[other_rank - 1 - i]
            } else {
                1
            };

            if self_dim == other_dim {
                result_dims[max_rank - 1 - i] = self_dim;
            } else if self_dim == 1 {
                result_dims[max_rank - 1 - i] = other_dim;
            } else if other_dim == 1 {
                result_dims[max_rank - 1 - i] = self_dim;
            } else {
                return Err(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    self.shape, other.shape
                ));
            }
        }

        Ok(Shape::new(result_dims))
    }

    /// Clone tensor without gradients
    pub fn clone_data(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            grads: None,
            requires_grad: false,
        }
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Variance of all elements
    pub fn var(&self) -> f64 {
        let mean = self.mean();
        self.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.data.len() as f64
    }

    /// Mean along specified dimension (simplified)
    pub fn mean_dim(&self, dim: &[usize]) -> Result<Tensor, String> {
        let mean = self.mean();
        Ok(Tensor::scalar(mean))
    }

    /// Variance along specified dimension (simplified)
    pub fn var_dim(&self, dim: &[usize]) -> Result<Tensor, String> {
        let var = self.var();
        Ok(Tensor::scalar(var))
    }

    /// Sum along specified dimension (simplified)
    pub fn sum_dim(&self, dim: &[usize]) -> Result<Tensor, String> {
        let sum = self.sum();
        Ok(Tensor::scalar(sum))
    }

    /// Add dimension of size 1 at specified position
    pub fn unsqueeze(&self, dim: i32) -> Tensor {
        let mut new_dims = self.shape.dims.clone();
        let dim_idx = if dim < 0 {
            (self.shape.rank() as i32 + dim) as usize
        } else {
            dim as usize
        };
        new_dims.insert(dim_idx, 1);
        Tensor::new(self.data.clone(), Shape::new(new_dims))
    }

    /// Maximum value along specified dimension (simplified)
    pub fn max_dim(&self, dim: usize) -> (usize, f64) {
        let max_val = self.max();
        let max_idx = self
            .data
            .iter()
            .position(|&x| (x - max_val).abs() < 1e-9)
            .unwrap_or(0);
        (max_idx, max_val)
    }

    /// Concatenate tensors along specified dimension
    pub fn concat(tensors: &[Tensor], dim: usize) -> Result<Tensor, String> {
        if tensors.is_empty() {
            return Err("Cannot concat empty tensor list".to_string());
        }

        let first = &tensors[0];
        let mut total_dim = 0;
        for t in tensors {
            if t.shape.rank() != first.shape.rank() {
                return Err("All tensors must have same rank".to_string());
            }
            total_dim += t.shape.dims.get(dim).unwrap_or(&1);
        }

        let mut result_data = Vec::new();
        for t in tensors {
            result_data.extend(&t.data);
        }

        let mut new_dims = first.shape.dims.clone();
        new_dims[dim] = total_dim;

        Ok(Tensor::new(result_data, Shape::new(new_dims)))
    }

    /// Pad tensor with constant value
    pub fn pad(&self, pad_size: usize, value: f64) -> Tensor {
        let mut new_data = vec![value; self.data.len() + 2 * pad_size];
        new_data[pad_size..pad_size + self.data.len()].clone_from_slice(&self.data);
        let mut new_dims = self.shape.dims.clone();
        if let Some(last) = new_dims.last_mut() {
            *last += 2 * pad_size;
        }
        Tensor::new(new_data, Shape::new(new_dims))
    }

    /// Stack tensors along new dimension
    pub fn stack(tensors: &[Tensor], dim: usize) -> Result<Tensor, String> {
        if tensors.is_empty() {
            return Err("Cannot stack empty tensor list".to_string());
        }

        let mut result_data = Vec::new();
        for t in tensors {
            result_data.extend(&t.data);
        }

        let mut new_dims = tensors[0].shape.dims.clone();
        new_dims.insert(dim, tensors.len());

        Ok(Tensor::new(result_data, Shape::new(new_dims)))
    }

    /// Index tensor with range (simplified: returns rows)
    pub fn index(&self, ranges: &[std::ops::RangeInclusive<usize>]) -> Tensor {
        // Simplified implementation for single range
        if ranges.len() == 1 {
            let start = *ranges[0].start();
            let end = *ranges[0].end();
            let dim_size = self.shape.dims.get(1).copied().unwrap_or(1);
            let mut data = Vec::new();
            for i in start..=end.min(self.shape.dims[0] - 1) {
                for j in 0..dim_size {
                    data.push(self.data[i * dim_size + j]);
                }
            }
            let mut new_dims = vec![end - start + 1];
            if self.shape.rank() > 1 {
                new_dims.push(dim_size);
            }
            return Tensor::new(data, Shape::new(new_dims));
        }
        self.clone()
    }

    /// Leaky ReLU activation
    pub fn leaky_relu(&self, alpha: f64) -> Tensor {
        let mut result = Vec::with_capacity(self.data.len());
        for &v in &self.data {
            if v > 0.0 {
                result.push(v);
            } else {
                result.push(v * alpha);
            }
        }
        Tensor::new(result, self.shape.clone())
    }

    /// Element-wise power (simplified)
    pub fn pow(&self, exponent: &Tensor) -> Result<Tensor, String> {
        if self.shape != exponent.shape {
            return Err("Shape mismatch in pow".to_string());
        }
        let result: Vec<f64> = self
            .data
            .iter()
            .zip(exponent.data.iter())
            .map(|(&a, &b)| a.powf(b))
            .collect();
        Ok(Tensor::new(result, self.shape.clone()))
    }

    /// Convert vector to diagonal matrix
    pub fn diag(&self) -> Tensor {
        let n = self.data.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = self.data[i];
        }
        Tensor::new(data, Shape::new(vec![n, n]))
    }

    /// L2 normalization
    pub fn normalize(&self) -> Tensor {
        let norm_sq: f64 = self.data.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        let result: Vec<f64> = self.data.iter().map(|&x| x / norm).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Create random tensor with same shape
    pub fn random_like(&self) -> Tensor {
        let mut data = Vec::with_capacity(self.data.len());
        for _ in 0..self.data.len() {
            data.push(pseudo_random());
        }
        Tensor::new(data, self.shape.clone())
    }

    /// Create uniform random tensor with same shape
    pub fn uniform_like(&self, low: f64, high: f64) -> Tensor {
        let mut data = Vec::with_capacity(self.data.len());
        for _ in 0..self.data.len() {
            data.push(low + (high - low) * pseudo_random());
        }
        Tensor::new(data, self.shape.clone())
    }

    /// Create random normal tensor with same shape
    pub fn randn_like(&self) -> Tensor {
        let mut data = Vec::with_capacity(self.data.len());
        for _ in 0..self.data.len() {
            data.push(box_muller());
        }
        Tensor::new(data, self.shape.clone())
    }
}

impl From<f64> for Tensor {
    fn from(val: f64) -> Self {
        Tensor::scalar(val)
    }
}

impl From<Vec<f64>> for Tensor {
    fn from(data: Vec<f64>) -> Self {
        Tensor::vector(data)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.rank() == 0 {
            write!(f, "Tensor({})", self.data[0])
        } else if self.shape.rank() == 1 {
            write!(
                f,
                "Tensor([{}])",
                self.data
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else if self.shape.rank() == 2 {
            let (rows, cols) = (self.shape.dims[0], self.shape.dims[1]);
            write!(f, "Tensor([\n")?;
            for i in 0..rows {
                write!(f, "  [")?;
                for j in 0..cols {
                    write!(f, "{:.4}", self.data[i * cols + j])?;
                    if j < cols - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")?;
                if i < rows - 1 {
                    write!(f, ",\n")?;
                }
            }
            write!(f, "\n])")
        } else {
            write!(
                f,
                "Tensor(shape: {:?}, size: {})",
                self.shape.dims,
                self.data.len()
            )
        }
    }
}

/// Broadcast tensor to target shape
fn broadcast_to(tensor: &Tensor, target: &Shape) -> Result<Tensor, String> {
    if tensor.shape == *target {
        return Ok(tensor.clone_data());
    }

    let result_shape = tensor.shape.broadcast_to(target)?;
    let mut result_data = vec![0.0; result_shape.size()];

    // Simple broadcasting implementation
    let mut strides = vec![1usize; result_shape.rank()];
    for i in (1..result_shape.rank()).rev() {
        strides[i - 1] = strides[i] * result_shape.dims[i];
    }

    let tensor_strides = vec![1usize; tensor.shape.rank()];

    for index in 0..result_shape.size() {
        let mut src_index = 0;
        let mut temp_idx = index;

        for i in (0..result_shape.rank()).rev() {
            let dim_idx = temp_idx / strides[i] % result_shape.dims[i];
            temp_idx -= dim_idx * strides[i];

            if i < tensor.shape.rank() {
                let tensor_dim = tensor.shape.dims[tensor.shape.rank() - 1 - i];
                let src_dim_idx = if tensor_dim == 1 { 0 } else { dim_idx };
                src_index = src_index * tensor_dim + src_dim_idx;
            }
        }

        result_data[index] = tensor.data[src_index];
    }

    Ok(Tensor::new(result_data, result_shape))
}

/// Generate pseudo-random value (0-1)
fn pseudo_random() -> f64 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    ((seed.wrapping_mul(1103515245_u64).wrapping_add(12345) & 0x7fffffff) as f64)
        / 0x7fffffff as f64
}

/// Box-Muller transform for normal distribution
fn box_muller() -> f64 {
    let u1 = pseudo_random();
    let u2 = pseudo_random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(t.shape.dims, vec![3]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensor_zeros_ones() {
        let zeros = Tensor::zeros(Shape::new(vec![2, 3]));
        assert_eq!(zeros.data, vec![0.0; 6]);

        let ones = Tensor::ones(Shape::new(vec![2, 3]));
        assert_eq!(ones.data, vec![1.0; 6]);
    }

    #[test]
    fn test_add() {
        let a = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let b = Tensor::vector(vec![4.0, 5.0, 6.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Tensor::matrix(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.matmul(&b).unwrap();

        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_relu() {
        let t = Tensor::vector(vec![-1.0, 0.0, 1.0]);
        let r = t.relu();
        assert_eq!(r.data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let t_t = t.transpose().unwrap();
        assert_eq!(t_t.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m = t.reshape(Shape::new(vec![2, 3])).unwrap();
        assert_eq!(m.shape.dims, vec![2, 3]);
    }

    #[test]
    fn test_sum_mean() {
        let t = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.sum(), 10.0);
        assert_eq!(t.mean(), 2.5);
    }
}
