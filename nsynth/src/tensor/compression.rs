//! Model Compression Primitives for nCPU/nSynth
//!
//! Comprehensive compression techniques for neural network models:
//! - Structured pruning (channel/layer removal)
//! - Unstructured pruning (weight-level sparsity)
//! - Quantization (post-training compression)
//! - Knowledge distillation (teacher-student training)

use crate::tensor::ops::Tensor;

/// Quantization mode for different quantization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantMode {
    /// Symmetric quantization (zero-point = 0)
    Symmetric,
    /// Asymmetric quantization (custom zero-point)
    Asymmetric,
    /// Per-channel quantization
    PerChannel,
}

/// Structured Pruning - Channel and layer-level pruning
///
/// Removes entire channels or layers based on importance ranking,
/// maintaining model structure while reducing computational cost.
#[derive(Debug, Clone)]
pub struct StructuredPruning {
    /// Ratio of channels to prune (0.0 to 1.0)
    pub pruning_ratio: f64,
    /// Indicates which layers should be pruned
    pub prune_layers: Vec<bool>,
}

impl StructuredPruning {
    /// Create new structured pruning configuration
    ///
    /// # Arguments
    /// * `pruning_ratio` - Fraction of channels to prune (0.0 to 1.0)
    ///
    /// # Example
    /// ```
    /// let pruning = StructuredPruning::new(0.3); // Prune 30% of channels
    /// ```
    pub fn new(pruning_ratio: f64) -> Self {
        assert!(
            pruning_ratio >= 0.0 && pruning_ratio <= 1.0,
            "Pruning ratio must be between 0 and 1"
        );
        Self {
            pruning_ratio,
            prune_layers: Vec::new(),
        }
    }

    /// Mark specific layers for pruning
    pub fn with_layer_mask(mut self, layer_indices: Vec<bool>) -> Self {
        self.prune_layers = layer_indices;
        self
    }

    /// Compute L1 norm for each channel (importance ranking)
    ///
    /// For convolutional layers, computes sum of absolute weights per channel.
    /// Higher L1 norm indicates more important channels.
    ///
    /// # Arguments
    /// * `layer_weights` - Weight tensor (typically shape: [out_channels, in_channels, ...])
    ///
    /// # Returns
    /// Vector of L1 norms, one per channel
    pub fn compute_l1_norm(&self, layer_weights: &Tensor) -> Tensor {
        let rank = layer_weights.shape.rank();
        if rank < 2 {
            // For 1D/flat weights, return sum of absolute values
            let sum: f64 = layer_weights.data.iter().map(|&x| x.abs()).sum();
            return Tensor::scalar(sum);
        }

        // For 2D+ tensors, compute L1 norm along first dimension (channels)
        let num_channels = layer_weights.shape.dims[0];
        let channel_size = layer_weights.shape.size() / num_channels;

        let mut norms = Vec::with_capacity(num_channels);
        for ch in 0..num_channels {
            let start = ch * channel_size;
            let end = start + channel_size;
            let channel_sum: f64 = layer_weights.data[start..end]
                .iter()
                .map(|&x| x.abs())
                .sum();
            norms.push(channel_sum);
        }

        Tensor::vector(norms)
    }

    /// Prune channels based on L1 norm ranking
    ///
    /// Removes the lowest-importance channels and returns a binary mask.
    ///
    /// # Arguments
    /// * `weights` - Weight tensor to prune
    /// * `num_channels` - Number of channels in the layer
    ///
    /// # Returns
    /// Binary mask tensor (1 = keep, 0 = prune)
    pub fn prune_channels(&mut self, weights: &mut Tensor, num_channels: usize) -> Tensor {
        let l1_norms = self.compute_l1_norm(weights);

        // Determine number of channels to prune
        let num_to_prune = (num_channels as f64 * self.pruning_ratio) as usize;
        let num_to_keep = num_channels - num_to_prune;

        // Create ranking based on L1 norms
        let mut indexed_norms: Vec<(usize, f64)> =
            l1_norms.data.iter().cloned().enumerate().collect();
        indexed_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create mask (1 for keep, 0 for prune)
        let mut mask = vec![0.0f64; num_channels];
        for &(idx, _) in indexed_norms.iter().take(num_to_keep) {
            mask[idx] = 1.0;
        }

        Tensor::vector(mask)
    }

    /// Apply binary mask to weights
    ///
    /// Zeroes out weights where mask is 0.
    ///
    /// # Arguments
    /// * `weights` - Weight tensor
    /// * `mask` - Binary mask tensor (same shape as weights along first dim)
    ///
    /// # Returns
    /// Masked weight tensor
    pub fn apply_mask(&self, weights: &Tensor, mask: &Tensor) -> Tensor {
        let rank = weights.shape.rank();
        if rank == 1 {
            // Element-wise multiplication for 1D
            let mut result_data = Vec::with_capacity(weights.data.len());
            for (i, &w) in weights.data.iter().enumerate() {
                let mask_val = mask.data.get(i).copied().unwrap_or(1.0);
                result_data.push(w * mask_val);
            }
            return Tensor::new(result_data, weights.shape.clone());
        }

        // For higher dimensions, broadcast mask along first dimension
        let num_channels = weights.shape.dims[0];
        let channel_size = weights.shape.size() / num_channels;

        let mut result_data = Vec::with_capacity(weights.data.len());
        for ch in 0..num_channels {
            let mask_val = mask.data.get(ch).copied().unwrap_or(1.0);
            let start = ch * channel_size;
            let end = start + channel_size;
            for &w in &weights.data[start..end] {
                result_data.push(w * mask_val);
            }
        }

        Tensor::new(result_data, weights.shape.clone())
    }

    /// Get number of pruned channels
    pub fn pruned_count(&self, total_channels: usize) -> usize {
        (total_channels as f64 * self.pruning_ratio) as usize
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self, total_channels: usize) -> f64 {
        let kept = total_channels - self.pruned_count(total_channels);
        total_channels as f64 / kept as f64
    }
}

/// Unstructured Pruning - Weight-level pruning
///
/// Creates sparse models by zeroing out individual weights based on magnitude.
#[derive(Debug, Clone)]
pub struct UnstructuredPruning {
    /// Target sparsity level (0.0 = dense, 1.0 = all zeros)
    pub sparsity: f64,
}

impl UnstructuredPruning {
    /// Create new unstructured pruning configuration
    ///
    /// # Arguments
    /// * `sparsity` - Target sparsity (0.0 to 1.0)
    ///
    /// # Example
    /// ```
    /// let pruning = UnstructuredPruning::new(0.5); // 50% sparse
    /// ```
    pub fn new(sparsity: f64) -> Self {
        assert!(
            sparsity >= 0.0 && sparsity <= 1.0,
            "Sparsity must be between 0 and 1"
        );
        Self { sparsity }
    }

    /// Generate binary pruning mask based on weight magnitudes
    ///
    /// Keeps the top (1-sparsity) fraction of weights by magnitude.
    ///
    /// # Arguments
    /// * `weights` - Weight tensor to analyze
    ///
    /// # Returns
    /// Binary mask tensor (same shape as weights)
    pub fn generate_mask(&self, weights: &Tensor) -> Tensor {
        let num_weights = weights.data.len();
        let num_to_keep = (num_weights as f64 * (1.0 - self.sparsity)) as usize;

        // Create indexed weights for sorting
        let mut indexed_weights: Vec<(usize, f64)> =
            weights.data.iter().cloned().enumerate().collect();

        // Sort by absolute value (descending)
        indexed_weights.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create mask
        let mut mask = vec![0.0f64; num_weights];
        for &(idx, _) in indexed_weights.iter().take(num_to_keep) {
            mask[idx] = 1.0;
        }

        Tensor::new(mask, weights.shape.clone())
    }

    /// Apply pruning to weights in-place
    ///
    /// Zeroes out weights based on magnitude threshold.
    ///
    /// # Arguments
    /// * `weights` - Weight tensor to prune (modified in-place)
    pub fn prune_weights(&self, weights: &mut Tensor) {
        let mask = self.generate_mask(weights);
        for (i, &m) in mask.data.iter().enumerate() {
            if m == 0.0 {
                weights.data[i] = 0.0;
            }
        }
    }

    /// Fine-tune mask based on gradient information
    ///
    /// Can be used during training to gradually increase sparsity.
    ///
    /// # Arguments
    /// * `weights` - Current weight tensor
    ///
    /// # Returns
    /// Updated mask tensor
    pub fn finetune_mask(&mut self, weights: &Tensor) -> Tensor {
        self.generate_mask(weights)
    }

    /// Calculate achieved sparsity
    pub fn achieved_sparsity(&self, weights: &Tensor) -> f64 {
        let zero_count = weights.data.iter().filter(|&&x| x.abs() < 1e-9).count();
        zero_count as f64 / weights.data.len() as f64
    }

    /// Progressive pruning: increase sparsity gradually
    pub fn increase_sparsity(&mut self, delta: f64) {
        self.sparsity = (self.sparsity + delta).min(1.0);
    }
}

/// Quantization - Post-training quantization
///
/// Reduces model size by using lower precision representations.
#[derive(Debug, Clone)]
pub struct Quantization {
    /// Number of bits for quantization (typically 8, 4, or 2)
    pub num_bits: usize,
    /// Quantization mode
    pub quant_mode: QuantMode,
}

impl Quantization {
    /// Create new quantization configuration
    ///
    /// # Arguments
    /// * `num_bits` - Bit width for quantization
    ///
    /// # Example
    /// ```
    /// let quant = Quantization::new(8); // 8-bit quantization
    /// ```
    pub fn new(num_bits: usize) -> Self {
        assert!(num_bits > 0 && num_bits <= 32, "Bits must be 1-32");
        Self {
            num_bits,
            quant_mode: QuantMode::Symmetric,
        }
    }

    /// Set quantization mode
    pub fn with_mode(mut self, mode: QuantMode) -> Self {
        self.quant_mode = mode;
        self
    }

    /// Quantize per-tensor (entire tensor shares scale/zero-point)
    ///
    /// # Arguments
    /// * `weights` - Weight tensor to quantize
    ///
    /// # Returns
    /// Tuple of (quantized_tensor, scale, zero_point)
    pub fn quantize_per_tensor(&self, weights: &Tensor) -> (Tensor, Tensor, Tensor) {
        let q_min = -(2_f64.powi(self.num_bits as i32 - 1));
        let q_max = 2_f64.powi(self.num_bits as i32 - 1) - 1.0;

        let (scale, zero_point) = match self.quant_mode {
            QuantMode::Symmetric => {
                let max_val = weights.data.iter().cloned().fold(f64::NAN, f64::max) / q_max;
                let min_val = weights.data.iter().cloned().fold(f64::NAN, f64::min) / q_min;
                let scale = max_val.abs().max(min_val.abs());
                (scale, 0.0)
            }
            QuantMode::Asymmetric => {
                let min_val = weights.data.iter().cloned().fold(f64::NAN, f64::min);
                let max_val = weights.data.iter().cloned().fold(f64::NAN, f64::max);
                let scale = (max_val - min_val) / (q_max - q_min);
                let zero_point = q_min - min_val / scale;
                (scale, zero_point)
            }
            QuantMode::PerChannel => {
                // Fallback to symmetric for per-tensor
                let max_val = weights.data.iter().cloned().fold(f64::NAN, f64::max) / q_max;
                let scale = max_val.abs();
                (scale, 0.0)
            }
        };

        // Quantize
        let quantized_data: Vec<f64> = weights
            .data
            .iter()
            .map(|&x| (x / scale + zero_point).round().clamp(q_min, q_max))
            .collect();

        let quantized = Tensor::new(quantized_data, weights.shape.clone());
        let scale_tensor = Tensor::scalar(scale);
        let zero_point_tensor = Tensor::scalar(zero_point);

        (quantized, scale_tensor, zero_point_tensor)
    }

    /// Quantize per-channel along specified dimension
    ///
    /// Each channel along the dimension gets its own scale.
    ///
    /// # Arguments
    /// * `weights` - Weight tensor to quantize
    /// * `dim` - Dimension along which to quantize separately
    ///
    /// # Returns
    /// Quantized tensor
    pub fn quantize_per_channel(&self, weights: &Tensor, dim: usize) -> Tensor {
        let rank = weights.shape.rank();
        assert!(dim < rank, "Dimension out of bounds");

        let dim_size = weights.shape.dims[dim];
        let stride = weights.shape.size() / dim_size;
        let outer_size = weights.data.len() / weights.shape.dims[dim] / stride;

        let q_min = -(2_f64.powi(self.num_bits as i32 - 1));
        let q_max = 2_f64.powi(self.num_bits as i32 - 1) - 1.0;

        let mut quantized_data = Vec::with_capacity(weights.data.len());

        for outer in 0..outer_size {
            for ch in 0..dim_size {
                // Extract channel data
                let start = (outer * dim_size + ch) * stride;
                let end = start + stride;
                let channel_data = &weights.data[start..end];

                // Compute scale for this channel
                let max_val = channel_data.iter().cloned().fold(f64::NAN, f64::max);
                let min_val = channel_data.iter().cloned().fold(f64::NAN, f64::min);
                let scale = (max_val - min_val) / (q_max - q_min);

                // Quantize channel
                for &x in channel_data {
                    let q = ((x - min_val) / scale + q_min).round().clamp(q_min, q_max);
                    quantized_data.push(q);
                }
            }
        }

        Tensor::new(quantized_data, weights.shape.clone())
    }

    /// Dequantize tensor back to floating point
    ///
    /// # Arguments
    /// * `quantized` - Quantized tensor
    /// * `scale` - Scale factor
    /// * `zero_point` - Zero-point offset
    ///
    /// # Returns
    /// Dequantized floating-point tensor
    pub fn dequantize(&self, quantized: &Tensor, scale: &Tensor, zero_point: &Tensor) -> Tensor {
        let s = scale.data[0];
        let zp = zero_point.data[0];

        let dequantized_data: Vec<f64> = quantized.data.iter().map(|&q| (q - zp) * s).collect();

        Tensor::new(dequantized_data, quantized.shape.clone())
    }

    /// Calculate theoretical size reduction
    pub fn size_reduction_ratio(&self) -> f64 {
        64.0 / self.num_bits as f64
    }

    /// Calculate quantization error (MSE)
    pub fn quantization_error(&self, original: &Tensor, quantized: &Tensor) -> f64 {
        let diff: Vec<f64> = original
            .data
            .iter()
            .zip(quantized.data.iter())
            .map(|(o, q)| o - q)
            .collect();

        let mse = diff.iter().map(|&x| x * x).sum::<f64>() / diff.len() as f64;
        mse.sqrt()
    }
}

/// Knowledge Distillation - Teacher-student training
///
/// Trains a smaller student model to mimic a larger teacher model.
#[derive(Debug, Clone)]
pub struct KnowledgeDistillation {
    /// Teacher model function (reference only, not executed)
    pub teacher_model: Option<String>,
    /// Temperature for soft targets (higher = softer distribution)
    pub temperature: f64,
    /// Balance between distillation and target loss (0.0 to 1.0)
    pub alpha: f64,
}

impl KnowledgeDistillation {
    /// Create new knowledge distillation configuration
    ///
    /// # Arguments
    /// * `temperature` - Softening temperature (typically 2.0-5.0)
    /// * `alpha` - Weight for distillation loss (0.0 to 1.0)
    ///
    /// # Example
    /// ```
    /// let kd = KnowledgeDistillation::new(3.0, 0.5);
    /// ```
    pub fn new(temperature: f64, alpha: f64) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        assert!(
            alpha >= 0.0 && alpha <= 1.0,
            "Alpha must be between 0 and 1"
        );
        Self {
            teacher_model: None,
            temperature,
            alpha,
        }
    }

    /// Set teacher model identifier
    pub fn with_teacher(mut self, teacher: &str) -> Self {
        self.teacher_model = Some(teacher.to_string());
        self
    }

    /// Compute distillation loss
    ///
    /// Combines KL divergence with student-teacher logits and
    /// standard cross-entropy with ground truth.
    ///
    /// # Arguments
    /// * `student_logits` - Raw output from student model
    /// * `teacher_logits` - Softened output from teacher model
    /// * `targets` - Ground truth labels
    ///
    /// # Returns
    /// Combined loss value
    pub fn distillation_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Tensor {
        // Soft targets from teacher
        let teacher_soft = self.soft_targets(teacher_logits);

        // Soft predictions from student
        let student_soft = self.soft_targets(student_logits);

        // KL divergence loss
        let kl_loss = self.kl_divergence(&student_soft, &teacher_soft);

        // Cross-entropy with targets (simplified)
        let ce_loss = self.cross_entropy_loss(student_logits, targets);

        // Combine: alpha * distillation + (1-alpha) * cross-entropy
        let combined = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss;

        Tensor::scalar(combined)
    }

    /// Generate soft targets using temperature scaling
    ///
    /// # Arguments
    /// * `logits` - Raw model outputs
    ///
    /// # Returns
    /// Softened probability distribution
    pub fn soft_targets(&self, logits: &Tensor) -> Tensor {
        let temp = self.temperature;

        // Apply temperature scaling
        let scaled: Vec<f64> = logits.data.iter().map(|&x| x / temp).collect();
        let scaled_tensor = Tensor::new(scaled, logits.shape.clone());

        // Softmax
        self.softmax(&scaled_tensor)
    }

    /// KL divergence between two probability distributions
    fn kl_divergence(&self, p: &Tensor, q: &Tensor) -> f64 {
        let mut kl: f64 = 0.0;
        for (pi, qi) in p.data.iter().zip(q.data.iter()) {
            // Add small epsilon for numerical stability
            let eps = 1e-8;
            let p_val = pi.max(eps);
            let q_val = qi.max(eps);
            kl += p_val * (p_val / q_val).ln();
        }
        (kl).max(0.0f64)
    }

    /// Cross-entropy loss (simplified)
    fn cross_entropy_loss(&self, logits: &Tensor, targets: &Tensor) -> f64 {
        let probs = self.softmax(logits);
        let mut ce = 0.0;
        for (prob, &target) in probs.data.iter().zip(targets.data.iter()) {
            ce -= target * prob.max(1e-8).ln();
        }
        ce
    }

    /// Softmax implementation
    fn softmax(&self, logits: &Tensor) -> Tensor {
        let max_val = logits.data.iter().cloned().fold(f64::NAN, f64::max);
        let exp_sum: f64 = logits.data.iter().map(|&x| (x - max_val).exp()).sum();

        let probs: Vec<f64> = logits
            .data
            .iter()
            .map(|&x| (x - max_val).exp() / exp_sum.max(1e-8))
            .collect();

        Tensor::new(probs, logits.shape.clone())
    }

    /// Adjust temperature during training
    pub fn set_temperature(&mut self, temp: f64) {
        assert!(temp > 0.0, "Temperature must be positive");
        self.temperature = temp;
    }

    /// Adjust alpha (balance between losses)
    pub fn set_alpha(&mut self, alpha: f64) {
        assert!(
            alpha >= 0.0 && alpha <= 1.0,
            "Alpha must be between 0 and 1"
        );
        self.alpha = alpha;
    }

    /// Cosine annealing schedule for temperature
    pub fn anneal_temperature(&mut self, epoch: usize, total_epochs: usize) {
        let progress = epoch as f64 / total_epochs as f64;
        let new_temp =
            self.temperature * (1.0 - 0.5 * (1.0 + (progress * std::f64::consts::PI).cos()));
        self.temperature = new_temp.max(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_pruning_creation() {
        let pruning = StructuredPruning::new(0.5);
        assert_eq!(pruning.pruning_ratio, 0.5);
        assert!(pruning.prune_layers.is_empty());
    }

    #[test]
    fn test_structured_pruning_with_mask() {
        let pruning = StructuredPruning::new(0.3).with_layer_mask(vec![true, false, true]);
        assert_eq!(pruning.prune_layers.len(), 3);
    }

    #[test]
    fn test_compute_l1_norm() {
        let pruning = StructuredPruning::new(0.3);
        let weights = Tensor::matrix(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], 3, 2);
        let norms = pruning.compute_l1_norm(&weights);

        // 3 channels, each with 2 elements
        assert_eq!(norms.shape.dims[0], 3);
        // Channel 0: |1| + |-2| = 3
        assert!((norms.data[0] - 3.0).abs() < 1e-6);
        // Channel 1: |3| + |-4| = 7
        assert!((norms.data[1] - 7.0).abs() < 1e-6);
        // Channel 2: |5| + |-6| = 11
        assert!((norms.data[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_prune_channels() {
        let mut pruning = StructuredPruning::new(0.5);
        let mut weights = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let mask = pruning.prune_channels(&mut weights, 3);

        // Should prune 1.5 channels rounded to 1
        assert_eq!(mask.shape.dims[0], 3);
        let kept = mask.data.iter().filter(|&&x| x > 0.5).count();
        assert_eq!(kept, 2);
    }

    #[test]
    fn test_apply_mask() {
        let pruning = StructuredPruning::new(0.3);
        let weights = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let mask = Tensor::vector(vec![1.0, 0.0, 1.0]);

        let masked = pruning.apply_mask(&weights, &mask);

        // Channel 1 should be zeroed
        assert_eq!(masked.data[2], 0.0);
        assert_eq!(masked.data[3], 0.0);
        // Channels 0 and 2 should remain
        assert_eq!(masked.data[0], 1.0);
        assert_eq!(masked.data[5], 6.0);
    }

    #[test]
    fn test_compression_ratio() {
        let pruning = StructuredPruning::new(0.5);
        let ratio = pruning.compression_ratio(100);
        assert!((ratio - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_unstructured_pruning_creation() {
        let pruning = UnstructuredPruning::new(0.8);
        assert_eq!(pruning.sparsity, 0.8);
    }

    #[test]
    fn test_generate_mask() {
        let pruning = UnstructuredPruning::new(0.5);
        let weights = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mask = pruning.generate_mask(&weights);

        // Should keep 3 weights (50%)
        let kept = mask.data.iter().filter(|&&x| x > 0.5).count();
        assert_eq!(kept, 3);

        // Largest values should be kept
        assert_eq!(mask.data[5], 1.0); // 6.0 (largest)
        assert_eq!(mask.data[4], 1.0); // 5.0
        assert_eq!(mask.data[3], 1.0); // 4.0
    }

    #[test]
    fn test_prune_weights() {
        let pruning = UnstructuredPruning::new(0.5);
        let mut weights = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        pruning.prune_weights(&mut weights);

        // Should have 3 zeros
        let zeros = weights.data.iter().filter(|&&x| x.abs() < 1e-6).count();
        assert_eq!(zeros, 3);
    }

    #[test]
    fn test_achieved_sparsity() {
        let pruning = UnstructuredPruning::new(0.5);
        let mut weights = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        pruning.prune_weights(&mut weights);

        let sparsity = pruning.achieved_sparsity(&weights);
        assert!((sparsity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_progressive_pruning() {
        let mut pruning = UnstructuredPruning::new(0.3);
        pruning.increase_sparsity(0.2);
        assert_eq!(pruning.sparsity, 0.5);

        pruning.increase_sparsity(1.0); // Cap at 1.0
        assert_eq!(pruning.sparsity, 1.0);
    }

    #[test]
    fn test_quantization_creation() {
        let quant = Quantization::new(8);
        assert_eq!(quant.num_bits, 8);
        assert_eq!(quant.quant_mode, QuantMode::Symmetric);
    }

    #[test]
    fn test_quantization_with_mode() {
        let quant = Quantization::new(4).with_mode(QuantMode::Asymmetric);
        assert_eq!(quant.num_bits, 4);
        assert_eq!(quant.quant_mode, QuantMode::Asymmetric);
    }

    #[test]
    fn test_quantize_per_tensor_symmetric() {
        let quant = Quantization::new(8);
        let weights = Tensor::vector(vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let (q, scale, zero_point) = quant.quantize_per_tensor(&weights);

        assert_eq!(zero_point.data[0], 0.0); // Symmetric
        assert!(scale.data[0] > 0.0);
        assert_eq!(q.shape, weights.shape);
    }

    #[test]
    fn test_quantize_per_channel() {
        let quant = Quantization::new(8);
        let weights = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let quantized = quant.quantize_per_channel(&weights, 0);

        assert_eq!(quantized.shape, weights.shape);
        // Values should be quantized (integers)
        for &val in &quantized.data {
            assert_eq!(val.round(), val);
        }
    }

    #[test]
    fn test_dequantize() {
        let quant = Quantization::new(8);
        let weights = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (q, scale, zero_point) = quant.quantize_per_tensor(&weights);
        let dequantized = quant.dequantize(&q, &scale, &zero_point);

        assert_eq!(dequantized.shape, weights.shape);
        // Should be close to original (some quantization error expected)
        let error: f64 = weights
            .data
            .iter()
            .zip(dequantized.data.iter())
            .map(|(o, d)| (o - d).abs())
            .sum();
        assert!(error < weights.data.len() as f64 * scale.data[0]);
    }

    #[test]
    fn test_size_reduction_ratio() {
        let quant8 = Quantization::new(8);
        assert!((quant8.size_reduction_ratio() - 8.0).abs() < 0.1);

        let quant4 = Quantization::new(4);
        assert!((quant4.size_reduction_ratio() - 16.0).abs() < 0.1);
    }

    #[test]
    fn test_quantization_error() {
        let quant = Quantization::new(8);
        let weights = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (q, scale, zero_point) = quant.quantize_per_tensor(&weights);
        let dequantized = quant.dequantize(&q, &scale, &zero_point);

        let error = quant.quantization_error(&weights, &dequantized);
        assert!(error >= 0.0);
        assert!(error < 1.0); // Should be reasonably small for this range
    }

    #[test]
    fn test_knowledge_distillation_creation() {
        let kd = KnowledgeDistillation::new(3.0, 0.5);
        assert_eq!(kd.temperature, 3.0);
        assert_eq!(kd.alpha, 0.5);
    }

    #[test]
    fn test_knowledge_distillation_with_teacher() {
        let kd = KnowledgeDistillation::new(2.0, 0.7).with_teacher("resnet50");
        assert_eq!(kd.teacher_model, Some("resnet50".to_string()));
    }

    #[test]
    fn test_soft_targets() {
        let kd = KnowledgeDistillation::new(2.0, 0.5);
        let logits = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let soft = kd.soft_targets(&logits);

        // Should sum to approximately 1
        let sum: f64 = soft.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All values should be positive
        assert!(soft.data.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_distillation_loss() {
        let kd = KnowledgeDistillation::new(2.0, 0.5);
        let student = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let teacher = Tensor::vector(vec![0.5, 1.5, 2.5]);
        let targets = Tensor::vector(vec![0.0, 1.0, 0.0]);

        let loss = kd.distillation_loss(&student, &teacher, &targets);

        // Loss should be non-negative
        assert!(loss.data[0] >= 0.0);

        // Alpha=0.5 means balanced contribution
        assert!(loss.data[0] < 10.0); // Should be reasonable
    }

    #[test]
    fn test_set_temperature() {
        let mut kd = KnowledgeDistillation::new(3.0, 0.5);
        kd.set_temperature(5.0);
        assert_eq!(kd.temperature, 5.0);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_invalid_temperature() {
        let mut kd = KnowledgeDistillation::new(3.0, 0.5);
        kd.set_temperature(-1.0);
    }

    #[test]
    fn test_anneal_temperature() {
        let mut kd = KnowledgeDistillation::new(5.0, 0.5);
        kd.anneal_temperature(5, 10);

        // Temperature should decrease
        assert!(kd.temperature < 5.0);
        assert!(kd.temperature >= 1.0);
    }

    #[test]
    fn test_pruning_ratio_validation() {
        // Valid ratios
        let p1 = StructuredPruning::new(0.0);
        let p2 = StructuredPruning::new(0.5);
        let p3 = StructuredPruning::new(1.0);

        assert_eq!(p1.pruning_ratio, 0.0);
        assert_eq!(p2.pruning_ratio, 0.5);
        assert_eq!(p3.pruning_ratio, 1.0);
    }

    #[test]
    #[should_panic(expected = "Pruning ratio must be between 0 and 1")]
    fn test_invalid_pruning_ratio() {
        let _ = StructuredPruning::new(1.5);
    }

    #[test]
    #[should_panic(expected = "Sparsity must be between 0 and 1")]
    fn test_invalid_sparsity() {
        let _ = UnstructuredPruning::new(-0.1);
    }

    #[test]
    fn test_mask_application_preserves_shape() {
        let pruning = StructuredPruning::new(0.3);
        let weights = Tensor::matrix(vec![1.0; 12], 3, 4);
        let mask = Tensor::vector(vec![1.0, 1.0, 0.0]);

        let masked = pruning.apply_mask(&weights, &mask);

        assert_eq!(masked.shape, weights.shape);
    }

    #[test]
    fn test_unstructured_mask_shape() {
        let pruning = UnstructuredPruning::new(0.5);
        let weights = Tensor::matrix(vec![1.0; 6], 2, 3);
        let mask = pruning.generate_mask(&weights);

        assert_eq!(mask.shape, weights.shape);
    }

    #[test]
    fn test_symmetric_vs_asymmetric_quantization() {
        let weights = Tensor::vector(vec![-5.0, -2.0, 0.0, 2.0, 5.0]);

        let quant_sym = Quantization::new(8).with_mode(QuantMode::Symmetric);
        let (q_sym, scale_sym, zp_sym) = quant_sym.quantize_per_tensor(&weights);

        let quant_asym = Quantization::new(8).with_mode(QuantMode::Asymmetric);
        let (q_asym, scale_asym, zp_asym) = quant_asym.quantize_per_tensor(&weights);

        // Symmetric should have zero_point near 0
        assert!(zp_sym.data[0].abs() < 1e-6);

        // Asymmetric may have non-zero zero_point
        // (unless data is perfectly symmetric around 0)
        assert_eq!(q_sym.shape, q_asym.shape);
    }
}
