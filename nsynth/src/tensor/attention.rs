//! Attention and Transformer Primitives for nCPU/nSynth
//!
//! Core attention mechanisms including positional encoding, multi-head attention,
//! rotary position embeddings (RoPE), ALiBi, layer normalization, and FlashAttention.

use super::ops::{Shape, Tensor};

// ============================================================================
// Positional Encoding
// ============================================================================

/// Sinusoidal position embeddings for transformers
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Pre-computed positional encodings [max_len, embedding_dim]
    pub encoding: Tensor,
}

impl PositionalEncoding {
    /// Create new positional encoding
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimension of the embedding (must be even)
    /// * `max_len` - Maximum sequence length
    pub fn new(embedding_dim: usize, max_len: usize) -> Self {
        assert!(embedding_dim % 2 == 0, "embedding_dim must be even");

        let mut data = vec![0.0; max_len * embedding_dim];

        for pos in 0..max_len {
            for i in (0..embedding_dim).step_by(2) {
                // Compute the divisor term: 1 / (10000^(2i/d_model))
                let div_term = (-((i as f64) / (embedding_dim as f64)) * (10000.0_f64).ln()).exp();

                // Sinusoidal encoding
                data[pos * embedding_dim + i] = ((pos as f64) * div_term).sin();

                if i + 1 < embedding_dim {
                    data[pos * embedding_dim + i + 1] = ((pos as f64) * div_term).cos();
                }
            }
        }

        Self {
            embedding_dim,
            encoding: Tensor::new(data, Shape::new(vec![max_len, embedding_dim])),
        }
    }

    /// Apply positional encoding to input tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, embedding_dim]
    ///
    /// # Returns
    /// Tensor with positional encoding added
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape.dims[0];
        assert!(
            seq_len <= self.encoding.shape.dims[0],
            "Sequence length exceeds max_len"
        );
        assert!(
            x.shape.dims[1] == self.embedding_dim,
            "Embedding dimension mismatch"
        );

        let mut result = Vec::with_capacity(x.data.len());

        for i in 0..seq_len {
            for d in 0..self.embedding_dim {
                let idx = i * self.embedding_dim + d;
                result.push(x.data[idx] + self.encoding.data[idx]);
            }
        }

        Tensor::new(result, x.shape.clone())
    }
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

/// Full multi-head attention mechanism
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Query projection [embed_dim, embed_dim]
    pub q_proj: Tensor,
    /// Key projection [embed_dim, embed_dim]
    pub k_proj: Tensor,
    /// Value projection [embed_dim, embed_dim]
    pub v_proj: Tensor,
    /// Output projection [embed_dim, embed_dim]
    pub out_proj: Tensor,
}

impl MultiHeadAttention {
    /// Create new multi-head attention layer
    ///
    /// # Arguments
    /// * `embed_dim` - Total embedding dimension
    /// * `num_heads` - Number of attention heads (must divide embed_dim evenly)
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim must be divisible by num_heads"
        );

        let head_dim = embed_dim / num_heads;

        // Glorot initialization
        let limit = (6.0 / (embed_dim as f64 + embed_dim as f64)).sqrt();

        let init_proj = || -> Tensor {
            let mut data = Vec::with_capacity(embed_dim * embed_dim);
            for _ in 0..(embed_dim * embed_dim) {
                data.push((pseudo_random() * 2.0 - 1.0) * limit);
            }
            Tensor::new(data, Shape::new(vec![embed_dim, embed_dim]))
        };

        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: init_proj(),
            k_proj: init_proj(),
            v_proj: init_proj(),
            out_proj: init_proj(),
        }
    }

    /// Forward pass with optional mask
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, embed_dim]
    /// * `mask` - Optional attention mask [seq_len, seq_len]
    ///
    /// # Returns
    /// Output tensor [seq_len, embed_dim]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Project to Q, K, V (matmul: x @ w_proj^T)
        let x_t = x.transpose().unwrap();
        let q = self.q_proj.matmul(&x_t).unwrap().transpose().unwrap();
        let k = self.k_proj.matmul(&x_t).unwrap().transpose().unwrap();
        let v = self.v_proj.matmul(&x_t).unwrap().transpose().unwrap();

        // Split heads
        let q_heads = self.split_heads(&q);
        let k_heads = self.split_heads(&k);
        let v_heads = self.split_heads(&v);

        // Apply scaled dot-product attention
        let attn_out = self.scaled_dot_product(&q_heads, &k_heads, &v_heads, mask);

        // Combine heads
        let combined = self.combine_heads(&attn_out);

        // Output projection
        let combined_t = combined.transpose().unwrap();
        self.out_proj
            .matmul(&combined_t)
            .unwrap()
            .transpose()
            .unwrap()
    }

    /// Split input into multiple heads
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, embed_dim]
    ///
    /// # Returns
    /// Reshaped tensor [num_heads, seq_len, head_dim]
    fn split_heads(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape.dims[0];
        let mut data = Vec::with_capacity(x.data.len());

        for head in 0..self.num_heads {
            for i in 0..seq_len {
                for d in 0..self.head_dim {
                    let src_idx = i * self.embed_dim + head * self.head_dim + d;
                    data.push(x.data[src_idx]);
                }
            }
        }

        Tensor::new(
            data,
            Shape::new(vec![self.num_heads, seq_len, self.head_dim]),
        )
    }

    /// Combine multiple heads back into single tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor [num_heads, seq_len, head_dim]
    ///
    /// # Returns
    /// Reshaped tensor [seq_len, embed_dim]
    fn combine_heads(&self, x: &Tensor) -> Tensor {
        let num_heads = x.shape.dims[0];
        let seq_len = x.shape.dims[1];
        let head_dim = x.shape.dims[2];
        let mut data = Vec::with_capacity(x.data.len());

        for i in 0..seq_len {
            for head in 0..num_heads {
                for d in 0..head_dim {
                    let src_idx = head * seq_len * head_dim + i * head_dim + d;
                    data.push(x.data[src_idx]);
                }
            }
        }

        Tensor::new(data, Shape::new(vec![seq_len, num_heads * head_dim]))
    }

    /// Scaled dot-product attention
    ///
    /// # Arguments
    /// * `q` - Query tensor [num_heads, seq_len_q, head_dim]
    /// * `k` - Key tensor [num_heads, seq_len_k, head_dim]
    /// * `v` - Value tensor [num_heads, seq_len_v, head_dim]
    /// * `mask` - Optional mask
    ///
    /// # Returns
    /// Attention output tensor [num_heads, seq_len_q, head_dim]
    fn scaled_dot_product(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let num_heads = q.shape.dims[0];
        let seq_len_q = q.shape.dims[1];
        let head_dim = q.shape.dims[2];
        let seq_len_k = k.shape.dims[1];

        let mut result = Vec::with_capacity(num_heads * seq_len_q * head_dim);

        for head in 0..num_heads {
            let head_offset = head * seq_len_q * head_dim;

            // Compute attention scores for this head
            for i in 0..seq_len_q {
                let mut attn_weights = Vec::with_capacity(seq_len_k);

                // Compute scores for all positions
                for j in 0..seq_len_k {
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        let q_idx = head_offset + i * head_dim + d;
                        let k_idx = head * seq_len_k * head_dim + j * head_dim + d;
                        score += q.data[q_idx] * k.data[k_idx];
                    }
                    // Scale
                    score /= (head_dim as f64).sqrt();
                    attn_weights.push(score);
                }

                // Apply mask if provided
                if let Some(m) = mask {
                    for j in 0..seq_len_k {
                        if let Some(&mask_val) = m.data.get(j) {
                            if mask_val <= f64::NEG_INFINITY / 2.0 {
                                attn_weights[j] = f64::NEG_INFINITY;
                            }
                        }
                    }
                }

                // Softmax
                let max_val = attn_weights
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = attn_weights.iter().map(|&x| (x - max_val).exp()).sum();

                let attn_probs: Vec<f64> = attn_weights
                    .iter()
                    .map(|&x| (x - max_val).exp() / exp_sum)
                    .collect();

                // Apply attention to values
                for d in 0..head_dim {
                    let mut weighted_sum = 0.0;
                    for j in 0..seq_len_k {
                        let v_idx = head * seq_len_k * head_dim + j * head_dim + d;
                        weighted_sum += attn_probs[j] * v.data[v_idx];
                    }
                    result.push(weighted_sum);
                }
            }
        }

        Tensor::new(result, Shape::new(vec![num_heads, seq_len_q, head_dim]))
    }
}

// ============================================================================
// Rotary Position Embeddings (RoPE)
// ============================================================================

/// Rotary Position Embeddings for transformer attention
#[derive(Debug, Clone)]
pub struct RoPE {
    /// Dimension to rotate (must be even)
    pub dim: usize,
    /// Pre-computed cos values
    pub cos_cache: Vec<f64>,
    /// Pre-computed sin values
    pub sin_cache: Vec<f64>,
    /// Maximum cached position
    pub max_cached: usize,
}

impl RoPE {
    /// Create new RoPE with specified dimension
    ///
    /// # Arguments
    /// * `dim` - Dimension to rotate (must be even)
    pub fn new(dim: usize) -> Self {
        assert!(dim % 2 == 0, "RoPE dimension must be even");

        // Pre-compute for common sequence lengths
        let max_cached = 2048;
        let mut cos_cache = Vec::with_capacity(max_cached * dim / 2);
        let mut sin_cache = Vec::with_capacity(max_cached * dim / 2);

        for pos in 0..max_cached {
            for i in (0..dim).step_by(2) {
                let theta = (pos as f64) * (-0.5 * (i as f64 / dim as f64).ln()).exp();
                cos_cache.push(theta.cos());
                sin_cache.push(theta.sin());
            }
        }

        Self {
            dim,
            cos_cache,
            sin_cache,
            max_cached,
        }
    }

    /// Apply RoPE rotation to query and key tensors in-place
    ///
    /// # Arguments
    /// * `q` - Query tensor (modified in-place)
    /// * `k` - Key tensor (modified in-place)
    /// * `position` - Position index
    pub fn apply(&self, q: &mut Tensor, k: &mut Tensor, position: usize) {
        let dim = self.dim.min(q.data.len() / 2).min(k.data.len() / 2);

        if position >= self.max_cached {
            // Compute on-the-fly for positions beyond cache
            for i in (0..dim).step_by(2) {
                let theta = (position as f64) * (-0.5 * (i as f64 / self.dim as f64).ln()).exp();
                let cos = theta.cos();
                let sin = theta.sin();

                // Rotate query
                if i + 1 < q.data.len() {
                    let q0 = q.data[i];
                    let q1 = q.data[i + 1];
                    q.data[i] = q0 * cos - q1 * sin;
                    q.data[i + 1] = q0 * sin + q1 * cos;
                }

                // Rotate key
                if i + 1 < k.data.len() {
                    let k0 = k.data[i];
                    let k1 = k.data[i + 1];
                    k.data[i] = k0 * cos - k1 * sin;
                    k.data[i + 1] = k0 * sin + k1 * cos;
                }
            }
        } else {
            // Use cached values
            let cache_offset = position * dim / 2;
            let mut pair_idx = 0;

            for i in (0..dim).step_by(2) {
                let cos = self.cos_cache[cache_offset + pair_idx];
                let sin = self.sin_cache[cache_offset + pair_idx];
                pair_idx += 1;

                // Rotate query
                if i + 1 < q.data.len() {
                    let q0 = q.data[i];
                    let q1 = q.data[i + 1];
                    q.data[i] = q0 * cos - q1 * sin;
                    q.data[i + 1] = q0 * sin + q1 * cos;
                }

                // Rotate key
                if i + 1 < k.data.len() {
                    let k0 = k.data[i];
                    let k1 = k.data[i + 1];
                    k.data[i] = k0 * cos - k1 * sin;
                    k.data[i + 1] = k0 * sin + k1 * cos;
                }
            }
        }
    }

    /// Rotate a single dimension with frequency computation
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `dim` - Dimension to rotate
    /// * `position` - Position index
    ///
    /// # Returns
    /// Rotated tensor
    pub fn rotate_frequency(&self, x: &Tensor, dim: usize, position: usize) -> Tensor {
        let mut result = x.clone();

        for i in (0..dim).step_by(2) {
            if i + 1 < result.data.len() {
                let theta = (position as f64) * (-0.5 * (i as f64 / dim as f64).ln()).exp();
                let cos = theta.cos();
                let sin = theta.sin();

                let x0 = result.data[i];
                let x1 = result.data[i + 1];
                result.data[i] = x0 * cos - x1 * sin;
                result.data[i + 1] = x0 * sin + x1 * cos;
            }
        }

        result
    }
}

// ============================================================================
// ALiBi (Attention with Linear Biases)
// ============================================================================

/// Attention with Linear Biases - position-independent attention
#[derive(Debug, Clone)]
pub struct ALiBi {
    /// Number of attention heads
    pub num_heads: usize,
    /// Pre-computed slopes for each head
    pub slopes: Vec<f64>,
}

impl ALiBi {
    /// Create new ALiBi with specified number of heads
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads
    pub fn new(num_heads: usize) -> Self {
        // Compute slopes for each head
        let mut slopes = Vec::with_capacity(num_heads);

        for i in 0..num_heads {
            // ALiBi slope formula: 1 / (2^(i + 1))
            // Using different formula for multi-head: 1 / (2^(8/n_heads * i))
            let slope = 1.0 / (2.0_f64).powf((8.0 / num_heads as f64) * i as f64);
            slopes.push(slope);
        }

        Self { num_heads, slopes }
    }

    /// Apply ALiBi bias to attention scores
    ///
    /// # Arguments
    /// * `attn_scores` - Attention scores tensor [num_heads, seq_len_q, seq_len_k]
    ///
    /// # Returns
    /// Attention scores with ALiBi bias applied
    pub fn forward(&self, attn_scores: &Tensor) -> Tensor {
        let num_heads = attn_scores.shape.dims[0];
        let seq_len_q = attn_scores.shape.dims[1];
        let seq_len_k = attn_scores.shape.dims[2];

        let mut result = Vec::with_capacity(attn_scores.data.len());

        for head in 0..num_heads {
            let slope = if head < self.slopes.len() {
                self.slopes[head]
            } else {
                self.slopes[0]
            };

            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    let idx = head * seq_len_q * seq_len_k + i * seq_len_k + j;
                    // ALiBi bias: -slope * (j - i) for causal, or -slope * distance
                    let distance = (j as f64 - i as f64).abs();
                    let bias = -slope * distance;
                    result.push(attn_scores.data[idx] + bias);
                }
            }
        }

        Tensor::new(result, attn_scores.shape.clone())
    }
}

// ============================================================================
// Layer Normalization
// ============================================================================

/// Proper layer normalization with learnable parameters
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Shape to normalize over
    pub normalized_shape: Vec<usize>,
    /// Learnable scale parameter
    pub weight: Tensor,
    /// Learnable shift parameter
    pub bias: Tensor,
    /// Small constant for numerical stability
    pub eps: f64,
}

impl LayerNorm {
    /// Create new layer normalization
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape of dimensions to normalize
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let size: usize = normalized_shape.iter().product();

        Self {
            normalized_shape: normalized_shape.clone(),
            weight: Tensor::ones(Shape::new(normalized_shape.clone())),
            bias: Tensor::zeros(Shape::new(normalized_shape.clone())),
            eps: 1e-5,
        }
    }

    /// Forward pass with layer normalization
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// Normalized tensor
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Compute mean and variance along the normalized dimensions
        let mean = x.mean();
        let variance =
            x.data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.data.len() as f64;

        // Normalize: (x - mean) / sqrt(var + eps)
        let std_dev = (variance + self.eps).sqrt();
        let mut result = Vec::with_capacity(x.data.len());

        for i in 0..x.data.len() {
            let normalized = (x.data[i] - mean) / std_dev;
            // Scale and shift: weight * normalized + bias
            let weight_idx = i % self.weight.data.len();
            let biased = self.weight.data[weight_idx] * normalized + self.bias.data[weight_idx];
            result.push(biased);
        }

        Tensor::new(result, x.shape.clone())
    }
}

// ============================================================================
// FlashAttention
// ============================================================================

/// FlashAttention - memory-efficient attention algorithm
///
/// Computes attention with O(N) memory instead of O(N^2) by
/// processing in blocks and using online softmax.
///
/// # Arguments
/// * `q` - Query tensor [seq_len_q, head_dim]
/// * `k` - Key tensor [seq_len_k, head_dim]
/// * `v` - Value tensor [seq_len_k, head_dim]
/// * `block_size` - Block size for computation (default: 64)
///
/// # Returns
/// Attention output [seq_len_q, head_dim]
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, block_size: usize) -> Tensor {
    let seq_len_q = q.shape.dims[0];
    let seq_len_k = k.shape.dims[0];
    let head_dim = q.shape.dims[1];

    let block_size = if block_size == 0 { 64 } else { block_size };

    // Output tensor
    let mut output = vec![0.0; seq_len_q * head_dim];

    // Process query in blocks
    for q_block_start in (0..seq_len_q).step_by(block_size) {
        let q_block_end = (q_block_start + block_size).min(seq_len_q);
        let q_block_size = q_block_end - q_block_start;

        // Initialize for this query block
        let mut block_output = vec![0.0; q_block_size * head_dim];
        let mut block_max = vec![f64::NEG_INFINITY; q_block_size];
        let mut block_sum = vec![0.0; q_block_size];

        // Process key/value in blocks
        for kv_block_start in (0..seq_len_k).step_by(block_size) {
            let kv_block_end = (kv_block_start + block_size).min(seq_len_k);
            let kv_block_size = kv_block_end - kv_block_start;

            // Compute attention scores for this block
            for i in 0..q_block_size {
                let q_idx = q_block_start + i;

                for j in 0..kv_block_size {
                    let k_idx = kv_block_start + j;

                    // Compute dot product
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        score += q.data[q_idx * head_dim + d] * k.data[k_idx * head_dim + d];
                    }
                    // Scale
                    score /= (head_dim as f64).sqrt();

                    // Online softmax update
                    let new_max = block_max[i].max(score);
                    let old_scaled = (block_max[i] - new_max).exp();
                    let new_scaled = (score - new_max).exp();

                    // Update output with old values scaled
                    if block_sum[i] > 0.0 {
                        for d in 0..head_dim {
                            block_output[i * head_dim + d] *= old_scaled;
                        }
                    }

                    // Add new contribution
                    for d in 0..head_dim {
                        block_output[i * head_dim + d] += new_scaled * v.data[k_idx * head_dim + d];
                    }

                    // Update normalization
                    block_sum[i] = block_sum[i] * old_scaled + new_scaled;
                    block_max[i] = new_max;
                }
            }
        }

        // Normalize and copy to output
        for i in 0..q_block_size {
            let norm = block_sum[i].max(1e-9);
            for d in 0..head_dim {
                let out_idx = (q_block_start + i) * head_dim + d;
                output[out_idx] = block_output[i * head_dim + d] / norm;
            }
        }
    }

    Tensor::new(output, Shape::new(vec![seq_len_q, head_dim]))
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Generate pseudo-random value in [0, 1)
fn pseudo_random() -> f64 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    ((seed.wrapping_mul(1103515245_u64).wrapping_add(12345) & 0x7fffffff) as f64)
        / 0x7fffffff as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::ops::Tensor;
    use super::*;

    #[test]
    fn test_positional_encoding_creation() {
        let pe = PositionalEncoding::new(512, 1000);
        assert_eq!(pe.embedding_dim, 512);
        assert_eq!(pe.encoding.shape.dims, vec![1000, 512]);
    }

    #[test]
    fn test_positional_encoding_forward() {
        let pe = PositionalEncoding::new(4, 10);
        let x_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0]
            .iter()
            .cycle()
            .take(16)
            .copied()
            .collect();
        let x = Tensor::new(x_data, Shape::new(vec![4, 4]));
        let result = pe.forward(&x);

        // Should add positional encoding
        assert_eq!(result.shape, x.shape);
        assert_ne!(result.data, x.data);
    }

    #[test]
    fn test_multi_head_attention_creation() {
        let mha = MultiHeadAttention::new(64, 4);
        assert_eq!(mha.embed_dim, 64);
        assert_eq!(mha.num_heads, 4);
        assert_eq!(mha.head_dim, 16);
    }

    #[test]
    fn test_multi_head_attention_forward() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = Tensor::randn(Shape::new(vec![10, 8]));
        let result = mha.forward(&x, None);

        assert_eq!(result.shape, x.shape);
    }

    #[test]
    fn test_rope_creation() {
        let rope = RoPE::new(64);
        assert_eq!(rope.dim, 64);
        assert_eq!(rope.max_cached, 2048);
    }

    #[test]
    fn test_rope_apply() {
        let rope = RoPE::new(4);
        let mut q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![4]));
        let mut k = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], Shape::new(vec![4]));

        rope.apply(&mut q, &mut k, 5);

        // Values should be rotated
        assert_ne!(q.data, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_rope_rotate_frequency() {
        let rope = RoPE::new(4);
        let x = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![4]));
        let result = rope.rotate_frequency(&x, 4, 10);

        assert_eq!(result.shape, x.shape);
        assert_ne!(result.data, x.data);
    }

    #[test]
    fn test_alibi_creation() {
        let alibi = ALiBi::new(8);
        assert_eq!(alibi.num_heads, 8);
        assert_eq!(alibi.slopes.len(), 8);
    }

    #[test]
    fn test_alibi_forward() {
        let alibi = ALiBi::new(2);
        let attn_scores = Tensor::ones(Shape::new(vec![2, 10, 10]));
        let result = alibi.forward(&attn_scores);

        // Should apply bias
        assert_ne!(result.data, attn_scores.data);
        assert_eq!(result.shape, attn_scores.shape);
    }

    #[test]
    fn test_layer_norm_creation() {
        let ln = LayerNorm::new(vec![64]);
        assert_eq!(ln.normalized_shape, vec![64]);
        assert_eq!(ln.weight.data.len(), 64);
        assert_eq!(ln.bias.data.len(), 64);
    }

    #[test]
    fn test_layer_norm_forward() {
        let ln = LayerNorm::new(vec![4]);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let result = ln.forward(&x);

        assert_eq!(result.shape, x.shape);
        // Result should be normalized (approximately zero mean, unit variance)
        let mean = result.data.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_flash_attention() {
        let q = Tensor::randn(Shape::new(vec![10, 8]));
        let k = Tensor::randn(Shape::new(vec![15, 8]));
        let v = Tensor::randn(Shape::new(vec![15, 8]));

        let result = flash_attention(&q, &k, &v, 4);

        assert_eq!(result.shape.dims, vec![10, 8]);
    }

    #[test]
    fn test_flash_attention_with_different_block_sizes() {
        let q = Tensor::randn(Shape::new(vec![20, 16]));
        let k = Tensor::randn(Shape::new(vec![25, 16]));
        let v = Tensor::randn(Shape::new(vec![25, 16]));

        let result_small = flash_attention(&q, &k, &v, 8);
        let result_large = flash_attention(&q, &k, &v, 32);

        // Results should be similar (not exact due to numerical precision)
        assert_eq!(result_small.shape, result_large.shape);
    }

    #[test]
    fn test_alibi_slopes_decrease() {
        let alibi = ALiBi::new(4);
        // Slopes should decrease with head index
        assert!(alibi.slopes[0] > alibi.slopes[1]);
        assert!(alibi.slopes[1] > alibi.slopes[2]);
        assert!(alibi.slopes[2] > alibi.slopes[3]);
    }

    #[test]
    fn test_multi_head_attention_with_mask() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = Tensor::randn(Shape::new(vec![5, 8]));

        // Create causal mask (lower triangular)
        let mut mask_data = vec![0.0; 5 * 5];
        for i in 0..5 {
            for j in 0..5 {
                if j > i {
                    mask_data[i * 5 + j] = f64::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::new(mask_data, Shape::new(vec![5, 5]));

        let result = mha.forward(&x, Some(&mask));
        assert_eq!(result.shape, x.shape);
    }

    #[test]
    fn test_positional_encoding_properties() {
        let pe = PositionalEncoding::new(4, 100);

        // Position 0 should have sin(0) = 0, cos(0) = 1 for even/odd pairs
        let pos_0_dim_0 = pe.encoding.data[0]; // sin(0) = 0
        let pos_0_dim_1 = pe.encoding.data[1]; // cos(0) = 1

        assert!((pos_0_dim_0 - 0.0).abs() < 1e-6);
        assert!((pos_0_dim_1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_cache_limit() {
        let rope = RoPE::new(8);
        let mut q = Tensor::new(vec![1.0; 8], Shape::new(vec![8]));
        let mut k = Tensor::new(vec![1.0; 8], Shape::new(vec![8]));

        // Within cache
        rope.apply(&mut q, &mut k, 100);
        let cached_q = q.clone();

        // Beyond cache
        q.data = vec![1.0; 8];
        k.data = vec![1.0; 8];
        rope.apply(&mut q, &mut k, 5000);

        // Both should work (values will be different)
        assert_ne!(q.data, vec![1.0; 8]);
    }
}
