//! Efficient Attention Variants for SOTA Transformers
//!
//! Implementation of modern efficient attention mechanisms:
//! - Linformer: Linear complexity O(n) attention
//! - Performer: Kernel-based attention with random feature maps
//! - SparseAttention: Longformer-style sparse patterns
//! - LocalAttention: Sliding window attention

use super::ops::{Shape, Tensor};

/// Linformer: Linear attention with O(n) complexity
///
/// Projects key and value matrices to lower dimension, enabling
/// linear complexity instead of quadratic O(n²).
#[derive(Debug)]
pub struct Linformer {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Projection dimension for low-rank approximation
    pub k: usize,
    /// Key projection matrix
    pub k_proj: Tensor,
    /// Value projection matrix
    pub v_proj: Tensor,
    /// Query projection
    pub w_q: Tensor,
    /// Output projection
    pub w_o: Tensor,
    /// Per-head dimension
    pub d_k: usize,
}

impl Linformer {
    /// Create a new Linformer attention layer
    ///
    /// # Arguments
    /// * `embed_dim` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `k` - Low-rank projection dimension (controls efficiency vs accuracy)
    pub fn new(embed_dim: usize, num_heads: usize, k: usize) -> Self {
        let d_k = embed_dim / num_heads;

        let glorot = |n: usize, m: usize| -> Tensor {
            let limit = (6.0 / (n as f64 + m as f64) as f64).sqrt();
            let mut data = Vec::with_capacity(n * m);
            for _ in 0..(n * m) {
                data.push((pseudo_random() * 2.0 - 1.0) * limit);
            }
            Tensor::new(data, Shape::new(vec![n, m]))
        };

        Self {
            embed_dim,
            num_heads,
            k,
            k_proj: glorot(k, d_k),
            v_proj: glorot(k, d_k),
            w_q: glorot(embed_dim, embed_dim),
            w_o: glorot(embed_dim, embed_dim),
            d_k,
        }
    }

    /// Forward pass with linear complexity attention
    ///
    /// Uses low-rank projection: K' = E * K, V' = E * V
    /// where E is a learned projection matrix of shape (k, seq_len)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x shape: [seq_len, embed_dim]
        let seq_len = x.shape.dims[0];

        // Project queries: x @ w_q gives [seq_len, embed_dim] result
        let q = x.matmul(&self.w_q).unwrap();

        // For each head, apply low-rank projection
        let mut outputs = Vec::with_capacity(self.num_heads * seq_len * self.d_k);

        for head in 0..self.num_heads {
            let head_start = head * self.d_k;
            let head_end = head_start + self.d_k;

            // Extract head-specific queries from the projected output
            // q has shape [seq_len, embed_dim], we need columns [head_start:head_end]
            let mut q_head_data = Vec::with_capacity(seq_len * self.d_k);
            for row in 0..seq_len {
                for col in head_start..head_end {
                    q_head_data.push(q.data[row * self.embed_dim + col]);
                }
            }
            let q_head = Tensor::new(q_head_data, Shape::new(vec![seq_len, self.d_k]));

            // Simulated key and value (in practice, these come from input projections)
            let k_head = Tensor::randn(Shape::new(vec![seq_len, self.d_k]));
            let v_head = Tensor::randn(Shape::new(vec![seq_len, self.d_k]));

            // Low-rank projection: Create projection matrix for this sequence length
            // In practice, these would be learned parameters or fixed projections
            let glorot = |n: usize, m: usize| -> Tensor {
                let limit = (6.0 / (n as f64 + m as f64) as f64).sqrt();
                let mut data = Vec::with_capacity(n * m);
                for _ in 0..(n * m) {
                    data.push((pseudo_random() * 2.0 - 1.0) * limit);
                }
                Tensor::new(data, Shape::new(vec![n, m]))
            };

            // Create sequence projection matrix E: [k, seq_len]
            let e_proj = glorot(self.k, seq_len);

            // K_proj = E @ K : [k, seq_len] @ [seq_len, d_k] = [k, d_k]
            let k_proj = e_proj.matmul(&k_head).unwrap(); // [k, d_k]
            let v_proj = e_proj.matmul(&v_head).unwrap(); // [k, d_k]

            // Compute attention: Q * K_proj^T / sqrt(d_k)
            // Q [seq_len, d_k] @ K_proj.T [d_k, k] = [seq_len, k]
            let k_proj_t = k_proj.transpose().unwrap(); // [d_k, k]
            let scores = q_head.matmul(&k_proj_t).unwrap(); // [seq_len, k]

            let d_k_sqrt = (self.d_k as f64).sqrt();
            let scaled_scores = Tensor::scalar(1.0 / d_k_sqrt).mul(&scores).unwrap();

            // Softmax over k dimension
            let attn_weights = self.softmax_over_last_dim(&scaled_scores);

            // Output: attention_weights * V_proj
            let head_out = attn_weights.matmul(&v_proj).unwrap(); // [seq_len, d_k]

            outputs.extend_from_slice(&head_out.data);
        }

        let concat = Tensor::new(outputs, Shape::new(vec![seq_len, self.embed_dim]));
        concat.matmul(&self.w_o).unwrap()
    }

    /// Softmax over the last dimension for attention weights
    fn softmax_over_last_dim(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape.dims[0];
        let last_dim = x.shape.dims[1];

        let mut result = Vec::with_capacity(x.data.len());

        for i in 0..seq_len {
            // Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..last_dim {
                let val = x.data[i * last_dim + j];
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute exp and sum
            let mut exp_sum = 0.0;
            let mut exp_vals = Vec::with_capacity(last_dim);
            for j in 0..last_dim {
                let exp_val = (x.data[i * last_dim + j] - max_val).exp();
                exp_sum += exp_val;
                exp_vals.push(exp_val);
            }

            // Normalize
            for exp_val in exp_vals {
                result.push(exp_val / exp_sum);
            }
        }

        Tensor::new(result, x.shape.clone())
    }
}

/// Performer: Kernel-based attention with random feature maps
///
/// Uses kernel trick with random feature mapping to achieve
/// linear complexity attention: attention(X, X) ≈ φ(X) φ(X)^T
#[derive(Debug)]
pub struct Performer {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Random feature map dimension
    pub feature_map_dim: usize,
    /// Query projection
    pub w_q: Tensor,
    /// Key projection
    pub w_k: Tensor,
    /// Value projection
    pub w_v: Tensor,
    /// Output projection
    pub w_o: Tensor,
    /// Random projection matrix for feature mapping
    pub random_proj: Tensor,
    /// Per-head dimension
    pub d_k: usize,
}

impl Performer {
    /// Create a new Performer attention layer
    ///
    /// # Arguments
    /// * `embed_dim` - Model dimension
    /// * `num_heads` - Number of attention heads
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let d_k = embed_dim / num_heads;
        let feature_map_dim = d_k * 2; // Common practice: use 2*d_k features

        let glorot = |n: usize, m: usize| -> Tensor {
            let limit = (6.0 / (n as f64 + m as f64) as f64).sqrt();
            let mut data = Vec::with_capacity(n * m);
            for _ in 0..(n * m) {
                data.push((pseudo_random() * 2.0 - 1.0) * limit);
            }
            Tensor::new(data, Shape::new(vec![n, m]))
        };

        // Random projection for feature mapping (sampled once, fixed)
        let random_proj = glorot(feature_map_dim, d_k);

        Self {
            embed_dim,
            num_heads,
            feature_map_dim,
            w_q: glorot(embed_dim, embed_dim),
            w_k: glorot(embed_dim, embed_dim),
            w_v: glorot(embed_dim, embed_dim),
            w_o: glorot(embed_dim, embed_dim),
            random_proj,
            d_k,
        }
    }

    /// Forward pass with kernel-based attention
    ///
    /// Uses random feature map: φ(x) = exp(-||x||²/2) * [cos(Wx), sin(Wx)]
    /// to approximate softmax kernel attention in O(n) time
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape.dims[0];

        // Project to Q, K, V using x @ projection
        let q = x.matmul(&self.w_q).unwrap();
        let k = x.matmul(&self.w_k).unwrap();
        let v = x.matmul(&self.w_v).unwrap();

        let mut outputs = Vec::with_capacity(self.num_heads * seq_len * self.d_k);

        for head in 0..self.num_heads {
            let head_start = head * self.d_k;
            let head_end = head_start + self.d_k;

            // Extract head-specific projections from the projected output
            let mut q_head_data = Vec::with_capacity(seq_len * self.d_k);
            let mut k_head_data = Vec::with_capacity(seq_len * self.d_k);
            let mut v_head_data = Vec::with_capacity(seq_len * self.d_k);

            for row in 0..seq_len {
                for col in head_start..head_end {
                    q_head_data.push(q.data[row * self.embed_dim + col]);
                    k_head_data.push(k.data[row * self.embed_dim + col]);
                    v_head_data.push(v.data[row * self.embed_dim + col]);
                }
            }

            let q_head = Tensor::new(q_head_data, Shape::new(vec![seq_len, self.d_k]));
            let k_head = Tensor::new(k_head_data, Shape::new(vec![seq_len, self.d_k]));
            let v_head = Tensor::new(v_head_data, Shape::new(vec![seq_len, self.d_k]));

            // Apply random feature mapping
            let q_phi = self.kernel_map(&q_head);
            let k_phi = self.kernel_map(&k_head);

            // Linear attention: φ(Q) * φ(K)^T * V
            // Compute: (φ(Q) @ φ(K)^T) @ V = φ(Q) @ (φ(K)^T @ V)
            let kt_v = k_phi.transpose().unwrap().matmul(&v_head).unwrap();
            let attn_out = q_phi.matmul(&kt_v).unwrap();

            // Normalize by sequence length
            let norm_factor = 1.0 / (seq_len as f64).sqrt();
            let normalized = attn_out.mul(&Tensor::scalar(norm_factor)).unwrap();

            // attn_out has shape [seq_len, d_k], so we take all seq_len * d_k elements
            outputs.extend_from_slice(&normalized.data);
        }

        let concat = Tensor::new(outputs, Shape::new(vec![seq_len, self.embed_dim]));
        concat.matmul(&self.w_o).unwrap()
    }

    /// Random feature mapping for kernel approximation
    ///
    /// Implements: φ(x) = exp(-||x||²/2) * [cos(Wx), sin(Wx)]
    /// which approximates the softmax kernel
    fn kernel_map(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape.dims[0];
        let d_k = x.shape.dims[1];

        let mut result = Vec::with_capacity(seq_len * self.feature_map_dim);

        for i in 0..seq_len {
            // Compute squared norm for this row
            let mut norm_sq = 0.0;
            for j in 0..d_k {
                let val = x.data[i * d_k + j];
                norm_sq += val * val;
            }
            let normalization = (-norm_sq / 2.0).exp();

            // Project and compute cos/sin features
            for f in 0..self.feature_map_dim {
                let mut projection = 0.0;
                for j in 0..d_k {
                    projection += x.data[i * d_k + j] * self.random_proj.data[f * d_k + j];
                }

                let feature = if f % 2 == 0 {
                    projection.cos()
                } else {
                    projection.sin()
                };

                result.push(feature * normalization);
            }
        }

        Tensor::new(result, Shape::new(vec![seq_len, self.feature_map_dim]))
    }
}

/// SparseAttention: Longformer-style sparse attention patterns
///
/// Combines local sliding window attention with global attention
/// on selected tokens for O(n) complexity.
#[derive(Debug)]
pub struct SparseAttention {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Sliding window size
    pub window_size: usize,
    /// Indices of global attention tokens
    pub global_indices: Vec<usize>,
    /// Query projection
    pub w_q: Tensor,
    /// Key projection
    pub w_k: Tensor,
    /// Value projection
    pub w_v: Tensor,
    /// Output projection
    pub w_o: Tensor,
    /// Per-head dimension
    pub d_k: usize,
}

impl SparseAttention {
    /// Create a new SparseAttention layer
    ///
    /// # Arguments
    /// * `embed_dim` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `window_size` - Size of sliding window for local attention
    pub fn new(embed_dim: usize, num_heads: usize, window_size: usize) -> Self {
        let d_k = embed_dim / num_heads;

        let glorot = |n: usize, m: usize| -> Tensor {
            let limit = (6.0 / (n as f64 + m as f64) as f64).sqrt();
            let mut data = Vec::with_capacity(n * m);
            for _ in 0..(n * m) {
                data.push((pseudo_random() * 2.0 - 1.0) * limit);
            }
            Tensor::new(data, Shape::new(vec![n, m]))
        };

        Self {
            embed_dim,
            num_heads,
            window_size,
            global_indices: Vec::new(), // Can be configured after creation
            w_q: glorot(embed_dim, embed_dim),
            w_k: glorot(embed_dim, embed_dim),
            w_v: glorot(embed_dim, embed_dim),
            w_o: glorot(embed_dim, embed_dim),
            d_k,
        }
    }

    /// Set global attention indices
    pub fn with_global_indices(mut self, indices: Vec<usize>) -> Self {
        self.global_indices = indices;
        self
    }

    /// Forward pass with sparse attention patterns
    ///
    /// Each token attends to:
    /// 1. Tokens within its sliding window
    /// 2. All global attention tokens
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape.dims[0];

        // Project to Q, K, V using x @ projection
        let q = x.matmul(&self.w_q).unwrap();
        let k = x.matmul(&self.w_k).unwrap();
        let v = x.matmul(&self.w_v).unwrap();

        let mut outputs = Vec::with_capacity(self.num_heads * seq_len * self.d_k);

        for head in 0..self.num_heads {
            let head_start = head * self.d_k;
            let head_end = head_start + self.d_k;

            // Extract head-specific projections from the projected output
            let mut q_head_data = Vec::with_capacity(seq_len * self.d_k);
            let mut k_head_data = Vec::with_capacity(seq_len * self.d_k);
            let mut v_head_data = Vec::with_capacity(seq_len * self.d_k);

            for row in 0..seq_len {
                for col in head_start..head_end {
                    q_head_data.push(q.data[row * self.embed_dim + col]);
                    k_head_data.push(k.data[row * self.embed_dim + col]);
                    v_head_data.push(v.data[row * self.embed_dim + col]);
                }
            }

            let q_head = Tensor::new(q_head_data, Shape::new(vec![seq_len, self.d_k]));
            let k_head = Tensor::new(k_head_data, Shape::new(vec![seq_len, self.d_k]));
            let v_head = Tensor::new(v_head_data, Shape::new(vec![seq_len, self.d_k]));

            // Compute sparse attention for each position
            for i in 0..seq_len {
                let mut attn_output = vec![0.0; self.d_k];
                let mut attn_sum = 0.0;

                // Window indices: [max(0, i-w), min(seq_len, i+w+1))
                let window_start = i.saturating_sub(self.window_size / 2);
                let window_end = (i + self.window_size / 2 + 1).min(seq_len);

                // Collect all indices this token attends to
                let mut attend_indices = Vec::new();
                for j in window_start..window_end {
                    attend_indices.push(j);
                }
                // Add global indices
                for &global_idx in &self.global_indices {
                    if global_idx < seq_len && !attend_indices.contains(&global_idx) {
                        attend_indices.push(global_idx);
                    }
                }

                // Compute attention for collected indices
                let d_k_sqrt = (self.d_k as f64).sqrt();

                for &j in &attend_indices {
                    // Compute attention score
                    let mut score = 0.0;
                    for d in 0..self.d_k {
                        score += q_head.data[i * self.d_k + d] * k_head.data[j * self.d_k + d];
                    }
                    let score = score / d_k_sqrt;
                    let attn_weight = score.exp();

                    // Accumulate weighted value
                    for d in 0..self.d_k {
                        attn_output[d] += attn_weight * v_head.data[j * self.d_k + d];
                    }
                    attn_sum += attn_weight;
                }

                // Normalize
                if attn_sum > 0.0 {
                    for d in 0..self.d_k {
                        attn_output[d] /= attn_sum;
                    }
                }

                outputs.extend_from_slice(&attn_output);
            }
        }

        let concat = Tensor::new(outputs, Shape::new(vec![seq_len, self.embed_dim]));
        concat.matmul(&self.w_o).unwrap()
    }
}

/// LocalAttention: Sliding window attention
///
/// Each token only attends to tokens within a fixed-size window
/// around it, achieving O(n * window_size) complexity.
#[derive(Debug)]
pub struct LocalAttention {
    /// Sliding window size
    pub window_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Per-head dimension
    pub d_k: usize,
    /// Projections
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
}

impl LocalAttention {
    /// Create a new LocalAttention layer
    ///
    /// # Arguments
    /// * `window_size` - Size of attention window (must be odd for symmetric windows)
    pub fn new(embed_dim: usize, num_heads: usize, window_size: usize) -> Self {
        let d_k = embed_dim / num_heads;

        let glorot = |n: usize, m: usize| -> Tensor {
            let limit = (6.0 / (n as f64 + m as f64) as f64).sqrt();
            let mut data = Vec::with_capacity(n * m);
            for _ in 0..(n * m) {
                data.push((pseudo_random() * 2.0 - 1.0) * limit);
            }
            Tensor::new(data, Shape::new(vec![n, m]))
        };

        Self {
            window_size,
            embed_dim,
            num_heads,
            d_k,
            w_q: glorot(embed_dim, embed_dim),
            w_k: glorot(embed_dim, embed_dim),
            w_v: glorot(embed_dim, embed_dim),
            w_o: glorot(embed_dim, embed_dim),
        }
    }

    /// Forward pass with sliding window attention
    ///
    /// Each token at position i only attends to tokens in range
    /// [i - window_size//2, i + window_size//2]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape.dims[0];
        let half_window = self.window_size / 2;

        // Project to Q, K, V using x @ projection
        let q = x.matmul(&self.w_q).unwrap();
        let k = x.matmul(&self.w_k).unwrap();
        let v = x.matmul(&self.w_v).unwrap();

        let mut outputs = Vec::with_capacity(self.num_heads * seq_len * self.d_k);

        for head in 0..self.num_heads {
            let head_start = head * self.d_k;
            let head_end = head_start + self.d_k;

            // Extract head-specific projections from the projected output
            let mut q_head_data = Vec::with_capacity(seq_len * self.d_k);
            let mut k_head_data = Vec::with_capacity(seq_len * self.d_k);
            let mut v_head_data = Vec::with_capacity(seq_len * self.d_k);

            for row in 0..seq_len {
                for col in head_start..head_end {
                    q_head_data.push(q.data[row * self.embed_dim + col]);
                    k_head_data.push(k.data[row * self.embed_dim + col]);
                    v_head_data.push(v.data[row * self.embed_dim + col]);
                }
            }

            let q_head = Tensor::new(q_head_data, Shape::new(vec![seq_len, self.d_k]));
            let k_head = Tensor::new(k_head_data, Shape::new(vec![seq_len, self.d_k]));
            let v_head = Tensor::new(v_head_data, Shape::new(vec![seq_len, self.d_k]));

            // Compute local attention for each position
            let d_k_sqrt = (self.d_k as f64).sqrt();

            for i in 0..seq_len {
                let mut attn_output = vec![0.0; self.d_k];
                let mut attn_sum = 0.0;

                // Window range
                let start = i.saturating_sub(half_window);
                let end = (i + half_window + 1).min(seq_len);

                // Compute attention within window
                for j in start..end {
                    // Compute dot product attention score
                    let mut score = 0.0;
                    for d in 0..self.d_k {
                        score += q_head.data[i * self.d_k + d] * k_head.data[j * self.d_k + d];
                    }
                    let score = score / d_k_sqrt;
                    let attn_weight = score.exp();

                    // Accumulate weighted value
                    for d in 0..self.d_k {
                        attn_output[d] += attn_weight * v_head.data[j * self.d_k + d];
                    }
                    attn_sum += attn_weight;
                }

                // Normalize
                if attn_sum > 0.0 {
                    for d in 0..self.d_k {
                        attn_output[d] /= attn_sum;
                    }
                }

                outputs.extend_from_slice(&attn_output);
            }
        }

        let concat = Tensor::new(outputs, Shape::new(vec![seq_len, self.embed_dim]));
        concat.matmul(&self.w_o).unwrap()
    }
}

/// Pseudo-random number generator for weight initialization
fn pseudo_random() -> f64 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    ((seed.wrapping_mul(1103515245_u64).wrapping_add(12345) & 0x7fffffff) as f64)
        / 0x7fffffff as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linformer_creation() {
        let linformer = Linformer::new(64, 4, 16);
        assert_eq!(linformer.embed_dim, 64);
        assert_eq!(linformer.num_heads, 4);
        assert_eq!(linformer.k, 16);
        assert_eq!(linformer.d_k, 16);
    }

    #[test]
    fn test_linformer_forward() {
        let linformer = Linformer::new(32, 2, 8);
        let x = Tensor::randn(Shape::new(vec![10, 32]));
        let output = linformer.forward(&x);
        // Output should have same shape as input
        assert_eq!(output.shape.dims, vec![10, 32]);
    }

    #[test]
    fn test_performer_creation() {
        let performer = Performer::new(64, 4);
        assert_eq!(performer.embed_dim, 64);
        assert_eq!(performer.num_heads, 4);
        assert_eq!(performer.d_k, 16);
        assert_eq!(performer.feature_map_dim, 32);
    }

    #[test]
    fn test_performer_forward() {
        let performer = Performer::new(32, 2);
        let x = Tensor::randn(Shape::new(vec![10, 32]));
        let output = performer.forward(&x);
        assert_eq!(output.shape.dims, vec![10, 32]);
    }

    #[test]
    fn test_performer_kernel_map() {
        let performer = Performer::new(64, 4);
        let x = Tensor::randn(Shape::new(vec![5, 16]));
        let phi = performer.kernel_map(&x);
        // Feature map should double the last dimension
        assert_eq!(phi.shape.dims, vec![5, 32]);
    }

    #[test]
    fn test_sparse_attention_creation() {
        let sparse = SparseAttention::new(64, 4, 8);
        assert_eq!(sparse.embed_dim, 64);
        assert_eq!(sparse.num_heads, 4);
        assert_eq!(sparse.window_size, 8);
        assert!(sparse.global_indices.is_empty());
    }

    #[test]
    fn test_sparse_attention_with_globals() {
        let sparse = SparseAttention::new(64, 4, 8).with_global_indices(vec![0, 5, 10]);
        assert_eq!(sparse.global_indices, vec![0, 5, 10]);
    }

    #[test]
    fn test_sparse_attention_forward() {
        let sparse = SparseAttention::new(32, 2, 4);
        let x = Tensor::randn(Shape::new(vec![10, 32]));
        let output = sparse.forward(&x);
        assert_eq!(output.shape.dims, vec![10, 32]);
    }

    #[test]
    fn test_local_attention_creation() {
        let local = LocalAttention::new(64, 4, 8);
        assert_eq!(local.window_size, 8);
        assert_eq!(local.embed_dim, 64);
        assert_eq!(local.num_heads, 4);
    }

    #[test]
    fn test_local_attention_forward() {
        let local = LocalAttention::new(32, 2, 4);
        let x = Tensor::randn(Shape::new(vec![10, 32]));
        let output = local.forward(&x);
        assert_eq!(output.shape.dims, vec![10, 32]);
    }

    #[test]
    fn test_local_attention_small_window() {
        let local = LocalAttention::new(16, 1, 3); // Window size 3
        let x = Tensor::matrix(vec![1.0; 50], 10, 5); // [10, 5] but embed is 16
        let x = Tensor::randn(Shape::new(vec![10, 16]));
        let output = local.forward(&x);
        assert_eq!(output.shape.dims, vec![10, 16]);
    }

    #[test]
    fn test_all_attention_types_shape_preservation() {
        let x = Tensor::randn(Shape::new(vec![8, 32]));

        let linformer = Linformer::new(32, 2, 8);
        let performer = Performer::new(32, 2);
        let sparse = SparseAttention::new(32, 2, 4);
        let local = LocalAttention::new(32, 2, 4);

        assert_eq!(linformer.forward(&x).shape.dims, vec![8, 32]);
        assert_eq!(performer.forward(&x).shape.dims, vec![8, 32]);
        assert_eq!(sparse.forward(&x).shape.dims, vec![8, 32]);
        assert_eq!(local.forward(&x).shape.dims, vec![8, 32]);
    }

    #[test]
    fn test_linformer_different_k_values() {
        let x = Tensor::randn(Shape::new(vec![10, 64]));

        let k4 = Linformer::new(64, 4, 4);
        let k8 = Linformer::new(64, 4, 8);
        let k16 = Linformer::new(64, 4, 16);

        // All should produce valid outputs
        assert_eq!(k4.forward(&x).shape.dims, vec![10, 64]);
        assert_eq!(k8.forward(&x).shape.dims, vec![10, 64]);
        assert_eq!(k16.forward(&x).shape.dims, vec![10, 64]);
    }

    #[test]
    fn test_complexity_properties() {
        // Test that mechanisms can handle longer sequences
        let long_seq = Tensor::randn(Shape::new(vec![100, 32]));

        let linformer = Linformer::new(32, 2, 8);
        let performer = Performer::new(32, 2);
        let sparse = SparseAttention::new(32, 2, 4).with_global_indices(vec![0, 50, 99]);
        let local = LocalAttention::new(32, 2, 8);

        // All should handle long sequences without issues
        assert!(linformer.forward(&long_seq).shape.dims[0] == 100);
        assert!(performer.forward(&long_seq).shape.dims[0] == 100);
        assert!(sparse.forward(&long_seq).shape.dims[0] == 100);
        assert!(local.forward(&long_seq).shape.dims[0] == 100);
    }
}
