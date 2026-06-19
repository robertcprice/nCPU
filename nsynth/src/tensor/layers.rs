//! Advanced Neural Network Layers for nCPU/nSynth
//!
//! RNN, LSTM, Attention, Pooling, Normalization, Embedding.

use super::ops::{Shape, Tensor};

/// LSTM (Long Short-Term Memory) layer
#[derive(Debug)]
pub struct LSTM {
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Input gate weights
    pub w_ii: Tensor,
    /// Forget gate weights
    pub w_if: Tensor,
    /// Output gate weights
    pub w_io: Tensor,
    /// Cell gate weights
    pub w_ic: Tensor,
    /// Hidden gate weights
    pub w_hi: Tensor,
    /// Bias terms
    pub b_i: Tensor,
    pub b_f: Tensor,
    pub b_o: Tensor,
    pub b_c: Tensor,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let glorot = |n: usize, m: usize| -> Tensor {
            let limit = (6.0 / (n as f64 + m as f64) as f64).sqrt();
            let mut data = Vec::with_capacity(n * m);
            for _ in 0..(n * m) {
                data.push((pseudo_random() * 2.0 - 1.0) * limit);
            }
            Tensor::new(data, Shape::new(vec![n, m]))
        };

        Self {
            input_size,
            hidden_size,
            w_ii: glorot(hidden_size, input_size),
            w_if: glorot(hidden_size, input_size),
            w_io: glorot(hidden_size, input_size),
            w_ic: glorot(hidden_size, input_size),
            w_hi: glorot(hidden_size, hidden_size),
            b_i: Tensor::zeros(Shape::new(vec![hidden_size])),
            b_f: Tensor::ones(Shape::new(vec![hidden_size])), // Initialize forget gate to 1
            b_o: Tensor::zeros(Shape::new(vec![hidden_size])),
            b_c: Tensor::zeros(Shape::new(vec![hidden_size])),
        }
    }

    /// LSTM cell step
    pub fn cell_step(&self, x: &Tensor, h_prev: &Tensor, c_prev: &Tensor) -> (Tensor, Tensor) {
        // Input gate: i = sigmoid(W_ii * x + W_hi * h + b_i)
        let i_gate = self
            .w_ii
            .matmul(x)
            .unwrap()
            .add(&self.w_hi.matmul(h_prev).unwrap())
            .unwrap()
            .add(&self.b_i)
            .unwrap()
            .sigmoid();

        // Forget gate: f = sigmoid(W_if * x + W_hf * h + b_f)
        let f_gate = self
            .w_if
            .matmul(x)
            .unwrap()
            .add(&self.w_hi.matmul(h_prev).unwrap())
            .unwrap()
            .add(&self.b_f)
            .unwrap()
            .sigmoid();

        // Output gate: o = sigmoid(W_io * x + W_ho * h + b_o)
        let o_gate = self
            .w_io
            .matmul(x)
            .unwrap()
            .add(&self.w_hi.matmul(h_prev).unwrap())
            .unwrap()
            .add(&self.b_o)
            .unwrap()
            .sigmoid();

        // Cell candidate: g = tanh(W_ic * x + W_hc * h + b_c)
        let g_gate = self
            .w_ic
            .matmul(x)
            .unwrap()
            .add(&self.w_hi.matmul(h_prev).unwrap())
            .unwrap()
            .add(&self.b_c)
            .unwrap()
            .tanh();

        // New cell state: c = f * c_prev + i * g
        let c_new = f_gate
            .mul(c_prev)
            .unwrap()
            .add(&i_gate.mul(&g_gate).unwrap())
            .unwrap();

        // New hidden state: h = o * tanh(c)
        let h_new = o_gate.mul(&c_new.tanh()).unwrap();

        (h_new, c_new)
    }
}

/// GRU (Gated Recurrent Unit) layer
#[derive(Debug)]
pub struct GRU {
    pub input_size: usize,
    pub hidden_size: usize,
    /// Update gate weights
    pub w_z: Tensor,
    pub u_z: Tensor,
    /// Reset gate weights
    pub w_r: Tensor,
    pub u_r: Tensor,
    /// Candidate weights
    pub w_h: Tensor,
    pub u_h: Tensor,
    /// Bias terms
    pub b_z: Tensor,
    pub b_r: Tensor,
    pub b_h: Tensor,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let glorot = |n: usize, m: usize| -> Tensor {
            let limit = (6.0 / (n as f64 + m as f64) as f64).sqrt();
            let mut data = Vec::with_capacity(n * m);
            for _ in 0..(n * m) {
                data.push((pseudo_random() * 2.0 - 1.0) * limit);
            }
            Tensor::new(data, Shape::new(vec![n, m]))
        };

        Self {
            input_size,
            hidden_size,
            w_z: glorot(hidden_size, input_size),
            u_z: glorot(hidden_size, hidden_size),
            w_r: glorot(hidden_size, input_size),
            u_r: glorot(hidden_size, hidden_size),
            w_h: glorot(hidden_size, input_size),
            u_h: glorot(hidden_size, hidden_size),
            b_z: Tensor::zeros(Shape::new(vec![hidden_size])),
            b_r: Tensor::zeros(Shape::new(vec![hidden_size])),
            b_h: Tensor::zeros(Shape::new(vec![hidden_size])),
        }
    }

    /// GRU cell step
    pub fn cell_step(&self, x: &Tensor, h_prev: &Tensor) -> Tensor {
        // Update gate: z = sigmoid(W_z * x + U_z * h + b_z)
        let z_gate = self
            .w_z
            .matmul(x)
            .unwrap()
            .add(&self.u_z.matmul(h_prev).unwrap())
            .unwrap()
            .add(&self.b_z)
            .unwrap()
            .sigmoid();

        // Reset gate: r = sigmoid(W_r * x + U_r * h + b_r)
        let r_gate = self
            .w_r
            .matmul(x)
            .unwrap()
            .add(&self.u_r.matmul(h_prev).unwrap())
            .unwrap()
            .add(&self.b_r)
            .unwrap()
            .sigmoid();

        // Candidate: h_hat = tanh(W_h * x + U_h * (r * h) + b_h)
        let h_hat = self
            .w_h
            .matmul(x)
            .unwrap()
            .add(&self.u_h.matmul(&r_gate.mul(h_prev).unwrap()).unwrap())
            .unwrap()
            .add(&self.b_h)
            .unwrap()
            .tanh();

        // New hidden: h = (1 - z) * h_prev + z * h_hat
        let one_minus_z = Tensor::ones(z_gate.shape.clone()).sub(&z_gate).unwrap();
        one_minus_z
            .mul(h_prev)
            .unwrap()
            .add(&z_gate.mul(&h_hat).unwrap())
            .unwrap()
    }
}

/// Multi-head attention mechanism
#[derive(Debug)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    /// Query projection
    pub w_q: Tensor,
    /// Key projection
    pub w_k: Tensor,
    /// Value projection
    pub w_v: Tensor,
    /// Output projection
    pub w_o: Tensor,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;

        let glorot = |n: usize, m: usize| -> Tensor {
            let limit = (6.0 / (n as f64 + m as f64) as f64).sqrt();
            let mut data = Vec::with_capacity(n * m);
            for _ in 0..(n * m) {
                data.push((pseudo_random() * 2.0 - 1.0) * limit);
            }
            Tensor::new(data, Shape::new(vec![n, m]))
        };

        Self {
            num_heads,
            d_model,
            d_k,
            w_q: glorot(d_model, d_model),
            w_k: glorot(d_model, d_model),
            w_v: glorot(d_model, d_model),
            w_o: glorot(d_model, d_model),
        }
    }

    /// Scaled dot-product attention
    pub fn scaled_dot_product(
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: Option<&Tensor>,
    ) -> Tensor {
        // scores = Q * K^T / sqrt(d_k)
        let k_t = keys.transpose().unwrap();
        let scores = queries.matmul(&k_t).unwrap();
        let d_k = keys.shape.dims[keys.shape.rank() - 1];
        let scaled_scores = Tensor::scalar(1.0 / (d_k as f64).sqrt())
            .mul(&scores)
            .unwrap();

        // Apply mask if provided
        let masked_scores = if let Some(m) = mask {
            scaled_scores.add(m).unwrap()
        } else {
            scaled_scores
        };

        // attention_weights = softmax(scores)
        let attn_weights = masked_scores.softmax();

        // output = attention_weights * V
        attn_weights.matmul(values).unwrap()
    }

    /// Forward pass with multi-head
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Simplified: assumes x is [seq_len, d_model]
        // In full implementation, would handle batch dimensions and multiple heads

        // Project to Q, K, V
        let q = self.w_q.matmul(x).unwrap();
        let k = self.w_k.matmul(x).unwrap();
        let v = self.w_v.matmul(x).unwrap();

        // Apply attention
        let attn_out = Self::scaled_dot_product(&q, &k, &v, None);

        // Output projection
        self.w_o.matmul(&attn_out).unwrap()
    }
}

/// Layer normalization
#[derive(Debug)]
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub gamma: Tensor, // Scale
    pub beta: Tensor,  // Shift
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let gamma = Tensor::ones(Shape::new(normalized_shape.clone()));
        let beta = Tensor::zeros(Shape::new(normalized_shape.clone()));

        Self {
            normalized_shape,
            gamma,
            beta,
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // mean and variance
        let mean = x.mean();
        let variance =
            x.data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.data.len() as f64;

        // normalize: (x - mean) / sqrt(var + eps)
        let normalized = x
            .data
            .iter()
            .map(|&v| (v - mean) / (variance + self.eps).sqrt())
            .collect::<Vec<f64>>();

        // scale and shift: gamma * normalized + beta
        let mut result = Vec::with_capacity(normalized.len());
        for i in 0..normalized.len() {
            result.push(self.gamma.data[i] * normalized[i] + self.beta.data[i]);
        }

        Tensor::new(result, x.shape.clone())
    }
}

/// Dropout layer
#[derive(Debug)]
pub struct Dropout {
    pub p: f64, // Dropout probability
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self { p, training: false }
    }

    pub fn train(mut self) -> Self {
        self.training = true;
        self
    }

    pub fn eval(mut self) -> Self {
        self.training = false;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training {
            return x.clone();
        }

        let mut result = Vec::with_capacity(x.data.len());
        for &v in &x.data {
            let keep = if pseudo_random() > self.p { 1.0 } else { 0.0 };
            result.push(v * keep / (1.0 - self.p));
        }
        Tensor::new(result, x.shape.clone())
    }
}

/// Max pooling layer
#[derive(Debug)]
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if x.shape.rank() != 2 {
            panic!("MaxPool2d requires 2D input");
        }

        let (h, w) = (x.shape.dims[0], x.shape.dims[1]);
        let (kh, kw) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
        let w_out = (w + 2 * pad_w - kw) / stride_w + 1;

        let mut result = Vec::with_capacity(h_out * w_out);

        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut max_val = f64::NEG_INFINITY;

                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        let ih = oh * stride_h + kh_idx - pad_h;
                        let iw = ow * stride_w + kw_idx - pad_w;

                        if ih < h && iw < w {
                            let val = x.data[ih * w + iw];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                }

                result.push(max_val);
            }
        }

        Tensor::new(result, Shape::new(vec![h_out, w_out]))
    }
}

/// Average pooling layer
#[derive(Debug)]
pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if x.shape.rank() != 2 {
            panic!("AvgPool2d requires 2D input");
        }

        let (h, w) = (x.shape.dims[0], x.shape.dims[1]);
        let (kh, kw) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
        let w_out = (w + 2 * pad_w - kw) / stride_w + 1;

        let mut result = Vec::with_capacity(h_out * w_out);

        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = 0.0;
                let mut count = 0.0;

                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        let ih = oh * stride_h + kh_idx - pad_h;
                        let iw = ow * stride_w + kw_idx - pad_w;

                        if ih < h && iw < w {
                            sum += x.data[ih * w + iw];
                            count += 1.0;
                        }
                    }
                }

                result.push(sum / count);
            }
        }

        Tensor::new(result, Shape::new(vec![h_out, w_out]))
    }
}

/// Embedding layer for categorical data
#[derive(Debug)]
pub struct Embedding {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub weights: Tensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let limit = (1.0 / embedding_dim as f64).sqrt();
        let mut data = Vec::with_capacity(num_embeddings * embedding_dim);
        for _ in 0..(num_embeddings * embedding_dim) {
            data.push((pseudo_random() * 2.0 - 1.0) * limit);
        }

        Self {
            num_embeddings,
            embedding_dim,
            weights: Tensor::new(data, Shape::new(vec![num_embeddings, embedding_dim])),
        }
    }

    /// Look up embeddings for indices
    pub fn forward(&self, indices: &[usize]) -> Tensor {
        let mut result = Vec::with_capacity(indices.len() * self.embedding_dim);

        for &idx in indices {
            if idx >= self.num_embeddings {
                panic!(
                    "Embedding index {} out of bounds for {}",
                    idx, self.num_embeddings
                );
            }
            for d in 0..self.embedding_dim {
                result.push(self.weights.data[idx * self.embedding_dim + d]);
            }
        }

        Tensor::new(result, Shape::new(vec![indices.len(), self.embedding_dim]))
    }
}

/// Positional encoding for transformer
#[derive(Debug)]
pub struct PositionalEncoding {
    pub max_len: usize,
    pub d_model: usize,
    pub encoding: Tensor,
}

impl PositionalEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut data = vec![0.0; max_len * d_model];

        for pos in 0..max_len {
            for i in (0..d_model).step_by(2) {
                let div_term = (i as f64 / d_model as f64 * 2.0 * std::f64::consts::PI).exp();
                data[pos * d_model + i] = ((pos as f64) / div_term).sin();
                if i + 1 < d_model {
                    data[pos * d_model + i + 1] = ((pos as f64) / div_term).cos();
                }
            }
        }

        Self {
            max_len,
            d_model,
            encoding: Tensor::new(data, Shape::new(vec![max_len, d_model])),
        }
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Tensor {
        // Add positional encoding to input
        let seq_len = x.shape.dims[0];
        let mut result = Vec::with_capacity(x.data.len());

        for i in 0..seq_len {
            let pos = offset + i;
            if pos < self.max_len {
                for d in 0..self.d_model {
                    let idx = i * self.d_model + d;
                    result.push(x.data[idx] + self.encoding.data[pos * self.d_model + d]);
                }
            }
        }

        Tensor::new(result, x.shape.clone())
    }
}

/// Transformer encoder block
#[derive(Debug)]
pub struct TransformerEncoder {
    pub self_attn: MultiHeadAttention,
    pub feed_forward: (Tensor, Tensor), // (weight, bias)
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub dropout: Dropout,
}

impl TransformerEncoder {
    pub fn new(d_model: usize, num_heads: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, num_heads),
            feed_forward: (
                Tensor::randn(Shape::new(vec![dim_feedforward, d_model])),
                Tensor::zeros(Shape::new(vec![dim_feedforward])),
            ),
            norm1: LayerNorm::new(vec![d_model]),
            norm2: LayerNorm::new(vec![d_model]),
            dropout: Dropout::new(0.1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Self-attention with residual
        let attn_out = self.self_attn.forward(x);
        let attn_out = self.dropout.forward(&attn_out);
        let norm1_out = self.norm1.forward(&x.add(&attn_out).unwrap());

        // Feed-forward with residual
        let ff_out = norm1_out
            .matmul(&self.feed_forward.0)
            .unwrap()
            .add(&self.feed_forward.1)
            .unwrap();
        let ff_out = ff_out.relu();
        let ff_out = ff_out
            .matmul(&self.feed_forward.0.transpose().unwrap())
            .unwrap();
        let ff_out = self.dropout.forward(&ff_out);
        self.norm2.forward(&norm1_out.add(&ff_out).unwrap())
    }
}

/// Residual connection with optional projection
#[derive(Debug)]
pub struct Residual {
    pub projection: Option<Tensor>,
}

impl Residual {
    pub fn new() -> Self {
        Self { projection: None }
    }

    pub fn with_projection(mut self, weight: Tensor) -> Self {
        self.projection = Some(weight);
        self
    }

    pub fn forward(&self, x: &Tensor, residual: &Tensor) -> Tensor {
        let proj_residual = if let Some(ref proj) = self.projection {
            residual.matmul(proj).unwrap()
        } else {
            residual.clone()
        };
        x.add(&proj_residual).unwrap()
    }
}

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
    fn test_lstm_creation() {
        let lstm = LSTM::new(10, 20);
        assert_eq!(lstm.input_size, 10);
        assert_eq!(lstm.hidden_size, 20);
    }

    #[test]
    fn test_maxpool() {
        let x = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let pool = MaxPool2d::new((2, 2));
        let out = pool.forward(&x);
        assert_eq!(out.data[0], 4.0);
    }

    #[test]
    fn test_avgpool() {
        let x = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let pool = AvgPool2d::new((2, 2));
        let out = pool.forward(&x);
        assert_eq!(out.data[0], 2.5);
    }

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(10, 5);
        let indices: &[usize] = &[0, 3, 7];
        let result = emb.forward(indices);
        assert_eq!(result.shape.dims, vec![3, 5]);
    }

    #[test]
    fn test_dropout_eval_mode() {
        let dropout = Dropout::new(0.5).eval();
        let x = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let out = dropout.forward(&x);
        assert_eq!(out.data, x.data); // No dropout in eval mode
    }
}
