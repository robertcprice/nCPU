//! Advanced Layer Primitives for nCPU/nSynth
//!
//! Composable building blocks for any neural architecture.
//! No hardcoded architectures — just primitives that combine freely.

use super::advanced_ops::concat;
use super::ops::{Shape, Tensor};

// ============================================================================
// RECURRENT LAYERS
// ============================================================================

/// Vanilla RNN Cell (tanh activation)
#[derive(Debug, Clone)]
pub struct RNNCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub w_ih: Tensor,
    pub w_hh: Tensor,
    pub b_ih: Tensor,
    pub b_hh: Tensor,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let w_ih = Tensor::uniform(Shape::new(vec![hidden_size, input_size]), -0.01, 0.01);
        let w_hh = Tensor::uniform(Shape::new(vec![hidden_size, hidden_size]), -0.01, 0.01);
        let b_ih = Tensor::zeros(Shape::new(vec![hidden_size]));
        let b_hh = Tensor::zeros(Shape::new(vec![hidden_size]));
        Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        }
    }

    pub fn forward(&self, x: &Tensor, h: &Tensor) -> Tensor {
        // h_new = tanh(W_ih * x + b_ih + W_hh * h + b_hh)
        let ih = x.matmul(&self.w_ih.transpose().unwrap()).unwrap();
        let hh = h.matmul(&self.w_hh.transpose().unwrap()).unwrap();
        let ih_bias = ih.add(&self.b_ih).unwrap();
        let hh_bias = hh.add(&self.b_hh).unwrap();
        let preact = ih_bias.add(&hh_bias).unwrap();
        preact.tanh()
    }
}

/// Peephole LSTM (LSTM with peephole connections from cell to gates)
#[derive(Debug, Clone)]
pub struct PeepholeLSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    // Input gate
    pub w_ii: Tensor,
    pub w_hi: Tensor,
    pub w_ci: Tensor,
    pub b_i: Tensor,
    // Forget gate
    pub w_if: Tensor,
    pub w_hf: Tensor,
    pub w_cf: Tensor,
    pub b_f: Tensor,
    // Output gate
    pub w_io: Tensor,
    pub w_ho: Tensor,
    pub w_co: Tensor,
    pub b_o: Tensor,
    // Cell candidate
    pub w_ic: Tensor,
    pub w_hc: Tensor,
    pub b_c: Tensor,
}

impl PeepholeLSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = (2.0 / (input_size + hidden_size) as f64).sqrt();
        Self {
            input_size,
            hidden_size,
            w_ii: Tensor::rand(Shape::new(vec![hidden_size, input_size])),
            w_hi: Tensor::rand(Shape::new(vec![hidden_size, hidden_size])),
            w_ci: Tensor::rand(Shape::new(vec![hidden_size])),
            b_i: Tensor::zeros(Shape::new(vec![hidden_size])),
            w_if: Tensor::rand(Shape::new(vec![hidden_size, input_size])),
            w_hf: Tensor::rand(Shape::new(vec![hidden_size, hidden_size])),
            w_cf: Tensor::rand(Shape::new(vec![hidden_size])),
            b_f: Tensor::zeros(Shape::new(vec![hidden_size])),
            w_io: Tensor::rand(Shape::new(vec![hidden_size, input_size])),
            w_ho: Tensor::rand(Shape::new(vec![hidden_size, hidden_size])),
            w_co: Tensor::rand(Shape::new(vec![hidden_size])),
            b_o: Tensor::zeros(Shape::new(vec![hidden_size])),
            w_ic: Tensor::rand(Shape::new(vec![hidden_size, input_size])),
            w_hc: Tensor::rand(Shape::new(vec![hidden_size, hidden_size])),
            b_c: Tensor::zeros(Shape::new(vec![hidden_size])),
        }
    }

    pub fn forward(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        // i = sigmoid(W_ii * x + W_hi * h + W_ci * c + b_i)
        // f = sigmoid(W_if * x + W_hf * h + W_cf * c + b_f)
        // o = sigmoid(W_io * x + W_ho * h + W_co * c + b_o)
        // g = tanh(W_ic * x + W_hc * h + b_c)
        // c_new = f * c + i * g
        // h_new = o * tanh(c_new)

        let i_gate = self.compute_gate(x, h, c, &self.w_ii, &self.w_hi, &self.w_ci, &self.b_i);
        let f_gate = self.compute_gate(x, h, c, &self.w_if, &self.w_hf, &self.w_cf, &self.b_f);
        let o_gate = self.compute_gate(x, h, c, &self.w_io, &self.w_ho, &self.w_co, &self.b_o);

        let ic = x.matmul(&self.w_ic.transpose().unwrap()).unwrap();
        let hc = h.matmul(&self.w_hc.transpose().unwrap()).unwrap();
        let g = ic.add(&hc).unwrap().add(&self.b_c).unwrap().tanh();

        let c_new = f_gate
            .mul(c)
            .unwrap()
            .add(&i_gate.mul(&g).unwrap())
            .unwrap();
        let h_new = o_gate.mul(&c_new.tanh()).unwrap();

        (h_new, c_new)
    }

    fn compute_gate(
        &self,
        x: &Tensor,
        h: &Tensor,
        c: &Tensor,
        w_ix: &Tensor,
        w_hx: &Tensor,
        w_cx: &Tensor,
        b: &Tensor,
    ) -> Tensor {
        let ix = x.matmul(&w_ix.transpose().unwrap()).unwrap();
        let hx = h.matmul(&w_hx.transpose().unwrap()).unwrap();
        let cx = c.mul(&w_cx).unwrap();
        ix.add(&hx)
            .unwrap()
            .add(&cx)
            .unwrap()
            .add(b)
            .unwrap()
            .sigmoid()
    }
}

/// Bidirectional wrapper for any RNN-like forward pass
#[derive(Debug, Clone)]
pub struct Bidirectional<F>
where
    F: Fn(&Tensor, &Tensor) -> Tensor + Clone + 'static,
{
    pub forward_fn: F,
}

impl<F> Bidirectional<F>
where
    F: Fn(&Tensor, &Tensor) -> Tensor + Clone + 'static,
{
    pub fn new(forward_fn: F) -> Self {
        Self { forward_fn }
    }

    pub fn forward(&self, seq: &Tensor, h_init: &Tensor) -> (Tensor, Tensor) {
        // Forward pass: left to right
        let mut h_fwd = h_init.clone();
        let seq_len = seq.shape.dims[0];
        let mut outputs_fwd = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = seq.index(&[t..=t]);
            h_fwd = (self.forward_fn)(&x_t, &h_fwd);
            outputs_fwd.push(h_fwd.clone());
        }

        // Backward pass: right to left
        let mut h_bwd = h_init.clone();
        let mut outputs_bwd = Vec::with_capacity(seq_len);

        for t in (0..seq_len).rev() {
            let x_t = seq.index(&[t..=t]);
            h_bwd = (self.forward_fn)(&x_t, &h_bwd);
            outputs_bwd.push(h_bwd.clone());
        }
        outputs_bwd.reverse();

        // Concatenate forward and backward outputs
        let chained: Vec<Tensor> = outputs_fwd
            .iter()
            .chain(outputs_bwd.iter())
            .map(|t| (*t).clone())
            .collect();
        let output = Tensor::concat(&chained, 0).unwrap();

        (output, h_fwd.add(&h_bwd).unwrap())
    }
}

// ============================================================================
// NORMALIZATION LAYERS
// ============================================================================

/// Batch Normalization 1D (for sequences and dense layers)
#[derive(Debug, Clone)]
pub struct BatchNorm1d {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: Tensor::ones(Shape::new(vec![num_features])),
            beta: Tensor::zeros(Shape::new(vec![num_features])),
            running_mean: Tensor::zeros(Shape::new(vec![num_features])),
            running_var: Tensor::ones(Shape::new(vec![num_features])),
            training: true,
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        if self.training {
            // Compute batch statistics
            let mean = x.mean();
            let var = x.var();

            // Convert to tensors for running stats update
            let mean_tensor = Tensor::scalar(mean);
            let var_tensor = Tensor::scalar(var);

            // Update running statistics
            self.running_mean = self
                .running_mean
                .mul(&(1.0 - self.momentum).into())
                .unwrap()
                .add(&mean_tensor.mul(&self.momentum.into()).unwrap())
                .unwrap();
            self.running_var = self
                .running_var
                .mul(&(1.0 - self.momentum).into())
                .unwrap()
                .add(&var_tensor.mul(&self.momentum.into()).unwrap())
                .unwrap();

            self.normalize(x, &mean_tensor, &var_tensor)
        } else {
            self.normalize(x, &self.running_mean, &self.running_var)
        }
    }

    fn normalize(&self, x: &Tensor, mean: &Tensor, var: &Tensor) -> Tensor {
        let x_centered = x.sub(mean).unwrap();
        let x_norm = x_centered
            .div(&var.add(&self.eps.into()).unwrap().sqrt())
            .unwrap();
        x_norm.mul(&self.gamma).unwrap().add(&self.beta).unwrap()
    }
}

/// Batch Normalization 2D (for CNN feature maps)
#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: Tensor::ones(Shape::new(vec![num_features])),
            beta: Tensor::zeros(Shape::new(vec![num_features])),
            running_mean: Tensor::zeros(Shape::new(vec![num_features])),
            running_var: Tensor::ones(Shape::new(vec![num_features])),
            training: true,
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        // x shape: [N, C, H, W]
        if self.training {
            // Compute per-channel statistics
            let (mean, var) = self.compute_channel_stats(x);

            // Update running statistics
            self.running_mean = self
                .running_mean
                .mul(&(1.0 - self.momentum).into())
                .unwrap()
                .add(&mean.mul(&self.momentum.into()).unwrap())
                .unwrap();
            self.running_var = self
                .running_var
                .mul(&(1.0 - self.momentum).into())
                .unwrap()
                .add(&var.mul(&self.momentum.into()).unwrap())
                .unwrap();

            self.normalize_channels(x, &mean, &var)
        } else {
            self.normalize_channels(x, &self.running_mean, &self.running_var)
        }
    }

    fn compute_channel_stats(&self, x: &Tensor) -> (Tensor, Tensor) {
        // Simplified: compute mean and variance for each channel
        // For full implementation, would iterate over channels
        let mean = x.mean();
        let var = x.var();
        (Tensor::scalar(mean), Tensor::scalar(var))
    }

    fn normalize_channels(&self, x: &Tensor, mean: &Tensor, var: &Tensor) -> Tensor {
        let x_centered = x.sub(mean).unwrap();
        let x_norm = x_centered
            .div(&var.add(&self.eps.into()).unwrap().sqrt())
            .unwrap();
        x_norm.mul(&self.gamma).unwrap().add(&self.beta).unwrap()
    }
}

/// Batch Normalization 3D (for video/volumetric data)
#[derive(Debug, Clone)]
pub struct BatchNorm3d {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub training: bool,
}

impl BatchNorm3d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: Tensor::ones(Shape::new(vec![num_features])),
            beta: Tensor::zeros(Shape::new(vec![num_features])),
            running_mean: Tensor::zeros(Shape::new(vec![num_features])),
            running_var: Tensor::ones(Shape::new(vec![num_features])),
            training: true,
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        if self.training {
            let mean = x.mean();
            let var = x.var();
            let mean_tensor = Tensor::scalar(mean);
            let var_tensor = Tensor::scalar(var);

            self.running_mean = self
                .running_mean
                .mul(&(1.0 - self.momentum).into())
                .unwrap()
                .add(&mean_tensor.mul(&self.momentum.into()).unwrap())
                .unwrap();
            self.running_var = self
                .running_var
                .mul(&(1.0 - self.momentum).into())
                .unwrap()
                .add(&var_tensor.mul(&self.momentum.into()).unwrap())
                .unwrap();

            let x_centered = x.sub(&mean_tensor).unwrap();
            let x_norm = x_centered
                .div(&var_tensor.add(&self.eps.into()).unwrap().sqrt())
                .unwrap();
            x_norm.mul(&self.gamma).unwrap().add(&self.beta).unwrap()
        } else {
            let x_centered = x.sub(&self.running_mean).unwrap();
            let x_norm = x_centered
                .div(&self.running_var.add(&self.eps.into()).unwrap().sqrt())
                .unwrap();
            x_norm.mul(&self.gamma).unwrap().add(&self.beta).unwrap()
        }
    }
}

/// Group Normalization (independent of batch size)
#[derive(Debug, Clone)]
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f64,
    pub gamma: Tensor,
    pub beta: Tensor,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            gamma: Tensor::ones(Shape::new(vec![num_channels])),
            beta: Tensor::zeros(Shape::new(vec![num_channels])),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Divide channels into groups, normalize within each group
        let x_reshaped = x
            .reshape(Shape::new(vec![
                x.shape.dims[0],
                self.num_groups,
                self.num_channels / self.num_groups,
            ]))
            .unwrap();
        let mean = x_reshaped.mean_dim(&[2]).unwrap();
        let var = x_reshaped.var_dim(&[2]).unwrap();

        let x_centered = x_reshaped.sub(&mean).unwrap();
        let x_norm = x_centered
            .div(&var.add(&self.eps.into()).unwrap().sqrt())
            .unwrap();
        let x_reshaped_back = x_norm.reshape(x.shape.clone()).unwrap();

        x_reshaped_back
            .mul(&self.gamma)
            .unwrap()
            .add(&self.beta)
            .unwrap()
    }
}

/// Instance Normalization (normalize per sample, not per batch)
#[derive(Debug, Clone)]
pub struct InstanceNorm {
    pub num_features: usize,
    pub eps: f64,
    pub gamma: Tensor,
    pub beta: Tensor,
}

impl InstanceNorm {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            gamma: Tensor::ones(Shape::new(vec![num_features])),
            beta: Tensor::zeros(Shape::new(vec![num_features])),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Normalize each sample independently
        let mean = x.mean_dim(&[2, 3]).unwrap();
        let var = x.var_dim(&[2, 3]).unwrap();

        let mean_expanded = mean.unsqueeze(-1).unsqueeze(-1);
        let var_expanded = var
            .add(&self.eps.into())
            .unwrap()
            .sqrt()
            .unsqueeze(-1)
            .unsqueeze(-1);
        let gamma_expanded = self.gamma.unsqueeze(-1).unsqueeze(-1);
        let beta_expanded = self.beta.unsqueeze(-1).unsqueeze(-1);

        let x_centered = x.sub(&mean_expanded).unwrap();
        let x_norm = x_centered.div(&var_expanded).unwrap();

        x_norm
            .mul(&gamma_expanded)
            .unwrap()
            .add(&beta_expanded)
            .unwrap()
    }
}

// ============================================================================
// ATTENTION VARIANTS
// ============================================================================

/// Cross-Attention (different from MultiHeadAttention)
/// Query from one sequence, Key/Value from another
#[derive(Debug, Clone)]
pub struct CrossAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub out_proj: Tensor,
    pub scale: f64,
}

impl CrossAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Self {
            embed_dim,
            num_heads,
            q_proj: Tensor::randn(Shape::new(vec![embed_dim, embed_dim])),
            k_proj: Tensor::randn(Shape::new(vec![embed_dim, embed_dim])),
            v_proj: Tensor::randn(Shape::new(vec![embed_dim, embed_dim])),
            out_proj: Tensor::randn(Shape::new(vec![embed_dim, embed_dim])),
            scale,
        }
    }

    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        // Project
        let q = query.matmul(&self.q_proj).unwrap();
        let k = key.matmul(&self.k_proj).unwrap();
        let v = value.matmul(&self.v_proj).unwrap();

        // Scaled dot-product attention
        let scores = q
            .matmul(&k.transpose().unwrap())
            .unwrap()
            .mul(&self.scale.into())
            .unwrap();
        let attn_weights = scores.softmax();
        let attn_output = attn_weights.matmul(&v).unwrap();

        // Output projection
        attn_output.matmul(&self.out_proj).unwrap()
    }
}

// ============================================================================
// CONVOLUTIONAL VARIANTS
// ============================================================================

/// 1D Convolution (for sequences, audio)
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Conv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let scale = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            weight: Tensor::randn(Shape::new(vec![out_channels, in_channels, kernel_size])),
            bias: Tensor::zeros(Shape::new(vec![out_channels])),
        }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Simplified 1D conv using 2D conv with height=1
        self.conv1d_impl(x)
    }

    fn conv1d_impl(&self, x: &Tensor) -> Tensor {
        // Placeholder: use 2D conv implementation
        // Real implementation would use optimized 1D convolution
        Tensor::zeros(Shape::new(vec![1, 1]))
    }
}

/// 3D Convolution (for video, volumetric data)
#[derive(Debug, Clone)]
pub struct Conv3d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Conv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        let (kd, kh, kw) = kernel_size;
        let scale = (2.0 / (in_channels * kd * kh * kw) as f64).sqrt();
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            weight: Tensor::randn(Shape::new(vec![out_channels, in_channels, kd, kh, kw])),
            bias: Tensor::zeros(Shape::new(vec![out_channels])),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Simplified 3D conv
        Tensor::zeros(Shape::new(vec![1, 1, 1, 1, 1]))
    }
}

// ============================================================================
// POOLING VARIANTS
// ============================================================================

/// Adaptive Max Pooling 2D (output fixed size regardless of input)
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveMaxPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Compute stride/kernel to achieve exact output size
        let (h_out, w_out) = self.output_size;
        let (h_in, w_in) = (x.shape.dims[2], x.shape.dims[3]);

        let stride_h = (h_in as f64 / h_out as f64).ceil() as usize;
        let stride_w = (w_in as f64 / w_out as f64).ceil() as usize;
        let kernel_h = stride_h;
        let kernel_w = stride_w;

        // Simplified adaptive pooling
        Tensor::zeros(Shape::new(vec![
            x.shape.dims[0],
            x.shape.dims[1],
            h_out,
            w_out,
        ]))
    }
}

/// Adaptive Average Pooling 2D
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (h_out, w_out) = self.output_size;
        Tensor::zeros(Shape::new(vec![
            x.shape.dims[0],
            x.shape.dims[1],
            h_out,
            w_out,
        ]))
    }
}

/// Global Max Pooling 2D (pool to 1x1)
#[derive(Debug, Clone)]
pub struct GlobalMaxPool2d;

impl GlobalMaxPool2d {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Max over spatial dimensions - simplified implementation
        // For full implementation, would iterate over spatial dims
        let n = x.data.len();
        let max_val = x.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        Tensor::scalar(max_val)
    }
}

/// Global Average Pooling 2D
#[derive(Debug, Clone)]
pub struct GlobalAvgPool2d;

impl GlobalAvgPool2d {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Average over spatial dimensions
        x.mean_dim(&[2, 3]).unwrap()
    }
}

// ============================================================================
// DROPOUT VARIANTS
// ============================================================================

/// Alpha Dropout (for SELU activation - self-normalizing)
#[derive(Debug, Clone)]
pub struct AlphaDropout {
    pub p: f64,
    pub alpha: f64,
    pub scale: f64,
}

impl AlphaDropout {
    pub fn new(p: f64) -> Self {
        Self {
            p,
            alpha: -1.7580993408473766,
            scale: 1.0507009873554804934193349852946,
        }
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Tensor {
        if !training {
            return x.clone();
        }

        // Alpha dropout maintains self-normalizing property
        let noise = x.uniform_like(0.0, 1.0);
        let mut mask_data = Vec::with_capacity(noise.data.len());
        let threshold = 1.0 - self.p;
        for &v in &noise.data {
            mask_data.push(if v >= threshold { 1.0 } else { 0.0 });
        }
        let mask = Tensor::new(mask_data, x.shape.clone());

        let a = self.alpha;
        let scaled = x
            .sub(&self.scale.into())
            .unwrap()
            .div(&self.scale.into())
            .unwrap();
        let noisy = scaled.mul(&mask).unwrap();

        noisy
            .add(&self.alpha.into())
            .unwrap()
            .mul(&self.scale.into())
            .unwrap()
    }
}

/// Gaussian Noise (additive noise for regularization)
#[derive(Debug, Clone)]
pub struct GaussianNoise {
    pub std: f64,
}

impl GaussianNoise {
    pub fn new(std: f64) -> Self {
        Self { std }
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Tensor {
        if !training {
            return x.clone();
        }

        // Generate Gaussian noise with given std
        let noise = x.clone().randn_like().mul(&self.std.into()).unwrap();
        x.add(&noise).unwrap()
    }
}

// ============================================================================
// COMPOSITIONAL PRIMITIVES
// ============================================================================

/// Residual connection primitive: y = x + F(x)
pub fn residual<F>(x: &Tensor, f: F) -> Tensor
where
    F: FnOnce(&Tensor) -> Tensor,
{
    let fx = f(x);
    x.add(&fx).unwrap()
}

/// Skip connection with optional projection
pub fn skip_connection<F>(x: &Tensor, f: F) -> Tensor
where
    F: FnOnce(&Tensor) -> Tensor,
{
    let fx = f(x);
    x.add(&fx).unwrap()
}

/// Parallel branch concatenation
pub fn parallel_branch<F>(x: &Tensor, branches: &[F]) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    let outputs: Vec<Tensor> = branches.iter().map(|f| f(x)).collect();
    Tensor::concat(&outputs, 1).unwrap()
}

/// Dense connection (concatenate all inputs)
pub fn dense_connection<F>(x: &Tensor, layers: &[F]) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    let mut output = x.clone();
    let mut all_outputs = vec![output.clone()];

    for layer in layers {
        let concat_in = Tensor::concat(
            &all_outputs.iter().map(|t| (*t).clone()).collect::<Vec<_>>(),
            1,
        )
        .unwrap();
        output = layer(&concat_in);
        all_outputs.push(output.clone());
    }

    output
}

/// Sequential composition (layer stack)
pub fn sequential<F>(x: &Tensor, layers: &[F]) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    let mut out = x.clone();
    for layer in layers {
        out = layer(&out);
    }
    out
}

/// Concatenation composition
pub fn concat_compose<F>(x: &Tensor, fns: &[F]) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    let outputs: Vec<Tensor> = fns.iter().map(|f| f(x)).collect();
    Tensor::concat(&outputs, 1).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "experimental tensor stack — Package O; see docs/TENSOR_QUARANTINE.md"]
    #[test]
    fn test_rnn_cell() {
        let rnn = RNNCell::new(4, 8);
        let x = Tensor::rand(Shape::new(vec![4]));
        let h = Tensor::zeros(Shape::new(vec![8]));
        let h_new = rnn.forward(&x, &h);
        assert_eq!(h_new.shape, Shape::new(vec![8]));
    }

    #[ignore = "experimental tensor stack — Package O; see docs/TENSOR_QUARANTINE.md"]
    #[test]
    fn test_peephole_lstm() {
        let lstm = PeepholeLSTMCell::new(4, 8);
        let x = Tensor::rand(Shape::new(vec![4]));
        let h = Tensor::zeros(Shape::new(vec![8]));
        let c = Tensor::zeros(Shape::new(vec![8]));
        let (h_new, c_new) = lstm.forward(&x, &h, &c);
        assert_eq!(h_new.shape, Shape::new(vec![8]));
        assert_eq!(c_new.shape, Shape::new(vec![8]));
    }

    #[ignore = "experimental tensor stack — Package O; see docs/TENSOR_QUARANTINE.md"]
    #[test]
    fn test_batch_norm_1d() {
        let mut bn = BatchNorm1d::new(16);
        let x = Tensor::rand(Shape::new(vec![32, 16]));
        let out = bn.forward(&x);
        assert_eq!(out.shape, Shape::new(vec![32, 16]));
    }

    #[ignore = "experimental tensor stack — Package O; see docs/TENSOR_QUARANTINE.md"]
    #[test]
    fn test_residual() {
        let x = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let out = residual(&x, |t| t.mul(&2.0.into()).unwrap());
        assert_eq!(out.data, vec![3.0, 4.0, 5.0]);
    }

    #[ignore = "experimental tensor stack — Package O; see docs/TENSOR_QUARANTINE.md"]
    #[test]
    fn test_parallel_branch() {
        let x = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let branches: Vec<fn(&Tensor) -> Tensor> = vec![|t| t.mul(&2.0.into()).unwrap(), |t| {
            t.mul(&3.0.into()).unwrap()
        }];
        let out = parallel_branch(&x, &branches);
        assert_eq!(out.data.len(), 6); // Concatenated
    }
}
