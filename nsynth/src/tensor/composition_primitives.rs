//! Composable Architecture Primitives for nCPU/nSynth
//!
//! Building blocks that combine into any architecture.
//! No hardcoded architectures — only composition patterns.

use super::ops::Shape;
use super::ops::Tensor;
use std::f64::consts::PI;

// ============================================================================
// COMPOSITIONAL PATTERNS
// ============================================================================

/// Residual Block: F(x) + x with optional projection
pub struct ResidualBlock<F>
where
    F: Fn(&Tensor) -> Tensor + Clone,
{
    pub forward_fn: F,
    pub projection: Option<Box<dyn Fn(&Tensor) -> Tensor>>,
    pub activation: Option<Box<dyn Fn(&Tensor) -> Tensor>>,
}

impl<F> ResidualBlock<F>
where
    F: Fn(&Tensor) -> Tensor + Clone + 'static,
{
    pub fn new(forward_fn: F) -> Self {
        Self {
            forward_fn,
            projection: None,
            activation: None,
        }
    }

    pub fn with_projection(mut self, proj: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.projection = Some(proj);
        self
    }

    pub fn with_activation(mut self, act: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.activation = Some(act);
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let fx = (self.forward_fn)(x);
        let shortcut = if let Some(ref proj) = self.projection {
            proj(x)
        } else {
            x.clone()
        };

        let out = shortcut.add(&fx).unwrap();

        if let Some(ref act) = self.activation {
            act(&out)
        } else {
            out
        }
    }
}

/// Skip Connection with multiple layers
pub struct SkipConnection {
    pub layers: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    pub combine_fn: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
}

impl SkipConnection {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            combine_fn: Box::new(|a, b| a.add(b).unwrap()),
        }
    }

    pub fn add_layer(mut self, layer: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn with_combine(mut self, f: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>) -> Self {
        self.combine_fn = f;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for layer in &self.layers {
            let layer_out = layer(x);
            out = (self.combine_fn)(&out, &layer_out);
        }
        out
    }
}

/// Parallel Branch Concatenation
pub struct ParallelBranch {
    pub branches: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
}

impl ParallelBranch {
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
        }
    }

    pub fn add_branch(mut self, branch: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.branches.push(branch);
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let outputs: Vec<Tensor> = self.branches.iter().map(|b| b(x)).collect();
        Tensor::concat(&outputs, 1).unwrap_or_else(|_| x.clone())
    }
}

/// Dense Connection (concatenate all previous outputs)
pub struct DenseBlock {
    pub layers: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    pub growth_rate: usize,
}

impl DenseBlock {
    pub fn new(growth_rate: usize) -> Self {
        Self {
            layers: Vec::new(),
            growth_rate,
        }
    }

    pub fn add_layer(mut self, layer: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut features = vec![x.clone()];
        let mut out = x.clone();

        for layer in &self.layers {
            let concat = Tensor::concat(&features, 1).unwrap();
            out = layer(&concat);
            features.push(out.clone());
        }

        out
    }
}

/// Inception-style parallel multi-scale branches
pub struct InceptionBlock {
    pub branches: Vec<(usize, Box<dyn Fn(&Tensor) -> Tensor>)>, // (kernel_size, branch)
}

impl InceptionBlock {
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
        }
    }

    pub fn add_branch(
        mut self,
        kernel_size: usize,
        branch: Box<dyn Fn(&Tensor) -> Tensor>,
    ) -> Self {
        self.branches.push((kernel_size, branch));
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let outputs: Vec<Tensor> = self.branches.iter().map(|(_, branch)| branch(x)).collect();

        Tensor::concat(&outputs, 1).unwrap()
    }
}

/// Encoder-Decoder with optional skip connections
pub struct EncoderDecoder {
    pub encoder: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    pub decoder: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    pub use_skip_connections: bool,
    pub bottleneck: Option<Box<dyn Fn(&Tensor) -> Tensor>>,
}

impl EncoderDecoder {
    pub fn new() -> Self {
        Self {
            encoder: Vec::new(),
            decoder: Vec::new(),
            use_skip_connections: true,
            bottleneck: None,
        }
    }

    pub fn add_encoder(mut self, layer: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.encoder.push(layer);
        self
    }

    pub fn add_decoder(mut self, layer: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.decoder.push(layer);
        self
    }

    pub fn with_bottleneck(mut self, bottleneck: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.bottleneck = Some(bottleneck);
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut encoder_outputs = Vec::new();
        let mut out = x.clone();

        // Encoder pass
        for layer in &self.encoder {
            out = layer(&out);
            if self.use_skip_connections {
                encoder_outputs.push(out.clone());
            }
        }

        // Bottleneck
        if let Some(ref bottleneck) = self.bottleneck {
            out = bottleneck(&out);
        }

        // Decoder pass with skip connections
        if self.use_skip_connections {
            for (i, layer) in self.decoder.iter().enumerate() {
                if let Some(skip) = encoder_outputs.get(encoder_outputs.len() - 1 - i) {
                    let concat = Tensor::concat(&[out.clone(), skip.clone()], 1).unwrap();
                    out = layer(&concat);
                } else {
                    out = layer(&out);
                }
            }
        } else {
            for layer in &self.decoder {
                out = layer(&out);
            }
        }

        out
    }
}

/// Sequence-to-Sequence with Attention Bridge
pub struct Seq2Seq {
    pub encoder_layers: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    pub decoder_layers: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    pub attention: Option<Box<dyn Fn(&Tensor, &Tensor) -> Tensor>>,
}

impl Seq2Seq {
    pub fn new() -> Self {
        Self {
            encoder_layers: Vec::new(),
            decoder_layers: Vec::new(),
            attention: None,
        }
    }

    pub fn add_encoder(mut self, layer: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.encoder_layers.push(layer);
        self
    }

    pub fn add_decoder(mut self, layer: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.decoder_layers.push(layer);
        self
    }

    pub fn with_attention(mut self, attn: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>) -> Self {
        self.attention = Some(attn);
        self
    }

    pub fn forward(&self, src: &Tensor, tgt: &Tensor) -> Tensor {
        // Encode source sequence
        let mut encoded = src.clone();
        for layer in &self.encoder_layers {
            encoded = layer(&encoded);
        }

        // Decode with attention
        let mut decoded = tgt.clone();
        for layer in &self.decoder_layers {
            if let Some(ref attn) = self.attention {
                let context = attn(&decoded, &encoded);
                let combined = Tensor::concat(&[decoded.clone(), context], 1).unwrap();
                decoded = layer(&combined);
            } else {
                decoded = layer(&decoded);
            }
        }

        decoded
    }
}

/// BERT-style Encoder with masking
pub struct TransformerEncoder {
    pub self_attention: Box<dyn Fn(&Tensor, Option<&Tensor>) -> Tensor>,
    pub feed_forward: Box<dyn Fn(&Tensor) -> Tensor>,
    pub layer_norm: Box<dyn Fn(&Tensor) -> Tensor>,
}

impl TransformerEncoder {
    pub fn new(
        sa: Box<dyn Fn(&Tensor, Option<&Tensor>) -> Tensor>,
        ff: Box<dyn Fn(&Tensor) -> Tensor>,
        ln: Box<dyn Fn(&Tensor) -> Tensor>,
    ) -> Self {
        Self {
            self_attention: sa,
            feed_forward: ff,
            layer_norm: ln,
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Self-attention with residual
        let attn_out = (self.self_attention)(x, mask);
        let attn_residual = x.add(&attn_out).unwrap();
        let attn_norm = (self.layer_norm)(&attn_residual);

        // Feed-forward with residual
        let ff_out = (self.feed_forward)(&attn_norm);
        let ff_residual = attn_norm.add(&ff_out).unwrap();
        (self.layer_norm)(&ff_residual)
    }
}

/// BERT-style Decoder with cross-attention
pub struct TransformerDecoder {
    pub self_attention: Box<dyn Fn(&Tensor, Option<&Tensor>) -> Tensor>,
    pub cross_attention: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
    pub feed_forward: Box<dyn Fn(&Tensor) -> Tensor>,
    pub layer_norm: Box<dyn Fn(&Tensor) -> Tensor>,
}

impl TransformerDecoder {
    pub fn new(
        sa: Box<dyn Fn(&Tensor, Option<&Tensor>) -> Tensor>,
        ca: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
        ff: Box<dyn Fn(&Tensor) -> Tensor>,
        ln: Box<dyn Fn(&Tensor) -> Tensor>,
    ) -> Self {
        Self {
            self_attention: sa,
            cross_attention: ca,
            feed_forward: ff,
            layer_norm: ln,
        }
    }

    pub fn forward(&self, x: &Tensor, encoder_out: &Tensor, self_mask: Option<&Tensor>) -> Tensor {
        // Self-attention with residual
        let self_attn_out = (self.self_attention)(x, self_mask);
        let self_attn_residual = x.add(&self_attn_out).unwrap();
        let self_attn_norm = (self.layer_norm)(&self_attn_residual);

        // Cross-attention with residual
        let cross_attn_out = (self.cross_attention)(&self_attn_norm, encoder_out);
        let cross_attn_residual = self_attn_norm.add(&cross_attn_out).unwrap();
        let cross_attn_norm = (self.layer_norm)(&cross_attn_residual);

        // Feed-forward with residual
        let ff_out = (self.feed_forward)(&cross_attn_norm);
        let ff_residual = cross_attn_norm.add(&ff_out).unwrap();
        (self.layer_norm)(&ff_residual)
    }
}

/// Attention Bridge for Seq2Seq
pub struct AttentionBridge {
    pub query_transform: Box<dyn Fn(&Tensor) -> Tensor>,
    pub key_transform: Box<dyn Fn(&Tensor) -> Tensor>,
    pub value_transform: Box<dyn Fn(&Tensor) -> Tensor>,
    pub scale: f64,
}

impl AttentionBridge {
    pub fn new(
        q: Box<dyn Fn(&Tensor) -> Tensor>,
        k: Box<dyn Fn(&Tensor) -> Tensor>,
        v: Box<dyn Fn(&Tensor) -> Tensor>,
    ) -> Self {
        Self {
            query_transform: q,
            key_transform: k,
            value_transform: v,
            scale: 1.0,
        }
    }

    pub fn forward(&self, query: &Tensor, keys_values: &Tensor) -> Tensor {
        let q = (self.query_transform)(query);
        let k = (self.key_transform)(keys_values);
        let v = (self.value_transform)(keys_values);

        let scores = q
            .matmul(&k.transpose().unwrap())
            .unwrap()
            .mul(&self.scale.into())
            .unwrap();
        let attn_weights = scores.softmax();
        attn_weights.matmul(&v).unwrap()
    }
}

/// Multi-Head Composition
pub struct MultiHeadComposition {
    pub heads: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
    pub combine_fn: Box<dyn Fn(&Vec<Tensor>) -> Tensor>,
}

impl MultiHeadComposition {
    pub fn new() -> Self {
        Self {
            heads: Vec::new(),
            combine_fn: Box::new(|heads| {
                if heads.is_empty() {
                    return Tensor::scalar(0.0);
                }
                // Concatenate all heads
                Tensor::concat(&heads, 1).unwrap()
            }),
        }
    }

    pub fn add_head(mut self, head: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.heads.push(head);
        self
    }

    pub fn with_combine(mut self, f: Box<dyn Fn(&Vec<Tensor>) -> Tensor>) -> Self {
        self.combine_fn = f;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let outputs: Vec<Tensor> = self.heads.iter().map(|h| h(x)).collect();
        (self.combine_fn)(&outputs)
    }
}

/// Gated Composition (learn to mix inputs)
pub struct GatedComposition {
    pub gate_fn: Box<dyn Fn(&Tensor) -> Tensor>,
    pub input_transforms: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
}

impl GatedComposition {
    pub fn new(gate_fn: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        Self {
            gate_fn,
            input_transforms: Vec::new(),
        }
    }

    pub fn add_input(mut self, transform: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.input_transforms.push(transform);
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = (self.gate_fn)(x);
        let transformed: Vec<Tensor> = self.input_transforms.iter().map(|t| t(x)).collect();

        if transformed.len() == 2 {
            // Mix two inputs
            let g1 = gate.data.get(0).copied().unwrap_or(0.5);
            let g2 = 1.0 - g1;
            transformed[0]
                .mul(&g1.into())
                .unwrap()
                .add(&transformed[1].mul(&g2.into()).unwrap())
                .unwrap()
        } else {
            x.clone()
        }
    }
}

// ============================================================================
// FUNCTIONAL HELPERS
// ============================================================================

/// Simple residual: x + f(x)
pub fn residual<F>(x: &Tensor, f: F) -> Tensor
where
    F: FnOnce(&Tensor) -> Tensor,
{
    let fx = f(x);
    x.add(&fx).unwrap()
}

/// Highway network: gated residual
pub fn highway<F>(x: &Tensor, f: F, gate: &Tensor) -> Tensor
where
    F: FnOnce(&Tensor) -> Tensor,
{
    let fx = f(x);
    let t = gate.sigmoid();
    let one_minus_t = t
        .mul(&(-1.0_f64).into())
        .unwrap()
        .add(&1.0_f64.into())
        .unwrap();
    t.mul(&fx)
        .unwrap()
        .add(&one_minus_t.mul(x).unwrap())
        .unwrap()
}

/// Dense block: concatenate all intermediate outputs
pub fn dense_block<F>(x: &Tensor, layers: &[F]) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    let mut outputs = vec![x.clone()];
    let mut out = x.clone();

    for layer in layers {
        let concat =
            Tensor::concat(&outputs.iter().map(|t| (*t).clone()).collect::<Vec<_>>(), 1).unwrap();
        out = layer(&concat);
        outputs.push(out.clone());
    }

    out
}

/// Parallel execution and concatenation
pub fn parallel<F>(x: &Tensor, branches: &[F]) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    let outputs: Vec<Tensor> = branches.iter().map(|b| b(x)).collect();
    Tensor::concat(&outputs, 1).unwrap()
}

/// Sequential composition
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual() {
        let x = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let out = residual(&x, |t| t.mul(&2.0.into()).unwrap());
        assert_eq!(out.data, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_parallel() {
        let x = Tensor::vector(vec![1.0, 2.0]);
        let branches: Vec<fn(&Tensor) -> Tensor> = vec![|t| t.mul(&2.0.into()).unwrap(), |t| {
            t.mul(&3.0.into()).unwrap()
        }];
        let out = parallel(&x, &branches);
        assert_eq!(out.data.len(), 4); // Concatenated
    }

    #[test]
    fn test_sequential() {
        let x = Tensor::vector(vec![1.0, 2.0]);
        let layers: Vec<fn(&Tensor) -> Tensor> = vec![|t| t.add(&1.0.into()).unwrap(), |t| {
            t.mul(&2.0.into()).unwrap()
        }];
        let out = sequential(&x, &layers);
        // (1+1)*2=4, (2+1)*2=6
        assert_eq!(out.data, vec![4.0, 6.0]);
    }
}
