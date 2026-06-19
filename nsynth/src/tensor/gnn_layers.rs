//! Graph Neural Network Layers for nCPU/nSynth
//!
//! GCN, GAT, and message passing primitives.

use super::ops::Shape;
use super::ops::Tensor;

// ============================================================================
// GRAPH CONVOLUTION (GCN)
// ============================================================================

/// Graph Convolutional Network Layer
/// H_out = D^-0.5 * A * D^-0.5 * H * W
pub struct GCNLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl GCNLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let scale = (2.0 / (in_features + out_features) as f64).sqrt();
        Self {
            in_features,
            out_features,
            weight: Tensor::uniform(Shape::new(vec![in_features, out_features]), -scale, scale),
            bias: Tensor::zeros(Shape::new(vec![out_features])),
        }
    }

    pub fn forward(&self, node_features: &Tensor, adjacency: &Tensor) -> Tensor {
        // Compute symmetric normalization: D^-0.5 * A * D^-0.5
        let degree = adjacency.sum_dim(&[1]).unwrap();
        let neg_half = Tensor::scalar(-0.5);
        let d_inv_sqrt = degree.pow(&neg_half).unwrap();
        let d_inv_sqrt_diag = d_inv_sqrt.diag();

        // A_norm = D^-0.5 * A * D^-0.5
        let a_norm = d_inv_sqrt_diag
            .matmul(adjacency)
            .unwrap()
            .matmul(&d_inv_sqrt_diag)
            .unwrap();

        // H_out = A_norm * H * W + b
        let h_out = a_norm
            .matmul(node_features)
            .unwrap()
            .matmul(&self.weight)
            .unwrap();
        h_out.add(&self.bias).unwrap()
    }
}

// ============================================================================
// GRAPH ATTENTION (GAT)
// ============================================================================

/// Graph Attention Network Layer
pub struct GATLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub num_heads: usize,
    pub weight: Tensor,
    pub attention: Tensor,
}

impl GATLayer {
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let scale = (2.0 / in_features as f64).sqrt();
        Self {
            in_features,
            out_features,
            num_heads,
            weight: Tensor::uniform(
                Shape::new(vec![in_features, out_features * num_heads]),
                -scale,
                scale,
            ),
            attention: Tensor::uniform(Shape::new(vec![2 * out_features, 1]), -scale, scale),
        }
    }

    pub fn forward(&self, node_features: &Tensor, adjacency: &Tensor) -> Tensor {
        let h = node_features.matmul(&self.weight).unwrap();
        let n = node_features.shape.dims[0];

        // Compute attention coefficients
        let mut attention_scores = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                // Concatenate h_i and h_j, then apply attention
                let concatenated =
                    Tensor::concat(&[h.index(&[i..=i]), h.index(&[j..=j])], 0).unwrap();
                let e = concatenated
                    .matmul(&self.attention)
                    .unwrap()
                    .leaky_relu(0.01)
                    .data[0];
                attention_scores.push(e);
            }
        }

        // Apply softmax and adjacency mask
        let mut attn_matrix = vec![0.0; n * n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if adjacency.data[i * n + j] > 0.0 {
                    sum += attention_scores[i * n + j].exp();
                }
            }
            for j in 0..n {
                if adjacency.data[i * n + j] > 0.0 && sum > 0.0 {
                    attn_matrix[i * n + j] = attention_scores[i * n + j].exp() / sum;
                }
            }
        }

        // Apply attention to features
        let mut output = vec![0.0; self.out_features * self.num_heads];
        for i in 0..self.out_features * self.num_heads {
            for j in 0..n {
                output[i] +=
                    attn_matrix[j * n + i] * h.data[j * (self.out_features * self.num_heads) + i];
            }
        }

        Tensor::new(
            output,
            Shape::new(vec![n, self.out_features * self.num_heads]),
        )
    }
}

// ============================================================================
// MESSAGE PASSING
// ============================================================================

/// Generic Message Passing primitive
pub struct MessagePassing<F, G>
where
    F: Fn(&Tensor, &Tensor) -> Tensor + Clone,
    G: Fn(&Tensor, &Tensor) -> Tensor + Clone,
{
    pub message_fn: F,
    pub aggregate_fn: G,
    pub update_fn: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
}

impl<F, G> MessagePassing<F, G>
where
    F: Fn(&Tensor, &Tensor) -> Tensor + Clone + 'static,
    G: Fn(&Tensor, &Tensor) -> Tensor + Clone + 'static,
{
    pub fn new(
        message_fn: F,
        aggregate_fn: G,
        update_fn: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
    ) -> Self {
        Self {
            message_fn,
            aggregate_fn,
            update_fn,
        }
    }

    pub fn forward(&self, node_features: &Tensor, edge_index: &Tensor) -> Tensor {
        let n = node_features.shape.dims[0];
        let mut messages = Vec::new();

        // Compute messages for each edge
        for e in 0..edge_index.shape.dims[1] {
            let src = edge_index.data[2 * e] as usize;
            let dst = edge_index.data[2 * e + 1] as usize;
            let src_feat = node_features.index(&[src..=src]);
            let dst_feat = node_features.index(&[dst..=dst]);
            let msg = (self.message_fn)(&src_feat, &dst_feat);
            messages.push((dst, msg));
        }

        // Aggregate messages per node
        let mut aggregated = vec![0.0; node_features.data.len()];
        for i in 0..n {
            let node_msgs: Vec<Tensor> = messages
                .iter()
                .filter(|(dst, _)| *dst == i)
                .map(|(_, msg)| msg.clone())
                .collect();

            if !node_msgs.is_empty() {
                let agg =
                    (self.aggregate_fn)(&Tensor::stack(&node_msgs, 0).unwrap(), node_features);
                for (j, &v) in agg.data.iter().enumerate() {
                    aggregated[i * node_features.shape.dims[1] + j] = v;
                }
            }
        }

        let agg_tensor = Tensor::new(aggregated, node_features.shape.clone());
        (self.update_fn)(&agg_tensor, node_features)
    }
}

/// Edge feature handling
pub struct EdgeFeatureConv {
    pub node_to_edge: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
    pub edge_to_node: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
}

impl EdgeFeatureConv {
    pub fn new(
        node_to_edge: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
        edge_to_node: Box<dyn Fn(&Tensor, &Tensor) -> Tensor>,
    ) -> Self {
        Self {
            node_to_edge,
            edge_to_node,
        }
    }

    pub fn forward(
        &self,
        node_features: &Tensor,
        edge_features: &Tensor,
        edge_index: &Tensor,
    ) -> Tensor {
        // node -> edge messages
        let mut edge_msgs = Vec::new();
        for e in 0..edge_index.shape.dims[1] {
            let src = edge_index.data[2 * e] as usize;
            let dst = edge_index.data[2 * e + 1] as usize;
            let src_feat = node_features.index(&[src..=src]);
            let dst_feat = node_features.index(&[dst..=dst]);
            let edge_feat = edge_features.index(&[e..=e]);
            let msg = (self.node_to_edge)(
                &Tensor::concat(&[src_feat.clone(), dst_feat.clone(), edge_feat.clone()], 0)
                    .unwrap(),
                &edge_feat,
            );
            edge_msgs.push(msg);
        }

        // edge -> node aggregation
        let mut node_out = vec![0.0; node_features.data.len()];
        for e in 0..edge_index.shape.dims[1] {
            let dst = edge_index.data[2 * e + 1] as usize;
            let msg = &edge_msgs[e];
            for (j, &v) in msg.data.iter().enumerate() {
                node_out[dst * node_features.shape.dims[1] + j] += v;
            }
        }

        Tensor::new(node_out, node_features.shape.clone())
    }
}

/// Graph pooling (readout)
pub struct GraphPool {
    pub pool_type: PoolType,
}

pub enum PoolType {
    Mean,
    Max,
    Sum,
    Attention,
}

impl GraphPool {
    pub fn new(pool_type: PoolType) -> Self {
        Self { pool_type }
    }

    pub fn forward(&self, node_features: &Tensor, batch: &Tensor) -> Tensor {
        let num_graphs = batch.data.iter().fold(0.0_f64, |a, &b| a.max(b)) as usize + 1;

        match self.pool_type {
            PoolType::Mean => self.mean_pool(node_features, batch, num_graphs),
            PoolType::Max => self.max_pool(node_features, batch, num_graphs),
            PoolType::Sum => self.sum_pool(node_features, batch, num_graphs),
            PoolType::Attention => self.attention_pool(node_features, batch, num_graphs),
        }
    }

    fn mean_pool(&self, node_features: &Tensor, batch: &Tensor, num_graphs: usize) -> Tensor {
        let mut pooled = vec![0.0; num_graphs * node_features.shape.dims[1]];
        let mut counts = vec![0; num_graphs];

        for (i, &b) in batch.data.iter().enumerate() {
            let graph_idx = b as usize;
            counts[graph_idx] += 1;
            for j in 0..node_features.shape.dims[1] {
                pooled[graph_idx * node_features.shape.dims[1] + j] +=
                    node_features.data[i * node_features.shape.dims[1] + j];
            }
        }

        for i in 0..num_graphs {
            let count = counts[i].max(1) as f64;
            for j in 0..node_features.shape.dims[1] {
                pooled[i * node_features.shape.dims[1] + j] /= count;
            }
        }

        Tensor::new(
            pooled,
            Shape::new(vec![num_graphs, node_features.shape.dims[1]]),
        )
    }

    fn max_pool(&self, node_features: &Tensor, batch: &Tensor, num_graphs: usize) -> Tensor {
        let mut pooled = vec![f64::NEG_INFINITY; num_graphs * node_features.shape.dims[1]];

        for (i, &b) in batch.data.iter().enumerate() {
            let graph_idx = b as usize;
            for j in 0..node_features.shape.dims[1] {
                let val = node_features.data[i * node_features.shape.dims[1] + j];
                pooled[graph_idx * node_features.shape.dims[1] + j] =
                    pooled[graph_idx * node_features.shape.dims[1] + j].max(val);
            }
        }

        Tensor::new(
            pooled,
            Shape::new(vec![num_graphs, node_features.shape.dims[1]]),
        )
    }

    fn sum_pool(&self, node_features: &Tensor, batch: &Tensor, num_graphs: usize) -> Tensor {
        let mut pooled = vec![0.0; num_graphs * node_features.shape.dims[1]];

        for (i, &b) in batch.data.iter().enumerate() {
            let graph_idx = b as usize;
            for j in 0..node_features.shape.dims[1] {
                pooled[graph_idx * node_features.shape.dims[1] + j] +=
                    node_features.data[i * node_features.shape.dims[1] + j];
            }
        }

        Tensor::new(
            pooled,
            Shape::new(vec![num_graphs, node_features.shape.dims[1]]),
        )
    }

    fn attention_pool(&self, node_features: &Tensor, batch: &Tensor, num_graphs: usize) -> Tensor {
        // Simplified attention pooling
        let attention_gate = node_features.mean_dim(&[1]).unwrap();
        let weights = attention_gate.softmax();

        let mut pooled = vec![0.0; num_graphs * node_features.shape.dims[1]];
        for (i, &b) in batch.data.iter().enumerate() {
            let graph_idx = b as usize;
            let w = weights.data[i];
            for j in 0..node_features.shape.dims[1] {
                pooled[graph_idx * node_features.shape.dims[1] + j] +=
                    w * node_features.data[i * node_features.shape.dims[1] + j];
            }
        }

        Tensor::new(
            pooled,
            Shape::new(vec![num_graphs, node_features.shape.dims[1]]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcn_layer() {
        let gcn = GCNLayer::new(4, 8);
        let features = Tensor::uniform(Shape::new(vec![3, 4]), -1.0, 1.0);
        let adj = Tensor::vector(vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .reshape(Shape::new(vec![3, 3]))
            .unwrap();
        let out = gcn.forward(&features, &adj);
        assert_eq!(out.shape, Shape::new(vec![3, 8]));
    }

    #[test]
    fn test_graph_pool() {
        let pool = GraphPool::new(PoolType::Mean);
        let features = Tensor::uniform(Shape::new(vec![4, 3]), -1.0, 1.0);
        let batch = Tensor::vector(vec![0.0, 0.0, 1.0, 1.0]);
        let out = pool.forward(&features, &batch);
        assert_eq!(out.shape, Shape::new(vec![2, 3]));
    }
}
