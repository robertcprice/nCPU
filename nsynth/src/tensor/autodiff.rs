//! Automatic Differentiation for nCPU/nSynth
//!
//! Compute graph and backpropagation.

use super::ops::Tensor;
use std::collections::HashMap;

/// Unique identifier for compute graph nodes
pub type NodeId = usize;

/// Node in compute graph
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub value: Tensor,
    pub grad: Option<Tensor>,
    pub op: Option<Op>,
    pub inputs: Vec<NodeId>,
}

/// Operation in compute graph
#[derive(Debug, Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Conv2D {
        stride: (usize, usize),
        padding: (usize, usize),
    },
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Sum,
    Mean,
    Reshape {
        shape: Vec<usize>,
    },
    Transpose,
    None, // For input/leaf nodes
}

impl Op {
    /// Get gradient function for this operation
    pub fn grad_fn(&self) -> GradFn {
        match self {
            Op::Add => GradFn::Add,
            Op::Sub => GradFn::Sub,
            Op::Mul => GradFn::Mul,
            Op::Div => GradFn::Div,
            Op::MatMul => GradFn::MatMul,
            Op::Conv2D { stride, padding } => GradFn::Conv2D {
                stride: *stride,
                padding: *padding,
            },
            Op::Relu => GradFn::Relu,
            Op::Sigmoid => GradFn::Sigmoid,
            Op::Tanh => GradFn::Tanh,
            Op::Softmax => GradFn::Softmax,
            Op::Sum => GradFn::Sum,
            Op::Mean => GradFn::Mean,
            Op::Reshape { .. } => GradFn::Reshape,
            Op::Transpose => GradFn::Transpose,
            Op::None => GradFn::None,
        }
    }
}

/// Gradient computation function
#[derive(Debug, Clone)]
pub enum GradFn {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Conv2D {
        stride: (usize, usize),
        padding: (usize, usize),
    },
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Sum,
    Mean,
    Reshape,
    Transpose,
    None,
}

/// Compute graph for automatic differentiation
#[derive(Debug)]
pub struct ComputeGraph {
    nodes: HashMap<NodeId, Node>,
    next_id: NodeId,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add node to graph
    pub fn add_node(&mut self, value: Tensor, op: Op, inputs: Vec<NodeId>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let node = Node {
            id,
            value,
            grad: None,
            op: Some(op),
            inputs,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add input/leaf node
    pub fn add_input(&mut self, value: Tensor) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let node = Node {
            id,
            value,
            grad: None,
            op: None,
            inputs: Vec::new(),
        };

        self.nodes.insert(id, node);
        id
    }

    /// Get node by id
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get mutable node by id
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Forward pass - compute output of node
    pub fn forward(&mut self, output_id: NodeId) -> Result<Tensor, String> {
        let node = self.get_node(output_id).ok_or("Node not found")?;

        // If this is a leaf node, just return its value
        if node.op.is_none() {
            return Ok(node.value.clone());
        }

        // Clone needed data to release borrow
        let inputs = node.inputs.clone();
        let op = node.op.clone();

        // Recursively compute inputs
        let mut input_values = Vec::new();
        for &input_id in &inputs {
            input_values.push(self.forward(input_id)?);
        }

        // Apply operation
        let result = match op.as_ref().unwrap() {
            Op::Add => input_values[0].add(&input_values[1])?,
            Op::Sub => input_values[0].sub(&input_values[1])?,
            Op::Mul => input_values[0].mul(&input_values[1])?,
            Op::Div => input_values[0].div(&input_values[1])?,
            Op::MatMul => input_values[0].matmul(&input_values[1])?,
            Op::Conv2D { stride, padding } => {
                input_values[0].conv2d(&input_values[1], *stride, *padding)?
            }
            Op::Relu => input_values[0].relu(),
            Op::Sigmoid => input_values[0].sigmoid(),
            Op::Tanh => input_values[0].tanh(),
            Op::Softmax => input_values[0].softmax(),
            Op::Sum => Tensor::scalar(input_values[0].sum()),
            Op::Mean => Tensor::scalar(input_values[0].mean()),
            Op::Reshape { shape } => {
                input_values[0].reshape(super::ops::Shape::new(shape.clone()))?
            }
            Op::Transpose => input_values[0].transpose()?,
            Op::None => input_values[0].clone(),
        };

        // Update node value
        if let Some(node) = self.get_node_mut(output_id) {
            node.value = result.clone();
        }

        Ok(result)
    }

    /// Backward pass - compute gradients
    pub fn backward(&mut self, output_id: NodeId) -> Result<(), String> {
        // Initialize output gradient to 1.0
        let output_shape = self
            .get_node(output_id)
            .ok_or("Output node not found")?
            .value
            .shape
            .clone();

        if let Some(node) = self.get_node_mut(output_id) {
            node.grad = Some(Tensor::ones(output_shape));
        }

        // Reverse topological order
        let mut sorted = self.topological_sort(output_id)?;
        sorted.reverse();

        // Process each node
        for &node_id in &sorted {
            self.compute_grad(node_id)?;
        }

        Ok(())
    }

    /// Compute gradient for a node
    fn compute_grad(&mut self, node_id: NodeId) -> Result<(), String> {
        let node = self.get_node(node_id).ok_or("Node not found")?;

        if node.op.is_none() || node.grad.is_none() {
            return Ok(()); // Leaf node or no gradient
        }

        let grad = node.grad.as_ref().unwrap().clone();
        let grad_fn = node.op.as_ref().unwrap().grad_fn();

        // Get input values and gradients
        let mut input_values = Vec::new();
        let mut input_grads = Vec::new();

        for &input_id in &node.inputs {
            let input_node = self.get_node(input_id).ok_or("Input node not found")?;
            input_values.push(input_node.value.clone());
            input_grads.push(input_node.grad.clone());
        }

        // Compute gradients based on operation
        let grads = match grad_fn {
            GradFn::Add => vec![
                Some(grad.clone()), // d(a+b)/da = 1
                Some(grad.clone()), // d(a+b)/db = 1
            ],
            GradFn::Sub => vec![
                Some(grad.clone()), // d(a-b)/da = 1
                Some(
                    Tensor::zeros(grad.shape)
                        .mul(&Tensor::scalar(-1.0))
                        .unwrap(),
                ), // d(a-b)/db = -1
            ],
            GradFn::Mul => {
                // d(a*b)/da = b, d(a*b)/db = a
                vec![
                    Some(grad.mul(&input_values[1])?),
                    Some(grad.mul(&input_values[0])?),
                ]
            }
            GradFn::Div => {
                // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
                let b = &input_values[1];
                let a = &input_values[0];
                vec![
                    Some(grad.div(b)?),
                    Some(grad.mul(&a)?.div(b)?.div(b)?.mul(&Tensor::scalar(-1.0))?),
                ]
            }
            GradFn::MatMul => {
                // For matrix multiplication C = A @ B
                // dL/dA = dL/dC @ B.T
                // dL/dB = A.T @ dL/dC
                let b_t = input_values[1].transpose()?;
                let a_t = input_values[0].transpose()?;
                vec![Some(grad.matmul(&b_t)?), Some(a_t.matmul(&grad)?)]
            }
            GradFn::Relu => {
                // d(relu)/dx = 1 if x > 0 else 0
                let mask = input_values[0].relu(); // Simple mask (not exactly derivative)
                vec![Some(grad.mul(&mask)?)]
            }
            GradFn::Sigmoid => {
                // d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
                let sigmoid = input_values[0].sigmoid();
                let ones = Tensor::ones(sigmoid.shape.clone());
                let derivative = sigmoid.mul(&(ones.sub(&sigmoid.clone())?))?;
                vec![Some(grad.mul(&derivative)?)]
            }
            GradFn::Tanh => {
                // d(tanh)/dx = 1 - tanh^2(x)
                let tanh = input_values[0].tanh();
                let ones = Tensor::ones(tanh.shape.clone());
                let derivative = ones.sub(&tanh.clone().mul(&tanh)?)?;
                vec![Some(grad.mul(&derivative)?)]
            }
            GradFn::Conv2D { .. } => {
                // Simplified - would need full implementation
                vec![Some(grad.clone()), Some(grad.clone())]
            }
            GradFn::Softmax => {
                // Simplified - softmax derivative is complex
                vec![Some(grad.clone())]
            }
            GradFn::Sum => {
                // Broadcast gradient back to input shape
                let broadcast_grad = grad.clone();
                // Reshape to match input
                vec![Some(broadcast_grad)]
            }
            GradFn::Mean => {
                // d(mean)/dx = 1/n
                let n = input_values[0].data.len() as f64;
                let scaled_grad = grad.mul(&Tensor::scalar(1.0 / n))?;
                vec![Some(scaled_grad)]
            }
            GradFn::Reshape { .. } => {
                // Gradient just gets reshaped back
                vec![Some(grad.clone())]
            }
            GradFn::Transpose => {
                // Gradient of transpose is transpose of gradient
                vec![Some(grad.transpose()?)]
            }
            GradFn::None => vec![Some(grad.clone())],
        };

        // Accumulate gradients to input nodes
        let inputs = node.inputs.clone();
        for (i, input_id) in inputs.iter().enumerate() {
            if i < grads.len() {
                if let Some(grad) = &grads[i] {
                    if let Some(input_node) = self.get_node_mut(*input_id) {
                        match &input_node.grad {
                            Some(existing_grad) => {
                                // Accumulate gradient
                                input_node.grad = Some(existing_grad.add(grad).unwrap());
                            }
                            None => {
                                input_node.grad = Some(grad.clone());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Topological sort of compute graph
    fn topological_sort(&self, output_id: NodeId) -> Result<Vec<NodeId>, String> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        self.dfs(output_id, &mut visited, &mut result)?;
        Ok(result)
    }

    /// Depth-first search for topological sort
    fn dfs(
        &self,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) -> Result<(), String> {
        if visited.contains(&node_id) {
            return Ok(());
        }

        visited.insert(node_id);

        let node = self.get_node(node_id).ok_or("Node not found")?;
        for &input_id in &node.inputs {
            self.dfs(input_id, visited, result)?;
        }

        result.push(node_id);
        Ok(())
    }

    /// Clear all gradients
    pub fn zero_grad(&mut self) {
        for node in self.nodes.values_mut() {
            node.grad = None;
        }
    }

    /// Get gradient for a node
    pub fn grad(&self, node_id: NodeId) -> Option<Tensor> {
        self.nodes.get(&node_id)?.grad.clone()
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::ops::Tensor;
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_input(Tensor::scalar(2.0));
        let b = graph.add_input(Tensor::scalar(3.0));
        let c = graph.add_node(Tensor::scalar(0.0), Op::Add, vec![a, b]);

        let result = graph.forward(c).unwrap();
        assert_eq!(result.data[0], 5.0);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_input(Tensor::scalar(1.0));
        let b = graph.add_input(Tensor::scalar(2.0));
        let c = graph.add_node(Tensor::scalar(0.0), Op::Add, vec![a, b]);

        let sorted = graph.topological_sort(c).unwrap();
        assert_eq!(sorted.len(), 3);
        assert!(sorted.contains(&c));
    }

    #[test]
    fn test_backward_add() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_input(Tensor::scalar(2.0));
        let b = graph.add_input(Tensor::scalar(3.0));
        let c = graph.add_node(Tensor::scalar(0.0), Op::Add, vec![a, b]);

        graph.forward(c).unwrap();
        graph.backward(c).unwrap();

        let grad_a = graph.grad(a).unwrap();
        let grad_b = graph.grad(b).unwrap();

        assert_eq!(grad_a.data[0], 1.0);
        assert_eq!(grad_b.data[0], 1.0);
    }

    #[test]
    fn test_backward_mul() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_input(Tensor::scalar(3.0));
        let b = graph.add_input(Tensor::scalar(4.0));
        let c = graph.add_node(Tensor::scalar(0.0), Op::Mul, vec![a, b]);

        graph.forward(c).unwrap();
        graph.backward(c).unwrap();

        let grad_a = graph.grad(a).unwrap();
        let grad_b = graph.grad(b).unwrap();

        // d(a*b)/da = b = 4, d(a*b)/db = a = 3
        assert_eq!(grad_a.data[0], 4.0);
        assert_eq!(grad_b.data[0], 3.0);
    }
}
