//! Neural Network Models for nCPU/nSynth
//!
//! Model abstractions and common layer types.

use super::ops::{Shape, Tensor};

/// Model trait
pub trait Model {
    /// Forward pass through the model
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Get learnable parameters
    fn parameters(&self) -> Vec<Tensor>;

    /// Get parameter gradients
    fn gradients(&self) -> Vec<Option<Tensor>>;
}

/// Activation function
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    None,
}

impl Activation {
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Relu => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Tanh => x.tanh(),
            Activation::Softmax => x.softmax(),
            Activation::None => x.clone(),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Activation::Relu => "relu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::Softmax => "softmax",
            Activation::None => "none",
        }
    }
}

/// Linear (fully connected) layer
#[derive(Debug)]
pub struct Linear {
    /// Weight matrix
    pub weight: Tensor,
    /// Bias vector
    pub bias: Tensor,
    /// Input size
    pub in_features: usize,
    /// Output size
    pub out_features: usize,
}

impl Linear {
    /// Create new linear layer
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Tensor::randn(Shape::new(vec![out_features, in_features]));
        let bias = Tensor::randn(Shape::new(vec![out_features]));

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Create linear layer with custom weights
    pub fn from_weights(weight: Tensor, bias: Tensor) -> Self {
        let out_features = weight.shape.dims[0];
        let in_features = weight.shape.dims[1];

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = x @ W.T + b
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: (batch, in_features), weight: (out_features, in_features)
        // result: (batch, out_features)
        let output = x.matmul(&self.weight.transpose().unwrap()).unwrap();

        // Add bias (broadcast across batch)
        let batch_size = x.shape.dims[0];
        let bias_reshaped = self
            .bias
            .clone()
            .reshape(Shape::new(vec![1, self.out_features]))
            .unwrap();
        let bias_broadcast = broadcast_bias(&bias_reshaped, batch_size);

        output.add(&bias_broadcast).unwrap()
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Update parameters
    pub fn update(&mut self, weight: Tensor, bias: Tensor) {
        self.weight = weight;
        self.bias = bias;
    }
}

impl Model for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn gradients(&self) -> Vec<Option<Tensor>> {
        vec![
            if self.weight.requires_grad {
                self.weight
                    .grads
                    .as_ref()
                    .map(|g| Tensor::new(g.clone(), self.weight.shape.clone()))
            } else {
                None
            },
            if self.bias.requires_grad {
                self.bias
                    .grads
                    .as_ref()
                    .map(|g| Tensor::new(g.clone(), self.bias.shape.clone()))
            } else {
                None
            },
        ]
    }
}

/// 2D Convolution layer
#[derive(Debug)]
pub struct Conv2D {
    /// Kernel weights
    pub kernel: Tensor,
    /// Bias
    pub bias: Tensor,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Padding
    pub padding: (usize, usize),
}

impl Conv2D {
    /// Create new 2D convolution layer
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        let kernel = Tensor::randn(Shape::new(vec![
            out_channels,
            in_channels,
            kernel_size.0,
            kernel_size.1,
        ]));

        let bias = Tensor::randn(Shape::new(vec![out_channels]));

        Self {
            kernel,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
        }
    }

    /// Set stride
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Simplified - single channel case
        x.conv2d(&self.kernel, self.stride, self.padding).unwrap()
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.kernel.clone(), self.bias.clone()]
    }
}

impl Model for Conv2D {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn gradients(&self) -> Vec<Option<Tensor>> {
        vec![
            if self.kernel.requires_grad {
                self.kernel
                    .grads
                    .as_ref()
                    .map(|g| Tensor::new(g.clone(), self.kernel.shape.clone()))
            } else {
                None
            },
            if self.bias.requires_grad {
                self.bias
                    .grads
                    .as_ref()
                    .map(|g| Tensor::new(g.clone(), self.bias.shape.clone()))
            } else {
                None
            },
        ]
    }
}

/// Multi-Layer Perceptron (MLP)
#[derive(Debug)]
pub struct MLP {
    /// Layers
    pub layers: Vec<Linear>,
    /// Activations (one per layer except output)
    pub activations: Vec<Activation>,
    /// Input size
    pub input_size: usize,
    /// Hidden sizes
    pub hidden_sizes: Vec<usize>,
    /// Output size
    pub output_size: usize,
}

impl MLP {
    /// Create new MLP
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        let mut layers = Vec::new();
        let mut activations = Vec::new();

        let mut prev_size = input_size;
        for &hidden_size in &hidden_sizes {
            layers.push(Linear::new(prev_size, hidden_size));
            activations.push(Activation::Relu);
            prev_size = hidden_size;
        }

        // Output layer
        layers.push(Linear::new(prev_size, output_size));
        activations.push(Activation::None);

        Self {
            layers,
            activations,
            input_size,
            hidden_sizes,
            output_size,
        }
    }

    /// Create MLP with custom activations
    pub fn with_activations(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        activations: Vec<Activation>,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;

        for &hidden_size in &hidden_sizes {
            layers.push(Linear::new(prev_size, hidden_size));
            prev_size = hidden_size;
        }

        layers.push(Linear::new(prev_size, output_size));

        Self {
            layers,
            activations,
            input_size,
            hidden_sizes,
            output_size,
        }
    }

    /// Forward pass through all layers
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut current = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current);
            if i < self.activations.len() {
                current = self.activations[i].apply(&current);
            }
        }

        current
    }

    /// Get all parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    /// Update a specific layer
    pub fn update_layer(&mut self, index: usize, layer: Linear) {
        if index < self.layers.len() {
            self.layers[index] = layer;
        }
    }
}

impl Model for MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn gradients(&self) -> Vec<Option<Tensor>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.gradients())
            .collect()
    }
}

/// Simple CNN
#[derive(Debug)]
pub struct CNN {
    /// Convolutional layers
    pub conv_layers: Vec<Conv2D>,
    /// Activation after each conv
    pub activations: Vec<Activation>,
    /// Linear layers after flattening
    pub linear_layers: Vec<Linear>,
    /// Input channels
    pub input_channels: usize,
}

impl CNN {
    /// Create simple CNN
    pub fn new(input_channels: usize) -> Self {
        Self {
            conv_layers: Vec::new(),
            activations: Vec::new(),
            linear_layers: Vec::new(),
            input_channels,
        }
    }

    /// Add convolutional layer
    pub fn add_conv(mut self, out_channels: usize, kernel_size: usize) -> Self {
        let in_channels = if self.conv_layers.is_empty() {
            self.input_channels
        } else {
            self.conv_layers.last().unwrap().out_channels
        };

        self.conv_layers.push(Conv2D::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
        ));
        self.activations.push(Activation::Relu);
        self
    }

    /// Add linear layer after convolutions
    pub fn add_linear(mut self, input_size: usize, output_size: usize) -> Self {
        if self.linear_layers.is_empty() {
            self.linear_layers
                .push(Linear::new(input_size, output_size));
        } else {
            let prev_output = self.linear_layers.last().unwrap().out_features;
            self.linear_layers
                .push(Linear::new(prev_output, output_size));
        }
        self
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut current = x.clone();

        // Conv layers
        for (i, conv) in self.conv_layers.iter().enumerate() {
            current = conv.forward(&current);
            current = self.activations[i].apply(&current);
        }

        // Flatten
        let flat_size = current.shape.size();
        current = current.reshape(Shape::new(vec![1, flat_size])).unwrap();

        // Linear layers
        for linear in &self.linear_layers {
            current = linear.forward(&current);
        }

        current
    }

    /// Get all parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        self.conv_layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .chain(
                self.linear_layers
                    .iter()
                    .flat_map(|layer| layer.parameters()),
            )
            .collect()
    }
}

impl Model for CNN {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }

    fn gradients(&self) -> Vec<Option<Tensor>> {
        self.conv_layers
            .iter()
            .flat_map(|layer| layer.gradients())
            .chain(
                self.linear_layers
                    .iter()
                    .flat_map(|layer| layer.gradients()),
            )
            .collect()
    }
}

/// Helper: broadcast bias across batch
fn broadcast_bias(bias: &Tensor, batch_size: usize) -> Tensor {
    let mut data = Vec::with_capacity(bias.data.len() * batch_size);
    for _ in 0..batch_size {
        data.extend_from_slice(&bias.data);
    }
    Tensor::new(data, Shape::new(vec![batch_size, bias.shape.dims[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let layer = Linear::new(3, 5);
        assert_eq!(layer.in_features, 3);
        assert_eq!(layer.out_features, 5);
        assert_eq!(layer.weight.shape, Shape::new(vec![5, 3]));
        assert_eq!(layer.bias.shape, Shape::new(vec![5]));
    }

    #[test]
    fn test_mlp_creation() {
        let mlp = MLP::new(4, vec![8, 8], 2);
        assert_eq!(mlp.input_size, 4);
        assert_eq!(mlp.hidden_sizes, vec![8, 8]);
        assert_eq!(mlp.output_size, 2);
        assert_eq!(mlp.layers.len(), 3); // 2 hidden + 1 output
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = MLP::new(2, vec![3], 2);
        let x = Tensor::matrix(vec![1.0, 2.0], 1, 2);
        let output = mlp.forward(&x);

        // Output should have shape (1, 2)
        assert_eq!(output.shape, Shape::new(vec![1, 2]));
    }

    #[test]
    fn test_activation() {
        let x = Tensor::vector(vec![-1.0, 0.0, 1.0]);

        assert_eq!(Activation::Relu.apply(&x).data, vec![0.0, 0.0, 1.0]);

        let sigmoid = Activation::Sigmoid.apply(&x);
        assert!(sigmoid.data[0] < sigmoid.data[1]);
        assert!(sigmoid.data[1] < sigmoid.data[2]);
    }

    #[test]
    fn test_conv2d() {
        let conv = Conv2D::new(1, 3, (3, 3));
        assert_eq!(conv.in_channels, 1);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3));
    }

    #[test]
    fn test_cnn_creation() {
        let cnn = CNN::new(1).add_conv(16, 3).add_conv(32, 3);

        assert_eq!(cnn.conv_layers.len(), 2);
        assert_eq!(cnn.conv_layers[0].out_channels, 16);
        assert_eq!(cnn.conv_layers[1].out_channels, 32);
    }
}
