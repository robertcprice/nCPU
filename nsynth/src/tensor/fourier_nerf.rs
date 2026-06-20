//! Fourier Features and NeRF (Neural Radiance Fields) Primitives
//!
//! This module implements:
//! - Fourier feature mappings for high-frequency signal representation
//! - Random Fourier Features for kernel approximation
//! - Neural Radiance Field (NeRF) networks for volumetric rendering
//! - Differentiable volume rendering with ray marching
//! - Ray sampling primitives for camera geometry
//!
//! Based on:
//! - "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (Tancik et al., 2020)
//! - "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (Mildenhall et al., 2020)

use crate::tensor::{Shape, Tensor};
use rand_distr::Distribution;
use std::f64::consts::PI;

// ============================================================================
// Fourier Features
// ============================================================================

/// Fourier feature mapping for high-frequency signal representation
///
/// Maps input coordinates to a higher-dimensional space using sinusoidal functions:
/// γ(x) = [cos(2πk₁·x), sin(2πk₁·x), ..., cos(2πk_B·x), sin(2πk_B·x)]
///
/// where k are frequency bands, typically powers of 2: [1, 2, 4, 8, ..., 2^(B-1)]
///
/// # Example
/// ```rust
/// let features = FourierFeatures::new(4, 1.0);
/// let x = Tensor::vector(vec![0.5]);
/// let encoded = features.forward(&x);  // 9-dim: identity + 4*sin + 4*cos
/// ```
#[derive(Debug, Clone)]
pub struct FourierFeatures {
    /// Number of frequency bands
    pub num_bands: usize,
    /// Maximum frequency (multiplier for the highest band)
    pub max_freq: f64,
    /// Whether to include the original input (identity mapping)
    pub include_identity: bool,
    /// Precomputed frequency bands
    pub bands: Vec<f64>,
}

impl FourierFeatures {
    /// Create new Fourier feature mapping
    ///
    /// # Arguments
    /// * `num_bands` - Number of frequency bands (creates 2*num_bands output features)
    /// * `max_freq` - Maximum frequency (typically 1.0 for normalized coordinates)
    ///
    /// # Example
    /// ```
    /// let ff = FourierFeatures::new(6, 1.0);  // Creates bands [1, 2, 4, 8, 16, 32]
    /// ```
    pub fn new(num_bands: usize, max_freq: f64) -> Self {
        let mut bands = Vec::with_capacity(num_bands);
        for k in 0..num_bands {
            // Powers of 2: 2^0, 2^1, ..., 2^(num_bands-1), scaled by max_freq
            bands.push(max_freq * (1 << k) as f64);
        }

        Self {
            num_bands,
            max_freq,
            include_identity: true,
            bands,
        }
    }

    /// Create Fourier features without identity mapping
    pub fn without_identity(num_bands: usize, max_freq: f64) -> Self {
        let mut ff = Self::new(num_bands, max_freq);
        ff.include_identity = false;
        ff
    }

    /// Forward pass: apply Fourier feature mapping
    ///
    /// Input: tensor of shape (..., input_dim)
    /// Output: tensor of shape (..., output_dim)
    ///   where output_dim = input_dim + 2*input_dim*num_bands (with identity)
    ///   or output_dim = 2*input_dim*num_bands (without identity)
    ///
    /// # Mathematical Formulation
    /// For input x with d dimensions:
    /// γ(x) = [x, sin(2πk₁·x), cos(2πk₁·x), ..., sin(2πk_B·x), cos(2πk_B·x)]
    ///
    /// where k_j are the frequency bands and the sin/cos are applied elementwise.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let input_shape = &x.shape;
        let input_dim = *input_shape.dims.last().unwrap_or(&1);
        let batch_dims = &input_shape.dims[..input_shape.dims.len().saturating_sub(1)];
        let batch_size = batch_dims.iter().product::<usize>();

        // Calculate output dimension
        let fourier_dim = 2 * input_dim * self.num_bands;
        let output_dim = if self.include_identity {
            input_dim + fourier_dim
        } else {
            fourier_dim
        };

        // Build output shape
        let mut output_dims = batch_dims.to_vec();
        output_dims.push(output_dim);
        let output_shape = Shape::new(output_dims);

        let mut output_data = Vec::with_capacity(batch_size * output_dim);

        // Process each sample in the batch
        for b in 0..batch_size {
            let start = b * input_dim;
            let end = start + input_dim;
            let sample = &x.data[start..end];

            // Add identity (original input) if enabled
            if self.include_identity {
                output_data.extend_from_slice(sample);
            }

            // Add Fourier features: sin(2π * band * x) and cos(2π * band * x)
            for &band in &self.bands {
                for &xi in sample {
                    let angle = 2.0 * PI * band * xi;
                    output_data.push(angle.sin());
                    output_data.push(angle.cos());
                }
            }
        }

        Tensor::new(output_data, output_shape)
    }

    /// Get output dimension for a given input dimension
    pub fn output_dim(&self, input_dim: usize) -> usize {
        if self.include_identity {
            input_dim + 2 * input_dim * self.num_bands
        } else {
            2 * input_dim * self.num_bands
        }
    }
}

// ============================================================================
// Random Fourier Features (RFF)
// ============================================================================

/// Random Fourier Features for kernel approximation
///
/// Approximates shift-invariant kernels using random feature maps.
/// Based on Rahimi & Recht (2007): "Random Features for Large-Scale Kernel Machines"
///
/// Maps input x to: sqrt(2/D) * cos(W·x + b)
/// where W ~ N(0, σ²) and b ~ Uniform(0, 2π)
///
/// This provides a finite-dimensional approximation of the kernel:
/// K(x, y) ≈ φ(x)·φ(y) where φ is the RFF mapping
#[derive(Debug, Clone)]
pub struct RandomFourierFeatures {
    /// Input dimension
    pub dim: usize,
    /// Number of random features (output dimension)
    pub num_features: usize,
    /// Random weight matrix: shape (num_features, dim)
    pub W: Tensor,
    /// Random bias vector: shape (num_features,)
    pub b: Tensor,
    /// Scaling factor (sqrt(2/num_features))
    pub scale: f64,
}

impl RandomFourierFeatures {
    /// Create new Random Fourier Features
    ///
    /// # Arguments
    /// * `dim` - Input dimension
    /// * `num_features` - Number of random features (output dimension)
    /// * `kernel_scale` - Scale parameter for the Gaussian kernel (σ)
    ///
    /// The kernel approximated is: K(x, y) = exp(-||x-y||² / (2σ²))
    pub fn new(dim: usize, num_features: usize, kernel_scale: f64) -> Self {
        use rand::Rng;
        use rand_distr::{Normal, Uniform};

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, kernel_scale).unwrap();
        let uniform = Uniform::new(0.0, 2.0 * PI);

        // Initialize W from N(0, kernel_scale²)
        let mut w_data = Vec::with_capacity(num_features * dim);
        for _ in 0..(num_features * dim) {
            w_data.push(normal.sample(&mut rng));
        }
        let W = Tensor::new(w_data, Shape::new(vec![num_features, dim]));

        // Initialize b from Uniform(0, 2π)
        let mut b_data = Vec::with_capacity(num_features);
        for _ in 0..num_features {
            b_data.push(uniform.sample(&mut rng));
        }
        let b = Tensor::new(b_data, Shape::new(vec![num_features]));

        let scale = (2.0 / num_features as f64).sqrt();

        Self {
            dim,
            num_features,
            W,
            b,
            scale,
        }
    }

    /// Create RFF with fixed random seed for reproducibility
    pub fn with_seed(dim: usize, num_features: usize, kernel_scale: f64, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        use rand_distr::{Normal, Uniform};

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, kernel_scale).unwrap();
        let uniform = Uniform::new(0.0, 2.0 * PI);

        let mut w_data = Vec::with_capacity(num_features * dim);
        for _ in 0..(num_features * dim) {
            w_data.push(normal.sample(&mut rng));
        }
        let W = Tensor::new(w_data, Shape::new(vec![num_features, dim]));

        let mut b_data = Vec::with_capacity(num_features);
        for _ in 0..num_features {
            b_data.push(uniform.sample(&mut rng));
        }
        let b = Tensor::new(b_data, Shape::new(vec![num_features]));

        let scale = (2.0 / num_features as f64).sqrt();

        Self {
            dim,
            num_features,
            W,
            b,
            scale,
        }
    }

    /// Forward pass: compute random Fourier features
    ///
    /// Input: tensor of shape (..., dim)
    /// Output: tensor of shape (..., num_features)
    ///
    /// Output: sqrt(2/D) * cos(W·x + b)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let input_shape = &x.shape;
        let batch_dims = &input_shape.dims[..input_shape.dims.len().saturating_sub(1)];
        let batch_size = batch_dims.iter().product::<usize>();

        // Reshape input to (batch_size, dim) for matrix multiplication
        let x_reshaped = if x.shape.rank() > 1 {
            x.clone()
        } else {
            // Single sample: (dim,) -> (1, dim)
            let mut new_dims = vec![1];
            new_dims.extend_from_slice(&x.shape.dims);
            Tensor {
                data: x.data.clone(),
                shape: Shape::new(new_dims),
                dtype: x.dtype,
                grads: x.grads.clone(),
                requires_grad: x.requires_grad,
            }
        };

        // Compute W·x + b
        // W: (num_features, dim), x: (batch_size, dim) -> (num_features, batch_size)
        let mut output_data = Vec::with_capacity(batch_size * self.num_features);

        for b in 0..batch_size {
            for f in 0..self.num_features {
                let mut dot = self.b.data[f]; // Add bias
                for d in 0..self.dim {
                    let w_idx = f * self.dim + d;
                    let x_idx = b * self.dim + d;
                    dot += self.W.data[w_idx] * x_reshaped.data[x_idx];
                }
                output_data.push(self.scale * dot.cos());
            }
        }

        let mut output_dims = batch_dims.to_vec();
        output_dims.push(self.num_features);
        let output_shape = Shape::new(output_dims);

        Tensor::new(output_data, output_shape)
    }
}

// ============================================================================
// Neural Radiance Field (NeRF)
// ============================================================================

/// Neural Radiance Field network for volumetric rendering
///
/// A NeRF network predicts RGB color and volume density at any 3D point.
/// The network uses positional encoding (Fourier features) for both
/// position (x, y, z) and viewing direction.
///
/// Architecture:
/// - Position encoder: FourierFeatures with 10+ bands (for high-frequency spatial detail)
/// - Direction encoder: FourierFeatures with 4 bands (for view-dependent appearance)
/// - MLP backbone: 8-layer fully connected with ReLU and skip connections
/// - RGB head: additional layers with direction encoding
/// - Density head: single linear layer (position only)
///
/// # References
/// - Mildenhall et al. (2020): "NeRF: Representing Scenes as Neural Radiance Fields"
pub struct NeRF {
    /// Position encoder (high-frequency)
    pub pos_encoder: FourierFeatures,
    /// Direction encoder (lower frequency)
    pub dir_encoder: FourierFeatures,
    /// MLP backbone: maps encoded position to features + density
    pub mlp_backbone: Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of hidden layers
    pub num_layers: usize,
}

impl NeRF {
    /// Create new NeRF network
    ///
    /// # Arguments
    /// * `hidden_dim` - Hidden layer dimension (default: 256)
    /// * `num_layers` - Number of hidden layers before density head (default: 8)
    ///
    /// # Architecture
    /// - Position encoding: 10 frequency bands (60-dim output with identity)
    /// - Direction encoding: 4 frequency bands (27-dim output with identity)
    /// - Skip connection at layer 4
    /// - Density output at layer 8
    /// - RGB output with direction encoding at layer 9
    pub fn new(hidden_dim: usize, num_layers: usize) -> Self {
        // Position encoder: 10 bands (typically used for x, y, z)
        let pos_encoder = FourierFeatures::new(10, 1.0);

        // Direction encoder: 4 bands (for viewing direction)
        let dir_encoder = FourierFeatures::new(4, 1.0);

        // Note: In a real implementation, this would be a proper MLP with
        // trainable weights. Here we use a closure placeholder.
        let hd = hidden_dim; // Copy for closure to avoid lifetime issues
        let mlp_backbone_fn = Box::new(move |encoded_pos: &Tensor| {
            // Placeholder: would be full MLP with skip connections
            // Returns (features, density)
            let batch_size = encoded_pos.shape.dims[0];
            let density = Tensor::new(vec![0.0; batch_size], Shape::new(vec![batch_size]));
            let features =
                Tensor::new(vec![0.0; batch_size * hd], Shape::new(vec![batch_size, hd]));
            (features, density)
        });

        Self {
            pos_encoder,
            dir_encoder,
            mlp_backbone: mlp_backbone_fn,
            hidden_dim,
            num_layers,
        }
    }

    /// Forward pass: query the NeRF at given positions and directions
    ///
    /// # Arguments
    /// * `rays_o` - Ray origins, shape (N_rays, 3)
    /// * `rays_d` - Ray directions (normalized), shape (N_rays, 3)
    ///
    /// # Returns
    /// * `(rgb, density)` - RGB color (N_rays, 3) and volume density (N_rays,)
    ///
    /// Note: In actual usage, you'd sample points along rays and query
    /// the network at each point, then use volume_rendering() to integrate.
    pub fn forward(&self, rays_o: &Tensor, rays_d: &Tensor) -> (Tensor, Tensor) {
        // Encode position (rays_o) with high-frequency Fourier features
        let encoded_pos = self.pos_encoder.forward(rays_o);

        // Encode direction (rays_d) with lower-frequency features
        let _encoded_dir = self.dir_encoder.forward(rays_d);

        // Forward through MLP (placeholder)
        let (_features, density) = (self.mlp_backbone)(&encoded_pos);

        // In full implementation: concatenate features with encoded_dir
        // and pass through RGB head to get final RGB output

        // Placeholder RGB output
        let rgb = Tensor::new(
            vec![0.5; rays_o.shape.dims[0] * 3],
            Shape::new(vec![rays_o.shape.dims[0], 3]),
        );

        (rgb, density)
    }

    /// Query NeRF at specific 3D points with viewing directions
    ///
    /// # Arguments
    /// * `points` - 3D positions, shape (N, 3)
    /// * `directions` - Viewing directions, shape (N, 3)
    ///
    /// # Returns
    /// * `(rgb, sigma)` - RGB color (N, 3) and density (N,)
    pub fn query(&self, points: &Tensor, directions: &Tensor) -> (Tensor, Tensor) {
        self.forward(points, directions)
    }

    /// Volume rendering along rays using the predicted colors and densities
    ///
    /// This implements the differentiable volume rendering integral:
    /// C(r) = ∫ T(t) * σ(t) * c(t) dt
    ///
    /// where T(t) is transmittance (probability of light reaching point t)
    ///
    /// # Arguments
    /// * `rgb` - RGB colors at sample points, shape (N_rays, N_samples, 3)
    /// * `sigma` - Volume density at sample points, shape (N_rays, N_samples)
    /// * `t` - Distance along ray for each sample, shape (N_rays, N_samples)
    ///
    /// # Returns
    /// * Rendered RGB color for each ray, shape (N_rays, 3)
    pub fn volume_rendering(&self, rgb: &Tensor, sigma: &Tensor, t: &Tensor) -> Tensor {
        volume_rendering::integrate_weights(sigma, rgb, t)
    }
}

// ============================================================================
// Volume Rendering Primitives
// ============================================================================

/// Volume rendering module for NeRF integration
pub mod volume_rendering {
    use super::*;

    /// Differentiable volume rendering integral
    ///
    /// Computes: C(r) = Σ T_i * (1 - exp(-δ_i * σ_i)) * c_i
    ///
    /// where:
    /// - δ_i = t_{i+1} - t_i (distance between samples)
    /// - T_i = exp(-Σ_{j<i} δ_j * σ_j) (transmittance to point i)
    /// - σ_i = density at point i
    /// - c_i = RGB color at point i
    ///
    /// # Arguments
    /// * `sigma` - Volume density (N_rays, N_samples)
    /// * `rgb` - RGB color (N_rays, N_samples, 3)
    /// * `t` - Distance values (N_rays, N_samples)
    ///
    /// # Returns
    /// * Rendered color (N_rays, 3)
    pub fn integrate_weights(sigma: &Tensor, rgb: &Tensor, t: &Tensor) -> Tensor {
        let n_rays = sigma.shape.dims[0];
        let n_samples = sigma.shape.dims[1];
        let n_channels = rgb.shape.dims.get(2).copied().unwrap_or(3);

        let mut output_data = vec![0.0; n_rays * n_channels];

        for r in 0..n_rays {
            let mut transmittance = 1.0;

            for s in 0..n_samples {
                // Distance between samples (δ)
                let delta = if s < n_samples - 1 {
                    let t_curr = t.data[r * n_samples + s];
                    let t_next = t.data[r * n_samples + s + 1];
                    t_next - t_curr
                } else {
                    // For last sample, use a small default distance
                    1e-3
                };

                // Density at this sample
                let sigma_i = sigma.data[r * n_samples + s];

                // Absorption probability: α_i = 1 - exp(-δ_i * σ_i)
                let alpha = 1.0 - (-delta * sigma_i).exp();

                // Weight for this sample: w_i = T_i * α_i
                let weight = transmittance * alpha;

                // Accumulate color
                for c in 0..n_channels {
                    let rgb_idx = (r * n_samples + s) * n_channels + c;
                    let out_idx = r * n_channels + c;
                    output_data[out_idx] += weight * rgb.data[rgb_idx];
                }

                // Update transmittance: T_{i+1} = T_i * (1 - α_i)
                transmittance *= 1.0 - alpha;
            }
        }

        Tensor::new(output_data, Shape::new(vec![n_rays, n_channels]))
    }

    /// Compute transmittance (accumulated opacity) along rays
    ///
    /// T_i = exp(-Σ_{j<i} δ_j * σ_j)
    ///
    /// This is the probability of light reaching point i without being absorbed.
    ///
    /// # Arguments
    /// * `sigma` - Volume density (N_rays, N_samples)
    ///
    /// # Returns
    /// * Transmittance values (N_rays, N_samples)
    pub fn transmittance(sigma: &Tensor) -> Tensor {
        let n_rays = sigma.shape.dims[0];
        let n_samples = sigma.shape.dims[1];

        let mut transmittance_data = vec![0.0; n_rays * n_samples];

        for r in 0..n_rays {
            let mut accumulated_density = 0.0;

            for s in 0..n_samples {
                // Transmittance at this point
                transmittance_data[r * n_samples + s] = f64::exp(-accumulated_density);

                // Accumulate density for next point (assuming uniform spacing)
                accumulated_density += sigma.data[r * n_samples + s];
            }
        }

        Tensor::new(transmittance_data, Shape::new(vec![n_rays, n_samples]))
    }

    /// Compute absorption (alpha) values from density
    ///
    /// α_i = 1 - exp(-δ_i * σ_i)
    ///
    /// # Arguments
    /// * `sigma` - Volume density (N_rays, N_samples)
    /// * `deltas` - Distance between samples (N_rays, N_samples)
    ///
    /// # Returns
    /// * Alpha values (N_rays, N_samples)
    pub fn compute_alpha(sigma: &Tensor, deltas: &Tensor) -> Tensor {
        let n_rays = sigma.shape.dims[0];
        let n_samples = sigma.shape.dims[1];

        let mut alpha_data = Vec::with_capacity(n_rays * n_samples);

        for i in 0..(n_rays * n_samples) {
            let alpha = 1.0 - (-deltas.data[i] * sigma.data[i]).exp();
            alpha_data.push(alpha);
        }

        Tensor::new(alpha_data, sigma.shape.clone())
    }

    /// Render depth map from density predictions
    ///
    /// Depth = Σ w_i * t_i
    ///
    /// Useful for regularization and multi-resolution training.
    pub fn render_depth(weights: &Tensor, t: &Tensor) -> Tensor {
        let n_rays = weights.shape.dims[0];
        let n_samples = weights.shape.dims[1];

        let mut depth_data = vec![0.0; n_rays];

        for r in 0..n_rays {
            for s in 0..n_samples {
                let w_idx = r * n_samples + s;
                depth_data[r] += weights.data[w_idx] * t.data[w_idx];
            }
        }

        Tensor::new(depth_data, Shape::new(vec![n_rays]))
    }
}

// ============================================================================
// Ray Sampling Primitives
// ============================================================================

/// Ray sampling module for camera geometry and ray generation
pub mod ray_sampling {
    use super::*;

    /// Generate rays from camera parameters
    ///
    /// Creates rays passing through each pixel of an image plane.
    ///
    /// # Arguments
    /// * `camera_params` - Camera parameters tensor containing:
    ///   - Position (3,) or (4,) for homogeneous
    ///   - Rotation matrix (3, 3) or quaternion (4,)
    ///   - Focal length (1,) or full intrinsics (3, 3)
    /// * `height` - Image height in pixels
    /// * `width` - Image width in pixels
    ///
    /// # Returns
    /// * `(rays_o, rays_d)` - Ray origins and directions
    ///   - rays_o: (height * width, 3)
    ///   - rays_d: (height * width, 3)
    ///
    /// # Camera Model
    /// Uses pinhole camera model:
    /// - Rays originate from camera center
    /// - Pass through pixels on the image plane
    /// - Image plane at distance = focal_length from camera
    pub fn generate_rays(_camera_params: &Tensor, height: usize, width: usize) -> (Tensor, Tensor) {
        // This is a simplified implementation
        // Full version would parse camera_params for position, rotation, intrinsics

        let n_pixels = height * width;
        let mut rays_o_data = Vec::with_capacity(n_pixels * 3);
        let mut rays_d_data = Vec::with_capacity(n_pixels * 3);

        // Camera at origin, pointing along +Z axis
        let camera_origin = [0.0, 0.0, 0.0];
        let focal_length = 1.0; // Default

        // Image plane coordinates (centered at origin)
        for y in 0..height {
            for x in 0..width {
                // Convert pixel coordinates to normalized device coordinates
                let u = (x as f64 + 0.5) / width as f64 - 0.5; // [-0.5, 0.5]
                let v = 0.5 - (y as f64 + 0.5) / height as f64; // [0.5, -0.5] (flip Y)

                // Aspect ratio correction
                let aspect = width as f64 / height as f64;
                let u_corrected = u * aspect;

                // Ray direction (normalized)
                let dir = [u_corrected, v, focal_length];

                // Normalize direction
                let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                let dir_normalized = [dir[0] / norm, dir[1] / norm, dir[2] / norm];

                // Store ray origin (same for all rays in this simple setup)
                rays_o_data.extend_from_slice(&camera_origin);

                // Store ray direction
                rays_d_data.extend_from_slice(&dir_normalized);
            }
        }

        let rays_o = Tensor::new(rays_o_data, Shape::new(vec![n_pixels, 3]));
        let rays_d = Tensor::new(rays_d_data, Shape::new(vec![n_pixels, 3]));

        (rays_o, rays_d)
    }

    /// Stratified sampling along rays
    ///
    /// Samples points uniformly along each ray with stratified noise
    /// for better integration during training.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples per ray
    /// * `near` - Near plane distance
    /// * `far` - Far plane distance
    ///
    /// # Returns
    /// * Sample distances tensor (N_rays, num_samples)
    ///
    /// # Sampling Strategy
    /// Uses stratified sampling: divide [near, far] into num_samples bins
    /// and sample uniformly within each bin, then add small random noise.
    pub fn stratified_sampling(num_samples: usize, near: f64, far: f64) -> Tensor {
        use rand::Rng;
        use rand_distr::Uniform;

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0, 1.0);

        // Bin edges
        let bin_edges = (far - near) / num_samples as f64;

        let mut sample_data = Vec::new();

        for i in 0..num_samples {
            // Uniform sample within bin
            let bin_center = near + (i as f64 + 0.5) * bin_edges;

            // Add small stratified noise (in practice, this varies per ray)
            let noise = (uniform.sample(&mut rng) - 0.5) * bin_edges;

            sample_data.push((bin_center + noise).max(near).min(far));
        }

        // Shape: (1, num_samples) - would be broadcasted across rays
        Tensor::new(sample_data, Shape::new(vec![1, num_samples]))
    }

    /// Generate samples for multiple rays with per-ray stratification
    ///
    /// # Arguments
    /// * `n_rays` - Number of rays
    /// * `num_samples` - Samples per ray
    /// * `near` - Near bound
    /// * `far` - Far bound
    ///
    /// # Returns
    /// * Sample distances (n_rays, num_samples)
    pub fn stratified_sampling_rays(
        n_rays: usize,
        num_samples: usize,
        near: f64,
        far: f64,
    ) -> Tensor {
        use rand::Rng;
        use rand_distr::Uniform;

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0, 1.0);

        let bin_edges = (far - near) / num_samples as f64;
        let mut sample_data = Vec::with_capacity(n_rays * num_samples);

        for _ in 0..n_rays {
            for i in 0..num_samples {
                let bin_center = near + (i as f64 + 0.5) * bin_edges;
                let noise = (uniform.sample(&mut rng) - 0.5) * bin_edges;
                sample_data.push((bin_center + noise).max(near).min(far));
            }
        }

        Tensor::new(sample_data, Shape::new(vec![n_rays, num_samples]))
    }

    /// Generate sample points along rays in 3D space
    ///
    /// # Arguments
    /// * `rays_o` - Ray origins (N_rays, 3)
    /// * `rays_d` - Ray directions (N_rays, 3)
    /// * `t` - Sample distances (N_rays, N_samples)
    ///
    /// # Returns
    /// * 3D points (N_rays, N_samples, 3)
    ///
    /// Points are computed as: p = rays_o + rays_d * t
    pub fn generate_points(rays_o: &Tensor, rays_d: &Tensor, t: &Tensor) -> Tensor {
        let n_rays = rays_o.shape.dims[0];
        let n_samples = t.shape.dims[1];

        let mut points_data = Vec::with_capacity(n_rays * n_samples * 3);

        for r in 0..n_rays {
            let ox = rays_o.data[r * 3 + 0];
            let oy = rays_o.data[r * 3 + 1];
            let oz = rays_o.data[r * 3 + 2];

            let dx = rays_d.data[r * 3 + 0];
            let dy = rays_d.data[r * 3 + 1];
            let dz = rays_d.data[r * 3 + 2];

            for s in 0..n_samples {
                let t_val = t.data[r * n_samples + s];

                points_data.push(ox + dx * t_val);
                points_data.push(oy + dy * t_val);
                points_data.push(oz + dz * t_val);
            }
        }

        Tensor::new(points_data, Shape::new(vec![n_rays, n_samples, 3]))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fourier_features_identity() {
        let ff = FourierFeatures::new(2, 1.0);
        let x = Tensor::vector(vec![0.5]);

        let encoded = ff.forward(&x);

        // Output: [x, sin(2π*1*x), cos(2π*1*x), sin(2π*2*x), cos(2π*2*x)]
        // = [0.5, sin(π), cos(π), sin(2π), cos(2π)]
        // = [0.5, ~0, -1, ~0, 1]
        assert_eq!(encoded.shape.dims, &[5]);
        assert!((encoded.data[0] - 0.5).abs() < 1e-10);
        assert!((encoded.data[2] - (-1.0)).abs() < 1e-10);
        assert!((encoded.data[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fourier_features_batch() {
        let ff = FourierFeatures::new(2, 1.0);
        let x = Tensor::new(vec![0.0, 0.5, 1.0], Shape::new(vec![3, 1]));

        let encoded = ff.forward(&x);

        // Shape: (3, 5) - 3 samples, 5 output dimensions
        assert_eq!(encoded.shape.dims, &[3, 5]);
    }

    #[test]
    fn test_fourier_features_output_dim() {
        let ff = FourierFeatures::new(4, 1.0);

        // Input dim 3, 4 bands, with identity
        // Output: 3 + 2*3*4 = 27
        assert_eq!(ff.output_dim(3), 27);
    }

    #[test]
    fn test_fourier_features_no_identity() {
        let ff = FourierFeatures::without_identity(2, 1.0);
        let x = Tensor::vector(vec![0.5]);

        let encoded = ff.forward(&x);

        // Output: [sin(2π*1*x), cos(2π*1*x), sin(2π*2*x), cos(2π*2*x)]
        assert_eq!(encoded.shape.dims, &[4]);
        assert_eq!(ff.output_dim(1), 4);
    }

    #[test]
    fn test_fourier_features_multidim() {
        let ff = FourierFeatures::new(1, 1.0);
        let x = Tensor::new(vec![0.25, 0.5], Shape::new(vec![1, 2]));

        let encoded = ff.forward(&x);

        // Input: (1, 2), 1 band, with identity
        // Output: (1, 2 + 2*2*1) = (1, 6)
        assert_eq!(encoded.shape.dims, &[1, 6]);
    }

    #[test]
    fn test_random_fourier_features_dim() {
        let rff = RandomFourierFeatures::new(3, 10, 1.0);

        assert_eq!(rff.dim, 3);
        assert_eq!(rff.num_features, 10);
        assert_eq!(rff.W.shape.dims, &[10, 3]);
        assert_eq!(rff.b.shape.dims, &[10]);
    }

    #[test]
    fn test_random_fourier_features_forward() {
        let rff = RandomFourierFeatures::new(2, 5, 1.0);
        let x = Tensor::new(vec![1.0, 2.0], Shape::new(vec![1, 2]));

        let encoded = rff.forward(&x);

        assert_eq!(encoded.shape.dims, &[1, 5]);
    }

    #[test]
    fn test_random_fourier_features_batch() {
        let rff = RandomFourierFeatures::new(2, 5, 1.0);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));

        let encoded = rff.forward(&x);

        assert_eq!(encoded.shape.dims, &[2, 5]);
    }

    #[test]
    fn test_volume_rendering_transmittance() {
        let sigma = Tensor::new(vec![0.0, 1.0, 0.0], Shape::new(vec![1, 3]));

        let T = volume_rendering::transmittance(&sigma);

        // T[0] = exp(0) = 1
        // T[1] = exp(0) = 1 (no previous density)
        // T[2] = exp(-1) ≈ 0.368
        assert!((T.data[0] - 1.0).abs() < 1e-6);
        assert!((T.data[1] - 1.0).abs() < 1e-6);
        assert!((T.data[2] - (-1.0f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_volume_rendering_integrate() {
        let sigma = Tensor::new(vec![0.0, 2.0, 0.0], Shape::new(vec![1, 3]));
        let rgb = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            Shape::new(vec![1, 3, 3]),
        );
        let t = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));

        let rendered = volume_rendering::integrate_weights(&sigma, &rgb, &t);

        // Single ray, 3 channels
        assert_eq!(rendered.shape.dims, &[1, 3]);
    }

    #[test]
    fn test_volume_rendering_alpha() {
        let sigma = Tensor::new(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let deltas = Tensor::new(vec![0.5, 0.5], Shape::new(vec![1, 2]));

        let alpha = volume_rendering::compute_alpha(&sigma, &deltas);

        // α = 1 - exp(-δ * σ)
        // α[0] = 1 - exp(-0.5) ≈ 0.393
        // α[1] = 1 - exp(-1.0) ≈ 0.632
        let expected0 = 1.0 - (-0.5f64).exp();
        let expected1 = 1.0 - (-1.0f64).exp();
        assert!((alpha.data[0] - expected0).abs() < 1e-6);
        assert!((alpha.data[1] - expected1).abs() < 1e-6);
    }

    #[test]
    fn test_ray_generation() {
        let camera = Tensor::new(vec![0.0, 0.0, 0.0], Shape::new(vec![3]));

        let (rays_o, rays_d) = ray_sampling::generate_rays(&camera, 10, 10);

        // 100 pixels
        assert_eq!(rays_o.shape.dims, &[100, 3]);
        assert_eq!(rays_d.shape.dims, &[100, 3]);

        // All rays originate from camera origin
        for i in 0..100 {
            assert!((rays_o.data[i * 3 + 0] - 0.0).abs() < 1e-10);
            assert!((rays_o.data[i * 3 + 1] - 0.0).abs() < 1e-10);
            assert!((rays_o.data[i * 3 + 2] - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stratified_sampling() {
        let t = ray_sampling::stratified_sampling(10, 0.0, 1.0);

        assert_eq!(t.shape.dims, &[1, 10]);

        // Check monotonic increase
        for i in 1..10 {
            assert!(t.data[i] > t.data[i - 1]);
        }
    }

    #[test]
    fn test_stratified_sampling_bounds() {
        let t = ray_sampling::stratified_sampling(5, 2.0, 5.0);

        // All samples should be in [2, 5]
        for &val in &t.data {
            assert!(val >= 2.0 && val <= 5.0);
        }
    }

    #[test]
    fn test_generate_points() {
        let rays_o = Tensor::new(vec![0.0, 0.0, 0.0], Shape::new(vec![1, 3]));
        let rays_d = Tensor::new(vec![1.0, 0.0, 0.0], Shape::new(vec![1, 3]));
        let t = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));

        let points = ray_sampling::generate_points(&rays_o, &rays_d, &t);

        assert_eq!(points.shape.dims, &[1, 3, 3]);

        // Points should be at x=1, 2, 3
        assert!((points.data[0] - 1.0).abs() < 1e-10);
        assert!((points.data[3] - 2.0).abs() < 1e-10);
        assert!((points.data[6] - 3.0).abs() < 1e-10);
    }
}
