//! Special Vision Operations for nCPU/nSynth
//!
//! 3D convolutions, pooling, and deformable convolutions for advanced vision tasks.
//! Supports medical imaging (CT/MRI), video processing, and 3D data analysis.

use super::ops::{Shape, Tensor};

// ============================================================================
// 3D Convolution
// ============================================================================

/// 3D Convolution layer
///
/// Operates on 5D tensors: (batch, channels, depth, height, width)
/// Used for volumetric data (CT/MRI scans), video processing, and 3D feature extraction.
#[derive(Debug)]
pub struct Conv3D {
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width)
    pub stride: (usize, usize, usize),
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
    /// Weight tensor: (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
    pub weight: Tensor,
    /// Bias tensor: (out_channels,)
    pub bias: Tensor,
}

impl Conv3D {
    /// Create new 3D convolution layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        let (kd, kh, kw) = kernel_size;
        let weight_shape = Shape::new(vec![out_channels, in_channels, kd, kh, kw]);

        // Glorot initialization for 3D conv
        let fan_in = (in_channels * kd * kh * kw) as f64;
        let fan_out = (out_channels * kd * kh * kw) as f64;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        let mut weight_data = Vec::with_capacity(out_channels * in_channels * kd * kh * kw);
        for _ in 0..weight_shape.size() {
            weight_data.push((pseudo_random() * 2.0 - 1.0) * limit);
        }

        let weight = Tensor::new(weight_data, weight_shape);
        let bias = Tensor::zeros(Shape::new(vec![out_channels]));

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            weight,
            bias,
        }
    }

    /// Set stride
    pub fn with_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn with_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Forward pass through 3D convolution
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.shape.rank(), 5, "Input must be 5D tensor (NCDHW)");

        let (batch_size, in_ch, d_in, h_in, w_in) = (
            x.shape.dims[0],
            x.shape.dims[1],
            x.shape.dims[2],
            x.shape.dims[3],
            x.shape.dims[4],
        );

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;

        // Output spatial dimensions
        let d_out = (d_in + 2 * pd - kd) / sd + 1;
        let h_out = (h_in + 2 * ph - kh) / sh + 1;
        let w_out = (w_in + 2 * pw - kw) / sw + 1;

        // Padded input
        let x_padded = pad3d(x, pd, ph, pw);

        // Output tensor
        let mut output = Vec::with_capacity(batch_size * self.out_channels * d_out * h_out * w_out);

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for od in 0..d_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut sum = 0.0;

                            // Convolution window
                            for ic in 0..in_ch {
                                for kd_idx in 0..kd {
                                    for kh_idx in 0..kh {
                                        for kw_idx in 0..kw {
                                            let id = od * sd + kd_idx;
                                            let ih = oh * sh + kh_idx;
                                            let iw = ow * sw + kw_idx;

                                            let in_idx = b
                                                * x_padded.shape.dims[1]
                                                * x_padded.shape.dims[2]
                                                * x_padded.shape.dims[3]
                                                * x_padded.shape.dims[4]
                                                + ic * x_padded.shape.dims[2]
                                                    * x_padded.shape.dims[3]
                                                    * x_padded.shape.dims[4]
                                                + id * x_padded.shape.dims[3]
                                                    * x_padded.shape.dims[4]
                                                + ih * x_padded.shape.dims[4]
                                                + iw;

                                            let w_idx = oc
                                                * self.weight.shape.dims[1]
                                                * self.weight.shape.dims[2]
                                                * self.weight.shape.dims[3]
                                                * self.weight.shape.dims[4]
                                                + ic * self.weight.shape.dims[2]
                                                    * self.weight.shape.dims[3]
                                                    * self.weight.shape.dims[4]
                                                + kd_idx
                                                    * self.weight.shape.dims[3]
                                                    * self.weight.shape.dims[4]
                                                + kh_idx * self.weight.shape.dims[4]
                                                + kw_idx;

                                            sum += x_padded.data[in_idx] * self.weight.data[w_idx];
                                        }
                                    }
                                }
                            }

                            output.push(sum + self.bias.data[oc]);
                        }
                    }
                }
            }
        }

        Tensor::new(
            output,
            Shape::new(vec![batch_size, self.out_channels, d_out, h_out, w_out]),
        )
    }
}

// ============================================================================
// 3D Transposed Convolution
// ============================================================================

/// 3D Transposed Convolution (Deconvolution) layer
///
/// Performs upsampling by learning interpolation kernels.
/// Used for decoder architectures, segmentation, and generative models.
#[derive(Debug)]
pub struct Conv3DTranspose {
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width)
    pub stride: (usize, usize, usize),
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
    /// Output padding (for handling stride > 1)
    pub output_padding: (usize, usize, usize),
    /// Weight tensor: (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
    pub weight: Tensor,
    /// Bias tensor: (out_channels,)
    pub bias: Tensor,
}

impl Conv3DTranspose {
    /// Create new 3D transposed convolution layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        let (kd, kh, kw) = kernel_size;
        let weight_shape = Shape::new(vec![in_channels, out_channels, kd, kh, kw]);

        // Glorot initialization
        let fan_in = (in_channels * kd * kh * kw) as f64;
        let fan_out = (out_channels * kd * kh * kw) as f64;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        let mut weight_data = Vec::with_capacity(in_channels * out_channels * kd * kh * kw);
        for _ in 0..weight_shape.size() {
            weight_data.push((pseudo_random() * 2.0 - 1.0) * limit);
        }

        let weight = Tensor::new(weight_data, weight_shape);
        let bias = Tensor::zeros(Shape::new(vec![out_channels]));

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            weight,
            bias,
        }
    }

    /// Set stride
    pub fn with_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn with_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Set output padding
    pub fn with_output_padding(mut self, output_padding: (usize, usize, usize)) -> Self {
        self.output_padding = output_padding;
        self
    }

    /// Forward pass through 3D transposed convolution
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.shape.rank(), 5, "Input must be 5D tensor (NCDHW)");

        let (batch_size, in_ch, d_in, h_in, w_in) = (
            x.shape.dims[0],
            x.shape.dims[1],
            x.shape.dims[2],
            x.shape.dims[3],
            x.shape.dims[4],
        );

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let (opd, oph, opw) = self.output_padding;

        // Output spatial dimensions for transposed conv
        let d_out = (d_in - 1) * sd - 2 * pd + kd + opd;
        let h_out = (h_in - 1) * sh - 2 * ph + kh + oph;
        let w_out = (w_in - 1) * sw - 2 * pw + kw + opw;

        // Initialize output with zeros
        let mut output_data = vec![0.0f64; batch_size * self.out_channels * d_out * h_out * w_out];

        // Perform transposed convolution (gradient of convolution w.r.t input)
        for b in 0..batch_size {
            for ic in 0..in_ch {
                for id in 0..d_in {
                    for ih in 0..h_in {
                        for iw in 0..w_in {
                            let input_val = x.data[b * in_ch * d_in * h_in * w_in
                                + ic * d_in * h_in * w_in
                                + id * h_in * w_in
                                + ih * w_in
                                + iw];

                            // Apply to output region
                            for oc in 0..self.out_channels {
                                for kd_idx in 0..kd {
                                    for kh_idx in 0..kh {
                                        for kw_idx in 0..kw {
                                            let od = if id * sd >= pd {
                                                id * sd - pd
                                            } else {
                                                continue;
                                            };
                                            let oh = if ih * sh >= ph {
                                                ih * sh - ph
                                            } else {
                                                continue;
                                            };
                                            let ow = if iw * sw >= pw {
                                                iw * sw - pw
                                            } else {
                                                continue;
                                            };

                                            if od < d_out && oh < h_out && ow < w_out {
                                                let w_idx = ic * self.out_channels * kd * kh * kw
                                                    + oc * kd * kh * kw
                                                    + kd_idx * kh * kw
                                                    + kh_idx * kw
                                                    + kw_idx;

                                                let out_idx =
                                                    b * self.out_channels * d_out * h_out * w_out
                                                        + oc * d_out * h_out * w_out
                                                        + od * h_out * w_out
                                                        + oh * w_out
                                                        + ow;

                                                output_data[out_idx] +=
                                                    input_val * self.weight.data[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for od in 0..d_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let idx = b * self.out_channels * d_out * h_out * w_out
                                + oc * d_out * h_out * w_out
                                + od * h_out * w_out
                                + oh * w_out
                                + ow;
                            output_data[idx] += self.bias.data[oc];
                        }
                    }
                }
            }
        }

        Tensor::new(
            output_data,
            Shape::new(vec![batch_size, self.out_channels, d_out, h_out, w_out]),
        )
    }
}

// ============================================================================
// 3D Pooling
// ============================================================================

/// 3D Max Pooling layer
///
/// Reduces spatial dimensions by taking maximum values in local 3D windows.
#[derive(Debug)]
pub struct MaxPool3D {
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width)
    pub stride: (usize, usize, usize),
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
}

impl MaxPool3D {
    /// Create new 3D max pooling layer
    pub fn new(kernel_size: (usize, usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0, 0),
        }
    }

    /// Set stride
    pub fn with_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn with_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Forward pass through 3D max pooling
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.shape.rank(), 5, "Input must be 5D tensor (NCDHW)");

        let (batch_size, channels, d_in, h_in, w_in) = (
            x.shape.dims[0],
            x.shape.dims[1],
            x.shape.dims[2],
            x.shape.dims[3],
            x.shape.dims[4],
        );

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;

        // Output dimensions
        let d_out = (d_in + 2 * pd - kd) / sd + 1;
        let h_out = (h_in + 2 * ph - kh) / sh + 1;
        let w_out = (w_in + 2 * pw - kw) / sw + 1;

        let x_padded = pad3d(x, pd, ph, pw);
        let mut output = Vec::with_capacity(batch_size * channels * d_out * h_out * w_out);

        for b in 0..batch_size {
            for c in 0..channels {
                for od in 0..d_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut max_val = f64::NEG_INFINITY;

                            for kd_idx in 0..kd {
                                for kh_idx in 0..kh {
                                    for kw_idx in 0..kw {
                                        let id = od * sd + kd_idx;
                                        let ih = oh * sh + kh_idx;
                                        let iw = ow * sw + kw_idx;

                                        let idx = b
                                            * x_padded.shape.dims[1]
                                            * x_padded.shape.dims[2]
                                            * x_padded.shape.dims[3]
                                            * x_padded.shape.dims[4]
                                            + c * x_padded.shape.dims[2]
                                                * x_padded.shape.dims[3]
                                                * x_padded.shape.dims[4]
                                            + id * x_padded.shape.dims[3] * x_padded.shape.dims[4]
                                            + ih * x_padded.shape.dims[4]
                                            + iw;

                                        max_val = max_val.max(x_padded.data[idx]);
                                    }
                                }
                            }

                            output.push(max_val);
                        }
                    }
                }
            }
        }

        Tensor::new(
            output,
            Shape::new(vec![batch_size, channels, d_out, h_out, w_out]),
        )
    }
}

/// 3D Average Pooling layer
///
/// Reduces spatial dimensions by averaging values in local 3D windows.
#[derive(Debug)]
pub struct AvgPool3D {
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width)
    pub stride: (usize, usize, usize),
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
}

impl AvgPool3D {
    /// Create new 3D average pooling layer
    pub fn new(kernel_size: (usize, usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0, 0),
        }
    }

    /// Set stride
    pub fn with_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn with_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Forward pass through 3D average pooling
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.shape.rank(), 5, "Input must be 5D tensor (NCDHW)");

        let (batch_size, channels, d_in, h_in, w_in) = (
            x.shape.dims[0],
            x.shape.dims[1],
            x.shape.dims[2],
            x.shape.dims[3],
            x.shape.dims[4],
        );

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;

        // Output dimensions
        let d_out = (d_in + 2 * pd - kd) / sd + 1;
        let h_out = (h_in + 2 * ph - kh) / sh + 1;
        let w_out = (w_in + 2 * pw - kw) / sw + 1;

        let x_padded = pad3d(x, pd, ph, pw);
        let mut output = Vec::with_capacity(batch_size * channels * d_out * h_out * w_out);
        let window_size = (kd * kh * kw) as f64;

        for b in 0..batch_size {
            for c in 0..channels {
                for od in 0..d_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut sum_val = 0.0;

                            for kd_idx in 0..kd {
                                for kh_idx in 0..kh {
                                    for kw_idx in 0..kw {
                                        let id = od * sd + kd_idx;
                                        let ih = oh * sh + kh_idx;
                                        let iw = ow * sw + kw_idx;

                                        let idx = b
                                            * x_padded.shape.dims[1]
                                            * x_padded.shape.dims[2]
                                            * x_padded.shape.dims[3]
                                            * x_padded.shape.dims[4]
                                            + c * x_padded.shape.dims[2]
                                                * x_padded.shape.dims[3]
                                                * x_padded.shape.dims[4]
                                            + id * x_padded.shape.dims[3] * x_padded.shape.dims[4]
                                            + ih * x_padded.shape.dims[4]
                                            + iw;

                                        sum_val += x_padded.data[idx];
                                    }
                                }
                            }

                            output.push(sum_val / window_size);
                        }
                    }
                }
            }
        }

        Tensor::new(
            output,
            Shape::new(vec![batch_size, channels, d_out, h_out, w_out]),
        )
    }
}

// ============================================================================
// Deformable Convolution 2D
// ============================================================================

/// Deformable Convolution 2D layer
///
/// Augments standard convolution with learnable spatial sampling offsets.
/// Enables adaptive receptive fields for geometric transformations.
/// Used for object detection, semantic segmentation, and alignment tasks.
#[derive(Debug)]
pub struct DeformableConv2D {
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Number of offset parameters (2 * kernel_h * kernel_w)
    pub num_offsets: usize,
    /// Weight tensor: (out_channels, in_channels, kernel_h, kernel_w)
    pub weight: Tensor,
    /// Bias tensor: (out_channels,)
    pub bias: Tensor,
}

impl DeformableConv2D {
    /// Create new deformable convolution layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        let (kh, kw) = (kernel_size.1, kernel_size.2);
        let num_offsets = 2 * kh * kw;

        let weight_shape = Shape::new(vec![out_channels, in_channels, kh, kw]);

        // Glorot initialization
        let fan_in = (in_channels * kh * kw) as f64;
        let fan_out = (out_channels * kh * kw) as f64;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        let mut weight_data = Vec::with_capacity(out_channels * in_channels * kh * kw);
        for _ in 0..weight_shape.size() {
            weight_data.push((pseudo_random() * 2.0 - 1.0) * limit);
        }

        let weight = Tensor::new(weight_data, weight_shape);
        let bias = Tensor::zeros(Shape::new(vec![out_channels]));

        Self {
            in_channels,
            out_channels,
            kernel_size: (kh, kw),
            stride: (1, 1),
            padding: (0, 0),
            num_offsets,
            weight,
            bias,
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

    /// Generate sampling grid from offsets
    pub fn generate_grid(&self, offsets: &Tensor) -> Tensor {
        // offsets shape: (batch, 2 * kh * kw, out_h, out_w)
        let (batch, _, out_h, out_w) = (
            offsets.shape.dims[0],
            offsets.shape.dims[1],
            offsets.shape.dims[2],
            offsets.shape.dims[3],
        );

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // Generate base grid
        let mut grid_data = Vec::with_capacity(batch * 2 * kh * kw * out_h * out_w);

        for b in 0..batch {
            for kk in 0..(kh * kw) {
                let ky = (kk / kw) as isize - (kh / 2) as isize;
                let kx = (kk % kw) as isize - (kw / 2) as isize;

                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Base coordinates
                        let y_base = (oh * sh + ph) as f64;
                        let x_base = (ow * sw + pw) as f64;

                        // Apply offsets
                        let offset_y = offsets.data[b * offsets.shape.dims[1] * out_h * out_w
                            + (2 * kk) * out_h * out_w
                            + oh * out_w
                            + ow];
                        let offset_x = offsets.data[b * offsets.shape.dims[1] * out_h * out_w
                            + (2 * kk + 1) * out_h * out_w
                            + oh * out_w
                            + ow];

                        grid_data.push(y_base + ky as f64 + offset_y);
                        grid_data.push(x_base + kx as f64 + offset_x);
                    }
                }
            }
        }

        Tensor::new(grid_data, Shape::new(vec![batch, 2, kh * kw, out_h, out_w]))
    }

    /// Forward pass through deformable convolution
    pub fn forward(&self, x: &Tensor, offsets: &Tensor) -> Tensor {
        assert_eq!(x.shape.rank(), 4, "Input must be 4D tensor (NCHW)");
        assert_eq!(offsets.shape.rank(), 4, "Offsets must be 4D tensor");

        let (batch_size, in_ch, h_in, w_in) = (
            x.shape.dims[0],
            x.shape.dims[1],
            x.shape.dims[2],
            x.shape.dims[3],
        );

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // Output dimensions
        let h_out = (h_in + 2 * ph - kh) / sh + 1;
        let w_out = (w_in + 2 * pw - kw) / sw + 1;

        // Generate sampling grid
        let grid = self.generate_grid(offsets);

        // Padded input
        let x_padded = pad2d(x, ph, pw);

        let mut output = Vec::with_capacity(batch_size * self.out_channels * h_out * w_out);

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;

                        for ic in 0..in_ch {
                            for kk in 0..(kh * kw) {
                                // Get sampling location from grid
                                let grid_y = grid.data[b * 2 * kh * kw * h_out * w_out
                                    + 0 * kh * kw * h_out * w_out
                                    + kk * h_out * w_out
                                    + oh * w_out
                                    + ow];
                                let grid_x = grid.data[b * 2 * kh * kw * h_out * w_out
                                    + 1 * kh * kw * h_out * w_out
                                    + kk * h_out * w_out
                                    + oh * w_out
                                    + ow];

                                // Bilinear interpolation
                                let y0 = grid_y.floor() as isize;
                                let x0 = grid_x.floor() as isize;
                                let y1 = y0 + 1;
                                let x1 = x0 + 1;

                                let ay = grid_y - y0 as f64;
                                let ax = grid_x - x0 as f64;

                                // Sample with boundary checking
                                let sample =
                                    |x_padded: &Tensor, c: usize, y: isize, x: isize| -> f64 {
                                        if y >= 0
                                            && x >= 0
                                            && y < x_padded.shape.dims[2] as isize
                                            && x < x_padded.shape.dims[3] as isize
                                        {
                                            let idx = b
                                                * x_padded.shape.dims[1]
                                                * x_padded.shape.dims[2]
                                                * x_padded.shape.dims[3]
                                                + c * x_padded.shape.dims[2]
                                                    * x_padded.shape.dims[3]
                                                + y as usize * x_padded.shape.dims[3]
                                                + x as usize;
                                            x_padded.data[idx]
                                        } else {
                                            0.0
                                        }
                                    };

                                let v00 = sample(&x_padded, ic, y0, x0);
                                let v01 = sample(&x_padded, ic, y0, x1);
                                let v10 = sample(&x_padded, ic, y1, x0);
                                let v11 = sample(&x_padded, ic, y1, x1);

                                // Bilinear interpolation
                                let interp_y0 = v00 * (1.0 - ax) + v01 * ax;
                                let interp_y1 = v10 * (1.0 - ax) + v11 * ax;
                                let interp = interp_y0 * (1.0 - ay) + interp_y1 * ay;

                                let w_idx = oc
                                    * self.weight.shape.dims[1]
                                    * self.weight.shape.dims[2]
                                    * self.weight.shape.dims[3]
                                    + ic * kh * kw
                                    + kk;

                                sum += interp * self.weight.data[w_idx];
                            }
                        }

                        output.push(sum + self.bias.data[oc]);
                    }
                }
            }
        }

        Tensor::new(
            output,
            Shape::new(vec![batch_size, self.out_channels, h_out, w_out]),
        )
    }
}

// ============================================================================
// Deformable Pooling
// ============================================================================

/// Deformable Pooling layer
///
/// Applies pooling with learnable spatial offsets for adaptive receptive fields.
/// Used for ROI pooling, object detection, and spatially-aware feature aggregation.
#[derive(Debug)]
pub struct DeformablePooling {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Number of offset parameters (2 * kernel_h * kernel_w)
    pub num_offsets: usize,
    /// Pool type: max or avg
    pub pool_type: PoolType,
}

/// Pooling type for deformable pooling
#[derive(Debug, Clone, Copy)]
pub enum PoolType {
    Max,
    Avg,
}

impl DeformablePooling {
    /// Create new deformable pooling layer
    pub fn new(kernel_size: (usize, usize), pool_type: PoolType) -> Self {
        let (kh, kw) = kernel_size;
        let num_offsets = 2 * kh * kw;

        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
            num_offsets,
            pool_type,
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

    /// Generate sampling grid (similar to DeformableConv2D)
    pub fn generate_grid(&self, offsets: &Tensor) -> Tensor {
        let (batch, _, out_h, out_w) = (
            offsets.shape.dims[0],
            offsets.shape.dims[1],
            offsets.shape.dims[2],
            offsets.shape.dims[3],
        );

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let mut grid_data = Vec::with_capacity(batch * 2 * kh * kw * out_h * out_w);

        for b in 0..batch {
            for kk in 0..(kh * kw) {
                let ky = (kk / kw) as isize - (kh / 2) as isize;
                let kx = (kk % kw) as isize - (kw / 2) as isize;

                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let y_base = (oh * sh + ph) as f64;
                        let x_base = (ow * sw + pw) as f64;

                        let offset_y = offsets.data[b * offsets.shape.dims[1] * out_h * out_w
                            + (2 * kk) * out_h * out_w
                            + oh * out_w
                            + ow];
                        let offset_x = offsets.data[b * offsets.shape.dims[1] * out_h * out_w
                            + (2 * kk + 1) * out_h * out_w
                            + oh * out_w
                            + ow];

                        grid_data.push(y_base + ky as f64 + offset_y);
                        grid_data.push(x_base + kx as f64 + offset_x);
                    }
                }
            }
        }

        Tensor::new(grid_data, Shape::new(vec![batch, 2, kh * kw, out_h, out_w]))
    }

    /// Forward pass through deformable pooling
    pub fn forward(&self, x: &Tensor, offsets: &Tensor) -> Tensor {
        assert_eq!(x.shape.rank(), 4, "Input must be 4D tensor (NCHW)");

        let (batch_size, channels, h_in, w_in) = (
            x.shape.dims[0],
            x.shape.dims[1],
            x.shape.dims[2],
            x.shape.dims[3],
        );

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // Output dimensions
        let h_out = (h_in + 2 * ph - kh) / sh + 1;
        let w_out = (w_in + 2 * pw - kw) / sw + 1;

        // Generate sampling grid
        let grid = self.generate_grid(offsets);

        // Padded input
        let x_padded = pad2d(x, ph, pw);

        let mut output = Vec::with_capacity(batch_size * channels * h_out * w_out);

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        match self.pool_type {
                            PoolType::Max => {
                                let mut max_val = f64::NEG_INFINITY;

                                for kk in 0..(kh * kw) {
                                    let grid_y = grid.data[b * 2 * kh * kw * h_out * w_out
                                        + 0 * kh * kw * h_out * w_out
                                        + kk * h_out * w_out
                                        + oh * w_out
                                        + ow];
                                    let grid_x = grid.data[b * 2 * kh * kw * h_out * w_out
                                        + 1 * kh * kw * h_out * w_out
                                        + kk * h_out * w_out
                                        + oh * w_out
                                        + ow];

                                    // Bilinear interpolation
                                    let y0 = grid_y.floor() as isize;
                                    let x0 = grid_x.floor() as isize;
                                    let y1 = y0 + 1;
                                    let x1 = x0 + 1;

                                    let ay = grid_y - y0 as f64;
                                    let ax = grid_x - x0 as f64;

                                    let sample = |y: isize, x: isize| -> f64 {
                                        if y >= 0
                                            && x >= 0
                                            && y < x_padded.shape.dims[2] as isize
                                            && x < x_padded.shape.dims[3] as isize
                                        {
                                            let idx = b
                                                * channels
                                                * x_padded.shape.dims[2]
                                                * x_padded.shape.dims[3]
                                                + c * x_padded.shape.dims[2]
                                                    * x_padded.shape.dims[3]
                                                + y as usize * x_padded.shape.dims[3]
                                                + x as usize;
                                            x_padded.data[idx]
                                        } else {
                                            f64::NEG_INFINITY
                                        }
                                    };

                                    let v00 = sample(y0, x0);
                                    let v01 = sample(y0, x1);
                                    let v10 = sample(y1, x0);
                                    let v11 = sample(y1, x1);

                                    let interp_y0 = if v00.is_finite() && v01.is_finite() {
                                        v00 * (1.0 - ax) + v01 * ax
                                    } else if v00.is_finite() {
                                        v00
                                    } else {
                                        v01
                                    };
                                    let interp_y1 = if v10.is_finite() && v11.is_finite() {
                                        v10 * (1.0 - ax) + v11 * ax
                                    } else if v10.is_finite() {
                                        v10
                                    } else {
                                        v11
                                    };
                                    let interp = interp_y0 * (1.0 - ay) + interp_y1 * ay;

                                    max_val = max_val.max(interp);
                                }

                                output.push(max_val);
                            }
                            PoolType::Avg => {
                                let mut sum_val = 0.0;
                                let mut count = 0.0;

                                for kk in 0..(kh * kw) {
                                    let grid_y = grid.data[b * 2 * kh * kw * h_out * w_out
                                        + 0 * kh * kw * h_out * w_out
                                        + kk * h_out * w_out
                                        + oh * w_out
                                        + ow];
                                    let grid_x = grid.data[b * 2 * kh * kw * h_out * w_out
                                        + 1 * kh * kw * h_out * w_out
                                        + kk * h_out * w_out
                                        + oh * w_out
                                        + ow];

                                    let y0 = grid_y.floor() as isize;
                                    let x0 = grid_x.floor() as isize;
                                    let y1 = y0 + 1;
                                    let x1 = x0 + 1;

                                    let ay = grid_y - y0 as f64;
                                    let ax = grid_x - x0 as f64;

                                    let sample = |y: isize, x: isize| -> f64 {
                                        if y >= 0
                                            && x >= 0
                                            && y < x_padded.shape.dims[2] as isize
                                            && x < x_padded.shape.dims[3] as isize
                                        {
                                            let idx = b
                                                * channels
                                                * x_padded.shape.dims[2]
                                                * x_padded.shape.dims[3]
                                                + c * x_padded.shape.dims[2]
                                                    * x_padded.shape.dims[3]
                                                + y as usize * x_padded.shape.dims[3]
                                                + x as usize;
                                            x_padded.data[idx]
                                        } else {
                                            0.0
                                        }
                                    };

                                    let v00 = sample(y0, x0);
                                    let v01 = sample(y0, x1);
                                    let v10 = sample(y1, x0);
                                    let v11 = sample(y1, x1);

                                    let interp_y0 = v00 * (1.0 - ax) + v01 * ax;
                                    let interp_y1 = v10 * (1.0 - ax) + v11 * ax;
                                    let interp = interp_y0 * (1.0 - ay) + interp_y1 * ay;

                                    sum_val += interp;
                                    count += 1.0;
                                }

                                output.push(sum_val / count);
                            }
                        }
                    }
                }
            }
        }

        Tensor::new(output, Shape::new(vec![batch_size, channels, h_out, w_out]))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// 3D padding helper
fn pad3d(x: &Tensor, pd: usize, ph: usize, pw: usize) -> Tensor {
    let (batch, channels, d, h, w) = (
        x.shape.dims[0],
        x.shape.dims[1],
        x.shape.dims[2],
        x.shape.dims[3],
        x.shape.dims[4],
    );

    let d_pad = d + 2 * pd;
    let h_pad = h + 2 * ph;
    let w_pad = w + 2 * pw;

    let mut padded = vec![0.0; batch * channels * d_pad * h_pad * w_pad];

    for b in 0..batch {
        for c in 0..channels {
            for id in 0..d {
                for ih in 0..h {
                    for iw in 0..w {
                        let src_idx =
                            b * channels * d * h * w + c * d * h * w + id * h * w + ih * w + iw;

                        let dst_idx = b * channels * d_pad * h_pad * w_pad
                            + c * d_pad * h_pad * w_pad
                            + (id + pd) * h_pad * w_pad
                            + (ih + ph) * w_pad
                            + (iw + pw);

                        padded[dst_idx] = x.data[src_idx];
                    }
                }
            }
        }
    }

    Tensor::new(
        padded,
        Shape::new(vec![batch, channels, d_pad, h_pad, w_pad]),
    )
}

/// 2D padding helper
fn pad2d(x: &Tensor, ph: usize, pw: usize) -> Tensor {
    let (batch, channels, h, w) = (
        x.shape.dims[0],
        x.shape.dims[1],
        x.shape.dims[2],
        x.shape.dims[3],
    );

    let h_pad = h + 2 * ph;
    let w_pad = w + 2 * pw;

    let mut padded = vec![0.0; batch * channels * h_pad * w_pad];

    for b in 0..batch {
        for c in 0..channels {
            for ih in 0..h {
                for iw in 0..w {
                    let src_idx = b * channels * h * w + c * h * w + ih * w + iw;

                    let dst_idx = b * channels * h_pad * w_pad
                        + c * h_pad * w_pad
                        + (ih + ph) * w_pad
                        + (iw + pw);

                    padded[dst_idx] = x.data[src_idx];
                }
            }
        }
    }

    Tensor::new(padded, Shape::new(vec![batch, channels, h_pad, w_pad]))
}

/// Pseudo-random number generator (0-1)
fn pseudo_random() -> f64 {
    // Simple LCG for reproducibility
    static mut STATE: u64 = 123456789;
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;

    unsafe {
        STATE = STATE.wrapping_mul(A).wrapping_add(C);
        (STATE >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Box-Muller transform for normal distribution
fn box_muller() -> f64 {
    let u1 = pseudo_random();
    let u2 = pseudo_random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv3d_basic() {
        let conv = Conv3D::new(3, 16, (3, 3, 3));

        // Input: (batch=2, channels=3, depth=10, height=10, width=10)
        let x = Tensor::rand(Shape::new(vec![2, 3, 10, 10, 10]));

        let output = conv.forward(&x);

        // Output shape: (batch=2, channels=16, depth=8, height=8, width=8)
        assert_eq!(output.shape.dims, vec![2, 16, 8, 8, 8]);
    }

    #[test]
    fn test_conv3d_with_stride_padding() {
        let conv = Conv3D::new(4, 8, (3, 3, 3))
            .with_stride((2, 2, 2))
            .with_padding((1, 1, 1));

        let x = Tensor::rand(Shape::new(vec![1, 4, 16, 16, 16]));
        let output = conv.forward(&x);

        // With stride 2 and padding 1: (16 - 3 + 2) / 2 + 1 = 8
        assert_eq!(output.shape.dims, vec![1, 8, 8, 8, 8]);
    }

    #[test]
    fn test_conv3d_transpose_basic() {
        let conv_t = Conv3DTranspose::new(16, 3, (3, 3, 3))
            .with_stride((2, 2, 2))
            .with_padding((1, 1, 1));

        let x = Tensor::rand(Shape::new(vec![1, 16, 8, 8, 8]));
        let output = conv_t.forward(&x);

        // Upsampling: (8 - 1) * 2 - 2 * 1 + 3 = 15
        assert_eq!(output.shape.dims[0], 1);
        assert_eq!(output.shape.dims[1], 3);
    }

    #[test]
    fn test_maxpool3d_basic() {
        let pool = MaxPool3D::new((2, 2, 2));

        let x = Tensor::rand(Shape::new(vec![2, 4, 16, 16, 16]));
        let output = pool.forward(&x);

        // Output: (16 - 2) / 2 + 1 = 8
        assert_eq!(output.shape.dims, vec![2, 4, 8, 8, 8]);
    }

    #[test]
    fn test_avgpool3d_basic() {
        let pool = AvgPool3D::new((2, 2, 2));

        let x = Tensor::rand(Shape::new(vec![2, 4, 16, 16, 16]));
        let output = pool.forward(&x);

        assert_eq!(output.shape.dims, vec![2, 4, 8, 8, 8]);
    }

    #[test]
    fn test_deformable_conv2d_basic() {
        let deform_conv = DeformableConv2D::new(3, 16, (1, 3, 3));

        let x = Tensor::rand(Shape::new(vec![1, 3, 32, 32]));

        // Offsets: (batch=1, 2*3*3=18, out_h=30, out_w=30)
        let offsets = Tensor::rand(Shape::new(vec![1, 18, 30, 30]));

        let output = deform_conv.forward(&x, &offsets);

        assert_eq!(output.shape.dims, vec![1, 16, 30, 30]);
    }

    #[test]
    fn test_deformable_pooling_max() {
        let deform_pool = DeformablePooling::new((2, 2), PoolType::Max);

        let x = Tensor::rand(Shape::new(vec![1, 4, 16, 16]));

        // Offsets: (batch=1, 2*2*2=8, out_h=8, out_w=8)
        let offsets = Tensor::rand(Shape::new(vec![1, 8, 8, 8]));

        let output = deform_pool.forward(&x, &offsets);

        assert_eq!(output.shape.dims, vec![1, 4, 8, 8]);
    }

    #[test]
    fn test_deformable_pooling_avg() {
        let deform_pool = DeformablePooling::new((2, 2), PoolType::Avg);

        let x = Tensor::rand(Shape::new(vec![1, 4, 16, 16]));
        let offsets = Tensor::rand(Shape::new(vec![1, 8, 8, 8]));

        let output = deform_pool.forward(&x, &offsets);

        assert_eq!(output.shape.dims, vec![1, 4, 8, 8]);
    }

    #[test]
    fn test_deformable_generate_grid() {
        let deform_conv = DeformableConv2D::new(3, 16, (1, 3, 3));

        let offsets = Tensor::rand(Shape::new(vec![1, 18, 4, 4]));
        let grid = deform_conv.generate_grid(&offsets);

        // Grid shape: (batch=1, 2, 3*3=9, out_h=4, out_w=4)
        assert_eq!(grid.shape.dims, vec![1, 2, 9, 4, 4]);
    }

    #[test]
    fn test_conv3d_parameter_validation() {
        let conv = Conv3D::new(4, 8, (3, 3, 3));

        assert_eq!(conv.in_channels, 4);
        assert_eq!(conv.out_channels, 8);
        assert_eq!(conv.kernel_size, (3, 3, 3));
        assert_eq!(conv.stride, (1, 1, 1));
        assert_eq!(conv.padding, (0, 0, 0));
    }

    #[test]
    fn test_3d_operations_with_padding() {
        let pool = MaxPool3D::new((3, 3, 3))
            .with_stride((1, 1, 1))
            .with_padding((1, 1, 1));

        let x = Tensor::rand(Shape::new(vec![1, 2, 10, 10, 10]));
        let output = pool.forward(&x);

        // With padding 1 and stride 1: (10 + 2 - 3) / 1 + 1 = 10
        assert_eq!(output.shape.dims, vec![1, 2, 10, 10, 10]);
    }
}
