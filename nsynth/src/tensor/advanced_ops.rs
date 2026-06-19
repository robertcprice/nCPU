//! Advanced Tensor Operations for nCPU/nSynth
//!
//! Concatenation, splitting, indexing, padding, advanced convolutions.

use super::ops::{Shape, Tensor};

/// Concatenate tensors along specified dimension
pub fn concat(tensors: &[&Tensor], dim: usize) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("Cannot concat empty tensor list".to_string());
    }

    let first = &tensors[0];
    let rank = first.shape.rank();

    if dim >= rank {
        return Err(format!("Dim {} out of bounds for rank {}", dim, rank));
    }

    // Verify all tensors have same shape except concat dimension
    for t in tensors {
        if t.shape.rank() != rank {
            return Err("All tensors must have same rank".to_string());
        }
        for (i, (&d1, &d2)) in first.shape.dims.iter().zip(t.shape.dims.iter()).enumerate() {
            if i != dim && d1 != d2 {
                return Err("Tensor shapes don't match for concat".to_string());
            }
        }
    }

    // Calculate output shape
    let mut out_dims = first.shape.dims.clone();
    out_dims[dim] = tensors.iter().map(|t| t.shape.dims[dim]).sum();

    let mut out_data = Vec::with_capacity(first.shape.size() * tensors.len());

    // Simple concatenation for 1D and 2D
    if rank == 1 {
        for t in tensors {
            out_data.extend_from_slice(&t.data);
        }
    } else if rank == 2 {
        if dim == 0 {
            // Concat rows: stack vertically
            for t in tensors {
                out_data.extend_from_slice(&t.data);
            }
        } else {
            // Concat columns: interleave rows
            let cols = out_dims[1];
            let mut row_idx = 0;
            let max_rows = first.shape.dims[0];

            for row in 0..max_rows {
                for t in tensors {
                    let t_cols = t.shape.dims[1];
                    for col in 0..t_cols {
                        out_data.push(t.data[row * t_cols + col]);
                    }
                }
            }
        }
    } else {
        return Err("Concat for rank > 2 not implemented".to_string());
    }

    Ok(Tensor::new(out_data, Shape::new(out_dims)))
}

/// Stack tensors along new dimension
pub fn stack(tensors: &[&Tensor], dim: usize) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("Cannot stack empty tensor list".to_string());
    }

    let first = &tensors[0];

    // All tensors must have identical shape
    for t in tensors {
        if t.shape != first.shape {
            return Err("All tensors must have same shape for stack".to_string());
        }
    }

    // Output rank = input rank + 1
    let mut out_dims = first.shape.dims.clone();
    out_dims.insert(dim, tensors.len());

    let mut out_data = vec![0.0; out_dims.iter().product()];
    let elem_size = first.shape.size();

    for (i, t) in tensors.iter().enumerate() {
        let offset = i * elem_size;
        out_data[offset..offset + elem_size].copy_from_slice(&t.data);
    }

    Ok(Tensor::new(out_data, Shape::new(out_dims)))
}

/// Split tensor into chunks along dimension
pub fn split(tensor: &Tensor, dim: usize, chunk_sizes: &[usize]) -> Result<Vec<Tensor>, String> {
    if dim >= tensor.shape.rank() {
        return Err("Dim out of bounds".to_string());
    }

    let dim_size = tensor.shape.dims[dim];
    let total: usize = chunk_sizes.iter().sum();

    if total != dim_size {
        return Err(format!(
            "Chunk sizes sum {} doesn't match dim size {}",
            total, dim_size
        ));
    }

    let mut results = Vec::new();
    let mut offset = 0;

    for &chunk_size in chunk_sizes {
        let mut out_dims = tensor.shape.dims.clone();
        out_dims[dim] = chunk_size;

        let mut out_data = Vec::with_capacity(tensor.shape.size() / dim_size * chunk_size);

        if tensor.shape.rank() == 1 {
            out_data.extend_from_slice(&tensor.data[offset..offset + chunk_size]);
        } else if tensor.shape.rank() == 2 {
            if dim == 0 {
                // Split rows
                let cols = tensor.shape.dims[1];
                out_data
                    .extend_from_slice(&tensor.data[offset * cols..(offset + chunk_size) * cols]);
            } else {
                // Split columns
                let rows = tensor.shape.dims[0];
                let cols_before = offset;
                let cols_chunk = chunk_size;
                for row in 0..rows {
                    let row_start = row * dim_size;
                    out_data.extend_from_slice(
                        &tensor.data[row_start + cols_before..row_start + cols_before + cols_chunk],
                    );
                }
            }
        } else {
            return Err("Split for rank > 2 not implemented".to_string());
        }

        results.push(Tensor::new(out_data, Shape::new(out_dims)));
        offset += chunk_size;
    }

    Ok(results)
}

/// Chunk tensor into n equal parts
pub fn chunk(tensor: &Tensor, dim: usize, n: usize) -> Result<Vec<Tensor>, String> {
    let dim_size = tensor.shape.dims[dim];
    let chunk_size = dim_size / n;
    let remainder = dim_size % n;

    let mut chunk_sizes = vec![chunk_size; n];
    for i in 0..remainder {
        chunk_sizes[n - 1 - i] += 1;
    }

    split(tensor, dim, &chunk_sizes)
}

/// Gather elements along axis
pub fn gather(tensor: &Tensor, dim: usize, indices: &[usize]) -> Result<Tensor, String> {
    if dim >= tensor.shape.rank() {
        return Err("Dim out of bounds".to_string());
    }

    let dim_size = tensor.shape.dims[dim];

    for &idx in indices {
        if idx >= dim_size {
            return Err(format!("Index {} out of bounds for dim {}", idx, dim_size));
        }
    }

    let mut out_dims = tensor.shape.dims.clone();
    out_dims[dim] = indices.len();

    let mut out_data = Vec::with_capacity(tensor.shape.size());

    if tensor.shape.rank() == 1 {
        for &idx in indices {
            out_data.push(tensor.data[idx]);
        }
    } else if tensor.shape.rank() == 2 {
        let cols = tensor.shape.dims[1];
        if dim == 0 {
            // Gather rows
            for &idx in indices {
                out_data.extend_from_slice(&tensor.data[idx * cols..(idx + 1) * cols]);
            }
        } else {
            // Gather columns
            let rows = tensor.shape.dims[0];
            for row in 0..rows {
                for &idx in indices {
                    out_data.push(tensor.data[row * cols + idx]);
                }
            }
        }
    } else {
        return Err("Gather for rank > 2 not implemented".to_string());
    }

    Ok(Tensor::new(out_data, Shape::new(out_dims)))
}

/// Index tensor with advanced indexing
pub fn index_select(tensor: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor, String> {
    let idx_data: Vec<usize> = indices.data.iter().map(|&x| x as usize).collect();
    gather(tensor, dim, &idx_data)
}

/// Pad tensor with specified padding
pub fn pad(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Tensor, String> {
    if padding.len() != tensor.shape.rank() {
        return Err("Padding must match tensor rank".to_string());
    }

    let mut out_dims = Vec::with_capacity(tensor.shape.rank());
    let mut offsets = Vec::with_capacity(tensor.shape.rank());

    for (i, &(before, after)) in padding.iter().enumerate() {
        out_dims.push(tensor.shape.dims[i] + before + after);
        offsets.push(before);
    }

    let mut out_data = vec![0.0; out_dims.iter().product()];

    if tensor.shape.rank() == 1 {
        let before = padding[0].0;
        out_data[before..before + tensor.data.len()].copy_from_slice(&tensor.data);
    } else if tensor.shape.rank() == 2 {
        let (pad_h_before, _pad_h_after) = padding[0];
        let (pad_w_before, _pad_w_after) = padding[1];
        let (h, w) = (tensor.shape.dims[0], tensor.shape.dims[1]);
        let (_out_h, out_w) = (out_dims[0], out_dims[1]);

        for i in 0..h {
            let out_row = pad_h_before + i;
            for j in 0..w {
                let out_col = pad_w_before + j;
                out_data[out_row * out_w + out_col] = tensor.data[i * w + j];
            }
        }
    } else {
        return Err("Pad for rank > 2 not implemented".to_string());
    }

    Ok(Tensor::new(out_data, Shape::new(out_dims)))
}

/// Upsample tensor using nearest neighbor
pub fn upsample_nearest(tensor: &Tensor, scale_factor: (usize, usize)) -> Result<Tensor, String> {
    if tensor.shape.rank() != 2 {
        return Err("Upsample requires 2D tensor".to_string());
    }

    let (h, w) = (tensor.shape.dims[0], tensor.shape.dims[1]);
    let (scale_h, scale_w) = scale_factor;

    let h_out = h * scale_h;
    let w_out = w * scale_w;

    let mut out_data = vec![0.0; h_out * w_out];

    for i in 0..h_out {
        for j in 0..w_out {
            let src_i = i / scale_h;
            let src_j = j / scale_w;
            out_data[i * w_out + j] = tensor.data[src_i * w + src_j];
        }
    }

    Ok(Tensor::new(out_data, Shape::new(vec![h_out, w_out])))
}

/// Upsample tensor using bilinear interpolation
pub fn upsample_bilinear(tensor: &Tensor, scale_factor: (usize, usize)) -> Result<Tensor, String> {
    if tensor.shape.rank() != 2 {
        return Err("Upsample requires 2D tensor".to_string());
    }

    let (h, w) = (tensor.shape.dims[0], tensor.shape.dims[1]);
    let (scale_h, scale_w) = scale_factor;

    let h_out = h * scale_h;
    let w_out = w * scale_w;

    let mut out_data = vec![0.0; h_out * w_out];

    for i in 0..h_out {
        for j in 0..w_out {
            let src_y = (i as f64) / (scale_h as f64);
            let src_x = (j as f64) / (scale_w as f64);

            let y0 = src_y.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);

            let dy = src_y - y0 as f64;
            let dx = src_x - x0 as f64;

            let v00 = tensor.data[y0 * w + x0];
            let v01 = tensor.data[y0 * w + x1];
            let v10 = tensor.data[y1 * w + x0];
            let v11 = tensor.data[y1 * w + x1];

            let top = v00 * (1.0 - dx) + v01 * dx;
            let bottom = v10 * (1.0 - dx) + v11 * dx;
            out_data[i * w_out + j] = top * (1.0 - dy) + bottom * dy;
        }
    }

    Ok(Tensor::new(out_data, Shape::new(vec![h_out, w_out])))
}

/// Dilated convolution
pub fn conv2d_dilated(
    input: &Tensor,
    kernel: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor, String> {
    if input.shape.rank() != 2 || kernel.shape.rank() != 2 {
        return Err("Conv2D requires 2D tensors".to_string());
    }

    let (h, w) = (input.shape.dims[0], input.shape.dims[1]);
    let (kh, kw) = (kernel.shape.dims[0], kernel.shape.dims[1]);

    // Effective kernel size with dilation
    let kh_eff = (kh - 1) * dilation.0 + 1;
    let kw_eff = (kw - 1) * dilation.1 + 1;

    let h_out = (h + 2 * padding.0 - kh_eff) / stride.0 + 1;
    let w_out = (w + 2 * padding.1 - kw_eff) / stride.1 + 1;

    let mut result = vec![0.0; h_out * w_out];

    for oh in 0..h_out {
        for ow in 0..w_out {
            let mut sum = 0.0;

            for kh_idx in 0..kh {
                for kw_idx in 0..kw {
                    let ih = oh * stride.0 + kh_idx * dilation.0 - padding.0;
                    let iw = ow * stride.1 + kw_idx * dilation.1 - padding.1;

                    if ih < h && iw < w {
                        let input_val = input.data[ih * w + iw];
                        let kernel_val = kernel.data[kh_idx * kw + kw_idx];
                        sum += input_val * kernel_val;
                    }
                }
            }

            result[oh * w_out + ow] = sum;
        }
    }

    Ok(Tensor::new(result, Shape::new(vec![h_out, w_out])))
}

/// Transposed convolution (aka deconvolution)
pub fn conv2d_transpose(
    input: &Tensor,
    kernel: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
) -> Result<Tensor, String> {
    if input.shape.rank() != 2 || kernel.shape.rank() != 2 {
        return Err("Conv2DTranspose requires 2D tensors".to_string());
    }

    let (h_in, w_in) = (input.shape.dims[0], input.shape.dims[1]);
    let (kh, kw) = (kernel.shape.dims[0], kernel.shape.dims[1]);

    let h_out = (h_in - 1) * stride.0 - 2 * padding.0 + kh + output_padding.0;
    let w_out = (w_in - 1) * stride.1 - 2 * padding.1 + kw + output_padding.1;

    let mut result = vec![0.0; h_out * w_out];

    for ih in 0..h_in {
        for iw in 0..w_in {
            let val = input.data[ih * w_in + iw];

            for kh_idx in 0..kh {
                for kw_idx in 0..kw {
                    let oh = ih * stride.0 + kh_idx - padding.0;
                    let ow = iw * stride.1 + kw_idx - padding.1;

                    if oh < h_out && ow < w_out {
                        result[oh * w_out + ow] += val * kernel.data[kh_idx * kw + kw_idx];
                    }
                }
            }
        }
    }

    Ok(Tensor::new(result, Shape::new(vec![h_out, w_out])))
}

/// Depthwise separable convolution
pub fn conv2d_depthwise_separable(
    input: &Tensor,
    depthwise_kernel: &Tensor,
    pointwise_kernel: &Tensor,
    stride: (usize, usize),
) -> Result<Tensor, String> {
    // Depthwise convolution (spatial)
    let depthwise_out = input.conv2d(depthwise_kernel, stride, (0, 0))?;

    // Pointwise convolution (1x1 cross-channel)
    // Simplified: assumes proper shape handling
    pointwise_kernel.matmul(&depthwise_out.transpose().unwrap())
}

/// Grouped convolution
pub fn conv2d_grouped(
    input: &Tensor,
    kernel: &Tensor,
    groups: usize,
    stride: (usize, usize),
) -> Result<Tensor, String> {
    if input.shape.rank() != 2 || kernel.shape.rank() != 2 {
        return Err("Conv2D requires 2D tensors".to_string());
    }

    let (h, w) = (input.shape.dims[0], input.shape.dims[1]);
    let (kh, kw) = (kernel.shape.dims[0], kernel.shape.dims[1]);

    let channels_per_group = w / groups;
    let kh_per_group = kh / groups;

    let h_out = (h - kh) / stride.0 + 1;
    let w_out = (w - kw) / stride.1 + 1;

    let mut result = vec![0.0; h_out * w_out];

    for oh in 0..h_out {
        for ow in 0..w_out {
            let mut sum = 0.0;

            for g in 0..groups {
                let group_offset_in = g * channels_per_group;
                let group_offset_k = g * kh_per_group;

                for kh_idx in 0..kh_per_group {
                    for kw_idx in 0..kw {
                        let ih = oh * stride.0 + kh_idx;
                        let iw = ow * stride.1 + kw_idx;

                        if ih < h && iw < w {
                            let input_val = input.data[ih * w + group_offset_in + iw];
                            let kernel_val = kernel.data[group_offset_k + kh_idx * kw + kw_idx];
                            sum += input_val * kernel_val;
                        }
                    }
                }
            }

            result[oh * w_out + ow] = sum;
        }
    }

    Ok(Tensor::new(result, Shape::new(vec![h_out, w_out])))
}

/// Flatten tensor to 1D
pub fn flatten(tensor: &Tensor) -> Tensor {
    Tensor::vector(tensor.data.clone())
}

/// Squeeze dimension (remove size-1 dimensions)
pub fn squeeze(tensor: &Tensor, dim: usize) -> Result<Tensor, String> {
    if dim >= tensor.shape.rank() {
        return Err("Dim out of bounds".to_string());
    }

    if tensor.shape.dims[dim] != 1 {
        return Ok(tensor.clone()); // Cannot squeeze non-unit dimension
    }

    let mut out_dims = tensor.shape.dims.clone();
    out_dims.remove(dim);

    Ok(Tensor::new(tensor.data.clone(), Shape::new(out_dims)))
}

/// Unsqueeze dimension (add size-1 dimension)
pub fn unsqueeze(tensor: &Tensor, dim: usize) -> Result<Tensor, String> {
    if dim > tensor.shape.rank() {
        return Err("Dim out of bounds".to_string());
    }

    let mut out_dims = tensor.shape.dims.clone();
    out_dims.insert(dim, 1);

    Ok(Tensor::new(tensor.data.clone(), Shape::new(out_dims)))
}

/// Permute dimensions
pub fn permute(tensor: &Tensor, dims: &[usize]) -> Result<Tensor, String> {
    if dims.len() != tensor.shape.rank() {
        return Err("Permutation dims must match tensor rank".to_string());
    }

    let mut seen = vec![false; tensor.shape.rank()];
    for &d in dims {
        if d >= tensor.shape.rank() {
            return Err("Invalid dim in permutation".to_string());
        }
        if seen[d] {
            return Err("Dim repeated in permutation".to_string());
        }
        seen[d] = true;
    }

    let mut out_dims = Vec::with_capacity(dims.len());
    for &d in dims {
        out_dims.push(tensor.shape.dims[d]);
    }

    // For 2D transpose: permute([1, 0])
    if tensor.shape.rank() == 2 && dims == &[1, 0] {
        return tensor.transpose();
    }

    Err("Permute for >2D not implemented".to_string())
}

/// Batch matrix multiplication for 3D tensors
pub fn bmm(batch1: &Tensor, batch2: &Tensor) -> Result<Tensor, String> {
    if batch1.shape.rank() != 3 || batch2.shape.rank() != 3 {
        return Err("bmm requires 3D tensors".to_string());
    }

    let (batch_size, m, k1) = (
        batch1.shape.dims[0],
        batch1.shape.dims[1],
        batch1.shape.dims[2],
    );
    let (_, k2, n) = (
        batch2.shape.dims[0],
        batch2.shape.dims[1],
        batch2.shape.dims[2],
    );

    if batch_size != batch2.shape.dims[0] || k1 != k2 {
        return Err("Incompatible shapes for bmm".to_string());
    }

    let mut result = vec![0.0; batch_size * m * n];

    for b in 0..batch_size {
        let batch1_offset = b * m * k1;
        let batch2_offset = b * k2 * n;
        let result_offset = b * m * n;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += batch1.data[batch1_offset + i * k1 + k]
                        * batch2.data[batch2_offset + k * n + j];
                }
                result[result_offset + i * n + j] = sum;
            }
        }
    }

    Ok(Tensor::new(result, Shape::new(vec![batch_size, m, n])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_1d() {
        let a = Tensor::vector(vec![1.0, 2.0]);
        let b = Tensor::vector(vec![3.0, 4.0]);
        let tensors: Vec<&Tensor> = vec![&a, &b];
        let c = concat(&tensors, 0).unwrap();
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_split() {
        let t = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let sizes: &[usize] = &[2, 2];
        let parts = split(&t, 0, sizes).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].data, vec![1.0, 2.0]);
        assert_eq!(parts[1].data, vec![3.0, 4.0]);
    }

    #[test]
    fn test_gather() {
        let t = Tensor::vector(vec![10.0, 20.0, 30.0, 40.0]);
        let indices: &[usize] = &[1, 3];
        let g = gather(&t, 0, indices).unwrap();
        assert_eq!(g.data, vec![20.0, 40.0]);
    }

    #[test]
    fn test_pad() {
        let t = Tensor::vector(vec![1.0, 2.0]);
        let padding: &[(usize, usize)] = &[(1, 2)];
        let p = pad(&t, padding).unwrap();
        assert_eq!(p.data, vec![0.0, 1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_maxpool2d() {
        use super::super::layers::MaxPool2d;
        let x = Tensor::matrix(vec![1.0, 3.0, 2.0, 4.0], 2, 2);
        let pool = MaxPool2d::new((2, 2));
        let out = pool.forward(&x);
        assert_eq!(out.data[0], 4.0);
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let flat = flatten(&t);
        assert_eq!(flat.shape.dims, vec![4]);
    }
}
