//! Neural Display Kernel — runs trained glyph MLP + color palette on Metal GPU
//!
//! Three-pass architecture (eliminates GPU thread-local memory overflow):
//!
//!   Pass 1: char_code → Embedding(256,64) → Linear(64,256)+GELU → h1_buf [device memory]
//!   Pass 2: h1_buf → Linear(256,256)+GELU → h2_buf [device memory]
//!   Pass 3: h2_buf → Linear(256,128)+Sigmoid → alpha → blend(fg,bg) → pixels
//!
//! Each pass uses ZERO thread-local arrays — all intermediate results go through
//! device-memory MTLBuffers (~1.9 MB each).  This avoids Apple Silicon's per-thread
//! stack limit (~1 KB) which caused NaN corruption with the original single-pass
//! approach that stored h1[256] in thread-local memory.
//!
//! Weight layout in the weights buffer (131,760 f32):
//!   [0        .. 16383  ]  embed.weight   [256, 64]   row-major
//!   [16384    .. 32767  ]  net.0.weight   [256, 64]   (FC1)
//!   [32768    .. 33023  ]  net.0.bias     [256]
//!   [33024    .. 98559  ]  net.2.weight   [256, 256]  (FC2)
//!   [98560    .. 98815  ]  net.2.bias     [256]
//!   [98816    .. 131583 ]  net.4.weight   [128, 256]  (FC3)
//!   [131584   .. 131711 ]  net.4.bias     [128]
//!   [131712   .. 131759 ]  palette.weight [16, 3]
//!
//! Output framebuffer: 384 × 640 × 3 = 737,280 uint8 (RGB)

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{get_default_device, MetalError};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const TERM_ROWS: usize = 24;
const TERM_COLS: usize = 80;
const CELL_H: usize = 16;
const CELL_W: usize = 8;
const FRAME_H: usize = TERM_ROWS * CELL_H; // 384
const FRAME_W: usize = TERM_COLS * CELL_W;  // 640
const FRAME_SIZE: usize = FRAME_H * FRAME_W * 3; // 737,280 bytes

const PALETTE_OFFSET: usize = 131_712;        // offset in weight buffer (16 colors × 3 RGB)
const PALETTE_FLOATS: usize = 48;             // 16 × 3
const N_WEIGHT_FLOATS_BASE: usize = 131_760;  // glyph MLP + palette only
const N_WEIGHT_FLOATS_FULL: usize = 143_539;  // + compositor ConvNet
const N_CELLS: usize = TERM_ROWS * TERM_COLS; // 1920
const H_BUF_FLOATS: usize = N_CELLS * 256;    // 491,520 floats per intermediate buffer
const H_BUF_BYTES: usize = H_BUF_FLOATS * 4;  // ~1.9 MB
const COMP_BUF_FLOATS: usize = FRAME_H * FRAME_W * 32; // 7,864,320 floats
const COMP_BUF_BYTES: usize = COMP_BUF_FLOATS * 2;     // ~15 MB per compositor buffer (half precision)
const MAX_BATCH: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Metal shader source — three-pass architecture
// ─────────────────────────────────────────────────────────────────────────────

const NEURAL_DISPLAY_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// ── Weight buffer offsets ──────────────────────────────────────────────────
constant int EMBED_W   = 0;        // [256, 64]  = 16384 floats
constant int FC1_W     = 16384;    // [256, 64]  = 16384 floats
constant int FC1_B     = 32768;    // [256]      = 256 floats
constant int FC2_W     = 33024;    // [256, 256] = 65536 floats
constant int FC2_B     = 98560;    // [256]      = 256 floats
constant int FC3_W     = 98816;    // [128, 256] = 32768 floats
constant int FC3_B     = 131584;   // [128]      = 128 floats
constant int PALETTE   = 131712;   // [16, 3]    = 48 floats

constant int TERM_COLS_C = 80;
constant int CELL_H_C    = 16;
constant int CELL_W_C    = 8;
constant int FRAME_W_C   = 640;
constant int FRAME_H_C   = 384;

// ── Compositor weight offsets (appended after palette) ──────────────
constant int COMP_CONV1_W = 131760;  // [32, 3, 5, 5] = 2400 floats
constant int COMP_CONV1_B = 134160;  // [32]
constant int COMP_CONV2_W = 134192;  // [32, 32, 3, 3] = 9216 floats
constant int COMP_CONV2_B = 143408;  // [32]
constant int COMP_CONV3_W = 143440;  // [3, 32, 1, 1] = 96 floats
constant int COMP_CONV3_B = 143536;  // [3]

// ── Activation functions ──────────────────────────────────────────────────
inline float neural_gelu(float x) {
    // Guard: for |x| > 10, GELU saturates to 0 (negative) or x (positive).
    // Without this, x^3 can overflow to ±inf for large |x|, and then
    // GELU(-inf) = 0.5 * (-inf) * (1 + tanh(-inf)) = (-inf) * 0 = NaN.
    if (x < -10.0f) return 0.0f;
    if (x >  10.0f) return x;
    float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

inline float neural_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -15.0f, 15.0f)));
}

// ══════════════════════════════════════════════════════════════════════════
// Pass 1: Embedding + FC1 → h1_buf
//   Thread-local memory: 0 arrays (just scalar accumulator)
//   buffer(0): char_codes [1920] uint8
//   buffer(1): weights    [131760] f32
//   buffer(2): h1_buf     [1920*256] f32  (output)
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_pass1_h1(
    device const uint8_t* char_codes [[buffer(0)]],
    device const float*   weights    [[buffer(1)]],
    device       float*   h1_buf     [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int col = (int)tid.x;
    int row = (int)tid.y;
    if (col >= 80 || row >= 24) return;

    int cell_idx = row * TERM_COLS_C + col;
    int ch = (int)char_codes[cell_idx];
    int h1_base = cell_idx * 256;

    // FC1: embed[ch] (64-dim) → Linear(64→256) + GELU
    // Each h1[i] is computed and written to device memory immediately.
    for (int i = 0; i < 256; i++) {
        float s = weights[FC1_B + i];
        for (int j = 0; j < 64; j++)
            s += weights[FC1_W + i * 64 + j] * weights[EMBED_W + ch * 64 + j];
        h1_buf[h1_base + i] = neural_gelu(s);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Pass 2: FC2 (h1 → h2)
//   Thread-local memory: 0 arrays (just scalar accumulator)
//   buffer(0): weights [131760] f32
//   buffer(1): h1_buf  [1920*256] f32  (input)
//   buffer(2): h2_buf  [1920*256] f32  (output)
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_pass2_h2(
    device const float* weights [[buffer(0)]],
    device const float* h1_buf  [[buffer(1)]],
    device       float* h2_buf  [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int col = (int)tid.x;
    int row = (int)tid.y;
    if (col >= 80 || row >= 24) return;

    int cell_idx = row * TERM_COLS_C + col;
    int base = cell_idx * 256;

    // FC2: h1 (256-dim) → Linear(256→256) + GELU
    for (int i = 0; i < 256; i++) {
        float s = weights[FC2_B + i];
        for (int j = 0; j < 256; j++)
            s += weights[FC2_W + i * 256 + j] * h1_buf[base + j];
        h2_buf[base + i] = neural_gelu(s);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Pass 3: FC3 + Palette + Alpha Blend → Pixels
//   Thread-local memory: 0 arrays (just scalar accumulators)
//   buffer(0): fg_codes  [1920] uint8
//   buffer(1): bg_codes  [1920] uint8
//   buffer(2): weights   [131760] f32
//   buffer(3): h2_buf    [1920*256] f32  (input)
//   buffer(4): framebuf  [384*640*3] uint8  (output RGB)
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_pass3_pixels(
    device const uint8_t* fg_codes  [[buffer(0)]],
    device const uint8_t* bg_codes  [[buffer(1)]],
    device const float*   weights   [[buffer(2)]],
    device const float*   h2_buf    [[buffer(3)]],
    device       uint8_t* framebuf  [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int col = (int)tid.x;
    int row = (int)tid.y;
    if (col >= 80 || row >= 24) return;

    int cell_idx = row * TERM_COLS_C + col;
    int fg_code = (int)fg_codes[cell_idx];
    int bg_code = (int)bg_codes[cell_idx];
    int h2_base = cell_idx * 256;

    // Color palette lookup
    float fg_r = weights[PALETTE + fg_code * 3 + 0];
    float fg_g = weights[PALETTE + fg_code * 3 + 1];
    float fg_b = weights[PALETTE + fg_code * 3 + 2];
    float bg_r = weights[PALETTE + bg_code * 3 + 0];
    float bg_g = weights[PALETTE + bg_code * 3 + 1];
    float bg_b = weights[PALETTE + bg_code * 3 + 2];

    int frame_row_start = row * CELL_H_C;
    int frame_col_start = col * CELL_W_C;

    // FC3: h2 (256-dim) → Linear(256→128) + Sigmoid → alpha blend → pixel
    for (int pi = 0; pi < 128; pi++) {
        float alpha_val = weights[FC3_B + pi];
        for (int j = 0; j < 256; j++)
            alpha_val += weights[FC3_W + pi * 256 + j] * h2_buf[h2_base + j];
        float a = neural_sigmoid(alpha_val);

        int py = pi / CELL_W_C;
        int px = pi % CELL_W_C;
        float r = a * fg_r + (1.0f - a) * bg_r;
        float g = a * fg_g + (1.0f - a) * bg_g;
        float b = a * fg_b + (1.0f - a) * bg_b;

        int frame_y = frame_row_start + py;
        int frame_x = frame_col_start + px;
        int pixel_idx = (frame_y * FRAME_W_C + frame_x) * 3;

        framebuf[pixel_idx + 0] = (uint8_t)clamp(r * 255.0f + 0.5f, 0.0f, 255.0f);
        framebuf[pixel_idx + 1] = (uint8_t)clamp(g * 255.0f + 0.5f, 0.0f, 255.0f);
        framebuf[pixel_idx + 2] = (uint8_t)clamp(b * 255.0f + 0.5f, 0.0f, 255.0f);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Pass 4: Compositor Conv1(5×5, 3→32) + GELU — half-precision output
//   Each thread handles ALL 32 output channels (2,400 MADs total).
//   2D dispatch is optimal here: only 3 input channels (uint8 pixels), so
//   per-thread work is already light. 3D dispatch (32x more threads) was
//   benchmarked but slower due to thread dispatch overhead exceeding the
//   benefit of splitting 75-MAD inner loops.
//   buffer(0): framebuf  [384*640*3] uint8   (input RGB)
//   buffer(1): weights   [143539] f32
//   buffer(2): comp_buf  [384*640*32] half    (output)
// ══════════════════════════════════════════════════════════════════════════

kernel void compositor_conv1(
    device const uint8_t* framebuf [[buffer(0)]],
    device const float*   weights  [[buffer(1)]],
    device       half*    comp_buf [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int x = (int)tid.x;
    int y = (int)tid.y;
    if (x >= FRAME_W_C || y >= FRAME_H_C) return;

    int out_base = (y * FRAME_W_C + x) * 32;

    for (int c_out = 0; c_out < 32; c_out++) {
        float sum = weights[COMP_CONV1_B + c_out];
        for (int c_in = 0; c_in < 3; c_in++) {
            for (int ky = -2; ky <= 2; ky++) {
                int ny = y + ky;
                if (ny < 0 || ny >= FRAME_H_C) continue;
                for (int kx = -2; kx <= 2; kx++) {
                    int nx = x + kx;
                    if (nx < 0 || nx >= FRAME_W_C) continue;
                    float pixel = (float)framebuf[(ny * FRAME_W_C + nx) * 3 + c_in] / 255.0f;
                    int w_idx = COMP_CONV1_W + ((c_out * 3 + c_in) * 5 + (ky + 2)) * 5 + (kx + 2);
                    sum += weights[w_idx] * pixel;
                }
            }
        }
        comp_buf[out_base + c_out] = (half)neural_gelu(sum);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Pass 5: Compositor Conv2(3×3, 32→32) + GELU — 3D parallel half-precision
//   The main bottleneck: 32×32×9 = 9,216 MADs per pixel.
//   Key optimization: parallelize output channels across threads.
//     OLD: each thread → ALL 32 c_out → 9,216 MADs per thread
//     NEW: each thread → ONE c_out    → 288 MADs per thread (32x lighter)
//   Grid: (640, 384, 32) = 7.86M threads via 3D dispatch.
//   Threads at same (x,y) form a SIMD group reading identical comp_in data →
//   one L1 cache fetch serves all 32 channel threads (coalesced read).
//   Writes to adjacent half values are also coalesced.
//   buffer(0): comp_in  [384*640*32] half  (input)
//   buffer(1): weights  [143539] f32
//   buffer(2): comp_out [384*640*32] half  (output)
// ══════════════════════════════════════════════════════════════════════════

kernel void compositor_conv2(
    device const half*  comp_in  [[buffer(0)]],
    device const float* weights  [[buffer(1)]],
    device       half*  comp_out [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int x = (int)tid.x;
    int y = (int)tid.y;
    int c_out = (int)tid.z;
    if (x >= FRAME_W_C || y >= FRAME_H_C || c_out >= 32) return;

    float sum = weights[COMP_CONV2_B + c_out];
    for (int c_in = 0; c_in < 32; c_in++) {
        for (int ky = -1; ky <= 1; ky++) {
            int ny = y + ky;
            if (ny < 0 || ny >= FRAME_H_C) continue;
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                if (nx < 0 || nx >= FRAME_W_C) continue;
                int in_idx = (ny * FRAME_W_C + nx) * 32 + c_in;
                int w_idx = COMP_CONV2_W + ((c_out * 32 + c_in) * 3 + (ky + 1)) * 3 + (kx + 1);
                sum += weights[w_idx] * (float)comp_in[in_idx];
            }
        }
    }
    comp_out[(y * FRAME_W_C + x) * 32 + c_out] = (half)neural_gelu(sum);
}

// ══════════════════════════════════════════════════════════════════════════
// Pass 6: Compositor Conv3(1×1, 32→3) + residual — reads half input
//   Reads original pixel from framebuf, adds conv output, writes back.
//   Float accumulator from half input: no precision loss.
//   buffer(0): comp_in  [384*640*32] half    (input)
//   buffer(1): weights  [143539] f32
//   buffer(2): framebuf [384*640*3] uint8    (in/out — residual)
// ══════════════════════════════════════════════════════════════════════════

kernel void compositor_conv3_residual(
    device const half*    comp_in  [[buffer(0)]],
    device const float*   weights  [[buffer(1)]],
    device       uint8_t* framebuf [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int x = (int)tid.x;
    int y = (int)tid.y;
    if (x >= FRAME_W_C || y >= FRAME_H_C) return;

    int pixel_idx = (y * FRAME_W_C + x) * 3;
    int comp_base = (y * FRAME_W_C + x) * 32;

    // Read original pixel (before overwrite)
    float orig[3];
    orig[0] = (float)framebuf[pixel_idx + 0] / 255.0f;
    orig[1] = (float)framebuf[pixel_idx + 1] / 255.0f;
    orig[2] = (float)framebuf[pixel_idx + 2] / 255.0f;

    // Conv3: 1×1 (dot product over 32 channels for each output channel)
    for (int c_out = 0; c_out < 3; c_out++) {
        float sum = weights[COMP_CONV3_B + c_out];
        for (int c_in = 0; c_in < 32; c_in++) {
            sum += weights[COMP_CONV3_W + c_out * 32 + c_in] * (float)comp_in[comp_base + c_in];
        }
        // Residual: output = clamp(original + conv_output, 0, 1)
        float result = clamp(orig[c_out] + sum, 0.0f, 1.0f);
        framebuf[pixel_idx + c_out] = (uint8_t)(result * 255.0f + 0.5f);
    }
}
"##;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn compile_display_lib(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
    let source = NSString::from_str(NEURAL_DISPLAY_SHADER);
    device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|e| MetalError::ShaderCompilationFailed(format!("{e:?}")))
}

fn make_pipeline(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    lib: &Retained<ProtocolObject<dyn MTLLibrary>>,
    fn_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    let name = NSString::from_str(fn_name);
    let func = lib
        .newFunctionWithName(&name)
        .ok_or_else(|| MetalError::ShaderCompilationFailed(format!("{fn_name} not found")))?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| MetalError::PipelineCreationFailed(format!("{e:?}")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Python-exposed struct
// ─────────────────────────────────────────────────────────────────────────────

/// Metal-based neural display — runs trained glyph MLP + color palette on GPU.
///
/// Three-pass architecture avoids GPU thread-local memory overflow:
///   Pass 1: char → Embedding+FC1 → h1_buf (device memory, ~1.9 MB)
///   Pass 2: h1_buf → FC2 → h2_buf (device memory, ~1.9 MB)
///   Pass 3: h2_buf → FC3 → alpha → blend → pixels
///
/// Usage from Python:
///   kernel = NeuralDisplayKernel()
///   kernel.load_weights(weights_flat)   # 131,760 f32 values
///   rgb_bytes = kernel.render(char_codes, fg_codes, bg_codes)
#[pyclass(unsendable)]
pub struct NeuralDisplayKernel {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pass1_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pass2_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pass3_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    // Compositor pipelines (always compiled, activated when compositor weights loaded)
    comp_conv1_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    comp_conv2_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    comp_conv3_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    weights_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    has_compositor: bool,
    // Pre-allocated buffers for zero-alloc rendering
    char_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    fg_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    bg_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    h1_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    h2_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    frame_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    // Compositor intermediate buffers (384×640×32 floats each, ~30 MB)
    comp_buf1: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    comp_buf2: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    // Batch rendering buffers (pre-allocated for MAX_BATCH frames)
    batch_char_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    batch_fg_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    batch_bg_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    batch_frame_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

#[pymethods]
impl NeuralDisplayKernel {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = get_default_device()
            .ok_or_else(|| PyRuntimeError::new_err("No Metal device available"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create Metal command queue"))?;

        let lib = compile_display_lib(&device)
            .map_err(|e| PyRuntimeError::new_err(format!("display shader: {e:?}")))?;
        let pass1_pipeline = make_pipeline(&device, &lib, "neural_pass1_h1")
            .map_err(|e| PyRuntimeError::new_err(format!("pass1 pipeline: {e:?}")))?;
        let pass2_pipeline = make_pipeline(&device, &lib, "neural_pass2_h2")
            .map_err(|e| PyRuntimeError::new_err(format!("pass2 pipeline: {e:?}")))?;
        let pass3_pipeline = make_pipeline(&device, &lib, "neural_pass3_pixels")
            .map_err(|e| PyRuntimeError::new_err(format!("pass3 pipeline: {e:?}")))?;
        let comp_conv1_pipeline = make_pipeline(&device, &lib, "compositor_conv1")
            .map_err(|e| PyRuntimeError::new_err(format!("comp_conv1 pipeline: {e:?}")))?;
        let comp_conv2_pipeline = make_pipeline(&device, &lib, "compositor_conv2")
            .map_err(|e| PyRuntimeError::new_err(format!("comp_conv2 pipeline: {e:?}")))?;
        let comp_conv3_pipeline = make_pipeline(&device, &lib, "compositor_conv3_residual")
            .map_err(|e| PyRuntimeError::new_err(format!("comp_conv3 pipeline: {e:?}")))?;

        // Pre-allocate all buffers (reused across renders)
        let shared = MTLResourceOptions::StorageModeShared;
        let char_buf = device.newBufferWithLength_options(N_CELLS, shared);
        let fg_buf = device.newBufferWithLength_options(N_CELLS, shared);
        let bg_buf = device.newBufferWithLength_options(N_CELLS, shared);
        let h1_buf = device.newBufferWithLength_options(H_BUF_BYTES, shared);
        let h2_buf = device.newBufferWithLength_options(H_BUF_BYTES, shared);
        let frame_buf = device.newBufferWithLength_options(FRAME_SIZE, shared);

        // Compositor intermediate buffers (384×640×32 floats each)
        let comp_buf1 = device.newBufferWithLength_options(COMP_BUF_BYTES, shared);
        let comp_buf2 = device.newBufferWithLength_options(COMP_BUF_BYTES, shared);

        // Batch rendering buffers (MAX_BATCH frames)
        let batch_char_buf = device.newBufferWithLength_options(MAX_BATCH * N_CELLS, shared);
        let batch_fg_buf = device.newBufferWithLength_options(MAX_BATCH * N_CELLS, shared);
        let batch_bg_buf = device.newBufferWithLength_options(MAX_BATCH * N_CELLS, shared);
        let batch_frame_buf = device.newBufferWithLength_options(MAX_BATCH * FRAME_SIZE, shared);

        Ok(Self {
            device,
            queue,
            pass1_pipeline,
            pass2_pipeline,
            pass3_pipeline,
            comp_conv1_pipeline,
            comp_conv2_pipeline,
            comp_conv3_pipeline,
            weights_buf: None,
            has_compositor: false,
            char_buf,
            fg_buf,
            bg_buf,
            h1_buf,
            h2_buf,
            frame_buf,
            comp_buf1,
            comp_buf2,
            batch_char_buf,
            batch_fg_buf,
            batch_bg_buf,
            batch_frame_buf,
        })
    }

    /// Load glyph MLP weights + color palette into a GPU buffer.
    ///
    /// weights_flat: flat f32 list of length 131,760
    ///   (embed + FC1 w/b + FC2 w/b + FC3 w/b + palette)
    /// Load glyph MLP weights + color palette (+ optional compositor) into GPU buffer.
    ///
    /// Accepts either 131,760 floats (glyph+palette only) or 143,539 (+ compositor).
    fn load_weights(&mut self, weights_flat: Vec<f32>) -> PyResult<()> {
        let n = weights_flat.len();
        if n != N_WEIGHT_FLOATS_BASE && n != N_WEIGHT_FLOATS_FULL {
            return Err(PyRuntimeError::new_err(format!(
                "weights must be {} (base) or {} (full) floats, got {}",
                N_WEIGHT_FLOATS_BASE, N_WEIGHT_FLOATS_FULL, n
            )));
        }
        self.has_compositor = n == N_WEIGHT_FLOATS_FULL;
        let bytes = n * 4;
        let buf = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("weights buffer alloc failed"))?;
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(weights_flat.as_ptr(), ptr, n);
        }
        self.weights_buf = Some(buf);
        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.weights_buf.is_some()
            && self.char_buf.is_some()
            && self.fg_buf.is_some()
            && self.bg_buf.is_some()
            && self.h1_buf.is_some()
            && self.h2_buf.is_some()
            && self.frame_buf.is_some()
    }

    /// Whether compositor weights are loaded (full 143,539 floats).
    fn has_compositor(&self) -> bool {
        self.has_compositor
    }

    /// Update the color palette in the GPU weight buffer without reloading all weights.
    ///
    /// palette: flat list of 48 f32 values (16 colors × 3 RGB, each 0.0-1.0)
    ///
    /// This allows real-time theme switching at full GPU speed — no need to
    /// re-extract/re-upload the entire weight buffer. Writes directly to the
    /// palette region at offset 131,712 in the shared MTLBuffer.
    fn set_palette(&self, palette: Vec<f32>) -> PyResult<()> {
        if palette.len() != PALETTE_FLOATS {
            return Err(PyRuntimeError::new_err(format!(
                "palette must be {} floats (16 colors × 3 RGB), got {}",
                PALETTE_FLOATS, palette.len()
            )));
        }
        let buf = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("weights not loaded — call load_weights() first")
        })?;
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut f32;
            let dst = ptr.add(PALETTE_OFFSET);
            std::ptr::copy_nonoverlapping(palette.as_ptr(), dst, PALETTE_FLOATS);
        }
        Ok(())
    }

    /// Read the current palette from the GPU weight buffer.
    ///
    /// Returns: list of 48 f32 values (16 colors × 3 RGB)
    fn get_palette(&self) -> PyResult<Vec<f32>> {
        let buf = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("weights not loaded")
        })?;
        let ptr = buf.contents().as_ptr() as *const f32;
        let mut out = Vec::with_capacity(PALETTE_FLOATS);
        for i in 0..PALETTE_FLOATS {
            out.push(unsafe { *ptr.add(PALETTE_OFFSET + i) });
        }
        Ok(out)
    }

    /// Read back weight values at specific indices (debug/verification).
    fn read_weights(&self, indices: Vec<usize>) -> PyResult<Vec<f32>> {
        let buf = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("weights not loaded")
        })?;
        let ptr = buf.contents().as_ptr() as *const f32;
        let max_idx = if self.has_compositor { N_WEIGHT_FLOATS_FULL } else { N_WEIGHT_FLOATS_BASE };
        let mut out = Vec::with_capacity(indices.len());
        for &idx in &indices {
            if idx >= max_idx {
                return Err(PyRuntimeError::new_err(format!(
                    "index {} out of range (max {})", idx, max_idx - 1
                )));
            }
            out.push(unsafe { *ptr.add(idx) });
        }
        Ok(out)
    }

    /// CPU-side single-character glyph computation for debugging.
    /// Returns 128 alpha values computed on CPU using the GPU weight buffer.
    fn debug_glyph(&self, char_code: u8) -> PyResult<Vec<f32>> {
        let buf = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("weights not loaded")
        })?;
        let w = buf.contents().as_ptr() as *const f32;

        let ch = char_code as usize;
        const EMBED_W: usize = 0;
        const FC1_W: usize = 16384;
        const FC1_B: usize = 32768;
        const FC2_W: usize = 33024;
        const FC2_B: usize = 98560;
        const FC3_W: usize = 98816;
        const FC3_B: usize = 131584;

        fn gelu(x: f32) -> f32 {
            let c: f32 = 0.7978845608028654;
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        }
        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x.clamp(-15.0, 15.0)).exp())
        }

        unsafe {
            let mut e = [0.0f32; 64];
            for i in 0..64 {
                e[i] = *w.add(EMBED_W + ch * 64 + i);
            }

            let mut h1 = [0.0f32; 256];
            for i in 0..256 {
                let mut s = *w.add(FC1_B + i);
                for j in 0..64 {
                    s += *w.add(FC1_W + i * 64 + j) * e[j];
                }
                h1[i] = gelu(s);
            }

            let mut h2 = [0.0f32; 256];
            for i in 0..256 {
                let mut s = *w.add(FC2_B + i);
                for j in 0..256 {
                    s += *w.add(FC2_W + i * 256 + j) * h1[j];
                }
                h2[i] = gelu(s);
            }

            let mut alpha = vec![0.0f32; 128];
            for i in 0..128 {
                let mut s = *w.add(FC3_B + i);
                for j in 0..256 {
                    s += *w.add(FC3_W + i * 256 + j) * h2[j];
                }
                alpha[i] = sigmoid(s);
            }

            Ok(alpha)
        }
    }

    /// Render terminal state to RGB framebuffer via three-pass GPU dispatch.
    ///
    /// char_codes: list of 1920 uint8 (24×80, row-major)
    /// fg_codes:   list of 1920 uint8
    /// bg_codes:   list of 1920 uint8
    ///
    /// Returns: bytes of length 737,280 (384×640×3 RGB)
    fn render<'py>(
        &self,
        py: Python<'py>,
        char_codes: Vec<u8>,
        fg_codes: Vec<u8>,
        bg_codes: Vec<u8>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if char_codes.len() != N_CELLS || fg_codes.len() != N_CELLS || bg_codes.len() != N_CELLS {
            return Err(PyRuntimeError::new_err(format!(
                "char/fg/bg must each be {} bytes, got {}/{}/{}",
                N_CELLS,
                char_codes.len(),
                fg_codes.len(),
                bg_codes.len()
            )));
        }

        let weights = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("weights not loaded — call load_weights() first")
        })?;
        let char_buf = self.char_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("char buffer not allocated")
        })?;
        let fg_buf = self.fg_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("fg buffer not allocated")
        })?;
        let bg_buf = self.bg_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("bg buffer not allocated")
        })?;
        let h1_buf = self.h1_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("h1 buffer not allocated")
        })?;
        let h2_buf = self.h2_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("h2 buffer not allocated")
        })?;
        let frame_buf = self.frame_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("frame buffer not allocated")
        })?;

        // Copy input data to pre-allocated GPU buffers
        unsafe {
            let cp = char_buf.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(char_codes.as_ptr(), cp, N_CELLS);
            let fp = fg_buf.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(fg_codes.as_ptr(), fp, N_CELLS);
            let bp = bg_buf.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(bg_codes.as_ptr(), bp, N_CELLS);
        }

        // Single command buffer, three compute passes (Metal guarantees serial order)
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| PyRuntimeError::new_err("command buffer creation failed"))?;

        // ── Pass 1: char → Embedding + FC1 → h1_buf ──
        {
            let enc = cmd
                .computeCommandEncoder()
                .ok_or_else(|| PyRuntimeError::new_err("pass1 encoder failed"))?;
            enc.setComputePipelineState(&self.pass1_pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(char_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                enc.setBuffer_offset_atIndex(Some(h1_buf), 0, 2);
            }
            // Use smaller threadgroups for better GPU occupancy.
            // tg_w=20 gives 4×24=96 threadgroups (vs 1×24=24 with tg_w=80).
            let tg_w = 20usize;
            let groups_x = (80 + tg_w - 1) / tg_w;
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: groups_x, height: 24, depth: 1 },
                MTLSize { width: tg_w, height: 1, depth: 1 },
            );
            enc.endEncoding();
        }

        // ── Pass 2: h1_buf → FC2 → h2_buf ──
        {
            let enc = cmd
                .computeCommandEncoder()
                .ok_or_else(|| PyRuntimeError::new_err("pass2 encoder failed"))?;
            enc.setComputePipelineState(&self.pass2_pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(weights), 0, 0);
                enc.setBuffer_offset_atIndex(Some(h1_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(h2_buf), 0, 2);
            }
            let max_tg = self.pass2_pipeline.maxTotalThreadsPerThreadgroup() as usize;
            let tg_w = max_tg.min(80);
            let groups_x = (80 + tg_w - 1) / tg_w;
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: groups_x, height: 24, depth: 1 },
                MTLSize { width: tg_w, height: 1, depth: 1 },
            );
            enc.endEncoding();
        }

        // ── Pass 3: h2_buf → FC3 + palette blend → pixels ──
        {
            let enc = cmd
                .computeCommandEncoder()
                .ok_or_else(|| PyRuntimeError::new_err("pass3 encoder failed"))?;
            enc.setComputePipelineState(&self.pass3_pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(fg_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(bg_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(weights), 0, 2);
                enc.setBuffer_offset_atIndex(Some(h2_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(frame_buf), 0, 4);
            }
            let max_tg = self.pass3_pipeline.maxTotalThreadsPerThreadgroup() as usize;
            let tg_w = max_tg.min(80);
            let groups_x = (80 + tg_w - 1) / tg_w;
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: groups_x, height: 24, depth: 1 },
                MTLSize { width: tg_w, height: 1, depth: 1 },
            );
            enc.endEncoding();
        }

        // ── Optional compositor passes (if full weights loaded) ──
        if self.has_compositor {
            let comp_buf1 = self.comp_buf1.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("comp_buf1 not allocated")
            })?;
            let comp_buf2 = self.comp_buf2.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("comp_buf2 not allocated")
            })?;

            // Conv1/Conv3: 2D grid over pixels (16×16 threadgroups)
            let comp_2d_tg = MTLSize { width: 16, height: 16, depth: 1 };
            let comp_2d_grid = MTLSize {
                width: (FRAME_W + 15) / 16,
                height: (FRAME_H + 15) / 16,
                depth: 1,
            };
            // Conv2: 3D grid — parallelize 32 output channels across threads
            // (8, 1, 32) = 256 threads/group; grid covers (W/8, H, 1)
            let conv3d_tg = MTLSize { width: 8, height: 1, depth: 32 };
            let conv3d_grid = MTLSize {
                width: (FRAME_W + 7) / 8,
                height: FRAME_H,
                depth: 1,
            };

            // ── Pass 4: Conv1(5×5, 3→32) + GELU ──
            {
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    PyRuntimeError::new_err("comp_conv1 encoder failed")
                })?;
                enc.setComputePipelineState(&self.comp_conv1_pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(frame_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(comp_buf1), 0, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(comp_2d_grid, comp_2d_tg);
                enc.endEncoding();
            }

            // ── Pass 5: Conv2(3×3, 32→32) + GELU — 3D parallelized ──
            {
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    PyRuntimeError::new_err("comp_conv2 encoder failed")
                })?;
                enc.setComputePipelineState(&self.comp_conv2_pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(comp_buf1), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(comp_buf2), 0, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(conv3d_grid, conv3d_tg);
                enc.endEncoding();
            }

            // ── Pass 6: Conv3(1×1, 32→3) + residual → framebuf ──
            {
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    PyRuntimeError::new_err("comp_conv3 encoder failed")
                })?;
                enc.setComputePipelineState(&self.comp_conv3_pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(comp_buf2), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(frame_buf), 0, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(comp_2d_grid, comp_2d_tg);
                enc.endEncoding();
            }
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        // Read back framebuffer
        let bytes = unsafe {
            let ptr = frame_buf.contents().as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr, FRAME_SIZE)
        };

        Ok(PyBytes::new(py, bytes))
    }

    /// Read back h1_buf values after render() for diagnostics.
    /// cell_idx: 0..1919, count: number of floats to read (max 256)
    fn read_h1(&self, cell_idx: usize, count: usize) -> PyResult<Vec<f32>> {
        let buf = self.h1_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("h1 buffer not allocated")
        })?;
        let start = cell_idx * 256;
        let n = count.min(256);
        if start + n > H_BUF_FLOATS {
            return Err(PyRuntimeError::new_err("cell_idx out of range"));
        }
        let ptr = buf.contents().as_ptr() as *const f32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(unsafe { *ptr.add(start + i) });
        }
        Ok(out)
    }

    /// Read back h2_buf values after render() for diagnostics.
    fn read_h2(&self, cell_idx: usize, count: usize) -> PyResult<Vec<f32>> {
        let buf = self.h2_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("h2 buffer not allocated")
        })?;
        let start = cell_idx * 256;
        let n = count.min(256);
        if start + n > H_BUF_FLOATS {
            return Err(PyRuntimeError::new_err("cell_idx out of range"));
        }
        let ptr = buf.contents().as_ptr() as *const f32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(unsafe { *ptr.add(start + i) });
        }
        Ok(out)
    }

    /// Return frame dimensions as (height, width, channels).
    fn frame_shape(&self) -> (usize, usize, usize) {
        (FRAME_H, FRAME_W, 3)
    }

    /// Return number of weight floats (base or full depending on load).
    fn weight_count(&self) -> usize {
        if self.has_compositor { N_WEIGHT_FLOATS_FULL } else { N_WEIGHT_FLOATS_BASE }
    }

    /// Return maximum batch size for render_batch().
    fn max_batch_size(&self) -> usize {
        MAX_BATCH
    }

    /// Render multiple frames in a single Metal command buffer.
    ///
    /// Amortizes command buffer overhead for animation/streaming.
    /// batch_size: number of frames (max 16)
    /// all_char_codes: flat uint8 of length batch_size * 1920
    /// all_fg_codes:   flat uint8 of length batch_size * 1920
    /// all_bg_codes:   flat uint8 of length batch_size * 1920
    ///
    /// Returns: list of PyBytes, each 737,280 bytes (384×640×3 RGB)
    fn render_batch<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        all_char_codes: Vec<u8>,
        all_fg_codes: Vec<u8>,
        all_bg_codes: Vec<u8>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        if batch_size == 0 || batch_size > MAX_BATCH {
            return Err(PyRuntimeError::new_err(format!(
                "batch_size must be 1..{}, got {}", MAX_BATCH, batch_size
            )));
        }
        let expected = batch_size * N_CELLS;
        if all_char_codes.len() != expected || all_fg_codes.len() != expected
            || all_bg_codes.len() != expected
        {
            return Err(PyRuntimeError::new_err(format!(
                "inputs must each be {} bytes for batch_size={}, got {}/{}/{}",
                expected, batch_size,
                all_char_codes.len(), all_fg_codes.len(), all_bg_codes.len()
            )));
        }

        let weights = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("weights not loaded")
        })?;
        let h1_buf = self.h1_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("h1 not allocated")
        })?;
        let h2_buf = self.h2_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("h2 not allocated")
        })?;
        let b_char = self.batch_char_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("batch_char not allocated")
        })?;
        let b_fg = self.batch_fg_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("batch_fg not allocated")
        })?;
        let b_bg = self.batch_bg_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("batch_bg not allocated")
        })?;
        let b_frame = self.batch_frame_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("batch_frame not allocated")
        })?;

        // Copy ALL input data to batch buffers (before commit)
        unsafe {
            let cp = b_char.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(all_char_codes.as_ptr(), cp, expected);
            let fp = b_fg.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(all_fg_codes.as_ptr(), fp, expected);
            let bp = b_bg.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(all_bg_codes.as_ptr(), bp, expected);
        }

        // Single command buffer for all frames
        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            PyRuntimeError::new_err("batch command buffer failed")
        })?;

        let tg_w = 20usize;
        let groups_x = (80 + tg_w - 1) / tg_w;
        let cell_tg = MTLSize { width: tg_w, height: 1, depth: 1 };
        let cell_grid = MTLSize { width: groups_x, height: 24, depth: 1 };

        for i in 0..batch_size {
            let char_off = (i * N_CELLS) as usize;
            let fg_off = (i * N_CELLS) as usize;
            let bg_off = (i * N_CELLS) as usize;
            let frame_off = (i * FRAME_SIZE) as usize;

            // ── Pass 1: Embed + FC1 → h1 ──
            {
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    PyRuntimeError::new_err("batch pass1 encoder failed")
                })?;
                enc.setComputePipelineState(&self.pass1_pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(b_char), char_off, 0);
                    enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(h1_buf), 0, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(cell_grid, cell_tg);
                enc.endEncoding();
            }

            // ── Pass 2: FC2 ──
            {
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    PyRuntimeError::new_err("batch pass2 encoder failed")
                })?;
                enc.setComputePipelineState(&self.pass2_pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(weights), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(h1_buf), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(h2_buf), 0, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(cell_grid, cell_tg);
                enc.endEncoding();
            }

            // ── Pass 3: FC3 + palette → pixels ──
            {
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    PyRuntimeError::new_err("batch pass3 encoder failed")
                })?;
                enc.setComputePipelineState(&self.pass3_pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(b_fg), fg_off, 0);
                    enc.setBuffer_offset_atIndex(Some(b_bg), bg_off, 1);
                    enc.setBuffer_offset_atIndex(Some(weights), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(h2_buf), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(b_frame), frame_off, 4);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(cell_grid, cell_tg);
                enc.endEncoding();
            }

            // ── Optional compositor passes ──
            if self.has_compositor {
                let comp_buf1 = self.comp_buf1.as_ref().unwrap();
                let comp_buf2 = self.comp_buf2.as_ref().unwrap();
                // Conv1/Conv3: 2D grid over pixels (16×16 threadgroups)
                let comp_2d_tg = MTLSize { width: 16, height: 16, depth: 1 };
                let comp_2d_grid = MTLSize {
                    width: (FRAME_W + 15) / 16,
                    height: (FRAME_H + 15) / 16,
                    depth: 1,
                };
                // Conv2: 3D grid (output channels parallelized)
                let conv3d_tg = MTLSize { width: 8, height: 1, depth: 32 };
                let conv3d_grid = MTLSize {
                    width: (FRAME_W + 7) / 8,
                    height: FRAME_H,
                    depth: 1,
                };

                // Pass 4: Conv1
                {
                    let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                        PyRuntimeError::new_err("batch comp1 encoder failed")
                    })?;
                    enc.setComputePipelineState(&self.comp_conv1_pipeline);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(b_frame), frame_off, 0);
                        enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(comp_buf1), 0, 2);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(comp_2d_grid, comp_2d_tg);
                    enc.endEncoding();
                }

                // Pass 5: Conv2 (3D parallelized)
                {
                    let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                        PyRuntimeError::new_err("batch comp2 encoder failed")
                    })?;
                    enc.setComputePipelineState(&self.comp_conv2_pipeline);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(comp_buf1), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(comp_buf2), 0, 2);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(conv3d_grid, conv3d_tg);
                    enc.endEncoding();
                }

                // Pass 6: Conv3 + residual
                {
                    let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                        PyRuntimeError::new_err("batch comp3 encoder failed")
                    })?;
                    enc.setComputePipelineState(&self.comp_conv3_pipeline);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(comp_buf2), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(b_frame), frame_off, 2);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(comp_2d_grid, comp_2d_tg);
                    enc.endEncoding();
                }
            }
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        // Read back all frames
        let mut results = Vec::with_capacity(batch_size);
        let base_ptr = b_frame.contents().as_ptr() as *const u8;
        for i in 0..batch_size {
            let offset = i * FRAME_SIZE;
            let slice = unsafe {
                std::slice::from_raw_parts(base_ptr.add(offset), FRAME_SIZE)
            };
            results.push(PyBytes::new(py, slice));
        }
        Ok(results)
    }
}

pub fn register_neural_display(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuralDisplayKernel>()?;
    Ok(())
}
