//! Standalone Metal shaders for Watchdog LSTM and GIC MLP neural OS models.
//!
//! Separated from the combined neural_os.rs to allow independent loading and
//! benchmarking of individual neurOS models. Each model gets its own Metal
//! compute pipeline and GPU weight buffer.
//!
//! Model architectures (matching the trained .pt files):
//!
//!   WatchdogMetalKernel — LSTM anomaly detector:
//!     LSTM(input=8, hidden=32, 1 layer)
//!       → Linear(32, 16) + ReLU
//!       → Linear(16, 1) + Sigmoid
//!     Input:  [seq_len, 8] system metrics window (flattened)
//!     Output: scalar anomaly score in [0, 1]
//!     5,921 params
//!
//!   GICMetalKernel — MLP interrupt priority controller:
//!     Linear(96, 64) + ReLU
//!       → Linear(64, 64) + ReLU
//!       → Linear(64, 32)
//!     Input:  [96] float (IRR + ISR + IMR register bits)
//!     Output: [32] float priority scores
//!     12,448 params
//!
//! Weight buffer layouts:
//!
//!   Watchdog weights (5,921 f32):
//!     [0      .. 1023  ]  lstm.weight_ih_l0 [128, 8]   (4*hidden × input)
//!     [1024   .. 5119  ]  lstm.weight_hh_l0 [128, 32]  (4*hidden × hidden)
//!     [5120   .. 5247  ]  lstm.bias_ih_l0   [128]
//!     [5248   .. 5375  ]  lstm.bias_hh_l0   [128]
//!     [5376   .. 5887  ]  scorer.0.weight   [16, 32]
//!     [5888   .. 5903  ]  scorer.0.bias     [16]
//!     [5904   .. 5919  ]  scorer.2.weight   [1, 16]
//!     [5920   .. 5920  ]  scorer.2.bias     [1]
//!
//!   GIC weights (12,448 f32):
//!     [0      .. 6143  ]  net.0.weight [64, 96] row-major
//!     [6144   .. 6207  ]  net.0.bias   [64]
//!     [6208   .. 10303 ]  net.2.weight [64, 64] row-major
//!     [10304  .. 10367 ]  net.2.bias   [64]
//!     [10368  .. 12415 ]  net.4.weight [32, 64] row-major
//!     [12416  .. 12447 ]  net.4.bias   [32]

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
use std::time::Instant;

use crate::{get_default_device, MetalError};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const WATCHDOG_N_WEIGHTS: usize = 5921;
const GIC_N_WEIGHTS: usize = 12448;

// ─────────────────────────────────────────────────────────────────────────────
// Metal shader source — Watchdog LSTM + GIC MLP
// ─────────────────────────────────────────────────────────────────────────────

const NEURAL_OS_MODELS_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// ── Shared helpers ──────────────────────────────────────────────────────────

inline float neural_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -15.0f, 15.0f)));
}

// ══════════════════════════════════════════════════════════════════════════════
// Watchdog LSTM Kernel
//
// LSTM(input=8, hidden=32, 1 layer)
//   → Linear(32, 16) + ReLU
//   → Linear(16, 1) + Sigmoid
//
// Buffers:
//   buffer(0): weights        [5921] f32  — LSTM + scorer weights
//   buffer(1): metrics_window [seq_len * 8] f32  — flattened [seq_len, 8]
//   buffer(2): output_score   [1] f32  — anomaly score [0, 1]
//   buffer(3): params         [1] uint32  — seq_len
//
// Single-threaded: unrolls LSTM sequentially over the time dimension.
// LSTM gate order: i(input), f(forget), g(cell), o(output) — each [32].
// ══════════════════════════════════════════════════════════════════════════════

// Watchdog weight buffer offsets
constant int WD_WIH   = 0;       // lstm.weight_ih_l0 [128, 8]  = 1024 floats
constant int WD_WHH   = 1024;    // lstm.weight_hh_l0 [128, 32] = 4096 floats
constant int WD_BIH   = 5120;    // lstm.bias_ih_l0   [128]
constant int WD_BHH   = 5248;    // lstm.bias_hh_l0   [128]
constant int WD_SC1_W = 5376;    // scorer.0.weight   [16, 32]  = 512 floats
constant int WD_SC1_B = 5888;    // scorer.0.bias     [16]
constant int WD_SC2_W = 5904;    // scorer.2.weight   [1, 16]   = 16 floats
constant int WD_SC2_B = 5920;    // scorer.2.bias     [1]

kernel void watchdog_lstm_check(
    device const float*  weights        [[buffer(0)]],
    device const float*  metrics_window [[buffer(1)]],
    device       float*  output_score   [[buffer(2)]],
    device const uint*   params         [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    int seq_len = (int)params[0];

    // Initialize hidden state h[32] and cell state c[32] to zero
    float h[32];
    float c[32];
    for (int i = 0; i < 32; i++) {
        h[i] = 0.0f;
        c[i] = 0.0f;
    }

    // Unroll LSTM over the sequence
    // LSTM equations:
    //   gates = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh
    //   i_t = sigmoid(gates[0:32])      — input gate
    //   f_t = sigmoid(gates[32:64])      — forget gate
    //   g_t = tanh(gates[64:96])         — cell candidate
    //   o_t = sigmoid(gates[96:128])     — output gate
    //   c_t = f_t * c_{t-1} + i_t * g_t
    //   h_t = o_t * tanh(c_t)
    for (int t = 0; t < seq_len; t++) {
        int inp_base = t * 8;

        // Compute all 128 gate values (4 gates × 32 hidden units)
        float gates[128];
        for (int g = 0; g < 128; g++) {
            float s = weights[WD_BIH + g] + weights[WD_BHH + g];
            // W_ih @ x_t: weight_ih is [128, 8], input is [8]
            for (int j = 0; j < 8; j++)
                s += weights[WD_WIH + g * 8 + j] * metrics_window[inp_base + j];
            // W_hh @ h_{t-1}: weight_hh is [128, 32], hidden is [32]
            for (int j = 0; j < 32; j++)
                s += weights[WD_WHH + g * 32 + j] * h[j];
            gates[g] = s;
        }

        // Apply gate activations and update cell/hidden state
        for (int i = 0; i < 32; i++) {
            float i_gate = neural_sigmoid(gates[i]);           // input gate
            float f_gate = neural_sigmoid(gates[32 + i]);      // forget gate
            float g_gate = tanh(gates[64 + i]);                // cell candidate
            float o_gate = neural_sigmoid(gates[96 + i]);      // output gate

            c[i] = f_gate * c[i] + i_gate * g_gate;
            h[i] = o_gate * tanh(c[i]);
        }
    }

    // Scorer MLP: h[32] → Linear(32, 16) + ReLU → Linear(16, 1) + Sigmoid

    // FC1: [32] → [16] + ReLU
    float sc1[16];
    for (int i = 0; i < 16; i++) {
        float s = weights[WD_SC1_B + i];
        for (int j = 0; j < 32; j++)
            s += weights[WD_SC1_W + i * 32 + j] * h[j];
        sc1[i] = max(0.0f, s);
    }

    // FC2: [16] → [1] + Sigmoid
    float score = weights[WD_SC2_B];
    for (int j = 0; j < 16; j++)
        score += weights[WD_SC2_W + j] * sc1[j];
    output_score[0] = neural_sigmoid(score);
}

// ══════════════════════════════════════════════════════════════════════════════
// Watchdog LSTM Batched Kernel
//
// Runs anomaly detection on multiple metrics windows in parallel.
// Each thread processes one window independently.
//
// Buffers:
//   buffer(0): weights          [5921] f32  — shared LSTM + scorer weights
//   buffer(1): metrics_windows  [batch * seq_len * 8] f32  — concatenated windows
//   buffer(2): output_scores    [batch] f32  — per-window anomaly scores
//   buffer(3): params           [2] uint32  — [seq_len, batch_size]
// ══════════════════════════════════════════════════════════════════════════════

kernel void watchdog_lstm_batch(
    device const float*  weights         [[buffer(0)]],
    device const float*  metrics_windows [[buffer(1)]],
    device       float*  output_scores   [[buffer(2)]],
    device const uint*   params          [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint seq_len    = params[0];
    uint batch_size = params[1];
    if (tid >= batch_size) return;

    int window_offset = (int)(tid * seq_len * 8u);

    // Initialize hidden state and cell state
    float h[32];
    float c[32];
    for (int i = 0; i < 32; i++) {
        h[i] = 0.0f;
        c[i] = 0.0f;
    }

    // LSTM unroll
    for (uint t = 0; t < seq_len; t++) {
        int inp_base = window_offset + (int)(t * 8u);

        float gates[128];
        for (int g = 0; g < 128; g++) {
            float s = weights[WD_BIH + g] + weights[WD_BHH + g];
            for (int j = 0; j < 8; j++)
                s += weights[WD_WIH + g * 8 + j] * metrics_windows[inp_base + j];
            for (int j = 0; j < 32; j++)
                s += weights[WD_WHH + g * 32 + j] * h[j];
            gates[g] = s;
        }

        for (int i = 0; i < 32; i++) {
            float i_gate = neural_sigmoid(gates[i]);
            float f_gate = neural_sigmoid(gates[32 + i]);
            float g_gate = tanh(gates[64 + i]);
            float o_gate = neural_sigmoid(gates[96 + i]);

            c[i] = f_gate * c[i] + i_gate * g_gate;
            h[i] = o_gate * tanh(c[i]);
        }
    }

    // Scorer
    float sc1[16];
    for (int i = 0; i < 16; i++) {
        float s = weights[WD_SC1_B + i];
        for (int j = 0; j < 32; j++)
            s += weights[WD_SC1_W + i * 32 + j] * h[j];
        sc1[i] = max(0.0f, s);
    }

    float score = weights[WD_SC2_B];
    for (int j = 0; j < 16; j++)
        score += weights[WD_SC2_W + j] * sc1[j];
    output_scores[tid] = neural_sigmoid(score);
}

// ══════════════════════════════════════════════════════════════════════════════
// GIC MLP Kernel — Single dispatch
//
// MLP [96 → 64] + ReLU → [64 → 64] + ReLU → [64 → 32]
//
// Buffers:
//   buffer(0): weights       [12448] f32
//   buffer(1): input_state   [96] f32  (IRR + ISR + IMR float bits)
//   buffer(2): output_scores [32] f32  (priority scores per IRQ)
//   buffer(3): pending_mask  [32] f32  (1.0 = pending, 0.0 = not)
//
// Single-threaded: one interrupt dispatch per call.
// Non-pending IRQs are masked to -1e30 (effectively -inf).
// ══════════════════════════════════════════════════════════════════════════════

// GIC weight buffer offsets
constant int GIC_FC1_W = 0;       // net.0.weight [64, 96]  = 6144 floats
constant int GIC_FC1_B = 6144;    // net.0.bias   [64]
constant int GIC_FC2_W = 6208;    // net.2.weight [64, 64]  = 4096 floats
constant int GIC_FC2_B = 10304;   // net.2.bias   [64]
constant int GIC_FC3_W = 10368;   // net.4.weight [32, 64]  = 2048 floats
constant int GIC_FC3_B = 12416;   // net.4.bias   [32]

kernel void gic_mlp_dispatch(
    device const float* weights       [[buffer(0)]],
    device const float* input_state   [[buffer(1)]],
    device       float* output_scores [[buffer(2)]],
    device const float* pending_mask  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // FC1: [96] → [64] + ReLU
    float h1[64];
    for (int i = 0; i < 64; i++) {
        float s = weights[GIC_FC1_B + i];
        for (int j = 0; j < 96; j++)
            s += weights[GIC_FC1_W + i * 96 + j] * input_state[j];
        h1[i] = max(0.0f, s);
    }

    // FC2: [64] → [64] + ReLU
    float h2[64];
    for (int i = 0; i < 64; i++) {
        float s = weights[GIC_FC2_B + i];
        for (int j = 0; j < 64; j++)
            s += weights[GIC_FC2_W + i * 64 + j] * h1[j];
        h2[i] = max(0.0f, s);
    }

    // FC3: [64] → [32] (raw scores, masked by pending)
    for (int i = 0; i < 32; i++) {
        float s = weights[GIC_FC3_B + i];
        for (int j = 0; j < 64; j++)
            s += weights[GIC_FC3_W + i * 64 + j] * h2[j];
        // Mask non-pending IRQs to -inf so argmax naturally selects pending ones
        output_scores[i] = (pending_mask[i] > 0.5f) ? s : -1e30f;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GIC MLP Batched Kernel — Multiple dispatches in parallel
//
// Buffers:
//   buffer(0): weights       [12448] f32  — shared weights
//   buffer(1): input_states  [batch * 96] f32  — concatenated state vectors
//   buffer(2): output_scores [batch * 32] f32  — concatenated score vectors
//   buffer(3): pending_masks [batch * 32] f32  — concatenated pending masks
//   buffer(4): params        [1] uint32  — batch_size
// ══════════════════════════════════════════════════════════════════════════════

kernel void gic_mlp_batch(
    device const float* weights       [[buffer(0)]],
    device const float* input_states  [[buffer(1)]],
    device       float* output_scores [[buffer(2)]],
    device const float* pending_masks [[buffer(3)]],
    device const uint*  params        [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    if (tid >= batch_size) return;

    int inp_base  = (int)(tid * 96u);
    int out_base  = (int)(tid * 32u);
    int mask_base = (int)(tid * 32u);

    // FC1: [96] → [64] + ReLU
    float h1[64];
    for (int i = 0; i < 64; i++) {
        float s = weights[GIC_FC1_B + i];
        for (int j = 0; j < 96; j++)
            s += weights[GIC_FC1_W + i * 96 + j] * input_states[inp_base + j];
        h1[i] = max(0.0f, s);
    }

    // FC2: [64] → [64] + ReLU
    float h2[64];
    for (int i = 0; i < 64; i++) {
        float s = weights[GIC_FC2_B + i];
        for (int j = 0; j < 64; j++)
            s += weights[GIC_FC2_W + i * 64 + j] * h1[j];
        h2[i] = max(0.0f, s);
    }

    // FC3: [64] → [32] (masked)
    for (int i = 0; i < 32; i++) {
        float s = weights[GIC_FC3_B + i];
        for (int j = 0; j < 64; j++)
            s += weights[GIC_FC3_W + i * 64 + j] * h2[j];
        output_scores[out_base + i] = (pending_masks[mask_base + i] > 0.5f) ? s : -1e30f;
    }
}
"##;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn compile_os_models_lib(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
    let source = NSString::from_str(NEURAL_OS_MODELS_SHADER);
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

fn new_buf_f32(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let bytes = data.len() * 4;
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferCreationFailed)?;
    unsafe {
        let ptr = buf.contents().as_ptr() as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
    Ok(buf)
}

fn new_buf_u32(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    val: u32,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let buf = device
        .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferCreationFailed)?;
    unsafe {
        *(buf.contents().as_ptr() as *mut u32) = val;
    }
    Ok(buf)
}

fn new_buf_u32_pair(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    val0: u32,
    val1: u32,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let buf = device
        .newBufferWithLength_options(8, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferCreationFailed)?;
    unsafe {
        let ptr = buf.contents().as_ptr() as *mut u32;
        *ptr = val0;
        *ptr.add(1) = val1;
    }
    Ok(buf)
}

fn read_buf_f32(buf: &Retained<ProtocolObject<dyn MTLBuffer>>, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    unsafe {
        let ptr = buf.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
    }
    out
}

fn dispatch_1d(
    queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    bufs: &[&Retained<ProtocolObject<dyn MTLBuffer>>],
    n_threads: usize,
) {
    let cmd = queue.commandBuffer().unwrap();
    let enc = cmd.computeCommandEncoder().unwrap();
    enc.setComputePipelineState(pipeline);

    unsafe {
        for (i, buf) in bufs.iter().enumerate() {
            enc.setBuffer_offset_atIndex(Some(buf), 0, i);
        }
    }

    let max_tg = pipeline.maxTotalThreadsPerThreadgroup() as usize;
    let tg_size = max_tg.min(256).min(n_threads).max(1);
    let groups = n_threads.div_ceil(tg_size);

    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: groups, height: 1, depth: 1 },
        MTLSize { width: tg_size, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
}

// ─────────────────────────────────────────────────────────────────────────────
// WatchdogMetalKernel
// ─────────────────────────────────────────────────────────────────────────────

/// Metal-based Watchdog LSTM anomaly detector.
///
/// Implements the same LSTM(8→32) + Scorer(32→16→1) architecture as
/// `WatchdogNet` in `ncpu/os/neuros/watchdog.py`, but executes entirely
/// as a Metal compute shader — no PyTorch at inference time.
///
/// Usage from Python:
///   kernel = WatchdogMetalKernel()
///   kernel.load_weights(flat_5921_floats)
///   score = kernel.check(metrics_flat, seq_len)
///   scores = kernel.check_batch(windows_flat, seq_len, batch_size)
#[pyclass(unsendable)]
pub struct WatchdogMetalKernel {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    single_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    batch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    weights_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

#[pymethods]
impl WatchdogMetalKernel {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = get_default_device()
            .ok_or_else(|| PyRuntimeError::new_err("No Metal device available"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create Metal command queue"))?;

        let lib = compile_os_models_lib(&device)
            .map_err(|e| PyRuntimeError::new_err(format!("os_models shader: {e:?}")))?;

        let single_pipeline = make_pipeline(&device, &lib, "watchdog_lstm_check")
            .map_err(|e| PyRuntimeError::new_err(format!("watchdog single pipeline: {e:?}")))?;
        let batch_pipeline = make_pipeline(&device, &lib, "watchdog_lstm_batch")
            .map_err(|e| PyRuntimeError::new_err(format!("watchdog batch pipeline: {e:?}")))?;

        Ok(Self {
            device,
            queue,
            single_pipeline,
            batch_pipeline,
            weights_buf: None,
        })
    }

    /// Load LSTM + scorer weights into a GPU buffer.
    ///
    /// weights: flat f32 list of length 5921
    ///   Layout: weight_ih[128,8] + weight_hh[128,32] + bias_ih[128] +
    ///           bias_hh[128] + scorer_fc1_w[16,32] + scorer_fc1_b[16] +
    ///           scorer_fc2_w[1,16] + scorer_fc2_b[1]
    fn load_weights(&mut self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != WATCHDOG_N_WEIGHTS {
            return Err(PyRuntimeError::new_err(format!(
                "Watchdog weights must be {} floats, got {}",
                WATCHDOG_N_WEIGHTS, weights.len()
            )));
        }
        self.weights_buf = Some(
            new_buf_f32(&self.device, &weights)
                .map_err(|e| PyRuntimeError::new_err(format!("watchdog weights buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Check whether weights are loaded and ready for inference.
    fn is_ready(&self) -> bool {
        self.weights_buf.is_some()
    }

    /// Run anomaly detection on a single metrics window.
    ///
    /// metrics_window: [seq_len * 8] f32 — flattened system metrics
    /// seq_len: number of timesteps in the window
    ///
    /// Returns: anomaly score in [0, 1] (higher = more anomalous)
    fn check(&self, metrics_window: Vec<f32>, seq_len: u32) -> PyResult<f32> {
        let expected = seq_len as usize * 8;
        if metrics_window.len() != expected {
            return Err(PyRuntimeError::new_err(format!(
                "Watchdog input must be {} floats ({}*8), got {}",
                expected, seq_len, metrics_window.len()
            )));
        }
        let weights = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Watchdog weights not loaded — call load_weights() first")
        })?;

        let input_buf = new_buf_f32(&self.device, &metrics_window)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output_buf = self.device
            .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("output buf alloc failed"))?;
        let params_buf = new_buf_u32(&self.device, seq_len)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.single_pipeline,
                    &[weights, &input_buf, &output_buf, &params_buf], 1);
        let result = read_buf_f32(&output_buf, 1);
        Ok(result[0])
    }

    /// Run anomaly detection on a batch of metrics windows in parallel.
    ///
    /// metrics_windows: [batch_size * seq_len * 8] f32 — concatenated windows
    /// seq_len: number of timesteps per window
    /// batch_size: number of windows
    ///
    /// Returns: [batch_size] f32 anomaly scores
    fn check_batch(
        &self,
        metrics_windows: Vec<f32>,
        seq_len: u32,
        batch_size: u32,
    ) -> PyResult<Vec<f32>> {
        let expected = batch_size as usize * seq_len as usize * 8;
        if metrics_windows.len() != expected {
            return Err(PyRuntimeError::new_err(format!(
                "Watchdog batch input must be {} floats ({}*{}*8), got {}",
                expected, batch_size, seq_len, metrics_windows.len()
            )));
        }
        let weights = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Watchdog weights not loaded — call load_weights() first")
        })?;

        let input_buf = new_buf_f32(&self.device, &metrics_windows)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output_buf = self.device
            .newBufferWithLength_options(batch_size as usize * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("output buf alloc failed"))?;
        let params_buf = new_buf_u32_pair(&self.device, seq_len, batch_size)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.batch_pipeline,
                    &[weights, &input_buf, &output_buf, &params_buf],
                    batch_size as usize);
        Ok(read_buf_f32(&output_buf, batch_size as usize))
    }

    /// Benchmark watchdog inference. Returns (single_us, batch_16_us).
    fn benchmark(&self) -> PyResult<(f64, f64)> {
        if !self.is_ready() {
            return Err(PyRuntimeError::new_err("Watchdog weights not loaded"));
        }

        // Single window: 64 timesteps × 8 metrics
        let single_input = vec![0.5f32; 64 * 8];
        let t0 = Instant::now();
        for _ in 0..1000 {
            let _ = self.check(single_input.clone(), 64);
        }
        let single_us = t0.elapsed().as_micros() as f64 / 1000.0;

        // Batch: 16 windows × 64 timesteps × 8 metrics
        let batch_input = vec![0.5f32; 16 * 64 * 8];
        let t0 = Instant::now();
        for _ in 0..1000 {
            let _ = self.check_batch(batch_input.clone(), 64, 16);
        }
        let batch_us = t0.elapsed().as_micros() as f64 / 1000.0;

        Ok((single_us, batch_us))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GICMetalKernel
// ─────────────────────────────────────────────────────────────────────────────

/// Metal-based GIC (Generic Interrupt Controller) neural priority encoder.
///
/// Implements the same MLP [96→64→64→32] architecture as `NeuralPriorityEncoder`
/// in `ncpu/os/neuros/interrupts.py`, but executes entirely as a Metal
/// compute shader — no PyTorch at inference time.
///
/// Usage from Python:
///   kernel = GICMetalKernel()
///   kernel.load_weights(flat_12448_floats)
///   scores = kernel.dispatch(input_state_96, pending_mask_32)
///   batch_scores = kernel.dispatch_batch(states_flat, masks_flat, batch_size)
#[pyclass(unsendable)]
pub struct GICMetalKernel {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    single_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    batch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    weights_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

#[pymethods]
impl GICMetalKernel {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = get_default_device()
            .ok_or_else(|| PyRuntimeError::new_err("No Metal device available"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create Metal command queue"))?;

        let lib = compile_os_models_lib(&device)
            .map_err(|e| PyRuntimeError::new_err(format!("os_models shader: {e:?}")))?;

        let single_pipeline = make_pipeline(&device, &lib, "gic_mlp_dispatch")
            .map_err(|e| PyRuntimeError::new_err(format!("gic single pipeline: {e:?}")))?;
        let batch_pipeline = make_pipeline(&device, &lib, "gic_mlp_batch")
            .map_err(|e| PyRuntimeError::new_err(format!("gic batch pipeline: {e:?}")))?;

        Ok(Self {
            device,
            queue,
            single_pipeline,
            batch_pipeline,
            weights_buf: None,
        })
    }

    /// Load GIC MLP weights into a GPU buffer.
    ///
    /// weights: flat f32 list of length 12448
    ///   Layout: FC1_w[64,96] + FC1_b[64] + FC2_w[64,64] + FC2_b[64] +
    ///           FC3_w[32,64] + FC3_b[32]
    fn load_weights(&mut self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != GIC_N_WEIGHTS {
            return Err(PyRuntimeError::new_err(format!(
                "GIC weights must be {} floats, got {}",
                GIC_N_WEIGHTS, weights.len()
            )));
        }
        self.weights_buf = Some(
            new_buf_f32(&self.device, &weights)
                .map_err(|e| PyRuntimeError::new_err(format!("gic weights buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Check whether weights are loaded and ready for inference.
    fn is_ready(&self) -> bool {
        self.weights_buf.is_some()
    }

    /// Score interrupt priorities for a single dispatch call.
    ///
    /// input_state: [96] f32 — concatenation of IRR + ISR + IMR float bits
    /// pending_mask: [32] f32 — 1.0 where IRQ is pending, 0.0 otherwise
    ///
    /// Returns: [32] f32 priority scores (non-pending masked to -1e30)
    fn dispatch(&self, input_state: Vec<f32>, pending_mask: Vec<f32>) -> PyResult<Vec<f32>> {
        if input_state.len() != 96 {
            return Err(PyRuntimeError::new_err(format!(
                "GIC input must be 96 floats, got {}", input_state.len()
            )));
        }
        if pending_mask.len() != 32 {
            return Err(PyRuntimeError::new_err(format!(
                "GIC pending_mask must be 32 floats, got {}", pending_mask.len()
            )));
        }
        let weights = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("GIC weights not loaded — call load_weights() first")
        })?;

        let input_buf = new_buf_f32(&self.device, &input_state)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output_buf = self.device
            .newBufferWithLength_options(32 * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("output buf alloc failed"))?;
        let mask_buf = new_buf_f32(&self.device, &pending_mask)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.single_pipeline,
                    &[weights, &input_buf, &output_buf, &mask_buf], 1);
        Ok(read_buf_f32(&output_buf, 32))
    }

    /// Score interrupt priorities for a batch of dispatch calls in parallel.
    ///
    /// input_states: [batch_size * 96] f32 — concatenated state vectors
    /// pending_masks: [batch_size * 32] f32 — concatenated pending masks
    /// batch_size: number of dispatches
    ///
    /// Returns: [batch_size * 32] f32 priority scores
    fn dispatch_batch(
        &self,
        input_states: Vec<f32>,
        pending_masks: Vec<f32>,
        batch_size: u32,
    ) -> PyResult<Vec<f32>> {
        let n = batch_size as usize;
        if input_states.len() != n * 96 {
            return Err(PyRuntimeError::new_err(format!(
                "GIC batch input must be {} floats ({}*96), got {}",
                n * 96, batch_size, input_states.len()
            )));
        }
        if pending_masks.len() != n * 32 {
            return Err(PyRuntimeError::new_err(format!(
                "GIC batch masks must be {} floats ({}*32), got {}",
                n * 32, batch_size, pending_masks.len()
            )));
        }
        let weights = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("GIC weights not loaded — call load_weights() first")
        })?;

        let input_buf = new_buf_f32(&self.device, &input_states)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output_buf = self.device
            .newBufferWithLength_options(n * 32 * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("output buf alloc failed"))?;
        let mask_buf = new_buf_f32(&self.device, &pending_masks)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let params_buf = new_buf_u32(&self.device, batch_size)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.batch_pipeline,
                    &[weights, &input_buf, &output_buf, &mask_buf, &params_buf], n);
        Ok(read_buf_f32(&output_buf, n * 32))
    }

    /// Benchmark GIC inference. Returns (single_us, batch_32_us).
    fn benchmark(&self) -> PyResult<(f64, f64)> {
        if !self.is_ready() {
            return Err(PyRuntimeError::new_err("GIC weights not loaded"));
        }

        // Single dispatch
        let input = vec![0.5f32; 96];
        let mask = vec![1.0f32; 32];
        let t0 = Instant::now();
        for _ in 0..1000 {
            let _ = self.dispatch(input.clone(), mask.clone());
        }
        let single_us = t0.elapsed().as_micros() as f64 / 1000.0;

        // Batch: 32 dispatches
        let batch_input = vec![0.5f32; 32 * 96];
        let batch_mask = vec![1.0f32; 32 * 32];
        let t0 = Instant::now();
        for _ in 0..1000 {
            let _ = self.dispatch_batch(batch_input.clone(), batch_mask.clone(), 32);
        }
        let batch_us = t0.elapsed().as_micros() as f64 / 1000.0;

        Ok((single_us, batch_us))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────────────────────────

pub fn register_neural_os_models(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WatchdogMetalKernel>()?;
    m.add_class::<GICMetalKernel>()?;
    Ok(())
}
