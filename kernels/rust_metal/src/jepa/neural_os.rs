//! Neural OS Kernels — runs trained neurOS models (GIC, Watchdog, Compiler Optimizer)
//! on Metal GPU, eliminating PyTorch per-syscall overhead.
//!
//! These three models account for ~60% overhead in the neurOS ablation study when
//! run via Python/PyTorch inference.  By porting them to Metal compute shaders
//! (same pattern as neural_alu.rs), they become near-free GPU operations.
//!
//! Model architectures (from trained .pt files):
//!
//!   GIC (Neural Priority Encoder):
//!     MLP [96 → 64] + ReLU → [64 → 64] + ReLU → [64 → 32]
//!     Input: [32*3] = 96 (IRR + ISR + IMR float bits)
//!     Output: [32] priority scores (higher = handle first)
//!     12,448 params
//!
//!   Watchdog (Neural Anomaly Detector):
//!     LSTM(input=8, hidden=32, 1 layer) → Linear(32,16) + ReLU → Linear(16,1) + Sigmoid
//!     Input: [seq_len, 8] system metrics window
//!     Output: scalar anomaly score [0,1]
//!     5,921 params
//!
//!   Compiler Optimizer (Neural Peephole):
//!     MLP [15 → 64] + ReLU → [64 → 32] + ReLU → [32 → 5]
//!     Input: [3*5] = 15 (window of 3 instructions × 5 features)
//!     Output: [5] optimization class scores
//!     3,269 params
//!
//! Weight buffer layouts:
//!
//!   GIC weights (12,448 f32):
//!     [0      .. 6143  ]  FC1 weight [64, 96]  row-major
//!     [6144   .. 6207  ]  FC1 bias   [64]
//!     [6208   .. 10303 ]  FC2 weight [64, 64]  row-major
//!     [10304  .. 10367 ]  FC2 bias   [64]
//!     [10368  .. 12415 ]  FC3 weight [32, 64]  row-major
//!     [12416  .. 12447 ]  FC3 bias   [32]
//!
//!   Watchdog weights (5,921 f32):
//!     [0      .. 1023  ]  lstm.weight_ih [128, 8]   (4*hidden × input)
//!     [1024   .. 5119  ]  lstm.weight_hh [128, 32]  (4*hidden × hidden)
//!     [5120   .. 5247  ]  lstm.bias_ih   [128]
//!     [5248   .. 5375  ]  lstm.bias_hh   [128]
//!     [5376   .. 5887  ]  scorer FC1 weight [16, 32]
//!     [5888   .. 5903  ]  scorer FC1 bias   [16]
//!     [5904   .. 5919  ]  scorer FC2 weight [1, 16]
//!     [5920   .. 5920  ]  scorer FC2 bias   [1]
//!
//!   Compiler weights (3,269 f32):
//!     [0      .. 959   ]  FC1 weight [64, 15]  row-major
//!     [960    .. 1023  ]  FC1 bias   [64]
//!     [1024   .. 3071  ]  FC2 weight [32, 64]  row-major
//!     [3072   .. 3103  ]  FC2 bias   [32]
//!     [3104   .. 3263  ]  FC3 weight [5, 32]   row-major
//!     [3264   .. 3268  ]  FC3 bias   [5]

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

use crate::{get_default_device, MetalError};

// ─────────────────────────────────────────────────────────────────────────────
// Metal shader source
// ─────────────────────────────────────────────────────────────────────────────

const NEURAL_OS_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// ── Helpers ──────────────────────────────────────────────────────────────────

inline float neural_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -15.0f, 15.0f)));
}

// ══════════════════════════════════════════════════════════════════════════════
// Kernel 1: neural_gic_dispatch
//
// MLP [96 → 64] + ReLU → [64 → 64] + ReLU → [64 → 32]
//
//   buffer(0): gic_weights   [12448] f32
//   buffer(1): input_state   [96] f32  (IRR + ISR + IMR as floats)
//   buffer(2): output_scores [32] f32  (priority scores per IRQ)
//   buffer(3): pending_mask  [32] f32  (1.0 = pending, 0.0 = not pending)
//
// Single-threaded (one interrupt dispatch per call).
// ══════════════════════════════════════════════════════════════════════════════

// Weight offsets for GIC
constant int GIC_FC1_W = 0;       // [64, 96] = 6144 floats
constant int GIC_FC1_B = 6144;    // [64]
constant int GIC_FC2_W = 6208;    // [64, 64] = 4096 floats
constant int GIC_FC2_B = 10304;   // [64]
constant int GIC_FC3_W = 10368;   // [32, 64] = 2048 floats
constant int GIC_FC3_B = 12416;   // [32]

kernel void neural_gic_dispatch(
    device const float* weights      [[buffer(0)]],
    device const float* input_state  [[buffer(1)]],
    device       float* output_scores[[buffer(2)]],
    device const float* pending_mask [[buffer(3)]],
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

    // FC3: [64] → [32] (raw scores)
    for (int i = 0; i < 32; i++) {
        float s = weights[GIC_FC3_B + i];
        for (int j = 0; j < 64; j++)
            s += weights[GIC_FC3_W + i * 64 + j] * h2[j];
        // Mask non-pending IRQs to -inf
        output_scores[i] = (pending_mask[i] > 0.5f) ? s : -1e30f;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Kernel 2: neural_watchdog_check
//
// LSTM(input=8, hidden=32, 1 layer) → Linear(32,16) + ReLU → Linear(16,1) + Sigmoid
//
//   buffer(0): watchdog_weights [5921] f32
//   buffer(1): metrics_window   [seq_len * 8] f32  (flattened [seq_len, 8])
//   buffer(2): output_score     [1] f32  (anomaly score 0-1)
//   buffer(3): params           [1] uint32  (seq_len)
//
// Single-threaded LSTM unroll over the sequence.
// ══════════════════════════════════════════════════════════════════════════════

// Weight offsets for Watchdog
constant int WD_WIH = 0;       // lstm.weight_ih [128, 8]  = 1024 floats
constant int WD_WHH = 1024;    // lstm.weight_hh [128, 32] = 4096 floats
constant int WD_BIH = 5120;    // lstm.bias_ih   [128]
constant int WD_BHH = 5248;    // lstm.bias_hh   [128]
constant int WD_SC1_W = 5376;  // scorer.0.weight [16, 32] = 512
constant int WD_SC1_B = 5888;  // scorer.0.bias   [16]
constant int WD_SC2_W = 5904;  // scorer.2.weight [1, 16]  = 16
constant int WD_SC2_B = 5920;  // scorer.2.bias   [1]

kernel void neural_watchdog_check(
    device const float*  weights       [[buffer(0)]],
    device const float*  metrics_window[[buffer(1)]],
    device       float*  output_score  [[buffer(2)]],
    device const uint*   params        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    int seq_len = (int)params[0];

    // LSTM hidden state and cell state (both [32])
    float h[32];
    float c[32];
    for (int i = 0; i < 32; i++) { h[i] = 0.0f; c[i] = 0.0f; }

    // Unroll LSTM over sequence
    // LSTM gates: i, f, g, o  (each size hidden=32)
    // weight_ih [4*32, 8] — maps input to gates
    // weight_hh [4*32, 32] — maps hidden to gates
    // bias_ih [4*32], bias_hh [4*32]
    for (int t = 0; t < seq_len; t++) {
        // Input at timestep t: metrics_window[t*8 .. t*8+7]
        int inp_base = t * 8;

        float gates[128];

        // gates = weight_ih @ x + bias_ih + weight_hh @ h + bias_hh
        for (int g = 0; g < 128; g++) {
            float s = weights[WD_BIH + g] + weights[WD_BHH + g];
            // weight_ih contribution
            for (int j = 0; j < 8; j++)
                s += weights[WD_WIH + g * 8 + j] * metrics_window[inp_base + j];
            // weight_hh contribution
            for (int j = 0; j < 32; j++)
                s += weights[WD_WHH + g * 32 + j] * h[j];
            gates[g] = s;
        }

        // Split gates: i=[0..31], f=[32..63], g=[64..95], o=[96..127]
        // Apply activations: sigmoid(i), sigmoid(f), tanh(g), sigmoid(o)
        for (int i = 0; i < 32; i++) {
            float i_gate = neural_sigmoid(gates[i]);       // input gate
            float f_gate = neural_sigmoid(gates[32 + i]);  // forget gate
            float g_gate = tanh(gates[64 + i]);            // cell gate
            float o_gate = neural_sigmoid(gates[96 + i]);  // output gate

            c[i] = f_gate * c[i] + i_gate * g_gate;
            h[i] = o_gate * tanh(c[i]);
        }
    }

    // Scorer MLP: h[32] → Linear(32,16) + ReLU → Linear(16,1) + Sigmoid

    // FC1: [32] → [16] + ReLU
    float sc1[16];
    for (int i = 0; i < 16; i++) {
        float s = weights[WD_SC1_B + i];
        for (int j = 0; j < 32; j++)
            s += weights[WD_SC1_W + i * 32 + j] * h[j];
        sc1[i] = max(0.0f, s);
    }

    // FC2: [16] → [1] + Sigmoid
    float s = weights[WD_SC2_B];
    for (int j = 0; j < 16; j++)
        s += weights[WD_SC2_W + j] * sc1[j];
    output_score[0] = neural_sigmoid(s);
}

// ══════════════════════════════════════════════════════════════════════════════
// Kernel 3: neural_compiler_score
//
// MLP [15 → 64] + ReLU → [64 → 32] + ReLU → [32 → 5]
//
//   buffer(0): compiler_weights [3269] f32
//   buffer(1): input_window     [N * 15] f32  (N windows of 3 instructions × 5 features)
//   buffer(2): output_scores    [N * 5] f32   (N × 5 optimization class scores)
//   buffer(3): params           [1] uint32     (N = number of windows)
//
// Parallelized: one thread per instruction window.
// ══════════════════════════════════════════════════════════════════════════════

// Weight offsets for Compiler Optimizer
constant int CO_FC1_W = 0;       // [64, 15] = 960 floats
constant int CO_FC1_B = 960;     // [64]
constant int CO_FC2_W = 1024;    // [32, 64] = 2048 floats
constant int CO_FC2_B = 3072;    // [32]
constant int CO_FC3_W = 3104;    // [5, 32]  = 160 floats
constant int CO_FC3_B = 3264;    // [5]

kernel void neural_compiler_score(
    device const float* weights       [[buffer(0)]],
    device const float* input_windows [[buffer(1)]],
    device       float* output_scores [[buffer(2)]],
    device const uint*  params        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n_windows = params[0];
    if (tid >= n_windows) return;

    int inp_base = (int)(tid * 15u);
    int out_base = (int)(tid * 5u);

    // FC1: [15] → [64] + ReLU
    float h1[64];
    for (int i = 0; i < 64; i++) {
        float s = weights[CO_FC1_B + i];
        for (int j = 0; j < 15; j++)
            s += weights[CO_FC1_W + i * 15 + j] * input_windows[inp_base + j];
        h1[i] = max(0.0f, s);
    }

    // FC2: [64] → [32] + ReLU
    float h2[32];
    for (int i = 0; i < 32; i++) {
        float s = weights[CO_FC2_B + i];
        for (int j = 0; j < 64; j++)
            s += weights[CO_FC2_W + i * 64 + j] * h1[j];
        h2[i] = max(0.0f, s);
    }

    // FC3: [32] → [5] (raw logits)
    for (int i = 0; i < 5; i++) {
        float s = weights[CO_FC3_B + i];
        for (int j = 0; j < 32; j++)
            s += weights[CO_FC3_W + i * 32 + j] * h2[j];
        output_scores[out_base + i] = s;
    }
}
"##;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (reuse pattern from neural_alu.rs)
// ─────────────────────────────────────────────────────────────────────────────

fn compile_neural_os_lib(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
    let source = NSString::from_str(NEURAL_OS_SHADER);
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
// Python-exposed struct
// ─────────────────────────────────────────────────────────────────────────────

/// Metal-based neural OS kernels — GIC, Watchdog, Compiler Optimizer on GPU.
///
/// Usage from Python:
///   kernel = NeuralOSKernels()
///   kernel.load_gic_weights(flat_weights)
///   kernel.load_watchdog_weights(flat_weights)
///   kernel.load_compiler_weights(flat_weights)
///   scores = kernel.execute_gic(input_state, pending_mask)
///   anomaly = kernel.execute_watchdog(metrics_window, seq_len)
///   opt_classes = kernel.execute_compiler(windows_flat, n_windows)
#[pyclass(unsendable)]
pub struct NeuralOSKernels {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    gic_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    watchdog_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    compiler_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gic_weights_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    watchdog_weights_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    compiler_weights_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

#[pymethods]
impl NeuralOSKernels {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = get_default_device()
            .ok_or_else(|| PyRuntimeError::new_err("No Metal device available"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create Metal command queue"))?;

        let lib = compile_neural_os_lib(&device)
            .map_err(|e| PyRuntimeError::new_err(format!("neural_os shader: {e:?}")))?;

        let gic_pipeline = make_pipeline(&device, &lib, "neural_gic_dispatch")
            .map_err(|e| PyRuntimeError::new_err(format!("gic pipeline: {e:?}")))?;
        let watchdog_pipeline = make_pipeline(&device, &lib, "neural_watchdog_check")
            .map_err(|e| PyRuntimeError::new_err(format!("watchdog pipeline: {e:?}")))?;
        let compiler_pipeline = make_pipeline(&device, &lib, "neural_compiler_score")
            .map_err(|e| PyRuntimeError::new_err(format!("compiler pipeline: {e:?}")))?;

        Ok(Self {
            device,
            queue,
            gic_pipeline,
            watchdog_pipeline,
            compiler_pipeline,
            gic_weights_buf: None,
            watchdog_weights_buf: None,
            compiler_weights_buf: None,
        })
    }

    // ── Weight loading ────────────────────────────────────────────────────────

    /// Load GIC MLP weights (12,448 f32).
    fn load_gic_weights(&mut self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != 12448 {
            return Err(PyRuntimeError::new_err(format!(
                "GIC weights must be 12448 floats, got {}", weights.len()
            )));
        }
        self.gic_weights_buf = Some(
            new_buf_f32(&self.device, &weights)
                .map_err(|e| PyRuntimeError::new_err(format!("GIC weights buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Load Watchdog LSTM + scorer weights (5,921 f32).
    fn load_watchdog_weights(&mut self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != 5921 {
            return Err(PyRuntimeError::new_err(format!(
                "Watchdog weights must be 5921 floats, got {}", weights.len()
            )));
        }
        self.watchdog_weights_buf = Some(
            new_buf_f32(&self.device, &weights)
                .map_err(|e| PyRuntimeError::new_err(format!("Watchdog weights buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Load Compiler Optimizer MLP weights (3,269 f32).
    fn load_compiler_weights(&mut self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != 3269 {
            return Err(PyRuntimeError::new_err(format!(
                "Compiler weights must be 3269 floats, got {}", weights.len()
            )));
        }
        self.compiler_weights_buf = Some(
            new_buf_f32(&self.device, &weights)
                .map_err(|e| PyRuntimeError::new_err(format!("Compiler weights buf: {e:?}")))?,
        );
        Ok(())
    }

    // ── Readiness checks ──────────────────────────────────────────────────────

    fn gic_ready(&self) -> bool { self.gic_weights_buf.is_some() }
    fn watchdog_ready(&self) -> bool { self.watchdog_weights_buf.is_some() }
    fn compiler_ready(&self) -> bool { self.compiler_weights_buf.is_some() }

    // ── Execution ─────────────────────────────────────────────────────────────

    /// Run neural GIC dispatch.
    ///
    /// input_state: [96] f32 — concatenation of IRR + ISR + IMR float bits
    /// pending_mask: [32] f32 — 1.0 where IRQ is pending, 0.0 otherwise
    ///
    /// Returns: [32] f32 priority scores (non-pending masked to -inf)
    fn execute_gic(
        &self,
        input_state: Vec<f32>,
        pending_mask: Vec<f32>,
    ) -> PyResult<Vec<f32>> {
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
        let weights = self.gic_weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("GIC weights not loaded — call load_gic_weights() first")
        })?;

        let input_buf = new_buf_f32(&self.device, &input_state)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output_buf = self.device
            .newBufferWithLength_options(32 * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("output buf alloc failed"))?;
        let mask_buf = new_buf_f32(&self.device, &pending_mask)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.gic_pipeline,
                    &[weights, &input_buf, &output_buf, &mask_buf], 1);
        Ok(read_buf_f32(&output_buf, 32))
    }

    /// Run neural Watchdog anomaly detection.
    ///
    /// metrics_window: [seq_len * 8] f32 — flattened metrics window
    /// seq_len: number of timesteps in the window
    ///
    /// Returns: [1] f32 anomaly score in [0, 1]
    fn execute_watchdog(
        &self,
        metrics_window: Vec<f32>,
        seq_len: u32,
    ) -> PyResult<f32> {
        let expected = seq_len as usize * 8;
        if metrics_window.len() != expected {
            return Err(PyRuntimeError::new_err(format!(
                "Watchdog input must be {} floats ({}*8), got {}",
                expected, seq_len, metrics_window.len()
            )));
        }
        let weights = self.watchdog_weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Watchdog weights not loaded — call load_watchdog_weights() first")
        })?;

        let input_buf = new_buf_f32(&self.device, &metrics_window)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output_buf = self.device
            .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("output buf alloc failed"))?;
        let params_buf = new_buf_u32(&self.device, seq_len)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.watchdog_pipeline,
                    &[weights, &input_buf, &output_buf, &params_buf], 1);
        let result = read_buf_f32(&output_buf, 1);
        Ok(result[0])
    }

    /// Run neural Compiler Optimizer scoring.
    ///
    /// windows_flat: [N * 15] f32 — N windows of (3 instructions × 5 features)
    /// n_windows: number of windows
    ///
    /// Returns: [N * 5] f32 optimization class scores (one row per window)
    fn execute_compiler(
        &self,
        windows_flat: Vec<f32>,
        n_windows: u32,
    ) -> PyResult<Vec<f32>> {
        let expected = n_windows as usize * 15;
        if windows_flat.len() != expected {
            return Err(PyRuntimeError::new_err(format!(
                "Compiler input must be {} floats ({}*15), got {}",
                expected, n_windows, windows_flat.len()
            )));
        }
        let weights = self.compiler_weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Compiler weights not loaded — call load_compiler_weights() first")
        })?;

        let n = n_windows as usize;
        let input_buf = new_buf_f32(&self.device, &windows_flat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output_buf = self.device
            .newBufferWithLength_options(n * 5 * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("output buf alloc failed"))?;
        let params_buf = new_buf_u32(&self.device, n_windows)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.compiler_pipeline,
                    &[weights, &input_buf, &output_buf, &params_buf], n.max(1));
        Ok(read_buf_f32(&output_buf, n * 5))
    }

    /// Benchmark all three kernels. Returns (gic_us, watchdog_us, compiler_us).
    fn benchmark(&self) -> PyResult<(f64, f64, f64)> {
        use std::time::Instant;

        // GIC benchmark: score 32 IRQs
        let gic_us = if self.gic_ready() {
            let input = vec![0.5f32; 96];
            let mask = vec![1.0f32; 32];
            let t0 = Instant::now();
            for _ in 0..1000 {
                let _ = self.execute_gic(input.clone(), mask.clone());
            }
            t0.elapsed().as_micros() as f64 / 1000.0
        } else { -1.0 };

        // Watchdog benchmark: score 64-step window
        let wd_us = if self.watchdog_ready() {
            let input = vec![0.5f32; 64 * 8];
            let t0 = Instant::now();
            for _ in 0..1000 {
                let _ = self.execute_watchdog(input.clone(), 64);
            }
            t0.elapsed().as_micros() as f64 / 1000.0
        } else { -1.0 };

        // Compiler benchmark: score 100 windows
        let co_us = if self.compiler_ready() {
            let input = vec![0.5f32; 100 * 15];
            let t0 = Instant::now();
            for _ in 0..1000 {
                let _ = self.execute_compiler(input.clone(), 100);
            }
            t0.elapsed().as_micros() as f64 / 1000.0
        } else { -1.0 };

        Ok((gic_us, wd_us, co_us))
    }
}

pub fn register_neural_os(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuralOSKernels>()?;
    Ok(())
}
