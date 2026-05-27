//! Neural ALU Kernel — runs trained .pt neural models on Metal GPU
//!
//! Ports the exact same forward passes as NeuralWeaveBatchALU (Python/PyTorch)
//! into Metal GPU shaders:
//!
//!   ADD/SUB  → neural Kogge-Stone CLA (carry_combine.pt + logical.pt)
//!   AND/OR/XOR → neural truth table lookup (logical.pt)
//!   MUL      → neural byte-pair LUT (multiply.pt)
//!   LSL/LSR  → neural shift precomputed LUT (lsl.pt / lsr.pt)
//!
//! Weight layout in the neural weights buffer (2494 f32):
//!   [0    .. 255  ]  carry_combiner FC1 weight [64 × 4]  row-major
//!   [256  .. 319  ]  carry_combiner FC1 bias   [64]
//!   [320  .. 2367 ]  carry_combiner FC2 weight [32 × 64] row-major
//!   [2368 .. 2399 ]  carry_combiner FC2 bias   [32]
//!   [2400 .. 2463 ]  carry_combiner FC3 weight [2 × 32]  row-major
//!   [2464 .. 2465 ]  carry_combiner FC3 bias   [2]
//!   [2466 .. 2493 ]  truth_tables [7 × 4]  row-major (AND=0, OR=1, XOR=2 …)
//!
//! MUL LUT buffer: 256 × 256 × 16 f32 = 1 048 576 floats (4 MB)
//!   Indexed as lut[(a_byte * 256 + b_byte) * 16 + bit]
//!   Each entry is a logit; sigmoid > 0.5 → bit = 1
//!
//! Shift LUT buffers (LSL + LSR): each 64 × 64 × 64 f32 = 262 144 floats (1 MB)
//!   Precomputed from NeuralShiftNet forward passes for all 64 shift amounts.
//!   shift_lut[k * 64*64 + i * 64 + j] = effective weight: source bit j → output bit i
//!   for shift_amount = k.  At runtime: output[i] = Σ_j(lut[k,i,j] * val_bits[j]) > 0.5

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
// Metal shader source
// ─────────────────────────────────────────────────────────────────────────────

const NEURAL_ALU_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// ── Weight buffer offsets ──────────────────────────────────────────────────
constant int CC_FC1_W = 0;      // [64, 4]  = 256 floats
constant int CC_FC1_B = 256;    // [64]
constant int CC_FC2_W = 320;    // [32, 64] = 2048 floats
constant int CC_FC2_B = 2368;   // [32]
constant int CC_FC3_W = 2400;   // [2, 32]  = 64 floats
constant int CC_FC3_B = 2464;   // [2]
constant int TT_BASE  = 2466;   // [7, 4]   = 28 floats

// ── Helpers ────────────────────────────────────────────────────────────────

inline float neural_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -15.0f, 15.0f)));
}

// Truth table lookup: row ∈ {0..6}, idx = a_bit*2 + b_bit
// AND=row0, OR=row1, XOR=row2, BIC=row3, ORN=row4, EON=row5, NOT=row6
inline int neural_tt(int row, int a_bit, int b_bit,
                     device const float* w) {
    float logit = w[TT_BASE + row * 4 + a_bit * 2 + b_bit];
    return neural_sigmoid(logit) > 0.5f ? 1 : 0;
}

// ── Carry combiner MLP [4 → 64 → 32 → 2] ─────────────────────────────────
// Input: (G_i, P_i, G_j, P_j) — each 0.0 or 1.0
// Writes *g_out, *p_out ← 0.0 or 1.0

void neural_carry_combine(float g_i, float p_i, float g_j, float p_j,
                           device const float* w,
                           thread float* g_out, thread float* p_out) {
    float inp[4] = {g_i, p_i, g_j, p_j};

    // FC1: [4] → [64] + ReLU
    float h1[64];
    for (int i = 0; i < 64; i++) {
        float s = w[CC_FC1_B + i];
        for (int j = 0; j < 4; j++)
            s += w[CC_FC1_W + i * 4 + j] * inp[j];
        h1[i] = max(0.0f, s);
    }

    // FC2: [64] → [32] + ReLU
    float h2[32];
    for (int i = 0; i < 32; i++) {
        float s = w[CC_FC2_B + i];
        for (int j = 0; j < 64; j++)
            s += w[CC_FC2_W + i * 64 + j] * h1[j];
        h2[i] = max(0.0f, s);
    }

    // FC3: [32] → [2]  (logits)
    float out0 = w[CC_FC3_B + 0];
    float out1 = w[CC_FC3_B + 1];
    for (int j = 0; j < 32; j++) {
        out0 += w[CC_FC3_W + 0 * 32 + j] * h2[j];
        out1 += w[CC_FC3_W + 1 * 32 + j] * h2[j];
    }

    *g_out = neural_sigmoid(out0) > 0.5f ? 1.0f : 0.0f;
    *p_out = neural_sigmoid(out1) > 0.5f ? 1.0f : 0.0f;
}

// ── Neural 32-bit Kogge-Stone CLA ─────────────────────────────────────────

void neural_cla(uint32_t a, uint32_t b, int carry_in,
                device const float* w,
                thread int* result_bits) {
    int a_bits[32], b_bits[32];
    for (int i = 0; i < 32; i++) {
        a_bits[i] = (int)((a >> i) & 1u);
        b_bits[i] = (int)((b >> i) & 1u);
    }

    // Initial G = AND(a,b),  P = XOR(a,b)
    float G[32], P[32];
    for (int i = 0; i < 32; i++) {
        G[i] = (float)neural_tt(0, a_bits[i], b_bits[i], w);
        P[i] = (float)neural_tt(2, a_bits[i], b_bits[i], w);
    }

    // Inject carry_in into bit 0: G[0]' = G[0] OR (P[0] AND carry_in)
    if (carry_in && (P[0] > 0.5f || G[0] > 0.5f)) G[0] = 1.0f;

    // Kogge-Stone prefix tree (5 stages, strides 1 2 4 8 16)
    for (int stage = 0; stage < 5; stage++) {
        int stride = 1 << stage;
        if (stride >= 32) break;

        float newG[32], newP[32];
        for (int i = 0; i < 32; i++) { newG[i] = G[i]; newP[i] = P[i]; }

        for (int i = stride; i < 32; i++) {
            float g_o, p_o;
            neural_carry_combine(G[i], P[i], G[i - stride], P[i - stride],
                                 w, &g_o, &p_o);
            newG[i] = g_o;
            newP[i] = p_o;
        }

        for (int i = 0; i < 32; i++) { G[i] = newG[i]; P[i] = newP[i]; }
    }

    // Carry vector: carry[0] = carry_in, carry[i] = G[i-1]
    int carry[32];
    carry[0] = carry_in;
    for (int i = 1; i < 32; i++)
        carry[i] = G[i - 1] > 0.5f ? 1 : 0;

    // Result bits: P_orig XOR carry
    for (int i = 0; i < 32; i++) {
        int p_orig = neural_tt(2, a_bits[i], b_bits[i], w);
        result_bits[i] = neural_tt(2, p_orig, carry[i], w);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Kernel 1: neural_add_batch
//   buffer(0): a_vals   [N] int64
//   buffer(1): b_vals   [N] int64
//   buffer(2): results  [N] int64   (output)
//   buffer(3): weights  [2494] f32
//   buffer(4): op_flags [1] uint32  bit0=is_sub, bit1=is_w32
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_add_batch(
    device const long* a_vals    [[buffer(0)]],
    device const long* b_vals    [[buffer(1)]],
    device       long* results   [[buffer(2)]],
    device const float* weights  [[buffer(3)]],
    device const uint* op_flags  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint flags = op_flags[0];
    int is_sub = (int)((flags >> 0u) & 1u);
    int is_w32 = (int)((flags >> 1u) & 1u);

    long av = a_vals[tid];
    long bv = b_vals[tid];

    uint32_t a = (uint32_t)((ulong)av & 0xFFFFFFFFUL);
    uint32_t b = (uint32_t)((ulong)bv & 0xFFFFFFFFUL);

    if (is_sub) b = ~b;
    int carry_in = is_sub ? 1 : 0;

    int rbits[32];
    neural_cla(a, b, carry_in, weights, rbits);

    long result = 0L;
    for (int i = 0; i < 32; i++)
        if (rbits[i]) result |= (long)(1u << i);

    if (!is_w32 && (result & 0x80000000L))
        result |= (long)0xFFFFFFFF00000000UL;
    else if (is_w32)
        result &= 0xFFFFFFFFL;

    results[tid] = result;
}

// ══════════════════════════════════════════════════════════════════════════
// Kernel 2: neural_logical_batch
//   buffer(0): a_vals   [N] int64
//   buffer(1): b_vals   [N] int64
//   buffer(2): results  [N] int64   (output)
//   buffer(3): weights  [2494] f32
//   buffer(4): op_idx   [1] uint32  (0=AND, 1=OR, 2=XOR, 3=BIC, 4=ORN, 5=EON)
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_logical_batch(
    device const long* a_vals    [[buffer(0)]],
    device const long* b_vals    [[buffer(1)]],
    device       long* results   [[buffer(2)]],
    device const float* weights  [[buffer(3)]],
    device const uint* op_idx    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int row = (int)op_idx[0];
    long av = a_vals[tid];
    long bv = b_vals[tid];

    long result = 0L;
    for (int i = 0; i < 32; i++) {
        int a_bit = (int)((av >> i) & 1L);
        int b_bit = (int)((bv >> i) & 1L);
        if (neural_tt(row, a_bit, b_bit, weights))
            result |= (long)(1LL << i);
    }

    if (result & 0x80000000L)
        result |= (long)0xFFFFFFFF00000000UL;

    results[tid] = result;
}

// ══════════════════════════════════════════════════════════════════════════
// Kernel 3: neural_mul_batch
//   buffer(0): a_vals    [N] int64
//   buffer(1): b_vals    [N] int64
//   buffer(2): results   [N] int64   (output)
//   buffer(3): lut_table [256*256*16] f32  (multiply.pt LUT, logits)
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_mul_batch(
    device const long* a_vals     [[buffer(0)]],
    device const long* b_vals     [[buffer(1)]],
    device       long* results    [[buffer(2)]],
    device const float* lut_table [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    long av = a_vals[tid];
    long bv = b_vals[tid];

    int neg_a = av < 0L;
    int neg_b = bv < 0L;
    int neg_result = neg_a ^ neg_b;

    ulong ua = (ulong)(neg_a ? -av : av) & 0xFFFFFFFFUL;
    ulong ub = (ulong)(neg_b ? -bv : bv) & 0xFFFFFFFFUL;

    uint a_bytes[4] = {
        (uint)(ua         & 0xFFUL),
        (uint)((ua >>  8) & 0xFFUL),
        (uint)((ua >> 16) & 0xFFUL),
        (uint)((ua >> 24) & 0xFFUL)
    };
    uint b_bytes[4] = {
        (uint)(ub         & 0xFFUL),
        (uint)((ub >>  8) & 0xFFUL),
        (uint)((ub >> 16) & 0xFFUL),
        (uint)((ub >> 24) & 0xFFUL)
    };

    ulong result = 0UL;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int lut_base = (int)((a_bytes[i] * 256u + b_bytes[j]) * 16u);
            uint pair_product = 0u;
            for (int bit = 0; bit < 16; bit++) {
                if (neural_sigmoid(lut_table[lut_base + bit]) > 0.5f)
                    pair_product |= (1u << bit);
            }
            result += (ulong)pair_product << ((i + j) * 8);
        }
    }

    result &= 0xFFFFFFFFUL;
    long signed_result = neg_result ? -(long)result : (long)result;
    results[tid] = signed_result;
}

// ══════════════════════════════════════════════════════════════════════════
// Kernel 4: neural_shift_batch  (LSL / LSR / ASR — 32-bit values)
//   buffer(0): a_vals     [N] int64   — values to shift (32-bit in int64)
//   buffer(1): shift_amts [N] int64   — shift amounts (lower 6 bits used)
//   buffer(2): results    [N] int64   — output
//   buffer(3): shift_lut  [64*64*64] f32
//              shift_lut[k*4096 + i*64 + j] = effective weight:
//              source bit j → output bit i, for shift_amount k ∈ [0,63]
//              Precomputed from NeuralShiftNet for all 64 amounts.
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_shift_batch(
    device const long*  a_vals     [[buffer(0)]],
    device const long*  shift_amts [[buffer(1)]],
    device       long*  results    [[buffer(2)]],
    device const float* shift_lut  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    long av = a_vals[tid];
    int k = (int)(shift_amts[tid] & 63L);

    // Unpack value bits (treat as unsigned 32-bit; upper bits of int64 zeroed)
    uint32_t av32 = (uint32_t)((ulong)av & 0xFFFFFFFFUL);
    float val_bits[64];
    for (int i = 0; i < 32; i++)
        val_bits[i] = (float)((int)((av32 >> i) & 1u));
    for (int i = 32; i < 64; i++)
        val_bits[i] = 0.0f;

    // Apply shift LUT: output[i] = Σ_j(lut[k,i,j] * val_bits[j]) > 0.5
    int lut_base = k * 64 * 64;
    long result = 0L;
    for (int i = 0; i < 32; i++) {
        float s = 0.0f;
        int row_base = lut_base + i * 64;
        for (int j = 0; j < 64; j++)
            s += shift_lut[row_base + j] * val_bits[j];
        if (s > 0.5f)
            result |= (long)(1L << i);
    }
    results[tid] = result;
}

// ══════════════════════════════════════════════════════════════════════════
// Kernel 5: neural_asr_batch  (32-bit arithmetic shift right, sign-extended)
//   Identical to neural_shift_batch except input is sign-extended to 64 bits
//   so the 64-bit ASR model correctly sees the sign bit at position 63.
//   buffer(0): a_vals    [N] int64   — 32-bit values (may be negative in int64)
//   buffer(1): shift_amts[N] int64   — shift amounts (lower 6 bits used)
//   buffer(2): results   [N] int64   — 32-bit sign-extended result
//   buffer(3): asr_lut   [64*64*64] f32
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_asr_batch(
    device const long*  a_vals     [[buffer(0)]],
    device const long*  shift_amts [[buffer(1)]],
    device       long*  results    [[buffer(2)]],
    device const float* asr_lut    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    long av = a_vals[tid];
    int k = (int)(shift_amts[tid] & 63L);

    // Sign-extend 32-bit value to 64 bits: bits 32..63 = bit 31
    uint32_t av32 = (uint32_t)((ulong)av & 0xFFFFFFFFUL);
    int sign_bit = (int)((av32 >> 31) & 1u);
    float val_bits[64];
    for (int i = 0; i < 32; i++)
        val_bits[i] = (float)((int)((av32 >> i) & 1u));
    for (int i = 32; i < 64; i++)
        val_bits[i] = (float)sign_bit;  // sign extension

    // Apply ASR LUT: output[i] = Σ_j(lut[k,i,j] * val_bits[j]) > 0.5
    int lut_base = k * 64 * 64;
    long result = 0L;
    for (int i = 0; i < 32; i++) {
        float s = 0.0f;
        int row_base = lut_base + i * 64;
        for (int j = 0; j < 64; j++)
            s += asr_lut[row_base + j] * val_bits[j];
        if (s > 0.5f)
            result |= (long)(1L << i);
    }
    // Sign-extend result to int64
    if (result & 0x80000000L)
        result |= (long)0xFFFFFFFF00000000UL;
    results[tid] = result;
}

// ══════════════════════════════════════════════════════════════════════════
// Kernel 6: neural_rol_batch  (64-bit rotate-left)
//   buffer(0): a_vals     [N] int64   — full 64-bit values to rotate
//   buffer(1): rot_amts   [N] int64   — rotation amounts (lower 6 bits used)
//   buffer(2): results    [N] int64   — 64-bit rotated output
//   buffer(3): rol_lut    [64*64*64] f32  — precomputed from rol.pt
// ══════════════════════════════════════════════════════════════════════════

kernel void neural_rol_batch(
    device const long*  a_vals    [[buffer(0)]],
    device const long*  rot_amts  [[buffer(1)]],
    device       long*  results   [[buffer(2)]],
    device const float* rol_lut   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    long av = a_vals[tid];
    int k = (int)(rot_amts[tid] & 63L);

    // Unpack all 64 bits of the int64 value
    float val_bits[64];
    for (int i = 0; i < 64; i++)
        val_bits[i] = (float)((int)((av >> i) & 1L));

    // Apply ROL LUT: output[i] = Σ_j(lut[k,i,j] * val_bits[j]) > 0.5
    int lut_base = k * 64 * 64;
    long result = 0L;
    for (int i = 0; i < 64; i++) {
        float s = 0.0f;
        int row_base = lut_base + i * 64;
        for (int j = 0; j < 64; j++)
            s += rol_lut[row_base + j] * val_bits[j];
        if (s > 0.5f)
            result |= (long)(1L << i);
    }
    results[tid] = result;
}
"##;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn compile_neural_alu_lib(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
    let source = NSString::from_str(NEURAL_ALU_SHADER);
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

fn new_buf_i64(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[i64],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let bytes = data.len() * 8;
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferCreationFailed)?;
    unsafe {
        let ptr = buf.contents().as_ptr() as *mut i64;
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

fn read_buf_i64(buf: &Retained<ProtocolObject<dyn MTLBuffer>>, n: usize) -> Vec<i64> {
    let mut out = vec![0i64; n];
    unsafe {
        let ptr = buf.contents().as_ptr() as *const i64;
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

/// Metal-based neural ALU — runs trained carry_combine + logical + MUL + shift on GPU.
///
/// Usage from Python:
///   kernel = NeuralALUKernel()
///   kernel.load_weights(cc_weights_flat, truth_tables_flat)
///   kernel.load_mul_lut(lut_flat)                    # optional, for MUL
///   kernel.load_shift_luts(lsl_flat, lsr_flat)       # optional, for LSL/LSR
///   results = kernel.execute_add(a_list, b_list, is_sub=False, is_w32=False)
///   results = kernel.execute_logical(a_list, b_list, op_idx=2)  # 0=AND 1=OR 2=XOR
///   results = kernel.execute_mul(a_list, b_list)
///   results = kernel.execute_shift(a_list, shift_list, is_left=True)
#[pyclass(unsendable)]
pub struct NeuralALUKernel {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    add_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    logical_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mul_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    shift_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    asr_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rol_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    weights_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    lut_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    shift_lsl_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    shift_lsr_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    shift_asr_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    rol_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

#[pymethods]
impl NeuralALUKernel {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = get_default_device()
            .ok_or_else(|| PyRuntimeError::new_err("No Metal device available"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create Metal command queue"))?;

        let lib = compile_neural_alu_lib(&device)
            .map_err(|e| PyRuntimeError::new_err(format!("neural_alu shader: {e:?}")))?;

        let add_pipeline = make_pipeline(&device, &lib, "neural_add_batch")
            .map_err(|e| PyRuntimeError::new_err(format!("add pipeline: {e:?}")))?;
        let logical_pipeline = make_pipeline(&device, &lib, "neural_logical_batch")
            .map_err(|e| PyRuntimeError::new_err(format!("logical pipeline: {e:?}")))?;
        let mul_pipeline = make_pipeline(&device, &lib, "neural_mul_batch")
            .map_err(|e| PyRuntimeError::new_err(format!("mul pipeline: {e:?}")))?;
        let shift_pipeline = make_pipeline(&device, &lib, "neural_shift_batch")
            .map_err(|e| PyRuntimeError::new_err(format!("shift pipeline: {e:?}")))?;
        let asr_pipeline = make_pipeline(&device, &lib, "neural_asr_batch")
            .map_err(|e| PyRuntimeError::new_err(format!("asr pipeline: {e:?}")))?;
        let rol_pipeline = make_pipeline(&device, &lib, "neural_rol_batch")
            .map_err(|e| PyRuntimeError::new_err(format!("rol pipeline: {e:?}")))?;

        Ok(Self {
            device,
            queue,
            add_pipeline,
            logical_pipeline,
            mul_pipeline,
            shift_pipeline,
            asr_pipeline,
            rol_pipeline,
            weights_buf: None,
            lut_buf: None,
            shift_lsl_buf: None,
            shift_lsr_buf: None,
            shift_asr_buf: None,
            rol_buf: None,
        })
    }

    /// Load carry_combiner weights + truth tables into a GPU buffer.
    ///
    /// cc_weights: flat f32 list of length 2466 (FC1 w/b, FC2 w/b, FC3 w/b)
    /// truth_tables: flat f32 list of length 28 ([7, 4] row-major, raw logits)
    fn load_weights(&mut self, cc_weights: Vec<f32>, truth_tables: Vec<f32>) -> PyResult<()> {
        if cc_weights.len() != 2466 {
            return Err(PyRuntimeError::new_err(format!(
                "cc_weights must be 2466 floats, got {}",
                cc_weights.len()
            )));
        }
        if truth_tables.len() != 28 {
            return Err(PyRuntimeError::new_err(format!(
                "truth_tables must be 28 floats, got {}",
                truth_tables.len()
            )));
        }
        let mut combined = cc_weights;
        combined.extend_from_slice(&truth_tables);  // total: 2494 floats
        self.weights_buf = Some(
            new_buf_f32(&self.device, &combined)
                .map_err(|e| PyRuntimeError::new_err(format!("weights buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Load the multiply LUT into a GPU buffer.
    ///
    /// lut_flat: flat f32 list of length 256*256*16 = 1 048 576
    fn load_mul_lut(&mut self, lut_flat: Vec<f32>) -> PyResult<()> {
        if lut_flat.len() != 256 * 256 * 16 {
            return Err(PyRuntimeError::new_err(format!(
                "lut_flat must be {} floats, got {}",
                256 * 256 * 16,
                lut_flat.len()
            )));
        }
        self.lut_buf = Some(
            new_buf_f32(&self.device, &lut_flat)
                .map_err(|e| PyRuntimeError::new_err(format!("lut buf: {e:?}")))?,
        );
        Ok(())
    }

    fn is_ready(&self) -> bool { self.weights_buf.is_some() }
    fn mul_ready(&self) -> bool { self.lut_buf.is_some() }
    fn shift_ready(&self) -> bool { self.shift_lsl_buf.is_some() && self.shift_lsr_buf.is_some() }
    fn asr_ready(&self) -> bool { self.shift_asr_buf.is_some() }
    fn rol_ready(&self) -> bool { self.rol_buf.is_some() }

    /// Execute batched neural ADD or SUB.
    /// Returns a Python list of int64 results.
    fn execute_add(
        &self,
        a_vals: Vec<i64>,
        b_vals: Vec<i64>,
        is_sub: bool,
        is_w32: bool,
    ) -> PyResult<Vec<i64>> {
        let n = a_vals.len();
        if n == 0 { return Ok(vec![]); }
        let weights = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("weights not loaded — call load_weights() first")
        })?;

        let a_buf = new_buf_i64(&self.device, &a_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let b_buf = new_buf_i64(&self.device, &b_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let r_buf = self.device
            .newBufferWithLength_options(n * 8, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("result buf alloc failed"))?;
        let flags_buf = new_buf_u32(&self.device, (is_sub as u32) | ((is_w32 as u32) << 1))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.add_pipeline,
                    &[&a_buf, &b_buf, &r_buf, weights, &flags_buf], n);
        Ok(read_buf_i64(&r_buf, n))
    }

    /// Execute batched neural logical (AND=0, OR=1, XOR=2, BIC=3, ORN=4, EON=5).
    fn execute_logical(
        &self,
        a_vals: Vec<i64>,
        b_vals: Vec<i64>,
        op_idx: u32,
    ) -> PyResult<Vec<i64>> {
        let n = a_vals.len();
        if n == 0 { return Ok(vec![]); }
        let weights = self.weights_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("weights not loaded — call load_weights() first")
        })?;

        let a_buf = new_buf_i64(&self.device, &a_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let b_buf = new_buf_i64(&self.device, &b_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let r_buf = self.device
            .newBufferWithLength_options(n * 8, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("result buf alloc failed"))?;
        let idx_buf = new_buf_u32(&self.device, op_idx)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        dispatch_1d(&self.queue, &self.logical_pipeline,
                    &[&a_buf, &b_buf, &r_buf, weights, &idx_buf], n);
        Ok(read_buf_i64(&r_buf, n))
    }

    /// Execute batched neural MUL using the byte-pair LUT.
    fn execute_mul(&self, a_vals: Vec<i64>, b_vals: Vec<i64>) -> PyResult<Vec<i64>> {
        let n = a_vals.len();
        if n == 0 { return Ok(vec![]); }
        let lut = self.lut_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("MUL LUT not loaded — call load_mul_lut() first")
        })?;

        let a_buf = new_buf_i64(&self.device, &a_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let b_buf = new_buf_i64(&self.device, &b_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let r_buf = self.device
            .newBufferWithLength_options(n * 8, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("result buf alloc failed"))?;

        dispatch_1d(&self.queue, &self.mul_pipeline,
                    &[&a_buf, &b_buf, &r_buf, lut], n);
        Ok(read_buf_i64(&r_buf, n))
    }

    /// Load precomputed shift LUTs (LSL + LSR) into GPU buffers.
    ///
    /// Each flat list must be 64 * 64 * 64 = 262 144 floats.
    /// lsl_flat[k*4096 + i*64 + j] = effective weight: source bit j → output bit i
    /// for left-shift amount k.  Same layout for lsr_flat (right shift).
    fn load_shift_luts(&mut self, lsl_flat: Vec<f32>, lsr_flat: Vec<f32>) -> PyResult<()> {
        const SHIFT_LUT_LEN: usize = 64 * 64 * 64;
        if lsl_flat.len() != SHIFT_LUT_LEN {
            return Err(PyRuntimeError::new_err(format!(
                "lsl_flat must be {} floats, got {}", SHIFT_LUT_LEN, lsl_flat.len()
            )));
        }
        if lsr_flat.len() != SHIFT_LUT_LEN {
            return Err(PyRuntimeError::new_err(format!(
                "lsr_flat must be {} floats, got {}", SHIFT_LUT_LEN, lsr_flat.len()
            )));
        }
        self.shift_lsl_buf = Some(
            new_buf_f32(&self.device, &lsl_flat)
                .map_err(|e| PyRuntimeError::new_err(format!("lsl buf: {e:?}")))?,
        );
        self.shift_lsr_buf = Some(
            new_buf_f32(&self.device, &lsr_flat)
                .map_err(|e| PyRuntimeError::new_err(format!("lsr buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Execute batched neural shift (LSL or LSR) using the precomputed LUT.
    /// is_left=True → LSL, is_left=False → LSR.
    fn execute_shift(
        &self,
        a_vals: Vec<i64>,
        shift_amts: Vec<i64>,
        is_left: bool,
    ) -> PyResult<Vec<i64>> {
        let n = a_vals.len();
        if n == 0 { return Ok(vec![]); }
        let lut = if is_left {
            self.shift_lsl_buf.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("shift LUTs not loaded — call load_shift_luts() first")
            })?
        } else {
            self.shift_lsr_buf.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("shift LUTs not loaded — call load_shift_luts() first")
            })?
        };

        let a_buf = new_buf_i64(&self.device, &a_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let s_buf = new_buf_i64(&self.device, &shift_amts)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let r_buf = self.device
            .newBufferWithLength_options(n * 8, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("result buf alloc failed"))?;

        dispatch_1d(&self.queue, &self.shift_pipeline,
                    &[&a_buf, &s_buf, &r_buf, lut], n);
        Ok(read_buf_i64(&r_buf, n))
    }

    /// Load ASR and ROL precomputed LUTs into GPU buffers.
    ///
    /// asr_flat: 64*64*64 = 262 144 floats — ASR LUT with fill baked in
    ///   (fill positions have all weight on source bit 31, the sign bit)
    /// rol_flat: 64*64*64 = 262 144 floats — 64-bit rotate-left LUT
    fn load_asr_rol_luts(&mut self, asr_flat: Vec<f32>, rol_flat: Vec<f32>) -> PyResult<()> {
        const LUT_LEN: usize = 64 * 64 * 64;
        if asr_flat.len() != LUT_LEN {
            return Err(PyRuntimeError::new_err(format!(
                "asr_flat must be {} floats, got {}", LUT_LEN, asr_flat.len()
            )));
        }
        if rol_flat.len() != LUT_LEN {
            return Err(PyRuntimeError::new_err(format!(
                "rol_flat must be {} floats, got {}", LUT_LEN, rol_flat.len()
            )));
        }
        self.shift_asr_buf = Some(
            new_buf_f32(&self.device, &asr_flat)
                .map_err(|e| PyRuntimeError::new_err(format!("asr buf: {e:?}")))?,
        );
        self.rol_buf = Some(
            new_buf_f32(&self.device, &rol_flat)
                .map_err(|e| PyRuntimeError::new_err(format!("rol buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Execute batched neural ASR (arithmetic shift right) — 32-bit values.
    fn execute_asr(&self, a_vals: Vec<i64>, shift_amts: Vec<i64>) -> PyResult<Vec<i64>> {
        let n = a_vals.len();
        if n == 0 { return Ok(vec![]); }
        let lut = self.shift_asr_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("ASR LUT not loaded — call load_asr_rol_luts() first")
        })?;
        let a_buf = new_buf_i64(&self.device, &a_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let s_buf = new_buf_i64(&self.device, &shift_amts)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let r_buf = self.device
            .newBufferWithLength_options(n * 8, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("result buf alloc failed"))?;
        dispatch_1d(&self.queue, &self.asr_pipeline,
                    &[&a_buf, &s_buf, &r_buf, lut], n);
        Ok(read_buf_i64(&r_buf, n))
    }

    /// Execute batched neural ROL (64-bit rotate left).
    fn execute_rol(&self, a_vals: Vec<i64>, rot_amts: Vec<i64>) -> PyResult<Vec<i64>> {
        let n = a_vals.len();
        if n == 0 { return Ok(vec![]); }
        let lut = self.rol_buf.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("ROL LUT not loaded — call load_asr_rol_luts() first")
        })?;
        let a_buf = new_buf_i64(&self.device, &a_vals)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let r_amts_buf = new_buf_i64(&self.device, &rot_amts)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let r_buf = self.device
            .newBufferWithLength_options(n * 8, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("result buf alloc failed"))?;
        dispatch_1d(&self.queue, &self.rol_pipeline,
                    &[&a_buf, &r_amts_buf, &r_buf, lut], n);
        Ok(read_buf_i64(&r_buf, n))
    }

    /// Benchmark: run N additions and return (n_ops, elapsed_secs, IPS).
    fn benchmark_add(&self, n: usize) -> PyResult<(usize, f64, f64)> {
        if !self.is_ready() {
            return Err(PyRuntimeError::new_err("weights not loaded"));
        }
        let a: Vec<i64> = (0..n as i64).map(|i| i * 13 + 7).collect();
        let b: Vec<i64> = (0..n as i64).map(|i| i * 7 + 3).collect();

        // Warmup
        let _ = self.execute_add(a[..n.min(32)].to_vec(), b[..n.min(32)].to_vec(), false, false)?;

        let t0 = Instant::now();
        let _ = self.execute_add(a, b, false, false)?;
        let elapsed = t0.elapsed().as_secs_f64();
        let ips = if elapsed > 0.0 { n as f64 / elapsed } else { 0.0 };
        Ok((n, elapsed, ips))
    }
}

pub fn register_neural_alu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuralALUKernel>()?;
    Ok(())
}
