# Deploying MLPs on Thread-Limited GPU Hardware: The Three-Pass Technique

A practical guide to deploying multi-layer perceptrons on Apple Silicon, mobile GPUs, and embedded GPU hardware where per-thread stack memory is severely constrained. This technique was developed during the nCPU project's Metal neural display work and is generalizable to any MLP deployment on thread-limited hardware.

**Author:** Robert Price
**Origin:** nCPU Neural Display (April 2026)
**Measured on:** Apple M-series, Metal compute shaders

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [The Three-Pass Solution](#the-three-pass-solution)
3. [Architecture Diagram](#architecture-diagram)
4. [Weight Buffer Layout](#weight-buffer-layout)
5. [The GELU NaN Guard](#the-gelu-nan-guard)
6. [Memory Budget Analysis](#memory-budget-analysis)
7. [Performance Characteristics](#performance-characteristics)
8. [When to Use This Technique](#when-to-use-this-technique)
9. [When NOT to Use This Technique](#when-not-to-use-this-technique)
10. [Generalizing Beyond nCPU](#generalizing-beyond-ncpu)
11. [Code Template: Metal Shader Language](#code-template-metal-shader-language)
12. [Code Template: Rust + PyO3 Host Code](#code-template-rust--pyo3-host-code)
13. [Weight Extraction from PyTorch](#weight-extraction-from-pytorch)
14. [Verification and Debugging](#verification-and-debugging)

---

## Problem Statement

Apple Silicon (and many mobile and embedded GPUs) limit per-thread stack memory to approximately 1 KB. This is a hard constraint imposed by the GPU scheduler: each thread in a compute kernel has access to a small, fixed-size region of fast memory for local variables. Exceeding this budget does not produce a clean error --- the hardware silently aliases stack writes, corrupting register spills and producing garbage output.

A standard 3-layer MLP with `hidden_dim=256` needs 256 x 4 = 1,024 bytes just for one hidden layer's activation array. A naive single-pass implementation stores hidden activations in thread-local arrays like this:

```metal
// BROKEN on Apple Silicon: h1[256] alone consumes the entire ~1 KB stack
kernel void naive_mlp(
    device const float* weights [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float h1[256];  // 1,024 bytes -- at or over the stack limit
    float h2[256];  // another 1,024 bytes -- definitely over

    // ... compute FC1 into h1, FC2 into h2, FC3 into output ...
}
```

**The failure mode is insidious.** The kernel does not crash. It does not raise an error. It does not report a stack overflow. It silently produces wrong results for a subset of inputs --- in the nCPU case, approximately 40% of characters rendered with NaN-corrupted pixels. The corruption pattern depends on thread scheduling and register pressure, making it non-deterministic and extremely difficult to diagnose.

This is not an Apple-specific quirk. The same constraint exists on:

- **Apple Silicon (M1/M2/M3/M4):** ~1 KB per-thread stack
- **ARM Mali GPUs (mobile Android):** typically 512 bytes -- 1 KB per-thread stack
- **Qualcomm Adreno GPUs (mobile Android):** similarly constrained
- **Embedded GPU accelerators:** often even more limited

Any MLP with `hidden_dim >= 256` running on these platforms faces this problem.

---

## The Three-Pass Solution

The core insight: **decompose the MLP forward pass into N sequential compute dispatches, one per hidden layer that exceeds the stack budget, and store all intermediate activations in device-memory buffers (GPU VRAM) instead of thread-local arrays.**

Each pass:
1. Reads its input from a device-memory `MTLBuffer`
2. Computes one layer's transformation using only scalar accumulators (a single `float` variable, not an array)
3. Writes each output element directly to a device-memory `MTLBuffer`

The critical property: **zero bytes of thread-local array storage per thread.** Each thread holds at most one scalar accumulator at a time, computes one output element, writes it to device memory, and moves to the next.

For a 3-layer MLP (FC1 -> FC2 -> FC3), this produces three compute passes:

| Pass | Input | Computation | Output | Thread-Local Storage |
|------|-------|-------------|--------|---------------------|
| **Pass 1** | Input data | Embedding + FC1 + GELU | `h1_buf` (device memory) | 0 arrays (1 scalar) |
| **Pass 2** | `h1_buf` | FC2 + GELU | `h2_buf` (device memory) | 0 arrays (1 scalar) |
| **Pass 3** | `h2_buf` | FC3 + Sigmoid + postprocessing | Final output | 0 arrays (1 scalar) |

All three passes are encoded into a **single Metal command buffer**. The GPU guarantees serial execution between compute command encoders within the same command buffer, so no explicit synchronization (barriers, fences, events) is needed between passes.

---

## Architecture Diagram

```
                          GPU VRAM (Device Memory)
                    ┌─────────────────────────────────┐
                    │                                 │
 ┌──────────┐      │  ┌──────────────────────────┐   │
 │  Input   │      │  │     weights_buf           │   │
 │ char_buf │      │  │  (all layers, contiguous) │   │
 │ (1920 B) │      │  │     527,040 bytes         │   │
 └────┬─────┘      │  └────────────┬─────────────┘   │
      │            │               │                  │
      ▼            │               ▼                  │
 ┌─────────────────────────────────────────────────┐  │
 │              PASS 1  (1920 threads)             │  │
 │                                                 │  │
 │  for each cell:                                 │  │
 │    s = bias[i]                 ← scalar only    │  │
 │    for j in 0..64:                              │  │
 │      s += W[i,j] * embed[ch,j]                 │  │
 │    h1_buf[cell*256 + i] = GELU(s)              │  │
 └───────────────────┬─────────────────────────────┘  │
                     │                                │
                     ▼                                │
              ┌─────────────┐                         │
              │   h1_buf    │  1,966,080 bytes         │
              │ (1920×256)  │  (~1.9 MB)              │
              └──────┬──────┘                         │
                     │                                │
                     ▼                                │
 ┌─────────────────────────────────────────────────┐  │
 │              PASS 2  (1920 threads)             │  │
 │                                                 │  │
 │  for each cell:                                 │  │
 │    s = bias[i]                 ← scalar only    │  │
 │    for j in 0..256:                             │  │
 │      s += W[i,j] * h1_buf[cell*256 + j]        │  │
 │    h2_buf[cell*256 + i] = GELU(s)              │  │
 └───────────────────┬─────────────────────────────┘  │
                     │                                │
                     ▼                                │
              ┌─────────────┐                         │
              │   h2_buf    │  1,966,080 bytes         │
              │ (1920×256)  │  (~1.9 MB)              │
              └──────┬──────┘                         │
                     │                                │
                     ▼                                │
 ┌─────────────────────────────────────────────────┐  │
 │              PASS 3  (1920 threads)             │  │
 │                                                 │  │
 │  for each cell:                                 │  │
 │    alpha = sigmoid(FC3(h2))    ← scalar only    │  │
 │    pixel = alpha*fg + (1-a)*bg                  │  │
 │    framebuf[...] = pixel                        │  │
 └───────────────────┬─────────────────────────────┘  │
                     │                                │
                     ▼                                │
              ┌─────────────┐                         │
              │  framebuf   │  737,280 bytes            │
              │ (384×640×3) │  RGB output             │
              └─────────────┘                         │
                    │                                 │
                    └─────────────────────────────────┘
```

Key points:
- `h1_buf` and `h2_buf` are pre-allocated once at initialization and reused across renders
- All buffers use `StorageModeShared` for zero-copy CPU readback (Apple Silicon unified memory)
- No thread-local arrays exist at any point --- each thread holds one `float s` accumulator

---

## Weight Buffer Layout

All model weights are packed into a single contiguous `MTLBuffer` in a known order. This eliminates per-layer buffer management and minimizes the number of buffer bindings per compute encoder.

For the nCPU neural display (131,760 floats = 527,040 bytes):

```
Offset (floats)   Size (floats)   Layer                Shape
─────────────────────────────────────────────────────────────────
0                  16,384          embed.weight         [256, 64]    row-major
16,384             16,384          net.0.weight (FC1)   [256, 64]    row-major
32,768                256          net.0.bias   (FC1)   [256]
33,024             65,536          net.2.weight (FC2)   [256, 256]   row-major
98,560                256          net.2.bias   (FC2)   [256]
98,816             32,768          net.4.weight (FC3)   [128, 256]   row-major
131,584               128          net.4.bias   (FC3)   [128]
131,712                48          palette.weight       [16, 3]
─────────────────────────────────────────────────────────────────
Total:            131,760 floats = 527,040 bytes
```

The shader accesses weights via compile-time constant offsets:

```metal
constant int EMBED_W   = 0;        // [256, 64]  = 16384 floats
constant int FC1_W     = 16384;    // [256, 64]  = 16384 floats
constant int FC1_B     = 32768;    // [256]      = 256 floats
constant int FC2_W     = 33024;    // [256, 256] = 65536 floats
constant int FC2_B     = 98560;    // [256]      = 256 floats
constant int FC3_W     = 98816;    // [128, 256] = 32768 floats
constant int FC3_B     = 131584;   // [128]      = 128 floats
constant int PALETTE   = 131712;   // [16, 3]    = 48 floats
```

**Convention:** Weights are stored in row-major order matching PyTorch's default layout. For a linear layer with shape `[out_features, in_features]`, element `W[i][j]` is at offset `base + i * in_features + j`. This means the shader's inner loop iterates over contiguous memory when computing dot products, which is cache-friendly on the GPU.

---

## The GELU NaN Guard

The FC2 layer computes dot products over 256 terms. For certain input patterns (especially rare character embeddings), these sums can overflow to negative infinity. Under IEEE 754 arithmetic, the standard GELU formula hits a trap:

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

When `x` approaches negative infinity:
1. `x^3` overflows to `-inf`
2. `tanh(-inf)` = `-1`
3. `(1 + (-1))` = `0`
4. `0.5 * (-inf) * 0` = **NaN** (IEEE 754: infinity times zero is undefined)

This NaN propagates through all subsequent layers, silently corrupting the output. The failure is intermittent because it depends on which character codes produce embedding vectors that cause the overflow.

**The fix: a saturating guard.**

```metal
inline float neural_gelu(float x) {
    // Guard: for |x| > 10, GELU saturates to 0 (negative) or x (positive).
    // Without this, x^3 can overflow to +/-inf for large |x|, and then
    // GELU(-inf) = 0.5 * (-inf) * (1 + tanh(-inf)) = (-inf) * 0 = NaN.
    if (x < -10.0f) return 0.0f;
    if (x >  10.0f) return x;
    float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}
```

**Why -10/+10?** This is mathematically exact to well beyond floating-point precision:
- `GELU(-10) = 2.27 x 10^-23` (effectively zero in float32, which has ~7 decimal digits)
- `GELU(10) = 10 - 2.27 x 10^-23` (effectively 10)

The guard introduces zero error for any realistic input while completely eliminating the NaN pathway. Apply the same pattern to any activation function that involves products of potentially-infinite terms.

Similarly, clamp sigmoid inputs to prevent `exp(-inf)` and `exp(inf)`:

```metal
inline float neural_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -15.0f, 15.0f)));
}
```

---

## Memory Budget Analysis

### Thread-Local Memory

| Approach | Per-Thread Storage | Status on Apple Silicon |
|----------|-------------------|------------------------|
| Naive single-pass | `h1[256]` + `h2[256]` = 2,048 bytes | Silent corruption |
| Naive single-pass | `h1[256]` = 1,024 bytes | At the limit, unreliable |
| **Three-pass** | **0 bytes** (scalar accumulators only) | **Safe on any GPU** |

### Device Memory (GPU VRAM)

For the nCPU display case (N = 1,920 cells, hidden_dim = 256):

| Buffer | Size | Calculation |
|--------|------|-------------|
| `weights_buf` | 527,040 bytes (0.5 MB) | 131,760 floats x 4 bytes |
| `h1_buf` | 1,966,080 bytes (1.9 MB) | 1,920 x 256 x 4 bytes |
| `h2_buf` | 1,966,080 bytes (1.9 MB) | 1,920 x 256 x 4 bytes |
| `char_buf` | 1,920 bytes | 1,920 x 1 byte |
| `fg_buf` | 1,920 bytes | 1,920 x 1 byte |
| `bg_buf` | 1,920 bytes | 1,920 x 1 byte |
| `frame_buf` | 737,280 bytes (0.7 MB) | 384 x 640 x 3 bytes |
| **Total** | **~5.2 MB** | |

5.2 MB is trivial compared to GPU VRAM (8--192 GB on Apple Silicon). Even on mobile GPUs with 1--4 GB of shared memory, intermediate buffers for typical MLP deployments will be in the single-digit megabyte range.

**General formula for intermediate buffer size:**

```
buffer_bytes = N_items x hidden_dim x sizeof(float)
```

Where `N_items` is the batch size (number of independent inputs processed in parallel) and `hidden_dim` is the size of the hidden layer. You need one intermediate buffer per pass boundary.

---

## Performance Characteristics

Measured on Apple M-series hardware:

| Metric | Value |
|--------|-------|
| **Throughput** | 305 FPS (integrated path), 331 FPS (raw kernel, dense scenes) |
| **Speedup vs. PyTorch CPU** | 4.4x |
| **Fidelity (PSNR)** | 68.7 dB against PyTorch reference |
| **Exact pixel match** | 99.13% across all 95 printable ASCII characters |
| **Max per-pixel error** | 1 (out of 255) |
| **Items per frame** | 1,920 (24 rows x 80 columns) |
| **Threads per dispatch** | 1,920 (one per cell) |
| **Dispatches per frame** | 3 (one per pass) |
| **GPU command buffers per frame** | 1 |

### Why Three Passes Are Fast

The overhead of encoding three compute passes instead of one is negligible for two reasons:

1. **Encoder setup is cheap.** Creating a `MTLComputeCommandEncoder`, setting a pipeline state, binding 3--5 buffers, and calling `dispatchThreadgroups` is on the order of microseconds. The actual GPU compute (matrix-vector products over 256 dimensions) dominates by 1000x or more.

2. **No synchronization overhead.** Metal guarantees that compute command encoders within a single command buffer execute serially. There is no need for `MTLFence`, `MTLEvent`, or memory barriers between passes. The GPU hardware handles the ordering automatically.

3. **Full parallelism within each pass.** All 1,920 threads execute simultaneously within each pass. The GPU's thread scheduler distributes work across all available compute units with no coordination between threads.

### Threadgroup Sizing

For optimal GPU occupancy, use smaller threadgroups rather than mapping the entire grid to a single group:

```rust
// tg_w=20 gives 4x24=96 threadgroups (vs 1x24=24 with tg_w=80)
let tg_w = 20usize;
let groups_x = (80 + tg_w - 1) / tg_w;
enc.dispatchThreadgroups_threadsPerThreadgroup(
    MTLSize { width: groups_x, height: 24, depth: 1 },
    MTLSize { width: tg_w, height: 1, depth: 1 },
);
```

More threadgroups give the GPU scheduler more flexibility to fill compute units.

---

## When to Use This Technique

Use the three-pass technique when **all** of the following are true:

- **Your MLP's `hidden_dim x sizeof(float)` exceeds the GPU's per-thread stack limit.** On Apple Silicon this is approximately 1 KB. For float32, that means `hidden_dim >= 256`.
- **You are targeting mobile or embedded GPUs.** Apple Silicon (M-series, A-series), ARM Mali, Qualcomm Adreno, or embedded NPU/GPU accelerators.
- **You want framework-free inference.** Weights as flat buffers, no PyTorch/TensorFlow/ONNX runtime dependency. Just native GPU compute.
- **You need deterministic, bit-reproducible results.** No framework version sensitivity, no Python overhead, no dynamic memory allocation at inference time.
- **Your workload is inference-only.** This technique stores intermediate activations in device memory for forward-pass use only; it does not retain the computation graph for backpropagation.

---

## When NOT to Use This Technique

- **Desktop GPUs (NVIDIA, AMD) with large per-thread stack.** CUDA devices typically allow 1 KB -- 512 KB per-thread local memory (configurable). If your hidden activations fit, a single-pass kernel is simpler and equally fast.
- **Very small MLPs where hidden activations fit in approximately 256 bytes.** If `hidden_dim <= 64`, thread-local arrays are fine.
- **When you need backpropagation.** This is an inference-only deployment technique. For training, use PyTorch or a framework that manages the computation graph.
- **Extremely deep networks (50+ layers).** Each layer that exceeds the stack budget adds one compute pass. For very deep networks, consider chunking multiple layers per pass where intermediate dimensions are small enough.

---

## Generalizing Beyond nCPU

The three-pass technique works for **any MLP** deployed on thread-limited GPU hardware. Common use cases:

- **NLP embedding lookup + classification:** Character or token embeddings followed by a few dense layers for sentiment, NER, or intent classification.
- **Recommendation models:** User/item embedding dot products with MLP towers.
- **Small vision classifiers:** Flattened feature vectors from a small CNN processed by dense layers.
- **Audio feature extraction:** Frame-level MLP processing of spectral features.
- **Sensor fusion on edge devices:** Combining multiple sensor inputs through dense layers on mobile GPUs.

### Adaptation Recipe

1. **Identify layers that exceed the stack budget.** For each hidden layer, compute `hidden_dim x sizeof(float)`. Any layer where this exceeds your target GPU's per-thread stack limit needs its own pass.

2. **Pack all weights into a single contiguous buffer.** Flatten each layer's weight matrix and bias vector in row-major order, concatenate them, and record the byte offset of each layer.

3. **Write one kernel function per pass.** Each kernel reads from one device-memory buffer, performs one layer's computation using only scalar accumulators, and writes to another device-memory buffer.

4. **Encode all passes into a single command buffer.** This gives you free serialization guarantees from the GPU.

5. **Pre-allocate all intermediate buffers at initialization.** Reuse them across inference calls. Do not allocate per-frame.

### Combining with Quantization

The intermediate buffers are the main memory cost. You can reduce them with quantized storage:

| Format | Buffer size (N=1920, dim=256) | Precision loss |
|--------|-------------------------------|----------------|
| float32 | 1.9 MB | None |
| float16 | 0.95 MB | Minimal for inference |
| int8 | 0.47 MB | Requires calibration |

Weight buffers can also be quantized. For float16 weights, replace `device const float*` with `device const half*` in the shader and cast to float for computation.

---

## Code Template: Metal Shader Language

The following is a generic Metal shader for a 3-layer MLP using the three-pass technique. Replace the constants with your model's dimensions.

```metal
#include <metal_stdlib>
using namespace metal;

// ── Model dimensions (customize these) ─────────────────────────────────────
constant int INPUT_DIM   = 64;    // input feature dimension
constant int HIDDEN_DIM  = 256;   // hidden layer dimension
constant int OUTPUT_DIM  = 128;   // output dimension
constant int N_ITEMS     = 1920;  // batch size (number of parallel items)

// ── Weight buffer offsets (compute from your model) ────────────────────────
//    Pack order: FC1_weight, FC1_bias, FC2_weight, FC2_bias, FC3_weight, FC3_bias
constant int FC1_W = 0;                                    // [HIDDEN, INPUT]
constant int FC1_B = HIDDEN_DIM * INPUT_DIM;               // [HIDDEN]
constant int FC2_W = FC1_B + HIDDEN_DIM;                   // [HIDDEN, HIDDEN]
constant int FC2_B = FC2_W + HIDDEN_DIM * HIDDEN_DIM;      // [HIDDEN]
constant int FC3_W = FC2_B + HIDDEN_DIM;                   // [OUTPUT, HIDDEN]
constant int FC3_B = FC3_W + OUTPUT_DIM * HIDDEN_DIM;      // [OUTPUT]

// ── Activation functions with NaN guards ───────────────────────────────────

inline float safe_gelu(float x) {
    if (x < -10.0f) return 0.0f;
    if (x >  10.0f) return x;
    float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

inline float safe_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -15.0f, 15.0f)));
}

// You can substitute safe_relu, safe_silu, etc. as needed:
// inline float safe_relu(float x) { return max(x, 0.0f); }

// ══════════════════════════════════════════════════════════════════════════════
// Pass 1: Input -> FC1 (INPUT_DIM -> HIDDEN_DIM) + GELU -> h1_buf
//   Thread-local storage: 0 arrays, 1 scalar accumulator
// ══════════════════════════════════════════════════════════════════════════════

kernel void mlp_pass1(
    device const float* input    [[buffer(0)]],  // [N_ITEMS, INPUT_DIM]
    device const float* weights  [[buffer(1)]],  // all weights, contiguous
    device       float* h1_buf   [[buffer(2)]],  // [N_ITEMS, HIDDEN_DIM] output
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N_ITEMS) return;

    int in_base = tid * INPUT_DIM;
    int h1_base = tid * HIDDEN_DIM;

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = weights[FC1_B + i];  // bias
        for (int j = 0; j < INPUT_DIM; j++) {
            s += weights[FC1_W + i * INPUT_DIM + j] * input[in_base + j];
        }
        h1_buf[h1_base + i] = safe_gelu(s);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Pass 2: h1_buf -> FC2 (HIDDEN_DIM -> HIDDEN_DIM) + GELU -> h2_buf
//   Thread-local storage: 0 arrays, 1 scalar accumulator
// ══════════════════════════════════════════════════════════════════════════════

kernel void mlp_pass2(
    device const float* weights  [[buffer(0)]],
    device const float* h1_buf   [[buffer(1)]],  // [N_ITEMS, HIDDEN_DIM] input
    device       float* h2_buf   [[buffer(2)]],  // [N_ITEMS, HIDDEN_DIM] output
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N_ITEMS) return;

    int base = tid * HIDDEN_DIM;

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float s = weights[FC2_B + i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            s += weights[FC2_W + i * HIDDEN_DIM + j] * h1_buf[base + j];
        }
        h2_buf[base + i] = safe_gelu(s);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Pass 3: h2_buf -> FC3 (HIDDEN_DIM -> OUTPUT_DIM) + Sigmoid -> output_buf
//   Thread-local storage: 0 arrays, 1 scalar accumulator
// ══════════════════════════════════════════════════════════════════════════════

kernel void mlp_pass3(
    device const float* weights    [[buffer(0)]],
    device const float* h2_buf     [[buffer(1)]],  // [N_ITEMS, HIDDEN_DIM] input
    device       float* output_buf [[buffer(2)]],  // [N_ITEMS, OUTPUT_DIM] output
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)N_ITEMS) return;

    int h2_base  = tid * HIDDEN_DIM;
    int out_base = tid * OUTPUT_DIM;

    for (int i = 0; i < OUTPUT_DIM; i++) {
        float s = weights[FC3_B + i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            s += weights[FC3_W + i * HIDDEN_DIM + j] * h2_buf[h2_base + j];
        }
        output_buf[out_base + i] = safe_sigmoid(s);
    }
}
```

**Adapting for your model:**

1. Replace `INPUT_DIM`, `HIDDEN_DIM`, `OUTPUT_DIM`, and `N_ITEMS` with your values.
2. Update the weight offset constants to match your packing order.
3. Substitute activation functions as needed (ReLU, SiLU, tanh, etc.), always including NaN guards for unbounded activations.
4. If your model has more than 3 layers, add additional passes. If a hidden layer's dimension is small enough to fit in thread-local memory (dim x 4 < 512 bytes to be safe), you can merge that layer into an adjacent pass.

---

## Code Template: Rust + PyO3 Host Code

The following Rust boilerplate sets up the three Metal compute pipelines, allocates buffers, and dispatches the passes. This uses the `objc2-metal` crate for Metal bindings and `pyo3` for Python interop.

```rust
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

// ── Constants (match your shader) ──────────────────────────────────────────
const N_ITEMS: usize = 1920;
const INPUT_DIM: usize = 64;
const HIDDEN_DIM: usize = 256;
const OUTPUT_DIM: usize = 128;

const H_BUF_FLOATS: usize = N_ITEMS * HIDDEN_DIM;
const H_BUF_BYTES: usize = H_BUF_FLOATS * 4;
const OUTPUT_BYTES: usize = N_ITEMS * OUTPUT_DIM * 4;

// Include your Metal shader source as a string literal
const SHADER_SOURCE: &str = r##"
    // ... your Metal shader code here ...
"##;

// ── Helper: compile shader library ─────────────────────────────────────────
fn compile_library(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    let source = NSString::from_str(SHADER_SOURCE);
    device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|e| format!("Shader compilation failed: {e:?}"))
}

// ── Helper: create compute pipeline ────────────────────────────────────────
fn make_pipeline(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    lib: &Retained<ProtocolObject<dyn MTLLibrary>>,
    fn_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let name = NSString::from_str(fn_name);
    let func = lib
        .newFunctionWithName(&name)
        .ok_or_else(|| format!("Function '{fn_name}' not found in shader library"))?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| format!("Pipeline creation failed: {e:?}"))
}

#[pyclass(unsendable)]
pub struct ThreePassMLP {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pass1_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pass2_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pass3_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    weights_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    // Pre-allocated intermediate buffers (reused across inference calls)
    h1_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    h2_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

#[pymethods]
impl ThreePassMLP {
    #[new]
    pub fn new() -> PyResult<Self> {
        // Get the default Metal device
        let device = objc2_metal::MTLCreateSystemDefaultDevice()
            .ok_or_else(|| PyRuntimeError::new_err("No Metal device available"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create command queue"))?;

        // Compile shader and create pipelines
        let lib = compile_library(&device)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        let pass1_pipeline = make_pipeline(&device, &lib, "mlp_pass1")
            .map_err(|e| PyRuntimeError::new_err(e))?;
        let pass2_pipeline = make_pipeline(&device, &lib, "mlp_pass2")
            .map_err(|e| PyRuntimeError::new_err(e))?;
        let pass3_pipeline = make_pipeline(&device, &lib, "mlp_pass3")
            .map_err(|e| PyRuntimeError::new_err(e))?;

        // Pre-allocate intermediate buffers with StorageModeShared
        // (zero-copy on Apple Silicon unified memory)
        let shared = MTLResourceOptions::StorageModeShared;
        let h1_buf = device.newBufferWithLength_options(H_BUF_BYTES, shared);
        let h2_buf = device.newBufferWithLength_options(H_BUF_BYTES, shared);

        Ok(Self {
            device,
            queue,
            pass1_pipeline,
            pass2_pipeline,
            pass3_pipeline,
            weights_buf: None,
            h1_buf,
            h2_buf,
        })
    }

    /// Load flattened weight buffer (all layers concatenated).
    fn load_weights(&mut self, weights_flat: Vec<f32>) -> PyResult<()> {
        let bytes = weights_flat.len() * 4;
        let buf = self.device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("Weight buffer allocation failed"))?;
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(weights_flat.as_ptr(), ptr, weights_flat.len());
        }
        self.weights_buf = Some(buf);
        Ok(())
    }

    /// Run inference: input [N_ITEMS x INPUT_DIM] -> output [N_ITEMS x OUTPUT_DIM]
    fn infer(&self, input_data: Vec<f32>) -> PyResult<Vec<f32>> {
        let weights = self.weights_buf.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Weights not loaded"))?;
        let h1_buf = self.h1_buf.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("h1 buffer not allocated"))?;
        let h2_buf = self.h2_buf.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("h2 buffer not allocated"))?;

        // Create input buffer and copy data
        let input_bytes = input_data.len() * 4;
        let input_buf = self.device
            .newBufferWithLength_options(input_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("Input buffer failed"))?;
        unsafe {
            let ptr = input_buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(input_data.as_ptr(), ptr, input_data.len());
        }

        // Create output buffer
        let output_buf = self.device
            .newBufferWithLength_options(OUTPUT_BYTES, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| PyRuntimeError::new_err("Output buffer failed"))?;

        // Single command buffer for all three passes
        let cmd = self.queue.commandBuffer()
            .ok_or_else(|| PyRuntimeError::new_err("Command buffer failed"))?;

        let tg_size = 64usize;  // threads per threadgroup
        let n_groups = (N_ITEMS + tg_size - 1) / tg_size;

        // ── Pass 1: input -> FC1 -> h1_buf ──
        {
            let enc = cmd.computeCommandEncoder()
                .ok_or_else(|| PyRuntimeError::new_err("Pass 1 encoder failed"))?;
            enc.setComputePipelineState(&self.pass1_pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&input_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
                enc.setBuffer_offset_atIndex(Some(h1_buf), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups, height: 1, depth: 1 },
                MTLSize { width: tg_size, height: 1, depth: 1 },
            );
            enc.endEncoding();
        }

        // ── Pass 2: h1_buf -> FC2 -> h2_buf ──
        {
            let enc = cmd.computeCommandEncoder()
                .ok_or_else(|| PyRuntimeError::new_err("Pass 2 encoder failed"))?;
            enc.setComputePipelineState(&self.pass2_pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(weights), 0, 0);
                enc.setBuffer_offset_atIndex(Some(h1_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(h2_buf), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups, height: 1, depth: 1 },
                MTLSize { width: tg_size, height: 1, depth: 1 },
            );
            enc.endEncoding();
        }

        // ── Pass 3: h2_buf -> FC3 -> output ──
        {
            let enc = cmd.computeCommandEncoder()
                .ok_or_else(|| PyRuntimeError::new_err("Pass 3 encoder failed"))?;
            enc.setComputePipelineState(&self.pass3_pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(weights), 0, 0);
                enc.setBuffer_offset_atIndex(Some(h2_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&output_buf), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups, height: 1, depth: 1 },
                MTLSize { width: tg_size, height: 1, depth: 1 },
            );
            enc.endEncoding();
        }

        // Submit and wait
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read back results
        let out_floats = N_ITEMS * OUTPUT_DIM;
        let mut result = vec![0.0f32; out_floats];
        unsafe {
            let ptr = output_buf.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), out_floats);
        }
        Ok(result)
    }
}
```

**Key implementation details:**

- **`#[pyclass(unsendable)]`**: Metal objects are not `Send`-safe. This annotation tells PyO3 to enforce single-threaded access from Python.
- **`StorageModeShared`**: On Apple Silicon, CPU and GPU share unified memory. `StorageModeShared` avoids copies. On discrete GPUs, use `StorageModeManaged` with explicit synchronization.
- **Pre-allocated buffers**: `h1_buf` and `h2_buf` are created once in `new()` and reused across all inference calls. Do not allocate per-frame.
- **Single command buffer**: All three passes go into one `commandBuffer()`. The GPU guarantees serial execution within a single command buffer.

---

## Weight Extraction from PyTorch

The bridge between a trained PyTorch model and the Metal shader is a flat weight file. Extract weights once, cache them, and load them at runtime without any PyTorch dependency.

```python
"""Extract MLP weights from a PyTorch checkpoint into a flat .npy file."""

import numpy as np
import torch
from pathlib import Path


def extract_weights(model_path: str, output_path: str = None) -> np.ndarray:
    """Extract and flatten MLP weights in the order expected by the shader.

    Args:
        model_path: Path to .pt checkpoint
        output_path: Optional .npy path. Defaults to model_path with .metal_weights.npy suffix.

    Returns:
        Flat float32 numpy array of all weights.
    """
    if output_path is None:
        output_path = str(Path(model_path).with_suffix('.metal_weights.npy'))

    sd = torch.load(model_path, map_location='cpu', weights_only=True)

    # Flatten weights in the SAME ORDER as the shader's offset constants.
    # Each weight matrix is stored row-major (PyTorch default).
    flat = []

    # -- Customize this section for your model's state_dict keys --
    layer_keys = [
        ('fc1.weight', None),    # [out, in] -> flatten row-major
        ('fc1.bias',   None),    # [out]
        ('fc2.weight', None),    # [out, in]
        ('fc2.bias',   None),    # [out]
        ('fc3.weight', None),    # [out, in]
        ('fc3.bias',   None),    # [out]
    ]

    for key, transform in layer_keys:
        tensor = sd[key]
        if transform:
            tensor = transform(tensor)
        flat.extend(tensor.flatten().tolist())

    result = np.array(flat, dtype=np.float32)
    np.save(output_path, result)
    print(f"Extracted {len(flat)} floats ({len(flat)*4:,} bytes) -> {output_path}")
    return result


def load_weights(npy_path: str) -> list:
    """Load cached weights as a flat Python list (for passing to Rust via PyO3)."""
    arr = np.load(npy_path)
    assert arr.dtype == np.float32
    return arr.tolist()
```

The nCPU project's actual extraction code (from `metal_neural_display.py`) demonstrates this pattern with automatic caching:

```python
# Try numpy cache first (no torch dependency)
cache = _cache_path(model_path)
if cache.exists():
    arr = np.load(str(cache))
    if arr.shape == (N_WEIGHT_FLOATS,) and arr.dtype == np.float32:
        return arr.tolist()

# Fall back to torch extraction (creates cache for next time)
sd = torch.load(str(path), map_location='cpu', weights_only=True)
flat = []
flat.extend(sd['glyphs.embed.weight'].flatten().tolist())
flat.extend(sd['glyphs.net.0.weight'].flatten().tolist())
flat.extend(sd['glyphs.net.0.bias'].tolist())
# ... remaining layers ...

np.save(str(cache), np.array(flat, dtype=np.float32))
```

This two-tier approach means the Metal kernel runs completely PyTorch-free after the first weight extraction.

---

## Verification and Debugging

Silent corruption is the primary risk with GPU MLP deployment. Build verification into your pipeline from the start.

### CPU Reference Implementation

Maintain a CPU-side forward pass using the same weight buffer for bit-level comparison. The nCPU project includes this as the `debug_glyph` method:

```rust
/// CPU-side single-item forward pass for debugging.
/// Uses the same GPU weight buffer to ensure weight packing correctness.
fn debug_forward(&self, input_idx: usize) -> PyResult<Vec<f32>> {
    let w = self.weights_buf.as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("weights not loaded"))?;
    let w = w.contents().as_ptr() as *const f32;

    unsafe {
        // FC1
        let mut h1 = [0.0f32; 256];
        for i in 0..256 {
            let mut s = *w.add(FC1_B + i);
            for j in 0..64 {
                s += *w.add(FC1_W + i * 64 + j) * *w.add(EMBED_W + input_idx * 64 + j);
            }
            h1[i] = gelu(s);
        }

        // FC2
        let mut h2 = [0.0f32; 256];
        for i in 0..256 {
            let mut s = *w.add(FC2_B + i);
            for j in 0..256 {
                s += *w.add(FC2_W + i * 256 + j) * h1[j];
            }
            h2[i] = gelu(s);
        }

        // FC3
        let mut output = vec![0.0f32; 128];
        for i in 0..128 {
            let mut s = *w.add(FC3_B + i);
            for j in 0..256 {
                s += *w.add(FC3_W + i * 256 + j) * h2[j];
            }
            output[i] = sigmoid(s);
        }

        Ok(output)
    }
}
```

### Intermediate Buffer Inspection

Expose methods to read back `h1_buf` and `h2_buf` after a GPU pass. This lets you compare intermediate activations between CPU and GPU to isolate which pass introduces errors:

```rust
/// Read back h1_buf values after a GPU pass for diagnostics.
fn read_h1(&self, item_idx: usize, count: usize) -> PyResult<Vec<f32>> {
    let buf = self.h1_buf.as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("h1 buffer not allocated"))?;
    let start = item_idx * 256;
    let n = count.min(256);
    let ptr = buf.contents().as_ptr() as *const f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(unsafe { *ptr.add(start + i) });
    }
    Ok(out)
}
```

### Verification Protocol

1. **Weight packing test.** After `load_weights()`, read back specific indices and compare against the original PyTorch state dict. Off-by-one errors in weight offsets are the most common bug.

2. **Per-pass comparison.** Run one pass on GPU, read back the intermediate buffer, compare against the CPU reference. Do this for each pass independently.

3. **Full pipeline PSNR.** Run the full three-pass pipeline on GPU and compute PSNR against PyTorch CPU output. The nCPU display achieves 68.7 dB PSNR (99.13% exact pixel match). Anything below 40 dB indicates a systematic error.

4. **NaN sweep.** Run inference on all possible inputs (or a large random sample) and check for NaN values in the output. If any appear, your activation function NaN guards are missing or insufficient.

5. **Boundary conditions.** Test with all-zero inputs, all-max inputs, and adversarial patterns that maximize dot product magnitudes. These are the inputs most likely to trigger overflow in activation functions.

---

## References

- **nCPU Paper, Section 14.8:** Metal GPU Acceleration of the Neural Display --- describes the original discovery and measurement of this technique.
- **nCPU Source, `kernels/rust_metal/src/neural_display.rs`:** Production implementation of the three-pass neural display shader.
- **nCPU Source, `ncpu/neural/metal_neural_display.py`:** Python wrapper with weight extraction and caching.
- **Apple Metal Best Practices Guide:** Thread execution and memory model documentation.
- **Metal Shading Language Specification:** Compute kernel authoring reference.
