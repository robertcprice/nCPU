# Multi-Pass MLP Inference on Thread-Limited GPUs: Eliminating Silent Stack Overflow on Apple Silicon

**Robert Price**

April 2026

---

## Abstract

Deploying multi-layer perceptrons on mobile and embedded GPUs presents a subtle correctness hazard: per-thread stack memory on Apple Silicon, ARM Mali, and Qualcomm Adreno GPUs is limited to approximately 1 KB, and exceeding this limit produces no error --- only silently corrupted output. We present a multi-pass decomposition technique that replaces thread-local activation arrays with device-memory intermediate buffers, enabling framework-free MLP inference with zero bytes of per-thread array storage. Applied to a 143K-parameter neural glyph renderer on Apple M-series hardware, the technique achieves 361 frames per second with 68.7 dB PSNR fidelity against a PyTorch float32 reference (99.13% exact pixel match, maximum per-pixel error of 1). We additionally describe a saturating GELU activation guard that eliminates a NaN propagation chain caused by IEEE 754 indeterminate forms arising from large dot products. The technique generalizes to any MLP whose hidden dimensions exceed the target GPU's per-thread stack budget, and requires no inference framework at runtime --- only flat weight buffers and native GPU compute.

---

## 1. Introduction

The trend toward on-device neural inference is accelerating across mobile applications, edge computing, and embedded systems. Apple Silicon's Metal compute shader architecture offers a compelling inference target: unified CPU-GPU memory eliminates data transfer overhead, and the GPU's massively parallel execution model maps naturally to batched inference workloads. However, a largely undocumented hardware constraint makes naive MLP deployment on these GPUs unreliable.

Apple Silicon limits per-thread stack memory to approximately 1 KB. This is a hard constraint imposed by the GPU thread scheduler: each thread in a compute kernel has access to a small, fixed-size region of fast memory for local variables. A standard 3-layer MLP with a hidden dimension of 256 requires 256 x 4 = 1,024 bytes for a single hidden layer's activation array. Two hidden layers require 2,048 bytes --- double the budget. Exceeding this limit does not produce a crash, an error code, or a diagnostic message. The hardware silently aliases stack writes, corrupting register spills and producing garbage output.

We encountered this problem while developing a neural terminal renderer that uses a trained MLP to convert character codes to glyph pixels on the GPU. The single-pass implementation corrupted approximately 40% of rendered characters, with the corruption pattern varying non-deterministically across frames. The absence of any error signal made the problem exceptionally difficult to diagnose.

This paper makes three contributions:

1. **Characterization of the silent stack overflow problem** on mobile GPUs, including the NaN propagation chain that converts stack corruption into visually detectable but logically opaque rendering errors (Section 3).

2. **A multi-pass decomposition technique** that eliminates all thread-local array storage by routing intermediate activations through device-memory buffers, with formal analysis of the memory trade-offs (Section 4).

3. **A saturating activation guard** for GELU and related functions that prevents IEEE 754 indeterminate forms from introducing NaN values, even when the multi-pass fix is applied (Section 5).

We provide complete, annotated Metal shader and Rust host code, empirical performance measurements on Apple M-series hardware, and guidance for applying the technique to arbitrary MLPs on any thread-limited GPU.

---

## 2. Background and Related Work

### 2.1 Metal Compute Shader Architecture

Apple's Metal [1] is the low-level GPU programming framework for Apple Silicon. Metal compute shaders execute as kernel functions dispatched over a grid of threads, organized into threadgroups. Each threadgroup executes on a single GPU compute unit, and threads within a threadgroup share a small amount of threadgroup memory (up to 32 KB on Apple Silicon). Individual threads have access to registers and a per-thread stack for local variables.

The thread execution model on Apple Silicon organizes threads into SIMD groups of 32 threads that execute in lockstep. The GPU scheduler maps threadgroups to compute units and manages the concurrent execution of multiple threadgroups per compute unit, subject to register pressure and memory constraints.

### 2.2 Per-Thread Memory Hierarchy

Metal compute threads access four tiers of memory, in order of increasing latency:

| Tier | Scope | Typical Size | Access Pattern |
|------|-------|-------------|----------------|
| Registers | Per-thread | ~256 bytes | Compiler-managed |
| Thread-local stack | Per-thread | ~1 KB | Automatic variables, array spills |
| Threadgroup memory | Per-threadgroup | Up to 32 KB | Explicit `threadgroup` qualifier |
| Device memory | Global | 8--192 GB | `device` qualifier, `MTLBuffer` |

When a kernel function declares thread-local arrays that exceed the register file, the Metal compiler spills them to the per-thread stack. When the stack itself is exhausted, writes alias into adjacent memory --- potentially other threads' stacks or control data. This aliasing produces no hardware exception.

### 2.3 Existing Inference Frameworks

Several frameworks target Apple Silicon for neural inference:

**CoreML** [2] is Apple's high-level inference framework. It accepts models in `.mlmodel` or `.mlpackage` format, performs automatic graph optimization, and dispatches to the Neural Engine, GPU, or CPU. CoreML manages memory allocation and kernel scheduling internally, abstracting away per-thread stack constraints. However, it requires the CoreML runtime, limits model architecture to its supported operation set, and provides no visibility into kernel-level execution.

**ONNX Runtime** [3] supports a CoreML execution provider for Apple Silicon. Like CoreML, it abstracts kernel-level details and manages memory through its own allocator. The abstraction prevents the stack overflow problem but introduces framework overhead and version coupling.

**MPSGraph** [4] provides a lower-level graph-based API backed by Metal Performance Shaders. It offers more control than CoreML but still manages memory allocation and kernel decomposition internally. MPSGraph operations use Apple-optimized kernels that account for per-thread memory limits.

None of these frameworks are suitable when the goal is framework-free deployment: weights as flat GPU buffers, inference as native compute shaders, no Python runtime, no dynamic memory allocation, and deterministic bit-reproducible results. The multi-pass technique presented here fills this gap.

### 2.4 Custom Metal Inference

Prior work on custom Metal compute for neural inference is sparse in the academic literature. Most Metal inference implementations target convolutional neural networks via Metal Performance Shaders [4] or use the Neural Engine through CoreML. For small, latency-sensitive models where framework overhead is significant (sub-millisecond inference budgets), direct Metal compute is attractive but requires manual attention to hardware constraints that frameworks handle automatically.

---

## 3. The Silent Stack Overflow Problem

### 3.1 Thread-Local Array Declaration

In Metal Shading Language (MSL), local variables declared inside a kernel function are allocated on the per-thread stack. Consider a naive single-pass MLP kernel:

```metal
// BROKEN on Apple Silicon: h1[256] alone consumes the entire ~1 KB stack
kernel void naive_mlp(
    device const float* weights [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float h1[256];  // 1,024 bytes -- at or over the stack limit
    float h2[256];  // another 1,024 bytes -- definitely over

    // Compute FC1 into h1 ...
    for (int i = 0; i < 256; i++) {
        h1[i] = gelu(dot_product(weights, input, i));
    }
    // Compute FC2 into h2 using h1 ...
    for (int i = 0; i < 256; i++) {
        h2[i] = gelu(dot_product(weights, h1, i));
    }
    // Compute FC3 output using h2 ...
}
```

The arrays `h1[256]` and `h2[256]` each require 1,024 bytes. Together they exceed the per-thread stack budget by a factor of two.

### 3.2 Failure Mode: Stack Aliasing

When thread-local storage exceeds the hardware stack limit, the Metal compiler and GPU hardware do not raise an error. Instead, writes to addresses beyond the stack boundary alias into adjacent memory regions. The specific aliasing behavior depends on the GPU's stack allocation strategy and varies with:

- **Thread scheduling order**: different SIMD group assignments produce different aliasing patterns.
- **Register pressure**: the compiler's register allocation decisions affect which variables spill to the stack and in what order.
- **Threadgroup size**: larger threadgroups may increase stack aliasing severity.

The result is that a subset of threads produce corrupted output while others appear correct. In our neural display application, this manifested as approximately 40% of character glyphs containing NaN or garbage pixel values, with the specific corrupted characters changing between runs.

### 3.3 The NaN Propagation Chain

Stack aliasing corrupts intermediate activation values, which then propagate through nonlinear activation functions. The GELU activation [5] is particularly susceptible due to its polynomial approximation:

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

When a corrupted intermediate value causes a dot product to overflow to negative infinity, the following chain occurs:

1. The cubic term `x^3` overflows to `-inf`.
2. `tanh(-inf) = -1`.
3. `(1 + (-1)) = 0`.
4. `0.5 * (-inf) * 0 = NaN` (IEEE 754 [6]: infinity multiplied by zero is an indeterminate form).

This NaN propagates through all subsequent layers, replacing every downstream computed value with NaN. The final output contains a mix of valid pixels (from threads with correct stack behavior) and NaN pixels (from threads with corrupted stacks), producing a visually distinctive but logically opaque corruption pattern.

### 3.4 Why This Is Hard to Debug

The silent stack overflow is difficult to diagnose for several reasons:

- **No error signal**: Metal's error reporting (`MTLCommandBuffer.error`, `MTLCommandBuffer.status`) returns success. GPU validation layers do not flag the kernel.
- **Partial corruption**: Only a subset of threads are affected. The output looks "mostly right" --- close enough to suggest a numerical bug rather than a memory corruption issue.
- **Non-determinism**: The corruption pattern varies across runs and across different input data, because thread scheduling and register allocation are non-deterministic.
- **Framework absence**: When writing custom Metal shaders (rather than using CoreML or MPSGraph), there is no framework-level validation of per-thread memory usage.

We identified the root cause only by systematically comparing intermediate activation values between a CPU reference implementation and GPU output, pass by pass, until we observed that FC2's hidden state was corrupted on the GPU despite correct FC1 output.

---

## 4. Multi-Pass Decomposition

### 4.1 Core Idea

The key insight is to replace thread-local arrays with device-memory `MTLBuffer` objects. Instead of computing the entire MLP forward pass in a single kernel (requiring thread-local storage for intermediate activations), we decompose the forward pass into N sequential compute dispatches, one per layer that would otherwise exceed the stack budget. Each dispatch reads its input from a device-memory buffer, computes one layer's transformation using only scalar accumulators, and writes each output element directly to a device-memory buffer.

The critical property is that each thread holds at most one scalar accumulator (`float s`) at any point during execution. It computes one output element of the layer's activation, writes it to device memory, and moves to the next element. Zero bytes of thread-local array storage are required.

### 4.2 Decomposition Strategy

For a 3-layer MLP with architecture `Input -> FC1 -> FC2 -> FC3`, the decomposition produces three compute passes:

| Pass | Input Source | Computation | Output Destination | Thread-Local Storage |
|------|-------------|-------------|-------------------|---------------------|
| Pass 1 | Input data (device) | FC1 + activation | `h1_buf` (device) | 0 arrays, 1 scalar |
| Pass 2 | `h1_buf` (device) | FC2 + activation | `h2_buf` (device) | 0 arrays, 1 scalar |
| Pass 3 | `h2_buf` (device) | FC3 + activation | Output (device) | 0 arrays, 1 scalar |

Each pass is implemented as a separate Metal kernel function. The three passes are encoded into a single `MTLCommandBuffer` using three sequential `MTLComputeCommandEncoder` instances. Metal guarantees that compute command encoders within a single command buffer execute serially [1], so no explicit synchronization primitives (fences, events, memory barriers) are needed between passes.

### 4.3 Buffer Sizing

Each intermediate buffer stores the full batch of activations for one layer:

```
buffer_bytes = N_items * hidden_dim * sizeof(float)
```

where `N_items` is the number of independent inputs processed in parallel (the batch size) and `hidden_dim` is the layer's output dimension. One intermediate buffer is required per pass boundary.

For our neural display application with `N_items = 1920` cells and `hidden_dim = 256`:

```
h1_buf = h2_buf = 1920 * 256 * 4 = 1,966,080 bytes (~1.9 MB each)
```

### 4.4 Memory Trade-Off Analysis

The multi-pass technique trades thread-local memory for device memory:

| Resource | Single-Pass | Multi-Pass | Change |
|----------|------------|------------|--------|
| Per-thread stack | 2,048 bytes (two hidden layers) | 0 bytes | Eliminated |
| Device memory (intermediate) | 0 bytes | ~3.8 MB (two buffers) | +3.8 MB |
| Total device memory | ~1.3 MB (weights + I/O) | ~5.1 MB (weights + I/O + intermediates) | +3.8 MB |

The 3.8 MB device memory cost is trivial compared to GPU VRAM capacity (8--192 GB on Apple Silicon, 1--4 GB on mobile GPUs). The intermediate buffers are allocated once at initialization and reused across all inference calls. No per-frame allocation occurs.

### 4.5 Implementation: Metal Shader

The following annotated Metal shader implements the three-pass decomposition for a glyph-rendering MLP. Each pass uses zero thread-local arrays.

**Pass 1: Embedding + FC1 (64 -> 256) + GELU**

```metal
kernel void neural_pass1_h1(
    device const uint8_t* char_codes [[buffer(0)]],  // [1920] input characters
    device const float*   weights    [[buffer(1)]],  // all weights, contiguous
    device       float*   h1_buf     [[buffer(2)]],  // [1920 * 256] output
    uint2 tid [[thread_position_in_grid]]
) {
    int col = (int)tid.x;
    int row = (int)tid.y;
    if (col >= 80 || row >= 24) return;         // bounds check: 80 cols x 24 rows

    int cell_idx = row * 80 + col;              // linear cell index
    int ch = (int)char_codes[cell_idx];          // character code for embedding lookup
    int h1_base = cell_idx * 256;                // output offset in h1_buf

    // FC1: embed[ch] (64-dim) -> Linear(64, 256) + GELU
    // Each output element computed and written immediately -- no local array.
    for (int i = 0; i < 256; i++) {
        float s = weights[FC1_B + i];            // bias: 1 scalar
        for (int j = 0; j < 64; j++)
            s += weights[FC1_W + i*64 + j] * weights[EMBED_W + ch*64 + j];
        h1_buf[h1_base + i] = neural_gelu(s);   // write to device memory
    }
}
```

**Pass 2: FC2 (256 -> 256) + GELU**

```metal
kernel void neural_pass2_h2(
    device const float* weights [[buffer(0)]],  // all weights
    device const float* h1_buf  [[buffer(1)]],  // [1920 * 256] input from Pass 1
    device       float* h2_buf  [[buffer(2)]],  // [1920 * 256] output
    uint2 tid [[thread_position_in_grid]]
) {
    int col = (int)tid.x;
    int row = (int)tid.y;
    if (col >= 80 || row >= 24) return;

    int cell_idx = row * 80 + col;
    int base = cell_idx * 256;

    // FC2: h1 (256-dim) -> Linear(256, 256) + GELU
    for (int i = 0; i < 256; i++) {
        float s = weights[FC2_B + i];
        for (int j = 0; j < 256; j++)
            s += weights[FC2_W + i*256 + j] * h1_buf[base + j];
        h2_buf[base + i] = neural_gelu(s);
    }
}
```

**Pass 3: FC3 (256 -> 128) + Sigmoid + Alpha Blend -> Pixels**

```metal
kernel void neural_pass3_pixels(
    device const uint8_t* fg_codes  [[buffer(0)]],  // foreground color indices
    device const uint8_t* bg_codes  [[buffer(1)]],  // background color indices
    device const float*   weights   [[buffer(2)]],  // all weights + palette
    device const float*   h2_buf    [[buffer(3)]],  // [1920 * 256] input from Pass 2
    device       uint8_t* framebuf  [[buffer(4)]],  // [384 * 640 * 3] RGB output
    uint2 tid [[thread_position_in_grid]]
) {
    int col = (int)tid.x;
    int row = (int)tid.y;
    if (col >= 80 || row >= 24) return;

    int cell_idx = row * 80 + col;
    int h2_base = cell_idx * 256;

    // Palette lookup for foreground and background colors
    int fg_code = (int)fg_codes[cell_idx];
    int bg_code = (int)bg_codes[cell_idx];
    float fg_r = weights[PALETTE + fg_code*3 + 0];
    float fg_g = weights[PALETTE + fg_code*3 + 1];
    float fg_b = weights[PALETTE + fg_code*3 + 2];
    float bg_r = weights[PALETTE + bg_code*3 + 0];
    float bg_g = weights[PALETTE + bg_code*3 + 1];
    float bg_b = weights[PALETTE + bg_code*3 + 2];

    // FC3: h2 (256-dim) -> Linear(256, 128) + Sigmoid -> alpha per pixel
    // Each of 128 outputs maps to one pixel in the 8x16 glyph cell.
    for (int pi = 0; pi < 128; pi++) {
        float alpha_val = weights[FC3_B + pi];     // bias: 1 scalar
        for (int j = 0; j < 256; j++)
            alpha_val += weights[FC3_W + pi*256 + j] * h2_buf[h2_base + j];
        float a = neural_sigmoid(alpha_val);

        // Map linear index to 2D pixel position within the 8x16 cell
        int py = pi / 8;
        int px = pi % 8;
        float r = a * fg_r + (1.0f - a) * bg_r;
        float g = a * fg_g + (1.0f - a) * bg_g;
        float b = a * fg_b + (1.0f - a) * bg_b;

        int frame_y = row * 16 + py;
        int frame_x = col * 8 + px;
        int pixel_idx = (frame_y * 640 + frame_x) * 3;
        framebuf[pixel_idx + 0] = (uint8_t)clamp(r * 255.0f + 0.5f, 0.0f, 255.0f);
        framebuf[pixel_idx + 1] = (uint8_t)clamp(g * 255.0f + 0.5f, 0.0f, 255.0f);
        framebuf[pixel_idx + 2] = (uint8_t)clamp(b * 255.0f + 0.5f, 0.0f, 255.0f);
    }
}
```

### 4.6 Command Buffer Structure

The three passes are encoded into a single `MTLCommandBuffer`. Each pass uses a separate `MTLComputeCommandEncoder`:

```rust
// Single command buffer for all three passes
let cmd = queue.commandBuffer()?;

// Pass 1: input -> FC1 -> h1_buf
{
    let enc = cmd.computeCommandEncoder()?;
    enc.setComputePipelineState(&pass1_pipeline);
    enc.setBuffer_offset_atIndex(Some(&char_buf), 0, 0);
    enc.setBuffer_offset_atIndex(Some(&weights_buf), 0, 1);
    enc.setBuffer_offset_atIndex(Some(&h1_buf), 0, 2);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
    enc.endEncoding();
}

// Pass 2: h1_buf -> FC2 -> h2_buf
{
    let enc = cmd.computeCommandEncoder()?;
    enc.setComputePipelineState(&pass2_pipeline);
    enc.setBuffer_offset_atIndex(Some(&weights_buf), 0, 0);
    enc.setBuffer_offset_atIndex(Some(&h1_buf), 0, 1);
    enc.setBuffer_offset_atIndex(Some(&h2_buf), 0, 2);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
    enc.endEncoding();
}

// Pass 3: h2_buf -> FC3 -> pixels
{
    let enc = cmd.computeCommandEncoder()?;
    enc.setComputePipelineState(&pass3_pipeline);
    enc.setBuffer_offset_atIndex(Some(&fg_buf), 0, 0);
    enc.setBuffer_offset_atIndex(Some(&bg_buf), 0, 1);
    enc.setBuffer_offset_atIndex(Some(&weights_buf), 0, 2);
    enc.setBuffer_offset_atIndex(Some(&h2_buf), 0, 3);
    enc.setBuffer_offset_atIndex(Some(&frame_buf), 0, 4);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
    enc.endEncoding();
}

// Submit all three passes as a single GPU workload
cmd.commit();
cmd.waitUntilCompleted();
```

Metal guarantees serial execution between `endEncoding()` and the next `computeCommandEncoder()` call within the same command buffer. This provides implicit ordering: Pass 2 reads `h1_buf` only after Pass 1 has finished writing it. No explicit barriers or fences are required.

---

## 5. The GELU Activation Guard

### 5.1 IEEE 754 Analysis

Even with the multi-pass decomposition eliminating stack corruption, large dot products can still produce extreme activation values. The FC2 layer computes dot products over 256 terms. For certain input patterns --- particularly rare character embeddings with high-magnitude entries --- the accumulated sum can reach values where the GELU polynomial approximation encounters IEEE 754 indeterminate forms.

The standard GELU formula [5]:

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

When `x` approaches negative infinity:

1. `x^3` overflows to `-inf` (IEEE 754 overflow).
2. `tanh(-inf) = -1` (well-defined).
3. `(1 + (-1)) = 0` (well-defined).
4. `0.5 * (-inf) * 0 = NaN` (IEEE 754 Section 7.1: infinity times zero is undefined [6]).

This NaN propagates through all subsequent layers and post-processing, silently corrupting the final output.

### 5.2 The Saturating Guard

We apply a saturation clamp based on the asymptotic behavior of GELU:

```metal
inline float neural_gelu(float x) {
    // For |x| > 10, GELU saturates:
    //   GELU(-10) = 2.27e-23  (effectively 0 in float32)
    //   GELU(+10) = 10 - 2.27e-23  (effectively 10 in float32)
    if (x < -10.0f) return 0.0f;
    if (x >  10.0f) return x;
    float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}
```

The thresholds +/-10 are chosen because `GELU(-10) ~ 2.27 x 10^-23`, which is zero to well beyond float32 precision (~7 decimal digits), and `GELU(10) ~ 10 - 2.27 x 10^-23`, which is indistinguishable from 10 in float32. The guard introduces zero error for any realistic network activation while completely eliminating the NaN pathway.

### 5.3 Sigmoid Guard

The same principle applies to sigmoid, where `exp(-inf)` and `exp(inf)` can produce NaN through overflow:

```metal
inline float neural_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -15.0f, 15.0f)));
}
```

Clamping the input to `[-15, 15]` ensures `exp()` operates within its well-behaved range while introducing zero practical error (`sigmoid(-15) ~ 3.06 x 10^-7`, `sigmoid(15) ~ 1 - 3.06 x 10^-7`).

### 5.4 Generalization

This guard pattern applies to any activation function that involves products of potentially infinite terms. For common activations:

| Activation | Risk | Guard |
|-----------|------|-------|
| GELU | `inf * 0 = NaN` via cubic overflow | Saturate at +/-10 |
| Sigmoid | `exp(inf)` overflow | Clamp input to +/-15 |
| Tanh | `exp(inf)` in internal sigmoid | Clamp input to +/-10 |
| Softmax | `exp(inf)` in numerator | Subtract max before exp |
| SiLU/Swish | `x * sigmoid(x)`, same as sigmoid | Clamp sigmoid input |

ReLU and its variants (Leaky ReLU, PReLU) are inherently safe because they involve no products of potentially infinite terms.

---

## 6. Implementation

### 6.1 Weight Buffer Layout

All model weights are packed into a single contiguous `MTLBuffer` in a documented order. This eliminates per-layer buffer management and minimizes the number of buffer bindings per compute encoder.

For the neural display model (131,760 floats = 527,040 bytes):

```
Offset (floats)   Size (floats)   Layer                Shape
------------------------------------------------------------------------
0                  16,384          embed.weight         [256, 64]   row-major
16,384             16,384          net.0.weight (FC1)   [256, 64]   row-major
32,768                256          net.0.bias   (FC1)   [256]
33,024             65,536          net.2.weight (FC2)   [256, 256]  row-major
98,560                256          net.2.bias   (FC2)   [256]
98,816             32,768          net.4.weight (FC3)   [128, 256]  row-major
131,584               128          net.4.bias   (FC3)   [128]
131,712                48          palette.weight       [16, 3]
------------------------------------------------------------------------
Total:            131,760 floats = 527,040 bytes
```

Weights are stored row-major, matching PyTorch's default layout. For a linear layer with shape `[out_features, in_features]`, element `W[i][j]` is at offset `base + i * in_features + j`. This means the inner loop of each dot product iterates over contiguous memory, which is cache-friendly on the GPU.

### 6.2 Rust/PyO3 Host Code

The host-side implementation uses Rust with the `objc2-metal` crate for Metal bindings and `pyo3` for optional Python interoperability. Key design decisions:

**Pre-allocated buffers.** The intermediate buffers `h1_buf` and `h2_buf` are allocated once during initialization using `MTLResourceOptions::StorageModeShared` (zero-copy on Apple Silicon unified memory) and reused across all inference calls. No per-frame allocation occurs.

**Single command buffer.** All passes are encoded into one `MTLCommandBuffer`, which amortizes command buffer creation overhead and provides implicit serialization.

**Threadgroup sizing.** We use threadgroup width of 20 (yielding 4 threadgroups per row, 96 total) rather than 80 (one threadgroup per row, 24 total). Smaller threadgroups give the GPU scheduler more flexibility to fill compute units:

```rust
let tg_w = 20usize;
let groups_x = (80 + tg_w - 1) / tg_w;  // = 4
enc.dispatchThreadgroups_threadsPerThreadgroup(
    MTLSize { width: groups_x, height: 24, depth: 1 },
    MTLSize { width: tg_w, height: 1, depth: 1 },
);
```

### 6.3 PyTorch-Free Deployment

The weight extraction pipeline operates in two tiers:

1. **First run (requires PyTorch):** Load the `.pt` checkpoint, flatten all parameter tensors in the documented order, and save as a `.npy` file (515 KB for the display model).

2. **Subsequent runs (PyTorch-free):** Load the `.npy` cache using only NumPy, copy to the GPU weight buffer, and dispatch. No `import torch` is required at inference time.

```python
# Weight extraction (one-time, requires torch)
sd = torch.load(model_path, map_location='cpu', weights_only=True)
flat = []
flat.extend(sd['glyphs.embed.weight'].flatten().tolist())
flat.extend(sd['glyphs.net.0.weight'].flatten().tolist())
flat.extend(sd['glyphs.net.0.bias'].tolist())
# ... remaining layers in documented order ...
np.save(cache_path, np.array(flat, dtype=np.float32))

# Inference (PyTorch-free)
weights = np.load(cache_path).tolist()
renderer.load_weights(weights)
frame = renderer.render(char_codes, fg_codes, bg_codes)
```

This two-tier approach decouples the training environment (Python, PyTorch, CUDA) from the deployment environment (native Rust/Metal, no Python runtime required at the core).

---

## 7. Evaluation

### 7.1 Experimental Setup

We evaluate the multi-pass technique using the nCPU neural terminal display, a 143K-parameter MLP that converts character codes to 8x16 pixel glyphs with foreground/background color blending. The evaluation hardware is Apple M-series (unified memory architecture). The display renders a 24-row by 80-column terminal (1,920 cells) into a 384x640 RGB framebuffer at each frame.

The reference implementation is a PyTorch float32 CPU forward pass using the identical trained weights. We compare the Metal multi-pass shader output against this reference across all 95 printable ASCII characters (codes 32--126) rendered in all 16 ANSI color combinations.

### 7.2 Fidelity

| Metric | Value |
|--------|-------|
| PSNR | 68.7 dB |
| Exact pixel match | 99.13% |
| Maximum per-pixel error | 1 (out of 255) |
| Mean absolute error | 0.0087 per pixel |

A PSNR of 68.7 dB is exceptionally high --- well above the 40 dB threshold typically considered visually lossless [7]. The residual 1-pixel differences arise from floating-point rounding between Metal's compute pipeline and PyTorch's CPU float32 path, not from the multi-pass decomposition itself. We verified this by comparing intermediate activation values (`h1_buf`, `h2_buf`) between the Metal shader and a CPU reference implementation that reads the same GPU weight buffer; the activations match to within float32 epsilon.

### 7.3 Performance

| Configuration | Throughput | Notes |
|--------------|-----------|-------|
| Metal 3-pass (glyph only) | 361 FPS | 1,920 threads x 3 dispatches |
| Metal 6-pass (glyph + compositor) | 15 FPS | Additional 3 conv passes |
| PyTorch CPU | ~82 FPS | Same model, float32 |
| PyTorch MPS | ~180 FPS | Metal Performance Shaders backend |

The 3-pass Metal kernel is 4.4x faster than PyTorch CPU inference and 2.0x faster than PyTorch's MPS backend for the same model. The overhead of encoding three compute passes versus one is negligible: creating a `MTLComputeCommandEncoder`, setting a pipeline state, binding 3--5 buffers, and calling `dispatchThreadgroups` is on the order of microseconds, while the actual GPU compute (matrix-vector products over 256 dimensions for 1,920 items) dominates by three orders of magnitude.

The 6-pass compositor path is slower (15 FPS) due to the Conv2 layer's 2.26 billion multiply-adds per frame (32 x 32 x 9 per pixel x 245,760 pixels). Since the compositor contributes less than 1 pixel difference on typical scenes, the 3-pass path is the production default.

### 7.4 Memory

| Component | Size |
|-----------|------|
| Weight buffer (glyph MLP) | 527 KB (131,760 floats) |
| Weight buffer (full, with compositor) | 574 KB (143,539 floats) |
| h1_buf intermediate | 1.9 MB (491,520 floats) |
| h2_buf intermediate | 1.9 MB (491,520 floats) |
| Weight cache (.npy) | 515 KB |
| Framebuffer output | 720 KB (737,280 bytes) |
| **Total GPU allocation** | **~5.1 MB** |

The total GPU memory footprint of 5.1 MB is well within the capacity of even the most constrained mobile GPUs (1 GB minimum on modern devices).

---

## 8. Generalization and Limitations

### 8.1 Applicability

The multi-pass technique applies to any MLP where `hidden_dim * sizeof(element) > stack_limit`:

| GPU Family | Approximate Stack Limit | Threshold (float32) | Threshold (float16) |
|-----------|------------------------|--------------------|--------------------|
| Apple Silicon (M1--M4) | ~1 KB | hidden_dim >= 256 | hidden_dim >= 512 |
| ARM Mali (mobile Android) | 512 bytes -- 1 KB | hidden_dim >= 128 | hidden_dim >= 256 |
| Qualcomm Adreno (mobile Android) | ~1 KB | hidden_dim >= 256 | hidden_dim >= 512 |
| Embedded GPU accelerators | Often < 512 bytes | hidden_dim >= 64 | hidden_dim >= 128 |

Common deployment scenarios that benefit from this technique include:

- **NLP classification:** Character or token embeddings followed by dense layers for sentiment analysis, named entity recognition, or intent classification.
- **Recommendation models:** User/item embedding dot products with MLP towers.
- **Sensor fusion on edge devices:** Combining multiple sensor inputs through dense layers on mobile GPUs.
- **Audio feature extraction:** Frame-level MLP processing of spectral features.

### 8.2 Desktop GPUs

Desktop GPUs from NVIDIA and AMD typically provide 1 KB -- 512 KB of configurable per-thread local memory through CUDA's `__local__` or OpenCL's private memory. On these platforms, the multi-pass technique is unnecessary for typical MLP hidden dimensions. A single-pass kernel is simpler and equally fast.

### 8.3 Limitations

**Inference only.** The multi-pass technique stores intermediate activations in device memory for forward-pass use only. It does not retain the computation graph for automatic differentiation. Backpropagation through the passes would require a separate backward-pass kernel sequence and careful buffer management, which is outside the scope of this work.

**Compute overhead for very deep networks.** Each layer that exceeds the stack budget adds one compute dispatch. For networks with 50+ layers, the cumulative dispatch overhead may become measurable. In practice, we observe that dispatch overhead is negligible for networks up to approximately 10 layers. For deeper architectures, multiple consecutive small-dimension layers can be merged into a single pass when their combined thread-local storage fits within the stack budget.

**No shared-memory optimization.** The current implementation does not use threadgroup shared memory for weight tiling or activation reuse. For large hidden dimensions where each thread recomputes the same weight accesses, threadgroup-level tiling could improve cache utilization. We leave this optimization to future work, noting that the current approach already achieves 361 FPS --- well above interactive rates.

### 8.4 Quantization

The intermediate buffers are the primary memory cost. Quantized storage can reduce them:

| Format | Buffer Size (N=1920, dim=256) | Precision Impact |
|--------|------------------------------|-----------------|
| float32 | 1.9 MB | None |
| float16 | 0.95 MB | Minimal for inference |
| int8 | 0.47 MB | Requires calibration |

Weight buffers can also be quantized. For float16 weights, replace `device const float*` with `device const half*` in the shader and cast to float for computation. Metal's native `half` support on Apple Silicon makes this a straightforward optimization.

---

## 9. Conclusion

We have presented a multi-pass decomposition technique for deploying MLPs on GPUs with per-thread stack limits of approximately 1 KB --- a constraint affecting Apple Silicon, ARM Mali, Qualcomm Adreno, and embedded GPU accelerators. The technique eliminates all thread-local array storage by routing intermediate activations through device-memory buffers, replacing a single large kernel with N sequential compute dispatches encoded into one command buffer.

Applied to a 143K-parameter neural glyph renderer, the technique achieves 361 FPS on Apple M-series hardware with 68.7 dB PSNR fidelity against a PyTorch float32 reference. The saturating GELU activation guard eliminates a NaN propagation chain caused by IEEE 754 indeterminate forms in large dot products. Together, these two fixes --- multi-pass decomposition and activation guards --- convert a silently broken single-pass kernel into a production-quality inference pipeline.

The technique is applicable to any MLP deployment on thread-limited GPUs and requires no inference framework at runtime. Weights are extracted once from a PyTorch checkpoint and loaded as flat GPU buffers. The entire inference path --- from weight loading to pixel output --- runs as native GPU compute with zero Python overhead.

The implementation is open source as part of the nCPU project, available at the project repository.

---

## References

[1] Apple Inc. "Metal." Apple Developer Documentation. https://developer.apple.com/metal/

[2] Apple Inc. "Core ML." Apple Developer Documentation. https://developer.apple.com/documentation/coreml

[3] ONNX Runtime Contributors. "ONNX Runtime." https://onnxruntime.ai/

[4] Apple Inc. "Metal Performance Shaders." Apple Developer Documentation. https://developer.apple.com/documentation/metalperformanceshaders

[5] D. Hendrycks and K. Gimpel, "Gaussian Error Linear Units (GELUs)," arXiv:1606.08415, 2016.

[6] IEEE Computer Society. "IEEE Standard for Floating-Point Arithmetic," IEEE Std 754-2019.

[7] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600--612, 2004.

[8] A. Graves, G. Wayne, and I. Danihelka, "Neural Turing Machines," arXiv:1410.5401, 2014.

[9] Apple Inc. "Metal Shading Language Specification," Version 3.1. https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

[10] Apple Inc. "Metal Best Practices Guide." https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/

[11] J. Nickolls, I. Buck, M. Garland, and K. Skadron, "Scalable Parallel Programming with CUDA," ACM Queue, vol. 6, no. 2, pp. 40--53, 2008.

[12] Khronos Group. "OpenCL Specification," Version 3.0. https://www.khronos.org/opencl/

[13] R. Price, "nCPU: A Neural Central Processing Unit," Technical Report, 2026.

---

## Appendix A: Adaptation Checklist

For engineers applying this technique to their own models:

1. **Measure your hidden dimensions.** For each hidden layer, compute `hidden_dim * sizeof(float)`. Any layer where this exceeds your target GPU's per-thread stack limit (~1 KB on Apple Silicon) needs its own pass.

2. **Pack weights into a single contiguous buffer.** Flatten each layer's weight matrix and bias vector in row-major order, concatenate them, and record the byte offset of each layer.

3. **Write one kernel function per pass.** Each kernel reads from one device-memory buffer, performs one layer's computation using only scalar accumulators, and writes to another device-memory buffer.

4. **Add activation guards.** For any unbounded activation (GELU, sigmoid, tanh, softmax), add saturation guards that clamp inputs to the mathematically safe range.

5. **Encode all passes into a single command buffer.** This gives free serialization guarantees from the GPU.

6. **Pre-allocate all intermediate buffers at initialization.** Reuse them across inference calls. Never allocate per-frame.

7. **Build a CPU reference.** Maintain a CPU-side forward pass using the same flat weight buffer for bit-level comparison. Run PSNR or MSE checks against the GPU output. Anything below 40 dB PSNR indicates a systematic error.

## Appendix B: Verification Protocol

Silent corruption is the primary risk. Build verification into the deployment pipeline from the start:

1. **Weight packing test.** After loading weights into the GPU buffer, read back specific indices and compare against the original PyTorch state dictionary. Off-by-one errors in weight offsets are the most common bug.

2. **Per-pass comparison.** Run one pass on GPU, read back the intermediate buffer via `StorageModeShared` (zero-copy on Apple Silicon), and compare against the CPU reference. Do this for each pass independently to isolate which layer introduces errors.

3. **Full pipeline PSNR.** Run the complete multi-pass pipeline on GPU and compute PSNR against PyTorch CPU output. Our neural display achieves 68.7 dB. Anything below 40 dB warrants investigation.

4. **NaN sweep.** Run inference on all possible inputs (or a large random sample) and check for NaN values in the output. Any NaN indicates missing activation guards or residual stack overflow.

5. **Boundary conditions.** Test with inputs that maximize dot product magnitudes: all-max embedding entries, adversarial patterns, and rare codebook entries. These are the inputs most likely to trigger overflow in activation functions.
