# metal-mlp: Deploy PyTorch MLPs on Apple Silicon Metal GPU

Run trained PyTorch MLPs at native Metal GPU speed without PyTorch installed at
inference time. Train your model in PyTorch, extract weights once to a flat numpy
cache, then load them directly into Metal GPU buffers via a compiled Rust/Metal
shared library. This eliminates Python framework overhead and enables
sub-millisecond inference -- the nCPU neural display achieves 361 FPS rendering a
143K-parameter glyph MLP entirely on the GPU.

## Quick Start

```python
from metal_mlp import WeightCache, MetalKernelLoader

# Cache weights from a .pt file (needs torch once)
cache = WeightCache("model.pt", expected_floats=131760)
weights = cache.load()

# Load Metal kernel (no torch needed after caching)
loader = MetalKernelLoader(so_name="my_kernels.so", search_paths=[...])
kernel = loader.get_class("MyKernel")()
kernel.load_weights(weights.tolist())
```

## Installation

```bash
pip install metal-mlp
```

For weight extraction from .pt checkpoints (first-time only):

```bash
pip install metal-mlp[torch]
```

## Usage

### WeightCache: Extract and Cache Weights

`WeightCache` converts PyTorch .pt checkpoints into flat .npy arrays. Once
cached, all subsequent loads are numpy-only -- no torch import required.

```python
from metal_mlp import WeightCache

cache = WeightCache("models/my_model.pt", expected_floats=49536)

# Auto-discover keys from state dict and cache
weights = cache.load()

# Or specify exact key order for your Metal shader layout
weights = cache.extract_from_state_dict([
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
])

# Check cache status
print(cache.is_cached())    # True after first extraction
print(cache.cache_path)     # models/my_model.metal_weights.npy
```

### MetalKernelLoader: Load Metal Shared Libraries

`MetalKernelLoader` handles dynamic loading of compiled Rust/Metal .so files
without polluting `sys.path`. It searches multiple candidate directories and
caches the loaded module.

```python
from pathlib import Path
from metal_mlp import MetalKernelLoader

loader = MetalKernelLoader(
    so_name="my_kernels.abi3.so",
    search_paths=[Path("build/release")],
)

if loader.available:
    kernel = loader.get_class("MyMLPKernel")()
    print(loader.list_classes())  # all exported kernel classes
else:
    print(loader.load_error)
```

### MetalMLPInference: High-Level API

`MetalMLPInference` combines weight caching and kernel loading into a single
object that handles the full lifecycle: `.pt` -> `.npy` cache -> Metal GPU
buffers -> inference.

```python
from metal_mlp import MetalMLPInference, MetalKernelLoader

mlp = MetalMLPInference(
    model_path="model.pt",
    kernel_class="MyMLPKernel",
    weight_keys=["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
    expected_floats=49536,
    kernel_loader=MetalKernelLoader(
        so_name="my_kernels.abi3.so",
        search_paths=[...],
    ),
)

if mlp.available:
    result = mlp.kernel.forward(input_data)
```

### Weight Layout Analysis

Use `print_weight_layout` to inspect a checkpoint before writing your Metal
shader. It shows the exact buffer offset for each weight tensor:

```python
from metal_mlp import print_weight_layout

print_weight_layout("model.pt")
# Key                                Shape               Offset     Count
# fc1.weight                         [256, 64]               0     16384
# fc1.bias                           [256]               16384       256
# fc2.weight                         [128, 256]          16640     32768
# fc2.bias                           [128]               49408       128
# Total: 49,536 floats = 198,144 bytes = 0.19 MB
```

### Benchmarking

Compare Metal inference speed against a PyTorch baseline:

```python
from metal_mlp import benchmark_inference

results = benchmark_inference(
    metal_fn=lambda: kernel.forward(data),
    torch_fn=lambda: model(tensor),
    n_iterations=5000,
    warmup=100,
)
print(f"Metal: {results['metal_fps']:.0f} FPS")
print(f"PyTorch: {results['torch_fps']:.0f} FPS")
print(f"Speedup: {results['speedup']:.1f}x")
```

## The Three-Pass Metal Inference Technique

The Metal kernels use a multi-pass compute shader approach to execute neural
network layers entirely on the GPU without CPU round-trips:

1. **Pass 1 (Load)**: Transfer input data into GPU shared memory.
2. **Pass 2 (Compute)**: Execute neural network layers -- matrix multiply,
   bias add, activation functions -- using the pre-loaded weight buffers.
3. **Pass 3 (Output)**: Post-process results and write to the output buffer.

Each pass is a separate `dispatchThreadgroups` call within one Metal command
buffer. Metal's implicit barriers between compute dispatches provide
synchronization. This avoids the CPU-GPU synchronization overhead that
dominates PyTorch's dispatch path on MPS.

## Performance

From the nCPU neural display benchmark (143K-parameter glyph MLP, 384x640
RGB output, Apple Silicon):

| Backend  | FPS  | Latency  |
|----------|------|----------|
| Metal    | 361  | 2.8 ms   |
| PyTorch  | ~3   | ~330 ms  |
| Speedup  | 120x | --       |

The speedup comes from eliminating Python/PyTorch dispatch overhead. The Metal
shader executes the same trained weights as native GPU compute kernels.

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python** >= 3.10
- **numpy** >= 1.24
- **torch** >= 2.0 (optional, only for first-time weight extraction)
- A compiled Rust/Metal shared library exposing your kernel classes via PyO3

## Writing Your Own Metal Kernel

This package handles the Python side (weight extraction, caching, library
loading). You provide the Metal compute shader compiled into a shared library.
The typical stack:

1. **Rust + PyO3**: Python bindings for your kernel.
2. **metal-rs or objc2-metal**: Rust crate for Metal GPU access.
3. **Metal Shading Language**: The actual compute kernel that reads from the
   flat weight buffer and executes your MLP layers.

See the nCPU project's `kernels/rust_metal/src/neural_display.rs` for a
complete example of a three-pass neural rendering kernel.

## License

MIT
