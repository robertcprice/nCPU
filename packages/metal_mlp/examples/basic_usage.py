#!/usr/bin/env python3
"""Basic usage example for metal-mlp.

Demonstrates the two-step workflow:
  1. Extract and cache weights from a .pt checkpoint (requires torch once).
  2. Load cached weights and deploy on Metal GPU (no torch needed).

This example uses a hypothetical MLP model. Replace the paths and kernel
names with your own trained model and compiled Metal shader.
"""

from pathlib import Path

from metal_mlp import MetalKernelLoader, WeightCache, print_weight_layout

# ---- Step 0: Inspect the weight layout of your model ----
# This helps you design the Metal shader's buffer offset constants.
# Requires torch installed.
#
# print_weight_layout("path/to/your/model.pt")
#
# Output:
#   Key                                Shape               Offset     Count
#   fc1.weight                         [256, 64]               0     16384
#   fc1.bias                           [256]               16384       256
#   fc2.weight                         [128, 256]          16640     32768
#   fc2.bias                           [128]               49408       128
#   Total: 49,536 floats = 198,144 bytes = 0.19 MB

# ---- Step 1: Cache weights (run once with torch installed) ----
MODEL_PATH = "path/to/your/model.pt"
EXPECTED_FLOATS = 49_536  # from print_weight_layout output

cache = WeightCache(MODEL_PATH, expected_floats=EXPECTED_FLOATS)

# Option A: Auto-discover keys from state dict
weights = cache.load()

# Option B: Explicit key order (if your shader expects a specific layout)
# weights = cache.extract_from_state_dict([
#     "fc1.weight", "fc1.bias",
#     "fc2.weight", "fc2.bias",
# ])

if weights is not None:
    print(f"Loaded {len(weights):,} weight floats ({weights.nbytes / 1024:.1f} KB)")
    print(f"Cache location: {cache.cache_path}")
    print(f"Is cached: {cache.is_cached()}")

# ---- Step 2: Load Metal kernel (no torch needed from here) ----
# Point the loader at your compiled Rust/Metal shared library.
loader = MetalKernelLoader(
    so_name="my_kernels.abi3.so",
    search_paths=[
        Path("build/release"),  # your build output directory
    ],
)

if loader.available:
    print(f"\nMetal kernel module loaded. Available classes: {loader.list_classes()}")

    # Get your kernel class and instantiate
    kernel_cls = loader.get_class("MyMLPKernel")
    if kernel_cls is not None and weights is not None:
        kernel = kernel_cls()
        kernel.load_weights(weights.tolist())
        print("Weights loaded into Metal GPU buffers -- ready for inference!")

        # Call your kernel's inference method
        # result = kernel.forward(input_data)
else:
    print(f"\nMetal not available: {loader.load_error}")
    print("This is expected if running without a compiled Metal kernel.")

# ---- Step 3: High-level API (combines steps 1+2) ----
from metal_mlp import MetalMLPInference

mlp = MetalMLPInference(
    model_path=MODEL_PATH,
    kernel_class="MyMLPKernel",
    weight_keys=["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
    expected_floats=EXPECTED_FLOATS,
    kernel_loader=loader,
    auto_init=True,
)

if mlp.available:
    print(f"\nHigh-level API ready: {mlp}")
    print(f"Info: {mlp.info()}")
    # result = mlp.kernel.forward(input_data)
else:
    print(f"\nHigh-level API not available: {mlp.init_error}")
