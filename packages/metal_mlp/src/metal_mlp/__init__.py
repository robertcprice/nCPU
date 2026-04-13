"""metal-mlp: Deploy trained PyTorch MLPs on Apple Silicon Metal GPU without PyTorch.

This package generalizes the weight-cache-to-Metal-shader pattern into a
reusable library. The core workflow:

    1. **Train** a neural network in PyTorch and save a .pt checkpoint.
    2. **Extract** weights once into a flat f32 numpy array, cached as .npy.
    3. **Load** the .npy weights into Metal GPU buffers via a compiled Rust/Metal
       shared library -- no PyTorch dependency at inference time.

This enables sub-millisecond neural inference on Apple Silicon GPUs without
Python framework overhead. Metal shaders execute the trained weights as native
compute kernels.

Quick start::

    from metal_mlp import WeightCache, MetalKernelLoader

    cache = WeightCache("model.pt", expected_floats=131760)
    weights = cache.load()

    loader = MetalKernelLoader(so_name="my_kernels.so", search_paths=[...])
    if loader.available:
        kernel = loader.get_class("MyKernel")()
        kernel.load_weights(weights.tolist())
        result = kernel.forward(input_data)
"""

from __future__ import annotations

from .analysis import benchmark_inference, print_weight_layout, weight_layout_from_state_dict
from .inference import MetalMLPInference
from .kernel_loader import MetalKernelLoader
from .weight_cache import WeightCache

__all__ = [
    "WeightCache",
    "MetalKernelLoader",
    "MetalMLPInference",
    "weight_layout_from_state_dict",
    "print_weight_layout",
    "benchmark_inference",
]

__version__ = "0.1.0"
