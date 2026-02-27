"""Tensor-first runtime variants that prioritize tensor execution over inference."""

from .tensor_native_cpu import TensorNativeCPU
from .tensor_native_kernel import TensorNativeKernel

__all__ = ["TensorNativeCPU", "TensorNativeKernel"]
