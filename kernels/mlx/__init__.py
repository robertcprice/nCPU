"""
MLX / GPU backend package.

This package intentionally avoids importing MLX-backed modules at import time.
Some environments have ``mlx`` installed but unusable, and importing
``mlx.core`` can abort the interpreter instead of raising ImportError.

Use attribute access (or direct submodule imports) to load the actual backend
objects lazily.
"""

from __future__ import annotations

import importlib


_EXPORTS = {
    "MLXKernelCPU": ("kernels.mlx.cpu_kernel", "MLXKernelCPU"),
    "StopReason": ("kernels.mlx.cpu_kernel", "StopReason"),
    "ExecutionResult": ("kernels.mlx.cpu_kernel", "ExecutionResult"),
    "create_cpu": ("kernels.mlx.cpu_kernel", "create_cpu"),
    "KERNEL_HEADER": ("kernels.mlx.cpu_kernel_source", "KERNEL_HEADER"),
    "KERNEL_SOURCE": ("kernels.mlx.cpu_kernel_source", "KERNEL_SOURCE"),
    "STOP_RUNNING": ("kernels.mlx.cpu_kernel_source", "STOP_RUNNING"),
    "STOP_HALT": ("kernels.mlx.cpu_kernel_source", "STOP_HALT"),
    "STOP_SYSCALL": ("kernels.mlx.cpu_kernel_source", "STOP_SYSCALL"),
    "STOP_MAX_CYCLES": ("kernels.mlx.cpu_kernel_source", "STOP_MAX_CYCLES"),
    "get_kernel_source": ("kernels.mlx.cpu_kernel_source", "get_kernel_source"),
    "get_full_kernel_source": ("kernels.mlx.cpu_kernel_source", "get_full_kernel_source"),
    "MLXKernelCPUv2": ("kernels.mlx.cpu_kernel_v2", "MLXKernelCPUv2"),
    "StopReasonV2": ("kernels.mlx.cpu_kernel_v2", "StopReasonV2"),
    "ExecutionResultV2": ("kernels.mlx.cpu_kernel_v2", "ExecutionResultV2"),
    "create_cpu_v2": ("kernels.mlx.cpu_kernel_v2", "create_cpu_v2"),
    "NCPUComputeKernel": ("kernels.mlx.ncpu_kernel", "NCPUComputeKernel"),
    "ComputeResult": ("kernels.mlx.ncpu_kernel", "ComputeResult"),
    "NCPU_KERNEL_HEADER": ("kernels.mlx.ncpu_kernel_source", "NCPU_KERNEL_HEADER"),
    "NCPU_KERNEL_SOURCE": ("kernels.mlx.ncpu_kernel_source", "NCPU_KERNEL_SOURCE"),
    "NCPU_STOP_RUNNING": ("kernels.mlx.ncpu_kernel_source", "NCPU_STOP_RUNNING"),
    "NCPU_STOP_HALT": ("kernels.mlx.ncpu_kernel_source", "NCPU_STOP_HALT"),
    "NCPU_STOP_MAX_CYCLES": ("kernels.mlx.ncpu_kernel_source", "NCPU_STOP_MAX_CYCLES"),
    "MuxleqVM": ("kernels.mlx.muxleq_kernel", "MuxleqVM"),
    "MuxleqResult": ("kernels.mlx.muxleq_kernel", "MuxleqResult"),
    "MUXLEQ_KERNEL_HEADER": ("kernels.mlx.muxleq_kernel_source", "MUXLEQ_KERNEL_HEADER"),
    "MUXLEQ_KERNEL_SOURCE": ("kernels.mlx.muxleq_kernel_source", "MUXLEQ_KERNEL_SOURCE"),
    "MUXLEQ_STOP_RUNNING": ("kernels.mlx.muxleq_kernel_source", "MUXLEQ_STOP_RUNNING"),
    "MUXLEQ_STOP_HALT": ("kernels.mlx.muxleq_kernel_source", "MUXLEQ_STOP_HALT"),
    "MUXLEQ_STOP_MAX_CYCLES": ("kernels.mlx.muxleq_kernel_source", "MUXLEQ_STOP_MAX_CYCLES"),
    "MUXLEQ_STOP_IO_READ": ("kernels.mlx.muxleq_kernel_source", "MUXLEQ_STOP_IO_READ"),
    "MUXLEQ_STOP_IO_WRITE": ("kernels.mlx.muxleq_kernel_source", "MUXLEQ_STOP_IO_WRITE"),
}

__all__ = sorted(_EXPORTS)
__version__ = "1.1.0"


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
