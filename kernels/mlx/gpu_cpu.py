#!/usr/bin/env python3
"""
Unified GPU CPU Factory -- Single import point for GPU kernel backend selection.

Selects between Rust Metal (fast, ~500x speedup) and Python MLX (legacy) backends
via the NCPU_GPU_BACKEND environment variable:

    auto  (default) -- try Rust Metal, fall back to Python MLX
    rust  -- force Rust Metal (fail if unavailable)
    mlx   -- force Python MLX (useful for debugging/testing)

Future backends can self-register via register_backend().

Usage:
    from kernels.mlx.gpu_cpu import GPUKernelCPU, StopReasonV2, ExecutionResultV2
    cpu = GPUKernelCPU(quiet=True)

    # Discovery API
    from kernels.mlx.gpu_cpu import list_backends, backend_info
    print(list_backends())     # [BackendInfo(name='rust', ...), ...]
    print(backend_info())      # BackendInfo(name='rust', ...)
"""

import importlib
import os
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# BACKEND INFO
# =============================================================================

@dataclass
class BackendInfo:
    """Describes a registered GPU compute backend."""

    name: str
    available: bool
    active: bool
    priority: int  # lower = preferred
    description: str
    import_path: str
    error: Optional[str] = None


# =============================================================================
# BACKEND REGISTRY
# =============================================================================

class _BackendRegistry:
    """Registry for GPU compute backends with priority-based auto-selection.

    Backends register with a name, priority (lower = preferred), and import
    metadata.  The ``resolve`` method loads the highest-priority available
    backend (or a specifically requested one) and exposes the CPU class,
    StopReasonV2, ExecutionResultV2, and optional shared-instance factory.
    """

    def __init__(self):
        self._backends: dict[str, dict] = {}
        self._active: Optional[str] = None

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        priority: int,
        description: str,
        cpu_import: str,
        cpu_class: str,
        stop_reason_import: str = None,
        result_import: str = None,
        shared_cpu_fn: str = None,
        probe: str = None,
    ):
        """Register a backend.  Lower *priority* = preferred."""
        self._backends[name] = {
            "priority": priority,
            "description": description,
            "cpu_import": cpu_import,
            "cpu_class": cpu_class,
            "stop_reason_import": stop_reason_import or cpu_import,
            "result_import": result_import or cpu_import,
            "shared_cpu_fn": shared_cpu_fn,
            "probe": probe,
        }

    # -- resolution ---------------------------------------------------------

    def resolve(self, requested: str = "auto"):
        """Resolve which backend to use.

        Returns ``(cpu_class, StopReasonV2, ExecutionResultV2, name, shared_fn)``
        where *shared_fn* may be ``None``.
        """
        if requested != "auto":
            if requested not in self._backends:
                raise ValueError(
                    f"Unknown backend: {requested!r}. "
                    f"Available: {sorted(self._backends.keys())}"
                )
            return self._try_load(requested, fail_hard=True)

        # Auto: try in priority order (lowest number first).
        sorted_backends = sorted(
            self._backends.items(), key=lambda item: item[1]["priority"]
        )
        for name, _ in sorted_backends:
            result = self._try_load(name, fail_hard=False)
            if result is not None:
                return result

        raise RuntimeError(
            "No GPU backend available. Tried: "
            + ", ".join(n for n, _ in sorted_backends)
        )

    def _try_load(self, name, fail_hard=False):
        """Attempt to import *name* and return the backend tuple, or ``None``."""
        info = self._backends[name]
        probe_error = self._probe_error(info)
        if probe_error is not None:
            if fail_hard:
                raise ImportError(f"Backend {name!r} not available: {probe_error}")
            return None
        try:
            cpu_mod = importlib.import_module(info["cpu_import"])
            cpu_cls = getattr(cpu_mod, info["cpu_class"])

            sr_mod = importlib.import_module(info["stop_reason_import"])
            stop_cls = getattr(sr_mod, "StopReasonV2")
            result_cls = getattr(sr_mod, "ExecutionResultV2")

            shared_fn = None
            if info.get("shared_cpu_fn"):
                shared_fn = getattr(cpu_mod, info["shared_cpu_fn"], None)

            self._active = name
            return cpu_cls, stop_cls, result_cls, name, shared_fn
        except (ImportError, OSError, AttributeError) as exc:
            if fail_hard:
                raise ImportError(
                    f"Backend {name!r} not available: {exc}"
                ) from exc
            return None

    def _probe_error(self, info: dict) -> Optional[str]:
        probe = info.get("probe")
        if not probe:
            return None
        from .availability import mlx_probe, rust_metal_probe

        if probe == "mlx":
            ok, error = mlx_probe()
        elif probe == "rust":
            ok, error = rust_metal_probe()
        else:
            return f"Unknown probe {probe!r}"
        return None if ok else error

    # -- introspection ------------------------------------------------------

    def list_backends(self) -> list[BackendInfo]:
        """Return info for every registered backend, sorted by priority."""
        result = []
        for name, info in sorted(
            self._backends.items(), key=lambda item: item[1]["priority"]
        ):
            error = self._probe_error(info)
            available = error is None
            if available and not info.get("probe"):
                try:
                    mod = importlib.import_module(info["cpu_import"])
                    getattr(mod, info["cpu_class"])
                except (ImportError, OSError, AttributeError) as exc:
                    available = False
                    error = str(exc)
            result.append(
                BackendInfo(
                    name=name,
                    available=available,
                    active=(name == self._active),
                    priority=info["priority"],
                    description=info["description"],
                    import_path=f"{info['cpu_import']}.{info['cpu_class']}",
                    error=error,
                )
            )
        return result

    def backend_info(self) -> Optional[BackendInfo]:
        """Return info about the currently active backend, or ``None``."""
        for entry in self.list_backends():
            if entry.active:
                return entry
        return None


# =============================================================================
# REGISTRY SETUP -- built-in backends
# =============================================================================

_registry = _BackendRegistry()

_registry.register(
    name="rust",
    priority=10,
    description="Rust + Metal with StorageModeShared (zero-copy, ~500x faster)",
    cpu_import="kernels.mlx.rust_runner",
    cpu_class="RustMetalCPU",
    shared_cpu_fn="get_shared_cpu",
    probe="rust",
)

_registry.register(
    name="mlx",
    priority=20,
    description="Python MLX with double-buffer architecture (legacy)",
    cpu_import="kernels.mlx.cpu_kernel_v2",
    cpu_class="MLXKernelCPUv2",
    probe="mlx",
)


# =============================================================================
# RESOLVE ACTIVE BACKEND
# =============================================================================

_BACKEND_ENV = os.environ.get("NCPU_GPU_BACKEND", "auto").lower()
_cpu_cls, StopReasonV2, ExecutionResultV2, BACKEND, _shared_fn = _registry.resolve(
    _BACKEND_ENV
)
GPUKernelCPU = _cpu_cls


# =============================================================================
# PUBLIC API
# =============================================================================

def get_shared_cpu(
    memory_size: int = 16 * 1024 * 1024,
    batch_size: int = 10_000_000,
) -> "GPUKernelCPU":
    """Get a shared GPU CPU instance (reuses Metal device/pipeline on Rust backend)."""
    if _shared_fn is not None:
        return _shared_fn(memory_size, batch_size)
    return GPUKernelCPU(memory_size=memory_size, quiet=True)


def register_backend(
    name: str,
    priority: int,
    description: str,
    cpu_import: str,
    cpu_class: str,
    **kwargs,
):
    """Register a new GPU backend for future use.

    The backend will be available for selection via the ``NCPU_GPU_BACKEND``
    environment variable or programmatic calls to ``list_backends()``.

    Example::

        register_backend(
            name="webgpu",
            priority=15,
            description="WebGPU compute backend via wgpu-py",
            cpu_import="kernels.webgpu.runner",
            cpu_class="WebGPUKernelCPU",
        )
    """
    _registry.register(name, priority, description, cpu_import, cpu_class, **kwargs)


def list_backends() -> list[BackendInfo]:
    """List all registered backends with availability status.

    Returns a list of ``BackendInfo`` dataclasses sorted by priority (lowest
    first).  Each entry indicates whether the backend is importable and
    whether it is the currently active backend.
    """
    return _registry.list_backends()


def backend_info() -> Optional[BackendInfo]:
    """Get info about the currently active backend.

    Returns a ``BackendInfo`` for the backend that was resolved at import
    time, or ``None`` if no backend is active (should not happen in normal
    operation).
    """
    return _registry.backend_info()


__all__ = [
    "GPUKernelCPU",
    "StopReasonV2",
    "ExecutionResultV2",
    "BACKEND",
    "get_shared_cpu",
    "register_backend",
    "list_backends",
    "backend_info",
    "BackendInfo",
]
