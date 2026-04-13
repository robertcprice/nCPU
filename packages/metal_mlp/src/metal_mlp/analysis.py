"""Weight layout analysis and inference benchmarking utilities.

These tools help with designing Metal shaders by inspecting checkpoint
weight layouts, and with measuring inference performance to validate
that the Metal deployment matches or exceeds the PyTorch baseline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Union

__all__ = [
    "weight_layout_from_state_dict",
    "print_weight_layout",
    "benchmark_inference",
]

logger = logging.getLogger(__name__)


def weight_layout_from_state_dict(model_path: Union[str, Path]) -> dict:
    """Analyze a .pt checkpoint and return the weight layout for Metal shader design.

    For each tensor in the state dict, computes its shape, byte offset, and
    element count in a hypothetical flat f32 buffer. This is the information
    needed to write the corresponding Metal shader that reads from the weight
    buffer at the correct offsets.

    Parameters
    ----------
    model_path : str or Path
        Path to the .pt checkpoint file.

    Returns
    -------
    dict
        Mapping ``key -> {'shape': tuple, 'offset': int, 'count': int,
        'dtype': str}`` for each key in the state dict, plus a ``'_total'``
        key with summary statistics.

    Raises
    ------
    ImportError
        If torch is not installed.
    FileNotFoundError
        If the model path does not exist.

    Examples
    --------
    >>> layout = weight_layout_from_state_dict("model.pt")
    >>> for key, info in layout.items():
    ...     if key == "_total":
    ...         continue
    ...     print(f"  [{info['offset']:>8d}]  {key:<40s}  {list(info['shape'])}")
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for weight layout analysis. "
            "Install with: pip install torch"
        ) from exc

    sd = torch.load(str(path), map_location="cpu", weights_only=True)

    layout: dict = {}
    offset = 0

    for key in sd:
        val = sd[key]
        if isinstance(val, torch.Tensor):
            shape = tuple(val.shape)
            count = int(val.numel())
            dtype_str = str(val.dtype)
        else:
            # Scalar parameter (e.g., temperature)
            shape = ()
            count = 1
            dtype_str = type(val).__name__

        layout[key] = {
            "shape": shape,
            "offset": offset,
            "count": count,
            "dtype": dtype_str,
        }
        offset += count

    layout["_total"] = {
        "total_floats": offset,
        "total_bytes_f32": offset * 4,
        "total_mb_f32": offset * 4 / (1024 * 1024),
        "n_keys": len(sd),
    }

    return layout


def print_weight_layout(model_path: Union[str, Path]) -> None:
    """Pretty-print the weight layout of a .pt checkpoint.

    Convenience wrapper around ``weight_layout_from_state_dict`` that
    formats the output as a table suitable for designing Metal shader
    buffer offsets.

    Parameters
    ----------
    model_path : str or Path
        Path to the .pt checkpoint file.
    """
    layout = weight_layout_from_state_dict(model_path)
    total_info = layout.pop("_total")

    print(f"\nWeight layout for: {model_path}")
    print(f"{'Key':<50s}  {'Shape':<20s}  {'Offset':>8s}  {'Count':>8s}")
    print("-" * 92)

    for key, info in layout.items():
        shape_str = str(list(info["shape"])) if info["shape"] else "(scalar)"
        print(
            f"  {key:<48s}  {shape_str:<20s}  {info['offset']:>8d}  {info['count']:>8d}"
        )

    print("-" * 92)
    print(
        f"  Total: {total_info['total_floats']:,d} floats "
        f"= {total_info['total_bytes_f32']:,d} bytes "
        f"= {total_info['total_mb_f32']:.2f} MB"
    )
    print(f"  Keys:  {total_info['n_keys']}")
    print()


def benchmark_inference(
    metal_fn: Callable[[], object],
    torch_fn: Optional[Callable[[], object]] = None,
    n_iterations: int = 1000,
    warmup: int = 100,
) -> dict:
    """Benchmark Metal inference, optionally comparing against a PyTorch baseline.

    Runs both functions for ``warmup`` iterations (discarded), then times
    ``n_iterations`` iterations and reports statistics.

    Parameters
    ----------
    metal_fn : callable
        Zero-argument callable that executes one Metal inference pass.
    torch_fn : callable or None
        Zero-argument callable that executes one PyTorch inference pass.
        If None, only Metal is benchmarked.
    n_iterations : int
        Number of timed iterations.
    warmup : int
        Number of warmup iterations (not timed).

    Returns
    -------
    dict
        Keys: ``metal_fps``, ``metal_ms``, ``n_iterations``, ``warmup``.
        If ``torch_fn`` is provided, also includes: ``torch_fps``,
        ``torch_ms``, ``speedup``.

    Examples
    --------
    >>> results = benchmark_inference(
    ...     metal_fn=lambda: kernel.forward(data),
    ...     torch_fn=lambda: model(tensor),
    ...     n_iterations=5000,
    ... )
    >>> print(f"Metal: {results['metal_fps']:.0f} FPS, "
    ...       f"Speedup: {results['speedup']:.1f}x")
    """
    # Warmup Metal
    for _ in range(warmup):
        metal_fn()

    # Time Metal
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        metal_fn()
    metal_elapsed = time.perf_counter() - t0

    metal_fps = n_iterations / metal_elapsed if metal_elapsed > 0 else float("inf")
    metal_ms = (metal_elapsed / n_iterations) * 1000.0

    result: dict = {
        "metal_fps": metal_fps,
        "metal_ms": metal_ms,
        "n_iterations": n_iterations,
        "warmup": warmup,
    }

    if torch_fn is not None:
        # Warmup PyTorch
        for _ in range(warmup):
            torch_fn()

        # Time PyTorch
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            torch_fn()
        torch_elapsed = time.perf_counter() - t0

        torch_fps = (
            n_iterations / torch_elapsed if torch_elapsed > 0 else float("inf")
        )
        torch_ms = (torch_elapsed / n_iterations) * 1000.0

        result["torch_fps"] = torch_fps
        result["torch_ms"] = torch_ms
        result["speedup"] = metal_fps / torch_fps if torch_fps > 0 else float("inf")

    return result
