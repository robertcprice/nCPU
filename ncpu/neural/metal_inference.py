"""Generic PyTorch-free neural inference on Apple Metal GPU.

This module generalizes the weight-cache-to-Metal-shader pattern used throughout
nCPU (MetalNeuralDisplay, MetalNeuralALU) into a reusable library. The core idea:

    1. **Train** a neural network in PyTorch and save a .pt checkpoint.
    2. **Extract** weights once into a flat f32 numpy array, cached as .npy.
    3. **Load** the .npy weights directly into Metal GPU buffers via the Rust
       ncpu_metal shared library -- no PyTorch dependency at inference time.

This enables sub-millisecond neural inference on Apple Silicon GPUs without
Python framework overhead. The Metal shaders execute the same trained weights
as native compute kernels, achieving 100-1000x speedups over PyTorch dispatch.

Weight Buffer Layout
--------------------
Metal shaders expect weights as a single contiguous f32 buffer. The layout is
defined by the order of ``weight_keys`` passed to the extraction functions.
Each key's tensor is flattened in row-major (C) order and concatenated:

    [key_0.flatten(), key_1.flatten(), ..., key_N.flatten()]

The Rust Metal shader must read weights at the matching offsets. Use
``weight_layout_from_state_dict()`` to inspect a checkpoint's layout when
designing new Metal kernels.

Three-Pass Metal Inference Technique
------------------------------------
The nCPU Metal kernels use a multi-pass compute shader approach:

    Pass 1: Load input data into GPU shared memory
    Pass 2: Execute neural network layers (matrix multiply + activation)
    Pass 3: Post-process outputs and write results

Each pass is a separate ``dispatchThreadgroups`` call within one command buffer,
synchronized via Metal's implicit barriers between compute dispatches. This
avoids CPU-GPU round-trips between layers.

Dependencies
------------
- **numpy** (required): weight caching, array operations
- **torch** (optional): only needed for first-time weight extraction from .pt
- **ncpu_metal** (optional): Rust/Metal shared library for GPU inference

Example
-------
    from ncpu.neural.metal_inference import WeightCache, MetalKernelLoader

    # Cache weights from a .pt file (needs torch once)
    cache = WeightCache('models/display/terminal_renderer.pt', expected_floats=131760)
    weights = cache.load()

    # Load Metal kernel (no torch needed)
    loader = MetalKernelLoader()
    if loader.available:
        kernel = loader.get_class('NeuralDisplayKernel')()
        kernel.load_weights(weights.tolist())
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = [
    "WeightCache",
    "MetalKernelLoader",
    "MetalMLPInference",
    "weight_layout_from_state_dict",
    "benchmark_inference",
]

logger = logging.getLogger(__name__)

# Default search paths for the ncpu_metal shared library, ordered by priority.
# The first path found with the target .so file wins. These cover:
#   1. Project build output (maturin develop / cargo build)
#   2. Installed venv package
_DEFAULT_SO_SEARCH_PATHS: list[Path] = [
    Path(__file__).resolve().parent.parent.parent / "kernels" / "rust_metal",
    # Common venv install location (Python 3.13 on macOS)
    Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "ncpu_metal",
]


# ---------------------------------------------------------------------------
# WeightCache
# ---------------------------------------------------------------------------

class WeightCache:
    """Extract weights from .pt checkpoints and cache as .npy for PyTorch-free loading.

    The cache file sits alongside the .pt file with a configurable suffix.
    Once the .npy cache exists, subsequent loads require only numpy -- no
    torch import, no checkpoint parsing, no GPU transfer overhead.

    Parameters
    ----------
    model_path : str or Path
        Path to the PyTorch .pt checkpoint file.
    expected_floats : int
        Exact number of f32 values expected in the flattened weight buffer.
        Used for validation after extraction and on cache load.
    cache_suffix : str
        Suffix appended to the model path stem for the cache file.
        Default: ``'.metal_weights.npy'``

    Example
    -------
        cache = WeightCache('models/display/terminal_renderer.pt', expected_floats=131760)
        if cache.is_cached():
            weights = cache.load()  # numpy-only, no torch
        else:
            # First time: needs torch to extract, then caches automatically
            weights = cache.extract_from_state_dict([
                'glyphs.embed.weight', 'glyphs.net.0.weight', 'glyphs.net.0.bias',
                'glyphs.net.2.weight', 'glyphs.net.2.bias',
                'glyphs.net.4.weight', 'glyphs.net.4.bias',
                'colors.palette.weight',
            ])
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        expected_floats: int,
        cache_suffix: str = ".metal_weights.npy",
    ) -> None:
        self._model_path = Path(model_path)
        self._expected_floats = expected_floats
        self._cache_suffix = cache_suffix

    @property
    def cache_path(self) -> Path:
        """Return the .npy cache file path derived from the model path."""
        return self._model_path.with_suffix(self._cache_suffix)

    @property
    def model_path(self) -> Path:
        """Return the original .pt model path."""
        return self._model_path

    @property
    def expected_floats(self) -> int:
        """Return the expected number of f32 values in the flat weight buffer."""
        return self._expected_floats

    def is_cached(self) -> bool:
        """Check whether a valid .npy cache file exists."""
        cache = self.cache_path
        if not cache.exists():
            return False
        try:
            arr = np.load(str(cache), mmap_mode="r")
            return arr.shape == (self._expected_floats,) and arr.dtype == np.float32
        except Exception:
            return False

    def invalidate_cache(self) -> None:
        """Delete the .npy cache file if it exists."""
        cache = self.cache_path
        if cache.exists():
            cache.unlink()
            logger.info("Invalidated weight cache: %s", cache)

    def load(self) -> Optional[np.ndarray]:
        """Load weights as a flat f32 numpy array.

        Resolution order:
            1. Try the .npy cache (numpy-only, fast).
            2. Fall back to torch extraction from the .pt checkpoint.
               On success, the .npy cache is written automatically.

        Returns
        -------
        np.ndarray or None
            Flat f32 array of shape ``(expected_floats,)``, or ``None`` if
            neither cache nor torch extraction succeeded.
        """
        # Strategy 1: numpy cache
        weights = self._load_from_cache()
        if weights is not None:
            return weights

        # Strategy 2: torch extraction (auto-discovers key order)
        weights = self._load_from_torch_auto()
        if weights is not None:
            return weights

        logger.warning(
            "Could not load weights for %s. "
            "Run once with torch installed to create cache, or call "
            "extract_from_state_dict() with explicit key order.",
            self._model_path,
        )
        return None

    def extract_from_state_dict(
        self, key_order: list[str], *, save_cache: bool = True
    ) -> Optional[np.ndarray]:
        """Extract weights from the .pt checkpoint in the specified key order.

        Each key's tensor is flattened (row-major) and concatenated. The total
        number of floats must match ``expected_floats``.

        Parameters
        ----------
        key_order : list[str]
            State dict keys in the order they should appear in the flat buffer.
        save_cache : bool
            Whether to write the .npy cache file after successful extraction.

        Returns
        -------
        np.ndarray or None
            Flat f32 array, or ``None`` on failure.
        """
        if not self._model_path.exists():
            logger.error("Model file not found: %s", self._model_path)
            return None

        try:
            import torch
        except ImportError:
            logger.error(
                "torch is required for weight extraction but is not installed. "
                "Install torch or provide a pre-built .npy cache at %s",
                self.cache_path,
            )
            return None

        try:
            sd = torch.load(str(self._model_path), map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.error("Failed to load checkpoint %s: %s", self._model_path, exc)
            return None

        return self._extract_keys(sd, key_order, save_cache=save_cache)

    # -- Private helpers ----------------------------------------------------

    def _load_from_cache(self) -> Optional[np.ndarray]:
        """Attempt to load from the .npy cache file."""
        cache = self.cache_path
        if not cache.exists():
            return None
        try:
            arr = np.load(str(cache))
            if arr.shape == (self._expected_floats,) and arr.dtype == np.float32:
                logger.debug("Loaded weight cache: %s", cache)
                return arr
            logger.warning(
                "Cache %s has wrong shape/dtype (got %s %s, expected (%d,) float32). "
                "Invalidating.",
                cache, arr.shape, arr.dtype, self._expected_floats,
            )
            self.invalidate_cache()
        except Exception as exc:
            logger.warning("Failed to read cache %s: %s", cache, exc)
        return None

    def _load_from_torch_auto(self) -> Optional[np.ndarray]:
        """Load from .pt via torch, extracting all keys in sorted order."""
        if not self._model_path.exists():
            return None

        try:
            import torch
        except ImportError:
            return None

        try:
            sd = torch.load(str(self._model_path), map_location="cpu", weights_only=True)
            # Use natural key order from the state dict (insertion-ordered in Python 3.7+)
            all_keys = list(sd.keys())
            return self._extract_keys(sd, all_keys, save_cache=True)
        except Exception as exc:
            logger.debug("Auto-extraction from %s failed: %s", self._model_path, exc)
            return None

    def _extract_keys(
        self, sd: dict, key_order: list[str], *, save_cache: bool
    ) -> Optional[np.ndarray]:
        """Extract and concatenate tensors from a state dict."""
        try:
            import torch
        except ImportError:
            return None

        parts: list[np.ndarray] = []
        total = 0
        for key in key_order:
            if key not in sd:
                logger.error("Key '%s' not found in state dict. Available keys: %s", key, list(sd.keys()))
                return None
            tensor = sd[key]
            if isinstance(tensor, torch.Tensor):
                flat = tensor.detach().cpu().float().numpy().ravel()
            else:
                # Handle non-tensor values (scalars, etc.)
                flat = np.array([float(tensor)], dtype=np.float32)
            parts.append(flat)
            total += len(flat)

        combined = np.concatenate(parts).astype(np.float32)

        if combined.shape[0] != self._expected_floats:
            logger.error(
                "Weight count mismatch: extracted %d floats but expected %d. "
                "Keys: %s",
                combined.shape[0], self._expected_floats, key_order,
            )
            return None

        if save_cache:
            self._write_cache(combined)

        return combined

    def _write_cache(self, arr: np.ndarray) -> None:
        """Write the .npy cache file, logging on failure."""
        cache = self.cache_path
        try:
            np.save(str(cache), arr)
            logger.info("Cached %d weights to %s", len(arr), cache)
        except Exception as exc:
            logger.warning("Failed to write weight cache %s: %s (non-fatal)", cache, exc)

    def __repr__(self) -> str:
        cached = "cached" if self.is_cached() else "uncached"
        return (
            f"WeightCache({self._model_path.name!r}, "
            f"expected_floats={self._expected_floats}, {cached})"
        )


# ---------------------------------------------------------------------------
# MetalKernelLoader
# ---------------------------------------------------------------------------

class MetalKernelLoader:
    """Load kernel classes from the ncpu_metal Rust/Metal shared library.

    The ncpu_metal .so is a PyO3 module compiled from Rust that exposes
    Metal GPU kernels (NeuralALUKernel, NeuralDisplayKernel, etc.) to Python.
    This loader handles the tricky import mechanics:

    - Uses ``importlib.util.spec_from_file_location`` to avoid polluting
      ``sys.path`` with venv paths that carry incompatible torch versions.
    - Caches the loaded module in ``sys.modules`` for subsequent calls.
    - Searches multiple candidate paths (build dir, venv install).

    Parameters
    ----------
    so_name : str
        Filename of the shared library. Default: ``'ncpu_metal.abi3.so'``
    search_paths : list[Path] or None
        Directories to search for the .so file. If None, uses the default
        project paths.

    Example
    -------
        loader = MetalKernelLoader()
        if loader.available:
            alu_kernel = loader.get_class('NeuralALUKernel')()
            display_kernel = loader.get_class('NeuralDisplayKernel')()
    """

    def __init__(
        self,
        so_name: str = "ncpu_metal.abi3.so",
        search_paths: Optional[list[Path]] = None,
    ) -> None:
        self._so_name = so_name
        self._search_paths = search_paths or _DEFAULT_SO_SEARCH_PATHS
        self._module: Optional[object] = None
        self._load_attempted = False
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        """Whether the ncpu_metal module was successfully loaded."""
        self._ensure_loaded()
        return self._module is not None

    @property
    def load_error(self) -> Optional[str]:
        """Human-readable error message if loading failed, or None on success."""
        self._ensure_loaded()
        return self._load_error

    def get_class(self, class_name: str) -> Optional[type]:
        """Retrieve a kernel class from the loaded module by name.

        Parameters
        ----------
        class_name : str
            The PyO3 class name, e.g. ``'NeuralALUKernel'``,
            ``'NeuralDisplayKernel'``, ``'MetalCPU'``.

        Returns
        -------
        type or None
            The class object, or ``None`` if the module is unavailable or the
            class does not exist.
        """
        self._ensure_loaded()
        if self._module is None:
            return None
        cls = getattr(self._module, class_name, None)
        if cls is None:
            logger.warning(
                "Class '%s' not found in ncpu_metal. Available: %s",
                class_name,
                [a for a in dir(self._module) if not a.startswith("_")],
            )
        return cls

    def get_module(self) -> Optional[object]:
        """Return the raw ncpu_metal module, or None if unavailable."""
        self._ensure_loaded()
        return self._module

    def list_classes(self) -> list[str]:
        """List all public names exported by the ncpu_metal module.

        Returns
        -------
        list[str]
            Sorted list of public attribute names, or empty list if unavailable.
        """
        self._ensure_loaded()
        if self._module is None:
            return []
        return sorted(a for a in dir(self._module) if not a.startswith("_"))

    # -- Private helpers ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the module on first access."""
        if self._load_attempted:
            return
        self._load_attempted = True

        # Check sys.modules first (may have been loaded by another codepath)
        if "ncpu_metal" in sys.modules:
            self._module = sys.modules["ncpu_metal"]
            logger.debug("Using already-imported ncpu_metal from sys.modules")
            return

        # Search for the .so file
        import importlib.util

        for search_dir in self._search_paths:
            so_path = search_dir / self._so_name
            if not so_path.exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location("ncpu_metal", str(so_path))
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules["ncpu_metal"] = mod
                self._module = mod
                logger.debug("Loaded ncpu_metal from %s", so_path)
                return
            except Exception as exc:
                logger.debug("Failed to load %s: %s", so_path, exc)
                continue

        # Last resort: plain import
        try:
            import ncpu_metal as _m  # type: ignore[import-not-found]
            self._module = _m
            logger.debug("Loaded ncpu_metal via standard import")
        except ImportError:
            self._load_error = (
                f"ncpu_metal shared library ({self._so_name}) not found. "
                f"Searched: {[str(p / self._so_name) for p in self._search_paths]}. "
                "Build with: cd kernels/rust_metal && maturin develop --release"
            )
            logger.info(self._load_error)

    def __repr__(self) -> str:
        status = "available" if self.available else "unavailable"
        return f"MetalKernelLoader({self._so_name!r}, {status})"


# ---------------------------------------------------------------------------
# MetalMLPInference
# ---------------------------------------------------------------------------

class MetalMLPInference:
    """High-level abstraction for running a pre-trained MLP on Metal GPU.

    Combines weight extraction/caching (``WeightCache``) with Metal kernel
    loading (``MetalKernelLoader``) into a single class. Handles the full
    lifecycle:

        .pt checkpoint --> .npy cache --> Metal GPU buffers --> inference

    Parameters
    ----------
    model_path : str or Path
        Path to the PyTorch .pt checkpoint.
    kernel_class : str
        Name of the PyO3 kernel class in ncpu_metal (e.g.
        ``'NeuralDisplayKernel'``, ``'NeuralALUKernel'``).
    weight_keys : list[str]
        State dict keys in the order the Metal shader expects them in the
        flat weight buffer.
    expected_floats : int
        Exact number of f32 values in the flat weight buffer.
    cache_suffix : str
        Suffix for the .npy cache file.
    kernel_loader : MetalKernelLoader or None
        Shared loader instance. If None, a new one is created.
    weight_load_method : str
        Name of the method on the kernel instance to call with the weight
        list. Default: ``'load_weights'``.
    auto_init : bool
        Whether to immediately attempt loading weights and kernel. Set to
        False for deferred initialization via ``initialize()``.

    Example
    -------
        mlp = MetalMLPInference(
            model_path='models/display/terminal_renderer.pt',
            kernel_class='NeuralDisplayKernel',
            weight_keys=[
                'glyphs.embed.weight', 'glyphs.net.0.weight', 'glyphs.net.0.bias',
                'glyphs.net.2.weight', 'glyphs.net.2.bias',
                'glyphs.net.4.weight', 'glyphs.net.4.bias',
                'colors.palette.weight',
            ],
            expected_floats=131760,
        )
        if mlp.available:
            rgb = mlp.kernel.render(chars, fg, bg)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        kernel_class: str,
        weight_keys: list[str],
        expected_floats: int,
        cache_suffix: str = ".metal_weights.npy",
        kernel_loader: Optional[MetalKernelLoader] = None,
        weight_load_method: str = "load_weights",
        auto_init: bool = True,
    ) -> None:
        self._weight_cache = WeightCache(
            model_path, expected_floats, cache_suffix=cache_suffix
        )
        self._kernel_class_name = kernel_class
        self._weight_keys = list(weight_keys)
        self._weight_load_method = weight_load_method
        self._loader = kernel_loader or MetalKernelLoader()

        self._kernel_instance: Optional[object] = None
        self._available = False
        self._init_error: Optional[str] = None

        if auto_init:
            self.initialize()

    def initialize(self) -> bool:
        """Attempt to load weights and create the Metal kernel.

        Safe to call multiple times; subsequent calls are no-ops if already
        initialized.

        Returns
        -------
        bool
            True if the kernel is ready for inference.
        """
        if self._available:
            return True

        # Step 1: Load Metal kernel class
        kernel_cls = self._loader.get_class(self._kernel_class_name)
        if kernel_cls is None:
            self._init_error = (
                f"Kernel class '{self._kernel_class_name}' not available. "
                f"Metal loader error: {self._loader.load_error}"
            )
            logger.info("[MetalMLPInference] %s", self._init_error)
            return False

        # Step 2: Load weights (cache first, then torch extraction)
        weights = self._weight_cache.load()
        if weights is None:
            # Try explicit key extraction
            weights = self._weight_cache.extract_from_state_dict(self._weight_keys)
        if weights is None:
            self._init_error = (
                f"Could not load weights from {self._weight_cache.model_path}. "
                "Ensure the .pt file exists and torch is installed for first extraction."
            )
            logger.info("[MetalMLPInference] %s", self._init_error)
            return False

        # Step 3: Create kernel and load weights into GPU buffers
        try:
            kernel = kernel_cls()
            load_fn = getattr(kernel, self._weight_load_method, None)
            if load_fn is None:
                self._init_error = (
                    f"Kernel class '{self._kernel_class_name}' has no method "
                    f"'{self._weight_load_method}'"
                )
                logger.error("[MetalMLPInference] %s", self._init_error)
                return False

            load_fn(weights.tolist())

            # Check readiness if the kernel exposes an is_ready() method
            is_ready_fn = getattr(kernel, "is_ready", None)
            if is_ready_fn is not None and not is_ready_fn():
                self._init_error = (
                    f"Kernel '{self._kernel_class_name}' loaded weights but "
                    "is_ready() returned False"
                )
                logger.warning("[MetalMLPInference] %s", self._init_error)
                return False

            self._kernel_instance = kernel
            self._available = True
            logger.info(
                "[MetalMLPInference] Ready: %s with %d weights on Metal GPU",
                self._kernel_class_name,
                len(weights),
            )
            return True

        except Exception as exc:
            self._init_error = (
                f"Failed to initialize kernel '{self._kernel_class_name}': {exc}"
            )
            logger.error("[MetalMLPInference] %s", self._init_error)
            return False

    @property
    def available(self) -> bool:
        """Whether the Metal kernel is loaded, weights are set, and inference is ready."""
        return self._available

    @property
    def kernel(self) -> Optional[object]:
        """The underlying Metal kernel instance, or None if not available.

        Use this to call kernel-specific inference methods, e.g.::

            mlp.kernel.render(chars, fg, bg)      # NeuralDisplayKernel
            mlp.kernel.execute_add(a, b, False)    # NeuralALUKernel
        """
        return self._kernel_instance

    @property
    def init_error(self) -> Optional[str]:
        """Human-readable error from initialization, or None on success."""
        return self._init_error

    def info(self) -> dict:
        """Return metadata about the loaded model and kernel state.

        Returns
        -------
        dict
            Keys: ``available``, ``kernel_class``, ``model_path``,
            ``cache_path``, ``is_cached``, ``expected_floats``,
            ``weight_keys``, ``init_error``.
        """
        return {
            "available": self._available,
            "kernel_class": self._kernel_class_name,
            "model_path": str(self._weight_cache.model_path),
            "cache_path": str(self._weight_cache.cache_path),
            "is_cached": self._weight_cache.is_cached(),
            "expected_floats": self._weight_cache.expected_floats,
            "weight_keys": self._weight_keys,
            "init_error": self._init_error,
        }

    def __repr__(self) -> str:
        status = "ready" if self._available else "not ready"
        return (
            f"MetalMLPInference({self._kernel_class_name!r}, "
            f"{self._weight_cache.model_path.name!r}, {status})"
        )


# ---------------------------------------------------------------------------
# Utility: weight_layout_from_state_dict
# ---------------------------------------------------------------------------

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
        key with the total float count.

    Raises
    ------
    ImportError
        If torch is not installed.
    FileNotFoundError
        If the model path does not exist.

    Example
    -------
        layout = weight_layout_from_state_dict('models/display/terminal_renderer.pt')
        for key, info in layout.items():
            if key == '_total':
                continue
            print(f"  [{info['offset']:>8d} .. {info['offset'] + info['count'] - 1:<8d}]  "
                  f"{key:<40s}  {list(info['shape'])}")
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

    Convenience wrapper around ``weight_layout_from_state_dict`` that formats
    the output as a table suitable for designing Metal shader buffer offsets.

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
        end_offset = info["offset"] + info["count"] - 1
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


# ---------------------------------------------------------------------------
# Utility: benchmark_inference
# ---------------------------------------------------------------------------

def benchmark_inference(
    metal_fn,
    torch_fn=None,
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

    Example
    -------
        results = benchmark_inference(
            metal_fn=lambda: kernel.execute_add(a_batch, b_batch, False, False),
            torch_fn=lambda: model(a_tensor, b_tensor),
            n_iterations=5000,
        )
        print(f"Metal: {results['metal_fps']:.0f} FPS, "
              f"PyTorch: {results['torch_fps']:.0f} FPS, "
              f"Speedup: {results['speedup']:.1f}x")
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

        torch_fps = n_iterations / torch_elapsed if torch_elapsed > 0 else float("inf")
        torch_ms = (torch_elapsed / n_iterations) * 1000.0

        result["torch_fps"] = torch_fps
        result["torch_ms"] = torch_ms
        result["speedup"] = metal_fps / torch_fps if torch_fps > 0 else float("inf")

    return result
