"""High-level Metal MLP inference abstraction.

Combines weight extraction/caching with Metal kernel loading into a single
class that handles the full lifecycle:

    .pt checkpoint --> .npy cache --> Metal GPU buffers --> inference

This is the primary entry point for most users who want to deploy a
trained PyTorch MLP on Apple Silicon Metal GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .kernel_loader import MetalKernelLoader
from .weight_cache import WeightCache

__all__ = ["MetalMLPInference"]

logger = logging.getLogger(__name__)


class MetalMLPInference:
    """Deploy a pre-trained PyTorch MLP on Metal GPU with automatic weight caching.

    Orchestrates ``WeightCache`` for weight extraction/caching and
    ``MetalKernelLoader`` for Metal kernel loading. After initialization,
    the kernel is ready for inference calls without any PyTorch dependency.

    Parameters
    ----------
    model_path : str or Path
        Path to the PyTorch .pt checkpoint.
    kernel_class : str
        Name of the kernel class in the shared library (e.g.
        ``'NeuralDisplayKernel'``, ``'NeuralALUKernel'``).
    weight_keys : list[str]
        State dict keys in the order the Metal shader expects them in the
        flat weight buffer. If empty or None, all keys are extracted in
        the order they appear in the state dict.
    expected_floats : int
        Exact number of f32 values in the flat weight buffer.
    cache_suffix : str
        Suffix for the .npy cache file.
    kernel_loader : MetalKernelLoader or None
        Shared loader instance. If None, a new one is created with no
        search paths (relies on standard import).
    weight_load_method : str
        Name of the method on the kernel instance to call with the weight
        list. Default: ``'load_weights'``.
    auto_init : bool
        Whether to immediately attempt loading weights and kernel. Set to
        False for deferred initialization via ``initialize()``.

    Examples
    --------
    >>> mlp = MetalMLPInference(
    ...     model_path="models/my_model.pt",
    ...     kernel_class="MyMLPKernel",
    ...     weight_keys=["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
    ...     expected_floats=65536,
    ...     kernel_loader=MetalKernelLoader(
    ...         so_name="my_kernels.abi3.so",
    ...         search_paths=[Path("build/")],
    ...     ),
    ... )
    >>> if mlp.available:
    ...     result = mlp.kernel.forward(input_data)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        kernel_class: str,
        weight_keys: Optional[list[str]] = None,
        expected_floats: int = 0,
        cache_suffix: str = ".metal_weights.npy",
        kernel_loader: Optional[MetalKernelLoader] = None,
        weight_load_method: str = "load_weights",
        auto_init: bool = True,
    ) -> None:
        self._weight_cache = WeightCache(
            model_path, expected_floats, cache_suffix=cache_suffix
        )
        self._kernel_class_name = kernel_class
        self._weight_keys = list(weight_keys) if weight_keys else []
        self._weight_load_method = weight_load_method
        self._loader = kernel_loader or MetalKernelLoader()

        self._kernel_instance: Optional[object] = None
        self._available = False
        self._init_error: Optional[str] = None

        if auto_init:
            self.initialize()

    def initialize(self) -> bool:
        """Attempt to load weights and create the Metal kernel.

        Safe to call multiple times; subsequent calls are no-ops if
        already initialized.

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
        if weights is None and self._weight_keys:
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
        """Whether the Metal kernel is loaded and ready for inference."""
        return self._available

    @property
    def kernel(self) -> Optional[object]:
        """The underlying Metal kernel instance, or None if not available.

        Use this to call kernel-specific inference methods::

            mlp.kernel.render(chars, fg, bg)        # display kernel
            mlp.kernel.execute_add(a, b, False)     # ALU kernel
            mlp.kernel.forward(input_data)           # generic kernel
        """
        return self._kernel_instance

    @property
    def init_error(self) -> Optional[str]:
        """Human-readable error from initialization, or None on success."""
        return self._init_error

    @property
    def weight_cache(self) -> WeightCache:
        """The underlying WeightCache instance for direct cache management."""
        return self._weight_cache

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
