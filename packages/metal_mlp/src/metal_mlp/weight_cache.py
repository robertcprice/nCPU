"""Weight extraction and caching for PyTorch-free Metal inference.

Converts .pt checkpoint weights into flat .npy arrays that can be loaded
without PyTorch. Once the .npy cache exists on disk, subsequent loads
require only numpy -- no torch import, no checkpoint parsing, no GPU
transfer overhead.

The cache file sits alongside the .pt file with a configurable suffix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["WeightCache"]

logger = logging.getLogger(__name__)


class WeightCache:
    """Extract weights from .pt checkpoints and cache as .npy for PyTorch-free loading.

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

    Examples
    --------
    Load from an existing cache (no torch required):

    >>> cache = WeightCache("model.pt", expected_floats=131760)
    >>> if cache.is_cached():
    ...     weights = cache.load()  # numpy-only, fast

    Extract from a checkpoint (requires torch, caches automatically):

    >>> weights = cache.extract_from_state_dict([
    ...     "layer1.weight", "layer1.bias",
    ...     "layer2.weight", "layer2.bias",
    ... ])
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
        """Check whether a valid .npy cache file exists.

        Validates both shape and dtype of the cached array against
        ``expected_floats``.
        """
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
            2. Fall back to torch extraction from the .pt checkpoint,
               auto-discovering keys from the state dict.
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

        Raises
        ------
        ImportError
            If torch is not installed (logged as error, returns None).
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
            sd = torch.load(
                str(self._model_path), map_location="cpu", weights_only=True
            )
        except Exception as exc:
            logger.error("Failed to load checkpoint %s: %s", self._model_path, exc)
            return None

        return self._extract_keys(sd, key_order, save_cache=save_cache)

    # -- Private helpers --------------------------------------------------------

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
                cache,
                arr.shape,
                arr.dtype,
                self._expected_floats,
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
            sd = torch.load(
                str(self._model_path), map_location="cpu", weights_only=True
            )
            # Use natural key order from the state dict (insertion-ordered)
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
                logger.error(
                    "Key '%s' not found in state dict. Available keys: %s",
                    key,
                    list(sd.keys()),
                )
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
                combined.shape[0],
                self._expected_floats,
                key_order,
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
            logger.warning(
                "Failed to write weight cache %s: %s (non-fatal)", cache, exc
            )

    def __repr__(self) -> str:
        cached = "cached" if self.is_cached() else "uncached"
        return (
            f"WeightCache({self._model_path.name!r}, "
            f"expected_floats={self._expected_floats}, {cached})"
        )
