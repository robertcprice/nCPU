"""Metal Neural OS Models — Watchdog LSTM and GIC MLP on Metal GPU.

Standalone Python wrappers for the WatchdogMetalKernel and GICMetalKernel
Metal compute shaders. These port the trained PyTorch models from
ncpu/os/neuros/watchdog.py (WatchdogNet) and ncpu/os/neuros/interrupts.py
(NeuralPriorityEncoder) to native Metal GPU execution.

Each wrapper:
  - Extracts trained weights from .pt checkpoints
  - Caches weights as .npy for PyTorch-free subsequent loads
  - Loads weights into Metal GPU buffers via the Rust ncpu_metal library
  - Provides a Python API matching the original PyTorch models
  - Auto-detects Metal availability and falls back to PyTorch

Architecture:
  WatchdogNet (5,921 params):
    LSTM(input=8, hidden=32, 1 layer)
      -> Linear(32, 16) + ReLU
      -> Linear(16, 1) + Sigmoid
    Input:  [seq_len, 8] system metrics (flattened)
    Output: scalar anomaly score [0, 1]

  NeuralPriorityEncoder / GIC (12,448 params):
    Linear(96, 64) + ReLU
      -> Linear(64, 64) + ReLU
      -> Linear(64, 32)
    Input:  [96] float (IRR[32] + ISR[32] + IMR[32])
    Output: [32] priority scores

Usage:
    from ncpu.neural.metal_neural_os_models import (
        MetalWatchdog, MetalGIC,
        load_metal_watchdog, load_metal_gic,
    )

    # Watchdog
    watchdog = load_metal_watchdog()
    if watchdog.available:
        score = watchdog.check(metrics_flat, seq_len=64)
        scores = watchdog.check_batch(windows_flat, seq_len=64, batch_size=16)

    # GIC
    gic = load_metal_gic()
    if gic.available:
        priority_scores = gic.dispatch(irr, isr, imr, pending)
        best_irq = gic.dispatch_best(irr, isr, imr)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from ncpu.neural.metal_inference import WeightCache, MetalKernelLoader

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Watchdog: LSTM(8->32) + Scorer(32->16->1)
N_WATCHDOG_WEIGHTS = 5921
_WATCHDOG_WEIGHT_KEYS = [
    "lstm.weight_ih_l0",   # [128, 8]  = 1024 floats
    "lstm.weight_hh_l0",   # [128, 32] = 4096 floats
    "lstm.bias_ih_l0",     # [128]
    "lstm.bias_hh_l0",     # [128]
    "scorer.0.weight",     # [16, 32]  = 512 floats
    "scorer.0.bias",       # [16]
    "scorer.2.weight",     # [1, 16]   = 16 floats
    "scorer.2.bias",       # [1]
]

# GIC: MLP [96 -> 64 -> 64 -> 32]
N_GIC_WEIGHTS = 12448
_GIC_WEIGHT_KEYS = [
    "net.0.weight",   # [64, 96]  = 6144 floats
    "net.0.bias",     # [64]
    "net.2.weight",   # [64, 64]  = 4096 floats
    "net.2.bias",     # [64]
    "net.4.weight",   # [32, 64]  = 2048 floats
    "net.4.bias",     # [32]
]

# Shared kernel loader (singleton avoids repeated .so discovery)
_kernel_loader = MetalKernelLoader()


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_watchdog_weights(
    model_path: str = "models/os/watchdog.pt",
) -> Optional[list[float]]:
    """Load watchdog weights from .npy cache or .pt checkpoint.

    Resolution order:
        1. .npy cache (numpy-only, fast)
        2. Torch extraction with explicit key ordering
        3. Direct torch.load with manual concatenation (fallback)
    """
    p = Path(model_path) if Path(model_path).is_absolute() else MODELS_DIR.parent / model_path
    cache = WeightCache(
        str(p), N_WATCHDOG_WEIGHTS,
        cache_suffix=".metal_watchdog_weights.npy",
    )

    # Try cache
    weights = cache.load()
    if weights is not None:
        return weights.tolist()

    # Try extraction with explicit keys
    weights = cache.extract_from_state_dict(_WATCHDOG_WEIGHT_KEYS)
    if weights is not None:
        return weights.tolist()

    # Manual fallback (in case state dict key names vary)
    if not p.exists():
        return None
    try:
        import torch
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        flat: list[float] = []
        flat.extend(sd["lstm.weight_ih_l0"].flatten().tolist())
        flat.extend(sd["lstm.weight_hh_l0"].flatten().tolist())
        flat.extend(sd["lstm.bias_ih_l0"].tolist())
        flat.extend(sd["lstm.bias_hh_l0"].tolist())
        flat.extend(sd["scorer.0.weight"].flatten().tolist())
        flat.extend(sd["scorer.0.bias"].tolist())
        flat.extend(sd["scorer.2.weight"].flatten().tolist())
        flat.extend(sd["scorer.2.bias"].tolist())
        if len(flat) != N_WATCHDOG_WEIGHTS:
            logger.error(
                "Watchdog weights: expected %d, got %d", N_WATCHDOG_WEIGHTS, len(flat)
            )
            return None
        return flat
    except Exception as e:
        logger.debug("Watchdog manual extraction failed: %s", e)
        return None


def _load_gic_weights(
    model_path: str = "models/os/gic.pt",
) -> Optional[list[float]]:
    """Load GIC weights from .npy cache or .pt checkpoint.

    Resolution order:
        1. .npy cache (numpy-only, fast)
        2. Torch extraction with explicit key ordering
        3. Direct torch.load with manual concatenation (fallback)
    """
    p = Path(model_path) if Path(model_path).is_absolute() else MODELS_DIR.parent / model_path
    cache = WeightCache(
        str(p), N_GIC_WEIGHTS,
        cache_suffix=".metal_gic_weights.npy",
    )

    # Try cache
    weights = cache.load()
    if weights is not None:
        return weights.tolist()

    # Try extraction with explicit keys
    weights = cache.extract_from_state_dict(_GIC_WEIGHT_KEYS)
    if weights is not None:
        return weights.tolist()

    # Manual fallback
    if not p.exists():
        return None
    try:
        import torch
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        flat: list[float] = []
        flat.extend(sd["net.0.weight"].flatten().tolist())
        flat.extend(sd["net.0.bias"].tolist())
        flat.extend(sd["net.2.weight"].flatten().tolist())
        flat.extend(sd["net.2.bias"].tolist())
        flat.extend(sd["net.4.weight"].flatten().tolist())
        flat.extend(sd["net.4.bias"].tolist())
        if len(flat) != N_GIC_WEIGHTS:
            logger.error(
                "GIC weights: expected %d, got %d", N_GIC_WEIGHTS, len(flat)
            )
            return None
        return flat
    except Exception as e:
        logger.debug("GIC manual extraction failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MetalWatchdog
# ─────────────────────────────────────────────────────────────────────────────

class MetalWatchdog:
    """Metal GPU-accelerated Watchdog LSTM anomaly detector.

    Wraps WatchdogMetalKernel (Rust/Metal) with automatic weight loading
    and a PyTorch fallback. Provides the same interface as NeuralWatchdog
    but executes on Metal when available.

    Falls back to PyTorch inference when Metal is unavailable.
    """

    def __init__(
        self,
        model_path: str = "models/os/watchdog.pt",
        verbose: bool = False,
    ):
        self._metal_kernel = None
        self._torch_model = None
        self._available = False
        self._backend = "none"
        self._model_path = model_path

        # Try Metal first
        kernel_cls = _kernel_loader.get_class("WatchdogMetalKernel")
        if kernel_cls is not None:
            weights = _load_watchdog_weights(model_path)
            if weights is not None:
                try:
                    kernel = kernel_cls()
                    kernel.load_weights(weights)
                    if kernel.is_ready():
                        self._metal_kernel = kernel
                        self._available = True
                        self._backend = "metal"
                        if verbose:
                            logger.info(
                                "[MetalWatchdog] Metal backend ready (%d weights)",
                                len(weights),
                            )
                        return
                except Exception as e:
                    if verbose:
                        logger.warning("[MetalWatchdog] Metal init failed: %s", e)

        # Fall back to PyTorch
        self._try_torch_fallback(verbose)

    def _try_torch_fallback(self, verbose: bool) -> None:
        """Attempt to load the PyTorch model as fallback."""
        p = Path(self._model_path)
        if not p.is_absolute():
            p = MODELS_DIR.parent / self._model_path
        if not p.exists():
            if verbose:
                logger.info("[MetalWatchdog] No model file at %s", p)
            return
        try:
            import torch
            from ncpu.os.neuros.watchdog import WatchdogNet

            net = WatchdogNet()
            net.load_state_dict(
                torch.load(str(p), map_location="cpu", weights_only=True)
            )
            net.eval()
            self._torch_model = net
            self._available = True
            self._backend = "pytorch"
            if verbose:
                logger.info("[MetalWatchdog] PyTorch fallback ready")
        except Exception as e:
            if verbose:
                logger.warning("[MetalWatchdog] PyTorch fallback failed: %s", e)

    @property
    def available(self) -> bool:
        """Whether the watchdog is ready for inference (Metal or PyTorch)."""
        return self._available

    @property
    def backend(self) -> str:
        """Active backend: 'metal', 'pytorch', or 'none'."""
        return self._backend

    def check(self, metrics_window_flat: list[float], seq_len: int) -> float:
        """Run anomaly detection on a metrics window.

        Args:
            metrics_window_flat: [seq_len * 8] float — flattened metrics
            seq_len: number of timesteps

        Returns:
            Anomaly score in [0, 1] (higher = more anomalous)
        """
        if self._metal_kernel is not None:
            return self._metal_kernel.check(metrics_window_flat, seq_len)

        if self._torch_model is not None:
            import torch
            t = torch.tensor(metrics_window_flat, dtype=torch.float32)
            t = t.reshape(1, seq_len, 8)
            with torch.no_grad():
                score = self._torch_model(t)
            return float(score.item())

        raise RuntimeError("Watchdog not available — no Metal or PyTorch backend")

    def check_batch(
        self,
        metrics_windows_flat: list[float],
        seq_len: int,
        batch_size: int,
    ) -> list[float]:
        """Run anomaly detection on a batch of metrics windows.

        Args:
            metrics_windows_flat: [batch_size * seq_len * 8] float
            seq_len: timesteps per window
            batch_size: number of windows

        Returns:
            [batch_size] float anomaly scores
        """
        if self._metal_kernel is not None:
            return self._metal_kernel.check_batch(
                metrics_windows_flat, seq_len, batch_size
            )

        if self._torch_model is not None:
            import torch
            t = torch.tensor(metrics_windows_flat, dtype=torch.float32)
            t = t.reshape(batch_size, seq_len, 8)
            with torch.no_grad():
                scores = self._torch_model(t)
            return scores.tolist()

        raise RuntimeError("Watchdog not available — no Metal or PyTorch backend")

    def check_tensor(self, metrics_buffer: "torch.Tensor") -> float:
        """Run anomaly detection from a torch tensor [window_size, 8].

        Convenience method matching NeuralWatchdog._neural_check interface.
        """
        if self._metal_kernel is not None:
            flat = metrics_buffer.flatten().tolist()
            seq_len = metrics_buffer.shape[0]
            return self._metal_kernel.check(flat, seq_len)

        if self._torch_model is not None:
            import torch
            with torch.no_grad():
                score = self._torch_model(metrics_buffer.unsqueeze(0))
            return float(score.item())

        return 0.0

    def benchmark(self) -> dict:
        """Benchmark inference. Returns per-call latency in microseconds."""
        if self._metal_kernel is not None:
            single_us, batch_us = self._metal_kernel.benchmark()
            return {
                "backend": "metal",
                "single_64step_us": single_us,
                "batch_16x64step_us": batch_us,
            }
        return {"backend": self._backend, "error": "Metal not available for benchmark"}

    def __repr__(self) -> str:
        return f"MetalWatchdog(backend={self._backend!r}, available={self._available})"


# ─────────────────────────────────────────────────────────────────────────────
# MetalGIC
# ─────────────────────────────────────────────────────────────────────────────

class MetalGIC:
    """Metal GPU-accelerated GIC (Generic Interrupt Controller) neural priority encoder.

    Wraps GICMetalKernel (Rust/Metal) with automatic weight loading
    and a PyTorch fallback. Provides the same interface as NeuralGIC
    but executes on Metal when available.

    Falls back to PyTorch inference when Metal is unavailable.
    """

    def __init__(
        self,
        model_path: str = "models/os/gic.pt",
        verbose: bool = False,
    ):
        self._metal_kernel = None
        self._torch_model = None
        self._available = False
        self._backend = "none"
        self._model_path = model_path

        # Try Metal first
        kernel_cls = _kernel_loader.get_class("GICMetalKernel")
        if kernel_cls is not None:
            weights = _load_gic_weights(model_path)
            if weights is not None:
                try:
                    kernel = kernel_cls()
                    kernel.load_weights(weights)
                    if kernel.is_ready():
                        self._metal_kernel = kernel
                        self._available = True
                        self._backend = "metal"
                        if verbose:
                            logger.info(
                                "[MetalGIC] Metal backend ready (%d weights)",
                                len(weights),
                            )
                        return
                except Exception as e:
                    if verbose:
                        logger.warning("[MetalGIC] Metal init failed: %s", e)

        # Fall back to PyTorch
        self._try_torch_fallback(verbose)

    def _try_torch_fallback(self, verbose: bool) -> None:
        """Attempt to load the PyTorch model as fallback."""
        p = Path(self._model_path)
        if not p.is_absolute():
            p = MODELS_DIR.parent / self._model_path
        if not p.exists():
            if verbose:
                logger.info("[MetalGIC] No model file at %s", p)
            return
        try:
            import torch
            from ncpu.os.neuros.interrupts import NeuralPriorityEncoder

            encoder = NeuralPriorityEncoder()
            encoder.load_state_dict(
                torch.load(str(p), map_location="cpu", weights_only=True)
            )
            encoder.eval()
            self._torch_model = encoder
            self._available = True
            self._backend = "pytorch"
            if verbose:
                logger.info("[MetalGIC] PyTorch fallback ready")
        except Exception as e:
            if verbose:
                logger.warning("[MetalGIC] PyTorch fallback failed: %s", e)

    @property
    def available(self) -> bool:
        """Whether the GIC is ready for inference (Metal or PyTorch)."""
        return self._available

    @property
    def backend(self) -> str:
        """Active backend: 'metal', 'pytorch', or 'none'."""
        return self._backend

    def dispatch(
        self,
        irr: list[float],
        isr: list[float],
        imr: list[float],
        pending: list[float],
    ) -> list[float]:
        """Score interrupt priorities.

        Args:
            irr: [32] float — interrupt request register bits
            isr: [32] float — in-service register bits
            imr: [32] float — interrupt mask register bits
            pending: [32] float — 1.0 where IRQ is pending, 0.0 otherwise

        Returns:
            [32] float priority scores (non-pending masked to -inf)
        """
        if self._metal_kernel is not None:
            input_state = irr + isr + imr  # [96]
            return self._metal_kernel.dispatch(input_state, pending)

        if self._torch_model is not None:
            import torch
            state = torch.tensor(irr + isr + imr, dtype=torch.float32)
            with torch.no_grad():
                scores = self._torch_model(state)
            # Mask non-pending
            pending_t = torch.tensor(pending, dtype=torch.float32)
            scores[pending_t < 0.5] = float("-inf")
            return scores.tolist()

        raise RuntimeError("GIC not available — no Metal or PyTorch backend")

    def dispatch_best(
        self,
        irr: list[float],
        isr: list[float],
        imr: list[float],
    ) -> int:
        """Score priorities and return the best (highest-scoring) IRQ index.

        Computes the pending mask automatically from irr/isr/imr:
        an IRQ is pending if raised (irr), not masked (imr), and not in-service (isr).

        Args:
            irr: [32] float — interrupt request register bits
            isr: [32] float — in-service register bits
            imr: [32] float — interrupt mask register bits

        Returns:
            Best IRQ index (0-31), or -1 if no IRQs are pending.
        """
        pending = [
            (1.0 if (irr[i] > 0.5 and imr[i] < 0.5 and isr[i] < 0.5) else 0.0)
            for i in range(32)
        ]
        if sum(pending) == 0:
            return -1

        scores = self.dispatch(irr, isr, imr, pending)
        return int(max(range(32), key=lambda i: scores[i]))

    def dispatch_tensors(
        self,
        irr_tensor: "torch.Tensor",
        isr_tensor: "torch.Tensor",
        imr_tensor: "torch.Tensor",
    ) -> int:
        """Score priorities from torch tensors, return best IRQ index.

        Convenience method matching NeuralGIC._neural_dispatch interface.
        """
        irr = irr_tensor.float().tolist()
        isr = isr_tensor.float().tolist()
        imr = imr_tensor.float().tolist()
        return self.dispatch_best(irr, isr, imr)

    def dispatch_batch(
        self,
        input_states_flat: list[float],
        pending_masks_flat: list[float],
        batch_size: int,
    ) -> list[float]:
        """Score priorities for a batch of dispatches in parallel.

        Args:
            input_states_flat: [batch_size * 96] float — concatenated states
            pending_masks_flat: [batch_size * 32] float — concatenated masks
            batch_size: number of dispatches

        Returns:
            [batch_size * 32] float priority scores
        """
        if self._metal_kernel is not None:
            return self._metal_kernel.dispatch_batch(
                input_states_flat, pending_masks_flat, batch_size
            )

        if self._torch_model is not None:
            import torch
            states = torch.tensor(input_states_flat, dtype=torch.float32)
            states = states.reshape(batch_size, 96)
            masks = torch.tensor(pending_masks_flat, dtype=torch.float32)
            masks = masks.reshape(batch_size, 32)
            results = []
            with torch.no_grad():
                for i in range(batch_size):
                    scores = self._torch_model(states[i])
                    scores[masks[i] < 0.5] = float("-inf")
                    results.extend(scores.tolist())
            return results

        raise RuntimeError("GIC not available — no Metal or PyTorch backend")

    def benchmark(self) -> dict:
        """Benchmark inference. Returns per-call latency in microseconds."""
        if self._metal_kernel is not None:
            single_us, batch_us = self._metal_kernel.benchmark()
            return {
                "backend": "metal",
                "single_dispatch_us": single_us,
                "batch_32_dispatch_us": batch_us,
            }
        return {"backend": self._backend, "error": "Metal not available for benchmark"}

    def __repr__(self) -> str:
        return f"MetalGIC(backend={self._backend!r}, available={self._available})"


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def load_metal_watchdog(
    model_path: str = "models/os/watchdog.pt",
    verbose: bool = False,
) -> MetalWatchdog:
    """Load the Metal Watchdog LSTM anomaly detector.

    Tries Metal first, falls back to PyTorch if Metal is unavailable.
    Returns a MetalWatchdog with .available indicating readiness.

    Args:
        model_path: path to trained watchdog.pt checkpoint
        verbose: print loading progress

    Returns:
        MetalWatchdog instance (check .available and .backend)
    """
    return MetalWatchdog(model_path=model_path, verbose=verbose)


def load_metal_gic(
    model_path: str = "models/os/gic.pt",
    verbose: bool = False,
) -> MetalGIC:
    """Load the Metal GIC neural priority encoder.

    Tries Metal first, falls back to PyTorch if Metal is unavailable.
    Returns a MetalGIC with .available indicating readiness.

    Args:
        model_path: path to trained gic.pt checkpoint
        verbose: print loading progress

    Returns:
        MetalGIC instance (check .available and .backend)
    """
    return MetalGIC(model_path=model_path, verbose=verbose)
