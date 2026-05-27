"""Metal Neural OS Kernels — runs trained neurOS models (GIC, Watchdog, Compiler Optimizer)
on Metal GPU via native Rust shaders, eliminating Python/PyTorch per-syscall overhead.

Ablation study showed these three models cause ~60% overhead when run via PyTorch.
By porting to Metal compute shaders (same pattern as MetalNeuralALU), they become
near-free GPU operations — one Metal dispatch per call instead of Python round-trips.

Usage:
    from ncpu.neural.metal_neural_os import MetalNeuralOS, load_metal_neural_os
    os_kernels = load_metal_neural_os(verbose=True)
    if os_kernels.available:
        # GIC: score interrupt priorities
        scores = os_kernels.gic_dispatch(irr_bits, isr_bits, imr_bits, pending_bits)
        best_irq = scores.argmax()

        # Watchdog: anomaly detection on metrics window
        anomaly_score = os_kernels.watchdog_check(metrics_window_flat, seq_len=64)

        # Compiler optimizer: score IR windows
        opt_scores = os_kernels.compiler_score(windows_flat, n_windows=10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gic_weights(path: str = "models/os/gic.pt") -> Optional[list[float]]:
    """Extract GIC MLP weights as flat float list (12,448 values).

    Layout:
      FC1 weight [64, 96] + FC1 bias [64] +
      FC2 weight [64, 64] + FC2 bias [64] +
      FC3 weight [32, 64] + FC3 bias [32]
    """
    p = Path(path) if Path(path).is_absolute() else MODELS_DIR.parent / path
    if not p.exists():
        return None
    try:
        import torch
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        flat: list[float] = []
        # FC1
        flat.extend(sd["net.0.weight"].flatten().tolist())   # [64, 96]
        flat.extend(sd["net.0.bias"].tolist())               # [64]
        # FC2
        flat.extend(sd["net.2.weight"].flatten().tolist())   # [64, 64]
        flat.extend(sd["net.2.bias"].tolist())               # [64]
        # FC3
        flat.extend(sd["net.4.weight"].flatten().tolist())   # [32, 64]
        flat.extend(sd["net.4.bias"].tolist())               # [32]
        assert len(flat) == 12448, f"gic_weights: expected 12448, got {len(flat)}"
        return flat
    except Exception:
        return None


def _extract_watchdog_weights(path: str = "models/os/watchdog.pt") -> Optional[list[float]]:
    """Extract Watchdog LSTM + scorer weights as flat float list (5,921 values).

    Layout:
      lstm.weight_ih [128, 8] + lstm.weight_hh [128, 32] +
      lstm.bias_ih [128] + lstm.bias_hh [128] +
      scorer FC1 weight [16, 32] + scorer FC1 bias [16] +
      scorer FC2 weight [1, 16] + scorer FC2 bias [1]
    """
    p = Path(path) if Path(path).is_absolute() else MODELS_DIR.parent / path
    if not p.exists():
        return None
    try:
        import torch
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        flat: list[float] = []
        # LSTM
        flat.extend(sd["lstm.weight_ih_l0"].flatten().tolist())   # [128, 8]
        flat.extend(sd["lstm.weight_hh_l0"].flatten().tolist())   # [128, 32]
        flat.extend(sd["lstm.bias_ih_l0"].tolist())               # [128]
        flat.extend(sd["lstm.bias_hh_l0"].tolist())               # [128]
        # Scorer
        flat.extend(sd["scorer.0.weight"].flatten().tolist())     # [16, 32]
        flat.extend(sd["scorer.0.bias"].tolist())                 # [16]
        flat.extend(sd["scorer.2.weight"].flatten().tolist())     # [1, 16]
        flat.extend(sd["scorer.2.bias"].tolist())                 # [1]
        assert len(flat) == 5921, f"watchdog_weights: expected 5921, got {len(flat)}"
        return flat
    except Exception:
        return None


def _extract_compiler_weights(path: str = "models/os/compiler_optimizer.pt") -> Optional[list[float]]:
    """Extract Compiler Optimizer MLP weights as flat float list (3,269 values).

    Layout:
      FC1 weight [64, 15] + FC1 bias [64] +
      FC2 weight [32, 64] + FC2 bias [32] +
      FC3 weight [5, 32] + FC3 bias [5]
    """
    p = Path(path) if Path(path).is_absolute() else MODELS_DIR.parent / path
    if not p.exists():
        return None
    try:
        import torch
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        flat: list[float] = []
        # FC1
        flat.extend(sd["net.0.weight"].flatten().tolist())   # [64, 15]
        flat.extend(sd["net.0.bias"].tolist())               # [64]
        # FC2
        flat.extend(sd["net.2.weight"].flatten().tolist())   # [32, 64]
        flat.extend(sd["net.2.bias"].tolist())               # [32]
        # FC3
        flat.extend(sd["net.4.weight"].flatten().tolist())   # [5, 32]
        flat.extend(sd["net.4.bias"].tolist())               # [5]
        assert len(flat) == 3269, f"compiler_weights: expected 3269, got {len(flat)}"
        return flat
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MetalNeuralOS — main class
# ─────────────────────────────────────────────────────────────────────────────

class MetalNeuralOS:
    """Wraps NeuralOSKernels (Rust/Metal) with automatic weight loading.

    Falls back to None if Metal is unavailable or weights can't be extracted.
    Provides the same interface signatures as the PyTorch-based NeuralGIC,
    NeuralWatchdog, and PeepholeOptimizerNet but executes entirely on Metal.
    """

    def __init__(self, kernel, has_gic: bool = False, has_watchdog: bool = False,
                 has_compiler: bool = False):
        self._kernel = kernel
        self._has_gic = has_gic
        self._has_watchdog = has_watchdog
        self._has_compiler = has_compiler

    @property
    def available(self) -> bool:
        """True if at least one model is loaded on Metal."""
        return self._kernel is not None and (
            self._has_gic or self._has_watchdog or self._has_compiler
        )

    @property
    def gic_available(self) -> bool:
        return self._has_gic and self._kernel is not None and self._kernel.gic_ready()

    @property
    def watchdog_available(self) -> bool:
        return self._has_watchdog and self._kernel is not None and self._kernel.watchdog_ready()

    @property
    def compiler_available(self) -> bool:
        return self._has_compiler and self._kernel is not None and self._kernel.compiler_ready()

    # ── GIC ────────────────────────────────────────────────────────────────────

    def gic_dispatch(self, irr: list[float], isr: list[float],
                     imr: list[float], pending: list[float]) -> list[float]:
        """Score interrupt priorities via Metal GPU.

        Args:
            irr: [32] float — interrupt request register bits
            isr: [32] float — in-service register bits
            imr: [32] float — interrupt mask register bits
            pending: [32] float — 1.0 where IRQ is pending, 0.0 otherwise

        Returns:
            [32] float priority scores (non-pending masked to -inf)
        """
        if not self.gic_available:
            raise RuntimeError("GIC weights not loaded on Metal")
        input_state = irr + isr + imr  # [96]
        return self._kernel.execute_gic(input_state, pending)

    def gic_dispatch_tensors(self, irr_tensor, isr_tensor, imr_tensor) -> int:
        """Score interrupt priorities from torch tensors, return best IRQ index.

        Convenience method that accepts the same tensor format as NeuralGIC._neural_dispatch.
        """
        if not self.gic_available:
            return -1
        irr = irr_tensor.float().tolist()
        isr = isr_tensor.float().tolist()
        imr = imr_tensor.float().tolist()
        pending = [(1.0 if (irr[i] > 0.5 and imr[i] < 0.5 and isr[i] < 0.5) else 0.0)
                   for i in range(32)]
        if sum(pending) == 0:
            return -1
        scores = self.gic_dispatch(irr, isr, imr, pending)
        return int(max(range(32), key=lambda i: scores[i]))

    # ── Watchdog ───────────────────────────────────────────────────────────────

    def watchdog_check(self, metrics_window_flat: list[float], seq_len: int) -> float:
        """Run anomaly detection on a metrics window via Metal GPU.

        Args:
            metrics_window_flat: [seq_len * 8] float — flattened metrics
            seq_len: number of timesteps

        Returns:
            Anomaly score in [0, 1] (higher = more anomalous)
        """
        if not self.watchdog_available:
            raise RuntimeError("Watchdog weights not loaded on Metal")
        return self._kernel.execute_watchdog(metrics_window_flat, seq_len)

    def watchdog_check_tensor(self, metrics_buffer) -> float:
        """Run anomaly detection from a torch tensor [window_size, 8].

        Convenience method matching NeuralWatchdog._neural_check interface.
        """
        if not self.watchdog_available:
            return 0.0
        flat = metrics_buffer.flatten().tolist()
        seq_len = metrics_buffer.shape[0]
        return self._kernel.execute_watchdog(flat, seq_len)

    # ── Compiler Optimizer ─────────────────────────────────────────────────────

    def compiler_score(self, windows_flat: list[float], n_windows: int) -> list[float]:
        """Score IR instruction windows for optimization via Metal GPU.

        Args:
            windows_flat: [N * 15] float — N windows of (3 instr x 5 features)
            n_windows: number of windows

        Returns:
            [N * 5] float optimization class scores (reshaped as flat list)
        """
        if not self.compiler_available:
            raise RuntimeError("Compiler weights not loaded on Metal")
        return self._kernel.execute_compiler(windows_flat, n_windows)

    def compiler_score_tensor(self, window_tensor) -> int:
        """Score a single window tensor [1, 15] and return best optimization class.

        Convenience method matching PeepholeOptimizerNet.forward interface.
        Returns: int optimization class (0=none, 1=const_fold, 2=strength_red,
                 3=dead_store, 4=identity)
        """
        if not self.compiler_available:
            return 0
        flat = window_tensor.flatten().tolist()
        n = len(flat) // 15
        scores = self._kernel.execute_compiler(flat, n)
        # Return argmax of first window
        if n > 0:
            row = scores[:5]
            return int(max(range(5), key=lambda i: row[i]))
        return 0

    # ── Benchmark ──────────────────────────────────────────────────────────────

    def benchmark(self) -> dict:
        """Benchmark all three kernels. Returns per-call latency in microseconds."""
        if self._kernel is None:
            return {"error": "No Metal kernel available"}
        gic_us, wd_us, co_us = self._kernel.benchmark()
        return {
            "gic_us_per_call": gic_us,
            "watchdog_us_per_call": wd_us,
            "compiler_us_per_call": co_us,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_metal_neural_os(
    load_gic: bool = True,
    load_watchdog: bool = True,
    load_compiler: bool = True,
    verbose: bool = False,
) -> MetalNeuralOS:
    """Load neural OS model weights and create a MetalNeuralOS instance.

    Loads trained .pt models from models/os/ and transfers their weights
    to Metal GPU buffers for native shader execution.

    Args:
        load_gic:      Load GIC interrupt controller model (51 KB)
        load_watchdog:  Load Watchdog anomaly detector model (4.5 KB)
        load_compiler:  Load Compiler optimizer model (16 KB)
        verbose:        Print loading progress

    Returns:
        MetalNeuralOS with .available = True if at least one model loaded,
        or .available = False if Metal / weights are unavailable.
    """
    try:
        import importlib.util
        import sys as _sys

        # Locate ncpu_metal — same strategy as metal_neural_alu.py
        ncpu_metal = None
        _so_candidates = [
            "/Users/bobbyprice/projects/.venv/lib/python3.13/site-packages/ncpu_metal/ncpu_metal.abi3.so",
            "/Users/bobbyprice/projects/nCPU/kernels/rust_metal/ncpu_metal.abi3.so",
        ]
        if "ncpu_metal" in _sys.modules and hasattr(_sys.modules["ncpu_metal"], "NeuralOSKernels"):
            ncpu_metal = _sys.modules["ncpu_metal"]
        else:
            for _so_path in _so_candidates:
                try:
                    spec = importlib.util.spec_from_file_location("ncpu_metal", _so_path)
                    if spec is not None and spec.loader is not None:
                        _m = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(_m)  # type: ignore[union-attr]
                        ncpu_metal = _m
                        _sys.modules["ncpu_metal"] = _m
                        break
                except Exception:
                    continue
            if ncpu_metal is None:
                try:
                    import ncpu_metal as _m
                    ncpu_metal = _m
                except ImportError:
                    pass

        if ncpu_metal is None or not hasattr(ncpu_metal, "NeuralOSKernels"):
            if verbose:
                print("[MetalNeuralOS] NeuralOSKernels not available")
            return MetalNeuralOS(None)

        kernel = ncpu_metal.NeuralOSKernels()

        has_gic = False
        if load_gic:
            gic_w = _extract_gic_weights()
            if gic_w is not None:
                kernel.load_gic_weights(gic_w)
                has_gic = True
                if verbose:
                    print(f"[MetalNeuralOS] Loaded GIC weights ({len(gic_w)} params) -> GPU buffer")
            elif verbose:
                print("[MetalNeuralOS] GIC model not found")

        has_watchdog = False
        if load_watchdog:
            wd_w = _extract_watchdog_weights()
            if wd_w is not None:
                kernel.load_watchdog_weights(wd_w)
                has_watchdog = True
                if verbose:
                    print(f"[MetalNeuralOS] Loaded Watchdog weights ({len(wd_w)} params) -> GPU buffer")
            elif verbose:
                print("[MetalNeuralOS] Watchdog model not found")

        has_compiler = False
        if load_compiler:
            co_w = _extract_compiler_weights()
            if co_w is not None:
                kernel.load_compiler_weights(co_w)
                has_compiler = True
                if verbose:
                    print(f"[MetalNeuralOS] Loaded Compiler weights ({len(co_w)} params) -> GPU buffer")
            elif verbose:
                print("[MetalNeuralOS] Compiler optimizer model not found")

        result = MetalNeuralOS(kernel, has_gic, has_watchdog, has_compiler)
        if verbose:
            loaded = []
            if has_gic: loaded.append("GIC")
            if has_watchdog: loaded.append("Watchdog")
            if has_compiler: loaded.append("Compiler")
            print(f"[MetalNeuralOS] Ready: {', '.join(loaded) if loaded else 'none'}")
        return result

    except Exception as e:
        if verbose:
            print(f"[MetalNeuralOS] Failed to initialize: {e}")
        return MetalNeuralOS(None)
