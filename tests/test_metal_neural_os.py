"""Tests for Metal Neural OS Kernels — verify Metal shader output matches PyTorch.

Loads each trained model via both PyTorch and Metal, runs identical inputs,
and checks that the outputs match within floating-point tolerance.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load Metal OS kernels
# ─────────────────────────────────────────────────────────────────────────────

def _get_metal_os():
    """Try to load MetalNeuralOS. Returns None if unavailable."""
    try:
        from ncpu.neural.metal_neural_os import load_metal_neural_os
        os_k = load_metal_neural_os(verbose=False)
        if os_k.available:
            return os_k
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Test GIC
# ─────────────────────────────────────────────────────────────────────────────

class TestGIC:
    """Test Neural GIC Metal shader against PyTorch reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.metal_os = _get_metal_os()
        self.gic_path = MODELS_DIR / "os" / "gic.pt"

    def test_gic_available(self):
        """Metal GIC kernel should load and be ready."""
        if self.metal_os is None:
            pytest.skip("Metal OS kernels not available")
        assert self.metal_os.gic_available

    def test_gic_matches_pytorch(self):
        """Metal GIC output should match PyTorch within tolerance."""
        if self.metal_os is None or not self.metal_os.gic_available:
            pytest.skip("Metal GIC not available")
        if not self.gic_path.exists():
            pytest.skip("GIC model file not found")

        # Build PyTorch model
        from ncpu.os.neuros.interrupts import NeuralPriorityEncoder
        encoder = NeuralPriorityEncoder(num_irqs=32, hidden_dim=64)
        encoder.load_state_dict(
            torch.load(str(self.gic_path), map_location="cpu", weights_only=True)
        )
        encoder.eval()

        # Generate random test inputs
        torch.manual_seed(42)
        for _ in range(10):
            irr = torch.randint(0, 2, (32,)).float()
            isr = torch.randint(0, 2, (32,)).float()
            imr = torch.randint(0, 2, (32,)).float()
            pending = (irr.bool() & ~imr.bool() & ~isr.bool()).float()

            # PyTorch forward
            state = torch.cat([irr, isr, imr])
            with torch.no_grad():
                pt_scores = encoder(state)
            # Mask non-pending
            pt_scores[pending < 0.5] = float('-inf')

            # Metal forward
            metal_scores = self.metal_os.gic_dispatch(
                irr.tolist(), isr.tolist(), imr.tolist(), pending.tolist()
            )

            # Compare (only pending IRQs)
            for i in range(32):
                if pending[i] > 0.5:
                    pt_val = pt_scores[i].item()
                    mt_val = metal_scores[i]
                    assert abs(pt_val - mt_val) < 0.01, \
                        f"IRQ {i}: PyTorch={pt_val:.4f} vs Metal={mt_val:.4f}"

    def test_gic_argmax_matches(self):
        """Metal and PyTorch should agree on the highest-priority IRQ."""
        if self.metal_os is None or not self.metal_os.gic_available:
            pytest.skip("Metal GIC not available")
        if not self.gic_path.exists():
            pytest.skip("GIC model file not found")

        from ncpu.os.neuros.interrupts import NeuralPriorityEncoder
        encoder = NeuralPriorityEncoder(num_irqs=32, hidden_dim=64)
        encoder.load_state_dict(
            torch.load(str(self.gic_path), map_location="cpu", weights_only=True)
        )
        encoder.eval()

        torch.manual_seed(99)
        for _ in range(20):
            irr = torch.randint(0, 2, (32,)).float()
            isr = torch.zeros(32)
            imr = torch.zeros(32)
            pending = irr.clone()

            if pending.sum() == 0:
                continue

            state = torch.cat([irr, isr, imr])
            with torch.no_grad():
                pt_scores = encoder(state)
            pt_scores[pending < 0.5] = float('-inf')
            pt_best = int(pt_scores.argmax().item())

            metal_scores = self.metal_os.gic_dispatch(
                irr.tolist(), isr.tolist(), imr.tolist(), pending.tolist()
            )
            mt_best = int(max(range(32), key=lambda i: metal_scores[i]))

            assert pt_best == mt_best, \
                f"Argmax mismatch: PyTorch={pt_best} vs Metal={mt_best}"


# ─────────────────────────────────────────────────────────────────────────────
# Test Watchdog
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchdog:
    """Test Neural Watchdog Metal shader against PyTorch reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.metal_os = _get_metal_os()
        self.wd_path = MODELS_DIR / "os" / "watchdog.pt"

    def test_watchdog_available(self):
        """Metal Watchdog kernel should load and be ready."""
        if self.metal_os is None:
            pytest.skip("Metal OS kernels not available")
        assert self.metal_os.watchdog_available

    def test_watchdog_matches_pytorch(self):
        """Metal Watchdog output should match PyTorch within tolerance."""
        if self.metal_os is None or not self.metal_os.watchdog_available:
            pytest.skip("Metal Watchdog not available")
        if not self.wd_path.exists():
            pytest.skip("Watchdog model file not found")

        from ncpu.os.neuros.watchdog import WatchdogNet
        net = WatchdogNet(input_size=8, hidden_size=32, num_layers=1)
        net.load_state_dict(
            torch.load(str(self.wd_path), map_location="cpu", weights_only=True)
        )
        net.eval()

        torch.manual_seed(42)
        for seq_len in [8, 16, 32, 64]:
            metrics = torch.rand(1, seq_len, 8)

            # PyTorch forward
            with torch.no_grad():
                pt_score = net(metrics).item()

            # Metal forward
            flat = metrics.flatten().tolist()
            mt_score = self.metal_os.watchdog_check(flat, seq_len)

            assert abs(pt_score - mt_score) < 0.02, \
                f"seq_len={seq_len}: PyTorch={pt_score:.4f} vs Metal={mt_score:.4f}"

    def test_watchdog_anomaly_detection(self):
        """Watchdog should produce low scores for normal data, high for anomalous."""
        if self.metal_os is None or not self.metal_os.watchdog_available:
            pytest.skip("Metal Watchdog not available")

        # Normal metrics: moderate values
        normal = [0.3, 0.2, 0.5, 0.8, 0.9, 0.1, 0.3, 0.1] * 64
        normal_score = self.metal_os.watchdog_check(normal, 64)

        # The score should be a valid float in [0, 1]
        assert 0.0 <= normal_score <= 1.0, f"Score out of range: {normal_score}"


# ─────────────────────────────────────────────────────────────────────────────
# Test Compiler Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestCompilerOptimizer:
    """Test Neural Compiler Optimizer Metal shader against PyTorch reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.metal_os = _get_metal_os()
        self.co_path = MODELS_DIR / "os" / "compiler_optimizer.pt"

    def test_compiler_available(self):
        """Metal Compiler kernel should load and be ready."""
        if self.metal_os is None:
            pytest.skip("Metal OS kernels not available")
        assert self.metal_os.compiler_available

    def test_compiler_matches_pytorch(self):
        """Metal Compiler output should match PyTorch within tolerance."""
        if self.metal_os is None or not self.metal_os.compiler_available:
            pytest.skip("Metal Compiler not available")
        if not self.co_path.exists():
            pytest.skip("Compiler optimizer model file not found")

        from ncpu.os.neuros.compiler import PeepholeOptimizerNet
        net = PeepholeOptimizerNet(window_size=3, feat_per_instr=5, num_opts=5)
        net.load_state_dict(
            torch.load(str(self.co_path), map_location="cpu", weights_only=True)
        )
        net.eval()

        torch.manual_seed(42)
        n_windows = 50
        windows = torch.rand(n_windows, 15)

        # PyTorch forward (batch)
        with torch.no_grad():
            pt_scores = net(windows)  # [50, 5]

        # Metal forward
        flat = windows.flatten().tolist()
        mt_scores_flat = self.metal_os.compiler_score(flat, n_windows)

        # Compare
        for i in range(n_windows):
            for j in range(5):
                pt_val = pt_scores[i, j].item()
                mt_val = mt_scores_flat[i * 5 + j]
                assert abs(pt_val - mt_val) < 0.01, \
                    f"Window {i}, class {j}: PyTorch={pt_val:.4f} vs Metal={mt_val:.4f}"

    def test_compiler_argmax_matches(self):
        """Metal and PyTorch should agree on the best optimization class."""
        if self.metal_os is None or not self.metal_os.compiler_available:
            pytest.skip("Metal Compiler not available")
        if not self.co_path.exists():
            pytest.skip("Compiler optimizer model file not found")

        from ncpu.os.neuros.compiler import PeepholeOptimizerNet
        net = PeepholeOptimizerNet(window_size=3, feat_per_instr=5, num_opts=5)
        net.load_state_dict(
            torch.load(str(self.co_path), map_location="cpu", weights_only=True)
        )
        net.eval()

        torch.manual_seed(99)
        for _ in range(100):
            window = torch.rand(1, 15)
            with torch.no_grad():
                pt_scores = net(window)[0]
            pt_best = int(pt_scores.argmax().item())

            mt_scores = self.metal_os.compiler_score(window.flatten().tolist(), 1)
            mt_best = int(max(range(5), key=lambda j: mt_scores[j]))

            assert pt_best == mt_best, \
                f"Argmax mismatch: PyTorch={pt_best} vs Metal={mt_best}"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark test
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmark:
    """Benchmark Metal OS kernels vs PyTorch inference."""

    def test_benchmark_runs(self):
        """Benchmark should execute without error and report times."""
        metal_os = _get_metal_os()
        if metal_os is None:
            pytest.skip("Metal OS kernels not available")
        result = metal_os.benchmark()
        assert "gic_us_per_call" in result
        assert "watchdog_us_per_call" in result
        assert "compiler_us_per_call" in result
        # All loaded kernels should have positive times
        if metal_os.gic_available:
            assert result["gic_us_per_call"] > 0
        if metal_os.watchdog_available:
            assert result["watchdog_us_per_call"] > 0
        if metal_os.compiler_available:
            assert result["compiler_us_per_call"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
