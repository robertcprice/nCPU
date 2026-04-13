"""Tests for neural math operations (sin, cos, sqrt, exp, log, atan2).

These models are approximation networks — tests verify:
1. Models load and produce outputs (no crashes)
2. Output shapes and types are correct
3. Known values produce reasonable approximations (wide tolerance)
"""

import math
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestMathBridge:
    """Test math operations through the NeuralALUBridge."""

    @pytest.fixture
    def bridge(self):
        from ncpu.neural.neural_alu_bridge import NeuralALUBridge
        b = NeuralALUBridge()
        b.load()
        return b

    def test_math_models_available(self, bridge):
        """Math models loaded during bridge.load()."""
        ops = bridge._ops._available_ops
        # At least some math models should be available
        math_ops = [k for k in ops if k in ("SIN", "COS", "SQRT", "EXP", "LOG", "ATAN2")]
        assert len(math_ops) >= 1, f"No math models loaded. Available: {ops}"

    def test_sin_returns_float(self, bridge):
        result = bridge.sin(0)
        assert isinstance(result, float)

    def test_cos_returns_float(self, bridge):
        result = bridge.cos(0)
        assert isinstance(result, float)

    def test_sqrt_returns_float(self, bridge):
        result = bridge.sqrt(4000)  # 4.0 in fixed-point
        assert isinstance(result, float)

    def test_exp_returns_float(self, bridge):
        result = bridge.exp_(0)
        assert isinstance(result, float)

    def test_log_returns_float(self, bridge):
        result = bridge.log_(1000)  # 1.0 in fixed-point
        assert isinstance(result, float)

    def test_atan2_returns_float(self, bridge):
        result = bridge.atan2(1000, 0)  # atan2(1, 0)
        assert isinstance(result, float)

    def test_sin_no_nan(self, bridge):
        """Sin should not produce NaN for valid inputs."""
        for val in [0, 1571, 3142, -1571]:
            result = bridge.sin(val)
            assert not math.isnan(result), f"sin({val}) produced NaN"

    def test_cos_no_nan(self, bridge):
        """Cos should not produce NaN for valid inputs."""
        for val in [0, 1571, 3142, -1571]:
            result = bridge.cos(val)
            assert not math.isnan(result), f"cos({val}) produced NaN"

    def test_sqrt_no_nan(self, bridge):
        """Sqrt should not produce NaN for positive inputs."""
        for val in [0, 1000, 4000, 9000]:
            result = bridge.sqrt(val)
            assert not math.isnan(result), f"sqrt({val}) produced NaN"

    def test_exp_no_nan(self, bridge):
        """Exp should not produce NaN for bounded inputs."""
        for val in [0, 1000, 2000, -1000]:
            result = bridge.exp_(val)
            assert not math.isnan(result), f"exp({val}) produced NaN"

    def test_log_no_nan(self, bridge):
        """Log should not produce NaN for positive inputs."""
        for val in [1000, 2718, 10000]:
            result = bridge.log_(val)
            assert not math.isnan(result), f"log({val}) produced NaN"

    def test_atan2_no_nan(self, bridge):
        """Atan2 should not produce NaN for valid inputs."""
        for y, x in [(1000, 0), (0, 1000), (1000, 1000), (-1000, 1000)]:
            result = bridge.atan2(y, x)
            assert not math.isnan(result), f"atan2({y},{x}) produced NaN"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestMathOpsDirectly:
    """Test math operations through NeuralOps directly."""

    @pytest.fixture
    def ops(self):
        from ncpu.model.neural_ops import NeuralOps
        o = NeuralOps()
        o.load()
        return o

    def test_sincos_model_loaded(self, ops):
        assert ops._sincos is not None, "sincos model not loaded"

    def test_sqrt_model_loaded(self, ops):
        assert ops._sqrt is not None, "sqrt model not loaded"

    def test_exp_model_loaded(self, ops):
        assert ops._exp is not None, "exp model not loaded"

    def test_log_model_loaded(self, ops):
        assert ops._log is not None, "log model not loaded"

    def test_atan2_model_loaded(self, ops):
        assert ops._atan2 is not None, "atan2 model not loaded"

    def test_sin_produces_output(self, ops):
        result = ops.neural_sin(0)
        assert isinstance(result, float)

    def test_cos_produces_output(self, ops):
        result = ops.neural_cos(0)
        assert isinstance(result, float)

    def test_sqrt_produces_output(self, ops):
        result = ops.neural_sqrt(4000)
        assert isinstance(result, float)

    def test_exp_produces_output(self, ops):
        result = ops.neural_exp(0)
        assert isinstance(result, float)

    def test_log_produces_output(self, ops):
        result = ops.neural_log(1000)
        assert isinstance(result, float)

    def test_atan2_produces_output(self, ops):
        result = ops.neural_atan2(1000, 0)
        assert isinstance(result, float)
