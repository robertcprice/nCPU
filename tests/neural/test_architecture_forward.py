"""Forward-pass smoke tests for all 11 reconstructed nCPU model architectures.

Tests instantiate each model, load trained weights (strict=True where applicable),
set to eval mode, feed representative input, and assert:
    - Output shape matches expected dimensions
    - Output dtype is float32
    - No NaN or Inf values in output

Models are loaded from the trained checkpoints under models/.
Tests are skipped when torch is unavailable or checkpoint files are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip the entire module when torch is not installed
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")

from ncpu.model.neural_ops import (
    NeuralCompare,
    NeuralFullAdder,
    NeuralLogical,
    NeuralMultiplierLUT,
    NeuralShiftNet,
)
from ncpu.model.architectures import (
    DoomTrigLUT,
    NeuralARM64Decoder,
    NeuralAtan2,
    NeuralExp,
    NeuralFunctionCall,
    NeuralLog,
    NeuralPointer,
    NeuralRegisterFile,
    NeuralSinCos,
    NeuralSqrt,
    NeuralStack,
)

# ---------------------------------------------------------------------------
# Resolve model directory from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def _load_weights(path: Path, *, weights_only: bool = False) -> dict:
    """Load a checkpoint from disk, raising FileNotFoundError when missing."""
    if not path.exists():
        pytest.skip(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def _assert_valid_output(tensor: torch.Tensor) -> None:
    """Assert the tensor contains finite float32 values."""
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    assert not torch.isnan(tensor).any(), "Output contains NaN"
    assert not torch.isinf(tensor).any(), "Output contains Inf"


# ===================================================================
# ALU models (from ncpu.model.neural_ops)
# ===================================================================


class TestNeuralFullAdderArithmetic:
    """arithmetic.pt -- NeuralFullAdder(hidden_dim=128): bit-serial full adder."""

    CKPT = MODELS_DIR / "alu" / "arithmetic.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralFullAdder(hidden_dim=128)
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.tensor([[1.0, 0.0, 0.0]])  # (bit_a, bit_b, carry_in)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"
        _assert_valid_output(out)

    def test_batch_forward(self, model):
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)
        _assert_valid_output(out)

    def test_output_in_sigmoid_range(self, model):
        x = torch.tensor([[1.0, 0.0, 1.0]])
        with torch.no_grad():
            out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all(), "Sigmoid output out of [0, 1]"


class TestNeuralFullAdderDivide:
    """divide.pt -- NeuralFullAdder(hidden_dim=64): division adder."""

    CKPT = MODELS_DIR / "alu" / "divide.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralFullAdder(hidden_dim=64)
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.tensor([[0.0, 1.0, 1.0]])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)
        _assert_valid_output(out)

    def test_output_in_sigmoid_range(self, model):
        x = torch.tensor([[1.0, 1.0, 0.0]])
        with torch.no_grad():
            out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()


class TestNeuralMultiplierLUT:
    """multiply.pt -- NeuralMultiplierLUT: 256x256x16 byte-pair lookup table."""

    CKPT = MODELS_DIR / "alu" / "multiply.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralMultiplierLUT()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_lut_table_shape(self, model):
        assert model.lut.table.shape == (256, 256, 16)

    def test_lookup_zero(self, model):
        result = model.lookup(0, 0)
        assert result == 0

    def test_lookup_identity(self, model):
        result = model.lookup(1, 1)
        assert result == 1

    def test_lookup_known_product(self, model):
        result = model.lookup(7, 8)
        assert result == 56


class TestNeuralCompare:
    """compare.pt -- NeuralCompare: 3-feature refinement layer."""

    CKPT = MODELS_DIR / "alu" / "compare.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralCompare()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.tensor([[1.0, 0.0, 0.0]])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3)
        _assert_valid_output(out)

    def test_output_in_sigmoid_range(self, model):
        x = torch.tensor([[0.0, 1.0, 0.0]])
        with torch.no_grad():
            out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()


class TestNeuralLogical:
    """logical.pt -- NeuralLogical: 7x4 truth table parameter."""

    CKPT = MODELS_DIR / "alu" / "logical.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralLogical()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_truth_table_shape(self, model):
        assert model.truth_tables.shape == (7, 4)

    def test_apply_and(self, model):
        assert model.apply_op(0, 1, 1) == 1  # AND(1,1) = 1
        assert model.apply_op(0, 1, 0) == 0  # AND(1,0) = 0

    def test_apply_or(self, model):
        assert model.apply_op(1, 0, 1) == 1  # OR(0,1) = 1
        assert model.apply_op(1, 0, 0) == 0  # OR(0,0) = 0

    def test_apply_xor(self, model):
        assert model.apply_op(2, 1, 0) == 1  # XOR(1,0) = 1
        assert model.apply_op(2, 1, 1) == 0  # XOR(1,1) = 0


class TestNeuralShiftNetLSL:
    """lsl.pt -- NeuralShiftNet: trained left shift network."""

    CKPT = MODELS_DIR / "shifts" / "lsl.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralShiftNet()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        value_bits = torch.zeros(64)
        value_bits[0] = 1.0  # value = 1
        shift_bits = torch.zeros(64)
        shift_bits[0] = 1.0  # shift by 1
        out = model.forward(value_bits, shift_bits)
        assert out.shape == (64,)
        _assert_valid_output(out)

    def test_temperature_parameter(self, model):
        assert hasattr(model, "temperature")
        assert model.temperature.shape == ()


class TestNeuralShiftNetLSR:
    """lsr.pt -- NeuralShiftNet: trained right shift network."""

    CKPT = MODELS_DIR / "shifts" / "lsr.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralShiftNet()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        value_bits = torch.zeros(64)
        value_bits[3] = 1.0  # value = 8
        shift_bits = torch.zeros(64)
        shift_bits[1] = 1.0  # shift by 2
        out = model.forward(value_bits, shift_bits)
        assert out.shape == (64,)
        _assert_valid_output(out)


# ===================================================================
# Register, Memory, Decoder models (from ncpu.model.architectures)
# ===================================================================


class TestNeuralRegisterFile:
    """register_file.pt -- NeuralRegisterFile: ARM64 register file emulation."""

    CKPT = MODELS_DIR / "register" / "register_file.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralRegisterFile()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.randn(1, 5)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 32)
        _assert_valid_output(out)

    def test_batch_forward(self, model):
        x = torch.randn(4, 5)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 32)
        _assert_valid_output(out)


class TestNeuralStack:
    """stack.pt -- NeuralStack: neural stack push/pop operations."""

    CKPT = MODELS_DIR / "memory" / "stack.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralStack()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.randn(1, 65)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)
        _assert_valid_output(out)


class TestNeuralPointer:
    """pointer.pt -- NeuralPointer: neural pointer dereference via _MemAddr."""

    CKPT = MODELS_DIR / "memory" / "pointer.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralPointer()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.randn(1, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 256)
        _assert_valid_output(out)


class TestNeuralFunctionCall:
    """function_call.pt -- NeuralFunctionCall: BL/RET branch target selection."""

    CKPT = MODELS_DIR / "memory" / "function_call.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralFunctionCall()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.randn(1, 64)
        with torch.no_grad():
            out = model(x)
        # target_selector: Linear(64,128) -> ReLU -> Linear(128,64) => output dim 64
        assert out.shape == (1, 64)
        _assert_valid_output(out)


class TestNeuralARM64Decoder:
    """arm64_decoder.pt -- NeuralARM64Decoder: transformer-based instruction decoder."""

    CKPT = MODELS_DIR / "decoder" / "arm64_decoder.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=True)
        m = NeuralARM64Decoder()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        # Embedding layer requires LongTensor with values in {0, 1}
        x = torch.randint(0, 2, (1, 32), dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        # FieldExtractor returns [batch, 6_fields, 256_dim]
        assert out.shape == (1, 6, 256), f"Expected (1, 6, 256), got {out.shape}"
        _assert_valid_output(out)

    def test_batch_forward(self, model):
        x = torch.randint(0, 2, (4, 32), dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 6, 256)
        _assert_valid_output(out)


# ===================================================================
# Math models (from ncpu.model.architectures)
# ===================================================================


class TestNeuralSinCos:
    """sincos.pt -- NeuralSinCos: sine-activated network for sin/cos approximation.

    Checkpoint format: {model: state_dict, ...}
    """

    CKPT = MODELS_DIR / "math" / "sincos.pt"

    @pytest.fixture
    def model(self):
        ckpt = _load_weights(self.CKPT, weights_only=False)
        m = NeuralSinCos()
        m.load_state_dict(ckpt["model"], strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.tensor([[1.57]])  # ~pi/2
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)
        _assert_valid_output(out)

    def test_zero_input(self, model):
        x = torch.tensor([[0.0]])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)
        _assert_valid_output(out)


class TestNeuralSqrt:
    """sqrt.pt -- NeuralSqrt: two-stage Newton-style sqrt approximation.

    Uses BatchNorm(track_running_stats=False) -- requires batch_size >= 2 in eval mode.
    Checkpoint format: {model: state_dict, ...}
    """

    CKPT = MODELS_DIR / "math" / "sqrt.pt"

    @pytest.fixture
    def model(self):
        ckpt = _load_weights(self.CKPT, weights_only=False)
        m = NeuralSqrt()
        m.load_state_dict(ckpt["model"], strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.tensor([[4.0], [9.0]])  # batch >= 2 for BatchNorm
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1)
        _assert_valid_output(out)

    def test_positive_inputs(self, model):
        x = torch.tensor([[1.0], [16.0], [25.0], [100.0]])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)
        _assert_valid_output(out)


class TestNeuralExp:
    """exp.pt -- NeuralExp: four-layer MLP for exponential approximation.

    Checkpoint format: {model: state_dict, ...}
    """

    CKPT = MODELS_DIR / "math" / "exp.pt"

    @pytest.fixture
    def model(self):
        ckpt = _load_weights(self.CKPT, weights_only=False)
        m = NeuralExp()
        m.load_state_dict(ckpt["model"], strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.tensor([[1.0]])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)
        _assert_valid_output(out)

    def test_zero_input(self, model):
        x = torch.tensor([[0.0]])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)
        _assert_valid_output(out)


class TestNeuralLog:
    """log.pt -- NeuralLog: four-layer MLP for logarithm approximation.

    Checkpoint format: {model: state_dict, ...}
    """

    CKPT = MODELS_DIR / "math" / "log.pt"

    @pytest.fixture
    def model(self):
        ckpt = _load_weights(self.CKPT, weights_only=False)
        m = NeuralLog()
        m.load_state_dict(ckpt["model"], strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.tensor([[1.0]])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)
        _assert_valid_output(out)

    def test_positive_input(self, model):
        x = torch.tensor([[2.718]])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)
        _assert_valid_output(out)


class TestNeuralAtan2:
    """atan2.pt -- NeuralAtan2: deep BatchNorm network for atan2 approximation.

    Uses BatchNorm(track_running_stats=False) -- requires batch_size >= 2 in eval mode.
    Checkpoint format: direct state_dict (not wrapped in a dict).
    """

    CKPT = MODELS_DIR / "math" / "atan2.pt"

    @pytest.fixture
    def model(self):
        state = _load_weights(self.CKPT, weights_only=False)
        m = NeuralAtan2()
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def test_forward_shape(self, model):
        # Input: [batch, 6] = (sin_a, cos_a, q1, q2, q3, q4)
        x = torch.tensor([
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)
        _assert_valid_output(out)

    def test_larger_batch(self, model):
        x = torch.randn(8, 6)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 2)
        _assert_valid_output(out)


class TestDoomTrigLUT:
    """doom_trig.pt -- DoomTrigLUT: 8192-entry fixed-point sine/cosine tables.

    Checkpoint format: plain dict with tensors (not a state_dict).
    Loaded via load_from_dict() instead of load_state_dict().
    """

    CKPT = MODELS_DIR / "math" / "doom_trig.pt"

    @pytest.fixture
    def model(self):
        data = _load_weights(self.CKPT, weights_only=False)
        m = DoomTrigLUT()
        m.load_from_dict(data)
        m.eval()
        return m

    def test_table_sizes(self, model):
        assert model.sine_table.shape == (8192,)
        assert model.cosine_table.shape == (8192,)
        assert model.n_angles == 8192

    def test_forward_single(self, model):
        idx = torch.tensor([0])
        out = model(idx)
        assert out.shape == (1,)
        _assert_valid_output(out)

    def test_forward_batch(self, model):
        idx = torch.tensor([0, 1024, 2048, 4096, 8191])
        out = model(idx)
        assert out.shape == (5,)
        _assert_valid_output(out)

    def test_forward_wraps_at_n_angles(self, model):
        # Index >= 8192 should wrap via modulo
        idx_a = torch.tensor([0])
        idx_b = torch.tensor([8192])
        out_a = model(idx_a)
        out_b = model(idx_b)
        assert torch.equal(out_a, out_b), "Modulo wrap not working"
