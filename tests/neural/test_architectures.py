"""Tests that every reconstructed nn.Module architecture loads its trained .pt checkpoint
with strict=True -- zero missing keys, zero unexpected keys.

Organized by model category: register, memory, decoder, math.
"""

from pathlib import Path

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")

# Lazy imports guarded by the module-level skip
if TORCH_AVAILABLE:
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
# Helpers
# ---------------------------------------------------------------------------

def _load_strict(model, path, *, weights_only=True):
    """Load a state_dict into *model* with strict=True and return the result."""
    state = torch.load(path, map_location="cpu", weights_only=weights_only)
    result = model.load_state_dict(state, strict=True)
    return result


def _load_dict_strict(model, path, *, state_key="model"):
    """Load a dict-wrapped checkpoint, extract state_dict, and load strictly."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert state_key in data, f"Missing key '{state_key}' in checkpoint"
    result = model.load_state_dict(data[state_key], strict=True)
    return result


def _assert_no_key_errors(result):
    """Assert that load_state_dict returned no missing or unexpected keys."""
    assert len(result.missing_keys) == 0, f"Missing keys: {result.missing_keys}"
    assert len(result.unexpected_keys) == 0, f"Unexpected keys: {result.unexpected_keys}"


# ===========================================================================
# Register
# ===========================================================================


class TestRegisterArchitectures:
    """Tests for register-related model architectures."""

    def test_register_file_loads(self):
        model = NeuralRegisterFile()
        result = _load_strict(model, MODELS_DIR / "register" / "register_file.pt")
        _assert_no_key_errors(result)

    def test_register_file_parameter_count(self):
        model = NeuralRegisterFile()
        state = torch.load(
            MODELS_DIR / "register" / "register_file.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        # Every tensor in the state dict should be accounted for
        assert len(list(model.parameters())) + len(list(model.buffers())) > 0


# ===========================================================================
# Memory
# ===========================================================================


class TestMemoryArchitectures:
    """Tests for memory-related model architectures (stack, pointer, function_call)."""

    def test_stack_loads(self):
        model = NeuralStack()
        result = _load_strict(model, MODELS_DIR / "memory" / "stack.pt")
        _assert_no_key_errors(result)

    def test_pointer_loads(self):
        model = NeuralPointer()
        result = _load_strict(model, MODELS_DIR / "memory" / "pointer.pt")
        _assert_no_key_errors(result)

    def test_function_call_loads(self):
        model = NeuralFunctionCall()
        result = _load_strict(model, MODELS_DIR / "memory" / "function_call.pt")
        _assert_no_key_errors(result)

    def test_stack_has_op_net(self):
        """NeuralStack should have an op_net that NeuralPointer does not."""
        stack = NeuralStack()
        pointer = NeuralPointer()
        assert hasattr(stack, "op_net")
        assert not hasattr(pointer, "op_net")

    def test_shared_addr_arith_structure(self):
        """Stack, Pointer, and FunctionCall all share the same addr_arith submodule."""
        for cls in (NeuralStack, NeuralPointer, NeuralFunctionCall):
            model = cls()
            assert hasattr(model, "addr_arith")
            assert hasattr(model.addr_arith, "full_adder")
            assert hasattr(model.addr_arith.full_adder, "net")


# ===========================================================================
# Decoder
# ===========================================================================


class TestDecoderArchitectures:
    """Tests for the ARM64 instruction decoder architecture."""

    def test_arm64_decoder_loads(self):
        model = NeuralARM64Decoder()
        result = _load_strict(model, MODELS_DIR / "decoder" / "arm64_decoder.pt")
        _assert_no_key_errors(result)

    def test_decoder_head_output_dims(self):
        """Verify the decoder head output dimensions match ARM64 semantics."""
        model = NeuralARM64Decoder()
        state = torch.load(
            MODELS_DIR / "decoder" / "arm64_decoder.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        # category_head final layer outputs 10 categories
        assert model.decoder_head.category_head[-1].out_features == 10
        # rd/rn/rm heads output 32 (one-hot register index)
        assert model.decoder_head.rd_head.out_features == 32
        assert model.decoder_head.rn_head.out_features == 32
        assert model.decoder_head.rm_head.out_features == 32
        # operation_head final layer outputs 128
        assert model.decoder_head.operation_head[-1].out_features == 128

    def test_decoder_field_queries_shape(self):
        """The field extractor should have 6 learned query vectors of dim 256."""
        model = NeuralARM64Decoder()
        assert model.field_extractor.field_queries.shape == (6, 256)


# ===========================================================================
# Math
# ===========================================================================


class TestMathArchitectures:
    """Tests for math function approximation models."""

    def test_atan2_loads(self):
        model = NeuralAtan2()
        result = _load_strict(model, MODELS_DIR / "math" / "atan2.pt")
        _assert_no_key_errors(result)

    def test_sincos_loads(self):
        model = NeuralSinCos()
        result = _load_dict_strict(model, MODELS_DIR / "math" / "sincos.pt")
        _assert_no_key_errors(result)

    def test_sqrt_loads(self):
        model = NeuralSqrt()
        result = _load_dict_strict(model, MODELS_DIR / "math" / "sqrt.pt")
        _assert_no_key_errors(result)

    def test_exp_loads(self):
        model = NeuralExp()
        result = _load_dict_strict(model, MODELS_DIR / "math" / "exp.pt")
        _assert_no_key_errors(result)

    def test_log_loads(self):
        model = NeuralLog()
        result = _load_dict_strict(model, MODELS_DIR / "math" / "log.pt")
        _assert_no_key_errors(result)

    def test_doom_trig_loads(self):
        model = DoomTrigLUT()
        data = torch.load(
            MODELS_DIR / "math" / "doom_trig.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_from_dict(data)
        assert model.n_angles == 8192
        assert model.sine_table.shape == (8192,)
        assert model.cosine_table.shape == (8192,)
        assert model.format_str != ""

    def test_sincos_checkpoint_metadata(self):
        """sincos.pt should contain max_err and epoch alongside the state dict."""
        data = torch.load(
            MODELS_DIR / "math" / "sincos.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert "max_err" in data
        assert "epoch" in data

    def test_sqrt_checkpoint_metadata(self):
        """sqrt.pt should contain rel_err, abs_err, and epoch."""
        data = torch.load(
            MODELS_DIR / "math" / "sqrt.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert "rel_err" in data
        assert "abs_err" in data
        assert "epoch" in data

    def test_exp_log_checkpoint_metadata(self):
        """exp.pt and log.pt should contain an error metric."""
        for name in ("exp", "log"):
            data = torch.load(
                MODELS_DIR / "math" / f"{name}.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert "error" in data, f"{name}.pt missing 'error' key"

    def test_atan2_batchnorm_no_running_stats(self):
        """Atan2 BatchNorm layers should have track_running_stats=False."""
        model = NeuralAtan2()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                assert not module.track_running_stats, (
                    f"{name} has track_running_stats=True but checkpoint has none"
                )

    def test_sqrt_batchnorm_no_running_stats(self):
        """Sqrt BatchNorm layers should have track_running_stats=False."""
        model = NeuralSqrt()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                assert not module.track_running_stats, (
                    f"{name} has track_running_stats=True but checkpoint has none"
                )
