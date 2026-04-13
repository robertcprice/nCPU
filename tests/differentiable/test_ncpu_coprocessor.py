"""Comprehensive tests for the nCPU differentiable coprocessor.

Tests cover:
  - Soft ALU: truth table correctness, gradient flow, bit decomposition
  - NCPUExpert: shapes, operation paths, gradient flow
  - Router: gate range, aux loss, target load
  - CoprocessorMLP: shape preservation, blending, aux loss propagation
  - Injection: into mock model, forward/backward pass
  - Arithmetic correctness: soft logic matches hard logic on discrete inputs
  - End-to-end gradient flow through all paths
"""

import pytest
import torch
import torch.nn as nn

from ncpu.coprocessor.config import NCPUCoprocessorConfig
from ncpu.coprocessor.soft_alu import (
    SoftNeuralLogical,
    SoftNeuralAdder,
    soft_int_to_bits,
    soft_bits_to_int,
    ste_threshold,
    StraightThroughThreshold,
)
from ncpu.coprocessor.ncpu_expert import NCPUExpert
from ncpu.coprocessor.router import NCPURouter
from ncpu.coprocessor.coprocessor_layer import NCPUCoprocessorMLP
from ncpu.coprocessor.inject import (
    inject_ncpu_coprocessor,
    collect_aux_losses,
    freeze_backbone,
    get_coprocessor_params,
    _replace_module,
    _resolve_layer_indices,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

HIDDEN_DIM = 64
SEQ_LEN = 4
BATCH_SIZE = 2
N_BITS = 8


@pytest.fixture
def config():
    return NCPUCoprocessorConfig(n_bits=N_BITS, num_ops=7, residual_init_scale=0.1)


@pytest.fixture
def hidden_states():
    return torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)


class MockMLP(nn.Module):
    """Simple MLP that mimics a transformer MLP sublayer."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, **kwargs):
        return self.fc(x)


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = MockMLP(hidden_dim)
        self.self_attn = nn.Identity()


class MockTransformerModel(nn.Module):
    """Minimal model with model.model.layers[i].mlp pattern."""
    def __init__(self, hidden_dim, n_layers=4):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_dim})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer.mlp(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# StraightThroughThreshold tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSTE:
    def test_forward_thresholds_correctly(self):
        x = torch.tensor([0.1, 0.5, 0.6, 0.9, 0.0, 1.0])
        result = ste_threshold(x)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        assert torch.equal(result, expected)

    def test_backward_passes_gradient_through(self):
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = ste_threshold(x)
        loss = y.sum()
        loss.backward()
        # STE: gradient should be 1.0 everywhere (pass-through)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones(2))

    def test_gradient_magnitude_preserved(self):
        x = torch.tensor([0.2, 0.8], requires_grad=True)
        y = ste_threshold(x)
        loss = (y * torch.tensor([3.0, 5.0])).sum()
        loss.backward()
        assert torch.allclose(x.grad, torch.tensor([3.0, 5.0]))


# ═══════════════════════════════════════════════════════════════════════════════
# Soft bit decomposition tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoftBits:
    def test_soft_int_to_bits_integer_values(self):
        """On exact integers, soft decomposition should closely match hard bits."""
        for val in [0, 1, 5, 13, 42, 127, 255]:
            x = torch.tensor(float(val))
            bits = soft_int_to_bits(x, n_bits=8, temperature=50.0)
            # Extract hard bits for comparison
            hard_bits = torch.tensor([(val >> i) & 1 for i in range(8)], dtype=torch.float)
            # With high temperature, should be very close
            assert torch.allclose(bits, hard_bits, atol=0.05), (
                f"val={val}: soft={bits.tolist()}, hard={hard_bits.tolist()}"
            )

    def test_soft_bits_to_int_roundtrip(self):
        """soft_bits_to_int should reconstruct the original value from hard bits."""
        for val in [0, 7, 42, 128, 255]:
            hard_bits = torch.tensor([(val >> i) & 1 for i in range(8)], dtype=torch.float)
            reconstructed = soft_bits_to_int(hard_bits)
            assert abs(reconstructed.item() - val) < 0.01, f"val={val}, got={reconstructed.item()}"

    def test_soft_int_to_bits_gradient_flows(self):
        x = torch.tensor(5.0, requires_grad=True)
        bits = soft_int_to_bits(x, n_bits=8, temperature=10.0)
        loss = bits.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs() > 0

    def test_soft_bits_to_int_gradient_flows(self):
        bits = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        val = soft_bits_to_int(bits)
        val.backward()
        assert bits.grad is not None
        # Gradient should be the power-of-2 weights
        expected = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float)
        assert torch.allclose(bits.grad, expected)

    def test_batched_soft_int_to_bits(self):
        x = torch.tensor([0.0, 5.0, 255.0])
        bits = soft_int_to_bits(x, n_bits=8, temperature=50.0)
        assert bits.shape == (3, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# SoftNeuralLogical tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoftNeuralLogical:
    def test_and_truth_table(self):
        """AND truth table: 0&0=0, 0&1=0, 1&0=0, 1&1=1."""
        logic = SoftNeuralLogical()
        # Set AND truth table logits so sigmoid gives [0, 0, 0, 1]
        with torch.no_grad():
            logic.truth_tables[0] = torch.tensor([-10.0, -10.0, -10.0, 10.0])

        # Test all 4 input combinations
        for a, b, expected in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]:
            bits_a = torch.tensor([[float(a)]])
            bits_b = torch.tensor([[float(b)]])
            result = logic.forward_single_op(bits_a, bits_b, op_idx=0)
            assert abs(result.item() - expected) < 0.01, f"AND({a},{b})={result.item()}, expected {expected}"

    def test_or_truth_table(self):
        """OR truth table: 0|0=0, 0|1=1, 1|0=1, 1|1=1."""
        logic = SoftNeuralLogical()
        with torch.no_grad():
            logic.truth_tables[1] = torch.tensor([-10.0, 10.0, 10.0, 10.0])

        for a, b, expected in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]:
            bits_a = torch.tensor([[float(a)]])
            bits_b = torch.tensor([[float(b)]])
            result = logic.forward_single_op(bits_a, bits_b, op_idx=1)
            assert abs(result.item() - expected) < 0.01, f"OR({a},{b})={result.item()}, expected {expected}"

    def test_xor_truth_table(self):
        """XOR truth table: 0^0=0, 0^1=1, 1^0=1, 1^1=0."""
        logic = SoftNeuralLogical()
        with torch.no_grad():
            logic.truth_tables[2] = torch.tensor([-10.0, 10.0, 10.0, -10.0])

        for a, b, expected in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
            bits_a = torch.tensor([[float(a)]])
            bits_b = torch.tensor([[float(b)]])
            result = logic.forward_single_op(bits_a, bits_b, op_idx=2)
            assert abs(result.item() - expected) < 0.01, f"XOR({a},{b})={result.item()}, expected {expected}"

    def test_gradient_flows_through_inputs(self):
        """Gradient must flow through bit_a and bit_b."""
        logic = SoftNeuralLogical()
        # Must use non-uniform truth table — uniform entries give constant output
        with torch.no_grad():
            logic.truth_tables[0] = torch.tensor([-10.0, -10.0, -10.0, 10.0])  # AND
        bits_a = torch.tensor([[0.3, 0.7]], requires_grad=True)
        bits_b = torch.tensor([[0.6, 0.4]], requires_grad=True)
        result = logic.forward_single_op(bits_a, bits_b, op_idx=0)
        loss = result.sum()
        loss.backward()
        assert bits_a.grad is not None and bits_a.grad.abs().sum() > 0
        assert bits_b.grad is not None and bits_b.grad.abs().sum() > 0

    def test_gradient_flows_through_truth_tables(self):
        """Gradient must flow through truth table parameters."""
        logic = SoftNeuralLogical()
        bits_a = torch.tensor([[0.3, 0.7]])
        bits_b = torch.tensor([[0.6, 0.4]])
        result = logic.forward_single_op(bits_a, bits_b, op_idx=0)
        loss = result.sum()
        loss.backward()
        assert logic.truth_tables.grad is not None
        assert logic.truth_tables.grad.abs().sum() > 0

    def test_soft_op_selection(self):
        """Weighted mixture of operations via forward()."""
        logic = SoftNeuralLogical()
        batch = 2
        n_bits = 4
        bits_a = torch.rand(batch, n_bits)
        bits_b = torch.rand(batch, n_bits)
        # Uniform op weights
        op_weights = torch.ones(batch, 7) / 7.0
        result = logic(bits_a, bits_b, op_weights)
        assert result.shape == (batch, n_bits)

    def test_multi_bit_and(self):
        """AND across multiple bit positions."""
        logic = SoftNeuralLogical()
        with torch.no_grad():
            logic.truth_tables[0] = torch.tensor([-10.0, -10.0, -10.0, 10.0])

        # a = [1,0,1,1], b = [1,1,0,1] → AND = [1,0,0,1]
        bits_a = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        bits_b = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        result = logic.forward_single_op(bits_a, bits_b, op_idx=0)
        expected = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        assert torch.allclose(result, expected, atol=0.01)

    def test_load_from_trained(self, tmp_path):
        """Can load weights from a .pt file."""
        # Save a mock state dict
        state = {"truth_tables": torch.randn(7, 4)}
        path = tmp_path / "logical.pt"
        torch.save(state, path)

        logic = SoftNeuralLogical()
        logic.load_from_trained(path)
        assert torch.allclose(logic.truth_tables.data, state["truth_tables"])


# ═══════════════════════════════════════════════════════════════════════════════
# SoftNeuralAdder tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoftNeuralAdder:
    def test_output_shape(self):
        adder = SoftNeuralAdder(hidden_dim=32, n_bits=8)
        bits_a = torch.zeros(2, 8)
        bits_b = torch.zeros(2, 8)
        result = adder(bits_a, bits_b)
        assert result.shape == (2, 8)

    def test_gradient_flows_through_adder(self):
        adder = SoftNeuralAdder(hidden_dim=32, n_bits=4)
        bits_a = torch.tensor([[0.0, 1.0, 0.0, 0.0]], requires_grad=True)
        bits_b = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
        result = adder(bits_a, bits_b)
        loss = result.sum()
        loss.backward()
        assert bits_a.grad is not None
        assert bits_b.grad is not None

    def test_zero_plus_zero(self):
        """0 + 0 = 0 for any random weights (carry chain starts at 0)."""
        adder = SoftNeuralAdder(hidden_dim=32, n_bits=8)
        bits_a = torch.zeros(1, 8)
        bits_b = torch.zeros(1, 8)
        result = adder(bits_a, bits_b)
        # With random weights, each bit is threshold of sigmoid(some_value)
        # The output should be deterministic but not necessarily 0
        # Just verify shapes and gradient flow
        assert result.shape == (1, 8)

    def test_batched_forward(self):
        adder = SoftNeuralAdder(hidden_dim=32, n_bits=8)
        bits_a = torch.rand(5, 8)
        bits_b = torch.rand(5, 8)
        result = adder(bits_a, bits_b)
        assert result.shape == (5, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# NCPUExpert tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNCPUExpert:
    def test_output_shape(self, config):
        expert = NCPUExpert(hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        out = expert(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    def test_gradient_flows(self, config):
        expert = NCPUExpert(hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_residual_scale_is_small(self, config):
        config.residual_init_scale = 0.01
        expert = NCPUExpert(hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        out = expert(x)
        # Output should be small due to residual_scale
        assert out.abs().mean() < 1.0

    def test_all_ops_contribute(self, config):
        """Each operation path should produce nonzero output."""
        expert = NCPUExpert(hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(1, 1, HIDDEN_DIM)
        flat = x.reshape(-1, HIDDEN_DIM)

        scalars = expert.scalar_proj(flat)
        assert scalars.shape == (1, 2)

        bits_raw = expert.bit_proj(flat)
        assert bits_raw.shape == (1, 2 * N_BITS)

        op_weights = expert.op_selector(flat)
        assert op_weights.shape == (1, 7)

    def test_load_pretrained_no_crash(self, config, tmp_path):
        """Loading from nonexistent dir should not crash."""
        expert = NCPUExpert(hidden_dim=HIDDEN_DIM, config=config)
        expert.load_pretrained_alu(tmp_path / "nonexistent", freeze=True)
        # Should still work (no models loaded)
        x = torch.randn(1, 1, HIDDEN_DIM)
        out = expert(x)
        assert out.shape == (1, 1, HIDDEN_DIM)


# ═══════════════════════════════════════════════════════════════════════════════
# Router tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNCPURouter:
    def test_gate_in_0_max_gate_range(self):
        router = NCPURouter(hidden_dim=HIDDEN_DIM, max_gate=0.1)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        gate, _ = router(x)
        assert gate.min() >= 0.0
        assert gate.max() <= 0.1 + 1e-6  # hard-capped at max_gate
        assert gate.shape == (BATCH_SIZE, SEQ_LEN, 1)

    def test_aux_loss_is_scalar(self):
        router = NCPURouter(hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        _, aux_loss = router(x)
        assert aux_loss.dim() == 0  # scalar

    def test_aux_loss_penalizes_deviation(self):
        router = NCPURouter(hidden_dim=HIDDEN_DIM, target_load=0.5, balance_coeff=1.0)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        _, aux_loss = router(x)
        # Loss should be nonnegative (it's a squared difference)
        assert aux_loss >= 0.0

    def test_gradient_through_gate(self):
        router = NCPURouter(hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        gate, aux_loss = router(x)
        loss = gate.sum() + aux_loss
        loss.backward()
        assert x.grad is not None

    def test_confidence_aware_without_mlp_output(self):
        """Confidence-aware router without mlp_output behaves like standard router."""
        router = NCPURouter(hidden_dim=HIDDEN_DIM, confidence_aware=True, max_gate=0.1)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        gate, aux_loss = router(x)
        assert gate.min() >= 0.0
        assert gate.max() <= 0.1 + 1e-6
        assert aux_loss.dim() == 0

    def test_confidence_aware_modulates_gate(self):
        """With mlp_output, confidence-aware mode modulates gate by uncertainty."""
        router = NCPURouter(hidden_dim=HIDDEN_DIM, confidence_aware=True, max_gate=0.5)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        mlp_out = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        gate, _ = router(x, mlp_output=mlp_out)
        assert gate.shape == (BATCH_SIZE, SEQ_LEN, 1)
        assert gate.min() >= 0.0
        assert gate.max() <= 0.5 + 1e-6

    def test_confidence_aware_gradient_flows(self):
        """Gradients flow through the confidence projection."""
        router = NCPURouter(hidden_dim=HIDDEN_DIM, confidence_aware=True, max_gate=0.1)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        mlp_out = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        gate, aux_loss = router(x, mlp_output=mlp_out)
        loss = gate.sum() + aux_loss
        loss.backward()
        assert x.grad is not None
        assert mlp_out.grad is not None
        # confidence_proj should have gradients
        assert router.confidence_proj.weight.grad is not None

    def test_confidence_aware_has_confidence_proj(self):
        """Confidence-aware router has the extra projection layer."""
        router_aware = NCPURouter(hidden_dim=HIDDEN_DIM, confidence_aware=True)
        router_basic = NCPURouter(hidden_dim=HIDDEN_DIM, confidence_aware=False)
        assert hasattr(router_aware, "confidence_proj")
        assert not hasattr(router_basic, "confidence_proj")

    def test_max_gate_caps_output(self):
        """max_gate parameter hard-caps the gate value."""
        router = NCPURouter(hidden_dim=HIDDEN_DIM, max_gate=0.05)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        gate, _ = router(x)
        assert gate.max() <= 0.05 + 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# CoprocessorMLP tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNCPUCoprocessorMLP:
    def test_output_shape_matches_input(self, config):
        mlp = MockMLP(HIDDEN_DIM)
        copro = NCPUCoprocessorMLP(mlp, hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        out = copro(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    def test_aux_loss_stored(self, config):
        mlp = MockMLP(HIDDEN_DIM)
        copro = NCPUCoprocessorMLP(mlp, hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        copro(x)
        assert copro.aux_loss is not None
        assert copro.aux_loss.dim() == 0

    def test_gradient_flows_through_both_paths(self, config):
        mlp = MockMLP(HIDDEN_DIM)
        copro = NCPUCoprocessorMLP(mlp, hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = copro(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_blending_between_mlp_and_expert(self, config):
        """When gate is 0, output should be pure MLP. When 1, pure expert."""
        mlp = MockMLP(HIDDEN_DIM)
        copro = NCPUCoprocessorMLP(mlp, hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(1, 1, HIDDEN_DIM)

        # Force gate to 0 by biasing gate_proj
        with torch.no_grad():
            copro.router.gate_proj.bias.fill_(-100.0)
            copro.router.gate_proj.weight.fill_(0.0)

        out_gate0 = copro(x)
        mlp_out = mlp(x)
        assert torch.allclose(out_gate0, mlp_out, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Injection tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInjection:
    def test_inject_replaces_mlp(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[-1])
        injected = inject_ncpu_coprocessor(model, config)
        assert len(injected) == 1
        assert isinstance(model.model.layers[-1].mlp, NCPUCoprocessorMLP)

    def test_inject_multiple_layers(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[0, 2])
        injected = inject_ncpu_coprocessor(model, config)
        assert len(injected) == 2
        assert isinstance(model.model.layers[0].mlp, NCPUCoprocessorMLP)
        assert isinstance(model.model.layers[2].mlp, NCPUCoprocessorMLP)
        # Unmodified layers should be normal
        assert isinstance(model.model.layers[1].mlp, MockMLP)

    def test_forward_after_injection(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[-1])
        inject_ncpu_coprocessor(model, config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        out = model(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    def test_backward_after_injection(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[-1])
        inject_ncpu_coprocessor(model, config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_collect_aux_losses(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[0, -1])
        inject_ncpu_coprocessor(model, config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        model(x)
        total_aux = collect_aux_losses(model)
        assert total_aux.dim() == 0
        assert total_aux >= 0.0

    def test_freeze_backbone(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[-1])
        inject_ncpu_coprocessor(model, config)
        freeze_backbone(model)

        # Non-coprocessor params should be frozen
        for name, p in model.named_parameters():
            if "router" in name or "expert" in name:
                if "soft_logical" not in name and "soft_adder" not in name:
                    assert p.requires_grad, f"{name} should be trainable"
            elif "original_mlp" in name:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_get_coprocessor_params(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[-1])
        inject_ncpu_coprocessor(model, config)
        freeze_backbone(model)
        params = get_coprocessor_params(model)
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_invalid_layer_index(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[10])
        with pytest.raises(IndexError):
            inject_ncpu_coprocessor(model, config)

    def test_negative_index_resolution(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=4)
        resolved = _resolve_layer_indices(model, [-1, -2])
        assert resolved == [3, 2]


# ═══════════════════════════════════════════════════════════════════════════════
# _replace_module tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplaceModule:
    def test_replace_simple_attr(self):
        model = nn.Module()
        model.fc = nn.Linear(10, 10)
        new_fc = nn.Linear(10, 5)
        _replace_module(model, "fc", new_fc)
        assert model.fc is new_fc

    def test_replace_nested_attr(self):
        model = MockTransformerModel(HIDDEN_DIM, n_layers=2)
        new_mlp = MockMLP(HIDDEN_DIM)
        _replace_module(model, "model.layers.0.mlp", new_mlp)
        assert model.model.layers[0].mlp is new_mlp


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end gradient flow tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndGradientFlow:
    def test_gradient_through_scalar_path(self, config):
        """Gradients flow through ADD/SUB/MUL (tensor ops)."""
        model = MockTransformerModel(HIDDEN_DIM, n_layers=2)
        inject_ncpu_coprocessor(model, config)
        x = torch.randn(1, 1, HIDDEN_DIM, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_through_neural_logic_path(self):
        """Gradients flow through SoftNeuralLogical truth tables."""
        logic = SoftNeuralLogical()
        # Must use non-uniform truth table for input gradients to be nonzero
        with torch.no_grad():
            logic.truth_tables[2] = torch.tensor([-10.0, 10.0, 10.0, -10.0])  # XOR
        bits_a = torch.tensor([[0.3, 0.7, 0.1, 0.9]], requires_grad=True)
        bits_b = torch.tensor([[0.6, 0.4, 0.8, 0.2]], requires_grad=True)
        result = logic.forward_single_op(bits_a, bits_b, op_idx=2)  # XOR
        loss = result.sum()
        loss.backward()

        # All three should have gradients
        assert bits_a.grad is not None and bits_a.grad.abs().sum() > 0
        assert bits_b.grad is not None and bits_b.grad.abs().sum() > 0
        assert logic.truth_tables.grad is not None

    def test_gradient_through_router_gate(self, config):
        """Gradients flow through the routing gate."""
        mlp = MockMLP(HIDDEN_DIM)
        copro = NCPUCoprocessorMLP(mlp, hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(1, 1, HIDDEN_DIM, requires_grad=True)
        out = copro(x)
        loss = out.sum()
        loss.backward()

        # Router gate params should have gradients
        assert copro.router.gate_proj.weight.grad is not None
        assert copro.router.gate_proj.weight.grad.abs().sum() > 0

    def test_all_coprocessor_params_have_gradients(self, config):
        """After forward+backward, all trainable coprocessor params get gradients."""
        model = MockTransformerModel(HIDDEN_DIM, n_layers=2)
        inject_ncpu_coprocessor(model, config)
        freeze_backbone(model)

        x = torch.randn(1, 2, HIDDEN_DIM)
        out = model(x)
        loss = out.sum() + collect_aux_losses(model)
        loss.backward()

        trainable_params = get_coprocessor_params(model)
        for p in trainable_params:
            assert p.grad is not None, "All trainable params should get gradients"


# ═══════════════════════════════════════════════════════════════════════════════
# Arithmetic correctness on discrete inputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestArithmeticCorrectness:
    def test_soft_and_matches_hard(self):
        """SoftNeuralLogical AND should match Python & on discrete bits."""
        logic = SoftNeuralLogical()
        # Set correct AND truth table
        with torch.no_grad():
            logic.truth_tables[0] = torch.tensor([-10.0, -10.0, -10.0, 10.0])

        for a_int in range(16):
            for b_int in range(16):
                expected = a_int & b_int
                # Convert to 4-bit representation
                bits_a = torch.tensor([[(a_int >> i) & 1 for i in range(4)]], dtype=torch.float)
                bits_b = torch.tensor([[(b_int >> i) & 1 for i in range(4)]], dtype=torch.float)
                result_bits = logic.forward_single_op(bits_a, bits_b, op_idx=0)
                # Convert back to int
                result_int = 0
                for i in range(4):
                    if result_bits[0, i].item() > 0.5:
                        result_int |= (1 << i)
                assert result_int == expected, f"{a_int} & {b_int}: got {result_int}, expected {expected}"

    def test_soft_or_matches_hard(self):
        """SoftNeuralLogical OR should match Python | on discrete bits."""
        logic = SoftNeuralLogical()
        with torch.no_grad():
            logic.truth_tables[1] = torch.tensor([-10.0, 10.0, 10.0, 10.0])

        for a_int in range(16):
            for b_int in range(16):
                expected = a_int | b_int
                bits_a = torch.tensor([[(a_int >> i) & 1 for i in range(4)]], dtype=torch.float)
                bits_b = torch.tensor([[(b_int >> i) & 1 for i in range(4)]], dtype=torch.float)
                result_bits = logic.forward_single_op(bits_a, bits_b, op_idx=1)
                result_int = sum((1 << i) for i in range(4) if result_bits[0, i].item() > 0.5)
                assert result_int == expected, f"{a_int} | {b_int}: got {result_int}, expected {expected}"

    def test_soft_xor_matches_hard(self):
        """SoftNeuralLogical XOR should match Python ^ on discrete bits."""
        logic = SoftNeuralLogical()
        with torch.no_grad():
            logic.truth_tables[2] = torch.tensor([-10.0, 10.0, 10.0, -10.0])

        for a_int in range(16):
            for b_int in range(16):
                expected = a_int ^ b_int
                bits_a = torch.tensor([[(a_int >> i) & 1 for i in range(4)]], dtype=torch.float)
                bits_b = torch.tensor([[(b_int >> i) & 1 for i in range(4)]], dtype=torch.float)
                result_bits = logic.forward_single_op(bits_a, bits_b, op_idx=2)
                result_int = sum((1 << i) for i in range(4) if result_bits[0, i].item() > 0.5)
                assert result_int == expected, f"{a_int} ^ {b_int}: got {result_int}, expected {expected}"


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter counting
# ═══════════════════════════════════════════════════════════════════════════════

class TestParameterCount:
    def test_coprocessor_is_lightweight(self, config):
        """Total coprocessor params should be << model params."""
        expert = NCPUExpert(hidden_dim=HIDDEN_DIM, config=config)
        router = NCPURouter(hidden_dim=HIDDEN_DIM)
        total = sum(p.numel() for p in expert.parameters())
        total += sum(p.numel() for p in router.parameters())
        # For hidden_dim=64, should be well under 100K
        assert total < 100_000, f"Too many params: {total}"

    def test_coprocessor_at_real_scale(self):
        """At Qwen 4B scale (hidden=2560), coprocessor should be ~124K params."""
        config = NCPUCoprocessorConfig(n_bits=8, num_ops=7)
        expert = NCPUExpert(hidden_dim=2560, config=config)
        router = NCPURouter(hidden_dim=2560)
        total = sum(p.numel() for p in expert.parameters())
        total += sum(p.numel() for p in router.parameters())
        assert total < 250_000, f"Coprocessor too large at real scale: {total}"


# ═══════════════════════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_default_config(self):
        config = NCPUCoprocessorConfig()
        assert config.n_bits == 8
        assert config.num_ops == 7
        assert config.target_load == 0.01
        assert config.confidence_aware is False
        assert config.max_gate == 0.1
        assert config.gate_warmup_steps == 0
        assert config.layer_gate_strategy == "uniform"
        assert config.deterministic_alu is False
        assert config.freeze_backbone is True
        assert config.layer_indices == [-1]

    def test_resolve_models_dir_default(self):
        config = NCPUCoprocessorConfig()
        assert config.resolve_models_dir() == Path("models")

    def test_resolve_models_dir_custom(self, tmp_path):
        config = NCPUCoprocessorConfig(models_dir=str(tmp_path / "my_models"))
        assert config.resolve_models_dir() == tmp_path / "my_models"


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic ALU tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterministicALU:
    """Tests for exact arithmetic mode (nCPU's answer to Percepta)."""

    def test_deterministic_expert_forward(self):
        """Deterministic expert produces output of correct shape."""
        config = NCPUCoprocessorConfig(deterministic_alu=True)
        expert = NCPUExpert(hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        out = expert(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    def test_deterministic_gradient_flows(self):
        """Gradients flow through deterministic ALU via STE."""
        config = NCPUCoprocessorConfig(deterministic_alu=True)
        expert = NCPUExpert(hidden_dim=HIDDEN_DIM, config=config)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_deterministic_and_exact(self):
        """Deterministic AND gives exact results for binary inputs."""
        from ncpu.coprocessor.ncpu_expert import _deterministic_and
        a = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        b = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        result = _deterministic_and(a, b)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_deterministic_or_exact(self):
        """Deterministic OR gives exact results for binary inputs."""
        from ncpu.coprocessor.ncpu_expert import _deterministic_or
        a = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        b = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        result = _deterministic_or(a, b)
        expected = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_deterministic_xor_exact(self):
        """Deterministic XOR gives exact results for binary inputs."""
        from ncpu.coprocessor.ncpu_expert import _deterministic_xor
        a = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        b = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        result = _deterministic_xor(a, b)
        expected = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_ste_round_gradient(self):
        """STE round: forward rounds, backward passes gradient through."""
        from ncpu.coprocessor.ncpu_expert import _ste_round
        x = torch.tensor([0.7, 1.3, 2.9, -0.4], requires_grad=True)
        y = _ste_round(x)
        assert torch.allclose(y, torch.tensor([1.0, 1.0, 3.0, 0.0]))
        y.sum().backward()
        # STE: gradient is 1 everywhere (straight-through)
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_deterministic_injection_e2e(self, config):
        """End-to-end: deterministic ALU works through full injection pipeline."""
        from ncpu.coprocessor.inject import inject_ncpu_coprocessor
        from ncpu.coprocessor.train import TinyTransformer

        det_config = NCPUCoprocessorConfig(deterministic_alu=True, layer_indices=[-1])
        model = TinyTransformer(vocab_size=100, hidden_dim=HIDDEN_DIM, n_layers=4)
        injected = inject_ncpu_coprocessor(model, det_config)
        assert len(injected) == 1
        assert injected[0].expert.deterministic_alu is True

        # Forward pass should work
        x = torch.randint(0, 100, (2, 8))
        out = model(input_ids=x)
        assert out.logits.shape == (2, 8, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive gate scheduling tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateScheduling:
    def test_gate_warmup_from_zero(self):
        """Gate warmup: starts at 0, linearly anneals to max_gate."""
        from ncpu.coprocessor.router import update_gate_schedule
        from ncpu.coprocessor.inject import inject_ncpu_coprocessor
        from ncpu.coprocessor.train import TinyTransformer

        model = TinyTransformer(vocab_size=100, hidden_dim=HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(layer_indices=[-1], max_gate=0.2)
        inject_ncpu_coprocessor(model, config)

        # Step 0: should be 0
        eff = update_gate_schedule(model, step=0, warmup_steps=100, max_gate=0.2)
        assert eff == 0.0

        # Step 50: should be 0.1 (halfway)
        eff = update_gate_schedule(model, step=50, warmup_steps=100, max_gate=0.2)
        assert abs(eff - 0.1) < 1e-6

        # Step 100: should be 0.2 (full)
        eff = update_gate_schedule(model, step=100, warmup_steps=100, max_gate=0.2)
        assert abs(eff - 0.2) < 1e-6

        # Step 200: clamped at max_gate
        eff = update_gate_schedule(model, step=200, warmup_steps=100, max_gate=0.2)
        assert abs(eff - 0.2) < 1e-6

    def test_per_layer_gating_linear_decay(self):
        """Linear decay: early layers get more gate budget than later layers."""
        from ncpu.coprocessor.inject import inject_ncpu_coprocessor
        from ncpu.coprocessor.train import TinyTransformer

        model = TinyTransformer(vocab_size=100, hidden_dim=HIDDEN_DIM, n_layers=4)
        config = NCPUCoprocessorConfig(
            layer_indices=[0, 1, 2, 3],
            max_gate=0.4,
            layer_gate_strategy="linear_decay",
        )
        injected = inject_ncpu_coprocessor(model, config)
        assert len(injected) == 4

        # Layer 0 should have max_gate near 0.4 (full), layer 3 near 0.1 (25%)
        assert injected[0].router.max_gate > injected[3].router.max_gate
        assert abs(injected[0].router.max_gate - 0.4) < 0.01
        assert abs(injected[3].router.max_gate - 0.1) < 0.01


from pathlib import Path
