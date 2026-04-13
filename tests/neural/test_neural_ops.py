"""Tests for neural ALU operations.

Verifies that the trained neural network models produce correct results
for all arithmetic, logical, comparison, and shift operations.
"""

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ncpu.model.neural_ops import NeuralOps, NeuralRegistry


# ═══════════════════════════════════════════════════════════════════════════════
# Loading & Availability
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralOpsLoading:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_loads_without_error(self, ops):
        assert ops.is_loaded

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_add_available(self, ops):
        assert ops._available_ops.get("ADD") is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_sub_available(self, ops):
        assert ops._available_ops.get("SUB") is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_mul_available(self, ops):
        assert ops._available_ops.get("MUL") is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_logical_available(self, ops):
        assert ops._available_ops.get("AND") is True
        assert ops._available_ops.get("OR") is True
        assert ops._available_ops.get("XOR") is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shift_available(self, ops):
        assert ops._available_ops.get("SHL") is True
        assert ops._available_ops.get("SHR") is True

    def test_missing_models_graceful(self):
        ops = NeuralOps(models_dir="/nonexistent")
        available = ops.load()
        assert len(available) == 0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_registry_valid_keys(self):
        reg = NeuralRegistry()
        reg.load()
        keys = reg.get_valid_keys()
        assert "OP_DIV" in keys
        assert "OP_AND" in keys
        assert "OP_OR" in keys
        assert "OP_XOR" in keys
        assert "OP_SHL" in keys
        assert "OP_SHR" in keys


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Addition
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralAdd:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0),
        (1, 1, 2),
        (100, 200, 300),
        (0, 42, 42),
        (1000, 2000, 3000),
        (255, 1, 256),
        (127, 128, 255),
    ])
    def test_add(self, ops, a, b, expected):
        assert ops.neural_add(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_add_negative(self, ops):
        assert ops.neural_add(-5, 5) == 0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_add_both_negative(self, ops):
        assert ops.neural_add(-10, -20) == -30


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Subtraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralSub:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (10, 5, 5),
        (100, 100, 0),
        (0, 0, 0),
        (300, 100, 200),
        (1, 2, -1),
    ])
    def test_sub(self, ops, a, b, expected):
        assert ops.neural_sub(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_sub_negative_result(self, ops):
        assert ops.neural_sub(5, 10) == -5


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Multiplication
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralMul:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0),
        (1, 1, 1),
        (7, 6, 42),
        (10, 10, 100),
        (0, 999, 0),
        (256, 256, 65536),
    ])
    def test_mul(self, ops, a, b, expected):
        assert ops.neural_mul(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_mul_negative(self, ops):
        assert ops.neural_mul(-3, 4) == -12

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_mul_both_negative(self, ops):
        assert ops.neural_mul(-5, -5) == 25


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Division
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralDiv:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (42, 7, 6),
        (100, 10, 10),
        (0, 5, 0),
        (7, 2, 3),
        (1, 1, 1),
        (255, 5, 51),
        (1000, 100, 10),
    ])
    def test_div(self, ops, a, b, expected):
        assert ops.neural_div(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_div_by_zero(self, ops):
        """Division by zero should return 0."""
        assert ops.neural_div(42, 0) == 0
        assert ops.neural_div(0, 0) == 0
        assert ops.neural_div(-5, 0) == 0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (-12, 3, -4),
        (12, -3, -4),
        (-12, -3, 4),
        (-100, 10, -10),
        (100, -10, -10),
    ])
    def test_div_signed(self, ops, a, b, expected):
        """Division with signed operands truncates toward zero."""
        assert ops.neural_div(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_div_remainder_discarded(self, ops):
        """Integer division discards remainder."""
        assert ops.neural_div(10, 3) == 3
        assert ops.neural_div(17, 5) == 3
        assert ops.neural_div(-7, 2) == -3


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Compare
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralCmp:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_equal(self, ops):
        zf, sf = ops.neural_cmp(50, 50)
        assert zf is True and sf is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_less(self, ops):
        zf, sf = ops.neural_cmp(10, 50)
        assert zf is False and sf is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_greater(self, ops):
        zf, sf = ops.neural_cmp(100, 50)
        assert zf is False and sf is False


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Inc/Dec
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralIncDec:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("value,expected", [
        (0, 1),
        (99, 100),
        (-1, 0),
        (255, 256),
    ])
    def test_inc(self, ops, value, expected):
        assert ops.neural_inc(value) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("value,expected", [
        (1, 0),
        (100, 99),
        (0, -1),
        (256, 255),
    ])
    def test_dec(self, ops, value, expected):
        assert ops.neural_dec(value) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Logical (AND, OR, XOR)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralLogical:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (0xFF, 0x0F, 0x0F),
        (0xFF, 0xFF, 0xFF),
        (0xFF, 0x00, 0x00),
        (0xAA, 0x55, 0x00),
        (0x0F, 0x0F, 0x0F),
    ])
    def test_and(self, ops, a, b, expected):
        assert ops.neural_and(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (0xF0, 0x0F, 0xFF),
        (0x00, 0x00, 0x00),
        (0xFF, 0x00, 0xFF),
        (0xAA, 0x55, 0xFF),
    ])
    def test_or(self, ops, a, b, expected):
        assert ops.neural_or(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (0xFF, 0xFF, 0x00),
        (0xFF, 0x00, 0xFF),
        (0xAA, 0x55, 0xFF),
        (0x0F, 0x0F, 0x00),
        (0xF0, 0x0F, 0xFF),
    ])
    def test_xor(self, ops, a, b, expected):
        assert ops.neural_xor(a, b) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Shifts (SHL, SHR)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralShift:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("value,amount,expected", [
        (1, 0, 1),
        (1, 1, 2),
        (1, 3, 8),
        (1, 4, 16),
        (3, 2, 12),
        (0xFF, 8, 0xFF00),
    ])
    def test_shl(self, ops, value, amount, expected):
        assert ops.neural_shl(value, amount) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("value,amount,expected", [
        (16, 2, 4),
        (256, 4, 16),
        (8, 3, 1),
        (0xFF00, 8, 0xFF),
        (1, 0, 1),
    ])
    def test_shr(self, ops, value, amount, expected):
        assert ops.neural_shr(value, amount) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shl_zero(self, ops):
        assert ops.neural_shl(0, 5) == 0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shr_zero(self, ops):
        assert ops.neural_shr(0, 5) == 0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shl_by_zero(self, ops):
        """Shift by 0 should return original value."""
        assert ops.neural_shl(42, 0) == 42

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shr_by_zero(self, ops):
        """Shift by 0 should return original value."""
        assert ops.neural_shr(42, 0) == 42

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shl_by_31(self, ops):
        """Shift left by max amount (31 bits). Wraps to INT32_MIN in 32-bit signed."""
        assert ops.neural_shl(1, 31) == -2147483648

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shr_by_31(self, ops):
        """Shift right by max amount."""
        # 0x80000000 as signed = -2147483648, unsigned = 2147483648
        # Logical shift right by 31: 0x80000000 >> 31 = 1
        assert ops.neural_shr(-2147483648, 31) == 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shl_large_value(self, ops):
        """Shift a multi-bit value."""
        assert ops.neural_shl(0xABCD, 4) == 0xABCD0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shr_large_value(self, ops):
        """Shift a multi-bit value right."""
        assert ops.neural_shr(0xABCD0, 4) == 0xABCD


# ═══════════════════════════════════════════════════════════════════════════════
# CLA (Carry-Lookahead Addition) Correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLACorrectness:
    """Verify the Kogge-Stone CLA produces identical results to ripple-carry."""

    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_loaded(self, ops):
        """Verify carry_combine.pt loaded successfully."""
        assert ops._carry_combiner is not None, "carry_combine.pt not loaded"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 2),
        (127, 1, 128),       # carry propagation across byte boundary
        (255, 1, 256),       # carry propagation into second byte
        (65535, 1, 65536),   # carry propagation into third byte
        (0xFFFF, 0xFFFF, 0x1FFFE),  # large carry chain
        (0xFFFFFF, 1, 0x1000000),   # 24-bit carry chain
        (100, 200, 300),
        (1000, 2000, 3000),
        (123456, 654321, 777777),
    ])
    def test_cla_add_positive(self, ops, a, b, expected):
        """CLA addition with positive operands."""
        assert ops.neural_add(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (-1, 1, 0),
        (-5, 5, 0),
        (-10, -20, -30),
        (-100, 50, -50),
        (50, -100, -50),
        (-1, -1, -2),
        (-128, -128, -256),
    ])
    def test_cla_add_negative(self, ops, a, b, expected):
        """CLA addition with negative operands."""
        assert ops.neural_add(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_add_overflow(self, ops):
        """CLA handles 32-bit signed overflow correctly."""
        assert ops.neural_add(2147483647, 1) == -2147483648

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_add_underflow(self, ops):
        """CLA handles 32-bit signed underflow correctly."""
        assert ops.neural_add(-2147483648, -1) == 2147483647

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.parametrize("a,b,expected", [
        (10, 5, 5),
        (100, 100, 0),
        (5, 10, -5),
        (0, 1, -1),
        (-10, -5, -5),
        (-5, -10, 5),
        (1000, 999, 1),
    ])
    def test_cla_sub(self, ops, a, b, expected):
        """CLA subtraction (a + ~b + 1)."""
        assert ops.neural_sub(a, b) == expected

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_cmp_equal(self, ops):
        """CLA-based CMP: equal values."""
        zf, sf = ops.neural_cmp(42, 42)
        assert zf is True and sf is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_cmp_less(self, ops):
        """CLA-based CMP: a < b."""
        zf, sf = ops.neural_cmp(10, 50)
        assert zf is False and sf is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_cmp_greater(self, ops):
        """CLA-based CMP: a > b."""
        zf, sf = ops.neural_cmp(100, 50)
        assert zf is False and sf is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_inc(self, ops):
        """INC through CLA path."""
        assert ops.neural_inc(99) == 100
        assert ops.neural_inc(-1) == 0
        assert ops.neural_inc(255) == 256

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_dec(self, ops):
        """DEC through CLA path."""
        assert ops.neural_dec(100) == 99
        assert ops.neural_dec(0) == -1
        assert ops.neural_dec(256) == 255

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_cla_wide_range(self, ops):
        """Test CLA across a wide range of values."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            a = rng.randint(-(2**30), 2**30)
            b = rng.randint(-(2**30), 2**30)
            expected = a + b
            # Clamp to 32-bit signed
            if expected > 2147483647:
                expected -= 2**32
            elif expected < -2147483648:
                expected += 2**32
            assert ops.neural_add(a, b) == expected, f"Failed: {a} + {b}"


# ═══════════════════════════════════════════════════════════════════════════════
# Batch API
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchAPI:
    @pytest.fixture
    def ops(self):
        ops = NeuralOps()
        ops.load()
        return ops

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_batch_add(self, ops):
        pairs = [(1, 2), (100, 200), (-5, 5), (0, 0)]
        results = ops.batch_neural_add(pairs)
        assert results == [3, 300, 0, 0]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_batch_mul(self, ops):
        pairs = [(7, 6), (10, 10), (-3, 4), (0, 999)]
        results = ops.batch_neural_mul(pairs)
        assert results == [42, 100, -12, 0]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_batch_bitwise(self, ops):
        triples = [(0xFF, 0x0F, 0), (0xF0, 0x0F, 1), (0xFF, 0xFF, 2)]
        results = ops.batch_neural_bitwise(triples)
        assert results == [0x0F, 0xFF, 0x00]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_batch_mul_empty(self, ops):
        assert ops.batch_neural_mul([]) == []

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_batch_mul_single(self, ops):
        assert ops.batch_neural_mul([(5, 5)]) == [25]
