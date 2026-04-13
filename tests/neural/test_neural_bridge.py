"""Tests for the Neural ALU Bridge connecting ncpu.model's trained networks to ncpu.neural.

Verifies that:
1. The bridge loads and initializes correctly
2. Individual operations produce correct results through the bridge
3. NeuralCPU automatically uses the neural ALU bridge (always on)
"""

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Bridge Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralALUBridge:
    @pytest.fixture
    def bridge(self):
        from ncpu.neural.neural_alu_bridge import NeuralALUBridge
        b = NeuralALUBridge()
        b.load()
        return b

    def test_bridge_loads(self, bridge):
        assert bridge.is_loaded

    def test_add_ints(self, bridge):
        assert bridge.add(10, 20) == 30

    def test_add_tensors(self, bridge):
        a = torch.tensor(10, dtype=torch.int64)
        b = torch.tensor(20, dtype=torch.int64)
        assert bridge.add(a, b) == 30

    def test_sub(self, bridge):
        assert bridge.sub(100, 60) == 40

    def test_sub_negative(self, bridge):
        assert bridge.sub(5, 10) == -5

    def test_mul(self, bridge):
        assert bridge.mul(7, 6) == 42

    def test_and(self, bridge):
        assert bridge.and_(0xFF, 0x0F) == 0x0F

    def test_or(self, bridge):
        assert bridge.or_(0xF0, 0x0F) == 0xFF

    def test_xor(self, bridge):
        assert bridge.xor_(0xFF, 0xFF) == 0x00

    def test_shl(self, bridge):
        assert bridge.shl(1, 4) == 16

    def test_shr(self, bridge):
        assert bridge.shr(16, 2) == 4

    def test_cmp_equal(self, bridge):
        n, z, c = bridge.cmp(50, 50)
        assert z == 1.0
        assert n == 0.0

    def test_cmp_less(self, bridge):
        n, z, c = bridge.cmp(10, 50)
        assert n == 1.0
        assert z == 0.0

    def test_cmp_greater(self, bridge):
        n, z, c = bridge.cmp(100, 50)
        assert n == 0.0
        assert z == 0.0

    def test_64bit_narrowing(self, bridge):
        """Bridge should narrow 64-bit values to 32-bit for the models."""
        big = torch.tensor(0x100000001, dtype=torch.int64)
        result = bridge.add(big, torch.tensor(1, dtype=torch.int64))
        assert result == 2  # 0x100000001 & 0xFFFFFFFF = 1, then 1 + 1 = 2


# ═══════════════════════════════════════════════════════════════════════════════
# NeuralCPU Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralCPUWithBridge:
    @pytest.fixture
    def cpu(self):
        from ncpu.neural import NeuralCPU
        return NeuralCPU(device_override="cpu")

    def test_neural_alu_always_on(self, cpu):
        """Neural ALU is enabled by default — no toggle needed."""
        assert cpu.use_neural_alu is True

    def test_neural_alu_bridge_loaded(self, cpu):
        """Bridge should be loaded and ready on construction."""
        assert cpu._neural_alu is not None
        assert cpu._neural_alu.is_loaded

    def test_add_with_neural_alu(self, cpu):
        """Test ADD instruction routed through neural ALU.

        ARM64 encoding: ADD X0, X1, X2
        31 = 1 (sf=64bit), 00 = opc, 01011 = op, 00 = shift, 0 = N
        Rm=X2(00010), imm6=0(000000), Rn=X1(00001), Rd=X0(00000)
        = 0x8B020020
        """
        # Set registers
        cpu.regs[1] = 10
        cpu.regs[2] = 20

        # Encode ADD X0, X1, X2 = 0x8B020020
        inst = 0x8B020020
        cpu.load_binary(inst.to_bytes(4, 'little'), addr=0)
        cpu.pc = torch.tensor(0, dtype=torch.int64, device=cpu.device)

        cpu.step()
        assert int(cpu.regs[0].item()) == 30

    def test_sub_with_neural_alu(self, cpu):
        """Test SUB instruction routed through neural ALU.

        ARM64 encoding: SUB X0, X1, X2
        = 0xCB020020
        """
        cpu.regs[1] = 100
        cpu.regs[2] = 40

        inst = 0xCB020020
        cpu.load_binary(inst.to_bytes(4, 'little'), addr=0)
        cpu.pc = torch.tensor(0, dtype=torch.int64, device=cpu.device)

        cpu.step()
        assert int(cpu.regs[0].item()) == 60

    def test_cmp_flags_with_neural_alu(self, cpu):
        """Test CMP instruction sets flags through neural ALU.

        ARM64 encoding: CMP X0, X1 = SUBS XZR, X0, X1
        = 0xEB01001F
        """
        cpu.regs[0] = 10
        cpu.regs[1] = 20

        inst = 0xEB01001F
        cpu.load_binary(inst.to_bytes(4, 'little'), addr=0)
        cpu.pc = torch.tensor(0, dtype=torch.int64, device=cpu.device)

        cpu.step()
        # 10 - 20 = -10: N=1, Z=0
        assert cpu.flags[0].item() == 1.0  # N flag
        assert cpu.flags[1].item() == 0.0  # Z flag
