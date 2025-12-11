"""Tests for CPURegistry primitives."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from kvrm_cpu.state import CPUState, create_initial_state
from kvrm_cpu.registry import CPURegistry, get_registry


class TestRegistryInitialization:
    """Test registry creation and freezing."""

    def test_registry_is_frozen(self):
        """Registry is frozen after initialization."""
        registry = get_registry()
        assert registry.is_frozen() is True

    def test_cannot_register_after_freeze(self):
        """Cannot add primitives after freeze."""
        registry = get_registry()
        with pytest.raises(RuntimeError):
            registry.register("OP_NEW", lambda s, p: s)

    def test_valid_keys(self):
        """All expected keys are registered."""
        registry = get_registry()
        keys = registry.get_valid_keys()
        expected = {
            "OP_MOV_REG_IMM", "OP_MOV_REG_REG",
            "OP_ADD", "OP_SUB", "OP_MUL",
            "OP_CMP",
            "OP_JMP", "OP_JZ", "OP_JNZ",
            "OP_HALT", "OP_NOP", "OP_INVALID"
        }
        assert keys == expected


class TestDataMovementPrimitives:
    """Test MOV operations."""

    def test_mov_reg_imm(self):
        """OP_MOV_REG_IMM loads immediate into register."""
        registry = get_registry()
        state = CPUState()

        new_state = registry.execute(state, "OP_MOV_REG_IMM", {
            "dest": "R0", "value": 42
        })

        assert new_state.registers["R0"] == 42
        assert new_state.pc == 1  # PC incremented
        assert new_state.cycle_count == 1

    def test_mov_reg_reg(self):
        """OP_MOV_REG_REG copies register to register."""
        registry = get_registry()
        state = CPUState()
        state = state.set_register("R1", 99)

        new_state = registry.execute(state, "OP_MOV_REG_REG", {
            "dest": "R0", "src": "R1"
        })

        assert new_state.registers["R0"] == 99
        assert new_state.registers["R1"] == 99  # Source unchanged


class TestArithmeticPrimitives:
    """Test ADD, SUB, MUL operations."""

    def test_add(self):
        """OP_ADD adds two registers."""
        registry = get_registry()
        state = CPUState()
        state = state.set_register("R1", 10)
        state = state.set_register("R2", 20)

        new_state = registry.execute(state, "OP_ADD", {
            "dest": "R0", "src1": "R1", "src2": "R2"
        })

        assert new_state.registers["R0"] == 30

    def test_sub(self):
        """OP_SUB subtracts src2 from src1."""
        registry = get_registry()
        state = CPUState()
        state = state.set_register("R1", 100)
        state = state.set_register("R2", 40)

        new_state = registry.execute(state, "OP_SUB", {
            "dest": "R0", "src1": "R1", "src2": "R2"
        })

        assert new_state.registers["R0"] == 60

    def test_mul(self):
        """OP_MUL multiplies two registers."""
        registry = get_registry()
        state = CPUState()
        state = state.set_register("R1", 7)
        state = state.set_register("R2", 6)

        new_state = registry.execute(state, "OP_MUL", {
            "dest": "R0", "src1": "R1", "src2": "R2"
        })

        assert new_state.registers["R0"] == 42


class TestComparisonPrimitive:
    """Test CMP operation."""

    def test_cmp_equal(self):
        """OP_CMP sets ZF when equal."""
        registry = get_registry()
        state = CPUState()
        state = state.set_register("R1", 50)
        state = state.set_register("R2", 50)

        new_state = registry.execute(state, "OP_CMP", {
            "src1": "R1", "src2": "R2"
        })

        assert new_state.flags["ZF"] is True
        assert new_state.flags["SF"] is False

    def test_cmp_less(self):
        """OP_CMP sets SF when src1 < src2."""
        registry = get_registry()
        state = CPUState()
        state = state.set_register("R1", 10)
        state = state.set_register("R2", 50)

        new_state = registry.execute(state, "OP_CMP", {
            "src1": "R1", "src2": "R2"
        })

        assert new_state.flags["ZF"] is False
        assert new_state.flags["SF"] is True

    def test_cmp_greater(self):
        """OP_CMP clears flags when src1 > src2."""
        registry = get_registry()
        state = CPUState()
        state = state.set_register("R1", 100)
        state = state.set_register("R2", 50)

        new_state = registry.execute(state, "OP_CMP", {
            "src1": "R1", "src2": "R2"
        })

        assert new_state.flags["ZF"] is False
        assert new_state.flags["SF"] is False


class TestControlFlowPrimitives:
    """Test JMP, JZ, JNZ operations."""

    def test_jmp(self):
        """OP_JMP sets PC unconditionally."""
        registry = get_registry()
        state = CPUState()

        new_state = registry.execute(state, "OP_JMP", {"addr": 5})

        assert new_state.pc == 5

    def test_jz_when_zero(self):
        """OP_JZ jumps when ZF is set."""
        registry = get_registry()
        state = CPUState(flags={"ZF": True, "SF": False})

        new_state = registry.execute(state, "OP_JZ", {"addr": 10})

        assert new_state.pc == 10

    def test_jz_when_not_zero(self):
        """OP_JZ falls through when ZF is clear."""
        registry = get_registry()
        state = CPUState(flags={"ZF": False, "SF": False})

        new_state = registry.execute(state, "OP_JZ", {"addr": 10})

        assert new_state.pc == 1  # PC incremented

    def test_jnz_when_not_zero(self):
        """OP_JNZ jumps when ZF is clear."""
        registry = get_registry()
        state = CPUState(flags={"ZF": False, "SF": False})

        new_state = registry.execute(state, "OP_JNZ", {"addr": 10})

        assert new_state.pc == 10

    def test_jnz_when_zero(self):
        """OP_JNZ falls through when ZF is set."""
        registry = get_registry()
        state = CPUState(flags={"ZF": True, "SF": False})

        new_state = registry.execute(state, "OP_JNZ", {"addr": 10})

        assert new_state.pc == 1


class TestSpecialPrimitives:
    """Test HALT, NOP, INVALID operations."""

    def test_halt(self):
        """OP_HALT sets halted flag."""
        registry = get_registry()
        state = CPUState()

        new_state = registry.execute(state, "OP_HALT", {})

        assert new_state.halted is True

    def test_nop(self):
        """OP_NOP only increments PC."""
        registry = get_registry()
        state = CPUState()
        state = state.set_register("R0", 42)

        new_state = registry.execute(state, "OP_NOP", {})

        assert new_state.pc == 1
        assert new_state.registers["R0"] == 42  # Unchanged

    def test_invalid(self):
        """OP_INVALID halts execution."""
        registry = get_registry()
        state = CPUState()

        new_state = registry.execute(state, "OP_INVALID", {"raw": "BADOP"})

        assert new_state.halted is True


class TestUnknownKey:
    """Test error handling for unknown keys."""

    def test_unknown_key_raises(self):
        """Unknown operation key raises KeyError."""
        registry = get_registry()
        state = CPUState()

        with pytest.raises(KeyError):
            registry.execute(state, "OP_UNKNOWN", {})
