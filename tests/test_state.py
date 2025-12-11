"""Tests for CPUState dataclass."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from kvrm_cpu.state import CPUState, create_initial_state, INT32_MIN, INT32_MAX


class TestCPUStateCreation:
    """Test CPUState initialization and defaults."""

    def test_default_state(self):
        """Default state has zeroed registers and flags."""
        state = CPUState()
        assert state.pc == 0
        assert state.cycle_count == 0
        assert state.halted is False
        for reg in ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
            assert state.registers[reg] == 0
        assert state.flags["ZF"] is False
        assert state.flags["SF"] is False

    def test_create_initial_state(self):
        """create_initial_state loads program into memory."""
        program = ["MOV R0, 1", "HALT"]
        state = create_initial_state(program)
        assert state.memory == program
        assert len(state.memory) == 2
        assert state.pc == 0
        assert state.halted is False


class TestCPUStateValidation:
    """Test state validation."""

    def test_valid_state(self):
        """Valid state passes validation."""
        state = CPUState()
        assert state.validate() is True

    def test_invalid_register_value(self):
        """Register value out of bounds fails validation."""
        state = CPUState()
        state.registers["R0"] = INT32_MAX + 1
        assert state.validate() is False

    def test_negative_pc(self):
        """Negative PC fails validation."""
        state = CPUState(pc=-1)
        assert state.validate() is False


class TestCPUStateImmutability:
    """Test immutable state operations."""

    def test_set_register_returns_new_state(self):
        """set_register returns a new state, original unchanged."""
        state = CPUState()
        new_state = state.set_register("R0", 42)
        assert state.registers["R0"] == 0  # Original unchanged
        assert new_state.registers["R0"] == 42

    def test_set_register_clamps_value(self):
        """set_register clamps values to 32-bit signed range."""
        state = CPUState()
        new_state = state.set_register("R0", INT32_MAX + 100)
        assert new_state.registers["R0"] == INT32_MAX

        new_state = state.set_register("R0", INT32_MIN - 100)
        assert new_state.registers["R0"] == INT32_MIN

    def test_set_flags(self):
        """set_flags correctly updates ZF and SF."""
        state = CPUState()

        # Zero value sets ZF
        new_state = state.set_flags(0)
        assert new_state.flags["ZF"] is True
        assert new_state.flags["SF"] is False

        # Positive value clears both
        new_state = state.set_flags(42)
        assert new_state.flags["ZF"] is False
        assert new_state.flags["SF"] is False

        # Negative value sets SF
        new_state = state.set_flags(-1)
        assert new_state.flags["ZF"] is False
        assert new_state.flags["SF"] is True

    def test_increment_pc(self):
        """increment_pc increases PC by 1."""
        state = CPUState()
        new_state = state.increment_pc()
        assert state.pc == 0  # Original unchanged
        assert new_state.pc == 1

    def test_set_pc(self):
        """set_pc sets PC to arbitrary value."""
        state = CPUState()
        new_state = state.set_pc(5)
        assert new_state.pc == 5

    def test_set_halted(self):
        """set_halted sets halted flag."""
        state = CPUState()
        new_state = state.set_halted(True)
        assert state.halted is False  # Original unchanged
        assert new_state.halted is True

    def test_increment_cycle(self):
        """increment_cycle increases cycle_count."""
        state = CPUState()
        new_state = state.increment_cycle()
        assert state.cycle_count == 0
        assert new_state.cycle_count == 1


class TestCPUStateSnapshot:
    """Test state snapshot for tracing."""

    def test_snapshot_is_deep_copy(self):
        """Snapshot is a deep copy of state."""
        state = CPUState()
        state = state.set_register("R0", 42)
        snapshot = state.snapshot()

        assert snapshot["registers"]["R0"] == 42
        assert snapshot["pc"] == 0
        assert snapshot["halted"] is False

        # Modifying snapshot doesn't affect state
        snapshot["registers"]["R0"] = 999
        assert state.registers["R0"] == 42


class TestCPUStateAccessors:
    """Test register and flag accessors."""

    def test_get_register(self):
        """get_register returns register value."""
        state = CPUState()
        state = state.set_register("R3", 100)
        assert state.get_register("R3") == 100

    def test_get_register_case_insensitive(self):
        """get_register is case insensitive."""
        state = CPUState()
        state = state.set_register("R0", 42)
        assert state.get_register("r0") == 42
        assert state.get_register("R0") == 42

    def test_get_register_invalid(self):
        """get_register raises KeyError for invalid register."""
        state = CPUState()
        with pytest.raises(KeyError):
            state.get_register("R9")

    def test_dump_registers(self):
        """dump_registers returns copy of all registers."""
        state = CPUState()
        state = state.set_register("R0", 1)
        state = state.set_register("R7", 2)

        regs = state.dump_registers()
        assert regs["R0"] == 1
        assert regs["R7"] == 2

        # Modifying copy doesn't affect state
        regs["R0"] = 999
        assert state.registers["R0"] == 1
