"""Tests for CPUState dataclass."""

import pytest
from ncpu.model.state import CPUState, create_initial_state, INT32_MIN, INT32_MAX


class TestCPUStateCreation:
    def test_default_state(self):
        state = CPUState()
        assert state.pc == 0
        assert state.cycle_count == 0
        assert state.halted is False
        for reg in ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
            assert state.registers[reg] == 0
        assert state.flags["ZF"] is False
        assert state.flags["SF"] is False

    def test_create_initial_state(self):
        program = ["MOV R0, 1", "HALT"]
        state = create_initial_state(program)
        assert state.memory == program
        assert len(state.memory) == 2
        assert state.pc == 0
        assert state.halted is False


class TestCPUStateValidation:
    def test_valid_state(self):
        assert CPUState().validate() is True

    def test_invalid_register_value(self):
        state = CPUState()
        state.registers["R0"] = INT32_MAX + 1
        assert state.validate() is False

    def test_negative_pc(self):
        assert CPUState(pc=-1).validate() is False


class TestCPUStateImmutability:
    def test_set_register_returns_new_state(self):
        state = CPUState()
        new_state = state.set_register("R0", 42)
        assert state.registers["R0"] == 0
        assert new_state.registers["R0"] == 42

    def test_set_register_clamps_value(self):
        state = CPUState()
        assert state.set_register("R0", INT32_MAX + 100).registers["R0"] == INT32_MAX
        assert state.set_register("R0", INT32_MIN - 100).registers["R0"] == INT32_MIN

    def test_set_flags(self):
        state = CPUState()
        z = state.set_flags(0)
        assert z.flags["ZF"] is True and z.flags["SF"] is False

        p = state.set_flags(42)
        assert p.flags["ZF"] is False and p.flags["SF"] is False

        n = state.set_flags(-1)
        assert n.flags["ZF"] is False and n.flags["SF"] is True

    def test_increment_pc(self):
        state = CPUState()
        new_state = state.increment_pc()
        assert state.pc == 0
        assert new_state.pc == 1

    def test_set_pc(self):
        assert CPUState().set_pc(5).pc == 5

    def test_set_halted(self):
        state = CPUState()
        new_state = state.set_halted(True)
        assert state.halted is False
        assert new_state.halted is True

    def test_increment_cycle(self):
        state = CPUState()
        new_state = state.increment_cycle()
        assert state.cycle_count == 0
        assert new_state.cycle_count == 1


class TestCPUStateSnapshot:
    def test_snapshot_is_deep_copy(self):
        state = CPUState().set_register("R0", 42)
        snapshot = state.snapshot()
        assert snapshot["registers"]["R0"] == 42
        snapshot["registers"]["R0"] = 999
        assert state.registers["R0"] == 42


class TestCPUStateAccessors:
    def test_get_register(self):
        assert CPUState().set_register("R3", 100).get_register("R3") == 100

    def test_get_register_case_insensitive(self):
        state = CPUState().set_register("R0", 42)
        assert state.get_register("r0") == 42
        assert state.get_register("R0") == 42

    def test_get_register_invalid(self):
        with pytest.raises(KeyError):
            CPUState().get_register("R9")

    def test_dump_registers(self):
        state = CPUState().set_register("R0", 1).set_register("R7", 2)
        regs = state.dump_registers()
        assert regs["R0"] == 1 and regs["R7"] == 2
        regs["R0"] = 999
        assert state.registers["R0"] == 1
