"""Tests for CPURegistry primitives."""

import pytest
from ncpu.model.state import CPUState
from ncpu.model.registry import get_registry


class TestRegistryInit:
    def test_is_frozen(self):
        assert get_registry().is_frozen() is True

    def test_cannot_register_after_freeze(self):
        with pytest.raises(RuntimeError):
            get_registry().register("OP_NEW", lambda s, p: s)

    def test_valid_keys(self):
        expected = {
            "OP_MOV_REG_IMM", "OP_MOV_REG_REG",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
            "OP_AND", "OP_OR", "OP_XOR",
            "OP_SHL", "OP_SHR",
            "OP_INC", "OP_DEC",
            "OP_CMP",
            "OP_JMP", "OP_JZ", "OP_JNZ", "OP_JS", "OP_JNS",
            "OP_HALT", "OP_NOP", "OP_INVALID"
        }
        assert get_registry().get_valid_keys() == expected


class TestDataMovement:
    def test_mov_reg_imm(self):
        s = get_registry().execute(CPUState(), "OP_MOV_REG_IMM", {"dest": "R0", "value": 42})
        assert s.registers["R0"] == 42 and s.pc == 1 and s.cycle_count == 1

    def test_mov_reg_reg(self):
        state = CPUState().set_register("R1", 99)
        s = get_registry().execute(state, "OP_MOV_REG_REG", {"dest": "R0", "src": "R1"})
        assert s.registers["R0"] == 99 and s.registers["R1"] == 99


class TestArithmetic:
    def test_add(self):
        state = CPUState().set_register("R1", 10).set_register("R2", 20)
        s = get_registry().execute(state, "OP_ADD", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 30

    def test_sub(self):
        state = CPUState().set_register("R1", 100).set_register("R2", 40)
        s = get_registry().execute(state, "OP_SUB", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 60

    def test_mul(self):
        state = CPUState().set_register("R1", 7).set_register("R2", 6)
        s = get_registry().execute(state, "OP_MUL", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 42

    def test_div(self):
        state = CPUState().set_register("R1", 42).set_register("R2", 7)
        s = get_registry().execute(state, "OP_DIV", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 6

    def test_div_by_zero(self):
        state = CPUState().set_register("R1", 42).set_register("R2", 0)
        s = get_registry().execute(state, "OP_DIV", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 0

    def test_inc(self):
        state = CPUState().set_register("R0", 10)
        s = get_registry().execute(state, "OP_INC", {"dest": "R0"})
        assert s.registers["R0"] == 11

    def test_dec(self):
        state = CPUState().set_register("R0", 10)
        s = get_registry().execute(state, "OP_DEC", {"dest": "R0"})
        assert s.registers["R0"] == 9


class TestBitwise:
    def test_and(self):
        state = CPUState().set_register("R1", 0xFF).set_register("R2", 0x0F)
        s = get_registry().execute(state, "OP_AND", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 0x0F

    def test_or(self):
        state = CPUState().set_register("R1", 0xF0).set_register("R2", 0x0F)
        s = get_registry().execute(state, "OP_OR", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 0xFF

    def test_xor(self):
        state = CPUState().set_register("R1", 0xFF).set_register("R2", 0xFF)
        s = get_registry().execute(state, "OP_XOR", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 0

    def test_xor_nonzero(self):
        state = CPUState().set_register("R1", 0xFF).set_register("R2", 0x0F)
        s = get_registry().execute(state, "OP_XOR", {"dest": "R0", "src1": "R1", "src2": "R2"})
        assert s.registers["R0"] == 0xF0


class TestShifts:
    def test_shl_immediate(self):
        state = CPUState().set_register("R1", 1)
        s = get_registry().execute(state, "OP_SHL", {"dest": "R0", "src": "R1", "amount": 3})
        assert s.registers["R0"] == 8

    def test_shr_immediate(self):
        state = CPUState().set_register("R1", 16)
        s = get_registry().execute(state, "OP_SHR", {"dest": "R0", "src": "R1", "amount": 2})
        assert s.registers["R0"] == 4

    def test_shl_register(self):
        state = CPUState().set_register("R1", 1).set_register("R2", 4)
        s = get_registry().execute(state, "OP_SHL", {"dest": "R0", "src": "R1", "amount_reg": "R2"})
        assert s.registers["R0"] == 16

    def test_shr_register(self):
        state = CPUState().set_register("R1", 256).set_register("R2", 4)
        s = get_registry().execute(state, "OP_SHR", {"dest": "R0", "src": "R1", "amount_reg": "R2"})
        assert s.registers["R0"] == 16


class TestComparison:
    def test_cmp_equal(self):
        state = CPUState().set_register("R1", 50).set_register("R2", 50)
        s = get_registry().execute(state, "OP_CMP", {"src1": "R1", "src2": "R2"})
        assert s.flags["ZF"] is True and s.flags["SF"] is False

    def test_cmp_less(self):
        state = CPUState().set_register("R1", 10).set_register("R2", 50)
        s = get_registry().execute(state, "OP_CMP", {"src1": "R1", "src2": "R2"})
        assert s.flags["ZF"] is False and s.flags["SF"] is True

    def test_cmp_greater(self):
        state = CPUState().set_register("R1", 100).set_register("R2", 50)
        s = get_registry().execute(state, "OP_CMP", {"src1": "R1", "src2": "R2"})
        assert s.flags["ZF"] is False and s.flags["SF"] is False


class TestControlFlow:
    def test_jmp(self):
        assert get_registry().execute(CPUState(), "OP_JMP", {"addr": 5}).pc == 5

    def test_jz_when_zero(self):
        state = CPUState(flags={"ZF": True, "SF": False})
        assert get_registry().execute(state, "OP_JZ", {"addr": 10}).pc == 10

    def test_jz_when_not_zero(self):
        state = CPUState(flags={"ZF": False, "SF": False})
        assert get_registry().execute(state, "OP_JZ", {"addr": 10}).pc == 1

    def test_jnz_when_not_zero(self):
        state = CPUState(flags={"ZF": False, "SF": False})
        assert get_registry().execute(state, "OP_JNZ", {"addr": 10}).pc == 10

    def test_jnz_when_zero(self):
        state = CPUState(flags={"ZF": True, "SF": False})
        assert get_registry().execute(state, "OP_JNZ", {"addr": 10}).pc == 1

    def test_js_when_negative(self):
        state = CPUState(flags={"ZF": False, "SF": True})
        assert get_registry().execute(state, "OP_JS", {"addr": 5}).pc == 5

    def test_jns_when_positive(self):
        state = CPUState(flags={"ZF": False, "SF": False})
        assert get_registry().execute(state, "OP_JNS", {"addr": 5}).pc == 5


class TestSpecial:
    def test_halt(self):
        assert get_registry().execute(CPUState(), "OP_HALT", {}).halted is True

    def test_nop(self):
        s = get_registry().execute(CPUState().set_register("R0", 42), "OP_NOP", {})
        assert s.pc == 1 and s.registers["R0"] == 42

    def test_invalid(self):
        assert get_registry().execute(CPUState(), "OP_INVALID", {"raw": "BADOP"}).halted is True

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            get_registry().execute(CPUState(), "OP_UNKNOWN", {})
