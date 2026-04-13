"""Tests for the instruction decoder."""

import pytest
from ncpu.model.decode import Decoder, DecodeResult, parse_program


class TestDecodeResult:
    def test_valid_result(self):
        result = DecodeResult("OP_ADD", {"dest": "R0"}, True)
        assert result.key == "OP_ADD"
        assert result.valid is True
        assert result.error is None

    def test_invalid_result(self):
        result = DecodeResult("OP_INVALID", {}, False, error="Unknown")
        assert result.valid is False
        assert result.error == "Unknown"


class TestMockDecodeMov:
    @pytest.fixture
    def decoder(self):
        return Decoder(mock_mode=True)

    def test_mov_reg_imm_decimal(self, decoder):
        result = decoder.decode("MOV R0, 42")
        assert result.valid and result.key == "OP_MOV_REG_IMM"
        assert result.params == {"dest": "R0", "value": 42}

    def test_mov_reg_imm_hex(self, decoder):
        result = decoder.decode("MOV R1, 0xFF")
        assert result.valid and result.params == {"dest": "R1", "value": 255}

    def test_mov_reg_imm_negative(self, decoder):
        result = decoder.decode("MOV R2, -10")
        assert result.valid and result.params == {"dest": "R2", "value": -10}

    def test_mov_reg_reg(self, decoder):
        result = decoder.decode("MOV R0, R1")
        assert result.valid and result.key == "OP_MOV_REG_REG"
        assert result.params == {"dest": "R0", "src": "R1"}


class TestMockDecodeArithmetic:
    @pytest.fixture
    def decoder(self):
        return Decoder(mock_mode=True)

    def test_add(self, decoder):
        result = decoder.decode("ADD R3, R1, R2")
        assert result.valid and result.key == "OP_ADD"
        assert result.params == {"dest": "R3", "src1": "R1", "src2": "R2"}

    def test_sub(self, decoder):
        result = decoder.decode("SUB R0, R5, R4")
        assert result.valid and result.key == "OP_SUB"

    def test_mul(self, decoder):
        result = decoder.decode("MUL R7, R3, R2")
        assert result.valid and result.key == "OP_MUL"

    def test_div(self, decoder):
        result = decoder.decode("DIV R0, R1, R2")
        assert result.valid and result.key == "OP_DIV"
        assert result.params == {"dest": "R0", "src1": "R1", "src2": "R2"}

    def test_inc(self, decoder):
        result = decoder.decode("INC R0")
        assert result.valid and result.key == "OP_INC"
        assert result.params == {"dest": "R0"}

    def test_dec(self, decoder):
        result = decoder.decode("DEC R5")
        assert result.valid and result.key == "OP_DEC"
        assert result.params == {"dest": "R5"}


class TestMockDecodeBitwise:
    @pytest.fixture
    def decoder(self):
        return Decoder(mock_mode=True)

    def test_and(self, decoder):
        result = decoder.decode("AND R0, R1, R2")
        assert result.valid and result.key == "OP_AND"
        assert result.params == {"dest": "R0", "src1": "R1", "src2": "R2"}

    def test_or(self, decoder):
        result = decoder.decode("OR R3, R4, R5")
        assert result.valid and result.key == "OP_OR"
        assert result.params == {"dest": "R3", "src1": "R4", "src2": "R5"}

    def test_xor(self, decoder):
        result = decoder.decode("XOR R6, R0, R7")
        assert result.valid and result.key == "OP_XOR"
        assert result.params == {"dest": "R6", "src1": "R0", "src2": "R7"}


class TestMockDecodeShifts:
    @pytest.fixture
    def decoder(self):
        return Decoder(mock_mode=True)

    def test_shl_immediate(self, decoder):
        result = decoder.decode("SHL R0, R1, 3")
        assert result.valid and result.key == "OP_SHL"
        assert result.params == {"dest": "R0", "src": "R1", "amount": 3}

    def test_shr_immediate(self, decoder):
        result = decoder.decode("SHR R2, R3, 4")
        assert result.valid and result.key == "OP_SHR"
        assert result.params == {"dest": "R2", "src": "R3", "amount": 4}

    def test_shl_register(self, decoder):
        result = decoder.decode("SHL R0, R1, R2")
        assert result.valid and result.key == "OP_SHL"
        assert result.params == {"dest": "R0", "src": "R1", "amount_reg": "R2"}

    def test_shr_register(self, decoder):
        result = decoder.decode("SHR R4, R5, R6")
        assert result.valid and result.key == "OP_SHR"
        assert result.params == {"dest": "R4", "src": "R5", "amount_reg": "R6"}

    def test_shl_hex_immediate(self, decoder):
        result = decoder.decode("SHL R0, R1, 0x10")
        assert result.valid and result.params == {"dest": "R0", "src": "R1", "amount": 16}


class TestMockDecodeComparison:
    def test_cmp(self):
        result = Decoder(mock_mode=True).decode("CMP R1, R2")
        assert result.valid and result.key == "OP_CMP"
        assert result.params == {"src1": "R1", "src2": "R2"}


class TestMockDecodeControlFlow:
    @pytest.fixture
    def decoder(self):
        return Decoder(mock_mode=True)

    def test_jmp_numeric(self, decoder):
        result = decoder.decode("JMP 5")
        assert result.valid and result.params == {"addr": 5}

    def test_jmp_label(self, decoder):
        decoder.set_labels({"loop": 3})
        result = decoder.decode("JMP loop")
        assert result.valid and result.params == {"addr": 3}

    def test_jz(self, decoder):
        result = decoder.decode("JZ 10")
        assert result.valid and result.key == "OP_JZ"

    def test_jnz_label(self, decoder):
        decoder.set_labels({"done": 7})
        result = decoder.decode("JNZ done")
        assert result.valid and result.params == {"addr": 7}

    def test_js(self, decoder):
        result = decoder.decode("JS 3")
        assert result.valid and result.key == "OP_JS"

    def test_jns(self, decoder):
        result = decoder.decode("JNS 8")
        assert result.valid and result.key == "OP_JNS"

    def test_unknown_label(self, decoder):
        result = decoder.decode("JMP unknown")
        assert not result.valid
        assert "Unknown label" in result.error


class TestMockDecodeSpecial:
    @pytest.fixture
    def decoder(self):
        return Decoder(mock_mode=True)

    def test_halt(self, decoder):
        result = decoder.decode("HALT")
        assert result.valid and result.key == "OP_HALT"

    def test_nop(self, decoder):
        result = decoder.decode("NOP")
        assert result.valid and result.key == "OP_NOP"


class TestMockDecodeInvalid:
    @pytest.fixture
    def decoder(self):
        return Decoder(mock_mode=True)

    def test_empty_instruction(self, decoder):
        result = decoder.decode("")
        assert not result.valid and result.key == "OP_INVALID"

    def test_unknown_instruction(self, decoder):
        result = decoder.decode("BADOP R1, R2")
        assert not result.valid

    def test_malformed_mov(self, decoder):
        result = decoder.decode("MOV R9, 10")
        assert not result.valid


class TestMockDecodeCaseInsensitivity:
    @pytest.fixture
    def decoder(self):
        return Decoder(mock_mode=True)

    def test_lowercase(self, decoder):
        assert decoder.decode("mov r0, 42").valid

    def test_mixed_case(self, decoder):
        assert decoder.decode("Add R0, r1, R2").valid


class TestProgramParser:
    def test_simple_program(self):
        instructions, labels = parse_program("MOV R0, 1\nHALT")
        assert instructions == ["MOV R0, 1", "HALT"]
        assert labels == {}

    def test_program_with_labels(self):
        source = "start:\n  MOV R0, 1\nloop:\n  ADD R0, R0, R1\n  JNZ loop\n  HALT"
        instructions, labels = parse_program(source)
        assert len(instructions) == 4
        assert labels["start"] == 0
        assert labels["loop"] == 1

    def test_comments_removed(self):
        source = "MOV R0, 1  ; load 1\n# comment\nHALT ; done"
        instructions, _ = parse_program(source)
        assert instructions == ["MOV R0, 1", "HALT"]

    def test_empty_lines_ignored(self):
        source = "\n\nMOV R0, 1\n\nHALT\n\n"
        instructions, _ = parse_program(source)
        assert instructions == ["MOV R0, 1", "HALT"]
