"""Tests for DecodeLLM instruction decoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from kvrm_cpu.decode_llm import DecodeLLM, DecodeResult, parse_program


class TestDecodeResultDataclass:
    """Test DecodeResult structure."""

    def test_valid_result(self):
        """Valid decode result has key and params."""
        result = DecodeResult("OP_ADD", {"dest": "R0"}, True)
        assert result.key == "OP_ADD"
        assert result.params == {"dest": "R0"}
        assert result.valid is True
        assert result.error is None

    def test_invalid_result(self):
        """Invalid decode result has error message."""
        result = DecodeResult("OP_INVALID", {}, False, error="Unknown")
        assert result.valid is False
        assert result.error == "Unknown"


class TestMockDecodeMovInstructions:
    """Test mock decoder for MOV instructions."""

    @pytest.fixture
    def decoder(self):
        return DecodeLLM(mock_mode=True)

    def test_mov_reg_imm_decimal(self, decoder):
        """MOV Rd, imm with decimal value."""
        result = decoder.decode("MOV R0, 42")
        assert result.valid is True
        assert result.key == "OP_MOV_REG_IMM"
        assert result.params == {"dest": "R0", "value": 42}

    def test_mov_reg_imm_hex(self, decoder):
        """MOV Rd, imm with hex value."""
        result = decoder.decode("MOV R1, 0xFF")
        assert result.valid is True
        assert result.key == "OP_MOV_REG_IMM"
        assert result.params == {"dest": "R1", "value": 255}

    def test_mov_reg_imm_negative(self, decoder):
        """MOV Rd, imm with negative value."""
        result = decoder.decode("MOV R2, -10")
        assert result.valid is True
        assert result.key == "OP_MOV_REG_IMM"
        assert result.params == {"dest": "R2", "value": -10}

    def test_mov_reg_reg(self, decoder):
        """MOV Rd, Rs copies register."""
        result = decoder.decode("MOV R0, R1")
        assert result.valid is True
        assert result.key == "OP_MOV_REG_REG"
        assert result.params == {"dest": "R0", "src": "R1"}


class TestMockDecodeArithmetic:
    """Test mock decoder for arithmetic instructions."""

    @pytest.fixture
    def decoder(self):
        return DecodeLLM(mock_mode=True)

    def test_add(self, decoder):
        """ADD Rd, Rs1, Rs2."""
        result = decoder.decode("ADD R3, R1, R2")
        assert result.valid is True
        assert result.key == "OP_ADD"
        assert result.params == {"dest": "R3", "src1": "R1", "src2": "R2"}

    def test_sub(self, decoder):
        """SUB Rd, Rs1, Rs2."""
        result = decoder.decode("SUB R0, R5, R4")
        assert result.valid is True
        assert result.key == "OP_SUB"
        assert result.params == {"dest": "R0", "src1": "R5", "src2": "R4"}

    def test_mul(self, decoder):
        """MUL Rd, Rs1, Rs2."""
        result = decoder.decode("MUL R7, R3, R2")
        assert result.valid is True
        assert result.key == "OP_MUL"
        assert result.params == {"dest": "R7", "src1": "R3", "src2": "R2"}


class TestMockDecodeComparison:
    """Test mock decoder for CMP instruction."""

    @pytest.fixture
    def decoder(self):
        return DecodeLLM(mock_mode=True)

    def test_cmp(self, decoder):
        """CMP Rs1, Rs2."""
        result = decoder.decode("CMP R1, R2")
        assert result.valid is True
        assert result.key == "OP_CMP"
        assert result.params == {"src1": "R1", "src2": "R2"}


class TestMockDecodeControlFlow:
    """Test mock decoder for control flow instructions."""

    @pytest.fixture
    def decoder(self):
        return DecodeLLM(mock_mode=True)

    def test_jmp_numeric(self, decoder):
        """JMP with numeric address."""
        result = decoder.decode("JMP 5")
        assert result.valid is True
        assert result.key == "OP_JMP"
        assert result.params == {"addr": 5}

    def test_jmp_label(self, decoder):
        """JMP with label."""
        decoder.set_labels({"loop": 3})
        result = decoder.decode("JMP loop")
        assert result.valid is True
        assert result.key == "OP_JMP"
        assert result.params == {"addr": 3}

    def test_jz(self, decoder):
        """JZ with numeric address."""
        result = decoder.decode("JZ 10")
        assert result.valid is True
        assert result.key == "OP_JZ"
        assert result.params == {"addr": 10}

    def test_jnz_label(self, decoder):
        """JNZ with label."""
        decoder.set_labels({"done": 7})
        result = decoder.decode("JNZ done")
        assert result.valid is True
        assert result.key == "OP_JNZ"
        assert result.params == {"addr": 7}

    def test_unknown_label(self, decoder):
        """Unknown label returns invalid result."""
        result = decoder.decode("JMP unknown")
        assert result.valid is False
        assert "Unknown label" in result.error


class TestMockDecodeSpecial:
    """Test mock decoder for special instructions."""

    @pytest.fixture
    def decoder(self):
        return DecodeLLM(mock_mode=True)

    def test_halt(self, decoder):
        """HALT instruction."""
        result = decoder.decode("HALT")
        assert result.valid is True
        assert result.key == "OP_HALT"
        assert result.params == {}

    def test_nop(self, decoder):
        """NOP instruction."""
        result = decoder.decode("NOP")
        assert result.valid is True
        assert result.key == "OP_NOP"
        assert result.params == {}


class TestMockDecodeInvalid:
    """Test mock decoder error handling."""

    @pytest.fixture
    def decoder(self):
        return DecodeLLM(mock_mode=True)

    def test_empty_instruction(self, decoder):
        """Empty instruction returns invalid."""
        result = decoder.decode("")
        assert result.valid is False
        assert result.key == "OP_INVALID"

    def test_unknown_instruction(self, decoder):
        """Unknown instruction returns invalid."""
        result = decoder.decode("BADOP R1, R2")
        assert result.valid is False
        assert result.key == "OP_INVALID"

    def test_malformed_mov(self, decoder):
        """Malformed MOV returns invalid."""
        result = decoder.decode("MOV R9, 10")  # R9 doesn't exist
        assert result.valid is False


class TestMockDecodeCaseInsensitivity:
    """Test case-insensitive parsing."""

    @pytest.fixture
    def decoder(self):
        return DecodeLLM(mock_mode=True)

    def test_lowercase(self, decoder):
        """Lowercase instructions are parsed."""
        result = decoder.decode("mov r0, 42")
        assert result.valid is True
        assert result.key == "OP_MOV_REG_IMM"

    def test_mixed_case(self, decoder):
        """Mixed case instructions are parsed."""
        result = decoder.decode("Add R0, r1, R2")
        assert result.valid is True
        assert result.key == "OP_ADD"


class TestProgramParser:
    """Test parse_program function."""

    def test_simple_program(self):
        """Parse simple program without labels."""
        source = """
        MOV R0, 1
        HALT
        """
        instructions, labels = parse_program(source)
        assert instructions == ["MOV R0, 1", "HALT"]
        assert labels == {}

    def test_program_with_labels(self):
        """Parse program with labels."""
        source = """
        start:
            MOV R0, 1
        loop:
            ADD R0, R0, R1
            JNZ loop
            HALT
        """
        instructions, labels = parse_program(source)
        assert len(instructions) == 4
        assert labels["start"] == 0
        assert labels["loop"] == 1

    def test_comments_removed(self):
        """Comments are stripped."""
        source = """
        MOV R0, 1  ; load 1
        # This is a comment
        HALT ; done
        """
        instructions, labels = parse_program(source)
        assert instructions == ["MOV R0, 1", "HALT"]

    def test_empty_lines_ignored(self):
        """Empty lines are ignored."""
        source = """

        MOV R0, 1

        HALT

        """
        instructions, labels = parse_program(source)
        assert instructions == ["MOV R0, 1", "HALT"]
