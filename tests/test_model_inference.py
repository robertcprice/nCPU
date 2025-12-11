"""Validation tests for trained Qwen2.5-Coder decode LLM.

This module tests the fine-tuned model's ability to decode assembly instructions
into verified registry keys and parameters in real (non-mock) mode.
"""

import pytest
import json
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kvrm_cpu.decode_llm import DecodeLLM


# Path to trained checkpoint
CHECKPOINT_PATH = Path(__file__).parent.parent / "models" / "decode_llm" / "checkpoint-2500"


class TestModelLoading:
    """Test that the trained model loads correctly."""

    def test_checkpoint_exists(self):
        """Verify checkpoint directory exists."""
        assert CHECKPOINT_PATH.exists(), f"Checkpoint not found at {CHECKPOINT_PATH}"

    def test_adapter_config_exists(self):
        """Verify adapter config exists and has correct base model."""
        adapter_path = CHECKPOINT_PATH / "adapter_config.json"
        assert adapter_path.exists()

        with open(adapter_path) as f:
            config = json.load(f)

        assert config["base_model_name_or_path"] == "Qwen/Qwen2.5-Coder-1.5B"
        assert config["peft_type"] == "LORA"
        assert config["r"] == 16

    def test_model_weights_exist(self):
        """Verify model weights file exists."""
        weights_path = CHECKPOINT_PATH / "adapter_model.safetensors"
        assert weights_path.exists()


@pytest.mark.slow
class TestRealModeInference:
    """Test model inference in real (non-mock) mode.

    These tests require GPU/MPS and take significant time to run.
    Use: pytest tests/test_model_inference.py -m slow
    """

    @pytest.fixture(scope="class")
    def decoder(self):
        """Load the trained model once for all tests."""
        decoder = DecodeLLM(mock_mode=False, model_path=str(CHECKPOINT_PATH))
        decoder.load()
        return decoder

    def test_mov_reg_imm(self, decoder):
        """Test MOV register immediate decoding."""
        result = decoder.decode("MOV R3, 42")
        assert result.key == "OP_MOV_REG_IMM"
        assert result.params["dest"] == "R3"
        assert result.params["value"] == 42

    def test_mov_reg_reg(self, decoder):
        """Test MOV register to register decoding."""
        result = decoder.decode("MOV R0, R1")
        assert result.key == "OP_MOV_REG_REG"
        assert result.params["dest"] == "R0"
        assert result.params["src"] == "R1"

    def test_add(self, decoder):
        """Test ADD instruction decoding."""
        result = decoder.decode("ADD R0, R1, R2")
        assert result.key == "OP_ADD"
        assert result.params["dest"] == "R0"
        assert result.params["src1"] == "R1"
        assert result.params["src2"] == "R2"

    def test_sub(self, decoder):
        """Test SUB instruction decoding."""
        result = decoder.decode("SUB R5, R3, R1")
        assert result.key == "OP_SUB"
        assert result.params["dest"] == "R5"

    def test_cmp(self, decoder):
        """Test CMP instruction decoding."""
        result = decoder.decode("CMP R3, R4")
        assert result.key == "OP_CMP"
        assert result.params["src1"] == "R3"
        assert result.params["src2"] == "R4"

    def test_jmp(self, decoder):
        """Test JMP instruction decoding."""
        result = decoder.decode("JMP 10")
        assert result.key == "OP_JMP"
        assert result.params["target"] == 10

    def test_halt(self, decoder):
        """Test HALT instruction decoding."""
        result = decoder.decode("HALT")
        assert result.key == "OP_HALT"

    def test_nop(self, decoder):
        """Test NOP instruction decoding."""
        result = decoder.decode("NOP")
        assert result.key == "OP_NOP"

    def test_case_insensitive(self, decoder):
        """Test case-insensitive instruction parsing."""
        result = decoder.decode("add r0, r1, r2")
        assert result.key == "OP_ADD"

    def test_invalid_instruction(self, decoder):
        """Test that invalid instructions are handled."""
        result = decoder.decode("BADOP R1")
        assert result.key == "OP_INVALID"


class TestMockVsRealComparison:
    """Compare mock mode output with real model output."""

    def test_mock_mode_baseline(self):
        """Verify mock mode works for baseline comparison."""
        decoder = DecodeLLM(mock_mode=True)

        test_cases = [
            ("MOV R0, 10", "OP_MOV_REG_IMM"),
            ("ADD R1, R2, R3", "OP_ADD"),
            ("SUB R4, R5, R6", "OP_SUB"),
            ("CMP R0, R1", "OP_CMP"),
            ("JMP 100", "OP_JMP"),
            ("HALT", "OP_HALT"),
        ]

        for instruction, expected_key in test_cases:
            result = decoder.decode(instruction)
            assert result.key == expected_key, f"Mock failed for {instruction}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
