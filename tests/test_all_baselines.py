"""Unified test suite for all baseline decoders.

This module runs the same test cases against all decoder implementations
to ensure consistency and measure accuracy.
"""

import pytest
import time
from typing import Type, List, Tuple, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines import BaseDecoder, DecodeResult, FSMDecoder, RegexDecoder, RuleBasedDecoder


# All decoder classes to test
DECODER_CLASSES: List[Type[BaseDecoder]] = [
    FSMDecoder,
    RegexDecoder,
    RuleBasedDecoder,
]


# Test cases: (instruction, expected_key, expected_params)
# params can be partial match (subset of actual params)
TEST_CASES: List[Tuple[str, str, dict]] = [
    # HALT and NOP
    ("HALT", "OP_HALT", {}),
    ("halt", "OP_HALT", {}),  # Case insensitive
    ("NOP", "OP_NOP", {}),
    ("nop", "OP_NOP", {}),

    # MOV register to immediate
    ("MOV R0, 10", "OP_MOV_REG_IMM", {"dest": "R0", "value": 10}),
    ("MOV R3, 42", "OP_MOV_REG_IMM", {"dest": "R3", "value": 42}),
    ("MOV R7, -5", "OP_MOV_REG_IMM", {"dest": "R7", "value": -5}),
    ("mov r0, 100", "OP_MOV_REG_IMM", {"dest": "R0", "value": 100}),
    ("MOV R1, 0x1F", "OP_MOV_REG_IMM", {"dest": "R1", "value": 31}),
    ("MOV R2, 0b1010", "OP_MOV_REG_IMM", {"dest": "R2", "value": 10}),

    # MOV register to register
    ("MOV R0, R1", "OP_MOV_REG_REG", {"dest": "R0", "src": "R1"}),
    ("MOV R5, R3", "OP_MOV_REG_REG", {"dest": "R5", "src": "R3"}),
    ("mov r2, r4", "OP_MOV_REG_REG", {"dest": "R2", "src": "R4"}),

    # ADD
    ("ADD R0, R1, R2", "OP_ADD", {"dest": "R0", "src1": "R1", "src2": "R2"}),
    ("add r3, r4, r5", "OP_ADD", {"dest": "R3", "src1": "R4", "src2": "R5"}),
    ("ADD R7, R0, R1", "OP_ADD", {"dest": "R7", "src1": "R0", "src2": "R1"}),

    # SUB
    ("SUB R0, R1, R2", "OP_SUB", {"dest": "R0", "src1": "R1", "src2": "R2"}),
    ("sub r5, r3, r1", "OP_SUB", {"dest": "R5", "src1": "R3", "src2": "R1"}),

    # MUL
    ("MUL R0, R1, R2", "OP_MUL", {"dest": "R0", "src1": "R1", "src2": "R2"}),
    ("mul r6, r2, r3", "OP_MUL", {"dest": "R6", "src1": "R2", "src2": "R3"}),

    # INC and DEC
    ("INC R0", "OP_INC", {"dest": "R0"}),
    ("inc r5", "OP_INC", {"dest": "R5"}),
    ("DEC R1", "OP_DEC", {"dest": "R1"}),
    ("dec r7", "OP_DEC", {"dest": "R7"}),

    # CMP
    ("CMP R0, R1", "OP_CMP", {"src1": "R0", "src2": "R1"}),
    ("cmp r3, r4", "OP_CMP", {"src1": "R3", "src2": "R4"}),

    # JMP (numeric address)
    ("JMP 10", "OP_JMP", {"addr": 10}),
    ("jmp 0", "OP_JMP", {"addr": 0}),
    ("JMP 100", "OP_JMP", {"addr": 100}),

    # JZ (numeric address)
    ("JZ 5", "OP_JZ", {"addr": 5}),
    ("jz 20", "OP_JZ", {"addr": 20}),

    # JNZ (numeric address)
    ("JNZ 15", "OP_JNZ", {"addr": 15}),
    ("jnz 8", "OP_JNZ", {"addr": 8}),

    # JS (numeric address)
    ("JS 3", "OP_JS", {"addr": 3}),
    ("js 12", "OP_JS", {"addr": 12}),

    # JNS (numeric address)
    ("JNS 7", "OP_JNS", {"addr": 7}),
    ("jns 25", "OP_JNS", {"addr": 25}),
]


# Invalid test cases: (instruction, should be OP_INVALID)
INVALID_CASES: List[str] = [
    "",  # Empty
    "   ",  # Whitespace only
    "BADOP R1",  # Unknown mnemonic
    "JUMP 10",  # Typo in mnemonic
    "ADD R0, R1",  # Missing operand
    "MOV R8, 10",  # Invalid register R8
    "MOV R9, R0",  # Invalid register R9
]


class TestDecoderInterface:
    """Test that all decoders implement the expected interface."""

    @pytest.mark.parametrize("decoder_class", DECODER_CLASSES)
    def test_has_decode_method(self, decoder_class: Type[BaseDecoder]):
        """Verify decode method exists."""
        decoder = decoder_class()
        assert hasattr(decoder, "decode")
        assert callable(decoder.decode)

    @pytest.mark.parametrize("decoder_class", DECODER_CLASSES)
    def test_has_set_labels(self, decoder_class: Type[BaseDecoder]):
        """Verify set_labels method exists."""
        decoder = decoder_class()
        assert hasattr(decoder, "set_labels")
        decoder.set_labels({"LOOP": 5, "END": 10})
        assert decoder.labels == {"LOOP": 5, "END": 10}

    @pytest.mark.parametrize("decoder_class", DECODER_CLASSES)
    def test_decode_returns_result(self, decoder_class: Type[BaseDecoder]):
        """Verify decode returns DecodeResult."""
        decoder = decoder_class()
        result = decoder.decode("HALT")
        assert isinstance(result, DecodeResult)
        assert hasattr(result, "key")
        assert hasattr(result, "params")
        assert hasattr(result, "valid")


class TestDecoderAccuracy:
    """Test accuracy of all decoders on standard test cases."""

    @pytest.mark.parametrize("decoder_class", DECODER_CLASSES)
    @pytest.mark.parametrize("instruction,expected_key,expected_params", TEST_CASES)
    def test_valid_instruction(
        self,
        decoder_class: Type[BaseDecoder],
        instruction: str,
        expected_key: str,
        expected_params: dict,
    ):
        """Test valid instruction decoding."""
        decoder = decoder_class()
        result = decoder.decode(instruction)

        assert result.valid, f"{decoder_class.__name__} failed on '{instruction}': {result.error}"
        assert result.key == expected_key, (
            f"{decoder_class.__name__} wrong key for '{instruction}': "
            f"expected {expected_key}, got {result.key}"
        )

        # Check params (partial match)
        for key, value in expected_params.items():
            assert key in result.params, (
                f"{decoder_class.__name__} missing param '{key}' for '{instruction}'"
            )
            assert result.params[key] == value, (
                f"{decoder_class.__name__} wrong param '{key}' for '{instruction}': "
                f"expected {value}, got {result.params[key]}"
            )

    @pytest.mark.parametrize("decoder_class", DECODER_CLASSES)
    @pytest.mark.parametrize("instruction", INVALID_CASES)
    def test_invalid_instruction(
        self, decoder_class: Type[BaseDecoder], instruction: str
    ):
        """Test invalid instruction handling."""
        decoder = decoder_class()
        result = decoder.decode(instruction)

        assert result.key == "OP_INVALID", (
            f"{decoder_class.__name__} should return OP_INVALID for '{instruction}', "
            f"got {result.key}"
        )


class TestLabelResolution:
    """Test label resolution for jump instructions."""

    @pytest.mark.parametrize("decoder_class", DECODER_CLASSES)
    def test_jump_with_label(self, decoder_class: Type[BaseDecoder]):
        """Test jump instruction with label resolution."""
        decoder = decoder_class()
        decoder.set_labels({"LOOP": 5, "END": 15})

        result = decoder.decode("JMP LOOP")
        assert result.valid
        assert result.key == "OP_JMP"
        assert result.params["addr"] == 5

        result = decoder.decode("JZ END")
        assert result.valid
        assert result.key == "OP_JZ"
        assert result.params["addr"] == 15

    @pytest.mark.parametrize("decoder_class", DECODER_CLASSES)
    def test_jump_unknown_label(self, decoder_class: Type[BaseDecoder]):
        """Test jump with unknown label returns invalid."""
        decoder = decoder_class()
        decoder.set_labels({"LOOP": 5})

        result = decoder.decode("JMP UNKNOWN")
        assert result.key == "OP_INVALID"


class TestDecoderPerformance:
    """Benchmark decoder performance."""

    ITERATIONS = 1000

    @pytest.mark.parametrize("decoder_class", DECODER_CLASSES)
    def test_decode_latency(self, decoder_class: Type[BaseDecoder]):
        """Measure average decode latency."""
        decoder = decoder_class()
        instructions = [tc[0] for tc in TEST_CASES[:10]]  # Use first 10 cases

        # Warmup
        for instr in instructions:
            decoder.decode(instr)

        # Timed run
        start = time.perf_counter()
        for _ in range(self.ITERATIONS):
            for instr in instructions:
                decoder.decode(instr)
        end = time.perf_counter()

        total_decodes = self.ITERATIONS * len(instructions)
        avg_latency_ms = (end - start) / total_decodes * 1000

        print(f"\n{decoder_class.__name__}: {avg_latency_ms:.4f}ms per decode")

        # All deterministic decoders should be under 0.1ms
        assert avg_latency_ms < 0.1, (
            f"{decoder_class.__name__} too slow: {avg_latency_ms:.4f}ms"
        )


class TestDecoderConsistency:
    """Verify all decoders produce consistent results."""

    def test_all_decoders_agree(self):
        """All decoders should produce identical results for all test cases."""
        decoders = [cls() for cls in DECODER_CLASSES]

        for instruction, expected_key, expected_params in TEST_CASES:
            results = [d.decode(instruction) for d in decoders]

            # All should have same key
            keys = [r.key for r in results]
            assert len(set(keys)) == 1, (
                f"Inconsistent keys for '{instruction}': {keys}"
            )

            # All should have same validity
            validities = [r.valid for r in results]
            assert len(set(validities)) == 1, (
                f"Inconsistent validity for '{instruction}': {validities}"
            )

            # All should have same params (for expected keys)
            for key, value in expected_params.items():
                param_values = [r.params.get(key) for r in results]
                assert len(set(param_values)) == 1, (
                    f"Inconsistent param '{key}' for '{instruction}': {param_values}"
                )


def run_accuracy_report():
    """Generate accuracy report for all decoders."""
    print("\n" + "=" * 60)
    print("BASELINE DECODER ACCURACY REPORT")
    print("=" * 60)

    for decoder_class in DECODER_CLASSES:
        decoder = decoder_class()
        correct = 0
        total = len(TEST_CASES) + len(INVALID_CASES)

        # Test valid cases
        for instruction, expected_key, expected_params in TEST_CASES:
            result = decoder.decode(instruction)
            if result.key == expected_key and result.valid:
                params_correct = all(
                    result.params.get(k) == v for k, v in expected_params.items()
                )
                if params_correct:
                    correct += 1

        # Test invalid cases
        for instruction in INVALID_CASES:
            result = decoder.decode(instruction)
            if result.key == "OP_INVALID":
                correct += 1

        accuracy = correct / total * 100
        print(f"\n{decoder_class.__name__}:")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"  Parameters: {decoder.parameter_count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run accuracy report
    run_accuracy_report()

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
