"""Comprehensive real LLM model tests for complex programs.

These tests use the ACTUAL trained KVRM decode LLM (not mock mode) to run
complex assembly programs. This validates that the neural network can
correctly decode all instructions in real-world computational scenarios.

Author: Bobby Price (blackWeb Research)
Date: January 2025
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kvrm_cpu import KVRMCPU

CHECKPOINT_PATH = Path(__file__).parent.parent / "models" / "decode_llm"


@pytest.fixture(scope="module")
def real_cpu():
    """Create a KVRM-CPU using the real trained LLM model."""
    cpu = KVRMCPU(mock_mode=False, model_path=str(CHECKPOINT_PATH), max_cycles=5000)
    cpu.load()
    yield cpu
    cpu.unload()


# =============================================================================
# BASIC OPERATIONS - Real LLM Decode
# =============================================================================

class TestRealModelBasicOperations:
    """Test basic operations with real LLM decode."""

    def test_simple_mov_and_halt(self, real_cpu):
        """Simple MOV followed by HALT."""
        real_cpu.load_program("""
            MOV R0, 42
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 42

    def test_register_to_register_copy(self, real_cpu):
        """Copy value between registers."""
        real_cpu.load_program("""
            MOV R0, 100
            MOV R1, R0
            MOV R2, R1
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 100
        assert real_cpu.get_register("R1") == 100
        assert real_cpu.get_register("R2") == 100

    def test_arithmetic_chain(self, real_cpu):
        """Chain of arithmetic operations."""
        real_cpu.load_program("""
            MOV R0, 10
            MOV R1, 5
            ADD R2, R0, R1      ; 15
            SUB R3, R0, R1      ; 5
            MUL R4, R2, R3      ; 75
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R2") == 15
        assert real_cpu.get_register("R3") == 5
        assert real_cpu.get_register("R4") == 75


# =============================================================================
# CONDITIONAL LOGIC - Real LLM Decode
# =============================================================================

class TestRealModelConditionalLogic:
    """Test conditional jumps with real LLM decode."""

    def test_equal_comparison_jz(self, real_cpu):
        """JZ taken when values are equal."""
        real_cpu.load_program("""
            MOV R0, 50
            MOV R1, 50
            CMP R0, R1
            JZ equal
            MOV R7, 0
            JMP done
        equal:
            MOV R7, 1
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R7") == 1  # Equal path

    def test_less_than_comparison_js(self, real_cpu):
        """JS taken when first < second."""
        real_cpu.load_program("""
            MOV R0, 10
            MOV R1, 20
            CMP R0, R1
            JS less
            MOV R7, 0
            JMP done
        less:
            MOV R7, 1
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R7") == 1  # Less path

    def test_greater_comparison_jns(self, real_cpu):
        """JNS taken when first >= second."""
        real_cpu.load_program("""
            MOV R0, 30
            MOV R1, 20
            CMP R0, R1
            JNS greater_or_equal
            MOV R7, 0
            JMP done
        greater_or_equal:
            MOV R7, 1
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R7") == 1


# =============================================================================
# LOOPS - Real LLM Decode
# =============================================================================

class TestRealModelLoops:
    """Test loop constructs with real LLM decode."""

    def test_countdown_loop(self, real_cpu):
        """Count down from 5 to 0."""
        real_cpu.load_program("""
            MOV R0, 5
            MOV R1, 0
            MOV R2, 1
        loop:
            CMP R0, R1
            JZ done
            SUB R0, R0, R2
            JMP loop
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 0

    def test_sum_loop(self, real_cpu):
        """Sum 1 + 2 + 3 + 4 + 5 = 15."""
        real_cpu.load_program("""
            MOV R0, 0       ; sum
            MOV R1, 1       ; counter
            MOV R2, 6       ; limit
            MOV R3, 1       ; increment
        loop:
            ADD R0, R0, R1  ; sum += counter
            ADD R1, R1, R3  ; counter++
            CMP R1, R2
            JNZ loop
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 15

    def test_multiply_via_addition(self, real_cpu):
        """6 * 7 = 42 via repeated addition."""
        real_cpu.load_program("""
            MOV R0, 0       ; result
            MOV R1, 7       ; value to add
            MOV R2, 6       ; times to add
            MOV R3, 1       ; decrement
            MOV R4, 0       ; zero
        loop:
            CMP R2, R4
            JZ done
            ADD R0, R0, R1
            SUB R2, R2, R3
            JMP loop
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 42


# =============================================================================
# FIBONACCI - Real LLM Decode
# =============================================================================

class TestRealModelFibonacci:
    """Test Fibonacci computation with real LLM decode."""

    def test_fibonacci_8_iterations(self, real_cpu):
        """Compute F(9) = 34 after 8 iterations."""
        real_cpu.load_program("""
            MOV R0, 0       ; F(n-2)
            MOV R1, 1       ; F(n-1)
            MOV R2, 8       ; iterations
            MOV R3, 0       ; counter
            MOV R4, 1       ; constant
        loop:
            CMP R3, R2
            JZ done
            MOV R5, R1
            ADD R1, R0, R1
            MOV R0, R5
            ADD R3, R3, R4
            JMP loop
        done:
            HALT
        """)
        real_cpu.run()
        # F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, F(6)=8, F(7)=13, F(8)=21, F(9)=34
        assert real_cpu.get_register("R1") == 34


# =============================================================================
# FACTORIAL - Real LLM Decode
# =============================================================================

class TestRealModelFactorial:
    """Test factorial computation with real LLM decode."""

    def test_factorial_5(self, real_cpu):
        """5! = 120."""
        real_cpu.load_program("""
            MOV R0, 1       ; result
            MOV R1, 5       ; n
            MOV R2, 1       ; decrement
            MOV R3, 1       ; compare
        loop:
            CMP R1, R3
            JS done
            JZ done
            MUL R0, R0, R1
            SUB R1, R1, R2
            JMP loop
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 120


# =============================================================================
# POWER - Real LLM Decode
# =============================================================================

class TestRealModelPower:
    """Test exponentiation with real LLM decode."""

    def test_power_2_8(self, real_cpu):
        """2^8 = 256."""
        real_cpu.load_program("""
            MOV R0, 1       ; result
            MOV R1, 2       ; base
            MOV R2, 8       ; exponent
            MOV R3, 1       ; decrement
            MOV R4, 0       ; zero
        loop:
            CMP R2, R4
            JZ done
            MUL R0, R0, R1
            SUB R2, R2, R3
            JMP loop
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 256


# =============================================================================
# GCD (EUCLIDEAN ALGORITHM) - Real LLM Decode
# =============================================================================

class TestRealModelGCD:
    """Test GCD computation with real LLM decode."""

    def test_gcd_48_18(self, real_cpu):
        """GCD(48, 18) = 6."""
        real_cpu.load_program("""
            MOV R0, 48
            MOV R1, 18
            MOV R3, 0
        gcd_loop:
            CMP R1, R3
            JZ gcd_done
            MOV R2, R0
        mod_loop:
            CMP R2, R1
            JS mod_done
            SUB R2, R2, R1
            JMP mod_loop
        mod_done:
            MOV R0, R1
            MOV R1, R2
            JMP gcd_loop
        gcd_done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 6


# =============================================================================
# INTEGER DIVISION - Real LLM Decode
# =============================================================================

class TestRealModelDivision:
    """Test integer division with real LLM decode."""

    def test_divide_100_by_7(self, real_cpu):
        """100 / 7 = 14 remainder 2."""
        real_cpu.load_program("""
            MOV R0, 100     ; dividend
            MOV R1, 7       ; divisor
            MOV R2, 0       ; quotient
            MOV R3, 1       ; increment
        loop:
            CMP R0, R1
            JS done
            SUB R0, R0, R1
            ADD R2, R2, R3
            JMP loop
        done:
            ; R2 = quotient, R0 = remainder
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R2") == 14  # quotient
        assert real_cpu.get_register("R0") == 2   # remainder


# =============================================================================
# PRIME CHECK - Real LLM Decode
# =============================================================================

class TestRealModelPrimeCheck:
    """Test prime checking with real LLM decode."""

    def test_is_prime_7(self, real_cpu):
        """7 is prime."""
        real_cpu.load_program("""
            MOV R0, 7       ; number
            MOV R1, 2       ; divisor
            MOV R2, 1       ; increment
            MOV R7, 1       ; assume prime
        check:
            CMP R1, R0
            JNS is_prime
            JZ is_prime
            MOV R3, R0
        mod:
            CMP R3, R1
            JS mod_done
            SUB R3, R3, R1
            JMP mod
        mod_done:
            MOV R4, 0
            CMP R3, R4
            JZ not_prime
            ADD R1, R1, R2
            JMP check
        not_prime:
            MOV R7, 0
        is_prime:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R7") == 1

    def test_is_not_prime_9(self, real_cpu):
        """9 is not prime (divisible by 3)."""
        real_cpu.load_program("""
            MOV R0, 9
            MOV R1, 2
            MOV R2, 1
            MOV R7, 1
        check:
            CMP R1, R0
            JNS is_prime
            JZ is_prime
            MOV R3, R0
        mod:
            CMP R3, R1
            JS mod_done
            SUB R3, R3, R1
            JMP mod
        mod_done:
            MOV R4, 0
            CMP R3, R4
            JZ not_prime
            ADD R1, R1, R2
            JMP check
        not_prime:
            MOV R7, 0
        is_prime:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R7") == 0


# =============================================================================
# NESTED LOOPS - Real LLM Decode
# =============================================================================

class TestRealModelNestedLoops:
    """Test nested loop structures with real LLM decode."""

    def test_double_nested_loop(self, real_cpu):
        """Count 3 * 4 = 12 iterations."""
        real_cpu.load_program("""
            MOV R0, 0       ; counter
            MOV R1, 3       ; outer limit
            MOV R2, 4       ; inner limit
            MOV R3, 1       ; constant
            MOV R4, 0       ; zero
        outer:
            CMP R1, R4
            JZ done
            MOV R5, R2      ; reset inner
        inner:
            CMP R5, R4
            JZ inner_done
            ADD R0, R0, R3  ; count++
            SUB R5, R5, R3
            JMP inner
        inner_done:
            SUB R1, R1, R3
            JMP outer
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 12


# =============================================================================
# COMPLEX MULTI-ALGORITHM - Real LLM Decode
# =============================================================================

class TestRealModelComplexAlgorithms:
    """Test complex multi-step algorithms with real LLM decode."""

    def test_sum_of_squares_1_to_5(self, real_cpu):
        """1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55."""
        real_cpu.load_program("""
            MOV R0, 0       ; sum
            MOV R1, 1       ; i
            MOV R2, 5       ; limit
            MOV R3, 1       ; increment
            MOV R4, 0       ; temp
        loop:
            MUL R4, R1, R1  ; i^2
            ADD R0, R0, R4  ; sum += i^2
            ADD R1, R1, R3  ; i++
            CMP R1, R2
            JS loop
            JZ last
            JMP done
        last:
            MUL R4, R1, R1
            ADD R0, R0, R4
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 55

    def test_find_max_of_three(self, real_cpu):
        """max(15, 42, 27) = 42."""
        real_cpu.load_program("""
            MOV R0, 15
            MOV R1, 42
            MOV R2, 27
            MOV R3, R0      ; max candidate
            CMP R3, R1
            JNS skip1
            MOV R3, R1
        skip1:
            CMP R3, R2
            JNS skip2
            MOV R3, R2
        skip2:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R3") == 42


# =============================================================================
# STRESS TESTS - Real LLM Decode
# =============================================================================

class TestRealModelStress:
    """Stress tests with real LLM decode."""

    def test_long_computation_sum_1_to_20(self, real_cpu):
        """Sum 1 to 20 = 210."""
        real_cpu.load_program("""
            MOV R0, 0
            MOV R1, 1
            MOV R2, 21
            MOV R3, 1
        loop:
            ADD R0, R0, R1
            ADD R1, R1, R3
            CMP R1, R2
            JNZ loop
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 210

    def test_power_2_10(self, real_cpu):
        """2^10 = 1024."""
        real_cpu.load_program("""
            MOV R0, 1
            MOV R1, 2
            MOV R2, 10
            MOV R3, 1
            MOV R4, 0
        loop:
            CMP R2, R4
            JZ done
            MUL R0, R0, R1
            SUB R2, R2, R3
            JMP loop
        done:
            HALT
        """)
        real_cpu.run()
        assert real_cpu.get_register("R0") == 1024


# =============================================================================
# ALL INSTRUCTION TYPES - Real LLM Decode
# =============================================================================

class TestRealModelAllInstructions:
    """Test program using all instruction types with real LLM decode."""

    def test_all_instruction_types(self, real_cpu):
        """Program using MOV, ADD, SUB, MUL, CMP, JMP, JZ, JNZ, JS, JNS, INC, DEC, NOP, HALT."""
        real_cpu.load_program("""
            ; Test all instruction types
            MOV R0, 10      ; MOV imm
            MOV R1, R0      ; MOV reg
            ADD R2, R0, R1  ; ADD -> 20
            SUB R3, R2, R0  ; SUB -> 10
            MUL R4, R3, R1  ; MUL -> 100
            INC R4          ; INC -> 101
            DEC R4          ; DEC -> 100
            NOP             ; NOP
            CMP R0, R1      ; CMP equal
            JZ equal_branch
            MOV R7, 0
            JMP skip1
        equal_branch:
            MOV R7, 1
        skip1:
            CMP R0, R2      ; R0 < R2
            JS less_branch
            JMP skip2
        less_branch:
            ADD R7, R7, R7  ; R7 = 2
        skip2:
            MOV R5, 0
            CMP R5, R0
            JNS skip3       ; 0 >= 10 is false
            ADD R7, R7, R7  ; R7 = 4
        skip3:
            MOV R6, 1
            CMP R6, R0
            JNZ skip4       ; 1 != 10
            MOV R7, 0
        skip4:
            HALT
        """)
        real_cpu.run()

        # Verify key registers
        assert real_cpu.get_register("R2") == 20
        assert real_cpu.get_register("R3") == 10
        assert real_cpu.get_register("R4") == 100
        assert real_cpu.get_register("R7") == 4  # 1 -> 2 -> 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
