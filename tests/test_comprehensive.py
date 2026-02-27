"""Comprehensive test suite for KVRM-CPU.

This test suite validates KVRM-CPU with complex programs including:
- Mathematical algorithms (GCD, factorial, prime checking, powers)
- Nested loops and complex control flow
- Array-like operations and sorting
- Edge cases (negative numbers, zero, boundaries)
- Stress tests with long-running computations

Author: Bobby Price (blackWeb Research)
Date: January 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from kvrm_cpu import KVRMCPU


# =============================================================================
# MATHEMATICAL ALGORITHMS
# =============================================================================

class TestGCDAlgorithm:
    """Test Greatest Common Divisor using Euclidean algorithm."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=1000)

    def test_gcd_48_18(self, cpu):
        """GCD(48, 18) = 6 using Euclidean algorithm."""
        # Euclidean: 48 mod 18 = 12, 18 mod 12 = 6, 12 mod 6 = 0
        program = """
            ; GCD(R0, R1) -> R0
            ; Uses R2 as temp, R3 for zero check
            MOV R0, 48      ; a = 48
            MOV R1, 18      ; b = 18
            MOV R3, 0       ; zero constant
        gcd_loop:
            CMP R1, R3      ; if b == 0
            JZ gcd_done     ; done, result in R0
            ; compute a mod b using repeated subtraction
            MOV R2, R0      ; temp = a
        mod_loop:
            CMP R2, R1      ; while temp >= b
            JS mod_done     ; if temp < b, done with mod
            SUB R2, R2, R1  ; temp -= b
            JMP mod_loop
        mod_done:
            MOV R0, R1      ; a = b
            MOV R1, R2      ; b = a mod b
            JMP gcd_loop
        gcd_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 6
        assert cpu.is_halted()

    def test_gcd_105_45(self, cpu):
        """GCD(105, 45) = 15."""
        program = """
            MOV R0, 105
            MOV R1, 45
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 15

    def test_gcd_same_numbers(self, cpu):
        """GCD(24, 24) = 24."""
        program = """
            MOV R0, 24
            MOV R1, 24
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 24

    def test_gcd_coprime(self, cpu):
        """GCD(17, 13) = 1 (coprime numbers)."""
        program = """
            MOV R0, 17
            MOV R1, 13
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 1


class TestFactorialAlgorithm:
    """Test factorial computation."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=1000)

    def test_factorial_5(self, cpu):
        """5! = 120."""
        program = """
            ; factorial(N) -> R0
            ; N in R1, result accumulates in R0
            MOV R0, 1       ; result = 1
            MOV R1, 5       ; N = 5
            MOV R2, 1       ; decrement constant
            MOV R3, 1       ; compare constant
        fact_loop:
            CMP R1, R3      ; if N <= 1
            JS fact_done    ; done
            JZ fact_done    ; done if equal
            MUL R0, R0, R1  ; result *= N
            SUB R1, R1, R2  ; N--
            JMP fact_loop
        fact_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 120

    def test_factorial_6(self, cpu):
        """6! = 720."""
        program = """
            MOV R0, 1
            MOV R1, 6
            MOV R2, 1
            MOV R3, 1
        fact_loop:
            CMP R1, R3
            JS fact_done
            JZ fact_done
            MUL R0, R0, R1
            SUB R1, R1, R2
            JMP fact_loop
        fact_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 720

    def test_factorial_1(self, cpu):
        """1! = 1."""
        program = """
            MOV R0, 1
            MOV R1, 1
            MOV R2, 1
            MOV R3, 1
        fact_loop:
            CMP R1, R3
            JS fact_done
            JZ fact_done
            MUL R0, R0, R1
            SUB R1, R1, R2
            JMP fact_loop
        fact_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 1

    def test_factorial_0(self, cpu):
        """0! = 1 by definition."""
        program = """
            MOV R0, 1       ; result = 1 (0! = 1)
            MOV R1, 0       ; N = 0
            MOV R2, 1
            MOV R3, 1
        fact_loop:
            CMP R1, R3
            JS fact_done    ; 0 < 1, so done immediately
            JZ fact_done
            MUL R0, R0, R1
            SUB R1, R1, R2
            JMP fact_loop
        fact_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 1


class TestPowerAlgorithm:
    """Test exponentiation (a^n)."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=1000)

    def test_power_2_10(self, cpu):
        """2^10 = 1024."""
        program = """
            ; power(base, exp) -> R0
            ; base in R1, exp in R2, result in R0
            MOV R0, 1       ; result = 1
            MOV R1, 2       ; base = 2
            MOV R2, 10      ; exp = 10
            MOV R3, 1       ; decrement constant
            MOV R4, 0       ; zero for comparison
        pow_loop:
            CMP R2, R4      ; if exp == 0
            JZ pow_done     ; done
            MUL R0, R0, R1  ; result *= base
            SUB R2, R2, R3  ; exp--
            JMP pow_loop
        pow_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 1024

    def test_power_3_4(self, cpu):
        """3^4 = 81."""
        program = """
            MOV R0, 1
            MOV R1, 3
            MOV R2, 4
            MOV R3, 1
            MOV R4, 0
        pow_loop:
            CMP R2, R4
            JZ pow_done
            MUL R0, R0, R1
            SUB R2, R2, R3
            JMP pow_loop
        pow_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 81

    def test_power_5_3(self, cpu):
        """5^3 = 125."""
        program = """
            MOV R0, 1
            MOV R1, 5
            MOV R2, 3
            MOV R3, 1
            MOV R4, 0
        pow_loop:
            CMP R2, R4
            JZ pow_done
            MUL R0, R0, R1
            SUB R2, R2, R3
            JMP pow_loop
        pow_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 125

    def test_power_any_0(self, cpu):
        """Any number to the 0 power = 1."""
        program = """
            MOV R0, 1
            MOV R1, 7
            MOV R2, 0
            MOV R3, 1
            MOV R4, 0
        pow_loop:
            CMP R2, R4
            JZ pow_done
            MUL R0, R0, R1
            SUB R2, R2, R3
            JMP pow_loop
        pow_done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 1


class TestSumOfSquares:
    """Test sum of squares: 1^2 + 2^2 + ... + n^2."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=1000)

    def test_sum_squares_5(self, cpu):
        """1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55."""
        program = """
            ; Sum of squares 1 to N
            ; N in R1, result in R0
            MOV R0, 0       ; sum = 0
            MOV R1, 1       ; i = 1
            MOV R2, 5       ; N = 5
            MOV R3, 1       ; increment
            MOV R4, 0       ; temp for square
        loop:
            MUL R4, R1, R1  ; temp = i^2
            ADD R0, R0, R4  ; sum += i^2
            ADD R1, R1, R3  ; i++
            CMP R1, R2      ; compare i to N
            JS loop         ; if i < N, continue
            JZ last         ; if i == N, do one more
            JMP done
        last:
            MUL R4, R1, R1
            ADD R0, R0, R4
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 55

    def test_sum_squares_10(self, cpu):
        """1^2 + ... + 10^2 = 385."""
        program = """
            MOV R0, 0
            MOV R1, 1
            MOV R2, 10
            MOV R3, 1
            MOV R4, 0
        loop:
            MUL R4, R1, R1
            ADD R0, R0, R4
            ADD R1, R1, R3
            CMP R1, R2
            JS loop
            JZ last
            JMP done
        last:
            MUL R4, R1, R1
            ADD R0, R0, R4
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 385


# =============================================================================
# NESTED LOOPS
# =============================================================================

class TestNestedLoops:
    """Test programs with nested loop structures."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=2000)

    def test_multiplication_table_entry(self, cpu):
        """Calculate 7 * 8 via nested counting."""
        # Outer loop counts 7 times, inner loop adds 8 each time
        program = """
            MOV R0, 0       ; result
            MOV R1, 7       ; outer counter
            MOV R2, 8       ; inner limit
            MOV R3, 1       ; decrement/increment constant
            MOV R4, 0       ; zero for compare
        outer:
            CMP R1, R4
            JZ done
            MOV R5, R2      ; inner counter = 8
        inner:
            CMP R5, R4
            JZ inner_done
            ADD R0, R0, R3  ; result++
            SUB R5, R5, R3  ; inner--
            JMP inner
        inner_done:
            SUB R1, R1, R3  ; outer--
            JMP outer
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 56  # 7 * 8

    def test_triple_nested_loop(self, cpu):
        """Count iterations: 3 * 4 * 2 = 24."""
        program = """
            MOV R0, 0       ; counter
            MOV R1, 3       ; outer limit
            MOV R2, 4       ; middle limit
            MOV R3, 2       ; inner limit
            MOV R4, 1       ; constant
            MOV R5, 0       ; zero
        outer:
            CMP R1, R5
            JZ done
            MOV R6, R2      ; reset middle
        middle:
            CMP R6, R5
            JZ middle_done
            MOV R7, R3      ; reset inner
        inner:
            CMP R7, R5
            JZ inner_done
            ADD R0, R0, R4  ; count++
            SUB R7, R7, R4
            JMP inner
        inner_done:
            SUB R6, R6, R4
            JMP middle
        middle_done:
            SUB R1, R1, R4
            JMP outer
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 24  # 3 * 4 * 2


class TestComplexControlFlow:
    """Test programs with complex branching."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=500)

    def test_find_max_of_three(self, cpu):
        """Find maximum of three numbers: max(15, 42, 27) = 42."""
        program = """
            ; Find max of R0, R1, R2 -> store in R3
            MOV R0, 15
            MOV R1, 42
            MOV R2, 27

            ; Start with R0 as candidate max
            MOV R3, R0

            ; Compare with R1
            CMP R3, R1
            JNS skip1       ; if R3 >= R1, skip
            MOV R3, R1      ; R3 = R1
        skip1:
            ; Compare with R2
            CMP R3, R2
            JNS skip2
            MOV R3, R2
        skip2:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R3") == 42

    def test_find_min_of_three(self, cpu):
        """Find minimum of three numbers: min(15, 42, 27) = 15."""
        program = """
            MOV R0, 15
            MOV R1, 42
            MOV R2, 27
            MOV R3, R0

            CMP R1, R3
            JNS skip1       ; if R1 >= R3, skip
            MOV R3, R1
        skip1:
            CMP R2, R3
            JNS skip2
            MOV R3, R2
        skip2:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R3") == 15

    def test_absolute_difference(self, cpu):
        """Calculate |a - b| for any a, b."""
        # Test with a=10, b=25 -> |10-25| = 15
        program = """
            MOV R0, 10      ; a
            MOV R1, 25      ; b

            CMP R0, R1
            JS a_smaller    ; if a < b
            ; a >= b
            SUB R2, R0, R1
            JMP done
        a_smaller:
            SUB R2, R1, R0
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R2") == 15


class TestDivisionBySubtraction:
    """Test integer division using repeated subtraction."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=500)

    def test_divide_100_by_7(self, cpu):
        """100 / 7 = 14 remainder 2."""
        program = """
            ; dividend in R0, divisor in R1
            ; quotient in R2, remainder in R3
            MOV R0, 100     ; dividend
            MOV R1, 7       ; divisor
            MOV R2, 0       ; quotient = 0
            MOV R4, 1       ; increment
        div_loop:
            CMP R0, R1      ; if dividend < divisor
            JS div_done     ; done, remainder in R0
            SUB R0, R0, R1  ; dividend -= divisor
            ADD R2, R2, R4  ; quotient++
            JMP div_loop
        div_done:
            MOV R3, R0      ; remainder = what's left
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R2") == 14  # quotient
        assert cpu.get_register("R3") == 2   # remainder

    def test_divide_45_by_9(self, cpu):
        """45 / 9 = 5 remainder 0."""
        program = """
            MOV R0, 45
            MOV R1, 9
            MOV R2, 0
            MOV R4, 1
        div_loop:
            CMP R0, R1
            JS div_done
            SUB R0, R0, R1
            ADD R2, R2, R4
            JMP div_loop
        div_done:
            MOV R3, R0
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R2") == 5
        assert cpu.get_register("R3") == 0

    def test_divide_7_by_10(self, cpu):
        """7 / 10 = 0 remainder 7."""
        program = """
            MOV R0, 7
            MOV R1, 10
            MOV R2, 0
            MOV R4, 1
        div_loop:
            CMP R0, R1
            JS div_done
            SUB R0, R0, R1
            ADD R2, R2, R4
            JMP div_loop
        div_done:
            MOV R3, R0
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R2") == 0
        assert cpu.get_register("R3") == 7


# =============================================================================
# ARRAY-LIKE OPERATIONS (using registers as array elements)
# =============================================================================

class TestArrayOperations:
    """Test array-like operations using registers."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=500)

    def test_sum_array_elements(self, cpu):
        """Sum elements stored in R0-R4: 10+20+30+40+50 = 150."""
        program = """
            ; Store array elements
            MOV R0, 10
            MOV R1, 20
            MOV R2, 30
            MOV R3, 40
            MOV R4, 50

            ; Sum into R5
            MOV R5, 0
            ADD R5, R5, R0
            ADD R5, R5, R1
            ADD R5, R5, R2
            ADD R5, R5, R3
            ADD R5, R5, R4

            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R5") == 150

    def test_product_array_elements(self, cpu):
        """Product of R0-R3: 2*3*4*5 = 120."""
        program = """
            MOV R0, 2
            MOV R1, 3
            MOV R2, 4
            MOV R3, 5

            MOV R4, R0      ; product = R0
            MUL R4, R4, R1
            MUL R4, R4, R2
            MUL R4, R4, R3

            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R4") == 120

    def test_count_elements_greater_than(self, cpu):
        """Count elements > 25: R0=10, R1=30, R2=20, R3=40, R4=15 -> 2."""
        program = """
            MOV R0, 10
            MOV R1, 30
            MOV R2, 20
            MOV R3, 40
            MOV R4, 15
            MOV R5, 25      ; threshold
            MOV R6, 0       ; count
            MOV R7, 1       ; increment

            ; Check R0
            CMP R0, R5
            JS skip0
            JZ skip0
            ADD R6, R6, R7
        skip0:
            ; Check R1
            CMP R1, R5
            JS skip1
            JZ skip1
            ADD R6, R6, R7
        skip1:
            ; Check R2
            CMP R2, R5
            JS skip2
            JZ skip2
            ADD R6, R6, R7
        skip2:
            ; Check R3
            CMP R3, R5
            JS skip3
            JZ skip3
            ADD R6, R6, R7
        skip3:
            ; Check R4
            CMP R4, R5
            JS skip4
            JZ skip4
            ADD R6, R6, R7
        skip4:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R6") == 2  # R1=30 and R3=40 are > 25


class TestBubbleSortSimulation:
    """Test bubble sort on a small 'array' using registers."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=500)

    def test_sort_three_elements(self, cpu):
        """Sort R0, R1, R2: [30, 10, 20] -> [10, 20, 30]."""
        program = """
            MOV R0, 30
            MOV R1, 10
            MOV R2, 20
            MOV R3, 0       ; temp

            ; Pass 1: Compare adjacent pairs
            ; Compare R0, R1
            CMP R0, R1
            JS no_swap01    ; if R0 < R1, don't swap
            JZ no_swap01
            ; Swap R0, R1
            MOV R3, R0
            MOV R0, R1
            MOV R1, R3
        no_swap01:
            ; Compare R1, R2
            CMP R1, R2
            JS no_swap12
            JZ no_swap12
            MOV R3, R1
            MOV R1, R2
            MOV R2, R3
        no_swap12:
            ; Pass 2: One more pass to ensure sorted
            CMP R0, R1
            JS no_swap01b
            JZ no_swap01b
            MOV R3, R0
            MOV R0, R1
            MOV R1, R3
        no_swap01b:
            CMP R1, R2
            JS done
            JZ done
            MOV R3, R1
            MOV R1, R2
            MOV R2, R3
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 10
        assert cpu.get_register("R1") == 20
        assert cpu.get_register("R2") == 30

    def test_sort_already_sorted(self, cpu):
        """Already sorted [10, 20, 30] should stay the same."""
        program = """
            MOV R0, 10
            MOV R1, 20
            MOV R2, 30
            MOV R3, 0

            CMP R0, R1
            JS no_swap01
            JZ no_swap01
            MOV R3, R0
            MOV R0, R1
            MOV R1, R3
        no_swap01:
            CMP R1, R2
            JS no_swap12
            JZ no_swap12
            MOV R3, R1
            MOV R1, R2
            MOV R2, R3
        no_swap12:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 10
        assert cpu.get_register("R1") == 20
        assert cpu.get_register("R2") == 30

    def test_sort_reverse_order(self, cpu):
        """Reverse [30, 20, 10] -> [10, 20, 30]."""
        program = """
            MOV R0, 30
            MOV R1, 20
            MOV R2, 10
            MOV R3, 0

            ; Pass 1
            CMP R0, R1
            JS p1_01
            JZ p1_01
            MOV R3, R0
            MOV R0, R1
            MOV R1, R3
        p1_01:
            CMP R1, R2
            JS p1_12
            JZ p1_12
            MOV R3, R1
            MOV R1, R2
            MOV R2, R3
        p1_12:
            ; Pass 2
            CMP R0, R1
            JS p2_01
            JZ p2_01
            MOV R3, R0
            MOV R0, R1
            MOV R1, R3
        p2_01:
            CMP R1, R2
            JS done
            JZ done
            MOV R3, R1
            MOV R1, R2
            MOV R2, R3
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 10
        assert cpu.get_register("R1") == 20
        assert cpu.get_register("R2") == 30


# =============================================================================
# FIBONACCI VARIANTS
# =============================================================================

class TestFibonacciVariants:
    """Test various Fibonacci computations."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=1000)

    def test_fibonacci_5(self, cpu):
        """After 5 iterations: F(6) = 8."""
        program = """
            MOV R0, 0
            MOV R1, 1
            MOV R2, 5
            MOV R3, 0
            MOV R4, 1
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
        """
        cpu.load_program(program)
        cpu.run()

        # After 5 iterations starting from F(0)=0, F(1)=1: F(6) = 8
        assert cpu.get_register("R1") == 8

    def test_fibonacci_7(self, cpu):
        """After 7 iterations: F(8) = 21."""
        program = """
            MOV R0, 0
            MOV R1, 1
            MOV R2, 7
            MOV R3, 0
            MOV R4, 1
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R1") == 21

    def test_fibonacci_12(self, cpu):
        """After 12 iterations: F(13) = 233."""
        program = """
            MOV R0, 0
            MOV R1, 1
            MOV R2, 12
            MOV R3, 0
            MOV R4, 1
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R1") == 233


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=500)

    def test_zero_operations(self, cpu):
        """Operations with zero."""
        program = """
            MOV R0, 0
            MOV R1, 42
            ADD R2, R0, R1      ; 0 + 42 = 42
            SUB R3, R1, R0      ; 42 - 0 = 42
            MUL R4, R0, R1      ; 0 * 42 = 0
            MUL R5, R1, R0      ; 42 * 0 = 0
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R2") == 42
        assert cpu.get_register("R3") == 42
        assert cpu.get_register("R4") == 0
        assert cpu.get_register("R5") == 0

    def test_identity_operations(self, cpu):
        """Identity operations (add 0, multiply by 1)."""
        program = """
            MOV R0, 100
            MOV R1, 0
            MOV R2, 1

            ADD R3, R0, R1      ; 100 + 0 = 100
            MUL R4, R0, R2      ; 100 * 1 = 100

            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R3") == 100
        assert cpu.get_register("R4") == 100

    def test_compare_equal(self, cpu):
        """Compare equal numbers."""
        program = """
            MOV R0, 50
            MOV R1, 50
            MOV R2, 0

            CMP R0, R1
            JZ are_equal
            MOV R2, 1       ; not equal
            JMP done
        are_equal:
            MOV R2, 100     ; equal
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R2") == 100  # Equal path taken

    def test_large_numbers(self, cpu):
        """Operations with larger numbers."""
        program = """
            MOV R0, 1000
            MOV R1, 999
            ADD R2, R0, R1      ; 1999
            MUL R3, R0, R1      ; 999000
            SUB R4, R0, R1      ; 1
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R2") == 1999
        assert cpu.get_register("R3") == 999000
        assert cpu.get_register("R4") == 1

    def test_self_operations(self, cpu):
        """Operations with same register."""
        program = """
            MOV R0, 5
            ADD R0, R0, R0      ; 5 + 5 = 10
            MUL R0, R0, R0      ; 10 * 10 = 100
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 100

    def test_increment_decrement_sequence(self, cpu):
        """Complex INC/DEC sequence."""
        program = """
            MOV R0, 10
            INC R0          ; 11
            INC R0          ; 12
            INC R0          ; 13
            DEC R0          ; 12
            INC R0          ; 13
            DEC R0          ; 12
            DEC R0          ; 11
            DEC R0          ; 10
            DEC R0          ; 9
            INC R0          ; 10
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 10


class TestFlagBehavior:
    """Test CPU flag behavior."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=100)

    def test_zero_flag_on_equal(self, cpu):
        """ZF should be set when values are equal."""
        program = """
            MOV R0, 42
            MOV R1, 42
            CMP R0, R1
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        flags = cpu.get_flags()
        assert flags["ZF"] == True
        assert flags["SF"] == False

    def test_sign_flag_on_less(self, cpu):
        """SF should be set when first < second."""
        program = """
            MOV R0, 10
            MOV R1, 20
            CMP R0, R1
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        flags = cpu.get_flags()
        assert flags["ZF"] == False
        assert flags["SF"] == True

    def test_flags_on_greater(self, cpu):
        """No flags set when first > second."""
        program = """
            MOV R0, 30
            MOV R1, 20
            CMP R0, R1
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        flags = cpu.get_flags()
        assert flags["ZF"] == False
        assert flags["SF"] == False


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStressTests:
    """Stress tests with longer computations."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=5000)

    def test_count_to_100(self, cpu):
        """Count from 0 to 100."""
        program = """
            MOV R0, 0       ; counter
            MOV R1, 100     ; target
            MOV R2, 1       ; increment
        loop:
            CMP R0, R1
            JZ done
            ADD R0, R0, R2
            JMP loop
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 100

    def test_sum_1_to_50(self, cpu):
        """Sum of 1 to 50 = 1275."""
        program = """
            MOV R0, 0       ; sum
            MOV R1, 1       ; counter
            MOV R2, 51      ; limit
            MOV R3, 1       ; increment
        loop:
            ADD R0, R0, R1
            ADD R1, R1, R3
            CMP R1, R2
            JNZ loop
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 1275

    def test_power_2_15(self, cpu):
        """2^15 = 32768."""
        program = """
            MOV R0, 1
            MOV R1, 2
            MOV R2, 15
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 32768

    def test_fibonacci_15(self, cpu):
        """After 15 iterations: F(16) = 987."""
        program = """
            MOV R0, 0
            MOV R1, 1
            MOV R2, 15
            MOV R3, 0
            MOV R4, 1
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R1") == 987


# =============================================================================
# REAL-WORLD ALGORITHMS
# =============================================================================

class TestRealWorldAlgorithms:
    """Test implementations of real-world algorithms."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=2000)

    def test_collatz_sequence_length(self, cpu):
        """Count Collatz sequence steps for n=7 (should be 16 steps to reach 1)."""
        # Collatz: if even, n/2; if odd, 3n+1
        # For 7: 7->22->11->34->17->52->26->13->40->20->10->5->16->8->4->2->1
        program = """
            MOV R0, 7       ; starting number
            MOV R1, 0       ; step counter
            MOV R2, 1       ; constant 1
            MOV R3, 2       ; constant 2
            MOV R4, 3       ; constant 3
        loop:
            CMP R0, R2      ; if n == 1
            JZ done
            ADD R1, R1, R2  ; steps++

            ; Check if even (using mod 2 via subtraction)
            MOV R5, R0
        mod2:
            CMP R5, R3
            JS is_even_check    ; if remainder < 2, check
            SUB R5, R5, R3      ; subtract 2
            JMP mod2
        is_even_check:
            CMP R5, R2          ; if remainder == 1, odd
            JZ is_odd
            ; Even: n = n / 2 (using repeated subtraction)
            MOV R5, 0           ; quotient
            MOV R6, R0          ; dividend
        div2:
            CMP R6, R3
            JS div2_done
            SUB R6, R6, R3
            ADD R5, R5, R2
            JMP div2
        div2_done:
            MOV R0, R5
            JMP loop
        is_odd:
            ; Odd: n = 3n + 1
            MUL R0, R0, R4
            ADD R0, R0, R2
            JMP loop
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R1") == 16


class TestPrimeCheck:
    """Test prime number checking."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=2000)

    def test_is_prime_7(self, cpu):
        """7 is prime (R7 = 1)."""
        # Check divisibility from 2 to sqrt(n), simplified here
        program = """
            MOV R0, 7       ; number to check
            MOV R1, 2       ; divisor
            MOV R2, 1       ; constant 1
            MOV R7, 1       ; assume prime
        check:
            ; if divisor >= number, it's prime
            CMP R1, R0
            JNS is_prime    ; divisor >= n, prime
            JZ is_prime

            ; Check if n mod divisor == 0
            MOV R3, R0      ; dividend
        mod_loop:
            CMP R3, R1
            JS mod_done
            SUB R3, R3, R1
            JMP mod_loop
        mod_done:
            ; R3 is remainder
            MOV R4, 0
            CMP R3, R4
            JZ not_prime    ; divisible, not prime

            ADD R1, R1, R2  ; next divisor
            JMP check

        not_prime:
            MOV R7, 0
        is_prime:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R7") == 1

    def test_is_not_prime_9(self, cpu):
        """9 is not prime (divisible by 3)."""
        program = """
            MOV R0, 9
            MOV R1, 2
            MOV R2, 1
            MOV R7, 1
        check:
            CMP R1, R0
            JNS is_prime
            JZ is_prime

            MOV R3, R0
        mod_loop:
            CMP R3, R1
            JS mod_done
            SUB R3, R3, R1
            JMP mod_loop
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R7") == 0

    def test_is_prime_13(self, cpu):
        """13 is prime."""
        program = """
            MOV R0, 13
            MOV R1, 2
            MOV R2, 1
            MOV R7, 1
        check:
            CMP R1, R0
            JNS is_prime
            JZ is_prime

            MOV R3, R0
        mod_loop:
            CMP R3, R1
            JS mod_done
            SUB R3, R3, R1
            JMP mod_loop
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
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R7") == 1


# =============================================================================
# SUMMARY TEST
# =============================================================================

class TestComprehensiveSummary:
    """Final comprehensive test combining multiple concepts."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True, max_cycles=5000)

    def test_complex_computation(self, cpu):
        """
        Complex computation:
        1. Compute factorial(5) = 120
        2. Divide by 4 to get 30
        3. Add 25 to get 55
        4. Verify this equals sum(1..10)
        """
        program = """
            ; Step 1: factorial(5) -> R0
            MOV R0, 1
            MOV R1, 5
            MOV R2, 1
            MOV R3, 1
        fact_loop:
            CMP R1, R3
            JS fact_done
            JZ fact_done
            MUL R0, R0, R1
            SUB R1, R1, R2
            JMP fact_loop
        fact_done:
            ; R0 = 120

            ; Step 2: divide by 4 -> R4
            MOV R1, 4
            MOV R4, 0
        div_loop:
            CMP R0, R1
            JS div_done
            SUB R0, R0, R1
            ADD R4, R4, R2
            JMP div_loop
        div_done:
            ; R4 = 30

            ; Step 3: add 25
            MOV R5, 25
            ADD R4, R4, R5
            ; R4 = 55

            ; Step 4: compute sum(1..10) in R6
            MOV R6, 0
            MOV R1, 1
            MOV R7, 11
        sum_loop:
            ADD R6, R6, R1
            ADD R1, R1, R2
            CMP R1, R7
            JNZ sum_loop
            ; R6 = 55

            ; Verify R4 == R6
            CMP R4, R6
            JZ verified
            MOV R7, 0       ; mismatch
            JMP done
        verified:
            MOV R7, 1       ; match!
        done:
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R4") == 55
        assert cpu.get_register("R6") == 55
        assert cpu.get_register("R7") == 1  # Verified match
