"""Integration tests for example programs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from kvrm_cpu import KVRMCPU


class TestSumProgram:
    """Test sum_1_to_10.asm - calculates sum of 1 to 10."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True)

    def test_sum_1_to_10_result(self, cpu):
        """Sum of 1 to 10 should be 55."""
        program = """
            MOV R0, 0       ; sum = 0
            MOV R1, 1       ; counter = 1
            MOV R2, 11      ; limit
            MOV R3, 1       ; increment
        loop:
            ADD R0, R0, R1  ; sum += counter
            ADD R1, R1, R3  ; counter++
            CMP R1, R2      ; compare to limit
            JNZ loop        ; continue if not equal
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 55
        assert cpu.is_halted() is True

    def test_sum_correct_cycles(self, cpu):
        """Verify execution takes expected number of cycles."""
        program = """
            MOV R0, 0
            MOV R1, 1
            MOV R2, 11
            MOV R3, 1
        loop:
            ADD R0, R0, R1
            ADD R1, R1, R3
            CMP R1, R2
            JNZ loop
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        # 4 init + 10 * (4 loop ops) + 1 halt = 45 cycles
        assert cpu.get_cycle_count() == 45


class TestFibonacciProgram:
    """Test fibonacci.asm - calculates Nth Fibonacci number."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True)

    def test_fibonacci_10(self, cpu):
        """After 10 iterations: F(11) = 89."""
        program = """
            MOV R0, 0       ; fib(0)
            MOV R1, 1       ; fib(1)
            MOV R2, 10      ; N iterations
            MOV R3, 0       ; counter
            MOV R4, 1       ; constant 1
        loop:
            MOV R5, R1      ; temp = fib_curr
            ADD R1, R0, R1  ; fib_curr = fib_prev + fib_curr
            MOV R0, R5      ; fib_prev = temp
            ADD R3, R3, R4  ; counter++
            CMP R3, R2
            JNZ loop
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        # After 10 iterations starting from F(0)=0, F(1)=1, we get F(11)=89
        assert cpu.get_register("R1") == 89
        assert cpu.is_halted() is True


class TestMultiplyProgram:
    """Test multiply.asm - 7 * 6 via repeated addition."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True)

    def test_multiply_7_times_6(self, cpu):
        """7 * 6 should be 42."""
        program = """
            MOV R0, 0       ; result = 0
            MOV R1, 7       ; multiplicand
            MOV R2, 6       ; multiplier (countdown)
            MOV R3, 1       ; decrement constant
            MOV R4, 0       ; zero for comparison
        loop:
            ADD R0, R0, R1  ; result += multiplicand
            SUB R2, R2, R3  ; multiplier--
            CMP R2, R4      ; compare to zero
            JNZ loop        ; continue if not zero
            HALT
        """
        cpu.load_program(program)
        cpu.run()

        assert cpu.get_register("R0") == 42
        assert cpu.is_halted() is True


class TestSimplePrograms:
    """Test simple edge case programs."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True)

    def test_immediate_halt(self, cpu):
        """Program with just HALT."""
        cpu.load_program("HALT")
        cpu.run()

        assert cpu.is_halted() is True
        assert cpu.get_cycle_count() == 1

    def test_single_mov(self, cpu):
        """Single MOV then HALT."""
        cpu.load_program("""
            MOV R7, 123
            HALT
        """)
        cpu.run()

        assert cpu.get_register("R7") == 123
        assert cpu.get_cycle_count() == 2

    def test_register_swap(self, cpu):
        """Swap two registers using a third."""
        cpu.load_program("""
            MOV R0, 10
            MOV R1, 20
            MOV R2, R0      ; temp = R0
            MOV R0, R1      ; R0 = R1
            MOV R1, R2      ; R1 = temp
            HALT
        """)
        cpu.run()

        assert cpu.get_register("R0") == 20
        assert cpu.get_register("R1") == 10


class TestExecutionTrace:
    """Test execution trace functionality."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True)

    def test_trace_records_all_cycles(self, cpu):
        """Trace has entry for each cycle."""
        cpu.load_program("""
            MOV R0, 1
            MOV R1, 2
            HALT
        """)
        trace = cpu.run()

        assert len(trace) == 3
        assert trace[0].instruction == "MOV R0, 1"
        assert trace[1].instruction == "MOV R1, 2"
        assert trace[2].instruction == "HALT"

    def test_trace_captures_state_changes(self, cpu):
        """Trace captures pre and post state."""
        cpu.load_program("""
            MOV R0, 42
            HALT
        """)
        trace = cpu.run()

        # First instruction changes R0
        assert trace[0].pre_state["registers"]["R0"] == 0
        assert trace[0].post_state["registers"]["R0"] == 42


class TestMaxCyclesSafety:
    """Test max cycles safety limit."""

    def test_max_cycles_stops_execution(self):
        """Infinite loop stops at max cycles."""
        cpu = KVRMCPU(mock_mode=True, max_cycles=10)

        cpu.load_program("""
        loop:
            JMP loop
        """)

        with pytest.raises(RuntimeError, match="Max cycles"):
            cpu.run()

        assert cpu.get_cycle_count() == 10


class TestProgramFromFile:
    """Test loading programs from files."""

    @pytest.fixture
    def cpu(self):
        return KVRMCPU(mock_mode=True)

    def test_load_sum_file(self, cpu):
        """Load and run sum_1_to_10.asm from file."""
        program_path = Path(__file__).parent.parent / "programs" / "sum_1_to_10.asm"
        if program_path.exists():
            source = program_path.read_text()
            cpu.load_program(source)
            cpu.run()

            assert cpu.get_register("R0") == 55

    def test_load_fibonacci_file(self, cpu):
        """Load and run fibonacci.asm from file."""
        program_path = Path(__file__).parent.parent / "programs" / "fibonacci.asm"
        if program_path.exists():
            source = program_path.read_text()
            cpu.load_program(source)
            cpu.run()

            # After 10 iterations: F(11) = 89
            assert cpu.get_register("R1") == 89

    def test_load_multiply_file(self, cpu):
        """Load and run multiply.asm from file."""
        program_path = Path(__file__).parent.parent / "programs" / "multiply.asm"
        if program_path.exists():
            source = program_path.read_text()
            cpu.load_program(source)
            cpu.run()

            assert cpu.get_register("R0") == 42
