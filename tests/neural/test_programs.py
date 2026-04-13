"""Integration tests for assembly programs."""

from pathlib import Path
import pytest
from ncpu.model import CPU


class TestSumProgram:
    @pytest.fixture
    def cpu(self):
        return CPU()

    def test_sum_1_to_10(self, cpu):
        cpu.load_program("""
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
        """)
        cpu.run()
        assert cpu.get_register("R0") == 55
        assert cpu.is_halted()

    def test_sum_correct_cycles(self, cpu):
        cpu.load_program("""
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
        """)
        cpu.run()
        assert cpu.get_cycle_count() == 45


class TestFibonacciProgram:
    def test_fibonacci_10(self):
        cpu = CPU()
        cpu.load_program("""
            MOV R0, 0
            MOV R1, 1
            MOV R2, 10
            MOV R3, 0
            MOV R4, 1
        loop:
            MOV R5, R1
            ADD R1, R0, R1
            MOV R0, R5
            ADD R3, R3, R4
            CMP R3, R2
            JNZ loop
            HALT
        """)
        cpu.run()
        assert cpu.get_register("R1") == 89
        assert cpu.is_halted()


class TestMultiplyProgram:
    def test_multiply_7_times_6(self):
        cpu = CPU()
        cpu.load_program("""
            MOV R0, 0
            MOV R1, 7
            MOV R2, 6
            MOV R3, 1
            MOV R4, 0
        loop:
            ADD R0, R0, R1
            SUB R2, R2, R3
            CMP R2, R4
            JNZ loop
            HALT
        """)
        cpu.run()
        assert cpu.get_register("R0") == 42
        assert cpu.is_halted()


class TestSimplePrograms:
    @pytest.fixture
    def cpu(self):
        return CPU()

    def test_immediate_halt(self, cpu):
        cpu.load_program("HALT")
        cpu.run()
        assert cpu.is_halted() and cpu.get_cycle_count() == 1

    def test_single_mov(self, cpu):
        cpu.load_program("MOV R7, 123\nHALT")
        cpu.run()
        assert cpu.get_register("R7") == 123 and cpu.get_cycle_count() == 2

    def test_register_swap(self, cpu):
        cpu.load_program("""
            MOV R0, 10
            MOV R1, 20
            MOV R2, R0
            MOV R0, R1
            MOV R1, R2
            HALT
        """)
        cpu.run()
        assert cpu.get_register("R0") == 20
        assert cpu.get_register("R1") == 10

    def test_inc_dec(self, cpu):
        cpu.load_program("""
            MOV R0, 5
            INC R0
            INC R0
            DEC R0
            HALT
        """)
        cpu.run()
        assert cpu.get_register("R0") == 6


class TestExecutionTrace:
    def test_trace_records_all_cycles(self):
        cpu = CPU()
        cpu.load_program("MOV R0, 1\nMOV R1, 2\nHALT")
        trace = cpu.run()
        assert len(trace) == 3
        assert trace[0].instruction == "MOV R0, 1"
        assert trace[2].instruction == "HALT"

    def test_trace_captures_state_changes(self):
        cpu = CPU()
        cpu.load_program("MOV R0, 42\nHALT")
        trace = cpu.run()
        assert trace[0].pre_state["registers"]["R0"] == 0
        assert trace[0].post_state["registers"]["R0"] == 42


class TestMaxCyclesSafety:
    def test_max_cycles_stops_execution(self):
        cpu = CPU(max_cycles=10)
        cpu.load_program("loop:\n    JMP loop")
        with pytest.raises(RuntimeError, match="Max cycles"):
            cpu.run()
        assert cpu.get_cycle_count() == 10


class TestProgramFromFile:
    @pytest.fixture
    def cpu(self):
        return CPU()

    def test_load_sum_file(self, cpu):
        path = Path(__file__).parent.parent.parent / "programs" / "arithmetic" / "sum_1_to_10.asm"
        if path.exists():
            cpu.load_program(path.read_text())
            cpu.run()
            assert cpu.get_register("R0") == 55

    def test_load_fibonacci_file(self, cpu):
        path = Path(__file__).parent.parent.parent / "programs" / "arithmetic" / "fibonacci.asm"
        if path.exists():
            cpu.load_program(path.read_text())
            cpu.run()
            assert cpu.get_register("R1") == 89

    def test_load_multiply_file(self, cpu):
        path = Path(__file__).parent.parent.parent / "programs" / "arithmetic" / "multiply.asm"
        if path.exists():
            cpu.load_program(path.read_text())
            cpu.run()
            assert cpu.get_register("R0") == 42
