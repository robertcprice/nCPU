"""Integration tests: neural execution mode vs mock mode.

Verifies that programs produce identical results whether running through
Python arithmetic (mock) or trained neural networks (neural).
"""

from pathlib import Path
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ncpu.model import CPU


PROGRAMS_DIR = Path(__file__).parent.parent.parent / "programs"


# ═══════════════════════════════════════════════════════════════════════════════
# Neural vs Mock Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralVsMock:
    """Run every program in both mock and neural modes, assert identical registers."""

    def _run_both_modes(self, source: str):
        """Run a program in mock and neural mode, return both register dumps."""
        mock_cpu = CPU(neural_execution=False)
        mock_cpu.load_program(source)
        mock_cpu.run()

        neural_cpu = CPU(neural_execution=True)
        neural_cpu.load_program(source)
        neural_cpu.run()

        return mock_cpu.dump_registers(), neural_cpu.dump_registers()

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_sum_1_to_10(self):
        source = """
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
        mock_regs, neural_regs = self._run_both_modes(source)
        assert mock_regs == neural_regs
        assert mock_regs["R0"] == 55

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_fibonacci(self):
        source = """
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
        """
        mock_regs, neural_regs = self._run_both_modes(source)
        assert mock_regs == neural_regs
        assert mock_regs["R1"] == 89

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_multiply_loop(self):
        source = """
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
        """
        mock_regs, neural_regs = self._run_both_modes(source)
        assert mock_regs == neural_regs
        assert mock_regs["R0"] == 42

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_inc_dec(self):
        source = """
            MOV R0, 5
            INC R0
            INC R0
            DEC R0
            HALT
        """
        mock_regs, neural_regs = self._run_both_modes(source)
        assert mock_regs == neural_regs
        assert mock_regs["R0"] == 6

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_register_swap(self):
        source = """
            MOV R0, 10
            MOV R1, 20
            MOV R2, R0
            MOV R0, R1
            MOV R1, R2
            HALT
        """
        mock_regs, neural_regs = self._run_both_modes(source)
        assert mock_regs == neural_regs

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_from_asm_files(self):
        """Run all .asm files in programs/ and verify mock == neural."""
        for asm_file in sorted(PROGRAMS_DIR.rglob("*.asm")):
            source = asm_file.read_text()
            mock_cpu = CPU(neural_execution=False)
            mock_cpu.load_program(source)
            mock_cpu.run()

            neural_cpu = CPU(neural_execution=True)
            neural_cpu.load_program(source)
            neural_cpu.run()

            assert mock_cpu.dump_registers() == neural_cpu.dump_registers(), \
                f"Mismatch in {asm_file.name}"


# ═══════════════════════════════════════════════════════════════════════════════
# Bitwise Programs (Neural)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralBitwiseProgram:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_bitwise_and_or_xor(self):
        cpu = CPU(neural_execution=True)
        cpu.load_program("""
            MOV R0, 0xFF
            MOV R1, 0x0F
            AND R2, R0, R1
            OR  R3, R0, R1
            XOR R4, R0, R1
            HALT
        """)
        cpu.run()
        assert cpu.get_register("R2") == 0x0F   # AND
        assert cpu.get_register("R3") == 0xFF   # OR
        assert cpu.get_register("R4") == 0xF0   # XOR

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_shift_program(self):
        cpu = CPU(neural_execution=True)
        cpu.load_program("""
            MOV R0, 1
            SHL R1, R0, 3
            MOV R2, 16
            SHR R3, R2, 2
            HALT
        """)
        cpu.run()
        assert cpu.get_register("R1") == 8   # 1 << 3
        assert cpu.get_register("R3") == 4   # 16 >> 2

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_bitwise_file(self):
        path = PROGRAMS_DIR / "bitwise.asm"
        if path.exists():
            source = path.read_text()
            mock_cpu = CPU(neural_execution=False)
            mock_cpu.load_program(source)
            mock_cpu.run()

            neural_cpu = CPU(neural_execution=True)
            neural_cpu.load_program(source)
            neural_cpu.run()

            assert mock_cpu.dump_registers() == neural_cpu.dump_registers()

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_power_of_two_file(self):
        path = PROGRAMS_DIR / "power_of_two.asm"
        if path.exists():
            source = path.read_text()
            mock_cpu = CPU(neural_execution=False)
            mock_cpu.load_program(source)
            mock_cpu.run()

            neural_cpu = CPU(neural_execution=True)
            neural_cpu.load_program(source)
            neural_cpu.run()

            assert mock_cpu.dump_registers() == neural_cpu.dump_registers()
            # 2^8 = 256
            assert mock_cpu.get_register("R0") == 256
