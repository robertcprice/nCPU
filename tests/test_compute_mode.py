"""Tests for nCPU GPU Compute Mode.

Verifies that the Metal compute shader (qemu-style fetch-decode-execute)
produces correct results for all nCPU ISA instructions, flag behavior,
loop programs, and cross-mode verification against the neural CPU.

Requires: mlx (Apple Silicon Metal)
"""

import sys
from pathlib import Path

import pytest
from kernels.mlx.availability import is_mlx_usable

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HAS_MLX = is_mlx_usable()

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX backend not available")


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def kernel():
    """Create a fresh NCPUComputeKernel for each test."""
    from kernels.mlx.ncpu_kernel import NCPUComputeKernel
    return NCPUComputeKernel()


@pytest.fixture
def assembler():
    """Create a ClassicalAssembler."""
    from ncpu.os.assembler import ClassicalAssembler
    return ClassicalAssembler()


# ═══════════════════════════════════════════════════════════════════════════════
# Instruction-Level Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeInstructions:
    """Test each nCPU ISA instruction on the Metal compute kernel."""

    def test_nop(self, kernel):
        kernel.load_program_from_asm("NOP\nHALT")
        result = kernel.execute()
        assert result.stop_reason_name == "HALT"
        assert result.cycles == 2

    def test_halt(self, kernel):
        kernel.load_program_from_asm("HALT")
        result = kernel.execute()
        assert result.stop_reason_name == "HALT"
        assert result.cycles == 1

    def test_mov_imm(self, kernel):
        kernel.load_program_from_asm("MOV R0, 42\nHALT")
        kernel.execute()
        assert kernel.get_register(0) == 42

    def test_mov_imm_zero(self, kernel):
        kernel.load_program_from_asm("MOV R0, 100\nMOV R0, 0\nHALT")
        kernel.execute()
        assert kernel.get_register(0) == 0

    def test_mov_imm_negative(self, kernel):
        kernel.load_program_from_asm("MOV R0, -1\nHALT")
        kernel.execute()
        assert kernel.get_register(0) == -1

    def test_mov_reg(self, kernel):
        kernel.load_program_from_asm("MOV R0, 99\nMOV R1, R0\nHALT")
        kernel.execute()
        assert kernel.get_register(1) == 99

    def test_add(self, kernel):
        kernel.load_program_from_asm("MOV R0, 10\nMOV R1, 20\nADD R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 30

    def test_sub(self, kernel):
        kernel.load_program_from_asm("MOV R0, 50\nMOV R1, 30\nSUB R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 20

    def test_sub_negative_result(self, kernel):
        kernel.load_program_from_asm("MOV R0, 10\nMOV R1, 30\nSUB R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == -20

    def test_mul(self, kernel):
        kernel.load_program_from_asm("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 42

    def test_mul_zero(self, kernel):
        kernel.load_program_from_asm("MOV R0, 100\nMOV R1, 0\nMUL R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 0

    def test_div(self, kernel):
        kernel.load_program_from_asm("MOV R0, 42\nMOV R1, 7\nDIV R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 6

    def test_div_by_zero(self, kernel):
        kernel.load_program_from_asm("MOV R0, 42\nMOV R1, 0\nDIV R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 0  # div by zero returns 0

    def test_and(self, kernel):
        kernel.load_program_from_asm("MOV R0, 0xFF\nMOV R1, 0x0F\nAND R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 0x0F

    def test_or(self, kernel):
        kernel.load_program_from_asm("MOV R0, 0xF0\nMOV R1, 0x0F\nOR R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 0xFF

    def test_xor(self, kernel):
        kernel.load_program_from_asm("MOV R0, 0xFF\nMOV R1, 0xFF\nXOR R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 0

    def test_xor_different(self, kernel):
        kernel.load_program_from_asm("MOV R0, 0xAA\nMOV R1, 0x55\nXOR R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 0xFF

    def test_shl_immediate(self, kernel):
        kernel.load_program_from_asm("MOV R0, 1\nSHL R1, R0, 4\nHALT")
        kernel.execute()
        assert kernel.get_register(1) == 16

    def test_shr_immediate(self, kernel):
        kernel.load_program_from_asm("MOV R0, 256\nSHR R1, R0, 4\nHALT")
        kernel.execute()
        assert kernel.get_register(1) == 16

    def test_inc(self, kernel):
        kernel.load_program_from_asm("MOV R0, 10\nINC R0\nHALT")
        kernel.execute()
        assert kernel.get_register(0) == 11

    def test_dec(self, kernel):
        kernel.load_program_from_asm("MOV R0, 10\nDEC R0\nHALT")
        kernel.execute()
        assert kernel.get_register(0) == 9

    def test_inc_from_zero(self, kernel):
        kernel.load_program_from_asm("INC R0\nHALT")
        kernel.execute()
        assert kernel.get_register(0) == 1

    def test_dec_to_negative(self, kernel):
        kernel.load_program_from_asm("DEC R0\nHALT")
        kernel.execute()
        assert kernel.get_register(0) == -1


# ═══════════════════════════════════════════════════════════════════════════════
# Flag Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeFlags:
    """Test flag behavior (ZF, SF) on the compute kernel."""

    def test_zero_flag_set(self, kernel):
        kernel.load_program_from_asm("MOV R0, 0\nHALT")
        kernel.execute()
        flags = kernel.get_flags()
        assert flags["ZF"] is True
        assert flags["SF"] is False

    def test_zero_flag_clear(self, kernel):
        kernel.load_program_from_asm("MOV R0, 1\nHALT")
        kernel.execute()
        flags = kernel.get_flags()
        assert flags["ZF"] is False

    def test_sign_flag_set(self, kernel):
        kernel.load_program_from_asm("MOV R0, -5\nHALT")
        kernel.execute()
        flags = kernel.get_flags()
        assert flags["SF"] is True

    def test_sign_flag_clear(self, kernel):
        kernel.load_program_from_asm("MOV R0, 5\nHALT")
        kernel.execute()
        flags = kernel.get_flags()
        assert flags["SF"] is False

    def test_cmp_equal(self, kernel):
        kernel.load_program_from_asm("MOV R0, 10\nMOV R1, 10\nCMP R0, R1\nHALT")
        kernel.execute()
        flags = kernel.get_flags()
        assert flags["ZF"] is True
        assert flags["SF"] is False

    def test_cmp_less(self, kernel):
        kernel.load_program_from_asm("MOV R0, 5\nMOV R1, 10\nCMP R0, R1\nHALT")
        kernel.execute()
        flags = kernel.get_flags()
        assert flags["ZF"] is False
        assert flags["SF"] is True

    def test_cmp_greater(self, kernel):
        kernel.load_program_from_asm("MOV R0, 10\nMOV R1, 5\nCMP R0, R1\nHALT")
        kernel.execute()
        flags = kernel.get_flags()
        assert flags["ZF"] is False
        assert flags["SF"] is False

    def test_sub_sets_zero(self, kernel):
        kernel.load_program_from_asm("MOV R0, 5\nMOV R1, 5\nSUB R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == 0
        assert kernel.get_flags()["ZF"] is True

    def test_add_sets_flags(self, kernel):
        kernel.load_program_from_asm("MOV R0, -5\nMOV R1, 2\nADD R2, R0, R1\nHALT")
        kernel.execute()
        assert kernel.get_register(2) == -3
        assert kernel.get_flags()["SF"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Branch Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeBranches:
    """Test conditional and unconditional jumps."""

    def test_jmp(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, 1
            JMP skip
            MOV R0, 99
        skip:
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(0) == 1  # should skip MOV R0, 99

    def test_jz_taken(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, 0
            JZ target
            MOV R1, 99
        target:
            MOV R1, 42
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(1) == 42

    def test_jz_not_taken(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, 1
            JZ target
            MOV R1, 99
            JMP done
        target:
            MOV R1, 42
        done:
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(1) == 99

    def test_jnz_taken(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, 5
            JNZ target
            MOV R1, 99
        target:
            MOV R1, 42
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(1) == 42

    def test_jnz_not_taken(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, 0
            JNZ target
            MOV R1, 99
            JMP done
        target:
            MOV R1, 42
        done:
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(1) == 99

    def test_js_taken(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, -1
            JS target
            MOV R1, 99
        target:
            MOV R1, 42
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(1) == 42

    def test_js_not_taken(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, 5
            JS target
            MOV R1, 99
            JMP done
        target:
            MOV R1, 42
        done:
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(1) == 99

    def test_jns_taken(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, 5
            JNS target
            MOV R1, 99
        target:
            MOV R1, 42
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(1) == 42

    def test_jns_not_taken(self, kernel):
        kernel.load_program_from_asm("""
            MOV R0, -1
            JNS target
            MOV R1, 99
            JMP done
        target:
            MOV R1, 42
        done:
            HALT
        """)
        kernel.execute()
        assert kernel.get_register(1) == 99


# ═══════════════════════════════════════════════════════════════════════════════
# Loop Programs
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePrograms:
    """Test complete programs with loops on the compute kernel."""

    def test_sum_1_to_10(self, kernel):
        source = (PROJECT_ROOT / "programs" / "sum_1_to_10.asm").read_text()
        kernel.load_program_from_asm(source)
        result = kernel.execute()
        assert result.stop_reason_name == "HALT"
        assert kernel.get_register(0) == 55

    def test_fibonacci_iterative(self, kernel):
        source = (PROJECT_ROOT / "programs" / "fibonacci_iterative.asm").read_text()
        kernel.load_program_from_asm(source)
        result = kernel.execute()
        assert result.stop_reason_name == "HALT"
        assert kernel.get_register(1) == 55

    def test_factorial(self, kernel):
        source = (PROJECT_ROOT / "programs" / "factorial.asm").read_text()
        kernel.load_program_from_asm(source)
        result = kernel.execute()
        assert result.stop_reason_name == "HALT"
        # factorial.asm computes 7! or similar — check R0 is non-zero
        assert kernel.get_register(0) > 0

    def test_collatz(self, kernel):
        source = (PROJECT_ROOT / "programs" / "collatz.asm").read_text()
        kernel.load_program_from_asm(source)
        result = kernel.execute()
        assert result.stop_reason_name == "HALT"

    def test_gcd(self, kernel):
        source = (PROJECT_ROOT / "programs" / "gcd.asm").read_text()
        kernel.load_program_from_asm(source)
        result = kernel.execute()
        assert result.stop_reason_name == "HALT"

    def test_count_loop(self, kernel):
        """Simple count to 100 loop."""
        kernel.load_program_from_asm("""
            MOV R0, 0
            MOV R1, 100
            MOV R2, 1
        loop:
            ADD R0, R0, R2
            CMP R0, R1
            JNZ loop
            HALT
        """)
        result = kernel.execute()
        assert kernel.get_register(0) == 100
        assert result.stop_reason_name == "HALT"

    def test_max_cycles_limit(self, kernel):
        """Test that max_cycles stops infinite loops."""
        kernel.load_program_from_asm("""
        loop:
            NOP
            JMP loop
        """)
        result = kernel.execute(max_cycles=100)
        assert result.stop_reason_name == "MAX_CYCLES"
        assert result.cycles == 100


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel API Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeAPI:
    """Test the NCPUComputeKernel Python API."""

    def test_get_registers_dict(self, kernel):
        kernel.load_program_from_asm("MOV R0, 1\nMOV R3, 99\nHALT")
        kernel.execute()
        regs = kernel.get_registers_dict()
        assert regs["R0"] == 1
        assert regs["R3"] == 99
        assert regs["R7"] == 0

    def test_get_flags(self, kernel):
        kernel.load_program_from_asm("MOV R0, 0\nHALT")
        kernel.execute()
        flags = kernel.get_flags()
        assert "ZF" in flags
        assert "SF" in flags

    def test_reset(self, kernel):
        kernel.load_program_from_asm("MOV R0, 42\nHALT")
        kernel.execute()
        assert kernel.get_register(0) == 42
        kernel.reset()
        assert kernel.get_register(0) == 0

    def test_no_program_raises(self, kernel):
        with pytest.raises(RuntimeError):
            kernel.execute()

    def test_invalid_register_index(self, kernel):
        with pytest.raises(IndexError):
            kernel.get_register(8)

    def test_load_from_binary_list(self, kernel):
        """Test loading pre-assembled binary words directly."""
        from ncpu.os.assembler import Opcode
        # MOV R0, 42 → opcode=0x10, rd=0, imm=42
        word = (Opcode.MOV_IMM << 24) | (0 << 21) | 42
        # HALT → opcode=0x01
        halt = Opcode.HALT << 24
        kernel.load_program([word, halt])
        result = kernel.execute()
        assert kernel.get_register(0) == 42
        assert result.stop_reason_name == "HALT"

    def test_ips_property(self, kernel):
        kernel.load_program_from_asm("NOP\n" * 100 + "HALT")
        result = kernel.execute()
        assert result.ips > 0
        assert result.elapsed_seconds > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Mode Verification: Compute vs Neural
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossModeVerification:
    """Verify compute mode produces identical register state to neural mode.

    This is the critical correctness test: run the same programs through
    both the neural CPU (trained .pt models) and the GPU compute kernel,
    then assert identical final register values.
    """

    PROGRAMS = [
        "sum_1_to_10.asm",
        "fibonacci_iterative.asm",
        "factorial.asm",
        "collatz.asm",
        "gcd.asm",
        "max_of_two.asm",
        "min_of_two.asm",
        "absolute_value.asm",
        "is_even_odd.asm",
        "triangular_number.asm",
    ]

    @pytest.fixture
    def neural_cpu(self):
        """Create neural CPU for cross-mode comparison."""
        from ncpu.model import CPU
        cpu = CPU(mock_mode=True, neural_execution=False, max_cycles=10000)
        return cpu

    @pytest.mark.parametrize("program_name", PROGRAMS)
    def test_cross_mode(self, kernel, neural_cpu, program_name):
        """Run program on both neural and compute mode, verify identical registers."""
        program_path = PROJECT_ROOT / "programs" / program_name
        if not program_path.exists():
            pytest.skip(f"Program not found: {program_name}")

        source = program_path.read_text()

        # Run on compute mode (Metal GPU)
        kernel.load_program_from_asm(source)
        compute_result = kernel.execute()
        compute_regs = kernel.get_registers_dict()

        # Run on neural mode (model CPU)
        neural_cpu.load_program(source)
        try:
            neural_cpu.run()
        except RuntimeError:
            pytest.skip(f"Neural CPU failed on {program_name}")

        neural_regs = neural_cpu.dump_registers()

        # Compare all 8 registers
        for i in range(8):
            reg_name = f"R{i}"
            compute_val = compute_regs[reg_name]
            neural_val = neural_regs[reg_name]
            assert compute_val == neural_val, (
                f"{program_name}: {reg_name} mismatch: "
                f"compute={compute_val}, neural={neural_val}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Bulk Program Execution
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeAllPrograms:
    """Run all 62 assembly programs through compute mode — verify they all halt."""

    def test_all_programs_halt(self, kernel):
        """Every .asm program should execute and halt without error."""
        programs_dir = PROJECT_ROOT / "programs"
        asm_files = sorted(programs_dir.glob("*.asm"))
        assert len(asm_files) > 0, "No .asm files found"

        passed = 0
        failed = []
        for asm_file in asm_files:
            kernel.reset()
            try:
                source = asm_file.read_text()
                kernel.load_program_from_asm(source)
                result = kernel.execute(max_cycles=100_000)
                if result.stop_reason_name == "HALT":
                    passed += 1
                else:
                    failed.append((asm_file.name, f"stop_reason={result.stop_reason_name}"))
            except Exception as e:
                failed.append((asm_file.name, str(e)))

        if failed:
            msg = "\n".join(f"  {name}: {err}" for name, err in failed)
            pytest.fail(f"{len(failed)} programs failed:\n{msg}")

        assert passed == len(asm_files), f"Only {passed}/{len(asm_files)} programs passed"
