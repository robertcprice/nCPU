"""Tests for MUXLEQ VM — all 4 instruction cases, I/O, halt, .dec loading.

Tests cover the fast execution mode (pure Python). Neural mode is tested
via a subset that verifies neural ops produce identical results to fast.

Run:
    pytest tests/gpu/test_muxleq.py -v
"""

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kernels.mlx.muxleq_kernel import (
    MuxleqVM,
    MuxleqResult,
    MUXLEQ_STOP_HALT,
    MUXLEQ_STOP_MAX_CYCLES,
    MUXLEQ_STOP_IO_READ,
    MUXLEQ_STOP_IO_WRITE,
    MEMORY_SIZE,
    SENTINEL,
    HALT_THRESHOLD,
    MUX_FLAG,
    MUX_ADDR_MASK,
    WORD_MASK,
)


# ─── Helpers ───

def make_vm(program: list[int], mode: str = "fast") -> MuxleqVM:
    """Create a VM with the given program loaded."""
    vm = MuxleqVM(mode=mode)
    vm.load_program(program)
    return vm


def run_to_halt(vm: MuxleqVM, max_cycles: int = 100_000,
                input_bytes: bytes = b"",
                capture_output: bool = True) -> tuple[MuxleqResult, bytes]:
    """Run VM until halt, providing input and capturing output."""
    input_pos = [0]
    output_buf = bytearray()

    def input_fn():
        if input_pos[0] < len(input_bytes):
            b = input_bytes[input_pos[0]]
            input_pos[0] += 1
            return b
        return SENTINEL  # EOF

    def output_fn(byte_val):
        output_buf.append(byte_val)

    while vm.total_cycles < max_cycles:
        result = vm.step(
            max_cycles=max_cycles - vm.total_cycles,
            input_fn=input_fn,
            output_fn=output_fn,
        )
        if result.stop_reason == MUXLEQ_STOP_HALT:
            return result, bytes(output_buf)
        if result.stop_reason == MUXLEQ_STOP_MAX_CYCLES:
            return result, bytes(output_buf)
        # IO handled, continue

    return MuxleqResult(vm.total_cycles, 0.0, MUXLEQ_STOP_MAX_CYCLES, vm.pc), bytes(output_buf)


# ─── SUBLEQ instruction tests ───

class TestSUBLEQ:
    """Test the SUBLEQ (subtract and branch if <= 0) instruction."""

    def test_subtract_no_branch(self):
        """m[b] = m[b] - m[a], result > 0 so no branch."""
        # m[0]=2, m[1]=5, m[2]=next_instr(3)
        # After: m[1] = 5 - 2 = 3, PC continues to 3
        # m[3..5] = halt triple (0xFFFF, 0xFFFF, 0xFFFF → SUBLEQ with c=FFFF)
        vm = make_vm([
            2, 1, 3,         # SUBLEQ: m[1] -= m[0], branch to 3 if <= 0
            0, 0, SENTINEL,  # SUBLEQ: m[0] -= m[0] = 0, branch to FFFF (halt)
        ])
        # m[0]=2 (addr 0 holds value 2, also operand a)
        # Wait, the program IS the memory. Let me think again.
        # m[0]=2, m[1]=1, m[2]=3 — these are the instruction triple
        # a=2, b=1, c=3
        # m[a]=m[2]=3, m[b]=m[1]=1
        # r = m[1] - m[2] = 1 - 3 = -2 (unsigned: 0xFFFE)
        # 0xFFFE & 0x8000 = true → branch to c=3
        # That's not what I want. Let me set up separate data and code areas.

        # Better: put data in high addresses, code at 0
        vm = make_vm([0] * 100)
        # Data at addresses 90, 91
        vm.write_memory(90, 3)    # m[90] = 3 (subtrahend)
        vm.write_memory(91, 10)   # m[91] = 10 (minuend)
        # Code at address 0: SUBLEQ a=90, b=91, c=6 (next instruction)
        vm.write_memory(0, 90)    # a
        vm.write_memory(1, 91)    # b
        vm.write_memory(2, 6)     # c (no branch target, just next instr)
        # Halt instruction at address 3-5
        vm.write_memory(3, 91)    # a (self)
        vm.write_memory(4, 91)    # b (self)
        vm.write_memory(5, SENTINEL)  # c = FFFF → halt when result <= 0
        # Wait — at addr 3: a=91, b=91, m[91]-m[91] = 0, branch to FFFF → halt

        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT
        assert vm.read_memory(91) == 0  # 10 - 3 = 7, then 7 - 7 = 0

    def test_subtract_branch_on_zero(self):
        """Branch when result == 0."""
        vm = make_vm([0] * 100)
        vm.write_memory(90, 5)   # m[90] = 5
        vm.write_memory(91, 5)   # m[91] = 5
        # SUBLEQ: m[91] -= m[90] → 5-5=0, branch to c
        vm.write_memory(0, 90)   # a
        vm.write_memory(1, 91)   # b
        vm.write_memory(2, SENTINEL)  # c = FFFF → halt on zero
        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT
        assert vm.read_memory(91) == 0

    def test_subtract_branch_on_negative(self):
        """Branch when result is negative (bit 15 set)."""
        vm = make_vm([0] * 100)
        vm.write_memory(90, 10)  # m[90] = 10
        vm.write_memory(91, 3)   # m[91] = 3
        # SUBLEQ: m[91] = 3 - 10 = -7 (0xFFF9), bit 15 set → branch
        vm.write_memory(0, 90)   # a
        vm.write_memory(1, 91)   # b
        vm.write_memory(2, SENTINEL)  # branch to halt
        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT
        assert vm.read_memory(91) == (3 - 10) & WORD_MASK  # 0xFFF9

    def test_subtract_no_branch_positive(self):
        """No branch when result > 0."""
        vm = make_vm([0] * 100)
        vm.write_memory(90, 3)   # m[90] = 3
        vm.write_memory(91, 10)  # m[91] = 10
        # SUBLEQ: m[91] = 10 - 3 = 7, no branch (result > 0)
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, 50)   # branch target (not taken)
        # Next instruction at 3: halt
        vm.write_memory(3, 91)
        vm.write_memory(4, 91)
        vm.write_memory(5, SENTINEL)
        result, _ = run_to_halt(vm)
        assert vm.read_memory(91) == 0  # 7 - 7 = 0 (second instruction)


# ─── MUX instruction tests ───

class TestMUX:
    """Test the MUX (bitwise multiplex) instruction."""

    def test_mux_all_from_a(self):
        """When mask is all zeros, result = m[a]."""
        vm = make_vm([0] * 200)
        vm.write_memory(90, 0xABCD)  # m[a] = source
        vm.write_memory(91, 0x1234)  # m[b] = destination
        vm.write_memory(95, 0x0000)  # mask = 0 → all bits from m[a]
        vm.write_memory(99, 0)       # zero cell for halt (don't clobber result)
        vm.write_memory(0, 90)       # a
        vm.write_memory(1, 91)       # b
        vm.write_memory(2, 0x8000 | 95)  # c (MUX, mask at addr 95)
        # Halt at 3 — use addr 99 (zero) to avoid clobbering m[91]
        vm.write_memory(3, 99)
        vm.write_memory(4, 99)
        vm.write_memory(5, SENTINEL)
        result, _ = run_to_halt(vm)
        assert vm.read_memory(91) == 0xABCD

    def test_mux_all_from_b(self):
        """When mask is all ones, result = m[b] (unchanged)."""
        vm = make_vm([0] * 200)
        vm.write_memory(90, 0xABCD)  # m[a]
        vm.write_memory(91, 0x1234)  # m[b]
        vm.write_memory(95, 0xFFFF)  # mask = all 1s → keep m[b]
        vm.write_memory(99, 0)
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, 0x8000 | 95)
        vm.write_memory(3, 99)
        vm.write_memory(4, 99)
        vm.write_memory(5, SENTINEL)
        result, _ = run_to_halt(vm)
        assert vm.read_memory(91) == 0x1234

    def test_mux_partial_mask(self):
        """Selective bit mixing."""
        vm = make_vm([0] * 200)
        vm.write_memory(90, 0xFF00)  # m[a] = 1111_1111_0000_0000
        vm.write_memory(91, 0x00FF)  # m[b] = 0000_0000_1111_1111
        vm.write_memory(95, 0x0F0F)  # mask = 0000_1111_0000_1111
        vm.write_memory(99, 0)
        # Result: (m[a] & ~mask) | (m[b] & mask)
        # = (0xFF00 & 0xF0F0) | (0x00FF & 0x0F0F)
        # = 0xF000 | 0x000F = 0xF00F
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, 0x8000 | 95)
        vm.write_memory(3, 99)
        vm.write_memory(4, 99)
        vm.write_memory(5, SENTINEL)
        result, _ = run_to_halt(vm)
        assert vm.read_memory(91) == 0xF00F

    def test_mux_not_triggered_when_c_is_sentinel(self):
        """c == 0xFFFF should trigger SUBLEQ, not MUX."""
        vm = make_vm([0] * 100)
        vm.write_memory(90, 5)
        vm.write_memory(91, 5)
        # c = 0xFFFF has bit 15 set, but c == SENTINEL → SUBLEQ path
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, SENTINEL)
        result, _ = run_to_halt(vm)
        # SUBLEQ: 5-5=0, branch to 0xFFFF → halt
        assert result.stop_reason == MUXLEQ_STOP_HALT
        assert vm.read_memory(91) == 0


# ─── I/O tests ───

class TestIO:
    """Test INPUT and OUTPUT instructions."""

    def test_output(self):
        """Output m[a] when b == 0xFFFF."""
        vm = make_vm([0] * 100)
        vm.write_memory(90, ord('H'))
        vm.write_memory(91, ord('i'))
        # Output m[90]: a=90, b=FFFF, c=anything
        vm.write_memory(0, 90)
        vm.write_memory(1, SENTINEL)
        vm.write_memory(2, 0)
        # Output m[91]: a=91, b=FFFF, c=anything
        vm.write_memory(3, 91)
        vm.write_memory(4, SENTINEL)
        vm.write_memory(5, 0)
        # Halt
        vm.write_memory(6, 90)
        vm.write_memory(7, 90)
        vm.write_memory(8, SENTINEL)

        result, output = run_to_halt(vm)
        assert output == b"Hi"

    def test_input(self):
        """Read byte into m[b] when a == 0xFFFF."""
        vm = make_vm([0] * 100)
        vm.write_memory(99, 0)  # zero cell for halt
        # Input to m[90]: a=FFFF, b=90, c=anything
        vm.write_memory(0, SENTINEL)
        vm.write_memory(1, 90)
        vm.write_memory(2, 0)
        # Halt — use addr 99 to avoid clobbering m[90]
        vm.write_memory(3, 99)
        vm.write_memory(4, 99)
        vm.write_memory(5, SENTINEL)

        result, _ = run_to_halt(vm, input_bytes=b"X")
        assert vm.read_memory(90) == ord('X')

    def test_input_eof(self):
        """EOF returns 0xFFFF."""
        vm = make_vm([0] * 100)
        vm.write_memory(99, 0)
        vm.write_memory(0, SENTINEL)
        vm.write_memory(1, 90)
        vm.write_memory(2, 0)
        vm.write_memory(3, 99)
        vm.write_memory(4, 99)
        vm.write_memory(5, SENTINEL)

        result, _ = run_to_halt(vm, input_bytes=b"")  # no input → EOF
        assert vm.read_memory(90) == SENTINEL

    def test_echo_byte(self):
        """Read a byte, then output it."""
        vm = make_vm([0] * 100)
        # Input to m[90]
        vm.write_memory(0, SENTINEL)
        vm.write_memory(1, 90)
        vm.write_memory(2, 0)
        # Output m[90]
        vm.write_memory(3, 90)
        vm.write_memory(4, SENTINEL)
        vm.write_memory(5, 0)
        # Halt
        vm.write_memory(6, 90)
        vm.write_memory(7, 90)
        vm.write_memory(8, SENTINEL)

        result, output = run_to_halt(vm, input_bytes=b"Z")
        assert output == b"Z"


# ─── Halt and control flow tests ───

class TestHalt:
    """Test halting and PC behavior."""

    def test_halt_via_branch_to_sentinel(self):
        """Branch to 0xFFFF halts because 0xFFFF >= 0x8000."""
        vm = make_vm([0] * 100)
        vm.write_memory(90, 1)
        vm.write_memory(91, 0)  # m[91] = 0, so 0-1 = negative → branch
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, SENTINEL)
        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT

    def test_halt_via_sentinel_branch(self):
        """Branch to 0xFFFF (the only SUBLEQ halt path) halts.

        Note: c values 0x8000-0xFFFE trigger MUX, so the only way to
        halt via SUBLEQ branch is c=0xFFFF (SENTINEL), which is >= 0x8000.
        """
        vm = make_vm([0] * 100)
        vm.write_memory(90, 1)
        vm.write_memory(91, 0)
        # SUBLEQ: m[91] = 0 - 1 = 0xFFFF (negative), branch to SENTINEL → halt
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, SENTINEL)
        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT

    def test_max_cycles_limit(self):
        """VM stops at max_cycles if no halt reached."""
        # Infinite loop: SUBLEQ that never branches
        vm = make_vm([0] * 100)
        vm.write_memory(90, 0)   # m[90] = 0
        vm.write_memory(91, 1)   # m[91] = 1 (will keep being positive)
        # SUBLEQ: m[91] = 1 - 0 = 1 (positive, no branch), loop back to 0
        # Wait — this would halt on second pass since 1-0=1, then 1-0=1...
        # Actually m[90]=0 always, m[91]-=0 each time → 1 forever → no branch
        vm.write_memory(0, 90)   # a
        vm.write_memory(1, 91)   # b
        vm.write_memory(2, 0)    # c (branch back to 0, but won't branch since result > 0)
        # PC advances to 3, which has zeros → SUBLEQ m[0]-m[0]=0, branch to m[2]=0
        # This is complex. Let me just make a simple non-halting loop.
        # Use MUX which doesn't affect PC:
        vm.write_memory(95, 0xFFFF)  # mask = all 1s → m[b] unchanged
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, 0x8000 | 95)  # MUX (no-op with all-1s mask)
        # PC goes to 3, which is all zeros → SUBLEQ m[0]-m[0]=0 → branch to 0
        # So this loops: 0→MUX→3→SUBLEQ(0,0,0)→branch to 0
        result, _ = run_to_halt(vm, max_cycles=100)
        assert result.stop_reason == MUXLEQ_STOP_MAX_CYCLES


# ─── .dec file loading tests ───

class TestDecFile:
    """Test .dec file loading."""

    def test_load_dec_file(self):
        """Load comma-separated decimal file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dec', delete=False) as f:
            f.write("10, 20, 30, -1, 0, 6,\n")
            f.flush()
            vm = MuxleqVM(mode="fast")
            count = vm.load_dec(f.name)
            assert count == 6
            assert vm.read_memory(0) == 10
            assert vm.read_memory(1) == 20
            assert vm.read_memory(2) == 30
            assert vm.read_memory(3) == SENTINEL  # -1 → 0xFFFF
            assert vm.read_memory(4) == 0
            assert vm.read_memory(5) == 6

    def test_load_negative_values(self):
        """Negative values wrap to unsigned 16-bit."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dec', delete=False) as f:
            f.write("-1, -2, -32768,\n")
            f.flush()
            vm = MuxleqVM(mode="fast")
            vm.load_dec(f.name)
            assert vm.read_memory(0) == 0xFFFF
            assert vm.read_memory(1) == 0xFFFE
            assert vm.read_memory(2) == 0x8000


# ─── Loop / program tests ───

class TestPrograms:
    """Test small complete programs."""

    def test_countdown(self):
        """Count down from 5 to 0 using SUBLEQ loop."""
        # Memory layout:
        #   addr 90: counter = 5
        #   addr 91: one = 1
        #   addr 92: zero = 0
        #
        # Code at 0:
        #   [0,1,2] SUBLEQ a=91, b=90, c=6 → m[90] -= 1, if <= 0 goto 6
        #   [3,4,5] SUBLEQ a=92, b=92, c=0 → m[92] -= 0 = 0 → branch to 0 (loop)
        #   [6,7,8] HALT: SUBLEQ a=92, b=92, c=FFFF → 0-0=0, branch to FFFF
        vm = make_vm([0] * 100)
        vm.write_memory(90, 5)    # counter
        vm.write_memory(91, 1)    # constant 1
        vm.write_memory(92, 0)    # constant 0
        # Instruction 0: subtract 1 from counter
        vm.write_memory(0, 91)    # a = addr of 1
        vm.write_memory(1, 90)    # b = addr of counter
        vm.write_memory(2, 6)     # c = branch to halt when <= 0
        # Instruction 1: unconditional jump to 0 (subtract 0 from 0 = 0, branch)
        vm.write_memory(3, 92)    # a = addr of 0
        vm.write_memory(4, 92)    # b = addr of 0
        vm.write_memory(5, 0)     # c = branch to 0
        # Instruction 2: halt
        vm.write_memory(6, 92)
        vm.write_memory(7, 92)
        vm.write_memory(8, SENTINEL)

        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT
        assert vm.read_memory(90) == 0  # counted down to 0

    def test_add_via_subleq(self):
        """Add two numbers using SUBLEQ: a + b = a - (0 - b)."""
        # To compute m[X] = m[X] + m[Y]:
        #   Step 1: m[T] = 0 - m[Y]  (negate Y into T)
        #   Step 2: m[X] = m[X] - m[T]  (X - (-Y) = X + Y)
        vm = make_vm([0] * 100)
        vm.write_memory(90, 7)    # X = 7
        vm.write_memory(91, 3)    # Y = 3
        vm.write_memory(92, 0)    # T (temp, starts at 0)
        vm.write_memory(93, 0)    # zero

        # Step 1: SUBLEQ a=91, b=92, c=3 → m[92] = 0 - 3 = -3 (0xFFFD)
        # Result is negative → branch to 3 (next instruction)
        vm.write_memory(0, 91)
        vm.write_memory(1, 92)
        vm.write_memory(2, 3)

        # Step 2: SUBLEQ a=92, b=90, c=6 → m[90] = 7 - (-3) = 10
        # Result is 10 (positive) → no branch, falls to 6
        vm.write_memory(3, 92)
        vm.write_memory(4, 90)
        vm.write_memory(5, 6)

        # Halt
        vm.write_memory(6, 93)
        vm.write_memory(7, 93)
        vm.write_memory(8, SENTINEL)

        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT
        assert vm.read_memory(90) == 10  # 7 + 3

    def test_output_hello(self):
        """Output 'Hello' using OUTPUT instructions."""
        vm = make_vm([0] * 100)
        msg = "Hello"
        # Store character data
        for i, ch in enumerate(msg):
            vm.write_memory(80 + i, ord(ch))

        # Generate output instructions
        pc = 0
        for i in range(len(msg)):
            vm.write_memory(pc, 80 + i)   # a = address of character
            vm.write_memory(pc + 1, SENTINEL)  # b = FFFF (output)
            vm.write_memory(pc + 2, 0)     # c (ignored)
            pc += 3

        # Halt
        vm.write_memory(pc, 80)
        vm.write_memory(pc + 1, 80)
        vm.write_memory(pc + 2, SENTINEL)

        result, output = run_to_halt(vm)
        assert output == b"Hello"


# ─── Neural mode tests ───

class TestNeuralMode:
    """Test neural mode produces identical results to fast mode."""

    @pytest.fixture
    def neural_available(self):
        """Check if neural models are available."""
        try:
            import torch
            models_dir = Path(__file__).resolve().parent.parent.parent / "models"
            if not (models_dir / "alu" / "arithmetic.pt").exists():
                pytest.skip("Neural models not available")
            return True
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_subleq_neural_vs_fast(self, neural_available):
        """SUBLEQ produces same result in neural and fast modes."""
        for mode in ("fast", "neural"):
            vm = MuxleqVM(mode=mode)
            vm.load_program([0] * 100)
            vm.write_memory(90, 3)
            vm.write_memory(91, 10)
            vm.write_memory(92, 0)
            # Instruction 0: SUBLEQ m[91] -= m[90] → 10-3=7, positive, no branch
            vm.write_memory(0, 90)
            vm.write_memory(1, 91)
            vm.write_memory(2, 6)     # branch target (not taken)
            # Instruction 1: halt (use zero cell 92 to not clobber m[91])
            vm.write_memory(3, 92)
            vm.write_memory(4, 92)
            vm.write_memory(5, SENTINEL)

            result, _ = run_to_halt(vm)
            assert result.stop_reason == MUXLEQ_STOP_HALT
            # m[91] = 10 - 3 = 7 (preserved because halt uses m[92])
            assert vm.read_memory(91) == 7, f"Failed in {mode} mode"

    def test_mux_neural_vs_fast(self, neural_available):
        """MUX produces same result in neural and fast modes."""
        for mode in ("fast", "neural"):
            vm = MuxleqVM(mode=mode)
            vm.load_program([0] * 200)
            vm.write_memory(90, 0xFF00)
            vm.write_memory(91, 0x00FF)
            vm.write_memory(95, 0x0F0F)
            vm.write_memory(99, 0)
            vm.write_memory(0, 90)
            vm.write_memory(1, 91)
            vm.write_memory(2, 0x8000 | 95)
            vm.write_memory(3, 99)
            vm.write_memory(4, 99)
            vm.write_memory(5, SENTINEL)

            result, _ = run_to_halt(vm)
            assert vm.read_memory(91) == 0xF00F, f"Failed in {mode} mode"

    def test_add_neural_vs_fast(self, neural_available):
        """Addition via SUBLEQ matches between neural and fast modes."""
        for mode in ("fast", "neural"):
            vm = MuxleqVM(mode=mode)
            vm.load_program([0] * 100)
            vm.write_memory(90, 7)
            vm.write_memory(91, 3)
            vm.write_memory(92, 0)
            vm.write_memory(93, 0)
            vm.write_memory(0, 91)
            vm.write_memory(1, 92)
            vm.write_memory(2, 3)
            vm.write_memory(3, 92)
            vm.write_memory(4, 90)
            vm.write_memory(5, 6)
            vm.write_memory(6, 93)
            vm.write_memory(7, 93)
            vm.write_memory(8, SENTINEL)

            result, _ = run_to_halt(vm)
            assert vm.read_memory(90) == 10, f"Failed in {mode} mode: expected 10, got {vm.read_memory(90)}"


# ─── Edge case tests ───

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_subtract_wraps_unsigned(self):
        """Subtraction wraps at 16-bit boundary."""
        vm = make_vm([0] * 100)
        vm.write_memory(90, 1)
        vm.write_memory(91, 0)
        vm.write_memory(0, 90)  # a
        vm.write_memory(1, 91)  # b
        vm.write_memory(2, SENTINEL)  # c
        result, _ = run_to_halt(vm)
        assert vm.read_memory(91) == WORD_MASK  # 0 - 1 = 0xFFFF

    def test_mux_with_self(self):
        """MUX where a == b should produce m[b] (identity regardless of mask)."""
        vm = make_vm([0] * 200)
        vm.write_memory(90, 0xABCD)
        vm.write_memory(95, 0x5555)  # any mask
        vm.write_memory(99, 0)       # zero cell for halt
        vm.write_memory(0, 90)       # a = b = 90
        vm.write_memory(1, 90)       # same address
        vm.write_memory(2, 0x8000 | 95)
        vm.write_memory(3, 99)
        vm.write_memory(4, 99)
        vm.write_memory(5, SENTINEL)
        result, _ = run_to_halt(vm)
        # (m[90] & ~mask) | (m[90] & mask) = m[90]
        assert vm.read_memory(90) == 0xABCD

    def test_pc_starts_at_zero(self):
        """PC starts at 0 after load."""
        vm = MuxleqVM(mode="fast")
        vm.load_program([0, 0, SENTINEL])
        assert vm.pc == 0

    def test_memory_size(self):
        """Memory is exactly 65536 words."""
        vm = MuxleqVM(mode="fast")
        assert len(vm.memory) == 65536

    def test_mode_validation(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            MuxleqVM(mode="invalid")


# ─── Compute mode tests (requires MLX) ───

class TestComputeMode:
    """Test Metal GPU compute mode."""

    @pytest.fixture
    def mlx_available(self):
        try:
            import mlx.core
            return True
        except ImportError:
            pytest.skip("MLX not available")

    def test_subleq_compute(self, mlx_available):
        """SUBLEQ works in compute mode."""
        vm = MuxleqVM(mode="compute")
        vm.load_program([0] * 100)
        vm.write_memory(90, 5)
        vm.write_memory(91, 5)
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, SENTINEL)
        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT
        assert vm.read_memory(91) == 0

    def test_mux_compute(self, mlx_available):
        """MUX works in compute mode."""
        vm = MuxleqVM(mode="compute")
        vm.load_program([0] * 200)
        vm.write_memory(90, 0xFF00)
        vm.write_memory(91, 0x00FF)
        vm.write_memory(95, 0x0F0F)
        vm.write_memory(99, 0)
        vm.write_memory(0, 90)
        vm.write_memory(1, 91)
        vm.write_memory(2, 0x8000 | 95)
        vm.write_memory(3, 99)
        vm.write_memory(4, 99)
        vm.write_memory(5, SENTINEL)
        result, _ = run_to_halt(vm)
        assert vm.read_memory(91) == 0xF00F

    def test_output_compute(self, mlx_available):
        """Output works in compute mode (via I/O traps)."""
        vm = MuxleqVM(mode="compute")
        vm.load_program([0] * 100)
        vm.write_memory(80, ord('A'))
        vm.write_memory(0, 80)
        vm.write_memory(1, SENTINEL)
        vm.write_memory(2, 0)
        vm.write_memory(3, 80)
        vm.write_memory(4, 80)
        vm.write_memory(5, SENTINEL)
        result, output = run_to_halt(vm)
        assert output == b"A"

    def test_countdown_compute(self, mlx_available):
        """Countdown program works in compute mode."""
        vm = MuxleqVM(mode="compute")
        vm.load_program([0] * 100)
        vm.write_memory(90, 5)
        vm.write_memory(91, 1)
        vm.write_memory(92, 0)
        vm.write_memory(0, 91)
        vm.write_memory(1, 90)
        vm.write_memory(2, 6)
        vm.write_memory(3, 92)
        vm.write_memory(4, 92)
        vm.write_memory(5, 0)
        vm.write_memory(6, 92)
        vm.write_memory(7, 92)
        vm.write_memory(8, SENTINEL)
        result, _ = run_to_halt(vm)
        assert result.stop_reason == MUXLEQ_STOP_HALT
        assert vm.read_memory(90) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
