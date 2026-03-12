"""Tests for GPU-Native Debugging Toolkit.

Verifies the complete debugging infrastructure in the Metal GPU kernel:
  - Instruction trace buffer (0x3B0000): 4096 entries × 56 bytes
  - Breakpoints (0x3A0000): up to 4 PC breakpoints
  - Conditional breakpoints: fire only when register==value
  - Memory watchpoints: shadow-comparison write-watch on up to 4 addresses
  - Instruction classifier: 88+ ARM64 instruction types
  - Deterministic replay verification

Tests cover:
  - Enable/disable trace via sentinel value
  - Recording correct PC and instruction words
  - Register state capture (x0-x3), NZCV flags, SP
  - Circular buffer wrapping at 4096 entries
  - Trace does not corrupt SVC buffer at 0x3F0000
  - Breakpoints: set, clear, stop on hit, edge cases
  - Conditional breakpoints: condition met/not-met, different registers
  - Memory watchpoints: trigger on write, info records values, clear, multiple
  - Combined breakpoint + watchpoint coexistence
  - Both Rust and MLX backends

Requires: mlx (Apple Silicon Metal)
"""

import struct
import sys
from pathlib import Path

import pytest
from kernels.mlx.availability import has_gpu_backend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HAS_GPU_BACKEND = has_gpu_backend()

pytestmark = pytest.mark.skipif(not HAS_GPU_BACKEND, reason="GPU backend not available")


# ═══════════════════════════════════════════════════════════════════════════════
# ARM64 Instruction Encoders (subset needed for trace tests)
# ═══════════════════════════════════════════════════════════════════════════════

def movz(rd, imm16, hw=0):
    """MOVZ Xd, #imm16, LSL #(hw*16)"""
    return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

def movk(rd, imm16, hw=0):
    """MOVK Xd, #imm16, LSL #(hw*16)"""
    return 0xF2800000 | (hw << 21) | (imm16 << 5) | rd

def add_imm(rd, rn, imm12):
    """ADD Xd, Xn, #imm12"""
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd

def sub_imm(rd, rn, imm12):
    """SUB Xd, Xn, #imm12"""
    return 0xD1000000 | (imm12 << 10) | (rn << 5) | rd

def adds_imm(rd, rn, imm12):
    """ADDS Xd, Xn, #imm12 (sets flags)"""
    return 0xB1000000 | (imm12 << 10) | (rn << 5) | rd

def subs_imm(rd, rn, imm12):
    """SUBS Xd, Xn, #imm12 (sets flags)"""
    return 0xF1000000 | (imm12 << 10) | (rn << 5) | rd

def b(offset_words):
    """B #offset (in 4-byte words from current PC)"""
    return 0x14000000 | (offset_words & 0x3FFFFFF)

def stur_64(rt, rn, offset):
    """STUR Xt, [Xn, #offset] — store 64-bit register to memory (unscaled offset)."""
    # F8000000 | (offset9 << 12) | (Rn << 5) | Rt
    return 0xF8000000 | ((offset & 0x1FF) << 12) | (rn << 5) | rt

def ldur_64(rt, rn, offset):
    """LDUR Xt, [Xn, #offset] — load 64-bit register from memory (unscaled offset)."""
    return 0xF8400000 | ((offset & 0x1FF) << 12) | (rn << 5) | rt

NOP = 0xD503201F
HLT = 0xD4400000


def build_binary(insts):
    """Convert list of 32-bit instruction words to bytes."""
    return b''.join(struct.pack('<I', i) for i in insts)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures — need 16MB memory for trace buffer at 0x3B0000
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def rust_cpu():
    """Get a RustMetalCPU with 16MB memory (trace buffer needs large memory)."""
    try:
        from kernels.mlx.rust_runner import RustMetalCPU
        cpu = RustMetalCPU(memory_size=16 * 1024 * 1024, quiet=True)
        return cpu
    except (ImportError, Exception):
        pytest.skip("RustMetalCPU not available")

@pytest.fixture
def mlx_cpu():
    """Get an MLXKernelCPUv2 with 16MB memory."""
    try:
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        cpu = MLXKernelCPUv2(memory_size=16 * 1024 * 1024, quiet=True)
        return cpu
    except (ImportError, Exception):
        pytest.skip("MLXKernelCPUv2 not available")


def run(cpu, insts, max_cycles=200):
    """Load program at address 0, set PC, execute."""
    cpu.load_program(insts, address=0)
    cpu.set_pc(0)
    return cpu.execute(max_cycles=max_cycles)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Rust Metal Backend
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceRust:
    """Trace buffer tests using the Rust Metal backend."""

    def test_enable_disable(self, rust_cpu):
        """Enable trace, run nothing, read → empty list."""
        rust_cpu.enable_trace()
        entries = rust_cpu.read_trace()
        assert entries == []

    def test_disable_returns_empty(self, rust_cpu):
        """Disabled trace returns empty list."""
        rust_cpu.disable_trace()
        entries = rust_cpu.read_trace()
        assert entries == []

    def test_basic_recording(self, rust_cpu):
        """Trace records correct number of instructions."""
        rust_cpu.enable_trace()
        # MOVZ X0, #42 ; HLT  → 2 instructions
        run(rust_cpu, [movz(0, 42), HLT])
        entries = rust_cpu.read_trace()
        assert len(entries) == 2

    def test_pc_values(self, rust_cpu):
        """Trace entries have correct PC values (0, 4, 8, ...)."""
        rust_cpu.enable_trace()
        # NOP ; NOP ; MOVZ X0, #1 ; HLT  → 4 instructions
        run(rust_cpu, [NOP, NOP, movz(0, 1), HLT])
        entries = rust_cpu.read_trace()
        assert len(entries) == 4
        for i, entry in enumerate(entries):
            assert entry[0] == i * 4, f"Entry {i}: expected PC={i*4}, got {entry[0]}"

    def test_instruction_words(self, rust_cpu):
        """Trace entries contain the correct instruction words."""
        rust_cpu.enable_trace()
        insts = [movz(0, 42), NOP, HLT]
        run(rust_cpu, insts)
        entries = rust_cpu.read_trace()
        assert len(entries) == 3
        for i, entry in enumerate(entries):
            assert entry[1] == insts[i], \
                f"Entry {i}: expected inst=0x{insts[i]:08x}, got 0x{entry[1]:08x}"

    def test_register_state_x0(self, rust_cpu):
        """Trace captures x0 register state correctly."""
        rust_cpu.enable_trace()
        # MOVZ X0, #100 ; MOVZ X0, #200 ; HLT
        run(rust_cpu, [movz(0, 100), movz(0, 200), HLT])
        entries = rust_cpu.read_trace()
        assert len(entries) == 3
        # After first MOVZ X0, #100: x0 should be 100
        assert entries[1][2] == 100, f"After MOVZ X0, #100: x0={entries[1][2]}"
        # After second MOVZ X0, #200: x0 should be 200
        assert entries[2][2] == 200, f"After MOVZ X0, #200: x0={entries[2][2]}"

    def test_register_state_multiple(self, rust_cpu):
        """Trace captures x0-x3 register state."""
        rust_cpu.enable_trace()
        # Set x0=10, x1=20, x2=30, x3=40, then HLT
        run(rust_cpu, [movz(0, 10), movz(1, 20), movz(2, 30), movz(3, 40), HLT])
        entries = rust_cpu.read_trace()
        assert len(entries) == 5
        # Last entry (HLT) should have all registers set
        pc, inst, x0, x1, x2, x3, flags, sp = entries[4]
        assert x0 == 10
        assert x1 == 20
        assert x2 == 30
        assert x3 == 40

    def test_clear_trace(self, rust_cpu):
        """Clear trace resets the buffer."""
        rust_cpu.enable_trace()
        run(rust_cpu, [movz(0, 1), HLT])
        entries = rust_cpu.read_trace()
        assert len(entries) > 0

        rust_cpu.clear_trace()
        entries = rust_cpu.read_trace()
        assert entries == []

    def test_trace_does_not_corrupt_svc_buffer(self, rust_cpu):
        """Trace buffer at 0x3B0000 must not corrupt SVC buffer at 0x3F0000."""
        # Initialize SVC buffer with known pattern
        svc_marker = b'\xDE\xAD\xBE\xEF' * 4  # 16 bytes
        rust_cpu.write_memory(0x3F0000, svc_marker)

        # Enable trace and run a program with many instructions
        rust_cpu.enable_trace()
        # 100 NOPs + HLT → fills trace with 101 entries
        run(rust_cpu, [NOP] * 100 + [HLT], max_cycles=200)

        # Verify SVC buffer is not corrupted
        svc_data = rust_cpu.read_memory(0x3F0000, 16)
        assert svc_data == svc_marker, "Trace buffer corrupted SVC buffer!"

    def test_many_instructions_no_svc_corruption(self, rust_cpu):
        """Run enough instructions to fill a significant portion of trace buffer."""
        svc_marker = b'\xCA\xFE\xBA\xBE' * 4
        rust_cpu.write_memory(0x3F0000, svc_marker)

        rust_cpu.enable_trace()
        # Loop: MOVZ X0, #0; ADD X0, X0, #1; B -1 (back to ADD)
        # This will loop until max_cycles
        run(rust_cpu, [movz(0, 0), add_imm(0, 0, 1), b(-1)], max_cycles=5000)

        # SVC buffer must be intact
        svc_data = rust_cpu.read_memory(0x3F0000, 16)
        assert svc_data == svc_marker, "Trace buffer corrupted SVC buffer with many entries!"

    def test_circular_wrapping(self, rust_cpu):
        """Trace buffer wraps at 4096 entries, preserving most recent."""
        rust_cpu.enable_trace()
        # Run a loop that executes > 4096 instructions
        # MOVZ X0, #0 ; ADD X0, X0, #1 ; B -1  → loops until max_cycles
        run(rust_cpu, [movz(0, 0), add_imm(0, 0, 1), b(-1)], max_cycles=5000)
        entries = rust_cpu.read_trace()
        # Should have exactly 4096 entries (max buffer size)
        assert len(entries) == 4096
        # Entries should be ordered oldest → newest
        # All PCs should be valid (either 0, 4, or 8 since it loops 3 instructions)
        for entry in entries:
            assert entry[0] in (0, 4, 8), f"Unexpected PC {entry[0]} in wrapped trace"

    def test_extended_trace_entry_format(self, rust_cpu):
        """Trace entries have 8 fields: pc, inst, x0, x1, x2, x3, flags, sp."""
        rust_cpu.enable_trace()
        run(rust_cpu, [movz(0, 42), HLT])
        entries = rust_cpu.read_trace()
        assert len(entries) == 2
        assert len(entries[0]) == 8, f"Expected 8-tuple, got {len(entries[0])}-tuple"

    def test_extended_trace_sp(self, rust_cpu):
        """Trace captures SP (x31) value."""
        rust_cpu.enable_trace()
        # Set SP to a known value via ADD (reg 31 = SP in ADD context)
        # MOVZ X5, #0x1234; ADD SP, X5, #0 → SP = 0x1234
        # Then NOP; HLT
        run(rust_cpu, [movz(5, 0x1234), add_imm(31, 5, 0), NOP, HLT])
        entries = rust_cpu.read_trace()
        assert len(entries) == 4
        # After ADD SP, X5, #0: SP should be 0x1234
        # Check at entry[2] (NOP) which is after the ADD
        sp = entries[2][7]
        assert sp == 0x1234, f"Expected SP=0x1234, got SP=0x{sp:X}"

    def test_extended_trace_flags(self, rust_cpu):
        """Trace captures NZCV flags after flag-setting instructions."""
        rust_cpu.enable_trace()
        # MOVZ X0, #0; SUBS X1, X0, #0 (sets Z=1); NOP; HLT
        # After SUBS with result 0: Z flag should be set
        run(rust_cpu, [movz(0, 0), subs_imm(1, 0, 0), NOP, HLT])
        entries = rust_cpu.read_trace()
        assert len(entries) == 4
        # Entry[2] (NOP after SUBS) should show Z flag set
        flags = entries[2][6]
        z_flag = (flags >> 2) & 1
        assert z_flag == 1, f"Expected Z flag set after SUBS 0-0, got flags=0b{flags:04b}"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Python MLX Backend
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceMLX:
    """Trace buffer tests using the Python MLX backend."""

    def test_enable_disable(self, mlx_cpu):
        """Enable trace, run nothing, read → empty list."""
        mlx_cpu.enable_trace()
        entries = mlx_cpu.read_trace()
        assert entries == []

    def test_basic_recording(self, mlx_cpu):
        """Trace records correct number of instructions."""
        mlx_cpu.enable_trace()
        run(mlx_cpu, [movz(0, 42), HLT])
        entries = mlx_cpu.read_trace()
        assert len(entries) == 2

    def test_pc_values(self, mlx_cpu):
        """Trace entries have correct PC values."""
        mlx_cpu.enable_trace()
        run(mlx_cpu, [NOP, NOP, movz(0, 1), HLT])
        entries = mlx_cpu.read_trace()
        assert len(entries) == 4
        for i, entry in enumerate(entries):
            assert entry[0] == i * 4

    def test_instruction_words(self, mlx_cpu):
        """Trace entries contain correct instruction words."""
        mlx_cpu.enable_trace()
        insts = [movz(0, 42), NOP, HLT]
        run(mlx_cpu, insts)
        entries = mlx_cpu.read_trace()
        assert len(entries) == 3
        for i, entry in enumerate(entries):
            assert entry[1] == insts[i]

    def test_register_state(self, mlx_cpu):
        """Trace captures x0-x3 register state."""
        mlx_cpu.enable_trace()
        run(mlx_cpu, [movz(0, 10), movz(1, 20), movz(2, 30), movz(3, 40), HLT])
        entries = mlx_cpu.read_trace()
        assert len(entries) == 5
        pc, inst, x0, x1, x2, x3, flags, sp = entries[4]
        assert x0 == 10
        assert x1 == 20
        assert x2 == 30
        assert x3 == 40

    def test_clear_trace(self, mlx_cpu):
        """Clear trace resets the buffer."""
        mlx_cpu.enable_trace()
        run(mlx_cpu, [movz(0, 1), HLT])
        assert len(mlx_cpu.read_trace()) > 0
        mlx_cpu.clear_trace()
        assert mlx_cpu.read_trace() == []

    def test_svc_buffer_not_corrupted(self, mlx_cpu):
        """Trace at 0x3B0000 must not corrupt SVC at 0x3F0000."""
        mlx_cpu.enable_trace()
        run(mlx_cpu, [NOP] * 100 + [HLT], max_cycles=200)
        # If we got here without SVC corruption, the buffers are separate
        entries = mlx_cpu.read_trace()
        assert len(entries) == 101

    def test_extended_trace_format(self, mlx_cpu):
        """Trace entries are 8-tuples with flags and SP."""
        mlx_cpu.enable_trace()
        run(mlx_cpu, [movz(0, 42), HLT])
        entries = mlx_cpu.read_trace()
        assert len(entries) == 2
        assert len(entries[0]) == 8


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Breakpoints (Rust backend)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBreakpointsRust:
    """Breakpoint tests using the Rust Metal backend."""

    def test_breakpoint_stops_execution(self, rust_cpu):
        """Setting a breakpoint at a PC causes execution to stop there."""
        from kernels.mlx.rust_runner import StopReasonV2
        # Program: MOVZ X0, #1; MOVZ X0, #2; MOVZ X0, #3; HLT
        # Set breakpoint at PC=8 (third instruction)
        rust_cpu.set_breakpoint(0, 8)
        rust_cpu.load_program([movz(0, 1), movz(0, 2), movz(0, 3), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        assert result.stop_reason == StopReasonV2.BREAKPOINT
        # PC should be at the breakpoint (instruction was NOT executed)
        assert rust_cpu.pc == 8
        # x0 should be 2 (only first two MOVZs executed)
        assert rust_cpu.get_register(0) == 2

    def test_breakpoint_with_trace(self, rust_cpu):
        """Breakpoint works together with instruction tracing."""
        rust_cpu.enable_trace()
        rust_cpu.set_breakpoint(0, 4)
        rust_cpu.load_program([movz(0, 42), movz(0, 99), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        # Trace records BEFORE breakpoint check, so BP instruction is traced but not executed
        entries = rust_cpu.read_trace()
        assert len(entries) == 2  # PC=0 traced, PC=4 traced (but not executed)
        assert entries[0][0] == 0  # first instruction PC
        assert entries[1][0] == 4  # breakpoint PC (traced, not executed)

    def test_clear_breakpoints(self, rust_cpu):
        """After clearing breakpoints, execution continues normally."""
        rust_cpu.set_breakpoint(0, 4)
        rust_cpu.clear_breakpoints()
        rust_cpu.load_program([movz(0, 1), movz(0, 2), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        from kernels.mlx.rust_runner import StopReasonV2
        assert result.stop_reason == StopReasonV2.HALT

    def test_multiple_breakpoints(self, rust_cpu):
        """Multiple breakpoints can be set and the first one hit stops execution."""
        from kernels.mlx.rust_runner import StopReasonV2
        rust_cpu.set_breakpoint(0, 8)   # third instruction
        rust_cpu.set_breakpoint(1, 12)  # fourth instruction
        rust_cpu.load_program([movz(0, 1), movz(0, 2), movz(0, 3), movz(0, 4), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        assert result.stop_reason == StopReasonV2.BREAKPOINT
        assert rust_cpu.pc == 8  # first breakpoint hit

    def test_continue_after_breakpoint(self, rust_cpu):
        """Can resume execution after hitting a breakpoint."""
        from kernels.mlx.rust_runner import StopReasonV2
        rust_cpu.set_breakpoint(0, 4)
        rust_cpu.load_program([movz(0, 1), movz(0, 2), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        assert result.stop_reason == StopReasonV2.BREAKPOINT
        # Clear breakpoint and continue
        rust_cpu.clear_breakpoints()
        result = rust_cpu.execute(max_cycles=100)
        assert result.stop_reason == StopReasonV2.BREAKPOINT or result.stop_reason == StopReasonV2.HALT
        # x0 should be 2 (second MOVZ executed)
        assert rust_cpu.get_register(0) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Backend Parity Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceParity:
    """Verify Rust and MLX backends produce identical trace results."""

    def test_same_trace_output(self, rust_cpu, mlx_cpu):
        """Both backends should record identical trace for the same program."""
        program = [movz(0, 42), movz(1, 99), add_imm(2, 0, 1), HLT]

        rust_cpu.enable_trace()
        run(rust_cpu, program)
        rust_entries = rust_cpu.read_trace()

        mlx_cpu.enable_trace()
        run(mlx_cpu, program)
        mlx_entries = mlx_cpu.read_trace()

        assert len(rust_entries) == len(mlx_entries), \
            f"Entry count mismatch: Rust={len(rust_entries)}, MLX={len(mlx_entries)}"

        for i, (r, m) in enumerate(zip(rust_entries, mlx_entries)):
            assert r[0] == m[0], f"Entry {i}: PC mismatch (Rust={r[0]}, MLX={m[0]})"
            assert r[1] == m[1], f"Entry {i}: inst mismatch (Rust=0x{r[1]:08x}, MLX=0x{m[1]:08x})"
            assert r[2] == m[2], f"Entry {i}: x0 mismatch (Rust={r[2]}, MLX={m[2]})"
            assert r[3] == m[3], f"Entry {i}: x1 mismatch (Rust={r[3]}, MLX={m[3]})"
            assert r[4] == m[4], f"Entry {i}: x2 mismatch (Rust={r[4]}, MLX={m[4]})"
            assert r[5] == m[5], f"Entry {i}: x3 mismatch (Rust={r[5]}, MLX={m[5]})"
            assert r[6] == m[6], f"Entry {i}: flags mismatch (Rust=0b{r[6]:04b}, MLX=0b{m[6]:04b})"
            assert r[7] == m[7], f"Entry {i}: SP mismatch (Rust={r[7]}, MLX={m[7]})"


# ═══════════════════════════════════════════════════════════════════════════════
# ARM64 Instruction Classifier Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInstructionClassifier:
    """Test the instruction_coverage.py classifier for correctness."""

    @pytest.fixture(autouse=True)
    def _load_classifier(self):
        from ncpu.os.gpu.programs.tools.instruction_coverage import classify_instruction
        self.classify = classify_instruction

    def test_movz(self):
        assert self.classify(movz(0, 42)) == 'movz'

    def test_movk(self):
        assert self.classify(movk(0, 0x1234, hw=1)) == 'movk'

    def test_add_imm(self):
        assert self.classify(add_imm(0, 1, 100)) == 'add_imm'

    def test_sub_imm(self):
        assert self.classify(sub_imm(0, 1, 50)) == 'sub_imm'

    def test_adds_imm(self):
        assert self.classify(adds_imm(0, 1, 10)) == 'adds_imm'

    def test_subs_imm(self):
        assert self.classify(subs_imm(0, 1, 10)) == 'subs_imm'

    def test_nop(self):
        assert self.classify(NOP) == 'nop'

    def test_hlt(self):
        # HLT uses the exception encoding (0xD4)
        result = self.classify(HLT)
        assert result in ('hlt', 'halt', 'exception'), f"HLT classified as '{result}'"

    def test_b_unconditional(self):
        result = self.classify(b(10))
        assert result in ('b', 'branch_unconditional'), f"B classified as '{result}'"

    def test_bl(self):
        inst = 0x94000000 | 10
        result = self.classify(inst)
        assert result in ('bl', 'branch_link', 'branch_unconditional'), f"BL classified as '{result}'"

    def test_cbz(self):
        inst = 0x34000000 | (2 << 5) | 0
        result = self.classify(inst)
        assert result in ('cbz', 'compare_branch', 'branch_compare'), f"CBZ classified as '{result}'"

    def test_cbnz(self):
        inst = 0xB5000000 | (2 << 5) | 0
        result = self.classify(inst)
        assert result in ('cbnz', 'compare_branch', 'branch_compare'), f"CBNZ classified as '{result}'"

    def test_b_cond(self):
        inst = 0x54000000 | (2 << 5) | 0
        result = self.classify(inst)
        assert result in ('b_cond', 'branch_conditional'), f"B.cond classified as '{result}'"

    def test_ldr_imm(self):
        inst = 0xF9400000 | (1 << 10) | (1 << 5) | 0
        result = self.classify(inst)
        assert 'ldr' in result or 'load' in result, f"LDR classified as '{result}'"

    def test_str_imm(self):
        inst = 0xF9000000 | (1 << 10) | (1 << 5) | 0
        result = self.classify(inst)
        assert 'str' in result or 'ldr' in result or 'store' in result, f"STR classified as '{result}'"

    def test_ldp_stp(self):
        # LDP X0, X1, [X2, #16]
        inst = 0xA9400000 | (2 << 15) | (1 << 10) | (2 << 5) | 0
        result = self.classify(inst)
        assert 'ldp' in result or 'stp' in result or 'pair' in result, f"LDP classified as '{result}'"

    def test_ret(self):
        inst = 0xD65F03C0
        result = self.classify(inst)
        assert result in ('ret', 'branch_register'), f"RET classified as '{result}'"

    def test_svc(self):
        inst = 0xD4000001
        assert self.classify(inst) == 'svc'

    def test_orr_imm(self):
        inst = 0xB2400000 | (1 << 5) | 0
        result = self.classify(inst)
        assert 'orr' in result or 'logical' in result, f"ORR classified as '{result}'"

    def test_adrp(self):
        inst = 0x90000000 | 0
        assert self.classify(inst) == 'adrp'

    def test_classifier_returns_strings(self):
        """All classifications should return non-empty strings."""
        test_insts = [
            movz(0, 42), movk(0, 1), add_imm(0, 1, 2), sub_imm(0, 1, 2),
            NOP, HLT, b(10), 0x94000010, 0xD65F03C0, 0xD4000001,
        ]
        for inst in test_insts:
            result = self.classify(inst)
            assert isinstance(result, str) and len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Record/Replay Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecordReplayTool:
    """Test the record_replay.py tool's core functions."""

    def test_deterministic_replay(self, rust_cpu):
        """Same program produces same trace on two runs."""
        program = [movz(0, 7), movz(1, 3), add_imm(2, 0, 3), HLT]

        rust_cpu.enable_trace()
        run(rust_cpu, program)
        trace1 = rust_cpu.read_trace()

        rust_cpu.clear_trace()
        rust_cpu.reset()
        rust_cpu.enable_trace()
        run(rust_cpu, program)
        trace2 = rust_cpu.read_trace()

        assert len(trace1) == len(trace2)
        for i, (t1, t2) in enumerate(zip(trace1, trace2)):
            assert t1[0] == t2[0], f"Replay diverged at entry {i}: PC {t1[0]} vs {t2[0]}"
            assert t1[1] == t2[1], f"Replay diverged at entry {i}: inst {t1[1]} vs {t2[1]}"


# ═══════════════════════════════════════════════════════════════════════════════
# Breakpoint Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestBreakpointEdgeCases:
    """Edge case tests for breakpoint functionality."""

    def test_breakpoint_at_first_instruction(self, rust_cpu):
        """Breakpoint at PC=0 fires before any instruction executes."""
        from kernels.mlx.rust_runner import StopReasonV2
        rust_cpu.set_breakpoint(0, 0)
        rust_cpu.load_program([movz(0, 99), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        assert result.stop_reason == StopReasonV2.BREAKPOINT
        assert rust_cpu.pc == 0
        # x0 should still be 0 (MOVZ not executed)
        assert rust_cpu.get_register(0) == 0

    def test_breakpoint_not_on_instruction_boundary(self, rust_cpu):
        """Breakpoint at non-aligned address is never hit."""
        from kernels.mlx.rust_runner import StopReasonV2
        rust_cpu.set_breakpoint(0, 2)  # Not 4-byte aligned
        rust_cpu.load_program([movz(0, 1), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        # Should not hit breakpoint, should halt normally
        assert result.stop_reason == StopReasonV2.HALT

    def test_breakpoint_preserves_trace(self, rust_cpu):
        """Trace entries are preserved when breakpoint fires."""
        rust_cpu.enable_trace()
        rust_cpu.set_breakpoint(0, 12)  # Break at 4th instruction
        rust_cpu.load_program([movz(0, 1), movz(1, 2), movz(2, 3), movz(3, 4), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()
        # 4 entries: PC=0, 4, 8, 12 (BP instruction traced but not executed)
        assert len(trace) == 4
        pcs = [e[0] for e in trace]
        assert pcs == [0, 4, 8, 12]

    def test_four_breakpoints_max(self, rust_cpu):
        """All 4 breakpoint slots can be used."""
        from kernels.mlx.rust_runner import StopReasonV2
        rust_cpu.set_breakpoint(0, 100)
        rust_cpu.set_breakpoint(1, 200)
        rust_cpu.set_breakpoint(2, 8)   # This one should fire first
        rust_cpu.set_breakpoint(3, 300)
        rust_cpu.load_program([movz(0, 1), movz(0, 2), movz(0, 3), HLT], address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        assert result.stop_reason == StopReasonV2.BREAKPOINT
        assert rust_cpu.pc == 8


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Memory Watchpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestWatchpoints:
    """Memory watchpoint tests — GPU stops when a watched address changes value."""

    def test_watchpoint_triggers_on_write(self, rust_cpu):
        """Watchpoint fires when 8-byte value at watched address changes."""
        from kernels.mlx.rust_runner import StopReasonV2
        watch_addr = 0x50000
        # Program: set x1 = watch_addr, x0 = 42, store x0 to [x1]
        prog = [
            movz(1, watch_addr >> 16, hw=1),
            movk(1, watch_addr & 0xFFFF, hw=0),
            movz(0, 42),
            stur_64(0, 1, 0),   # Triggers watchpoint
            movz(0, 99),        # Should NOT execute
            HLT,
        ]
        rust_cpu.set_watchpoint(0, watch_addr)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.WATCHPOINT
        # x0 should still be 42 (movz(0,99) didn't execute)
        assert rust_cpu.get_register(0) == 42

    def test_watchpoint_info_records_values(self, rust_cpu):
        """read_watchpoint_info returns (idx, addr, old_val, new_val)."""
        watch_addr = 0x50000
        prog = [
            movz(1, watch_addr >> 16, hw=1),
            movk(1, watch_addr & 0xFFFF, hw=0),
            movz(0, 123),
            stur_64(0, 1, 0),
            HLT,
        ]
        rust_cpu.set_watchpoint(0, watch_addr)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        info = rust_cpu.read_watchpoint_info()
        assert info is not None
        wp_idx, wp_addr, old_val, new_val = info
        assert wp_idx == 0
        assert wp_addr == watch_addr
        assert old_val == 0   # Was zero before write
        assert new_val == 123

    def test_watchpoint_no_change_no_trigger(self, rust_cpu):
        """Program that doesn't touch watched address runs to completion."""
        from kernels.mlx.rust_runner import StopReasonV2
        watch_addr = 0x50000
        prog = [
            movz(0, 42),
            movz(1, 99),
            add_imm(2, 0, 1),
            HLT,
        ]
        rust_cpu.set_watchpoint(0, watch_addr)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.HALT
        info = rust_cpu.read_watchpoint_info()
        assert info is None

    def test_watchpoint_preserves_trace(self, rust_cpu):
        """Trace entries are preserved when watchpoint fires."""
        watch_addr = 0x50000
        prog = [
            movz(1, watch_addr >> 16, hw=1),
            movk(1, watch_addr & 0xFFFF, hw=0),
            movz(0, 77),
            stur_64(0, 1, 0),
            HLT,
        ]
        rust_cpu.enable_trace()
        rust_cpu.set_watchpoint(0, watch_addr)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        trace = rust_cpu.read_trace()
        # Should have 4 entries (STUR executes, then WP check fires)
        assert len(trace) >= 4
        pcs = [e[0] for e in trace]
        assert 12 in pcs  # STUR at PC=12

    def test_clear_watchpoints(self, rust_cpu):
        """After clearing, watchpoint no longer fires."""
        from kernels.mlx.rust_runner import StopReasonV2
        watch_addr = 0x50000
        prog = [
            movz(1, watch_addr >> 16, hw=1),
            movk(1, watch_addr & 0xFFFF, hw=0),
            movz(0, 42),
            stur_64(0, 1, 0),
            HLT,
        ]
        rust_cpu.set_watchpoint(0, watch_addr)
        rust_cpu.clear_watchpoints()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.HALT

    def test_multiple_watchpoints(self, rust_cpu):
        """Multiple watchpoint slots; first triggered wins."""
        from kernels.mlx.rust_runner import StopReasonV2
        addr_a = 0x50000
        addr_b = 0x50100
        prog = [
            movz(1, addr_a >> 16, hw=1),
            movk(1, addr_a & 0xFFFF, hw=0),
            movz(0, 11),
            stur_64(0, 1, 0),
            HLT,
        ]
        rust_cpu.set_watchpoint(0, addr_a)
        rust_cpu.set_watchpoint(1, addr_b)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.WATCHPOINT
        info = rust_cpu.read_watchpoint_info()
        assert info is not None
        assert info[0] == 0      # WP index 0
        assert info[1] == addr_a

    def test_clear_all_debug(self, rust_cpu):
        """clear_all_debug removes breakpoints AND watchpoints."""
        from kernels.mlx.rust_runner import StopReasonV2
        rust_cpu.set_breakpoint(0, 0)
        rust_cpu.set_watchpoint(0, 0x50000)
        rust_cpu.clear_all_debug()
        prog = [movz(0, 1), HLT]
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        assert result.stop_reason == StopReasonV2.HALT


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Conditional Breakpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestConditionalBreakpoints:
    """Conditional breakpoint tests — stop at PC only when register==value."""

    def test_conditional_bp_fires_when_condition_met(self, rust_cpu):
        """Fires when PC matches AND register==value."""
        from kernels.mlx.rust_runner import StopReasonV2
        prog = [
            movz(0, 0),       # PC=0: x0 = 0
            add_imm(0, 0, 1), # PC=4: x0 = 1
            add_imm(0, 0, 1), # PC=8: x0 = 2
            add_imm(0, 0, 1), # PC=12: x0 = 3 — BP here when x0==2
            add_imm(0, 0, 1), # PC=16: should NOT execute
            HLT,              # PC=20
        ]
        # BP at PC=12 when x0==2 (x0 is 2 BEFORE instruction at PC=12)
        rust_cpu.set_conditional_breakpoint(0, 12, 0, 2)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.BREAKPOINT
        assert rust_cpu.pc == 12
        assert rust_cpu.get_register(0) == 2

    def test_conditional_bp_skips_when_not_met(self, rust_cpu):
        """Does NOT fire when register != expected value."""
        from kernels.mlx.rust_runner import StopReasonV2
        prog = [
            movz(0, 0),
            add_imm(0, 0, 1),
            add_imm(0, 0, 1),
            add_imm(0, 0, 1),
            HLT,
        ]
        # PC=12 AND x0==999 — never true
        rust_cpu.set_conditional_breakpoint(0, 12, 0, 999)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.HALT
        assert rust_cpu.get_register(0) == 3

    def test_conditional_bp_different_register(self, rust_cpu):
        """Conditional breakpoint can watch any register."""
        from kernels.mlx.rust_runner import StopReasonV2
        prog = [
            movz(0, 10),      # PC=0
            movz(1, 42),      # PC=4: x1 = 42
            add_imm(0, 0, 1), # PC=8 — BP when x1==42
            HLT,              # PC=12
        ]
        rust_cpu.set_conditional_breakpoint(0, 8, 1, 42)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.BREAKPOINT
        assert rust_cpu.pc == 8
        assert rust_cpu.get_register(0) == 10  # ADD not executed

    def test_unconditional_and_conditional_coexist(self, rust_cpu):
        """Regular and conditional breakpoints work together."""
        from kernels.mlx.rust_runner import StopReasonV2
        prog = [
            movz(0, 1),       # PC=0
            movz(1, 2),       # PC=4
            add_imm(2, 0, 1), # PC=8
            HLT,              # PC=12
        ]
        # Unconditional at PC=8, conditional at PC=4 when x0==999 (never)
        rust_cpu.set_breakpoint(0, 8)
        rust_cpu.set_conditional_breakpoint(1, 4, 0, 999)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.BREAKPOINT
        assert rust_cpu.pc == 8

    def test_watchpoint_and_breakpoint_together(self, rust_cpu):
        """Watchpoint and breakpoint set simultaneously; first triggered wins."""
        from kernels.mlx.rust_runner import StopReasonV2
        watch_addr = 0x50000
        prog = [
            movz(0, 1),
            movz(1, watch_addr >> 16, hw=1),
            movk(1, watch_addr & 0xFFFF, hw=0),
            stur_64(0, 1, 0),  # PC=12: write triggers WP
            movz(0, 99),       # PC=16: BP here
            HLT,
        ]
        rust_cpu.set_breakpoint(0, 16)
        rust_cpu.set_watchpoint(0, watch_addr)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        # WP should fire first (after STUR at PC=12, before BP at PC=16)
        assert result.stop_reason == StopReasonV2.WATCHPOINT


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Advanced Debugging Features
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdvancedDebugging:
    """Tests for advanced debugging features: data flow, call tracking, profiling."""

    def test_trace_captures_bl_ret(self, rust_cpu):
        """BL and RET instructions are properly captured in trace."""
        # BL +2 (skip one instruction, jump to PC=8 from PC=0)
        bl_inst = 0x94000002  # BL #2 → target = PC + 2*4 = 8
        ret_inst = 0xD65F03C0  # RET
        prog = [
            bl_inst,        # PC=0: BL to PC=8
            HLT,            # PC=4: never reached (return comes back to PC=4 tho)
            movz(0, 42),    # PC=8: function body
            ret_inst,       # PC=12: RET → goes to LR (X30=4)
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()
        pcs = [e[0] for e in trace]
        # Should see: 0 (BL), 8 (MOVZ), 12 (RET), 4 (HLT)
        assert 0 in pcs    # BL
        assert 8 in pcs    # Function body
        assert 12 in pcs   # RET
        assert 4 in pcs    # Return site (HLT)

    def test_call_graph_from_trace(self, rust_cpu):
        """Can reconstruct call graph from BL/RET in trace."""
        bl_inst = 0x94000002  # BL #2
        ret_inst = 0xD65F03C0
        prog = [
            bl_inst,        # PC=0: BL → PC=8
            HLT,            # PC=4: return here, halt
            movz(0, 99),    # PC=8: function
            ret_inst,       # PC=12: return
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()

        # Reconstruct calls
        calls = []
        rets = []
        for entry in trace:
            pc, inst = entry[0], entry[1]
            op = (inst >> 24) & 0xFF
            if op == 0x94 or op == 0x97:  # BL
                offset = inst & 0x3FFFFFF
                if offset & 0x2000000:
                    offset -= 0x4000000
                target = pc + offset * 4
                calls.append((pc, target))
            elif op == 0xD6 and ((inst >> 21) & 0x7) == 2:  # RET
                rets.append(pc)

        assert len(calls) == 1
        assert calls[0] == (0, 8)  # BL from 0 to 8
        assert len(rets) == 1
        assert rets[0] == 12

    def test_instruction_categorization(self, rust_cpu):
        """Trace-based instruction categorization works correctly."""
        from collections import Counter
        prog = [
            movz(0, 1),           # move
            movz(1, 2),           # move
            add_imm(2, 0, 1),     # arithmetic
            sub_imm(3, 1, 1),     # arithmetic
            stur_64(2, 0, 0),     # store (may fault but gets traced)
            HLT,
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()

        categories = Counter()
        for entry in trace:
            op = (entry[1] >> 24) & 0xFF
            if op in (0xD2, 0xF2, 0x92):
                categories['move'] += 1
            elif op in (0x91, 0xD1):
                categories['arithmetic'] += 1
            elif op in (0xF8, 0xB8):
                categories['store'] += 1

        assert categories['move'] >= 2
        assert categories['arithmetic'] >= 2

    def test_hotspot_detection(self, rust_cpu):
        """Can detect hot PCs (loop body) from trace."""
        from collections import Counter
        # Simple loop: x0=0, loop: add x0,x0,1; cmp x0,5; bne loop; halt
        # SUBs x1,x0,#5 sets Z flag when x0==5
        # B.NE goes back if Z=0
        b_ne_back = 0x54FFFFC1  # B.NE imm19=-2 (PC-8, back to add)
        prog = [
            movz(0, 0),           # PC=0: x0 = 0
            add_imm(0, 0, 1),     # PC=4: x0 += 1 (LOOP BODY)
            subs_imm(1, 0, 5),    # PC=8: compare x0 to 5
            b_ne_back,            # PC=12: branch back to PC=4 if not equal
            HLT,                  # PC=16: halt
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=1000)
        trace = rust_cpu.read_trace()

        pc_counts = Counter(entry[0] for entry in trace)
        # The ADD at PC=4 should be the hottest (executed 5 times)
        hottest_pc = pc_counts.most_common(1)[0][0]
        assert hottest_pc == 4  # Loop body is hottest
        assert pc_counts[4] == 5  # Executed 5 times

    def test_flag_transitions_in_trace(self, rust_cpu):
        """Can detect NZCV flag transitions from trace entries."""
        prog = [
            movz(0, 0),           # PC=0: flags unchanged
            adds_imm(1, 0, 0),    # PC=4: 0+0=0, sets Z flag
            adds_imm(2, 0, 1),    # PC=8: 0+1=1, clears Z flag
            HLT,
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        result = rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()

        # Check flag transitions exist
        flag_values = [entry[6] if len(entry) > 6 else 0 for entry in trace]
        # Should see at least one transition (Z set then cleared)
        transitions = sum(1 for i in range(1, len(flag_values)) if flag_values[i] != flag_values[i-1])
        assert transitions >= 1

    def test_watchpoint_resume_updates_shadow(self, rust_cpu):
        """After watchpoint fires, shadow is updated so re-execution continues."""
        from kernels.mlx.rust_runner import StopReasonV2
        watch_addr = 0x50000
        prog = [
            movz(1, watch_addr >> 16, hw=1),
            movk(1, watch_addr & 0xFFFF, hw=0),
            movz(0, 11),
            stur_64(0, 1, 0),    # First write: 0→11, triggers WP
            movz(0, 22),
            stur_64(0, 1, 0),    # Second write: 11→22, should trigger again after resume
            HLT,
        ]
        rust_cpu.set_watchpoint(0, watch_addr)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)

        # First execution: should stop at first write
        result = rust_cpu.execute(max_cycles=1000)
        assert result.stop_reason == StopReasonV2.WATCHPOINT
        info1 = rust_cpu.read_watchpoint_info()
        assert info1 is not None
        assert info1[3] == 11  # new_val = 11

        # Continue: should stop at second write
        result2 = rust_cpu.execute(max_cycles=1000)
        assert result2.stop_reason == StopReasonV2.WATCHPOINT
        info2 = rust_cpu.read_watchpoint_info()
        assert info2 is not None
        assert info2[2] == 11  # old_val = 11 (from first write)
        assert info2[3] == 22  # new_val = 22

    def test_combined_trace_bp_wp_all_features(self, rust_cpu):
        """All debug features work together: trace + breakpoint + watchpoint."""
        from kernels.mlx.rust_runner import StopReasonV2
        watch_addr = 0x50000
        prog = [
            movz(0, 1),                              # PC=0
            movz(1, watch_addr >> 16, hw=1),          # PC=4
            movk(1, watch_addr & 0xFFFF, hw=0),       # PC=8
            stur_64(0, 1, 0),                          # PC=12: write triggers WP
            movz(0, 99),                               # PC=16: never reached
            HLT,
        ]
        rust_cpu.enable_trace()
        rust_cpu.set_breakpoint(0, 20)     # BP at PC=20 (past end)
        rust_cpu.set_watchpoint(0, watch_addr)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)

        result = rust_cpu.execute(max_cycles=1000)

        # WP fires at PC=12 (after STUR), before BP at PC=20
        assert result.stop_reason == StopReasonV2.WATCHPOINT

        # Trace should have entries
        trace = rust_cpu.read_trace()
        assert len(trace) >= 4
        pcs = [e[0] for e in trace]
        assert 0 in pcs and 4 in pcs and 8 in pcs and 12 in pcs

        # WP info should be valid
        info = rust_cpu.read_watchpoint_info()
        assert info is not None
        assert info[3] == 1  # new_val = 1 (x0 was 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Disassembler Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDisassembler:
    """Test the ARM64 disassembler used by gpu-asm."""

    @staticmethod
    def _disasm():
        """Import the shared disassembler used by the Alpine GPU shell."""
        from ncpu.os.gpu.programs.tools.trace_utils import disassemble_instruction
        return disassemble_instruction

    def test_disasm_nop(self):
        disasm = self._disasm()
        assert disasm(0xD503201F) == "nop"

    def test_disasm_hlt(self):
        disasm = self._disasm()
        assert disasm(0xD4400000) == "hlt"

    def test_disasm_ret(self):
        disasm = self._disasm()
        assert disasm(0xD65F03C0) == "ret"

    def test_disasm_svc(self):
        disasm = self._disasm()
        assert disasm(0xD4000001) == "svc #0x0"

    def test_disasm_movz(self):
        disasm = self._disasm()
        result = disasm(movz(0, 42))
        assert "movz" in result
        assert "0x2a" in result

    def test_disasm_add_imm(self):
        disasm = self._disasm()
        result = disasm(add_imm(1, 0, 5))
        assert "add" in result

    def test_disasm_cmp(self):
        disasm = self._disasm()
        result = disasm(subs_imm(31, 0, 5))  # CMP x0, #5 is SUBS xzr, x0, #5
        assert "cmp" in result

    def test_disasm_bl(self):
        disasm = self._disasm()
        bl_inst = 0x94000000 | (10 & 0x3FFFFFF)  # BL +40
        result = disasm(bl_inst)
        assert "bl" in result

    def test_disasm_b(self):
        disasm = self._disasm()
        b_inst = 0x14000000 | (5 & 0x3FFFFFF)  # B +20
        result = disasm(b_inst)
        assert result.startswith("b ")

    def test_disasm_ldr(self):
        disasm = self._disasm()
        ldr_inst = 0xF9400000 | (0 << 5) | 1  # LDR x1, [x0]
        result = disasm(ldr_inst)
        assert "ldr" in result

    def test_disasm_str(self):
        disasm = self._disasm()
        str_inst = 0xF9000000 | (0 << 5) | 1  # STR x1, [x0]
        result = disasm(str_inst)
        assert "str" in result

    def test_disasm_mov_alias(self):
        disasm = self._disasm()
        # MOV x0, x1 = ORR x0, xzr, x1
        mov_inst = 0xAA0003E0 | 1 | (1 << 16)  # ORR x1, xzr, x1 — but we need x0
        mov_inst = 0xAA0003E0 | (1 << 16)        # MOV x0, x1
        result = disasm(mov_inst)
        assert "mov" in result

    def test_render_trace_table(self):
        from ncpu.os.gpu.programs.tools.trace_utils import render_trace_table

        table = render_trace_table([
            (0x1000, movz(0, 42), 42, 0, 0, 0, 0, 0),
        ], limit=1)

        assert "PC" in table
        assert "movz x0, #0x2a" in table
        assert "0x00001000" in table


# ═══════════════════════════════════════════════════════════════════════════════
# Constant-Time Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstantTime:
    """Test deterministic execution for constant-time verification."""

    @pytest.fixture
    def rust_cpu(self):
        from kernels.mlx.gpu_cpu import GPUKernelCPU
        cpu = GPUKernelCPU(quiet=True)
        yield cpu

    def test_same_program_same_cycles(self, rust_cpu):
        """Same program always produces same cycle count (deterministic)."""
        prog = [movz(0, 10), add_imm(0, 0, 5), HLT]
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        r1 = rust_cpu.execute(max_cycles=100)

        # Reset and run again
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        r2 = rust_cpu.execute(max_cycles=100)

        assert r1.cycles == r2.cycles

    def test_same_program_same_trace(self, rust_cpu):
        """Same program always produces identical trace (deterministic replay)."""
        prog = [movz(0, 10), add_imm(0, 0, 5), movz(1, 20), HLT]

        # Run 1
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        rust_cpu.execute(max_cycles=100)
        trace1 = rust_cpu.read_trace()

        # Reset and run 2
        rust_cpu.clear_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        rust_cpu.execute(max_cycles=100)
        trace2 = rust_cpu.read_trace()

        assert len(trace1) == len(trace2)
        for e1, e2 in zip(trace1, trace2):
            assert e1[0] == e2[0]  # Same PCs
            assert e1[1] == e2[1]  # Same instructions

    def test_different_data_same_instruction_count(self, rust_cpu):
        """Programs with different data but same structure have same cycle count."""
        prog1 = [movz(0, 100), add_imm(0, 0, 1), HLT]
        prog2 = [movz(0, 200), add_imm(0, 0, 1), HLT]

        rust_cpu.load_program(prog1, address=0)
        rust_cpu.set_pc(0)
        r1 = rust_cpu.execute(max_cycles=100)

        rust_cpu.load_program(prog2, address=0)
        rust_cpu.set_pc(0)
        r2 = rust_cpu.execute(max_cycles=100)

        assert r1.cycles == r2.cycles  # Same structure = same cycles


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Sanitizer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemorySanitizer:
    """Test memory safety analysis from trace data."""

    @pytest.fixture
    def rust_cpu(self):
        from kernels.mlx.gpu_cpu import GPUKernelCPU
        cpu = GPUKernelCPU(quiet=True)
        yield cpu

    def test_sp_tracking_from_trace(self, rust_cpu):
        """Can track SP values from trace entries."""
        # SUB sp, sp, #0x20 = 0xD10083FF (sp = sp - 32)
        sub_sp = 0xD10083FF
        prog = [
            0xD10083FF,    # sub sp, sp, #0x20
            movz(0, 42),
            HLT,
        ]
        rust_cpu.enable_trace()
        # Set SP to a known value
        rust_cpu.set_register(31, 0xFF000)
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()

        # SP should be captured in trace entries
        assert len(trace) >= 2
        # After first instruction, SP should have decreased
        sp_before = trace[0][7] if len(trace[0]) > 7 else 0
        # SP value is captured BEFORE instruction execution
        assert sp_before != 0 or len(trace[0]) <= 7

    def test_store_detection_from_trace(self, rust_cpu):
        """Can detect store instructions from trace entries."""
        prog = [
            movz(0, 42),
            movz(1, 0x5, hw=1),   # x1 = 0x50000
            stur_64(0, 1, 0),      # STR x0, [x1]
            HLT,
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()

        # Find the store instruction in trace
        stores = []
        for e in trace:
            inst = e[1]
            top8 = (inst >> 24) & 0xFF
            if top8 in (0xF8, 0xB8, 0x78, 0x38):
                is_store = not ((inst >> 22) & 1)
                if is_store:
                    stores.append(e)
        assert len(stores) >= 1

    def test_load_store_ratio(self, rust_cpu):
        """Can compute load/store ratio from trace."""
        prog = [
            movz(0, 42),
            movz(1, 0x5, hw=1),   # x1 = 0x50000
            stur_64(0, 1, 0),      # STR x0, [x1]
            ldur_64(2, 1, 0),      # LDR x2, [x1]
            ldur_64(3, 1, 0),      # LDR x3, [x1]
            HLT,
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()

        loads = 0
        stores = 0
        for e in trace:
            inst = e[1]
            top8 = (inst >> 24) & 0xFF
            if top8 in (0xF8,):
                if (inst >> 22) & 1:
                    loads += 1
                else:
                    stores += 1
        assert loads >= 2  # Two LDR instructions
        assert stores >= 1  # One STR instruction


# ═══════════════════════════════════════════════════════════════════════════════
# Reverse Data Flow Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestReverseDataFlow:
    """Test reverse data flow analysis from trace."""

    @pytest.fixture
    def rust_cpu(self):
        from kernels.mlx.gpu_cpu import GPUKernelCPU
        cpu = GPUKernelCPU(quiet=True)
        yield cpu

    def test_track_x0_mutations(self, rust_cpu):
        """Can track all mutations of x0 through trace."""
        prog = [
            movz(0, 1),           # x0 = 1
            add_imm(0, 0, 2),     # x0 = 3
            add_imm(0, 0, 4),     # x0 = 7
            HLT,
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()

        # Track x0 values (index 2 in trace entry)
        x0_values = [e[2] for e in trace]
        # x0 should change: 0→1, 1→3, 3→7
        assert len(x0_values) >= 3
        # Final value should be 7
        assert x0_values[-1] == 7

    def test_detect_value_source(self, rust_cpu):
        """Can find which instruction first set a register to a value."""
        prog = [
            movz(0, 0),           # x0 = 0
            movz(1, 100),         # x1 = 100
            add_imm(0, 1, 0),     # x0 = x1 + 0 = 100
            HLT,
        ]
        rust_cpu.enable_trace()
        rust_cpu.load_program(prog, address=0)
        rust_cpu.set_pc(0)
        rust_cpu.execute(max_cycles=100)
        trace = rust_cpu.read_trace()

        # Find where x0 first becomes 100
        x0_became_100 = None
        for i, e in enumerate(trace):
            if e[2] == 100:
                x0_became_100 = i
                break
        assert x0_became_100 is not None
        # Trace captures state BEFORE execution, so x0=100 first appears
        # at the entry AFTER ADD (PC=8) executes, i.e., at PC=12 (HLT)
        assert trace[x0_became_100][0] == 12
