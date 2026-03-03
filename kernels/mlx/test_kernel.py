#!/usr/bin/env python3
"""
Comprehensive Tests for MLX ARM64 CPU Kernel.

This module tests all supported instructions and execution features
of the Metal GPU kernel. Tests are designed to:

1. Verify correctness of each instruction
2. Test edge cases (zero register, overflow, sign extension)
3. Validate control flow (branches, conditions)
4. Compare results with expected values

Run tests:
    python -m mlx_kernel.test_kernel
    # or
    pytest mlx_kernel/test_kernel.py -v

Author: KVRM Project
Date: 2024
"""

import sys
import numpy as np
from typing import Callable, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, '..')

from mlx_kernel import MLXKernelCPU, StopReason


# ═══════════════════════════════════════════════════════════════════════════════
# TEST UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class TestResult:
    """Result of a single test."""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


def run_test(name: str, test_fn: Callable) -> TestResult:
    """Run a test function and return result."""
    try:
        test_fn()
        return TestResult(name, True)
    except AssertionError as e:
        return TestResult(name, False, str(e))
    except Exception as e:
        return TestResult(name, False, f"Exception: {e}")


def encode_add_imm(rd: int, rn: int, imm12: int, sf: int = 1) -> int:
    """Encode ADD Rd, Rn, #imm12 instruction."""
    # ADD (immediate): sf=1 for 64-bit, op=0, S=0
    # 31 30 29 | 28:24 | 23:22 | 21:10     | 9:5 | 4:0
    # sf  0  0 | 10001 | shift | imm12     | Rn  | Rd
    return (sf << 31) | (0b10001 << 24) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd


def encode_sub_imm(rd: int, rn: int, imm12: int, sf: int = 1) -> int:
    """Encode SUB Rd, Rn, #imm12 instruction."""
    # SUB (immediate): sf=1 for 64-bit, op=1, S=0
    return (sf << 31) | (0b11010001 << 24) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd


def encode_subs_imm(rd: int, rn: int, imm12: int, sf: int = 1) -> int:
    """Encode SUBS Rd, Rn, #imm12 instruction (sets flags)."""
    # SUBS (immediate): sf=1 for 64-bit, op=1, S=1
    return (sf << 31) | (0b11110001 << 24) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd


def encode_movz(rd: int, imm16: int, hw: int = 0, sf: int = 1) -> int:
    """Encode MOVZ Rd, #imm16, LSL #(hw*16) instruction."""
    # MOVZ: sf=1 for 64-bit, opc=10
    return (sf << 31) | (0b10100101 << 23) | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd


def encode_movk(rd: int, imm16: int, hw: int = 0, sf: int = 1) -> int:
    """Encode MOVK Rd, #imm16, LSL #(hw*16) instruction."""
    # MOVK: sf=1 for 64-bit, opc=11
    return (sf << 31) | (0b11100101 << 23) | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd


def encode_add_reg(rd: int, rn: int, rm: int, sf: int = 1) -> int:
    """Encode ADD Rd, Rn, Rm instruction."""
    # ADD (shifted register): sf=1, op=0, S=0
    return (sf << 31) | (0b0001011 << 24) | (rm << 16) | (rn << 5) | rd


def encode_sub_reg(rd: int, rn: int, rm: int, sf: int = 1) -> int:
    """Encode SUB Rd, Rn, Rm instruction."""
    return (sf << 31) | (0b1001011 << 24) | (rm << 16) | (rn << 5) | rd


def encode_and_reg(rd: int, rn: int, rm: int, sf: int = 1) -> int:
    """Encode AND Rd, Rn, Rm instruction."""
    return (sf << 31) | (0b0001010 << 24) | (rm << 16) | (rn << 5) | rd


def encode_orr_reg(rd: int, rn: int, rm: int, sf: int = 1) -> int:
    """Encode ORR Rd, Rn, Rm instruction."""
    return (sf << 31) | (0b0101010 << 24) | (rm << 16) | (rn << 5) | rd


def encode_eor_reg(rd: int, rn: int, rm: int, sf: int = 1) -> int:
    """Encode EOR Rd, Rn, Rm instruction."""
    return (sf << 31) | (0b1001010 << 24) | (rm << 16) | (rn << 5) | rd


def encode_b(offset: int) -> int:
    """Encode B label instruction. Offset is in bytes, must be 4-byte aligned."""
    imm26 = (offset // 4) & 0x3FFFFFF
    return 0x14000000 | imm26


def encode_bl(offset: int) -> int:
    """Encode BL label instruction."""
    imm26 = (offset // 4) & 0x3FFFFFF
    return 0x94000000 | imm26


def encode_cbz(rt: int, offset: int, sf: int = 1) -> int:
    """Encode CBZ Rt, label instruction."""
    imm19 = (offset // 4) & 0x7FFFF
    return (sf << 31) | 0x34000000 | (imm19 << 5) | rt


def encode_cbnz(rt: int, offset: int, sf: int = 1) -> int:
    """Encode CBNZ Rt, label instruction."""
    imm19 = (offset // 4) & 0x7FFFF
    return (sf << 31) | 0x35000000 | (imm19 << 5) | rt


def encode_ret(rn: int = 30) -> int:
    """Encode RET instruction (RET Xn, default X30)."""
    return 0xD65F0000 | (rn << 5)


def encode_hlt(imm16: int = 0) -> int:
    """Encode HLT instruction."""
    return 0xD4400000 | (imm16 << 5)


def encode_svc(imm16: int = 0) -> int:
    """Encode SVC instruction (syscall)."""
    return 0xD4000001 | (imm16 << 5)


def encode_nop() -> int:
    """Encode NOP instruction."""
    return 0xD503201F


def encode_ldr(rt: int, rn: int, imm12: int) -> int:
    """Encode LDR Xt, [Xn, #imm12*8] instruction."""
    # imm12 is scaled by 8 for 64-bit loads
    return 0xF9400000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rt


def encode_str(rt: int, rn: int, imm12: int) -> int:
    """Encode STR Xt, [Xn, #imm12*8] instruction."""
    return 0xF9000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rt


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

def test_movz_basic():
    """Test MOVZ instruction with various immediates."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 0x1234),        # X0 = 0x1234
        encode_movz(1, 0xFFFF),        # X1 = 0xFFFF
        encode_movz(2, 0x5678, hw=1),  # X2 = 0x5678 << 16 = 0x56780000
        encode_movz(3, 0xABCD, hw=2),  # X3 = 0xABCD << 32
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 0x1234, f"X0: got {cpu.get_register(0):X}"
    assert cpu.get_register(1) == 0xFFFF, f"X1: got {cpu.get_register(1):X}"
    assert cpu.get_register(2) == 0x56780000, f"X2: got {cpu.get_register(2):X}"
    assert cpu.get_register(3) == 0xABCD00000000, f"X3: got {cpu.get_register(3):X}"


def test_movk_basic():
    """Test MOVK instruction (keep other bits)."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 0x1111),        # X0 = 0x1111
        encode_movk(0, 0x2222, hw=1),  # X0 = 0x22221111
        encode_movk(0, 0x3333, hw=2),  # X0 = 0x333322221111
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 0x333322221111, f"X0: got {cpu.get_register(0):X}"


def test_add_imm():
    """Test ADD immediate instruction."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 100),           # X0 = 100
        encode_add_imm(1, 0, 50),      # X1 = X0 + 50 = 150
        encode_add_imm(2, 1, 0xFFF),   # X2 = X1 + 4095 = 4245
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(1) == 150, f"X1: got {cpu.get_register(1)}"
    assert cpu.get_register(2) == 4245, f"X2: got {cpu.get_register(2)}"


def test_sub_imm():
    """Test SUB immediate instruction."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 1000),          # X0 = 1000
        encode_sub_imm(1, 0, 100),     # X1 = X0 - 100 = 900
        encode_sub_imm(2, 1, 500),     # X2 = X1 - 500 = 400
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(1) == 900, f"X1: got {cpu.get_register(1)}"
    assert cpu.get_register(2) == 400, f"X2: got {cpu.get_register(2)}"


def test_add_reg():
    """Test ADD register instruction."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 100),           # X0 = 100
        encode_movz(1, 200),           # X1 = 200
        encode_add_reg(2, 0, 1),       # X2 = X0 + X1 = 300
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(2) == 300, f"X2: got {cpu.get_register(2)}"


def test_sub_reg():
    """Test SUB register instruction."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 500),           # X0 = 500
        encode_movz(1, 200),           # X1 = 200
        encode_sub_reg(2, 0, 1),       # X2 = X0 - X1 = 300
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(2) == 300, f"X2: got {cpu.get_register(2)}"


def test_logical_ops():
    """Test AND, ORR, EOR register instructions."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 0xFF00),        # X0 = 0xFF00
        encode_movz(1, 0x0FF0),        # X1 = 0x0FF0
        encode_and_reg(2, 0, 1),       # X2 = X0 & X1 = 0x0F00
        encode_orr_reg(3, 0, 1),       # X3 = X0 | X1 = 0xFFF0
        encode_eor_reg(4, 0, 1),       # X4 = X0 ^ X1 = 0xF0F0
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(2) == 0x0F00, f"X2: got {cpu.get_register(2):X}"
    assert cpu.get_register(3) == 0xFFF0, f"X3: got {cpu.get_register(3):X}"
    assert cpu.get_register(4) == 0xF0F0, f"X4: got {cpu.get_register(4):X}"


def test_xzr_read():
    """Test that reading XZR (X31) returns 0."""
    cpu = MLXKernelCPU()

    # ORR X0, XZR, X31 should give X0 = 0 | 0 = 0
    # We'll use ADD X0, XZR, #100 - should be 0 + 100 = 100
    program = [
        encode_movz(0, 0xFFFF),        # X0 = 0xFFFF (set to non-zero first)
        encode_add_imm(0, 31, 100),    # X0 = XZR + 100 = 0 + 100 = 100
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 100, f"X0: got {cpu.get_register(0)}"


def test_xzr_write():
    """Test that writing to XZR (X31) is ignored."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(31, 0x1234),       # XZR = 0x1234 (should be ignored)
        encode_add_imm(0, 31, 0),      # X0 = XZR + 0 = 0
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 0, f"X0: got {cpu.get_register(0)} (XZR should be 0)"


def test_branch_unconditional():
    """Test B (unconditional branch)."""
    cpu = MLXKernelCPU()

    # B +8 (skip next instruction)
    program = [
        encode_movz(0, 1),             # X0 = 1
        encode_b(8),                   # B +8 (skip next)
        encode_movz(0, 99),            # X0 = 99 (should be skipped)
        encode_movz(1, 2),             # X1 = 2
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 1, f"X0: got {cpu.get_register(0)} (should be 1, not 99)"
    assert cpu.get_register(1) == 2, f"X1: got {cpu.get_register(1)}"


def test_branch_with_link():
    """Test BL (branch with link) - saves return address."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 1),             # 0x00: X0 = 1
        encode_bl(8),                  # 0x04: BL +8, X30 = 0x08
        encode_hlt(),                  # 0x08: HLT (skipped, then returned to)
        encode_movz(0, 2),             # 0x0C: X0 = 2 (called)
        encode_ret(),                  # 0x10: RET (returns to 0x08)
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 2, f"X0: got {cpu.get_register(0)} (should be 2 from called code)"


def test_cbz_taken():
    """Test CBZ when register is zero (branch taken)."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 0),             # X0 = 0
        encode_cbz(0, 8),              # CBZ X0, +8 (should branch)
        encode_movz(1, 99),            # X1 = 99 (should be skipped)
        encode_movz(2, 1),             # X2 = 1 (landed here)
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(1) == 0, f"X1: got {cpu.get_register(1)} (should be 0, skipped)"
    assert cpu.get_register(2) == 1, f"X2: got {cpu.get_register(2)}"


def test_cbz_not_taken():
    """Test CBZ when register is non-zero (branch not taken)."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 5),             # X0 = 5 (non-zero)
        encode_cbz(0, 8),              # CBZ X0, +8 (should NOT branch)
        encode_movz(1, 1),             # X1 = 1 (executed)
        encode_movz(2, 99),            # X2 = 99 (also executed)
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(1) == 1, f"X1: got {cpu.get_register(1)}"
    assert cpu.get_register(2) == 99, f"X2: got {cpu.get_register(2)}"


def test_cbnz_taken():
    """Test CBNZ when register is non-zero (branch taken)."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 5),             # X0 = 5 (non-zero)
        encode_cbnz(0, 8),             # CBNZ X0, +8 (should branch)
        encode_movz(1, 99),            # X1 = 99 (should be skipped)
        encode_movz(2, 1),             # X2 = 1 (landed here)
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(1) == 0, f"X1: got {cpu.get_register(1)} (should be 0, skipped)"
    assert cpu.get_register(2) == 1, f"X2: got {cpu.get_register(2)}"


def test_loop():
    """Test a simple counting loop."""
    cpu = MLXKernelCPU()

    # Count from 0 to 100
    program = [
        encode_movz(0, 0),             # X0 = 0 (counter)
        encode_movz(1, 100),           # X1 = 100 (limit)
        # loop:
        encode_add_imm(0, 0, 1),       # X0 = X0 + 1
        encode_subs_imm(2, 0, 100),    # X2 = X0 - 100, set flags
        encode_cbnz(2, -8),            # CBNZ X2, loop (while X0 != 100)
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=10000)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 100, f"X0: got {cpu.get_register(0)} (should be 100)"


def test_syscall():
    """Test SVC (syscall) instruction stops execution."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 1),             # X0 = 1
        encode_movz(8, 60),            # X8 = 60 (exit syscall number)
        encode_svc(0),                 # SVC #0
        encode_movz(0, 99),            # X0 = 99 (should not execute)
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.SYSCALL
    assert cpu.get_register(0) == 1, f"X0: got {cpu.get_register(0)} (should be 1, not 99)"
    assert cpu.get_register(8) == 60, f"X8: got {cpu.get_register(8)}"


def test_halt():
    """Test HLT instruction stops execution."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 42),
        encode_hlt(),
        encode_movz(0, 99),            # Should not execute
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 42, f"X0: got {cpu.get_register(0)}"


def test_max_cycles():
    """Test that execution stops at max_cycles."""
    cpu = MLXKernelCPU()

    # Infinite loop: MOVZ, ADD, B (back to ADD)
    # After 1000 cycles: 1 MOVZ + ~500 ADDs + ~499 Branches = 1000
    # So X0 should be around 500
    program = [
        encode_movz(0, 0),
        encode_add_imm(0, 0, 1),       # X0++
        encode_b(-4),                  # B back to ADD
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=1000)

    assert result.stop_reason == StopReason.MAX_CYCLES
    assert result.cycles == 1000
    # X0 should be close to 500 (each ADD+B pair is 2 cycles)
    # 1 MOVZ + (1000-1)/2 iterations = ~500
    x0 = cpu.get_register(0)
    assert 490 <= x0 <= 510, f"X0: got {x0}, expected around 500"


def test_nop():
    """Test NOP instruction."""
    cpu = MLXKernelCPU()

    program = [
        encode_movz(0, 1),
        encode_nop(),
        encode_nop(),
        encode_nop(),
        encode_movz(1, 2),
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 1
    assert cpu.get_register(1) == 2


def test_memory_store_load():
    """Test LDR instruction.

    NOTE: STR is disabled in Phase 1 (MLX inputs are read-only).
    This test only verifies LDR from pre-loaded memory.
    """
    cpu = MLXKernelCPU()

    # Pre-load data at address 0x100 (256 decimal)
    # Using smaller address that's easier to encode
    test_value = 0x56781234DEADBEEF
    data = test_value.to_bytes(8, 'little')
    cpu.write_memory(0x100, data)  # Address 256

    # Load from pre-loaded memory
    # We need X1 = 0x100 (256)
    program = [
        encode_movz(1, 0x100),         # X1 = 0x100 (256)
        encode_ldr(2, 1, 0),           # LDR X2, [X1, #0]
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    actual = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
    assert actual == test_value, f"X2: got {actual:X}, expected {test_value:X}"


def test_backward_branch():
    """Test backward branch (negative offset)."""
    cpu = MLXKernelCPU()

    # Simple backward loop: count down from 5 to 0
    program = [
        encode_movz(0, 5),             # X0 = 5
        # loop:
        encode_sub_imm(0, 0, 1),       # X0 = X0 - 1
        encode_cbnz(0, -4),            # CBNZ X0, loop
        encode_movz(1, 1),             # X1 = 1 (done)
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 0, f"X0: got {cpu.get_register(0)}"
    assert cpu.get_register(1) == 1, f"X1: got {cpu.get_register(1)}"


def test_large_immediate():
    """Test building large 64-bit values with MOVZ/MOVK."""
    cpu = MLXKernelCPU()

    # Build 0xDEADBEEFCAFEBABE
    program = [
        encode_movz(0, 0xBABE, hw=0),  # X0 = 0xBABE
        encode_movk(0, 0xCAFE, hw=1),  # X0 = 0xCAFEBABE
        encode_movk(0, 0xBEEF, hw=2),  # X0 = 0xBEEFCAFEBABE
        encode_movk(0, 0xDEAD, hw=3),  # X0 = 0xDEADBEEFCAFEBABE
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=100)

    assert result.stop_reason == StopReason.HALT
    expected = 0xDEADBEEFCAFEBABE
    actual = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF
    assert actual == expected, f"X0: got {actual:016X}, expected {expected:016X}"


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_performance():
    """Test performance with a large number of instructions."""
    cpu = MLXKernelCPU()

    # Simple counting loop for 100,000 iterations
    program = [
        encode_movz(0, 0),             # X0 = 0 (counter)
        encode_movz(1, 0),             # X1 = 0 (will hold 100000)
        encode_movk(1, 0x86A0 & 0xFFFF, hw=0),  # X1 low = 34464
        encode_movk(1, (0x186A0 >> 16) & 0xFFFF, hw=1),  # X1 = 100000
        # loop:
        encode_add_imm(0, 0, 1),       # X0++
        encode_sub_reg(2, 0, 1),       # X2 = X0 - X1
        encode_cbnz(2, -8),            # CBNZ X2, loop
        encode_hlt(),
    ]

    cpu.load_program(program)
    result = cpu.execute(max_cycles=1_000_000)

    print(f"\n  Performance: {result.cycles:,} cycles in {result.elapsed_seconds*1000:.2f}ms")
    print(f"  IPS: {result.ips:,.0f}")

    assert result.stop_reason == StopReason.HALT
    assert cpu.get_register(0) == 100000, f"X0: got {cpu.get_register(0)}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("MOVZ basic", test_movz_basic),
        ("MOVK basic", test_movk_basic),
        ("ADD immediate", test_add_imm),
        ("SUB immediate", test_sub_imm),
        ("ADD register", test_add_reg),
        ("SUB register", test_sub_reg),
        ("Logical ops (AND/ORR/EOR)", test_logical_ops),
        ("XZR read", test_xzr_read),
        ("XZR write", test_xzr_write),
        ("Branch unconditional", test_branch_unconditional),
        ("Branch with link", test_branch_with_link),
        ("CBZ taken", test_cbz_taken),
        ("CBZ not taken", test_cbz_not_taken),
        ("CBNZ taken", test_cbnz_taken),
        ("Loop", test_loop),
        ("Syscall", test_syscall),
        ("Halt", test_halt),
        ("Max cycles", test_max_cycles),
        ("NOP", test_nop),
        ("Memory store/load", test_memory_store_load),
        ("Backward branch", test_backward_branch),
        ("Large immediate", test_large_immediate),
        ("Performance", test_performance),
    ]

    print("=" * 70)
    print("MLX ARM64 CPU KERNEL - COMPREHENSIVE TESTS")
    print("=" * 70)

    passed = 0
    failed = 0
    results = []

    for name, test_fn in tests:
        result = run_test(name, test_fn)
        results.append(result)

        if result.passed:
            print(f"  [{chr(0x2713)}] {name}")
            passed += 1
        else:
            print(f"  [X] {name}: {result.message}")
            failed += 1

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
