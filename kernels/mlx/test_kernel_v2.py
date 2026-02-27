#!/usr/bin/env python3
"""
Comprehensive tests for MLX Kernel V2 (Double-Buffer Memory).

Tests all instructions including STR/STRB memory writes.
"""

import sys
sys.path.insert(0, '.')

from mlx_kernel.cpu_kernel_v2 import MLXKernelCPUv2, StopReasonV2


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION ENCODERS
# ═══════════════════════════════════════════════════════════════════════════════

def encode_movz(rd, imm16, hw=0):
    """MOVZ Xd, #imm16, LSL #(hw*16)"""
    return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

def encode_movk(rd, imm16, hw=0):
    """MOVK Xd, #imm16, LSL #(hw*16)"""
    return 0xF2800000 | (hw << 21) | (imm16 << 5) | rd

def encode_add_imm(rd, rn, imm12):
    """ADD Xd, Xn, #imm12"""
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd

def encode_sub_imm(rd, rn, imm12):
    """SUB Xd, Xn, #imm12"""
    return 0xD1000000 | (imm12 << 10) | (rn << 5) | rd

def encode_subs_imm(rd, rn, imm12):
    """SUBS Xd, Xn, #imm12"""
    return 0xF1000000 | (imm12 << 10) | (rn << 5) | rd

def encode_str_64(rt, rn, imm12=0):
    """STR Xt, [Xn, #imm12*8]"""
    return 0xF9000000 | (imm12 << 10) | (rn << 5) | rt

def encode_ldr_64(rt, rn, imm12=0):
    """LDR Xt, [Xn, #imm12*8]"""
    return 0xF9400000 | (imm12 << 10) | (rn << 5) | rt

def encode_strb(rt, rn, imm12=0):
    """STRB Wt, [Xn, #imm12]"""
    return 0x39000000 | (imm12 << 10) | (rn << 5) | rt

def encode_ldrb(rt, rn, imm12=0):
    """LDRB Wt, [Xn, #imm12]"""
    return 0x39400000 | (imm12 << 10) | (rn << 5) | rt

def encode_str_32(rt, rn, imm12=0):
    """STR Wt, [Xn, #imm12*4]"""
    return 0xB9000000 | (imm12 << 10) | (rn << 5) | rt

def encode_ldr_32(rt, rn, imm12=0):
    """LDR Wt, [Xn, #imm12*4]"""
    return 0xB9400000 | (imm12 << 10) | (rn << 5) | rt

def encode_b(offset_words):
    """B label (offset in words from current PC)"""
    imm26 = offset_words & 0x3FFFFFF
    return 0x14000000 | imm26

def encode_cbnz(rt, offset_words):
    """CBNZ Xt, label"""
    imm19 = offset_words & 0x7FFFF
    return 0xB5000000 | (imm19 << 5) | rt

HLT = 0xD4400000
SVC = 0xD4000001
NOP = 0xD503201F


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests():
    passed = 0
    failed = 0

    def test(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  [✓] {name}")
            passed += 1
        else:
            print(f"  [✗] {name}")
            failed += 1

    print("=" * 70)
    print("MLX KERNEL V2 - COMPREHENSIVE TESTS (with Memory Writes)")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY WRITE TESTS (NEW IN V2)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── MEMORY WRITE TESTS ──")

    # Test: STR 64-bit
    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 0xABCD),      # X0 = 0xABCD
        encode_movk(0, 0x1234, 1),   # X0 = 0x1234ABCD
        encode_movz(1, 0x2000),      # X1 = 0x2000
        encode_str_64(0, 1, 0),      # STR X0, [X1]
        HLT
    ], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    mem_val = cpu.read_memory_64(0x2000)
    test("STR 64-bit", mem_val == 0x1234ABCD)

    # Test: STR then LDR 64-bit
    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 0x5678),
        encode_movz(1, 0x3000),
        encode_str_64(0, 1, 0),
        encode_movz(0, 0),           # Clear X0
        encode_ldr_64(2, 1, 0),      # Load into X2
        HLT
    ], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    test("STR then LDR 64-bit", cpu.get_register(2) == 0x5678)

    # Test: STRB (byte store)
    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 0xFF),
        encode_movz(1, 0x4000),
        encode_strb(0, 1, 0),        # STRB X0, [X1]
        encode_strb(0, 1, 1),        # STRB X0, [X1, #1]
        HLT
    ], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    b0 = cpu.read_memory(0x4000, 1)[0]
    b1 = cpu.read_memory(0x4001, 1)[0]
    test("STRB byte store", b0 == 0xFF and b1 == 0xFF)

    # Test: STR 32-bit
    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 0xDEAD),
        encode_movk(0, 0xBEEF, 1),   # X0 = 0xBEEFDEAD
        encode_movz(1, 0x5000),
        encode_str_32(0, 1, 0),      # STR W0, [X1] (32-bit store)
        HLT
    ], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    mem_bytes = cpu.read_memory(0x5000, 4)
    mem_32 = int.from_bytes(mem_bytes, 'little')
    test("STR 32-bit", mem_32 == 0xBEEFDEAD)

    # Test: Multiple stores to different addresses
    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 0x1111),
        encode_movz(1, 0x2222),
        encode_movz(2, 0x3333),
        encode_movz(10, 0x6000),     # Base address
        encode_str_64(0, 10, 0),     # [0x6000] = 0x1111
        encode_str_64(1, 10, 1),     # [0x6008] = 0x2222
        encode_str_64(2, 10, 2),     # [0x6010] = 0x3333
        HLT
    ], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    v0 = cpu.read_memory_64(0x6000)
    v1 = cpu.read_memory_64(0x6008)
    v2 = cpu.read_memory_64(0x6010)
    test("Multiple STR to different addresses", v0 == 0x1111 and v1 == 0x2222 and v2 == 0x3333)

    # Test: Store in loop
    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 0),           # Counter
        encode_movz(1, 10),          # Loop count
        encode_movz(2, 0x7000),      # Base address
        # loop:
        encode_str_64(0, 2, 0),      # Store counter
        encode_add_imm(0, 0, 1),     # counter++
        encode_add_imm(2, 2, 8),     # addr += 8
        encode_subs_imm(1, 1, 1),    # loop_count--
        encode_cbnz(1, -4 & 0x7FFFF),  # if (loop_count != 0) goto loop
        HLT
    ], 0)
    cpu.set_pc(0)
    cpu.execute(1000)
    # Check first and last values
    v_first = cpu.read_memory_64(0x7000)
    v_last = cpu.read_memory_64(0x7000 + 9*8)
    test("STR in loop", v_first == 0 and v_last == 9)

    # ─────────────────────────────────────────────────────────────────────────
    # ALU TESTS (Same as V1)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── ALU TESTS ──")

    # Test: MOVZ
    cpu = MLXKernelCPUv2()
    cpu.load_program([encode_movz(0, 0x1234), HLT], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    test("MOVZ basic", cpu.get_register(0) == 0x1234)

    # Test: ADD immediate
    cpu = MLXKernelCPUv2()
    cpu.load_program([encode_movz(0, 100), encode_add_imm(1, 0, 50), HLT], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    test("ADD immediate", cpu.get_register(1) == 150)

    # Test: SUB immediate
    cpu = MLXKernelCPUv2()
    cpu.load_program([encode_movz(0, 100), encode_sub_imm(1, 0, 30), HLT], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    test("SUB immediate", cpu.get_register(1) == 70)

    # ─────────────────────────────────────────────────────────────────────────
    # BRANCH TESTS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── BRANCH TESTS ──")

    # Test: Unconditional branch
    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 1),
        encode_b(2),                 # Skip next instruction
        encode_movz(0, 99),          # Should be skipped
        HLT
    ], 0)
    cpu.set_pc(0)
    cpu.execute(100)
    test("B unconditional", cpu.get_register(0) == 1)

    # Test: Loop with CBNZ
    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 0),
        encode_movz(1, 5),
        encode_add_imm(0, 0, 1),
        encode_subs_imm(1, 1, 1),
        encode_cbnz(1, -2 & 0x7FFFF),
        HLT
    ], 0)
    cpu.set_pc(0)
    cpu.execute(1000)
    test("Loop with CBNZ", cpu.get_register(0) == 5)

    # ─────────────────────────────────────────────────────────────────────────
    # CONTROL FLOW TESTS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── CONTROL FLOW TESTS ──")

    # Test: HLT stops execution
    cpu = MLXKernelCPUv2()
    cpu.load_program([encode_movz(0, 1), HLT, encode_movz(0, 99)], 0)
    cpu.set_pc(0)
    result = cpu.execute(100)
    test("HLT stops execution", result.stop_reason == StopReasonV2.HALT and cpu.get_register(0) == 1)

    # Test: SVC triggers syscall
    cpu = MLXKernelCPUv2()
    cpu.load_program([encode_movz(0, 42), SVC, encode_movz(0, 99)], 0)
    cpu.set_pc(0)
    result = cpu.execute(100)
    test("SVC triggers syscall", result.stop_reason == StopReasonV2.SYSCALL and cpu.get_register(0) == 42)

    # Test: Max cycles reached
    cpu = MLXKernelCPUv2()
    cpu.load_program([encode_add_imm(0, 0, 1), encode_b(-1 & 0x3FFFFFF)], 0)
    cpu.set_pc(0)
    result = cpu.execute(max_cycles=100)
    test("Max cycles stops execution", result.stop_reason == StopReasonV2.MAX_CYCLES and result.cycles == 100)

    # ─────────────────────────────────────────────────────────────────────────
    # PERFORMANCE TEST
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── PERFORMANCE TEST ──")

    cpu = MLXKernelCPUv2()
    cpu.load_program([
        encode_movz(0, 0),
        encode_movz(1, 0x8000),
        encode_add_imm(0, 0, 1),
        encode_str_64(0, 1, 0),      # Memory write each iteration
        encode_b(-2 & 0x3FFFFFF),
    ], 0)
    cpu.set_pc(0)
    result = cpu.execute(max_cycles=1_000_000)
    test(f"1M cycles with STR: {result.ips:,.0f} IPS", result.ips > 500_000)

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
