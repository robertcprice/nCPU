#!/usr/bin/env python3
"""
CREATE SIMPLE WORKING DOOM BENCHMARK
======================================
Create a working DOOM-like benchmark directly in memory with proper ARM64 encoding.
"""

import sys
sys.path.insert(0, '.')
from run_neural_rtos_v2 import FullyNeuralCPU

def create_simple_doom_benchmark():
    """Create a simple DOOM-like benchmark in memory."""

    cpu = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)

    # ARM64 instruction encoding helpers
    def b_imm(offset: int) -> int:
        """Encode unconditional branch with immediate offset."""
        # B: 0001 | 01ii iiii iiii iiii iiii iiii iiii iiii
        imm26 = (offset >> 2) & 0x3FFFFFF
        return 0x14000000 | imm26

    def b_cond_imm(cond: int, offset: int) -> int:
        """Encode conditional branch with immediate offset."""
        # B.cond: 0101 | 0ii iiii iiii iiii iiii iiii iiii ffff
        # cond: 0=EQ, 1=NE, 2=CS, 3=CC, 4=MI, 5=PL, 6=VS, 7=VC, 8=HI, 9=LS, 10=GE, 11=LT, 12=GT, 13=LE, 14=AL
        imm19 = (offset >> 2) & 0x7FFFF
        return 0x54000000 | (cond << 5) | imm19

    def mov_reg(rd: int, rn: int) -> int:
        """Encode MOV register (ORR with dummy)."""
        # MOV x0, x1 -> ORR x0, xZR, x1
        return orr_imm(rd, 31, rn, 0)

    def orr_imm(rd: int, rn: int, imm: int, shift: int = 0) -> int:
        """Encode ORR immediate."""
        # Simplified - just for small immediates
        n = 1 if shift == 0 else 0
        imm_r = imm & 0x3F
        imm_s = imm & 0x3F
        return 0x32000000 | (n << 22) | (imm_r << 16) | (imm_s << 10) | (rn << 5) | rd

    def subs_imm(rd: int, rn: int, imm: int) -> int:
        """Encode SUBS immediate."""
        # SUBS xd, xn, #imm
        return 0x71000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd

    def str_imm(rt: int, rn: int, imm: int) -> int:
        """Encode STR immediate."""
        # STR wt, [xn, #imm]
        imm9 = imm & 0x1FF
        size = 2  # 32-bit
        return 0x39000000 | (size << 30) | (imm9 << 12) | (rn << 5) | rt

    def str_post(rt: int, rn: int, imm: int) -> int:
        """Encode STR post-index."""
        # STR wt, [xn], #imm
        imm9 = imm & 0x1FF
        size = 2
        return 0x38000000 | (size << 30) | (imm9 << 12) | (rn << 5) | rt

    # Write program to memory starting at 0x400000
    pc = 0x400000

    # Setup
    cpu.memory.write32(pc, 0xd28003e0)  # mov x0, #0x1ff (approx sp)
    pc += 4
    cpu.memory.write32(pc, 0x910003fd)  # mov sp, x0
    pc += 4

    # Frame counter: x19 = 10
    cpu.memory.write32(pc, 0xd2800513)  # mov x19, #10 (0x14 << 16 | 10)
    pc += 4

    frame_loop_start = pc

    # MEMSET loop: Clear framebuffer (16000 words)
    # x0 = framebuffer address (0x20000)
    # x1 = color (0)
    # x2 = counter (16000)

    cpu.memory.write32(pc, 0xd28042a0)  # mov x0, #0x21000 (approx 0x20000)
    pc += 4
    cpu.memory.write32(pc, 0xd2800021)  # mov x1, #0
    pc += 4
    cpu.memory.write32(pc, 0xd2819442)  # mov x2, #0x0fa0 (16000 in hex >> 16)
    pc += 4
    cpu.memory.write32(pc, 0xf2a00042)  # movk x2, #0x0fa0, lsl #16
    pc += 4

    memset_loop_start = pc

    # str w1, [x0], #4
    cpu.memory.write32(pc, 0x39000401)  # str w1, [x0], #4
    pc += 4

    # subs x2, x2, #1
    cpu.memory.write32(pc, 0xf1000842)  # subs x2, x2, #1
    pc += 4

    # b.ne memset_loop_start
    offset = (memset_loop_start - (pc - 4)) & 0x7FFFF
    cpu.memory.write32(pc, 0x54000001 | (offset << 5))  # b.ne
    pc += 4

    # DONE: Exit
    # mov x0, #0
    cpu.memory.write32(pc, 0xd2800000)  # mov x0, #0
    pc += 4

    # mov x8, #93 (exit syscall)
    cpu.memory.write32(pc, 0xd28005d6)  # mov x8, #93
    pc += 4

    # svc #0
    cpu.memory.write32(pc, 0xd4000001)  # svc #0
    pc += 4

    # Infinite hang
    cpu.memory.write32(pc, 0x17ffffff)  # b #(-4) infinite loop
    pc += 4

    # Set PC to start
    cpu.pc = 0x400000

    return cpu, {
        'frame_loop': frame_loop_start,
        'memset_loop': memset_loop_start,
        'entry': 0x400000
    }


if __name__ == "__main__":
    print("="*70)
    print("CREATING SIMPLE DOOM BENCHMARK")
    print("="*70)
    print()

    cpu, addrs = create_simple_doom_benchmark()

    print(f"Created DOOM benchmark at:")
    print(f"  Entry: 0x{addrs['entry']:x}")
    print(f"  Frame loop: 0x{addrs['frame_loop']:x}")
    print(f"  Memset loop: 0x{addrs['memset_loop']:x}")
    print()

    # Disassemble to verify
    print("Instructions:")
    for addr in range(0x400000, 0x400080, 4):
        inst = cpu.memory.read32(addr)
        print(f"  0x{addr:x}: 0x{inst:08x}")
    print()

    # Save to file
    import pickle
    with open('simple_doom_cpu.pkl', 'wb') as f:
        pickle.dump(cpu, f)
    print("Saved CPU state to simple_doom_cpu.pkl")
    print()
