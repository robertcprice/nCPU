#!/usr/bin/env python3
"""
Metal Shader Source for ARM64 CPU Emulator Kernel - Version 2.

Phase 2: DOUBLE-BUFFER MEMORY ARCHITECTURE
==========================================

Instead of a write buffer that requires Python to apply writes, we use
double-buffering: the kernel reads from memory_in and writes to memory_out.

Key insight: This keeps EVERYTHING on GPU. Python just swaps pointers
between calls - exactly like we already do for registers.

MEMORY MODEL:
=============

    memory_in (const device)  →  Read initial state
           ↓
    [Copy to memory_out at kernel start]
           ↓
    memory_out (device)       →  Read/Write during execution
           ↓
    [Returned to Python]
           ↓
    memory_in for next call   ←  Swap!

The 4MB copy at kernel start is:
- GPU-to-GPU (~0.4ms at 10GB/s)
- Amortized over millions of cycles
- Negligible per-instruction cost

Author: KVRM Project
Date: 2024
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STOP_RUNNING = 0
STOP_HALT = 1
STOP_SYSCALL = 2
STOP_MAX_CYCLES = 3

# ═══════════════════════════════════════════════════════════════════════════════
# METAL KERNEL HEADER (V2 - Double Buffer)
# ═══════════════════════════════════════════════════════════════════════════════

KERNEL_HEADER_V2 = """
// ════════════════════════════════════════════════════════════════════════════
// ARM64 CPU EMULATOR - METAL KERNEL V2 (Double-Buffer Memory)
// ════════════════════════════════════════════════════════════════════════════
//
// This kernel runs the ENTIRE CPU emulation loop on GPU:
// - Fetch, Decode, Execute, Memory Read/Write
// - Zero Python interaction during execution
// - Only syncs at halt/syscall/max_cycles
//
// Memory Architecture:
// - memory_in: Read-only input (initial state)
// - memory_out: Writable output (copied from input, then modified)
// - After kernel returns, memory_out becomes memory_in for next call
//
// ════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

// Stop reason codes
constant uint8_t STOP_RUNNING = 0;
constant uint8_t STOP_HALT = 1;
constant uint8_t STOP_SYSCALL = 2;
constant uint8_t STOP_MAX_CYCLES = 3;

// Memory size constant (passed as parameter in practice)
constant uint32_t MEMORY_SIZE = 4 * 1024 * 1024;  // 4MB

// ════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

// Fetch 32-bit instruction from memory (little-endian)
inline uint32_t fetch_instruction(device uint8_t* memory, uint64_t pc) {
    return uint32_t(memory[pc]) |
           (uint32_t(memory[pc + 1]) << 8) |
           (uint32_t(memory[pc + 2]) << 16) |
           (uint32_t(memory[pc + 3]) << 24);
}

// Load 64-bit value from memory (little-endian)
inline int64_t load64(device uint8_t* memory, uint64_t addr) {
    return int64_t(memory[addr]) |
           (int64_t(memory[addr + 1]) << 8) |
           (int64_t(memory[addr + 2]) << 16) |
           (int64_t(memory[addr + 3]) << 24) |
           (int64_t(memory[addr + 4]) << 32) |
           (int64_t(memory[addr + 5]) << 40) |
           (int64_t(memory[addr + 6]) << 48) |
           (int64_t(memory[addr + 7]) << 56);
}

// Store 64-bit value to memory (little-endian)
inline void store64(device uint8_t* memory, uint64_t addr, int64_t val) {
    memory[addr] = uint8_t(val & 0xFF);
    memory[addr + 1] = uint8_t((val >> 8) & 0xFF);
    memory[addr + 2] = uint8_t((val >> 16) & 0xFF);
    memory[addr + 3] = uint8_t((val >> 24) & 0xFF);
    memory[addr + 4] = uint8_t((val >> 32) & 0xFF);
    memory[addr + 5] = uint8_t((val >> 40) & 0xFF);
    memory[addr + 6] = uint8_t((val >> 48) & 0xFF);
    memory[addr + 7] = uint8_t((val >> 56) & 0xFF);
}

// Load 32-bit value from memory (little-endian)
inline int32_t load32(device uint8_t* memory, uint64_t addr) {
    return int32_t(memory[addr]) |
           (int32_t(memory[addr + 1]) << 8) |
           (int32_t(memory[addr + 2]) << 16) |
           (int32_t(memory[addr + 3]) << 24);
}

// Store 32-bit value to memory (little-endian)
inline void store32(device uint8_t* memory, uint64_t addr, int32_t val) {
    memory[addr] = uint8_t(val & 0xFF);
    memory[addr + 1] = uint8_t((val >> 8) & 0xFF);
    memory[addr + 2] = uint8_t((val >> 16) & 0xFF);
    memory[addr + 3] = uint8_t((val >> 24) & 0xFF);
}

// Sign extend 26-bit immediate (for B/BL)
inline int32_t sign_extend_26(uint32_t imm26) {
    if (imm26 & 0x2000000) {
        return int32_t(imm26 | 0xFC000000);
    }
    return int32_t(imm26);
}

// Sign extend 19-bit immediate (for CBZ/CBNZ/B.cond)
inline int32_t sign_extend_19(uint32_t imm19) {
    if (imm19 & 0x40000) {
        return int32_t(imm19 | 0xFFF80000);
    }
    return int32_t(imm19);
}

// Sign extend 21-bit immediate (for ADR)
inline int32_t sign_extend_21(uint32_t imm21) {
    if (imm21 & 0x100000) {
        return int32_t(imm21 | 0xFFE00000);
    }
    return int32_t(imm21);
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# METAL KERNEL SOURCE (V2 - Double Buffer)
# ═══════════════════════════════════════════════════════════════════════════════

KERNEL_SOURCE_V2 = """
    // ════════════════════════════════════════════════════════════════════════
    // KERNEL ENTRY - Single-threaded CPU emulation with writable memory
    // ════════════════════════════════════════════════════════════════════════

    uint tid = thread_position_in_grid.x;
    if (tid != 0) return;

    // ════════════════════════════════════════════════════════════════════════
    // MEMORY COPY: memory_in → memory_out
    // ════════════════════════════════════════════════════════════════════════
    // This is a GPU-to-GPU copy (~0.4ms for 4MB at 10GB/s)
    // Amortized over millions of cycles, the per-instruction cost is negligible

    uint32_t mem_size = memory_size_in[0];
    for (uint32_t i = 0; i < mem_size; i++) {
        memory_out[i] = memory_in[i];
    }

    // ════════════════════════════════════════════════════════════════════════
    // LOAD INITIAL STATE
    // ════════════════════════════════════════════════════════════════════════

    uint64_t pc = pc_in[0];
    uint32_t max_cycles = max_cycles_in[0];
    uint32_t cycles = 0;
    uint8_t reason = STOP_RUNNING;

    // Copy registers to thread-local array for faster access
    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers_in[i];
    }
    regs[31] = 0;  // XZR is always zero

    // Copy flags
    float flag_n = flags_in[0];
    float flag_z = flags_in[1];
    float flag_c = flags_in[2];
    float flag_v = flags_in[3];

    // ════════════════════════════════════════════════════════════════════════
    // MAIN EXECUTION LOOP - Runs entirely on GPU!
    // ════════════════════════════════════════════════════════════════════════

    while (cycles < max_cycles) {
        // ════════════════════════════════════════════════════════════════════
        // FETCH - Read from memory_out (our working copy)
        // ════════════════════════════════════════════════════════════════════
        uint32_t inst = fetch_instruction(memory_out, pc);

        // ════════════════════════════════════════════════════════════════════
        // DECODE
        // ════════════════════════════════════════════════════════════════════
        uint8_t op_byte = (inst >> 24) & 0xFF;
        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint8_t rm = (inst >> 16) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint16_t imm16 = (inst >> 5) & 0xFFFF;
        uint8_t hw = (inst >> 21) & 0x3;
        uint32_t imm26 = inst & 0x3FFFFFF;
        uint32_t imm19 = (inst >> 5) & 0x7FFFF;
        uint8_t cond = inst & 0xF;

        // ════════════════════════════════════════════════════════════════════
        // CHECK FOR HALT
        // ════════════════════════════════════════════════════════════════════
        if (inst == 0 || (inst & 0xFFE0001F) == 0xD4400000) {
            reason = STOP_HALT;
            break;
        }

        // ════════════════════════════════════════════════════════════════════
        // CHECK FOR SYSCALL
        // ════════════════════════════════════════════════════════════════════
        if ((inst & 0xFFE0001F) == 0xD4000001) {
            reason = STOP_SYSCALL;
            break;
        }

        // ════════════════════════════════════════════════════════════════════
        // EXECUTE
        // ════════════════════════════════════════════════════════════════════
        bool branch_taken = false;

        // NOP
        if (inst == 0xD503201F) {
            // NOP - do nothing
        }

        // ════════════════════════════════════════════════════════════════════
        // ALU OPERATIONS
        // ════════════════════════════════════════════════════════════════════

        // ADD immediate (64-bit: 0x91, 32-bit: 0x11)
        else if (op_byte == 0x91 || op_byte == 0x11) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val + imm12;
            if (rd != 31) regs[rd] = result;
        }

        // ADDS immediate (64-bit: 0xB1, 32-bit: 0x31)
        else if (op_byte == 0xB1 || op_byte == 0x31) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val + imm12;
            if (rd != 31) regs[rd] = result;
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = 0.0f;
            flag_v = 0.0f;
        }

        // SUB immediate (64-bit: 0xD1, 32-bit: 0x51)
        else if (op_byte == 0xD1 || op_byte == 0x51) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val - imm12;
            if (rd != 31) regs[rd] = result;
        }

        // SUBS immediate (64-bit: 0xF1, 32-bit: 0x71)
        else if (op_byte == 0xF1 || op_byte == 0x71) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val - imm12;
            if (rd != 31) regs[rd] = result;
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = (uint64_t(rn_val) >= uint64_t(imm12)) ? 1.0f : 0.0f;
            flag_v = 0.0f;
        }

        // MOVZ (64-bit: 0xD2, 32-bit: 0x52)
        else if (op_byte == 0xD2 || op_byte == 0x52) {
            int64_t result = int64_t(imm16) << (hw * 16);
            if (rd != 31) regs[rd] = result;
        }

        // MOVK (64-bit: 0xF2, 32-bit: 0x72)
        else if (op_byte == 0xF2 || op_byte == 0x72) {
            int64_t mask = ~(int64_t(0xFFFF) << (hw * 16));
            int64_t rd_val = (rd == 31) ? 0 : regs[rd];
            int64_t result = (rd_val & mask) | (int64_t(imm16) << (hw * 16));
            if (rd != 31) regs[rd] = result;
        }

        // MOVN (64-bit: 0x92, 32-bit: 0x12 with specific encoding)
        else if (op_byte == 0x92 || (op_byte == 0x12 && ((inst >> 29) & 3) == 1)) {
            int64_t result = ~(int64_t(imm16) << (hw * 16));
            if (rd != 31) regs[rd] = result;
        }

        // ADD register (64-bit: 0x8B, 32-bit: 0x0B)
        else if (op_byte == 0x8B || op_byte == 0x0B) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val + rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // SUB register (64-bit: 0xCB, 32-bit: 0x4B)
        else if (op_byte == 0xCB || op_byte == 0x4B) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val - rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // AND register (64-bit: 0x8A, 32-bit: 0x0A)
        else if (op_byte == 0x8A || op_byte == 0x0A) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val & rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // ORR register (64-bit: 0xAA, 32-bit: 0x2A)
        else if (op_byte == 0xAA || op_byte == 0x2A) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val | rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // EOR register (64-bit: 0xCA, 32-bit: 0x4A)
        else if (op_byte == 0xCA || op_byte == 0x4A) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val ^ rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // SUBS register (64-bit: 0xEB, 32-bit: 0x6B)
        else if (op_byte == 0xEB || op_byte == 0x6B) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val - rm_val;
            if (rd != 31) regs[rd] = result;
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = (uint64_t(rn_val) >= uint64_t(rm_val)) ? 1.0f : 0.0f;
            flag_v = 0.0f;
        }

        // ADR (PC-relative address): op=0x10 or 0x30
        else if (op_byte == 0x10 || op_byte == 0x30) {
            uint32_t immlo = (inst >> 29) & 0x3;
            uint32_t immhi = (inst >> 5) & 0x7FFFF;
            uint32_t imm21 = (immhi << 2) | immlo;
            int32_t offset = sign_extend_21(imm21);
            int64_t result = int64_t(pc) + offset;
            if (rd != 31) regs[rd] = result;
        }

        // ════════════════════════════════════════════════════════════════════
        // MEMORY OPERATIONS - Now with full read/write support!
        // ════════════════════════════════════════════════════════════════════

        // LDR 64-bit unsigned offset
        else if ((inst & 0xFFC00000) == 0xF9400000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + (imm12 << 3);
            int64_t val = load64(memory_out, addr);
            if (rd != 31) regs[rd] = val;
        }

        // STR 64-bit unsigned offset - NOW WORKS!
        else if ((inst & 0xFFC00000) == 0xF9000000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + (imm12 << 3);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            store64(memory_out, addr, val);
        }

        // LDRB unsigned offset
        else if ((inst & 0xFFC00000) == 0x39400000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + imm12;
            int64_t val = int64_t(memory_out[addr]);
            if (rd != 31) regs[rd] = val;
        }

        // STRB unsigned offset - NOW WORKS!
        else if ((inst & 0xFFC00000) == 0x39000000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + imm12;
            int64_t val = (rd == 31) ? 0 : regs[rd];
            memory_out[addr] = uint8_t(val & 0xFF);
        }

        // LDR 32-bit unsigned offset
        else if ((inst & 0xFFC00000) == 0xB9400000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + (imm12 << 2);
            int32_t val = load32(memory_out, addr);
            if (rd != 31) regs[rd] = int64_t(val);
        }

        // STR 32-bit unsigned offset - NOW WORKS!
        else if ((inst & 0xFFC00000) == 0xB9000000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + (imm12 << 2);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            store32(memory_out, addr, int32_t(val & 0xFFFFFFFF));
        }

        // ════════════════════════════════════════════════════════════════════
        // BRANCHES
        // ════════════════════════════════════════════════════════════════════

        // B (unconditional branch)
        else if ((inst & 0xFC000000) == 0x14000000) {
            int32_t offset = sign_extend_26(imm26) * 4;
            pc = uint64_t(int64_t(pc) + offset);
            branch_taken = true;
        }

        // BL (branch with link)
        else if ((inst & 0xFC000000) == 0x94000000) {
            regs[30] = int64_t(pc + 4);
            int32_t offset = sign_extend_26(imm26) * 4;
            pc = uint64_t(int64_t(pc) + offset);
            branch_taken = true;
        }

        // BR (branch to register)
        else if ((inst & 0xFFFFFC1F) == 0xD61F0000) {
            int64_t target = (rn == 31) ? 0 : regs[rn];
            pc = uint64_t(target);
            branch_taken = true;
        }

        // BLR (branch with link to register)
        else if ((inst & 0xFFFFFC1F) == 0xD63F0000) {
            regs[30] = int64_t(pc + 4);
            int64_t target = (rn == 31) ? 0 : regs[rn];
            pc = uint64_t(target);
            branch_taken = true;
        }

        // RET (return)
        else if ((inst & 0xFFFFFC1F) == 0xD65F0000) {
            int64_t target = (rn == 31) ? 0 : regs[rn];
            pc = uint64_t(target);
            branch_taken = true;
        }

        // CBZ
        else if ((inst & 0x7F000000) == 0x34000000) {
            uint8_t rt = inst & 0x1F;
            int64_t rt_val = (rt == 31) ? 0 : regs[rt];
            if (rt_val == 0) {
                int32_t offset = sign_extend_19(imm19) * 4;
                pc = uint64_t(int64_t(pc) + offset);
                branch_taken = true;
            }
        }

        // CBNZ
        else if ((inst & 0x7F000000) == 0x35000000) {
            uint8_t rt = inst & 0x1F;
            int64_t rt_val = (rt == 31) ? 0 : regs[rt];
            if (rt_val != 0) {
                int32_t offset = sign_extend_19(imm19) * 4;
                pc = uint64_t(int64_t(pc) + offset);
                branch_taken = true;
            }
        }

        // B.cond (conditional branch)
        else if ((inst & 0xFF000010) == 0x54000000) {
            bool take_branch = false;
            switch (cond) {
                case 0x0: take_branch = (flag_z > 0.5f); break;                    // EQ
                case 0x1: take_branch = (flag_z <= 0.5f); break;                   // NE
                case 0x2: take_branch = (flag_c > 0.5f); break;                    // CS/HS
                case 0x3: take_branch = (flag_c <= 0.5f); break;                   // CC/LO
                case 0x4: take_branch = (flag_n > 0.5f); break;                    // MI
                case 0x5: take_branch = (flag_n <= 0.5f); break;                   // PL
                case 0x6: take_branch = (flag_v > 0.5f); break;                    // VS
                case 0x7: take_branch = (flag_v <= 0.5f); break;                   // VC
                case 0x8: take_branch = (flag_c > 0.5f && flag_z <= 0.5f); break;  // HI
                case 0x9: take_branch = (flag_c <= 0.5f || flag_z > 0.5f); break;  // LS
                case 0xA: take_branch = ((flag_n > 0.5f) == (flag_v > 0.5f)); break; // GE
                case 0xB: take_branch = ((flag_n > 0.5f) != (flag_v > 0.5f)); break; // LT
                case 0xC: take_branch = (flag_z <= 0.5f && (flag_n > 0.5f) == (flag_v > 0.5f)); break; // GT
                case 0xD: take_branch = (flag_z > 0.5f || (flag_n > 0.5f) != (flag_v > 0.5f)); break;  // LE
                case 0xE: take_branch = true; break;  // AL
                case 0xF: take_branch = true; break;  // NV (always)
            }
            if (take_branch) {
                int32_t offset = sign_extend_19(imm19) * 4;
                pc = uint64_t(int64_t(pc) + offset);
                branch_taken = true;
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // UPDATE
        // ════════════════════════════════════════════════════════════════════

        if (!branch_taken) {
            pc += 4;
        }
        regs[31] = 0;
        cycles++;
    }

    // ════════════════════════════════════════════════════════════════════════
    // WRITE OUTPUTS
    // ════════════════════════════════════════════════════════════════════════

    for (int i = 0; i < 32; i++) {
        registers_out[i] = regs[i];
    }
    pc_out[0] = pc;
    flags_out[0] = flag_n;
    flags_out[1] = flag_z;
    flags_out[2] = flag_c;
    flags_out[3] = flag_v;
    cycles_out[0] = cycles;

    if (reason == STOP_RUNNING && cycles >= max_cycles) {
        reason = STOP_MAX_CYCLES;
    }
    stop_reason_out[0] = reason;
"""


def get_kernel_source_v2() -> tuple[str, str]:
    """Get the V2 Metal kernel header and source code."""
    return KERNEL_HEADER_V2, KERNEL_SOURCE_V2


if __name__ == "__main__":
    print("=" * 70)
    print("ARM64 CPU EMULATOR - METAL KERNEL V2 SOURCE (Double-Buffer)")
    print("=" * 70)

    header, source = get_kernel_source_v2()
    print(f"Header length: {len(header):,} characters")
    print(f"Source length: {len(source):,} characters")
    print(f"Total: {len(header) + len(source):,} characters")
    print()
    print("Features:")
    print("  - Full memory read/write support (STR, STRB, STR32)")
    print("  - Double-buffer architecture (memory_in → memory_out)")
    print("  - Zero Python interaction during execution")
    print("  - Pure GPU execution loop")
