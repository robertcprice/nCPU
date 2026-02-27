#!/usr/bin/env python3
"""
Metal Shader Source for ARM64 CPU Emulator Kernel.

This module contains the Metal Shading Language (MSL) source code for the
CPU emulator kernel. The kernel runs entirely on GPU, eliminating Python
loop overhead and GPU-CPU synchronization bottlenecks.

ARCHITECTURE:
=============

The kernel executes instructions in a tight loop on the GPU:
1. FETCH:  Read 4 bytes from memory at PC (little-endian)
2. DECODE: Extract opcode, register indices, immediates via bit operations
3. CHECK:  Detect halt/syscall conditions (break if found)
4. EXECUTE: Dispatch based on opcode, compute result
5. WRITEBACK: Update registers, flags, PC
6. REPEAT: Until max_cycles or halt/syscall

MEMORY LAYOUT:
==============

Inputs:
- memory[4MB]: uint8 - Unified memory (code + data)
- registers[32]: int64 - General purpose registers (x0-x30, xzr=x31)
- pc: uint64 - Program counter
- flags[4]: float32 - NZCV condition flags [N, Z, C, V]
- max_cycles: uint32 - Maximum instructions to execute

Outputs:
- registers_out[32]: int64 - Final register state
- pc_out: uint64 - Final program counter
- flags_out[4]: float32 - Final condition flags
- cycles_executed: uint32 - Number of instructions executed
- stop_reason: uint8 - Why execution stopped (0=running, 1=halt, 2=syscall, 3=max_cycles)

SUPPORTED INSTRUCTIONS:
=======================

ALU (Data Processing):
- ADD/ADDS (immediate and register)
- SUB/SUBS (immediate and register)
- MOVZ, MOVK, MOVN (wide immediate moves)
- AND, ORR, EOR (register)
- CMP (alias for SUBS with XZR destination)

Memory (Phase 1 - Read Only):
- LDR (64-bit load with unsigned offset)
- LDRB (byte load)
- NOTE: STR/STRB disabled in Phase 1 (MLX inputs are read-only)
- Memory writes will be added in Phase 2 via write buffer mechanism

Branches:
- B (unconditional)
- BL (branch with link)
- BR, BLR, RET (register branches)
- CBZ, CBNZ (compare and branch)
- B.cond (conditional branch: EQ, NE, GE, LT, GT, LE)

System:
- SVC (syscall - triggers early exit)
- HLT (halt - triggers early exit)
- NOP (no operation)

Address Calculation:
- ADR (PC-relative address)

PERFORMANCE:
============

Target: 10M-100M+ IPS on Apple Silicon
- Zero GPU-CPU sync during execution
- Single-threaded (parallel execution planned for Phase 2)
- Local register copy for faster access

Author: KVRM Project
Date: 2024
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STOP REASON CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STOP_RUNNING = 0      # Still running (shouldn't be returned)
STOP_HALT = 1         # Hit HLT instruction or PC=0 with inst=0
STOP_SYSCALL = 2      # Hit SVC instruction (needs kernel handling)
STOP_MAX_CYCLES = 3   # Reached max_cycles limit

# ═══════════════════════════════════════════════════════════════════════════════
# METAL KERNEL SOURCE CODE
# ═══════════════════════════════════════════════════════════════════════════════

KERNEL_HEADER = """
// ════════════════════════════════════════════════════════════════════════════
// ARM64 CPU EMULATOR - METAL KERNEL
// ════════════════════════════════════════════════════════════════════════════
//
// This kernel emulates an ARM64 CPU entirely on the GPU.
// Key features:
// - Zero CPU-GPU synchronization during execution
// - Full decode/execute loop in Metal
// - Exits only on halt, syscall, or max_cycles
//
// ════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

// Stop reason codes
constant uint8_t STOP_RUNNING = 0;
constant uint8_t STOP_HALT = 1;
constant uint8_t STOP_SYSCALL = 2;
constant uint8_t STOP_MAX_CYCLES = 3;

// ════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

// Fetch 32-bit instruction from memory (little-endian)
inline uint32_t fetch_instruction(device const uint8_t* memory, uint64_t pc) {
    return uint32_t(memory[pc]) |
           (uint32_t(memory[pc + 1]) << 8) |
           (uint32_t(memory[pc + 2]) << 16) |
           (uint32_t(memory[pc + 3]) << 24);
}

// Load 64-bit value from memory (little-endian)
inline int64_t load64(device const uint8_t* memory, uint64_t addr) {
    return int64_t(memory[addr]) |
           (int64_t(memory[addr + 1]) << 8) |
           (int64_t(memory[addr + 2]) << 16) |
           (int64_t(memory[addr + 3]) << 24) |
           (int64_t(memory[addr + 4]) << 32) |
           (int64_t(memory[addr + 5]) << 40) |
           (int64_t(memory[addr + 6]) << 48) |
           (int64_t(memory[addr + 7]) << 56);
}

// NOTE: store64 removed in Phase 1 - MLX inputs are read-only
// Memory writes will be added in Phase 2 via write buffer mechanism

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

KERNEL_SOURCE = """
    // ════════════════════════════════════════════════════════════════════════
    // KERNEL ENTRY - Single-threaded CPU emulation
    // ════════════════════════════════════════════════════════════════════════

    // Only thread 0 executes
    uint tid = thread_position_in_grid.x;
    if (tid != 0) return;

    // Load initial state
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
    // MAIN EXECUTION LOOP
    // ════════════════════════════════════════════════════════════════════════

    while (cycles < max_cycles) {
        // ════════════════════════════════════════════════════════════════════
        // FETCH
        // ════════════════════════════════════════════════════════════════════
        uint32_t inst = fetch_instruction(memory, pc);

        // ════════════════════════════════════════════════════════════════════
        // DECODE - Extract all possible fields
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
        bool sf = (inst >> 31) & 1;  // 64-bit flag

        // ════════════════════════════════════════════════════════════════════
        // CHECK FOR HALT
        // ════════════════════════════════════════════════════════════════════
        // HLT instruction: 0xD4400000 (with imm16)
        // Zero instruction also halts
        if (inst == 0 || (inst & 0xFFE0001F) == 0xD4400000) {
            reason = STOP_HALT;
            break;
        }

        // ════════════════════════════════════════════════════════════════════
        // CHECK FOR SYSCALL
        // ════════════════════════════════════════════════════════════════════
        // SVC instruction: 0xD4000001 (with imm16)
        if ((inst & 0xFFE0001F) == 0xD4000001) {
            reason = STOP_SYSCALL;
            break;
        }

        // ════════════════════════════════════════════════════════════════════
        // EXECUTE - Dispatch based on instruction type
        // ════════════════════════════════════════════════════════════════════
        bool branch_taken = false;

        // ────────────────────────────────────────────────────────────────────
        // NOP: 0xD503201F
        // ────────────────────────────────────────────────────────────────────
        if (inst == 0xD503201F) {
            // NOP - do nothing
        }

        // ────────────────────────────────────────────────────────────────────
        // ADD immediate (64-bit: 0x91, 32-bit: 0x11)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0x91 || op_byte == 0x11) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val + imm12;
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // ADDS immediate (64-bit: 0xB1, 32-bit: 0x31)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xB1 || op_byte == 0x31) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val + imm12;
            if (rd != 31) regs[rd] = result;
            // Update flags
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = 0.0f;  // Simplified
            flag_v = 0.0f;  // Simplified
        }

        // ────────────────────────────────────────────────────────────────────
        // SUB immediate (64-bit: 0xD1, 32-bit: 0x51)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xD1 || op_byte == 0x51) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val - imm12;
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // SUBS immediate (64-bit: 0xF1, 32-bit: 0x71)
        // Also handles CMP (SUBS with rd=XZR)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xF1 || op_byte == 0x71) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val - imm12;
            if (rd != 31) regs[rd] = result;
            // Update flags
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = (uint64_t(rn_val) >= uint64_t(imm12)) ? 1.0f : 0.0f;
            flag_v = 0.0f;  // Simplified
        }

        // ────────────────────────────────────────────────────────────────────
        // MOVZ (64-bit: 0xD2, 32-bit: 0x52)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xD2 || op_byte == 0x52) {
            int64_t result = int64_t(imm16) << (hw * 16);
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // MOVK (64-bit: 0xF2, 32-bit: 0x72)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xF2 || op_byte == 0x72) {
            int64_t mask = ~(int64_t(0xFFFF) << (hw * 16));
            int64_t rd_val = (rd == 31) ? 0 : regs[rd];
            int64_t result = (rd_val & mask) | (int64_t(imm16) << (hw * 16));
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // MOVN (64-bit: 0x92, 32-bit: 0x12)
        // Note: 0x12 can also be AND immediate, differentiated by other bits
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0x92 || (op_byte == 0x12 && ((inst >> 29) & 3) == 1)) {
            int64_t result = ~(int64_t(imm16) << (hw * 16));
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // ADD register (64-bit: 0x8B, 32-bit: 0x0B)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0x8B || op_byte == 0x0B) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val + rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // SUB register (64-bit: 0xCB, 32-bit: 0x4B)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xCB || op_byte == 0x4B) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val - rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // AND register (64-bit: 0x8A, 32-bit: 0x0A)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0x8A || op_byte == 0x0A) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val & rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // ORR register (64-bit: 0xAA, 32-bit: 0x2A)
        // Also handles MOV register (ORR Rd, XZR, Rm)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xAA || op_byte == 0x2A) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val | rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // EOR register (64-bit: 0xCA, 32-bit: 0x4A)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xCA || op_byte == 0x4A) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val ^ rm_val;
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // SUBS register (64-bit: 0xEB, 32-bit: 0x6B)
        // Also handles CMP register (SUBS with rd=XZR)
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0xEB || op_byte == 0x6B) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t result = rn_val - rm_val;
            if (rd != 31) regs[rd] = result;
            // Update flags
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = (uint64_t(rn_val) >= uint64_t(rm_val)) ? 1.0f : 0.0f;
            flag_v = 0.0f;  // Simplified
        }

        // ────────────────────────────────────────────────────────────────────
        // ADR (PC-relative address): op=0x10 or 0x30
        // ADR Rd, label  - calculates PC + offset
        // ────────────────────────────────────────────────────────────────────
        else if (op_byte == 0x10 || op_byte == 0x30) {
            // immlo = bits[30:29], immhi = bits[23:5]
            uint32_t immlo = (inst >> 29) & 0x3;
            uint32_t immhi = (inst >> 5) & 0x7FFFF;
            uint32_t imm21 = (immhi << 2) | immlo;
            int32_t offset = sign_extend_21(imm21);
            int64_t result = int64_t(pc) + offset;
            if (rd != 31) regs[rd] = result;
        }

        // ────────────────────────────────────────────────────────────────────
        // LDR 64-bit unsigned offset: 0xF9 with bit 22 = 1
        // STR 64-bit unsigned offset: 0xF9 with bit 22 = 0
        // Actually: LDR is 0xF94, STR is 0xF90 (different bit patterns)
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFFC00000) == 0xF9400000) {
            // LDR Xt, [Xn, #imm12*8]
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + (imm12 << 3);
            int64_t val = load64(memory, addr);
            if (rd != 31) regs[rd] = val;
        }
        // NOTE: STR 64-bit disabled in Phase 1 (MLX inputs are read-only)
        // else if ((inst & 0xFFC00000) == 0xF9000000) { STR Xt, [Xn, #imm12*8] }

        // ────────────────────────────────────────────────────────────────────
        // LDRB unsigned offset: 0x39400000
        // STRB unsigned offset: 0x39000000
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFFC00000) == 0x39400000) {
            // LDRB Wt, [Xn, #imm12]
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + imm12;
            int64_t val = int64_t(memory[addr]);
            if (rd != 31) regs[rd] = val;
        }
        // NOTE: STRB disabled in Phase 1 (MLX inputs are read-only)
        // else if ((inst & 0xFFC00000) == 0x39000000) { STRB Wt, [Xn, #imm12] }

        // ────────────────────────────────────────────────────────────────────
        // LDR 32-bit unsigned offset: 0xB9400000
        // STR 32-bit unsigned offset: 0xB9000000
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFFC00000) == 0xB9400000) {
            // LDR Wt, [Xn, #imm12*4]
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = uint64_t(base) + (imm12 << 2);
            int32_t val = int32_t(memory[addr]) |
                         (int32_t(memory[addr + 1]) << 8) |
                         (int32_t(memory[addr + 2]) << 16) |
                         (int32_t(memory[addr + 3]) << 24);
            if (rd != 31) regs[rd] = int64_t(val);  // Zero-extend
        }
        // NOTE: STR 32-bit disabled in Phase 1 (MLX inputs are read-only)
        // else if ((inst & 0xFFC00000) == 0xB9000000) { STR Wt, [Xn, #imm12*4] }

        // ────────────────────────────────────────────────────────────────────
        // B (unconditional branch): 0x14xxxxxx
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFC000000) == 0x14000000) {
            int32_t offset = sign_extend_26(imm26) * 4;
            pc = uint64_t(int64_t(pc) + offset);
            branch_taken = true;
        }

        // ────────────────────────────────────────────────────────────────────
        // BL (branch with link): 0x94xxxxxx
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFC000000) == 0x94000000) {
            regs[30] = int64_t(pc + 4);  // Link register
            int32_t offset = sign_extend_26(imm26) * 4;
            pc = uint64_t(int64_t(pc) + offset);
            branch_taken = true;
        }

        // ────────────────────────────────────────────────────────────────────
        // BR (branch to register): 0xD61F0xxx
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFFFFFC1F) == 0xD61F0000) {
            int64_t target = (rn == 31) ? 0 : regs[rn];
            pc = uint64_t(target);
            branch_taken = true;
        }

        // ────────────────────────────────────────────────────────────────────
        // BLR (branch with link to register): 0xD63F0xxx
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFFFFFC1F) == 0xD63F0000) {
            regs[30] = int64_t(pc + 4);
            int64_t target = (rn == 31) ? 0 : regs[rn];
            pc = uint64_t(target);
            branch_taken = true;
        }

        // ────────────────────────────────────────────────────────────────────
        // RET (return): 0xD65F0xxx
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFFFFFC1F) == 0xD65F0000) {
            int64_t target = (rn == 31) ? 0 : regs[rn];
            pc = uint64_t(target);
            branch_taken = true;
        }

        // ────────────────────────────────────────────────────────────────────
        // CBZ (compare and branch if zero): 0x34xxxxxx (64-bit), 0xB4xxxxxx
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0x7F000000) == 0x34000000) {
            uint8_t rt = inst & 0x1F;
            int64_t rt_val = (rt == 31) ? 0 : regs[rt];
            if (rt_val == 0) {
                int32_t offset = sign_extend_19(imm19) * 4;
                pc = uint64_t(int64_t(pc) + offset);
                branch_taken = true;
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // CBNZ (compare and branch if not zero): 0x35xxxxxx (64-bit), 0xB5xxxxxx
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0x7F000000) == 0x35000000) {
            uint8_t rt = inst & 0x1F;
            int64_t rt_val = (rt == 31) ? 0 : regs[rt];
            if (rt_val != 0) {
                int32_t offset = sign_extend_19(imm19) * 4;
                pc = uint64_t(int64_t(pc) + offset);
                branch_taken = true;
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // B.cond (conditional branch): 0x54xxxxxx
        // ────────────────────────────────────────────────────────────────────
        else if ((inst & 0xFF000010) == 0x54000000) {
            bool take_branch = false;

            // Evaluate condition based on cond code
            switch (cond) {
                case 0x0: take_branch = (flag_z > 0.5f); break;           // EQ: Z=1
                case 0x1: take_branch = (flag_z <= 0.5f); break;          // NE: Z=0
                case 0x2: take_branch = (flag_c > 0.5f); break;           // CS/HS: C=1
                case 0x3: take_branch = (flag_c <= 0.5f); break;          // CC/LO: C=0
                case 0x4: take_branch = (flag_n > 0.5f); break;           // MI: N=1
                case 0x5: take_branch = (flag_n <= 0.5f); break;          // PL: N=0
                case 0x6: take_branch = (flag_v > 0.5f); break;           // VS: V=1
                case 0x7: take_branch = (flag_v <= 0.5f); break;          // VC: V=0
                case 0x8: take_branch = (flag_c > 0.5f && flag_z <= 0.5f); break;  // HI: C=1 && Z=0
                case 0x9: take_branch = (flag_c <= 0.5f || flag_z > 0.5f); break;  // LS: C=0 || Z=1
                case 0xA: take_branch = ((flag_n > 0.5f) == (flag_v > 0.5f)); break;  // GE: N=V
                case 0xB: take_branch = ((flag_n > 0.5f) != (flag_v > 0.5f)); break;  // LT: N!=V
                case 0xC: take_branch = (flag_z <= 0.5f && (flag_n > 0.5f) == (flag_v > 0.5f)); break;  // GT: Z=0 && N=V
                case 0xD: take_branch = (flag_z > 0.5f || (flag_n > 0.5f) != (flag_v > 0.5f)); break;   // LE: Z=1 || N!=V
                case 0xE: take_branch = true; break;                       // AL: always
                case 0xF: take_branch = true; break;                       // NV: always (used for hints)
            }

            if (take_branch) {
                int32_t offset = sign_extend_19(imm19) * 4;
                pc = uint64_t(int64_t(pc) + offset);
                branch_taken = true;
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // Unknown instruction - skip
        // ────────────────────────────────────────────────────────────────────
        // else { /* Unknown - just increment PC */ }

        // ════════════════════════════════════════════════════════════════════
        // UPDATE PC (if not already updated by branch)
        // ════════════════════════════════════════════════════════════════════
        if (!branch_taken) {
            pc += 4;
        }

        // XZR must always be zero
        regs[31] = 0;

        cycles++;
    }

    // ════════════════════════════════════════════════════════════════════════
    // WRITE OUTPUTS
    // ════════════════════════════════════════════════════════════════════════

    // Copy registers back
    for (int i = 0; i < 32; i++) {
        registers_out[i] = regs[i];
    }

    // Write final state
    pc_out[0] = pc;
    flags_out[0] = flag_n;
    flags_out[1] = flag_z;
    flags_out[2] = flag_c;
    flags_out[3] = flag_v;
    cycles_out[0] = cycles;

    // Set stop reason
    if (reason == STOP_RUNNING && cycles >= max_cycles) {
        reason = STOP_MAX_CYCLES;
    }
    stop_reason_out[0] = reason;
"""

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

def get_kernel_source() -> tuple[str, str]:
    """
    Get the Metal kernel header and source code.

    Returns:
        Tuple of (header, source) strings for mx.fast.metal_kernel()
    """
    return KERNEL_HEADER, KERNEL_SOURCE


def get_full_kernel_source() -> str:
    """
    Get the complete Metal kernel source code (header + source combined).

    Useful for debugging or manual inspection.
    """
    return KERNEL_HEADER + "\n// ═══ KERNEL SOURCE ═══\n" + KERNEL_SOURCE


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ARM64 CPU EMULATOR - METAL KERNEL SOURCE")
    print("=" * 70)

    header, source = get_kernel_source()

    print(f"\nHeader length: {len(header):,} characters")
    print(f"Source length: {len(source):,} characters")
    print(f"Total:         {len(header) + len(source):,} characters")

    print("\n" + "=" * 70)
    print("SUPPORTED INSTRUCTIONS:")
    print("=" * 70)

    instructions = [
        ("ADD/ADDS imm", "0x91, 0x11, 0xB1, 0x31"),
        ("SUB/SUBS imm", "0xD1, 0x51, 0xF1, 0x71"),
        ("MOVZ", "0xD2, 0x52"),
        ("MOVK", "0xF2, 0x72"),
        ("MOVN", "0x92, 0x12"),
        ("ADD/SUB reg", "0x8B, 0x0B, 0xCB, 0x4B"),
        ("AND/ORR/EOR reg", "0x8A, 0x0A, 0xAA, 0x2A, 0xCA, 0x4A"),
        ("SUBS reg (CMP)", "0xEB, 0x6B"),
        ("ADR", "0x10, 0x30"),
        ("LDR 64-bit", "0xF9400000"),
        ("STR 64-bit", "0xF9000000"),
        ("LDRB", "0x39400000"),
        ("STRB", "0x39000000"),
        ("LDR 32-bit", "0xB9400000"),
        ("STR 32-bit", "0xB9000000"),
        ("B", "0x14xxxxxx"),
        ("BL", "0x94xxxxxx"),
        ("BR", "0xD61F0xxx"),
        ("BLR", "0xD63F0xxx"),
        ("RET", "0xD65F0xxx"),
        ("CBZ", "0x34xxxxxx"),
        ("CBNZ", "0x35xxxxxx"),
        ("B.cond", "0x54xxxxxx"),
        ("SVC (syscall)", "0xD4000001"),
        ("HLT (halt)", "0xD4400000"),
        ("NOP", "0xD503201F"),
    ]

    for name, encoding in instructions:
        print(f"  {name:20} {encoding}")
