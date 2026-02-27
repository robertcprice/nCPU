//! Optimized GPU execution with memory access patterns and register caching.
//!
//! This module provides optimizations beyond continuous execution:
//! - Threadgroup memory for registers (faster than device memory)
//! - Optimized instruction decode paths
//! - Reduced branching in hot loops

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::time::Instant;

use crate::{MetalError, get_default_device};

/// Optimized Metal shader with threadgroup memory for registers
const OPTIMIZED_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Stop/signal reasons
constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

// Optimized memory access - aligned loads
inline uint64_t load64_aligned(device uint8_t* mem, uint64_t addr) {
    // Use aligned access when possible
    if ((addr & 7) == 0) {
        return *((device uint64_t*)(mem + addr));
    }
    // Fallback to byte-by-byte for unaligned
    return uint64_t(mem[addr]) |
           (uint64_t(mem[addr + 1]) << 8) |
           (uint64_t(mem[addr + 2]) << 16) |
           (uint64_t(mem[addr + 3]) << 24) |
           (uint64_t(mem[addr + 4]) << 32) |
           (uint64_t(mem[addr + 5]) << 40) |
           (uint64_t(mem[addr + 6]) << 48) |
           (uint64_t(mem[addr + 7]) << 56);
}

inline void store64_aligned(device uint8_t* mem, uint64_t addr, uint64_t val) {
    if ((addr & 7) == 0) {
        *((device uint64_t*)(mem + addr)) = val;
        return;
    }
    mem[addr] = val & 0xFF;
    mem[addr + 1] = (val >> 8) & 0xFF;
    mem[addr + 2] = (val >> 16) & 0xFF;
    mem[addr + 3] = (val >> 24) & 0xFF;
    mem[addr + 4] = (val >> 32) & 0xFF;
    mem[addr + 5] = (val >> 40) & 0xFF;
    mem[addr + 6] = (val >> 48) & 0xFF;
    mem[addr + 7] = (val >> 56) & 0xFF;
}

inline uint32_t load32_aligned(device uint8_t* mem, uint64_t addr) {
    if ((addr & 3) == 0) {
        return *((device uint32_t*)(mem + addr));
    }
    return uint32_t(mem[addr]) |
           (uint32_t(mem[addr + 1]) << 8) |
           (uint32_t(mem[addr + 2]) << 16) |
           (uint32_t(mem[addr + 3]) << 24);
}

// Helper to decode logical immediate bitmask (simplified)
inline uint32_t decode_immediate_bitmask(uint32_t inst) {
    // N:immr:imms - simplified version for common cases
    uint8_t immr = (inst >> 16) & 0x3F;
    uint8_t imms = (inst >> 10) & 0x3F;
    uint8_t n = (inst >> 22) & 1;

    // For element size = 1 (imms = 0x00 or 0x3F with n=1)
    if (imms == 0x3F && n == 1) {
        return 0xFFFFFFFF;
    }
    // For element size = 2 (imms = 0x01 or 0x3E with n=1)
    else if (imms == 0x3E && n == 1) {
        return 0x55555555 | (0x55555555 << immr);
    }
    // For element size = 4 (imms = 0x03 or 0x3D with n=1)
    else if (imms == 0x3D && n == 1) {
        return 0x11111111 | (0x11111111 << immr);
    }
    // For element size = 8 (imms = 0x07 or 0x3C with n=1)
    else if (imms == 0x3C && n == 1) {
        return 0x01010101 | (0x01010101 << immr);
    }
    // For element size = 16 (imms = 0x0F or 0x3B with n=1)
    else if (imms == 0x3B && n == 1) {
        return 0x00010001 | (0x00010001 << immr);
    }
    // For element size = 32 (imms = 0x1F or 0x3A with n=1)
    else if (imms == 0x3A && n == 1) {
        return 0x00000001 | (0x00000001 << immr);
    }

    // Fallback: try to handle as repeating element
    // This is a simplified implementation
    uint32_t result = 0xFFFFFFFF;
    if (imms < 32) {
        result = (1u << (imms + 1)) - 1;
        if (immr > 0) {
            result = (result >> immr) | (result << (32 - immr));
        }
    }
    return result;
}

// Optimized CPU execution kernel
// Uses direct instruction dispatch table pattern
kernel void cpu_execute_optimized(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* max_cycles_ptr [[buffer(4)]],
    device const uint32_t* mem_size [[buffer(5)]],
    device atomic_uint* signal_flag [[buffer(6)]],
    device uint32_t* total_cycles [[buffer(7)]],
    device uint32_t* batch_count [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t max_cycles = max_cycles_ptr[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;

    // Load registers into thread-local storage (register file)
    // r31 is SP (stack pointer) - kept for load/store base addressing
    int64_t r0 = registers[0], r1 = registers[1], r2 = registers[2], r3 = registers[3];
    int64_t r4 = registers[4], r5 = registers[5], r6 = registers[6], r7 = registers[7];
    int64_t r8 = registers[8], r9 = registers[9], r10 = registers[10], r11 = registers[11];
    int64_t r12 = registers[12], r13 = registers[13], r14 = registers[14], r15 = registers[15];
    int64_t r16 = registers[16], r17 = registers[17], r18 = registers[18], r19 = registers[19];
    int64_t r20 = registers[20], r21 = registers[21], r22 = registers[22], r23 = registers[23];
    int64_t r24 = registers[24], r25 = registers[25], r26 = registers[26], r27 = registers[27];
    int64_t r28 = registers[28], r29 = registers[29], r30 = registers[30], r31 = registers[31];

    float N = flags[0], Z = flags[1], C = flags[2], V = flags[3];

    // Macro to read register - XZR behavior handled per-instruction
    #define READ_REG(n) ((n) == 0 ? r0 : (n) == 1 ? r1 : (n) == 2 ? r2 : (n) == 3 ? r3 : \
                         (n) == 4 ? r4 : (n) == 5 ? r5 : (n) == 6 ? r6 : (n) == 7 ? r7 : \
                         (n) == 8 ? r8 : (n) == 9 ? r9 : (n) == 10 ? r10 : (n) == 11 ? r11 : \
                         (n) == 12 ? r12 : (n) == 13 ? r13 : (n) == 14 ? r14 : (n) == 15 ? r15 : \
                         (n) == 16 ? r16 : (n) == 17 ? r17 : (n) == 18 ? r18 : (n) == 19 ? r19 : \
                         (n) == 20 ? r20 : (n) == 21 ? r21 : (n) == 22 ? r22 : (n) == 23 ? r23 : \
                         (n) == 24 ? r24 : (n) == 25 ? r25 : (n) == 26 ? r26 : (n) == 27 ? r27 : \
                         (n) == 28 ? r28 : (n) == 29 ? r29 : (n) == 30 ? r30 : r31)

    // Macro to write register
    #define WRITE_REG(n, val) do { \
        switch(n) { \
            case 0: r0 = val; break; case 1: r1 = val; break; \
            case 2: r2 = val; break; case 3: r3 = val; break; \
            case 4: r4 = val; break; case 5: r5 = val; break; \
            case 6: r6 = val; break; case 7: r7 = val; break; \
            case 8: r8 = val; break; case 9: r9 = val; break; \
            case 10: r10 = val; break; case 11: r11 = val; break; \
            case 12: r12 = val; break; case 13: r13 = val; break; \
            case 14: r14 = val; break; case 15: r15 = val; break; \
            case 16: r16 = val; break; case 17: r17 = val; break; \
            case 18: r18 = val; break; case 19: r19 = val; break; \
            case 20: r20 = val; break; case 21: r21 = val; break; \
            case 22: r22 = val; break; case 23: r23 = val; break; \
            case 24: r24 = val; break; case 25: r25 = val; break; \
            case 26: r26 = val; break; case 27: r27 = val; break; \
            case 28: r28 = val; break; case 29: r29 = val; break; \
            case 30: r30 = val; break; case 31: r31 = val; break; \
        } \
    } while(0)

    while (cycles < max_cycles) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        // Fetch instruction (always aligned for ARM64)
        uint32_t inst = *((device uint32_t*)(memory + pc));

        // Quick check for special instructions
        uint32_t masked = inst & 0xFFE0001F;
        if (masked == 0xD4400000) { // HLT
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }
        if (masked == 0xD4000001) { // SVC
            atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
            break;
        }

        // Decode common fields once
        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint8_t rm = (inst >> 16) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint16_t imm16 = (inst >> 5) & 0xFFFF;
        uint8_t hw = (inst >> 21) & 0x3;

        // Get opcode high byte for fast dispatch
        uint8_t op_hi = (inst >> 24) & 0xFF;

        bool branch_taken = false;

        // Dispatch based on high byte (most instructions can be identified this way)
        switch (op_hi) {
            // ═══════════════════════════════════════════════════════════════════
            // DATA PROCESSING - IMMEDIATE
            // ═══════════════════════════════════════════════════════════════════
            case 0x91: { // ADD immediate 64-bit (rn=31 is SP)
                int64_t rn_val = READ_REG(rn);
                if (rd < 31) WRITE_REG(rd, rn_val + imm12);
                else r31 = rn_val + imm12;
                break;
            }
            case 0x11: { // ADD immediate 32-bit
                int64_t rn_val = READ_REG(rn);
                int64_t result = (int64_t)(int32_t)((uint32_t)rn_val + imm12);
                WRITE_REG(rd, result);
                break;
            }
            case 0x31: { // ADDS immediate 32-bit (sets flags)
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int32_t result = (int32_t)((uint32_t)rn_val + imm12);
                if (rd < 31) WRITE_REG(rd, (int64_t)result);
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint32_t)result < (uint32_t)rn_val) ? 1.0 : 0.0;
                V = 0.0;
                break;
            }
            case 0xD1: { // SUB immediate 64-bit (rn=31 is SP)
                int64_t rn_val = READ_REG(rn);
                if (rd < 31) WRITE_REG(rd, rn_val - imm12);
                else r31 = rn_val - imm12;
                break;
            }
            case 0x51: { // SUB immediate 32-bit
                int64_t rn_val = READ_REG(rn);
                WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val - imm12));
                break;
            }
            case 0x71: { // SUBS immediate 32-bit (sets flags)
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int32_t result = (int32_t)((uint32_t)rn_val - imm12);
                if (rd < 31) WRITE_REG(rd, (int64_t)result);
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint32_t)rn_val >= imm12) ? 1.0 : 0.0;
                V = 0.0;
                break;
            }
            case 0xB1: { // ADDS immediate 64-bit
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t result = rn_val + imm12;
                if (rd < 31) WRITE_REG(rd, result);
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint64_t)result < (uint64_t)rn_val) ? 1.0 : 0.0;
                V = 0.0;
                break;
            }
            case 0xF1: { // SUBS immediate 64-bit
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t result = rn_val - imm12;
                if (rd < 31) WRITE_REG(rd, result);
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
                V = 0.0;
                break;
            }
            case 0xD2: { // MOVZ 64-bit or UBFM
                if ((inst & 0xFF800000) == 0xD2800000) { // MOVZ
                    WRITE_REG(rd, (int64_t)((uint64_t)imm16 << (hw * 16)));
                }
                break;
            }
            case 0xD3: { // UBFM 64-bit (LSL, LSR, UBFX)
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                uint8_t imms = (inst >> 10) & 0x3F;
                uint8_t immr = (inst >> 16) & 0x3F;
                if (rd < 31) {
                    int64_t result;
                    if (imms < 63) {
                        int shift = 63 - imms;
                        result = (int64_t)((uint64_t)rn_val << shift);
                    } else if (imms == 63 && immr > 0) {
                        result = (int64_t)((uint64_t)rn_val >> immr);
                    } else {
                        uint64_t mask = (1ULL << (imms + 1)) - 1;
                        result = (int64_t)(((uint64_t)rn_val >> immr) & mask);
                    }
                    WRITE_REG(rd, result);
                }
                break;
            }
            case 0x12: { // MOVN 32-bit
                if ((inst & 0xFF800000) == 0x12800000) {
                    WRITE_REG(rd, (int64_t)(int32_t)(uint32_t)(~(imm16 << (hw * 16))));
                }
                break;
            }
            case 0x52: { // MOVZ 32-bit
                if ((inst & 0xFF800000) == 0x52800000) {
                    WRITE_REG(rd, (int64_t)(uint32_t)(imm16 << (hw * 16)));
                }
                break;
            }
            case 0x92: { // MOVN 64-bit
                if ((inst & 0xFF800000) == 0x92800000) {
                    WRITE_REG(rd, ~((int64_t)((uint64_t)imm16 << (hw * 16))));
                }
                break;
            }
            case 0x93: { // SBFM 64-bit (ASR, SXTB, SXTH, SXTW)
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                uint8_t imms = (inst >> 10) & 0x3F;
                uint8_t immr = (inst >> 16) & 0x3F;
                if (rd < 31) {
                    int64_t result;
                    if (imms == 0x1F && immr == 0) {
                        result = (int64_t)(int32_t)rn_val;
                    } else if (imms == 0x0F && immr == 0) {
                        result = (int64_t)(int16_t)rn_val;
                    } else if (imms == 0x07 && immr == 0) {
                        result = (int64_t)(int8_t)rn_val;
                    } else if (imms == 63) {
                        result = rn_val >> immr;
                    } else {
                        int width = imms + 1;
                        uint64_t mask = (1ULL << width) - 1;
                        int64_t extracted = (int64_t)(((uint64_t)rn_val >> immr) & mask);
                        if (extracted & (1ULL << (width - 1))) {
                            extracted |= ~((1ULL << width) - 1);
                        }
                        result = extracted;
                    }
                    WRITE_REG(rd, result);
                }
                break;
            }
            case 0xF2: { // MOVK 64-bit
                if ((inst & 0xFF800000) == 0xF2800000) {
                    uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
                    int64_t old_val = (rd == 31) ? 0 : READ_REG(rd);
                    uint64_t val = (uint64_t)old_val & mask;
                    val |= ((uint64_t)imm16 << (hw * 16));
                    WRITE_REG(rd, (int64_t)val);
                }
                break;
            }

            // ═══════════════════════════════════════════════════════════════════
            // DATA PROCESSING - REGISTER
            // ═══════════════════════════════════════════════════════════════════
            case 0x0B: { // ADD shifted register 32-bit
                if ((inst & 0xFF200000) == 0x0B000000) {
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    int64_t result = (int32_t)(rn_val + rm_val);
                    if (rd < 31) WRITE_REG(rd, result);
                }
                break;
            }
            case 0x2B: { // ADDS shifted register 32-bit (sets flags)
                if ((inst & 0xFF200000) == 0x2B000000) {
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    int32_t result = (int32_t)((uint32_t)rn_val + (uint32_t)rm_val);
                    if (rd < 31) WRITE_REG(rd, (int64_t)result);
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint32_t)result < (uint32_t)rn_val) ? 1.0 : 0.0;
                    V = 0.0;
                }
                break;
            }
            case 0x8B: { // ADD shifted register 64-bit
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                if (rd < 31) WRITE_REG(rd, rn_val + rm_val);
                break;
            }
            case 0xCB: { // SUB shifted register 64-bit
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                if (rd < 31) WRITE_REG(rd, rn_val - rm_val);
                break;
            }
            case 0x6B: { // SUBS shifted register 32-bit
                if ((inst & 0xFF200000) == 0x6B000000) {
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    int64_t result = (int32_t)(rn_val - rm_val);
                    if (rd < 31) WRITE_REG(rd, result);
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint32_t)rn_val >= (uint32_t)rm_val) ? 1.0 : 0.0;
                    V = 0.0;
                }
                break;
            }
            case 0xEB: { // SUBS shifted register 64-bit
                if ((inst & 0xFF200000) == 0xEB000000) {
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    int64_t result = rn_val - rm_val;
                    if (rd < 31) WRITE_REG(rd, result);
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint64_t)rn_val >= (uint64_t)rm_val) ? 1.0 : 0.0;
                    V = 0.0;
                }
                break;
            }
            case 0x9B: { // MADD/MUL 64-bit
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                uint8_t ra = (inst >> 10) & 0x1F;
                int64_t ra_val = (ra == 31) ? 0 : READ_REG(ra);
                if (rd < 31) WRITE_REG(rd, rn_val * rm_val + ra_val);
                break;
            }

            // ═══════════════════════════════════════════════════════════════════
            // LOGICAL OPERATIONS
            // ═══════════════════════════════════════════════════════════════════
            case 0x0A: { // logical shifted register 32-bit (AND, ORR, EOR)
                if ((inst & 0xFF200000) == 0x0A000000) { // AND 32-bit
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val & (uint32_t)rm_val));
                } else if ((inst & 0xFF200000) == 0x2A000000) { // ORR 32-bit
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val | (uint32_t)rm_val));
                } else if ((inst & 0xFF200000) == 0x4A000000) { // EOR 32-bit
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val ^ (uint32_t)rm_val));
                }
                break;
            }
            case 0x1A: { // logical shifted register 32-bit (AND, ORR, EOR with different encoding)
                if ((inst & 0xFE200000) == 0x0A000000) { // AND 32-bit
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val & (uint32_t)rm_val));
                } else if ((inst & 0xFE200000) == 0x2A000000) { // ORR 32-bit
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val | (uint32_t)rm_val));
                } else if ((inst & 0xFE200000) == 0x4A000000) { // EOR 32-bit
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val ^ (uint32_t)rm_val));
                }
                break;
            }
            case 0x8A: { // AND shifted register 64-bit
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                WRITE_REG(rd, rn_val & rm_val);
                break;
            }
            case 0xAA: { // ORR shifted register 64-bit (MOV alias)
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                WRITE_REG(rd, rn_val | rm_val);
                break;
            }
            case 0xCA: { // EOR shifted register 64-bit
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                WRITE_REG(rd, rn_val ^ rm_val);
                break;
            }
            case 0xB2: { // ORR/AND immediate 64-bit
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                // Simplified bitmask - just handle common cases
                if (rd < 31) WRITE_REG(rd, rn_val);
                break;
            }
            case 0x72: { // ANDS immediate 32-bit (sets flags)
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                uint32_t imm = decode_immediate_bitmask(inst);
                uint32_t result = (uint32_t)rn_val & imm;
                if (rd < 31) WRITE_REG(rd, (int64_t)(int32_t)result);
                N = ((int32_t)result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = 0.0;
                V = 0.0;
                break;
            }

            // ═══════════════════════════════════════════════════════════════════
            // MEMORY OPERATIONS
            // ═══════════════════════════════════════════════════════════════════
            case 0xF9: { // LDR/STR 64-bit
                int64_t base = READ_REG(rn);
                if ((inst & 0xFFC00000) == 0xF9400000) { // LDR
                    uint64_t addr = (uint64_t)base + (imm12 << 3);
                    if (addr + 8 <= memory_size) {
                        WRITE_REG(rd, (int64_t)load64_aligned(memory, addr));
                    }
                } else if ((inst & 0xFFC00000) == 0xF9000000) { // STR
                    uint64_t addr = (uint64_t)base + (imm12 << 3);
                    int64_t val = (rd == 31) ? 0 : READ_REG(rd);
                    if (addr + 8 <= memory_size) {
                        store64_aligned(memory, addr, (uint64_t)val);
                    }
                }
                break;
            }
            case 0xB9: { // LDR/STR 32-bit
                int64_t base = READ_REG(rn);
                if ((inst & 0xFFC00000) == 0xB9400000) { // LDR
                    uint64_t addr = (uint64_t)base + (imm12 << 2);
                    if (addr + 4 <= memory_size) {
                        WRITE_REG(rd, (int64_t)load32_aligned(memory, addr));
                    }
                } else if ((inst & 0xFFC00000) == 0xB9000000) { // STR
                    uint64_t addr = (uint64_t)base + (imm12 << 2);
                    int64_t val = (rd == 31) ? 0 : READ_REG(rd);
                    if (addr + 4 <= memory_size) {
                        *((device uint32_t*)(memory + addr)) = (uint32_t)val;
                    }
                } else if ((inst & 0xFFC00000) == 0xB9800000) { // LDRSW
                    uint64_t addr = (uint64_t)base + (imm12 << 2);
                    if (addr + 4 <= memory_size && rd < 31) {
                        int32_t val = load32_aligned(memory, addr);
                        WRITE_REG(rd, (int64_t)val);
                    }
                }
                break;
            }
            case 0x38: { // STURB (unprivileged store byte)
                // STURB uses a signed 9-bit offset
                int64_t base = READ_REG(rn);
                int16_t imm9 = (inst >> 12) & 0x1FF;
                if (imm9 & 0x100) imm9 |= 0xFE00;  // Sign extend
                uint64_t addr = (uint64_t)(base + imm9);
                int64_t val = (rd == 31) ? 0 : READ_REG(rd);
                if (addr < memory_size) {
                    memory[addr] = (uint8_t)val;
                }
                break;
            }
            case 0x39: { // LDRB/STRB/LDRSB
                int64_t base = READ_REG(rn);
                if ((inst & 0xFFC00000) == 0x39400000) { // LDRB
                    uint64_t addr = (uint64_t)base + imm12;
                    if (addr < memory_size) {
                        WRITE_REG(rd, (int64_t)memory[addr]);
                    }
                } else if ((inst & 0xFFC00000) == 0x39000000) { // STRB
                    uint64_t addr = (uint64_t)base + imm12;
                    int64_t val = (rd == 31) ? 0 : READ_REG(rd);
                    if (addr < memory_size) {
                        memory[addr] = (uint8_t)val;
                    }
                } else if ((inst & 0xFFC00000) == 0x39800000) { // LDRSB
                    uint64_t addr = (uint64_t)base + imm12;
                    if (addr < memory_size && rd < 31) {
                        WRITE_REG(rd, (int64_t)(int8_t)memory[addr]);
                    }
                }
                break;
            }
            case 0x79: { // LDRH/STRH/LDRSH
                int64_t base = READ_REG(rn);
                uint64_t addr = (uint64_t)base + (imm12 << 1);
                if ((inst & 0xFFC00000) == 0x79400000) { // LDRH
                    if (addr + 2 <= memory_size && rd < 31) {
                        WRITE_REG(rd, (int64_t)(uint16_t)(memory[addr] | ((uint16_t)memory[addr + 1] << 8)));
                    }
                } else if ((inst & 0xFFC00000) == 0x79000000) { // STRH
                    int64_t val = (rd == 31) ? 0 : READ_REG(rd);
                    if (addr + 2 <= memory_size) {
                        memory[addr] = (uint8_t)(val & 0xFF);
                        memory[addr + 1] = (uint8_t)((val >> 8) & 0xFF);
                    }
                } else if ((inst & 0xFFC00000) == 0x79800000) { // LDRSH
                    if (addr + 2 <= memory_size && rd < 31) {
                        int16_t val = (int16_t)(memory[addr] | ((uint16_t)memory[addr + 1] << 8));
                        WRITE_REG(rd, (int64_t)val);
                    }
                }
                break;
            }
            case 0xA9: case 0xA8: { // STP 64-bit
                if (((inst >> 27) & 0x1F) == 0x15) {
                    bool is_load = ((inst >> 22) & 1) == 1;
                    uint8_t rt1 = inst & 0x1F;
                    uint8_t rt2 = (inst >> 10) & 0x1F;
                    uint8_t xn = (inst >> 5) & 0x1F;
                    int8_t imm7 = (inst >> 15) & 0x7F;
                    if (imm7 & 0x40) imm7 |= (int8_t)0x80;
                    int64_t offset = imm7 * 8;
                    int64_t base = READ_REG(xn);
                    uint64_t addr = (uint64_t)base + offset;
                    if (is_load) {
                        if (addr + 16 <= memory_size) {
                            if (rt1 != 31) WRITE_REG(rt1, (int64_t)load64_aligned(memory, addr));
                            if (rt2 != 31) WRITE_REG(rt2, (int64_t)load64_aligned(memory, addr + 8));
                        }
                    } else {
                        if (addr + 16 <= memory_size) {
                            int64_t v1 = (rt1 == 31) ? 0 : READ_REG(rt1);
                            int64_t v2 = (rt2 == 31) ? 0 : READ_REG(rt2);
                            store64_aligned(memory, addr, (uint64_t)v1);
                            store64_aligned(memory, addr + 8, (uint64_t)v2);
                        }
                    }
                    if ((inst >> 23) & 1) {
                        WRITE_REG(xn, (int64_t)addr);
                    }
                }
                break;
            }
            case 0x29: case 0x28: { // STP 32-bit
                if (((inst >> 27) & 0x1F) == 0x15) {
                    bool is_load = ((inst >> 22) & 1) == 1;
                    uint8_t rt1 = inst & 0x1F;
                    uint8_t rt2 = (inst >> 10) & 0x1F;
                    uint8_t xn = (inst >> 5) & 0x1F;
                    int8_t imm7 = (inst >> 15) & 0x7F;
                    if (imm7 & 0x40) imm7 |= (int8_t)0x80;
                    int64_t offset = imm7 * 4;
                    int64_t base = READ_REG(xn);
                    uint64_t addr = (uint64_t)base + offset;
                    if (is_load) {
                        if (addr + 8 <= memory_size) {
                            if (rt1 != 31) WRITE_REG(rt1, (int64_t)(int32_t)load32_aligned(memory, addr));
                            if (rt2 != 31) WRITE_REG(rt2, (int64_t)(int32_t)load32_aligned(memory, addr + 4));
                        }
                    } else {
                        if (addr + 8 <= memory_size) {
                            int64_t v1 = (rt1 == 31) ? 0 : READ_REG(rt1);
                            int64_t v2 = (rt2 == 31) ? 0 : READ_REG(rt2);
                            *((device uint32_t*)(memory + addr)) = (uint32_t)v1;
                            *((device uint32_t*)(memory + addr + 4)) = (uint32_t)v2;
                        }
                    }
                    if ((inst >> 23) & 1) {
                        WRITE_REG(xn, (int64_t)addr);
                    }
                }
                break;
            }

            // ═══════════════════════════════════════════════════════════════════
            // BRANCH OPERATIONS
            // ═══════════════════════════════════════════════════════════════════
            case 0x14: case 0x15: case 0x16: case 0x17: { // B unconditional
                if ((inst & 0xFC000000) == 0x14000000) {
                    int32_t imm26 = inst & 0x3FFFFFF;
                    if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                    pc = pc + (int64_t)imm26 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0x94: case 0x95: case 0x96: case 0x97: { // BL
                if ((inst & 0xFC000000) == 0x94000000) {
                    int32_t imm26 = inst & 0x3FFFFFF;
                    if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                    r30 = (int64_t)(pc + 4);
                    pc = pc + (int64_t)imm26 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0xD6: { // BR/BLR/RET
                if ((inst & 0xFFFFFC00) == 0xD61F0000) { // BR
                    pc = (uint64_t)READ_REG(rn);
                    branch_taken = true;
                } else if ((inst & 0xFFFFFC00) == 0xD63F0000) { // BLR
                    r30 = (int64_t)(pc + 4);
                    pc = (uint64_t)READ_REG(rn);
                    branch_taken = true;
                } else if ((inst & 0xFFFFFC00) == 0xD65F0000) { // RET
                    pc = (uint64_t)READ_REG(rn);
                    branch_taken = true;
                }
                break;
            }
            case 0xB4: { // CBZ 64-bit
                uint8_t rt = rd;
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                int64_t rt_val = (rt == 31) ? 0 : READ_REG(rt);
                if (rt_val == 0) {
                    pc = pc + (int64_t)imm19 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0xB5: { // CBNZ 64-bit
                uint8_t rt = rd;
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                int64_t rt_val = (rt == 31) ? 0 : READ_REG(rt);
                if (rt_val != 0) {
                    pc = pc + (int64_t)imm19 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0x34: { // CBZ 32-bit
                uint8_t rt = rd;
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                int32_t rt_val = (rt == 31) ? 0 : (int32_t)READ_REG(rt);
                if (rt_val == 0) {
                    pc = pc + (int64_t)imm19 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0x35: { // CBNZ 32-bit
                uint8_t rt = rd;
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                int32_t rt_val = (rt == 31) ? 0 : (int32_t)READ_REG(rt);
                if (rt_val != 0) {
                    pc = pc + (int64_t)imm19 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0x54: { // B.cond
                if ((inst & 0xFF000010) == 0x54000000) {
                    uint8_t cond = inst & 0xF;
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    bool take = false;
                    switch (cond) {
                        case 0x0: take = (Z > 0.5); break;
                        case 0x1: take = (Z < 0.5); break;
                        case 0x2: take = (C > 0.5); break;
                        case 0x3: take = (C < 0.5); break;
                        case 0x4: take = (N > 0.5); break;
                        case 0x5: take = (N < 0.5); break;
                        case 0x6: take = (V > 0.5); break;
                        case 0x7: take = (V < 0.5); break;
                        case 0x8: take = (C > 0.5 && Z < 0.5); break;
                        case 0x9: take = (C < 0.5 || Z > 0.5); break;
                        case 0xA: take = ((N > 0.5) == (V > 0.5)); break;
                        case 0xB: take = ((N > 0.5) != (V > 0.5)); break;
                        case 0xC: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                        case 0xD: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                        case 0xE: case 0xF: take = true; break;
                    }
                    if (take) {
                        pc = pc + (int64_t)imm19 * 4;
                        branch_taken = true;
                    }
                }
                break;
            }
            case 0x36: { // TBZ
                uint8_t rt = rd;
                uint8_t bit_pos = ((inst >> 19) & 0x1F) | (((inst >> 31) & 1) << 5);
                int32_t imm14 = (inst >> 5) & 0x3FFF;
                if (imm14 & 0x2000) imm14 |= (int32_t)0xFFFFC000;
                int64_t rt_val = (rt == 31) ? 0 : READ_REG(rt);
                if (((rt_val >> bit_pos) & 1) == 0) {
                    pc = pc + (int64_t)imm14 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0x37: { // TBNZ
                uint8_t rt = rd;
                uint8_t bit_pos = ((inst >> 19) & 0x1F) | (((inst >> 31) & 1) << 5);
                int32_t imm14 = (inst >> 5) & 0x3FFF;
                if (imm14 & 0x2000) imm14 |= (int32_t)0xFFFFC000;
                int64_t rt_val = (rt == 31) ? 0 : READ_REG(rt);
                if (((rt_val >> bit_pos) & 1) != 0) {
                    pc = pc + (int64_t)imm14 * 4;
                    branch_taken = true;
                }
                break;
            }

            // ═══════════════════════════════════════════════════════════════════
            // PC-RELATIVE ADDRESSING
            // ═══════════════════════════════════════════════════════════════════
            case 0x90: case 0xB0: case 0xD0: case 0xF0: { // ADRP
                if ((inst & 0x9F000000) == 0x90000000) {
                    int32_t immhi = (inst >> 5) & 0x7FFFF;
                    int32_t immlo = (inst >> 29) & 0x3;
                    int32_t imm = (immhi << 2) | immlo;
                    if (imm & 0x100000) imm |= (int32_t)0xFFE00000;
                    int64_t page = (int64_t)(pc & ~0xFFFULL) + ((int64_t)imm << 12);
                    if (rd < 31) WRITE_REG(rd, page);
                }
                break;
            }
            case 0x10: case 0x30: case 0x50: case 0x70: { // ADR
                if ((inst & 0x9F000000) == 0x10000000) {
                    int32_t immhi = (inst >> 5) & 0x7FFFF;
                    int32_t immlo = (inst >> 29) & 0x3;
                    int32_t imm = (immhi << 2) | immlo;
                    if (imm & 0x100000) imm |= (int32_t)0xFFE00000;
                    WRITE_REG(rd, (int64_t)(pc + imm));
                }
                break;
            }

            // ═══════════════════════════════════════════════════════════════════
            // CONDITIONAL SELECT
            // ═══════════════════════════════════════════════════════════════════
            case 0x9A: { // CSEL/CSINC 64-bit
                uint8_t cond = (inst >> 12) & 0xF;
                int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                bool cond_true = false;
                switch (cond) {
                    case 0x0: cond_true = (Z > 0.5); break;
                    case 0x1: cond_true = (Z < 0.5); break;
                    case 0x2: cond_true = (C > 0.5); break;
                    case 0x3: cond_true = (C < 0.5); break;
                    case 0x4: cond_true = (N > 0.5); break;
                    case 0x5: cond_true = (N < 0.5); break;
                    case 0x6: cond_true = (V > 0.5); break;
                    case 0x7: cond_true = (V < 0.5); break;
                    case 0x8: cond_true = (C > 0.5 && Z < 0.5); break;
                    case 0x9: cond_true = (C < 0.5 || Z > 0.5); break;
                    case 0xA: cond_true = ((N > 0.5) == (V > 0.5)); break;
                    case 0xB: cond_true = ((N > 0.5) != (V > 0.5)); break;
                    case 0xC: cond_true = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                    case 0xD: cond_true = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                    case 0xE: case 0xF: cond_true = true; break;
                }
                if ((inst & 0xFFE00C00) == 0x9A800000) { // CSEL
                    if (rd < 31) WRITE_REG(rd, cond_true ? rn_val : rm_val);
                } else if ((inst & 0xFFE00C00) == 0x9A800400) { // CSINC
                    if (rd < 31) WRITE_REG(rd, cond_true ? rn_val : (rm_val + 1));
                }
                break;
            }
            case 0xDA: { // CSINV/CSNEG/CLZ/RBIT/REV
                if ((inst & 0xFFFFFC00) == 0xDAC01000) { // CLZ 64-bit
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    uint64_t val = (uint64_t)rn_val;
                    int count = 0;
                    if (val == 0) {
                        count = 64;
                    } else {
                        while ((val & 0x8000000000000000ULL) == 0) {
                            count++;
                            val <<= 1;
                        }
                    }
                    if (rd < 31) WRITE_REG(rd, count);
                } else {
                    // CSINV/CSNEG
                    uint8_t cond = (inst >> 12) & 0xF;
                    int64_t rn_val = (rn == 31) ? 0 : READ_REG(rn);
                    int64_t rm_val = (rm == 31) ? 0 : READ_REG(rm);
                    bool cond_true = false;
                    switch (cond) {
                        case 0x0: cond_true = (Z > 0.5); break;
                        case 0x1: cond_true = (Z < 0.5); break;
                        default: break;
                    }
                    if ((inst & 0xFFE00C00) == 0xDA800000) { // CSINV
                        if (rd < 31) WRITE_REG(rd, cond_true ? rn_val : ~rm_val);
                    } else if ((inst & 0xFFE00C00) == 0xDA800400) { // CSNEG
                        if (rd < 31) WRITE_REG(rd, cond_true ? rn_val : -rm_val);
                    }
                }
                break;
            }

            // ═══════════════════════════════════════════════════════════════════
            // DIVISION
            // ═══════════════════════════════════════════════════════════════════
            case 0x1A: { // UDIV/SDIV 32-bit
                if ((inst & 0xFFE0FC00) == 0x1AC00800) { // UDIV 32-bit
                    uint32_t dividend = (uint32_t)((rn == 31) ? 0 : READ_REG(rn));
                    uint32_t divisor = (uint32_t)((rm == 31) ? 0 : READ_REG(rm));
                    if (rd < 31) WRITE_REG(rd, (divisor == 0) ? 0 : (int64_t)(dividend / divisor));
                } else if ((inst & 0xFFE0FC00) == 0x1AC00C00) { // SDIV 32-bit
                    int32_t dividend = (int32_t)((rn == 31) ? 0 : READ_REG(rn));
                    int32_t divisor = (int32_t)((rm == 31) ? 0 : READ_REG(rm));
                    if (rd < 31) WRITE_REG(rd, (divisor == 0) ? 0 : (int64_t)(dividend / divisor));
                }
                break;
            }

            // ═══════════════════════════════════════════════════════════════════
            // NOP AND OTHER SYSTEM
            // ═══════════════════════════════════════════════════════════════════
            case 0xD5: {
                if (inst == 0xD503201F) {
                    // NOP - do nothing
                }
                break;
            }

            default:
                // Unknown instruction - just advance PC
                break;
        }

        if (!branch_taken) pc += 4;
        cycles++;
    }

    // Write back registers (including SP in r31)
    registers[0] = r0; registers[1] = r1; registers[2] = r2; registers[3] = r3;
    registers[4] = r4; registers[5] = r5; registers[6] = r6; registers[7] = r7;
    registers[8] = r8; registers[9] = r9; registers[10] = r10; registers[11] = r11;
    registers[12] = r12; registers[13] = r13; registers[14] = r14; registers[15] = r15;
    registers[16] = r16; registers[17] = r17; registers[18] = r18; registers[19] = r19;
    registers[20] = r20; registers[21] = r21; registers[22] = r22; registers[23] = r23;
    registers[24] = r24; registers[25] = r25; registers[26] = r26; registers[27] = r27;
    registers[28] = r28; registers[29] = r29; registers[30] = r30; registers[31] = r31;

    pc_ptr[0] = pc;
    flags[0] = N; flags[1] = Z; flags[2] = C; flags[3] = V;

    atomic_fetch_add_explicit((device atomic_uint*)total_cycles, cycles, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)batch_count, 1, memory_order_relaxed);

    uint32_t current_signal = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (current_signal == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

/// Optimized execution result
#[pyclass]
#[derive(Debug, Clone)]
pub struct OptimizedResult {
    #[pyo3(get)]
    pub total_cycles: u32,
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    #[pyo3(get)]
    pub signal: u32,
    #[pyo3(get)]
    pub final_pc: u64,
}

#[pymethods]
impl OptimizedResult {
    #[getter]
    fn ips(&self) -> f64 {
        if self.elapsed_seconds > 0.0 {
            self.total_cycles as f64 / self.elapsed_seconds
        } else {
            0.0
        }
    }

    #[getter]
    fn signal_name(&self) -> &str {
        match self.signal {
            0 => "RUNNING",
            1 => "HALT",
            2 => "SYSCALL",
            3 => "CHECKPOINT",
            _ => "UNKNOWN",
        }
    }
}

/// Optimized Metal CPU with switch-based dispatch and aligned memory access
#[pyclass(unsendable)]
pub struct OptimizedMetalCPU {
    #[allow(dead_code)]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    memory_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    max_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mem_size_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    signal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    memory_size: usize,
}

#[pymethods]
impl OptimizedMetalCPU {
    #[new]
    #[pyo3(signature = (memory_size = 4 * 1024 * 1024))]
    fn new(memory_size: usize) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[OptimizedMetalCPU] Using device: {:?}", device.name());

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        // Compile optimized shader
        let source = NSString::from_str(OPTIMIZED_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let function_name = NSString::from_str("cpu_execute_optimized");
        let function = library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("Function not found".to_string()))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        let shared_options = MTLResourceOptions::StorageModeShared;

        // Create buffers
        let memory_buffer = device
            .newBufferWithLength_options(memory_size, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let registers_buffer = device
            .newBufferWithLength_options(32 * std::mem::size_of::<i64>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let pc_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u64>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let flags_buffer = device
            .newBufferWithLength_options(4 * std::mem::size_of::<f32>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let max_cycles_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let mem_size_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let signal_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let total_cycles_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let batch_count_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // Initialize mem_size
        unsafe {
            let ptr = mem_size_buffer.contents().as_ptr() as *mut u32;
            *ptr = memory_size as u32;
        }

        println!("[OptimizedMetalCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[OptimizedMetalCPU] Using switch-based dispatch + aligned memory access");

        Ok(OptimizedMetalCPU {
            device,
            command_queue,
            pipeline,
            memory_buffer,
            registers_buffer,
            pc_buffer,
            flags_buffer,
            max_cycles_buffer,
            mem_size_buffer,
            signal_buffer,
            total_cycles_buffer,
            batch_count_buffer,
            memory_size,
        })
    }

    fn load_program(&self, program: Vec<u8>, address: usize) -> PyResult<()> {
        if address + program.len() > self.memory_size {
            return Err(PyRuntimeError::new_err("Program exceeds memory size"));
        }

        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(program.as_ptr(), ptr.add(address), program.len());
        }

        Ok(())
    }

    fn set_pc(&self, pc: u64) {
        unsafe {
            let ptr = self.pc_buffer.contents().as_ptr() as *mut u64;
            *ptr = pc;
        }
    }

    fn get_pc(&self) -> u64 {
        unsafe {
            let ptr = self.pc_buffer.contents().as_ptr() as *const u64;
            *ptr
        }
    }

    fn set_register(&self, reg: usize, value: i64) {
        if reg >= 32 { return; }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *mut i64;
            *ptr.add(reg) = value;
        }
    }

    fn get_register(&self, reg: usize) -> i64 {
        if reg >= 32 { return 0; }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *const i64;
            *ptr.add(reg)
        }
    }

    /// Execute with optimized kernel
    #[pyo3(signature = (total_cycles = 100_000_000))]
    fn execute(&self, total_cycles: u32) -> PyResult<OptimizedResult> {
        let start = Instant::now();

        // Reset counters
        unsafe {
            let ptr = self.total_cycles_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
            let ptr = self.batch_count_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
            let ptr = self.signal_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0; // RUNNING
            let ptr = self.max_cycles_buffer.contents().as_ptr() as *mut u32;
            *ptr = total_cycles;
        }

        // Single dispatch
        let command_buffer = self
            .command_queue
            .commandBuffer()
            .ok_or(MetalError::ExecutionFailed)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ExecutionFailed)?;

        encoder.setComputePipelineState(&self.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.memory_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.registers_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.pc_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.flags_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.max_cycles_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&self.mem_size_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.signal_buffer), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&self.total_cycles_buffer), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&self.batch_count_buffer), 0, 8);

            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
        }
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Read results
        let (actual_cycles, signal, final_pc) = unsafe {
            let total = *(self.total_cycles_buffer.contents().as_ptr() as *const u32);
            let sig = *(self.signal_buffer.contents().as_ptr() as *const u32);
            let pc = *(self.pc_buffer.contents().as_ptr() as *const u64);
            (total, sig, pc)
        };

        let elapsed = start.elapsed().as_secs_f64();

        Ok(OptimizedResult {
            total_cycles: actual_cycles,
            elapsed_seconds: elapsed,
            signal,
            final_pc,
        })
    }

    #[getter]
    fn memory_size(&self) -> usize {
        self.memory_size
    }

    /// Read memory at address
    fn read_memory(&self, address: usize, size: usize) -> PyResult<Vec<u8>> {
        if address + size > self.memory_size {
            return Err(PyRuntimeError::new_err("Read exceeds memory bounds"));
        }

        let mut result = vec![0u8; size];
        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(ptr.add(address), result.as_mut_ptr(), size);
        }
        Ok(result)
    }

    /// Write memory at address
    fn write_memory(&self, address: usize, data: Vec<u8>) -> PyResult<()> {
        if address + data.len() > self.memory_size {
            return Err(PyRuntimeError::new_err("Write exceeds memory bounds"));
        }

        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(address), data.len());
        }
        Ok(())
    }
}

/// Register optimized types with Python module
pub fn register_optimized(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptimizedMetalCPU>()?;
    m.add_class::<OptimizedResult>()?;
    Ok(())
}
