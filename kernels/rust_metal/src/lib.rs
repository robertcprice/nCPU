//! KVRM Metal - High-performance Metal GPU kernel for ARM64 CPU emulation
//!
//! This crate provides direct Metal API access via objc2-metal for:
//! - Zero-copy shared memory between CPU and GPU
//! - Indirect Command Buffers for GPU-autonomous dispatch
//! - Maximum performance CPU emulation
//!
//! Exposed to Python via PyO3.

mod continuous;
mod async_gpu;
mod parallel;
mod multi_kernel;
mod neural_dispatch;
mod neural_weights;
mod pure_gpu;
mod optimized;
mod ultra_optimized;
mod fusion;
mod bb_cache;
mod ultra;
mod trace_jit;
mod ooo_exec;
mod neural_ooo;
mod neural_hybrid;
mod differentiable_ooo;
mod jit_compiler;
mod diff_jit;
mod unified_diff_cpu;
mod unified_test_kernel;

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
use thiserror::Error;

/// Metal shader source for ARM64 CPU emulation kernel
const METAL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Stop reasons
constant uint8_t STOP_RUNNING = 0;
constant uint8_t STOP_HALT = 1;
constant uint8_t STOP_SYSCALL = 2;
constant uint8_t STOP_MAX_CYCLES = 3;

// Helper functions
inline uint64_t load64(device uint8_t* mem, uint64_t addr) {
    return uint64_t(mem[addr]) |
           (uint64_t(mem[addr + 1]) << 8) |
           (uint64_t(mem[addr + 2]) << 16) |
           (uint64_t(mem[addr + 3]) << 24) |
           (uint64_t(mem[addr + 4]) << 32) |
           (uint64_t(mem[addr + 5]) << 40) |
           (uint64_t(mem[addr + 6]) << 48) |
           (uint64_t(mem[addr + 7]) << 56);
}

inline void store64(device uint8_t* mem, uint64_t addr, uint64_t val) {
    mem[addr] = val & 0xFF;
    mem[addr + 1] = (val >> 8) & 0xFF;
    mem[addr + 2] = (val >> 16) & 0xFF;
    mem[addr + 3] = (val >> 24) & 0xFF;
    mem[addr + 4] = (val >> 32) & 0xFF;
    mem[addr + 5] = (val >> 40) & 0xFF;
    mem[addr + 6] = (val >> 48) & 0xFF;
    mem[addr + 7] = (val >> 56) & 0xFF;
}

inline uint32_t load32(device uint8_t* mem, uint64_t addr) {
    return uint32_t(mem[addr]) |
           (uint32_t(mem[addr + 1]) << 8) |
           (uint32_t(mem[addr + 2]) << 16) |
           (uint32_t(mem[addr + 3]) << 24);
}

inline void store32(device uint8_t* mem, uint64_t addr, uint32_t val) {
    mem[addr] = val & 0xFF;
    mem[addr + 1] = (val >> 8) & 0xFF;
    mem[addr + 2] = (val >> 16) & 0xFF;
    mem[addr + 3] = (val >> 24) & 0xFF;
}

// Main CPU execution kernel - runs entirely on GPU
kernel void cpu_execute(
    device uint8_t* memory [[buffer(0)]],          // Read-write memory (shared)
    device int64_t* registers [[buffer(1)]],       // Read-write registers
    device uint64_t* pc_ptr [[buffer(2)]],         // Read-write program counter
    device float* flags [[buffer(3)]],             // Read-write NZCV flags
    device const uint32_t* max_cycles [[buffer(4)]],
    device const uint32_t* mem_size [[buffer(5)]],
    device uint32_t* cycles_out [[buffer(6)]],     // Output: cycles executed
    device uint8_t* stop_reason [[buffer(7)]],     // Output: stop reason
    device uint8_t* syscall_write [[buffer(8)]],   // Syscall write output buffer
    device uint8_t* syscall_read [[buffer(9)]],    // Syscall read input buffer
    device uint64_t* syscall_info [[buffer(10)]],  // Syscall info [num, fd, buf, count, result, key_state]
    uint tid [[thread_position_in_grid]]
) {
    // Single-threaded execution
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t cycles = 0;
    uint32_t max = max_cycles[0];
    uint32_t memory_size = mem_size[0];

    // Copy registers to local for faster access
    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers[i];
    }
    // CRITICAL: Preserve x31 (SP) value explicitly
    // x31 is SP when used as destination/base register, not XZR
    int64_t sp_value = registers[31];  // Save SP value DIRECT from input buffer

    // Copy flags
    float N = flags[0];
    float Z = flags[1];
    float C = flags[2];
    float V = flags[3];

    *stop_reason = STOP_RUNNING;

    // Main execution loop - runs autonomously on GPU
    while (cycles < max) {
        // Bounds check
        if (pc + 4 > memory_size) {
            *stop_reason = STOP_HALT;
            break;
        }

        // FETCH instruction (little-endian)
        uint32_t inst = load32(memory, pc);

        // DECODE opcode byte
        uint8_t op_byte = (inst >> 24) & 0xFF;

        // Check for HALT (HLT #imm16)
        if ((inst & 0xFFE0001F) == 0xD4400000) {
            *stop_reason = STOP_HALT;
            break;
        }

        // Check for SYSCALL (SVC #imm16) - GPU-side syscall handling
        if ((inst & 0xFFE0001F) == 0xD4000001) {
            // ARM64 syscall number is in x8
            int64_t syscall_num = regs[8];
            int64_t result = 0;

            // Syscall info: [num, fd, buf, count, result, key_state]
            syscall_info[0] = (uint64_t)syscall_num;  // syscall number
            syscall_info[1] = (uint64_t)regs[0];      // fd (x0)
            syscall_info[2] = (uint64_t)regs[1];      // buf (x1)
            syscall_info[3] = (uint64_t)regs[2];      // count (x2)
            syscall_info[5] = (uint64_t)regs[3];      // key_state (x3) - for DOOM input

            // Handle syscalls on GPU
            if (syscall_num == 64) {  // SYS_WRITE
                int64_t fd = regs[0];
                uint64_t buf = (uint64_t)regs[1];
                int64_t count = regs[2];
                // Write to syscall_write buffer for Python to read
                if ((fd == 1 || fd == 2) && count > 0 && count < 4096) {
                    for (int i = 0; i < count && i < 4096; i++) {
                        syscall_write[i] = memory[buf + i];
                    }
                    syscall_info[4] = (uint64_t)count;  // result = bytes written
                } else {
                    syscall_info[4] = 0;
                }
                result = (int64_t)syscall_info[4];
            }
            else if (syscall_num == 63) {  // SYS_READ
                int64_t fd = regs[0];
                uint64_t buf = (uint64_t)regs[1];
                int64_t count = regs[2];
                // Read from syscall_read buffer (populated by Python)
                if (fd == 0 && count > 0) {
                    int bytes_to_read = (count < 256) ? count : 256;
                    // Check if input available (key_state > 0)
                    uint64_t key_state = syscall_info[5];
                    if (key_state > 0 && key_state < 6) {
                        // Map key_state to character directly (1=w, 2=s, 3=a, 4=d, 5=q)
                        uint8_t key_char = 0;
                        if (key_state == 1) key_char = 119;       // 'w'
                        else if (key_state == 2) key_char = 115;   // 's'
                        else if (key_state == 3) key_char = 97;    // 'a'
                        else if (key_state == 4) key_char = 100;   // 'd'
                        else if (key_state == 5) key_char = 113;   // 'q'

                        memory[buf] = key_char;
                        result = 1;
                        syscall_info[4] = 1;
                    } else {
                        result = 0;  // No input available
                        syscall_info[4] = 0;
                    }
                } else {
                    result = 0;
                    syscall_info[4] = 0;
                }
            }
            else if (syscall_num == 93) {  // SYS_EXIT
                *stop_reason = STOP_HALT;  // Use HALT for exit
                result = 0;
                syscall_info[4] = 0;
                break;
            }
            else if (syscall_num == 29) {  // SYS_IOCTL
                // Just return success (0)
                result = 0;
                syscall_info[4] = 0;
            }
            else if (syscall_num == 172) {  // SYS_GETPID
                // Return fake PID
                result = 1;
                syscall_info[4] = 1;
            }
            else if (syscall_num == 214) {  // SYS_BRK
                // Simple brk - return address or default
                int64_t addr = regs[0];
                result = (addr > 0) ? addr : 0x1000000;
                syscall_info[4] = (uint64_t)result;
            }
            else {
                // Unknown syscall - just continue
                result = 0;
                syscall_info[4] = 0;
            }

            // Set return value in x0
            regs[0] = result;
            pc += 4;  // Advance PC past syscall
            continue;  // Continue execution
        }

        // Extract common fields
        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint8_t rm = (inst >> 16) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint16_t imm16 = (inst >> 5) & 0xFFFF;
        uint8_t hw = (inst >> 21) & 0x3;

        bool branch_taken = false;

        // ═══════════════════════════════════════════════════════════════════
        // DATA PROCESSING - IMMEDIATE
        // ═══════════════════════════════════════════════════════════════════

        // ADD (immediate) - 64-bit: 0x91xxxxxx
        if ((inst & 0xFF000000) == 0x91000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            regs[rd] = rn_val + imm12;
        }
        // ADD (immediate) - 32-bit: 0x11xxxxxx
        else if ((inst & 0xFF000000) == 0x11000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val + imm12);
        }
        // ADDS (immediate) - 64-bit: 0xB1xxxxxx
        else if ((inst & 0xFF000000) == 0xB1000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val + imm12;
            regs[rd] = result;
            N = (result < 0) ? 1.0 : 0.0;
            Z = (result == 0) ? 1.0 : 0.0;
            C = ((uint64_t)result < (uint64_t)rn_val) ? 1.0 : 0.0;
            V = ((rn_val > 0 && imm12 > 0 && result < 0) ||
                 (rn_val < 0 && (int64_t)imm12 < 0 && result > 0)) ? 1.0 : 0.0;
        }
        // SUB (immediate) - 64-bit: 0xD1xxxxxx
        else if ((inst & 0xFF000000) == 0xD1000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            regs[rd] = rn_val - imm12;
        }
        // SUB (immediate) - 32-bit: 0x51xxxxxx
        else if ((inst & 0xFF000000) == 0x51000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val - imm12);
        }
        // SUBS (immediate) - 64-bit: 0xF1xxxxxx
        else if ((inst & 0xFF000000) == 0xF1000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val - imm12;
            regs[rd] = result;
            N = (result < 0) ? 1.0 : 0.0;
            Z = (result == 0) ? 1.0 : 0.0;
            C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
            V = ((rn_val > 0 && (int64_t)imm12 < 0 && result < 0) ||
                 (rn_val < 0 && (int64_t)imm12 > 0 && result > 0)) ? 1.0 : 0.0;
        }
        // SUBS (shifted register) - 32-bit: 0x6Bxxxxxx
        else if ((inst & 0xFF200000) == 0x6B000000) {
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = (int32_t)(rn_val - rm_val);  // 32-bit result
            if (rd < 31) {
                regs[rd] = result;
            }
            // Update flags for 32-bit operation
            N = (result < 0) ? 1.0 : 0.0;
            Z = (result == 0) ? 1.0 : 0.0;
            C = ((uint32_t)rn_val >= (uint32_t)rm_val) ? 1.0 : 0.0;
            V = (((rn_val ^ rm_val) & (rn_val ^ result)) < 0) ? 1.0 : 0.0;  // Signed overflow
        }
        // SUBS (shifted register) - 64-bit: 0xEBxxxxxx
        else if ((inst & 0xFF200000) == 0xEB000000) {
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val - rm_val;
            if (rd < 31) {
                regs[rd] = result;
            }
            N = (result < 0) ? 1.0 : 0.0;
            Z = (result == 0) ? 1.0 : 0.0;
            C = ((uint64_t)rn_val >= (uint64_t)rm_val) ? 1.0 : 0.0;
            V = (((rn_val ^ rm_val) & (rn_val ^ result)) < 0) ? 1.0 : 0.0;  // Signed overflow
        }
        // MOVZ - 64-bit: 0xD28xxxxx
        else if ((inst & 0xFF800000) == 0xD2800000) {
            regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
        }
        // MOVZ - 32-bit: 0x528xxxxx
        else if ((inst & 0xFF800000) == 0x52800000) {
            regs[rd] = (int64_t)(uint32_t)(imm16 << (hw * 16));
        }
        // MOVK - 64-bit: 0xF28xxxxx
        else if ((inst & 0xFF800000) == 0xF2800000) {
            uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
            uint64_t val = (uint64_t)regs[rd] & mask;
            val |= ((uint64_t)imm16 << (hw * 16));
            regs[rd] = (int64_t)val;
        }
        // MOVN - 64-bit: 0x928xxxxx
        else if ((inst & 0xFF800000) == 0x92800000) {
            regs[rd] = ~((int64_t)((uint64_t)imm16 << (hw * 16)));
        }

        // ADD (shifted register) - 64-bit: 0x8B0xxxxx
        else if ((inst & 0xFF200000) == 0x8B000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = rn_val + rm_val;
        }
        // SUB (shifted register) - 64-bit: 0xCB0xxxxx
        else if ((inst & 0xFF200000) == 0xCB000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = rn_val - rm_val;
        }

        // ═══════════════════════════════════════════════════════════════════
        // LOGICAL OPERATIONS
        // ═══════════════════════════════════════════════════════════════════

        // AND (shifted register) - 64-bit: 0x8A0xxxxx
        else if ((inst & 0xFF200000) == 0x8A000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            regs[rd] = rn_val & rm_val;
        }
        // ORR (shifted register) - 64-bit: 0xAA0xxxxx
        else if ((inst & 0xFF200000) == 0xAA000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            regs[rd] = rn_val | rm_val;
        }
        // EOR (shifted register) - 64-bit: 0xCA0xxxxx
        else if ((inst & 0xFF200000) == 0xCA000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            regs[rd] = rn_val ^ rm_val;
        }
        // ORR (immediate): 0xB2xxxxxx - check bit 29 to distinguish AND from ORR
        else if ((inst & 0xFF800000) == 0xB2000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            bool is_orr = ((inst >> 29) & 1) == 1;
            if (is_orr) {
                // ORR (immediate) - simplified for DOOM's orr sp, xzr, #0x80000
                uint8_t immr = (inst >> 16) & 0x3F;
                uint8_t imms = (inst >> 10) & 0x3F;
                uint64_t imm = 0;
                if (imms == immr) {
                    imm = 1ULL << immr;
                }
                if (rd < 31) regs[rd] = rn_val | (int64_t)imm;
            } else {
                // AND (immediate)
                uint32_t imm = (inst >> 10) & 0xFFF;
                if (rd < 31) regs[rd] = rn_val & imm;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // MULTIPLY AND DIVIDE
        // ═══════════════════════════════════════════════════════════════════

        // MADD/MUL (multiply-add): 0x9B - rd = rn * rm + ra
        else if ((inst & 0xFF000000) == 0x9B000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            uint8_t ra = (inst >> 10) & 0x1F;
            int64_t ra_val = (ra == 31) ? 0 : regs[ra];
            if (rd < 31) regs[rd] = rn_val * rm_val + ra_val;
        }

        // ═══════════════════════════════════════════════════════════════════
        // BITFIELD OPERATIONS
        // ═══════════════════════════════════════════════════════════════════

        // UBFM (Unsigned Bitfield Move) - used for LSL/LSR/ASR: 0xD3
        else if ((inst & 0xFF000000) == 0xD3000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            uint8_t imms = (inst >> 10) & 0x3F;
            uint8_t immr = (inst >> 16) & 0x3F;
            if (rd < 31) {
                int64_t shifted = 0;
                if (imms < 63) {  // LSL case: imms = (63 - shift), immr = 0
                    int shift = 63 - imms;
                    shifted = (int64_t)((uint64_t)rn_val << shift);
                } else if (imms == 63 && immr > 0) {  // LSR case
                    shifted = (int64_t)((uint64_t)rn_val >> immr);
                } else {
                    // General UBFM
                    uint64_t mask = (1ULL << (imms + 1)) - 1;
                    shifted = (int64_t)(((uint64_t)rn_val >> immr) & mask);
                }
                regs[rd] = shifted;
            }
        }
        // SBFM (Signed Bitfield Move) - used for ASR, SXTB, SXTH, SXTW: 0x93
        else if ((inst & 0xFF000000) == 0x93000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            uint8_t imms = (inst >> 10) & 0x3F;
            uint8_t immr = (inst >> 16) & 0x3F;
            uint8_t n = (inst >> 22) & 0x1;
            if (rd < 31) {
                int64_t result;
                if (imms == 0x1F && immr == 0) {
                    // SXTW (sign extend 32-bit to 64-bit)
                    result = (int64_t)(int32_t)rn_val;
                } else if (imms == 0x0F && immr == 0) {
                    // SXTH (sign extend 16-bit to 64-bit)
                    result = (int64_t)(int16_t)rn_val;
                } else if (imms == 0x07 && immr == 0) {
                    // SXTB (sign extend 8-bit to 64-bit)
                    result = (int64_t)(int8_t)rn_val;
                } else if (imms == 63) {
                    // ASR (arithmetic shift right)
                    result = rn_val >> immr;
                } else {
                    // General SBFM: extract and sign-extend bits
                    int width = imms + 1;
                    uint64_t mask = (1ULL << width) - 1;
                    int64_t extracted = (int64_t)(((uint64_t)rn_val >> immr) & mask);
                    // Sign extend
                    if (extracted & (1ULL << (width - 1))) {
                        extracted |= ~((1ULL << width) - 1);
                    }
                    result = extracted;
                }
                regs[rd] = result;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // MEMORY OPERATIONS
        // ═══════════════════════════════════════════════════════════════════

        // LDR (64-bit, unsigned offset): 0xF94xxxxx
        else if ((inst & 0xFFC00000) == 0xF9400000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + (imm12 << 3);
            if (addr + 8 <= memory_size) {
                regs[rd] = (int64_t)load64(memory, addr);
            }
        }
        // LDR (32-bit, unsigned offset): 0xB94xxxxx
        else if ((inst & 0xFFC00000) == 0xB9400000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + (imm12 << 2);
            if (addr + 4 <= memory_size) {
                regs[rd] = (int64_t)load32(memory, addr);
            }
        }
        // LDRB (unsigned offset): 0x394xxxxx
        else if ((inst & 0xFFC00000) == 0x39400000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + imm12;
            if (addr < memory_size) {
                regs[rd] = (int64_t)memory[addr];
            }
        }
        // STR (64-bit, unsigned offset): 0xF90xxxxx
        else if ((inst & 0xFFC00000) == 0xF9000000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + (imm12 << 3);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 8 <= memory_size) {
                store64(memory, addr, (uint64_t)val);
            }
        }
        // STR (32-bit, unsigned offset): 0xB90xxxxx
        else if ((inst & 0xFFC00000) == 0xB9000000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + (imm12 << 2);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 4 <= memory_size) {
                store32(memory, addr, (uint32_t)val);
            }
        }
        // STRB (unsigned offset): 0x390xxxxx
        else if ((inst & 0xFFC00000) == 0x39000000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + imm12;
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr < memory_size) {
                memory[addr] = (uint8_t)val;
            }
        }
        // LDRB/STRB with register offset: 0x38xxxxxx
        // Load/Store Byte (register offset) - e.g., LDRB xt, [xn, xm, LSL #amount]
        else if ((inst & 0xFFC00000) == 0x38000000) {
            bool is_load = ((inst >> 22) & 1) == 1;
            uint8_t rm = (inst >> 16) & 0x1F;
            uint8_t shift = (inst >> 12) & 0x3;
            int64_t base = (rn == 31) ? 0 : regs[rn];
            int64_t offset = (rm == 31) ? 0 : regs[rm];

            // Apply shift if specified
            if (shift == 1) {
                offset = offset << 12;
            } else if (shift == 2) {
                offset = offset << 12;  // LSL #12 is common for page alignment
            }

            uint64_t addr = (uint64_t)base + offset;

            if (is_load) {
                // LDRB (register offset)
                if (addr < memory_size && rd < 31) {
                    regs[rd] = (int64_t)memory[addr];
                }
            } else {
                // STRB (register offset)
                if (addr < memory_size) {
                    int64_t val = (rd == 31) ? 0 : regs[rd];
                    memory[addr] = (uint8_t)val;
                }
            }
        }
        // STP/LDP (Store/Load Pair): bits 31-27 == 0x15 (10101b)
        // STP xt1, xt2, [xn, #imm] or [xn, #imm]!
        else if (((inst >> 27) & 0x1F) == 0x15) {
            bool is_load = ((inst >> 22) & 1) == 1;  // LDP if set, STP if clear
            bool is_64bit = ((inst >> 31) & 1) == 1;  // 64-bit if set, 32-bit if clear
            uint8_t rt1 = inst & 0x1F;               // First register
            uint8_t rt2 = (inst >> 10) & 0x1F;       // Second register
            uint8_t xn = (inst >> 5) & 0x1F;         // Base register

            // Extract 7-bit signed immediate
            int8_t imm7 = (inst >> 15) & 0x7F;
            if (imm7 & 0x40) imm7 |= (int8_t)0x80;  // Sign extend

            // Calculate offset (imm7 * data_size)
            int64_t offset = imm7 * (is_64bit ? 8 : 4);

            // Get base address (SP is NOT zero for load/store pair base!)
            int64_t base = (xn == 31) ? regs[31] : regs[xn];
            uint64_t addr = (uint64_t)base + offset;

            if (is_64bit) {
                // 64-bit pair
                if (is_load) {
                    // LDP
                    if (addr + 16 <= memory_size) {
                        if (rt1 != 31) regs[rt1] = (int64_t)load64(memory, addr);
                        if (rt2 != 31) regs[rt2] = (int64_t)load64(memory, addr + 8);
                    }
                } else {
                    // STP
                    if (addr + 16 <= memory_size) {
                        int64_t val1 = (rt1 == 31) ? 0 : regs[rt1];
                        int64_t val2 = (rt2 == 31) ? 0 : regs[rt2];
                        store64(memory, addr, (uint64_t)val1);
                        store64(memory, addr + 8, (uint64_t)val2);
                    }
                }
            } else {
                // 32-bit pair
                if (is_load) {
                    // LDP (32-bit)
                    if (addr + 8 <= memory_size) {
                        if (rt1 != 31) regs[rt1] = (int64_t)(int32_t)load32(memory, addr);
                        if (rt2 != 31) regs[rt2] = (int64_t)(int32_t)load32(memory, addr + 4);
                    }
                } else {
                    // STP (32-bit)
                    if (addr + 8 <= memory_size) {
                        int64_t val1 = (rt1 == 31) ? 0 : regs[rt1];
                        int64_t val2 = (rt2 == 31) ? 0 : regs[rt2];
                        store32(memory, addr, (uint32_t)val1);
                        store32(memory, addr + 4, (uint32_t)val2);
                    }
                }
            }

            // Check for writeback (bit 23 set means pre-indexed with writeback)
            // For STP x29, x30, [sp, #-16]!, the '!' indicates writeback
            if ((inst >> 23) & 1) {
                // Writeback: update base register with new address
                if (xn != 31) {
                    regs[xn] = (int64_t)addr;
                } else {
                    regs[31] = (int64_t)addr;  // Update SP
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // BRANCH OPERATIONS
        // ═══════════════════════════════════════════════════════════════════

        // B (unconditional): 0x14xxxxxx or 0x17xxxxxx
        else if ((inst & 0xFC000000) == 0x14000000) {
            int32_t imm26 = inst & 0x3FFFFFF;
            if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;  // Sign extend
            pc = pc + (int64_t)imm26 * 4;
            branch_taken = true;
        }
        // BL (branch with link): 0x94xxxxxx or 0x97xxxxxx
        else if ((inst & 0xFC000000) == 0x94000000) {
            int32_t imm26 = inst & 0x3FFFFFF;
            if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
            regs[30] = (int64_t)(pc + 4);  // Link register
            pc = pc + (int64_t)imm26 * 4;
            branch_taken = true;
        }
        // BR (branch to register): 0xD61F0xxx
        else if ((inst & 0xFFFFFC00) == 0xD61F0000) {
            pc = (uint64_t)regs[rn];
            branch_taken = true;
        }
        // BLR (branch with link to register): 0xD63F0xxx
        else if ((inst & 0xFFFFFC00) == 0xD63F0000) {
            regs[30] = (int64_t)(pc + 4);
            pc = (uint64_t)regs[rn];
            branch_taken = true;
        }
        // RET: 0xD65F03C0 (common form)
        else if ((inst & 0xFFFFFC00) == 0xD65F0000) {
            pc = (uint64_t)regs[rn];
            branch_taken = true;
        }
        // CBZ (64-bit): 0xB4xxxxxx
        else if ((inst & 0xFF000000) == 0xB4000000) {
            uint8_t rt = rd;
            int32_t imm19 = (inst >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            int64_t rt_val = (rt == 31) ? 0 : regs[rt];
            if (rt_val == 0) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }
        // CBNZ (64-bit): 0xB5xxxxxx
        else if ((inst & 0xFF000000) == 0xB5000000) {
            uint8_t rt = rd;
            int32_t imm19 = (inst >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            int64_t rt_val = (rt == 31) ? 0 : regs[rt];
            if (rt_val != 0) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }
        // B.cond (conditional branch): 0x54xxxxxx
        else if ((inst & 0xFF000010) == 0x54000000) {
            uint8_t cond = inst & 0xF;
            int32_t imm19 = (inst >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;

            bool take_branch = false;
            switch (cond) {
                case 0x0: take_branch = (Z > 0.5); break;           // EQ
                case 0x1: take_branch = (Z < 0.5); break;           // NE
                case 0x2: take_branch = (C > 0.5); break;           // CS/HS
                case 0x3: take_branch = (C < 0.5); break;           // CC/LO
                case 0x4: take_branch = (N > 0.5); break;           // MI
                case 0x5: take_branch = (N < 0.5); break;           // PL
                case 0x6: take_branch = (V > 0.5); break;           // VS
                case 0x7: take_branch = (V < 0.5); break;           // VC
                case 0x8: take_branch = (C > 0.5 && Z < 0.5); break; // HI
                case 0x9: take_branch = (C < 0.5 || Z > 0.5); break; // LS
                case 0xA: take_branch = ((N > 0.5) == (V > 0.5)); break; // GE
                case 0xB: take_branch = ((N > 0.5) != (V > 0.5)); break; // LT
                case 0xC: take_branch = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break; // GT
                case 0xD: take_branch = (Z > 0.5) || ((N > 0.5) != (V > 0.5)); break; // LE - fixed operator precedence
                case 0xE: take_branch = true; break;                 // AL
                case 0xF: take_branch = true; break;                 // NV (always)
            }
            if (take_branch) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // BIT MANIPULATION OPERATIONS
        // ═══════════════════════════════════════════════════════════════════

        // CLZ (Count Leading Zeros) - 64-bit: 0xDAC01000
        else if ((inst & 0xFFFFFC00) == 0xDAC01000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
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
            if (rd < 31) regs[rd] = count;
        }
        // CLZ (Count Leading Zeros) - 32-bit: 0x5AC01000
        else if ((inst & 0xFFFFFC00) == 0x5AC01000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            uint32_t val = (uint32_t)rn_val;
            int count = 0;
            if (val == 0) {
                count = 32;
            } else {
                while ((val & 0x80000000U) == 0) {
                    count++;
                    val <<= 1;
                }
            }
            if (rd < 31) regs[rd] = count;
        }
        // RBIT (Reverse Bits) - 64-bit: 0xDAC00000
        else if ((inst & 0xFFFFFC00) == 0xDAC00000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            uint64_t val = (uint64_t)rn_val;
            uint64_t result = 0;
            for (int i = 0; i < 64; i++) {
                result = (result << 1) | (val & 1);
                val >>= 1;
            }
            if (rd < 31) regs[rd] = (int64_t)result;
        }
        // RBIT (Reverse Bits) - 32-bit: 0x5AC00000
        else if ((inst & 0xFFFFFC00) == 0x5AC00000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            uint32_t val = (uint32_t)rn_val;
            uint32_t result = 0;
            for (int i = 0; i < 32; i++) {
                result = (result << 1) | (val & 1);
                val >>= 1;
            }
            if (rd < 31) regs[rd] = (int64_t)result;
        }
        // REV (Reverse Bytes) - 64-bit: 0xDAC00C00
        else if ((inst & 0xFFFFFC00) == 0xDAC00C00) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            uint64_t val = (uint64_t)rn_val;
            uint64_t result = ((val & 0xFF) << 56) |
                              ((val & 0xFF00) << 40) |
                              ((val & 0xFF0000) << 24) |
                              ((val & 0xFF000000) << 8) |
                              ((val >> 8) & 0xFF000000) |
                              ((val >> 24) & 0xFF0000) |
                              ((val >> 40) & 0xFF00) |
                              ((val >> 56) & 0xFF);
            if (rd < 31) regs[rd] = (int64_t)result;
        }
        // REV32 (Reverse Bytes in Words) - 64-bit: 0xDAC00800
        else if ((inst & 0xFFFFFC00) == 0xDAC00800) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            uint64_t val = (uint64_t)rn_val;
            uint32_t lo = (uint32_t)val;
            uint32_t hi = (uint32_t)(val >> 32);
            lo = ((lo & 0xFF) << 24) | ((lo & 0xFF00) << 8) | ((lo >> 8) & 0xFF00) | ((lo >> 24) & 0xFF);
            hi = ((hi & 0xFF) << 24) | ((hi & 0xFF00) << 8) | ((hi >> 8) & 0xFF00) | ((hi >> 24) & 0xFF);
            if (rd < 31) regs[rd] = (int64_t)(((uint64_t)hi << 32) | lo);
        }
        // REV16 (Reverse Bytes in Halfwords) - 64-bit: 0xDAC00400
        else if ((inst & 0xFFFFFC00) == 0xDAC00400) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            uint64_t val = (uint64_t)rn_val;
            uint64_t result = 0;
            for (int i = 0; i < 4; i++) {
                uint16_t hw = (val >> (i * 16)) & 0xFFFF;
                hw = ((hw & 0xFF) << 8) | ((hw >> 8) & 0xFF);
                result |= ((uint64_t)hw << (i * 16));
            }
            if (rd < 31) regs[rd] = (int64_t)result;
        }

        // ═══════════════════════════════════════════════════════════════════
        // CONDITIONAL SELECT OPERATIONS
        // ═══════════════════════════════════════════════════════════════════

        // CSEL (Conditional Select) - 64-bit: 0x9A800000
        else if ((inst & 0xFFE00C00) == 0x9A800000) {
            uint8_t cond = (inst >> 12) & 0xF;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
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
                case 0xD: cond_true = (Z > 0.5) || ((N > 0.5) != (V > 0.5)); break;
                case 0xE: cond_true = true; break;
                case 0xF: cond_true = true; break;
            }
            if (rd < 31) regs[rd] = cond_true ? rn_val : rm_val;
        }
        // CSINC (Conditional Select Increment) - 64-bit: 0x9A800400
        else if ((inst & 0xFFE00C00) == 0x9A800400) {
            uint8_t cond = (inst >> 12) & 0xF;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
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
                case 0xD: cond_true = (Z > 0.5) || ((N > 0.5) != (V > 0.5)); break;
                case 0xE: cond_true = true; break;
                case 0xF: cond_true = true; break;
            }
            if (rd < 31) regs[rd] = cond_true ? rn_val : (rm_val + 1);
        }
        // CSINV (Conditional Select Invert) - 64-bit: 0xDA800000
        else if ((inst & 0xFFE00C00) == 0xDA800000) {
            uint8_t cond = (inst >> 12) & 0xF;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
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
                case 0xD: cond_true = (Z > 0.5) || ((N > 0.5) != (V > 0.5)); break;
                case 0xE: cond_true = true; break;
                case 0xF: cond_true = true; break;
            }
            if (rd < 31) regs[rd] = cond_true ? rn_val : ~rm_val;
        }
        // CSNEG (Conditional Select Negate) - 64-bit: 0xDA800400
        else if ((inst & 0xFFE00C00) == 0xDA800400) {
            uint8_t cond = (inst >> 12) & 0xF;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
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
                case 0xD: cond_true = (Z > 0.5) || ((N > 0.5) != (V > 0.5)); break;
                case 0xE: cond_true = true; break;
                case 0xF: cond_true = true; break;
            }
            if (rd < 31) regs[rd] = cond_true ? rn_val : -rm_val;
        }

        // ═══════════════════════════════════════════════════════════════════
        // DIVISION OPERATIONS
        // ═══════════════════════════════════════════════════════════════════

        // UDIV (Unsigned Divide) - 64-bit: 0x9AC00800
        else if ((inst & 0xFFE0FC00) == 0x9AC00800) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            uint64_t dividend = (uint64_t)rn_val;
            uint64_t divisor = (uint64_t)rm_val;
            if (rd < 31) {
                regs[rd] = (divisor == 0) ? 0 : (int64_t)(dividend / divisor);
            }
        }
        // SDIV (Signed Divide) - 64-bit: 0x9AC00C00
        else if ((inst & 0xFFE0FC00) == 0x9AC00C00) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) {
                regs[rd] = (rm_val == 0) ? 0 : (rn_val / rm_val);
            }
        }
        // UDIV (Unsigned Divide) - 32-bit: 0x1AC00800
        else if ((inst & 0xFFE0FC00) == 0x1AC00800) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            uint32_t dividend = (uint32_t)rn_val;
            uint32_t divisor = (uint32_t)rm_val;
            if (rd < 31) {
                regs[rd] = (divisor == 0) ? 0 : (int64_t)(dividend / divisor);
            }
        }
        // SDIV (Signed Divide) - 32-bit: 0x1AC00C00
        else if ((inst & 0xFFE0FC00) == 0x1AC00C00) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            int32_t dividend = (int32_t)rn_val;
            int32_t divisor = (int32_t)rm_val;
            if (rd < 31) {
                regs[rd] = (divisor == 0) ? 0 : (int64_t)(dividend / divisor);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // TEST AND BRANCH OPERATIONS
        // ═══════════════════════════════════════════════════════════════════

        // TBZ (Test Bit and Branch if Zero): 0x36xxxxxx
        else if ((inst & 0x7F000000) == 0x36000000) {
            uint8_t rt = rd;
            uint8_t bit_pos = ((inst >> 19) & 0x1F) | (((inst >> 31) & 1) << 5);
            int32_t imm14 = (inst >> 5) & 0x3FFF;
            if (imm14 & 0x2000) imm14 |= (int32_t)0xFFFFC000;  // Sign extend
            int64_t rt_val = (rt == 31) ? 0 : regs[rt];
            if (((rt_val >> bit_pos) & 1) == 0) {
                pc = pc + (int64_t)imm14 * 4;
                branch_taken = true;
            }
        }
        // TBNZ (Test Bit and Branch if Not Zero): 0x37xxxxxx
        else if ((inst & 0x7F000000) == 0x37000000) {
            uint8_t rt = rd;
            uint8_t bit_pos = ((inst >> 19) & 0x1F) | (((inst >> 31) & 1) << 5);
            int32_t imm14 = (inst >> 5) & 0x3FFF;
            if (imm14 & 0x2000) imm14 |= (int32_t)0xFFFFC000;  // Sign extend
            int64_t rt_val = (rt == 31) ? 0 : regs[rt];
            if (((rt_val >> bit_pos) & 1) != 0) {
                pc = pc + (int64_t)imm14 * 4;
                branch_taken = true;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // HALFWORD LOAD/STORE
        // ═══════════════════════════════════════════════════════════════════

        // LDRH (Load Halfword unsigned offset): 0x79400000
        else if ((inst & 0xFFC00000) == 0x79400000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + (imm12 << 1);
            if (addr + 2 <= memory_size && rd < 31) {
                regs[rd] = (int64_t)(uint16_t)(memory[addr] | ((uint16_t)memory[addr + 1] << 8));
            }
        }
        // STRH (Store Halfword unsigned offset): 0x79000000
        else if ((inst & 0xFFC00000) == 0x79000000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + (imm12 << 1);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 2 <= memory_size) {
                memory[addr] = (uint8_t)(val & 0xFF);
                memory[addr + 1] = (uint8_t)((val >> 8) & 0xFF);
            }
        }
        // LDRSB (Load Signed Byte) - 64-bit: 0x39800000
        else if ((inst & 0xFFC00000) == 0x39800000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + imm12;
            if (addr < memory_size && rd < 31) {
                regs[rd] = (int64_t)(int8_t)memory[addr];
            }
        }
        // LDRSH (Load Signed Halfword) - 64-bit: 0x79800000
        else if ((inst & 0xFFC00000) == 0x79800000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + (imm12 << 1);
            if (addr + 2 <= memory_size && rd < 31) {
                int16_t val = (int16_t)(memory[addr] | ((uint16_t)memory[addr + 1] << 8));
                regs[rd] = (int64_t)val;
            }
        }
        // LDRSW (Load Signed Word) - 64-bit: 0xB9800000
        else if ((inst & 0xFFC00000) == 0xB9800000) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base + (imm12 << 2);
            if (addr + 4 <= memory_size && rd < 31) {
                int32_t val = load32(memory, addr);
                regs[rd] = (int64_t)val;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // ATOMIC OPERATIONS (for basic atomicity support)
        // ═══════════════════════════════════════════════════════════════════

        // LDXR (Load Exclusive Register) - 64-bit: 0xC85F7C00
        else if ((inst & 0xFFFFFC00) == 0xC85F7C00) {
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base;
            if (addr + 8 <= memory_size && rd < 31) {
                regs[rd] = (int64_t)load64(memory, addr);
            }
        }
        // STXR (Store Exclusive Register) - 64-bit: 0xC8007C00
        else if ((inst & 0xFFE0FC00) == 0xC8007C00) {
            uint8_t rs = (inst >> 16) & 0x1F;  // Status register
            int64_t base = (rn == 31) ? 0 : regs[rn];
            uint64_t addr = (uint64_t)base;
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 8 <= memory_size) {
                store64(memory, addr, (uint64_t)val);
                // Always succeed in this simplified emulation
                if (rs < 31) regs[rs] = 0;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // ADRP (PC-relative Page Address)
        // ═══════════════════════════════════════════════════════════════════

        // ADRP: 0x90xxxxxx (also 0xB0, 0xD0, 0xF0)
        else if ((inst & 0x9F000000) == 0x90000000) {
            int32_t immhi = (inst >> 5) & 0x7FFFF;
            int32_t immlo = (inst >> 29) & 0x3;
            int32_t imm = (immhi << 2) | immlo;
            if (imm & 0x100000) imm |= (int32_t)0xFFE00000;
            int64_t page = (int64_t)(pc & ~0xFFFULL) + ((int64_t)imm << 12);
            if (rd < 31) regs[rd] = page;
        }

        // ═══════════════════════════════════════════════════════════════════
        // OTHER
        // ═══════════════════════════════════════════════════════════════════

        // ADR: 0x10xxxxxx
        else if ((inst & 0x9F000000) == 0x10000000) {
            int32_t immhi = (inst >> 5) & 0x7FFFF;
            int32_t immlo = (inst >> 29) & 0x3;
            int32_t imm = (immhi << 2) | immlo;
            if (imm & 0x100000) imm |= (int32_t)0xFFE00000;  // Sign extend 21-bit
            regs[rd] = (int64_t)(pc + imm);
        }
        // NOP: 0xD503201F
        else if (inst == 0xD503201F) {
            // Do nothing
        }

        // Update PC if not branching
        if (!branch_taken) {
            pc += 4;
        }

        cycles++;
    }
    // End of while loop

    // Write back results
    // CRITICAL: For x31 (SP), preserve the original input value explicitly
    // This works around a potential Metal cache coherency issue with StorageModeShared
    regs[31] = sp_value;
    registers[31] = sp_value;  // Explicit write-back for SP
    for (int i = 0; i < 31; i++) {  // Skip x31, already handled
        registers[i] = regs[i];
    }
    pc_ptr[0] = pc;
    flags[0] = N;
    flags[1] = Z;
    flags[2] = C;
    flags[3] = V;
    *cycles_out = cycles;

    if (*stop_reason == STOP_RUNNING && cycles >= max) {
        *stop_reason = STOP_MAX_CYCLES;
    }
}
"#;

/// Error types for Metal operations
#[derive(Error, Debug)]
pub enum MetalError {
    #[error("No Metal device found")]
    NoDevice,
    #[error("Failed to create command queue")]
    NoCommandQueue,
    #[error("Failed to compile shader: {0}")]
    ShaderCompilationFailed(String),
    #[error("Failed to create pipeline state: {0}")]
    PipelineCreationFailed(String),
    #[error("Failed to create buffer")]
    BufferCreationFailed,
    #[error("Command buffer execution failed")]
    ExecutionFailed,
}

impl From<MetalError> for PyErr {
    fn from(err: MetalError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// Stop reason enum matching Metal shader constants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StopReason {
    Running = 0,
    Halt = 1,
    Syscall = 2,
    MaxCycles = 3,
}

impl From<u8> for StopReason {
    fn from(val: u8) -> Self {
        match val {
            0 => StopReason::Running,
            1 => StopReason::Halt,
            2 => StopReason::Syscall,
            3 => StopReason::MaxCycles,
            _ => StopReason::MaxCycles,
        }
    }
}

/// Execution result from GPU kernel
#[pyclass]
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    #[pyo3(get)]
    pub cycles: u32,
    #[pyo3(get)]
    pub stop_reason: u8,
    #[pyo3(get)]
    pub final_pc: u64,
}

#[pymethods]
impl ExecutionResult {
    #[getter]
    fn stop_reason_name(&self) -> &str {
        match StopReason::from(self.stop_reason) {
            StopReason::Running => "RUNNING",
            StopReason::Halt => "HALT",
            StopReason::Syscall => "SYSCALL",
            StopReason::MaxCycles => "MAX_CYCLES",
        }
    }
}

/// Metal GPU CPU emulator with zero-copy shared memory
/// Note: unsendable because Metal objects are not thread-safe
#[pyclass(unsendable)]
pub struct MetalCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Memory buffers
    memory_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Registers with explicit staging buffer for reliable CPU-GPU sync
    registers_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,      // GPU-side (Managed)
    registers_staging: Retained<ProtocolObject<dyn MTLBuffer>>,    // CPU-side (Shared)

    pc_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,            // GPU-side (Managed)
    pc_staging: Retained<ProtocolObject<dyn MTLBuffer>>,           // CPU-side (Shared)

    flags_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Config buffers
    max_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mem_size_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Output buffers
    cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    stop_reason_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Syscall buffers for GPU-side syscall handling
    syscall_write_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,      // GPU->CPU: write() output
    syscall_read_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,       // CPU->GPU: read() input
    syscall_info_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,       // Syscall metadata [num, fd, buf, count, result, key_state]

    memory_size: usize,
    total_instructions: u64,
}

#[pymethods]
impl MetalCPU {
    /// Create a new Metal CPU emulator
    #[new]
    #[pyo3(signature = (memory_size = 4 * 1024 * 1024))]
    fn new(memory_size: usize) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[MetalCPU] Using device: {:?}", device.name());

        // Create command queue
        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        // Compile shader
        let source = NSString::from_str(METAL_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let function_name = NSString::from_str("cpu_execute");
        let function = library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("Function not found".to_string()))?;

        // Create compute pipeline
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        // Create managed memory buffers with explicit synchronization
        // StorageModeManaged requires explicit CPU-GPU synchronization but is more reliable
        let managed_options = MTLResourceOptions::StorageModeManaged;

        // For memory buffer, use Shared since it's large and accessed frequently by GPU
        let shared_options = MTLResourceOptions::StorageModeShared;

        // Critical buffers that need reliable CPU-GPU sync use Managed mode
        let registers_buffer = device
            .newBufferWithLength_options(32 * std::mem::size_of::<i64>(), managed_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // Create a CPU-side staging buffer for registers
        let registers_staging = device
            .newBufferWithLength_options(32 * std::mem::size_of::<i64>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let memory_buffer = device
            .newBufferWithLength_options(memory_size, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // PC buffer with explicit staging for reliable CPU-GPU sync
        let pc_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u64>(), managed_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // CPU-side staging buffer for PC
        let pc_staging = device
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

        let cycles_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        let stop_reason_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u8>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // Syscall buffers for GPU-side syscall handling
        // write_buffer: GPU writes text here for write() syscall (4KB max)
        let syscall_write_buffer = device
            .newBufferWithLength_options(4096, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // read_buffer: CPU provides input here for read() syscall (256B max)
        let syscall_read_buffer = device
            .newBufferWithLength_options(256, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // info_buffer: Syscall metadata [num, fd, buf, count, result, key_state]
        let syscall_info_buffer = device
            .newBufferWithLength_options(6 * std::mem::size_of::<u64>(), shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // Initialize memory size buffer
        unsafe {
            let ptr = mem_size_buffer.contents().as_ptr() as *mut u32;
            *ptr = memory_size as u32;
        }

        println!("[MetalCPU] Initialized with {} bytes memory", memory_size);
        println!("[MetalCPU] Using StorageModeManaged for registers with explicit sync");

        Ok(MetalCPU {
            device,
            command_queue,
            pipeline,
            memory_buffer,
            registers_buffer,
            registers_staging,
            pc_buffer,
            pc_staging,
            flags_buffer,
            max_cycles_buffer,
            mem_size_buffer,
            cycles_buffer,
            stop_reason_buffer,
            syscall_write_buffer,
            syscall_read_buffer,
            syscall_info_buffer,
            memory_size,
            total_instructions: 0,
        })
    }

    /// Load program bytes into memory
    fn load_program(&self, program: Vec<u8>, address: usize) -> PyResult<()> {
        if address + program.len() > self.memory_size {
            return Err(PyRuntimeError::new_err("Program exceeds memory size"));
        }

        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(program.as_ptr(), ptr.add(address), program.len());
        }

        println!("[MetalCPU] Loaded {} bytes at 0x{:X}", program.len(), address);
        Ok(())
    }

    /// Synchronize registers from staging to GPU buffer (before GPU execution)
    fn sync_registers_to_gpu(&self) {
        unsafe {
            let src = self.registers_staging.contents().as_ptr() as *const i64;
            let dst = self.registers_buffer.contents().as_ptr() as *mut i64;
            std::ptr::copy_nonoverlapping(src, dst, 32);
        }
    }

    /// Synchronize PC from staging to GPU buffer (before GPU execution)
    fn sync_pc_to_gpu(&self) {
        unsafe {
            let src = self.pc_staging.contents().as_ptr() as *const u64;
            let dst = self.pc_buffer.contents().as_ptr() as *mut u64;
            std::ptr::copy_nonoverlapping(src, dst, 1);
        }
    }

    /// Synchronize registers from GPU to staging buffer (after GPU execution)
    fn sync_registers_from_gpu(&self) {
        unsafe {
            let src = self.registers_buffer.contents().as_ptr() as *const i64;
            let dst = self.registers_staging.contents().as_ptr() as *mut i64;
            std::ptr::copy_nonoverlapping(src, dst, 32);
        }
    }

    /// Synchronize PC from GPU to staging buffer (after GPU execution)
    fn sync_pc_from_gpu(&self) {
        unsafe {
            let src = self.pc_buffer.contents().as_ptr() as *const u64;
            let dst = self.pc_staging.contents().as_ptr() as *mut u64;
            std::ptr::copy_nonoverlapping(src, dst, 1);
        }
    }

    /// Set program counter (writes to staging buffer)
    fn set_pc(&self, pc: u64) {
        unsafe {
            let ptr = self.pc_staging.contents().as_ptr() as *mut u64;
            *ptr = pc;
        }
    }

    /// Get program counter (reads from staging buffer)
    fn get_pc(&self) -> u64 {
        unsafe {
            let ptr = self.pc_staging.contents().as_ptr() as *const u64;
            *ptr
        }
    }

    /// Set register value (writes to staging buffer)
    fn set_register(&self, reg: usize, value: i64) {
        if reg >= 32 {
            return;
        }
        unsafe {
            let ptr = self.registers_staging.contents().as_ptr() as *mut i64;
            *ptr.add(reg) = value;
        }
    }

    /// Get register value (reads from staging buffer)
    fn get_register(&self, reg: usize) -> i64 {
        if reg >= 32 {
            return 0;
        }
        unsafe {
            let ptr = self.registers_staging.contents().as_ptr() as *const i64;
            *ptr.add(reg)
        }
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

    /// Read 64-bit value from memory
    fn read_memory_64(&self, address: usize) -> PyResult<u64> {
        if address + 8 > self.memory_size {
            return Err(PyRuntimeError::new_err("Read exceeds memory bounds"));
        }

        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *const u8;
            let mut bytes = [0u8; 8];
            std::ptr::copy_nonoverlapping(ptr.add(address), bytes.as_mut_ptr(), 8);
            Ok(u64::from_le_bytes(bytes))
        }
    }

    /// Execute ARM64 instructions on GPU
    #[pyo3(signature = (max_cycles = 100000))]
    fn execute(&mut self, max_cycles: u32) -> PyResult<ExecutionResult> {
        // CRITICAL: Sync registers from staging (CPU) to GPU buffer before execution
        self.sync_registers_to_gpu();

        // CRITICAL: Sync PC from staging to GPU buffer before execution
        self.sync_pc_to_gpu();

        // Set max_cycles
        unsafe {
            let ptr = self.max_cycles_buffer.contents().as_ptr() as *mut u32;
            *ptr = max_cycles;
        }

        // Create command buffer
        let command_buffer = self
            .command_queue
            .commandBuffer()
            .ok_or(MetalError::ExecutionFailed)?;

        // Create compute encoder
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ExecutionFailed)?;

        // Set pipeline and buffers
        encoder.setComputePipelineState(&self.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.memory_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.registers_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.pc_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.flags_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.max_cycles_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&self.mem_size_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.cycles_buffer), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&self.stop_reason_buffer), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&self.syscall_write_buffer), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(&self.syscall_read_buffer), 0, 9);
            encoder.setBuffer_offset_atIndex(Some(&self.syscall_info_buffer), 0, 10);
        }

        // Dispatch single thread (CPU emulation is inherently sequential)
        let threads_per_grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        unsafe {
            encoder.dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
        }
        encoder.endEncoding();

        // Execute and wait
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // CRITICAL: Sync registers from GPU back to staging (CPU) after execution
        self.sync_registers_from_gpu();

        // CRITICAL: Sync PC from GPU back to staging (CPU) after execution
        self.sync_pc_from_gpu();

        // Read results (from staging buffer for CPU access)
        let (cycles, stop_reason, final_pc) = unsafe {
            let cycles = *(self.cycles_buffer.contents().as_ptr() as *const u32);
            let stop_reason = *(self.stop_reason_buffer.contents().as_ptr() as *const u8);
            let final_pc = *(self.pc_staging.contents().as_ptr() as *const u64);
            (cycles, stop_reason, final_pc)
        };

        self.total_instructions += cycles as u64;

        Ok(ExecutionResult {
            cycles,
            stop_reason,
            final_pc,
        })
    }

    /// Reset CPU state
    fn reset(&mut self) {
        // Zero out both GPU and staging register buffers
        unsafe {
            let gpu_ptr = self.registers_buffer.contents().as_ptr() as *mut u8;
            let staging_ptr = self.registers_staging.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(gpu_ptr, 0, 32 * std::mem::size_of::<i64>());
            std::ptr::write_bytes(staging_ptr, 0, 32 * std::mem::size_of::<i64>());
        }

        // Reset PC
        self.set_pc(0);

        // Reset flags
        unsafe {
            let ptr = self.flags_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 4 * std::mem::size_of::<f32>());
        }

        self.total_instructions = 0;
    }

    /// Get total instructions executed
    #[getter]
    fn total_instructions(&self) -> u64 {
        self.total_instructions
    }

    /// Get memory size
    #[getter]
    fn memory_size(&self) -> usize {
        self.memory_size
    }

    /// Get syscall write output (data written by GPU via write() syscall)
    fn get_syscall_write_output(&self) -> PyResult<Vec<u8>> {
        unsafe {
            let ptr = self.syscall_write_buffer.contents().as_ptr() as *const u8;
            // Read until null terminator or max 4096 bytes
            let mut data = Vec::new();
            for i in 0..4096 {
                let byte = *ptr.add(i);
                if byte == 0 {
                    break;
                }
                data.push(byte);
            }
            Ok(data)
        }
    }

    /// Set syscall read input (data for GPU to read via read() syscall)
    fn set_syscall_read_input(&self, data: &[u8]) -> PyResult<()> {
        unsafe {
            let ptr = self.syscall_read_buffer.contents().as_ptr() as *mut u8;
            let len = data.len().min(256);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
            // Null terminate
            if len < 256 {
                *ptr.add(len) = 0;
            }
        }
        Ok(())
    }

    /// Get syscall info after execution [syscall_num, fd, buf, count, result, key_state]
    fn get_syscall_info(&self) -> PyResult<Vec<u64>> {
        unsafe {
            let ptr = self.syscall_info_buffer.contents().as_ptr() as *const u64;
            Ok(vec![
                *ptr,         // syscall_num
                *ptr.add(1),  // fd
                *ptr.add(2),  // buf
                *ptr.add(3),  // count
                *ptr.add(4),  // result
                *ptr.add(5),  // key_state
            ])
        }
    }

    /// Set key state for DOOM input (1=w, 2=s, 3=a, 4=d, 5=q, 0=none)
    fn set_key_state(&self, key_state: u8) {
        unsafe {
            let ptr = self.syscall_info_buffer.contents().as_ptr() as *mut u64;
            *ptr.add(5) = key_state as u64;
        }
    }
}

/// Get the default Metal device
pub fn get_default_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    objc2_metal::MTLCreateSystemDefaultDevice()
}

/// Python module
#[pymodule]
fn kvrm_metal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MetalCPU>()?;
    m.add_class::<ExecutionResult>()?;
    continuous::register_continuous(m)?;
    async_gpu::register_async(m)?;
    parallel::register_parallel(m)?;
    multi_kernel::register_multi_kernel(m)?;
    neural_dispatch::register_neural(m)?;
    neural_weights::register_weights(m)?;
    pure_gpu::register_pure_gpu(m)?;
    optimized::register_optimized(m)?;
    ultra_optimized::register_hyper_optimized(m)?;
    fusion::register_fusion(m)?;
    bb_cache::register_bb_cache(m)?;
    ultra::register_ultra(m)?;
    trace_jit::register_trace_jit(m)?;
    ooo_exec::register_ooo_exec(m)?;
    neural_ooo::register_neural_ooo(m)?;
    neural_hybrid::register_neural_hybrid(m)?;
    differentiable_ooo::register_diff_ooo(m)?;
    jit_compiler::register_jit_compiler(m)?;
    diff_jit::register_diff_jit(m)?;
    unified_diff_cpu::register_unified_diff_cpu(m)?;
    unified_test_kernel::register_minimal_test_cpu(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let device = get_default_device();
        assert!(device.is_some(), "Should have a Metal device");
    }
}
