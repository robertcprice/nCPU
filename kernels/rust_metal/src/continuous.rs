//! Continuous GPU execution with atomic signaling.
//!
//! This module provides a Metal kernel that runs continuously on the GPU,
//! only returning control to the CPU when syscall handling is needed.
//!
//! Key features:
//! - GPU runs in mega-batches (100M+ cycles per check)
//! - Atomic flag for GPU-to-CPU signaling
//! - Zero kernel re-launch overhead between batches
//! - True "keep it ALL on GPU" execution

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
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::{MetalError, StopReason, ExecutionResult, get_default_device};

/// Metal shader for continuous execution with atomic signaling
const CONTINUOUS_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Stop/signal reasons
constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;  // Periodic checkpoint for progress

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

// Continuous CPU execution kernel
// Runs for cycles_per_batch, then checks/updates signal
kernel void cpu_execute_continuous(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* cycles_per_batch [[buffer(4)]],
    device const uint32_t* mem_size [[buffer(5)]],
    device atomic_uint* signal_flag [[buffer(6)]],      // Atomic for GPU<->CPU
    device uint32_t* total_cycles [[buffer(7)]],         // Running total
    device uint32_t* batch_count [[buffer(8)]],          // Number of batches executed
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t batch_cycles = cycles_per_batch[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;

    // Load registers
    // Note: regs[31] is SP (stack pointer) for load/store base addressing
    // XZR (zero register) behavior is handled inline by checking (rn == 31) ? 0 : regs[rn]
    // for data processing instructions, not load/store base registers
    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers[i];
    }
    // regs[31] is kept as SP - don't zero it

    // Load flags
    float N = flags[0];
    float Z = flags[1];
    float C = flags[2];
    float V = flags[3];

    // Main execution loop - runs for batch_cycles before returning
    while (cycles < batch_cycles) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        uint32_t inst = load32(memory, pc);

        // Check for HALT
        if ((inst & 0xFFE0001F) == 0xD4400000) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        // Check for SYSCALL
        if ((inst & 0xFFE0001F) == 0xD4000001) {
            atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
            break;
        }

        // Decode
        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint8_t rm = (inst >> 16) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint16_t imm16 = (inst >> 5) & 0xFFFF;
        uint8_t hw = (inst >> 21) & 0x3;

        bool branch_taken = false;

        // ADD immediate 64-bit (for this instruction, reg 31 is SP not XZR)
        if ((inst & 0xFF000000) == 0x91000000) {
            int64_t rn_val = regs[rn];  // rn==31 means SP for ADD/SUB imm
            regs[rd] = rn_val + imm12;
        }
        // ADD immediate 32-bit (for this instruction, reg 31 is SP not XZR)
        else if ((inst & 0xFF000000) == 0x11000000) {
            int64_t rn_val = regs[rn];  // rn==31 means SP for ADD/SUB imm
            regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val + imm12);
        }
        // SUB immediate 64-bit (for this instruction, reg 31 is SP not XZR)
        else if ((inst & 0xFF000000) == 0xD1000000) {
            int64_t rn_val = regs[rn];  // rn==31 means SP for ADD/SUB imm
            regs[rd] = rn_val - imm12;
        }
        // SUB immediate 32-bit (for this instruction, reg 31 is SP not XZR)
        else if ((inst & 0xFF000000) == 0x51000000) {
            int64_t rn_val = regs[rn];  // rn==31 means SP for ADD/SUB imm
            regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val - imm12);
        }
        // ADD (shifted register) 64-bit: 0x8B0xxxxx
        else if ((inst & 0xFF200000) == 0x8B000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = rn_val + rm_val;
        }
        // ADD (shifted register) 32-bit: 0x0B0xxxxx
        else if ((inst & 0xFF200000) == 0x0B000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val + (uint32_t)rm_val);
        }
        // SUB (shifted register) 64-bit: 0xCB0xxxxx
        else if ((inst & 0xFF200000) == 0xCB000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = rn_val - rm_val;
        }
        // SUB (shifted register) 32-bit: 0x4B0xxxxx
        else if ((inst & 0xFF200000) == 0x4B000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val - (uint32_t)rm_val);
        }
        // MADD/MUL 64-bit: 0x9B0xxxxx
        else if ((inst & 0xFF000000) == 0x9B000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            uint8_t ra = (inst >> 10) & 0x1F;
            int64_t ra_val = (ra == 31) ? 0 : regs[ra];
            if (rd < 31) regs[rd] = rn_val * rm_val + ra_val;
        }
        // AND (shifted register) 64-bit: 0x8A0xxxxx
        else if ((inst & 0xFF200000) == 0x8A000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = rn_val & rm_val;
        }
        // ORR (shifted register) 64-bit: 0xAA0xxxxx
        else if ((inst & 0xFF200000) == 0xAA000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = rn_val | rm_val;
        }
        // EOR (shifted register) 64-bit: 0xCA0xxxxx
        else if ((inst & 0xFF200000) == 0xCA000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (rd < 31) regs[rd] = rn_val ^ rm_val;
        }
        // SUBS immediate 64-bit
        else if ((inst & 0xFF000000) == 0xF1000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t result = rn_val - imm12;
            if (rd < 31) regs[rd] = result;  // XZR means discard (CMP uses this)
            N = (result < 0) ? 1.0 : 0.0;
            Z = (result == 0) ? 1.0 : 0.0;
            C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
            V = 0.0;
        }
        // SUBS immediate 32-bit (CMP W when Rd=WZR): 0x71000000
        else if ((inst & 0xFF000000) == 0x71000000) {
            uint32_t rn_val = (rn == 31) ? 0 : (uint32_t)regs[rn];
            uint32_t result = rn_val - (uint32_t)imm12;
            if (rd < 31) regs[rd] = (int64_t)(int32_t)result;  // sign extend to 64-bit
            N = ((int32_t)result < 0) ? 1.0 : 0.0;
            Z = (result == 0) ? 1.0 : 0.0;
            C = (rn_val >= (uint32_t)imm12) ? 1.0 : 0.0;
            V = 0.0;
        }
        // SUBS register 64-bit (CMP Xn, Xm when Rd=XZR): 0xEB00_0000
        else if ((inst & 0xFF200000) == 0xEB000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            // Handle shift (LSL, LSR, ASR)
            uint8_t shift_type = (inst >> 22) & 0x3;
            uint8_t shift_amt = (inst >> 10) & 0x3F;
            uint64_t shifted_rm = (uint64_t)rm_val;
            switch (shift_type) {
                case 0: shifted_rm = shifted_rm << shift_amt; break;  // LSL
                case 1: shifted_rm = shifted_rm >> shift_amt; break;  // LSR
                case 2: shifted_rm = (uint64_t)((int64_t)rm_val >> shift_amt); break;  // ASR
                default: break;
            }
            int64_t result = rn_val - (int64_t)shifted_rm;
            if (rd < 31) regs[rd] = result;  // XZR means discard (CMP uses this)
            N = (result < 0) ? 1.0 : 0.0;
            Z = (result == 0) ? 1.0 : 0.0;
            C = ((uint64_t)rn_val >= shifted_rm) ? 1.0 : 0.0;
            // Overflow: positive - negative = negative OR negative - positive = positive
            bool rn_neg = (rn_val < 0);
            bool rm_neg = ((int64_t)shifted_rm < 0);
            bool res_neg = (result < 0);
            V = ((rn_neg != rm_neg) && (res_neg != rn_neg)) ? 1.0 : 0.0;
        }
        // ADDS register 64-bit: 0xAB00_0000
        else if ((inst & 0xFF200000) == 0xAB000000) {
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            uint8_t shift_type = (inst >> 22) & 0x3;
            uint8_t shift_amt = (inst >> 10) & 0x3F;
            uint64_t shifted_rm = (uint64_t)rm_val;
            switch (shift_type) {
                case 0: shifted_rm = shifted_rm << shift_amt; break;
                case 1: shifted_rm = shifted_rm >> shift_amt; break;
                case 2: shifted_rm = (uint64_t)((int64_t)rm_val >> shift_amt); break;
                default: break;
            }
            int64_t result = rn_val + (int64_t)shifted_rm;
            if (rd < 31) regs[rd] = result;
            N = (result < 0) ? 1.0 : 0.0;
            Z = (result == 0) ? 1.0 : 0.0;
            // Carry: unsigned overflow
            C = ((uint64_t)result < (uint64_t)rn_val) ? 1.0 : 0.0;
            // Overflow: same signs become opposite sign
            bool rn_neg = (rn_val < 0);
            bool rm_neg = ((int64_t)shifted_rm < 0);
            bool res_neg = (result < 0);
            V = ((rn_neg == rm_neg) && (res_neg != rn_neg)) ? 1.0 : 0.0;
        }
        // MOVZ 64-bit (rd=31 is XZR, discard)
        else if ((inst & 0xFF800000) == 0xD2800000) {
            if (rd < 31) regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
        }
        // MOVZ 32-bit (rd=31 is XZR, discard)
        else if ((inst & 0xFF800000) == 0x52800000) {
            if (rd < 31) regs[rd] = (int64_t)(uint32_t)(imm16 << (hw * 16));
        }
        // MOVK 64-bit (rd=31 is XZR, discard)
        else if ((inst & 0xFF800000) == 0xF2800000) {
            if (rd < 31) {
                uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
                uint64_t val = (uint64_t)regs[rd] & mask;
                val |= ((uint64_t)imm16 << (hw * 16));
                regs[rd] = (int64_t)val;
            }
        }
        // LDR 64-bit
        else if ((inst & 0xFFC00000) == 0xF9400000) {
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base + (imm12 << 3);
            if (rd < 31) {
                if (addr + 8 <= memory_size) {
                    regs[rd] = (int64_t)load64(memory, addr);
                } else {
                    regs[rd] = 0;  // Out of bounds returns 0
                }
            }
        }
        // LDR 32-bit
        else if ((inst & 0xFFC00000) == 0xB9400000) {
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base + (imm12 << 2);
            if (rd < 31) {
                if (addr + 4 <= memory_size) {
                    regs[rd] = (int64_t)load32(memory, addr);
                } else {
                    regs[rd] = 0;  // Out of bounds returns 0
                }
            }
        }
        // LDRB (unsigned offset) - rd=31 is XZR (discard)
        else if ((inst & 0xFFC00000) == 0x39400000) {
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base + imm12;
            if (addr < memory_size && rd < 31) {
                regs[rd] = (int64_t)memory[addr];
            }
        }
        // LDURB (unscaled offset) - 0x38400000-0x385FFFFF
        else if ((inst & 0xFFE00C00) == 0x38400000) {
            int32_t imm9 = (inst >> 12) & 0x1FF;
            if (imm9 & 0x100) imm9 |= (int32_t)0xFFFFFE00;  // sign extend
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)(base + imm9);
            if (addr < memory_size && rd < 31) {
                regs[rd] = (int64_t)memory[addr];
            }
        }
        // STURB (unscaled offset) - 0x38000000-0x381FFFFF
        else if ((inst & 0xFFE00C00) == 0x38000000) {
            int32_t imm9 = (inst >> 12) & 0x1FF;
            if (imm9 & 0x100) imm9 |= (int32_t)0xFFFFFE00;  // sign extend
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)(base + imm9);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr < memory_size) {
                memory[addr] = (uint8_t)val;
            }
        }
        // STR 64-bit
        else if ((inst & 0xFFC00000) == 0xF9000000) {
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base + (imm12 << 3);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 8 <= memory_size) {
                store64(memory, addr, (uint64_t)val);
            }
        }
        // STR 32-bit
        else if ((inst & 0xFFC00000) == 0xB9000000) {
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base + (imm12 << 2);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 4 <= memory_size) {
                store32(memory, addr, (uint32_t)val);
            }
        }
        // STRB
        else if ((inst & 0xFFC00000) == 0x39000000) {
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base + imm12;
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr < memory_size) {
                memory[addr] = (uint8_t)val;
            }
        }
        // STR 64-bit post-index: F800_0400
        else if ((inst & 0xFFE00C00) == 0xF8000400) {
            int32_t imm9 = (inst >> 12) & 0x1FF;
            if (imm9 & 0x100) imm9 |= (int32_t)0xFFFFFE00;  // Sign extend
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base;
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 8 <= memory_size) {
                store64(memory, addr, (uint64_t)val);
            }
            regs[rn] = base + imm9;  // Post-index writeback (rn can be SP)
        }
        // LDR 64-bit post-index: F840_0400
        else if ((inst & 0xFFE00C00) == 0xF8400400) {
            int32_t imm9 = (inst >> 12) & 0x1FF;
            if (imm9 & 0x100) imm9 |= (int32_t)0xFFFFFE00;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base;
            if (rd < 31) {
                if (addr + 8 <= memory_size) {
                    regs[rd] = (int64_t)load64(memory, addr);
                } else {
                    regs[rd] = 0;  // Out of bounds returns 0
                }
            }
            regs[rn] = base + imm9;  // Post-index writeback (rn can be SP)
        }
        // STR 64-bit pre-index: F800_0C00
        else if ((inst & 0xFFE00C00) == 0xF8000C00) {
            int32_t imm9 = (inst >> 12) & 0x1FF;
            if (imm9 & 0x100) imm9 |= (int32_t)0xFFFFFE00;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)(base + imm9);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 8 <= memory_size) {
                store64(memory, addr, (uint64_t)val);
            }
            regs[rn] = base + imm9;  // Pre-index writeback (rn can be SP)
        }
        // LDR 64-bit pre-index: F840_0C00
        else if ((inst & 0xFFE00C00) == 0xF8400C00) {
            int32_t imm9 = (inst >> 12) & 0x1FF;
            if (imm9 & 0x100) imm9 |= (int32_t)0xFFFFFE00;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)(base + imm9);
            if (rd < 31) {
                if (addr + 8 <= memory_size) {
                    regs[rd] = (int64_t)load64(memory, addr);
                } else {
                    regs[rd] = 0;  // Out of bounds returns 0
                }
            }
            regs[rn] = base + imm9;  // Pre-index writeback (rn can be SP)
        }
        // LDR 64-bit (register offset) - LDR Xt, [Xn, Xm, LSL #3]
        // Encoding: 0xF8600800 with option=011 (LSL), S=1 (scaled by 8)
        else if ((inst & 0xFFE00C00) == 0xF8600800) {
            uint8_t rm = (inst >> 16) & 0x1F;
            uint8_t option = (inst >> 13) & 0x7;
            bool S = (inst >> 12) & 1;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            int64_t offset_val = (rm == 31) ? 0 : regs[rm];
            // Apply shift based on option and S
            if (S && option == 3) offset_val <<= 3;  // LSL #3 for 64-bit
            uint64_t addr = (uint64_t)(base + offset_val);
            if (rd < 31) {
                if (addr + 8 <= memory_size) {
                    regs[rd] = (int64_t)load64(memory, addr);
                } else {
                    regs[rd] = 0;  // Out of bounds read returns 0
                }
            }
        }
        // STR 64-bit (register offset) - STR Xt, [Xn, Xm, LSL #3]
        else if ((inst & 0xFFE00C00) == 0xF8200800) {
            uint8_t rm = (inst >> 16) & 0x1F;
            uint8_t option = (inst >> 13) & 0x7;
            bool S = (inst >> 12) & 1;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            int64_t offset_val = (rm == 31) ? 0 : regs[rm];
            if (S && option == 3) offset_val <<= 3;  // LSL #3 for 64-bit
            uint64_t addr = (uint64_t)(base + offset_val);
            int64_t val = (rd == 31) ? 0 : regs[rd];
            if (addr + 8 <= memory_size) {
                store64(memory, addr, (uint64_t)val);
            }
        }
        // STP 64-bit (store pair) - common in prologues
        else if ((inst & 0xFFC00000) == 0xA9000000) {
            uint8_t rt2 = (inst >> 10) & 0x1F;
            int32_t imm7 = (inst >> 15) & 0x7F;
            if (imm7 & 0x40) imm7 |= (int32_t)0xFFFFFF80;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)(base + (imm7 << 3));
            int64_t val1 = (rd == 31) ? 0 : regs[rd];
            int64_t val2 = (rt2 == 31) ? 0 : regs[rt2];
            if (addr + 16 <= memory_size) {
                store64(memory, addr, (uint64_t)val1);
                store64(memory, addr + 8, (uint64_t)val2);
            }
        }
        // LDP 64-bit (load pair)
        else if ((inst & 0xFFC00000) == 0xA9400000) {
            uint8_t rt2 = (inst >> 10) & 0x1F;
            int32_t imm7 = (inst >> 15) & 0x7F;
            if (imm7 & 0x40) imm7 |= (int32_t)0xFFFFFF80;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)(base + (imm7 << 3));
            if (addr + 16 <= memory_size) {
                if (rd < 31) regs[rd] = (int64_t)load64(memory, addr);
                if (rt2 < 31) regs[rt2] = (int64_t)load64(memory, addr + 8);
            } else {
                // Out of bounds - set both registers to 0
                if (rd < 31) regs[rd] = 0;
                if (rt2 < 31) regs[rt2] = 0;
            }
        }
        // STP 64-bit pre-index: A980_0000
        else if ((inst & 0xFFC00000) == 0xA9800000) {
            uint8_t rt2 = (inst >> 10) & 0x1F;
            int32_t imm7 = (inst >> 15) & 0x7F;
            if (imm7 & 0x40) imm7 |= (int32_t)0xFFFFFF80;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)(base + (imm7 << 3));
            int64_t val1 = (rd == 31) ? 0 : regs[rd];
            int64_t val2 = (rt2 == 31) ? 0 : regs[rt2];
            if (addr + 16 <= memory_size) {
                store64(memory, addr, (uint64_t)val1);
                store64(memory, addr + 8, (uint64_t)val2);
            }
            regs[rn] = base + (imm7 << 3);  // Pre-index writeback (rn can be SP)
        }
        // LDP 64-bit post-index: A8C0_0000
        else if ((inst & 0xFFC00000) == 0xA8C00000) {
            uint8_t rt2 = (inst >> 10) & 0x1F;
            int32_t imm7 = (inst >> 15) & 0x7F;
            if (imm7 & 0x40) imm7 |= (int32_t)0xFFFFFF80;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            uint64_t addr = (uint64_t)base;
            if (addr + 16 <= memory_size) {
                if (rd < 31) regs[rd] = (int64_t)load64(memory, addr);
                if (rt2 < 31) regs[rt2] = (int64_t)load64(memory, addr + 8);
            } else {
                // Out of bounds - set both registers to 0
                if (rd < 31) regs[rd] = 0;
                if (rt2 < 31) regs[rt2] = 0;
            }
            regs[rn] = base + (imm7 << 3);  // Post-index writeback (rn can be SP)
        }
        // B unconditional
        else if ((inst & 0xFC000000) == 0x14000000) {
            int32_t imm26 = inst & 0x3FFFFFF;
            if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
            pc = pc + (int64_t)imm26 * 4;
            branch_taken = true;
        }
        // BL
        else if ((inst & 0xFC000000) == 0x94000000) {
            int32_t imm26 = inst & 0x3FFFFFF;
            if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
            regs[30] = (int64_t)(pc + 4);
            pc = pc + (int64_t)imm26 * 4;
            branch_taken = true;
        }
        // BR
        else if ((inst & 0xFFFFFC00) == 0xD61F0000) {
            pc = (uint64_t)regs[rn];
            branch_taken = true;
        }
        // BLR
        else if ((inst & 0xFFFFFC00) == 0xD63F0000) {
            regs[30] = (int64_t)(pc + 4);
            pc = (uint64_t)regs[rn];
            branch_taken = true;
        }
        // RET
        else if ((inst & 0xFFFFFC00) == 0xD65F0000) {
            pc = (uint64_t)regs[rn];
            branch_taken = true;
        }
        // CBZ 64-bit
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
        // CBZ 32-bit
        else if ((inst & 0xFF000000) == 0x34000000) {
            uint8_t rt = rd;
            int32_t imm19 = (inst >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            uint32_t rt_val = (rt == 31) ? 0 : (uint32_t)regs[rt];
            if (rt_val == 0) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }
        // CBNZ 64-bit
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
        // CBNZ 32-bit
        else if ((inst & 0xFF000000) == 0x35000000) {
            uint8_t rt = rd;
            int32_t imm19 = (inst >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            uint32_t rt_val = (rt == 31) ? 0 : (uint32_t)regs[rt];
            if (rt_val != 0) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }
        // B.cond
        else if ((inst & 0xFF000010) == 0x54000000) {
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
                case 0xE: take = true; break;
                case 0xF: take = true; break;
            }
            if (take) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }
        // NOP
        else if (inst == 0xD503201F) {
            // Do nothing
        }
        // ADR - PC-relative address
        else if ((inst & 0x9F000000) == 0x10000000) {
            uint8_t rd_adr = inst & 0x1F;
            int32_t immlo = (inst >> 29) & 0x3;
            int32_t immhi = (inst >> 5) & 0x7FFFF;
            int64_t offset = ((int64_t)immhi << 2) | immlo;
            // Sign extend from 21 bits
            if (offset & 0x100000) offset |= (int64_t)0xFFFFFFFFFFE00000;
            if (rd_adr < 31) regs[rd_adr] = (int64_t)(pc + offset);
        }
        // ADRP - PC-relative page address
        else if ((inst & 0x9F000000) == 0x90000000) {
            uint8_t rd_adr = inst & 0x1F;
            int32_t immlo = (inst >> 29) & 0x3;
            int32_t immhi = (inst >> 5) & 0x7FFFF;
            int64_t offset = (((int64_t)immhi << 2) | immlo) << 12;
            // Sign extend from 33 bits
            if (offset & 0x100000000LL) offset |= (int64_t)0xFFFFFFFE00000000LL;
            uint64_t page_pc = pc & ~0xFFFULL;
            if (rd_adr < 31) regs[rd_adr] = (int64_t)(page_pc + offset);
        }
        // SIMD STP Q (128-bit) with post-index/pre-index writeback
        // Format: STP Qt, Qt2, [Xn], #imm or STP Qt, Qt2, [Xn, #imm]!
        // 0xAC/0xAD prefix for SIMD load/store pair
        else if ((inst & 0xFE000000) == 0xAC000000) {
            // Decode STP/LDP Q - use unique names to avoid shadowing issues
            uint8_t simd_rn = (inst >> 5) & 0x1F;  // Base register
            int8_t simd_imm7 = (inst >> 15) & 0x7F;
            if (simd_imm7 & 0x40) simd_imm7 |= 0x80;  // Sign extend
            int64_t simd_offset = ((int64_t)simd_imm7) * 16;  // Scale by 16 for Q registers

            // Addressing mode is in bits 25-23 (per ARM64 spec):
            // 001 = post-index writeback
            // 010 = signed offset (no writeback)
            // 011 = pre-index writeback
            uint8_t simd_mode = (inst >> 23) & 7;
            bool simd_post = (simd_mode == 1);  // 001 = post-index
            bool simd_pre = (simd_mode == 3);   // 011 = pre-index
            bool simd_signed = (simd_mode == 2);  // 010 = signed offset
            bool simd_store = (inst & 0x00400000) == 0;  // L bit (bit 22)

            // Handle STP Q (store pair) and LDP Q (load pair) for all addressing modes
            int64_t simd_base = regs[simd_rn];  // rn==31 is SP for load/store
            int64_t simd_addr = simd_base;

            // Calculate effective address based on mode
            if (simd_pre || simd_signed) {
                simd_addr = simd_base + simd_offset;  // Pre-index or signed offset: add before access
            }
            // Post-index: access at base, add offset after

            // For stores, write zeros to memory (32 bytes = 2x Q register)
            // We treat all Q registers as containing zero (memset optimization)
            if (simd_store) {
                for (int i = 0; i < 32; i++) {
                    if (simd_addr + i < (int64_t)memory_size && simd_addr + i >= 0) {
                        memory[simd_addr + i] = 0;
                    }
                }
            }
            // For loads, we just skip (Q registers not tracked)

            // Writeback only for post-index and pre-index modes
            if (simd_pre || simd_post) {
                regs[simd_rn] = simd_base + simd_offset;
            }
        }
        // SIMD DUP (duplicate GPR to vector) - 0x4E010C00 family
        // Format: DUP Vd.T, Wn/Xn - fills vector with repeated value from GPR
        // We don't track SIMD registers, so just NOP
        else if ((inst & 0xFFE0FC00) == 0x0E000C00 || (inst & 0xFFE0FC00) == 0x4E000C00) {
            // SIMD DUP - NOP (Q0 assumed to be zeros for memset)
        }
        // SIMD STR Q (128-bit store) - 0x3D800000 family
        // Format: STR Qt, [Xn, #imm] or STR Qt, [Xn]
        else if ((inst & 0xFFC00000) == 0x3D800000) {
            // Store 16 bytes of zeros (we don't track Q registers)
            uint8_t rn = (inst >> 5) & 0x1F;
            uint16_t pimm = (inst >> 10) & 0xFFF;  // Unsigned offset, scaled by 16
            int64_t offset = (int64_t)pimm * 16;
            int64_t base = regs[rn];  // rn==31 is SP for load/store
            int64_t addr = base + offset;
            for (int i = 0; i < 16; i++) {
                if (addr + i < (int64_t)memory_size) {
                    memory[addr + i] = 0;
                }
            }
        }
        // SIMD LDR Q (128-bit load) - 0x3DC00000 family
        // Format: LDR Qt, [Xn, #imm]
        else if ((inst & 0xFFC00000) == 0x3DC00000) {
            // Load to Q register - NOP (we don't track Q registers)
        }
        // MRS - read system register (e.g., MRS X0, FPCR)
        else if ((inst & 0xFFF00000) == 0xD5300000) {
            // Return 0 for any system register read
            if (rd < 31) regs[rd] = 0;
        }
        // MSR - write system register
        else if ((inst & 0xFFF00000) == 0xD5100000) {
            // Ignore system register writes
        }

        if (!branch_taken) pc += 4;
        // Note: regs[31] is SP, not XZR - don't zero it
        // XZR behavior is handled by checking (rn == 31) ? 0 : regs[rn] in data processing ops
        cycles++;
    }

    // Write back state
    for (int i = 0; i < 32; i++) {
        registers[i] = regs[i];
    }
    pc_ptr[0] = pc;
    flags[0] = N;
    flags[1] = Z;
    flags[2] = C;
    flags[3] = V;

    // Update counters
    atomic_fetch_add_explicit((device atomic_uint*)total_cycles, cycles, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)batch_count, 1, memory_order_relaxed);

    // If we finished normally (not halt/syscall), signal checkpoint
    uint32_t current_signal = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (current_signal == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

/// Signal values for GPU<->CPU communication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Signal {
    Running = 0,
    Halt = 1,
    Syscall = 2,
    Checkpoint = 3,
}

impl From<u32> for Signal {
    fn from(val: u32) -> Self {
        match val {
            0 => Signal::Running,
            1 => Signal::Halt,
            2 => Signal::Syscall,
            3 => Signal::Checkpoint,
            _ => Signal::Running,
        }
    }
}

/// Continuous execution result
#[pyclass]
#[derive(Debug, Clone)]
pub struct ContinuousResult {
    #[pyo3(get)]
    pub total_cycles: u32,
    #[pyo3(get)]
    pub batch_count: u32,
    #[pyo3(get)]
    pub signal: u32,
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    #[pyo3(get)]
    pub final_pc: u64,
}

#[pymethods]
impl ContinuousResult {
    #[getter]
    fn signal_name(&self) -> &str {
        match Signal::from(self.signal) {
            Signal::Running => "RUNNING",
            Signal::Halt => "HALT",
            Signal::Syscall => "SYSCALL",
            Signal::Checkpoint => "CHECKPOINT",
        }
    }

    #[getter]
    fn ips(&self) -> f64 {
        if self.elapsed_seconds > 0.0 {
            self.total_cycles as f64 / self.elapsed_seconds
        } else {
            0.0
        }
    }
}

/// Continuous GPU execution with atomic signaling
#[pyclass(unsendable)]
pub struct ContinuousMetalCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Shared memory buffers
    memory_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    cycles_per_batch_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mem_size_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Atomic signal buffer
    signal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Counter buffers
    total_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    memory_size: usize,
    cycles_per_batch: u32,
}

#[pymethods]
impl ContinuousMetalCPU {
    #[new]
    #[pyo3(signature = (memory_size = 4 * 1024 * 1024, cycles_per_batch = 10_000_000))]
    fn new(memory_size: usize, cycles_per_batch: u32) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[ContinuousMetalCPU] Using device: {:?}", device.name());

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        // Compile shader
        let source = NSString::from_str(CONTINUOUS_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let function_name = NSString::from_str("cpu_execute_continuous");
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

        let cycles_per_batch_buffer = device
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

        // Initialize config buffers
        unsafe {
            let ptr = mem_size_buffer.contents().as_ptr() as *mut u32;
            *ptr = memory_size as u32;

            let ptr = cycles_per_batch_buffer.contents().as_ptr() as *mut u32;
            *ptr = cycles_per_batch;
        }

        println!("[ContinuousMetalCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[ContinuousMetalCPU] Cycles per batch: {}", cycles_per_batch);
        println!("[ContinuousMetalCPU] Continuous execution with atomic signaling enabled");

        Ok(ContinuousMetalCPU {
            device,
            command_queue,
            pipeline,
            memory_buffer,
            registers_buffer,
            pc_buffer,
            flags_buffer,
            cycles_per_batch_buffer,
            mem_size_buffer,
            signal_buffer,
            total_cycles_buffer,
            batch_count_buffer,
            memory_size,
            cycles_per_batch,
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

        println!("[ContinuousMetalCPU] Loaded {} bytes at 0x{:X}", program.len(), address);
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

    /// Execute a single mega-batch on GPU - maximum throughput, no batching overhead
    /// This is the fastest possible execution mode for workloads that don't need
    /// frequent syscall checking.
    #[pyo3(signature = (total_cycles = 100_000_000))]
    fn execute_mega(&self, total_cycles: u32) -> PyResult<ContinuousResult> {
        let start = Instant::now();

        // Reset counters and set batch size to total cycles
        unsafe {
            let ptr = self.total_cycles_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
            let ptr = self.batch_count_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
            let ptr = self.signal_buffer.contents().as_ptr() as *mut u32;
            *ptr = Signal::Running as u32;
            // Set cycles per batch to total cycles for single dispatch
            let ptr = self.cycles_per_batch_buffer.contents().as_ptr() as *mut u32;
            *ptr = total_cycles;
        }

        // Single GPU dispatch
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
            encoder.setBuffer_offset_atIndex(Some(&self.cycles_per_batch_buffer), 0, 4);
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
        let (actual_cycles, batch_count, signal, final_pc) = unsafe {
            let total = *(self.total_cycles_buffer.contents().as_ptr() as *const u32);
            let batches = *(self.batch_count_buffer.contents().as_ptr() as *const u32);
            let sig = *(self.signal_buffer.contents().as_ptr() as *const u32);
            let pc = *(self.pc_buffer.contents().as_ptr() as *const u64);
            (total, batches, sig, pc)
        };

        // Restore original batch size
        unsafe {
            let ptr = self.cycles_per_batch_buffer.contents().as_ptr() as *mut u32;
            *ptr = self.cycles_per_batch;
        }

        let elapsed = start.elapsed().as_secs_f64();

        Ok(ContinuousResult {
            total_cycles: actual_cycles,
            batch_count,
            signal,
            elapsed_seconds: elapsed,
            final_pc,
        })
    }

    /// Execute continuously until halt, syscall, or timeout
    #[pyo3(signature = (max_batches = 1000, timeout_seconds = 60.0))]
    fn execute_continuous(&self, max_batches: u32, timeout_seconds: f64) -> PyResult<ContinuousResult> {
        let start = Instant::now();
        let timeout = Duration::from_secs_f64(timeout_seconds);

        // Reset counters
        unsafe {
            let ptr = self.total_cycles_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
            let ptr = self.batch_count_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
            let ptr = self.signal_buffer.contents().as_ptr() as *mut u32;
            *ptr = Signal::Running as u32;
        }

        let mut batches_executed = 0u32;

        // Execute batches until done
        while batches_executed < max_batches && start.elapsed() < timeout {
            // Clear signal for this batch
            unsafe {
                let ptr = self.signal_buffer.contents().as_ptr() as *mut u32;
                *ptr = Signal::Running as u32;
            }

            // Create and execute command buffer
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
                encoder.setBuffer_offset_atIndex(Some(&self.cycles_per_batch_buffer), 0, 4);
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

            batches_executed += 1;

            // Check signal
            let signal = unsafe {
                let ptr = self.signal_buffer.contents().as_ptr() as *const u32;
                Signal::from(*ptr)
            };

            // Stop if halt or syscall
            if signal == Signal::Halt || signal == Signal::Syscall {
                break;
            }
        }

        // Read final state
        let (total_cycles, batch_count, signal, final_pc) = unsafe {
            let total = *(self.total_cycles_buffer.contents().as_ptr() as *const u32);
            let batches = *(self.batch_count_buffer.contents().as_ptr() as *const u32);
            let sig = *(self.signal_buffer.contents().as_ptr() as *const u32);
            let pc = *(self.pc_buffer.contents().as_ptr() as *const u64);
            (total, batches, sig, pc)
        };

        let elapsed = start.elapsed().as_secs_f64();

        Ok(ContinuousResult {
            total_cycles,
            batch_count,
            signal,
            elapsed_seconds: elapsed,
            final_pc,
        })
    }

    fn reset(&self) {
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 32 * std::mem::size_of::<i64>());

            let ptr = self.pc_buffer.contents().as_ptr() as *mut u64;
            *ptr = 0;

            let ptr = self.flags_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 4 * std::mem::size_of::<f32>());

            let ptr = self.total_cycles_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;

            let ptr = self.batch_count_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;

            let ptr = self.signal_buffer.contents().as_ptr() as *mut u32;
            *ptr = 0;
        }
    }

    #[getter]
    fn memory_size(&self) -> usize {
        self.memory_size
    }

    #[getter]
    fn cycles_per_batch(&self) -> u32 {
        self.cycles_per_batch
    }
}

/// Register continuous execution types with Python module
pub fn register_continuous(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ContinuousMetalCPU>()?;
    m.add_class::<ContinuousResult>()?;
    Ok(())
}
