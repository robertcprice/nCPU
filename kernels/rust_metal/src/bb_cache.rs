//! Basic Block Caching - Cache decoded instruction blocks for faster loop execution
//!
//! Optimizations:
//! 1. Pre-decode basic blocks on first visit
//! 2. Cache decoded blocks in GPU memory
//! 3. Skip decode on subsequent visits to same PC
//! 4. Combined with fusion for maximum performance

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

/// Basic Block Cache shader with fusion
const BB_CACHE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

// Basic Block Cache Configuration
constant uint32_t CACHE_SIZE = 256;        // Number of cache entries
constant uint32_t MAX_BLOCK_SIZE = 16;     // Max instructions per block

// Cached basic block entry
struct CacheEntry {
    uint64_t start_pc;                     // Starting PC of this block
    uint32_t inst_count;                   // Number of instructions in block
    uint32_t hit_count;                    // Number of times this block was hit
    uint32_t instructions[MAX_BLOCK_SIZE]; // Pre-fetched instructions
    uint8_t is_loop;                       // Is this a loop block?
};

// Hash function for PC -> cache index
inline uint32_t hash_pc(uint64_t pc) {
    return (uint32_t)((pc >> 2) % CACHE_SIZE);
}

// Decode helpers for cached execution
struct DecodedInst {
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint16_t imm12;
    uint16_t imm16;
    uint8_t hw;
    uint8_t opcode_class;  // 0=alu, 1=mem, 2=branch, 3=other
};

inline DecodedInst decode_inst(uint32_t inst) {
    DecodedInst d;
    d.rd = inst & 0x1F;
    d.rn = (inst >> 5) & 0x1F;
    d.rm = (inst >> 16) & 0x1F;
    d.imm12 = (inst >> 10) & 0xFFF;
    d.imm16 = (inst >> 5) & 0xFFFF;
    d.hw = (inst >> 21) & 0x3;

    // Classify opcode
    uint8_t top = (inst >> 24) & 0xFF;
    if (top == 0x91 || top == 0x11 || top == 0xD1 || top == 0x51 ||
        top == 0x8B || top == 0xCB || top == 0x9B || top == 0xF1) {
        d.opcode_class = 0;  // ALU
    } else if (top == 0xF9 || top == 0xB9 || top == 0x39) {
        d.opcode_class = 1;  // Memory
    } else if ((top & 0xFC) == 0x14 || (top & 0xFC) == 0x94 ||
               top == 0x54 || top == 0xB4 || top == 0xB5) {
        d.opcode_class = 2;  // Branch
    } else {
        d.opcode_class = 3;  // Other
    }
    return d;
}

// Fusion check functions
inline bool can_fuse_add_add(uint32_t inst1, uint32_t inst2) {
    if ((inst1 & 0xFF000000) != 0x91000000) return false;
    if ((inst2 & 0xFF000000) != 0x91000000) return false;
    uint8_t rd1 = inst1 & 0x1F;
    uint8_t rn1 = (inst1 >> 5) & 0x1F;
    uint8_t rd2 = inst2 & 0x1F;
    uint8_t rn2 = (inst2 >> 5) & 0x1F;
    return (rd1 == rn1) && (rd2 == rn2) && (rd1 == rd2);
}

inline bool can_fuse_cmp_branch(uint32_t inst1, uint32_t inst2) {
    bool is_cmp = ((inst1 & 0xFF000000) == 0xF1000000) && ((inst1 & 0x1F) == 31);
    bool is_bcond = (inst2 & 0xFF000010) == 0x54000000;
    return is_cmp && is_bcond;
}

// Main execution kernel with BB cache + fusion
kernel void cpu_execute_bb_cache(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* cycles_per_batch [[buffer(4)]],
    device const uint32_t* mem_size [[buffer(5)]],
    device atomic_uint* signal_flag [[buffer(6)]],
    device uint32_t* total_cycles [[buffer(7)]],
    device uint32_t* batch_count [[buffer(8)]],
    device uint32_t* cache_hits [[buffer(9)]],
    device uint32_t* cache_misses [[buffer(10)]],
    device uint32_t* fused_count [[buffer(11)]],
    device CacheEntry* bb_cache [[buffer(12)]],  // Basic block cache
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t batch_cycles = cycles_per_batch[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;
    uint32_t hits = 0;
    uint32_t misses = 0;
    uint32_t fusions = 0;
    bool should_exit = false;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers[i];
    }

    float N = flags[0];
    float Z = flags[1];
    float C = flags[2];
    float V = flags[3];

    while (cycles < batch_cycles && !should_exit) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        // Check BB cache
        uint32_t cache_idx = hash_pc(pc);
        device CacheEntry* entry = &bb_cache[cache_idx];

        bool cache_hit = (entry->start_pc == pc && entry->inst_count > 0);

        if (cache_hit) {
            hits++;
            entry->hit_count++;

            // Execute from cache with fusion
            uint32_t block_cycles = 0;
            uint32_t i = 0;
            bool block_branch_taken = false;

            while (i < entry->inst_count && cycles + block_cycles < batch_cycles && !should_exit) {
                uint32_t inst1 = entry->instructions[i];
                uint32_t inst2 = (i + 1 < entry->inst_count) ? entry->instructions[i + 1] : 0;
                bool has_inst2 = (i + 1 < entry->inst_count);

                // Check for halt/syscall
                if ((inst1 & 0xFFE0001F) == 0xD4400000) {
                    atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
                    should_exit = true;
                    break;
                }
                if ((inst1 & 0xFFE0001F) == 0xD4000001) {
                    atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
                    should_exit = true;
                    break;
                }

                uint8_t rd = inst1 & 0x1F;
                uint8_t rn = (inst1 >> 5) & 0x1F;
                uint8_t rm = (inst1 >> 16) & 0x1F;
                uint16_t imm12 = (inst1 >> 10) & 0xFFF;
                uint16_t imm16 = (inst1 >> 5) & 0xFFFF;
                uint8_t hw = (inst1 >> 21) & 0x3;

                bool branch_taken = false;
                uint32_t advance = 1;

                // Fusion: ADD + ADD
                if (has_inst2 && can_fuse_add_add(inst1, inst2)) {
                    uint16_t imm12_2 = (inst2 >> 10) & 0xFFF;
                    uint32_t combined = imm12 + imm12_2;
                    if (combined <= 0xFFF) {
                        if (rd < 31) regs[rd] = regs[rn] + combined;
                        advance = 2;
                        fusions++;
                        block_cycles++;
                    }
                }
                // Fusion: CMP + B.cond
                else if (has_inst2 && can_fuse_cmp_branch(inst1, inst2)) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t result = rn_val - imm12;
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
                    V = 0.0;

                    uint8_t cond = inst2 & 0xF;
                    int32_t imm19 = (inst2 >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;

                    bool take = false;
                    switch (cond) {
                        case 0x0: take = (Z > 0.5); break;
                        case 0x1: take = (Z < 0.5); break;
                        case 0xA: take = ((N > 0.5) == (V > 0.5)); break;
                        case 0xB: take = ((N > 0.5) != (V > 0.5)); break;
                        case 0xC: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                        case 0xD: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                        default: break;
                    }

                    if (take) {
                        pc = (pc + (i + 1) * 4) + (int64_t)imm19 * 4;
                        branch_taken = true;
                    } else {
                        advance = 2;
                    }
                    fusions++;
                    block_cycles++;
                }
                // Regular execution
                else if ((inst1 & 0xFF000000) == 0x91000000) {
                    regs[rd] = regs[rn] + imm12;
                }
                else if ((inst1 & 0xFF000000) == 0xD1000000) {
                    regs[rd] = regs[rn] - imm12;
                }
                else if ((inst1 & 0xFF200000) == 0x8B000000) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                    if (rd < 31) regs[rd] = rn_val + rm_val;
                }
                else if ((inst1 & 0xFF200000) == 0xCB000000) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                    if (rd < 31) regs[rd] = rn_val - rm_val;
                }
                else if ((inst1 & 0xFF000000) == 0x9B000000) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                    uint8_t ra = (inst1 >> 10) & 0x1F;
                    int64_t ra_val = (ra == 31) ? 0 : regs[ra];
                    if (rd < 31) regs[rd] = rn_val * rm_val + ra_val;
                }
                else if ((inst1 & 0xFF000000) == 0xF1000000) {
                    // SUBS immediate
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t result = rn_val - imm12;
                    if (rd < 31) regs[rd] = result;
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
                    V = 0.0;
                }
                else if ((inst1 & 0xFF200000) == 0xEB000000) {
                    // SUBS register (CMP X, X)
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                    int64_t result = rn_val - rm_val;
                    if (rd < 31) regs[rd] = result;
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint64_t)rn_val >= (uint64_t)rm_val) ? 1.0 : 0.0;
                    V = (((rn_val ^ rm_val) & (rn_val ^ result)) < 0) ? 1.0 : 0.0;
                }
                else if ((inst1 & 0xFF800000) == 0xD2800000) {
                    if (rd < 31) regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
                }
                else if ((inst1 & 0xFF800000) == 0xF2800000) {
                    if (rd < 31) {
                        uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
                        uint64_t val = (uint64_t)regs[rd] & mask;
                        val |= ((uint64_t)imm16 << (hw * 16));
                        regs[rd] = (int64_t)val;
                    }
                }
                else if ((inst1 & 0xFC000000) == 0x14000000) {
                    int32_t imm26 = inst1 & 0x3FFFFFF;
                    if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                    pc = (pc + i * 4) + (int64_t)imm26 * 4;
                    branch_taken = true;
                }
                else if ((inst1 & 0xFF000010) == 0x54000000) {
                    uint8_t cond = inst1 & 0xF;
                    int32_t imm19 = (inst1 >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    bool take = false;
                    switch (cond) {
                        case 0x0: take = (Z > 0.5); break;
                        case 0x1: take = (Z < 0.5); break;
                        case 0xA: take = ((N > 0.5) == (V > 0.5)); break;
                        case 0xB: take = ((N > 0.5) != (V > 0.5)); break;
                        case 0xC: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                        case 0xD: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                        default: break;
                    }
                    if (take) {
                        pc = (pc + i * 4) + (int64_t)imm19 * 4;
                        branch_taken = true;
                    }
                }
                else if ((inst1 & 0xFF000000) == 0xB4000000) {
                    int32_t imm19 = (inst1 >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    int64_t rt_val = (rd == 31) ? 0 : regs[rd];
                    if (rt_val == 0) {
                        pc = (pc + i * 4) + (int64_t)imm19 * 4;
                        branch_taken = true;
                    }
                }
                else if ((inst1 & 0xFF000000) == 0xB5000000) {
                    int32_t imm19 = (inst1 >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    int64_t rt_val = (rd == 31) ? 0 : regs[rd];
                    if (rt_val != 0) {
                        pc = (pc + i * 4) + (int64_t)imm19 * 4;
                        branch_taken = true;
                    }
                }

                if (branch_taken) {
                    cycles += block_cycles + 1;
                    block_branch_taken = true;
                    break;
                }

                i += advance;
                block_cycles++;
            }

            if (!block_branch_taken && !should_exit) {
                pc += entry->inst_count * 4;
                cycles += block_cycles;
            }
        }
        else {
            // Cache miss - fetch and cache the block
            misses++;

            entry->start_pc = pc;
            entry->hit_count = 0;
            entry->inst_count = 0;
            entry->is_loop = 0;

            // Fetch instructions until branch or max size
            uint32_t block_pc = pc;
            while (entry->inst_count < MAX_BLOCK_SIZE && block_pc + 4 <= memory_size) {
                uint32_t inst = uint32_t(memory[block_pc]) |
                               (uint32_t(memory[block_pc + 1]) << 8) |
                               (uint32_t(memory[block_pc + 2]) << 16) |
                               (uint32_t(memory[block_pc + 3]) << 24);

                entry->instructions[entry->inst_count++] = inst;
                block_pc += 4;

                // Stop at branches, halt, syscall
                uint8_t top = (inst >> 24) & 0xFF;
                if ((top & 0xFC) == 0x14 || (top & 0xFC) == 0x94 ||
                    top == 0x54 || top == 0xB4 || top == 0xB5 ||
                    top == 0xD6 ||  // BR/BLR/RET
                    (inst & 0xFFE0001F) == 0xD4400000 ||  // HLT
                    (inst & 0xFFE0001F) == 0xD4000001) {  // SVC

                    // Check if backward branch (loop)
                    if (top == 0x54) {
                        int32_t imm19 = (inst >> 5) & 0x7FFFF;
                        if (imm19 & 0x40000) entry->is_loop = 1;
                    }
                    break;
                }
            }

            // Execute the entire cached block (same as hit path)
            // This fixes the bug where only first instruction was executed on miss
            uint32_t block_cycles = 0;
            uint32_t i = 0;
            bool block_branch_taken = false;

            while (i < entry->inst_count && cycles + block_cycles < batch_cycles && !should_exit) {
                uint32_t inst1 = entry->instructions[i];
                uint32_t inst2 = (i + 1 < entry->inst_count) ? entry->instructions[i + 1] : 0;
                bool has_inst2 = (i + 1 < entry->inst_count);

                // Check for halt/syscall
                if ((inst1 & 0xFFE0001F) == 0xD4400000) {
                    atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
                    should_exit = true;
                    break;
                }
                if ((inst1 & 0xFFE0001F) == 0xD4000001) {
                    atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
                    should_exit = true;
                    break;
                }

                uint8_t rd = inst1 & 0x1F;
                uint8_t rn = (inst1 >> 5) & 0x1F;
                uint8_t rm = (inst1 >> 16) & 0x1F;
                uint16_t imm12 = (inst1 >> 10) & 0xFFF;
                uint16_t imm16 = (inst1 >> 5) & 0xFFFF;
                uint8_t hw = (inst1 >> 21) & 0x3;

                bool branch_taken = false;
                uint32_t advance = 1;

                // Fusion: CMP + B.cond (most common in loops)
                if (has_inst2 && can_fuse_cmp_branch(inst1, inst2)) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t result = rn_val - imm12;
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
                    V = 0.0;

                    uint8_t cond = inst2 & 0xF;
                    int32_t imm19 = (inst2 >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;

                    bool take = false;
                    switch (cond) {
                        case 0x0: take = (Z > 0.5); break;
                        case 0x1: take = (Z < 0.5); break;
                        case 0xA: take = ((N > 0.5) == (V > 0.5)); break;
                        case 0xB: take = ((N > 0.5) != (V > 0.5)); break;
                        case 0xC: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                        case 0xD: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                        default: break;
                    }
                    if (take) {
                        pc = (pc + (i + 1) * 4) + (int64_t)imm19 * 4;
                        branch_taken = true;
                    } else {
                        advance = 2;
                    }
                    fusions++;
                    block_cycles++;
                }
                // Regular instruction execution
                else if ((inst1 & 0xFF000000) == 0x91000000) {
                    if (rd < 31) regs[rd] = regs[rn] + imm12;
                }
                else if ((inst1 & 0xFF000000) == 0xD1000000) {
                    if (rd < 31) regs[rd] = regs[rn] - imm12;
                }
                else if ((inst1 & 0xFF800000) == 0xD2800000) {
                    if (rd < 31) regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
                }
                else if ((inst1 & 0xFF800000) == 0xF2800000) {
                    if (rd < 31) {
                        uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
                        uint64_t val = (uint64_t)regs[rd] & mask;
                        val |= ((uint64_t)imm16 << (hw * 16));
                        regs[rd] = (int64_t)val;
                    }
                }
                else if ((inst1 & 0xFF200000) == 0xEB000000) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                    int64_t result = rn_val - rm_val;
                    if (rd < 31) regs[rd] = result;
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint64_t)rn_val >= (uint64_t)rm_val) ? 1.0 : 0.0;
                    V = (((rn_val ^ rm_val) & (rn_val ^ result)) < 0) ? 1.0 : 0.0;
                }
                else if ((inst1 & 0xFF000000) == 0xF1000000) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t result = rn_val - imm12;
                    if (rd < 31) regs[rd] = result;
                    N = (result < 0) ? 1.0 : 0.0;
                    Z = (result == 0) ? 1.0 : 0.0;
                    C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
                    V = 0.0;
                }
                else if ((inst1 & 0xFF200000) == 0x8B000000) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                    if (rd < 31) regs[rd] = rn_val + rm_val;
                }
                else if ((inst1 & 0xFF200000) == 0xCB000000) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                    if (rd < 31) regs[rd] = rn_val - rm_val;
                }
                else if ((inst1 & 0xFFC00000) == 0xF9400000) {
                    int64_t base = regs[rn];
                    int64_t offset = (int64_t)imm12 * 8;
                    uint64_t addr = (uint64_t)(base + offset);
                    if (addr + 8 <= memory_size && rd < 31) {
                        regs[rd] = (int64_t)(uint64_t(memory[addr]) |
                                   (uint64_t(memory[addr + 1]) << 8) |
                                   (uint64_t(memory[addr + 2]) << 16) |
                                   (uint64_t(memory[addr + 3]) << 24) |
                                   (uint64_t(memory[addr + 4]) << 32) |
                                   (uint64_t(memory[addr + 5]) << 40) |
                                   (uint64_t(memory[addr + 6]) << 48) |
                                   (uint64_t(memory[addr + 7]) << 56));
                    }
                }
                else if ((inst1 & 0xFFC00000) == 0xF9000000) {
                    int64_t base = regs[rn];
                    int64_t offset = (int64_t)imm12 * 8;
                    uint64_t addr = (uint64_t)(base + offset);
                    int64_t val = (rd == 31) ? 0 : regs[rd];
                    if (addr + 8 <= memory_size) {
                        memory[addr] = val & 0xFF;
                        memory[addr + 1] = (val >> 8) & 0xFF;
                        memory[addr + 2] = (val >> 16) & 0xFF;
                        memory[addr + 3] = (val >> 24) & 0xFF;
                        memory[addr + 4] = (val >> 32) & 0xFF;
                        memory[addr + 5] = (val >> 40) & 0xFF;
                        memory[addr + 6] = (val >> 48) & 0xFF;
                        memory[addr + 7] = (val >> 56) & 0xFF;
                    }
                }
                else if ((inst1 & 0xFF000010) == 0x54000000) {
                    uint8_t cond = inst1 & 0xF;
                    int32_t imm19 = (inst1 >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    bool take = false;
                    switch (cond) {
                        case 0x0: take = (Z > 0.5); break;
                        case 0x1: take = (Z < 0.5); break;
                        case 0x2: take = (C > 0.5); break;
                        case 0x3: take = (C < 0.5); break;
                        case 0xA: take = ((N > 0.5) == (V > 0.5)); break;
                        case 0xB: take = ((N > 0.5) != (V > 0.5)); break;
                        case 0xC: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                        case 0xD: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                        default: break;
                    }
                    if (take) {
                        pc = (pc + i * 4) + (int64_t)imm19 * 4;
                        branch_taken = true;
                    }
                }
                else if ((inst1 & 0xFC000000) == 0x14000000) {
                    int32_t imm26 = inst1 & 0x3FFFFFF;
                    if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                    pc = (pc + i * 4) + (int64_t)imm26 * 4;
                    branch_taken = true;
                }

                if (branch_taken) {
                    cycles += block_cycles + 1;
                    block_branch_taken = true;
                    break;
                }

                i += advance;
                block_cycles++;
            }

            if (!block_branch_taken && !should_exit) {
                pc += entry->inst_count * 4;
                cycles += block_cycles;
            }
        }
    }

    // Write back results
    for (int i = 0; i < 32; i++) {
        registers[i] = regs[i];
    }
    pc_ptr[0] = pc;
    flags[0] = N;
    flags[1] = Z;
    flags[2] = C;
    flags[3] = V;

    atomic_fetch_add_explicit((device atomic_uint*)total_cycles, cycles, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)batch_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)cache_hits, hits, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)cache_misses, misses, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)fused_count, fusions, memory_order_relaxed);

    uint32_t current_signal = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (current_signal == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

/// Cache entry size in bytes
const CACHE_ENTRY_SIZE: usize = 8 + 4 + 4 + (16 * 4) + 1 + 7; // Align to 88 bytes
const CACHE_SIZE: usize = 256;

/// BB Cache execution result
#[pyclass]
#[derive(Debug, Clone)]
pub struct BBCacheResult {
    #[pyo3(get)]
    pub total_cycles: u32,
    #[pyo3(get)]
    pub batch_count: u32,
    #[pyo3(get)]
    pub cache_hits: u32,
    #[pyo3(get)]
    pub cache_misses: u32,
    #[pyo3(get)]
    pub fused_count: u32,
    #[pyo3(get)]
    pub signal: u32,
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    #[pyo3(get)]
    pub ips: f64,
    #[pyo3(get)]
    pub cache_hit_rate: f64,
}

#[pymethods]
impl BBCacheResult {
    #[getter]
    fn signal_name(&self) -> &'static str {
        match self.signal {
            0 => "RUNNING",
            1 => "HALT",
            2 => "SYSCALL",
            3 => "CHECKPOINT",
            _ => "UNKNOWN",
        }
    }
}

/// BB Cache Metal CPU
#[pyclass(unsendable)]
pub struct BBCacheMetalCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    memory_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    cycles_per_batch_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mem_size_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    signal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    cache_hits_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    cache_misses_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    fused_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    bb_cache_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    memory_size: usize,
    cycles_per_batch: u32,
}

#[pymethods]
impl BBCacheMetalCPU {
    #[new]
    #[pyo3(signature = (memory_size = 4 * 1024 * 1024, cycles_per_batch = 10_000_000))]
    pub fn new(memory_size: usize, cycles_per_batch: u32) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[BBCacheMetalCPU] Using device: {:?}", device.name());

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        let source = NSString::from_str(BB_CACHE_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let function_name = NSString::from_str("cpu_execute_bb_cache");
        let function = library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("Function not found".to_string()))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        let shared_options = MTLResourceOptions::StorageModeShared;

        let memory_buffer = device
            .newBufferWithLength_options(memory_size, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let registers_buffer = device
            .newBufferWithLength_options(32 * 8, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let pc_buffer = device
            .newBufferWithLength_options(8, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let flags_buffer = device
            .newBufferWithLength_options(16, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cycles_per_batch_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let mem_size_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let signal_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let total_cycles_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let batch_count_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cache_hits_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cache_misses_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let fused_count_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // BB Cache buffer - 256 entries * 88 bytes each
        let bb_cache_buffer = device
            .newBufferWithLength_options(CACHE_SIZE * CACHE_ENTRY_SIZE, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        unsafe {
            let ptr = cycles_per_batch_buffer.contents().as_ptr() as *mut u32;
            *ptr = cycles_per_batch;

            let ptr = mem_size_buffer.contents().as_ptr() as *mut u32;
            *ptr = memory_size as u32;

            // Clear cache
            std::ptr::write_bytes(bb_cache_buffer.contents().as_ptr() as *mut u8, 0, CACHE_SIZE * CACHE_ENTRY_SIZE);
        }

        println!("[BBCacheMetalCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[BBCacheMetalCPU] BB Cache: {} entries, {} bytes", CACHE_SIZE, CACHE_SIZE * CACHE_ENTRY_SIZE);
        println!("[BBCacheMetalCPU] Fusion + BB Cache enabled");

        Ok(Self {
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
            cache_hits_buffer,
            cache_misses_buffer,
            fused_count_buffer,
            bb_cache_buffer,
            memory_size,
            cycles_per_batch,
        })
    }

    pub fn load_program(&self, program: Vec<u8>, address: usize) -> PyResult<()> {
        if address + program.len() > self.memory_size {
            return Err(PyRuntimeError::new_err("Program too large"));
        }
        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(program.as_ptr(), ptr.add(address), program.len());
        }
        println!("[BBCacheMetalCPU] Loaded {} bytes at 0x{:X}", program.len(), address);
        Ok(())
    }

    pub fn set_pc(&self, pc: u64) {
        unsafe {
            let ptr = self.pc_buffer.contents().as_ptr() as *mut u64;
            *ptr = pc;
        }
    }

    pub fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buffer.contents().as_ptr() as *const u64) }
    }

    pub fn set_register(&self, index: usize, value: i64) {
        if index >= 32 { return; }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *mut i64;
            *ptr.add(index) = value;
        }
    }

    pub fn get_register(&self, index: usize) -> i64 {
        if index >= 32 { return 0; }
        unsafe { *(self.registers_buffer.contents().as_ptr() as *const i64).add(index) }
    }

    pub fn clear_cache(&self) {
        unsafe {
            std::ptr::write_bytes(self.bb_cache_buffer.contents().as_ptr() as *mut u8, 0, CACHE_SIZE * CACHE_ENTRY_SIZE);
        }
    }

    #[pyo3(signature = (max_batches = 1000, timeout_seconds = 60.0))]
    pub fn execute(&self, max_batches: u32, timeout_seconds: f64) -> PyResult<BBCacheResult> {
        unsafe {
            *(self.signal_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.total_cycles_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.batch_count_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.cache_hits_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.cache_misses_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.fused_count_buffer.contents().as_ptr() as *mut u32) = 0;
        }

        let start = Instant::now();
        let timeout = std::time::Duration::from_secs_f64(timeout_seconds);
        let mut batches = 0u32;

        while batches < max_batches && start.elapsed() < timeout {
            unsafe { *(self.signal_buffer.contents().as_ptr() as *mut u32) = 0; }

            let cmd_buffer = self.command_queue.commandBuffer()
                .ok_or_else(|| PyRuntimeError::new_err("Failed to create command buffer"))?;
            let encoder = cmd_buffer.computeCommandEncoder()
                .ok_or_else(|| PyRuntimeError::new_err("Failed to create encoder"))?;

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
                encoder.setBuffer_offset_atIndex(Some(&self.cache_hits_buffer), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(&self.cache_misses_buffer), 0, 10);
                encoder.setBuffer_offset_atIndex(Some(&self.fused_count_buffer), 0, 11);
                encoder.setBuffer_offset_atIndex(Some(&self.bb_cache_buffer), 0, 12);

                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: 1, height: 1, depth: 1 },
                    MTLSize { width: 1, height: 1, depth: 1 },
                );
            }
            encoder.endEncoding();
            cmd_buffer.commit();
            cmd_buffer.waitUntilCompleted();

            batches += 1;
            let signal = unsafe { *(self.signal_buffer.contents().as_ptr() as *const u32) };
            if signal == 1 || signal == 2 { break; }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let total_cycles = unsafe { *(self.total_cycles_buffer.contents().as_ptr() as *const u32) };
        let batch_count = unsafe { *(self.batch_count_buffer.contents().as_ptr() as *const u32) };
        let cache_hits = unsafe { *(self.cache_hits_buffer.contents().as_ptr() as *const u32) };
        let cache_misses = unsafe { *(self.cache_misses_buffer.contents().as_ptr() as *const u32) };
        let fused_count = unsafe { *(self.fused_count_buffer.contents().as_ptr() as *const u32) };
        let signal = unsafe { *(self.signal_buffer.contents().as_ptr() as *const u32) };

        let ips = if elapsed > 0.0 { total_cycles as f64 / elapsed } else { 0.0 };
        let total_accesses = cache_hits + cache_misses;
        let cache_hit_rate = if total_accesses > 0 { cache_hits as f64 / total_accesses as f64 * 100.0 } else { 0.0 };

        Ok(BBCacheResult {
            total_cycles,
            batch_count,
            cache_hits,
            cache_misses,
            fused_count,
            signal,
            elapsed_seconds: elapsed,
            ips,
            cache_hit_rate,
        })
    }

    pub fn reset(&self) {
        unsafe {
            std::ptr::write_bytes(self.registers_buffer.contents().as_ptr() as *mut u8, 0, 32 * 8);
            std::ptr::write_bytes(self.pc_buffer.contents().as_ptr() as *mut u8, 0, 8);
            std::ptr::write_bytes(self.flags_buffer.contents().as_ptr() as *mut u8, 0, 16);
        }
        self.clear_cache();
    }
}

pub fn register_bb_cache(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BBCacheMetalCPU>()?;
    m.add_class::<BBCacheResult>()?;
    Ok(())
}
