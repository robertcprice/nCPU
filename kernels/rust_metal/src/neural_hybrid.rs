//! Neural Hybrid Engine - BBCache speed + Neural OoO scheduling
//!
//! Combines:
//! 1. BBCache's basic block caching for fast block lookups
//! 2. Neural lookup table dependency prediction for OoO within blocks
//! 3. Device-persistent caching for dependency matrices

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

/// Neural Hybrid shader - BBCache + Neural OoO
const NEURAL_HYBRID_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

// Cache Configuration
constant uint32_t CACHE_SIZE = 256;
constant uint32_t MAX_BLOCK_SIZE = 8;

// Instruction type IDs for lookup table
constant uint8_t ITYPE_UNKNOWN = 0;
constant uint8_t ITYPE_ALU_IMM = 1;
constant uint8_t ITYPE_ALU_REG = 2;
constant uint8_t ITYPE_MOV = 3;
constant uint8_t ITYPE_CMP = 4;
constant uint8_t ITYPE_LOAD = 5;
constant uint8_t ITYPE_STORE = 6;
constant uint8_t ITYPE_CBRANCH = 7;
constant uint8_t ITYPE_BRANCH = 8;
constant uint8_t ITYPE_CCBRANCH = 9;
constant uint8_t ITYPE_MADD = 10;
constant uint8_t NUM_ITYPES = 11;

// Classify instruction type for fast lookup
inline uint8_t classify_inst(uint32_t inst) {
    if ((inst & 0xFFC00000) == 0xF9400000) return ITYPE_LOAD;
    if ((inst & 0xFFC00000) == 0xF9000000) return ITYPE_STORE;
    if ((inst & 0xFF000000) == 0x91000000) return ITYPE_ALU_IMM;
    if ((inst & 0xFF000000) == 0xD1000000) return ITYPE_ALU_IMM;
    if ((inst & 0xFF200000) == 0x8B000000) return ITYPE_ALU_REG;
    if ((inst & 0xFF200000) == 0xCB000000) return ITYPE_ALU_REG;
    if ((inst & 0xFF000000) == 0xF1000000) return ITYPE_CMP;
    if ((inst & 0xFF200000) == 0xEB000000) return ITYPE_CMP;
    if ((inst & 0xFF800000) == 0xD2800000) return ITYPE_MOV;
    if ((inst & 0xFF800000) == 0xF2800000) return ITYPE_MOV;
    if ((inst & 0xFF000000) == 0xB4000000) return ITYPE_CBRANCH;
    if ((inst & 0xFF000000) == 0xB5000000) return ITYPE_CBRANCH;
    if ((inst & 0xFF000010) == 0x54000000) return ITYPE_CCBRANCH;
    if ((inst & 0xFC000000) == 0x14000000) return ITYPE_BRANCH;
    if ((inst & 0xFC000000) == 0x94000000) return ITYPE_BRANCH;
    if ((inst & 0xFF000000) == 0x9B000000) return ITYPE_MADD;
    return ITYPE_UNKNOWN;
}

// Pre-computed dependency lookup table (distilled from neural weights)
constant float DEP_LOOKUP[NUM_ITYPES][NUM_ITYPES] = {
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8},
    {0.8, 0.2, 0.2, 0.1, 0.3, 0.5, 0.5, 0.9, 0.1, 0.1, 0.3},
    {0.8, 0.2, 0.2, 0.1, 0.3, 0.5, 0.5, 0.9, 0.1, 0.1, 0.3},
    {0.8, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9},
    {0.8, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.95, 0.1},
    {0.8, 0.9, 0.9, 0.1, 0.9, 0.3, 0.5, 0.9, 0.1, 0.1, 0.9},
    {0.8, 0.1, 0.1, 0.1, 0.1, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1},
    {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
    {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
    {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
    {0.8, 0.3, 0.3, 0.1, 0.3, 0.5, 0.5, 0.9, 0.1, 0.1, 0.3}
};

// Decoded instruction
struct DecodedInst {
    uint32_t raw;
    uint8_t rd, rn, rm;
    uint8_t sets_flags, uses_flags, is_branch;
};

// Cached block with instructions and pre-computed dependencies
struct CachedBlock {
    uint64_t pc;
    DecodedInst insts[MAX_BLOCK_SIZE];
    float deps[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
    uint32_t count;
    uint32_t valid;
    uint32_t hits;
};

// Hash PC to cache index
inline uint32_t hash_pc(uint64_t pc) {
    return (uint32_t)((pc >> 2) % CACHE_SIZE);
}

// Decode instruction
inline DecodedInst decode(uint32_t inst) {
    DecodedInst d;
    d.raw = inst;
    d.rd = inst & 0x1F;
    d.rn = (inst >> 5) & 0x1F;
    d.rm = (inst >> 16) & 0x1F;
    d.uses_flags = 0;
    d.sets_flags = 0;
    d.is_branch = 0;

    uint8_t top = (inst >> 24) & 0xFF;

    if ((inst & 0xFF000000) == 0xF1000000 || (inst & 0xFF200000) == 0xEB000000) {
        d.sets_flags = 1;
    }
    if ((top & 0xFC) == 0x14 || (top & 0xFC) == 0x94 || top == 0x54 ||
        top == 0xB4 || top == 0xB5 || top == 0xD6) {
        d.is_branch = 1;
        if (top == 0x54) d.uses_flags = 1;
    }
    if ((inst & 0xFFC00000) == 0xF9000000 || (inst & 0xFFE0001F) == 0xD4400000) {
        d.rd = 32;
    }
    if ((inst & 0xFF000000) == 0xB4000000 || (inst & 0xFF000000) == 0xB5000000) {
        d.rn = inst & 0x1F;
        d.rm = 32;
        d.rd = 32;
    }
    if ((inst & 0xFF000010) == 0x54000000 || (inst & 0xFC000000) == 0x14000000) {
        d.rd = 32;
    }

    return d;
}

// Fast dependency prediction using lookup table
inline float predict_dep_fast(DecodedInst prod, DecodedInst cons) {
    bool possible_raw = (prod.rd < 31) && (prod.rd == cons.rn || prod.rd == cons.rm);
    bool possible_waw = (prod.rd < 31) && (cons.rd < 31) && (prod.rd == cons.rd);
    bool possible_flag = prod.sets_flags && cons.uses_flags;

    if (!possible_raw && !possible_waw && !possible_flag) {
        return 0.05;
    }

    uint8_t pt = classify_inst(prod.raw);
    uint8_t ct = classify_inst(cons.raw);
    float base = DEP_LOOKUP[pt][ct];

    if (possible_raw) base = max(base, 0.9f);
    if (possible_flag) base = max(base, 0.95f);

    return base;
}

// Get register with forwarding
inline int64_t get_fwd(uint8_t reg, uint8_t prod_idx, thread int64_t* regs,
                       thread int64_t* results, thread bool* executed) {
    if (reg == 31) return 0;
    if (prod_idx != 0xFF && executed[prod_idx]) {
        return results[prod_idx];
    }
    return regs[reg];
}

// Execute single instruction with forwarding
inline void exec_inst(DecodedInst d, thread int64_t* regs,
                      thread float& N, thread float& Z, thread float& C, thread float& V,
                      device uint8_t* memory, uint32_t mem_size,
                      thread int64_t& result, thread bool& branch_taken,
                      uint8_t prod_rn, uint8_t prod_rm,
                      thread int64_t* results, thread bool* executed) {
    uint32_t inst = d.raw;
    uint8_t rd = d.rd, rn = d.rn, rm = d.rm;
    uint16_t imm12 = (inst >> 10) & 0xFFF;
    uint16_t imm16 = (inst >> 5) & 0xFFFF;
    uint8_t hw = (inst >> 21) & 0x3;

    int64_t rn_val = get_fwd(rn, prod_rn, regs, results, executed);
    int64_t rm_val = get_fwd(rm, prod_rm, regs, results, executed);
    int64_t rd_val = (rd < 31) ? regs[rd] : 0;

    result = 0;
    branch_taken = false;

    // ADD imm
    if ((inst & 0xFF000000) == 0x91000000) { result = rn_val + imm12; return; }
    // SUB imm
    if ((inst & 0xFF000000) == 0xD1000000) { result = rn_val - imm12; return; }
    // ADD reg
    if ((inst & 0xFF200000) == 0x8B000000) { result = rn_val + rm_val; return; }
    // SUB reg
    if ((inst & 0xFF200000) == 0xCB000000) { result = rn_val - rm_val; return; }
    // SUBS imm
    if ((inst & 0xFF000000) == 0xF1000000) {
        result = rn_val - imm12;
        N = (result < 0) ? 1.0 : 0.0;
        Z = (result == 0) ? 1.0 : 0.0;
        C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
        V = 0.0;
        return;
    }
    // SUBS reg
    if ((inst & 0xFF200000) == 0xEB000000) {
        result = rn_val - rm_val;
        N = (result < 0) ? 1.0 : 0.0;
        Z = (result == 0) ? 1.0 : 0.0;
        C = ((uint64_t)rn_val >= (uint64_t)rm_val) ? 1.0 : 0.0;
        V = 0.0;
        return;
    }
    // MOVZ
    if ((inst & 0xFF800000) == 0xD2800000) {
        result = (int64_t)((uint64_t)imm16 << (hw * 16));
        return;
    }
    // MOVK
    if ((inst & 0xFF800000) == 0xF2800000) {
        uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
        result = (int64_t)(((uint64_t)rd_val & mask) | ((uint64_t)imm16 << (hw * 16)));
        return;
    }
    // MADD
    if ((inst & 0xFF000000) == 0x9B000000) {
        uint8_t ra = (inst >> 10) & 0x1F;
        int64_t ra_val = (ra == 31) ? 0 : regs[ra];
        result = ra_val + rn_val * rm_val;
        return;
    }
    // LDR 64-bit
    if ((inst & 0xFFC00000) == 0xF9400000) {
        uint64_t addr = (uint64_t)(rn_val + (int64_t)imm12 * 8);
        if (addr + 8 <= mem_size) {
            result = (int64_t)(uint64_t(memory[addr]) | (uint64_t(memory[addr+1])<<8) |
                     (uint64_t(memory[addr+2])<<16) | (uint64_t(memory[addr+3])<<24) |
                     (uint64_t(memory[addr+4])<<32) | (uint64_t(memory[addr+5])<<40) |
                     (uint64_t(memory[addr+6])<<48) | (uint64_t(memory[addr+7])<<56));
        }
        return;
    }
    // STR 64-bit
    if ((inst & 0xFFC00000) == 0xF9000000) {
        uint64_t addr = (uint64_t)(rn_val + (int64_t)imm12 * 8);
        if (addr + 8 <= mem_size) {
            memory[addr] = rd_val & 0xFF;
            memory[addr+1] = (rd_val >> 8) & 0xFF;
            memory[addr+2] = (rd_val >> 16) & 0xFF;
            memory[addr+3] = (rd_val >> 24) & 0xFF;
            memory[addr+4] = (rd_val >> 32) & 0xFF;
            memory[addr+5] = (rd_val >> 40) & 0xFF;
            memory[addr+6] = (rd_val >> 48) & 0xFF;
            memory[addr+7] = (rd_val >> 56) & 0xFF;
        }
        return;
    }
    // B.cond
    if ((inst & 0xFF000010) == 0x54000000) {
        uint8_t cond = inst & 0xF;
        bool take = false;
        switch (cond) {
            case 0x0: take = (Z > 0.5); break;
            case 0x1: take = (Z < 0.5); break;
            case 0xA: take = ((N > 0.5) == (V > 0.5)); break;
            case 0xB: take = ((N > 0.5) != (V > 0.5)); break;
            case 0xC: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
            case 0xD: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
        }
        if (take) branch_taken = true;
        return;
    }
    // B
    if ((inst & 0xFC000000) == 0x14000000) { branch_taken = true; return; }
    // BL
    if ((inst & 0xFC000000) == 0x94000000) { branch_taken = true; return; }
    // CBZ
    if ((inst & 0xFF000000) == 0xB4000000) {
        if (rn_val == 0) branch_taken = true;
        return;
    }
    // CBNZ
    if ((inst & 0xFF000000) == 0xB5000000) {
        if (rn_val != 0) branch_taken = true;
        return;
    }
}

// Main hybrid kernel: BBCache + Neural OoO
kernel void cpu_execute_neural_hybrid(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* cycles_per_batch [[buffer(4)]],
    device const uint32_t* mem_size [[buffer(5)]],
    device atomic_uint* signal_flag [[buffer(6)]],
    device uint32_t* total_cycles [[buffer(7)]],
    device uint32_t* batch_count [[buffer(8)]],
    device uint32_t* stats [[buffer(9)]],
    device CachedBlock* cache [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t batch_cycles = cycles_per_batch[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;
    uint32_t parallel_execs = 0;
    bool should_exit = false;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) regs[i] = registers[i];
    regs[31] = 0;

    float N = flags[0], Z = flags[1], C = flags[2], V = flags[3];

    while (cycles < batch_cycles && !should_exit) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        // Lookup in BBCache
        uint32_t cache_idx = hash_pc(pc);
        device CachedBlock* block = &cache[cache_idx];
        bool cache_hit = (block->valid == 1 && block->pc == pc);

        DecodedInst window[MAX_BLOCK_SIZE];
        float dep_matrix[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
        uint32_t window_count = 0;

        if (cache_hit) {
            // Use cached block and dependencies
            cache_hits++;
            block->hits++;
            window_count = block->count;
            for (uint32_t i = 0; i < window_count; i++) {
                window[i] = block->insts[i];
                for (uint32_t j = 0; j < window_count; j++) {
                    dep_matrix[i][j] = block->deps[i][j];
                }
            }
        } else {
            // Fetch and decode new block
            cache_misses++;
            uint64_t bpc = pc;
            while (window_count < MAX_BLOCK_SIZE && bpc + 4 <= memory_size) {
                uint32_t inst = uint32_t(memory[bpc]) | (uint32_t(memory[bpc+1])<<8) |
                                (uint32_t(memory[bpc+2])<<16) | (uint32_t(memory[bpc+3])<<24);

                if ((inst & 0xFFE0001F) == 0xD4400000) {
                    atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
                    should_exit = true;
                    break;
                }
                if ((inst & 0xFFE0001F) == 0xD4000001) {
                    atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
                    should_exit = true;
                    break;
                }

                window[window_count] = decode(inst);
                bpc += 4;
                window_count++;

                if (window[window_count-1].is_branch) break;
            }

            if (should_exit || window_count == 0) break;

            // Compute dependency matrix using fast lookup
            for (uint32_t i = 0; i < window_count; i++) {
                for (uint32_t j = 0; j < window_count; j++) {
                    if (j <= i) {
                        dep_matrix[i][j] = 0.0;
                    } else {
                        dep_matrix[i][j] = predict_dep_fast(window[i], window[j]);
                    }
                }
            }

            // Cache the block
            block->pc = pc;
            block->count = window_count;
            for (uint32_t i = 0; i < window_count; i++) {
                block->insts[i] = window[i];
                for (uint32_t j = 0; j < window_count; j++) {
                    block->deps[i][j] = dep_matrix[i][j];
                }
            }
            block->valid = 1;
            block->hits = 0;
        }

        if (should_exit || window_count == 0) break;

        // Build producer map for forwarding
        uint8_t producer[32];
        for (int r = 0; r < 32; r++) producer[r] = 0xFF;

        uint8_t prod_rn[MAX_BLOCK_SIZE], prod_rm[MAX_BLOCK_SIZE];
        for (uint32_t i = 0; i < window_count; i++) {
            prod_rn[i] = producer[window[i].rn];
            prod_rm[i] = producer[window[i].rm];
            if (window[i].rd < 31) {
                producer[window[i].rd] = i;
            }
        }

        // OoO execution with soft dependencies
        int64_t results[MAX_BLOCK_SIZE];
        bool executed[MAX_BLOCK_SIZE];
        bool branch_taken_arr[MAX_BLOCK_SIZE];
        for (uint32_t i = 0; i < MAX_BLOCK_SIZE; i++) {
            executed[i] = false;
            results[i] = 0;
            branch_taken_arr[i] = false;
        }

        uint32_t committed = 0;
        bool branch_encountered = false;
        uint32_t branch_idx = 0;

        while (committed < window_count && !should_exit) {
            uint32_t executed_this_cycle = 0;

            // Find ready instructions
            for (uint32_t i = 0; i < window_count; i++) {
                if (executed[i]) continue;

                float readiness = 1.0;
                for (uint32_t j = 0; j < i; j++) {
                    if (!executed[j]) {
                        readiness *= (1.0 - dep_matrix[j][i]);
                    }
                }

                if (readiness > 0.5) {
                    bool taken = false;
                    exec_inst(window[i], regs, N, Z, C, V, memory, memory_size,
                              results[i], taken, prod_rn[i], prod_rm[i], results, executed);

                    executed[i] = true;
                    branch_taken_arr[i] = taken;
                    executed_this_cycle++;

                    if (taken && !branch_encountered) {
                        branch_encountered = true;
                        branch_idx = i;
                    }
                }
            }

            // In-order commit
            while (committed < window_count && executed[committed]) {
                if (window[committed].rd < 31) {
                    regs[window[committed].rd] = results[committed];
                }
                committed++;
                cycles++;

                if (branch_encountered && committed > branch_idx) break;
            }

            if (executed_this_cycle > 1) parallel_execs += executed_this_cycle;

            if (branch_encountered && committed > branch_idx) break;

            // Force progress if stuck
            if (executed_this_cycle == 0 && committed < window_count) {
                for (uint32_t i = committed; i < window_count; i++) {
                    if (!executed[i]) {
                        bool taken = false;
                        exec_inst(window[i], regs, N, Z, C, V, memory, memory_size,
                                  results[i], taken, prod_rn[i], prod_rm[i], results, executed);
                        executed[i] = true;
                        branch_taken_arr[i] = taken;
                        break;
                    }
                }
            }
        }

        // Update PC
        if (branch_encountered) {
            uint32_t inst = window[branch_idx].raw;
            if ((inst & 0xFF000010) == 0x54000000) {
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                pc = pc + branch_idx * 4 + imm19 * 4;
            }
            else if ((inst & 0xFC000000) == 0x14000000) {
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                pc = pc + branch_idx * 4 + imm26 * 4;
            }
            else if ((inst & 0xFC000000) == 0x94000000) {
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                regs[30] = (int64_t)(pc + (branch_idx + 1) * 4);
                pc = pc + branch_idx * 4 + imm26 * 4;
            }
            else if ((inst & 0xFF000000) == 0xB4000000 || (inst & 0xFF000000) == 0xB5000000) {
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                pc = pc + branch_idx * 4 + imm19 * 4;
            }
            else {
                pc += committed * 4;
            }
        } else {
            pc += committed * 4;
        }
    }

    // Write back
    for (int i = 0; i < 32; i++) registers[i] = regs[i];
    pc_ptr[0] = pc;
    flags[0] = N; flags[1] = Z; flags[2] = C; flags[3] = V;

    atomic_fetch_add_explicit((device atomic_uint*)total_cycles, cycles, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)batch_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[0], cache_hits, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[1], cache_misses, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[2], parallel_execs, memory_order_relaxed);

    uint32_t sig = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (sig == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

const CACHE_ENTRIES: usize = 256;
const CACHED_BLOCK_SIZE: usize = 512;
const CACHE_SIZE: usize = CACHE_ENTRIES * CACHED_BLOCK_SIZE;

/// Neural Hybrid result
#[pyclass]
#[derive(Debug, Clone)]
pub struct NeuralHybridResult {
    #[pyo3(get)]
    pub total_cycles: u32,
    #[pyo3(get)]
    pub batch_count: u32,
    #[pyo3(get)]
    pub cache_hits: u32,
    #[pyo3(get)]
    pub cache_misses: u32,
    #[pyo3(get)]
    pub parallel_executions: u32,
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
impl NeuralHybridResult {
    fn __repr__(&self) -> String {
        format!("NeuralHybridResult(cycles={}, ips={:.0}, cache_hit={:.1}%)",
                self.total_cycles, self.ips, self.cache_hit_rate * 100.0)
    }
}

/// Neural Hybrid CPU - BBCache speed + Neural OoO
#[pyclass(unsendable)]
pub struct NeuralHybridCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    memory_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    cycles_per_batch_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    mem_size_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    signal_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_cycles_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_count_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    stats_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    cache_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    memory_size: usize,
    cycles_per_batch: u32,
}

#[pymethods]
impl NeuralHybridCPU {
    #[new]
    #[pyo3(signature = (memory_size=4*1024*1024, cycles_per_batch=10_000_000))]
    fn new(memory_size: usize, cycles_per_batch: u32) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[NeuralHybridCPU] Using device: {:?}", device.name());
        println!("[NeuralHybridCPU] Compiling Neural Hybrid shader...");

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        let source = NSString::from_str(NEURAL_HYBRID_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let func_name = NSString::from_str("cpu_execute_neural_hybrid");
        let function = library.newFunctionWithName(&func_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("Function not found".to_string()))?;

        let pipeline = device.newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        let opts = MTLResourceOptions::StorageModeShared;

        let memory_buf = device.newBufferWithLength_options(memory_size, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let registers_buf = device.newBufferWithLength_options(32 * 8, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let pc_buf = device.newBufferWithLength_options(8, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let flags_buf = device.newBufferWithLength_options(16, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cycles_per_batch_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let mem_size_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let signal_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let total_cycles_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let batch_count_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let stats_buf = device.newBufferWithLength_options(4 * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cache_buf = device.newBufferWithLength_options(CACHE_SIZE, opts)
            .ok_or(MetalError::BufferCreationFailed)?;

        unsafe {
            std::ptr::write_bytes(cache_buf.contents().as_ptr() as *mut u8, 0, CACHE_SIZE);
            let ptr = cycles_per_batch_buf.contents().as_ptr() as *mut u32;
            *ptr = cycles_per_batch;
            let ptr = mem_size_buf.contents().as_ptr() as *mut u32;
            *ptr = memory_size as u32;
        }

        println!("[NeuralHybridCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[NeuralHybridCPU] Features: BBCache + Neural OoO + Lookup Tables");

        Ok(Self {
            device,
            command_queue,
            pipeline,
            memory_buf,
            registers_buf,
            pc_buf,
            flags_buf,
            cycles_per_batch_buf,
            mem_size_buf,
            signal_buf,
            total_cycles_buf,
            batch_count_buf,
            stats_buf,
            cache_buf,
            memory_size,
            cycles_per_batch,
        })
    }

    fn load_program(&self, program: Vec<u8>, address: u64) -> PyResult<()> {
        if address as usize + program.len() > self.memory_size {
            return Err(PyRuntimeError::new_err("Program exceeds memory"));
        }
        unsafe {
            let mem = self.memory_buf.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(program.as_ptr(), mem.add(address as usize), program.len());
        }
        println!("[NeuralHybridCPU] Loaded {} bytes at 0x{:x}", program.len(), address);
        Ok(())
    }

    fn set_pc(&self, pc: u64) {
        unsafe { *(self.pc_buf.contents().as_ptr() as *mut u64) = pc; }
    }

    fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buf.contents().as_ptr() as *const u64) }
    }

    fn set_register(&self, reg: usize, value: i64) -> PyResult<()> {
        if reg >= 32 { return Err(PyRuntimeError::new_err("Invalid register")); }
        unsafe { *(self.registers_buf.contents().as_ptr() as *mut i64).add(reg) = value; }
        Ok(())
    }

    fn get_register(&self, reg: usize) -> PyResult<i64> {
        if reg >= 32 { return Err(PyRuntimeError::new_err("Invalid register")); }
        unsafe { Ok(*(self.registers_buf.contents().as_ptr() as *const i64).add(reg)) }
    }

    fn reset(&self) {
        unsafe {
            std::ptr::write_bytes(self.registers_buf.contents().as_ptr() as *mut u8, 0, 32 * 8);
            std::ptr::write_bytes(self.flags_buf.contents().as_ptr() as *mut u8, 0, 16);
            *(self.pc_buf.contents().as_ptr() as *mut u64) = 0;
        }
    }

    #[pyo3(signature = (max_batches=100, timeout_seconds=10.0))]
    fn execute(&self, max_batches: u32, timeout_seconds: f64) -> PyResult<NeuralHybridResult> {
        let start = Instant::now();

        unsafe {
            *(self.signal_buf.contents().as_ptr() as *mut u32) = 0;
            *(self.total_cycles_buf.contents().as_ptr() as *mut u32) = 0;
            *(self.batch_count_buf.contents().as_ptr() as *mut u32) = 0;
            std::ptr::write_bytes(self.stats_buf.contents().as_ptr() as *mut u8, 0, 16);
        }

        let mut batch = 0u32;
        while batch < max_batches {
            if start.elapsed().as_secs_f64() > timeout_seconds { break; }

            let cmd = self.command_queue.commandBuffer()
                .ok_or_else(|| PyRuntimeError::new_err("Failed to create command buffer"))?;

            let encoder = cmd.computeCommandEncoder()
                .ok_or_else(|| PyRuntimeError::new_err("Failed to create encoder"))?;

            encoder.setComputePipelineState(&self.pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.memory_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.registers_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.pc_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&self.flags_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&self.cycles_per_batch_buf), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&self.mem_size_buf), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&self.signal_buf), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(&self.total_cycles_buf), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(&self.batch_count_buf), 0, 8);
                encoder.setBuffer_offset_atIndex(Some(&self.stats_buf), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(&self.cache_buf), 0, 10);

                let grid = MTLSize { width: 1, height: 1, depth: 1 };
                let tg = MTLSize { width: 1, height: 1, depth: 1 };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            }
            encoder.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let signal = unsafe { *(self.signal_buf.contents().as_ptr() as *const u32) };
            if signal == 1 || signal == 2 { break; }

            unsafe { *(self.signal_buf.contents().as_ptr() as *mut u32) = 0; }
            batch += 1;
        }

        let elapsed = start.elapsed().as_secs_f64();
        let total_cycles = unsafe { *(self.total_cycles_buf.contents().as_ptr() as *const u32) };
        let batch_count = unsafe { *(self.batch_count_buf.contents().as_ptr() as *const u32) };
        let signal = unsafe { *(self.signal_buf.contents().as_ptr() as *const u32) };

        let stats = unsafe { std::slice::from_raw_parts(self.stats_buf.contents().as_ptr() as *const u32, 4) };
        let cache_hits = stats[0];
        let cache_misses = stats[1];
        let parallel_executions = stats[2];

        let ips = if elapsed > 0.0 { total_cycles as f64 / elapsed } else { 0.0 };
        let total_accesses = cache_hits + cache_misses;
        let cache_hit_rate = if total_accesses > 0 {
            cache_hits as f64 / total_accesses as f64
        } else { 0.0 };

        Ok(NeuralHybridResult {
            total_cycles,
            batch_count,
            cache_hits,
            cache_misses,
            parallel_executions,
            signal,
            elapsed_seconds: elapsed,
            ips,
            cache_hit_rate,
        })
    }
}

pub fn register_neural_hybrid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuralHybridCPU>()?;
    m.add_class::<NeuralHybridResult>()?;
    Ok(())
}
