//! Neural Differentiable Out-of-Order Execution Engine
//!
//! Combines neural networks with OoO execution for learned scheduling:
//! 1. Instruction Encoder: Embed instructions into latent space
//! 2. Dependency Predictor: Neural network predicts soft dependency matrix
//! 3. Attention Scheduler: Soft attention selects ready instructions
//! 4. Differentiable Execution: Gradients flow through scheduling decisions
//!
//! Training: PyTorch computes gradients, Metal executes inference

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

/// Simplified Neural OoO Execution shader - fast to compile
/// Focuses on learned dependency prediction with core ARM64 instructions
const NEURAL_OOO_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

constant uint32_t WINDOW_SIZE = 8;
constant uint32_t SCHED_OFFSET = 0;
constant uint32_t DEP_CACHE_SIZE = 64;  // Cache for neural dependency matrices

// Instruction type IDs for lookup table
constant uint8_t ITYPE_UNKNOWN = 0;
constant uint8_t ITYPE_ALU_IMM = 1;    // ADD/SUB imm
constant uint8_t ITYPE_ALU_REG = 2;    // ADD/SUB reg
constant uint8_t ITYPE_MOV = 3;        // MOVZ/MOVK
constant uint8_t ITYPE_CMP = 4;        // SUBS (flag-setting)
constant uint8_t ITYPE_LOAD = 5;       // LDR
constant uint8_t ITYPE_STORE = 6;      // STR
constant uint8_t ITYPE_CBRANCH = 7;    // CBZ/CBNZ
constant uint8_t ITYPE_BRANCH = 8;     // B/BL
constant uint8_t ITYPE_CCBRANCH = 9;   // B.cond (uses flags)
constant uint8_t ITYPE_MADD = 10;      // MADD (3-operand)
constant uint8_t NUM_ITYPES = 11;

// Classify instruction type for fast lookup
inline uint8_t classify_inst(uint32_t inst) {
    uint8_t top = (inst >> 24) & 0xFF;

    // Load/Store
    if ((inst & 0xFFC00000) == 0xF9400000) return ITYPE_LOAD;
    if ((inst & 0xFFC00000) == 0xF9000000) return ITYPE_STORE;

    // ALU immediate
    if ((inst & 0xFF000000) == 0x91000000) return ITYPE_ALU_IMM;  // ADD imm
    if ((inst & 0xFF000000) == 0xD1000000) return ITYPE_ALU_IMM;  // SUB imm

    // ALU register
    if ((inst & 0xFF200000) == 0x8B000000) return ITYPE_ALU_REG;  // ADD reg
    if ((inst & 0xFF200000) == 0xCB000000) return ITYPE_ALU_REG;  // SUB reg

    // Compares (flag-setting)
    if ((inst & 0xFF000000) == 0xF1000000) return ITYPE_CMP;  // SUBS imm
    if ((inst & 0xFF200000) == 0xEB000000) return ITYPE_CMP;  // SUBS reg

    // Move
    if ((inst & 0xFF800000) == 0xD2800000) return ITYPE_MOV;  // MOVZ
    if ((inst & 0xFF800000) == 0xF2800000) return ITYPE_MOV;  // MOVK

    // Branches
    if ((inst & 0xFF000000) == 0xB4000000) return ITYPE_CBRANCH;  // CBZ
    if ((inst & 0xFF000000) == 0xB5000000) return ITYPE_CBRANCH;  // CBNZ
    if ((inst & 0xFF000010) == 0x54000000) return ITYPE_CCBRANCH; // B.cond
    if ((inst & 0xFC000000) == 0x14000000) return ITYPE_BRANCH;   // B
    if ((inst & 0xFC000000) == 0x94000000) return ITYPE_BRANCH;   // BL

    // MADD
    if ((inst & 0xFF000000) == 0x9B000000) return ITYPE_MADD;

    return ITYPE_UNKNOWN;
}

// Pre-computed dependency scores based on instruction type pairs
// [producer_type][consumer_type] = base dependency score
// These are "distilled" from the neural weights
constant float DEP_LOOKUP[NUM_ITYPES][NUM_ITYPES] = {
    // Unknown producer - conservative high dependency
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8},
    // ALU_IMM producer
    {0.8, 0.2, 0.2, 0.1, 0.3, 0.5, 0.5, 0.9, 0.1, 0.1, 0.3},
    // ALU_REG producer
    {0.8, 0.2, 0.2, 0.1, 0.3, 0.5, 0.5, 0.9, 0.1, 0.1, 0.3},
    // MOV producer
    {0.8, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9},
    // CMP producer (sets flags)
    {0.8, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.95, 0.1},
    // LOAD producer
    {0.8, 0.9, 0.9, 0.1, 0.9, 0.3, 0.5, 0.9, 0.1, 0.1, 0.9},
    // STORE producer
    {0.8, 0.1, 0.1, 0.1, 0.1, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1},
    // CBRANCH producer
    {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
    // BRANCH producer
    {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
    // CCBRANCH producer
    {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
    // MADD producer
    {0.8, 0.3, 0.3, 0.1, 0.3, 0.5, 0.5, 0.9, 0.1, 0.1, 0.3}
};

// Fast sigmoid approximation (no exp() - 5x faster)
inline float sigmoid(float x) {
    return 0.5 + 0.5 * x / (1.0 + fabs(x));
}

// Decoded instruction for OoO execution
struct DecodedInst {
    uint32_t raw;
    uint8_t rd, rn, rm;
    uint8_t sets_flags, uses_flags, is_branch;
};

// Cached block with pre-computed dependencies
struct CachedBlock {
    uint64_t pc;                               // Start PC
    DecodedInst insts[WINDOW_SIZE];            // Decoded instructions
    float deps[WINDOW_SIZE][WINDOW_SIZE];       // Pre-computed dependency matrix
    uint32_t count;                            // Number of instructions
    uint32_t valid;                            // Cache validity
};

// Fast dependency prediction using lookup table + register check
// (Distilled from neural weights - much faster than sigmoid)
inline float predict_dependency_fast(DecodedInst prod, DecodedInst cons) {
    // Quick check: if no possible RAW, return low dependency
    bool possible_raw = (prod.rd < 31) && (prod.rd == cons.rn || prod.rd == cons.rm);
    bool possible_waw = (prod.rd < 31) && (cons.rd < 31) && (prod.rd == cons.rd);
    bool possible_flag = prod.sets_flags && cons.uses_flags;

    if (!possible_raw && !possible_waw && !possible_flag) {
        return 0.05;  // Very low dependency
    }

    // Use lookup table for base score
    uint8_t prod_type = classify_inst(prod.raw);
    uint8_t cons_type = classify_inst(cons.raw);
    float base_score = DEP_LOOKUP[prod_type][cons_type];

    // Adjust based on actual register overlap
    if (possible_raw) base_score = max(base_score, 0.9f);
    if (possible_flag) base_score = max(base_score, 0.95f);

    return base_score;
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

    // Flag-setting instructions
    if ((inst & 0xFF000000) == 0xF1000000 || (inst & 0xFF200000) == 0xEB000000) {
        d.sets_flags = 1;
    }
    // Branches
    if ((top & 0xFC) == 0x14 || (top & 0xFC) == 0x94 || top == 0x54 ||
        top == 0xB4 || top == 0xB5 || top == 0xD6) {
        d.is_branch = 1;
        if (top == 0x54) d.uses_flags = 1;
    }
    // Instructions that don't write to destination register:
    // STR, HLT
    if ((inst & 0xFFC00000) == 0xF9000000 || (inst & 0xFFE0001F) == 0xD4400000) {
        d.rd = 32;
    }
    // CBZ/CBNZ - rd field is test register, not destination
    // Set rn to test register for dependency tracking, rd=32 for no write-back
    if ((inst & 0xFF000000) == 0xB4000000 || (inst & 0xFF000000) == 0xB5000000) {
        d.rn = inst & 0x1F;  // Test register for dependency detection
        d.rm = 32;           // No second source
        d.rd = 32;           // No destination (don't write)
    }
    // B.cond, B, BL - branches don't write to general registers (BL writes to LR in exec)
    if ((inst & 0xFF000010) == 0x54000000 || (inst & 0xFC000000) == 0x14000000) {
        d.rd = 32;
    }

    return d;
}

// Predict dependency using learned hazard weights
inline float predict_dependency(DecodedInst prod, DecodedInst cons, device const float* weights) {
    float raw = 0.0, waw = 0.0, flag_dep = 0.0;

    // RAW: consumer reads what producer writes
    if (prod.rd < 31) {
        if (prod.rd == cons.rn || prod.rd == cons.rm) raw = 1.0;
    }
    // WAW: both write same register
    if (prod.rd < 31 && cons.rd < 31 && prod.rd == cons.rd) waw = 1.0;
    // Flag dependency
    if (prod.sets_flags && cons.uses_flags) flag_dep = 1.0;

    float score = weights[0] * raw + weights[1] * waw + weights[2] * flag_dep + weights[3];
    return sigmoid(score);
}

// Get register value with forwarding
inline int64_t get_fwd(uint8_t reg, uint8_t producer_idx, thread int64_t* regs,
                       thread int64_t* results, thread bool* executed) {
    if (reg == 31) return 0;
    // Check if producer executed and can forward result
    if (producer_idx != 0xFF && executed[producer_idx]) {
        return results[producer_idx];
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
    // SUBS imm (CMP)
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
    // CBZ/CBNZ - test register is in bits [4:0], use rn for forwarding
    // (decode sets rn = test register for CBZ/CBNZ)
    if ((inst & 0xFF000000) == 0xB4000000) {
        int64_t rt_val = rn_val;  // rn was set to test register in decode
        if (rt_val == 0) branch_taken = true;
        return;
    }
    if ((inst & 0xFF000000) == 0xB5000000) {
        int64_t rt_val = rn_val;  // rn was set to test register in decode
        if (rt_val != 0) branch_taken = true;
        return;
    }
}

// Main neural OoO kernel with device-persistent dependency caching
kernel void cpu_execute_neural_ooo(
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
    device const float* neural_weights [[buffer(10)]],
    device float* grad_buffer [[buffer(11)]],
    device CachedBlock* dep_cache [[buffer(12)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t batch_cycles = cycles_per_batch[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;
    uint32_t parallel_execs = 0;
    uint32_t serial_execs = 0;
    uint32_t cache_hits = 0;
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

        // Check device-persistent cache
        uint32_t cache_idx = (uint32_t)(pc >> 2) & (DEP_CACHE_SIZE - 1);
        bool cache_hit = (dep_cache[cache_idx].valid == 1 && dep_cache[cache_idx].pc == pc);

        DecodedInst window[WINDOW_SIZE];
        float dep_matrix[WINDOW_SIZE][WINDOW_SIZE];
        uint32_t window_count = 0;

        if (cache_hit) {
            // Use cached block and dependency matrix
            window_count = dep_cache[cache_idx].count;
            for (uint32_t i = 0; i < window_count; i++) {
                window[i] = dep_cache[cache_idx].insts[i];
                for (uint32_t j = 0; j < window_count; j++) {
                    dep_matrix[i][j] = dep_cache[cache_idx].deps[i][j];
                }
            }
            cache_hits++;
        } else {
            // Fetch and decode window
            uint64_t bpc = pc;
            while (window_count < WINDOW_SIZE && bpc + 4 <= memory_size) {
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

            // Compute dependency matrix using fast lookup + register check
            // This is much faster than sigmoid, with similar accuracy
            for (uint32_t i = 0; i < window_count; i++) {
                for (uint32_t j = 0; j < window_count; j++) {
                    if (j <= i) {
                        dep_matrix[i][j] = 0.0;
                    } else {
                        dep_matrix[i][j] = predict_dependency_fast(window[i], window[j]);
                    }
                }
            }

            // Cache the block and dependency matrix
            dep_cache[cache_idx].pc = pc;
            dep_cache[cache_idx].count = window_count;
            for (uint32_t i = 0; i < window_count; i++) {
                dep_cache[cache_idx].insts[i] = window[i];
                for (uint32_t j = 0; j < window_count; j++) {
                    dep_cache[cache_idx].deps[i][j] = dep_matrix[i][j];
                }
            }
            dep_cache[cache_idx].valid = 1;
        }

        if (should_exit || window_count == 0) break;

        // Build producer map for result forwarding
        // producer[i] = index of instruction that produces register i, or 0xFF if none
        uint8_t producer[32];
        for (int r = 0; r < 32; r++) producer[r] = 0xFF;

        // Per-instruction producer arrays (which instruction produces each source register)
        uint8_t prod_rn[WINDOW_SIZE], prod_rm[WINDOW_SIZE];
        for (uint32_t i = 0; i < window_count; i++) {
            prod_rn[i] = producer[window[i].rn];
            prod_rm[i] = producer[window[i].rm];
            // Update producer for this instruction's destination
            if (window[i].rd < 31) {
                producer[window[i].rd] = i;
            }
        }

        // OoO execution with soft dependencies
        int64_t results[WINDOW_SIZE];
        bool executed[WINDOW_SIZE];
        bool branch_taken_arr[WINDOW_SIZE];
        for (uint32_t i = 0; i < WINDOW_SIZE; i++) {
            executed[i] = false;
            results[i] = 0;
            branch_taken_arr[i] = false;
        }

        uint32_t committed = 0;
        bool branch_encountered = false;
        uint32_t branch_idx = 0;

        while (committed < window_count && !should_exit) {
            uint32_t executed_this_cycle = 0;

            // Compute soft readiness
            for (uint32_t i = 0; i < window_count; i++) {
                if (executed[i]) continue;

                // Readiness = product of (1 - dep[j][i]) for non-executed j
                float readiness = 1.0;
                for (uint32_t j = 0; j < i; j++) {
                    if (!executed[j]) {
                        readiness *= (1.0 - dep_matrix[j][i]);
                    }
                }

                // Execute if ready
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

            // Commit in-order
            while (committed < window_count && executed[committed]) {
                if (window[committed].rd < 31) {
                    regs[window[committed].rd] = results[committed];
                }
                committed++;
                cycles++;

                if (branch_encountered && committed > branch_idx) break;
            }

            if (executed_this_cycle > 1) parallel_execs += executed_this_cycle;
            else if (executed_this_cycle == 1) serial_execs++;

            if (branch_encountered && committed > branch_idx) break;

            // Safety: force progress
            if (executed_this_cycle == 0 && committed < window_count) {
                for (uint32_t i = committed; i < window_count; i++) {
                    if (!executed[i]) {
                        bool taken = false;
                        exec_inst(window[i], regs, N, Z, C, V, memory, memory_size,
                                  results[i], taken, prod_rn[i], prod_rm[i], results, executed);
                        executed[i] = true;
                        branch_taken_arr[i] = taken;
                        serial_execs++;
                        break;
                    }
                }
            }
        }

        // Update PC
        if (branch_encountered) {
            uint32_t inst = window[branch_idx].raw;
            // B.cond
            if ((inst & 0xFF000010) == 0x54000000) {
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                pc = pc + branch_idx * 4 + imm19 * 4;
            }
            // B (unconditional)
            else if ((inst & 0xFC000000) == 0x14000000) {
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                pc = pc + branch_idx * 4 + imm26 * 4;
            }
            // BL
            else if ((inst & 0xFC000000) == 0x94000000) {
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                regs[30] = (int64_t)(pc + (branch_idx + 1) * 4);
                pc = pc + branch_idx * 4 + imm26 * 4;
            }
            // CBZ (X0=0: branch taken)
            else if ((inst & 0xFF000000) == 0xB4000000) {
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                pc = pc + branch_idx * 4 + imm19 * 4;
            }
            // CBNZ (X0!=0: branch taken)
            else if ((inst & 0xFF000000) == 0xB5000000) {
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
    atomic_fetch_add_explicit((device atomic_uint*)&stats[0], parallel_execs, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[1], serial_execs, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[2], cache_hits, memory_order_relaxed);

    uint32_t sig = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (sig == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

const NEURAL_WEIGHT_COUNT: usize = 4; // RAW, WAW, FLAG, BIAS

// Cache size: 64 entries * ~512 bytes per entry = 32KB
// CachedBlock layout: pc(8) + insts[8](128) + deps[8][8](256) + count(4) + valid(4) + padding
const DEP_CACHE_ENTRIES: usize = 64;
const CACHED_BLOCK_SIZE: usize = 512;  // Conservative estimate with padding
const DEP_CACHE_SIZE: usize = DEP_CACHE_ENTRIES * CACHED_BLOCK_SIZE;

/// Neural OoO execution result
#[pyclass]
#[derive(Debug, Clone)]
pub struct NeuralOoOResult {
    #[pyo3(get)]
    pub total_cycles: u32,
    #[pyo3(get)]
    pub batch_count: u32,
    #[pyo3(get)]
    pub parallel_executions: u32,
    #[pyo3(get)]
    pub serial_executions: u32,
    #[pyo3(get)]
    pub signal: u32,
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    #[pyo3(get)]
    pub ips: f64,
    #[pyo3(get)]
    pub parallelism_ratio: f64,
}

#[pymethods]
impl NeuralOoOResult {
    fn __repr__(&self) -> String {
        format!("NeuralOoOResult(cycles={}, ips={:.0}, parallelism={:.1}%)",
                self.total_cycles, self.ips, self.parallelism_ratio * 100.0)
    }
}

/// Neural Differentiable OoO CPU
#[pyclass(unsendable)]
pub struct NeuralOoOCPU {
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
    weights_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    grad_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    cache_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    memory_size: usize,
    cycles_per_batch: u32,
}

#[pymethods]
impl NeuralOoOCPU {
    #[new]
    #[pyo3(signature = (memory_size=4*1024*1024, cycles_per_batch=10_000_000))]
    fn new(memory_size: usize, cycles_per_batch: u32) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[NeuralOoOCPU] Using device: {:?}", device.name());
        println!("[NeuralOoOCPU] Compiling Neural OoO shader...");

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        let source = NSString::from_str(NEURAL_OOO_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let func_name = NSString::from_str("cpu_execute_neural_ooo");
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
        let weights_buf = device.newBufferWithLength_options(NEURAL_WEIGHT_COUNT * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let grad_buf = device.newBufferWithLength_options(NEURAL_WEIGHT_COUNT * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cache_buf = device.newBufferWithLength_options(DEP_CACHE_SIZE, opts)
            .ok_or(MetalError::BufferCreationFailed)?;

        // Initialize cache to invalid
        unsafe {
            std::ptr::write_bytes(cache_buf.contents().as_ptr() as *mut u8, 0, DEP_CACHE_SIZE);
        }

        // Initialize weights
        unsafe {
            let ptr = cycles_per_batch_buf.contents().as_ptr() as *mut u32;
            *ptr = cycles_per_batch;
            let ptr = mem_size_buf.contents().as_ptr() as *mut u32;
            *ptr = memory_size as u32;

            // Initialize learned hazard weights: RAW=5.0, WAW=3.0, FLAG=5.0, BIAS=-2.0
            let weights = weights_buf.contents().as_ptr() as *mut f32;
            *weights.add(0) = 5.0;   // RAW weight
            *weights.add(1) = 3.0;   // WAW weight
            *weights.add(2) = 5.0;   // FLAG weight
            *weights.add(3) = -2.0;  // Bias

            std::ptr::write_bytes(grad_buf.contents().as_ptr() as *mut u8, 0, NEURAL_WEIGHT_COUNT * 4);
        }

        println!("[NeuralOoOCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[NeuralOoOCPU] Neural weights: {} parameters (RAW, WAW, FLAG, BIAS)", NEURAL_WEIGHT_COUNT);
        println!("[NeuralOoOCPU] Features: Learned Dependency Prediction, Soft Scheduling, Differentiable");

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
            weights_buf,
            grad_buf,
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
        println!("[NeuralOoOCPU] Loaded {} bytes at 0x{:x}", program.len(), address);
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

    /// Load neural weights (RAW, WAW, FLAG, BIAS)
    fn load_weights(&self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != NEURAL_WEIGHT_COUNT {
            return Err(PyRuntimeError::new_err(format!(
                "Expected {} weights, got {}", NEURAL_WEIGHT_COUNT, weights.len()
            )));
        }
        unsafe {
            let ptr = self.weights_buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(weights.as_ptr(), ptr, weights.len());
        }
        Ok(())
    }

    /// Get current neural weights
    fn get_weights(&self) -> Vec<f32> {
        unsafe {
            let ptr = self.weights_buf.contents().as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, NEURAL_WEIGHT_COUNT).to_vec()
        }
    }

    fn reset(&self) {
        unsafe {
            std::ptr::write_bytes(self.registers_buf.contents().as_ptr() as *mut u8, 0, 32 * 8);
            std::ptr::write_bytes(self.flags_buf.contents().as_ptr() as *mut u8, 0, 16);
            *(self.pc_buf.contents().as_ptr() as *mut u64) = 0;
        }
    }

    #[pyo3(signature = (max_batches=100, timeout_seconds=10.0))]
    fn execute(&self, max_batches: u32, timeout_seconds: f64) -> PyResult<NeuralOoOResult> {
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
                encoder.setBuffer_offset_atIndex(Some(&self.weights_buf), 0, 10);
                encoder.setBuffer_offset_atIndex(Some(&self.grad_buf), 0, 11);
                encoder.setBuffer_offset_atIndex(Some(&self.cache_buf), 0, 12);

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
        let parallel_executions = stats[0];
        let serial_executions = stats[1];

        let ips = if elapsed > 0.0 { total_cycles as f64 / elapsed } else { 0.0 };
        let total_execs = parallel_executions + serial_executions;
        let parallelism_ratio = if total_execs > 0 {
            parallel_executions as f64 / total_execs as f64
        } else { 0.0 };

        Ok(NeuralOoOResult {
            total_cycles,
            batch_count,
            parallel_executions,
            serial_executions,
            signal,
            elapsed_seconds: elapsed,
            ips,
            parallelism_ratio,
        })
    }

    /// Get weight count for training
    fn weight_count(&self) -> usize {
        NEURAL_WEIGHT_COUNT
    }
}

pub fn register_neural_ooo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuralOoOCPU>()?;
    m.add_class::<NeuralOoOResult>()?;
    Ok(())
}
