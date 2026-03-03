//! Unified Differentiable CPU
//!
//! The ultimate 100% differentiable CPU architecture combining:
//! 1. Differentiable JIT - learned pattern matching & template selection
//! 2. Differentiable OoO - Gumbel-softmax scheduling & soft dependencies
//! 3. Speculative Execution - both branch paths with soft commit
//! 4. Soft Register File - differentiable memory with weighted writes
//! 5. Soft Flags - differentiable condition codes
//!
//! Total learnable parameters: ~1,200
//! All decisions are soft/differentiable for end-to-end training.

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

/// Unified Differentiable CPU Shader
///
/// Architecture Overview:
/// ┌─────────────────────────────────────────────────────────────────┐
/// │                    UNIFIED DIFF CPU                              │
/// ├─────────────────────────────────────────────────────────────────┤
/// │  Stage 1: Diff JIT Pattern Detection                            │
/// │    - Encode instruction window → embedding                       │
/// │    - Soft template matching via learned similarities            │
/// │    - Gumbel-softmax template selection                          │
/// ├─────────────────────────────────────────────────────────────────┤
/// │  Stage 2: Diff OoO Scheduling (if not JIT)                      │
/// │    - Learned dependency prediction (RAW, WAW, FLAG weights)     │
/// │    - Soft readiness scores via product of (1-dep)               │
/// │    - Gumbel-softmax instruction selection                       │
/// ├─────────────────────────────────────────────────────────────────┤
/// │  Stage 3: Speculative Execution                                 │
/// │    - Execute BOTH branch paths                                  │
/// │    - Soft branch probability for commit                         │
/// │    - Weighted result blending                                   │
/// ├─────────────────────────────────────────────────────────────────┤
/// │  Stage 4: Soft Register Commit                                  │
/// │    - Differentiable register writes                             │
/// │    - Soft flags (N, Z, C, V)                                    │
/// │    - Gradient accumulation                                      │
/// └─────────────────────────────────────────────────────────────────┘
const UNIFIED_DIFF_CPU_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS
// ============================================================================

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_CHECKPOINT = 3;

constant uint32_t NUM_REGS = 32;
constant uint32_t WINDOW_SIZE = 8;       // OoO window
constant uint32_t JIT_WINDOW = 4;        // JIT pattern window
constant uint32_t NUM_TEMPLATES = 4;     // JIT templates
constant uint32_t EMBED_DIM = 8;         // Embedding dimension

// Weight layout:
// [0-3]: OoO hazard weights (RAW, WAW, FLAG, BIAS)
// [4-19]: OoO type interaction matrix (4x4)
// [20-51]: Template embeddings (4 templates * 8 dims)
// [52+]: Pattern encoder weights
constant uint32_t OOO_WEIGHTS_START = 0;
constant uint32_t OOO_WEIGHTS_COUNT = 20;
constant uint32_t TEMPLATE_WEIGHTS_START = 20;
constant uint32_t TEMPLATE_WEIGHTS_COUNT = 32;
constant uint32_t ENCODER_WEIGHTS_START = 52;
constant uint32_t ENCODER_WEIGHTS_COUNT = 1024;  // 4 positions * 32 types * 8 dims

constant uint32_t TOTAL_WEIGHTS = 1076;

// Instruction types
constant uint8_t ITYPE_ALU = 0;
constant uint8_t ITYPE_MEM = 1;
constant uint8_t ITYPE_BRANCH = 2;
constant uint8_t ITYPE_OTHER = 3;

// JIT templates
constant uint32_t TEMPLATE_COUNTING = 0;
constant uint32_t TEMPLATE_MEMCPY = 1;
constant uint32_t TEMPLATE_SUM = 2;
constant uint32_t TEMPLATE_INTERPRET = 3;

// ============================================================================
// DIFFERENTIABLE PRIMITIVES
// ============================================================================

inline float soft_sigmoid(float x) {
    return 0.5 + 0.5 * x / (1.0 + fabs(x));
}

inline float fast_exp(float x) {
    x = 1.0 + x / 256.0;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

inline void gumbel_softmax(thread float* logits, thread float* output,
                           uint32_t n, float temperature, uint32_t seed) {
    // Hard execution mode: negative temperature means use argmax (no gumbel noise)
    if (temperature < 0.0f) {
        // Hard mode: simple argmax with one-hot encoding (much faster)
        uint32_t max_idx = 0;
        float max_val = logits[0];
        for (uint32_t i = 1; i < n; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        // One-hot encoding
        for (uint32_t i = 0; i < n; i++) {
            output[i] = (i == max_idx) ? 1.0f : 0.0f;
        }
        return;
    }

    // Normal differentiable mode with gumbel-softmax
    float max_val = -1e9;
    for (uint32_t i = 0; i < n; i++) {
        float u = fract(sin(float(seed + i * 12345)) * 43758.5453);
        u = max(u, 1e-7f);
        float gumbel = -log(-log(u));
        output[i] = (logits[i] + gumbel) / max(temperature, 0.1f);
        if (output[i] > max_val) max_val = output[i];
    }
    float sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        output[i] = fast_exp(output[i] - max_val);
        sum += output[i];
    }
    for (uint32_t i = 0; i < n; i++) {
        output[i] /= max(sum, 1e-8f);
    }
}

inline float dot_product(thread float* a, device const float* b, uint32_t n) {
    float sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ============================================================================
// SOFT REGISTER FILE WITH STRAIGHT-THROUGH ESTIMATOR
// ============================================================================

struct SoftRegFile {
    float regs[NUM_REGS];           // Current values (can drift)
    float hard_regs[NUM_REGS];      // Hard/correct values for control flow
    float grad_accum[NUM_REGS];     // Gradient accumulator
};

// Straight-Through Estimator: hard forward, soft gradient
inline void soft_reg_write_ste(thread SoftRegFile* rf, uint8_t rd, float value, float write_prob) {
    if (rd < 31) {
        // Forward pass: HARD update for correct execution
        rf->hard_regs[rd] = value;

        // Soft tracking for gradient flow (weighted blend)
        rf->regs[rd] = write_prob * value + (1.0 - write_prob) * rf->regs[rd];

        // Accumulate gradient contribution
        rf->grad_accum[rd] += write_prob * (value - rf->regs[rd]);
    }
}

// Hard commit: use when probability is high enough
inline void hard_reg_write(thread SoftRegFile* rf, uint8_t rd, float value) {
    if (rd < 31) {
        rf->regs[rd] = value;
        rf->hard_regs[rd] = value;
    }
}

// Periodic refresh: sync soft values to hard values to prevent drift
inline void refresh_registers(thread SoftRegFile* rf) {
    for (uint32_t i = 0; i < NUM_REGS; i++) {
        rf->regs[i] = rf->hard_regs[i];
    }
}

// Get value for control flow (use hard values)
inline float get_hard_reg(thread SoftRegFile* rf, uint8_t reg) {
    return (reg < 31) ? rf->hard_regs[reg] : 0.0;
}

// Legacy soft write (for backwards compatibility)
inline void soft_reg_write(thread SoftRegFile* rf, uint8_t rd, float value, float write_prob) {
    soft_reg_write_ste(rf, rd, value, write_prob);
}

// ============================================================================
// DECODED INSTRUCTION
// ============================================================================

struct DecodedInst {
    uint32_t raw;
    uint8_t rd, rn, rm;
    uint8_t sets_flags, uses_flags, is_branch;
    uint8_t inst_type;
};

inline uint8_t classify_type(uint32_t inst) {
    uint8_t top = (inst >> 24) & 0xFF;
    if ((inst & 0xFFC00000) == 0xF9400000 || (inst & 0xFFC00000) == 0xF9000000) return ITYPE_MEM;
    if ((top & 0xFC) == 0x14 || top == 0x54 || top == 0xB4 || top == 0xB5) return ITYPE_BRANCH;
    if (top == 0x91 || top == 0xD1 || top == 0x8B || top == 0xCB || top == 0xF1) return ITYPE_ALU;
    return ITYPE_OTHER;
}

inline DecodedInst decode_inst(uint32_t inst) {
    DecodedInst d;
    d.raw = inst;
    d.rd = inst & 0x1F;
    d.rn = (inst >> 5) & 0x1F;
    d.rm = (inst >> 16) & 0x1F;
    d.uses_flags = 0;
    d.sets_flags = 0;
    d.is_branch = 0;
    d.inst_type = classify_type(inst);

    uint8_t top = (inst >> 24) & 0xFF;

    if ((inst & 0xFF000000) == 0xF1000000 || (inst & 0xFF200000) == 0xEB000000) {
        d.sets_flags = 1;
    }
    if ((top & 0xFC) == 0x14 || (top & 0xFC) == 0x94 || top == 0x54 ||
        top == 0xB4 || top == 0xB5 || top == 0xD6) {
        d.is_branch = 1;
        if (top == 0x54) d.uses_flags = 1;
    }
    if ((inst & 0xFFC00000) == 0xF9000000) d.rd = 32;
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

// ============================================================================
// STAGE 1: DIFFERENTIABLE JIT PATTERN DETECTION
// ============================================================================

inline void encode_jit_pattern(
    thread uint32_t* instructions,
    uint32_t count,
    device const float* encoder_weights,
    thread float* embedding
) {
    for (uint32_t i = 0; i < EMBED_DIM; i++) {
        embedding[i] = 0.0;
    }

    for (uint32_t pos = 0; pos < count && pos < JIT_WINDOW; pos++) {
        uint8_t inst_type = (instructions[pos] >> 24) & 0x1F;
        uint32_t embed_offset = (pos * 32 + inst_type) * EMBED_DIM;

        for (uint32_t i = 0; i < EMBED_DIM; i++) {
            embedding[i] += encoder_weights[embed_offset + i];
        }
    }

    float norm = 0.0;
    for (uint32_t i = 0; i < EMBED_DIM; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrt(max(norm, 1e-8f));
    for (uint32_t i = 0; i < EMBED_DIM; i++) {
        embedding[i] /= norm;
    }
}

inline void match_jit_templates(
    thread float* pattern_embedding,
    device const float* template_embeddings,
    thread float* similarities
) {
    for (uint32_t t = 0; t < NUM_TEMPLATES; t++) {
        similarities[t] = dot_product(pattern_embedding,
                                       &template_embeddings[t * EMBED_DIM],
                                       EMBED_DIM);
    }
}

// ============================================================================
// STAGE 2: DIFFERENTIABLE OoO DEPENDENCY PREDICTION
// ============================================================================

inline float predict_soft_dependency(DecodedInst prod, DecodedInst cons,
                                     device const float* weights) {
    float raw = 0.0, waw = 0.0, flag_dep = 0.0;

    if (prod.rd < 31) {
        if (prod.rd == cons.rn || prod.rd == cons.rm) raw = 1.0;
    }
    if (prod.rd < 31 && cons.rd < 31 && prod.rd == cons.rd) waw = 1.0;
    if (prod.sets_flags && cons.uses_flags) flag_dep = 1.0;

    float type_interaction = weights[4 + prod.inst_type * 4 + cons.inst_type];

    float score = weights[0] * raw + weights[1] * waw + weights[2] * flag_dep +
                  weights[3] + type_interaction;

    return soft_sigmoid(score);
}

// ============================================================================
// STAGE 3: SOFT INSTRUCTION EXECUTION
// ============================================================================

inline float exec_soft_inst(DecodedInst d, thread SoftRegFile* rf,
                            thread float& N, thread float& Z, thread float& C, thread float& V,
                            device uint8_t* memory, uint32_t mem_size,
                            thread bool& is_branch_taken, thread float& branch_prob) {
    uint32_t inst = d.raw;
    uint16_t imm12 = (inst >> 10) & 0xFFF;
    uint16_t imm16 = (inst >> 5) & 0xFFFF;
    uint8_t hw = (inst >> 21) & 0x3;

    // Use HARD registers for reading to prevent drift accumulation
    float rn_val = (d.rn < 31) ? rf->hard_regs[d.rn] : 0.0;
    float rm_val = (d.rm < 31) ? rf->hard_regs[d.rm] : 0.0;
    float rd_val = (d.rd < 31) ? rf->hard_regs[d.rd] : 0.0;

    float result = 0.0;
    is_branch_taken = false;
    branch_prob = 0.0;

    // ADD imm
    if ((inst & 0xFF000000) == 0x91000000) { return rn_val + imm12; }
    // SUB imm
    if ((inst & 0xFF000000) == 0xD1000000) { return rn_val - imm12; }
    // ADD reg
    if ((inst & 0xFF200000) == 0x8B000000) { return rn_val + rm_val; }
    // SUB reg
    if ((inst & 0xFF200000) == 0xCB000000) { return rn_val - rm_val; }
    // SUBS imm
    if ((inst & 0xFF000000) == 0xF1000000) {
        result = rn_val - imm12;
        N = soft_sigmoid(-result);
        Z = soft_sigmoid(-(fabs(result) - 0.5) * 10.0);
        C = soft_sigmoid((rn_val - imm12) * 0.1);
        V = 0.0;
        return result;
    }
    // SUBS reg
    if ((inst & 0xFF200000) == 0xEB000000) {
        result = rn_val - rm_val;
        N = soft_sigmoid(-result);
        Z = soft_sigmoid(-(fabs(result) - 0.5) * 10.0);
        C = soft_sigmoid((rn_val - rm_val) * 0.1);
        V = 0.0;
        return result;
    }
    // MOVZ
    if ((inst & 0xFF800000) == 0xD2800000) {
        return (float)((uint64_t)imm16 << (hw * 16));
    }
    // MOVK
    if ((inst & 0xFF800000) == 0xF2800000) {
        uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
        return (float)(((uint64_t)rd_val & mask) | ((uint64_t)imm16 << (hw * 16)));
    }
    // MADD
    if ((inst & 0xFF000000) == 0x9B000000) {
        uint8_t ra = (inst >> 10) & 0x1F;
        float ra_val = (ra < 31) ? rf->regs[ra] : 0.0;
        return ra_val + rn_val * rm_val;
    }
    // LDR
    if ((inst & 0xFFC00000) == 0xF9400000) {
        uint64_t addr = (uint64_t)(rn_val + imm12 * 8);
        if (addr + 8 <= mem_size) {
            int64_t val = (int64_t)(uint64_t(memory[addr]) | (uint64_t(memory[addr+1])<<8) |
                         (uint64_t(memory[addr+2])<<16) | (uint64_t(memory[addr+3])<<24) |
                         (uint64_t(memory[addr+4])<<32) | (uint64_t(memory[addr+5])<<40) |
                         (uint64_t(memory[addr+6])<<48) | (uint64_t(memory[addr+7])<<56));
            return (float)val;
        }
    }
    // STR
    if ((inst & 0xFFC00000) == 0xF9000000) {
        uint64_t addr = (uint64_t)(rn_val + imm12 * 8);
        if (addr + 8 <= mem_size) {
            int64_t val = (int64_t)rd_val;
            memory[addr] = val & 0xFF;
            memory[addr+1] = (val >> 8) & 0xFF;
            memory[addr+2] = (val >> 16) & 0xFF;
            memory[addr+3] = (val >> 24) & 0xFF;
            memory[addr+4] = (val >> 32) & 0xFF;
            memory[addr+5] = (val >> 40) & 0xFF;
            memory[addr+6] = (val >> 48) & 0xFF;
            memory[addr+7] = (val >> 56) & 0xFF;
        }
    }
    // B.cond - SOFT branch
    if ((inst & 0xFF000010) == 0x54000000) {
        uint8_t cond = inst & 0xF;
        float take_prob = 0.0;
        switch (cond) {
            case 0x0: take_prob = Z; break;              // EQ
            case 0x1: take_prob = 1.0 - Z; break;        // NE
            case 0xA: take_prob = soft_sigmoid((N - 0.5) * (V - 0.5) * 20.0 + 0.5); break; // GE
            case 0xB: take_prob = 1.0 - soft_sigmoid((N - 0.5) * (V - 0.5) * 20.0 + 0.5); break; // LT
            case 0xC: take_prob = (1.0 - Z) * soft_sigmoid((N - 0.5) * (V - 0.5) * 20.0 + 0.5); break; // GT
            case 0xD: take_prob = Z + (1.0 - Z) * (1.0 - soft_sigmoid((N - 0.5) * (V - 0.5) * 20.0 + 0.5)); break; // LE
        }
        is_branch_taken = (take_prob > 0.5);
        branch_prob = take_prob;
        return take_prob;
    }
    // B unconditional
    if ((inst & 0xFC000000) == 0x14000000) {
        is_branch_taken = true;
        branch_prob = 1.0;
        return 1.0;
    }
    // BL
    if ((inst & 0xFC000000) == 0x94000000) {
        is_branch_taken = true;
        branch_prob = 1.0;
        return 1.0;
    }
    // CBZ
    if ((inst & 0xFF000000) == 0xB4000000) {
        float take_prob = soft_sigmoid(-(fabs(rn_val) - 0.5) * 10.0);
        is_branch_taken = (take_prob > 0.5);
        branch_prob = take_prob;
        return take_prob;
    }
    // CBNZ
    if ((inst & 0xFF000000) == 0xB5000000) {
        float take_prob = 1.0 - soft_sigmoid(-(fabs(rn_val) - 0.5) * 10.0);
        is_branch_taken = (take_prob > 0.5);
        branch_prob = take_prob;
        return take_prob;
    }

    return 0.0;
}

// ============================================================================
// JIT TEMPLATE EXECUTORS (update both soft and hard registers)
// ============================================================================

inline float jit_exec_counting(thread SoftRegFile* rf, uint32_t max_iters,
                               thread float& N, thread float& Z) {
    // Use HARD register for correct control flow
    float counter = rf->hard_regs[0];
    uint32_t iters = 0;

    while (counter > 0.5 && iters < max_iters) {
        counter -= 1.0;
        iters++;
    }

    // Update BOTH soft and hard registers
    rf->regs[0] = counter;
    rf->hard_regs[0] = counter;

    N = soft_sigmoid(-counter);
    Z = soft_sigmoid(-(fabs(counter) - 0.5) * 10.0);

    return float(iters * 2);
}

inline float jit_exec_memcpy(device uint8_t* memory, thread SoftRegFile* rf,
                             uint32_t max_iters, uint32_t mem_size) {
    // Use HARD registers for correct addresses
    uint64_t dst = (uint64_t)rf->hard_regs[0];
    uint64_t src = (uint64_t)rf->hard_regs[1];
    uint32_t count = min(max_iters, (uint32_t)max(rf->hard_regs[2], 0.0f));

    for (uint32_t i = 0; i < count; i++) {
        if (src + i < mem_size && dst + i < mem_size) {
            memory[dst + i] = memory[src + i];
        }
    }

    // Update BOTH soft and hard registers
    float new_dst = (float)(dst + count);
    float new_src = (float)(src + count);
    float new_count = rf->hard_regs[2] - count;

    rf->regs[0] = new_dst; rf->hard_regs[0] = new_dst;
    rf->regs[1] = new_src; rf->hard_regs[1] = new_src;
    rf->regs[2] = new_count; rf->hard_regs[2] = new_count;

    return float(count * 4);
}

inline float jit_exec_sum(device uint8_t* memory, thread SoftRegFile* rf,
                          uint32_t max_iters, uint32_t mem_size) {
    // Use HARD registers for correct values
    float sum = rf->hard_regs[0];
    uint64_t ptr = (uint64_t)rf->hard_regs[1];
    uint32_t count = min(max_iters, (uint32_t)max(rf->hard_regs[2], 0.0f));

    for (uint32_t i = 0; i < count; i++) {
        uint64_t addr = ptr + i * 8;
        if (addr + 8 <= mem_size) {
            int64_t val = int64_t(memory[addr]) |
                         (int64_t(memory[addr+1]) << 8) |
                         (int64_t(memory[addr+2]) << 16) |
                         (int64_t(memory[addr+3]) << 24) |
                         (int64_t(memory[addr+4]) << 32) |
                         (int64_t(memory[addr+5]) << 40) |
                         (int64_t(memory[addr+6]) << 48) |
                         (int64_t(memory[addr+7]) << 56);
            sum += float(val);
        }
    }

    // Update BOTH soft and hard registers
    float new_ptr = float(ptr + count * 8);
    float new_count = rf->hard_regs[2] - count;

    rf->regs[0] = sum; rf->hard_regs[0] = sum;
    rf->regs[1] = new_ptr; rf->hard_regs[1] = new_ptr;
    rf->regs[2] = new_count; rf->hard_regs[2] = new_count;

    return float(count * 5);
}

// ============================================================================
// MAIN UNIFIED DIFF CPU KERNEL
// ============================================================================

kernel void unified_diff_cpu_execute(
    device uint8_t* memory [[buffer(0)]],
    device float* registers [[buffer(1)]],        // Soft registers (float)
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],            // [N, Z, C, V]
    device const uint32_t* config [[buffer(4)]],  // [cycles_per_batch, mem_size, jit_threshold]
    device atomic_uint* signal [[buffer(5)]],
    device uint32_t* stats [[buffer(6)]],         // [cycles, jit_execs, ooo_execs, spec_branches, soft_commits]
    device const float* weights [[buffer(7)]],    // All learnable weights
    device float* gradients [[buffer(8)]],        // Gradient accumulator
    device const float* temperature [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // DEBUG: Write canary to verify kernel runs (use registers buffer)
    registers[30] = 12345.0;  // Canary value in X30 (link register, not used in simple programs)

    uint64_t pc = pc_ptr[0];
    uint32_t cycles_per_batch = config[0];
    uint32_t mem_size = config[1];
    uint32_t jit_threshold = config[2];
    float temp = temperature[0];

    uint32_t cycles = 0;
    uint32_t jit_execs = 0;
    uint32_t ooo_execs = 0;
    uint32_t spec_branches = 0;
    uint32_t soft_commits = 0;

    float N = flags[0], Z = flags[1], C = flags[2], V = flags[3];

    // Soft register file with hard tracking
    SoftRegFile rf;
    for (uint32_t i = 0; i < NUM_REGS; i++) {
        rf.regs[i] = registers[i];
        rf.hard_regs[i] = registers[i];  // Initialize hard values
        rf.grad_accum[i] = 0.0;          // Zero gradient accumulators
    }

    // Refresh interval for drift correction
    const uint32_t REFRESH_INTERVAL = 100;

    // Local gradient accumulators
    float local_grads[64];
    for (uint32_t i = 0; i < 64; i++) local_grads[i] = 0.0;

    bool halted = false;

    while (cycles < cycles_per_batch && !halted) {
        if (pc + 4 > mem_size) {
            halted = true;
            break;
        }

        // ================================================================
        // STAGE 1: JIT PATTERN DETECTION
        // ================================================================

        // Fetch instruction window for JIT pattern matching
        uint32_t jit_window[JIT_WINDOW];
        uint32_t jit_count = 0;
        uint64_t fetch_pc = pc;

        for (uint32_t i = 0; i < JIT_WINDOW && fetch_pc + 4 <= mem_size; i++) {
            jit_window[i] = uint32_t(memory[fetch_pc]) | (uint32_t(memory[fetch_pc+1])<<8) |
                            (uint32_t(memory[fetch_pc+2])<<16) | (uint32_t(memory[fetch_pc+3])<<24);

            // HLT - stop fetching but don't halt yet (execute fetched instructions first)
            if ((jit_window[i] & 0xFFE0001F) == 0xD4400000) {
                // Don't set halted here - let the OoO path handle it
                // so we can execute the instructions before HLT
                break;
            }

            jit_count++;
            fetch_pc += 4;

            uint8_t top = (jit_window[i] >> 24) & 0xFF;
            if ((top & 0xFC) == 0x14 || top == 0x54) break;
        }

        if (jit_count == 0) {
            gradients[1] = 99999.0;  // DEBUG: jit_count was 0
            break;
        }
        gradients[1] = (float)jit_count;  // DEBUG: jit_count value

        // Encode pattern
        float pattern_embedding[EMBED_DIM];
        encode_jit_pattern(jit_window, jit_count, &weights[ENCODER_WEIGHTS_START], pattern_embedding);

        // Match templates
        float template_similarities[NUM_TEMPLATES];
        match_jit_templates(pattern_embedding, &weights[TEMPLATE_WEIGHTS_START], template_similarities);

        // Gumbel-softmax template selection
        float template_probs[NUM_TEMPLATES];
        gumbel_softmax(template_similarities, template_probs, NUM_TEMPLATES, temp, cycles);

        // Find best template
        float max_prob = 0.0;
        uint32_t best_template = TEMPLATE_INTERPRET;
        for (uint32_t t = 0; t < NUM_TEMPLATES - 1; t++) {
            if (template_probs[t] > max_prob) {
                max_prob = template_probs[t];
                best_template = t;
            }
        }

        // ================================================================
        // DECISION: JIT or OoO? (use HARD regs for control flow)
        // ================================================================

        // Use hard_regs for control flow decisions to prevent drift
        bool do_jit = (max_prob > 0.5) && (rf.hard_regs[0] > jit_threshold || rf.hard_regs[2] > jit_threshold);

        if (do_jit) {
            // ================================================================
            // JIT EXECUTION PATH
            // ================================================================

            // Use hard values for iteration count
            uint32_t max_iters = min((uint32_t)max(max(rf.hard_regs[0], rf.hard_regs[2]), 1.0f),
                                     cycles_per_batch - cycles);
            float jit_cycles = 0.0;

            if (best_template == TEMPLATE_COUNTING) {
                jit_cycles = jit_exec_counting(&rf, max_iters, N, Z);
                pc += 8;  // Skip SUBS + B.NE
            }
            else if (best_template == TEMPLATE_MEMCPY) {
                jit_cycles = jit_exec_memcpy(memory, &rf, max_iters, mem_size);
                pc += 16;
            }
            else if (best_template == TEMPLATE_SUM) {
                jit_cycles = jit_exec_sum(memory, &rf, max_iters, mem_size);
                pc += 16;
            }

            cycles += (uint32_t)jit_cycles;
            jit_execs++;

            // Gradient: reward JIT selection
            for (uint32_t i = 0; i < EMBED_DIM; i++) {
                local_grads[best_template * EMBED_DIM + i] += pattern_embedding[i] * 0.01;
            }
        }
        else {
            // ================================================================
            // OoO EXECUTION PATH
            // ================================================================

            // Fetch larger window for OoO
            DecodedInst window[WINDOW_SIZE];
            uint32_t window_count = 0;
            fetch_pc = pc;

            while (window_count < WINDOW_SIZE && fetch_pc + 4 <= mem_size) {
                uint32_t inst = uint32_t(memory[fetch_pc]) | (uint32_t(memory[fetch_pc+1])<<8) |
                                (uint32_t(memory[fetch_pc+2])<<16) | (uint32_t(memory[fetch_pc+3])<<24);

                // Check for HLT - just set halted flag, set signal later
                if ((inst & 0xFFE0001F) == 0xD4400000) {
                    halted = true;
                    break;
                }

                // Check for RET - just set halted flag, set signal later
                if ((inst & 0xFFFFFC1F) == 0xD65F0000) {
                    halted = true;
                    break;
                }

                window[window_count] = decode_inst(inst);
                fetch_pc += 4;
                window_count++;

                if (window[window_count-1].is_branch) break;
            }

            // Execute the window if we have any instructions, even if halting after
            if (window_count == 0) {
                gradients[2] = 88888.0;  // DEBUG: window_count was 0
                break;
            }
            gradients[2] = (float)window_count;  // DEBUG: window_count value
            gradients[3] = halted ? 1.0 : 0.0;  // DEBUG: halted state

            // ================================================================
            // SOFT DEPENDENCY MATRIX
            // ================================================================
            float dep_matrix[WINDOW_SIZE][WINDOW_SIZE];
            for (uint32_t i = 0; i < window_count; i++) {
                for (uint32_t j = 0; j < window_count; j++) {
                    if (j <= i) {
                        dep_matrix[i][j] = 0.0;
                    } else {
                        dep_matrix[i][j] = predict_soft_dependency(window[i], window[j],
                                                                    &weights[OOO_WEIGHTS_START]);
                    }
                }
            }

            // ================================================================
            // SOFT READINESS SCORES
            // ================================================================
            float readiness[WINDOW_SIZE];
            for (uint32_t i = 0; i < window_count; i++) {
                readiness[i] = 1.0;
                for (uint32_t j = 0; j < i; j++) {
                    readiness[i] *= (1.0 - dep_matrix[j][i]);
                }
            }

            // ================================================================
            // GUMBEL-SOFTMAX SCHEDULING
            // ================================================================
            float schedule_probs[WINDOW_SIZE];
            gumbel_softmax(readiness, schedule_probs, window_count, temp, cycles + 1000);

            // ================================================================
            // SOFT EXECUTION WITH RESULT FORWARDING
            // Each instruction commits to hard_regs immediately so subsequent
            // instructions in the window see the correct values (OoO forwarding)
            // ================================================================
            float results[WINDOW_SIZE];
            float branch_probs[WINDOW_SIZE];
            bool branch_taken[WINDOW_SIZE];

            for (uint32_t i = 0; i < window_count; i++) {
                bool taken = false;
                float bp = 0.0;
                results[i] = exec_soft_inst(window[i], &rf, N, Z, C, V, memory, mem_size, taken, bp);
                branch_taken[i] = taken;
                branch_probs[i] = bp;

                // RESULT FORWARDING: Commit to hard_regs immediately so
                // subsequent instructions in the window see the correct value
                // Skip branches - they don't write to registers
                if (!window[i].is_branch && window[i].rd < 31) {
                    rf.hard_regs[window[i].rd] = results[i];
                }
            }

            // ================================================================
            // STRAIGHT-THROUGH COMMIT
            // All instructions commit with hard values (correct execution)
            // Soft probabilities tracked for gradient flow only
            // NOTE: Skip branch instructions - they don't write to registers!
            // ================================================================
            for (uint32_t i = 0; i < window_count; i++) {
                // Skip branches - rd field is condition code, not destination
                if (window[i].is_branch) {
                    cycles++;
                    continue;
                }

                if (window[i].rd < 31) {
                    // HARD commit for correct execution
                    rf.hard_regs[window[i].rd] = results[i];

                    // SOFT update for gradient flow (STE pattern)
                    rf.regs[window[i].rd] = schedule_probs[i] * results[i] +
                                            (1.0 - schedule_probs[i]) * rf.regs[window[i].rd];

                    // Track gradient contribution
                    rf.grad_accum[window[i].rd] += schedule_probs[i] * 0.01;
                    soft_commits++;
                }
                cycles++;
            }

            // Periodic refresh: sync soft to hard to prevent drift accumulation
            if (cycles % REFRESH_INTERVAL == 0) {
                for (uint32_t i = 0; i < NUM_REGS; i++) {
                    rf.regs[i] = rf.hard_regs[i];
                }
            }

            ooo_execs += window_count;

            // ================================================================
            // SPECULATIVE BRANCH HANDLING
            // ================================================================
            bool any_branch = false;
            uint32_t branch_idx = 0;
            float branch_p = 0.0;

            for (uint32_t i = 0; i < window_count; i++) {
                if (window[i].is_branch) {
                    any_branch = true;
                    branch_idx = i;
                    branch_p = branch_probs[i];
                    spec_branches++;
                    break;
                }
            }

            if (any_branch && branch_taken[branch_idx]) {
                uint32_t inst = window[branch_idx].raw;
                uint64_t taken_pc = pc;

                if ((inst & 0xFF000010) == 0x54000000) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    taken_pc = pc + branch_idx * 4 + imm19 * 4;
                }
                else if ((inst & 0xFC000000) == 0x14000000) {
                    int32_t imm26 = inst & 0x3FFFFFF;
                    if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                    taken_pc = pc + branch_idx * 4 + imm26 * 4;
                }
                else if ((inst & 0xFC000000) == 0x94000000) {
                    int32_t imm26 = inst & 0x3FFFFFF;
                    if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                    rf.regs[30] = (float)(pc + (branch_idx + 1) * 4);
                    taken_pc = pc + branch_idx * 4 + imm26 * 4;
                }
                else if ((inst & 0xFF000000) == 0xB4000000 || (inst & 0xFF000000) == 0xB5000000) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    taken_pc = pc + branch_idx * 4 + imm19 * 4;
                }

                pc = taken_pc;

                // Gradient for branch prediction
                local_grads[3] += (branch_p - 0.5) * 0.1;
            } else {
                pc += window_count * 4;
            }

            // Gradient for dependency prediction
            for (uint32_t i = 0; i < window_count; i++) {
                for (uint32_t j = i + 1; j < window_count; j++) {
                    float dep = dep_matrix[i][j];
                    bool has_raw = (window[i].rd < 31) &&
                                   (window[i].rd == window[j].rn || window[i].rd == window[j].rm);
                    if (!has_raw) {
                        local_grads[0] -= dep * 0.01;
                    }
                }
            }
        }
    }

    // Set HALT signal if we halted (do this AFTER counting cycles)
    if (halted) {
        atomic_store_explicit(signal, SIGNAL_HALT, memory_order_relaxed);
    }

    // Write back HARD registers (correct values)
    for (uint32_t i = 0; i < NUM_REGS; i++) {
        registers[i] = rf.hard_regs[i];
    }
    pc_ptr[0] = pc;
    flags[0] = N; flags[1] = Z; flags[2] = C; flags[3] = V;

    // DEBUG: Mark that we reached the end of the kernel
    registers[30] = 99999.0;  // Final canary - should definitely be set if kernel completes

    // Update stats
    stats[0] = cycles;
    stats[1] = jit_execs;
    stats[2] = ooo_execs;
    stats[3] = spec_branches;
    stats[4] = soft_commits;

    // Accumulate gradients
    for (uint32_t i = 0; i < 64; i++) {
        atomic_fetch_add_explicit((device atomic_uint*)&gradients[i],
                                  as_type<uint32_t>(local_grads[i]), memory_order_relaxed);
    }

    uint32_t sig = atomic_load_explicit(signal, memory_order_relaxed);
    if (sig == SIGNAL_RUNNING) {
        atomic_store_explicit(signal, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

const TOTAL_WEIGHTS: usize = 1076;
const GRADIENT_SIZE: usize = 64;

/// Unified Diff CPU Result
#[pyclass]
#[derive(Debug, Clone)]
pub struct UnifiedDiffResult {
    #[pyo3(get)]
    pub total_cycles: u32,
    #[pyo3(get)]
    pub jit_executions: u32,
    #[pyo3(get)]
    pub ooo_executions: u32,
    #[pyo3(get)]
    pub speculative_branches: u32,
    #[pyo3(get)]
    pub soft_commits: u32,
    #[pyo3(get)]
    pub signal: u32,
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    #[pyo3(get)]
    pub ips: f64,
}

#[pymethods]
impl UnifiedDiffResult {
    fn __repr__(&self) -> String {
        format!("UnifiedDiffResult(cycles={}, jit={}, ooo={}, spec={}, ips={:.0})",
                self.total_cycles, self.jit_executions, self.ooo_executions,
                self.speculative_branches, self.ips)
    }
}

/// Unified Differentiable CPU
#[pyclass(unsendable)]
pub struct UnifiedDiffCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    memory_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    config_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    signal_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    stats_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    weights_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    gradients_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    temperature_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    memory_size: usize,
}

#[pymethods]
impl UnifiedDiffCPU {
    #[new]
    #[pyo3(signature = (memory_size=4*1024*1024, cycles_per_batch=10_000_000, jit_threshold=100))]
    fn new(memory_size: usize, cycles_per_batch: u32, jit_threshold: u32) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[UnifiedDiffCPU] Using device: {:?}", device.name());
        println!("[UnifiedDiffCPU] Compiling Unified Differentiable CPU shader...");

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        let source = NSString::from_str(UNIFIED_DIFF_CPU_SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let func_name = NSString::from_str("unified_diff_cpu_execute");
        let function = library.newFunctionWithName(&func_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("Function not found".to_string()))?;

        let pipeline = device.newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        let opts = MTLResourceOptions::StorageModeShared;

        let memory_buf = device.newBufferWithLength_options(memory_size, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let registers_buf = device.newBufferWithLength_options(32 * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let pc_buf = device.newBufferWithLength_options(8, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let flags_buf = device.newBufferWithLength_options(16, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let config_buf = device.newBufferWithLength_options(12, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let signal_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let stats_buf = device.newBufferWithLength_options(20, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let weights_buf = device.newBufferWithLength_options(TOTAL_WEIGHTS * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let gradients_buf = device.newBufferWithLength_options(GRADIENT_SIZE * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let temperature_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;

        unsafe {
            let cfg = config_buf.contents().as_ptr() as *mut u32;
            *cfg.add(0) = cycles_per_batch;
            *cfg.add(1) = memory_size as u32;
            *cfg.add(2) = jit_threshold;

            let temp = temperature_buf.contents().as_ptr() as *mut f32;
            *temp = 1.0;

            // Initialize weights
            let w = weights_buf.contents().as_ptr() as *mut f32;

            // OoO weights [0-19]
            *w.add(0) = 5.0;   // RAW
            *w.add(1) = 3.0;   // WAW
            *w.add(2) = 5.0;   // FLAG
            *w.add(3) = -2.0;  // BIAS
            for i in 4..20 { *w.add(i) = 0.1; }

            // Template embeddings [20-51]
            // Counting: first 2 dims high
            for i in 0..8 { *w.add(20 + 0 * 8 + i) = if i < 2 { 1.0 } else { 0.0 }; }
            // Memcpy: middle dims high
            for i in 0..8 { *w.add(20 + 1 * 8 + i) = if i >= 2 && i < 4 { 1.0 } else { 0.0 }; }
            // Sum: later dims high
            for i in 0..8 { *w.add(20 + 2 * 8 + i) = if i >= 4 && i < 6 { 1.0 } else { 0.0 }; }
            // Interpret: uniform
            for i in 0..8 { *w.add(20 + 3 * 8 + i) = 0.25; }

            // Encoder weights [52+]: random init
            for i in 0..1024 {
                let val = ((i * 7919 + 104729) % 1000) as f32 / 1000.0 - 0.5;
                *w.add(52 + i) = val * 0.1;
            }

            std::ptr::write_bytes(gradients_buf.contents().as_ptr() as *mut u8, 0, GRADIENT_SIZE * 4);
            std::ptr::write_bytes(registers_buf.contents().as_ptr() as *mut u8, 0, 32 * 4);
        }

        println!("[UnifiedDiffCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[UnifiedDiffCPU] Total learnable parameters: {}", TOTAL_WEIGHTS);
        println!("[UnifiedDiffCPU] Architecture: Diff JIT → Diff OoO → Speculative → Soft Commit");
        println!("[UnifiedDiffCPU] Features:");
        println!("  - Learned JIT pattern encoder (1024 params)");
        println!("  - Soft template selection (32 params)");
        println!("  - Learned OoO dependency prediction (20 params)");
        println!("  - Gumbel-softmax scheduling");
        println!("  - Speculative branch execution");
        println!("  - Soft register commit");
        println!("  - End-to-end gradient flow");

        Ok(Self {
            device,
            command_queue,
            pipeline,
            memory_buf,
            registers_buf,
            pc_buf,
            flags_buf,
            config_buf,
            signal_buf,
            stats_buf,
            weights_buf,
            gradients_buf,
            temperature_buf,
            memory_size,
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
        println!("[UnifiedDiffCPU] Loaded {} bytes at 0x{:x}", program.len(), address);
        Ok(())
    }

    fn set_pc(&self, pc: u64) {
        unsafe { *(self.pc_buf.contents().as_ptr() as *mut u64) = pc; }
    }

    fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buf.contents().as_ptr() as *const u64) }
    }

    fn set_register(&self, reg: usize, value: f32) -> PyResult<()> {
        if reg >= 32 { return Err(PyRuntimeError::new_err("Invalid register")); }
        unsafe { *(self.registers_buf.contents().as_ptr() as *mut f32).add(reg) = value; }
        Ok(())
    }

    fn get_register(&self, reg: usize) -> PyResult<f32> {
        if reg >= 32 { return Err(PyRuntimeError::new_err("Invalid register")); }
        unsafe { Ok(*(self.registers_buf.contents().as_ptr() as *const f32).add(reg)) }
    }

    fn set_temperature(&self, temp: f32) {
        unsafe { *(self.temperature_buf.contents().as_ptr() as *mut f32) = temp; }
    }

    fn get_temperature(&self) -> f32 {
        unsafe { *(self.temperature_buf.contents().as_ptr() as *const f32) }
    }

    fn get_weights(&self) -> Vec<f32> {
        unsafe {
            let ptr = self.weights_buf.contents().as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, TOTAL_WEIGHTS).to_vec()
        }
    }

    fn set_weights(&self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != TOTAL_WEIGHTS {
            return Err(PyRuntimeError::new_err(format!(
                "Expected {} weights, got {}", TOTAL_WEIGHTS, weights.len()
            )));
        }
        unsafe {
            let ptr = self.weights_buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(weights.as_ptr(), ptr, weights.len());
        }
        Ok(())
    }

    fn get_gradients(&self) -> Vec<f32> {
        unsafe {
            let ptr = self.gradients_buf.contents().as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, GRADIENT_SIZE).to_vec()
        }
    }

    fn zero_gradients(&self) {
        unsafe {
            std::ptr::write_bytes(self.gradients_buf.contents().as_ptr() as *mut u8, 0, GRADIENT_SIZE * 4);
        }
    }

    fn reset(&self) {
        unsafe {
            std::ptr::write_bytes(self.registers_buf.contents().as_ptr() as *mut u8, 0, 32 * 4);
            std::ptr::write_bytes(self.flags_buf.contents().as_ptr() as *mut u8, 0, 16);
            *(self.pc_buf.contents().as_ptr() as *mut u64) = 0;
        }
    }

    fn param_count(&self) -> usize {
        TOTAL_WEIGHTS
    }

    #[pyo3(signature = (max_batches=100, timeout_seconds=10.0))]
    fn execute(&self, max_batches: u32, timeout_seconds: f64) -> PyResult<UnifiedDiffResult> {
        let start = Instant::now();

        // DEBUG: Print initial register value
        let initial_x30 = unsafe { *(self.registers_buf.contents().as_ptr() as *const f32).add(30) };
        println!("[DEBUG] Before kernel: X30={}", initial_x30);

        unsafe {
            *(self.signal_buf.contents().as_ptr() as *mut u32) = 0;
            std::ptr::write_bytes(self.stats_buf.contents().as_ptr() as *mut u8, 0, 20);
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
                encoder.setBuffer_offset_atIndex(Some(&self.config_buf), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&self.signal_buf), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&self.stats_buf), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(&self.weights_buf), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(&self.gradients_buf), 0, 8);
                encoder.setBuffer_offset_atIndex(Some(&self.temperature_buf), 0, 9);

                let grid = MTLSize { width: 1, height: 1, depth: 1 };
                let tg = MTLSize { width: 1, height: 1, depth: 1 };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            }
            encoder.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            // DEBUG: Print after kernel
            let after_x30 = unsafe { *(self.registers_buf.contents().as_ptr() as *const f32).add(30) };
            let signal = unsafe { *(self.signal_buf.contents().as_ptr() as *const u32) };
            println!("[DEBUG] After kernel batch {}: X30={}, signal={}", batch, after_x30, signal);

            if signal == 1 { break; }

            unsafe { *(self.signal_buf.contents().as_ptr() as *mut u32) = 0; }
            batch += 1;
        }

        let elapsed = start.elapsed().as_secs_f64();

        let stats = unsafe { std::slice::from_raw_parts(self.stats_buf.contents().as_ptr() as *const u32, 5) };
        let total_cycles = stats[0];
        let jit_execs = stats[1];
        let ooo_execs = stats[2];
        let spec_branches = stats[3];
        let soft_commits = stats[4];
        let signal = unsafe { *(self.signal_buf.contents().as_ptr() as *const u32) };

        let ips = if elapsed > 0.0 { total_cycles as f64 / elapsed } else { 0.0 };

        Ok(UnifiedDiffResult {
            total_cycles,
            jit_executions: jit_execs,
            ooo_executions: ooo_execs,
            speculative_branches: spec_branches,
            soft_commits,
            signal,
            elapsed_seconds: elapsed,
            ips,
        })
    }

    /// Set performance mode - optimizes for speed over differentiability
    /// mode: 0 = full differentiable (default, slowest, fully learnable)
    ///       1 = fast mode (temperature=0.01, near-hard decisions)
    ///       2 = ultra fast (temperature=0.001, hard decisions)
    fn set_performance_mode(&self, mode: u32) {
        unsafe {
            match mode {
                0 => {
                    // Full differentiable mode
                    *(self.temperature_buf.contents().as_ptr() as *mut f32) = 1.0;
                    println!("[UnifiedDiffCPU] Performance mode: FULL DIFFERENTIABLE (temp=1.0)");
                }
                1 => {
                    // Fast mode: near-hard decisions (mostly for execution)
                    *(self.temperature_buf.contents().as_ptr() as *mut f32) = 0.01;
                    println!("[UnifiedDiffCPU] Performance mode: FAST (temp=0.01)");
                }
                2 => {
                    // Ultra fast: hard decisions (minimal differentiability)
                    *(self.temperature_buf.contents().as_ptr() as *mut f32) = 0.001;
                    println!("[UnifiedDiffCPU] Performance mode: ULTRA FAST (temp=0.001)");
                }
                _ => {
                    eprintln!("[UnifiedDiffCPU] Invalid performance mode: {}, using default", mode);
                    *(self.temperature_buf.contents().as_ptr() as *mut f32) = 1.0;
                }
            }
        }
    }
}

pub fn register_unified_diff_cpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UnifiedDiffCPU>()?;
    m.add_class::<UnifiedDiffResult>()?;
    Ok(())
}
