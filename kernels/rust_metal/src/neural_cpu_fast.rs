//! Neural ARM64 Fast CPU Kernel — focused cooperative-threadgroup neural Metal shader
//!
//! A clean, focused implementation covering ~20 most common ARM64 instructions with
//! maximum performance. Every ALU operation uses trained neural network weights.
//!
//! Architecture: Two-Phase Per-Instruction (Phase A/B/C with barriers)
//!   Phase A (thread 0 only): fetch, decode, prepare operands, check for ALU op
//!   Phase B (all 64 threads): if ALU op, run cooperative CLA/logic/mul
//!   Phase C (thread 0 only): read result, writeback, update PC/flags
//!
//! Instructions covered (~20, 95%+ of real programs):
//!   ALU (cooperative): ADD/ADDS/SUB/SUBS (reg + imm, 32-bit), AND/ORR/EOR (reg), MUL (LUT)
//!   Data movement: MOVZ, MOVK
//!   Branches: B, B.cond (EQ/NE/GE/LT/GT/LE), CBZ, CBNZ
//!   Memory: LDR, STR, LDRB, STRB (unsigned offset)
//!   System: SVC (SYS_WRITE, SYS_EXIT), HLT
//!
//! Buffer layout (simplified — no shift LUTs):
//!   buffer(0):  memory          [4 MB, shared]
//!   buffer(1):  registers       [32 x int64]
//!   buffer(2):  pc_ptr          [1 x uint64]
//!   buffer(3):  flags           [4 x float]  (N, Z, C, V)
//!   buffer(4):  max_cycles_ptr  [1 x uint32]
//!   buffer(5):  signal_flag     [1 x uint32, atomic]
//!   buffer(6):  total_cycles    [1 x uint32]
//!   buffer(7):  neural_weights  [2,494 f32] — carry combine + truth tables
//!   buffer(8):  mul_lut         [1,048,576 f32] — byte-pair multiply

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::time::{Duration, Instant};

use crate::continuous::{ContinuousResult, Signal};
use crate::{get_default_device, MetalError};

// ─────────────────────────────────────────────────────────────────────────────
// Metal shader source — neural ARM64 fast CPU (cooperative threadgroup, focused)
// ─────────────────────────────────────────────────────────────────────────────

const NEURAL_FAST_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// ════════════════════════════════════════════════════════════════════════════
// NEURAL ARM64 FAST CPU — Metal Compute Shader (Cooperative Threadgroup)
//
// Focused on ~20 most common instructions with maximum performance.
// All ALU operations route through trained neural network weights.
//
// Architecture: 1 threadgroup of 64 threads.
//   Phase A (thread 0): fetch, decode, prepare operands
//   Phase B (all 64):   cooperative neural CLA / logic / mul
//   Phase C (thread 0): writeback results, update PC/flags
// ════════════════════════════════════════════════════════════════════════════

// ── Constants ────────────────────────────────────────────────────────────────

constant uint32_t SIGNAL_RUNNING    = 0;
constant uint32_t SIGNAL_HALT       = 1;
constant uint32_t SIGNAL_SYSCALL    = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

// SVC buffer for GPU-side SYS_WRITE
constant uint32_t SVC_BUF_BASE     = 0x3F0000;
constant uint32_t SVC_BUF_HDR      = 16;
constant uint32_t SVC_BUF_DATA     = SVC_BUF_BASE + SVC_BUF_HDR;
constant uint32_t SVC_BUF_CAPACITY = 0xFFF0;
constant uint32_t SVC_HEAP_BASE    = 0x60000;

constant int64_t SVC_SYS_WRITE      = 64;
constant int64_t SVC_SYS_EXIT       = 93;
constant int64_t SVC_SYS_EXIT_GROUP = 231;
constant int64_t SVC_SYS_BRK        = 214;
constant int64_t SVC_SYS_CLOSE      = 57;

// ── Neural weight offsets ────────────────────────────────────────────────────
constant int CC_FC1_W = 0;      // [64, 4]  = 256 floats
constant int CC_FC1_B = 256;    // [64]
constant int CC_FC2_W = 320;    // [32, 64] = 2048 floats
constant int CC_FC2_B = 2368;   // [32]
constant int CC_FC3_W = 2400;   // [2, 32]  = 64 floats
constant int CC_FC3_B = 2464;   // [2]
constant int TT_BASE  = 2466;   // [7, 4]   = 28 floats

// ── Memory helpers ───────────────────────────────────────────────────────────

inline uint32_t read_u32_le(device uint8_t* mem, uint32_t addr) {
    return uint32_t(mem[addr]) | (uint32_t(mem[addr+1]) << 8) |
           (uint32_t(mem[addr+2]) << 16) | (uint32_t(mem[addr+3]) << 24);
}

inline void write_u32_le(device uint8_t* mem, uint32_t addr, uint32_t val) {
    mem[addr]   = uint8_t(val & 0xFF);
    mem[addr+1] = uint8_t((val >> 8) & 0xFF);
    mem[addr+2] = uint8_t((val >> 16) & 0xFF);
    mem[addr+3] = uint8_t((val >> 24) & 0xFF);
}

inline uint64_t read_u64_le(device uint8_t* mem, uint32_t addr) {
    return uint64_t(read_u32_le(mem, addr)) | (uint64_t(read_u32_le(mem, addr+4)) << 32);
}

inline void write_u64_le(device uint8_t* mem, uint32_t addr, uint64_t val) {
    write_u32_le(mem, addr, uint32_t(val & 0xFFFFFFFF));
    write_u32_le(mem, addr+4, uint32_t(val >> 32));
}

// ════════════════════════════════════════════════════════════════════════════
// NEURAL ALU FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

inline float neural_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -15.0f, 15.0f)));
}

// Truth table lookup: row in {0..6}, idx = a_bit*2 + b_bit
// AND=row0, OR=row1, XOR=row2
inline int neural_tt(int row, int a_bit, int b_bit,
                     device const float* w) {
    float logit = w[TT_BASE + row * 4 + a_bit * 2 + b_bit];
    return neural_sigmoid(logit) > 0.5f ? 1 : 0;
}

// ────────────────────────────────────────────────────────────────────────────
// COOPERATIVE carry_combine MLP [4 -> 64 -> 32 -> 2]
//
// All 64 threads in the threadgroup participate:
//   FC1: thread tid (0..63) computes h1[tid]   — 4 MADs each, fully parallel
//   FC2: threads 0..31 compute h2[tid]          — 64 MADs each
//   FC3: threads 0..1 compute out[tid]          — 32 MADs each
//
// Critical path: 4 + 64 + 32 = 100 MADs (vs 2,368 serial)
// ────────────────────────────────────────────────────────────────────────────

void cooperative_carry_combine(
    threadgroup float* shared_G,
    threadgroup float* shared_P,
    int bit_i,
    int bit_j,
    device const float* w,
    uint tid,
    threadgroup float* shared_h1,
    threadgroup float* shared_h2,
    threadgroup float* shared_mlp_out
) {
    float g_i = shared_G[bit_i];
    float p_i = shared_P[bit_i];
    float g_j = shared_G[bit_j];
    float p_j = shared_P[bit_j];
    float inp[4] = {g_i, p_i, g_j, p_j};

    // FC1: [4] -> [64] + ReLU — each of 64 threads computes ONE neuron
    if (tid < 64) {
        float s = w[CC_FC1_B + tid];
        for (int j = 0; j < 4; j++)
            s += w[CC_FC1_W + tid * 4 + j] * inp[j];
        shared_h1[tid] = max(0.0f, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FC2: [64] -> [32] + ReLU — first 32 threads
    if (tid < 32) {
        float s = w[CC_FC2_B + tid];
        for (int j = 0; j < 64; j++)
            s += w[CC_FC2_W + tid * 64 + j] * shared_h1[j];
        shared_h2[tid] = max(0.0f, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FC3: [32] -> [2] — first 2 threads compute final logits
    if (tid < 2) {
        float s = w[CC_FC3_B + tid];
        for (int j = 0; j < 32; j++)
            s += w[CC_FC3_W + tid * 32 + j] * shared_h2[j];
        shared_mlp_out[tid] = neural_sigmoid(s) > 0.5f ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ────────────────────────────────────────────────────────────────────────────
// COOPERATIVE 32-bit Kogge-Stone CLA
//
// All 64 threads cooperate:
//   Initial G,P: 32 threads compute in parallel (truth table lookups)
//   5 Kogge-Stone stages: each carry_combine uses all 64 threads for MLP
//   Final XOR: 32 threads compute result bits in parallel
// ────────────────────────────────────────────────────────────────────────────

void cooperative_neural_cla(
    uint32_t a, uint32_t b, int carry_in,
    device const float* w,
    uint tid,
    threadgroup float* shared_G,
    threadgroup float* shared_P,
    threadgroup float* shared_h1,
    threadgroup float* shared_h2,
    threadgroup float* shared_mlp_out,
    threadgroup int*   shared_rbits
) {
    // Initial G = AND(a,b), P = XOR(a,b) — 32 threads in parallel
    if (tid < 32) {
        int a_bit = (int)((a >> tid) & 1u);
        int b_bit = (int)((b >> tid) & 1u);
        shared_G[tid] = (float)neural_tt(0, a_bit, b_bit, w);  // AND
        shared_P[tid] = (float)neural_tt(2, a_bit, b_bit, w);  // XOR
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inject carry_in at bit 0
    if (tid == 0 && carry_in) {
        if (shared_P[0] > 0.5f || shared_G[0] > 0.5f) shared_G[0] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5 Kogge-Stone prefix stages
    for (int stage = 0; stage < 5; stage++) {
        int stride = 1 << stage;
        if (stride >= 32) break;

        // Process high-to-low for safe in-place update
        for (int i = 31; i >= stride; i--) {
            cooperative_carry_combine(
                shared_G, shared_P,
                i, i - stride,
                w, tid,
                shared_h1, shared_h2, shared_mlp_out
            );

            if (tid == 0) {
                shared_G[i] = shared_mlp_out[0];
                shared_P[i] = shared_mlp_out[1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Compute final result bits — 32 threads in parallel
    if (tid < 32) {
        int carry = (tid == 0) ? carry_in : (shared_G[tid - 1] > 0.5f ? 1 : 0);
        int a_bit = (int)((a >> tid) & 1u);
        int b_bit = (int)((b >> tid) & 1u);
        int p_orig = neural_tt(2, a_bit, b_bit, w);         // XOR propagate
        shared_rbits[tid] = neural_tt(2, p_orig, carry, w); // XOR with carry
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Reassemble 32 result bits from threadgroup shared memory
inline uint32_t bits_to_u32_tg(threadgroup int* bits) {
    uint32_t r = 0;
    for (int i = 0; i < 32; i++)
        if (bits[i]) r |= (1u << i);
    return r;
}

// ── Neural multiply via byte-pair LUT (thread 0 only) ───────────────────────

inline uint32_t neural_mul_32(uint32_t a, uint32_t b, device const float* mul_lut) {
    uint a_bytes[4] = {
        a & 0xFFu, (a >> 8) & 0xFFu, (a >> 16) & 0xFFu, (a >> 24) & 0xFFu
    };
    uint b_bytes[4] = {
        b & 0xFFu, (b >> 8) & 0xFFu, (b >> 16) & 0xFFu, (b >> 24) & 0xFFu
    };

    ulong result = 0UL;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int lut_base = (int)((a_bytes[i] * 256u + b_bytes[j]) * 16u);
            uint pair_product = 0u;
            for (int bit = 0; bit < 16; bit++) {
                if (neural_sigmoid(mul_lut[lut_base + bit]) > 0.5f)
                    pair_product |= (1u << bit);
            }
            result += (ulong)pair_product << ((i + j) * 8);
        }
    }
    return uint32_t(result & 0xFFFFFFFFUL);
}

// ════════════════════════════════════════════════════════════════════════════
// NEURAL ARM64 FAST KERNEL — Two-Phase Per-Instruction
//
// Dispatched as 1 threadgroup of 64 threads.
// Phase A: Thread 0 fetches, decodes, prepares operands
// Phase B: All 64 threads cooperate on neural ALU (CLA/logic/mul)
// Phase C: Thread 0 writes back results, updates PC/flags
// ════════════════════════════════════════════════════════════════════════════

kernel void neural_arm64_fast(
    device uint8_t*       memory         [[buffer(0)]],
    device int64_t*       registers      [[buffer(1)]],
    device uint64_t*      pc_ptr         [[buffer(2)]],
    device float*         flags          [[buffer(3)]],
    device const uint32_t* max_cycles_ptr [[buffer(4)]],
    device atomic_uint*   signal_flag    [[buffer(5)]],
    device uint32_t*      total_cycles_ptr [[buffer(6)]],
    device const float*   neural_weights [[buffer(7)]],
    device const float*   mul_lut        [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // ── Threadgroup shared memory ────────────────────────────────────
    threadgroup float  sh_G[32];         // Kogge-Stone generate array
    threadgroup float  sh_P[32];         // Kogge-Stone propagate array
    threadgroup float  sh_h1[64];        // FC1 output scratch
    threadgroup float  sh_h2[32];        // FC2 output scratch
    threadgroup float  sh_mlp_out[2];    // FC3 output (G, P)
    threadgroup int    sh_rbits[32];     // CLA / logic result bits

    // Shared instruction state (written by thread 0, read by all)
    threadgroup uint32_t sh_inst;
    threadgroup uint8_t  sh_opcode;
    threadgroup uint8_t  sh_needs_cla;   // 1 = ADD/SUB needs cooperative CLA
    threadgroup uint8_t  sh_needs_logic; // 1=AND, 2=ORR, 3=EOR
    threadgroup uint8_t  sh_needs_mul;   // 1 = MUL via LUT
    threadgroup uint32_t sh_alu_a;       // operand A (32-bit)
    threadgroup uint32_t sh_alu_b;       // operand B
    threadgroup int      sh_carry_in;    // for SUB
    threadgroup uint32_t sh_alu_result;  // result written back
    threadgroup uint8_t  sh_done;        // halt/syscall/checkpoint

    device const float* w = neural_weights;

    // Thread 0's local execution state
    int64_t regs[32];
    uint64_t pc = 0;
    float flag_n = 0, flag_z = 0, flag_c = 0, flag_v = 0;
    uint32_t max_cycles = 0;
    uint32_t cycles = 0;

    if (tid == 0) {
        pc = pc_ptr[0];
        max_cycles = max_cycles_ptr[0];
        for (int i = 0; i < 32; i++) regs[i] = registers[i];
        flag_n = flags[0]; flag_z = flags[1]; flag_c = flags[2]; flag_v = flags[3];
        sh_done = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ════════════════════════════════════════════════════════════════
    // MAIN EXECUTION LOOP
    // ════════════════════════════════════════════════════════════════

    while (!sh_done) {
        // ══ PHASE A: Thread 0 fetches, decodes, prepares ═══════════
        if (tid == 0) {
            sh_needs_cla = 0;
            sh_needs_logic = 0;
            sh_needs_mul = 0;

            if (cycles >= max_cycles) {
                sh_done = 2; // checkpoint
            } else {
                // Fetch
                uint32_t inst = read_u32_le(memory, uint32_t(pc));
                sh_inst = inst;

                // Check halt
                if (inst == 0 || (inst & 0xFFE0001F) == 0xD4400000) {
                    sh_done = 1; // halt
                }
                // Check SVC
                else if ((inst & 0xFFE0001F) == 0xD4000001) {
                    int64_t svc_num = regs[8];

                    if (svc_num == SVC_SYS_WRITE && (regs[0] == 1 || regs[0] == 2)) {
                        // GPU-side SYS_WRITE buffering
                        uint32_t buf_pos = read_u32_le(memory, SVC_BUF_BASE);
                        int64_t src_addr = regs[1];
                        int64_t write_len = regs[2];
                        uint32_t entry_size = 3 + uint32_t(write_len);
                        if (write_len > 0 && buf_pos + entry_size <= SVC_BUF_CAPACITY) {
                            uint32_t base = SVC_BUF_DATA + buf_pos;
                            memory[base] = uint8_t(regs[0]);
                            memory[base + 1] = uint8_t(write_len & 0xFF);
                            memory[base + 2] = uint8_t((write_len >> 8) & 0xFF);
                            for (int64_t i = 0; i < write_len && i < 8192; i++) {
                                memory[base + 3 + uint32_t(i)] = memory[uint64_t(src_addr + i)];
                            }
                            buf_pos += entry_size;
                            uint32_t cnt = read_u32_le(memory, SVC_BUF_BASE + 4);
                            write_u32_le(memory, SVC_BUF_BASE, buf_pos);
                            write_u32_le(memory, SVC_BUF_BASE + 4, cnt + 1);
                            regs[0] = write_len;
                            pc += 4;
                            cycles++;
                        } else {
                            sh_done = 3; // syscall signal
                        }
                    } else if (svc_num == SVC_SYS_BRK) {
                        uint64_t brk = read_u64_le(memory, SVC_BUF_BASE + 8);
                        if (brk == 0) brk = SVC_HEAP_BASE;
                        if (regs[0] == 0) {
                            regs[0] = int64_t(brk);
                        } else if (uint64_t(regs[0]) >= SVC_HEAP_BASE) {
                            brk = uint64_t(regs[0]);
                            regs[0] = int64_t(brk);
                        } else {
                            regs[0] = int64_t(brk);
                        }
                        write_u64_le(memory, SVC_BUF_BASE + 8, brk);
                        pc += 4;
                        cycles++;
                    } else if (svc_num == SVC_SYS_CLOSE && regs[0] <= 2) {
                        regs[0] = 0;
                        pc += 4;
                        cycles++;
                    } else if (svc_num == SVC_SYS_EXIT || svc_num == SVC_SYS_EXIT_GROUP) {
                        sh_done = 1; // halt
                    } else {
                        sh_done = 3; // syscall signal for Python
                    }
                }
                else {
                    uint8_t op = (inst >> 24) & 0xFF;
                    sh_opcode = op;

                    uint8_t rd = inst & 0x1F;
                    uint8_t rn = (inst >> 5) & 0x1F;
                    uint8_t rm = (inst >> 16) & 0x1F;
                    uint16_t imm12 = (inst >> 10) & 0xFFF;

                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t rm_val = (rm == 31) ? 0 : regs[rm];

                    switch (op) {
                        // ── ADD register 32-bit ── COOPERATIVE CLA ────────
                        case 0x0B: {
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = uint32_t(rm_val);
                            sh_carry_in = 0;
                            sh_needs_cla = 1;
                            break;
                        }
                        // ── ADD immediate 32-bit ── COOPERATIVE CLA ───────
                        case 0x11: {
                            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = uint32_t(aimm);
                            sh_carry_in = 0;
                            sh_needs_cla = 1;
                            break;
                        }
                        // ── SUB register 32-bit ── COOPERATIVE CLA ────────
                        case 0x4B: {
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = ~uint32_t(rm_val);
                            sh_carry_in = 1;
                            sh_needs_cla = 1;
                            break;
                        }
                        // ── SUB immediate 32-bit ── COOPERATIVE CLA ───────
                        case 0x51: {
                            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = ~uint32_t(aimm);
                            sh_carry_in = 1;
                            sh_needs_cla = 1;
                            break;
                        }
                        // ── ADDS register 32-bit ── COOPERATIVE CLA ───────
                        case 0x2B: {
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = uint32_t(rm_val);
                            sh_carry_in = 0;
                            sh_needs_cla = 1;
                            break;
                        }
                        // ── ADDS immediate 32-bit ── COOPERATIVE CLA ──────
                        case 0x31: {
                            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = uint32_t(aimm);
                            sh_carry_in = 0;
                            sh_needs_cla = 1;
                            break;
                        }
                        // ── SUBS register 32-bit (CMP when rd=31) ─────────
                        case 0x6B: {
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = ~uint32_t(rm_val);
                            sh_carry_in = 1;
                            sh_needs_cla = 1;
                            break;
                        }
                        // ── SUBS immediate 32-bit (CMP imm) ──────────────
                        case 0x71: {
                            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = ~uint32_t(aimm);
                            sh_carry_in = 1;
                            sh_needs_cla = 1;
                            break;
                        }

                        // ── AND register 32-bit ── cooperative truth table ─
                        case 0x0A: {
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = uint32_t(rm_val);
                            sh_needs_logic = 1; // AND
                            break;
                        }
                        // ── ORR register 32-bit ── cooperative truth table ─
                        case 0x2A: {
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = uint32_t(rm_val);
                            sh_needs_logic = 2; // ORR
                            break;
                        }
                        // ── EOR register 32-bit ── cooperative truth table ─
                        case 0x4A: {
                            sh_alu_a = uint32_t(rn_val);
                            sh_alu_b = uint32_t(rm_val);
                            sh_needs_logic = 3; // EOR
                            break;
                        }

                        // ── MUL (MADD with Ra=WZR) ── byte-pair LUT ──────
                        case 0x1B: {
                            sh_alu_a = uint32_t(uint64_t(rn_val));
                            sh_alu_b = uint32_t(uint64_t(rm_val));
                            sh_needs_mul = 1;
                            break;
                        }

                        default:
                            // Non-ALU ops handled in Phase C
                            break;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Early exit check
        if (sh_done) break;

        // ══ PHASE B: All 64 threads cooperate on neural ALU ════════
        if (sh_needs_cla) {
            // Cooperative Kogge-Stone CLA — all 64 threads participate
            cooperative_neural_cla(
                sh_alu_a, sh_alu_b, sh_carry_in, w,
                tid, sh_G, sh_P, sh_h1, sh_h2, sh_mlp_out, sh_rbits
            );
            // Thread 0 assembles result
            if (tid == 0) {
                sh_alu_result = bits_to_u32_tg(sh_rbits);
            }
        }
        else if (sh_needs_logic) {
            // Cooperative truth table — 32 threads, one per bit
            if (tid < 32) {
                int a_bit = (int)((sh_alu_a >> tid) & 1u);
                int b_bit = (int)((sh_alu_b >> tid) & 1u);
                int row = (sh_needs_logic == 1) ? 0 : (sh_needs_logic == 2) ? 1 : 2;
                sh_rbits[tid] = neural_tt(row, a_bit, b_bit, w);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                sh_alu_result = bits_to_u32_tg(sh_rbits);
            }
        }
        else if (sh_needs_mul) {
            // Neural MUL via byte-pair LUT — thread 0 only
            if (tid == 0) {
                sh_alu_result = neural_mul_32(sh_alu_a, sh_alu_b, mul_lut);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ══ PHASE C: Thread 0 writes back results ═════════════════
        if (tid == 0) {
            uint8_t op = sh_opcode;
            uint32_t inst = sh_inst;
            uint8_t rd = inst & 0x1F;
            uint8_t rn = (inst >> 5) & 0x1F;
            uint8_t rm = (inst >> 16) & 0x1F;
            uint16_t imm12 = (inst >> 10) & 0xFFF;
            uint16_t imm16 = (inst >> 5) & 0xFFFF;
            uint8_t hw = (inst >> 21) & 0x3;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            bool branch_taken = false;

            if (sh_needs_cla || sh_needs_logic || sh_needs_mul) {
                // ── ALU result writeback ──────────────────────────
                if (rd != 31) regs[rd] = int64_t(sh_alu_result);

                // Flags for ADDS/SUBS/CMP variants
                if (op == 0x2B || op == 0x31 ||  // ADDS reg/imm
                    op == 0x6B || op == 0x71) {   // SUBS reg/imm
                    uint32_t r32 = sh_alu_result;
                    flag_n = (r32 & 0x80000000u) ? 1.0f : 0.0f;
                    flag_z = (r32 == 0) ? 1.0f : 0.0f;
                    // Carry: true if unsigned sum overflowed
                    flag_c = (uint64_t(sh_alu_a) + uint64_t(sh_alu_b) + uint64_t(sh_carry_in) > 0xFFFFFFFFu) ? 1.0f : 0.0f;
                    // Overflow: signed overflow detection
                    // For SUB: b_orig is the original (un-inverted) operand
                    uint32_t b_orig = (sh_carry_in) ? ~sh_alu_b : sh_alu_b;
                    flag_v = ((~(sh_alu_a ^ b_orig)) & (sh_alu_a ^ r32) & 0x80000000u) ? 1.0f : 0.0f;
                }

                pc += 4;
                cycles++;
            } else {
                // ── Non-ALU instruction handling ─────────────────

                // SVC was already handled in Phase A, skip
                if ((inst & 0xFFE0001F) == 0xD4000001) {
                    // already handled — do nothing here
                } else {
                    switch (op) {
                        // ── MOVZ 32-bit ──────────────────────────
                        case 0x52: {
                            if (rd != 31) regs[rd] = int64_t(uint64_t(imm16) << (hw * 16));
                            break;
                        }
                        // ── MOVZ 64-bit ──────────────────────────
                        case 0xD2: {
                            if (rd != 31) regs[rd] = int64_t(uint64_t(imm16) << (hw * 16));
                            break;
                        }
                        // ── MOVK 32-bit ──────────────────────────
                        case 0x72: {
                            uint64_t mask = ~(uint64_t(0xFFFF) << (hw * 16));
                            uint64_t val = uint64_t(regs[rd]) & mask;
                            regs[rd] = int64_t(val | (uint64_t(imm16) << (hw * 16)));
                            break;
                        }
                        // ── MOVK 64-bit ──────────────────────────
                        case 0xF2: {
                            uint64_t mask = ~(uint64_t(0xFFFF) << (hw * 16));
                            uint64_t val = uint64_t(regs[rd]) & mask;
                            regs[rd] = int64_t(val | (uint64_t(imm16) << (hw * 16)));
                            break;
                        }

                        // ── B (unconditional branch) ─────────────
                        case 0x14: case 0x15: case 0x16: case 0x17: {
                            uint32_t imm26 = inst & 0x3FFFFFF;
                            int32_t offset = (imm26 & 0x2000000)
                                ? int32_t(imm26 | 0xFC000000)
                                : int32_t(imm26);
                            pc = uint64_t(int64_t(pc) + int64_t(offset) * 4);
                            branch_taken = true;
                            break;
                        }
                        // ── BL (branch and link) ─────────────────
                        case 0x94: case 0x95: case 0x96: case 0x97: {
                            uint32_t imm26 = inst & 0x3FFFFFF;
                            int32_t offset = (imm26 & 0x2000000)
                                ? int32_t(imm26 | 0xFC000000)
                                : int32_t(imm26);
                            regs[30] = int64_t(pc + 4); // LR
                            pc = uint64_t(int64_t(pc) + int64_t(offset) * 4);
                            branch_taken = true;
                            break;
                        }

                        // ── B.cond ───────────────────────────────
                        case 0x54: {
                            uint32_t imm19 = (inst >> 5) & 0x7FFFF;
                            int32_t offset = (imm19 & 0x40000)
                                ? int32_t(imm19 | 0xFFF80000)
                                : int32_t(imm19);
                            uint8_t cond = inst & 0xF;
                            bool n = flag_n > 0.5f, z = flag_z > 0.5f;
                            bool c = flag_c > 0.5f, v = flag_v > 0.5f;
                            bool take = false;
                            switch (cond) {
                                case 0x0: take = z; break;             // EQ
                                case 0x1: take = !z; break;            // NE
                                case 0x2: take = c; break;             // CS/HS
                                case 0x3: take = !c; break;            // CC/LO
                                case 0x4: take = n; break;             // MI
                                case 0x5: take = !n; break;            // PL
                                case 0x8: take = c && !z; break;       // HI
                                case 0x9: take = !c || z; break;       // LS
                                case 0xA: take = (n == v); break;      // GE
                                case 0xB: take = (n != v); break;      // LT
                                case 0xC: take = !z && (n == v); break; // GT
                                case 0xD: take = z || (n != v); break;  // LE
                                case 0xE: take = true; break;          // AL
                                default:  take = true; break;
                            }
                            if (take) {
                                pc = uint64_t(int64_t(pc) + int64_t(offset) * 4);
                                branch_taken = true;
                            }
                            break;
                        }

                        // ── CBZ 32-bit ───────────────────────────
                        case 0x34: {
                            uint32_t imm19 = (inst >> 5) & 0x7FFFF;
                            int32_t off = (imm19 & 0x40000)
                                ? int32_t(imm19 | 0xFFF80000)
                                : int32_t(imm19);
                            int64_t rt_val = (rd == 31) ? 0 : regs[rd];
                            if ((rt_val & 0xFFFFFFFF) == 0) {
                                pc = uint64_t(int64_t(pc) + int64_t(off) * 4);
                                branch_taken = true;
                            }
                            break;
                        }
                        // ── CBZ 64-bit ───────────────────────────
                        case 0xB4: {
                            uint32_t imm19 = (inst >> 5) & 0x7FFFF;
                            int32_t off = (imm19 & 0x40000)
                                ? int32_t(imm19 | 0xFFF80000)
                                : int32_t(imm19);
                            int64_t rt_val = (rd == 31) ? 0 : regs[rd];
                            if (rt_val == 0) {
                                pc = uint64_t(int64_t(pc) + int64_t(off) * 4);
                                branch_taken = true;
                            }
                            break;
                        }
                        // ── CBNZ 32-bit ──────────────────────────
                        case 0x35: {
                            uint32_t imm19 = (inst >> 5) & 0x7FFFF;
                            int32_t off = (imm19 & 0x40000)
                                ? int32_t(imm19 | 0xFFF80000)
                                : int32_t(imm19);
                            int64_t rt_val = (rd == 31) ? 0 : regs[rd];
                            if ((rt_val & 0xFFFFFFFF) != 0) {
                                pc = uint64_t(int64_t(pc) + int64_t(off) * 4);
                                branch_taken = true;
                            }
                            break;
                        }
                        // ── CBNZ 64-bit ──────────────────────────
                        case 0xB5: {
                            uint32_t imm19 = (inst >> 5) & 0x7FFFF;
                            int32_t off = (imm19 & 0x40000)
                                ? int32_t(imm19 | 0xFFF80000)
                                : int32_t(imm19);
                            int64_t rt_val = (rd == 31) ? 0 : regs[rd];
                            if (rt_val != 0) {
                                pc = uint64_t(int64_t(pc) + int64_t(off) * 4);
                                branch_taken = true;
                            }
                            break;
                        }

                        // ── RET (BR X30 by convention) ───────────
                        case 0xD6: {
                            // RET is 0xD65F03C0; BR Xn is 0xD61F0000 | (rn << 5)
                            uint8_t br_rn = (inst >> 5) & 0x1F;
                            pc = uint64_t(regs[br_rn]);
                            branch_taken = true;
                            break;
                        }

                        // ── LDR 32-bit (unsigned offset) ─────────
                        case 0xB9: {
                            uint32_t opc = (inst >> 22) & 0x3;
                            if (opc == 1) { // LDR
                                uint64_t addr = uint64_t(rn_val) + uint64_t(imm12) * 4;
                                int32_t val = int32_t(read_u32_le(memory, uint32_t(addr)));
                                if (rd != 31) regs[rd] = int64_t(val);
                            } else if (opc == 0) { // STR (0xB9 with opc=0)
                                uint64_t addr = uint64_t(rn_val) + uint64_t(imm12) * 4;
                                write_u32_le(memory, uint32_t(addr), uint32_t(uint64_t(regs[rd])));
                            }
                            break;
                        }

                        // ── LDRB / STRB (unsigned offset) ────────
                        case 0x39: {
                            uint32_t opc = (inst >> 22) & 0x3;
                            uint64_t addr = uint64_t(rn_val) + uint64_t(imm12);
                            if (opc == 1) { // LDRB
                                if (rd != 31) regs[rd] = int64_t(memory[addr]);
                            } else if (opc == 0) { // STRB
                                memory[addr] = uint8_t(regs[rd]);
                            }
                            break;
                        }

                        // ── LDP (32-bit, signed offset) ──────────
                        case 0x29: {
                            uint8_t rt2 = (inst >> 10) & 0x1F;
                            int32_t imm7 = (inst >> 15) & 0x7F;
                            if (imm7 & 0x40) imm7 |= int32_t(0xFFFFFF80);
                            uint32_t opc = (inst >> 22) & 0x7;
                            uint64_t addr = uint64_t(rn_val) + int64_t(imm7) * 4;
                            if (opc == 5 || opc == 1) { // LDP
                                if (rd != 31)  regs[rd]  = int64_t(int32_t(read_u32_le(memory, uint32_t(addr))));
                                if (rt2 != 31) regs[rt2] = int64_t(int32_t(read_u32_le(memory, uint32_t(addr + 4))));
                            } else { // STP
                                write_u32_le(memory, uint32_t(addr),     uint32_t(uint64_t(regs[rd])));
                                write_u32_le(memory, uint32_t(addr + 4), uint32_t(uint64_t((rt2 == 31) ? 0 : regs[rt2])));
                            }
                            break;
                        }

                        // ── STP/LDP 64-bit (signed offset) ──────
                        case 0xA9: {
                            uint8_t rt2 = (inst >> 10) & 0x1F;
                            int32_t imm7 = (inst >> 15) & 0x7F;
                            if (imm7 & 0x40) imm7 |= int32_t(0xFFFFFF80);
                            uint32_t opc = (inst >> 22) & 0x7;
                            uint64_t addr = uint64_t(rn_val) + int64_t(imm7) * 8;
                            if (opc == 5 || opc == 1) { // LDP
                                if (rd != 31)  regs[rd]  = int64_t(read_u64_le(memory, uint32_t(addr)));
                                if (rt2 != 31) regs[rt2] = int64_t(read_u64_le(memory, uint32_t(addr + 8)));
                            } else { // STP
                                write_u64_le(memory, uint32_t(addr),     uint64_t(regs[rd]));
                                write_u64_le(memory, uint32_t(addr + 8), uint64_t((rt2 == 31) ? 0 : regs[rt2]));
                            }
                            break;
                        }

                        default:
                            // Unhandled opcode — signal for Python fallback
                            sh_done = 3;
                            break;
                    }

                    if (!branch_taken && !sh_done) {
                        pc += 4;
                    }
                    cycles++;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

    } // end main loop

    // ════════════════════════════════════════════════════════════════
    // WRITE OUTPUTS (thread 0 only)
    // ════════════════════════════════════════════════════════════════
    if (tid == 0) {
        for (int i = 0; i < 32; i++) registers[i] = regs[i];
        pc_ptr[0] = pc;
        flags[0] = flag_n; flags[1] = flag_z; flags[2] = flag_c; flags[3] = flag_v;
        *total_cycles_ptr = cycles;

        uint32_t sig = SIGNAL_RUNNING;
        if (sh_done == 1) sig = SIGNAL_HALT;
        else if (sh_done == 2) sig = SIGNAL_CHECKPOINT;
        else if (sh_done == 3) sig = SIGNAL_SYSCALL;
        atomic_store_explicit(signal_flag, sig, memory_order_relaxed);
    }
}
"##;

// ─────────────────────────────────────────────────────────────────────────────
// Rust struct: NeuralFastCPU
// ─────────────────────────────────────────────────────────────────────────────

/// SVC buffer constants (must match Metal shader)
const SVC_BUF_BASE: usize = 0x3F0000;
const SVC_BUF_HDR: usize = 16;
const SVC_HEAP_BASE: usize = 0x60000;

/// Weight buffer sizes
const MUL_LUT_LEN: usize = 256 * 256 * 16; // 1,048,576

/// Threadgroup size for cooperative MLP — must match shader
const TG_SIZE: usize = 64;

fn make_buf_f32_fast(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let bytes = data.len() * 4;
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferCreationFailed)?;
    unsafe {
        let ptr = buf.contents().as_ptr() as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
    Ok(buf)
}

/// Neural ARM64 Fast GPU CPU — focused cooperative-threadgroup kernel.
///
/// Covers ~20 most common instructions with maximum performance.
/// Every ALU operation routes through trained neural network weights.
/// Uses 1 threadgroup of 64 threads for cooperative Kogge-Stone CLA.
///
/// Simplified buffer layout (9 buffers vs 13 for NeuralFullARM64CPU):
///   No shift LUTs, no mem_size, no batch_count — lean and fast.
#[pyclass(unsendable)]
#[allow(dead_code)]
pub struct NeuralFastCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Shared memory buffers
    memory_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    max_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Atomic signal + counter
    signal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Neural weight buffers
    neural_weights_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    mul_lut_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    memory_size: usize,
    cycles_per_batch: u32,
}

#[pymethods]
impl NeuralFastCPU {
    #[new]
    #[pyo3(signature = (memory_size = 4 * 1024 * 1024, cycles_per_batch = 10_000_000))]
    fn new(memory_size: usize, cycles_per_batch: u32) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;
        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        // Compile the fast neural CPU shader
        let source = NSString::from_str(NEURAL_FAST_SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let function_name = NSString::from_str("neural_arm64_fast");
        let function = library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| {
                MetalError::ShaderCompilationFailed("neural_arm64_fast not found".to_string())
            })?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        let shared = MTLResourceOptions::StorageModeShared;

        let memory_buffer = device
            .newBufferWithLength_options(memory_size, shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let registers_buffer = device
            .newBufferWithLength_options(32 * std::mem::size_of::<i64>(), shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let pc_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u64>(), shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let flags_buffer = device
            .newBufferWithLength_options(4 * std::mem::size_of::<f32>(), shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let max_cycles_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let signal_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let total_cycles_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), shared)
            .ok_or(MetalError::BufferCreationFailed)?;

        // Initialize config
        unsafe {
            *(max_cycles_buffer.contents().as_ptr() as *mut u32) = cycles_per_batch;
            // Initialize SVC BRK heap pointer
            if memory_size > SVC_BUF_BASE + SVC_BUF_HDR {
                let mem_ptr = memory_buffer.contents().as_ptr() as *mut u8;
                let brk_bytes = (SVC_HEAP_BASE as u64).to_le_bytes();
                for (i, &b) in brk_bytes.iter().enumerate() {
                    *mem_ptr.add(SVC_BUF_BASE + 8 + i) = b;
                }
            }
        }

        Ok(NeuralFastCPU {
            device,
            command_queue,
            pipeline,
            memory_buffer,
            registers_buffer,
            pc_buffer,
            flags_buffer,
            max_cycles_buffer,
            signal_buffer,
            total_cycles_buffer,
            neural_weights_buffer: None,
            mul_lut_buffer: None,
            memory_size,
            cycles_per_batch,
        })
    }

    /// Load carry_combiner weights + truth tables into the neural weights buffer.
    ///
    /// cc_weights: flat f32 list of length 2466 (FC1 w/b, FC2 w/b, FC3 w/b)
    /// truth_tables: flat f32 list of length 28 ([7, 4] row-major, raw logits)
    fn load_neural_weights(
        &mut self,
        cc_weights: Vec<f32>,
        truth_tables: Vec<f32>,
    ) -> PyResult<()> {
        if cc_weights.len() != 2466 {
            return Err(PyRuntimeError::new_err(format!(
                "cc_weights must be 2466 floats, got {}",
                cc_weights.len()
            )));
        }
        if truth_tables.len() != 28 {
            return Err(PyRuntimeError::new_err(format!(
                "truth_tables must be 28 floats, got {}",
                truth_tables.len()
            )));
        }
        let mut combined = cc_weights;
        combined.extend_from_slice(&truth_tables); // total: 2494
        self.neural_weights_buffer = Some(
            make_buf_f32_fast(&self.device, &combined)
                .map_err(|e| PyRuntimeError::new_err(format!("neural weights buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Load the multiply LUT into the GPU buffer.
    ///
    /// lut_flat: flat f32 list of length 256*256*16 = 1,048,576
    fn load_mul_lut(&mut self, lut_flat: Vec<f32>) -> PyResult<()> {
        if lut_flat.len() != MUL_LUT_LEN {
            return Err(PyRuntimeError::new_err(format!(
                "mul_lut must be {} floats, got {}",
                MUL_LUT_LEN,
                lut_flat.len()
            )));
        }
        self.mul_lut_buffer = Some(
            make_buf_f32_fast(&self.device, &lut_flat)
                .map_err(|e| PyRuntimeError::new_err(format!("mul lut buf: {e:?}")))?,
        );
        Ok(())
    }

    /// Check if all neural weights are loaded and ready
    fn is_ready(&self) -> bool {
        self.neural_weights_buffer.is_some() && self.mul_lut_buffer.is_some()
    }

    /// Load binary data into GPU memory at the given address
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

    /// Write raw bytes to GPU memory
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

    /// Read bytes from GPU memory
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

    fn set_pc(&self, pc: u64) {
        unsafe {
            *(self.pc_buffer.contents().as_ptr() as *mut u64) = pc;
        }
    }

    fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buffer.contents().as_ptr() as *const u64) }
    }

    fn set_register(&self, reg: usize, value: i64) {
        if reg >= 32 {
            return;
        }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *mut i64;
            *ptr.add(reg) = value;
        }
    }

    fn get_register(&self, reg: usize) -> i64 {
        if reg >= 32 {
            return 0;
        }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *const i64;
            *ptr.add(reg)
        }
    }

    fn set_flag(&self, index: usize, value: f32) {
        if index >= 4 {
            return;
        }
        unsafe {
            let ptr = self.flags_buffer.contents().as_ptr() as *mut f32;
            *ptr.add(index) = value;
        }
    }

    fn get_flag(&self, index: usize) -> f32 {
        if index >= 4 {
            return 0.0;
        }
        unsafe {
            let ptr = self.flags_buffer.contents().as_ptr() as *const f32;
            *ptr.add(index)
        }
    }

    /// Drain the GPU SVC write buffer — returns list of (fd, data) tuples
    fn drain_svc_buffer(&self) -> PyResult<Vec<(u8, Vec<u8>)>> {
        let mut entries = Vec::new();
        unsafe {
            let mem = self.memory_buffer.contents().as_ptr() as *const u8;
            let write_pos = u32::from_le_bytes([
                *mem.add(SVC_BUF_BASE),
                *mem.add(SVC_BUF_BASE + 1),
                *mem.add(SVC_BUF_BASE + 2),
                *mem.add(SVC_BUF_BASE + 3),
            ]);
            let entry_count = u32::from_le_bytes([
                *mem.add(SVC_BUF_BASE + 4),
                *mem.add(SVC_BUF_BASE + 5),
                *mem.add(SVC_BUF_BASE + 6),
                *mem.add(SVC_BUF_BASE + 7),
            ]);

            if entry_count > 0 {
                let mut offset = 0u32;
                for _ in 0..entry_count {
                    if offset + 3 > write_pos {
                        break;
                    }
                    let base = SVC_BUF_BASE + SVC_BUF_HDR + offset as usize;
                    let fd = *mem.add(base);
                    let len =
                        u16::from_le_bytes([*mem.add(base + 1), *mem.add(base + 2)]) as usize;
                    let mut data = vec![0u8; len];
                    std::ptr::copy_nonoverlapping(mem.add(base + 3), data.as_mut_ptr(), len);
                    entries.push((fd, data));
                    offset += 3 + len as u32;
                }

                let mem_mut = self.memory_buffer.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(mem_mut.add(SVC_BUF_BASE), 0, 8);
            }
        }
        Ok(entries)
    }

    /// Reset CPU state (registers, PC, flags, counters, SVC buffer)
    fn reset(&self) -> PyResult<()> {
        unsafe {
            std::ptr::write_bytes(
                self.registers_buffer.contents().as_ptr() as *mut u8,
                0,
                32 * std::mem::size_of::<i64>(),
            );
            *(self.pc_buffer.contents().as_ptr() as *mut u64) = 0;
            std::ptr::write_bytes(
                self.flags_buffer.contents().as_ptr() as *mut u8,
                0,
                4 * std::mem::size_of::<f32>(),
            );
            *(self.total_cycles_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.signal_buffer.contents().as_ptr() as *mut u32) = 0;
            // Reset SVC buffer
            std::ptr::write_bytes(
                (self.memory_buffer.contents().as_ptr() as *mut u8).add(SVC_BUF_BASE),
                0,
                SVC_BUF_HDR,
            );
            let mem_ptr = self.memory_buffer.contents().as_ptr() as *mut u8;
            let brk_bytes = (SVC_HEAP_BASE as u64).to_le_bytes();
            for (i, &b) in brk_bytes.iter().enumerate() {
                *mem_ptr.add(SVC_BUF_BASE + 8 + i) = b;
            }
        }
        Ok(())
    }

    /// Execute a single mega-batch on GPU — all ALU via neural networks.
    /// Uses cooperative threadgroup parallelism (64 threads) for CLA operations.
    #[pyo3(signature = (max_cycles = 100_000_000))]
    fn execute(&self, max_cycles: u32) -> PyResult<ContinuousResult> {
        let neural_w = self.neural_weights_buffer.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Neural weights not loaded — call load_neural_weights()")
        })?;
        let mul_lut = self.mul_lut_buffer.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("MUL LUT not loaded — call load_mul_lut()")
        })?;

        let start = Instant::now();

        unsafe {
            *(self.total_cycles_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.signal_buffer.contents().as_ptr() as *mut u32) = Signal::Running as u32;
            *(self.max_cycles_buffer.contents().as_ptr() as *mut u32) = max_cycles;
        }

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
            encoder.setBuffer_offset_atIndex(Some(&self.signal_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.total_cycles_buffer), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(neural_w), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(mul_lut), 0, 8);

            // Dispatch 1 threadgroup of 64 threads for cooperative MLP
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let (cycles, signal, pc) = unsafe {
            let c = *(self.total_cycles_buffer.contents().as_ptr() as *const u32);
            let s = *(self.signal_buffer.contents().as_ptr() as *const u32);
            let p = *(self.pc_buffer.contents().as_ptr() as *const u64);
            (c, s, p)
        };

        Ok(ContinuousResult {
            total_cycles: cycles,
            batch_count: 1,
            signal,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            final_pc: pc,
        })
    }

    /// Execute continuously until halt, syscall, or timeout.
    /// Uses cooperative threadgroup parallelism (64 threads) for CLA operations.
    #[pyo3(signature = (max_batches = 1000, timeout_seconds = 60.0))]
    fn execute_continuous(
        &self,
        max_batches: u32,
        timeout_seconds: f64,
    ) -> PyResult<ContinuousResult> {
        let neural_w = self.neural_weights_buffer.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Neural weights not loaded")
        })?;
        let mul_lut = self.mul_lut_buffer.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("MUL LUT not loaded")
        })?;

        let start = Instant::now();
        let timeout = Duration::from_secs_f64(timeout_seconds);

        unsafe {
            *(self.total_cycles_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.signal_buffer.contents().as_ptr() as *mut u32) = Signal::Running as u32;
            *(self.max_cycles_buffer.contents().as_ptr() as *mut u32) = self.cycles_per_batch;
        }

        let mut batches_executed = 0u32;
        let mut total_cycles_sum = 0u32;

        while batches_executed < max_batches && start.elapsed() < timeout {
            unsafe {
                *(self.signal_buffer.contents().as_ptr() as *mut u32) = Signal::Running as u32;
                *(self.total_cycles_buffer.contents().as_ptr() as *mut u32) = 0;
            }

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
                encoder.setBuffer_offset_atIndex(Some(&self.signal_buffer), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&self.total_cycles_buffer), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(neural_w), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(mul_lut), 0, 8);

                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: TG_SIZE,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            batches_executed += 1;

            let batch_cycles =
                unsafe { *(self.total_cycles_buffer.contents().as_ptr() as *const u32) };
            total_cycles_sum += batch_cycles;

            let signal =
                unsafe { Signal::from(*(self.signal_buffer.contents().as_ptr() as *const u32)) };

            if signal == Signal::Halt || signal == Signal::Syscall {
                break;
            }
        }

        let (signal, pc) = unsafe {
            let s = *(self.signal_buffer.contents().as_ptr() as *const u32);
            let p = *(self.pc_buffer.contents().as_ptr() as *const u64);
            (s, p)
        };

        Ok(ContinuousResult {
            total_cycles: total_cycles_sum,
            batch_count: batches_executed,
            signal,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            final_pc: pc,
        })
    }

    /// Read a u64 from memory
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

    /// Read a u32 from memory
    fn read_memory_32(&self, address: usize) -> PyResult<u32> {
        if address + 4 > self.memory_size {
            return Err(PyRuntimeError::new_err("Read exceeds memory bounds"));
        }
        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *const u8;
            let mut bytes = [0u8; 4];
            std::ptr::copy_nonoverlapping(ptr.add(address), bytes.as_mut_ptr(), 4);
            Ok(u32::from_le_bytes(bytes))
        }
    }
}

/// Register neural fast CPU types with Python module
pub fn register_neural_cpu_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuralFastCPU>()?;
    Ok(())
}
