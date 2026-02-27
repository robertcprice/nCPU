//! Trace-Based JIT - Hot loop detection with specialized execution
//!
//! Optimizations:
//! 1. Hot spot detection via execution counting
//! 2. Loop unrolling for hot loops (execute multiple iterations per dispatch)
//! 3. Trace recording and specialized execution paths
//! 4. Adaptive optimization based on runtime profiling

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

/// Trace JIT shader with hot loop unrolling
const TRACE_JIT_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

// Configuration
constant uint32_t CACHE_SIZE = 256;
constant uint32_t MAX_BLOCK_SIZE = 8;       // Smaller blocks for better nested loop handling
constant uint32_t HOT_THRESHOLD = 20;       // Balanced threshold
constant uint32_t UNROLL_FACTOR = 4;        // Conservative unroll factor

struct CacheEntry {
    uint64_t start_pc;
    uint32_t inst_count;
    uint32_t hit_count;
    uint32_t instructions[MAX_BLOCK_SIZE];
    uint8_t is_loop;
    uint8_t is_hot;                        // Hot flag for JIT optimization
    uint8_t loop_iterations;              // Detected loop iteration count
    uint8_t pad;
};

inline uint32_t hash_pc(uint64_t pc) {
    // Simple hash matching BBCache for consistency
    return (uint32_t)((pc >> 2) % CACHE_SIZE);
}

inline uint64_t load64(device uint8_t* mem, uint64_t addr, uint32_t size) {
    if (addr + 8 > size) return 0;
    return uint64_t(mem[addr]) | (uint64_t(mem[addr+1])<<8) | (uint64_t(mem[addr+2])<<16) |
           (uint64_t(mem[addr+3])<<24) | (uint64_t(mem[addr+4])<<32) | (uint64_t(mem[addr+5])<<40) |
           (uint64_t(mem[addr+6])<<48) | (uint64_t(mem[addr+7])<<56);
}

inline void store64(device uint8_t* mem, uint64_t addr, uint64_t val, uint32_t size) {
    if (addr + 8 > size) return;
    mem[addr] = val & 0xFF; mem[addr+1] = (val>>8) & 0xFF;
    mem[addr+2] = (val>>16) & 0xFF; mem[addr+3] = (val>>24) & 0xFF;
    mem[addr+4] = (val>>32) & 0xFF; mem[addr+5] = (val>>40) & 0xFF;
    mem[addr+6] = (val>>48) & 0xFF; mem[addr+7] = (val>>56) & 0xFF;
}

inline bool check_cond(uint8_t cond, float N, float Z, float C, float V) {
    switch (cond) {
        case 0x0: return (Z > 0.5);        // EQ
        case 0x1: return (Z < 0.5);        // NE
        case 0x2: return (C > 0.5);        // CS/HS
        case 0x3: return (C < 0.5);        // CC/LO
        case 0xA: return ((N > 0.5) == (V > 0.5));  // GE
        case 0xB: return ((N > 0.5) != (V > 0.5));  // LT
        case 0xC: return (Z < 0.5 && (N > 0.5) == (V > 0.5));  // GT
        case 0xD: return (Z > 0.5 || (N > 0.5) != (V > 0.5));  // LE
        default: return false;
    }
}

// Execute a single instruction, return true if branch taken
inline bool exec_inst(uint32_t inst, thread int64_t* regs, thread float& N, thread float& Z,
                      thread float& C, thread float& V, device uint8_t* memory, uint32_t mem_size,
                      thread uint64_t& pc, uint32_t inst_offset) {
    uint8_t rd = inst & 0x1F;
    uint8_t rn = (inst >> 5) & 0x1F;
    uint8_t rm = (inst >> 16) & 0x1F;
    uint16_t imm12 = (inst >> 10) & 0xFFF;
    uint16_t imm16 = (inst >> 5) & 0xFFFF;
    uint8_t hw = (inst >> 21) & 0x3;

    // ADD immediate
    if ((inst & 0xFF000000) == 0x91000000) {
        if (rd < 31) regs[rd] = regs[rn] + imm12;
        return false;
    }
    // SUB immediate
    if ((inst & 0xFF000000) == 0xD1000000) {
        if (rd < 31) regs[rd] = regs[rn] - imm12;
        return false;
    }
    // SUBS immediate (including CMP)
    if ((inst & 0xFF000000) == 0xF1000000) {
        int64_t rv = (rn == 31) ? 0 : regs[rn];
        int64_t res = rv - imm12;
        if (rd < 31) regs[rd] = res;
        N = (res < 0) ? 1.0 : 0.0;
        Z = (res == 0) ? 1.0 : 0.0;
        C = ((uint64_t)rv >= imm12) ? 1.0 : 0.0;
        V = 0.0;
        return false;
    }
    // SUBS register
    if ((inst & 0xFF200000) == 0xEB000000) {
        int64_t rv = (rn == 31) ? 0 : regs[rn];
        int64_t rmv = (rm == 31) ? 0 : regs[rm];
        int64_t res = rv - rmv;
        if (rd < 31) regs[rd] = res;
        N = (res < 0) ? 1.0 : 0.0;
        Z = (res == 0) ? 1.0 : 0.0;
        C = ((uint64_t)rv >= (uint64_t)rmv) ? 1.0 : 0.0;
        V = (((rv ^ rmv) & (rv ^ res)) < 0) ? 1.0 : 0.0;
        return false;
    }
    // ADD register
    if ((inst & 0xFF200000) == 0x8B000000) {
        int64_t rv = (rn == 31) ? 0 : regs[rn];
        int64_t rmv = (rm == 31) ? 0 : regs[rm];
        if (rd < 31) regs[rd] = rv + rmv;
        return false;
    }
    // SUB register
    if ((inst & 0xFF200000) == 0xCB000000) {
        int64_t rv = (rn == 31) ? 0 : regs[rn];
        int64_t rmv = (rm == 31) ? 0 : regs[rm];
        if (rd < 31) regs[rd] = rv - rmv;
        return false;
    }
    // MOVZ
    if ((inst & 0xFF800000) == 0xD2800000) {
        if (rd < 31) regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
        return false;
    }
    // MOVK
    if ((inst & 0xFF800000) == 0xF2800000) {
        if (rd < 31) {
            uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
            regs[rd] = (int64_t)(((uint64_t)regs[rd] & mask) | ((uint64_t)imm16 << (hw * 16)));
        }
        return false;
    }
    // LDR 64-bit
    if ((inst & 0xFFC00000) == 0xF9400000) {
        int64_t base = regs[rn];
        uint64_t addr = (uint64_t)(base + (int64_t)imm12 * 8);
        if (rd < 31) regs[rd] = (int64_t)load64(memory, addr, mem_size);
        return false;
    }
    // STR 64-bit
    if ((inst & 0xFFC00000) == 0xF9000000) {
        int64_t base = regs[rn];
        uint64_t addr = (uint64_t)(base + (int64_t)imm12 * 8);
        int64_t val = (rd == 31) ? 0 : regs[rd];
        store64(memory, addr, (uint64_t)val, mem_size);
        return false;
    }
    // B.cond
    if ((inst & 0xFF000010) == 0x54000000) {
        uint8_t cond = inst & 0xF;
        if (check_cond(cond, N, Z, C, V)) {
            int32_t imm19 = (inst >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            pc = pc + inst_offset * 4 + (int64_t)imm19 * 4;
            return true;
        }
        return false;
    }
    // B unconditional
    if ((inst & 0xFC000000) == 0x14000000) {
        int32_t imm26 = inst & 0x3FFFFFF;
        if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
        pc = pc + inst_offset * 4 + (int64_t)imm26 * 4;
        return true;
    }
    // BL
    if ((inst & 0xFC000000) == 0x94000000) {
        int32_t imm26 = inst & 0x3FFFFFF;
        if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
        regs[30] = (int64_t)(pc + inst_offset * 4 + 4);
        pc = pc + inst_offset * 4 + (int64_t)imm26 * 4;
        return true;
    }
    // RET
    if ((inst & 0xFFFFFC00) == 0xD65F0000) {
        pc = (uint64_t)regs[rn];
        return true;
    }
    // CBZ
    if ((inst & 0xFF000000) == 0xB4000000) {
        int64_t rv = (rd == 31) ? 0 : regs[rd];
        if (rv == 0) {
            int32_t imm19 = (inst >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            pc = pc + inst_offset * 4 + (int64_t)imm19 * 4;
            return true;
        }
        return false;
    }
    // CBNZ
    if ((inst & 0xFF000000) == 0xB5000000) {
        int64_t rv = (rd == 31) ? 0 : regs[rd];
        if (rv != 0) {
            int32_t imm19 = (inst >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            pc = pc + inst_offset * 4 + (int64_t)imm19 * 4;
            return true;
        }
        return false;
    }
    return false;
}

kernel void cpu_execute_trace_jit(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* cycles_per_batch [[buffer(4)]],
    device const uint32_t* mem_size [[buffer(5)]],
    device atomic_uint* signal_flag [[buffer(6)]],
    device uint32_t* total_cycles [[buffer(7)]],
    device uint32_t* batch_count [[buffer(8)]],
    device uint32_t* stats [[buffer(9)]],      // [cache_hits, cache_misses, fusions, hot_executions, unrolled_iters]
    device CacheEntry* cache [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t batch_cycles = cycles_per_batch[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;
    uint32_t cache_hits = 0, cache_misses = 0, fusions = 0;
    uint32_t hot_executions = 0, unrolled_iterations = 0;
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

        uint32_t idx = hash_pc(pc);
        device CacheEntry* entry = &cache[idx];
        bool hit = (entry->start_pc == pc && entry->inst_count > 0);

        if (!hit) {
            cache_misses++;
            entry->start_pc = pc;
            entry->inst_count = 0;
            entry->hit_count = 0;
            entry->is_loop = 0;
            entry->is_hot = 0;

            // Build the block
            uint64_t bpc = pc;
            while (entry->inst_count < MAX_BLOCK_SIZE && bpc + 4 <= memory_size) {
                uint32_t inst = uint32_t(memory[bpc]) | (uint32_t(memory[bpc+1])<<8) |
                                (uint32_t(memory[bpc+2])<<16) | (uint32_t(memory[bpc+3])<<24);
                entry->instructions[entry->inst_count++] = inst;
                bpc += 4;

                uint8_t top = (inst >> 24) & 0xFF;
                // Check for block-ending instructions
                if ((top & 0xFC) == 0x14 || (top & 0xFC) == 0x94 || top == 0x54 ||
                    top == 0xB4 || top == 0xB5 || top == 0xD6 ||
                    (inst & 0xFFE0001F) == 0xD4400000 || (inst & 0xFFE0001F) == 0xD4000001) {
                    // Check if it's a tight loop (branches back to start of this block)
                    if (top == 0x54) {
                        int32_t imm19 = (inst >> 5) & 0x7FFFF;
                        if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;  // Sign extend
                        // Calculate target PC (bpc-4 because we already incremented)
                        uint64_t branch_pc = bpc - 4;
                        uint64_t target = branch_pc + (int64_t)imm19 * 4;
                        // Only mark as loop if it branches back to the block start
                        if (target == pc) entry->is_loop = 1;
                    }
                    break;
                }
            }
        } else {
            cache_hits++;
            entry->hit_count++;

            // Promote to hot if threshold reached
            if (entry->hit_count >= HOT_THRESHOLD && !entry->is_hot) {
                entry->is_hot = 1;
            }
        }

        // Check for halt/syscall in first instruction
        uint32_t first_inst = entry->instructions[0];
        if ((first_inst & 0xFFE0001F) == 0xD4400000) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            should_exit = true;
            break;
        }
        if ((first_inst & 0xFFE0001F) == 0xD4000001) {
            atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
            should_exit = true;
            break;
        }

        // Hot loop path with unrolling
        if (entry->is_hot && entry->is_loop) {
            hot_executions++;

            // Execute multiple iterations of the loop
            uint32_t unroll_count = 0;
            bool loop_done = false;

            while (unroll_count < UNROLL_FACTOR && !loop_done && cycles < batch_cycles && !should_exit) {
                uint32_t i = 0;
                bool branch_taken = false;

                // Execute all instructions in the block
                while (i < entry->inst_count && !should_exit) {
                    uint32_t inst = entry->instructions[i];

                    // Fusion: CMP + B.cond
                    if (i + 1 < entry->inst_count) {
                        uint32_t inst2 = entry->instructions[i + 1];

                        // CMP reg + B.cond fusion
                        if (((inst & 0xFF200000) == 0xEB000000) && ((inst & 0x1F) == 31) &&
                            ((inst2 & 0xFF000010) == 0x54000000)) {
                            uint8_t rn = (inst >> 5) & 0x1F;
                            uint8_t rm = (inst >> 16) & 0x1F;
                            int64_t rv = (rn == 31) ? 0 : regs[rn];
                            int64_t rmv = (rm == 31) ? 0 : regs[rm];
                            int64_t res = rv - rmv;
                            N = (res < 0) ? 1.0 : 0.0;
                            Z = (res == 0) ? 1.0 : 0.0;
                            C = ((uint64_t)rv >= (uint64_t)rmv) ? 1.0 : 0.0;
                            V = (((rv ^ rmv) & (rv ^ res)) < 0) ? 1.0 : 0.0;

                            uint8_t cond = inst2 & 0xF;
                            if (check_cond(cond, N, Z, C, V)) {
                                int32_t imm19 = (inst2 >> 5) & 0x7FFFF;
                                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                                pc = (pc + (i + 1) * 4) + (int64_t)imm19 * 4;
                                branch_taken = true;
                            } else {
                                // Loop exit - advance PC past the block
                                pc += entry->inst_count * 4;
                                loop_done = true;
                            }
                            fusions++;
                            cycles++;
                            if (branch_taken || loop_done) break;
                            continue;
                        }

                        // CMP imm + B.cond fusion
                        if (((inst & 0xFF000000) == 0xF1000000) && ((inst & 0x1F) == 31) &&
                            ((inst2 & 0xFF000010) == 0x54000000)) {
                            uint8_t rn = (inst >> 5) & 0x1F;
                            uint16_t imm12 = (inst >> 10) & 0xFFF;
                            int64_t rv = (rn == 31) ? 0 : regs[rn];
                            int64_t res = rv - imm12;
                            N = (res < 0) ? 1.0 : 0.0;
                            Z = (res == 0) ? 1.0 : 0.0;
                            C = ((uint64_t)rv >= imm12) ? 1.0 : 0.0;
                            V = 0.0;

                            uint8_t cond = inst2 & 0xF;
                            if (check_cond(cond, N, Z, C, V)) {
                                int32_t imm19 = (inst2 >> 5) & 0x7FFFF;
                                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                                pc = (pc + (i + 1) * 4) + (int64_t)imm19 * 4;
                                branch_taken = true;
                            } else {
                                // Loop exit - advance PC past the block
                                pc += entry->inst_count * 4;
                                loop_done = true;
                            }
                            fusions++;
                            cycles++;
                            if (branch_taken || loop_done) break;
                            continue;
                        }
                    }

                    // Normal instruction execution
                    branch_taken = exec_inst(inst, regs, N, Z, C, V, memory, memory_size, pc, i);
                    cycles++;

                    if (branch_taken) break;
                    i++;
                }

                if (!branch_taken && !loop_done) {
                    pc += entry->inst_count * 4;
                }

                unroll_count++;
                unrolled_iterations++;
            }
        }
        // Normal execution path (non-hot or non-loop)
        else {
            uint32_t i = 0;
            bool branch_taken = false;

            while (i < entry->inst_count && cycles < batch_cycles && !should_exit) {
                uint32_t inst = entry->instructions[i];

                // Fusion checks
                if (i + 1 < entry->inst_count) {
                    uint32_t inst2 = entry->instructions[i + 1];

                    // CMP reg + B.cond fusion
                    if (((inst & 0xFF200000) == 0xEB000000) && ((inst & 0x1F) == 31) &&
                        ((inst2 & 0xFF000010) == 0x54000000)) {
                        uint8_t rn = (inst >> 5) & 0x1F;
                        uint8_t rm = (inst >> 16) & 0x1F;
                        int64_t rv = (rn == 31) ? 0 : regs[rn];
                        int64_t rmv = (rm == 31) ? 0 : regs[rm];
                        int64_t res = rv - rmv;
                        N = (res < 0) ? 1.0 : 0.0;
                        Z = (res == 0) ? 1.0 : 0.0;
                        C = ((uint64_t)rv >= (uint64_t)rmv) ? 1.0 : 0.0;
                        V = (((rv ^ rmv) & (rv ^ res)) < 0) ? 1.0 : 0.0;

                        uint8_t cond = inst2 & 0xF;
                        if (check_cond(cond, N, Z, C, V)) {
                            int32_t imm19 = (inst2 >> 5) & 0x7FFFF;
                            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                            pc = (pc + (i + 1) * 4) + (int64_t)imm19 * 4;
                            branch_taken = true;
                        } else {
                            i += 2;
                        }
                        fusions++;
                        cycles++;
                        if (branch_taken) break;
                        continue;
                    }

                    // CMP imm + B.cond fusion
                    if (((inst & 0xFF000000) == 0xF1000000) && ((inst & 0x1F) == 31) &&
                        ((inst2 & 0xFF000010) == 0x54000000)) {
                        uint8_t rn = (inst >> 5) & 0x1F;
                        uint16_t imm12 = (inst >> 10) & 0xFFF;
                        int64_t rv = (rn == 31) ? 0 : regs[rn];
                        int64_t res = rv - imm12;
                        N = (res < 0) ? 1.0 : 0.0;
                        Z = (res == 0) ? 1.0 : 0.0;
                        C = ((uint64_t)rv >= imm12) ? 1.0 : 0.0;
                        V = 0.0;

                        uint8_t cond = inst2 & 0xF;
                        if (check_cond(cond, N, Z, C, V)) {
                            int32_t imm19 = (inst2 >> 5) & 0x7FFFF;
                            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                            pc = (pc + (i + 1) * 4) + (int64_t)imm19 * 4;
                            branch_taken = true;
                        } else {
                            i += 2;
                        }
                        fusions++;
                        cycles++;
                        if (branch_taken) break;
                        continue;
                    }
                }

                // Normal execution
                branch_taken = exec_inst(inst, regs, N, Z, C, V, memory, memory_size, pc, i);
                cycles++;

                if (branch_taken) break;
                i++;
            }

            if (!branch_taken && !should_exit) {
                pc += entry->inst_count * 4;
            }
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
    atomic_fetch_add_explicit((device atomic_uint*)&stats[2], fusions, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[3], hot_executions, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[4], unrolled_iterations, memory_order_relaxed);

    uint32_t sig = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (sig == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

const CACHE_SIZE: usize = 256;
const CACHE_ENTRY_SIZE: usize = 8 + 4 + 4 + (8 * 4) + 4; // 52 bytes (MAX_BLOCK_SIZE=8)

/// Trace JIT execution result
#[pyclass]
#[derive(Debug, Clone)]
pub struct TraceJITResult {
    #[pyo3(get)]
    pub total_cycles: u32,
    #[pyo3(get)]
    pub batch_count: u32,
    #[pyo3(get)]
    pub cache_hits: u32,
    #[pyo3(get)]
    pub cache_misses: u32,
    #[pyo3(get)]
    pub fusions: u32,
    #[pyo3(get)]
    pub hot_executions: u32,
    #[pyo3(get)]
    pub unrolled_iterations: u32,
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
impl TraceJITResult {
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

/// Trace JIT Metal CPU
#[pyclass(unsendable)]
pub struct TraceJITMetalCPU {
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
impl TraceJITMetalCPU {
    #[new]
    #[pyo3(signature = (memory_size=4*1024*1024, cycles_per_batch=10_000_000))]
    fn new(memory_size: usize, cycles_per_batch: u32) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[TraceJITMetalCPU] Using device: {:?}", device.name());

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        let source = NSString::from_str(TRACE_JIT_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let func_name = NSString::from_str("cpu_execute_trace_jit");
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
        let stats_buf = device.newBufferWithLength_options(5 * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cache_buf = device.newBufferWithLength_options(CACHE_SIZE * CACHE_ENTRY_SIZE, opts)
            .ok_or(MetalError::BufferCreationFailed)?;

        unsafe {
            let ptr = cycles_per_batch_buf.contents().as_ptr() as *mut u32;
            *ptr = cycles_per_batch;
            let ptr = mem_size_buf.contents().as_ptr() as *mut u32;
            *ptr = memory_size as u32;
            std::ptr::write_bytes(cache_buf.contents().as_ptr() as *mut u8, 0, CACHE_SIZE * CACHE_ENTRY_SIZE);
        }

        println!("[TraceJITMetalCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[TraceJITMetalCPU] Features: Hot Loop Detection, Loop Unrolling (4x), Fusion");

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
        println!("[TraceJITMetalCPU] Loaded {} bytes at 0x{:x}", program.len(), address);
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
            std::ptr::write_bytes(self.cache_buf.contents().as_ptr() as *mut u8, 0, CACHE_SIZE * CACHE_ENTRY_SIZE);
            *(self.pc_buf.contents().as_ptr() as *mut u64) = 0;
        }
    }

    fn clear_cache(&self) {
        unsafe {
            std::ptr::write_bytes(self.cache_buf.contents().as_ptr() as *mut u8, 0, CACHE_SIZE * CACHE_ENTRY_SIZE);
        }
    }

    #[pyo3(signature = (max_batches=100, timeout_seconds=10.0))]
    fn execute(&self, max_batches: u32, timeout_seconds: f64) -> PyResult<TraceJITResult> {
        let start = Instant::now();

        unsafe {
            *(self.signal_buf.contents().as_ptr() as *mut u32) = 0;
            *(self.total_cycles_buf.contents().as_ptr() as *mut u32) = 0;
            *(self.batch_count_buf.contents().as_ptr() as *mut u32) = 0;
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

        let stats = unsafe { std::slice::from_raw_parts(self.stats_buf.contents().as_ptr() as *const u32, 5) };
        let cache_hits = stats[0];
        let cache_misses = stats[1];
        let fusions = stats[2];
        let hot_executions = stats[3];
        let unrolled_iterations = stats[4];

        let ips = if elapsed > 0.0 { total_cycles as f64 / elapsed } else { 0.0 };
        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64 * 100.0
        } else { 0.0 };

        Ok(TraceJITResult {
            total_cycles,
            batch_count,
            cache_hits,
            cache_misses,
            fusions,
            hot_executions,
            unrolled_iterations,
            signal,
            elapsed_seconds: elapsed,
            ips,
            cache_hit_rate,
        })
    }
}

pub fn register_trace_jit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TraceJITMetalCPU>()?;
    m.add_class::<TraceJITResult>()?;
    Ok(())
}
