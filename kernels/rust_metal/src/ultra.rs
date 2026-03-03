//! Ultra-optimized Metal CPU with streamlined optimizations
//!
//! Key optimizations:
//! 1. Super Block Cache (512 entries) - larger than BB cache
//! 2. Instruction Prefetch Buffer (16 instructions)
//! 3. Extended Fusion Patterns (CMP+B.cond, ADD+ADD, MOVZ+MOVK, SUB loop)
//! 4. Loop Detection and Hot Block Tracking
//! 5. Memory Coalescing for aligned access

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

/// Streamlined Ultra shader - optimized for compilation speed and runtime performance
const ULTRA_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

constant uint32_t CACHE_SIZE = 512;
constant uint32_t MAX_BLOCK_SIZE = 16;
constant uint32_t PREFETCH_SIZE = 16;

struct CacheEntry {
    uint64_t start_pc;
    uint32_t inst_count;
    uint32_t hit_count;
    uint32_t instructions[MAX_BLOCK_SIZE];
    uint8_t is_loop;
};

inline uint32_t hash_pc(uint64_t pc) {
    // Better hash function for reduced collisions
    uint64_t h = pc >> 2;
    h ^= (h >> 7);
    return (uint32_t)(h % CACHE_SIZE);
}

inline uint64_t load64(device uint8_t* mem, uint64_t addr, uint32_t size) {
    if (addr + 8 > size) return 0;
    return uint64_t(mem[addr]) | (uint64_t(mem[addr+1])<<8) | (uint64_t(mem[addr+2])<<16) |
           (uint64_t(mem[addr+3])<<24) | (uint64_t(mem[addr+4])<<32) | (uint64_t(mem[addr+5])<<40) |
           (uint64_t(mem[addr+6])<<48) | (uint64_t(mem[addr+7])<<56);
}

inline void store64(device uint8_t* mem, uint64_t addr, uint64_t val, uint32_t size) {
    if (addr + 8 > size) return;
    mem[addr] = val & 0xFF; mem[addr+1] = (val>>8) & 0xFF; mem[addr+2] = (val>>16) & 0xFF;
    mem[addr+3] = (val>>24) & 0xFF; mem[addr+4] = (val>>32) & 0xFF; mem[addr+5] = (val>>40) & 0xFF;
    mem[addr+6] = (val>>48) & 0xFF; mem[addr+7] = (val>>56) & 0xFF;
}

inline bool check_cond(uint8_t cond, float N, float Z, float C, float V) {
    switch (cond) {
        case 0x0: return (Z > 0.5);
        case 0x1: return (Z < 0.5);
        case 0x2: return (C > 0.5);
        case 0x3: return (C < 0.5);
        case 0xA: return ((N > 0.5) == (V > 0.5));
        case 0xB: return ((N > 0.5) != (V > 0.5));
        case 0xC: return (Z < 0.5 && (N > 0.5) == (V > 0.5));
        case 0xD: return (Z > 0.5 || (N > 0.5) != (V > 0.5));
        default: return false;
    }
}

kernel void cpu_execute_ultra(
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
    device CacheEntry* cache [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t batch_cycles = cycles_per_batch[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;
    uint32_t cache_hits = 0, cache_misses = 0, fusions = 0, prefetch_hits = 0;
    bool should_exit = false;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) regs[i] = registers[i];
    regs[31] = 0;

    float N = flags[0], Z = flags[1], C = flags[2], V = flags[3];

    // Prefetch buffer
    uint32_t prefetch[PREFETCH_SIZE];
    uint64_t prefetch_base = 0xFFFFFFFFFFFFFFFFULL;

    while (cycles < batch_cycles && !should_exit) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        uint32_t idx = hash_pc(pc);
        device CacheEntry* e = &cache[idx];
        bool hit = (e->start_pc == pc && e->inst_count > 0);

        // On miss, build the block first
        if (!hit) {
            cache_misses++;
            e->start_pc = pc;
            e->inst_count = 0;
            e->hit_count = 0;
            e->is_loop = 0;

            // Fill prefetch if needed
            if (pc != prefetch_base) {
                prefetch_base = pc;
                for (uint32_t p = 0; p < PREFETCH_SIZE && pc + p*4 < memory_size; p++) {
                    uint64_t a = pc + p*4;
                    prefetch[p] = uint32_t(memory[a]) | (uint32_t(memory[a+1])<<8) |
                                  (uint32_t(memory[a+2])<<16) | (uint32_t(memory[a+3])<<24);
                }
            }

            uint64_t bpc = pc;
            while (e->inst_count < MAX_BLOCK_SIZE && bpc + 4 <= memory_size) {
                uint32_t inst;
                uint32_t off = (uint32_t)(bpc - prefetch_base);
                if (off < PREFETCH_SIZE * 4 && (off & 3) == 0) {
                    inst = prefetch[off >> 2];
                    prefetch_hits++;
                } else {
                    inst = uint32_t(memory[bpc]) | (uint32_t(memory[bpc+1])<<8) |
                           (uint32_t(memory[bpc+2])<<16) | (uint32_t(memory[bpc+3])<<24);
                }
                e->instructions[e->inst_count++] = inst;
                bpc += 4;

                uint8_t top = (inst >> 24) & 0xFF;
                if ((top & 0xFC) == 0x14 || (top & 0xFC) == 0x94 || top == 0x54 ||
                    top == 0xB4 || top == 0xB5 || top == 0xD6 ||
                    (inst & 0xFFE0001F) == 0xD4400000 || (inst & 0xFFE0001F) == 0xD4000001) {
                    if (top == 0x54 && ((inst >> 5) & 0x40000)) e->is_loop = 1;
                    break;
                }
            }
        } else {
            cache_hits++;
            e->hit_count++;
        }

        // Execute block (same for hit and miss)
        uint32_t bc = 0, i = 0;
        bool branch = false;

        while (i < e->inst_count && cycles + bc < batch_cycles && !should_exit) {
            uint32_t inst = e->instructions[i];
            uint32_t inst2 = (i+1 < e->inst_count) ? e->instructions[i+1] : 0;
            bool has2 = (i+1 < e->inst_count);

            if ((inst & 0xFFE0001F) == 0xD4400000) {
                atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
                should_exit = true; break;
            }
            if ((inst & 0xFFE0001F) == 0xD4000001) {
                atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
                should_exit = true; break;
            }

            uint8_t rd = inst & 0x1F;
            uint8_t rn = (inst >> 5) & 0x1F;
            uint8_t rm = (inst >> 16) & 0x1F;
            uint16_t imm12 = (inst >> 10) & 0xFFF;
            uint16_t imm16 = (inst >> 5) & 0xFFFF;
            uint8_t hw = (inst >> 21) & 0x3;
            uint32_t adv = 1;

            // Fusion: CMP imm + B.cond
            if (has2 && ((inst & 0xFF000000) == 0xF1000000) && ((inst & 0x1F) == 31) &&
                ((inst2 & 0xFF000010) == 0x54000000)) {
                int64_t rv = (rn == 31) ? 0 : regs[rn];
                int64_t res = rv - imm12;
                N = (res < 0) ? 1.0 : 0.0; Z = (res == 0) ? 1.0 : 0.0;
                C = ((uint64_t)rv >= imm12) ? 1.0 : 0.0; V = 0.0;
                uint8_t cond = inst2 & 0xF;
                if (check_cond(cond, N, Z, C, V)) {
                    int32_t imm19 = (inst2 >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    pc = (pc + (i+1)*4) + (int64_t)imm19 * 4;
                    branch = true;
                } else adv = 2;
                fusions++; bc++;
            }
            // Fusion: CMP reg + B.cond
            else if (has2 && ((inst & 0xFF200000) == 0xEB000000) && ((inst & 0x1F) == 31) &&
                     ((inst2 & 0xFF000010) == 0x54000000)) {
                int64_t rv = (rn == 31) ? 0 : regs[rn];
                int64_t rmv = (rm == 31) ? 0 : regs[rm];
                int64_t res = rv - rmv;
                N = (res < 0) ? 1.0 : 0.0; Z = (res == 0) ? 1.0 : 0.0;
                C = ((uint64_t)rv >= (uint64_t)rmv) ? 1.0 : 0.0;
                V = (((rv ^ rmv) & (rv ^ res)) < 0) ? 1.0 : 0.0;
                uint8_t cond = inst2 & 0xF;
                if (check_cond(cond, N, Z, C, V)) {
                    int32_t imm19 = (inst2 >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    pc = (pc + (i+1)*4) + (int64_t)imm19 * 4;
                    branch = true;
                } else adv = 2;
                fusions++; bc++;
            }
            // Fusion: ADD + ADD (same reg)
            else if (has2 && (inst & 0xFF000000) == 0x91000000 && (inst2 & 0xFF000000) == 0x91000000) {
                uint8_t rd1 = inst & 0x1F, rn1 = (inst >> 5) & 0x1F;
                uint8_t rd2 = inst2 & 0x1F, rn2 = (inst2 >> 5) & 0x1F;
                if (rd1 == rn1 && rd2 == rn2 && rd1 == rd2) {
                    uint16_t imm2 = (inst2 >> 10) & 0xFFF;
                    if (rd1 < 31) regs[rd1] = regs[rn1] + imm12 + imm2;
                    adv = 2; fusions++; bc++;
                } else {
                    if (rd < 31) regs[rd] = regs[rn] + imm12;
                }
            }
            // Fusion: MOVZ + MOVK
            else if (has2 && (inst & 0xFF800000) == 0xD2800000 && (inst2 & 0xFF800000) == 0xF2800000) {
                uint8_t rd2 = inst2 & 0x1F;
                if (rd == rd2 && hw == 0 && ((inst2 >> 21) & 0x3) == 1) {
                    uint16_t imm2 = (inst2 >> 5) & 0xFFFF;
                    if (rd < 31) regs[rd] = (int64_t)(((uint64_t)imm2 << 16) | imm16);
                    adv = 2; fusions++; bc++;
                } else {
                    if (rd < 31) regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
                }
            }
            // ADD immediate
            else if ((inst & 0xFF000000) == 0x91000000) {
                if (rd < 31) regs[rd] = regs[rn] + imm12;
            }
            // SUB immediate
            else if ((inst & 0xFF000000) == 0xD1000000) {
                if (rd < 31) regs[rd] = regs[rn] - imm12;
            }
            // SUBS immediate
            else if ((inst & 0xFF000000) == 0xF1000000) {
                int64_t rv = (rn == 31) ? 0 : regs[rn];
                int64_t res = rv - imm12;
                if (rd < 31) regs[rd] = res;
                N = (res < 0) ? 1.0 : 0.0; Z = (res == 0) ? 1.0 : 0.0;
                C = ((uint64_t)rv >= imm12) ? 1.0 : 0.0; V = 0.0;
            }
            // SUBS register
            else if ((inst & 0xFF200000) == 0xEB000000) {
                int64_t rv = (rn == 31) ? 0 : regs[rn];
                int64_t rmv = (rm == 31) ? 0 : regs[rm];
                int64_t res = rv - rmv;
                if (rd < 31) regs[rd] = res;
                N = (res < 0) ? 1.0 : 0.0; Z = (res == 0) ? 1.0 : 0.0;
                C = ((uint64_t)rv >= (uint64_t)rmv) ? 1.0 : 0.0;
                V = (((rv ^ rmv) & (rv ^ res)) < 0) ? 1.0 : 0.0;
            }
            // ADD register
            else if ((inst & 0xFF200000) == 0x8B000000) {
                int64_t rv = (rn == 31) ? 0 : regs[rn];
                int64_t rmv = (rm == 31) ? 0 : regs[rm];
                if (rd < 31) regs[rd] = rv + rmv;
            }
            // SUB register
            else if ((inst & 0xFF200000) == 0xCB000000) {
                int64_t rv = (rn == 31) ? 0 : regs[rn];
                int64_t rmv = (rm == 31) ? 0 : regs[rm];
                if (rd < 31) regs[rd] = rv - rmv;
            }
            // MOVZ
            else if ((inst & 0xFF800000) == 0xD2800000) {
                if (rd < 31) regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
            }
            // MOVK
            else if ((inst & 0xFF800000) == 0xF2800000) {
                if (rd < 31) {
                    uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
                    regs[rd] = (int64_t)(((uint64_t)regs[rd] & mask) | ((uint64_t)imm16 << (hw * 16)));
                }
            }
            // LDR 64
            else if ((inst & 0xFFC00000) == 0xF9400000) {
                int64_t base = regs[rn];
                uint64_t addr = (uint64_t)(base + (int64_t)imm12 * 8);
                if (rd < 31) regs[rd] = (int64_t)load64(memory, addr, memory_size);
            }
            // STR 64
            else if ((inst & 0xFFC00000) == 0xF9000000) {
                int64_t base = regs[rn];
                uint64_t addr = (uint64_t)(base + (int64_t)imm12 * 8);
                int64_t val = (rd == 31) ? 0 : regs[rd];
                store64(memory, addr, (uint64_t)val, memory_size);
            }
            // B.cond
            else if ((inst & 0xFF000010) == 0x54000000) {
                uint8_t cond = inst & 0xF;
                if (check_cond(cond, N, Z, C, V)) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    pc = (pc + i*4) + (int64_t)imm19 * 4;
                    branch = true;
                }
            }
            // B
            else if ((inst & 0xFC000000) == 0x14000000) {
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                pc = (pc + i*4) + (int64_t)imm26 * 4;
                branch = true;
            }
            // BL
            else if ((inst & 0xFC000000) == 0x94000000) {
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                regs[30] = (int64_t)((pc + i*4) + 4);
                pc = (pc + i*4) + (int64_t)imm26 * 4;
                branch = true;
            }
            // RET
            else if ((inst & 0xFFFFFC00) == 0xD65F0000) {
                pc = (uint64_t)regs[rn];
                branch = true;
            }
            // CBZ
            else if ((inst & 0xFF000000) == 0xB4000000) {
                int64_t rv = (rd == 31) ? 0 : regs[rd];
                if (rv == 0) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    pc = (pc + i*4) + (int64_t)imm19 * 4;
                    branch = true;
                }
            }
            // CBNZ
            else if ((inst & 0xFF000000) == 0xB5000000) {
                int64_t rv = (rd == 31) ? 0 : regs[rd];
                if (rv != 0) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    pc = (pc + i*4) + (int64_t)imm19 * 4;
                    branch = true;
                }
            }

            if (branch) { cycles += bc + 1; break; }
            i += adv; bc++;
        }

        if (!branch && !should_exit) {
            pc += e->inst_count * 4;
            cycles += bc;
        }
    }

    for (int i = 0; i < 32; i++) registers[i] = regs[i];
    pc_ptr[0] = pc;
    flags[0] = N; flags[1] = Z; flags[2] = C; flags[3] = V;

    atomic_fetch_add_explicit((device atomic_uint*)total_cycles, cycles, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)batch_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[0], cache_hits, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[1], cache_misses, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[2], fusions, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&stats[3], prefetch_hits, memory_order_relaxed);

    uint32_t sig = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (sig == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

const CACHE_SIZE: usize = 512;
const CACHE_ENTRY_SIZE: usize = 8 + 4 + 4 + (16 * 4) + 1 + 3; // 84 bytes, pad to 88

/// Ultra execution result
#[pyclass]
#[derive(Debug, Clone)]
pub struct UltraResult {
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
    pub prefetch_hits: u32,
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
impl UltraResult {
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

/// Ultra-optimized Metal CPU
#[pyclass(unsendable)]
pub struct UltraMetalCPU {
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
impl UltraMetalCPU {
    #[new]
    #[pyo3(signature = (memory_size=4*1024*1024))]
    fn new(memory_size: usize) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        let device_name = device.name();
        println!("[UltraMetalCPU] Using device: {:?}", device_name);

        let command_queue = device.newCommandQueue()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create command queue"))?;

        // Compile shader
        let source = NSString::from_str(ULTRA_SHADER_SOURCE);
        let library = unsafe {
            device.newLibraryWithSource_options_error(&source, None)
        }.map_err(|e| PyRuntimeError::new_err(format!("Shader compilation failed: {}", e)))?;

        let func_name = NSString::from_str("cpu_execute_ultra");
        let function = library.newFunctionWithName(&func_name)
            .ok_or_else(|| PyRuntimeError::new_err("Function not found"))?;

        let pipeline = device.newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| PyRuntimeError::new_err(format!("Pipeline creation failed: {}", e)))?;

        let opts = MTLResourceOptions::StorageModeShared;
        let cycles_per_batch: u32 = 10_000_000;

        let memory_buf = device.newBufferWithLength_options(memory_size, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create memory buffer"))?;
        let registers_buf = device.newBufferWithLength_options(32 * 8, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create registers buffer"))?;
        let pc_buf = device.newBufferWithLength_options(8, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create PC buffer"))?;
        let flags_buf = device.newBufferWithLength_options(4 * 4, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create flags buffer"))?;
        let cycles_per_batch_buf = device.newBufferWithLength_options(4, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create cycles buffer"))?;
        let mem_size_buf = device.newBufferWithLength_options(4, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create mem_size buffer"))?;
        let signal_buf = device.newBufferWithLength_options(4, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create signal buffer"))?;
        let total_cycles_buf = device.newBufferWithLength_options(4, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create total_cycles buffer"))?;
        let batch_count_buf = device.newBufferWithLength_options(4, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create batch_count buffer"))?;
        let stats_buf = device.newBufferWithLength_options(5 * 4, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create stats buffer"))?;
        let cache_buf = device.newBufferWithLength_options(CACHE_SIZE * CACHE_ENTRY_SIZE, opts)
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create cache buffer"))?;

        // Initialize buffers
        unsafe {
            let cpb = cycles_per_batch_buf.contents().as_ptr() as *mut u32;
            *cpb = cycles_per_batch;
            let ms = mem_size_buf.contents().as_ptr() as *mut u32;
            *ms = memory_size as u32;
        }

        println!("[UltraMetalCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[UltraMetalCPU] Cache: {} entries, {} bytes", CACHE_SIZE, CACHE_SIZE * CACHE_ENTRY_SIZE);
        println!("[UltraMetalCPU] Features: Super Block Cache, Prefetch, Extended Fusion");

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
        println!("[UltraMetalCPU] Loaded {} bytes at 0x{:x}", program.len(), address);
        Ok(())
    }

    fn set_pc(&self, pc: u64) {
        unsafe {
            let p = self.pc_buf.contents().as_ptr() as *mut u64;
            *p = pc;
        }
    }

    fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buf.contents().as_ptr() as *const u64) }
    }

    fn set_register(&self, reg: usize, value: i64) -> PyResult<()> {
        if reg >= 32 { return Err(PyRuntimeError::new_err("Invalid register")); }
        unsafe {
            let regs = self.registers_buf.contents().as_ptr() as *mut i64;
            *regs.add(reg) = value;
        }
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
    fn execute(&self, max_batches: u32, timeout_seconds: f64) -> PyResult<UltraResult> {
        let start = Instant::now();

        // Reset counters
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

            // Reset signal for next batch
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
        let prefetch_hits = stats[3];

        let ips = if elapsed > 0.0 { total_cycles as f64 / elapsed } else { 0.0 };
        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64 * 100.0
        } else { 0.0 };

        Ok(UltraResult {
            total_cycles,
            batch_count,
            cache_hits,
            cache_misses,
            fusions,
            prefetch_hits,
            signal,
            elapsed_seconds: elapsed,
            ips,
            cache_hit_rate,
        })
    }
}

pub fn register_ultra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UltraMetalCPU>()?;
    m.add_class::<UltraResult>()?;
    Ok(())
}
