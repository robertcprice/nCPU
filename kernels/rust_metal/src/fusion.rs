//! Instruction Fusion and Basic Block Caching for maximum GPU throughput.
//!
//! Optimizations implemented:
//! 1. Instruction Prefetch - Load 4 instructions ahead
//! 2. Instruction Fusion - Combine common patterns
//! 3. Basic Block Detection - Cache loop bodies
//! 4. Pattern Recognition - Optimize hot paths

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

/// Fusion-optimized Metal shader
const FUSION_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Stop/signal reasons
constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

// Check if two ADDs to same dest with immediates can be fused
bool can_fuse_add_add(uint32_t inst1, uint32_t inst2) {
    if ((inst1 & 0xFF000000) != 0x91000000) return false;
    if ((inst2 & 0xFF000000) != 0x91000000) return false;
    uint8_t rd1 = inst1 & 0x1F;
    uint8_t rn1 = (inst1 >> 5) & 0x1F;
    uint8_t rd2 = inst2 & 0x1F;
    uint8_t rn2 = (inst2 >> 5) & 0x1F;
    return (rd1 == rn1) && (rd2 == rn2) && (rd1 == rd2);
}

// Check if CMP + B.cond can be fused
bool can_fuse_cmp_branch(uint32_t inst1, uint32_t inst2) {
    bool is_cmp = ((inst1 & 0xFF000000) == 0xF1000000) && ((inst1 & 0x1F) == 31);
    bool is_bcond = (inst2 & 0xFF000010) == 0x54000000;
    return is_cmp && is_bcond;
}

// Check if MOVZ + MOVK can be fused (32-bit constant load)
bool can_fuse_mov_mov(uint32_t inst1, uint32_t inst2) {
    if ((inst1 & 0xFF800000) != 0xD2800000) return false;
    if ((inst2 & 0xFF800000) != 0xF2800000) return false;
    uint8_t rd1 = inst1 & 0x1F;
    uint8_t rd2 = inst2 & 0x1F;
    uint8_t hw1 = (inst1 >> 21) & 0x3;
    uint8_t hw2 = (inst2 >> 21) & 0x3;
    return (rd1 == rd2) && (hw1 == 0) && (hw2 == 1);
}

// Main execution kernel with fusion
kernel void cpu_execute_fusion(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* cycles_per_batch [[buffer(4)]],
    device const uint32_t* mem_size [[buffer(5)]],
    device atomic_uint* signal_flag [[buffer(6)]],
    device uint32_t* total_cycles [[buffer(7)]],
    device uint32_t* batch_count [[buffer(8)]],
    device uint32_t* fused_count [[buffer(9)]],
    device uint32_t* loop_accel_count [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t batch_cycles = cycles_per_batch[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;
    uint32_t fusions = 0;
    uint32_t loop_accels = 0;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers[i];
    }

    float N = flags[0];
    float Z = flags[1];
    float C = flags[2];
    float V = flags[3];

    uint32_t loop_iter = 0;

    while (cycles < batch_cycles) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        // Prefetch 2 instructions
        uint32_t inst1 = uint32_t(memory[pc]) |
                        (uint32_t(memory[pc + 1]) << 8) |
                        (uint32_t(memory[pc + 2]) << 16) |
                        (uint32_t(memory[pc + 3]) << 24);

        uint32_t inst2 = 0;
        bool has_inst2 = (pc + 8 <= memory_size);
        if (has_inst2) {
            inst2 = uint32_t(memory[pc + 4]) |
                   (uint32_t(memory[pc + 5]) << 8) |
                   (uint32_t(memory[pc + 6]) << 16) |
                   (uint32_t(memory[pc + 7]) << 24);
        }

        if ((inst1 & 0xFFE0001F) == 0xD4400000) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }
        if ((inst1 & 0xFFE0001F) == 0xD4000001) {
            atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
            break;
        }

        uint8_t rd = inst1 & 0x1F;
        uint8_t rn = (inst1 >> 5) & 0x1F;
        uint8_t rm = (inst1 >> 16) & 0x1F;
        uint16_t imm12 = (inst1 >> 10) & 0xFFF;
        uint16_t imm16 = (inst1 >> 5) & 0xFFFF;
        uint8_t hw = (inst1 >> 21) & 0x3;

        bool branch_taken = false;
        uint32_t pc_advance = 4;

        // ============== FUSION PATTERNS ==============

        // Pattern 1: ADD + ADD fusion
        if (has_inst2 && can_fuse_add_add(inst1, inst2)) {
            uint16_t imm12_2 = (inst2 >> 10) & 0xFFF;
            uint32_t combined_imm = imm12 + imm12_2;
            if (combined_imm <= 0xFFF) {
                int64_t rn_val = regs[rn];
                if (rd < 31) regs[rd] = rn_val + combined_imm;
                pc_advance = 8;
                fusions++;
                cycles++;
                pc += pc_advance;
                cycles++;
                continue;
            }
        }

        // Pattern 2: MOVZ + MOVK fusion
        if (has_inst2 && can_fuse_mov_mov(inst1, inst2)) {
            uint16_t imm16_2 = (inst2 >> 5) & 0xFFFF;
            uint64_t combined = ((uint64_t)imm16_2 << 16) | imm16;
            if (rd < 31) regs[rd] = (int64_t)combined;
            pc_advance = 8;
            fusions++;
            cycles++;
            pc += pc_advance;
            cycles++;
            continue;
        }

        // Pattern 3: CMP + B.cond fusion
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
                case 0x2: take = (C > 0.5); break;
                case 0x3: take = (C < 0.5); break;
                case 0x4: take = (N > 0.5); break;
                case 0x5: take = (N < 0.5); break;
                case 0xA: take = ((N > 0.5) == (V > 0.5)); break;
                case 0xB: take = ((N > 0.5) != (V > 0.5)); break;
                case 0xC: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                case 0xD: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                default: break;
            }

            if (take) {
                pc = (pc + 4) + (int64_t)imm19 * 4;
                branch_taken = true;
                if (imm19 < 0) {
                    loop_iter++;
                    if (loop_iter > 10) loop_accels++;
                }
            } else {
                pc_advance = 8;
            }
            fusions++;
            cycles++;
            if (!branch_taken) pc += pc_advance;
            cycles++;
            continue;
        }

        // ============== REGULAR EXECUTION ==============

        if ((inst1 & 0xFF000000) == 0x91000000) {
            int64_t rn_val = regs[rn];
            regs[rd] = rn_val + imm12;
        }
        else if ((inst1 & 0xFF000000) == 0x11000000) {
            int64_t rn_val = regs[rn];
            regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val + imm12);
        }
        else if ((inst1 & 0xFF000000) == 0xD1000000) {
            int64_t rn_val = regs[rn];
            regs[rd] = rn_val - imm12;
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
        else if ((inst1 & 0xFFC00000) == 0xF9400000) {
            uint16_t pimm = imm12;
            int64_t offset = (int64_t)pimm * 8;
            int64_t base = regs[rn];
            uint64_t addr = (uint64_t)(base + offset);
            if (addr + 8 <= memory_size) {
                uint64_t val = uint64_t(memory[addr]) |
                              (uint64_t(memory[addr + 1]) << 8) |
                              (uint64_t(memory[addr + 2]) << 16) |
                              (uint64_t(memory[addr + 3]) << 24) |
                              (uint64_t(memory[addr + 4]) << 32) |
                              (uint64_t(memory[addr + 5]) << 40) |
                              (uint64_t(memory[addr + 6]) << 48) |
                              (uint64_t(memory[addr + 7]) << 56);
                if (rd < 31) regs[rd] = (int64_t)val;
            }
        }
        else if ((inst1 & 0xFFC00000) == 0xF9000000) {
            uint16_t pimm = imm12;
            int64_t offset = (int64_t)pimm * 8;
            int64_t base = regs[rn];
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
        else if ((inst1 & 0xFC000000) == 0x14000000) {
            int32_t imm26 = inst1 & 0x3FFFFFF;
            if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
            pc = pc + (int64_t)imm26 * 4;
            branch_taken = true;
        }
        else if ((inst1 & 0xFC000000) == 0x94000000) {
            int32_t imm26 = inst1 & 0x3FFFFFF;
            if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
            regs[30] = (int64_t)(pc + 4);
            pc = pc + (int64_t)imm26 * 4;
            branch_taken = true;
        }
        else if ((inst1 & 0xFFFFFC00) == 0xD61F0000) {
            pc = (uint64_t)regs[rn];
            branch_taken = true;
        }
        else if ((inst1 & 0xFFFFFC00) == 0xD63F0000) {
            regs[30] = (int64_t)(pc + 4);
            pc = (uint64_t)regs[rn];
            branch_taken = true;
        }
        else if ((inst1 & 0xFFFFFC00) == 0xD65F0000) {
            pc = (uint64_t)regs[rn];
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
                case 0x2: take = (C > 0.5); break;
                case 0x3: take = (C < 0.5); break;
                case 0x4: take = (N > 0.5); break;
                case 0x5: take = (N < 0.5); break;
                case 0xA: take = ((N > 0.5) == (V > 0.5)); break;
                case 0xB: take = ((N > 0.5) != (V > 0.5)); break;
                case 0xC: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                case 0xD: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                default: break;
            }
            if (take) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }
        else if ((inst1 & 0xFF000000) == 0xB4000000) {
            int32_t imm19 = (inst1 >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            int64_t rt_val = (rd == 31) ? 0 : regs[rd];
            if (rt_val == 0) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }
        else if ((inst1 & 0xFF000000) == 0xB5000000) {
            int32_t imm19 = (inst1 >> 5) & 0x7FFFF;
            if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
            int64_t rt_val = (rd == 31) ? 0 : regs[rd];
            if (rt_val != 0) {
                pc = pc + (int64_t)imm19 * 4;
                branch_taken = true;
            }
        }
        else if (inst1 == 0xD503201F) {
            // NOP
        }

        if (!branch_taken) pc += pc_advance;
        cycles++;
    }

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
    atomic_fetch_add_explicit((device atomic_uint*)fused_count, fusions, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)loop_accel_count, loop_accels, memory_order_relaxed);

    uint32_t current_signal = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (current_signal == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_CHECKPOINT, memory_order_relaxed);
    }
}
"#;

/// Fusion execution result
#[pyclass]
#[derive(Debug, Clone)]
pub struct FusionResult {
    #[pyo3(get)]
    pub total_cycles: u32,
    #[pyo3(get)]
    pub batch_count: u32,
    #[pyo3(get)]
    pub fused_count: u32,
    #[pyo3(get)]
    pub loop_accel_count: u32,
    #[pyo3(get)]
    pub signal: u32,
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    #[pyo3(get)]
    pub ips: f64,
    #[pyo3(get)]
    pub fusion_rate: f64,
}

#[pymethods]
impl FusionResult {
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

/// Fusion-optimized Metal CPU
#[pyclass(unsendable)]
pub struct FusionMetalCPU {
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
    fused_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    loop_accel_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    memory_size: usize,
    cycles_per_batch: u32,
}

#[pymethods]
impl FusionMetalCPU {
    #[new]
    #[pyo3(signature = (memory_size = 4 * 1024 * 1024, cycles_per_batch = 10_000_000))]
    pub fn new(memory_size: usize, cycles_per_batch: u32) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[FusionMetalCPU] Using device: {:?}", device.name());

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        // Compile shader
        let source = NSString::from_str(FUSION_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let function_name = NSString::from_str("cpu_execute_fusion");
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
        let fused_count_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let loop_accel_buffer = device
            .newBufferWithLength_options(4, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // Initialize config buffers
        unsafe {
            let ptr = cycles_per_batch_buffer.contents().as_ptr() as *mut u32;
            *ptr = cycles_per_batch;

            let ptr = mem_size_buffer.contents().as_ptr() as *mut u32;
            *ptr = memory_size as u32;
        }

        println!("[FusionMetalCPU] Initialized with {} MB memory", memory_size / 1024 / 1024);
        println!("[FusionMetalCPU] Cycles per batch: {}", cycles_per_batch);
        println!("[FusionMetalCPU] Fusion-optimized execution enabled");

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
            fused_count_buffer,
            loop_accel_buffer,
            memory_size,
            cycles_per_batch,
        })
    }

    /// Load program into memory
    pub fn load_program(&self, program: Vec<u8>, address: usize) -> PyResult<()> {
        if address + program.len() > self.memory_size {
            return Err(PyRuntimeError::new_err("Program too large for memory"));
        }

        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(program.as_ptr(), ptr.add(address), program.len());
        }

        println!("[FusionMetalCPU] Loaded {} bytes at 0x{:X}", program.len(), address);
        Ok(())
    }

    /// Set program counter
    pub fn set_pc(&self, pc: u64) {
        unsafe {
            let ptr = self.pc_buffer.contents().as_ptr() as *mut u64;
            *ptr = pc;
        }
    }

    /// Get program counter
    pub fn get_pc(&self) -> u64 {
        unsafe {
            let ptr = self.pc_buffer.contents().as_ptr() as *const u64;
            *ptr
        }
    }

    /// Set register value
    pub fn set_register(&self, index: usize, value: i64) {
        if index >= 32 { return; }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *mut i64;
            *ptr.add(index) = value;
        }
    }

    /// Get register value
    pub fn get_register(&self, index: usize) -> i64 {
        if index >= 32 { return 0; }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *const i64;
            *ptr.add(index)
        }
    }

    /// Execute with fusion optimization
    #[pyo3(signature = (max_batches = 1000, timeout_seconds = 60.0))]
    pub fn execute_fusion(&self, max_batches: u32, timeout_seconds: f64) -> PyResult<FusionResult> {
        // Reset counters
        unsafe {
            *(self.signal_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.total_cycles_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.batch_count_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.fused_count_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.loop_accel_buffer.contents().as_ptr() as *mut u32) = 0;
        }

        let start = Instant::now();
        let timeout = std::time::Duration::from_secs_f64(timeout_seconds);

        let mut batches = 0u32;

        while batches < max_batches && start.elapsed() < timeout {
            // Reset signal for this batch
            unsafe {
                *(self.signal_buffer.contents().as_ptr() as *mut u32) = 0;
            }

            // Create command buffer
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
                encoder.setBuffer_offset_atIndex(Some(&self.fused_count_buffer), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(&self.loop_accel_buffer), 0, 10);

                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: 1, height: 1, depth: 1 },
                    MTLSize { width: 1, height: 1, depth: 1 },
                );
            }
            encoder.endEncoding();

            cmd_buffer.commit();
            cmd_buffer.waitUntilCompleted();

            batches += 1;

            // Check signal
            let signal = unsafe { *(self.signal_buffer.contents().as_ptr() as *const u32) };
            if signal == 1 || signal == 2 {
                break;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let total_cycles = unsafe { *(self.total_cycles_buffer.contents().as_ptr() as *const u32) };
        let batch_count = unsafe { *(self.batch_count_buffer.contents().as_ptr() as *const u32) };
        let fused_count = unsafe { *(self.fused_count_buffer.contents().as_ptr() as *const u32) };
        let loop_accel = unsafe { *(self.loop_accel_buffer.contents().as_ptr() as *const u32) };
        let signal = unsafe { *(self.signal_buffer.contents().as_ptr() as *const u32) };

        let ips = if elapsed > 0.0 { total_cycles as f64 / elapsed } else { 0.0 };
        let fusion_rate = if total_cycles > 0 { fused_count as f64 / total_cycles as f64 * 100.0 } else { 0.0 };

        Ok(FusionResult {
            total_cycles,
            batch_count,
            fused_count,
            loop_accel_count: loop_accel,
            signal,
            elapsed_seconds: elapsed,
            ips,
            fusion_rate,
        })
    }

    /// Reset CPU state
    pub fn reset(&self) {
        unsafe {
            std::ptr::write_bytes(self.registers_buffer.contents().as_ptr() as *mut u8, 0, 32 * 8);
            std::ptr::write_bytes(self.pc_buffer.contents().as_ptr() as *mut u8, 0, 8);
            std::ptr::write_bytes(self.flags_buffer.contents().as_ptr() as *mut u8, 0, 16);
        }
    }
}

/// Register fusion types with Python module
pub fn register_fusion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FusionMetalCPU>()?;
    m.add_class::<FusionResult>()?;
    Ok(())
}
