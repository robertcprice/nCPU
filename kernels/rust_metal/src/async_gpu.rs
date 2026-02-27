//! Async GPU execution - GPU runs completely independently.
//!
//! The GPU executes in the background, Python can poll for completion
//! or do other work while GPU crunches cycles.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::{MetalError, get_default_device};

/// Same optimized shader but designed for async operation
const ASYNC_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_MAX_CYCLES = 3;

kernel void cpu_execute_async(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* max_cycles_ptr [[buffer(4)]],
    device const uint32_t* mem_size [[buffer(5)]],
    device atomic_uint* signal_flag [[buffer(6)]],
    device atomic_uint* cycles_executed [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t max_cycles = max_cycles_ptr[0];
    uint32_t memory_size = mem_size[0];
    uint32_t cycles = 0;

    // Load registers to local
    int64_t regs[32];
    for (int i = 0; i < 32; i++) regs[i] = registers[i];
    regs[31] = 0;

    float N = flags[0], Z = flags[1], C = flags[2], V = flags[3];

    atomic_store_explicit(signal_flag, SIGNAL_RUNNING, memory_order_relaxed);

    while (cycles < max_cycles) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        uint32_t inst = *((device uint32_t*)(memory + pc));

        // Check special instructions
        if ((inst & 0xFFE0001F) == 0xD4400000) {
            atomic_store_explicit(signal_flag, SIGNAL_HALT, memory_order_relaxed);
            break;
        }
        if ((inst & 0xFFE0001F) == 0xD4000001) {
            atomic_store_explicit(signal_flag, SIGNAL_SYSCALL, memory_order_relaxed);
            break;
        }

        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint16_t imm16 = (inst >> 5) & 0xFFFF;
        uint8_t hw = (inst >> 21) & 0x3;
        uint8_t op_hi = (inst >> 24) & 0xFF;

        bool branch_taken = false;

        switch (op_hi) {
            case 0x91: { // ADD imm 64
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                if (rd < 31) regs[rd] = rn_val + imm12;
                break;
            }
            case 0x11: { // ADD imm 32
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                if (rd < 31) regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val + imm12);
                break;
            }
            case 0xD1: { // SUB imm 64
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                if (rd < 31) regs[rd] = rn_val - imm12;
                break;
            }
            case 0x51: { // SUB imm 32
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                if (rd < 31) regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val - imm12);
                break;
            }
            case 0xF1: { // SUBS imm 64
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t result = rn_val - imm12;
                if (rd < 31) regs[rd] = result;
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
                V = 0.0;
                break;
            }
            case 0xD2: case 0xD3: { // MOVZ 64
                if ((inst & 0xFF800000) == 0xD2800000 && rd < 31) {
                    regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
                }
                break;
            }
            case 0x52: case 0x53: { // MOVZ 32
                if ((inst & 0xFF800000) == 0x52800000 && rd < 31) {
                    regs[rd] = (int64_t)(uint32_t)(imm16 << (hw * 16));
                }
                break;
            }
            case 0xF2: case 0xF3: { // MOVK 64
                if ((inst & 0xFF800000) == 0xF2800000 && rd < 31) {
                    uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
                    uint64_t val = (uint64_t)regs[rd] & mask;
                    val |= ((uint64_t)imm16 << (hw * 16));
                    regs[rd] = (int64_t)val;
                }
                break;
            }
            case 0xF9: { // LDR/STR 64
                int64_t base = (rn == 31) ? 0 : regs[rn];
                uint64_t addr = (uint64_t)base + (imm12 << 3);
                if ((inst & 0xFFC00000) == 0xF9400000) { // LDR
                    if (addr + 8 <= memory_size && rd < 31) {
                        regs[rd] = *((device int64_t*)(memory + addr));
                    }
                } else if ((inst & 0xFFC00000) == 0xF9000000) { // STR
                    if (addr + 8 <= memory_size) {
                        int64_t val = (rd == 31) ? 0 : regs[rd];
                        *((device int64_t*)(memory + addr)) = val;
                    }
                }
                break;
            }
            case 0xB9: { // LDR/STR 32
                int64_t base = (rn == 31) ? 0 : regs[rn];
                uint64_t addr = (uint64_t)base + (imm12 << 2);
                if ((inst & 0xFFC00000) == 0xB9400000) { // LDR
                    if (addr + 4 <= memory_size && rd < 31) {
                        regs[rd] = (int64_t)*((device uint32_t*)(memory + addr));
                    }
                } else if ((inst & 0xFFC00000) == 0xB9000000) { // STR
                    if (addr + 4 <= memory_size) {
                        int64_t val = (rd == 31) ? 0 : regs[rd];
                        *((device uint32_t*)(memory + addr)) = (uint32_t)val;
                    }
                }
                break;
            }
            case 0x39: { // LDRB/STRB
                int64_t base = (rn == 31) ? 0 : regs[rn];
                uint64_t addr = (uint64_t)base + imm12;
                if ((inst & 0xFFC00000) == 0x39400000) { // LDRB
                    if (addr < memory_size && rd < 31) {
                        regs[rd] = (int64_t)memory[addr];
                    }
                } else if ((inst & 0xFFC00000) == 0x39000000) { // STRB
                    if (addr < memory_size) {
                        int64_t val = (rd == 31) ? 0 : regs[rd];
                        memory[addr] = (uint8_t)val;
                    }
                }
                break;
            }
            case 0x14: case 0x15: case 0x16: case 0x17: { // B
                if ((inst & 0xFC000000) == 0x14000000) {
                    int32_t imm26 = inst & 0x3FFFFFF;
                    if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                    pc = pc + (int64_t)imm26 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0x94: case 0x95: case 0x96: case 0x97: { // BL
                if ((inst & 0xFC000000) == 0x94000000) {
                    int32_t imm26 = inst & 0x3FFFFFF;
                    if (imm26 & 0x2000000) imm26 |= (int32_t)0xFC000000;
                    regs[30] = (int64_t)(pc + 4);
                    pc = pc + (int64_t)imm26 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0xD6: { // BR/BLR/RET
                if ((inst & 0xFFFFFC00) == 0xD61F0000) {
                    pc = (uint64_t)regs[rn];
                    branch_taken = true;
                } else if ((inst & 0xFFFFFC00) == 0xD63F0000) {
                    regs[30] = (int64_t)(pc + 4);
                    pc = (uint64_t)regs[rn];
                    branch_taken = true;
                } else if ((inst & 0xFFFFFC00) == 0xD65F0000) {
                    pc = (uint64_t)regs[rn];
                    branch_taken = true;
                }
                break;
            }
            case 0xB4: { // CBZ
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                int64_t rt_val = (rd == 31) ? 0 : regs[rd];
                if (rt_val == 0) {
                    pc = pc + (int64_t)imm19 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0xB5: { // CBNZ
                int32_t imm19 = (inst >> 5) & 0x7FFFF;
                if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                int64_t rt_val = (rd == 31) ? 0 : regs[rd];
                if (rt_val != 0) {
                    pc = pc + (int64_t)imm19 * 4;
                    branch_taken = true;
                }
                break;
            }
            case 0x54: { // B.cond
                if ((inst & 0xFF000010) == 0x54000000) {
                    uint8_t cond = inst & 0xF;
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= (int32_t)0xFFF80000;
                    bool take = false;
                    switch (cond) {
                        case 0: take = (Z > 0.5); break;
                        case 1: take = (Z < 0.5); break;
                        case 2: take = (C > 0.5); break;
                        case 3: take = (C < 0.5); break;
                        case 4: take = (N > 0.5); break;
                        case 5: take = (N < 0.5); break;
                        case 6: take = (V > 0.5); break;
                        case 7: take = (V < 0.5); break;
                        case 8: take = (C > 0.5 && Z < 0.5); break;
                        case 9: take = (C < 0.5 || Z > 0.5); break;
                        case 10: take = ((N > 0.5) == (V > 0.5)); break;
                        case 11: take = ((N > 0.5) != (V > 0.5)); break;
                        case 12: take = (Z < 0.5 && (N > 0.5) == (V > 0.5)); break;
                        case 13: take = (Z > 0.5 || (N > 0.5) != (V > 0.5)); break;
                        default: take = true; break;
                    }
                    if (take) {
                        pc = pc + (int64_t)imm19 * 4;
                        branch_taken = true;
                    }
                }
                break;
            }
            default:
                break;
        }

        if (!branch_taken) pc += 4;
        cycles++;

        // Periodically update cycles for monitoring (every 1M cycles)
        if ((cycles & 0xFFFFF) == 0) {
            atomic_store_explicit(cycles_executed, cycles, memory_order_relaxed);
        }
    }

    // Write back
    for (int i = 0; i < 32; i++) registers[i] = regs[i];
    pc_ptr[0] = pc;
    flags[0] = N; flags[1] = Z; flags[2] = C; flags[3] = V;

    atomic_store_explicit(cycles_executed, cycles, memory_order_relaxed);

    uint32_t sig = atomic_load_explicit(signal_flag, memory_order_relaxed);
    if (sig == SIGNAL_RUNNING) {
        atomic_store_explicit(signal_flag, SIGNAL_MAX_CYCLES, memory_order_relaxed);
    }
}
"#;

/// Async execution state
#[pyclass]
#[derive(Clone)]
pub struct AsyncStatus {
    #[pyo3(get)]
    pub is_running: bool,
    #[pyo3(get)]
    pub cycles_executed: u32,
    #[pyo3(get)]
    pub signal: u32,
    #[pyo3(get)]
    pub elapsed_seconds: f64,
}

#[pymethods]
impl AsyncStatus {
    #[getter]
    fn signal_name(&self) -> &str {
        match self.signal {
            0 => "RUNNING",
            1 => "HALT",
            2 => "SYSCALL",
            3 => "MAX_CYCLES",
            _ => "UNKNOWN",
        }
    }

    #[getter]
    fn ips(&self) -> f64 {
        if self.elapsed_seconds > 0.0 {
            self.cycles_executed as f64 / self.elapsed_seconds
        } else {
            0.0
        }
    }
}

/// Async GPU CPU - fire and forget execution
#[pyclass(unsendable)]
pub struct AsyncMetalCPU {
    #[allow(dead_code)]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    memory_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    max_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mem_size_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    signal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    memory_size: usize,

    // Async state
    current_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    start_time: Option<Instant>,
}

#[pymethods]
impl AsyncMetalCPU {
    #[new]
    #[pyo3(signature = (memory_size = 4 * 1024 * 1024))]
    fn new(memory_size: usize) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[AsyncMetalCPU] Using device: {:?}", device.name());

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        let source = NSString::from_str(ASYNC_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let function_name = NSString::from_str("cpu_execute_async");
        let function = library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("Function not found".to_string()))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        let shared = MTLResourceOptions::StorageModeShared;

        let memory_buffer = device.newBufferWithLength_options(memory_size, shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let registers_buffer = device.newBufferWithLength_options(32 * 8, shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let pc_buffer = device.newBufferWithLength_options(8, shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let flags_buffer = device.newBufferWithLength_options(16, shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let max_cycles_buffer = device.newBufferWithLength_options(4, shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let mem_size_buffer = device.newBufferWithLength_options(4, shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let signal_buffer = device.newBufferWithLength_options(4, shared)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cycles_buffer = device.newBufferWithLength_options(4, shared)
            .ok_or(MetalError::BufferCreationFailed)?;

        unsafe {
            *(mem_size_buffer.contents().as_ptr() as *mut u32) = memory_size as u32;
        }

        println!("[AsyncMetalCPU] TRUE ASYNC execution - GPU runs independently!");

        Ok(AsyncMetalCPU {
            device,
            command_queue,
            pipeline,
            memory_buffer,
            registers_buffer,
            pc_buffer,
            flags_buffer,
            max_cycles_buffer,
            mem_size_buffer,
            signal_buffer,
            cycles_buffer,
            memory_size,
            current_command_buffer: None,
            start_time: None,
        })
    }

    fn load_program(&self, program: Vec<u8>, address: usize) -> PyResult<()> {
        if address + program.len() > self.memory_size {
            return Err(PyRuntimeError::new_err("Program exceeds memory"));
        }
        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(program.as_ptr(), ptr.add(address), program.len());
        }
        Ok(())
    }

    fn set_pc(&self, pc: u64) {
        unsafe { *(self.pc_buffer.contents().as_ptr() as *mut u64) = pc; }
    }

    fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buffer.contents().as_ptr() as *const u64) }
    }

    fn set_register(&self, reg: usize, value: i64) {
        if reg < 32 {
            unsafe {
                let ptr = self.registers_buffer.contents().as_ptr() as *mut i64;
                *ptr.add(reg) = value;
            }
        }
    }

    fn get_register(&self, reg: usize) -> i64 {
        if reg < 32 {
            unsafe {
                let ptr = self.registers_buffer.contents().as_ptr() as *const i64;
                *ptr.add(reg)
            }
        } else { 0 }
    }

    /// Start async execution - returns immediately, GPU runs in background
    fn start(&mut self, max_cycles: u32) -> PyResult<()> {
        if self.current_command_buffer.is_some() {
            return Err(PyRuntimeError::new_err("Already running"));
        }

        // Reset state
        unsafe {
            *(self.max_cycles_buffer.contents().as_ptr() as *mut u32) = max_cycles;
            *(self.signal_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.cycles_buffer.contents().as_ptr() as *mut u32) = 0;
        }

        let command_buffer = self.command_queue.commandBuffer()
            .ok_or(MetalError::ExecutionFailed)?;

        let encoder = command_buffer.computeCommandEncoder()
            .ok_or(MetalError::ExecutionFailed)?;

        encoder.setComputePipelineState(&self.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.memory_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.registers_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.pc_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.flags_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.max_cycles_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&self.mem_size_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.signal_buffer), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&self.cycles_buffer), 0, 7);

            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
        }
        encoder.endEncoding();

        // Commit but DON'T wait - GPU runs async!
        command_buffer.commit();

        self.current_command_buffer = Some(command_buffer);
        self.start_time = Some(Instant::now());

        Ok(())
    }

    /// Poll status without blocking
    fn poll(&self) -> AsyncStatus {
        let elapsed = self.start_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);

        let (is_running, cycles, signal) = if let Some(ref cb) = self.current_command_buffer {
            let status = cb.status();
            let running = status == MTLCommandBufferStatus::Scheduled
                       || status == MTLCommandBufferStatus::Committed;

            let cycles = unsafe { *(self.cycles_buffer.contents().as_ptr() as *const u32) };
            let signal = unsafe { *(self.signal_buffer.contents().as_ptr() as *const u32) };

            (running, cycles, signal)
        } else {
            (false, 0, 0)
        };

        AsyncStatus {
            is_running,
            cycles_executed: cycles,
            signal,
            elapsed_seconds: elapsed,
        }
    }

    /// Wait for completion (blocking)
    fn wait(&mut self) -> PyResult<AsyncStatus> {
        if let Some(ref cb) = self.current_command_buffer {
            cb.waitUntilCompleted();
        }

        let status = self.poll();
        self.current_command_buffer = None;
        self.start_time = None;

        Ok(status)
    }

    /// Check if GPU is still running
    fn is_running(&self) -> bool {
        self.poll().is_running
    }

    #[getter]
    fn memory_size(&self) -> usize {
        self.memory_size
    }
}

pub fn register_async(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AsyncMetalCPU>()?;
    m.add_class::<AsyncStatus>()?;
    Ok(())
}
