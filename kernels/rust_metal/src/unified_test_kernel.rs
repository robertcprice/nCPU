//! Minimal test kernel to debug Metal dispatch issues
//!
//! This is a stripped down version of UnifiedDiffCPU to isolate the dispatch issue.

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

/// Minimal test shader - just sets some values
const MINIMAL_TEST_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void minimal_test_execute(
    device uint8_t* memory [[buffer(0)]],
    device float* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* config [[buffer(4)]],
    device atomic_uint* signal [[buffer(5)]],
    device uint32_t* stats [[buffer(6)]],
    device const float* weights [[buffer(7)]],
    device float* gradients [[buffer(8)]],
    device const float* temperature [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // IMMEDIATE canary writes at the very start
    registers[30] = 12345.0;  // Canary 1
    gradients[0] = 11111.0;   // Canary 2
    stats[0] = 99;            // Canary 3

    // Read some values to prove we can access buffers
    uint64_t pc = pc_ptr[0];
    uint32_t mem_size = config[1];
    float temp = temperature[0];

    // Write more debug values
    gradients[1] = (float)pc;
    gradients[2] = (float)mem_size;
    gradients[3] = temp;

    // Fetch first instruction
    if (pc + 4 <= mem_size) {
        uint32_t inst = uint32_t(memory[pc]) | (uint32_t(memory[pc+1])<<8) |
                        (uint32_t(memory[pc+2])<<16) | (uint32_t(memory[pc+3])<<24);
        gradients[4] = float(inst >> 16) / 65536.0;  // High bits as float

        // MOVZ Xd, #imm16: 1101 0010 1xhw imm16 rd
        // Check if it's MOVZ
        if ((inst & 0xFF800000) == 0xD2800000) {
            uint8_t rd = inst & 0x1F;
            uint16_t imm16 = (inst >> 5) & 0xFFFF;
            registers[rd] = (float)imm16;
            gradients[5] = 1.0;  // Executed MOVZ
            pc += 4;
        }

        // Check for HLT
        if (pc + 4 <= mem_size) {
            inst = uint32_t(memory[pc]) | (uint32_t(memory[pc+1])<<8) |
                   (uint32_t(memory[pc+2])<<16) | (uint32_t(memory[pc+3])<<24);
            if ((inst & 0xFFE0001F) == 0xD4400000) {
                atomic_store_explicit(signal, 1u, memory_order_relaxed);
                gradients[6] = 1.0;  // Found HLT
            }
        }
    }

    // Final canary to prove kernel completed
    registers[29] = 99999.0;
    stats[1] = 88;
    pc_ptr[0] = pc;
}
"#;

#[pyclass(unsendable)]
pub struct MinimalTestCPU {
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
impl MinimalTestCPU {
    #[new]
    #[pyo3(signature = (memory_size=1024))]
    fn new(memory_size: usize) -> PyResult<Self> {
        let device = get_default_device()
            .ok_or_else(|| PyRuntimeError::new_err("No Metal device available"))?;

        println!("[MinimalTestCPU] Using device: {:?}", device.name());
        println!("[MinimalTestCPU] Compiling minimal test shader...");

        let command_queue = device.newCommandQueue()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create command queue"))?;

        let source = NSString::from_str(MINIMAL_TEST_SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| {
                let err_msg = format!("Shader compilation failed: {:?}", e);
                println!("[MinimalTestCPU] ERROR: {}", err_msg);
                MetalError::ShaderCompilationFailed(err_msg)
            })?;

        println!("[MinimalTestCPU] Shader compiled successfully!");

        let func_name = NSString::from_str("minimal_test_execute");
        let function = library.newFunctionWithName(&func_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("Function not found".to_string()))?;

        println!("[MinimalTestCPU] Function found!");

        let pipeline = device.newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        println!("[MinimalTestCPU] Pipeline created!");

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
        let weights_buf = device.newBufferWithLength_options(4 * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let gradients_buf = device.newBufferWithLength_options(64 * 4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let temperature_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;

        unsafe {
            let cfg = config_buf.contents().as_ptr() as *mut u32;
            *cfg.add(0) = 256;  // cycles_per_batch
            *cfg.add(1) = memory_size as u32;
            *cfg.add(2) = 100;  // jit_threshold (unused)

            let temp = temperature_buf.contents().as_ptr() as *mut f32;
            *temp = 1.0;

            std::ptr::write_bytes(registers_buf.contents().as_ptr() as *mut u8, 0, 32 * 4);
            std::ptr::write_bytes(gradients_buf.contents().as_ptr() as *mut u8, 0, 64 * 4);
        }

        println!("[MinimalTestCPU] Initialized with {} bytes memory", memory_size);

        Ok(Self {
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
        println!("[MinimalTestCPU] Loaded {} bytes at 0x{:x}", program.len(), address);
        Ok(())
    }

    fn set_pc(&self, pc: u64) {
        unsafe { *(self.pc_buf.contents().as_ptr() as *mut u64) = pc; }
    }

    fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buf.contents().as_ptr() as *const u64) }
    }

    fn get_register(&self, reg: usize) -> PyResult<f32> {
        if reg >= 32 { return Err(PyRuntimeError::new_err("Invalid register")); }
        unsafe { Ok(*(self.registers_buf.contents().as_ptr() as *const f32).add(reg)) }
    }

    fn get_gradients(&self) -> Vec<f32> {
        unsafe {
            let ptr = self.gradients_buf.contents().as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, 64).to_vec()
        }
    }

    fn get_stats(&self) -> Vec<u32> {
        unsafe {
            let ptr = self.stats_buf.contents().as_ptr() as *const u32;
            std::slice::from_raw_parts(ptr, 5).to_vec()
        }
    }

    fn execute(&self) -> PyResult<()> {
        println!("[MinimalTestCPU] Starting execute...");

        // Clear buffers
        unsafe {
            *(self.signal_buf.contents().as_ptr() as *mut u32) = 0;
            std::ptr::write_bytes(self.stats_buf.contents().as_ptr() as *mut u8, 0, 20);
        }

        // Debug: show pre-execute state
        let pre_x30 = unsafe { *(self.registers_buf.contents().as_ptr() as *const f32).add(30) };
        println!("[MinimalTestCPU] Pre-execute X30 = {}", pre_x30);

        // Create command buffer
        let cmd = self.command_queue.commandBuffer()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create command buffer"))?;
        println!("[MinimalTestCPU] Command buffer created");

        // Create encoder
        let encoder = cmd.computeCommandEncoder()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create encoder"))?;
        println!("[MinimalTestCPU] Encoder created");

        // Set pipeline
        encoder.setComputePipelineState(&self.pipeline);
        println!("[MinimalTestCPU] Pipeline set");

        // Set buffers
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
            println!("[MinimalTestCPU] Buffers bound");

            let grid = MTLSize { width: 1, height: 1, depth: 1 };
            let tg = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            println!("[MinimalTestCPU] Dispatch submitted: grid={:?}, tg={:?}", grid, tg);
        }

        encoder.endEncoding();
        println!("[MinimalTestCPU] Encoding ended");

        cmd.commit();
        println!("[MinimalTestCPU] Command buffer committed");

        cmd.waitUntilCompleted();
        println!("[MinimalTestCPU] Execution completed");

        // Debug: show post-execute state
        let post_x30 = unsafe { *(self.registers_buf.contents().as_ptr() as *const f32).add(30) };
        let post_signal = unsafe { *(self.signal_buf.contents().as_ptr() as *const u32) };
        println!("[MinimalTestCPU] Post-execute X30 = {}, signal = {}", post_x30, post_signal);

        Ok(())
    }
}

pub fn register_minimal_test_cpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MinimalTestCPU>()?;
    Ok(())
}
