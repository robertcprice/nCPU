//! Hyper-Optimized GPU CPU Execution - DOOM-optimized shader with reduced branching

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use pyo3::prelude::*;

use crate::{MetalError, get_default_device};

const HYPER_OPTIMIZED_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

kernel void hyper_cpu_execute(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device float* flags [[buffer(3)]],
    device const uint32_t* config [[buffer(4)]],
    device atomic_uint* signal [[buffer(5)]],
    device uint32_t* stats [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    
    uint64_t pc = pc_ptr[0];
    uint32_t max_cycles = config[0];
    uint32_t memory_size = config[1];
    uint32_t cycles = 0;
    
    int64_t r0 = registers[0], r1 = registers[1], r2 = registers[2], r3 = registers[3];
    int64_t r4 = registers[4], r5 = registers[5], r6 = registers[6], r7 = registers[7];
    int64_t r8 = registers[8], r9 = registers[9], r10 = registers[10], r11 = registers[11];
    int64_t r12 = registers[12], r13 = registers[13], r14 = registers[14], r15 = registers[15];
    int64_t r16 = registers[16], r17 = registers[17], r18 = registers[18], r19 = registers[19];
    int64_t r20 = registers[20], r21 = registers[21], r22 = registers[22], r23 = registers[23];
    int64_t r24 = registers[24], r25 = registers[25], r26 = registers[26], r27 = registers[27];
    int64_t r28 = registers[28], r29 = registers[29], r30 = registers[30], r31 = registers[31];
    
    float N = flags[0], Z = flags[1], C = flags[2], V = flags[3];
    
    #define READ_REG(n) ((n) == 0 ? r0 : (n) == 1 ? r1 : (n) == 2 ? r2 : (n) == 3 ? r3 : \
                         (n) == 4 ? r4 : (n) == 5 ? r5 : (n) == 6 ? r6 : (n) == 7 ? r7 : \
                         (n) == 8 ? r8 : (n) == 9 ? r9 : (n) == 10 ? r10 : (n) == 11 ? r11 : \
                         (n) == 12 ? r12 : (n) == 13 ? r13 : (n) == 14 ? r14 : (n) == 15 ? r15 : \
                         (n) == 16 ? r16 : (n) == 17 ? r17 : (n) == 18 ? r18 : (n) == 19 ? r19 : \
                         (n) == 20 ? r20 : (n) == 21 ? r21 : (n) == 22 ? r22 : (n) == 23 ? r23 : \
                         (n) == 24 ? r24 : (n) == 25 ? r26 : (n) == 26 ? r27 : r27 : \
                         (n) == 28 ? r28 : (n) == 29 ? r30 : r31)
    
    #define WRITE_REG(n, val) do { \
        switch(n) { \
            case 0: r0 = val; break; case 1: r1 = val; break; case 2: r2 = val; break; case 3: r3 = val; break; \
            case 4: r4 = val; break; case 5: r5 = val; break; case 6: r6 = val; break; case 7: r7 = val; break; \
            case 8: r8 = val; break; case 9: r9 = val; break; case 10: r10 = val; break; case 11: r11 = val; break; \
            case 12: r12 = val; break; case 13: r13 = val; break; case 14: r14 = val; break; case 15: r15 = val; break; \
            case 16: r16 = val; break; case 17: r17 = val; break; case 18: r18 = val; break; case 19: r19 = val; break; \
            case 20: r20 = val; break; case 21: r21 = val; break; case 22: r22 = val; break; case 23: r23 = val; break; \
            case 24: r24 = val; break; case 25: r25 = val; break; case 26: r26 = val; break; case 27: r27 = val; break; \
            case 28: r28 = val; break; case 29: r29 = val; break; case 30: r30 = val; break; case 31: r31 = val; break; \
        } \
    } while(0)
    
    while (cycles < max_cycles) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal, SIGNAL_HALT, memory_order_relaxed);
            break;
        }
        
        uint32_t inst = *((device uint32_t*)(memory + pc));
        
        if ((inst & 0xFFE0001F) == 0xD4400000) {
            atomic_store_explicit(signal, SIGNAL_HALT, memory_order_relaxed);
            break;
        }
        if ((inst & 0xFFE0001F) == 0xD4000001) {
            atomic_store_explicit(signal, SIGNAL_SYSCALL, memory_order_relaxed);
            break;
        }
        
        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint8_t rm = (inst >> 16) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint16_t imm16 = (inst >> 5) & 0xFFFF;
        uint8_t hw = (inst >> 21) & 0x3;
        uint8_t op_hi = (inst >> 24) & 0xFF;
        
        cycles++;
        pc += 4;
        
        switch (op_hi) {
            case 0x91: {
                int64_t rn_val = (rn == 31) ? r31 : READ_REG(rn);
                if (rd < 31) WRITE_REG(rd, rn_val + imm12);
                else r31 = rn_val + imm12;
                continue;
            }
            case 0x11: {
                int64_t rn_val = READ_REG(rn);
                WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val + imm12));
                continue;
            }
            case 0xD1: {
                int64_t rn_val = (rn == 31) ? r31 : READ_REG(rn);
                if (rd < 31) WRITE_REG(rd, rn_val - imm12);
                else r31 = rn_val - imm12;
                continue;
            }
            case 0x51: {
                int64_t rn_val = READ_REG(rn);
                WRITE_REG(rd, (int64_t)(int32_t)((uint32_t)rn_val - imm12));
                continue;
            }
            case 0x31: {
                int64_t rn_val = READ_REG(rn);
                int32_t result = (int32_t)((uint32_t)rn_val + imm12);
                WRITE_REG(rd, (int64_t)result);
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint32_t)result < (uint32_t)rn_val) ? 1.0 : 0.0;
                continue;
            }
            case 0x71: {
                int64_t rn_val = READ_REG(rn);
                int32_t result = (int32_t)((uint32_t)rn_val - imm12);
                WRITE_REG(rd, (int64_t)result);
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint32_t)rn_val >= imm12) ? 1.0 : 0.0;
                continue;
            }
            case 0xB1: {
                int64_t rn_val = READ_REG(rn);
                int64_t result = rn_val + imm12;
                WRITE_REG(rd, result);
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint64_t)result < (uint64_t)rn_val) ? 1.0 : 0.0;
                continue;
            }
            case 0xF1: {
                int64_t rn_val = READ_REG(rn);
                int64_t result = rn_val - imm12;
                WRITE_REG(rd, result);
                N = (result < 0) ? 1.0 : 0.0;
                Z = (result == 0) ? 1.0 : 0.0;
                C = ((uint64_t)rn_val >= imm12) ? 1.0 : 0.0;
                continue;
            }
            case 0x14: {
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= 0xFE000000;
                pc += imm26 * 4;
                continue;
            }
            case 0x54: {
                uint8_t cond = inst & 0xF;
                bool take = false;
                switch (cond) {
                    case 0: take = (Z == 1.0); break;
                    case 1: take = (Z == 0.0); break;
                    case 10: take = (N == V); break;
                    case 11: take = (N != V); break;
                    case 12: take = (Z == 0.0 && N == V); break;
                    case 13: take = (Z == 1.0 || N != V); break;
                }
                if (take) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= 0xFFF80000;
                    pc += imm19 * 4;
                }
                continue;
            }
            case 0x97: {
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= 0xFE000000;
                r30 = pc;
                pc += imm26 * 4;
                continue;
            }
            case 0xD0: {
                int32_t immlo = (inst >> 29) & 0x3;
                int32_t immhi = (inst >> 5) & 0x7FFFF;
                if (immhi & 0x40000) immhi |= 0xFFF80000;
                int64_t offset = (immhi << 2) | immlo;
                WRITE_REG(rd, pc + offset - 4);
                continue;
            }
            case 0x39: {
                if ((inst & 0xFFC00000) == 0x39400000) {
                    int64_t rn_val = READ_REG(rn);
                    uint16_t imm12 = (inst >> 10) & 0xFFF;
                    uint8_t size = (inst >> 30) & 0x3;
                    uint64_t addr = rn_val + (imm12 << size);
                    if (size == 2) WRITE_REG(rd, (int64_t)*((device uint64_t*)(memory + addr));
                    else if (size == 1) WRITE_REG(rd, (int64_t)(int32_t)*((device uint32_t*)(memory + addr));
                    else if (size == 0) WRITE_REG(rd, (int64_t)(int32_t)*((device uint16_t*)(memory + addr));
                } else if ((inst & 0xFFC00000) == 0x39000000) {
                    int64_t rn_val = READ_REG(rn);
                    uint16_t imm12 = (inst >> 10) & 0xFFF;
                    uint64_t addr = rn_val + imm12;
                    WRITE_REG(rd, (int64_t)(int32_t)(int8_t)memory[addr]);
                }
                continue;
            }
            case 0x38: {
                if ((inst & 0xFFE00000) == 0x38000000) {
                    int64_t rn_val = READ_REG(rn);
                    int16_t imm9 = (inst >> 12) & 0x1FF;
                    if (imm9 & 0x100) imm9 |= 0xFE00;
                    uint64_t addr = rn_val + imm9;
                    memory[addr] = (uint8_t)READ_REG(rd);
                }
                continue;
            }
            case 0xD2: {
                if ((inst & 0xFF800000) == 0xD2800000) {
                    WRITE_REG(rd, (int64_t)((uint64_t)imm16 << (hw * 16)));
                }
                continue;
            }
            case 0x72: {
                if ((inst & 0xFF800000) == 0x72800000) {
                    int64_t current = READ_REG(rd);
                    uint64_t mask = ~(0xFFFFULL << (hw * 16));
                    WRITE_REG(rd, (current & (int64_t)mask) | (int64_t)((uint64_t)imm16 << (hw * 16)));
                }
                continue;
            }
            case 0x1A: {
                int64_t rn_val = READ_REG(rn);
                int64_t rm_val = READ_REG(rm);
                uint8_t cond = (inst >> 12) & 0xF;
                bool take = false;
                switch (cond) {
                    case 0: take = (Z == 1.0); break;
                    case 1: take = (Z == 0.0); break;
                    case 10: take = (N == V); break;
                    case 11: take = (N != V); break;
                    case 12: take = (Z == 0.0 && N == V); break;
                    case 13: take = (Z == 1.0 || N != V); break;
                }
                WRITE_REG(rd, take ? rn_val : rm_val);
                continue;
            }
            case 0xB5: {
                int64_t rn_val = (inst & 0x80000000) ? READ_REG(rn) : (int64_t)(int32_t)READ_REG(rn);
                if (rn_val != 0) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= 0xFFF80000;
                    pc += imm19 * 4;
                }
                continue;
            }
            case 0xB7: {
                int64_t rn_val = (inst & 0x80000000) ? READ_REG(rn) : (int64_t)(int32_t)READ_REG(rn);
                if (rn_val == 0) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= 0xFFF80000;
                    pc += imm19 * 4;
                }
                continue;
            }
            default:
                continue;
        }
        
        uint32_t sig = atomic_load_explicit(signal, memory_order_relaxed);
        if (sig != SIGNAL_RUNNING) break;
    }
    
    registers[0] = r0; registers[1] = r1; registers[2] = r2; registers[3] = r3;
    registers[4] = r4; registers[5] = r5; registers[6] = r6; registers[7] = r7;
    registers[8] = r8; registers[9] = r9; registers[10] = r10; registers[11] = r11;
    registers[12] = r12; registers[13] = r13; registers[14] = r14; registers[15] = r15;
    registers[16] = r16; registers[17] = r17; registers[18] = r18; registers[19] = r19;
    registers[20] = r20; registers[21] = r21; registers[22] = r22; registers[23] = r23;
    registers[24] = r24; registers[25] = r25; registers[26] = r26; registers[27] = r27;
    registers[28] = r28; registers[29] = r29; registers[30] = r30; registers[31] = r31;
    
    flags[0] = N;
    flags[1] = Z;
    flags[2] = C;
    flags[3] = V;
    
    pc_ptr[0] = pc;
    stats[0] = cycles;
}
"#;

#[pyclass(unsendable)]
pub struct HyperOptimizedMetalCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    _command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    memory_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    config_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    signal_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    stats_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    memory_size: usize,
}

#[pyclass]
pub struct HyperOptimizedResult {
    #[pyo3(get, set)]
    pub total_cycles: u64,
    #[pyo3(get, set)]
    pub signal: u32,
    #[pyo3(get, set)]
    pub final_pc: u64,
}

#[pymethods]
impl HyperOptimizedMetalCPU {
    #[new]
    #[pyo3(signature = (memory_size=8*1024*1024))]
    fn new(memory_size: usize) -> PyResult<Self> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[HyperOptimizedMetalCPU] Using device: {:?}", device.name());
        println!("[HyperOptimizedMetalCPU] Compiling ultra-optimized shader...");

        let command_queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;

        let source = NSString::from_str(HYPER_OPTIMIZED_SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let func_name = NSString::from_str("hyper_cpu_execute");
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
        let config_buf = device.newBufferWithLength_options(16, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let signal_buf = device.newBufferWithLength_options(4, opts)
            .ok_or(MetalError::BufferCreationFailed)?;
        let stats_buf = device.newBufferWithLength_options(32, opts)
            .ok_or(MetalError::BufferCreationFailed)?;

        unsafe {
            let cfg = config_buf.contents().as_ptr() as *mut u32;
            *cfg.add(0) = 1_000_000;
            *cfg.add(1) = memory_size as u32;

            let sig = signal_buf.contents().as_ptr() as *mut u32;
            *sig = 0;
        }

        println!("[HyperOptimizedMetalCPU] Initialized with {} MB memory", memory_size / (1024 * 1024));

        Ok(Self {
            device,
            _command_queue: command_queue,
            pipeline,
            memory_buf,
            registers_buf,
            pc_buf,
            flags_buf,
            config_buf,
            signal_buf,
            stats_buf,
            memory_size,
        })
    }

    fn execute(&self, total_cycles: u64) -> PyResult<HyperOptimizedResult> {
        unsafe {
            let cfg = self.config_buf.contents().as_ptr() as *mut u32;
            *cfg.add(0) = total_cycles as u32;
            let sig = self.signal_buf.contents().as_ptr() as *mut u32;
            *sig = 0;
        }

        let command_buffer = self
            ._command_queue
            .commandBuffer()
            .ok_or(MetalError::ExecutionFailed)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ExecutionFailed)?;

        encoder.setComputePipelineState(&self.pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.memory_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.registers_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.pc_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.flags_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.config_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&self.signal_buf), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.stats_buf), 0, 6);

            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
        }

        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let (total_cycles_result, signal, final_pc) = unsafe {
            let total = *(self.stats_buf.contents().as_ptr() as *const u32);
            let sig = *(self.signal_buf.contents().as_ptr() as *const u32);
            let pc = *(self.pc_buf.contents().as_ptr() as *const u64);
            (total, sig, pc)
        };

        Ok(HyperOptimizedResult {
            total_cycles: total_cycles_result as u64,
            signal,
            final_pc,
        })
    }

    fn write_memory(&self, addr: u64, data: &[u8]) {
        if addr as usize + data.len() <= self.memory_size {
            unsafe {
                let dst = self.memory_buf.contents().as_ptr().add(addr as usize) as *mut u8;
                for (i, &byte) in data.iter().enumerate() {
                    *dst.add(i) = byte;
                }
            }
        }
    }

    fn read_memory(&self, addr: u64, size: u64) -> Vec<u8> {
        let mut result = vec![0u8; size as usize];
        if addr as usize + size as usize <= self.memory_size {
            unsafe {
                let src = self.memory_buf.contents().as_ptr().add(addr as usize) as *const u8;
                for i in 0..size as usize {
                    result[i] = *src.add(i);
                }
            }
        }
        result
    }

    fn set_register(&self, reg: u64, value: i64) {
        unsafe {
            let regs = self.registers_buf.contents().as_ptr() as *mut i64;
            *regs.add(reg as usize) = value;
        }
    }

    fn get_register(&self, reg: u64) -> i64 {
        unsafe {
            let regs = self.registers_buf.contents().as_ptr() as *const i64;
            *regs.add(reg as usize)
        }
    }

    fn set_pc(&self, pc: u64) {
        unsafe {
            *(self.pc_buf.contents().as_ptr() as *mut u64) = pc;
        }
    }

    fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buf.contents().as_ptr() as *const u64) }
    }
}

pub fn register_hyper_optimized(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HyperOptimizedMetalCPU>()?;
    m.add_class::<HyperOptimizedResult>()?;
    Ok(())
}
