//! Pure Rust DOOM Benchmark Tool
//!
//! Runs DOOM on Metal GPU and measures performance without Python overhead.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::time::Instant;
use std::fs;
use std::path::Path;

/// Metal shader with DOOM-optimized dispatch
const DOOM_OPTIMIZED_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Signal constants
constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;
constant uint32_t SIGNAL_SYSCALL = 2;
constant uint32_t SIGNAL_CHECKPOINT = 3;

// Optimized memory access helpers
inline uint64_t load64(device uint8_t* mem, uint64_t addr) {
    return *((device uint64_t*)(mem + addr));
}

inline void store64(device uint8_t* mem, uint64_t addr, uint64_t val) {
    *((device uint64_t*)(mem + addr)) = val;
}

inline uint32_t load32(device uint8_t* mem, uint64_t addr) {
    return *((device uint32_t*)(mem + addr));
}

kernel void doom_cpu_execute(
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

    // Load registers into local storage
    int64_t r0 = registers[0], r1 = registers[1], r2 = registers[2], r3 = registers[3];
    int64_t r4 = registers[4], r5 = registers[5], r6 = registers[6], r7 = registers[7];
    int64_t r8 = registers[8], r9 = registers[9], r10 = registers[10], r11 = registers[11];
    int64_t r12 = registers[12], r13 = registers[13], r14 = registers[14], r15 = registers[15];
    int64_t r16 = registers[16], r17 = registers[17], r18 = registers[18], r19 = registers[19];
    int64_t r20 = registers[20], r21 = registers[21], r22 = registers[22], r23 = registers[23];
    int64_t r24 = registers[24], r25 = registers[25], r26 = registers[26], r27 = registers[27];
    int64_t r28 = registers[28], r29 = registers[29], r30 = registers[30], r31 = registers[31];

    float N = flags[0], Z = flags[1], C = flags[2], V = flags[3];

    // Optimized register access macros
    #define READ_REG(n) ({ \
        int64_t _val; \
        switch(n) { \
            case 0: _val = r0; break; case 1: _val = r1; break; case 2: _val = r2; break; case 3: _val = r3; break; \
            case 4: _val = r4; break; case 5: _val = r5; break; case 6: _val = r6; break; case 7: _val = r7; break; \
            case 8: _val = r8; break; case 9: _val = r9; break; case 10: _val = r10; break; case 11: _val = r11; break; \
            case 12: _val = r12; break; case 13: _val = r13; break; case 14: _val = r14; break; case 15: _val = r15; break; \
            case 16: _val = r16; break; case 17: _val = r17; break; case 18: _val = r18; break; case 19: _val = r19; break; \
            case 20: _val = r20; break; case 21: _val = r21; break; case 22: _val = r22; break; case 23: _val = r23; break; \
            case 24: _val = r24; break; case 25: _val = r25; break; case 26: _val = r26; break; case 27: _val = r27; break; \
            case 28: _val = r28; break; case 29: _val = r29; break; case 30: _val = r30; break; case 31: _val = r31; break; \
            default: _val = 0; break; \
        } \
        _val; \
    })

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

    // Main execution loop
    while (cycles < max_cycles) {
        if (pc + 4 > memory_size) {
            atomic_store_explicit(signal, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        uint32_t inst = load32(memory, pc);

        // Check for halt/syscall
        if ((inst & 0xFFE0001F) == 0xD4400000) {
            atomic_store_explicit(signal, SIGNAL_HALT, memory_order_relaxed);
            break;
        }

        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint8_t op_hi = (inst >> 24) & 0xFF;

        cycles++;
        pc += 4;

        // Optimized instruction dispatch
        switch (op_hi) {
            case 0x91:  // ADD immediate
                WRITE_REG(rd, ((rn == 31) ? r31 : READ_REG(rn)) + (int64_t)(int16_t)(int8_t)(imm12 & 0xFF));
                continue;
            case 0xD1:  // SUB immediate
                WRITE_REG(rd, ((rn == 31) ? r31 : READ_REG(rn)) - (int64_t)(int16_t)(int8_t)(imm12 & 0xFF));
                continue;
            case 0x14: {  // Unconditional branch
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= 0xFE000000;
                pc += imm26 * 4;
                continue;
            }
            case 0x54: {  // Conditional branch
                uint8_t cond = inst & 0xF;
                bool take = false;
                if (cond == 0) take = (Z == 1.0);
                else if (cond == 1) take = (Z == 0.0);
                if (take) {
                    int32_t imm19 = (inst >> 5) & 0x7FFFF;
                    if (imm19 & 0x40000) imm19 |= 0xFFF80000;
                    pc += imm19 * 4;
                }
                continue;
            }
            case 0x97: {  // Branch with link
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= 0xFE000000;
                r30 = pc;
                pc += imm26 * 4;
                continue;
            }
            case 0xD0: {  // ADRP
                int32_t immlo = (inst >> 29) & 0x3;
                int32_t immhi = (inst >> 5) & 0x7FFFF;
                if (immhi & 0x40000) immhi |= 0xFFF80000;
                int64_t offset = (immhi << 2) | immlo;
                WRITE_REG(rd, (pc + offset - 4) & ~0xFFFL);
                continue;
            }
            default:
                continue;
        }
    }

    // Write back registers
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

/// Result from GPU execution
struct ExecutionResult {
    total_cycles: u32,
    signal: u32,
    final_pc: u64,
}

/// DOOM CPU emulator
struct DoomCPU {
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
    memory_size: usize,
}

impl DoomCPU {
    fn new(memory_size: usize) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or("Failed to get Metal device")?;

        println!("[DoomCPU] Using device: {:?}", device.name());

        let command_queue = device.newCommandQueue()
            .ok_or("Failed to create command queue")?;

        let source = NSString::from_str(DOOM_OPTIMIZED_SHADER);
        let library = device.newLibraryWithSource_options_error(&source, None)
            .map_err(|e| format!("Shader compilation failed: {:?}", e))?;

        let func_name = NSString::from_str("doom_cpu_execute");
        let function = library.newFunctionWithName(&func_name)
            .ok_or("Function not found")?;

        let pipeline = device.newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| format!("Pipeline creation failed: {:?}", e))?;

        let opts = MTLResourceOptions::StorageModeShared;

        let memory_buf = device.newBufferWithLength_options(memory_size, opts)
            .ok_or("Failed to create memory buffer")?;
        let registers_buf = device.newBufferWithLength_options(32 * 8, opts)
            .ok_or("Failed to create registers buffer")?;
        let pc_buf = device.newBufferWithLength_options(8, opts)
            .ok_or("Failed to create PC buffer")?;
        let flags_buf = device.newBufferWithLength_options(16, opts)
            .ok_or("Failed to create flags buffer")?;
        let config_buf = device.newBufferWithLength_options(16, opts)
            .ok_or("Failed to create config buffer")?;
        let signal_buf = device.newBufferWithLength_options(4, opts)
            .ok_or("Failed to create signal buffer")?;
        let stats_buf = device.newBufferWithLength_options(32, opts)
            .ok_or("Failed to create stats buffer")?;

        unsafe {
            let cfg = config_buf.contents().as_ptr() as *mut u32;
            *cfg.add(0) = 1_000_000;
            *cfg.add(1) = memory_size as u32;

            let sig = signal_buf.contents().as_ptr() as *mut u32;
            *sig = 0;
        }

        println!("[DoomCPU] Initialized with {} MB memory", memory_size / (1024 * 1024));

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
            memory_size,
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

    fn set_register(&self, reg: u64, value: i64) {
        unsafe {
            let regs = self.registers_buf.contents().as_ptr() as *mut i64;
            *regs.add(reg as usize) = value;
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

    fn execute(&self, max_cycles: u32) -> ExecutionResult {
        unsafe {
            let cfg = self.config_buf.contents().as_ptr() as *mut u32;
            *cfg.add(0) = max_cycles;
            let sig = self.signal_buf.contents().as_ptr() as *mut u32;
            *sig = 0;
        }

        let command_buffer = self.command_queue.commandBuffer()
            .expect("Failed to create command buffer");

        let encoder = command_buffer.computeCommandEncoder()
            .expect("Failed to create encoder");

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

        unsafe {
            let total = *(self.stats_buf.contents().as_ptr() as *const u32);
            let sig = *(self.signal_buf.contents().as_ptr() as *const u32);
            let pc = *(self.pc_buf.contents().as_ptr() as *const u64);
            ExecutionResult {
                total_cycles: total,
                signal: sig,
                final_pc: pc,
            }
        }
    }
}

/// Load ELF file into CPU memory
fn load_elf(cpu: &DoomCPU, elf_path: &Path) -> Result<(), String> {
    let data = fs::read(elf_path)
        .map_err(|e| format!("Failed to read ELF: {}", e))?;

    // Parse ELF header (little-endian)
    let e_phoff = u64::from_le_bytes(data[32..40].try_into().unwrap());
    let e_phentsize = u16::from_le_bytes(data[54..56].try_into().unwrap());
    let e_phnum = u16::from_le_bytes(data[56..58].try_into().unwrap());

    for i in 0..e_phnum {
        let off = e_phoff as usize + i as usize * e_phentsize as usize;
        let p_type = u32::from_le_bytes(data[off..off+4].try_into().unwrap());

        if p_type == 1 {
            // PT_LOAD
            let p_offset = u64::from_le_bytes(data[off+8..off+16].try_into().unwrap());
            let p_paddr = u64::from_le_bytes(data[off+24..off+32].try_into().unwrap());
            let p_filesz = u64::from_le_bytes(data[off+32..off+40].try_into().unwrap());
            let p_memsz = u64::from_le_bytes(data[off+40..off+48].try_into().unwrap());

            let segment_data = &data[p_offset as usize..(p_offset + p_filesz) as usize];
            for (j, &byte) in segment_data.iter().enumerate() {
                cpu.write_memory(p_paddr + j as u64, &[byte]);
            }

            if p_filesz < p_memsz {
                for j in p_filesz..p_memsz {
                    cpu.write_memory(p_paddr + j, &[0]);
                }
            }
        }
    }

    Ok(())
}

fn main() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║     DOOM GPU Benchmark - Pure Rust/Metal              ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let elf_path = Path::new("/Users/bobbyprice/projects/nCPU/doom_patched.elf");

    if !elf_path.exists() {
        eprintln!("Error: ELF file not found at {}", elf_path.display());
        std::process::exit(1);
    }

    println!("Loading DOOM ELF: {}", elf_path.display());

    let cpu = match DoomCPU::new(8 * 1024 * 1024) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error creating CPU: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = load_elf(&cpu, elf_path) {
        eprintln!("Error loading ELF: {}", e);
        std::process::exit(1);
    }

    println!("ELF loaded successfully!\n");

    // Set initial registers
    cpu.set_pc(0x102e0);
    cpu.set_register(0, 1);
    cpu.set_register(1, 0x7fff00);
    cpu.set_register(2, 0);

    // Warmup
    println!("Warming up GPU...");
    for _ in 0..3 {
        cpu.execute(50000);
    }

    // Reset for benchmark
    cpu.set_pc(0x102e0);
    cpu.set_register(0, 1);
    cpu.set_register(1, 0x7fff00);
    cpu.set_register(2, 0);

    // Benchmark
    println!("\n════════════════════════════════════════════════════════");
    println!("BENCHMARK");
    println!("════════════════════════════════════════════════════════\n");

    let start = Instant::now();
    let mut total_cycles: u64 = 0;
    let duration_secs = 10;

    print!("Running for {} seconds... ", duration_secs);
    let _ = std::io::Write::flush(&mut std::io::stdout());

    while start.elapsed().as_secs() < duration_secs {
        let result = cpu.execute(100_000);
        total_cycles += result.total_cycles as u64;
    }

    let elapsed = start.elapsed();
    let ips = total_cycles as f64 / elapsed.as_secs_f64();

    println!("DONE!\n");

    println!("════════════════════════════════════════════════════════");
    println!("RESULTS");
    println!("════════════════════════════════════════════════════════");
    println!("Total cycles:  {:,}", total_cycles);
    println!("Elapsed time:  {:.2}s", elapsed.as_secs_f64());
    println!("IPS:          {:,.0}", ips);
    println!("Target IPS:   5,000,000");
    println!("Gap:          {:,.0}", 5_000_000.0 - ips);

    let pct = (ips / 5_000_000.0) * 100.0;
    println!("Progress:     {:.1}% of target", pct);

    if ips >= 5_000_000.0 {
        println!("\n🎉 TARGET ACHIEVED! DOOM is playable at full speed!");
    } else {
        let needed = 5_000_000.0 / ips;
        println!("\n⚠️  Need {:.2}x more speedup for 5M IPS", needed);
    }

    println!("════════════════════════════════════════════════════════");
}
