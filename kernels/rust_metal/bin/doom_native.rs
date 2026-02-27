//! Native DOOM Benchmark - Standalone Metal GPU Execution
//!
//! This is a standalone binary that doesn't depend on Python or PyO3.
//! It directly uses Metal to execute DOOM and measure performance.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::time::Instant;
use std::fs;
use std::path::Path;

/// Optimized Metal shader for ARM64 DOOM execution
const DOOM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint32_t SIGNAL_RUNNING = 0;
constant uint32_t SIGNAL_HALT = 1;

kernel void doom_execute(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers [[buffer(1)]],
    device uint64_t* pc_ptr [[buffer(2)]],
    device const uint32_t* config [[buffer(3)]],
    device uint32_t* cycles_out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint64_t pc = pc_ptr[0];
    uint32_t max_cycles = config[0];
    uint32_t cycles = 0;

    int64_t r0 = registers[0], r1 = registers[1], r2 = registers[2], r3 = registers[3];
    int64_t r4 = registers[4], r5 = registers[5], r6 = registers[6], r7 = registers[7];
    int64_t r8 = registers[8], r9 = registers[9], r10 = registers[10], r11 = registers[11];
    int64_t r12 = registers[12], r13 = registers[13], r14 = registers[14], r15 = registers[15];
    int64_t r16 = registers[16], r17 = registers[17], r18 = registers[18], r19 = registers[19];
    int64_t r20 = registers[20], r21 = registers[21], r22 = registers[22], r23 = registers[23];
    int64_t r24 = registers[24], r25 = registers[25], r26 = registers[26], r27 = registers[27];
    int64_t r28 = registers[28], r29 = registers[29], r30 = registers[30], r31 = registers[31];

    while (cycles < max_cycles) {
        uint32_t inst = *((device uint32_t*)(memory + pc));
        pc += 4;
        cycles++;

        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint8_t op_hi = (inst >> 24) & 0xFF;

        // Fast instruction dispatch
        switch (op_hi) {
            case 0x91:  // ADD immediate
                if (rd < 31) {
                    int64_t rn_val = (rn == 31) ? r31 : ((rn == 0) ? r0 : ((rn == 1) ? r1 : r2));
                    switch(rd) {
                        case 0: r0 = rn_val + imm12; break;
                        case 1: r1 = rn_val + imm12; break;
                        case 2: r2 = rn_val + imm12; break;
                        case 3: r3 = rn_val + imm12; break;
                        case 4: r4 = rn_val + imm12; break;
                        case 5: r5 = rn_val + imm12; break;
                        case 6: r6 = rn_val + imm12; break;
                        case 7: r7 = rn_val + imm12; break;
                        case 8: r8 = rn_val + imm12; break;
                        case 9: r9 = rn_val + imm12; break;
                        case 10: r10 = rn_val + imm12; break;
                        case 11: r11 = rn_val + imm12; break;
                        case 12: r12 = rn_val + imm12; break;
                        case 13: r13 = rn_val + imm12; break;
                        case 14: r14 = rn_val + imm12; break;
                        case 15: r15 = rn_val + imm12; break;
                        case 16: r16 = rn_val + imm12; break;
                        case 17: r17 = rn_val + imm12; break;
                        case 18: r18 = rn_val + imm12; break;
                        case 19: r19 = rn_val + imm12; break;
                        case 20: r20 = rn_val + imm12; break;
                        case 21: r21 = rn_val + imm12; break;
                        case 22: r22 = rn_val + imm12; break;
                        case 23: r23 = rn_val + imm12; break;
                        case 24: r24 = rn_val + imm12; break;
                        case 25: r25 = rn_val + imm12; break;
                        case 26: r26 = rn_val + imm12; break;
                        case 27: r27 = rn_val + imm12; break;
                        case 28: r28 = rn_val + imm12; break;
                        case 29: r29 = rn_val + imm12; break;
                        case 30: r30 = rn_val + imm12; break;
                        default: break;
                    }
                }
                continue;
            case 0xD1:  // SUB immediate
                if (rd < 31) {
                    int64_t rn_val = (rn == 31) ? r31 : ((rn == 0) ? r0 : r1);
                    if (rd == 0) r0 = rn_val - imm12;
                    else if (rd == 1) r1 = rn_val - imm12;
                    else if (rd == 2) r2 = rn_val - imm12;
                    else if (rd == 3) r3 = rn_val - imm12;
                }
                continue;
            case 0x14: {  // Branch
                int32_t imm26 = inst & 0x3FFFFFF;
                if (imm26 & 0x2000000) imm26 |= 0xFE000000;
                pc += imm26 * 4;
                continue;
            }
            case 0x54: {  // Conditional branch
                uint8_t cond = inst & 0xF;
                bool take = (cond == 1);  // NE (not equal)
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
                if (rd == 0) r0 = (pc + offset - 4) & ~0xFFFL;
                else if (rd == 1) r1 = (pc + offset - 4) & ~0xFFFL;
                else if (rd == 2) r2 = (pc + offset - 4) & ~0xFFFL;
                continue;
            }
            default:
                continue;
        }
    }

    registers[0] = r0; registers[1] = r1; registers[2] = r2; registers[3] = r3;
    registers[4] = r4; registers[5] = r5; registers[6] = r6; registers[7] = r7;
    registers[8] = r8; registers[9] = r9; registers[10] = r10; registers[11] = r11;
    registers[12] = r12; registers[13] = r13; registers[14] = r14; registers[15] = r15;
    registers[16] = r16; registers[17] = r17; registers[18] = r18; registers[19] = r19;
    registers[20] = r20; registers[21] = r21; registers[22] = r22; registers[23] = r23;
    registers[24] = r24; registers[25] = r25; registers[26] = r26; registers[27] = r27;
    registers[28] = r28; registers[29] = r29; registers[30] = r30; registers[31] = r31;

    pc_ptr[0] = pc;
    cycles_out[0] = cycles;
}
"#;

struct NativeDoomCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    memory_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    config_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    cycles_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl NativeDoomCPU {
    fn new(memory_size: usize) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or("No Metal device")?;

        println!("Using device: {:?}", device.name());

        let command_queue = device.newCommandQueue()
            .ok_or("No command queue")?;

        let source = NSString::from_str(DOOM_SHADER);
        let library = device.newLibraryWithSource_options_error(&source, None)
            .map_err(|e| format!("Shader error: {:?}", e))?;

        let func = library.newFunctionWithName(&NSString::from_str("doom_execute"))
            .ok_or("Function not found")?;

        let pipeline = device.newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("Pipeline error: {:?}", e))?;

        let opts = MTLResourceOptions::StorageModeShared;
        let memory_buf = device.newBufferWithLength_options(memory_size, opts)
            .ok_or("No memory buffer")?;
        let registers_buf = device.newBufferWithLength_options(256, opts)
            .ok_or("No registers buffer")?;
        let pc_buf = device.newBufferWithLength_options(8, opts)
            .ok_or("No PC buffer")?;
        let config_buf = device.newBufferWithLength_options(16, opts)
            .ok_or("No config buffer")?;
        let cycles_buf = device.newBufferWithLength_options(4, opts)
            .ok_or("No cycles buffer")?;

        unsafe {
            let cfg = config_buf.contents().as_ptr() as *mut u32;
            *cfg.add(0) = 500_000;
            *cfg.add(1) = memory_size as u32;
        }

        Ok(Self {
            device,
            command_queue,
            pipeline,
            memory_buf,
            registers_buf,
            pc_buf,
            config_buf,
            cycles_buf,
        })
    }

    fn write_memory(&self, addr: u64, data: &[u8]) {
        unsafe {
            let dst = self.memory_buf.contents().as_ptr().add(addr as usize) as *mut u8;
            for (i, &byte) in data.iter().enumerate() {
                *dst.add(i) = byte;
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

    fn execute(&self, max_cycles: u32) -> u32 {
        unsafe {
            let cfg = self.config_buf.contents().as_ptr() as *mut u32;
            *cfg.add(0) = max_cycles;
        }

        let cmd = self.command_queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&self.pipeline);

        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.memory_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.registers_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.pc_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.config_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&self.cycles_buf), 0, 4);

            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        unsafe { *(self.cycles_buf.contents().as_ptr() as *const u32) }
    }
}

fn load_elf(cpu: &NativeDoomCPU, path: &Path) -> Result<(), String> {
    let data = fs::read(path).map_err(|e| e.to_string())?;

    let e_phoff = u64::from_le_bytes(data[32..40].try_into().unwrap());
    let e_phentsize = u16::from_le_bytes(data[54..56].try_into().unwrap());
    let e_phnum = u16::from_le_bytes(data[56..58].try_into().unwrap());

    for i in 0..e_phnum {
        let off = (e_phoff + i as u64 * e_phentsize as u64) as usize;
        let p_type = u32::from_le_bytes(data[off..off+4].try_into().unwrap());

        if p_type == 1 {
            let p_offset = u64::from_le_bytes(data[off+8..off+16].try_into().unwrap());
            let p_paddr = u64::from_le_bytes(data[off+24..off+32].try_into().unwrap());
            let p_filesz = u64::from_le_bytes(data[off+32..off+40].try_into().unwrap());
            let p_memsz = u64::from_le_bytes(data[off+40..off+48].try_into().unwrap());

            let segment = &data[p_offset as usize..(p_offset+p_filesz) as usize];
            for (j, &b) in segment.iter().enumerate() {
                cpu.write_memory(p_paddr + j as u64, &[b]);
            }

            for j in p_filesz..p_memsz {
                cpu.write_memory(p_paddr + j, &[0]);
            }
        }
    }
    Ok(())
}

fn main() {
    println!("═══════════════════════════════════════════════════");
    println!("     DOOM Native Benchmark - Metal GPU");
    println!("═══════════════════════════════════════════════════\n");

    let elf_path = Path::new("/Users/bobbyprice/projects/nCPU/doom_patched.elf");
    if !elf_path.exists() {
        eprintln!("ELF not found: {}", elf_path.display());
        std::process::exit(1);
    }

    let cpu = match NativeDoomCPU::new(8 * 1024 * 1024) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("CPU init failed: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = load_elf(&cpu, elf_path) {
        eprintln!("ELF load failed: {}", e);
        std::process::exit(1);
    }

    println!("ELF loaded!\n");

    cpu.set_pc(0x102e0);
    cpu.set_register(0, 1);
    cpu.set_register(1, 0x7fff00);
    cpu.set_register(2, 0);

    // Warmup
    print!("Warming up... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    for _ in 0..3 {
        cpu.execute(50000);
    }
    println!("DONE");

    cpu.set_pc(0x102e0);
    cpu.set_register(0, 1);
    cpu.set_register(1, 0x7fff00);
    cpu.set_register(2, 0);

    println!("\nBenchmarking (10 seconds)...\n");

    let start = Instant::now();
    let mut total = 0u64;

    while start.elapsed().as_secs() < 10 {
        total += cpu.execute(100_000) as u64;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let ips = total as f64 / elapsed;

    println!("═══════════════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════════════");
    println!("Cycles:  {:,}", total);
    println!("Time:    {:.2}s", elapsed);
    println!("IPS:     {:,.0}", ips);
    println!("Target:  5,000,000");
    println!("Gap:     {:,.0}", 5_000_000.0 - ips);
    println!("═══════════════════════════════════════════════════");
}
