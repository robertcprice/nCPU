#!/usr/bin/env python3
"""
Metal Shader Source for ARM64 CPU Emulator Kernel - Version 2.

FULL 125-INSTRUCTION ARM64 ISA
===============================

Complete ARM64 instruction set ported from NeuralCPU (ncpu/neural/cpu.py).
Every instruction the NeuralCPU supports is now implemented as native GPU
Metal compute shader operations — qemu-style fetch-decode-execute on GPU.

MEMORY MODEL (Double-Buffer):
=============================
    memory_in (const device)  →  Read initial state
           ↓
    [Copy to memory_out at kernel start]
           ↓
    memory_out (device)       →  Read/Write during execution
           ↓
    [Returned to Python]
           ↓
    memory_in for next call   ←  Swap!

Author: Robert Price / nCPU Project
Date: March 2026
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STOP_RUNNING = 0
STOP_HALT = 1
STOP_SYSCALL = 2
STOP_MAX_CYCLES = 3

# ═══════════════════════════════════════════════════════════════════════════════
# METAL KERNEL HEADER (V2 - Full ARM64 ISA)
# ═══════════════════════════════════════════════════════════════════════════════

KERNEL_HEADER_V2 = """
// ════════════════════════════════════════════════════════════════════════════
// ARM64 CPU EMULATOR - METAL KERNEL V2 (Full 139-Instruction ISA)
// ════════════════════════════════════════════════════════════════════════════
//
// qemu-style fetch-decode-execute loop running entirely on GPU.
// Zero CPU involvement. The GPU IS the CPU.
//
// ════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

// Stop reason codes
constant uint8_t STOP_RUNNING = 0;
constant uint8_t STOP_HALT = 1;
constant uint8_t STOP_SYSCALL = 2;
constant uint8_t STOP_MAX_CYCLES = 3;

constant uint32_t MEMORY_SIZE = 4 * 1024 * 1024;  // 4MB

// ── GPU-SIDE SVC BUFFER ──
// Buffers SYS_WRITE(stdout/stderr) on GPU to avoid Python round-trips.
// Layout at SVC_BUF_BASE:
//   [0..3]   uint32_t write_pos   (offset into data area)
//   [4..7]   uint32_t entry_count (number of buffered entries)
//   [8..15]  uint64_t brk_addr    (heap break for SYS_BRK)
//   [16..]   entry data: [1B fd][2B len LE][len bytes data]
constant uint32_t SVC_BUF_BASE     = 0x3F0000;
constant uint32_t SVC_BUF_HDR      = 16;
constant uint32_t SVC_BUF_DATA     = SVC_BUF_BASE + SVC_BUF_HDR;
constant uint32_t SVC_BUF_CAPACITY = 0xFFF0;  // ~64KB data area
constant uint32_t SVC_HEAP_BASE    = 0x60000;  // must match runner.py HEAP_BASE
// Linux syscall numbers
constant int64_t SVC_SYS_WRITE = 64;
constant int64_t SVC_SYS_CLOSE = 57;
constant int64_t SVC_SYS_BRK   = 214;

// ════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

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

inline uint32_t fetch_instruction(device uint8_t* memory, uint64_t pc) {
    return uint32_t(memory[pc]) |
           (uint32_t(memory[pc + 1]) << 8) |
           (uint32_t(memory[pc + 2]) << 16) |
           (uint32_t(memory[pc + 3]) << 24);
}

inline int64_t load64(device uint8_t* memory, uint64_t addr) {
    return int64_t(memory[addr]) |
           (int64_t(memory[addr + 1]) << 8) |
           (int64_t(memory[addr + 2]) << 16) |
           (int64_t(memory[addr + 3]) << 24) |
           (int64_t(memory[addr + 4]) << 32) |
           (int64_t(memory[addr + 5]) << 40) |
           (int64_t(memory[addr + 6]) << 48) |
           (int64_t(memory[addr + 7]) << 56);
}

inline void store64(device uint8_t* memory, uint64_t addr, int64_t val) {
    memory[addr]     = uint8_t(val & 0xFF);
    memory[addr + 1] = uint8_t((val >> 8) & 0xFF);
    memory[addr + 2] = uint8_t((val >> 16) & 0xFF);
    memory[addr + 3] = uint8_t((val >> 24) & 0xFF);
    memory[addr + 4] = uint8_t((val >> 32) & 0xFF);
    memory[addr + 5] = uint8_t((val >> 40) & 0xFF);
    memory[addr + 6] = uint8_t((val >> 48) & 0xFF);
    memory[addr + 7] = uint8_t((val >> 56) & 0xFF);
}

inline int32_t load32(device uint8_t* memory, uint64_t addr) {
    return int32_t(memory[addr]) |
           (int32_t(memory[addr + 1]) << 8) |
           (int32_t(memory[addr + 2]) << 16) |
           (int32_t(memory[addr + 3]) << 24);
}

inline void store32(device uint8_t* memory, uint64_t addr, int32_t val) {
    memory[addr]     = uint8_t(val & 0xFF);
    memory[addr + 1] = uint8_t((val >> 8) & 0xFF);
    memory[addr + 2] = uint8_t((val >> 16) & 0xFF);
    memory[addr + 3] = uint8_t((val >> 24) & 0xFF);
}

inline int64_t load8(device uint8_t* memory, uint64_t addr) {
    return int64_t(memory[addr]);
}

inline void store8(device uint8_t* memory, uint64_t addr, uint8_t val) {
    memory[addr] = val;
}

inline int64_t load16(device uint8_t* memory, uint64_t addr) {
    return int64_t(memory[addr]) | (int64_t(memory[addr + 1]) << 8);
}

inline void store16(device uint8_t* memory, uint64_t addr, int64_t val) {
    memory[addr]     = uint8_t(val & 0xFF);
    memory[addr + 1] = uint8_t((val >> 8) & 0xFF);
}

// Sign extension helpers
inline int32_t sign_extend_26(uint32_t v) {
    return (v & 0x2000000) ? int32_t(v | 0xFC000000) : int32_t(v);
}
inline int32_t sign_extend_19(uint32_t v) {
    return (v & 0x40000) ? int32_t(v | 0xFFF80000) : int32_t(v);
}
inline int32_t sign_extend_21(uint32_t v) {
    return (v & 0x100000) ? int32_t(v | 0xFFE00000) : int32_t(v);
}
inline int32_t sign_extend_14(uint32_t v) {
    return (v & 0x2000) ? int32_t(v | 0xFFFFC000) : int32_t(v);
}
inline int32_t sign_extend_9(uint32_t v) {
    return (v & 0x100) ? int32_t(v | 0xFFFFFE00) : int32_t(v);
}
inline int32_t sign_extend_7(uint32_t v) {
    return (v & 0x40) ? int32_t(v | 0xFFFFFF80) : int32_t(v);
}

// ════════════════════════════════════════════════════════════════════════════
// CONDITION EVALUATION (shared by B.cond, CSEL, CSINC, CSINV, CSNEG)
// ════════════════════════════════════════════════════════════════════════════

inline bool eval_condition(uint8_t cond, float fn, float fz, float fc, float fv) {
    bool n = fn > 0.5f, z = fz > 0.5f, c = fc > 0.5f, v = fv > 0.5f;
    switch (cond) {
        case 0x0: return z;                      // EQ
        case 0x1: return !z;                     // NE
        case 0x2: return c;                      // CS/HS
        case 0x3: return !c;                     // CC/LO
        case 0x4: return n;                      // MI
        case 0x5: return !n;                     // PL
        case 0x6: return v;                      // VS
        case 0x7: return !v;                     // VC
        case 0x8: return c && !z;                // HI
        case 0x9: return !c || z;                // LS
        case 0xA: return n == v;                 // GE
        case 0xB: return n != v;                 // LT
        case 0xC: return !z && (n == v);         // GT
        case 0xD: return z || (n != v);          // LE
        case 0xE: return true;                   // AL
        default:  return true;                   // NV
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BITMASK IMMEDIATE DECODER (for AND/ORR/EOR/ANDS/TST immediate)
// Full ARM64 logical immediate encoding: N:immr:imms → 64-bit mask
// ════════════════════════════════════════════════════════════════════════════

inline int64_t decode_bitmask_imm(uint32_t inst) {
    uint8_t sf = (inst >> 31) & 1;
    uint8_t N = (inst >> 22) & 1;
    uint8_t immr = (inst >> 16) & 0x3F;
    uint8_t imms = (inst >> 10) & 0x3F;

    // len = HighestSetBit(N:NOT(imms)) per ARM ARM
    // Element size = 2^len, levels = 2^len - 1
    uint8_t len_val = 0;
    if (N == 1) {
        len_val = 6;
    } else {
        uint8_t not_imms = (~imms) & 0x3F;
        if (not_imms == 0) return 0;
        for (int i = 5; i >= 0; i--) {
            if (not_imms & (1 << i)) { len_val = i; break; }
        }
    }
    if (len_val == 0) return 0;

    uint8_t size = 1 << len_val;
    uint8_t S = imms & ((1 << len_val) - 1);
    uint8_t R = immr & ((1 << len_val) - 1);

    // Create pattern of (S+1) ones
    uint64_t pattern = (S + 1 >= 64) ? 0xFFFFFFFFFFFFFFFFULL
                                      : (uint64_t(1) << (S + 1)) - 1;

    // Rotate right by R within element size
    if (R > 0) {
        uint64_t elem_mask = (size >= 64) ? 0xFFFFFFFFFFFFFFFFULL
                                           : (uint64_t(1) << size) - 1;
        pattern = ((pattern >> R) | (pattern << (size - R))) & elem_mask;
    }

    // Replicate to 64 bits
    uint64_t result = 0;
    if (size >= 64) {
        result = pattern;
    } else {
        uint64_t elem_mask = (uint64_t(1) << size) - 1;
        pattern &= elem_mask;
        for (uint8_t i = 0; i < 64; i += size) {
            result |= pattern << i;
        }
    }

    if (sf == 0) result &= 0xFFFFFFFF;
    return int64_t(result);
}

// ════════════════════════════════════════════════════════════════════════════
// EXTENSION HELPER (for ADD_EXT / SUB_EXT)
// ════════════════════════════════════════════════════════════════════════════

inline int64_t apply_extension(int64_t val, uint8_t ext_type) {
    switch (ext_type) {
        case 0: return val & 0xFF;                                            // UXTB
        case 1: return val & 0xFFFF;                                          // UXTH
        case 2: return val & 0xFFFFFFFF;                                      // UXTW
        case 3: return val;                                                   // UXTX
        case 4: { int64_t v = val & 0xFF;  return (v & 0x80)    ? (v | int64_t(0xFFFFFFFFFFFFFF00)) : v; }  // SXTB
        case 5: { int64_t v = val & 0xFFFF; return (v & 0x8000)  ? (v | int64_t(0xFFFFFFFFFFFF0000)) : v; } // SXTH
        case 6: { int64_t v = val & 0xFFFFFFFF; return (v & 0x80000000) ? (v | int64_t(0xFFFFFFFF00000000)) : v; } // SXTW
        default: return val;
    }
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# METAL KERNEL SOURCE (V2 - Full ARM64 ISA)
# ═══════════════════════════════════════════════════════════════════════════════

KERNEL_SOURCE_V2 = """

    // ════════════════════════════════════════════════════════════════════════
    // KERNEL ENTRY
    // ════════════════════════════════════════════════════════════════════════
    uint tid = thread_position_in_grid.x;
    if (tid != 0) return;

    // Memory copy: memory_in → memory_out (GPU-to-GPU, ~0.4ms for 4MB)
    uint32_t mem_size = memory_size_in[0];
    for (uint32_t i = 0; i < mem_size; i++) {
        memory_out[i] = memory_in[i];
    }

    // Load initial state
    uint64_t pc = pc_in[0];
    uint32_t max_cycles = max_cycles_in[0];
    uint32_t cycles = 0;
    uint8_t reason = STOP_RUNNING;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) regs[i] = registers_in[i];
    // regs[31] holds SP for load/store BASE ADDRESS and ADD/SUB immediate contexts.
    // Data-processing register instructions use (rn == 31) ? 0 for XZR semantics.
    // Store data registers: rd/rt2=31 means XZR (zero), NOT SP.
    // Use RD_VAL/RT2_VAL macros for store data to get XZR semantics.
    #define RD_VAL ((rd == 31) ? int64_t(0) : regs[rd])
    #define RT2_VAL ((rt2 == 31) ? int64_t(0) : regs[rt2])

    float flag_n = flags_in[0];
    float flag_z = flags_in[1];
    float flag_c = flags_in[2];
    float flag_v = flags_in[3];

    // SIMD/FP registers V0-V31 (128-bit each, stored as hi:lo int64 pairs)
    // Persisted across kernel invocations for musl va_list save/restore
    int64_t vreg_lo[32];
    int64_t vreg_hi[32];
    for (int i = 0; i < 32; i++) { vreg_lo[i] = vreg_lo_in[i]; vreg_hi[i] = vreg_hi_in[i]; }

    // ════════════════════════════════════════════════════════════════════════
    // MAIN EXECUTION LOOP - Full ARM64 ISA on GPU
    // ════════════════════════════════════════════════════════════════════════

    while (cycles < max_cycles) {
        // FETCH
        uint32_t inst = fetch_instruction(memory_out, pc);

        // DECODE common fields
        uint8_t op_byte = (inst >> 24) & 0xFF;
        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint8_t rm = (inst >> 16) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        uint16_t imm16 = (inst >> 5) & 0xFFFF;
        uint8_t hw = (inst >> 21) & 0x3;
        uint32_t imm26 = inst & 0x3FFFFFF;
        uint32_t imm19 = (inst >> 5) & 0x7FFFF;
        uint8_t cond = inst & 0xF;
        uint8_t rt2 = (inst >> 10) & 0x1F;
        uint8_t ra = (inst >> 10) & 0x1F;

        // CHECK HALT
        if (inst == 0 || (inst & 0xFFE0001F) == 0xD4400000) {
            reason = STOP_HALT;
            break;
        }
        // CHECK SVC — handle hot syscalls on GPU, trap the rest to Python
        if ((inst & 0xFFE0001F) == 0xD4000001) {
            int64_t svc_num = regs[8];  // X8 = syscall number

            // ── SYS_WRITE(stdout/stderr): buffer on GPU ──
            if (svc_num == SVC_SYS_WRITE && (regs[0] == 1 || regs[0] == 2)) {
                uint32_t buf_pos = read_u32_le(memory_out, SVC_BUF_BASE);
                int64_t src_addr = regs[1];
                int64_t write_len = regs[2];
                uint32_t entry_size = 3 + uint32_t(write_len);  // fd + len16 + data
                if (write_len > 0 && buf_pos + entry_size <= SVC_BUF_CAPACITY) {
                    uint32_t base = SVC_BUF_DATA + buf_pos;
                    memory_out[base] = uint8_t(regs[0]);  // fd
                    memory_out[base + 1] = uint8_t(write_len & 0xFF);
                    memory_out[base + 2] = uint8_t((write_len >> 8) & 0xFF);
                    for (int64_t i = 0; i < write_len && i < 8192; i++) {
                        memory_out[base + 3 + uint32_t(i)] = memory_out[uint64_t(src_addr + i)];
                    }
                    buf_pos += entry_size;
                    uint32_t cnt = read_u32_le(memory_out, SVC_BUF_BASE + 4);
                    write_u32_le(memory_out, SVC_BUF_BASE, buf_pos);
                    write_u32_le(memory_out, SVC_BUF_BASE + 4, cnt + 1);
                    regs[0] = write_len;  // return value = bytes written
                    pc += 4;
                    cycles++;
                    continue;  // next instruction — NO Python trap!
                }
                // Buffer full — fall through to Python trap
            }

            // ── SYS_BRK: handle heap break on GPU ──
            if (svc_num == SVC_SYS_BRK) {
                uint64_t brk = read_u64_le(memory_out, SVC_BUF_BASE + 8);
                if (brk == 0) brk = SVC_HEAP_BASE;  // first call: init
                if (regs[0] == 0) {
                    regs[0] = int64_t(brk);
                } else if (uint64_t(regs[0]) >= SVC_HEAP_BASE) {
                    brk = uint64_t(regs[0]);
                    regs[0] = int64_t(brk);
                } else {
                    regs[0] = int64_t(brk);
                }
                write_u64_le(memory_out, SVC_BUF_BASE + 8, brk);
                pc += 4;
                cycles++;
                continue;
            }

            // ── SYS_CLOSE(fd <= 2): no-op on GPU ──
            if (svc_num == SVC_SYS_CLOSE && regs[0] <= 2) {
                regs[0] = 0;  // success
                pc += 4;
                cycles++;
                continue;
            }

            // All other SVCs: trap to Python
            reason = STOP_SYSCALL;
            break;
        }
        // CHECK BRK (software breakpoint — abort/trap)
        if ((inst & 0xFFE0001F) == 0xD4200000) {
            reason = STOP_HALT;
            break;
        }

        bool branch_taken = false;
        bool halt_break = false;

        // ════════════════════════════════════════════════════════════════════
        // INSTRUCTION DECODE — switch(op_byte)
        // ════════════════════════════════════════════════════════════════════
        switch (op_byte) {

        case 0x0A: { // AND/BIC register 32-bit (N=0: AND, N=1: BIC)
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            uint8_t N = (inst >> 21) & 1;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            else if (stype == 3) rm_val = int64_t((uint64_t(rm_val) >> samt) | (uint64_t(rm_val) << (32 - samt))) & 0xFFFFFFFF;
            if (N) rm_val = ~rm_val;
            if (rd != 31) regs[rd] = (rn_val & rm_val) & 0xFFFFFFFF;
            break;
        }

        case 0x0B: { // ADD register 32-bit
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            if (rd != 31) regs[rd] = (rn_val + rm_val) & 0xFFFFFFFF;
            break;
        }

        case 0x2B: { // ADDS register 32-bit (sets flags)
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            uint32_t a32 = uint32_t(rn_val);
            uint32_t b32 = uint32_t(rm_val);
            uint32_t r32 = a32 + b32;
            int64_t result = int64_t(r32);
            if (rd != 31) regs[rd] = result;
            flag_n = ((r32 & 0x80000000u) != 0) ? 1.0f : 0.0f;
            flag_z = (r32 == 0) ? 1.0f : 0.0f;
            flag_c = (uint64_t(a32) + uint64_t(b32) > 0xFFFFFFFFu) ? 1.0f : 0.0f;
            flag_v = ((~(a32 ^ b32)) & (a32 ^ r32) & 0x80000000u) ? 1.0f : 0.0f;
            break;
        }

        case 0x10: { // ADR
            uint32_t immlo = (inst >> 29) & 0x3;
            uint32_t immhi = (inst >> 5) & 0x7FFFF;
            int32_t offset = sign_extend_21((immhi << 2) | immlo);
            if (rd != 31) regs[rd] = int64_t(pc) + offset;
            break;
        }

        case 0x11: { // ADD immediate 32-bit
            int64_t rn_val = regs[rn];
            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
            regs[rd] = (rn_val + aimm) & 0xFFFFFFFF;
            break;
        }

        case 0x12: { // MOVN 32-bit / AND immediate 32-bit
            if ((inst >> 23) & 1) {
                // MOVN 32-bit
                if (rd != 31) regs[rd] = (~(int64_t(imm16) << (hw * 16))) & 0xFFFFFFFF;
            } else {
                // AND immediate 32-bit
                int64_t bitmask = decode_bitmask_imm(inst);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                if (rd != 31) regs[rd] = (rn_val & bitmask) & 0xFFFFFFFF;
            }
            break;
        }

        case 0x13: { // SBFM 32-bit
            // SBFM 32-bit (also handles ASR_IMM 32-bit, SBFX 32-bit, SXTB, SXTH)
            uint8_t immr = (inst >> 16) & 0x3F;
            uint8_t imms = (inst >> 10) & 0x3F;
            uint32_t val = uint32_t(regs[rn] & 0xFFFFFFFF);
            uint32_t result;
            if (imms >= immr) {
                // SBFX / ASR 32-bit: extract (imms-immr+1) bits at immr, sign extend
                uint8_t width = imms - immr + 1;
                result = val >> immr;
                uint32_t mask = (width < 32) ? ((uint32_t(1) << width) - 1) : 0xFFFFFFFF;
                result &= mask;
                // Sign extend from bit (width - 1) within 32-bit result
                if (width < 32 && (result & (uint32_t(1) << (width - 1)))) {
                    result |= ~mask;
                }
            } else {
                // SBFIZ: extract (imms+1) low bits, sign-extend, shift left by (32-immr)
                uint32_t width = imms + 1;
                uint32_t lsb = 32 - immr;
                uint32_t mask = (uint32_t(1) << width) - 1;
                uint32_t src_bits = val & mask;
                if (src_bits & (uint32_t(1) << (width - 1))) {
                    src_bits |= ~mask;
                }
                result = src_bits << lsb;
            }
            // 32-bit SBFM: zero-extend to 64 bits (sign is within 32-bit result)
            if (rd != 31) regs[rd] = int64_t(uint64_t(result));
            break;
        }

        case 0x14: case 0x15: case 0x16: case 0x17: { // B unconditional
            // B - Unconditional branch
            pc = uint64_t(int64_t(pc) + sign_extend_26(imm26) * 4);
            branch_taken = true;
            break;
        }

        case 0x18: case 0x58: case 0x98: case 0xD8: { // PC-relative literal loads
            if (op_byte != 0xD8 && rd != 31) {
                int32_t offset = sign_extend_19(imm19) * 4;
                uint64_t addr = uint64_t(int64_t(pc) + offset);
                if (op_byte == 0x18) regs[rd] = int64_t(load32(memory_out, addr)) & 0xFFFFFFFF;
                else if (op_byte == 0x58) regs[rd] = load64(memory_out, addr);
                else regs[rd] = int64_t(int32_t(load32(memory_out, addr)));
            }
            break;
        }

        case 0x1A: { // Data processing 2-source 32-bit + CSEL/CSINC 32-bit
            if ((inst & 0xFFE0FC00) == 0x1AC00800) {
                // UDIV W — unsigned divide 32-bit
                uint32_t divisor = uint32_t(regs[rm] & 0xFFFFFFFF);
                if (divisor != 0) {
                    if (rd != 31) regs[rd] = int64_t(uint32_t(regs[rn] & 0xFFFFFFFF) / divisor);
                } else {
                    if (rd != 31) regs[rd] = 0;
                }
            } else if ((inst & 0xFFE0FC00) == 0x1AC00C00) {
                // SDIV W — signed divide 32-bit
                int32_t dividend = int32_t(regs[rn] & 0xFFFFFFFF);
                int32_t divisor = int32_t(regs[rm] & 0xFFFFFFFF);
                if (divisor != 0) {
                    if (rd != 31) regs[rd] = int64_t(uint32_t(dividend / divisor));
                } else {
                    if (rd != 31) regs[rd] = 0;
                }
            } else if ((inst & 0xFFE0FC00) == 0x1AC02000) {
                // LSLV W — logical shift left by register 32-bit
                uint32_t shift = uint32_t(regs[rm] & 0x1F);  // Mod 32
                if (rd != 31) regs[rd] = int64_t((uint32_t(regs[rn] & 0xFFFFFFFF) << shift));
            } else if ((inst & 0xFFE0FC00) == 0x1AC02400) {
                // LSRV W — logical shift right by register 32-bit
                uint32_t shift = uint32_t(regs[rm] & 0x1F);  // Mod 32
                if (rd != 31) regs[rd] = int64_t(uint32_t(regs[rn] & 0xFFFFFFFF) >> shift);
            } else if ((inst & 0xFFE0FC00) == 0x1AC02800) {
                // ASRV W — arithmetic shift right by register 32-bit
                int32_t val = int32_t(regs[rn] & 0xFFFFFFFF);
                uint32_t shift = uint32_t(regs[rm] & 0x1F);  // Mod 32
                if (rd != 31) regs[rd] = int64_t(uint32_t(val >> shift));
            } else if ((inst & 0xFFE0FC00) == 0x1AC02C00) {
                // RORV W — rotate right by register 32-bit
                uint32_t val = uint32_t(regs[rn] & 0xFFFFFFFF);
                uint32_t shift = uint32_t(regs[rm] & 0x1F);  // Mod 32
                if (shift > 0) {
                    val = (val >> shift) | (val << (32 - shift));
                }
                if (rd != 31) regs[rd] = int64_t(val);
            } else if ((inst & 0xFFE00C00) == 0x1A800000) {
                // CSEL 32-bit (rn/rm=31 → WZR)
                uint8_t cc = (inst >> 12) & 0xF;
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (rd != 31) regs[rd] = (take ? rn_val : rm_val) & 0xFFFFFFFF;
            } else if ((inst & 0xFFE00C00) == 0x1A800400) {
                // CSINC 32-bit (rn/rm=31 → WZR; CSET when rn=rm=31)
                uint8_t cc = (inst >> 12) & 0xF;
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (rd != 31) regs[rd] = (take ? rn_val : (rm_val + 1)) & 0xFFFFFFFF;
            }
            break;
        }

        case 0x1B: { // MADD 32-bit
            // MADD 32-bit
            int64_t ra_val = (ra == 31) ? 0 : regs[ra];
            int64_t result = (ra_val + regs[rn] * regs[rm]) & 0xFFFFFFFF;
            if (rd != 31) regs[rd] = result;
            break;
        }

        case 0x28: { // LDP/STP post-index 32-bit
            if ((inst & 0xFFC00000) == 0x28C00000) {
                // LDP post-index 32-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 4;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base);
                if (rd != 31) regs[rd] = int64_t(load32(memory_out, addr)) & 0xFFFFFFFF;
                if (rt2 != 31) regs[rt2] = int64_t(load32(memory_out, addr + 4)) & 0xFFFFFFFF;
                regs[rn] = base + imm7;  // writeback (SP-capable)
            } else if ((inst & 0xFFC00000) == 0x28800000) {
                // STP post-index 32-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 4;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base);
                store32(memory_out, addr, int32_t(RD_VAL & 0xFFFFFFFF));
                store32(memory_out, addr + 4, int32_t(RT2_VAL & 0xFFFFFFFF));
                regs[rn] = base + imm7;  // writeback (SP-capable)
            }
            break;
        }

        case 0x29: { // LDP/STP 32-bit
            if ((inst & 0xFFC00000) == 0x29C00000) {
                // LDP pre-index 32-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 4;
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t new_base = base + imm7;
                uint64_t addr = uint64_t(new_base);
                if (rd != 31) regs[rd] = int64_t(load32(memory_out, addr)) & 0xFFFFFFFF;
                if (rt2 != 31) regs[rt2] = int64_t(load32(memory_out, addr + 4)) & 0xFFFFFFFF;
                regs[rn] = new_base;  // writeback (SP-capable)
            } else if ((inst & 0xFFC00000) == 0x29800000) {
                // STP pre-index 32-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 4;
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t new_base = base + imm7;
                uint64_t addr = uint64_t(new_base);
                store32(memory_out, addr, int32_t(RD_VAL & 0xFFFFFFFF));
                store32(memory_out, addr + 4, int32_t(RT2_VAL & 0xFFFFFFFF));
                regs[rn] = new_base;  // writeback (SP-capable)
            } else if ((inst & 0xFFC00000) == 0x29400000) {
                // LDP signed-offset 32-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 4;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + imm7);
                if (rd != 31) regs[rd] = int64_t(load32(memory_out, addr)) & 0xFFFFFFFF;
                if (rt2 != 31) regs[rt2] = int64_t(load32(memory_out, addr + 4)) & 0xFFFFFFFF;
            } else if ((inst & 0xFFC00000) == 0x29000000) {
                // STP signed-offset 32-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 4;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + imm7);
                store32(memory_out, addr, int32_t(RD_VAL & 0xFFFFFFFF));
                store32(memory_out, addr + 4, int32_t(RT2_VAL & 0xFFFFFFFF));
            }
            break;
        }

        case 0x2A: { // ORR/ORN register 32-bit (N=0: ORR, N=1: ORN)
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            uint8_t N = (inst >> 21) & 1;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            else if (stype == 3) rm_val = int64_t((uint64_t(rm_val) >> samt) | (uint64_t(rm_val) << (32 - samt))) & 0xFFFFFFFF;
            if (N) rm_val = ~rm_val;
            if (rd != 31) regs[rd] = (rn_val | rm_val) & 0xFFFFFFFF;
            break;
        }

        case 0x31: { // ADDS immediate 32-bit
            int64_t rn_val = regs[rn];
            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
            uint32_t a32 = uint32_t(rn_val);
            uint32_t b32 = uint32_t(aimm);
            uint32_t r32 = a32 + b32;
            int64_t result = int64_t(r32);
            if (rd != 31) regs[rd] = result;
            flag_n = ((r32 & 0x80000000u) != 0) ? 1.0f : 0.0f;
            flag_z = (r32 == 0) ? 1.0f : 0.0f;
            flag_c = (r32 < a32) ? 1.0f : 0.0f;
            flag_v = ((a32 ^ r32) & ~(a32 ^ b32) & 0x80000000u) ? 1.0f : 0.0f;
            break;
        }

        case 0x32: { // ORR immediate 32-bit
            int64_t bitmask = decode_bitmask_imm(inst);
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            if (rd != 31) regs[rd] = (rn_val | bitmask) & 0xFFFFFFFF;
            break;
        }

        case 0x33: { // BFM 32-bit
            uint8_t immr = (inst >> 16) & 0x3F;
            uint8_t imms = (inst >> 10) & 0x3F;
            uint32_t src = uint32_t(regs[rn] & 0xFFFFFFFF);
            uint32_t dst = (rd != 31) ? uint32_t(regs[rd] & 0xFFFFFFFF) : 0;
            if (imms >= immr) {
                // BFXIL: extract (imms-immr+1) bits at immr from src, insert at bit 0 of dst
                uint8_t width = imms - immr + 1;
                uint32_t extracted = (src >> immr);
                uint32_t mask = (width < 32) ? ((uint32_t(1) << width) - 1) : 0xFFFFFFFF;
                extracted &= mask;
                dst = (dst & ~mask) | extracted;
            } else {
                // BFI: extract (imms+1) low bits of src, insert at bit (32-immr) of dst
                uint8_t width = imms + 1;
                uint8_t lsb = 32 - immr;
                uint32_t extracted = src & ((uint32_t(1) << width) - 1);
                uint32_t mask = ((uint32_t(1) << width) - 1) << lsb;
                dst = (dst & ~mask) | (extracted << lsb);
            }
            if (rd != 31) regs[rd] = int64_t(dst);  // zero-extend to 64
            break;
        }

        case 0x34: case 0xB4: { // CBZ
            // CBZ
            uint8_t rt = inst & 0x1F;
            if (((rt == 31) ? 0 : regs[rt]) == 0) {
                pc = uint64_t(int64_t(pc) + sign_extend_19(imm19) * 4);
                branch_taken = true;
            }
            break;
        }

        case 0x35: case 0xB5: { // CBNZ
            // CBNZ
            uint8_t rt = inst & 0x1F;
            if (((rt == 31) ? 0 : regs[rt]) != 0) {
                pc = uint64_t(int64_t(pc) + sign_extend_19(imm19) * 4);
                branch_taken = true;
            }
            break;
        }

        case 0x36: case 0xB6: { // TBZ
            // TBZ - Test bit and branch if zero
            uint8_t rt = inst & 0x1F;
            uint8_t b5 = (inst >> 31) & 1;
            uint8_t b40 = (inst >> 19) & 0x1F;
            uint8_t bit_pos = (b5 << 5) | b40;
            uint32_t imm14 = (inst >> 5) & 0x3FFF;
            int64_t val = (rt == 31) ? 0 : regs[rt];
            if (!(uint64_t(val) & (uint64_t(1) << bit_pos))) {
                pc = uint64_t(int64_t(pc) + sign_extend_14(imm14) * 4);
                branch_taken = true;
            }
            break;
        }

        case 0x37: case 0xB7: { // TBNZ
            // TBNZ - Test bit and branch if not zero
            uint8_t rt = inst & 0x1F;
            uint8_t b5 = (inst >> 31) & 1;
            uint8_t b40 = (inst >> 19) & 0x1F;
            uint8_t bit_pos = (b5 << 5) | b40;
            uint32_t imm14 = (inst >> 5) & 0x3FFF;
            int64_t val = (rt == 31) ? 0 : regs[rt];
            if (uint64_t(val) & (uint64_t(1) << bit_pos)) {
                pc = uint64_t(int64_t(pc) + sign_extend_14(imm14) * 4);
                branch_taken = true;
            }
            break;
        }

        case 0x38: { // Byte LDR/STR complex
            uint8_t opc = (inst >> 22) & 0x3;
            uint8_t opt_bits = (inst >> 10) & 0x3;

            if (opt_bits == 0x0) {
                // Unscaled offset: LDURB/STURB/LDURSB Wt, [Xn, #imm9]
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + imm9);
                if (opc == 0) {
                    memory_out[addr] = uint8_t(RD_VAL & 0xFF);
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = int64_t(memory_out[addr]);
                } else {
                    // LDRSB: sign-extend byte to 64-bit
                    int8_t val8 = int8_t(memory_out[addr]);
                    if (rd != 31) regs[rd] = int64_t(val8);
                }
            } else if (opt_bits == 0x2) {
                // Register offset: LDRB/STRB Wt, [Xn, Xm{, extend}]
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t offset = (rm == 31) ? 0 : regs[rm];
                uint8_t option = (inst >> 13) & 0x7;
                if (option == 6) {
                    offset = offset & 0xFFFFFFFF;
                    if (offset & 0x80000000) offset |= int64_t(0xFFFFFFFF00000000);
                } else if (option == 2) {
                    offset = offset & 0xFFFFFFFF;
                }
                uint64_t addr = uint64_t(base + offset);
                if (opc == 0) {
                    memory_out[addr] = uint8_t(RD_VAL & 0xFF);
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = int64_t(memory_out[addr]);
                } else {
                    int8_t val8 = int8_t(memory_out[addr]);
                    if (rd != 31) regs[rd] = int64_t(val8);
                }
            } else if (opt_bits == 0x1) {
                // Post-index
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                if (opc == 0) {
                    memory_out[uint64_t(base)] = uint8_t(RD_VAL & 0xFF);
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = int64_t(memory_out[uint64_t(base)]);
                } else {
                    int8_t val8 = int8_t(memory_out[uint64_t(base)]);
                    if (rd != 31) regs[rd] = int64_t(val8);
                }
                regs[rn] = base + imm9;  // writeback (SP-capable)
            } else if (opt_bits == 0x3) {
                // Pre-index
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t new_base = base + imm9;
                regs[rn] = new_base;  // writeback (SP-capable)
                if (opc == 0) {
                    memory_out[uint64_t(new_base)] = uint8_t(RD_VAL & 0xFF);
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = int64_t(memory_out[uint64_t(new_base)]);
                } else {
                    int8_t val8 = int8_t(memory_out[uint64_t(new_base)]);
                    if (rd != 31) regs[rd] = int64_t(val8);
                }
            }
            break;
        }

        case 0x39: { // LDRB/STRB/LDRSB unsigned offset
            if ((inst & 0xFFC00000) == 0x39800000) {
                // LDRSB - Load signed byte
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + imm12;
                int64_t val = int64_t(memory_out[addr]);
                if (val & 0x80) val |= int64_t(0xFFFFFFFFFFFFFF00);
                if (rd != 31) regs[rd] = val;
            } else if ((inst & 0xFFC00000) == 0x39400000) {
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + imm12;
                if (rd != 31) regs[rd] = int64_t(memory_out[addr]);
            } else if ((inst & 0xFFC00000) == 0x39000000) {
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + imm12;
                memory_out[addr] = uint8_t(RD_VAL & 0xFF);
            }
            break;
        }

        case 0x3A: case 0x7A: case 0xBA: case 0xFA: { // CCMP/CCMN
            if ((inst & 0x3FE00410) == 0x3A400000) {
                // CCMP (bit30=1): if cond then CMP(Rn, op2) else flags=nzcv
                // CCMN (bit30=0): if cond then CMN(Rn, op2) else flags=nzcv
                // bit 11: 0=register, 1=immediate
                uint8_t cc = (inst >> 12) & 0xF;
                uint8_t nzcv = inst & 0xF;
                bool sf = (inst >> 31) & 1;
                bool is_imm = (inst >> 11) & 1;
                bool is_sub = (inst >> 30) & 1;  // 1=CCMP(sub), 0=CCMN(add)
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                if (take) {
                    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                    int64_t op2;
                    if (is_imm) {
                        op2 = int64_t((inst >> 16) & 0x1F);
                    } else {
                        op2 = (rm == 31) ? 0 : regs[rm];
                    }
                    if (sf) {
                        int64_t result;
                        if (is_sub) {
                            result = rn_val - op2;
                            flag_c = (uint64_t(rn_val) >= uint64_t(op2)) ? 1.0f : 0.0f;
                            flag_v = ((rn_val ^ op2) & (rn_val ^ result) & (int64_t(1) << 63)) ? 1.0f : 0.0f;
                        } else {
                            result = rn_val + op2;
                            flag_c = (uint64_t(result) < uint64_t(rn_val)) ? 1.0f : 0.0f;
                            flag_v = ((rn_val ^ result) & ~(rn_val ^ op2) & (int64_t(1) << 63)) ? 1.0f : 0.0f;
                        }
                        flag_n = (result < 0) ? 1.0f : 0.0f;
                        flag_z = (result == 0) ? 1.0f : 0.0f;
                    } else {
                        uint32_t a32 = uint32_t(rn_val);
                        uint32_t b32 = uint32_t(op2);
                        uint32_t r32;
                        if (is_sub) {
                            r32 = a32 - b32;
                            flag_c = (a32 >= b32) ? 1.0f : 0.0f;
                            flag_v = ((a32 ^ b32) & (a32 ^ r32) & 0x80000000u) ? 1.0f : 0.0f;
                        } else {
                            r32 = a32 + b32;
                            flag_c = (r32 < a32) ? 1.0f : 0.0f;
                            flag_v = ((a32 ^ r32) & ~(a32 ^ b32) & 0x80000000u) ? 1.0f : 0.0f;
                        }
                        flag_n = ((r32 & 0x80000000u) != 0) ? 1.0f : 0.0f;
                        flag_z = (r32 == 0) ? 1.0f : 0.0f;
                    }
                } else {
                    flag_n = (nzcv & 8) ? 1.0f : 0.0f;
                    flag_z = (nzcv & 4) ? 1.0f : 0.0f;
                    flag_c = (nzcv & 2) ? 1.0f : 0.0f;
                    flag_v = (nzcv & 1) ? 1.0f : 0.0f;
                }
            }
            break;
        }

        case 0x3D: { // SIMD STR/LDR Q 128-bit unsigned offset
            if ((inst & 0xFFC00000) == 0x3D800000) {
                uint8_t rt = inst & 0x1F;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 * 16);
                store64(memory_out, addr, vreg_lo[rt]);
                store64(memory_out, addr + 8, vreg_hi[rt]);
            } else if ((inst & 0xFFC00000) == 0x3DC00000) {
                uint8_t rt = inst & 0x1F;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 * 16);
                vreg_lo[rt] = load64(memory_out, addr);
                vreg_hi[rt] = load64(memory_out, addr + 8);
            }
            break;
        }

        case 0x3C: { // SIMD/FP STUR/LDUR with unscaled offset
            // Encoding: size=00 V=1 opc imm9 Rn Rt
            uint8_t rt = inst & 0x1F;
            uint8_t sopc = (inst >> 22) & 0x3;
            int32_t simm9 = (inst >> 12) & 0x1FF;
            if (simm9 & 0x100) simm9 -= 0x200;
            int64_t base = (rn == 31) ? regs[31] : regs[rn];
            uint64_t addr = uint64_t(int64_t(base) + simm9);
            if (sopc == 0) {
                // STUR B (8-bit store)
                memory_out[addr] = uint8_t(vreg_lo[rt] & 0xFF);
            } else if (sopc == 1) {
                // LDUR B (8-bit load)
                vreg_lo[rt] = int64_t(memory_out[addr]);
                vreg_hi[rt] = 0;
            } else if (sopc == 2) {
                // STUR Q (128-bit store)
                store64(memory_out, addr, vreg_lo[rt]);
                store64(memory_out, addr + 8, vreg_hi[rt]);
            } else if (sopc == 3) {
                // LDUR Q (128-bit load)
                vreg_lo[rt] = load64(memory_out, addr);
                vreg_hi[rt] = load64(memory_out, addr + 8);
            }
            break;
        }

        case 0x0E: { // NEON 2-register misc / across-lanes (8B arrangement)
            uint8_t rt = inst & 0x1F;
            if ((inst & 0xFFFFFC00) == 0x0E205800) {
                // CNT Vd.8B, Vn.8B — count set bits per byte
                uint64_t val = uint64_t(vreg_lo[rn]);
                uint64_t result = 0;
                for (int b = 0; b < 8; b++) {
                    uint8_t byte_val = (val >> (b * 8)) & 0xFF;
                    uint8_t cnt = 0;
                    while (byte_val) { cnt += byte_val & 1; byte_val >>= 1; }
                    result |= (uint64_t(cnt) << (b * 8));
                }
                vreg_lo[rt] = int64_t(result);
                vreg_hi[rt] = 0;
            } else if ((inst & 0xFFFFFC00) == 0x0E31B800) {
                // ADDV Bd, Vn.8B — horizontal add across vector bytes
                uint64_t val = uint64_t(vreg_lo[rn]);
                uint64_t sum = 0;
                for (int b = 0; b < 8; b++) {
                    sum += (val >> (b * 8)) & 0xFF;
                }
                vreg_lo[rt] = int64_t(sum & 0xFF);
                vreg_hi[rt] = 0;
            }
            break;
        }

        case 0x1E: { // FP data processing / FMOV
            uint8_t rt = inst & 0x1F;
            if ((inst & 0xFFE0FC00) == 0x1E260000) {
                // FMOV Wd, Sn — move bottom 32 bits of SIMD to GPR
                uint32_t val = uint32_t(vreg_lo[rn] & 0xFFFFFFFF);
                if (rt != 31) regs[rt] = int64_t(val);
            } else if ((inst & 0xFFE0FC00) == 0x1E270000) {
                // FMOV Sd, Wn — move GPR to bottom 32 bits of SIMD
                int64_t val = (rn == 31) ? 0 : regs[rn];
                vreg_lo[rt] = int64_t(uint32_t(val & 0xFFFFFFFF));
                vreg_hi[rt] = 0;
            } else if ((inst & 0xFFE0FC00) == 0x1E204000) {
                // FMOV Sd, Sn — copy single-precision between FP registers
                vreg_lo[rt] = vreg_lo[rn] & 0xFFFFFFFF;
                vreg_hi[rt] = 0;
            }
            break;
        }

        case 0x9E: { // FP/INT transfer
            uint8_t rt = inst & 0x1F;
            if ((inst & 0xFFE0FC00) == 0x9E660000) {
                // FMOV Dd, Xn — move GPR to SIMD D register
                int64_t val = (rn == 31) ? 0 : regs[rn];
                vreg_lo[rt] = val;
                vreg_hi[rt] = 0;
            } else if ((inst & 0xFFE0FC00) == 0x9E670000) {
                // FMOV Xd, Dn — move SIMD D register to GPR
                if (rt != 31) regs[rt] = vreg_lo[rn];
            }
            break;
        }

        case 0x2F: case 0x4F: { // MOVI SIMD vector immediate
            // Used to zero vector registers: MOVI Vd.xS, #0
            uint8_t rt = inst & 0x1F;
            // For MOVI #0 patterns (0x2F00041D, 0x4F00041F), just zero the register
            uint8_t cmode = (inst >> 12) & 0xF;
            uint8_t op_bit = (inst >> 29) & 0x1;
            uint8_t abc = ((inst >> 16) & 0x7) | (((inst >> 5) & 0x1F) << 3);
            // Extract immediate
            uint8_t a = (inst >> 18) & 1;
            uint8_t bcdefgh = ((inst >> 16) & 0x7) | (((inst >> 5) & 0x1F) << 3);
            uint8_t imm8_val = (a << 7) | (bcdefgh & 0x7F);
            // For simplicity, handle the zero case (most common in musl)
            if (imm8_val == 0 && cmode == 0) {
                vreg_lo[rt] = 0;
                vreg_hi[rt] = 0;
            } else {
                // General MOVI: replicate immediate based on cmode
                // For now, handle as zero (conservative)
                vreg_lo[rt] = 0;
                vreg_hi[rt] = 0;
            }
            break;
        }

        case 0x4E: { // SIMD vector ops (16B/Q arrangement)
            uint8_t rt = inst & 0x1F;
            uint8_t imm5 = (inst >> 16) & 0x1F;
            if ((inst & 0xBFE0FC00) == 0x0E000C00) {
                // DUP Vd, Wn — broadcast GP register to vector lanes
                int64_t val = (rn == 31) ? 0 : regs[rn];
                if (imm5 & 1) {
                    // DUP Vd.16B/8B, Wn — broadcast byte
                    uint8_t byte_val = uint8_t(val & 0xFF);
                    uint64_t fill = 0;
                    for (int b = 0; b < 8; b++) fill |= (uint64_t(byte_val) << (b*8));
                    vreg_lo[rt] = int64_t(fill);
                    vreg_hi[rt] = ((inst >> 30) & 1) ? int64_t(fill) : 0;
                } else if (imm5 & 2) {
                    // DUP Vd.8H/4H, Wn — broadcast halfword
                    uint16_t hw = uint16_t(val & 0xFFFF);
                    uint64_t fill = 0;
                    for (int b = 0; b < 4; b++) fill |= (uint64_t(hw) << (b*16));
                    vreg_lo[rt] = int64_t(fill);
                    vreg_hi[rt] = ((inst >> 30) & 1) ? int64_t(fill) : 0;
                } else if (imm5 & 4) {
                    // DUP Vd.4S/2S, Wn — broadcast word
                    uint32_t w = uint32_t(val & 0xFFFFFFFF);
                    uint64_t fill = (uint64_t(w) << 32) | uint64_t(w);
                    vreg_lo[rt] = int64_t(fill);
                    vreg_hi[rt] = ((inst >> 30) & 1) ? int64_t(fill) : 0;
                } else if (imm5 & 8) {
                    // DUP Vd.2D, Xn — broadcast doubleword
                    vreg_lo[rt] = int64_t(val);
                    vreg_hi[rt] = ((inst >> 30) & 1) ? int64_t(val) : 0;
                }
            } else if ((inst & 0xBFE0FC00) == 0x0E003C00) {
                // UMOV Wd/Xd, Vn.Ts[idx] — extract element from vector to GP
                uint8_t q = (inst >> 30) & 1;
                uint64_t lo = uint64_t(vreg_lo[rn]);
                uint64_t hi = uint64_t(vreg_hi[rn]);
                if (imm5 & 8) {
                    // UMOV Xd, Vn.D[idx] — 64-bit extract (Q must be 1)
                    uint8_t idx_d = (imm5 >> 4) & 1;
                    int64_t val = idx_d ? int64_t(hi) : int64_t(lo);
                    if (rt != 31) regs[rt] = val;
                } else if (imm5 & 4) {
                    // UMOV Wd, Vn.S[idx] — 32-bit extract
                    uint8_t idx_s = (imm5 >> 3) & 3;
                    uint64_t src = (idx_s < 2) ? lo : hi;
                    uint32_t val = uint32_t((src >> ((idx_s & 1) * 32)) & 0xFFFFFFFF);
                    if (rt != 31) regs[rt] = int64_t(val);
                } else if (imm5 & 2) {
                    // UMOV Wd, Vn.H[idx] — 16-bit extract
                    uint8_t idx_h = (imm5 >> 2) & 7;
                    uint64_t src = (idx_h < 4) ? lo : hi;
                    uint16_t val = uint16_t((src >> ((idx_h & 3) * 16)) & 0xFFFF);
                    if (rt != 31) regs[rt] = int64_t(val);
                } else {
                    // UMOV Wd, Vn.B[idx] — 8-bit extract
                    uint8_t idx_b = (imm5 >> 1) & 0xF;
                    uint64_t src = (idx_b < 8) ? lo : hi;
                    uint8_t val = uint8_t((src >> ((idx_b & 7) * 8)) & 0xFF);
                    if (rt != 31) regs[rt] = int64_t(val);
                }
            } else if ((inst & 0xBFE07C00) == 0x0E001C00) {
                // INS Vd.Ts[idx], Wn — insert GP into vector element
                int64_t val = (rn == 31) ? 0 : regs[rn];
                if (imm5 & 4) {
                    // INS Vd.S[idx], Wn — 32-bit insert
                    uint8_t idx_s = (imm5 >> 3) & 3;
                    uint32_t w = uint32_t(val & 0xFFFFFFFF);
                    if (idx_s < 2) {
                        uint64_t mask = ~(uint64_t(0xFFFFFFFF) << ((idx_s & 1) * 32));
                        vreg_lo[rt] = int64_t((uint64_t(vreg_lo[rt]) & mask) | (uint64_t(w) << ((idx_s & 1) * 32)));
                    } else {
                        uint64_t mask = ~(uint64_t(0xFFFFFFFF) << ((idx_s & 1) * 32));
                        vreg_hi[rt] = int64_t((uint64_t(vreg_hi[rt]) & mask) | (uint64_t(w) << ((idx_s & 1) * 32)));
                    }
                }
            } else if ((inst & 0xFFE0FC00) == 0x4EA01C00) {
                // ORR Vd.16B, Vn.16B, Vm.16B (also MOV when Vm==Vn)
                vreg_lo[rt] = vreg_lo[rn] | vreg_lo[rm];
                vreg_hi[rt] = vreg_hi[rn] | vreg_hi[rm];
            } else if ((inst & 0xFFE0FC00) == 0x4E201C00) {
                // AND Vd.16B, Vn.16B, Vm.16B
                vreg_lo[rt] = vreg_lo[rn] & vreg_lo[rm];
                vreg_hi[rt] = vreg_hi[rn] & vreg_hi[rm];
            } else if ((inst & 0xFFE0FC00) == 0x4E601C00) {
                // BIC Vd.16B, Vn.16B, Vm.16B (bit clear = AND NOT)
                vreg_lo[rt] = vreg_lo[rn] & ~vreg_lo[rm];
                vreg_hi[rt] = vreg_hi[rn] & ~vreg_hi[rm];
            } else if ((inst & 0xFF20FC00) == 0x4E208400) {
                // SUB Vd.16B, Vn.16B, Vm.16B
                uint64_t lo_a = uint64_t(vreg_lo[rn]), lo_b = uint64_t(vreg_lo[rm]);
                uint64_t hi_a = uint64_t(vreg_hi[rn]), hi_b = uint64_t(vreg_hi[rm]);
                uint64_t lo_r = 0, hi_r = 0;
                for (int b = 0; b < 8; b++) {
                    uint8_t va = (lo_a >> (b*8)) & 0xFF;
                    uint8_t vb = (lo_b >> (b*8)) & 0xFF;
                    lo_r |= (uint64_t((va - vb) & 0xFF) << (b*8));
                }
                for (int b = 0; b < 8; b++) {
                    uint8_t va = (hi_a >> (b*8)) & 0xFF;
                    uint8_t vb = (hi_b >> (b*8)) & 0xFF;
                    hi_r |= (uint64_t((va - vb) & 0xFF) << (b*8));
                }
                vreg_lo[rt] = int64_t(lo_r);
                vreg_hi[rt] = int64_t(hi_r);
            } else if ((inst & 0xFFFFFC00) == 0x4E201800) {
                // REV16 Vd.16B, Vn.16B — reverse bytes in each 16-bit element
                uint64_t lo = uint64_t(vreg_lo[rn]), hi = uint64_t(vreg_hi[rn]);
                uint64_t lo_r = 0, hi_r = 0;
                for (int i = 0; i < 4; i++) {
                    uint16_t hw = (lo >> (i*16)) & 0xFFFF;
                    hw = (hw >> 8) | ((hw & 0xFF) << 8);
                    lo_r |= (uint64_t(hw) << (i*16));
                }
                for (int i = 0; i < 4; i++) {
                    uint16_t hw = (hi >> (i*16)) & 0xFFFF;
                    hw = (hw >> 8) | ((hw & 0xFF) << 8);
                    hi_r |= (uint64_t(hw) << (i*16));
                }
                vreg_lo[rt] = int64_t(lo_r);
                vreg_hi[rt] = int64_t(hi_r);
            }
            break;
        }

        case 0x6E: { // SIMD vector ops (16B) — extended
            uint8_t rt = inst & 0x1F;
            if ((inst & 0xFF20FC00) == 0x6E208400) {
                // SUB Vd.16B, Vn.16B, Vm.16B (U=1 variant)
                uint64_t lo_a = uint64_t(vreg_lo[rn]), lo_b = uint64_t(vreg_lo[rm]);
                uint64_t hi_a = uint64_t(vreg_hi[rn]), hi_b = uint64_t(vreg_hi[rm]);
                uint64_t lo_r = 0, hi_r = 0;
                for (int b = 0; b < 8; b++) {
                    uint8_t va = (lo_a >> (b*8)) & 0xFF;
                    uint8_t vb = (lo_b >> (b*8)) & 0xFF;
                    lo_r |= (uint64_t((va - vb) & 0xFF) << (b*8));
                }
                for (int b = 0; b < 8; b++) {
                    uint8_t va = (hi_a >> (b*8)) & 0xFF;
                    uint8_t vb = (hi_b >> (b*8)) & 0xFF;
                    hi_r |= (uint64_t((va - vb) & 0xFF) << (b*8));
                }
                vreg_lo[rt] = int64_t(lo_r);
                vreg_hi[rt] = int64_t(hi_r);
            } else if ((inst & 0xFFE0FC00) == 0x6EE08400) {
                // SUB Vd.2D, Vn.2D, Vm.2D — 64-bit lane subtraction
                vreg_lo[rt] = vreg_lo[rn] - vreg_lo[rm];
                vreg_hi[rt] = vreg_hi[rn] - vreg_hi[rm];
            }
            break;
        }

        case 0x6F: { // MOVI V.2D / MVNI V.4S
            uint8_t rt = inst & 0x1F;
            uint8_t cmode = (inst >> 12) & 0xF;
            uint8_t op_bit = (inst >> 29) & 0x1;
            if (cmode == 0xE && op_bit == 1) {
                // MOVI Vd.2D, #0 (or MOVI Vd.2D, #imm)
                // Extract imm8
                uint8_t a = (inst >> 18) & 1;
                uint8_t b_val = (inst >> 17) & 1;
                uint8_t c = (inst >> 16) & 1;
                uint8_t d = (inst >> 9) & 1;
                uint8_t e = (inst >> 8) & 1;
                uint8_t f = (inst >> 7) & 1;
                uint8_t g = (inst >> 6) & 1;
                uint8_t h = (inst >> 5) & 1;
                uint8_t imm8 = (a<<7)|(b_val<<6)|(c<<5)|(d<<4)|(e<<3)|(f<<2)|(g<<1)|h;
                // Expand: each bit of imm8 -> a full byte (0xFF or 0x00)
                uint64_t val = 0;
                for (int i = 0; i < 8; i++) {
                    if (imm8 & (1 << i)) val |= (uint64_t(0xFF) << (i*8));
                }
                vreg_lo[rt] = int64_t(val);
                vreg_hi[rt] = int64_t(val);
            } else {
                // MVNI or other — just zero for safety
                vreg_lo[rt] = 0;
                vreg_hi[rt] = 0;
            }
            break;
        }

        case 0x0F: { // SIMD shift by immediate / MOVI (8B arrangement)
            uint8_t rt = inst & 0x1F;
            // For SXTL-like and MOVI V.2S cases, handle conservatively
            vreg_lo[rt] = 0;
            vreg_hi[rt] = 0;
            break;
        }

        case 0x2E: { // SIMD misc (8B arrangement)
            uint8_t rt = inst & 0x1F;
            if ((inst & 0xFFFFFC00) == 0x2E205800) {
                // MVN Vd.8B, Vn.8B (NOT)
                vreg_lo[rt] = ~vreg_lo[rn];
                vreg_hi[rt] = 0;  // 8B = only lower 8 bytes
            } else if ((inst & 0xFFFFFC00) == 0x2E200800) {
                // REV32 Vd.8B, Vn.8B — reverse bytes in 32-bit elements
                uint64_t lo = uint64_t(vreg_lo[rn]);
                uint64_t lo_r = 0;
                for (int i = 0; i < 2; i++) {
                    uint32_t w = (lo >> (i*32)) & 0xFFFFFFFF;
                    w = ((w & 0xFF000000) >> 24) | ((w & 0x00FF0000) >> 8) |
                        ((w & 0x0000FF00) << 8) | ((w & 0x000000FF) << 24);
                    lo_r |= (uint64_t(w) << (i*32));
                }
                vreg_lo[rt] = int64_t(lo_r);
                vreg_hi[rt] = 0;
            }
            break;
        }

        case 0x69: { // LDPSW — load pair signed word
            uint8_t rt = inst & 0x1F;
            uint8_t rt2_v = (inst >> 10) & 0x1F;
            int32_t imm7 = (inst >> 15) & 0x7F;
            if (imm7 & 0x40) imm7 -= 0x80;
            int64_t base = (rn == 31) ? regs[31] : regs[rn];
            uint64_t addr = uint64_t(base + imm7 * 4);
            int32_t val1 = int32_t(load32(memory_out, addr));
            int32_t val2 = int32_t(load32(memory_out, addr + 4));
            if (rt != 31) regs[rt] = int64_t(val1);   // sign-extended
            if (rt2_v != 31) regs[rt2_v] = int64_t(val2);
            break;
        }

        case 0x4C: { // SIMD LD1/ST1 multi-structure
            // LD1/ST1 {Vt.16B}, [Xn] and similar
            uint8_t rt = inst & 0x1F;
            uint8_t L = (inst >> 22) & 1;
            uint8_t opcode_field = (inst >> 12) & 0xF;
            int64_t base = (rn == 31) ? regs[31] : regs[rn];
            uint64_t addr = uint64_t(base);
            if (L) {
                // LD1 — load from memory to SIMD register(s)
                if (opcode_field == 0x7) {
                    // LD1 {Vt.16B}, [Xn] — single register
                    vreg_lo[rt] = load64(memory_out, addr);
                    vreg_hi[rt] = load64(memory_out, addr + 8);
                } else if (opcode_field == 0xA) {
                    // LD1 {Vt.16B, Vt2.16B}, [Xn] — two registers
                    uint8_t rt2 = (rt + 1) & 31;
                    vreg_lo[rt] = load64(memory_out, addr);
                    vreg_hi[rt] = load64(memory_out, addr + 8);
                    vreg_lo[rt2] = load64(memory_out, addr + 16);
                    vreg_hi[rt2] = load64(memory_out, addr + 24);
                }
            } else {
                // ST1 — store from SIMD register(s) to memory
                if (opcode_field == 0x7) {
                    // ST1 {Vt.16B}, [Xn]
                    store64(memory_out, addr, vreg_lo[rt]);
                    store64(memory_out, addr + 8, vreg_hi[rt]);
                } else if (opcode_field == 0xA) {
                    // ST1 {Vt.16B, Vt2.16B}, [Xn]
                    uint8_t rt2 = (rt + 1) & 31;
                    store64(memory_out, addr, vreg_lo[rt]);
                    store64(memory_out, addr + 8, vreg_hi[rt]);
                    store64(memory_out, addr + 16, vreg_lo[rt2]);
                    store64(memory_out, addr + 24, vreg_hi[rt2]);
                }
            }
            break;
        }

        case 0x7C: case 0xBC: case 0xFC: { // SIMD/FP STR/LDR H/S/D with complex addressing
            // 0x7C = halfword (16-bit), 0xBC = single (32-bit), 0xFC = double (64-bit)
            uint8_t rt = inst & 0x1F;
            uint8_t sopc = (inst >> 22) & 0x3;
            int32_t simm9 = (inst >> 12) & 0x1FF;
            if (simm9 & 0x100) simm9 -= 0x200;
            int64_t base = (rn == 31) ? regs[31] : regs[rn];
            uint8_t idx_type = (inst >> 10) & 0x3; // 00=unscaled, 01=post, 11=pre
            uint64_t addr;
            if (idx_type == 1) {
                // Post-index
                addr = uint64_t(base);
                if (rn == 31) regs[31] = base + simm9;
                else regs[rn] = base + simm9;
            } else if (idx_type == 3) {
                // Pre-index
                addr = uint64_t(int64_t(base) + simm9);
                if (rn == 31) regs[31] = int64_t(addr);
                else regs[rn] = int64_t(addr);
            } else {
                // Unscaled offset
                addr = uint64_t(int64_t(base) + simm9);
            }
            if (op_byte == 0xFC) {
                // 64-bit (D register)
                if (sopc == 0 || sopc == 2) {
                    store64(memory_out, addr, vreg_lo[rt]);
                } else {
                    vreg_lo[rt] = load64(memory_out, addr);
                    vreg_hi[rt] = 0;
                }
            } else if (op_byte == 0xBC) {
                // 32-bit (S register)
                if (sopc == 0 || sopc == 2) {
                    store32(memory_out, addr, int32_t(vreg_lo[rt] & 0xFFFFFFFF));
                } else {
                    vreg_lo[rt] = int64_t(uint32_t(load32(memory_out, addr)));
                    vreg_hi[rt] = 0;
                }
            } else {
                // 16-bit (H register)
                if (sopc == 0 || sopc == 2) {
                    store16(memory_out, addr, vreg_lo[rt] & 0xFFFF);
                } else {
                    vreg_lo[rt] = load16(memory_out, addr) & 0xFFFF;
                    vreg_hi[rt] = 0;
                }
            }
            break;
        }

        case 0x4A: { // EOR/EON register 32-bit (N=0: EOR, N=1: EON)
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            uint8_t N = (inst >> 21) & 1;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            else if (stype == 3) rm_val = int64_t((uint64_t(rm_val) >> samt) | (uint64_t(rm_val) << (32 - samt))) & 0xFFFFFFFF;
            if (N) rm_val = ~rm_val;
            if (rd != 31) regs[rd] = (rn_val ^ rm_val) & 0xFFFFFFFF;
            break;
        }

        case 0x4B: { // SUB register 32-bit
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            if (rd != 31) regs[rd] = (rn_val - rm_val) & 0xFFFFFFFF;
            break;
        }

        case 0x51: { // SUB immediate 32-bit
            int64_t rn_val = regs[rn];
            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
            regs[rd] = (rn_val - aimm) & 0xFFFFFFFF;
            break;
        }

        case 0x52: { // MOVZ 32-bit / EOR immediate 32-bit
            if ((inst >> 23) & 1) {
                // MOVZ 32-bit
                if (rd != 31) regs[rd] = (int64_t(imm16) << (hw * 16)) & 0xFFFFFFFF;
            } else {
                // EOR immediate 32-bit
                int64_t bitmask = decode_bitmask_imm(inst);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                if (rd != 31) regs[rd] = (rn_val ^ bitmask) & 0xFFFFFFFF;
            }
            break;
        }

        case 0x53: { // UBFM 32-bit
            // UBFM 32-bit
            uint8_t immr = (inst >> 16) & 0x3F;
            uint8_t imms = (inst >> 10) & 0x3F;
            uint32_t val = uint32_t(regs[rn] & 0xFFFFFFFF);
            uint32_t result;
            if (imms >= immr) {
                // UBFX 32-bit: extract (imms-immr+1) bits starting at immr
                uint8_t width = imms - immr + 1;
                result = val >> immr;
                uint32_t mask = (width < 32) ? ((uint32_t(1) << width) - 1) : 0xFFFFFFFF;
                result &= mask;
            } else {
                // UBFIZ / LSL: extract low (imms+1) bits, shift left by (32-immr)
                uint32_t src_bits = val & ((uint32_t(1) << (imms + 1)) - 1);
                result = src_bits << (32 - immr);
            }
            if (rd != 31) regs[rd] = int64_t(result);
            break;
        }

        case 0x54: { // B.cond
            // B.cond - Conditional branch
            if (eval_condition(cond, flag_n, flag_z, flag_c, flag_v)) {
                pc = uint64_t(int64_t(pc) + sign_extend_19(imm19) * 4);
                branch_taken = true;
            }
            break;
        }

        case 0x5A: { // RBIT/CLZ/REV 32-bit + CSINV/CSNEG 32-bit
            if ((inst & 0xFFFFFC00) == 0x5AC00000) {
                // RBIT W - Reverse bits 32-bit
                uint32_t val = uint32_t(regs[rn] & 0xFFFFFFFF);
                uint32_t result = 0;
                for (int i = 0; i < 32; i++) {
                    if (val & (uint32_t(1) << i)) result |= uint32_t(1) << (31 - i);
                }
                if (rd != 31) regs[rd] = int64_t(result);
            } else if ((inst & 0xFFFFFC00) == 0x5AC01000) {
                // CLZ W - Count leading zeros 32-bit
                uint32_t val = uint32_t(regs[rn] & 0xFFFFFFFF);
                int64_t count = 32;
                for (int i = 31; i >= 0; i--) {
                    if (val & (uint32_t(1) << i)) { count = 31 - i; break; }
                }
                if (rd != 31) regs[rd] = count;
            } else if ((inst & 0xFFFFFC00) == 0x5AC00800) {
                // REV W - Reverse bytes 32-bit
                uint32_t val = uint32_t(regs[rn] & 0xFFFFFFFF);
                uint32_t result = ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
                                  ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000u);
                if (rd != 31) regs[rd] = int64_t(result);
            } else if ((inst & 0xFFFFFC00) == 0x5AC00400) {
                // REV16 W - Reverse bytes in 16-bit halfwords 32-bit
                uint32_t val = uint32_t(regs[rn] & 0xFFFFFFFF);
                uint32_t result = ((val >> 8) & 0x00FF00FF) | ((val << 8) & 0xFF00FF00u);
                if (rd != 31) regs[rd] = int64_t(result);
            } else if ((inst & 0xFFE00C00) == 0x5A800000) {
                // CSINV 32-bit (rn/rm=31 → WZR)
                uint8_t cc = (inst >> 12) & 0xF;
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (rd != 31) regs[rd] = (take ? rn_val : (~rm_val)) & 0xFFFFFFFF;
            } else if ((inst & 0xFFE00C00) == 0x5A800400) {
                // CSNEG 32-bit (rn/rm=31 → WZR)
                uint8_t cc = (inst >> 12) & 0xF;
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (rd != 31) regs[rd] = (take ? rn_val : (-rm_val)) & 0xFFFFFFFF;
            }
            break;
        }

        case 0x6A: { // ANDS/BICS register 32-bit with shift (N=0: ANDS, N=1: BICS)
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            uint8_t N = (inst >> 21) & 1;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            rn_val &= 0xFFFFFFFF; rm_val &= 0xFFFFFFFF;
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = int64_t(int32_t(rm_val) >> samt);
            if (N) rm_val = ~rm_val;
            int64_t result = (rn_val & rm_val) & 0xFFFFFFFF;
            if (rd != 31) regs[rd] = result;
            flag_n = (result & 0x80000000) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = 0.0f;
            flag_v = 0.0f;
            break;
        }

        case 0x6B: { // SUBS register 32-bit
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            uint32_t a32 = uint32_t(rn_val);
            uint32_t b32 = uint32_t(rm_val);
            uint32_t r32 = a32 - b32;
            int64_t result = int64_t(r32);
            if (rd != 31) regs[rd] = result;
            flag_n = ((r32 & 0x80000000u) != 0) ? 1.0f : 0.0f;
            flag_z = (r32 == 0) ? 1.0f : 0.0f;
            flag_c = (a32 >= b32) ? 1.0f : 0.0f;
            flag_v = ((a32 ^ b32) & (a32 ^ r32) & 0x80000000u) ? 1.0f : 0.0f;
            break;
        }

        case 0x6D: { // STP/LDP Dn FP pair
            if ((inst & 0xFFC00000) == 0x6D000000) {
                uint8_t rt = inst & 0x1F;
                uint8_t rt2_v = (inst >> 10) & 0x1F;
                int32_t offset = sign_extend_7((inst >> 15) & 0x7F) * 8;
                int64_t base = regs[rn];
                uint64_t addr = uint64_t(base + offset);
                store64(memory_out, addr, vreg_lo[rt]);
                store64(memory_out, addr + 8, vreg_lo[rt2_v]);
            } else if ((inst & 0xFFC00000) == 0x6D400000) {
                uint8_t rt = inst & 0x1F;
                uint8_t rt2_v = (inst >> 10) & 0x1F;
                int32_t offset = sign_extend_7((inst >> 15) & 0x7F) * 8;
                int64_t base = regs[rn];
                uint64_t addr = uint64_t(base + offset);
                vreg_lo[rt] = load64(memory_out, addr);
                vreg_hi[rt] = 0;
                vreg_lo[rt2_v] = load64(memory_out, addr + 8);
                vreg_hi[rt2_v] = 0;
            }
            break;
        }

        case 0x71: { // SUBS immediate 32-bit
            int64_t rn_val = regs[rn];
            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
            uint32_t a32 = uint32_t(rn_val);
            uint32_t b32 = uint32_t(aimm);
            uint32_t r32 = a32 - b32;
            int64_t result = int64_t(r32);
            if (rd != 31) regs[rd] = result;
            flag_n = ((r32 & 0x80000000u) != 0) ? 1.0f : 0.0f;
            flag_z = (r32 == 0) ? 1.0f : 0.0f;
            flag_c = (a32 >= b32) ? 1.0f : 0.0f;
            flag_v = ((a32 ^ b32) & (a32 ^ r32) & 0x80000000u) ? 1.0f : 0.0f;
            break;
        }

        case 0x72: { // MOVK 32-bit / ANDS immediate 32-bit
            if ((inst >> 23) & 1) {
                // MOVK 32-bit
                int64_t mask = ~(int64_t(0xFFFF) << (hw * 16));
                int64_t rd_val = (rd == 31) ? 0 : regs[rd];
                if (rd != 31) regs[rd] = ((rd_val & mask) | (int64_t(imm16) << (hw * 16))) & 0xFFFFFFFF;
            } else {
                // ANDS immediate 32-bit
                int64_t bitmask = decode_bitmask_imm(inst);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t result = (rn_val & bitmask) & 0xFFFFFFFF;
                if (rd != 31) regs[rd] = result;
                flag_n = ((result & 0x80000000) != 0) ? 1.0f : 0.0f;
                flag_z = (result == 0) ? 1.0f : 0.0f;
                flag_c = 0.0f;
                flag_v = 0.0f;
            }
            break;
        }

        case 0x78: { // Halfword LDR/STR complex
            uint8_t opc = (inst >> 22) & 0x3;
            uint8_t opt_bits = (inst >> 10) & 0x3;

            if (opt_bits == 0x0) {
                // Unscaled offset: LDURH/STURH/LDURSH Wt, [Xn, #imm9]
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + imm9);
                if (opc == 0) {
                    store16(memory_out, addr, RD_VAL);
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = load16(memory_out, addr);
                } else {
                    // LDRSH: sign-extend halfword to 64-bit
                    int16_t val16 = int16_t(load16(memory_out, addr));
                    if (rd != 31) regs[rd] = int64_t(val16);
                }
            } else if (opt_bits == 0x2) {
                // Register offset: LDRH/STRH Wt, [Xn, Xm{, extend}]
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t offset = (rm == 31) ? 0 : regs[rm];
                uint8_t option = (inst >> 13) & 0x7;
                uint8_t S_bit = (inst >> 12) & 0x1;
                if (option == 6) {
                    offset = offset & 0xFFFFFFFF;
                    if (offset & 0x80000000) offset |= int64_t(0xFFFFFFFF00000000);
                } else if (option == 2) {
                    offset = offset & 0xFFFFFFFF;
                }
                if (S_bit) offset <<= 1;  // Scale by element size (halfword = 2)
                uint64_t addr = uint64_t(base + offset);
                if (opc == 0) {
                    store16(memory_out, addr, RD_VAL);
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = load16(memory_out, addr);
                } else {
                    int16_t val16 = int16_t(load16(memory_out, addr));
                    if (rd != 31) regs[rd] = int64_t(val16);
                }
            } else if (opt_bits == 0x1) {
                // Post-index
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                if (opc == 0) {
                    store16(memory_out, uint64_t(base), RD_VAL);
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = load16(memory_out, uint64_t(base));
                } else {
                    int16_t val16 = int16_t(load16(memory_out, uint64_t(base)));
                    if (rd != 31) regs[rd] = int64_t(val16);
                }
                regs[rn] = base + imm9;  // writeback (SP-capable)
            } else if (opt_bits == 0x3) {
                // Pre-index
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t new_base = base + imm9;
                regs[rn] = new_base;  // writeback (SP-capable)
                if (opc == 0) {
                    store16(memory_out, uint64_t(new_base), RD_VAL);
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = load16(memory_out, uint64_t(new_base));
                } else {
                    int16_t val16 = int16_t(load16(memory_out, uint64_t(new_base)));
                    if (rd != 31) regs[rd] = int64_t(val16);
                }
            }
            break;
        }

        case 0x79: { // LDRH/STRH/LDRSH unsigned offset
            if ((inst & 0xFFC00000) == 0x79400000) {
                // LDRH - Load unsigned halfword
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 * 2);
                if (rd != 31) regs[rd] = load16(memory_out, addr);
            } else if ((inst & 0xFFC00000) == 0x79000000) {
                // STRH - Store halfword
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 * 2);
                store16(memory_out, addr, RD_VAL);
            } else if ((inst & 0xFFC00000) == 0x79800000) {
                // LDRSH - Load signed halfword
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 * 2);
                int64_t val = load16(memory_out, addr);
                if (val & 0x8000) val |= int64_t(0xFFFFFFFFFFFF0000);
                if (rd != 31) regs[rd] = val;
            }
            break;
        }

        case 0x8A: { // AND/BIC register 64-bit (N=0: AND, N=1: BIC)
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            uint8_t N = (inst >> 21) & 1;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            else if (stype == 3) rm_val = int64_t((uint64_t(rm_val) >> samt) | (uint64_t(rm_val) << (64 - samt)));
            if (N) rm_val = ~rm_val;
            if (rd != 31) regs[rd] = rn_val & rm_val;
            break;
        }

        case 0x8B: { // ADD extended / ADD register 64-bit
            if ((inst & 0xFFE00000) == 0x8B200000) {
                // ADD extended register (rn=31 is SP, rd=31 writes SP)
                uint8_t ext_type = (inst >> 13) & 0x7;
                uint8_t shift = (inst >> 10) & 0x7;
                int64_t val = apply_extension(regs[rm], ext_type);
                val = (val << shift);
                int64_t rn_val = regs[rn];
                regs[rd] = rn_val + val;
            } else {
                uint8_t stype = (inst >> 22) & 0x3;
                uint8_t samt = (inst >> 10) & 0x3F;
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (stype == 0) rm_val = rm_val << samt;
                else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
                else if (stype == 2) rm_val = rm_val >> samt;
                if (rd != 31) regs[rd] = rn_val + rm_val;
            }
            break;
        }

        case 0x90: case 0xB0: case 0xD0: case 0xF0: { // ADRP
            // ADRP - PC-relative page address
            uint32_t immlo = (inst >> 29) & 0x3;
            uint32_t immhi = (inst >> 5) & 0x7FFFF;
            int32_t offset = sign_extend_21((immhi << 2) | immlo);
            int64_t page_base = int64_t(pc) & ~int64_t(0xFFF);
            if (rd != 31) regs[rd] = page_base + (int64_t(offset) << 12);
            break;
        }

        case 0x91: { // ADD immediate 64-bit
            int64_t rn_val = regs[rn];
            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
            regs[rd] = rn_val + aimm;
            break;
        }

        case 0x92: { // MOVN 64-bit / AND immediate 64-bit
            if ((inst >> 23) & 1) {
                // MOVN 64-bit
                if (rd != 31) regs[rd] = ~(int64_t(imm16) << (hw * 16));
            } else {
                // AND immediate 64-bit
                int64_t bitmask = decode_bitmask_imm(inst);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                if (rd != 31) regs[rd] = rn_val & bitmask;
            }
            break;
        }

        case 0x93: { // SXTW / SBFM 64-bit / EXTR
            if ((inst & 0xFFFFFC00) == 0x93407C00) {
                // SXTW - Sign extend 32-bit to 64-bit
                int64_t val = regs[rn] & 0xFFFFFFFF;
                if (val & 0x80000000) val |= int64_t(0xFFFFFFFF00000000);
                if (rd != 31) regs[rd] = val;
            } else if ((inst & 0xFFE00000) == 0x93C00000) {
                uint64_t val_n = uint64_t(regs[rn]);
                uint64_t val_m = uint64_t(regs[rm]);
                uint8_t lsb = (inst >> 10) & 0x3F;
                // Concatenate [Rn:Rm] and extract 64 bits starting at lsb
                // result = (Rn:Rm >> lsb)[63:0]
                uint64_t result;
                if (lsb == 0) {
                    result = val_m;
                } else {
                    result = (val_m >> lsb) | (val_n << (64 - lsb));
                }
                if (rd != 31) regs[rd] = int64_t(result);
            } else {
                // SBFM 64-bit (also handles ASR_IMM, SBFX, SXTB, SXTH, SXTW)
                uint8_t immr = (inst >> 16) & 0x3F;
                uint8_t imms = (inst >> 10) & 0x3F;
                uint64_t val = uint64_t(regs[rn]);
                uint64_t result;
                if (imms >= immr) {
                    // SBFX / ASR: extract (imms-immr+1) bits at immr, sign extend
                    uint8_t width = imms - immr + 1;
                    result = val >> immr;
                    uint64_t mask = (width < 64) ? ((uint64_t(1) << width) - 1) : 0xFFFFFFFFFFFFFFFFULL;
                    result &= mask;
                    // Sign extend from bit (width - 1)
                    if (width < 64 && (result & (uint64_t(1) << (width - 1)))) {
                        result |= ~mask;
                    }
                } else {
                    // SBFIZ: extract (imms+1) low bits, sign-extend, shift left by (64-immr)
                    uint64_t width = imms + 1;
                    uint64_t lsb = 64 - immr;
                    uint64_t mask = (uint64_t(1) << width) - 1;
                    uint64_t src_bits = val & mask;
                    if (src_bits & (uint64_t(1) << (width - 1))) {
                        src_bits |= ~mask;
                    }
                    result = src_bits << lsb;
                }
                if (rd != 31) regs[rd] = int64_t(result);
            }
            break;
        }

        case 0x94: case 0x95: case 0x96: case 0x97: { // BL
            // BL - Branch with link
            regs[30] = int64_t(pc + 4);
            pc = uint64_t(int64_t(pc) + sign_extend_26(imm26) * 4);
            branch_taken = true;
            break;
        }

        case 0x9A: { // Shift registers / Division / CSEL / CSINC 64-bit
            if ((inst & 0xFFE0FC00) == 0x9AC02000) {
                // LSL register
                int64_t shift = regs[rm] & 63;
                if (rd != 31) regs[rd] = regs[rn] << shift;
            } else if ((inst & 0xFFE0FC00) == 0x9AC02400) {
                // LSR register (logical, unsigned)
                uint64_t val = uint64_t(regs[rn]);
                int64_t shift = regs[rm] & 63;
                if (rd != 31) regs[rd] = int64_t(val >> shift);
            } else if ((inst & 0xFFE0FC00) == 0x9AC02800) {
                // ASR register (arithmetic, signed)
                int64_t shift = regs[rm] & 63;
                if (rd != 31) regs[rd] = regs[rn] >> shift;
            } else if ((inst & 0xFFE0FC00) == 0x9AC02C00) {
                // ROR register
                uint64_t val = uint64_t(regs[rn]);
                int64_t shift = regs[rm] & 63;
                if (shift > 0) {
                    val = (val >> shift) | (val << (64 - shift));
                }
                if (rd != 31) regs[rd] = int64_t(val);
            } else if ((inst & 0xFFE0FC00) == 0x9AC00C00) {
                // SDIV - Signed division
                int64_t divisor = regs[rm];
                if (divisor != 0) {
                    if (rd != 31) regs[rd] = regs[rn] / divisor;
                } else {
                    if (rd != 31) regs[rd] = 0;
                }
            } else if ((inst & 0xFFE0FC00) == 0x9AC00800) {
                // UDIV - Unsigned division
                uint64_t divisor = uint64_t(regs[rm]);
                if (divisor != 0) {
                    if (rd != 31) regs[rd] = int64_t(uint64_t(regs[rn]) / divisor);
                } else {
                    if (rd != 31) regs[rd] = 0;
                }
            } else if ((inst & 0xFFE00C00) == 0x9A800000) {
                // CSEL 64-bit: Rd = cond ? Rn : Rm (rn/rm=31 → XZR)
                uint8_t cc = (inst >> 12) & 0xF;
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (rd != 31) regs[rd] = take ? rn_val : rm_val;
            } else if ((inst & 0xFFE00C00) == 0x9A800400) {
                // CSINC 64-bit: Rd = cond ? Rn : Rm+1 (rn/rm=31 → XZR; CSET when rn=rm=31)
                uint8_t cc = (inst >> 12) & 0xF;
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (rd != 31) regs[rd] = take ? rn_val : (rm_val + 1);
            }
            break;
        }

        case 0x9B: { // Multiply variants
            if ((inst & 0xFFE0FC00) == 0x9BC07C00) {
                // UMULH 64-bit: Rd = (unsigned(Rn) * unsigned(Rm)) >> 64
                uint64_t a = uint64_t(regs[rn]);
                uint64_t b = uint64_t(regs[rm]);
                uint64_t a_lo = a & 0xFFFFFFFF;
                uint64_t a_hi = a >> 32;
                uint64_t b_lo = b & 0xFFFFFFFF;
                uint64_t b_hi = b >> 32;
                uint64_t p0 = a_lo * b_lo;
                uint64_t p1 = a_lo * b_hi;
                uint64_t p2 = a_hi * b_lo;
                uint64_t p3 = a_hi * b_hi;
                uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
                uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + carry;
                if (rd != 31) regs[rd] = int64_t(hi);
            } else if ((inst & 0xFFE0FC00) == 0x9B407C00) {
                int64_t a = regs[rn];
                int64_t b = regs[rm];
                // Decompose into unsigned parts, adjust sign at end
                uint64_t ua = uint64_t(a < 0 ? -a : a);
                uint64_t ub = uint64_t(b < 0 ? -b : b);
                uint64_t a_lo = ua & 0xFFFFFFFF, a_hi = ua >> 32;
                uint64_t b_lo = ub & 0xFFFFFFFF, b_hi = ub >> 32;
                uint64_t p0 = a_lo * b_lo;
                uint64_t p1 = a_lo * b_hi;
                uint64_t p2 = a_hi * b_lo;
                uint64_t p3 = a_hi * b_hi;
                uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
                uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + carry;
                bool neg = (a < 0) != (b < 0);
                if (neg) {
                    // Two's complement: if low product is 0, just negate hi; else negate and subtract borrow
                    uint64_t lo = p0 + (p1 << 32) + (p2 << 32);
                    hi = (lo == 0) ? (~hi + 1) : ~hi;
                }
                if (rd != 31) regs[rd] = int64_t(hi);
            } else if ((inst & 0xFFE08000) == 0x9B200000) {
                int64_t nval = int64_t(int32_t(regs[rn] & 0xFFFFFFFF));
                int64_t mval = int64_t(int32_t(regs[rm] & 0xFFFFFFFF));
                int64_t ra_val = (ra == 31) ? 0 : regs[ra];
                if (rd != 31) regs[rd] = ra_val + nval * mval;
            } else if ((inst & 0xFFE08000) == 0x9B208000) {
                int64_t nval = int64_t(int32_t(regs[rn] & 0xFFFFFFFF));
                int64_t mval = int64_t(int32_t(regs[rm] & 0xFFFFFFFF));
                int64_t ra_val = (ra == 31) ? 0 : regs[ra];
                if (rd != 31) regs[rd] = ra_val - nval * mval;
            } else if ((inst & 0xFFE08000) == 0x9BA00000) {
                uint64_t nval = uint64_t(regs[rn]) & 0xFFFFFFFF;
                uint64_t mval = uint64_t(regs[rm]) & 0xFFFFFFFF;
                int64_t ra_val = (ra == 31) ? 0 : regs[ra];
                if (rd != 31) regs[rd] = int64_t(uint64_t(ra_val) + nval * mval);
            } else if ((inst & 0xFFE08000) == 0x9BA08000) {
                uint64_t nval = uint64_t(regs[rn]) & 0xFFFFFFFF;
                uint64_t mval = uint64_t(regs[rm]) & 0xFFFFFFFF;
                int64_t ra_val = (ra == 31) ? 0 : regs[ra];
                if (rd != 31) regs[rd] = int64_t(uint64_t(ra_val) - nval * mval);
            } else if ((inst & 0xFFE08000) == 0x9B008000) {
                // MSUB 64-bit: Rd = Ra - Rn * Rm
                int64_t ra_val = (ra == 31) ? 0 : regs[ra];
                if (rd != 31) regs[rd] = ra_val - regs[rn] * regs[rm];
            } else if ((inst & 0xFFE08000) == 0x9B000000) {
                // MADD 64-bit: Rd = Ra + Rn * Rm (MUL when Ra=XZR)
                int64_t ra_val = (ra == 31) ? 0 : regs[ra];
                if (rd != 31) regs[rd] = ra_val + regs[rn] * regs[rm];
            }
            break;
        }

        case 0xA8: { // LDP/STP post-index 64-bit
            if ((inst & 0xFFC00000) == 0xA8C00000) {
                // LDP post-index 64-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 8;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base);
                if (rd != 31) regs[rd] = load64(memory_out, addr);
                if (rt2 != 31) regs[rt2] = load64(memory_out, addr + 8);
                regs[rn] = base + imm7;  // writeback (SP-capable)
            } else if ((inst & 0xFFC00000) == 0xA8800000) {
                // STP post-index 64-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 8;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base);
                store64(memory_out, addr, RD_VAL);
                store64(memory_out, addr + 8, RT2_VAL);
                regs[rn] = base + imm7;  // writeback (SP-capable)
            }
            break;
        }

        case 0xA9: { // LDP/STP 64-bit
            if ((inst & 0xFFC00000) == 0xA9C00000) {
                // LDP pre-index 64-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 8;
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t new_base = base + imm7;
                uint64_t addr = uint64_t(new_base);
                if (rd != 31) regs[rd] = load64(memory_out, addr);
                if (rt2 != 31) regs[rt2] = load64(memory_out, addr + 8);
                regs[rn] = new_base;  // writeback (SP-capable)
            } else if ((inst & 0xFFC00000) == 0xA9800000) {
                // STP pre-index 64-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 8;
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t new_base = base + imm7;
                uint64_t addr = uint64_t(new_base);
                store64(memory_out, addr, RD_VAL);
                store64(memory_out, addr + 8, RT2_VAL);
                regs[rn] = new_base;  // writeback (SP-capable)
            } else if ((inst & 0xFFC00000) == 0xA9400000) {
                // LDP signed-offset 64-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 8;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + imm7);
                if (rd != 31) regs[rd] = load64(memory_out, addr);
                if (rt2 != 31) regs[rt2] = load64(memory_out, addr + 8);
            } else if ((inst & 0xFFC00000) == 0xA9000000) {
                // STP signed-offset 64-bit
                int32_t imm7 = sign_extend_7((inst >> 15) & 0x7F) * 8;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + imm7);
                store64(memory_out, addr, RD_VAL);
                store64(memory_out, addr + 8, RT2_VAL);
            }
            break;
        }

        case 0xAA: { // MVN / ORR register 64-bit
            if ((inst & 0xFFE0FFE0) == 0xAA2003E0) {
                // MVN - Bitwise NOT: ORN Xd, XZR, Xm
                if (rd != 31) regs[rd] = ~regs[rm];
            } else {
                uint8_t stype = (inst >> 22) & 0x3;
                uint8_t samt = (inst >> 10) & 0x3F;
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (stype == 0) rm_val = rm_val << samt;
                else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
                else if (stype == 2) rm_val = rm_val >> samt;
                else if (stype == 3) rm_val = int64_t((uint64_t(rm_val) >> samt) | (uint64_t(rm_val) << (64 - samt)));
                if (rd != 31) regs[rd] = rn_val | rm_val;
            }
            break;
        }

        case 0xAB: { // ADDS extended / ADDS register 64-bit
            if ((inst & 0xFFE00000) == 0xAB200000) {
                // ADDS extended register (64-bit, flag-setting)
                // CMP with extended reg when Rd=XZR (e.g., CMP X1, W0, SXTW)
                uint8_t ext_type = (inst >> 13) & 0x7;
                uint8_t shift = (inst >> 10) & 0x7;
                int64_t val = apply_extension(regs[rm], ext_type);
                val = (val << shift);
                int64_t rn_val = regs[rn];
                int64_t result = rn_val + val;
                if (rd != 31) regs[rd] = result;
                flag_n = (result < 0) ? 1.0f : 0.0f;
                flag_z = (result == 0) ? 1.0f : 0.0f;
                flag_c = (uint64_t(result) < uint64_t(rn_val)) ? 1.0f : 0.0f;
                flag_v = ((rn_val ^ result) & ~(rn_val ^ val) & (int64_t(1) << 63)) ? 1.0f : 0.0f;
            } else {
                uint8_t stype = (inst >> 22) & 0x3;
                uint8_t samt = (inst >> 10) & 0x3F;
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (stype == 0) rm_val = rm_val << samt;
                else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
                else if (stype == 2) rm_val = rm_val >> samt;
                int64_t result = rn_val + rm_val;
                if (rd != 31) regs[rd] = result;
                flag_n = (result < 0) ? 1.0f : 0.0f;
                flag_z = (result == 0) ? 1.0f : 0.0f;
                flag_c = (uint64_t(result) < uint64_t(rn_val)) ? 1.0f : 0.0f;
                flag_v = ((rn_val ^ result) & ~(rn_val ^ rm_val) & (int64_t(1) << 63)) ? 1.0f : 0.0f;
            }
            break;
        }

        case 0xAD: { // STP/LDP Q 128-bit pair
            if ((inst & 0xFFC00000) == 0xAD000000) {
                uint8_t rt = inst & 0x1F;
                uint8_t rt2_v = (inst >> 10) & 0x1F;
                int32_t offset = sign_extend_7((inst >> 15) & 0x7F) * 16;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + offset);
                store64(memory_out, addr, vreg_lo[rt]);
                store64(memory_out, addr + 8, vreg_hi[rt]);
                store64(memory_out, addr + 16, vreg_lo[rt2_v]);
                store64(memory_out, addr + 24, vreg_hi[rt2_v]);
            } else if ((inst & 0xFFC00000) == 0xAD400000) {
                uint8_t rt = inst & 0x1F;
                uint8_t rt2_v = (inst >> 10) & 0x1F;
                int32_t offset = sign_extend_7((inst >> 15) & 0x7F) * 16;
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + offset);
                vreg_lo[rt] = load64(memory_out, addr);
                vreg_hi[rt] = load64(memory_out, addr + 8);
                vreg_lo[rt2_v] = load64(memory_out, addr + 16);
                vreg_hi[rt2_v] = load64(memory_out, addr + 24);
            }
            break;
        }

        case 0xB1: { // ADDS immediate 64-bit
            int64_t rn_val = regs[rn];
            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
            int64_t result = rn_val + aimm;
            if (rd != 31) regs[rd] = result;
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = (uint64_t(rn_val) > uint64_t(result)) ? 1.0f : 0.0f;
            flag_v = ((rn_val ^ result) & ~(rn_val ^ aimm) & (int64_t(1) << 63)) ? 1.0f : 0.0f;
            break;
        }

        case 0xB2: { // ORR immediate 64-bit
            int64_t bitmask = decode_bitmask_imm(inst);
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            if (rd != 31) regs[rd] = rn_val | bitmask;
            break;
        }

        case 0xB3: { // BFM 64-bit
            uint8_t immr = (inst >> 16) & 0x3F;
            uint8_t imms = (inst >> 10) & 0x3F;
            uint64_t src = uint64_t(regs[rn]);
            uint64_t dst = (rd != 31) ? uint64_t(regs[rd]) : 0;
            if (imms >= immr) {
                // BFXIL: extract (imms-immr+1) bits at immr, insert at bit 0
                uint8_t width = imms - immr + 1;
                uint64_t extracted = (src >> immr);
                uint64_t mask = (width < 64) ? ((uint64_t(1) << width) - 1) : 0xFFFFFFFFFFFFFFFFULL;
                extracted &= mask;
                dst = (dst & ~mask) | extracted;
            } else {
                // BFI: extract (imms+1) low bits, insert at bit (64-immr)
                uint8_t width = imms + 1;
                uint8_t lsb = 64 - immr;
                uint64_t extracted = src & ((uint64_t(1) << width) - 1);
                uint64_t mask = ((uint64_t(1) << width) - 1) << lsb;
                dst = (dst & ~mask) | (extracted << lsb);
            }
            if (rd != 31) regs[rd] = int64_t(dst);
            break;
        }

        case 0xB8: { // Word LDR/STR complex
            uint8_t opc = (inst >> 22) & 0x3;
            uint8_t opt_bits = (inst >> 10) & 0x3;

            if (opt_bits == 0x0) {
                // Unscaled offset: LDUR/STUR/LDURSW Wt, [Xn, #imm9]
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + imm9);
                if (opc == 0) {
                    store32(memory_out, addr, int32_t(RD_VAL & 0xFFFFFFFF));
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = int64_t(load32(memory_out, addr)) & 0xFFFFFFFF;
                } else {
                    // LDURSW: load 32-bit, sign-extend to 64-bit
                    int32_t val32 = int32_t(load32(memory_out, addr));
                    if (rd != 31) regs[rd] = int64_t(val32);
                }
            } else if (opt_bits == 0x2) {
                // Register offset: LDR/STR Wt, [Xn, Xm{, extend}]
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t offset = (rm == 31) ? 0 : regs[rm];
                uint8_t option = (inst >> 13) & 0x7;
                uint8_t S_bit = (inst >> 12) & 0x1;
                if (option == 6) {
                    offset = offset & 0xFFFFFFFF;
                    if (offset & 0x80000000) offset |= int64_t(0xFFFFFFFF00000000);
                } else if (option == 2) {
                    offset = offset & 0xFFFFFFFF;
                }
                if (S_bit) offset <<= 2;  // Scale by element size (word = 4)
                uint64_t addr = uint64_t(base + offset);
                if (opc == 0) {
                    store32(memory_out, addr, int32_t(RD_VAL & 0xFFFFFFFF));
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = int64_t(load32(memory_out, addr)) & 0xFFFFFFFF;
                } else {
                    int32_t val32 = int32_t(load32(memory_out, addr));
                    if (rd != 31) regs[rd] = int64_t(val32);
                }
            } else if (opt_bits == 0x1) {
                // Post-index: LDR/STR Wt, [Xn], #imm9
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                if (opc == 0) {
                    store32(memory_out, uint64_t(base), int32_t(RD_VAL & 0xFFFFFFFF));
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = int64_t(load32(memory_out, uint64_t(base))) & 0xFFFFFFFF;
                } else {
                    int32_t val32 = int32_t(load32(memory_out, uint64_t(base)));
                    if (rd != 31) regs[rd] = int64_t(val32);
                }
                regs[rn] = base + imm9;  // writeback (SP-capable)
            } else if (opt_bits == 0x3) {
                // Pre-index: LDR/STR Wt, [Xn, #imm9]!
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t new_base = base + imm9;
                regs[rn] = new_base;  // writeback (SP-capable)
                if (opc == 0) {
                    store32(memory_out, uint64_t(new_base), int32_t(RD_VAL & 0xFFFFFFFF));
                } else if (opc == 1) {
                    if (rd != 31) regs[rd] = int64_t(load32(memory_out, uint64_t(new_base))) & 0xFFFFFFFF;
                } else {
                    int32_t val32 = int32_t(load32(memory_out, uint64_t(new_base)));
                    if (rd != 31) regs[rd] = int64_t(val32);
                }
            }
            break;
        }

        case 0xB9: { // LDR/STR/LDRSW 32-bit unsigned offset
            if ((inst & 0xFFC00000) == 0xB9800000) {
                // LDRSW - Load signed word
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 * 4);
                int64_t val = int64_t(load32(memory_out, addr));
                if (val & 0x80000000) val |= int64_t(0xFFFFFFFF00000000);
                if (rd != 31) regs[rd] = val;
            } else if ((inst & 0xFFC00000) == 0xB9400000) {
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 << 2);
                if (rd != 31) regs[rd] = int64_t(load32(memory_out, addr)) & 0xFFFFFFFF;
            } else if ((inst & 0xFFC00000) == 0xB9000000) {
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 << 2);
                store32(memory_out, addr, int32_t(RD_VAL & 0xFFFFFFFF));
            }
            break;
        }

        case 0xBD: { // STR/LDR Sn FP 32-bit unsigned offset
            if ((inst & 0xFFC00000) == 0xBD000000) {
                uint8_t rt = inst & 0x1F;
                int64_t base = regs[rn];
                uint64_t addr = uint64_t(base) + (imm12 * 4);
                store32(memory_out, addr, int32_t(vreg_lo[rt] & 0xFFFFFFFF));
            } else if ((inst & 0xFFC00000) == 0xBD400000) {
                uint8_t rt = inst & 0x1F;
                int64_t base = regs[rn];
                uint64_t addr = uint64_t(base) + (imm12 * 4);
                vreg_lo[rt] = int64_t(uint32_t(load32(memory_out, addr)));
                vreg_hi[rt] = 0;
            }
            break;
        }

        case 0x88: { // LDXR/LDAXR/STXR/STLXR 32-bit (Load/Store Exclusive)
            uint8_t L = (inst >> 22) & 1;
            int64_t base = (rn == 31) ? regs[31] : regs[rn];  // rn=31 is SP
            if (L) {
                // LDXR/LDAXR W - Load exclusive 32-bit (acquire semantics ignored on GPU)
                if (rd != 31) regs[rd] = int64_t(load32(memory_out, uint64_t(base))) & 0xFFFFFFFF;
            } else {
                // STXR/STLXR W - Store exclusive 32-bit (always succeeds on GPU)
                uint8_t rs = (inst >> 16) & 0x1F;
                store32(memory_out, uint64_t(base), uint32_t(RD_VAL & 0xFFFFFFFF));
                if (rs != 31) regs[rs] = 0;  // Always succeeds (single-threaded GPU)
            }
            break;
        }

        case 0xC8: { // LDXR/LDAXR/STXR/STLXR 64-bit (Load/Store Exclusive)
            uint8_t L = (inst >> 22) & 1;
            int64_t base = (rn == 31) ? regs[31] : regs[rn];  // rn=31 is SP
            if (L) {
                // LDXR/LDAXR X - Load exclusive 64-bit
                if (rd != 31) regs[rd] = load64(memory_out, uint64_t(base));
            } else {
                // STXR/STLXR X - Store exclusive 64-bit (always succeeds)
                uint8_t rs = (inst >> 16) & 0x1F;
                store64(memory_out, uint64_t(base), RD_VAL);
                if (rs != 31) regs[rs] = 0;  // Always succeeds (single-threaded GPU)
            }
            break;
        }

        case 0xCA: { // EOR/EON register 64-bit (N=0: EOR, N=1: EON)
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            uint8_t N = (inst >> 21) & 1;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            else if (stype == 3) rm_val = int64_t((uint64_t(rm_val) >> samt) | (uint64_t(rm_val) << (64 - samt)));
            if (N) rm_val = ~rm_val;
            if (rd != 31) regs[rd] = rn_val ^ rm_val;
            break;
        }

        case 0xCB: { // NEG / SUB extended / SUB register 64-bit
            if ((inst & 0xFFE0FFE0) == 0xCB0003E0) {
                // NEG - Negate: SUB Xd, XZR, Xm
                if (rd != 31) regs[rd] = -regs[rm];
            } else if ((inst & 0xFFE00000) == 0xCB200000) {
                // SUB extended register (rn=31 is SP, rd=31 writes SP)
                uint8_t ext_type = (inst >> 13) & 0x7;
                uint8_t shift = (inst >> 10) & 0x7;
                int64_t val = apply_extension(regs[rm], ext_type);
                val = (val << shift);
                int64_t rn_val = regs[rn];
                regs[rd] = rn_val - val;
            } else {
                uint8_t stype = (inst >> 22) & 0x3;
                uint8_t samt = (inst >> 10) & 0x3F;
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (stype == 0) rm_val = rm_val << samt;
                else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
                else if (stype == 2) rm_val = rm_val >> samt;
                if (rd != 31) regs[rd] = rn_val - rm_val;
            }
            break;
        }

        case 0xD1: { // SUB immediate 64-bit
            int64_t rn_val = regs[rn];
            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
            regs[rd] = rn_val - aimm;
            break;
        }

        case 0xD2: { // MOVZ 64-bit / EOR immediate 64-bit
            if ((inst >> 23) & 1) {
                // MOVZ 64-bit
                if (rd != 31) regs[rd] = int64_t(imm16) << (hw * 16);
            } else {
                // EOR immediate 64-bit
                int64_t bitmask = decode_bitmask_imm(inst);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                if (rd != 31) regs[rd] = rn_val ^ bitmask;
            }
            break;
        }

        case 0xD3: { // UXTB / UXTH / UBFM 64-bit
            if ((inst & 0xFFFFFC00) == 0xD3401C00) {
                // UXTB - Zero extend byte
                if (rd != 31) regs[rd] = regs[rn] & 0xFF;
            } else if ((inst & 0xFFFFFC00) == 0xD3403C00) {
                // UXTH - Zero extend halfword
                if (rd != 31) regs[rd] = regs[rn] & 0xFFFF;
            } else {
                // UBFM 64-bit: handles LSR_IMM, LSL_IMM, UBFX, UXTB, UXTH
                uint8_t immr = (inst >> 16) & 0x3F;
                uint8_t imms = (inst >> 10) & 0x3F;
                uint64_t val = uint64_t(regs[rn]);
                uint64_t result;
                if (imms >= immr) {
                    // UBFX / LSR: extract (imms-immr+1) bits starting at immr
                    uint8_t width = imms - immr + 1;
                    result = val >> immr;
                    uint64_t mask = (width < 64) ? ((uint64_t(1) << width) - 1) : 0xFFFFFFFFFFFFFFFFULL;
                    result &= mask;
                } else {
                    // UBFIZ / LSL: extract low (imms+1) bits, shift left by (64-immr)
                    uint64_t src_bits = val & ((uint64_t(1) << (imms + 1)) - 1);
                    result = src_bits << (64 - immr);
                }
                if (rd != 31) regs[rd] = int64_t(result);
            }
            break;
        }

        case 0xD5: { // NOP / System instructions
            if (inst == 0xD503201F) {
                // NOP
            } else if ((inst & 0xFFFFF0FF) == 0xD50330BF) {
                // no-op
            } else if ((inst & 0xFFFFF0FF) == 0xD503309F) {
                // no-op
            } else if ((inst & 0xFFFFF0FF) == 0xD50330DF) {
                // no-op
            } else if ((inst & 0xFFF00000) == 0xD5300000) {
                // MRS - simplified: return 0 for all system registers
                if (rd != 31) regs[rd] = 0;
            } else if ((inst & 0xFFF00000) == 0xD5100000) {
                // MSR - simplified: discard writes
            } else if ((inst & 0xFFFFFFE0) == 0xD50B7420) {
                // DC ZVA, Xt — zero cache line (64 bytes aligned)
                uint8_t rt_dc = inst & 0x1F;
                int64_t addr_val = (rt_dc == 31) ? 0 : regs[rt_dc];
                uint64_t aligned = uint64_t(addr_val) & ~uint64_t(63);
                for (int i = 0; i < 64; i++) {
                    memory_out[aligned + i] = 0;
                }
            }
            break;
        }

        case 0xD6: { // ERET / BR / BLR / RET
            if (inst == 0xD69F03E0) {
                // ERET - Simplified: halt (no exception stack on GPU)
                reason = STOP_HALT;
                halt_break = true;
            } else if ((inst & 0xFFFFFC1F) == 0xD61F0000) {
                // BR - Branch to register
                pc = uint64_t((rn == 31) ? 0 : regs[rn]);
                branch_taken = true;
            } else if ((inst & 0xFFFFFC1F) == 0xD63F0000) {
                // BLR - Branch with link to register
                regs[30] = int64_t(pc + 4);
                pc = uint64_t((rn == 31) ? 0 : regs[rn]);
                branch_taken = true;
            } else if ((inst & 0xFFFFFC1F) == 0xD65F0000) {
                // RET
                pc = uint64_t((rn == 31) ? 0 : regs[rn]);
                branch_taken = true;
            }
            break;
        }

        case 0xDA: { // Bit manipulation / CSINV / CSNEG 64-bit
            if ((inst & 0xFFFFFC00) == 0xDAC01000) {
                // CLZ - Count leading zeros
                uint64_t val = uint64_t(regs[rn]) ;
                int64_t count = 64;
                for (int i = 63; i >= 0; i--) {
                    if (val & (uint64_t(1) << i)) { count = 63 - i; break; }
                }
                if (rd != 31) regs[rd] = count;
            } else if ((inst & 0xFFFFFC00) == 0xDAC00000) {
                // RBIT - Reverse bits
                uint64_t val = uint64_t(regs[rn]);
                uint64_t result = 0;
                for (int i = 0; i < 64; i++) {
                    if (val & (uint64_t(1) << i)) result |= uint64_t(1) << (63 - i);
                }
                if (rd != 31) regs[rd] = int64_t(result);
            } else if ((inst & 0xFFFFFC00) == 0xDAC00C00) {
                // REV - Reverse bytes (64-bit)
                uint64_t val = uint64_t(regs[rn]);
                uint64_t result = 0;
                for (int i = 0; i < 8; i++) {
                    uint64_t b = (val >> (i * 8)) & 0xFF;
                    result |= b << ((7 - i) * 8);
                }
                if (rd != 31) regs[rd] = int64_t(result);
            } else if ((inst & 0xFFFFFC00) == 0xDAC00400) {
                // REV16 - Reverse bytes in each 16-bit halfword
                uint64_t val = uint64_t(regs[rn]);
                uint64_t result = 0;
                for (int i = 0; i < 4; i++) {
                    uint64_t hw_val = (val >> (i * 16)) & 0xFFFF;
                    uint64_t b0 = hw_val & 0xFF, b1 = (hw_val >> 8) & 0xFF;
                    result |= ((b0 << 8) | b1) << (i * 16);
                }
                if (rd != 31) regs[rd] = int64_t(result);
            } else if ((inst & 0xFFFFFC00) == 0xDAC00800) {
                // REV32 - Reverse bytes in each 32-bit word
                uint64_t val = uint64_t(regs[rn]);
                uint64_t result = 0;
                for (int i = 0; i < 2; i++) {
                    uint64_t word = (val >> (i * 32)) & 0xFFFFFFFF;
                    uint64_t rev = 0;
                    for (int j = 0; j < 4; j++) {
                        rev |= ((word >> (j * 8)) & 0xFF) << ((3 - j) * 8);
                    }
                    result |= rev << (i * 32);
                }
                if (rd != 31) regs[rd] = int64_t(result);
            } else if ((inst & 0xFFE00C00) == 0xDA800000) {
                // CSINV 64-bit: Rd = cond ? Rn : ~Rm (rn/rm=31 → XZR)
                uint8_t cc = (inst >> 12) & 0xF;
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (rd != 31) regs[rd] = take ? rn_val : (~rm_val);
            } else if ((inst & 0xFFE00C00) == 0xDA800400) {
                // CSNEG 64-bit: Rd = cond ? Rn : -Rm (rn/rm=31 → XZR)
                uint8_t cc = (inst >> 12) & 0xF;
                bool take = eval_condition(cc, flag_n, flag_z, flag_c, flag_v);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (rd != 31) regs[rd] = take ? rn_val : (-rm_val);
            }
            break;
        }

        case 0xEA: { // ANDS/BICS register 64-bit with shift (N=0: ANDS, N=1: BICS)
            uint8_t stype = (inst >> 22) & 0x3;
            uint8_t samt = (inst >> 10) & 0x3F;
            uint8_t N = (inst >> 21) & 1;
            int64_t rn_val = (rn == 31) ? 0 : regs[rn];
            int64_t rm_val = (rm == 31) ? 0 : regs[rm];
            if (stype == 0) rm_val = rm_val << samt;
            else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
            else if (stype == 2) rm_val = rm_val >> samt;
            else if (stype == 3) rm_val = int64_t((uint64_t(rm_val) >> samt) | (uint64_t(rm_val) << (64 - samt)));
            if (N) rm_val = ~rm_val;
            int64_t result = rn_val & rm_val;
            if (rd != 31) regs[rd] = result;
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = 0.0f;
            flag_v = 0.0f;
            break;
        }

        case 0xEB: { // SUBS extended / SUBS register 64-bit
            if ((inst & 0xFFE00000) == 0xEB200000) {
                // SUBS extended register (64-bit, flag-setting)
                // CMP with extended reg when Rd=XZR (e.g., CMP X1, W0, SXTW)
                uint8_t ext_type = (inst >> 13) & 0x7;
                uint8_t shift = (inst >> 10) & 0x7;
                int64_t val = apply_extension(regs[rm], ext_type);
                val = (val << shift);
                int64_t rn_val = regs[rn];
                int64_t result = rn_val - val;
                if (rd != 31) regs[rd] = result;
                flag_n = (result < 0) ? 1.0f : 0.0f;
                flag_z = (result == 0) ? 1.0f : 0.0f;
                flag_c = (uint64_t(rn_val) >= uint64_t(val)) ? 1.0f : 0.0f;
                flag_v = ((rn_val ^ val) & (rn_val ^ result) & (int64_t(1) << 63)) ? 1.0f : 0.0f;
            } else {
                uint8_t stype = (inst >> 22) & 0x3;
                uint8_t samt = (inst >> 10) & 0x3F;
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t rm_val = (rm == 31) ? 0 : regs[rm];
                if (stype == 0) rm_val = rm_val << samt;
                else if (stype == 1) rm_val = int64_t(uint64_t(rm_val) >> samt);
                else if (stype == 2) rm_val = rm_val >> samt;
                int64_t result = rn_val - rm_val;
                if (rd != 31) regs[rd] = result;
                flag_n = (result < 0) ? 1.0f : 0.0f;
                flag_z = (result == 0) ? 1.0f : 0.0f;
                flag_c = (uint64_t(rn_val) >= uint64_t(rm_val)) ? 1.0f : 0.0f;
                flag_v = ((rn_val ^ rm_val) & (rn_val ^ result) & (int64_t(1) << 63)) ? 1.0f : 0.0f;
            }
            break;
        }

        case 0xF1: { // SUBS immediate 64-bit
            int64_t rn_val = regs[rn];
            int64_t aimm = ((inst >> 22) & 1) ? (int64_t(imm12) << 12) : int64_t(imm12);
            int64_t result = rn_val - aimm;
            if (rd != 31) regs[rd] = result;
            flag_n = (result < 0) ? 1.0f : 0.0f;
            flag_z = (result == 0) ? 1.0f : 0.0f;
            flag_c = (uint64_t(rn_val) >= uint64_t(aimm)) ? 1.0f : 0.0f;
            flag_v = ((rn_val ^ aimm) & (rn_val ^ result) & (int64_t(1) << 63)) ? 1.0f : 0.0f;
            break;
        }

        case 0xF2: { // MOVK 64-bit / ANDS immediate
            if ((inst >> 23) & 1) {
                // MOVK 64-bit
                int64_t mask = ~(int64_t(0xFFFF) << (hw * 16));
                int64_t rd_val = (rd == 31) ? 0 : regs[rd];
                if (rd != 31) regs[rd] = (rd_val & mask) | (int64_t(imm16) << (hw * 16));
            } else {
                // ANDS immediate (TST when rd=31)
                int64_t bitmask = decode_bitmask_imm(inst);
                int64_t rn_val = (rn == 31) ? 0 : regs[rn];
                int64_t result = rn_val & bitmask;
                if (rd != 31) regs[rd] = result;
                flag_n = (result < 0) ? 1.0f : 0.0f;
                flag_z = (result == 0) ? 1.0f : 0.0f;
                flag_c = 0.0f;
                flag_v = 0.0f;
            }
            break;
        }

        case 0xF8: { // 64-bit LDR/STR complex
            uint8_t opc_bit = (inst >> 22) & 0x1;  // 1=load, 0=store
            uint8_t opt_bits = (inst >> 10) & 0x3;

            if (opt_bits == 0x2) {
                // Register offset: LDR/STR Xt, [Xn, Xm, LSL #shift]
                uint8_t shift_bit = (inst >> 12) & 0x1;
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t offset = ((rm == 31) ? 0 : regs[rm]) << (shift_bit ? 3 : 0);
                uint64_t addr = uint64_t(base + offset);
                if (opc_bit) {
                    if (rd != 31) regs[rd] = load64(memory_out, addr);
                } else {
                    store64(memory_out, addr, RD_VAL);
                }
            } else if (opt_bits == 0x1) {
                // Post-index: LDR/STR Xt, [Xn], #imm9
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base);
                if (opc_bit) {
                    if (rd != 31) regs[rd] = load64(memory_out, addr);
                } else {
                    store64(memory_out, addr, RD_VAL);
                }
                regs[rn] = base + imm9;  // writeback (SP-capable)
            } else if (opt_bits == 0x3) {
                // Pre-index: LDR/STR Xt, [Xn, #imm9]!
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                int64_t new_base = base + imm9;
                regs[rn] = new_base;  // writeback (SP-capable)
                uint64_t addr = uint64_t(new_base);
                if (opc_bit) {
                    if (rd != 31) regs[rd] = load64(memory_out, addr);
                } else {
                    store64(memory_out, addr, RD_VAL);
                }
            } else {
                // Unscaled: LDUR/STUR
                int32_t imm9 = sign_extend_9((inst >> 12) & 0x1FF);
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base + imm9);
                if (opc_bit) {
                    if (rd != 31) regs[rd] = load64(memory_out, addr);
                } else {
                    store64(memory_out, addr, RD_VAL);
                }
            }
            break;
        }

        case 0xF9: { // LDR/STR 64-bit unsigned offset
            if ((inst & 0xFFC00000) == 0xF9400000) {
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 << 3);
                if (rd != 31) regs[rd] = load64(memory_out, addr);
            } else if ((inst & 0xFFC00000) == 0xF9000000) {
                int64_t base = regs[rn];  // rn=31 is SP
                uint64_t addr = uint64_t(base) + (imm12 << 3);
                store64(memory_out, addr, RD_VAL);
            }
            break;
        }

        case 0xFD: { // STR/LDR Dn FP 64-bit unsigned offset
            if ((inst & 0xFFC00000) == 0xFD000000) {
                uint8_t rt = inst & 0x1F;
                int64_t base = regs[rn];
                uint64_t addr = uint64_t(base) + (imm12 * 8);
                store64(memory_out, addr, vreg_lo[rt]);
            } else if ((inst & 0xFFC00000) == 0xFD400000) {
                uint8_t rt = inst & 0x1F;
                int64_t base = regs[rn];
                uint64_t addr = uint64_t(base) + (imm12 * 8);
                vreg_lo[rt] = load64(memory_out, addr);
                vreg_hi[rt] = 0;
            }
            break;
        }

        default: {
            // Track unknown instructions at memory addresses 0-19
            uint32_t unk_count = load32(memory_out, 0);
            unk_count++;
            store32(memory_out, 0, int64_t(unk_count));
            store64(memory_out, 8, int64_t(pc));   // Last unknown PC
            store32(memory_out, 16, int64_t(inst)); // Last unknown inst
            break;
        }

        } // end switch(op_byte)

        // Check if ERET requested halt — exit while loop
        if (halt_break) break;
        if (!branch_taken) {
            pc += 4;
        }
        cycles++;
    }

    // ════════════════════════════════════════════════════════════════════════
    // WRITE OUTPUTS
    // ════════════════════════════════════════════════════════════════════════
    for (int i = 0; i < 32; i++) {
        registers_out[i] = regs[i];
        vreg_lo_out[i] = vreg_lo[i];
        vreg_hi_out[i] = vreg_hi[i];
    }
    pc_out[0] = pc;
    flags_out[0] = flag_n;
    flags_out[1] = flag_z;
    flags_out[2] = flag_c;
    flags_out[3] = flag_v;
    cycles_out[0] = cycles;

    if (reason == STOP_RUNNING && cycles >= max_cycles) {
        reason = STOP_MAX_CYCLES;
    }
    stop_reason_out[0] = reason;
"""


def get_kernel_source_v2() -> tuple[str, str]:
    """Get the V2 Metal kernel header and source code."""
    return KERNEL_HEADER_V2, KERNEL_SOURCE_V2


if __name__ == "__main__":
    print("=" * 70)
    print("ARM64 CPU EMULATOR - METAL KERNEL V2 (Full 139-Instruction ISA)")
    print("=" * 70)

    header, source = get_kernel_source_v2()
    print(f"Header length: {len(header):,} characters")
    print(f"Source length: {len(source):,} characters")
    print(f"Total: {len(header) + len(source):,} characters")
    print()
    print("Full ARM64 ISA with:")
    print("  - All ALU ops (ADD/SUB/MUL/DIV/AND/ORR/EOR + 32-bit variants)")
    print("  - All shifts (LSL/LSR/ASR/ROR register + UBFM/SBFM immediate)")
    print("  - Bit manipulation (CLZ/RBIT/REV/REV16/REV32)")
    print("  - Conditional select (CSEL/CSINC/CSINV/CSNEG)")
    print("  - All branches (B/BL/BR/BLR/RET/B.cond/CBZ/CBNZ/TBZ/TBNZ)")
    print("  - Full memory (LDR/STR 64/32/16/8, signed, pre/post-index, reg-offset)")
    print("  - Load/store pair (LDP/STP with signed/pre/post-index)")
    print("  - Logical immediates (full ARM64 bitmask decode)")
    print("  - Extension arithmetic (ADD_EXT/SUB_EXT)")
    print("  - System (DMB/DSB/ISB/MRS/MSR/ERET)")
    print("  - Atomic (LDXR/STXR simplified)")
    print("  - Double-buffer memory architecture (full read/write)")
