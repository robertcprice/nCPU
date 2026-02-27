//! Multi-kernel ARM64 execution - splits instructions across multiple kernels
//!
//! This avoids Metal compiler complexity limits by compiling each instruction
//! category as a separate kernel/pipeline state.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::time::Instant;

use crate::{MetalError, get_default_device, ExecutionResult};

// Import PyModule type for the registration function
use pyo3::types::PyModule;

/// Arithmetic instruction opcodes
const ARITHMETIC_OPS: &[u8] = &[
    0x91, 0x11, 0xD1, 0x51, 0xD2, 0xF2, 0x92, 0x8B, 0xCB, 0x0B,
];

/// Logical instruction opcodes
const LOGICAL_OPS: &[u8] = &[
    0x0A, 0xAA, 0x4A, 0x72, 0x52, 0x32,
];

/// Load/Store instruction opcodes
const LOADSTORE_OPS: &[u8] = &[
    0xF9, 0xB9, 0x39, 0x29, 0x69, 0xF8, 0xB8, 0x78, 0x38, 0x28, 0x68,
    0x88, 0x48, 0xA8, 0xA9, 0xA4, 0xA0,
];

/// Branch instruction opcodes
const BRANCH_OPS: &[u8] = &[
    0x14, 0x34, 0x54, 0x17, 0x97, 0xD6, 0xD7,
];

/// Multiply/Divide instruction opcodes
const MULDIV_OPS: &[u8] = &[
    0x9B, 0x1B, 0x5B, 0xDB,
];

/// Extend/Shift/Bit instruction opcodes
const EXTEND_SHIFT_OPS: &[u8] = &[
    0x13, 0x53, 0x73, 0x33, 0x34, 0x14, 0x54, 0x74, 0x94, 0xB4, 0xD4,
    0x00, 0x08, 0x29, 0x49, 0x69, 0x0A, 0x4A, 0x5A, 0x2B, 0x6B, 0xAB, 0x4B,
];

/// System instruction opcodes
const SYSTEM_OPS: &[u8] = &[
    0xD4, 0xD5, 0x03, 0x6B, 0xEB, 0x1B,
];

/// Check if an opcode belongs to a category
fn opcode_in_category(op: u8, category_ops: &[u8]) -> bool {
    category_ops.contains(&op)
}

/// Arithmetic kernel shader source
const ARITHMETIC_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// Execute a single arithmetic instruction and return new PC
kernel void execute_arithmetic(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers_buf [[buffer(1)]],
    device uint64_t* pc_buf [[buffer(2)]],
    device uint32_t* inst_buf [[buffer(3)]],
    device uint64_t* pc_out [[buffer(4)]],
    device uint8_t* handled_out [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint lane_id = tid;

    // Load registers
    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers_buf[lane_id * 32 + i];
    }

    uint64_t pc = pc_buf[lane_id];
    uint32_t inst = inst_buf[lane_id];
    uint8_t op = (inst >> 24) & 0xFF;

    uint8_t rd = inst & 0x1F;
    uint8_t rn = (inst >> 5) & 0x1F;
    uint16_t imm12 = (inst >> 10) & 0xFFF;
    uint16_t imm16 = (inst >> 5) & 0xFFFF;
    uint8_t hw = (inst >> 21) & 0x3;
    int64_t rn_val = (rn == 31) ? 0 : regs[rn];

    bool handled = true;

    if (op == 0x91) {  // ADD imm 64-bit
        if (rd < 31) regs[rd] = rn_val + imm12;
        pc += 4;
    } else if (op == 0x11) {  // ADD imm 32-bit
        if (rd < 31) regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val + imm12);
        pc += 4;
    } else if (op == 0xD1) {  // SUB imm 64-bit
        if (rd < 31) regs[rd] = rn_val - imm12;
        pc += 4;
    } else if (op == 0x51) {  // SUB imm 32-bit
        if (rd < 31) regs[rd] = (int64_t)(int32_t)((uint32_t)rn_val - imm12);
        pc += 4;
    } else if (op == 0xD2) {  // MOVZ
        if (rd < 31) regs[rd] = (int64_t)((uint64_t)imm16 << (hw * 16));
        pc += 4;
    } else if (op == 0xF2) {  // MOVK
        if (rd < 31) {
            uint64_t mask = ~((uint64_t)0xFFFF << (hw * 16));
            regs[rd] = (regs[rd] & mask) | ((int64_t)imm16 << (hw * 16));
        }
        pc += 4;
    } else if (op == 0x92) {  // MOVN
        if (rd < 31) regs[rd] = (int64_t)(~((uint64_t)imm16 << (hw * 16)));
        pc += 4;
    } else if (op == 0x8B) {  // ADD reg 64-bit
        if (rd < 31) regs[rd] = rn_val + regs[(inst >> 16) & 0x1F];
        pc += 4;
    } else if (op == 0xCB) {  // SUB reg 64-bit
        if (rd < 31) regs[rd] = rn_val - regs[(inst >> 16) & 0x1F];
        pc += 4;
    } else if (op == 0x0B) {  // ADD extended
        uint8_t rm = (inst >> 16) & 0x1F;
        int64_t rm_val = (rm == 31) ? 0 : regs[rm];
        if (rd < 31) regs[rd] = rn_val + rm_val;
        pc += 4;
    } else {
        handled = false;
    }

    // Write back
    for (int i = 0; i < 32; i++) {
        registers_buf[lane_id * 32 + i] = regs[i];
    }
    pc_out[lane_id] = pc;
    handled_out[lane_id] = handled ? 1 : 0;
}
"##;

/// Logical kernel shader source
const LOGICAL_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// Execute a single logical instruction
kernel void execute_logical(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers_buf [[buffer(1)]],
    device uint64_t* pc_buf [[buffer(2)]],
    device uint32_t* inst_buf [[buffer(3)]],
    device uint64_t* pc_out [[buffer(4)]],
    device uint8_t* handled_out [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint lane_id = tid;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers_buf[lane_id * 32 + i];
    }

    uint64_t pc = pc_buf[lane_id];
    uint32_t inst = inst_buf[lane_id];
    uint8_t op = (inst >> 24) & 0xFF;

    uint8_t rd = inst & 0x1F;
    uint8_t rn = (inst >> 5) & 0x1F;
    int64_t rn_val = (rn == 31) ? 0 : regs[rn];

    bool handled = true;

    if (op == 0x0A) {  // AND register
        if (rd < 31) regs[rd] = rn_val & regs[(inst >> 16) & 0x1F];
        pc += 4;
    } else if (op == 0xAA) {  // ORR register / MOV
        if (rd < 31) regs[rd] = rn_val | regs[(inst >> 16) & 0x1F];
        pc += 4;
    } else if (op == 0x4A) {  // EOR register
        if (rd < 31) regs[rd] = rn_val ^ regs[(inst >> 16) & 0x1F];
        pc += 4;
    } else if (op == 0x72) {  // ORR immediate
        if (rd < 31) regs[rd] = rn_val | (int64_t)((inst >> 10) & 0xFFF);
        pc += 4;
    } else if (op == 0x52) {  // AND immediate
        if (rd < 31) regs[rd] = rn_val & (int64_t)((inst >> 10) & 0xFFF);
        pc += 4;
    } else if (op == 0x32) {  // EOR immediate
        if (rd < 31) regs[rd] = rn_val ^ (int64_t)((inst >> 10) & 0xFFF);
        pc += 4;
    } else {
        handled = false;
    }

    for (int i = 0; i < 32; i++) {
        registers_buf[lane_id * 32 + i] = regs[i];
    }
    pc_out[lane_id] = pc;
    handled_out[lane_id] = handled ? 1 : 0;
}
"##;

/// Load/Store kernel shader source (11 instructions)
const LOADSTORE_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// Execute a single load/store instruction
kernel void execute_loadstore(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers_buf [[buffer(1)]],
    device uint64_t* pc_buf [[buffer(2)]],
    device uint32_t* inst_buf [[buffer(3)]],
    device uint64_t* pc_out [[buffer(4)]],
    device uint8_t* handled_out [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint lane_id = tid;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers_buf[lane_id * 32 + i];
    }

    uint64_t pc = pc_buf[lane_id];
    uint32_t inst = inst_buf[lane_id];
    uint8_t op = (inst >> 24) & 0xFF;

    uint8_t rd = inst & 0x1F;
    uint8_t rn = (inst >> 5) & 0x1F;
    uint16_t imm12 = (inst >> 10) & 0xFFF;
    int64_t base = (rn == 31) ? 0 : regs[rn];
    bool handled = true;

    if (op == 0xF9) {  // LDR/STR 64-bit
        if ((inst & (1 << 22))) {
            if (rd < 31) regs[rd] = *((device int64_t*)(memory + base + (imm12 << 3)));
        } else {
            if (rd < 31) *((device int64_t*)(memory + base + (imm12 << 3))) = regs[rd];
        }
        pc += 4;
    } else if (op == 0xB9) {  // LDR/STR 32-bit
        if ((inst & (1 << 22))) {
            if (rd < 31) regs[rd] = (int64_t)(int32_t)(*((device uint32_t*)(memory + base + (imm12 << 2))));
        } else {
            if (rd < 31) *((device uint32_t*)(memory + base + (imm12 << 2))) = (uint32_t)regs[rd];
        }
        pc += 4;
    } else if (op == 0x39) {  // LDRB/STRB
        if ((inst & (1 << 22))) {
            if (rd < 31) regs[rd] = (int64_t)memory[base + imm12];
        } else {
            if (rd < 31) memory[base + imm12] = (uint8_t)regs[rd];
        }
        pc += 4;
    } else if (op == 0x29) {  // LDP/STP 64-bit
        int16_t offset = ((inst >> 15) & 0x1F);
        offset = (offset & 0x10) ? (offset | 0xFFE0) : offset;
        uint64_t addr = base + (offset * 8);
        uint8_t rt2 = (inst >> 10) & 0x1F;
        if ((inst & (1 << 22))) {
            if (rd < 31) regs[rd] = *((device int64_t*)(memory + addr));
            if (rt2 < 31) regs[rt2] = *((device int64_t*)(memory + addr + 8));
        } else {
            if (rd < 31) *((device int64_t*)(memory + addr)) = regs[rd];
            if (rt2 < 31) *((device int64_t*)(memory + addr + 8)) = regs[rt2];
        }
        pc += 4;
    } else if (op == 0x69) {  // LDP/STP 32-bit
        int16_t offset = ((inst >> 15) & 0x1F);
        offset = (offset & 0x10) ? (offset | 0xFFE0) : offset;
        uint64_t addr = base + (offset * 4);
        uint8_t rt2 = (inst >> 10) & 0x1F;
        if ((inst & (1 << 22))) {
            if (rd < 31) regs[rd] = (int64_t)(int32_t)(*((device uint32_t*)(memory + addr)));
            if (rt2 < 31) regs[rt2] = (int64_t)(int32_t)(*((device uint32_t*)(memory + addr + 4)));
        } else {
            if (rd < 31) *((device uint32_t*)(memory + addr)) = (uint32_t)regs[rd];
            if (rt2 < 31) *((device uint32_t*)(memory + addr + 4)) = (uint32_t)regs[rt2];
        }
        pc += 4;
    } else if (op == 0xF8) {  // LDUR/STUR 64-bit
        int16_t offset = (inst >> 12) & 0x1FF;
        if (offset & 0x100) offset |= 0xFE00;  // Sign-extend to 16 bits
        uint64_t addr = base + offset;
        if ((inst & (1 << 22))) {
            if (rd < 31) regs[rd] = *((device int64_t*)(memory + addr));
        } else {
            if (rd < 31) *((device int64_t*)(memory + addr)) = regs[rd];
        }
        pc += 4;
    } else if (op == 0xB8) {  // LDUR/STUR 32-bit
        int16_t offset = (inst >> 12) & 0x1FF;
        if (offset & 0x100) offset |= 0xFE00;  // Sign-extend to 16 bits
        uint64_t addr = base + offset;
        if ((inst & (1 << 22))) {
            if (rd < 31) regs[rd] = (int64_t)(int32_t)(*((device uint32_t*)(memory + addr)));
        } else {
            if (rd < 31) *((device uint32_t*)(memory + addr)) = (uint32_t)regs[rd];
        }
        pc += 4;
    } else {
        handled = false;
    }

    for (int i = 0; i < 32; i++) {
        registers_buf[lane_id * 32 + i] = regs[i];
    }
    pc_out[lane_id] = pc;
    handled_out[lane_id] = handled ? 1 : 0;
}
"##;

/// Branch kernel shader source (7 instructions)
const BRANCH_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// Execute a single branch instruction
kernel void execute_branch(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers_buf [[buffer(1)]],
    device uint64_t* pc_buf [[buffer(2)]],
    device uint32_t* inst_buf [[buffer(3)]],
    device uint64_t* pc_out [[buffer(4)]],
    device uint8_t* handled_out [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint lane_id = tid;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers_buf[lane_id * 32 + i];
    }

    uint64_t pc = pc_buf[lane_id];
    uint32_t inst = inst_buf[lane_id];
    uint8_t op = (inst >> 24) & 0xFF;

    bool handled = true;

    if (op == 0x14) {  // B
        int32_t offset = inst & 0x3FFFFFF;
        if (offset & 0x2000000) offset |= 0xFC000000;
        pc += offset * 4;
    } else if (op == 0x17) {  // B unconditional
        int32_t offset = inst & 0x3FFFFFF;
        if (offset & 0x2000000) offset |= 0xFC000000;
        pc += offset * 4;
    } else if (op == 0x54) {  // B.cond
        int32_t offset = ((inst >> 5) & 0x7FFFF);
        if (offset & 0x40000) offset |= 0xFFF80000;
        uint8_t cond = inst & 0xF;
        // Simplified: always take branch for now
        if (cond == 0 || cond == 14) {  // EQ or AL
            pc += offset * 4;
        } else {
            pc += 4;
        }
    } else if (op == 0x97) {  // BL
        int32_t offset = inst & 0x3FFFFFF;
        if (offset & 0x2000000) offset |= 0xFC000000;
        regs[30] = pc + 4;  // Save return address in LR
        pc += offset * 4;
    } else if (op == 0xD6) {  // RET
        pc = regs[30];  // Return to address in LR
    } else if (op == 0xD7) {  // RET alias
        pc = regs[30];
    } else {
        handled = false;
    }

    for (int i = 0; i < 32; i++) {
        registers_buf[lane_id * 32 + i] = regs[i];
    }
    pc_out[lane_id] = pc;
    handled_out[lane_id] = handled ? 1 : 0;
}
"##;

/// Multiply/Divide kernel shader source (4 instructions)
const MULDIV_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// Execute a single multiply/divide instruction
kernel void execute_muldiv(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers_buf [[buffer(1)]],
    device uint64_t* pc_buf [[buffer(2)]],
    device uint32_t* inst_buf [[buffer(3)]],
    device uint64_t* pc_out [[buffer(4)]],
    device uint8_t* handled_out [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint lane_id = tid;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers_buf[lane_id * 32 + i];
    }

    uint64_t pc = pc_buf[lane_id];
    uint32_t inst = inst_buf[lane_id];
    uint8_t op = (inst >> 24) & 0xFF;

    uint8_t rd = inst & 0x1F;
    uint8_t rn = (inst >> 5) & 0x1F;
    uint8_t rm = (inst >> 16) & 0x1F;
    uint8_t ra = (inst >> 10) & 0x1F;
    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
    int64_t ra_val = (ra == 31) ? 0 : regs[ra];
    bool handled = true;

    if (op == 0x9B) {  // MADD
        if (rd < 31) regs[rd] = rn_val * rm_val + ra_val;
        pc += 4;
    } else if (op == 0x1B) {  // MADD (alias)
        if (rd < 31) regs[rd] = rn_val * rm_val + ra_val;
        pc += 4;
    } else if (op == 0x5B) {  // MSUB
        if (rd < 31) regs[rd] = rn_val * rm_val - ra_val;
        pc += 4;
    } else if (op == 0xDB) {  // MSUB (alias)
        if (rd < 31) regs[rd] = rn_val * rm_val - ra_val;
        pc += 4;
    } else {
        handled = false;
    }

    for (int i = 0; i < 32; i++) {
        registers_buf[lane_id * 32 + i] = regs[i];
    }
    pc_out[lane_id] = pc;
    handled_out[lane_id] = handled ? 1 : 0;
}
"##;

/// Extend/Shift/Bit kernel shader source
const EXTEND_SHIFT_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// Execute extend/shift/bit instructions
kernel void execute_extend_shift(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers_buf [[buffer(1)]],
    device uint64_t* pc_buf [[buffer(2)]],
    device uint32_t* inst_buf [[buffer(3)]],
    device uint64_t* pc_out [[buffer(4)]],
    device uint8_t* handled_out [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint lane_id = tid;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers_buf[lane_id * 32 + i];
    }

    uint64_t pc = pc_buf[lane_id];
    uint32_t inst = inst_buf[lane_id];
    uint8_t op = (inst >> 24) & 0xFF;

    uint8_t rd = inst & 0x1F;
    uint8_t rn = (inst >> 5) & 0x1F;
    uint8_t rm = (inst >> 16) & 0x1F;
    int64_t rn_val = (rn == 31) ? 0 : regs[rn];
    int64_t rm_val = (rm == 31) ? 0 : regs[rm];
    bool handled = true;

    if (op == 0x13) {  // ORR shifted
        if (rd < 31) regs[rd] = rn_val | rm_val;
        pc += 4;
    } else if (op == 0x53) {  // EOR shifted
        if (rd < 31) regs[rd] = rn_val ^ rm_val;
        pc += 4;
    } else if (op == 0x73) {  // AND shifted
        if (rd < 31) regs[rd] = rn_val & rm_val;
        pc += 4;
    } else if (op == 0x33) {  // BIC shifted
        if (rd < 31) regs[rd] = rn_val & ~rm_val;
        pc += 4;
    } else if (op == 0x34) {  // SBFM
        if (rd < 31) regs[rd] = (int64_t)(int32_t)(rn_val | rm_val);
        pc += 4;
    } else if (op == 0x14) {  // BFM
        if (rd < 31) regs[rd] = (int64_t)(int32_t)(rn_val | rm_val);
        pc += 4;
    } else if (op == 0x54) {  // BFCM
        if (rd < 31) regs[rd] = rn_val | rm_val;
        pc += 4;
    } else if (op == 0x74) {  // SXTB (sign extend byte)
        if (rd < 31) regs[rd] = (int64_t)(int8_t)rn_val;
        pc += 4;
    } else if (op == 0x94) {  // SXTB (32 to 64)
        if (rd < 31) regs[rd] = (int64_t)(int32_t)(int8_t)(rn_val & 0xFF);
        pc += 4;
    } else if (op == 0xB4) {  // SXTH
        if (rd < 31) regs[rd] = (int64_t)(int16_t)rn_val;
        pc += 4;
    } else if (op == 0xD4) {  // SXTW
        if (rd < 31) regs[rd] = (int64_t)(int32_t)rn_val;
        pc += 4;
    } else if (op == 0x00) {  // ADRP
        uint64_t immhi = (inst & 0xFFFF) << 12;
        uint64_t immlo = ((inst >> 5) & 0xFFFF) << 2;
        if (rd < 31) regs[rd] = (pc & ~0xFFF) + immhi + immlo;
        pc += 4;
    } else {
        handled = false;
    }

    for (int i = 0; i < 32; i++) {
        registers_buf[lane_id * 32 + i] = regs[i];
    }
    pc_out[lane_id] = pc;
    handled_out[lane_id] = handled ? 1 : 0;
}
"##;

/// System kernel shader source (6 instructions)
const SYSTEM_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;

// Execute a single system instruction
kernel void execute_system(
    device uint8_t* memory [[buffer(0)]],
    device int64_t* registers_buf [[buffer(1)]],
    device uint64_t* pc_buf [[buffer(2)]],
    device uint32_t* inst_buf [[buffer(3)]],
    device uint64_t* pc_out [[buffer(4)]],
    device uint8_t* handled_out [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint lane_id = tid;

    int64_t regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = registers_buf[lane_id * 32 + i];
    }

    uint64_t pc = pc_buf[lane_id];
    uint32_t inst = inst_buf[lane_id];
    uint8_t op = (inst >> 24) & 0xFF;

    bool handled = true;

    if (op == 0xD4) {  // HLT
        // Stop execution
    } else if (op == 0xD5) {  // DCPS1
        // Debug hint - skip
        pc += 4;
    } else if (op == 0x03) {  // MRS/MSR (system register access)
        // Simplified: skip
        pc += 4;
    } else if (op == 0x6B) {  // SUBS with flags
        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint16_t imm12 = (inst >> 10) & 0xFFF;
        if (rd < 31) regs[rd] = (rn == 31) ? 0 : regs[rn] - imm12;
        pc += 4;
    } else if (op == 0xEB) {  // SUBS extended
        uint8_t rd = inst & 0x1F;
        uint8_t rn = (inst >> 5) & 0x1F;
        uint8_t rm = (inst >> 16) & 0x1F;
        if (rd < 31) regs[rd] = ((rn == 31) ? 0 : regs[rn]) - ((rm == 31) ? 0 : regs[rm]);
        pc += 4;
    } else if (op == 0x1B) {  // Other system instruction
        pc += 4;
    } else {
        handled = false;
    }

    for (int i = 0; i < 32; i++) {
        registers_buf[lane_id * 32 + i] = regs[i];
    }
    pc_out[lane_id] = pc;
    handled_out[lane_id] = handled ? 1 : 0;
}
"##;

/// Multi-kernel Metal CPU that dispatches to specialized kernels
pub struct MultiKernelMetalCPU {
    device: Retained<ProtocolObject<dyn MTLDevice>>,

    // Libraries (kept alive to prevent pipeline deallocation)
    _arithmetic_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    _logical_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    _loadstore_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    _branch_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    _muldiv_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    _extend_shift_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    _system_library: Retained<ProtocolObject<dyn MTLLibrary>>,

    // Pipeline states for each instruction category
    arithmetic_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    logical_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    loadstore_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    branch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    muldiv_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    extend_shift_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    system_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    num_lanes: u32,
    memory_size: u64,

    // Buffers
    memory_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    cycles_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    stop_reason_buf: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Temporary buffers for single-instruction execution
    inst_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_out_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    handled_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MultiKernelMetalCPU {
    pub fn new(num_lanes: u32, memory_size: u64) -> Result<Self, MetalError> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;

        println!("[MultiKernelMetalCPU] Using device: {:?}", device.name());

        // Compile arithmetic kernel
        let arithmetic_source = NSString::from_str(ARITHMETIC_SHADER);
        let arithmetic_library = device
            .newLibraryWithSource_options_error(&arithmetic_source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let arithmetic_fn_name = NSString::from_str("execute_arithmetic");
        let arithmetic_fn = arithmetic_library
            .newFunctionWithName(&arithmetic_fn_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("execute_arithmetic not found".to_string()))?;

        let arithmetic_pipeline = device
            .newComputePipelineStateWithFunction_error(&arithmetic_fn)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        println!("✅ Arithmetic pipeline compiled successfully!");

        // Compile logical kernel
        let logical_source = NSString::from_str(LOGICAL_SHADER);
        let logical_library = device
            .newLibraryWithSource_options_error(&logical_source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let logical_fn_name = NSString::from_str("execute_logical");
        let logical_fn = logical_library
            .newFunctionWithName(&logical_fn_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("execute_logical not found".to_string()))?;

        let logical_pipeline = device
            .newComputePipelineStateWithFunction_error(&logical_fn)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        println!("✅ Logical pipeline compiled successfully!");

        // Compile loadstore kernel
        let loadstore_source = NSString::from_str(LOADSTORE_SHADER);
        let loadstore_library = device
            .newLibraryWithSource_options_error(&loadstore_source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let loadstore_fn_name = NSString::from_str("execute_loadstore");
        let loadstore_fn = loadstore_library
            .newFunctionWithName(&loadstore_fn_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("execute_loadstore not found".to_string()))?;

        let loadstore_pipeline = device
            .newComputePipelineStateWithFunction_error(&loadstore_fn)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        println!("✅ Load/Store pipeline compiled successfully!");

        // Compile branch kernel
        let branch_source = NSString::from_str(BRANCH_SHADER);
        let branch_library = device
            .newLibraryWithSource_options_error(&branch_source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let branch_fn_name = NSString::from_str("execute_branch");
        let branch_fn = branch_library
            .newFunctionWithName(&branch_fn_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("execute_branch not found".to_string()))?;

        let branch_pipeline = device
            .newComputePipelineStateWithFunction_error(&branch_fn)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        println!("✅ Branch pipeline compiled successfully!");

        // Compile muldiv kernel
        let muldiv_source = NSString::from_str(MULDIV_SHADER);
        let muldiv_library = device
            .newLibraryWithSource_options_error(&muldiv_source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let muldiv_fn_name = NSString::from_str("execute_muldiv");
        let muldiv_fn = muldiv_library
            .newFunctionWithName(&muldiv_fn_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("execute_muldiv not found".to_string()))?;

        let muldiv_pipeline = device
            .newComputePipelineStateWithFunction_error(&muldiv_fn)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        println!("✅ Mul/Div pipeline compiled successfully!");

        // Compile extend_shift kernel
        let extend_shift_source = NSString::from_str(EXTEND_SHIFT_SHADER);
        let extend_shift_library = device
            .newLibraryWithSource_options_error(&extend_shift_source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let extend_shift_fn_name = NSString::from_str("execute_extend_shift");
        let extend_shift_fn = extend_shift_library
            .newFunctionWithName(&extend_shift_fn_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("execute_extend_shift not found".to_string()))?;

        let extend_shift_pipeline = device
            .newComputePipelineStateWithFunction_error(&extend_shift_fn)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        println!("✅ Extend/Shift pipeline compiled successfully!");

        // Compile system kernel
        let system_source = NSString::from_str(SYSTEM_SHADER);
        let system_library = device
            .newLibraryWithSource_options_error(&system_source, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let system_fn_name = NSString::from_str("execute_system");
        let system_fn = system_library
            .newFunctionWithName(&system_fn_name)
            .ok_or_else(|| MetalError::ShaderCompilationFailed("execute_system not found".to_string()))?;

        let system_pipeline = device
            .newComputePipelineStateWithFunction_error(&system_fn)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        println!("✅ System pipeline compiled successfully!");

        // Create buffers (same as before)
        let shared_options = MTLResourceOptions::StorageModeShared;
        let memory_buf = device.newBufferWithLength_options(memory_size as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let registers_buf = device.newBufferWithLength_options((num_lanes * 32 * 8) as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let pc_buf = device.newBufferWithLength_options((num_lanes * 8) as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let flags_buf = device.newBufferWithLength_options((num_lanes * 4 * 4) as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let cycles_buf = device.newBufferWithLength_options((num_lanes * 4) as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let stop_reason_buf = device.newBufferWithLength_options((num_lanes) as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        // Temporary buffers for single-instruction execution
        let inst_buf = device.newBufferWithLength_options((num_lanes * 4) as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let pc_out_buf = device.newBufferWithLength_options((num_lanes * 8) as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;
        let handled_buf = device.newBufferWithLength_options((num_lanes) as usize, shared_options)
            .ok_or(MetalError::BufferCreationFailed)?;

        Ok(Self {
            device,
            _arithmetic_library: arithmetic_library,
            _logical_library: logical_library,
            _loadstore_library: loadstore_library,
            _branch_library: branch_library,
            _muldiv_library: muldiv_library,
            _extend_shift_library: extend_shift_library,
            _system_library: system_library,
            arithmetic_pipeline,
            logical_pipeline,
            loadstore_pipeline,
            branch_pipeline,
            muldiv_pipeline,
            extend_shift_pipeline,
            system_pipeline,
            num_lanes,
            memory_size,
            memory_buf,
            registers_buf,
            pc_buf,
            flags_buf,
            cycles_buf,
            stop_reason_buf,
            inst_buf,
            pc_out_buf,
            handled_buf,
        })
    }

    /// Write a single 32-bit value to memory
    fn write_memory_u32(&self, address: u64, value: u32) -> Result<(), MetalError> {
        if address + 4 > self.memory_size {
            return Err(MetalError::ExecutionFailed);
        }
        unsafe {
            let mem_ptr = self.memory_buf.contents().as_ptr() as *mut u8;
            let bytes = value.to_le_bytes();
            for (i, &byte) in bytes.iter().enumerate() {
                *mem_ptr.add((address + i as u64) as usize) = byte;
            }
        }
        Ok(())
    }

    /// Write bytes to memory
    fn write_memory(&self, address: u64, data: Vec<u8>) -> Result<(), MetalError> {
        if address + data.len() as u64 > self.memory_size {
            return Err(MetalError::ExecutionFailed);
        }
        unsafe {
            let mem_ptr = self.memory_buf.contents().as_ptr() as *mut u8;
            for (i, &byte) in data.iter().enumerate() {
                *mem_ptr.add((address + i as u64) as usize) = byte;
            }
        }
        Ok(())
    }

    /// Read bytes from memory
    fn read_memory(&self, address: usize, length: usize) -> Result<Vec<u8>, MetalError> {
        if address + length > self.memory_size as usize {
            return Err(MetalError::ExecutionFailed);
        }
        unsafe {
            let mem_ptr = self.memory_buf.contents().as_ptr() as *const u8;
            let mut data = Vec::with_capacity(length);
            data.extend_from_slice(std::slice::from_raw_parts(mem_ptr.add(address), length));
            Ok(data)
        }
    }

    /// Set a register value for a specific lane
    fn set_register(&self, lane_id: u32, reg_id: u32, value: i64) -> Result<(), MetalError> {
        if lane_id >= self.num_lanes || reg_id >= 32 {
            return Err(MetalError::ExecutionFailed);
        }
        unsafe {
            let reg_ptr = self.registers_buf.contents().as_ptr() as *mut i64;
            *reg_ptr.add((lane_id * 32 + reg_id) as usize) = value;
        }
        Ok(())
    }

    /// Get a register value for a specific lane
    fn get_register(&self, lane_id: u32, reg_id: u32) -> Result<i64, MetalError> {
        if lane_id >= self.num_lanes || reg_id >= 32 {
            return Err(MetalError::ExecutionFailed);
        }
        unsafe {
            let reg_ptr = self.registers_buf.contents().as_ptr() as *const i64;
            Ok(*reg_ptr.add((lane_id * 32 + reg_id) as usize))
        }
    }

    /// Set PC for a specific lane
    fn set_pc(&self, lane_id: u32, pc: u64) -> Result<(), MetalError> {
        if lane_id >= self.num_lanes {
            return Err(MetalError::ExecutionFailed);
        }
        unsafe {
            let pc_ptr = self.pc_buf.contents().as_ptr() as *mut u64;
            *pc_ptr.add(lane_id as usize) = pc;
        }
        Ok(())
    }

    /// Get PC for a specific lane
    fn get_pc(&self, lane_id: u32) -> Result<u64, MetalError> {
        if lane_id >= self.num_lanes {
            return Err(MetalError::ExecutionFailed);
        }
        unsafe {
            let pc_ptr = self.pc_buf.contents().as_ptr() as *const u64;
            Ok(*pc_ptr.add(lane_id as usize))
        }
    }

    /// Execute instructions by dispatching to appropriate kernels
    pub fn execute(&self, max_cycles: u64) -> Result<ExecutionResult, MetalError> {
        let start = Instant::now();

        // Create command queue
        let command_queue = self
            .device
            .newCommandQueue()
            .ok_or(MetalError::ExecutionFailed)?;

        let mut total_cycles: u64 = 0;
        let mut any_stopped = false;

        // Main execution loop - CPU controlled, GPU executes instructions
        for cycle in 0..max_cycles {
            // Read current PCs from GPU memory
            let pcs = unsafe {
                let ptr = self.pc_buf.contents().as_ptr() as *const u64;
                (0..self.num_lanes)
                    .map(|i| *ptr.add(i as usize))
                    .collect::<Vec<_>>()
            };

            // Read instruction for each lane
            let mut instructions = vec![0u32; self.num_lanes as usize];
            for lane in 0..self.num_lanes {
                let pc = pcs[lane as usize] as usize;
                if pc + 4 <= self.memory_size as usize {
                    unsafe {
                        let mem_ptr = self.memory_buf.contents().as_ptr() as *const u8;
                        let inst_bytes = [
                            *mem_ptr.add(pc),
                            *mem_ptr.add(pc + 1),
                            *mem_ptr.add(pc + 2),
                            *mem_ptr.add(pc + 3),
                        ];
                        instructions[lane as usize] = u32::from_le_bytes(inst_bytes);
                    }
                }
            }

            // Copy instructions to inst_buf
            unsafe {
                let inst_ptr = self.inst_buf.contents().as_ptr() as *mut u32;
                for (i, &inst) in instructions.iter().enumerate() {
                    *inst_ptr.add(i) = inst;
                }
            }

            // Create command buffer
            let command_buffer = command_queue
                .commandBuffer()
                .ok_or(MetalError::ExecutionFailed)?;

            // Create compute encoder
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::ExecutionFailed)?;

            // Dispatch based on first instruction's opcode (simplified - all lanes execute same kernel)
            // In a more sophisticated version, we could group lanes by instruction type
            let first_inst = instructions[0];
            let opcode = (first_inst >> 24) & 0xFF;

            // Select appropriate kernel based on opcode
            let pipeline = match opcode {
                // Arithmetic ops: 0x91, 0x11, 0xD1, 0x51, 0xD2, 0xF2, 0x92, 0x8B, 0xCB, 0x0B
                0x91 | 0x11 | 0xD1 | 0x51 | 0xD2 | 0xF2 | 0x92 | 0x8B | 0xCB | 0x0B => {
                    &self.arithmetic_pipeline
                }
                // Logical ops: 0x0A, 0xAA, 0x4A, 0x72, 0x52, 0x32
                0x0A | 0xAA | 0x4A | 0x72 | 0x52 | 0x32 => &self.logical_pipeline,
                // Load/Store ops: 0xF9, 0xB8, 0x28, 0xA8, 0x78, 0x38, 0x3C, 0xBC, 0x7C, 0xFC, 0x00
                0xF9 | 0xB8 | 0x28 | 0xA8 | 0x78 | 0x38 | 0x3C | 0xBC | 0x7C | 0xFC | 0x00 => {
                    &self.loadstore_pipeline
                }
                // Branch ops: 0x14, 0x17, 0x54, 0x97, 0xD6, 0xD7
                0x14 | 0x17 | 0x54 | 0x97 | 0xD6 | 0xD7 => &self.branch_pipeline,
                // Mul/Div ops: 0x9B, 0x1B, 0x5B, 0xDB
                0x9B | 0x1B | 0x5B | 0xDB => &self.muldiv_pipeline,
                // Extend/Shift ops: 0x72, 0x52, 0x32, 0x13, 0x93, 0xB4, 0x94, 0x74, 0x34, 0x10
                0x13 | 0x93 | 0xB4 | 0x94 | 0x74 | 0x34 | 0x10 => &self.extend_shift_pipeline,
                // System ops: 0xD4, 0x0A, 0xD2, 0xEB
                0xD4 | 0x0A | 0xD2 | 0xEB => &self.system_pipeline,
                // Unknown opcode - try arithmetic as fallback
                _ => &self.arithmetic_pipeline,
            };

            // Set pipeline and buffers
            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.memory_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.registers_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.pc_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&self.inst_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&self.pc_out_buf), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&self.handled_buf), 0, 5);
            }

            // Dispatch threads
            let threads_per_grid = MTLSize {
                width: self.num_lanes as usize,
                height: 1,
                depth: 1,
            };
            let threads_per_threadgroup = MTLSize {
                width: self.num_lanes.min(256) as usize,
                height: 1,
                depth: 1,
            };

            unsafe {
                encoder.dispatchThreads_threadsPerThreadgroup(
                    threads_per_grid,
                    threads_per_threadgroup,
                );
            }

            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            // Read back updated PCs and handled flags
            let pc_out = unsafe {
                let ptr = self.pc_out_buf.contents().as_ptr() as *const u64;
                (0..self.num_lanes)
                    .map(|i| *ptr.add(i as usize))
                    .collect::<Vec<_>>()
            };

            let handled = unsafe {
                let ptr = self.handled_buf.contents().as_ptr() as *const u8;
                (0..self.num_lanes)
                    .map(|i| *ptr.add(i as usize))
                    .collect::<Vec<_>>()
            };

            // Update PCs from pc_out
            unsafe {
                let pc_ptr = self.pc_buf.contents().as_ptr() as *mut u64;
                for (i, &pc) in pc_out.iter().enumerate() {
                    *pc_ptr.add(i) = pc;
                }
            }

            // Check if any lane handled an instruction
            let any_handled = handled.iter().any(|&h| h != 0);
            if any_handled {
                total_cycles += 1;
            }

            // Check for halt/stop conditions
            let stop_reasons = unsafe {
                let ptr = self.stop_reason_buf.contents().as_ptr() as *const u8;
                (0..self.num_lanes)
                    .map(|i| *ptr.add(i as usize))
                    .collect::<Vec<_>>()
            };

            if stop_reasons.iter().any(|&r| r != 0) {
                any_stopped = true;
                break;
            }

            // Safety: prevent infinite loops
            if cycle > 10000 && total_cycles == 0 {
                break;
            }
        }

        let elapsed = start.elapsed();

        // Read final PC
        let final_pc = unsafe {
            let ptr = self.pc_buf.contents().as_ptr() as *const u64;
            *ptr
        };

        Ok(ExecutionResult {
            cycles: total_cycles as u32,
            stop_reason: if any_stopped { 1 } else { 0 },
            final_pc,
        })
    }
}

#[pyclass(unsendable)]
pub struct PyMultiKernelMetalCPU {
    inner: MultiKernelMetalCPU,
}

#[pymethods]
impl PyMultiKernelMetalCPU {
    #[new]
    fn new(num_lanes: u32, memory_size: u64) -> PyResult<Self> {
        MultiKernelMetalCPU::new(num_lanes, memory_size)
            .map(|inner| Self { inner })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn execute(&self, max_cycles: u64) -> PyResult<ExecutionResult> {
        self.inner.execute(max_cycles)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Write a single 32-bit value to memory
    fn write_memory_u32(&mut self, address: u64, value: u32) -> PyResult<()> {
        self.inner.write_memory_u32(address, value)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Write bytes to memory
    fn write_memory(&mut self, address: u64, data: Vec<u8>) -> PyResult<()> {
        self.inner.write_memory(address, data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Read bytes from memory
    fn read_memory(&self, address: u64, length: u64) -> PyResult<Vec<u8>> {
        self.inner.read_memory(address as usize, length as usize)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Set a register value for a specific lane
    fn set_register(&mut self, lane_id: u32, reg_id: u32, value: i64) -> PyResult<()> {
        self.inner.set_register(lane_id, reg_id, value)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get a register value for a specific lane
    fn get_register(&self, lane_id: u32, reg_id: u32) -> PyResult<i64> {
        self.inner.get_register(lane_id, reg_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Set PC for a specific lane
    fn set_pc(&mut self, lane_id: u32, pc: u64) -> PyResult<()> {
        self.inner.set_pc(lane_id, pc)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get PC for a specific lane
    fn get_pc(&self, lane_id: u32) -> PyResult<u64> {
        self.inner.get_pc(lane_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get number of lanes
    fn get_num_lanes(&self) -> u32 {
        self.inner.num_lanes
    }
}

/// Register multi-kernel classes with the Python module
pub fn register_multi_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiKernelMetalCPU>()?;
    Ok(())
}
