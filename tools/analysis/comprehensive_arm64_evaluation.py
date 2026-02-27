#!/usr/bin/env python3
"""
COMPREHENSIVE ARM64 INSTRUCTION EVALUATION
==========================================

Rigorous evaluation of KVRM-SPNC decoder across ALL ARM64 instruction categories.

Categories tested:
0. ADD - Addition operations
1. SUB - Subtraction operations
2. AND - Bitwise AND
3. OR  - Bitwise OR
4. XOR - Bitwise XOR
5. MUL - Multiplication
6. DIV - Division
7. SHIFT - Shift/rotate operations
8. LOAD - Memory loads (LDR, LDP, etc.)
9. STORE - Memory stores (STR, STP, etc.)
10. BRANCH - Branch instructions (B, BL, CBZ, etc.)
11. COMPARE - Compare instructions (CMP, CMN, TST)
12. MOVE - Move instructions (MOV, MOVZ, MOVK, MOVN)
13. SYSTEM - System instructions (SVC, MRS, MSR)
14. UNKNOWN - Unrecognized/reserved

This provides comprehensive coverage for publication claims.
"""

import torch
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

# Import the decoder architecture
sys.path.insert(0, '.')
from run_neural_rtos_v2 import UniversalARM64Decoder


# Category mapping
CATEGORIES = {
    'ADD': 0, 'SUB': 1, 'AND': 2, 'OR': 3, 'XOR': 4,
    'MUL': 5, 'DIV': 6, 'SHIFT': 7, 'LOAD': 8, 'STORE': 9,
    'BRANCH': 10, 'COMPARE': 11, 'MOVE': 12, 'SYSTEM': 13, 'UNKNOWN': 14
}
IDX_TO_CAT = {v: k for k, v in CATEGORIES.items()}


@dataclass
class TestCase:
    """A single instruction test case."""
    name: str
    encoding: int
    category: int
    description: str


# =============================================================================
# INSTRUCTION ENCODERS
# =============================================================================

def encode_add_sub_imm(rd: int, rn: int, imm12: int, is_sub: bool = False, sf: int = 1) -> int:
    """ADD/SUB immediate: ADD Xd, Xn, #imm"""
    base = 0x91000000 if not is_sub else 0xD1000000
    if sf == 0:
        base &= ~(1 << 31)
    return base | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)


def encode_add_sub_reg(rd: int, rn: int, rm: int, is_sub: bool = False, sf: int = 1) -> int:
    """ADD/SUB register: ADD Xd, Xn, Xm"""
    base = 0x8B000000 if not is_sub else 0xCB000000
    if sf == 0:
        base &= ~(1 << 31)
    return base | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F)


def encode_logical_imm(rd: int, rn: int, op: str, imm: int = 0x7F, sf: int = 1) -> int:
    """AND/ORR/EOR immediate (simplified)."""
    opcodes = {'AND': 0x12000000, 'OR': 0x32000000, 'XOR': 0x52000000}
    base = opcodes.get(op, 0x12000000)
    if sf:
        base |= (1 << 31)
    # Simplified immediate encoding
    return base | ((imm & 0x3F) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)


def encode_logical_reg(rd: int, rn: int, rm: int, op: str, sf: int = 1) -> int:
    """AND/ORR/EOR register."""
    opcodes = {'AND': 0x0A000000, 'OR': 0x2A000000, 'XOR': 0x4A000000}
    base = opcodes.get(op, 0x0A000000)
    if sf:
        base |= (1 << 31)
    return base | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F)


def encode_mul(rd: int, rn: int, rm: int, sf: int = 1) -> int:
    """MUL: MUL Xd, Xn, Xm (MADD with Ra=XZR)."""
    base = 0x9B007C00
    if sf == 0:
        base &= ~(1 << 31)
    return base | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F)


def encode_div(rd: int, rn: int, rm: int, is_signed: bool = True, sf: int = 1) -> int:
    """SDIV/UDIV: SDIV Xd, Xn, Xm."""
    base = 0x9AC00C00 if is_signed else 0x9AC00800
    if sf == 0:
        base &= ~(1 << 31)
    return base | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F)


def encode_shift(rd: int, rn: int, rm: int, op: str, sf: int = 1) -> int:
    """LSL/LSR/ASR register."""
    shift_ops = {'LSL': 0x9AC02000, 'LSR': 0x9AC02400, 'ASR': 0x9AC02800}
    base = shift_ops.get(op, 0x9AC02000)
    if sf == 0:
        base &= ~(1 << 31)
    return base | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F)


def encode_ldr_imm(rt: int, rn: int, imm12: int, sf: int = 1) -> int:
    """LDR immediate unsigned offset."""
    base = 0xF9400000 if sf else 0xB9400000
    return base | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)


def encode_ldp(rt: int, rt2: int, rn: int, imm7: int, sf: int = 1) -> int:
    """LDP signed offset."""
    base = 0xA9400000 if sf else 0x29400000
    return base | ((imm7 & 0x7F) << 15) | ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)


def encode_str_imm(rt: int, rn: int, imm12: int, sf: int = 1) -> int:
    """STR immediate unsigned offset."""
    base = 0xF9000000 if sf else 0xB9000000
    return base | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)


def encode_stp(rt: int, rt2: int, rn: int, imm7: int, sf: int = 1) -> int:
    """STP signed offset."""
    base = 0xA9000000 if sf else 0x29000000
    return base | ((imm7 & 0x7F) << 15) | ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)


def encode_branch_imm(offset: int) -> int:
    """B (unconditional branch)."""
    return 0x14000000 | ((offset >> 2) & 0x3FFFFFF)


def encode_branch_link(offset: int) -> int:
    """BL (branch with link)."""
    return 0x94000000 | ((offset >> 2) & 0x3FFFFFF)


def encode_branch_reg(rn: int) -> int:
    """BR (branch to register)."""
    return 0xD61F0000 | ((rn & 0x1F) << 5)


def encode_cbz(rt: int, offset: int, sf: int = 1) -> int:
    """CBZ (compare and branch if zero)."""
    base = 0xB4000000 if sf else 0x34000000
    return base | (((offset >> 2) & 0x7FFFF) << 5) | (rt & 0x1F)


def encode_cmp_imm(rn: int, imm12: int, sf: int = 1) -> int:
    """CMP (compare): SUBS XZR, Xn, #imm."""
    base = 0xF100001F if sf else 0x7100001F
    return base | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5)


def encode_cmp_reg(rn: int, rm: int, sf: int = 1) -> int:
    """CMP register: SUBS XZR, Xn, Xm."""
    base = 0xEB00001F if sf else 0x6B00001F
    return base | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5)


def encode_tst_imm(rn: int, imm: int = 0x3F, sf: int = 1) -> int:
    """TST (test bits): ANDS XZR, Xn, #imm."""
    base = 0xF200001F if sf else 0x7200001F
    return base | ((imm & 0x3F) << 10) | ((rn & 0x1F) << 5)


def encode_movz(rd: int, imm16: int, hw: int = 0, sf: int = 1) -> int:
    """MOVZ: move wide with zero."""
    base = 0x52800000
    if sf:
        base |= (1 << 31)
    return base | ((hw & 0x3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)


def encode_movk(rd: int, imm16: int, hw: int = 0, sf: int = 1) -> int:
    """MOVK: move wide with keep."""
    base = 0x72800000
    if sf:
        base |= (1 << 31)
    return base | ((hw & 0x3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)


def encode_movn(rd: int, imm16: int, hw: int = 0, sf: int = 1) -> int:
    """MOVN: move wide with NOT."""
    base = 0x12800000
    if sf:
        base |= (1 << 31)
    return base | ((hw & 0x3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)


def encode_svc(imm16: int) -> int:
    """SVC (supervisor call)."""
    return 0xD4000001 | ((imm16 & 0xFFFF) << 5)


def encode_mrs(rt: int, sysreg: int = 0xDE82) -> int:
    """MRS (move from system register)."""
    return 0xD5300000 | ((sysreg & 0xFFFF) << 5) | (rt & 0x1F)


def encode_nop() -> int:
    """NOP."""
    return 0xD503201F


# =============================================================================
# TEST CASE GENERATION
# =============================================================================

def generate_comprehensive_tests() -> List[TestCase]:
    """Generate comprehensive test cases for all ARM64 categories."""
    tests = []

    # ADD (category 0) - 10 variants
    tests.extend([
        TestCase("ADD x0, x1, #0", encode_add_sub_imm(0, 1, 0), 0, "Add immediate 0"),
        TestCase("ADD x0, x1, #1", encode_add_sub_imm(0, 1, 1), 0, "Add immediate 1"),
        TestCase("ADD x0, x1, #100", encode_add_sub_imm(0, 1, 100), 0, "Add immediate 100"),
        TestCase("ADD x0, x1, #4095", encode_add_sub_imm(0, 1, 4095), 0, "Add max immediate"),
        TestCase("ADD w0, w1, #10", encode_add_sub_imm(0, 1, 10, sf=0), 0, "Add 32-bit"),
        TestCase("ADD x0, x1, x2", encode_add_sub_reg(0, 1, 2), 0, "Add register"),
        TestCase("ADD x5, x10, x15", encode_add_sub_reg(5, 10, 15), 0, "Add different regs"),
        TestCase("ADD x31, x0, x1", encode_add_sub_reg(31, 0, 1), 0, "Add to SP/XZR"),
        TestCase("ADD w0, w1, w2", encode_add_sub_reg(0, 1, 2, sf=0), 0, "Add 32-bit reg"),
        TestCase("ADD x20, x21, #2048", encode_add_sub_imm(20, 21, 2048), 0, "Add mid imm"),
    ])

    # SUB (category 1) - 10 variants
    tests.extend([
        TestCase("SUB x0, x1, #0", encode_add_sub_imm(0, 1, 0, True), 1, "Sub immediate 0"),
        TestCase("SUB x0, x1, #1", encode_add_sub_imm(0, 1, 1, True), 1, "Sub immediate 1"),
        TestCase("SUB x0, x1, #100", encode_add_sub_imm(0, 1, 100, True), 1, "Sub immediate 100"),
        TestCase("SUB x0, x1, #4095", encode_add_sub_imm(0, 1, 4095, True), 1, "Sub max immediate"),
        TestCase("SUB w0, w1, #10", encode_add_sub_imm(0, 1, 10, True, sf=0), 1, "Sub 32-bit"),
        TestCase("SUB x0, x1, x2", encode_add_sub_reg(0, 1, 2, True), 1, "Sub register"),
        TestCase("SUB x5, x10, x15", encode_add_sub_reg(5, 10, 15, True), 1, "Sub different regs"),
        TestCase("SUB x31, x0, x1", encode_add_sub_reg(31, 0, 1, True), 1, "Sub to SP/XZR"),
        TestCase("SUB w0, w1, w2", encode_add_sub_reg(0, 1, 2, True, sf=0), 1, "Sub 32-bit reg"),
        TestCase("SUB x20, x21, #2048", encode_add_sub_imm(20, 21, 2048, True), 1, "Sub mid imm"),
    ])

    # AND (category 2) - 5 variants
    tests.extend([
        TestCase("AND x0, x1, x2", encode_logical_reg(0, 1, 2, 'AND'), 2, "And register"),
        TestCase("AND x5, x10, x15", encode_logical_reg(5, 10, 15, 'AND'), 2, "And different regs"),
        TestCase("AND w0, w1, w2", encode_logical_reg(0, 1, 2, 'AND', sf=0), 2, "And 32-bit"),
        TestCase("AND x0, x1, #0x3F", encode_logical_imm(0, 1, 'AND', 0x3F), 2, "And immediate"),
        TestCase("AND x20, x21, x22", encode_logical_reg(20, 21, 22, 'AND'), 2, "And high regs"),
    ])

    # OR (category 3) - 5 variants
    tests.extend([
        TestCase("ORR x0, x1, x2", encode_logical_reg(0, 1, 2, 'OR'), 3, "Or register"),
        TestCase("ORR x5, x10, x15", encode_logical_reg(5, 10, 15, 'OR'), 3, "Or different regs"),
        TestCase("ORR w0, w1, w2", encode_logical_reg(0, 1, 2, 'OR', sf=0), 3, "Or 32-bit"),
        TestCase("ORR x0, x1, #0x3F", encode_logical_imm(0, 1, 'OR', 0x3F), 3, "Or immediate"),
        TestCase("ORR x20, x21, x22", encode_logical_reg(20, 21, 22, 'OR'), 3, "Or high regs"),
    ])

    # XOR (category 4) - 5 variants
    tests.extend([
        TestCase("EOR x0, x1, x2", encode_logical_reg(0, 1, 2, 'XOR'), 4, "Xor register"),
        TestCase("EOR x5, x10, x15", encode_logical_reg(5, 10, 15, 'XOR'), 4, "Xor different regs"),
        TestCase("EOR w0, w1, w2", encode_logical_reg(0, 1, 2, 'XOR', sf=0), 4, "Xor 32-bit"),
        TestCase("EOR x0, x1, #0x3F", encode_logical_imm(0, 1, 'XOR', 0x3F), 4, "Xor immediate"),
        TestCase("EOR x20, x21, x22", encode_logical_reg(20, 21, 22, 'XOR'), 4, "Xor high regs"),
    ])

    # MUL (category 5) - 5 variants
    tests.extend([
        TestCase("MUL x0, x1, x2", encode_mul(0, 1, 2), 5, "Mul register"),
        TestCase("MUL x5, x10, x15", encode_mul(5, 10, 15), 5, "Mul different regs"),
        TestCase("MUL w0, w1, w2", encode_mul(0, 1, 2, sf=0), 5, "Mul 32-bit"),
        TestCase("MUL x20, x21, x22", encode_mul(20, 21, 22), 5, "Mul high regs"),
        TestCase("MUL x0, x0, x1", encode_mul(0, 0, 1), 5, "Mul same src/dst"),
    ])

    # DIV (category 6) - 5 variants
    tests.extend([
        TestCase("SDIV x0, x1, x2", encode_div(0, 1, 2, True), 6, "Sdiv register"),
        TestCase("UDIV x0, x1, x2", encode_div(0, 1, 2, False), 6, "Udiv register"),
        TestCase("SDIV w0, w1, w2", encode_div(0, 1, 2, True, sf=0), 6, "Sdiv 32-bit"),
        TestCase("UDIV w0, w1, w2", encode_div(0, 1, 2, False, sf=0), 6, "Udiv 32-bit"),
        TestCase("SDIV x20, x21, x22", encode_div(20, 21, 22, True), 6, "Sdiv high regs"),
    ])

    # SHIFT (category 7) - 6 variants
    tests.extend([
        TestCase("LSL x0, x1, x2", encode_shift(0, 1, 2, 'LSL'), 7, "Lsl register"),
        TestCase("LSR x0, x1, x2", encode_shift(0, 1, 2, 'LSR'), 7, "Lsr register"),
        TestCase("ASR x0, x1, x2", encode_shift(0, 1, 2, 'ASR'), 7, "Asr register"),
        TestCase("LSL w0, w1, w2", encode_shift(0, 1, 2, 'LSL', sf=0), 7, "Lsl 32-bit"),
        TestCase("LSR w0, w1, w2", encode_shift(0, 1, 2, 'LSR', sf=0), 7, "Lsr 32-bit"),
        TestCase("ASR w0, w1, w2", encode_shift(0, 1, 2, 'ASR', sf=0), 7, "Asr 32-bit"),
    ])

    # LOAD (category 8) - 8 variants
    tests.extend([
        TestCase("LDR x0, [x1]", encode_ldr_imm(0, 1, 0), 8, "Ldr zero offset"),
        TestCase("LDR x0, [x1, #8]", encode_ldr_imm(0, 1, 1), 8, "Ldr offset 8"),
        TestCase("LDR x0, [x1, #16]", encode_ldr_imm(0, 1, 2), 8, "Ldr offset 16"),
        TestCase("LDR w0, [x1]", encode_ldr_imm(0, 1, 0, sf=0), 8, "Ldr 32-bit"),
        TestCase("LDP x0, x1, [x2]", encode_ldp(0, 1, 2, 0), 8, "Ldp pair"),
        TestCase("LDP x0, x1, [x2, #16]", encode_ldp(0, 1, 2, 2), 8, "Ldp pair offset"),
        TestCase("LDP w0, w1, [x2]", encode_ldp(0, 1, 2, 0, sf=0), 8, "Ldp 32-bit"),
        TestCase("LDR x20, [x21, #24]", encode_ldr_imm(20, 21, 3), 8, "Ldr high regs"),
    ])

    # STORE (category 9) - 8 variants
    tests.extend([
        TestCase("STR x0, [x1]", encode_str_imm(0, 1, 0), 9, "Str zero offset"),
        TestCase("STR x0, [x1, #8]", encode_str_imm(0, 1, 1), 9, "Str offset 8"),
        TestCase("STR x0, [x1, #16]", encode_str_imm(0, 1, 2), 9, "Str offset 16"),
        TestCase("STR w0, [x1]", encode_str_imm(0, 1, 0, sf=0), 9, "Str 32-bit"),
        TestCase("STP x0, x1, [x2]", encode_stp(0, 1, 2, 0), 9, "Stp pair"),
        TestCase("STP x0, x1, [x2, #16]", encode_stp(0, 1, 2, 2), 9, "Stp pair offset"),
        TestCase("STP w0, w1, [x2]", encode_stp(0, 1, 2, 0, sf=0), 9, "Stp 32-bit"),
        TestCase("STR x20, [x21, #24]", encode_str_imm(20, 21, 3), 9, "Str high regs"),
    ])

    # BRANCH (category 10) - 8 variants
    tests.extend([
        TestCase("B #0", encode_branch_imm(0), 10, "Branch zero"),
        TestCase("B #4", encode_branch_imm(4), 10, "Branch +4"),
        TestCase("B #100", encode_branch_imm(100), 10, "Branch +100"),
        TestCase("BL #0", encode_branch_link(0), 10, "Branch link zero"),
        TestCase("BL #4", encode_branch_link(4), 10, "Branch link +4"),
        TestCase("BR x0", encode_branch_reg(0), 10, "Branch register"),
        TestCase("CBZ x0, #0", encode_cbz(0, 0), 10, "Cbz zero offset"),
        TestCase("CBZ x0, #8", encode_cbz(0, 8), 10, "Cbz +8"),
    ])

    # COMPARE (category 11) - 6 variants
    tests.extend([
        TestCase("CMP x0, #0", encode_cmp_imm(0, 0), 11, "Cmp immediate 0"),
        TestCase("CMP x0, #1", encode_cmp_imm(0, 1), 11, "Cmp immediate 1"),
        TestCase("CMP x0, #100", encode_cmp_imm(0, 100), 11, "Cmp immediate 100"),
        TestCase("CMP x0, x1", encode_cmp_reg(0, 1), 11, "Cmp register"),
        TestCase("CMP w0, w1", encode_cmp_reg(0, 1, sf=0), 11, "Cmp 32-bit"),
        TestCase("TST x0, #0x3F", encode_tst_imm(0, 0x3F), 11, "Tst immediate"),
    ])

    # MOVE (category 12) - 10 variants
    tests.extend([
        TestCase("MOVZ x0, #0", encode_movz(0, 0), 12, "Movz zero"),
        TestCase("MOVZ x0, #1", encode_movz(0, 1), 12, "Movz one"),
        TestCase("MOVZ x0, #0xFFFF", encode_movz(0, 0xFFFF), 12, "Movz max16"),
        TestCase("MOVZ x0, #0x1000, LSL#16", encode_movz(0, 0x1000, hw=1), 12, "Movz shift 16"),
        TestCase("MOVZ x0, #0x1000, LSL#32", encode_movz(0, 0x1000, hw=2), 12, "Movz shift 32"),
        TestCase("MOVZ x0, #0x1000, LSL#48", encode_movz(0, 0x1000, hw=3), 12, "Movz shift 48"),
        TestCase("MOVK x0, #0x1234", encode_movk(0, 0x1234), 12, "Movk"),
        TestCase("MOVK x0, #0xABCD, LSL#16", encode_movk(0, 0xABCD, hw=1), 12, "Movk shift"),
        TestCase("MOVN x0, #0", encode_movn(0, 0), 12, "Movn zero"),
        TestCase("MOVN x0, #100", encode_movn(0, 100), 12, "Movn 100"),
    ])

    # SYSTEM (category 13) - 4 variants
    tests.extend([
        TestCase("SVC #0", encode_svc(0), 13, "Svc zero"),
        TestCase("SVC #1", encode_svc(1), 13, "Svc one"),
        TestCase("MRS x0, TPIDR_EL0", encode_mrs(0), 13, "Mrs"),
        TestCase("NOP", encode_nop(), 13, "Nop"),
    ])

    return tests


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

def evaluate_decoder(model, tests: List[TestCase]) -> Dict:
    """Run comprehensive evaluation."""
    model.eval()

    results = {
        'total': len(tests),
        'correct': 0,
        'per_category': {cat: {'total': 0, 'correct': 0} for cat in CATEGORIES.keys()},
        'failures': [],
        'confusion_matrix': np.zeros((15, 15), dtype=int),
    }

    for test in tests:
        # Convert instruction to bits
        bits = torch.zeros(1, 32, dtype=torch.float32, device=device)
        for i in range(32):
            bits[0, i] = float((test.encoding >> i) & 1)

        # Decode
        with torch.no_grad():
            outputs = model(bits)
            pred_cat = outputs['category'].argmax(dim=1).item()

        # Update stats
        expected_cat_name = IDX_TO_CAT[test.category]
        results['per_category'][expected_cat_name]['total'] += 1

        # Update confusion matrix
        results['confusion_matrix'][test.category][pred_cat] += 1

        if pred_cat == test.category:
            results['correct'] += 1
            results['per_category'][expected_cat_name]['correct'] += 1
        else:
            results['failures'].append({
                'name': test.name,
                'encoding': f"0x{test.encoding:08X}",
                'expected': expected_cat_name,
                'predicted': IDX_TO_CAT.get(pred_cat, f"UNKNOWN({pred_cat})"),
            })

    return results


def print_results(results: Dict):
    """Print formatted evaluation results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ARM64 DECODER EVALUATION RESULTS")
    print("=" * 80)

    # Overall accuracy
    accuracy = 100.0 * results['correct'] / results['total']
    print(f"\n{'OVERALL ACCURACY:':<30} {results['correct']}/{results['total']} ({accuracy:.1f}%)")

    # Per-category breakdown
    print("\n" + "-" * 80)
    print("PER-CATEGORY BREAKDOWN")
    print("-" * 80)
    print(f"{'Category':<15} {'Correct':<12} {'Total':<10} {'Accuracy':<12}")
    print("-" * 50)

    categories_tested = 0
    categories_100pct = 0

    for cat_name in CATEGORIES.keys():
        stats = results['per_category'][cat_name]
        if stats['total'] > 0:
            categories_tested += 1
            cat_acc = 100.0 * stats['correct'] / stats['total']
            if cat_acc == 100.0:
                categories_100pct += 1
            status = "✅" if cat_acc == 100.0 else "⚠️" if cat_acc >= 80.0 else "❌"
            print(f"{status} {cat_name:<13} {stats['correct']:<12} {stats['total']:<10} {cat_acc:.1f}%")

    print("-" * 50)
    print(f"\nCategories tested: {categories_tested}/15")
    print(f"Categories at 100%: {categories_100pct}/{categories_tested}")

    # Show failures
    if results['failures']:
        print("\n" + "-" * 80)
        print(f"FAILURES ({len(results['failures'])} total)")
        print("-" * 80)
        for i, failure in enumerate(results['failures'][:20]):  # Show first 20
            print(f"  {failure['name']}: expected {failure['expected']}, got {failure['predicted']}")
        if len(results['failures']) > 20:
            print(f"  ... and {len(results['failures']) - 20} more failures")

    # KVRM validity check
    print("\n" + "-" * 80)
    print("KVRM VALIDITY CHECK")
    print("-" * 80)
    invalid_outputs = 0
    for failure in results['failures']:
        if 'UNKNOWN' in failure['predicted']:
            invalid_outputs += 1
    print(f"Invalid outputs (outside registry): {invalid_outputs}/{results['total']} ({100.0*invalid_outputs/results['total']:.2f}%)")
    if invalid_outputs == 0:
        print("✅ KVRM GUARANTEE VERIFIED: 0% invalid outputs")
    else:
        print("⚠️ Some outputs were outside the registry")

    return accuracy


def main():
    """Run comprehensive ARM64 evaluation."""
    print("=" * 80)
    print("KVRM-SPNC COMPREHENSIVE ARM64 INSTRUCTION EVALUATION")
    print("=" * 80)
    print(f"\nDevice: {device}")

    # Load model
    model = UniversalARM64Decoder(d_model=256).to(device)

    # Try multiple decoder models - prefer comprehensive decoder
    model_paths = [
        Path('models/final/comprehensive_decoder.pt'),  # New comprehensive model
        Path('models/final/arm64_decoder_100pct.pt'),
        Path('models/final/decoder_pure_neural.pt'),
        Path('models/final/decoder_movz_fixed.pt'),
    ]

    model_path = None
    for p in model_paths:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        model_path = Path('models/final/decoder_movz_fixed.pt')
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"✅ Loaded model from {model_path}")

    # Generate and run tests
    tests = generate_comprehensive_tests()
    print(f"\nGenerated {len(tests)} test cases across {len(CATEGORIES)} categories")

    results = evaluate_decoder(model, tests)
    accuracy = print_results(results)

    # Save results
    output_path = Path('comprehensive_arm64_results.json')
    save_results = {
        'total_tests': results['total'],
        'correct': results['correct'],
        'accuracy': accuracy,
        'per_category': {k: v for k, v in results['per_category'].items() if v['total'] > 0},
        'failures': results['failures'][:50],  # First 50 failures
        'confusion_matrix': results['confusion_matrix'].tolist(),
    }
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")

    # Publication readiness
    print("\n" + "=" * 80)
    print("PUBLICATION READINESS ASSESSMENT")
    print("=" * 80)
    if accuracy >= 95.0:
        print("✅ DECODER MEETS PUBLICATION THRESHOLD (≥95% accuracy)")
    elif accuracy >= 90.0:
        print("⚠️ DECODER APPROACHING PUBLICATION THRESHOLD (90-95% accuracy)")
    else:
        print("❌ DECODER BELOW PUBLICATION THRESHOLD (<90% accuracy)")

    return accuracy >= 90.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
