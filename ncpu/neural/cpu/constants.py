"""
Constants, enums, and helpers shared across the NeuralCPU package.
"""

import logging
import torch
from enum import IntEnum

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# DEVICE SETUP
# ════════════════════════════════════════════════════════════════════════════════

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"[Neural CPU] Device: {device}")


# ════════════════════════════════════════════════════════════════════════════════
# HELPER: UNSIGNED TO SIGNED 64-BIT CONVERSION
# ════════════════════════════════════════════════════════════════════════════════

def _u64_to_s64(val: int) -> int:
    """Convert unsigned 64-bit value to signed for torch.int64 storage."""
    val = val & 0xFFFFFFFFFFFFFFFF
    if val >= 0x8000000000000000:
        return val - 0x10000000000000000
    return val


# ════════════════════════════════════════════════════════════════════════════════
# OPERATION TYPES
# ════════════════════════════════════════════════════════════════════════════════


class OpType(IntEnum):
    NOP = 0
    ADD_IMM = 1
    SUB_IMM = 2
    ADD_REG = 3
    SUB_REG = 4
    MUL = 5
    MOVZ = 6
    MOVK = 7
    CMP_IMM = 8
    CMP_REG = 9
    B = 10
    BL = 11
    B_COND = 12
    CBZ = 13
    CBNZ = 14
    LDRB = 15
    STRB = 16
    LDR = 17
    STR = 18
    RET = 19
    MOV_REG = 20
    # === NEW INSTRUCTIONS FOR ALPINE LINUX SUPPORT ===
    AND_REG = 21      # AND (register)
    AND_IMM = 22      # AND (immediate)
    ORR_REG = 23      # ORR (register)
    ORR_IMM = 24      # ORR (immediate)
    EOR_REG = 25      # EOR (exclusive OR register)
    EOR_IMM = 26      # EOR (immediate)
    LSL_REG = 27      # LSL (register)
    LSL_IMM = 28      # LSL (immediate via UBFM)
    LSR_REG = 29      # LSR (register)
    LSR_IMM = 30      # LSR (immediate via UBFM)
    ASR_REG = 31      # ASR (register)
    ASR_IMM = 32      # ASR (immediate)
    ROR_REG = 33      # ROR (register)
    MVN = 34          # MVN (bitwise NOT)
    BIC = 35          # BIC (AND NOT)
    TST_REG = 36      # TST (AND, set flags, discard result)
    TST_IMM = 37      # TST immediate
    NEG = 38          # NEG (negate)
    BLR = 39          # BLR (branch with link to register)
    BR = 40           # BR (branch to register)
    SVC = 41          # SVC (syscall)
    LDUR = 42         # LDUR (load unscaled)
    STUR = 43         # STUR (store unscaled)
    LDP = 44          # LDP (load pair)
    STP = 45          # STP (store pair)
    MADD = 46         # MADD (multiply-add)
    MSUB = 47         # MSUB (multiply-subtract)
    SDIV = 48         # SDIV (signed divide)
    UDIV = 49         # UDIV (unsigned divide)
    CLZ = 50          # CLZ (count leading zeros)
    SXTW = 51         # SXTW (sign extend word)
    UXTB = 52         # UXTB (zero extend byte)
    UXTH = 53         # UXTH (zero extend halfword)
    # === ADDITIONAL INSTRUCTIONS FOR BUSYBOX SUPPORT ===
    ADDS_IMM = 54     # ADDS (add immediate, set flags)
    ADDS_REG = 55     # ADDS (add register, set flags)
    SUBS_IMM = 56     # SUBS (subtract immediate, set flags)
    SUBS_REG = 57     # SUBS (subtract register, set flags)
    LDRSB = 58        # LDRSB (load register signed byte)
    LDRSH = 59        # LDRSH (load register signed halfword)
    LDRSW = 60        # LDRSW (load register signed word)
    LDRH = 61         # LDRH (load halfword unsigned)
    STRH = 62         # STRH (store halfword)
    CSEL = 63         # CSEL (conditional select)
    CSINC = 64        # CSINC (conditional select increment)
    CSINV = 65        # CSINV (conditional select invert)
    CSNEG = 66        # CSNEG (conditional select negate)
    ADR = 67          # ADR (PC-relative address)
    ADRP = 68         # ADRP (PC-relative address, page)
    UBFM = 69         # UBFM (unsigned bitfield move)
    SBFM = 70         # SBFM (signed bitfield move)
    EXTR = 71         # EXTR (extract register)
    TBZ = 72          # TBZ (test bit and branch if zero)
    TBNZ = 73         # TBNZ (test bit and branch if not zero)
    RBIT = 74         # RBIT (reverse bits)
    REV = 75          # REV (reverse bytes)
    REV16 = 76        # REV16 (reverse bytes in halfwords)
    REV32 = 77        # REV32 (reverse bytes in words)
    ANDS_REG = 78     # ANDS (AND with flags)
    ANDS_IMM = 79     # ANDS (AND immediate with flags)
    LDXR = 80         # LDXR (load exclusive register)
    STXR = 81         # STXR (store exclusive register)
    DMB = 82          # DMB (data memory barrier)
    DSB = 83          # DSB (data synchronization barrier)
    ISB = 84          # ISB (instruction synchronization barrier)
    MRS = 85          # MRS (move from system register)
    MSR = 86          # MSR (move to system register)
    ERET = 87         # ERET (exception return)
    ADD_EXT = 88      # ADD with extension (UXTW, SXTW, etc.)
    SUB_EXT = 89      # SUB with extension
    # === 32-BIT (W) INSTRUCTION VARIANTS ===
    MOVZ_W = 90       # MOVZ 32-bit (W register)
    MOVK_W = 91       # MOVK 32-bit
    MOV_W = 92        # MOV 32-bit register
    ADD_IMM_W = 93    # ADD 32-bit immediate
    SUB_IMM_W = 94    # SUB 32-bit immediate
    ADD_REG_W = 95    # ADD 32-bit register
    SUB_REG_W = 96    # SUB 32-bit register
    ADDS_IMM_W = 97   # ADDS 32-bit immediate
    SUBS_IMM_W = 98   # SUBS 32-bit immediate (CMP_W when Rd=WZR)
    CMP_IMM_W = 99    # CMP 32-bit immediate
    CMP_REG_W = 100   # CMP 32-bit register
    LDR_W = 101       # LDR 32-bit (word)
    STR_W = 102       # STR 32-bit (word)
    LDRSW_IMM = 103   # LDRSW immediate (load signed word)
    LDR_REG_OFF = 113 # LDR 64-bit with register offset (LDR Xt, [Xn, Xm, LSL #shift])
    STR_REG_OFF = 114 # STR 64-bit with register offset (STR Xt, [Xn, Xm, LSL #shift])
    CSEL_W = 104      # CSEL 32-bit
    MADD_W = 105      # MADD 32-bit
    MOVN = 106        # MOVN (move NOT)
    MOVN_W = 107      # MOVN 32-bit
    # Post/pre-index addressing modes - CRITICAL for busybox!
    LDR_POST = 115    # LDR Xt, [Xn], #imm (load then update base)
    STR_POST = 116    # STR Xt, [Xn], #imm (store then update base)
    LDR_PRE = 117     # LDR Xt, [Xn, #imm]! (update base then load)
    STR_PRE = 118     # STR Xt, [Xn, #imm]! (update base then store)
    # Load/Store pair with pre/post-index - CRITICAL for function calls!
    LDP_POST = 119    # LDP Xt1, Xt2, [Xn], #imm (load pair then update base)
    STP_POST = 120    # STP Xt1, Xt2, [Xn], #imm (store pair then update base)
    LDP_PRE = 121     # LDP Xt1, Xt2, [Xn, #imm]! (update base then load pair)
    STP_PRE = 122     # STP Xt1, Xt2, [Xn, #imm]! (update base then store pair)
    # Byte load/store with post-index - used in string loops
    LDRB_POST = 123   # LDRB Wt, [Xn], #imm (load byte then update base)
    STRB_POST = 124   # STRB Wt, [Xn], #imm (store byte then update base)



# Memory map constants
FB_BASE = 0x40000
FB_WIDTH = 80
FB_HEIGHT = 25
FB_SIZE = FB_WIDTH * FB_HEIGHT
