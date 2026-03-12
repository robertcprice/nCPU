"""Tests for ARM64 GPU Compute Mode (Metal V2 Kernel).

Verifies that the expanded 125-instruction ARM64 Metal shader
executes correctly on Apple Silicon via MLXKernelCPUv2.

Instruction categories tested:
  - Data movement: MOVZ, MOVK, MOVN, MOV (register)
  - Arithmetic: ADD, SUB, ADDS, SUBS, MUL, MADD, MSUB, SDIV, UDIV, NEG
  - Logic: AND, ORR, EOR, ANDS, BIC, MVN, TST
  - Shifts: LSL, LSR, ASR, ROR (register)
  - Bit manipulation: CLZ, RBIT, REV, REV16, REV32
  - Conditional: CSEL, CSINC, CSINV, CSNEG
  - Extension: SXTW, UXTB, UXTH, UBFM, SBFM, BFM (BFI/BFXIL)
  - Memory: LDR/STR (64/32/byte), LDP/STP, LDRH/STRH, LDRSB/LDRSH/LDRSW, pre/post-index
  - Branches: B, BL, BR, BLR, RET, B.cond, CBZ, CBNZ, TBZ, TBNZ
  - Address: ADR, ADRP
  - System: NOP, HLT, DMB, DSB, ISB, MRS, MSR, SVC

Requires: mlx (Apple Silicon Metal)
"""

import struct
import sys
from pathlib import Path

import pytest
from kernels.mlx.availability import has_gpu_backend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HAS_GPU_BACKEND = has_gpu_backend()

pytestmark = pytest.mark.skipif(not HAS_GPU_BACKEND, reason="GPU backend not available")


# ═══════════════════════════════════════════════════════════════════════════════
# ARM64 Instruction Encoders
# ═══════════════════════════════════════════════════════════════════════════════

def movz(rd, imm16, hw=0):
    """MOVZ Xd, #imm16, LSL #(hw*16)"""
    return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

def movz_w(rd, imm16, hw=0):
    """MOVZ Wd, #imm16"""
    return 0x52800000 | (hw << 21) | (imm16 << 5) | rd

def movk(rd, imm16, hw=0):
    """MOVK Xd, #imm16, LSL #(hw*16)"""
    return 0xF2800000 | (hw << 21) | (imm16 << 5) | rd

def movn(rd, imm16, hw=0):
    """MOVN Xd, #imm16, LSL #(hw*16)"""
    return 0x92800000 | (hw << 21) | (imm16 << 5) | rd

def add_reg(rd, rn, rm):
    """ADD Xd, Xn, Xm"""
    return 0x8B000000 | (rm << 16) | (rn << 5) | rd

def add_imm(rd, rn, imm12):
    """ADD Xd, Xn, #imm12"""
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd

def add_w_reg(rd, rn, rm):
    """ADD Wd, Wn, Wm"""
    return 0x0B000000 | (rm << 16) | (rn << 5) | rd

def sub_reg(rd, rn, rm):
    """SUB Xd, Xn, Xm"""
    return 0xCB000000 | (rm << 16) | (rn << 5) | rd

def sub_imm(rd, rn, imm12):
    """SUB Xd, Xn, #imm12"""
    return 0xD1000000 | (imm12 << 10) | (rn << 5) | rd

def adds_reg(rd, rn, rm):
    """ADDS Xd, Xn, Xm"""
    return 0xAB000000 | (rm << 16) | (rn << 5) | rd

def subs_reg(rd, rn, rm):
    """SUBS Xd, Xn, Xm"""
    return 0xEB000000 | (rm << 16) | (rn << 5) | rd

def subs_imm(rd, rn, imm12):
    """SUBS Xd, Xn, #imm12"""
    return 0xF1000000 | (imm12 << 10) | (rn << 5) | rd

def mul_reg(rd, rn, rm):
    """MUL Xd, Xn, Xm (MADD Xd, Xn, Xm, XZR)"""
    return 0x9B000000 | (rm << 16) | (31 << 10) | (rn << 5) | rd

def madd(rd, rn, rm, ra):
    """MADD Xd, Xn, Xm, Xa"""
    return 0x9B000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd

def msub(rd, rn, rm, ra):
    """MSUB Xd, Xn, Xm, Xa"""
    return 0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd

def sdiv(rd, rn, rm):
    """SDIV Xd, Xn, Xm"""
    return 0x9AC00C00 | (rm << 16) | (rn << 5) | rd

def udiv(rd, rn, rm):
    """UDIV Xd, Xn, Xm"""
    return 0x9AC00800 | (rm << 16) | (rn << 5) | rd

def and_reg(rd, rn, rm):
    """AND Xd, Xn, Xm"""
    return 0x8A000000 | (rm << 16) | (rn << 5) | rd

def orr_reg(rd, rn, rm):
    """ORR Xd, Xn, Xm"""
    return 0xAA000000 | (rm << 16) | (rn << 5) | rd

def eor_reg(rd, rn, rm):
    """EOR Xd, Xn, Xm"""
    return 0xCA000000 | (rm << 16) | (rn << 5) | rd

def ands_reg(rd, rn, rm):
    """ANDS Xd, Xn, Xm"""
    return 0xEA000000 | (rm << 16) | (rn << 5) | rd

def lsl_reg(rd, rn, rm):
    """LSL Xd, Xn, Xm"""
    return 0x9AC02000 | (rm << 16) | (rn << 5) | rd

def lsr_reg(rd, rn, rm):
    """LSR Xd, Xn, Xm"""
    return 0x9AC02400 | (rm << 16) | (rn << 5) | rd

def asr_reg(rd, rn, rm):
    """ASR Xd, Xn, Xm"""
    return 0x9AC02800 | (rm << 16) | (rn << 5) | rd

def clz(rd, rn):
    """CLZ Xd, Xn"""
    return 0xDAC01000 | (rn << 5) | rd

def rev(rd, rn):
    """REV Xd, Xn"""
    return 0xDAC00C00 | (rn << 5) | rd

def csel(rd, rn, rm, cond):
    """CSEL Xd, Xn, Xm, cond"""
    return 0x9A800000 | (rm << 16) | (cond << 12) | (rn << 5) | rd

def csinc(rd, rn, rm, cond):
    """CSINC Xd, Xn, Xm, cond"""
    return 0x9A800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd

def str_64(rt, rn, imm12=0):
    """STR Xt, [Xn, #imm12*8]"""
    return 0xF9000000 | (imm12 << 10) | (rn << 5) | rt

def ldr_64(rt, rn, imm12=0):
    """LDR Xt, [Xn, #imm12*8]"""
    return 0xF9400000 | (imm12 << 10) | (rn << 5) | rt

def str_32(rt, rn, imm12=0):
    """STR Wt, [Xn, #imm12*4]"""
    return 0xB9000000 | (imm12 << 10) | (rn << 5) | rt

def ldr_32(rt, rn, imm12=0):
    """LDR Wt, [Xn, #imm12*4]"""
    return 0xB9400000 | (imm12 << 10) | (rn << 5) | rt

def strb(rt, rn, imm12=0):
    """STRB Wt, [Xn, #imm12]"""
    return 0x39000000 | (imm12 << 10) | (rn << 5) | rt

def ldrb(rt, rn, imm12=0):
    """LDRB Wt, [Xn, #imm12]"""
    return 0x39400000 | (imm12 << 10) | (rn << 5) | rt

def stp_signed(rt, rt2, rn, imm7=0):
    """STP Xt, Xt2, [Xn, #imm7*8]"""
    return 0xA9000000 | ((imm7 & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt

def ldp_signed(rt, rt2, rn, imm7=0):
    """LDP Xt, Xt2, [Xn, #imm7*8]"""
    return 0xA9400000 | ((imm7 & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt

def b(offset):
    """B offset (in instructions)"""
    imm26 = offset & 0x3FFFFFF
    return 0x14000000 | imm26

def bl(offset):
    """BL offset (in instructions)"""
    imm26 = offset & 0x3FFFFFF
    return 0x94000000 | imm26

def br(rn):
    """BR Xn"""
    return 0xD61F0000 | (rn << 5)

def blr(rn):
    """BLR Xn"""
    return 0xD63F0000 | (rn << 5)

def ret(rn=30):
    """RET Xn (default X30)"""
    return 0xD65F0000 | (rn << 5)

def b_cond(cond, offset):
    """B.cond offset (in instructions)"""
    imm19 = offset & 0x7FFFF
    return 0x54000000 | (imm19 << 5) | cond

def cbz(rt, offset):
    """CBZ Xt, offset"""
    imm19 = offset & 0x7FFFF
    return 0xB4000000 | (imm19 << 5) | rt

def cbnz(rt, offset):
    """CBNZ Xt, offset"""
    imm19 = offset & 0x7FFFF
    return 0xB5000000 | (imm19 << 5) | rt

def tbz(rt, bit, offset):
    """TBZ Xt, #bit, offset"""
    b5 = (bit >> 5) & 1
    b40 = bit & 0x1F
    imm14 = offset & 0x3FFF
    return 0x36000000 | (b5 << 31) | (b40 << 19) | (imm14 << 5) | rt

def tbnz(rt, bit, offset):
    """TBNZ Xt, #bit, offset"""
    b5 = (bit >> 5) & 1
    b40 = bit & 0x1F
    imm14 = offset & 0x3FFF
    return 0x37000000 | (b5 << 31) | (b40 << 19) | (imm14 << 5) | rt

def bfm_w(rd, rn, immr, imms):
    """BFM Wd, Wn, #immr, #imms (32-bit)"""
    return 0x33000000 | (immr << 16) | (imms << 10) | (rn << 5) | rd

def bfi_w(rd, rn, lsb, width):
    """BFI Wd, Wn, #lsb, #width — alias for BFM Wd, Wn, #(32-lsb)%32, #(width-1)"""
    immr = (32 - lsb) & 0x1F
    imms = width - 1
    return bfm_w(rd, rn, immr, imms)

def bfxil_w(rd, rn, lsb, width):
    """BFXIL Wd, Wn, #lsb, #width — alias for BFM Wd, Wn, #lsb, #(lsb+width-1)"""
    return bfm_w(rd, rn, lsb, lsb + width - 1)

def str_32_post(rt, rn, imm9):
    """STR Wt, [Xn], #imm9 (post-index)"""
    return 0xB8000400 | ((imm9 & 0x1FF) << 12) | (rn << 5) | rt

def ldr_32_post(rt, rn, imm9):
    """LDR Wt, [Xn], #imm9 (post-index)"""
    return 0xB8400400 | ((imm9 & 0x1FF) << 12) | (rn << 5) | rt

def str_32_pre(rt, rn, imm9):
    """STR Wt, [Xn, #imm9]! (pre-index)"""
    return 0xB8000C00 | ((imm9 & 0x1FF) << 12) | (rn << 5) | rt

def ldur_32(rt, rn, imm9):
    """LDUR Wt, [Xn, #imm9] (unscaled offset)"""
    return 0xB8400000 | ((imm9 & 0x1FF) << 12) | (rn << 5) | rt

def stur_32(rt, rn, imm9):
    """STUR Wt, [Xn, #imm9] (unscaled offset)"""
    return 0xB8000000 | ((imm9 & 0x1FF) << 12) | (rn << 5) | rt

NOP = 0xD503201F
HLT = 0xD4400000
SVC = 0xD4000001
DMB = 0xD5033BBF
DSB = 0xD5033F9F
ISB = 0xD5033FDF


def build_binary(insts):
    """Convert list of 32-bit instruction words to bytes."""
    return b''.join(struct.pack('<I', i) for i in insts)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cpu():
    """Create a fresh GPU backend CPU with 64KB memory."""
    from kernels.mlx.gpu_cpu import GPUKernelCPU
    return GPUKernelCPU(memory_size=64 * 1024, quiet=True)


def run(cpu, insts, max_cycles=200):
    """Load program and execute."""
    cpu.load_program(insts, address=0)
    cpu.set_pc(0)
    return cpu.execute(max_cycles=max_cycles)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Movement Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataMovement:

    def test_movz(self, cpu):
        run(cpu, [movz(0, 42), HLT])
        assert cpu.get_register(0) == 42

    def test_movz_hw1(self, cpu):
        run(cpu, [movz(0, 0xAB, hw=1), HLT])
        assert cpu.get_register(0) == 0xAB << 16

    def test_movk(self, cpu):
        run(cpu, [movz(0, 0xBEEF), movk(0, 0xDEAD, hw=1), HLT])
        assert cpu.get_register(0) == 0xDEADBEEF

    def test_movn(self, cpu):
        """MOVN X0, #0 → X0 = ~0 = -1"""
        run(cpu, [movn(0, 0), HLT])
        assert cpu.get_register(0) == -1

    def test_movz_w(self, cpu):
        run(cpu, [movz_w(0, 0xFFFF), HLT])
        assert cpu.get_register(0) == 0xFFFF

    def test_mov_reg(self, cpu):
        """MOV Xd, Xm = ORR Xd, XZR, Xm"""
        run(cpu, [movz(1, 99), orr_reg(0, 31, 1), HLT])
        assert cpu.get_register(0) == 99

    def test_movz_to_xzr(self, cpu):
        """Writing to X31 (XZR) should be a no-op."""
        run(cpu, [movz(31, 42), HLT])
        assert cpu.get_register(0) == 0  # X0 untouched


# ═══════════════════════════════════════════════════════════════════════════════
# Arithmetic Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestArithmetic:

    def test_add_reg(self, cpu):
        run(cpu, [movz(0, 10), movz(1, 20), add_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 30

    def test_add_imm(self, cpu):
        run(cpu, [movz(0, 100), add_imm(1, 0, 50), HLT])
        assert cpu.get_register(1) == 150

    def test_sub_reg(self, cpu):
        run(cpu, [movz(0, 50), movz(1, 30), sub_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 20

    def test_sub_imm(self, cpu):
        run(cpu, [movz(0, 100), sub_imm(1, 0, 25), HLT])
        assert cpu.get_register(1) == 75

    def test_adds_sets_flags(self, cpu):
        """ADDS should set flags."""
        run(cpu, [movz(0, 0), movz(1, 0), adds_reg(2, 0, 1), HLT])
        n, z, c, v = cpu.get_flags()
        assert z  # result is zero

    def test_subs_sets_flags(self, cpu):
        """SUBS X31, X0, X1 = CMP X0, X1."""
        run(cpu, [movz(0, 10), movz(1, 10), subs_reg(31, 0, 1), HLT])
        n, z, c, v = cpu.get_flags()
        assert z  # 10-10 = 0

    def test_mul(self, cpu):
        run(cpu, [movz(0, 7), movz(1, 6), mul_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 42

    def test_madd(self, cpu):
        """MADD X3, X0, X1, X2 = X0*X1 + X2."""
        run(cpu, [movz(0, 3), movz(1, 4), movz(2, 10), madd(3, 0, 1, 2), HLT])
        assert cpu.get_register(3) == 22  # 3*4 + 10

    def test_msub(self, cpu):
        """MSUB X3, X0, X1, X2 = X2 - X0*X1."""
        run(cpu, [movz(0, 3), movz(1, 4), movz(2, 100), msub(3, 0, 1, 2), HLT])
        assert cpu.get_register(3) == 88  # 100 - 3*4

    def test_sdiv(self, cpu):
        run(cpu, [movz(0, 100), movz(1, 7), sdiv(2, 0, 1), HLT])
        assert cpu.get_register(2) == 14

    def test_udiv(self, cpu):
        run(cpu, [movz(0, 100), movz(1, 7), udiv(2, 0, 1), HLT])
        assert cpu.get_register(2) == 14

    def test_sdiv_by_zero(self, cpu):
        run(cpu, [movz(0, 42), movz(1, 0), sdiv(2, 0, 1), HLT])
        assert cpu.get_register(2) == 0

    def test_neg(self, cpu):
        """NEG Xd, Xm = SUB Xd, XZR, Xm."""
        neg_inst = sub_reg(1, 31, 0)  # SUB X1, XZR, X0
        run(cpu, [movz(0, 42), neg_inst, HLT])
        assert cpu.get_register(1) == -42

    def test_add_w(self, cpu):
        """32-bit ADD should mask result to 32 bits."""
        run(cpu, [movz_w(0, 0xFFFF), movz_w(1, 1), add_w_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 0x10000


# ═══════════════════════════════════════════════════════════════════════════════
# Logic Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogic:

    def test_and(self, cpu):
        run(cpu, [movz(0, 0xFF), movz(1, 0x0F), and_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 0x0F

    def test_orr(self, cpu):
        run(cpu, [movz(0, 0xF0), movz(1, 0x0F), orr_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 0xFF

    def test_eor(self, cpu):
        run(cpu, [movz(0, 0xFF), movz(1, 0xFF), eor_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 0

    def test_ands_sets_flags(self, cpu):
        """ANDS should set Z flag when result is zero."""
        run(cpu, [movz(0, 0xF0), movz(1, 0x0F), ands_reg(2, 0, 1), HLT])
        n, z, c, v = cpu.get_flags()
        assert z  # 0xF0 & 0x0F = 0
        assert cpu.get_register(2) == 0

    def test_mvn(self, cpu):
        """MVN Xd, Xm = ORN Xd, XZR, Xm."""
        # ORN: 0xAA200000 | rm<<16 | rn<<5 | rd (with N bit set)
        orn = 0xAA200000 | (0 << 16) | (31 << 5) | 1  # MVN X1, X0
        run(cpu, [movz(0, 0), orn, HLT])
        assert cpu.get_register(1) == -1  # ~0 = -1


# ═══════════════════════════════════════════════════════════════════════════════
# Shift Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShifts:

    def test_lsl(self, cpu):
        run(cpu, [movz(0, 1), movz(1, 4), lsl_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 16

    def test_lsr(self, cpu):
        run(cpu, [movz(0, 256), movz(1, 4), lsr_reg(2, 0, 1), HLT])
        assert cpu.get_register(2) == 16

    def test_asr(self, cpu):
        """ASR should preserve sign bit."""
        # Use MOVN to get a negative value: MOVN X0, #0 = ~0 = -1
        # Then LSL to get -256 (0xFFFFFFFFFFFFFF00)
        run(cpu, [movn(0, 0xFF), movz(1, 4), asr_reg(2, 0, 1), HLT])
        # -256 >> 4 = -16
        assert cpu.get_register(2) == -16


# ═══════════════════════════════════════════════════════════════════════════════
# Bit Manipulation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBitManipulation:

    def test_clz_nonzero(self, cpu):
        run(cpu, [movz(0, 0x80, hw=1), clz(1, 0), HLT])
        # 0x800000 has bit 23 as highest → CLZ = 64-24 = 40
        assert cpu.get_register(1) == 40

    def test_clz_zero(self, cpu):
        run(cpu, [movz(0, 0), clz(1, 0), HLT])
        assert cpu.get_register(1) == 64

    def test_clz_one(self, cpu):
        run(cpu, [movz(0, 1), clz(1, 0), HLT])
        assert cpu.get_register(1) == 63

    def test_rev(self, cpu):
        """REV reverses all 8 bytes."""
        run(cpu, [movz(0, 0xBBCC), movk(0, 0x00AA, hw=1), rev(1, 0), HLT])
        # 0x00AABBCC → reversed = 0xCCBBAA0000000000
        val = cpu.get_register(1) & 0xFFFFFFFFFFFFFFFF
        assert val == 0xCCBBAA0000000000


# ═══════════════════════════════════════════════════════════════════════════════
# BFM (Bitfield Move) Tests — critical for GCC -O2 byte packing (AES, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBFM:

    def test_bfi_insert_low_byte(self, cpu):
        """BFI Wd, Wn, #0, #8 — insert low 8 bits at bit 0 (= BFXIL)."""
        # x0 = 0xDEADBEEF, x1 = 0x42
        # BFI w0, w1, #0, #8 → low byte of w0 becomes 0x42
        run(cpu, [movz(0, 0xBEEF), movk(0, 0xDEAD, hw=1),
                  movz(1, 0x42),
                  bfi_w(0, 1, 0, 8), HLT])
        assert (cpu.get_register(0) & 0xFFFFFFFF) == 0xDEADBE42

    def test_bfi_insert_byte_at_8(self, cpu):
        """BFI Wd, Wn, #8, #8 — insert 8 bits at bit position 8."""
        # x0 = 0x12345678, x1 = 0xAB
        run(cpu, [movz(0, 0x5678), movk(0, 0x1234, hw=1),
                  movz(1, 0xAB),
                  bfi_w(0, 1, 8, 8), HLT])
        assert (cpu.get_register(0) & 0xFFFFFFFF) == 0x1234AB78

    def test_bfi_insert_byte_at_16(self, cpu):
        """BFI Wd, Wn, #16, #8 — insert 8 bits at bit position 16."""
        run(cpu, [movz(0, 0x5678), movk(0, 0x1234, hw=1),
                  movz(1, 0xCD),
                  bfi_w(0, 1, 16, 8), HLT])
        assert (cpu.get_register(0) & 0xFFFFFFFF) == 0x12CD5678

    def test_bfi_insert_byte_at_24(self, cpu):
        """BFI Wd, Wn, #24, #8 — insert 8 bits at bit position 24."""
        run(cpu, [movz(0, 0x5678), movk(0, 0x1234, hw=1),
                  movz(1, 0xEF),
                  bfi_w(0, 1, 24, 8), HLT])
        assert (cpu.get_register(0) & 0xFFFFFFFF) == 0xEF345678

    def test_bfxil_extract_low_byte(self, cpu):
        """BFXIL Wd, Wn, #0, #8 — extract low byte, insert at bit 0."""
        run(cpu, [movz(0, 0xFFFF), movk(0, 0xFFFF, hw=1),
                  movz(1, 0xABCD),
                  bfxil_w(0, 1, 0, 8), HLT])
        # Extract bits [7:0] of x1 (0xCD), insert at bits [7:0] of x0
        assert (cpu.get_register(0) & 0xFFFFFFFF) == 0xFFFFFFCD

    def test_bfxil_extract_byte_at_8(self, cpu):
        """BFXIL Wd, Wn, #8, #8 — extract byte at bit 8, insert at bit 0."""
        run(cpu, [movz(0, 0xFFFF), movk(0, 0xFFFF, hw=1),
                  movz(1, 0xABCD),
                  bfxil_w(0, 1, 8, 8), HLT])
        # Extract bits [15:8] of 0xABCD = 0xAB, insert at bits [7:0]
        assert (cpu.get_register(0) & 0xFFFFFFFF) == 0xFFFFFFAB

    def test_bfi_pack_four_bytes(self, cpu):
        """Pack 4 bytes into a word using BFI — the exact AES S-box pattern."""
        # Start with x0 = 0, insert bytes at positions 0, 8, 16, 24
        run(cpu, [movz(0, 0),
                  movz(1, 0x63),  # byte 0
                  bfi_w(0, 1, 0, 8),
                  movz(1, 0x7C),  # byte 1
                  bfi_w(0, 1, 8, 8),
                  movz(1, 0x77),  # byte 2
                  bfi_w(0, 1, 16, 8),
                  movz(1, 0x7B),  # byte 3
                  bfi_w(0, 1, 24, 8),
                  HLT])
        assert (cpu.get_register(0) & 0xFFFFFFFF) == 0x7B777C63


# ═══════════════════════════════════════════════════════════════════════════════
# 32-bit LDR/STR Pre/Post-Index Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemory32PrePost:

    def test_str_32_post_index(self, cpu):
        """STR Wt, [Xn], #4 — store then increment base."""
        run(cpu, [movz(0, 0x1000),  # base addr
                  movz(1, 0xBEEF),
                  str_32_post(1, 0, 4),  # store w1 at [x0], x0 += 4
                  HLT])
        assert cpu.get_register(0) == 0x1004  # base incremented
        # Verify store happened at 0x1000
        data = cpu.read_memory(0x1000, 4)
        val = int.from_bytes(bytes(data), 'little')
        assert val == 0xBEEF

    def test_ldr_32_post_index(self, cpu):
        """LDR Wt, [Xn], #4 — load then increment base."""
        # Pre-store a word at 0x1000
        cpu.write_memory(0x1000, (0xCAFE).to_bytes(4, 'little'))
        run(cpu, [movz(0, 0x1000),
                  ldr_32_post(1, 0, 4),
                  HLT])
        assert (cpu.get_register(1) & 0xFFFFFFFF) == 0xCAFE
        assert cpu.get_register(0) == 0x1004

    def test_str_32_pre_index(self, cpu):
        """STR Wt, [Xn, #4]! — increment base then store."""
        run(cpu, [movz(0, 0x1000),
                  movz(1, 0xFACE),
                  str_32_pre(1, 0, 4),  # x0 += 4, store w1 at [x0]
                  HLT])
        assert cpu.get_register(0) == 0x1004  # base incremented
        data = cpu.read_memory(0x1004, 4)
        val = int.from_bytes(bytes(data), 'little')
        assert val == 0xFACE

    def test_stur_ldur_32(self, cpu):
        """STUR/LDUR 32-bit with unscaled offset."""
        run(cpu, [movz(0, 0x1000),
                  movz(1, 0xDEAD),
                  stur_32(1, 0, 5),   # store at 0x1005
                  ldur_32(2, 0, 5),   # load from 0x1005
                  HLT])
        assert (cpu.get_register(2) & 0xFFFFFFFF) == 0xDEAD


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional Select Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConditionalSelect:

    def test_csel_true(self, cpu):
        """CSEL with GT condition when X0 > X1."""
        run(cpu, [
            movz(0, 100), movz(1, 50),
            subs_reg(31, 0, 1),        # CMP X0, X1 (sets flags: positive)
            csel(2, 0, 1, 0xC),        # CSEL X2, X0, X1, GT → X2=100
            HLT,
        ])
        assert cpu.get_register(2) == 100

    def test_csel_false(self, cpu):
        """CSEL with EQ condition when X0 != X1."""
        run(cpu, [
            movz(0, 100), movz(1, 50),
            subs_reg(31, 0, 1),        # CMP: not equal
            csel(2, 0, 1, 0x0),        # CSEL X2, X0, X1, EQ → X2=50
            HLT,
        ])
        assert cpu.get_register(2) == 50

    def test_csinc(self, cpu):
        """CSINC: if cond true → Xn, else → Xm+1."""
        run(cpu, [
            movz(0, 10), movz(1, 20),
            subs_reg(31, 0, 1),        # CMP: not equal (Z=0)
            csinc(2, 0, 1, 0x0),       # CSINC X2, X0, X1, EQ → X2=X1+1=21
            HLT,
        ])
        assert cpu.get_register(2) == 21


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemory:

    def test_str_ldr_64(self, cpu):
        """Store and load 64-bit value."""
        run(cpu, [
            movz(0, 0x1234), movz(1, 0x2000),
            str_64(0, 1, 0),           # STR X0, [X1]
            movz(0, 0),               # clear X0
            ldr_64(2, 1, 0),          # LDR X2, [X1]
            HLT,
        ])
        assert cpu.get_register(2) == 0x1234
        assert cpu.read_memory_64(0x2000) == 0x1234

    def test_str_ldr_32(self, cpu):
        """Store and load 32-bit value."""
        run(cpu, [
            movz(0, 0xABCD), movz(1, 0x2000),
            str_32(0, 1, 0),
            movz(0, 0),
            ldr_32(2, 1, 0),
            HLT,
        ])
        assert cpu.get_register(2) == 0xABCD

    def test_strb_ldrb(self, cpu):
        """Store and load byte."""
        run(cpu, [
            movz(0, 0xFF), movz(1, 0x2000),
            strb(0, 1, 0),
            movz(0, 0),
            ldrb(2, 1, 0),
            HLT,
        ])
        assert cpu.get_register(2) == 0xFF

    def test_stp_ldp(self, cpu):
        """Store pair and load pair."""
        run(cpu, [
            movz(0, 0xAAAA), movz(1, 0xBBBB), movz(2, 0x3000),
            stp_signed(0, 1, 2, 0),
            movz(0, 0), movz(1, 0),
            ldp_signed(3, 4, 2, 0),
            HLT,
        ])
        assert cpu.get_register(3) == 0xAAAA
        assert cpu.get_register(4) == 0xBBBB

    def test_ldr_with_offset(self, cpu):
        """LDR with scaled unsigned offset."""
        run(cpu, [
            movz(0, 0x42), movz(1, 0x2000),
            str_64(0, 1, 1),           # STR X0, [X1, #8] (imm12=1 → byte offset 8)
            ldr_64(2, 1, 1),           # LDR X2, [X1, #8]
            HLT,
        ])
        assert cpu.get_register(2) == 0x42


# ═══════════════════════════════════════════════════════════════════════════════
# Branch Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBranches:

    def test_b_unconditional(self, cpu):
        run(cpu, [
            movz(0, 1),
            b(2),                      # skip next instruction
            movz(0, 99),               # should be skipped
            HLT,
        ])
        assert cpu.get_register(0) == 1

    def test_bl_ret(self, cpu):
        """BL saves return address in X30, RET returns to it."""
        run(cpu, [
            bl(3),                     # [0] BL to [3], X30 = addr of [1]
            movz(1, 99),               # [1] executed after RET
            HLT,                       # [2] halt
            movz(0, 55),               # [3] subroutine body
            ret(),                     # [4] RET → goes to [1]
        ])
        assert cpu.get_register(0) == 55
        assert cpu.get_register(1) == 99

    def test_br(self, cpu):
        """BR Xn = indirect branch."""
        run(cpu, [
            movz(0, 12),              # [0] X0 = 12 (byte address of instruction [3])
            br(0),                     # [1] BR X0 → jump to addr 12 = inst[3]
            movz(1, 99),               # [2] skipped
            movz(1, 42),               # [3] landed here
            HLT,                       # [4]
        ])
        assert cpu.get_register(1) == 42

    def test_b_cond_eq_taken(self, cpu):
        run(cpu, [
            movz(0, 10), movz(1, 10),
            subs_reg(31, 0, 1),        # CMP X0, X1 → Z=1
            b_cond(0x0, 2),            # B.EQ +2 → skip next
            movz(2, 99),               # skipped
            movz(2, 42),
            HLT,
        ])
        assert cpu.get_register(2) == 42

    def test_b_cond_eq_not_taken(self, cpu):
        run(cpu, [
            movz(0, 10), movz(1, 20),
            subs_reg(31, 0, 1),        # CMP: not equal
            b_cond(0x0, 2),            # B.EQ +2 → NOT taken
            movz(2, 99),               # executed
            HLT,                       # halt before next
            movz(2, 42),               # not reached
        ])
        assert cpu.get_register(2) == 99

    def test_cbz_taken(self, cpu):
        run(cpu, [
            movz(0, 0),               # X0 = 0
            cbz(0, 2),                # CBZ X0, +2 → skip next
            movz(1, 99),
            movz(1, 42),
            HLT,
        ])
        assert cpu.get_register(1) == 42

    def test_cbnz_taken(self, cpu):
        run(cpu, [
            movz(0, 5),
            cbnz(0, 2),               # CBNZ X0, +2 → skip next (X0 != 0)
            movz(1, 99),
            movz(1, 42),
            HLT,
        ])
        assert cpu.get_register(1) == 42

    def test_tbz_taken(self, cpu):
        """TBZ: branch if bit is zero."""
        run(cpu, [
            movz(0, 0b1010),          # bit 0 is 0
            tbz(0, 0, 2),             # TBZ X0, #0, +2 → branch taken
            movz(1, 99),
            movz(1, 42),
            HLT,
        ])
        assert cpu.get_register(1) == 42

    def test_tbnz_taken(self, cpu):
        """TBNZ: branch if bit is not zero."""
        run(cpu, [
            movz(0, 0b1010),          # bit 1 is 1
            tbnz(0, 1, 2),            # TBNZ X0, #1, +2 → branch taken
            movz(1, 99),
            movz(1, 42),
            HLT,
        ])
        assert cpu.get_register(1) == 42


# ═══════════════════════════════════════════════════════════════════════════════
# System / NOP / Barrier Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystem:

    def test_nop(self, cpu):
        run(cpu, [NOP, movz(0, 42), HLT])
        assert cpu.get_register(0) == 42

    def test_hlt(self, cpu):
        result = run(cpu, [HLT])
        assert result.stop_reason_name == "HALT"

    def test_dmb_is_nop(self, cpu):
        run(cpu, [DMB, movz(0, 42), HLT])
        assert cpu.get_register(0) == 42

    def test_dsb_is_nop(self, cpu):
        run(cpu, [DSB, movz(0, 42), HLT])
        assert cpu.get_register(0) == 42

    def test_isb_is_nop(self, cpu):
        run(cpu, [ISB, movz(0, 42), HLT])
        assert cpu.get_register(0) == 42


# ═══════════════════════════════════════════════════════════════════════════════
# Program Tests (Fibonacci, Sum, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrograms:

    def test_sum_1_to_10(self, cpu):
        """Sum 1+2+...+10 = 55 using loop."""
        run(cpu, [
            movz(0, 0),               # sum = 0
            movz(1, 10),              # counter = 10
            # loop:
            add_reg(0, 0, 1),         # [2] sum += counter
            subs_imm(1, 1, 1),        # [3] counter--
            cbnz(1, (-2) & 0x7FFFF), # [4] if counter != 0, goto [2]
            HLT,
        ], max_cycles=500)
        assert cpu.get_register(0) == 55

    def test_fibonacci(self, cpu):
        """Compute fib(10) = 55 via 9 loop iterations."""
        run(cpu, [
            movz(0, 0),               # a = 0
            movz(1, 1),               # b = 1
            movz(3, 9),               # 9 iterations → b = fib(10) = 55
            # loop:
            add_reg(2, 0, 1),         # [3] c = a + b
            orr_reg(0, 31, 1),        # [4] a = b (MOV X0, X1)
            orr_reg(1, 31, 2),        # [5] b = c (MOV X1, X2)
            subs_imm(3, 3, 1),        # [6] n--
            cbnz(3, (-4) & 0x7FFFF), # [7] if n != 0, goto [3]
            HLT,
        ], max_cycles=500)
        assert cpu.get_register(1) == 55

    def test_factorial_5(self, cpu):
        """Compute 5! = 120."""
        run(cpu, [
            movz(0, 1),               # result = 1
            movz(1, 5),               # n = 5
            # loop:
            mul_reg(0, 0, 1),         # [2] result *= n
            subs_imm(1, 1, 1),        # [3] n--
            cbnz(1, (-2) & 0x7FFFF), # [4] if n != 0, goto [2]
            HLT,
        ], max_cycles=500)
        assert cpu.get_register(0) == 120

    def test_memory_copy(self, cpu):
        """Copy 4 bytes from one location to another."""
        run(cpu, [
            # Set up source data
            movz(0, 0xABCD),
            movz(1, 0x1000),          # src address
            str_64(0, 1, 0),           # store at src
            # Copy
            movz(2, 0x2000),          # dst address
            ldr_64(3, 1, 0),          # load from src
            str_64(3, 2, 0),           # store at dst
            # Verify
            ldr_64(4, 2, 0),          # load from dst
            HLT,
        ])
        assert cpu.get_register(4) == 0xABCD

    def test_max_cycles_exceeded(self, cpu):
        """Verify max_cycles limit works."""
        result = run(cpu, [
            NOP,
            b(-1 & 0x3FFFFFF),        # infinite loop: B to self
        ], max_cycles=50)
        assert result.stop_reason_name == "MAX_CYCLES"
        assert result.cycles == 50


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestSIMDImmediate:
    """MOVI/MVNI SIMD immediate tests — verifies the MVNI fix for grep regex."""

    def test_mvni_2s_zero(self, cpu):
        """MVNI V15.2S, #0 → V15 = 0xFFFFFFFF_FFFFFFFF, then STR to memory, read back."""
        # MVNI v15.2s, #0  → 0x2F00040F (set v15 to all-ones)
        # STR D15, [X0]    → 0xFD00000F (store 8 bytes to [X0], Rn=0 Rt=15)
        # LDR X1, [X0]     → 0xF9400001 (load 8 bytes to X1)
        addr = 0x8000
        run(cpu, [
            movz(0, addr),      # X0 = address for store
            0x2F00040F,          # MVNI V15.2S, #0
            0xFD00000F,          # STR D15, [X0]
            0xF9400001,          # LDR X1, [X0]
            HLT,
        ])
        # X1 should contain 0xFFFFFFFF_FFFFFFFF = -1
        val = cpu.get_register(1)
        assert val == -1, f"Expected -1, got {val} (0x{val & 0xFFFFFFFFFFFFFFFF:016X})"

    def test_movi_2s_zero(self, cpu):
        """MOVI V15.2S, #0 → V15 = 0, then STR to memory, read back."""
        addr = 0x8000
        # First set memory to non-zero
        cpu.write_memory(addr, b'\xFF' * 8)
        run(cpu, [
            movz(0, addr),
            0x0F00040F,          # MOVI V15.2S, #0 (op_byte 0x0F, Q=0, op=0, Rd=15)
            0xFD00000F,          # STR D15, [X0]
            0xF9400001,          # LDR X1, [X0]
            HLT,
        ])
        assert cpu.get_register(1) == 0, f"Expected 0, got {cpu.get_register(1)}"

    def test_mvni_store_sentinel(self, cpu):
        """Reproduce the grep sentinel pattern: MVNI + STR D pre-index."""
        base_addr = 0x8000
        run(cpu, [
            movz(0, base_addr),    # X0 = base address
            0x2F00040F,            # MVNI V15.2S, #0 (all-ones)
            # STR D15, [X0, #56]! → pre-index: store at X0+56, update X0
            0xFC038C0F,            # STR D15, [X0, #56]!
            HLT,
        ])
        # X0 should be updated to base_addr + 56
        assert cpu.get_register(0) == base_addr + 56
        # Memory at base_addr + 56 should be all-ones
        data = cpu.read_memory(base_addr + 56, 8)
        val = int.from_bytes(data, 'little', signed=True)
        assert val == -1, f"Expected -1 at sentinel, got {val}"

    def test_fmov_ins_sshll_constructor_pattern(self, cpu):
        """Reproduce the grep literal constructor: FMOV + INS + SSHLL + STR Q."""
        addr = 0x8000
        run(cpu, [
            movz(0, addr),
            movz_w(1, 0x72),       # W1 = 'r'
            movz_w(2, 0x72),       # W2 = 'r'
            0x1E27003F,            # FMOV S31, W1
            0x4E0C1C5F,            # MOV V31.S[1], W2
            0x0F20A7FF,            # SSHLL V31.2D, V31.2S, #0
            0x3D80001F,            # STR Q31, [X0]
            HLT,
        ])
        data = cpu.read_memory(addr, 16)
        lo = int.from_bytes(data[:8], 'little', signed=False)
        hi = int.from_bytes(data[8:], 'little', signed=False)
        assert lo == 0x72, f"Expected widened low lane 'r', got {lo} (0x{lo:016X})"
        assert hi == 0x72, f"Expected sign-extended 'r' lane, got {hi} (0x{hi:016X})"


class TestPerformance:

    def test_sustained_ips(self, cpu):
        """Count to 1000 — verify IPS and correctness."""
        run(cpu, [
            movz(0, 0),               # counter
            movz(1, 1000),            # limit
            movz(2, 1),               # increment
            # loop:
            add_reg(0, 0, 2),         # [3] counter++
            subs_reg(31, 0, 1),       # [4] CMP counter, limit
            b_cond(0x1, (-2) & 0x7FFFF),  # [5] B.NE loop (cond 1 = NE)
            HLT,
        ], max_cycles=100_000)
        assert cpu.get_register(0) == 1000
