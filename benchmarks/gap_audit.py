"""
gap_audit.py — Static gap analysis for nCPU ARM64 GPU execution engine.

Reads a binary (ELF or raw ARM64), extracts every aligned 4-byte word as a
candidate ARM64 instruction, checks each against all known handler patterns
from gpu_only.py, and reports unmatched encodings with frequency counts.

Usage:
    python benchmarks/gap_audit.py [binary_path] [top_n]

Defaults:
    binary_path = demos/busybox.elf
    top_n       = 20
"""

from __future__ import annotations

import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Known handler patterns extracted from ncpu/neural/cpu/engines/gpu_only.py
# Each entry: (mask, pattern, name)
# A word W is "handled" when (W & mask) == pattern.
# ---------------------------------------------------------------------------
KNOWN_PATTERNS: list[tuple[int, int, str]] = [
    # === BRANCHES ===
    (0xFC000000, 0x14000000, "B"),
    (0xFC000000, 0x94000000, "BL"),
    (0xFF000010, 0x54000000, "B.cond"),
    (0xFF000000, 0xB4000000, "CBZ 64"),
    (0xFF000000, 0x34000000, "CBZ 32"),
    (0xFF000000, 0xB5000000, "CBNZ 64"),
    (0xFF000000, 0x35000000, "CBNZ 32"),
    (0xFFFFFC1F, 0xD61F0000, "BR"),
    (0xFFFFFC1F, 0xD63F0000, "BLR"),
    (0xFFFFFC1F, 0xD65F0000, "RET"),
    (0x7F000000, 0x36000000, "TBZ"),
    (0x7F000000, 0x37000000, "TBNZ"),
    # === HALTS / SYSCALLS ===
    (0xFFFFFFFF, 0x00000000, "HALT/NOP"),
    (0xFFE0001F, 0xD4000001, "SVC"),
    # === ADD/SUB IMMEDIATE ===
    (0xFF000000, 0x91000000, "ADD IMM 64"),
    (0xFF000000, 0xD1000000, "SUB IMM 64"),
    (0xFF000000, 0x11000000, "ADD IMM 32"),
    (0xFF000000, 0x51000000, "SUB IMM 32"),
    # === ADD/SUB REGISTER ===
    (0xFF200000, 0x8B000000, "ADD REG 64"),
    (0xFF200000, 0xCB000000, "SUB REG 64"),
    (0xFF200000, 0x0B000000, "ADD REG 32"),
    (0xFF200000, 0x4B000000, "SUB REG 32"),
    # === MOVZ/MOVK/MOVN ===
    (0xFF800000, 0xD2800000, "MOVZ 64"),
    (0xFF800000, 0x52800000, "MOVZ 32"),
    (0xFF800000, 0xF2800000, "MOVK 64"),
    (0xFF800000, 0x72800000, "MOVK 32"),
    (0xFF800000, 0x92800000, "MOVN 64"),
    (0xFF800000, 0x12800000, "MOVN 32"),
    # === ORR/AND/EOR REGISTER ===
    (0xFF200000, 0xAA000000, "ORR REG 64"),
    (0xFF200000, 0x2A000000, "ORR REG 32"),
    (0xFF200000, 0x8A000000, "AND REG 64"),
    (0xFF200000, 0x0A000000, "AND REG 32"),
    (0xFF200000, 0xCA000000, "EOR REG 64"),
    (0xFF200000, 0x4A000000, "EOR REG 32"),
    # === BIC/BICS/EON/ORN ===
    (0xFF200000, 0x8A200000, "BIC 64"),
    (0xFF200000, 0x0A200000, "BIC 32"),
    (0xFF200000, 0xEA200000, "BICS 64"),
    (0xFF200000, 0x6A200000, "BICS 32"),
    (0xFF200000, 0xCA200000, "EON 64"),
    (0xFF200000, 0x4A200000, "EON 32"),
    (0xFF200000, 0xAA200000, "ORN 64"),
    (0xFF200000, 0x2A200000, "ORN 32"),
    # === SHIFTS IMMEDIATE ===
    (0xFFC00000, 0xD3400000, "UBFM 64 (LSR/UBFX)"),
    (0xFFC0FC00, 0x9340FC00, "ASR IMM 64"),
    (0xFFC0FC00, 0x13007C00, "ASR IMM 32"),
    # === SHIFTS REGISTER ===
    (0xFFE0FC00, 0x9AC02000, "LSL REG 64"),
    (0xFFE0FC00, 0x1AC02000, "LSL REG 32"),
    (0xFFE0FC00, 0x9AC02400, "LSR REG 64"),
    (0xFFE0FC00, 0x1AC02400, "LSR REG 32"),
    (0xFFE0FC00, 0x9AC02800, "ASR REG 64"),
    (0xFFE0FC00, 0x1AC02800, "ASR REG 32"),
    # === ADRP/ADR ===
    (0x9F000000, 0x90000000, "ADRP"),
    (0x9F000000, 0x10000000, "ADR"),
    # === CMP/SUBS ===
    (0xFF200000, 0xEB000000, "SUBS REG 64"),
    (0xFF000000, 0xF1000000, "SUBS IMM 64"),
    (0xFF200000, 0x6B000000, "SUBS REG 32"),
    (0xFF000000, 0x71000000, "SUBS IMM 32"),
    # === ADDS ===
    (0xFF000000, 0xB1000000, "ADDS IMM 64"),
    (0xFF200000, 0xAB000000, "ADDS REG 64"),
    (0xFF000000, 0x31000000, "ADDS IMM 32"),
    (0xFF200000, 0x2B000000, "ADDS REG 32"),
    # === ADC/ADCS/SBC/SBCS ===
    (0xFFE0FC00, 0x9A000000, "ADC 64"),
    (0xFFE0FC00, 0x1A000000, "ADC 32"),
    (0xFFE0FC00, 0xBA000000, "ADCS 64"),
    (0xFFE0FC00, 0x3A000000, "ADCS 32"),
    (0xFFE0FC00, 0xDA000000, "SBC 64"),
    (0xFFE0FC00, 0x5A000000, "SBC 32"),
    (0xFFE0FC00, 0xFA000000, "SBCS 64"),
    (0xFFE0FC00, 0x7A000000, "SBCS 32"),
    # === ANDS REG ===
    (0xFF200000, 0xEA000000, "ANDS REG 64"),
    (0xFF200000, 0x6A000000, "ANDS REG 32"),
    # === MUL/MADD/MSUB ===
    (0xFFE0FC00, 0x9B007C00, "MUL 64"),
    (0xFFE0FC00, 0x1B007C00, "MUL 32"),
    (0xFFE08000, 0x9B000000, "MADD 64"),
    (0xFFE08000, 0x1B000000, "MADD 32"),
    (0xFFE08000, 0x9B008000, "MSUB 64"),
    (0xFFE08000, 0x1B008000, "MSUB 32"),
    # === SMULL/UMULL ===
    (0xFFE08000, 0x9B200000, "SMADDL"),
    (0xFFE08000, 0x9B208000, "SMSUBL"),
    (0xFFE08000, 0x9BA00000, "UMADDL"),
    (0xFFE08000, 0x9BA08000, "UMSUBL"),
    (0xFFE0FC00, 0x9B407C00, "SMULH"),
    (0xFFE0FC00, 0x9BC07C00, "UMULH"),
    # === UDIV/SDIV ===
    (0xFFE0FC00, 0x9AC00800, "UDIV 64"),
    (0xFFE0FC00, 0x1AC00800, "UDIV 32"),
    (0xFFE0FC00, 0x9AC00C00, "SDIV 64"),
    (0xFFE0FC00, 0x1AC00C00, "SDIV 32"),
    # === SXTW/SXTB/SXTH ===
    (0xFFFFFC00, 0x93407C00, "SXTW"),
    (0xFFFFFC00, 0x13001C00, "SXTB 32"),
    (0xFFFFFC00, 0x13003C00, "SXTH 32"),
    (0xFFFFFC00, 0x93401C00, "SXTB 64"),
    (0xFFFFFC00, 0x93403C00, "SXTH 64"),
    # === UXTB/UXTH ===
    (0xFFFFFC00, 0x53001C00, "UXTB"),
    (0xFFFFFC00, 0x53003C00, "UXTH"),
    # === UBFM/SBFM general ===
    (0xFFC00000, 0x53000000, "UBFM 32"),
    (0xFFC00000, 0x93400000, "SBFM 64"),
    (0xFFC00000, 0x13000000, "SBFM 32"),
    # === LDR/STR IMMEDIATE ===
    (0xFFC00000, 0xF9400000, "LDR IMM 64"),
    (0xFFC00000, 0xB9400000, "LDR IMM 32"),
    (0xFFC00000, 0xF9000000, "STR IMM 64"),
    (0xFFC00000, 0xB9000000, "STR IMM 32"),
    # === LDUR/STUR UNSCALED ===
    (0xFFE00C00, 0xF8400000, "LDUR 64"),
    (0xFFE00C00, 0xB8400000, "LDUR 32"),
    (0xFFE00C00, 0xF8000000, "STUR 64"),
    (0xFFE00C00, 0xB8000000, "STUR 32"),
    (0xFFE00C00, 0x78400000, "LDURH"),
    (0xFFE00C00, 0x78000000, "STURH"),
    (0xFFE00C00, 0x38400000, "LDURB"),
    (0xFFE00C00, 0x38000000, "STURB"),
    (0xFFE00C00, 0xB8800000, "LDURSW"),
    (0xFFE00C00, 0x78800000, "LDURSH 64"),
    (0xFFE00C00, 0x78C00000, "LDURSH 32"),
    (0xFFE00C00, 0x38800000, "LDURSB 64"),
    (0xFFE00C00, 0x38C00000, "LDURSB 32"),
    # === LDR/STR POST/PRE-INDEX ===
    (0xFFE00C00, 0xF8400400, "LDR POST 64"),
    (0xFFE00C00, 0xF8400C00, "LDR PRE 64"),
    (0xFFE00C00, 0xB8400400, "LDR POST 32"),
    (0xFFE00C00, 0xB8400C00, "LDR PRE 32"),
    (0xFFE00C00, 0xF8000400, "STR POST 64"),
    (0xFFE00C00, 0xF8000C00, "STR PRE 64"),
    (0xFFE00C00, 0xB8000400, "STR POST 32"),
    (0xFFE00C00, 0xB8000C00, "STR PRE 32"),
    (0xFFE00C00, 0x78400400, "LDRH POST"),
    (0xFFE00C00, 0x78400C00, "LDRH PRE"),
    (0xFFE00C00, 0x78000400, "STRH POST"),
    (0xFFE00C00, 0x78000C00, "STRH PRE"),
    (0xFFE00C00, 0x38400400, "LDRB POST"),
    (0xFFE00C00, 0x38400C00, "LDRB PRE"),
    (0xFFE00C00, 0x38000400, "STRB POST"),
    (0xFFE00C00, 0x38000C00, "STRB PRE"),
    # === LDRH/STRH/LDRB/STRB SCALED ===
    (0xFFC00000, 0x79400000, "LDRH SCALED"),
    (0xFFC00000, 0x79000000, "STRH SCALED"),
    (0xFFC00000, 0x39400000, "LDRB SCALED"),
    (0xFFC00000, 0x39000000, "STRB SCALED"),
    # === LDRSW/LDRSH/LDRSB ===
    (0xFFC00000, 0xB9800000, "LDRSW"),
    (0xFFC00000, 0x79800000, "LDRSH 64"),
    (0xFFC00000, 0x79C00000, "LDRSH 32"),
    (0xFFC00000, 0x39800000, "LDRSB 64"),
    (0xFFC00000, 0x39C00000, "LDRSB 32"),
    # === LDR REGISTER OFFSET ===
    (0xFFE00C00, 0xF8600800, "LDR REG 64"),
    (0xFFE00C00, 0xB8600800, "LDR REG 32"),
    (0xFFE00C00, 0xF8200800, "STR REG 64"),
    (0xFFE00C00, 0xB8200800, "STR REG 32"),
    # === LDR LITERAL ===
    (0xFF000000, 0x58000000, "LDR LIT 64"),
    (0xFF000000, 0x18000000, "LDR LIT 32"),
    # === LDP/STP ===
    (0xFFC00000, 0xA9400000, "LDP 64"),
    (0xFFC00000, 0xA9000000, "STP 64"),
    (0xFFC00000, 0x29400000, "LDP 32"),
    (0xFFC00000, 0x29000000, "STP 32"),
    (0xFFC00000, 0xA8C00000, "LDP POST 64"),
    (0xFFC00000, 0xA8800000, "STP POST 64"),
    (0xFFC00000, 0xA9800000, "STP PRE 64"),
    (0xFFC00000, 0xA9C00000, "LDP PRE 64"),
    (0xFFC00000, 0x28C00000, "LDP POST 32"),
    (0xFFC00000, 0x28800000, "STP PRE 32"),
    # === LDXR/STXR/LDAXR (atomics) ===
    (0xFFFFFC00, 0x885F7C00, "LDXR 64"),
    (0xFFFFFC00, 0x485F7C00, "LDXR 32"),
    (0xFFFFFC00, 0x88DF7C00, "LDAXR 64"),
    (0xFFFFFC00, 0x48DF7C00, "LDAXR 32"),
    (0xFFE00FFF, 0x88007C00, "STXR 64"),
    (0xFFE00FFF, 0x48007C00, "STXR 32"),
    (0xFFE0FC00, 0xC8A07C00, "CAS 64"),
    (0xFFE0FC00, 0x88A07C00, "CAS 32"),
    (0xFFE0FC00, 0xC8E07C00, "CASAL 64"),
    (0xFFE0FC00, 0x88E07C00, "CASAL 32"),
    # === CSEL/CSINC/CSINV/CSNEG/CSET ===
    (0xFF800000, 0x9A800000, "CSEL 64"),
    (0xFF800000, 0x1A800000, "CSEL 32"),
    (0xFF800000, 0x9A800400, "CSINC 64"),
    (0xFF800000, 0x1A800400, "CSINC 32"),
    (0xFF800000, 0xDA800000, "CSINV 64"),
    (0xFF800000, 0x5A800000, "CSINV 32"),
    (0xFF800000, 0xDA800400, "CSNEG 64"),
    (0xFF800000, 0x5A800400, "CSNEG 32"),
    # === CCMP/CCMN ===
    (0xFF800010, 0xFA000000, "CCMP 64"),
    (0xFF800010, 0x7A000000, "CCMP 32"),
    # === AND/ORR/EOR IMMEDIATE ===
    (0xFF800000, 0x92000000, "AND IMM 64"),
    (0xFF800000, 0x12000000, "AND IMM 32"),
    (0xFF800000, 0xB2000000, "ORR IMM 64"),
    (0xFF800000, 0x32000000, "ORR IMM 32"),
    (0xFF800000, 0xD2000000, "EOR IMM 64"),
    (0xFF800000, 0x52000000, "EOR IMM 32"),
    # === CLZ/RBIT/REV ===
    (0xFFFFFC00, 0xDAC01000, "CLZ 64"),
    (0xFFFFFC00, 0x5AC01000, "CLZ 32"),
    (0xFFFFFC00, 0xDAC00400, "RBIT 64"),
    (0xFFFFFC00, 0x5AC00400, "RBIT 32"),
    (0xFFFFFC00, 0xDAC00800, "REV16 64"),
    (0xFFFFFC00, 0x5AC00800, "REV16 32"),
    (0xFFFFFC00, 0xDAC00C00, "REV32 64"),
    # === LDR REGISTER OFFSET (extended) ===
    (0xFFE00C00, 0x78600800, "LDRH REG"),
    (0xFFE00C00, 0x38600800, "LDRB REG"),
    (0xFFE00C00, 0x78200800, "STRH REG"),
    (0xFFE00C00, 0x38200800, "STRB REG"),
]

# ---------------------------------------------------------------------------
# ARM64 family identification for unhandled encodings
# Keyed on the top byte (bits 31-24) of the little-endian word.
# ---------------------------------------------------------------------------
_FAMILY_BY_TOP_BYTE: dict[int, str] = {
    0x4E: "AdvSIMD vector (NEON)",
    0x4F: "AdvSIMD modified-immediate / MOVI",
    0x0E: "AdvSIMD vector (NEON) 32-bit",
    0x0F: "AdvSIMD modified-immediate 32-bit",
    0x6E: "AdvSIMD vector (NEON) w/ Q",
    0x6F: "AdvSIMD modified-immediate Q",
    0x2E: "AdvSIMD vector (NEON) 2",
    0x2F: "AdvSIMD modified-immediate 2",
    0x5E: "SIMD scalar",
    0x7E: "SIMD scalar 2",
    0x1E: "Float scalar (FMOV/FADD/FMUL/...)",
    0x1F: "Float scalar fused (FMADD/FMSUB/...)",
    0x9E: "Float scalar 64-bit",
    0xD5: "System/Hint (MRS/MSR/NOP/CLREX/DMB/ISB/...)",
    0xD4: "Exception gen (BRK/HLT/SVC other encodings)",
    0x6D: "LDP/STP SIMD (Q-register)",
    0x6C: "LDP/STP SIMD (D-register post-index)",
    0xAD: "LDP SIMD 64-bit",
    0xAC: "LDP/STP SIMD variant",
    0xFC: "LDR/STR SIMD 128-bit",
    0xFD: "LDR/STR SIMD 64-bit scaled",
    0xBC: "LDR/STR SIMD 32-bit scaled",
    0x3C: "LDR/STR SIMD 8-bit",
    0x7C: "LDR/STR SIMD 16-bit",
    0xBD: "LDR SIMD 32-bit",
    0x3D: "LDR/STR SIMD 8-bit (post/pre)",
    0x7D: "SIMD load/store",
    0x0C: "AdvSIMD load/store multiple",
    0x4C: "AdvSIMD load/store multiple (Q)",
    0x08: "Load/Store exclusive pair",
    0x48: "Load/Store exclusive pair 64",
    0xC8: "Load/Store exclusive pair 128",
    0xD8: "Load register (literal) SIMD",
    0x58: "LDR literal 64 (also prefetch PRFM)",
    0xD9: "PRFM / cache hint",
    0xF8: "Load/Store register (unscaled) — extended",
    0xB8: "Load/Store register 32 extended",
    # GP register pair (LDP/STP): top byte encodes opc+mode+L
    # 0xA8 = post-index (bits 24:23 = 01), 0xA9 = signed-offset OR pre-index (bits 24:23 = 10/11)
    0xA8: "LDP/STP GP 64-bit post-index (A8xx — check STP/LDP POST patterns)",
    0xA9: "LDP/STP GP 64-bit signed-offset or pre-index (A9xx — possible missing STP/LDP PRE 64)",
    0x28: "LDP/STP GP 32-bit post-index",
    0x29: "LDP/STP GP 32-bit signed-offset or pre-index",
    0xC4: "SVE (Scalable Vector Extension)",
    0xC5: "SVE2",
    0x25: "SVE predicate",
    0x05: "SVE 2",
    0x45: "SVE 3",
    0xA5: "SVE load/store",
    0xE5: "SVE load/store 2",
    0x85: "SVE load/store 3",
}

# Broader family detection using top-nibble ranges for less common encodings
_FAMILY_RANGES: list[tuple[int, int, str]] = [
    # SVE encodings (top 3 bits = 001 + bit pattern)
    (0xE0, 0x20, "SVE (Scalable Vector Extension)"),
    # Crypto / SHA / AES
    (0xFF, 0xCE, "Crypto AES/SHA (AESE/AESD/SHA1/SHA256)"),
    (0xFF, 0x5E, "Crypto SHA / SIMD scalar"),
]


def _identify_family(word: int) -> str:
    """Return a human-readable family string for an unhandled word."""
    top_byte = (word >> 24) & 0xFF
    if top_byte in _FAMILY_BY_TOP_BYTE:
        return _FAMILY_BY_TOP_BYTE[top_byte]

    # Try range-based matching
    for mask, val, name in _FAMILY_RANGES:
        if (top_byte & mask) == val:
            return name

    # Generic decode hints using bit fields from the ARM64 encoding spec
    op0 = (word >> 29) & 0x7          # bits 31:29
    op1 = (word >> 25) & 0xF          # bits 28:25

    # Encoding class from op0/op1 table (ARM DDI 0487)
    if op0 in (0b000, 0b001):
        return "Reserved / UNALLOCATED"
    if op1 == 0b0000:
        return "Reserved"
    if op1 in (0b1000, 0b1001):
        return "Data processing — immediate"
    if op1 in (0b1010, 0b1011):
        return "Branch / exception / system"
    if op1 in (0b0100, 0b0110, 0b0101, 0b0111):
        return "Load/Store"
    if op1 in (0b0001, 0b0011):
        return "Data processing — register"

    return f"Unknown (op0={op0:#05b} op1={op1:#06b})"


# ---------------------------------------------------------------------------
# ELF parsing helpers
# ---------------------------------------------------------------------------
ELF_MAGIC = b"\x7fELF"

# ELF64 program header field offsets (relative to start of phdr)
_PH_TYPE   = 0   # Elf64_Word  (4 bytes)
_PH_FLAGS  = 4   # Elf64_Word  (4 bytes)
_PH_OFFSET = 8   # Elf64_Off   (8 bytes)
_PH_FILESZ = 32  # Elf64_Xword (8 bytes)

PT_LOAD  = 1
PF_X     = 1  # execute permission flag


def _iter_elf_exec_bytes(data: bytes) -> Iterable[bytes]:
    """
    Yield raw byte slices for every executable PT_LOAD segment in an ELF64
    binary.  Raises ValueError if the data does not look like a valid ELF64.
    """
    if len(data) < 64:
        raise ValueError("File too short to be ELF64")
    if data[:4] != ELF_MAGIC:
        raise ValueError("Not an ELF file")
    ei_class = data[4]
    if ei_class != 2:
        raise ValueError(f"Only ELF64 supported (EI_CLASS={ei_class})")
    ei_data = data[5]
    if ei_data != 1:
        raise ValueError(f"Only little-endian ELF supported (EI_DATA={ei_data})")

    # ELF64 header fields we need
    (e_phoff,) = struct.unpack_from("<Q", data, 32)
    (e_phentsize,) = struct.unpack_from("<H", data, 54)
    (e_phnum,) = struct.unpack_from("<H", data, 56)

    if e_phentsize < 56:
        raise ValueError(f"Unexpected e_phentsize={e_phentsize}")

    for i in range(e_phnum):
        base = e_phoff + i * e_phentsize
        if base + e_phentsize > len(data):
            break
        p_type,  = struct.unpack_from("<I", data, base + _PH_TYPE)
        p_flags, = struct.unpack_from("<I", data, base + _PH_FLAGS)
        p_offset,= struct.unpack_from("<Q", data, base + _PH_OFFSET)
        p_filesz,= struct.unpack_from("<Q", data, base + _PH_FILESZ)

        if p_type == PT_LOAD and (p_flags & PF_X):
            seg_end = p_offset + p_filesz
            if seg_end > len(data):
                seg_end = len(data)
            yield data[p_offset:seg_end]


def extract_instructions(path: str | Path) -> list[int]:
    """
    Read a binary file and return every aligned 4-byte word as an int.

    For ELF files, only words from executable segments are returned.
    For raw files, every aligned word in the file is returned.
    """
    data = Path(path).read_bytes()
    words: list[int] = []

    is_elf = data[:4] == ELF_MAGIC

    if is_elf:
        try:
            segments = list(_iter_elf_exec_bytes(data))
        except ValueError as exc:
            print(f"[warn] ELF parse failed ({exc}); falling back to raw mode", file=sys.stderr)
            segments = [data]
    else:
        segments = [data]

    for seg in segments:
        # Align to 4-byte boundary by trimming leading bytes if needed
        # (segment offsets should already be aligned for ARM64)
        n = len(seg) // 4
        for i in range(n):
            (word,) = struct.unpack_from("<I", seg, i * 4)
            words.append(word)

    return words


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def is_handled(word: int) -> bool:
    """Return True if *word* matches at least one entry in KNOWN_PATTERNS."""
    for mask, pattern, _ in KNOWN_PATTERNS:
        if (word & mask) == pattern:
            return True
    return False


def audit(
    path: str | Path,
    top_n: int = 20,
) -> None:
    """Run the gap audit and print a formatted report."""
    path = Path(path)
    print(f"\n=== Gap Audit: {path.name} ===")

    instructions = extract_instructions(path)
    total = len(instructions)

    if total == 0:
        print("No instructions found.")
        return

    unhandled_counter: Counter[int] = Counter()
    n_handled = 0

    for word in instructions:
        if is_handled(word):
            n_handled += 1
        else:
            unhandled_counter[word] += 1

    n_unhandled = total - n_handled
    handled_pct = 100.0 * n_handled / total
    unhandled_pct = 100.0 * n_unhandled / total

    print(f"Total instructions analyzed: {total:,}")
    print(f"Handled   (matched):  {n_handled:,} ({handled_pct:.1f}%)")
    print(f"Unhandled (gap):      {n_unhandled:,} ({unhandled_pct:.1f}%)")
    print(f"Unique unhandled encodings: {len(unhandled_counter):,}")

    if not unhandled_counter:
        print("\nNo gaps found — all encodings are handled.")
        return

    # -----------------------------------------------------------------------
    # Top-N report
    # -----------------------------------------------------------------------
    top = unhandled_counter.most_common(top_n)

    col_enc   = max(len("Encoding"),   10)
    col_top   = max(len("Top byte"),    8)
    col_count = max(len("Count"),       6)
    col_fam   = max(len("Likely family"), 40)

    header = (
        f"{'Count':>{col_count}}  "
        f"{'Encoding':<{col_enc}}  "
        f"{'Top byte':<{col_top}}  "
        f"Likely family"
    )
    print(f"\nTop {min(top_n, len(top))} unhandled encodings:")
    print(f"  {header}")
    print(f"  {'-' * (col_count + col_enc + col_top + col_fam + 8)}")

    for word, count in top:
        top_byte = (word >> 24) & 0xFF
        family   = _identify_family(word)
        enc_str  = f"0x{word:08X}"
        tb_str   = f"0x{top_byte:02X}"
        print(
            f"  {count:>{col_count},}  "
            f"{enc_str:<{col_enc}}  "
            f"{tb_str:<{col_top}}  "
            f"{family}"
        )

    # -----------------------------------------------------------------------
    # Family roll-up summary
    # -----------------------------------------------------------------------
    family_totals: Counter[str] = Counter()
    for word, count in unhandled_counter.items():
        family_totals[_identify_family(word)] += count

    print(f"\nUnhandled by family ({len(family_totals)} distinct families):")
    print(f"  {'Count':>8}  {'Pct of gap':>10}  Family")
    print(f"  {'-' * 70}")
    for family, count in family_totals.most_common():
        pct = 100.0 * count / n_unhandled if n_unhandled else 0.0
        print(f"  {count:>8,}  {pct:>9.1f}%  {family}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _path  = sys.argv[1] if len(sys.argv) > 1 else "demos/busybox.elf"
    _top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    audit(_path, top_n=_top_n)
