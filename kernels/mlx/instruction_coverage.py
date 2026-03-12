#!/usr/bin/env python3
"""Backend-neutral ARM64 instruction coverage helpers."""

from typing import Callable


def is_known_instruction(inst: int, top: int | None = None) -> bool:
    """Check if an instruction is handled by the Metal kernel decode table."""
    if top is None:
        top = (inst >> 24) & 0xFF

    # HLT
    if (inst & 0xFFE0001F) == 0xD4400000:
        return True
    # SVC
    if (inst & 0xFFE0001F) == 0xD4000001:
        return True
    # NOP, DMB, DSB, ISB, CLREX
    if inst in (0xD503201F, 0xD50330BF, 0xD5033F9F, 0xD5033FDF, 0xD503305F):
        return True

    # Data processing — immediate
    if top in (0xD2, 0xD3, 0xF2, 0xF3, 0x52, 0x72):  # MOVZ, MOVK
        return True
    if top in (0x92, 0x12):  # MOVN
        return True
    if top in (0x91, 0x11, 0xB1, 0x31):  # ADD/ADDS imm
        return True
    if top in (0xD1, 0x51, 0xF1, 0x71):  # SUB/SUBS imm
        return True
    if top in (0x92, 0x12, 0xB2, 0x32):  # AND/ANDS imm (logical imm)
        return True
    if top in (0xD3, 0x53):  # UBFM/SBFM/BFM
        return True
    if top == 0x93:  # SBFM 64-bit / EXTR
        return True
    if top == 0x33:  # BFM 32-bit
        return True

    # Branches
    if top in (0x14, 0x15, 0x16, 0x17):  # B
        return True
    if top in (0x94, 0x95, 0x96, 0x97):  # BL
        return True
    if top == 0x54:  # B.cond
        return True
    if (inst & 0xFF000000) in (0xB4000000, 0xB5000000, 0x34000000, 0x35000000):  # CBZ/CBNZ
        return True
    if (inst & 0x7F000000) in (0x36000000, 0x37000000):  # TBZ/TBNZ
        return True
    if (inst & 0xFFFFFC1F) in (0xD61F0000, 0xD63F0000, 0xD65F0000):  # BR/BLR/RET
        return True

    # Load/Store
    if top in (0xF9, 0xB9, 0x39, 0x79):  # LDR/STR unsigned imm
        return True
    if top in (0xF8, 0xB8, 0x38, 0x78):  # LDR/STR reg offset / pre/post index
        return True
    if top in (0xA9, 0xA8, 0x29, 0x28):  # STP/LDP
        return True
    if top in (0x18, 0x1C, 0x58, 0x98, 0x9C, 0xD8):  # LDR literal / PRFM literal
        return True

    # Data processing — register
    if top in (0x8B, 0x0B, 0xAB, 0x2B):  # ADD reg
        return True
    if top in (0xCB, 0x4B, 0xEB, 0x6B):  # SUB/SUBS reg
        return True
    if top in (0x8A, 0x0A, 0xAA, 0x2A):  # AND/ORR reg
        return True
    if top in (0xCA, 0x4A, 0xEA, 0x6A):  # EOR/ANDS reg
        return True
    if top in (0x9A, 0x1A, 0xDA, 0x5A):  # CSEL/ADC/SBC
        return True
    if top in (0x9B, 0x1B):  # MUL/MADD/MSUB/SMULH/UMULH
        return True

    # Division / shifts
    if (inst & 0xFF200000) in (0x9AC00000, 0x1AC00000):  # SDIV/UDIV / shift-reg family
        return True

    # Bit manipulation
    if (inst & 0xFF200000) == 0xDAC00000:  # CLZ/RBIT/REV
        return True
    if (inst & 0x7F200000) == 0x5AC00000:  # 32-bit bit ops
        return True

    # ADR/ADRP
    if (inst & 0x9F000000) in (0x10000000, 0x90000000):
        return True

    # MRS/MSR
    if (inst & 0xFFF00000) in (0xD5300000, 0xD5100000):
        return True

    # LDXR/STXR
    if (inst & 0xFF200000) in (0xC8400000, 0xC8000000, 0x88400000, 0x88000000):
        return True

    # SIMD/FP pair load/store
    if (inst & 0xFFC00000) in (0xAD000000, 0xAD400000, 0x6D000000, 0x6D400000):
        return True

    return False


def analyze_instruction_coverage(
    read_memory: Callable[[int, int], bytes],
    text_base: int,
    text_size: int,
) -> list[dict]:
    """Analyze a code region for instruction coverage gaps."""
    data = read_memory(text_base, text_size)
    unknowns: dict[int, dict] = {}

    for off in range(0, len(data) - 3, 4):
        inst = int.from_bytes(data[off:off + 4], "little")
        if inst == 0:
            continue

        top = (inst >> 24) & 0xFF
        class_key = inst >> 21

        if not is_known_instruction(inst, top):
            unknowns.setdefault(class_key, {
                "opcode": inst,
                "pc": text_base + off,
                "opcode_hex": f"0x{inst:08X}",
                "top_byte": f"0x{top:02X}",
                "class_bits": f"0x{class_key:03X}",
            })

    return list(unknowns.values())
