"""Shared helpers for GPU trace inspection and ARM64 disassembly."""

from __future__ import annotations

from collections.abc import Sequence


def format_nzcv(flags: int) -> str:
    """Format NZCV flags as a compact four-character string."""
    return (
        f"{'N' if flags & 8 else '.'}"
        f"{'Z' if flags & 4 else '.'}"
        f"{'C' if flags & 2 else '.'}"
        f"{'V' if flags & 1 else '.'}"
    )


def disassemble_instruction(inst: int) -> str:
    """Disassemble a single ARM64 instruction to human-readable text."""
    top8 = (inst >> 24) & 0xFF
    rd = inst & 0x1F
    rn = (inst >> 5) & 0x1F
    rm = (inst >> 16) & 0x1F
    sf = (inst >> 31) & 1
    reg_prefix = 'x' if sf else 'w'

    if inst == 0xD503201F:
        return "nop"
    if (inst & 0xFFE0001F) == 0xD4400000:
        return "hlt"
    if (inst & 0xFFE0001F) == 0xD4000001:
        imm = (inst >> 5) & 0xFFFF
        return f"svc #0x{imm:x}"
    if (inst & 0xFFFFFC1F) == 0xD65F0000:
        return f"ret x{rn}" if rn != 30 else "ret"
    if top8 == 0xD6:
        op = (inst >> 21) & 0x3
        return f"{'blr' if op == 1 else 'br'} x{rn}"
    if (inst >> 26) in (0b000101, 0b100101):
        is_bl = (inst >> 31) & 1
        imm26 = inst & 0x3FFFFFF
        if imm26 & (1 << 25):
            imm26 -= (1 << 26)
        return f"{'bl' if is_bl else 'b'} #{imm26 * 4:+d}"
    if top8 == 0x54:
        conds = ['eq', 'ne', 'cs', 'cc', 'mi', 'pl', 'vs', 'vc',
                 'hi', 'ls', 'ge', 'lt', 'gt', 'le', 'al', 'nv']
        cond = inst & 0xF
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 & (1 << 18):
            imm19 -= (1 << 19)
        return f"b.{conds[cond]} #{imm19 * 4:+d}"
    if top8 in (0x34, 0x35, 0xB4, 0xB5):
        is_nz = top8 & 1
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 & (1 << 18):
            imm19 -= (1 << 19)
        return f"cb{'nz' if is_nz else 'z'} {reg_prefix}{rd}, #{imm19 * 4:+d}"
    if top8 in (0x36, 0x37, 0xB6, 0xB7):
        is_nz = top8 & 1
        b5 = (inst >> 31) & 1
        b40 = (inst >> 19) & 0x1F
        bit = (b5 << 5) | b40
        imm14 = (inst >> 5) & 0x3FFF
        if imm14 & (1 << 13):
            imm14 -= (1 << 14)
        return f"tb{'nz' if is_nz else 'z'} {reg_prefix}{rd}, #{bit}, #{imm14 * 4:+d}"
    if top8 in (0xD2, 0x52):
        imm16 = (inst >> 5) & 0xFFFF
        hw = (inst >> 21) & 3
        shift = f", lsl #{hw * 16}" if hw else ""
        return f"movz {reg_prefix}{rd}, #0x{imm16:x}{shift}"
    if top8 in (0xF2, 0x72):
        imm16 = (inst >> 5) & 0xFFFF
        hw = (inst >> 21) & 3
        shift = f", lsl #{hw * 16}" if hw else ""
        return f"movk {reg_prefix}{rd}, #0x{imm16:x}{shift}"
    if top8 in (0x92, 0x12):
        imm16 = (inst >> 5) & 0xFFFF
        hw = (inst >> 21) & 3
        shift = f", lsl #{hw * 16}" if hw else ""
        return f"movn {reg_prefix}{rd}, #0x{imm16:x}{shift}"
    if top8 in (0x91, 0x11):
        imm12 = (inst >> 10) & 0xFFF
        shift = (inst >> 22) & 1
        dest = 'sp' if rd == 31 else f'{reg_prefix}{rd}'
        src = 'sp' if rn == 31 else f'{reg_prefix}{rn}'
        shift_suffix = ", lsl #12" if shift else ""
        return f"add {dest}, {src}, #0x{imm12:x}{shift_suffix}"
    if top8 in (0xD1, 0x51):
        imm12 = (inst >> 10) & 0xFFF
        shift = (inst >> 22) & 1
        dest = 'sp' if rd == 31 else f'{reg_prefix}{rd}'
        src = 'sp' if rn == 31 else f'{reg_prefix}{rn}'
        shift_suffix = ", lsl #12" if shift else ""
        return f"sub {dest}, {src}, #0x{imm12:x}{shift_suffix}"
    if top8 in (0xB1, 0x31):
        imm12 = (inst >> 10) & 0xFFF
        if rd == 31:
            return f"cmn {reg_prefix}{rn}, #0x{imm12:x}"
        return f"adds {reg_prefix}{rd}, {reg_prefix}{rn}, #0x{imm12:x}"
    if top8 in (0xF1, 0x71):
        imm12 = (inst >> 10) & 0xFFF
        if rd == 31:
            return f"cmp {reg_prefix}{rn}, #0x{imm12:x}"
        return f"subs {reg_prefix}{rd}, {reg_prefix}{rn}, #0x{imm12:x}"
    if top8 in (0x8B, 0x0B):
        return f"add {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
    if top8 in (0xCB, 0x4B):
        return f"sub {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
    if top8 in (0xAB, 0x2B):
        if rd == 31:
            return f"cmn {reg_prefix}{rn}, {reg_prefix}{rm}"
        return f"adds {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
    if top8 in (0xEB, 0x6B):
        if rd == 31:
            return f"cmp {reg_prefix}{rn}, {reg_prefix}{rm}"
        return f"subs {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
    if top8 in (0x8A, 0x0A):
        n_bit = (inst >> 21) & 1
        return f"{'bic' if n_bit else 'and'} {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
    if top8 in (0xAA, 0x2A):
        n_bit = (inst >> 21) & 1
        if not n_bit and rn == 31:
            return f"mov {reg_prefix}{rd}, {reg_prefix}{rm}"
        return f"{'orn' if n_bit else 'orr'} {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
    if top8 in (0xCA, 0x4A):
        n_bit = (inst >> 21) & 1
        return f"{'eon' if n_bit else 'eor'} {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
    if top8 in (0xEA, 0x6A):
        n_bit = (inst >> 21) & 1
        op = 'bics' if n_bit else 'ands'
        if rd == 31:
            return f"tst {reg_prefix}{rn}, {reg_prefix}{rm}"
        return f"{op} {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
    if top8 in (0xB2, 0x32):
        return f"orr {reg_prefix}{rd}, {reg_prefix}{rn}, #<bitmask>"
    if top8 == 0xF9:
        is_ld = (inst >> 22) & 1
        imm12 = ((inst >> 10) & 0xFFF) * 8
        src = 'sp' if rn == 31 else f'x{rn}'
        return f"{'ldr' if is_ld else 'str'} x{rd}, [{src}, #0x{imm12:x}]"
    if top8 == 0xB9:
        is_ld = (inst >> 22) & 1
        imm12 = ((inst >> 10) & 0xFFF) * 4
        src = 'sp' if rn == 31 else f'x{rn}'
        return f"{'ldr' if is_ld else 'str'} w{rd}, [{src}, #0x{imm12:x}]"
    if top8 == 0x79:
        is_ld = (inst >> 22) & 1
        imm12 = ((inst >> 10) & 0xFFF) * 2
        src = 'sp' if rn == 31 else f'x{rn}'
        return f"{'ldrh' if is_ld else 'strh'} w{rd}, [{src}, #0x{imm12:x}]"
    if top8 == 0x39:
        is_ld = (inst >> 22) & 1
        imm12 = (inst >> 10) & 0xFFF
        src = 'sp' if rn == 31 else f'x{rn}'
        return f"{'ldrb' if is_ld else 'strb'} w{rd}, [{src}, #0x{imm12:x}]"
    if top8 in (0xF8, 0xB8, 0x78, 0x38):
        sizes = {0xF8: ('x', 8), 0xB8: ('w', 4), 0x78: ('h', 2), 0x38: ('b', 1)}
        reg_kind, _size = sizes[top8]
        is_ld = (inst >> 22) & 1
        op = 'ldr' if is_ld else 'str'
        if reg_kind == 'h':
            op = 'ldrh' if is_ld else 'strh'
        elif reg_kind == 'b':
            op = 'ldrb' if is_ld else 'strb'
        src = 'sp' if rn == 31 else f'x{rn}'
        if (inst >> 21) & 1:
            dst_reg = reg_kind if reg_kind in ('x', 'w') else 'w'
            return f"{op} {dst_reg}{rd}, [{src}, x{rm}]"
        imm9 = (inst >> 12) & 0x1FF
        if imm9 & 0x100:
            imm9 -= 0x200
        dst_reg = reg_kind if reg_kind in ('x', 'w') else 'w'
        return f"{op} {dst_reg}{rd}, [{src}, #{imm9}]"
    if top8 in (0xA9, 0x29, 0x69, 0xE9, 0xA8, 0x28, 0x68, 0xE8):
        is_ld = (inst >> 22) & 1
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 & 0x40:
            imm7 -= 0x80
        scale = 8 if sf else 4
        src = 'sp' if rn == 31 else f'x{rn}'
        return f"{'ldp' if is_ld else 'stp'} {reg_prefix}{rd}, {reg_prefix}{rt2}, [{src}, #{imm7 * scale}]"
    if ((inst >> 24) & 0x1F) == 0x10:
        is_page = (inst >> 31) & 1
        return f"{'adrp' if is_page else 'adr'} x{rd}, <label>"
    if top8 in (0x93, 0x13):
        if ((inst >> 21) & 0x7) == 0b100:
            imms = (inst >> 10) & 0x3F
            return f"extr {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}, #{imms}"
        return f"sbfm {reg_prefix}{rd}, {reg_prefix}{rn}, #<immr>, #<imms>"
    if top8 in (0xD3, 0x53):
        return f"ubfm {reg_prefix}{rd}, {reg_prefix}{rn}, #<immr>, #<imms>"
    if top8 in (0xB3, 0x33):
        return f"bfm {reg_prefix}{rd}, {reg_prefix}{rn}, #<immr>, #<imms>"
    if top8 in (0x9B, 0x1B):
        ra = (inst >> 10) & 0x1F
        o0 = (inst >> 15) & 1
        op31 = (inst >> 21) & 0x7
        if op31 in (0b001, 0b101):
            return f"{'smull' if op31 == 1 else 'umull'} x{rd}, w{rn}, w{rm}"
        if o0 == 0 and ra == 31:
            return f"mul {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
        if o0 == 1 and ra == 31:
            return f"mneg {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
        return f"{'msub' if o0 else 'madd'} {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}, {reg_prefix}{ra}"
    if top8 in (0x9A, 0x1A):
        bits_23_21 = (inst >> 21) & 0x7
        if bits_23_21 == 0b000:
            return f"adc {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
        if bits_23_21 == 0b110:
            op2 = (inst >> 10) & 0x3F
            if op2 == 2:
                return f"udiv {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
            if op2 == 3:
                return f"sdiv {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
            if 8 <= op2 <= 11:
                names = ['lsl', 'lsr', 'asr', 'ror']
                return f"{names[op2 & 3]} {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
        if bits_23_21 == 0b100:
            conds = ['eq', 'ne', 'cs', 'cc', 'mi', 'pl', 'vs', 'vc',
                     'hi', 'ls', 'ge', 'lt', 'gt', 'le', 'al', 'nv']
            cond = (inst >> 12) & 0xF
            op = 'csinc' if ((inst >> 10) & 1) else 'csel'
            return f"{op} {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}, {conds[cond]}"
    if top8 in (0xDA, 0x5A):
        bits_23_21 = (inst >> 21) & 0x7
        if bits_23_21 == 0b000:
            return f"sbc {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}"
        if bits_23_21 == 0b110:
            op2 = (inst >> 10) & 0x3F
            if op2 == 0:
                return f"rbit {reg_prefix}{rd}, {reg_prefix}{rn}"
            if op2 == 4:
                return f"clz {reg_prefix}{rd}, {reg_prefix}{rn}"
            if op2 == 5:
                return f"cls {reg_prefix}{rd}, {reg_prefix}{rn}"
            return f"rev {reg_prefix}{rd}, {reg_prefix}{rn}"
        if bits_23_21 == 0b100:
            conds = ['eq', 'ne', 'cs', 'cc', 'mi', 'pl', 'vs', 'vc',
                     'hi', 'ls', 'ge', 'lt', 'gt', 'le', 'al', 'nv']
            cond = (inst >> 12) & 0xF
            op = 'csneg' if ((inst >> 10) & 1) else 'csinv'
            return f"{op} {reg_prefix}{rd}, {reg_prefix}{rn}, {reg_prefix}{rm}, {conds[cond]}"
    if top8 in (0xBA, 0x3A, 0xFA, 0x7A):
        return f"ccmp/ccmn {reg_prefix}{rn}, {reg_prefix}{rm}, #<nzcv>, <cond>"
    if top8 in (0x0E, 0x4E, 0x2E, 0x6E):
        return "simd <vector op>"
    if top8 in (0x0F, 0x4F, 0x2F, 0x6F):
        return "movi/mvni <simd imm>"
    if top8 in (0xFD, 0xBD, 0x7D, 0x3D, 0xFC, 0xBC, 0x7C, 0x3C):
        return "ldr/str <simd reg>, [...]"
    if top8 in (0xAD, 0x2D, 0x6D, 0xED, 0xAC, 0x2C, 0x6C, 0xEC):
        return "ldp/stp <simd pair>, [...]"
    if top8 in (0x08, 0x48, 0x88, 0xC8):
        return "ldxr/stxr ..."
    if top8 == 0xD5:
        return "msr/mrs <system>"
    if top8 in (0x18, 0x58, 0x98):
        return f"ldr {reg_prefix}{rd}, <literal>"
    return f".word 0x{inst:08x}"


def render_trace_table(
    trace: Sequence[Sequence[int]],
    limit: int | None = 30,
    indent: str = "  ",
) -> str:
    """Render a compact disassembly table for a trace window."""
    rows = list(trace if limit is None else trace[-limit:])
    if not rows:
        return ""

    lines = [
        f"{indent}{'PC':<12s} {'Hex':>10s}  {'Assembly':<40s} {'x0':>12s}",
        f"{indent}{'─' * 12} {'─' * 10}  {'─' * 40} {'─' * 12}",
    ]
    for entry in rows:
        pc = entry[0]
        inst = entry[1]
        x0 = entry[2] if len(entry) > 2 else 0
        flags = entry[6] if len(entry) > 6 else 0
        lines.append(
            f"{indent}0x{pc:08X}  {inst:08X}  {disassemble_instruction(inst):<40s} "
            f"x0=0x{x0 & 0xFFFFFFFFFFFFFFFF:X} [{format_nzcv(flags)}]"
        )
    return "\n".join(lines)
