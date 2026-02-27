#!/usr/bin/env python3
"""Analyze DOOM binary to find all required ARM64 instructions"""

import sys
from pathlib import Path
import struct

# ELF header parsing
def parse_elf(filename):
    with open(filename, 'rb') as f:
        # Read ELF header
        ident = f.read(16)
        if ident[:4] != b'\x7fELF':
            print(f"Not an ELF file: {filename}")
            return None, None

        # Check if 64-bit
        is_64bit = ident[4] == 2
        if not is_64bit:
            print(f"Not 64-bit: {filename}")
            return None, None

        # Read rest of ELF header
        f.read(12)  # skip to section header offset
        section_header_offset = struct.unpack('<Q', f.read(8))[0]

        # Read program headers and section headers
        f.seek(section_header_offset)

        # Read section headers
        sections = []
        while True:
            try:
                name_offset = struct.unpack('<I', f.read(4))[0]
                sh_type = struct.unpack('<I', f.read(4))[0]
                sh_flags = struct.unpack('<Q', f.read(8))[0]
                sh_addr = struct.unpack('<Q', f.read(8))[0]
                sh_offset = struct.unpack('<Q', f.read(8))[0]
                sh_size = struct.unpack('<Q', f.read(8))[0]
                f.read(8)  # sh_link, sh_info
                sh_entsize = struct.unpack('<Q', f.read(8))[0]
                f.read(8)  # padding

                if sh_type == 1:  # SHT_PROGBITS
                    sections.append({
                        'offset': sh_offset,
                        'addr': sh_addr,
                        'size': sh_size,
                        'name_offset': name_offset
                    })
            except:
                break

        # Read section header string table
        f.seek(0)  # Go back to start
        f.read(32)  # Skip e_ident
        f.read(8)  # Skip e_type, e_machine
        e_shstrndx = struct.unpack('<H', f.read(2))[0]

        # Read section names
        sections_data = []
        for section in sections:
            f.seek(section['offset'])
            data = f.read(section['size'])
            sections_data.append((section, data))

        return sections_data

def decode_arm64_instruction(inst):
    """Decode ARM64 instruction and return mnemonic info"""
    op = (inst >> 24) & 0xFF

    # Common instruction patterns
    instructions = {
        # Data processing - immediate
        0xD100: 'SUBS (immediate)',
        0xF100: 'SUBS (immediate)',
        0x9100: 'ADD (immediate)',
        0x1100: 'ADDS (immediate)',
        0xD200: 'MOVZ',  # Actually 0xD2 is MOVZ
        0x5200: 'MOVZ',
        0x7200: 'MOVZ',
        0xF200: 'MOVZ',

        # Branches
        0x1400: 'B (unconditional)',
        0x1700: 'B (unconditional)',
        0x5400: 'B.cond',
        0x5401: 'B.cond',
        0xD600: 'RET',
        0xD61F: 'RET',
        0xD63F: 'RET',
        0x3400: 'CBZ',
        0x3500: 'CBNZ',
        0x3600: 'TBZ',
        0x3700: 'TBNZ',

        # Load/Store
        0xF940: 'LDR (literal)',
        0x1800: 'LDR (literal)',
        0x5800: 'LDR (literal)',
        0x9000: 'ADRP',
        0x9100: 'ADD (immediate)',  # ADRP is similar to ADD
        0xF900: 'STR (unsigned offset)',
        0xF940: 'LDR (unsigned offset)',
        0xB800: 'STRH (unsigned offset)',
        0x7800: 'STRH (unsigned offset)',
        0x7840: 'LDRH (unsigned offset)',
        0xB840: 'LDRH (unsigned offset)',
        0x3800: 'STRB (unsigned offset)',
        0x3C00: 'LDURB',
        0x3840: 'LDRB (unsigned offset)',
        0x3C40: 'LDRB (unsigned offset)',
        0xF840: 'LDUR',
        0xF800: 'STR (unsigned offset)',
        0x3900: 'LDRSB (unsigned offset)',
        0x3940: 'LDRSB (unsigned offset)',
        0x7900: 'LDRSH (unsigned offset)',
        0x7940: 'LDRSH (unsigned offset)',
        0xB900: 'STR (immediate)',
        0xB940: 'LDR (immediate)',
        0x3D00: 'LDRSW (literal)',
        0x9A00: 'ORR (shifted register)',
        0xAA00: 'ORR (shifted register)',
        0x0A00: 'AND (shifted register)',
        0x8A00: 'AND (shifted register)',
        0xCA00: 'EOR (shifted register)',

        # Data processing - register
        0x8B00: 'ADD (shifted register)',
        0x4B00: 'SUB (shifted register)',
        0x6B00: 'SUBS (shifted register)',
        0xCB00: 'SUBS (shifted register)',
        0x1A00: 'ADC (shifted register)',
        0x5A00: 'SBC (shifted register)',
        0x3A00: 'ADC (shifted register)',
        0x7A00: 'SBC (shifted register)',
        0x9A00: 'ORR (shifted register)',
        0xAA00: 'ORR (shifted register)',
        0x0A00: 'AND (shifted register)',
        0x8A00: 'AND (shifted register)',
        0xCA00: 'EOR (shifted register)',
        0xEA00: 'EON (shifted register)',

        # Logical immediate
        0x1200: 'AND (immediate)',
        0x3200: 'ORR (immediate)',
        0x5200: 'EOR (immediate)',
        0x7200: 'ANDS (immediate)',

        # Bitfield
        0x1300: 'SBFM',
        0x3300: 'BFM',
        0x5300: 'UBFM',

        # Extract
        0x3400: 'EXTR',

        # Compare and branch
        0x3400: 'CBZ',
        0x3500: 'CBNZ',
        0x3600: 'TBZ',
        0x3700: 'TBNZ',

        # Conditional branch
        0x5400: 'B.cond',

        # Unconditional branch
        0x1400: 'B',
        0x1700: 'B',

        # Branch with link
        0x9400: 'BL',

        # System
        0xD500: 'System',
        0xD503: 'NOP',
        0xD508: 'System',
        0xD538: 'System',

        # Multiply
        0x1B00: 'MADD',
        0x9B00: 'MADD',
        0x1F00: 'MSUB',
        0x9F00: 'MSUB',

        # Divide
        0x1AC0: 'SDIV',
        0x9AC0: 'SDIV',
        0x1A80: 'UDIV',
        0x9A80: 'UDIV',

        # PAuth
        0xD500: 'PAuth',

        # Load pair
        0x2800: 'STP',
        0x2840: 'LDP',
        0x2940: 'LDP',
        0x2980: 'LDPSW',

        # Load/store exclusive
        0x0800: 'STXR',
        0x0840: 'STXR',
        0x0880: 'STXR',
        0x4800: 'LDXR',
        0x4840: 'LDXR',
        0x4880: 'LDXR',

        # Atomic
        0x3800: 'LDADDAL',

        # Barrier
        0xD503: 'DMB',
        0xD503: 'DSB',
        0xD503: 'ISB',

        # Hint
        0xD503: 'HINT',

        # DP (2 source)
        0x1AC0: 'UDIV',
        0x1A00: 'DIV',

        # DP (1 source)
        0x5AC0: 'RBIT',

        # FP
        0x1E00: 'FP data',
    }

    # Check for instruction by top byte or pattern
    if op in instructions:
        return instructions[op]

    # Check for patterns
    if (op & 0x1F) == 0x0D:  # B from 0x1400 range
        return 'B (unconditional)'
    if (op & 0x7F) == 0x04:  # B from 0x1400 range
        return 'B (unconditional)'
    if op == 0xD4:
        return 'System (DCPS1/2/3)'
    if op == 0xD6:
        return 'RET/BR/BLR'
    if op == 0x97:
        return 'BL'

    # More specific patterns
    if op == 0xD6:
        bits_20_16 = (inst >> 16) & 0x1F
        if bits_20_16 == 0x1F:
            return 'RET'

    # Check by opcode ranges
    if 0x00 <= op <= 0x0F:
        return 'Unknown/Data processing'
    if 0x10 <= op <= 0x1F:
        return 'Unknown/Data processing'
    if 0x28 <= op <= 0x3F:
        return 'Load/Store pair'
    if 0x38 <= op <= 0x3F:
        return 'Load/Store exclusive'
    if 0x48 <= op <= 0x4F:
        return 'Load/store exclusive'
    if 0x68 <= op <= 0x6F:
        return 'Load/store'
    if 0x78 <= op <= 0x7F:
        return 'Load/store'
    if 0x88 <= op <= 0x8F:
        return 'Load/store (pair)'
    if 0x98 <= op <= 0x9F:
        return 'Load/store (register)'
    if 0xA8 <= op <= 0xBF:
        return 'Load/store (register)'
    if 0xC0 <= op <= 0xFF:
        return 'Load/store (register)'

    return f'Unknown (0x{op:02X})'

def main():
    doom_path = Path('/Users/bobbyprice/projects/KVRM/kvrm-cpu/doom_benchmark.elf')
    if not doom_path.exists():
        doom_path = Path('/Users/bobbyprice/projects/KVRM/kvrm-cpu/arm64_doom/doom_neural.elf')

    print(f"Analyzing: {doom_path}")
    print("=" * 80)

    sections = parse_elf(doom_path)
    if not sections:
        print("Failed to parse ELF")
        return

    all_instructions = []
    instruction_counts = {}

    for section, data in sections:
        print(f"\nSection at 0x{section['addr']:08X}, size={section['size']:,} bytes")

        # Extract instructions (4 bytes each)
        inst_count = min(10000, len(data) // 4)  # Limit to first 10k instructions
        for i in range(inst_count):
            offset = i * 4
            inst_bytes = data[offset:offset+4]
            if len(inst_bytes) < 4:
                break

            inst = struct.unpack('<I', inst_bytes)[0]

            # Skip zeros and invalid instructions
            if inst == 0:
                continue

            op = (inst >> 24) & 0xFF
            mnemonic = decode_arm64_instruction(inst)

            all_instructions.append((inst, mnemonic))
            instruction_counts[mnemonic] = instruction_counts.get(mnemonic, 0) + 1

    # Print summary
    print("\n" + "=" * 80)
    print("INSTRUCTION SUMMARY")
    print("=" * 80)

    # Sort by count
    sorted_insts = sorted(instruction_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTotal unique instruction types: {len(sorted_insts)}")
    print(f"Total instructions analyzed: {sum(instruction_counts.values()):,}")

    print("\nTop 50 instructions by frequency:")
    for mnemonic, count in sorted_insts[:50]:
        print(f"  {mnemonic:30s}: {count:6d}")

    # List all unique opcodes
    print("\n" + "=" * 80)
    print("ALL OPCODES NEEDED")
    print("=" * 80)

    opcodes = set()
    for inst, mnemonic in all_instructions:
        op = (inst >> 24) & 0xFF
        opcodes.add(op)

    print(f"\nTotal unique opcodes: {len(opcodes)}")
    print("\nOpcode list:")
    for op in sorted(opcodes):
        print(f"  0x{op:02X}")

if __name__ == '__main__':
    main()
