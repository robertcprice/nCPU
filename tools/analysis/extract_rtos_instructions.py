#!/usr/bin/env python3
"""
Extract all unique instructions from RTOS ELF for neural decoder training.

This extracts every instruction the RTOS uses so we can fine-tune
the neural decoder to handle them ALL neurally.
"""

import struct
from collections import defaultdict


def load_elf_code(path):
    """Load all code from ELF file."""
    with open(path, 'rb') as f:
        data = f.read()

    if data[:4] != b'\x7fELF':
        raise ValueError("Not ELF")

    phoff = struct.unpack('<Q', data[32:40])[0]
    phentsize = struct.unpack('<H', data[54:56])[0]
    phnum = struct.unpack('<H', data[56:58])[0]

    code_segments = []
    for i in range(phnum):
        ph = phoff + i * phentsize
        pt = struct.unpack('<I', data[ph:ph+4])[0]
        if pt == 1:  # PT_LOAD
            offset = struct.unpack('<Q', data[ph+8:ph+16])[0]
            vaddr = struct.unpack('<Q', data[ph+16:ph+24])[0]
            filesz = struct.unpack('<Q', data[ph+32:ph+40])[0]
            flags = struct.unpack('<I', data[ph+4:ph+8])[0]
            # Check if executable (flags & 1)
            if flags & 1:
                code_segments.append((vaddr, data[offset:offset+filesz]))

    return code_segments


def decode_arm64_op(inst):
    """Get the main opcode class and instruction details."""
    op = (inst >> 24) & 0xFF
    rd = inst & 0x1F
    rn = (inst >> 5) & 0x1F
    rm = (inst >> 16) & 0x1F
    imm12 = (inst >> 10) & 0xFFF
    imm7 = (inst >> 15) & 0x7F

    # Categorize instruction
    op_info = {
        'opcode': op,
        'rd': rd,
        'rn': rn,
        'rm': rm,
        'inst': inst,
    }

    # Known instruction types
    if op == 0x91:
        op_info['name'] = 'ADD_imm_64'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x11:
        op_info['name'] = 'ADD_imm_32'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0xD1:
        op_info['name'] = 'SUB_imm_64'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x51:
        op_info['name'] = 'SUB_imm_32'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x8B:
        op_info['name'] = 'ADD_reg_64'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x0B:
        op_info['name'] = 'ADD_reg_32'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0xCB:
        op_info['name'] = 'SUB_reg_64'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x4B:
        op_info['name'] = 'SUB_reg_32'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0xD2:
        op_info['name'] = 'MOVZ_64'
        op_info['category'] = 'MOVE'
    elif op == 0x52:
        op_info['name'] = 'MOVZ_32'
        op_info['category'] = 'MOVE'
    elif op == 0xF2:
        op_info['name'] = 'MOVK_64'
        op_info['category'] = 'MOVE'
    elif op == 0x72:
        op_info['name'] = 'MOVK_32'
        op_info['category'] = 'MOVE'
    elif op == 0xAA:
        op_info['name'] = 'ORR_reg_64'
        op_info['category'] = 'LOGICAL'
    elif op == 0x2A:
        op_info['name'] = 'ORR_reg_32'
        op_info['category'] = 'LOGICAL'
    elif op == 0x8A:
        op_info['name'] = 'AND_reg_64'
        op_info['category'] = 'LOGICAL'
    elif op == 0x0A:
        op_info['name'] = 'AND_reg_32'
        op_info['category'] = 'LOGICAL'
    elif op == 0xCA:
        op_info['name'] = 'EOR_reg_64'
        op_info['category'] = 'LOGICAL'
    elif op == 0x4A:
        op_info['name'] = 'EOR_reg_32'
        op_info['category'] = 'LOGICAL'
    elif op == 0xF9:
        op_info['name'] = 'LDR_STR_64'
        op_info['category'] = 'LOAD_STORE'
    elif op == 0xB9:
        op_info['name'] = 'LDR_STR_32'
        op_info['category'] = 'LOAD_STORE'
    elif op == 0x39:
        op_info['name'] = 'LDRB_STRB'
        op_info['category'] = 'LOAD_STORE'
    elif op == 0x38:
        op_info['name'] = 'LDRB_STRB_idx'
        op_info['category'] = 'LOAD_STORE'
    elif op == 0xB8:
        op_info['name'] = 'LDR_STR_32_idx'
        op_info['category'] = 'LOAD_STORE'
    elif op == 0xA9:
        op_info['name'] = 'STP_64'
        op_info['category'] = 'LOAD_STORE'
    elif op == 0x29:
        op_info['name'] = 'STP_32'
        op_info['category'] = 'LOAD_STORE'
    elif op == 0xA8:
        op_info['name'] = 'LDP_64'
        op_info['category'] = 'LOAD_STORE'
    elif op == 0x28:
        op_info['name'] = 'LDP_32'
        op_info['category'] = 'LOAD_STORE'
    elif (op >> 2) == 0x05:
        op_info['name'] = 'B'
        op_info['category'] = 'BRANCH'
    elif (op >> 2) == 0x25:
        op_info['name'] = 'BL'
        op_info['category'] = 'BRANCH'
    elif op == 0x54:
        op_info['name'] = 'B_cond'
        op_info['category'] = 'BRANCH'
    elif op == 0x34:
        op_info['name'] = 'CBZ_32'
        op_info['category'] = 'BRANCH'
    elif op == 0xB4:
        op_info['name'] = 'CBZ_64'
        op_info['category'] = 'BRANCH'
    elif op == 0x35:
        op_info['name'] = 'CBNZ_32'
        op_info['category'] = 'BRANCH'
    elif op == 0xB5:
        op_info['name'] = 'CBNZ_64'
        op_info['category'] = 'BRANCH'
    elif op == 0xF1:
        op_info['name'] = 'CMP_imm_64'
        op_info['category'] = 'COMPARE'
    elif op == 0x71:
        op_info['name'] = 'SUBS_imm_32'
        op_info['category'] = 'COMPARE'
    elif op == 0xEB:
        op_info['name'] = 'CMP_reg_64'
        op_info['category'] = 'COMPARE'
    elif op == 0x6B:
        op_info['name'] = 'SUBS_reg_32'
        op_info['category'] = 'COMPARE'
    elif (op & 0x9F) == 0x90:
        op_info['name'] = 'ADRP'
        op_info['category'] = 'ADDRESS'
    elif op == 0x93:
        op_info['name'] = 'SXTW'
        op_info['category'] = 'EXTEND'
    elif op == 0x9B:
        op_info['name'] = 'MUL_64'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x1B:
        op_info['name'] = 'MUL_32'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x9A:
        op_info['name'] = 'SDIV_UDIV_64'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x1A:
        op_info['name'] = 'SDIV_UDIV_32'
        op_info['category'] = 'ARITHMETIC'
    elif op == 0x72:
        op_info['name'] = 'ANDS_imm_32'
        op_info['category'] = 'LOGICAL'
    elif op == 0x7A:
        op_info['name'] = 'CCMP_32'
        op_info['category'] = 'COMPARE'
    elif inst == 0xD65F03C0:
        op_info['name'] = 'RET'
        op_info['category'] = 'BRANCH'
    elif inst == 0xD503201F:
        op_info['name'] = 'NOP'
        op_info['category'] = 'SYSTEM'
    else:
        op_info['name'] = f'UNKNOWN_0x{op:02x}'
        op_info['category'] = 'UNKNOWN'

    return op_info


def main():
    print("="*70)
    print("RTOS Instruction Extraction for Neural Decoder Training")
    print("="*70)

    rtos_path = "arm64_doom/neural_rtos.elf"

    try:
        segments = load_elf_code(rtos_path)
    except Exception as e:
        print(f"Error loading ELF: {e}")
        return

    print(f"\nFound {len(segments)} code segment(s)")

    # Extract all instructions
    all_instructions = []
    opcode_counts = defaultdict(int)
    category_counts = defaultdict(int)
    unknown_opcodes = set()

    for vaddr, code in segments:
        print(f"\nSegment at 0x{vaddr:x}, size {len(code)} bytes")

        for i in range(0, len(code) - 3, 4):
            inst = struct.unpack('<I', code[i:i+4])[0]
            addr = vaddr + i

            info = decode_arm64_op(inst)
            info['address'] = addr
            all_instructions.append(info)

            opcode_counts[info['name']] += 1
            category_counts[info['category']] += 1

            if info['category'] == 'UNKNOWN':
                unknown_opcodes.add(info['opcode'])

    print(f"\n{'='*70}")
    print("INSTRUCTION STATISTICS")
    print(f"{'='*70}")
    print(f"\nTotal instructions: {len(all_instructions)}")
    print(f"Unique instruction types: {len(opcode_counts)}")

    print(f"\n--- By Category ---")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_instructions) * 100
        print(f"  {cat:15s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n--- By Instruction Type ---")
    for name, count in sorted(opcode_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_instructions) * 100
        print(f"  {name:20s}: {count:5d} ({pct:5.1f}%)")

    if unknown_opcodes:
        print(f"\n{'='*70}")
        print("⚠️  UNKNOWN OPCODES (need to add to neural decoder)")
        print(f"{'='*70}")
        for op in sorted(unknown_opcodes):
            count = sum(1 for i in all_instructions if i['opcode'] == op)
            print(f"  0x{op:02x}: {count} occurrences")

    # Save training data
    print(f"\n{'='*70}")
    print("Saving training data for neural decoder fine-tuning...")
    print(f"{'='*70}")

    import torch

    training_data = []
    for info in all_instructions:
        # Convert instruction to bits
        inst = info['inst']
        bits = torch.tensor([float((inst >> i) & 1) for i in range(32)])
        training_data.append({
            'bits': bits,
            'inst': inst,
            'name': info['name'],
            'category': info['category'],
            'rd': info['rd'],
            'rn': info['rn'],
            'rm': info['rm'],
        })

    torch.save(training_data, 'rtos_instructions.pt')
    print(f"Saved {len(training_data)} instructions to rtos_instructions.pt")

    # Show sample instructions
    print(f"\n--- Sample Instructions ---")
    seen = set()
    for info in all_instructions:
        if info['name'] not in seen and len(seen) < 20:
            seen.add(info['name'])
            print(f"  0x{info['address']:05x}: 0x{info['inst']:08x} {info['name']}")


if __name__ == "__main__":
    main()
