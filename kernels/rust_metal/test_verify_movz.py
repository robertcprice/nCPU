#!/usr/bin/env python3
"""Verify MOVZ instruction encoding"""

def decode_movz(inst):
    """Decode MOVZ instruction"""
    op = (inst >> 24) & 0xFF
    hw = (inst >> 21) & 0x3
    imm16 = (inst >> 5) & 0xFFFF
    rd = inst & 0x1F
    result = imm16 << (hw * 16)
    return op, hw, imm16, rd, result

def encode_movz(rd, value, hw=0):
    """Encode MOVZ instruction"""
    if value >= 2**16:
        raise ValueError(f"Value {value} too large for MOVZ (max 65535)")
    if hw > 3:
        raise ValueError(f"hw {hw} must be 0-3")

    # MOVZ encoding: op=0xD2 (bits [31:24] with hw in [22:21])
    # Format: 11 001 00 hw imm16:rd (simplified)
    op = 0xD2
    imm16 = value & 0xFFFF

    # Encode: [31:24]=op, [23:22]=00, [21:20]=hw, [19:5]=imm16, [4:0]=rd
    inst = (op << 24) | (hw << 21) | (imm16 << 5) | rd
    return inst

print("=" * 60)
print("  MOVZ ENCODING VERIFICATION")
print("=" * 60)

# Test cases
test_cases = [
    (0, 50, 0),
    (1, 100, 0),
    (2, 150, 0),
    (3, 13, 0),
    (4, 15, 0),
]

print("\nGenerating MOVZ instructions:")
program = []
for rd, value, hw in test_cases:
    inst = encode_movz(rd, value, hw)
    program.append(inst)
    op, dec_hw, dec_imm16, dec_rd, dec_result = decode_movz(inst)
    status = "✅" if (dec_rd == rd and dec_result == value and dec_hw == hw) else "❌"
    print(f"  MOVZ X{rd}, #{value} -> 0x{inst:08X} {status}")
    print(f"    Decoded: op=0x{op:02X}, hw={dec_hw}, imm16={dec_imm16}, rd=X{dec_rd}, result={dec_result}")

print("\nTest program:")
for i, inst in enumerate(program):
    print(f"  program[{i}] = 0x{inst:08X}")

print("\n" + "=" * 60)
