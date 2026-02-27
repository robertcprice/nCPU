#!/usr/bin/env python3
"""Debug test for LDR/STR operations"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import kvrm_metal
import numpy as np

print("=" * 60)
print("  LDR/STR DEBUG TEST")
print("=" * 60)

cpu = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=16 * 1024 * 1024)

# Load weights
dispatch_path = Path(__file__).parent / "weights" / "dispatch_weights_embedding_100pct.npy"
if dispatch_path.exists():
    cpu.load_embedding_weights(np.load(dispatch_path).astype(np.float32).tolist())

# Set SP to a known value
cpu.set_register(0, 31, 0x10000)  # Set SP (X31) to 0x10000
print(f"Initial SP: {cpu.get_register(0, 31)} (expected: 0x10000)")

# Test program: MOVZ X0, #13; STR X0, [SP, #0]; LDR X1, [SP, #0]
program = [
    0xD20001A0,  # MOVZ X0, #13
    0xF90003E0,  # STR X0, [SP, #0]
    0xF94003E1,  # LDR X1, [SP, #0]
]

program_addr = 0x1000
for i, inst in enumerate(program):
    addr = program_addr + i * 4
    cpu.write_memory(addr, list(inst.to_bytes(4, 'little')))
    print(f"Wrote instruction {i} at 0x{addr:X}: 0x{inst:08X}")

cpu.set_pc(0, program_addr)

# Execute one instruction at a time
print("\n--- Cycle 1: MOVZ X0, #13 ---")
result = cpu.execute(1)
x0 = cpu.get_register(0, 0)
print(f"X0 = {x0} (expected: 13)")
print(f"PC = 0x{result.final_pc:X} (expected: 0x1004)")

print("\n--- Cycle 2: STR X0, [SP, #0] ---")
result = cpu.execute(1)
x0 = cpu.get_register(0, 0)
sp = cpu.get_register(0, 31)
print(f"X0 = {x0} (expected: 13)")
print(f"SP = {sp} (expected: 0x10000)")

# Read memory at SP to verify STR worked
mem_data = cpu.read_memory(0x10000, 8)
print(f"Memory at SP (0x10000): {mem_data}")
print(f"Expected: [0, 0, 0, 0, 13, 0, 0, 0, 0, 0] (little-endian 13)")

print("\n--- Cycle 3: LDR X1, [SP, #0] ---")
result = cpu.execute(1)
x0 = cpu.get_register(0, 0)
x1 = cpu.get_register(0, 1)
print(f"X0 = {x0} (expected: 13)")
print(f"X1 = {x1} (expected: 13)")
print(f"PC = 0x{result.final_pc:X} (expected: 0x100C)")

print("\n" + "=" * 60)
if x0 == 13 and x1 == 13:
    print("✅ PASS - LDR/STR working!")
else:
    print(f"❌ FAIL - X0={x0}, X1={x1}")
print("=" * 60)
