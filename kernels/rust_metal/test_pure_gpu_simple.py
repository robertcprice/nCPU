#!/usr/bin/env python3
"""
Simple pure GPU test - linear program (no branches)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kvrm_metal
import numpy as np

print("=" * 80)
print("  🚀 PURE GPU TEST - Simple Linear Program")
print("=" * 80)
print()

# Create PureGPUCPU
cpu = kvrm_metal.PyPureGPUCPU(num_lanes=1, memory_size=16 * 1024 * 1024)

# Load 100% accurate dispatch weights
weights_path = Path(__file__).parent / "weights" / "dispatch_weights_embedding_100pct.npy"
if weights_path.exists():
    weights = np.load(weights_path).tolist()
    cpu.load_dispatch_weights(weights)
    print("✅ Loaded 100% accurate dispatch weights")
else:
    print("⚠️  No weights found - will use zeros")

print()

# Simple linear program: MOVZ, MOVZ, MOVZ
# MOVZ encoding: imm16 is bits [20:5], hw is bits [22:21]
program = [
    0xD2800140,  # MOVZ X0, #10  (imm16=10, hw=0)
    0xD2800041,  # MOVZ X1, #2   (imm16=2, hw=0)
    0xD2800062,  # MOVZ X2, #3   (imm16=3, hw=0)
]

# Write program to memory
program_addr = 0x1000
for i, inst in enumerate(program):
    addr = program_addr + i * 4
    inst_bytes = inst.to_bytes(4, 'little')
    cpu.write_memory(addr, list(inst_bytes))

cpu.set_pc(0, program_addr)

print(f"Program: {len(program)} instructions")
print(f"  0xD2800140 = MOVZ X0, #10")
print(f"  0xD2800041 = MOVZ X1, #2")
print(f"  0xD2800062 = MOVZ X2, #3")
print()

# Execute
result = cpu.execute(100)

print("Results:")
print(f"  Cycles: {result.cycles}")
print(f"  Final PC: 0x{result.final_pc:08X}")
print(f"  X0 = {cpu.get_register(0, 0)} (expected: 10)")
print(f"  X1 = {cpu.get_register(0, 1)} (expected: 2)")
print(f"  X2 = {cpu.get_register(0, 2)} (expected: 3)")
print()

if result.cycles == 3:
    if cpu.get_register(0, 0) == 10 and cpu.get_register(0, 1) == 2 and cpu.get_register(0, 2) == 3:
        print("✅ PURE GPU EXECUTION CORRECT!")
    else:
        print("❌ Register values incorrect")
else:
    print(f"⚠️  Expected 3 cycles, got {result.cycles}")
