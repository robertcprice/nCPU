#!/usr/bin/env python3
"""Simple debug test - only MOVZ instructions"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import kvrm_metal
import numpy as np
import time

print("=" * 60)
print("  DEBUG TEST - Only MOVZ")
print("=" * 60)

cpu = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=16 * 1024 * 1024)

# Load weights
dispatch_path = Path(__file__).parent / "weights" / "dispatch_weights_embedding_100pct.npy"
if dispatch_path.exists():
    cpu.load_embedding_weights(np.load(dispatch_path).astype(np.float32).tolist())

# Test program: 5 MOVZ instructions (correctly encoded)
program = [
    0xD2000640,  # MOVZ X0, #50
    0xD2000C81,  # MOVZ X1, #100
    0xD20012C2,  # MOVZ X2, #150
    0xD20001A3,  # MOVZ X3, #13
    0xD20001E4,  # MOVZ X4, #15
]

program_addr = 0x1000
for i, inst in enumerate(program):
    addr = program_addr + i * 4
    cpu.write_memory(addr, list(inst.to_bytes(4, 'little')))

cpu.set_pc(0, program_addr)

start = time.time()
result = cpu.execute(100)  # Allow up to 100 cycles
elapsed = time.time() - start

x0 = cpu.get_register(0, 0)
x1 = cpu.get_register(0, 1)
x2 = cpu.get_register(0, 2)
x3 = cpu.get_register(0, 3)
x4 = cpu.get_register(0, 4)

print(f"X0 = {x0} (expected: 50)")
print(f"X1 = {x1} (expected: 100)")
print(f"X2 = {x2} (expected: 150)")
print(f"X3 = {x3} (expected: 13)")
print(f"X4 = {x4} (expected: 15)")
print(f"Cycles executed: {result.cycles} / 5 expected")
print(f"Final PC: 0x{result.final_pc:04X}")

if result.cycles == 5 and x0 == 50 and x1 == 100 and x2 == 150 and x3 == 13 and x4 == 15:
    print("✅ PASS - All MOVZ executed correctly")
else:
    print(f"❌ FAIL - Only {result.cycles}/5 cycles executed")

print("=" * 60)
