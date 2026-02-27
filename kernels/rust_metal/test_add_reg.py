#!/usr/bin/env python3
"""Test ADD(register-register) instruction"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import kvrm_metal
import numpy as np
import time

print("=" * 60)
print("  ADD (REGISTER-REGISTER) TEST")
print("=" * 60)

cpu = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=16 * 1024 * 1024)

dispatch_path = Path(__file__).parent / "weights" / "dispatch_weights_embedding_100pct.npy"
if dispatch_path.exists():
    cpu.load_embedding_weights(np.load(dispatch_path).astype(np.float32).tolist())

# Test: MOVZ X0, #50; MOVZ X1, #100; ADD X0, X1, X0
# Expected: X0 = 150, X1 = 100
program = [
    0xD2000640,  # MOVZ X0, #50
    0xD2000C81,  # MOVZ X1, #100
    0x8B000020,  # ADD X0, X1, X0  (X0 = X1 + X0)
]

program_addr = 0x1000
for i, inst in enumerate(program):
    addr = program_addr + i * 4
    cpu.write_memory(addr, list(inst.to_bytes(4, 'little')))

cpu.set_pc(0, program_addr)

start = time.time()
result = cpu.execute(100)
elapsed = time.time() - start

x0 = cpu.get_register(0, 0)
x1 = cpu.get_register(0, 1)

print(f"X0 = {x0} (expected: 150)")
print(f"X1 = {x1} (expected: 100)")
print(f"Cycles: {result.cycles} / 3 expected")
print(f"Final PC: 0x{result.final_pc:04X}")

if x0 == 150 and x1 == 100 and result.cycles == 3:
    print("✅ PASS - ADD(register-register) works!")
else:
    print(f"❌ FAIL - ADD(register-register) not working correctly")

print("=" * 60)
