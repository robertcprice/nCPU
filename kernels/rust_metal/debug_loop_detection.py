#!/usr/bin/env python3
"""Debug neural loop detection"""

import sys
sys.path.insert(0, "/Users/bobbyprice/projects/KVRM/kvrm-cpu/rust_metal")

import kvrm_metal
import numpy as np

print("=" * 80)
print("  🔍 NEURAL LOOP DETECTION DEBUG")
print("=" * 80)
print()

# Create CPU
cpu = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=16 * 1024 * 1024)

# Load weights
dispatch_path = "/Users/bobbyprice/projects/KVRM/kvrm-cpu/rust_metal/weights/dispatch_weights_embedding_100pct.npy"
dispatch_weights = np.load(dispatch_path).astype(np.float32)
cpu.load_embedding_weights(dispatch_weights.tolist())
print(f"✅ Loaded {len(dispatch_weights):,} dispatch weights")

loop_path = "/Users/bobbyprice/projects/KVRM/kvrm-cpu/rust_metal/weights/loop_detector_v2_weights.npy"
loop_weights = np.load(loop_path).astype(np.float32)
cpu.load_loop_weights(loop_weights.tolist())
print(f"✅ Loaded {len(loop_weights):,} loop detector weights")
print()

# Test program - single SUBS instruction
# This should be detected as part of a loop pattern
program = [
    0xD2800064,  # MOVZ X0, #100 (counter)
    0xD2800001,  # MOVZ X1, #0 (accumulator)
    0x91000421,  # ADD X1, X1, #1
    0xF1000400,  # SUBS X0, X0, #1 <- This is where loop detection should trigger
]

program_addr = 0x1000
for i, inst in enumerate(program):
    addr = program_addr + i * 4
    cpu.write_memory(addr, list(inst.to_bytes(4, 'little')))

cpu.set_pc(0, program_addr)

print("Test Instructions:")
print(f"  0x{program_addr:04X}: MOVZ X0, #100")
print(f"  0x{program_addr+4:04X}: MOVZ X1, #0")
print(f"  0x{program_addr+8:04X}: ADD X1, X1, #1")
print(f"  0x{program_addr+12:04X}: SUBS X0, X0, #1  <- Should detect loop here")
print()

# Execute step by step
print("Executing step by step:")
for cycle in range(10):
    result = cpu.execute(1)
    pc = cpu.get_pc(0)
    x0 = cpu.get_register(0, 0)
    x1 = cpu.get_register(0, 1)
    print(f"  Cycle {cycle}: PC=0x{pc:04X}, X0={x0}, X1={x1}")

    if pc >= program_addr + 16:
        break

print()
print("Expected: neural model should detect loop pattern at SUBS instruction")
print("          with high confidence and predict ~100 iterations")
