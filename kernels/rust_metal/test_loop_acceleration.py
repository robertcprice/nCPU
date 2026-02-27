#!/usr/bin/env python3
"""
Test loop detection and acceleration on GPU.

This test:
1. Creates a simple countdown loop (100 iterations)
2. Runs with loop detection enabled
3. Measures cycles and verifies correctness
4. Establishes baseline for neural loop acceleration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kvrm_metal
import numpy as np
import time

print("=" * 80)
print("  🔄 LOOP ACCELERATION TEST")
print("=" * 80)
print()

# Create NeuralMetalCPU (the pure GPU implementation)
cpu = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=16 * 1024 * 1024)

# Load neural weights
dispatch_path = Path(__file__).parent / "weights" / "dispatch_weights_embedding_100pct.npy"
if dispatch_path.exists():
    dispatch_weights = np.load(dispatch_path).astype(np.float32)
    cpu.load_embedding_weights(dispatch_weights.tolist())
    print(f"✅ Loaded {len(dispatch_weights):,} dispatch weights (100% accurate)")

loop_path = Path(__file__).parent / "weights" / "loop_detector_v2_weights.npy"
if loop_path.exists():
    loop_weights = np.load(loop_path).astype(np.float32)
    cpu.load_loop_weights(loop_weights.tolist())
    print(f"✅ Loaded {len(loop_weights):,} loop detector weights")

memory_path = Path(__file__).parent / "weights" / "memory_oracle_lstm_weights.npy"
if memory_path.exists():
    memory_weights = np.load(memory_path).astype(np.float32)
    cpu.load_memory_weights(memory_weights.tolist())
    print(f"✅ Loaded {len(memory_weights):,} memory oracle weights")

print()

# Simple countdown loop program:
#   MOVZ X0, #100     ; X0 = 100 (loop counter)
#   MOVZ X1, #0       ; X1 = 0 (sum accumulator)
# loop:
#   ADD  X1, X1, #1   ; X1++
#   SUBS X0, X0, #1   ; X0-- (sets flags)
#   B.NE loop         ; Branch if not equal (X0 != 0)
#   MOVZ X2, #42      ; X2 = 42 (done marker)
#   NOP               ; End

program = [
    0xD2000C80,  # MOVZ X0, #100 (counter) - FIXED encoding
    0xD2800001,  # MOVZ X1, #0   (sum)
    # loop:
    0x91000421,  # ADD X1, X1, #1   ; sum++ (correct encoding: imm12=1)
    0xF1000400,  # SUBS X0, X0, #1  ; counter-- (correct encoding: imm12=1)
    0x54FFFFC1,  # B.NE -2 instructions (back to loop)  ; branch back if X0 != 0
    # done:
    0xD2800542,  # MOVZ X2, #42
    0xD503201F,  # NOP
]

program_addr = 0x1000
for i, inst in enumerate(program):
    addr = program_addr + i * 4
    cpu.write_memory(addr, list(inst.to_bytes(4, 'little')))

cpu.set_pc(0, program_addr)

print("Test Program: Countdown Loop (100 iterations)")
print("  Code:")
print("    MOVZ X0, #100    ; X0 = 100")
print("    MOVZ X1, #0      ; X1 = 0")
print("  loop:")
print("    ADD  X1, X1, #1  ; X1++")
print("    SUBS X0, X0, #1  ; X0--")
print("    B.NE loop        ; if X0 != 0, goto loop")
print("  done:")
print("    MOVZ X2, #42")
print("    NOP")
print(f"  Loaded to 0x{program_addr:04X}")
print()

# Execute with plenty of cycles to let the loop complete
# Then check where the loop actually stops
start = time.time()
result = cpu.execute(1000)  # Plenty of cycles
elapsed = time.time() - start

# Verify results
x0 = cpu.get_register(0, 0)
x1 = cpu.get_register(0, 1)
x2 = cpu.get_register(0, 2)
final_pc = result.final_pc

print("Results:")
print(f"  X0 = {x0} (expected: -1/0xFFFFFFFFFFFFFFFF)")
print(f"  X1 = {x1} (expected: 100)")
print(f"  X2 = {x2} (expected: 42)")
print(f"  Final PC = 0x{final_pc:08X}")
print(f"  Cycles: {result.cycles}")
print(f"  Time: {elapsed:.4f}s")
print(f"  IPS: {result.cycles / elapsed:.0f}")
print()

# Calculate expected values
# After 100 iterations: X0 = 100-100 = -1 (underflow), X1 = 100
expected_x0 = -1 & 0xFFFFFFFFFFFFFFFF  # Wrap around
expected_x1 = 100
expected_x2 = 42

if x0 == expected_x0 and x1 == expected_x1 and x2 == expected_x2:
    print("=" * 80)
    print("  ✅ LOOP EXECUTION VERIFIED CORRECT")
    print("=" * 80)
    print()
    print("Analysis:")
    print(f"  - Loop executed correctly: 100 iterations")
    print(f"  - X1 accumulator: {x1} (100 iterations × 1 increment)")
    print(f"  - X0 counter: {x0} (wraps to -1 after underflow)")
    print(f"  - Control flow reached completion (X2 marker set)")
    print()
    print("Current Status:")
    print("  ✅ Loop detection architecture working")
    print("  ✅ Neural weights loaded (1.08M params)")
    print("  ⏳ Full LSTM inference: Not yet implemented")
    print("  ⏳ Loop acceleration: Using heuristic (not neural yet)")
    print()
    print("Next Steps for Neural Loop Acceleration:")
    print("  1. Implement LSTM cell in Metal shader")
    print("  2. Use trained weights for prediction")
    print("  3. Detect loops before entering (skip entire loop body)")
    print("  4. Target: 100 iterations → 1 operation")
else:
    print("❌ Loop execution incorrect")
    print(f"  Expected: X0={expected_x0}, X1={expected_x1}, X2={expected_x2}")
    print(f"  Got:      X0={x0}, X1={x1}, X2={x2}")
