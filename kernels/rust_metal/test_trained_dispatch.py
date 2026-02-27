#!/usr/bin/env python3
"""
Test neural dispatch system with TRAINED dispatch weights.

This script loads the newly trained dispatch network weights and tests
whether they improve the neural dispatch performance.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kvrm_metal
import numpy as np

print("=" * 80)
print("  🧠 NEURAL GPU DISPATCH - TRAINED WEIGHTS TEST")
print("=" * 80)
print()

# Load the trained weights
weights_path = Path(__file__).parent / "weights" / "dispatch_weights_trained.npy"

if not weights_path.exists():
    print(f"❌ Trained weights not found: {weights_path}")
    print("   Run train_dispatch_network.py first!")
    sys.exit(1)

print(f"Loading trained weights from: {weights_path}")
trained_weights = np.load(weights_path)
print(f"  Loaded {len(trained_weights)} weights")
print(f"  Weight range: [{trained_weights.min():.3f}, {trained_weights.max():.3f}]")
print()

# Create NeuralMetalCPU
print("Creating NeuralMetalCPU with 128 lanes...")
cpu = kvrm_metal.PyNeuralMetalCPU(num_lanes=128, memory_size=16*1024*1024)
print(f"  ✅ Created with {cpu.get_num_lanes()} lanes")
print()

# Load the trained weights into GPU
print("Loading trained dispatch weights into GPU...")
cpu.load_dispatch_weights(trained_weights.tolist())
print("  ✅ Weights loaded!")
print()

# Test instructions
print("=" * 80)
print("  TESTING WITH TRAINED WEIGHTS")
print("=" * 80)
print()

tests = [
    # (instruction, expected_x0, description)
    ((0xD2 << 24) | (0 << 21) | (100 << 5) | 0, 100, "MOVZ X0, #100"),
    ((0xD2 << 24) | (0 << 21) | (200 << 5) | 1, 200, "MOVZ X1, #200"),
    ((0xD2 << 24) | (0 << 21) | (50 << 5) | 2, 50, "MOVZ X2, #50"),
]

passed = 0
failed = 0

for inst, expected, description in tests:
    print(f"Test: {description}")
    print(f"  Instruction: 0x{inst:08X}")

    cpu.write_memory_u32(0, inst)
    for lane in range(128):
        cpu.set_pc(lane, 0)

    result = cpu.execute(max_cycles=5)
    x0 = cpu.get_register(0, ((inst >> 0) & 0x1F) if (inst & 0x1F) < 31 else 0)

    print(f"  Result: X{(inst >> 0) & 0x1F} = {x0} (expected {expected})")

    if x0 == expected:
        print("  ✅ PASSED!")
        passed += 1
    else:
        print(f"  ❌ FAILED: got {x0}")
        failed += 1

    print()

print("=" * 80)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 80)
print()

if failed == 0:
    print("✅ All tests passed with trained weights!")
    print()
    print("The neural dispatch system is working correctly with trained weights!")
else:
    print(f"⚠️  {failed} tests failed")
    print()
    print("The trained weights still have some inaccuracies (42.9% accuracy).")
    print()
    print("For production use, we should:")
    print("  1. Collect more diverse training samples")
    print("  2. Train for more epochs")
    print("  3. Increase network capacity")
    print("  4. Use opcode-based fallback for correctness")
