#!/usr/bin/env python3
"""Debug test for weight loading issue."""

import kvrm_metal
import numpy as np

print("=" * 80)
print("  DEBUG TEST: Weight Loading Issue")
print("=" * 80)
print()

# Test 1: Create CPU, test without loading weights
print("Test 1: WITHOUT loading weights")
print("-" * 80)
cpu1 = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=1024)

inst1 = (0xD2 << 24) | (0 << 21) | (100 << 5) | 0
print(f"Instruction: 0x{inst1:08X} (MOVZ X0, #100)")

cpu1.write_memory_u32(0, inst1)
cpu1.set_pc(0, 0)
cpu1.execute(max_cycles=5)

x0_1 = cpu1.get_register(0, 0)
pc_1 = cpu1.get_pc(0)

print(f"Result: X0 = {x0_1}, PC = 0x{pc_1:X}")
print(f"Expected: X0 = 100, PC = 0x4")
print(f"Status: {'✅ PASS' if x0_1 == 100 and pc_1 == 4 else '❌ FAIL'}")
print()

# Test 2: Create NEW CPU, load all-zero weights, test
print("Test 2: With ALL-ZERO weights")
print("-" * 80)
cpu2 = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=1024)

# Load all-zero weights (same as initial state)
zero_weights = np.zeros(135, dtype=np.float32)
cpu2.load_dispatch_weights(zero_weights.tolist())

print("Loaded 135 zero weights")

cpu2.write_memory_u32(0, inst1)
cpu2.set_pc(0, 0)
cpu2.execute(max_cycles=5)

x0_2 = cpu2.get_register(0, 0)
pc_2 = cpu2.get_pc(0)

print(f"Result: X0 = {x0_2}, PC = 0x{pc_2:X}")
print(f"Expected: X0 = 100, PC = 0x4")
print(f"Status: {'✅ PASS' if x0_2 == 100 and pc_2 == 4 else '❌ FAIL'}")
print()

# Test 3: Create NEW CPU, load random weights, test
print("Test 3: With RANDOM weights (from file)")
print("-" * 80)
cpu3 = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=1024)

random_weights = np.load('weights/dispatch_weights.npy')
print(f"Loaded {len(random_weights)} random weights")
print(f"Weight range: [{random_weights.min():.3f}, {random_weights.max():.3f}]")

cpu3.load_dispatch_weights(random_weights.tolist())

cpu3.write_memory_u32(0, inst1)
cpu3.set_pc(0, 0)
cpu3.execute(max_cycles=5)

x0_3 = cpu3.get_register(0, 0)
pc_3 = cpu3.get_pc(0)

print(f"Result: X0 = {x0_3}, PC = 0x{pc_3:X}")
print(f"Expected: X0 = 100, PC = 0x4")
print(f"Status: {'✅ PASS' if x0_3 == 100 and pc_3 == 4 else '❌ FAIL'}")
print()

# Test 4: Test with same CPU instance after loading weights
print("Test 4: Same CPU instance, test BEFORE and AFTER loading weights")
print("-" * 80)
cpu4 = kvrm_metal.PyNeuralMetalCPU(num_lanes=1, memory_size=1024)

# Test BEFORE loading weights
cpu4.write_memory_u32(0, inst1)
cpu4.set_pc(0, 0)
cpu4.execute(max_cycles=5)
x0_before = cpu4.get_register(0, 0)
print(f"BEFORE loading weights: X0 = {x0_before}")

# Load weights
cpu4.load_dispatch_weights(random_weights.tolist())

# Test AFTER loading weights (reset state first)
cpu4.write_memory_u32(0, inst1)
cpu4.set_pc(0, 0)
cpu4.execute(max_cycles=5)
x0_after = cpu4.get_register(0, 0)
print(f"AFTER loading weights:  X0 = {x0_after}")
print()

print("=" * 80)
print("  SUMMARY")
print("=" * 80)
print(f"Test 1 (no load):       X0 = {x0_1:3d} ... {'✅ PASS' if x0_1 == 100 else '❌ FAIL'}")
print(f"Test 2 (zero weights):  X0 = {x0_2:3d} ... {'✅ PASS' if x0_2 == 100 else '❌ FAIL'}")
print(f"Test 3 (random weights): X0 = {x0_3:3d} ... {'✅ PASS' if x0_3 == 100 else '❌ FAIL'}")
print(f"Test 4 (before/after):  Before={x0_before:3d}, After={x0_after:3d}")
print("=" * 80)
