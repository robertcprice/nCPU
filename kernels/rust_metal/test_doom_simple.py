#!/usr/bin/env python3
"""
Simple DOOM test for NeuralMetalCPU.
Loads ARM64 DOOM binary and executes for a limited number of instructions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import kvrm_metal
import numpy as np
import time

print("=" * 80)
print("  DOOM TEST - NeuralMetalCPU")
print("=" * 80)
print()

# Create CPU with enough memory for DOOM
cpu = kvrm_metal.PyNeuralMetalCPU(
    num_lanes=1,
    memory_size=32 * 1024 * 1024  # 32MB
)

# Load neural weights
dispatch_path = Path(__file__).parent / "weights" / "dispatch_weights_embedding_100pct.npy"
if dispatch_path.exists():
    dispatch_weights = np.load(dispatch_path).astype(np.float32)
    cpu.load_embedding_weights(dispatch_weights.tolist())
    print(f"✅ Loaded {len(dispatch_weights):,} dispatch weights")

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

# Load DOOM binary
doom_path = Path(__file__).parent.parent / "doom_benchmark.elf"
if not doom_path.exists():
    doom_path = Path(__file__).parent.parent / "arm64_doom" / "doom_neural.elf"

if not doom_path.exists():
    print("❌ DOOM binary not found!")
    print(f"   Searched for: {doom_path}")
    sys.exit(1)

print(f"📁 Loading DOOM binary: {doom_path.name}")
with open(doom_path, 'rb') as f:
    doom_binary = f.read()

print(f"   Size: {len(doom_binary):,} bytes ({len(doom_binary)/1024:.1f} KB)")
print()

# Load DOOM to memory at address 0x10000 (typical ARM64 load address)
load_addr = 0x10000
print(f"📋 Loading DOOM to 0x{load_addr:08X}")

# Load as raw bytes (simplified - not doing full ELF parsing)
for i, byte in enumerate(doom_binary):
    addr = load_addr + i
    if addr < 32 * 1024 * 1024:  # Within memory range
        cpu.write_memory(addr, [byte])

print(f"   Loaded {len(doom_binary):,} bytes")
print()

# Set PC to DOOM entry point
# ARM64 binaries typically start at offset 0x100 in the file
entry_point = load_addr + 0x100
cpu.set_pc(0, entry_point)
print(f"🎯 Starting execution at PC=0x{entry_point:08X}")
print()

# Execute for a limited number of cycles
max_cycles = 1_000_000
print(f"⚡ Executing up to {max_cycles:,} cycles...")
print()

start = time.time()
result = cpu.execute(max_cycles)
elapsed = time.time() - start

# Report results
print("=" * 80)
print("  EXECUTION RESULTS")
print("=" * 80)
print(f"  Cycles executed: {result.cycles:,}")
print(f"  Final PC: 0x{result.final_pc:08X}")
print(f"  Time: {elapsed:.4f}s")
print(f"  IPS: {result.cycles / elapsed:,.0f}")
print()

# Check some registers
x0 = cpu.get_register(0, 0)
x1 = cpu.get_register(0, 1)
x2 = cpu.get_register(0, 2)
x19 = cpu.get_register(0, 19)
sp = cpu.get_register(0, 31)  # X31 is SP in this context

print("  Register State:")
print(f"    X0  = 0x{x0:016X}")
print(f"    X1  = 0x{x1:016X}")
print(f"    X2  = 0x{x2:016X}")
print(f"    X19 = 0x{x19:016X}")
print(f"    SP  = 0x{sp:016X}")
print()

# Check if we made progress
if result.cycles > 1000:
    print("  ✅ DOOM is executing!")
    print(f"     System appears to be running at {result.cycles / elapsed:,.0f} IPS")
else:
    print("  ⚠️  DOOM may have encountered an issue")
    print(f"     Only {result.cycles} cycles executed")

print()
print("=" * 80)
