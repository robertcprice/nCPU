#!/usr/bin/env python3
"""Benchmark Rust Metal kernel vs MLX V2."""

import time
from kvrm_metal import MetalCPU

print("=" * 70)
print("RUST METAL KERNEL BENCHMARK - Zero-Copy Shared Memory")
print("=" * 70)

def encode_movz(rd, imm16, hw=0):
    return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

def encode_add_imm(rd, rn, imm12):
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd

def encode_str_64(rt, rn, imm12=0):
    return 0xF9000000 | (imm12 << 10) | (rn << 5) | rt

def encode_b(offset_words):
    imm26 = offset_words & 0x3FFFFFF
    return 0x14000000 | imm26

# Test 1: Loop with memory writes
print("\n--- Test 1: Loop with STR (1M cycles) ---")
cpu = MetalCPU()
program = [
    encode_movz(0, 0),           # Counter
    encode_movz(1, 0x8000),      # Address
    encode_add_imm(0, 0, 1),     # counter++
    encode_str_64(0, 1, 0),      # STR to memory
    encode_b(-2 & 0x3FFFFFF),    # Loop back
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program)
cpu.load_program(list(program_bytes), 0)
cpu.set_pc(0)

start = time.perf_counter()
result = cpu.execute(1_000_000)
elapsed = time.perf_counter() - start

print(f"Cycles: {result.cycles:,}")
print(f"Time: {elapsed:.3f}s")
print(f"IPS: {result.cycles / elapsed:,.0f}")

# Test 2: Higher cycle count
print("\n--- Test 2: Loop with STR (10M cycles) ---")
cpu = MetalCPU()
cpu.load_program(list(program_bytes), 0)
cpu.set_pc(0)

start = time.perf_counter()
result = cpu.execute(10_000_000)
elapsed = time.perf_counter() - start

print(f"Cycles: {result.cycles:,}")
print(f"Time: {elapsed:.3f}s")
print(f"IPS: {result.cycles / elapsed:,.0f}")

# Test 3: Even higher - 100M cycles
print("\n--- Test 3: Loop with STR (100M cycles) ---")
cpu = MetalCPU()
cpu.load_program(list(program_bytes), 0)
cpu.set_pc(0)

start = time.perf_counter()
result = cpu.execute(100_000_000)
elapsed = time.perf_counter() - start

print(f"Cycles: {result.cycles:,}")
print(f"Time: {elapsed:.3f}s")
print(f"IPS: {result.cycles / elapsed:,.0f}")

# Test 4: Loop without memory writes (baseline)
print("\n--- Test 4: Loop WITHOUT STR (100M cycles) ---")
program_no_str = [
    encode_movz(0, 0),           # Counter
    encode_add_imm(0, 0, 1),     # counter++
    encode_b(-1 & 0x3FFFFFF),    # Loop back
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program_no_str)
cpu = MetalCPU()
cpu.load_program(list(program_bytes), 0)
cpu.set_pc(0)

start = time.perf_counter()
result = cpu.execute(100_000_000)
elapsed = time.perf_counter() - start

print(f"Cycles: {result.cycles:,}")
print(f"Time: {elapsed:.3f}s")
print(f"IPS: {result.cycles / elapsed:,.0f}")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
