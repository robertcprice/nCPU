#!/usr/bin/env python3
"""Benchmark: Standard vs Continuous vs Mega (single dispatch)."""

import time
from kvrm_metal import ContinuousMetalCPU, MetalCPU

print("=" * 70)
print("MEGA BATCH BENCHMARK - Single GPU Dispatch")
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

# Test program: tight loop
program = [
    encode_movz(0, 0),           # x0 = 0
    encode_add_imm(0, 0, 1),     # x0++
    encode_b(-1 & 0x3FFFFFF),    # loop
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program)

total_cycles = 100_000_000

print(f"\nTest: {total_cycles:,} cycles - tight loop (ADD + B)\n")

# 1. Standard execution
cpu_std = MetalCPU()
cpu_std.load_program(list(program_bytes), 0)
cpu_std.set_pc(0)

start = time.perf_counter()
result = cpu_std.execute(total_cycles)
elapsed_std = time.perf_counter() - start
ips_std = result.cycles / elapsed_std

print(f"STANDARD:   {result.cycles:>12,} cycles in {elapsed_std:.3f}s = {ips_std:>12,.0f} IPS")

# 2. Continuous (batched)
cpu_cont = ContinuousMetalCPU(cycles_per_batch=10_000_000)
cpu_cont.load_program(list(program_bytes), 0)
cpu_cont.set_pc(0)

start = time.perf_counter()
result = cpu_cont.execute_continuous(max_batches=10, timeout_seconds=120.0)
elapsed_cont = time.perf_counter() - start
ips_cont = result.ips

print(f"CONTINUOUS: {result.total_cycles:>12,} cycles in {elapsed_cont:.3f}s = {ips_cont:>12,.0f} IPS ({(ips_cont-ips_std)/ips_std*100:+.1f}%)")

# 3. MEGA - single dispatch
cpu_mega = ContinuousMetalCPU()
cpu_mega.load_program(list(program_bytes), 0)
cpu_mega.set_pc(0)

start = time.perf_counter()
result = cpu_mega.execute_mega(total_cycles)
elapsed_mega = time.perf_counter() - start
ips_mega = result.ips

print(f"MEGA:       {result.total_cycles:>12,} cycles in {elapsed_mega:.3f}s = {ips_mega:>12,.0f} IPS ({(ips_mega-ips_std)/ips_std*100:+.1f}%)")

# With memory writes
print(f"\nTest: {total_cycles:,} cycles - loop with STR (memory writes)\n")

program_str = [
    encode_movz(0, 0),
    encode_movz(1, 0x8000),
    encode_add_imm(0, 0, 1),
    encode_str_64(0, 1, 0),
    encode_b(-2 & 0x3FFFFFF),
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program_str)

# Standard
cpu_std = MetalCPU()
cpu_std.load_program(list(program_bytes), 0)
cpu_std.set_pc(0)

start = time.perf_counter()
result = cpu_std.execute(total_cycles)
elapsed_std = time.perf_counter() - start
ips_std = result.cycles / elapsed_std

print(f"STANDARD:   {result.cycles:>12,} cycles in {elapsed_std:.3f}s = {ips_std:>12,.0f} IPS")

# Mega
cpu_mega = ContinuousMetalCPU()
cpu_mega.load_program(list(program_bytes), 0)
cpu_mega.set_pc(0)

start = time.perf_counter()
result = cpu_mega.execute_mega(total_cycles)
elapsed_mega = time.perf_counter() - start
ips_mega = result.ips

print(f"MEGA:       {result.total_cycles:>12,} cycles in {elapsed_mega:.3f}s = {ips_mega:>12,.0f} IPS ({(ips_mega-ips_std)/ips_std*100:+.1f}%)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Standard: One GPU dispatch per call, includes all setup overhead
Continuous: Multiple batched dispatches with Rust loop (minimal Python involvement)
MEGA: Single GPU dispatch for entire workload (maximum throughput)

The MEGA mode achieves peak IPS by eliminating all per-batch overhead.
""")
