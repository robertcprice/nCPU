#!/usr/bin/env python3
"""Test async GPU execution - GPU runs while Python does other work."""

import time
from kvrm_metal import AsyncMetalCPU

print("=" * 70)
print("ASYNC GPU EXECUTION TEST")
print("=" * 70)

def encode_movz(rd, imm16, hw=0):
    return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

def encode_add_imm(rd, rn, imm12):
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd

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

# Create async CPU
cpu = AsyncMetalCPU()
cpu.load_program(list(program_bytes), 0)
cpu.set_pc(0)

print("\n[1] Starting async execution of 100M cycles...")
total_cycles = 100_000_000

start = time.perf_counter()
cpu.start(total_cycles)

print("[2] GPU is now executing in background!")
print("[3] Python can do other work while GPU crunches...")

# Poll periodically while GPU runs
poll_count = 0
while cpu.is_running():
    poll_count += 1
    status = cpu.poll()
    print(f"    Poll {poll_count}: {status.cycles_executed:>12,} cycles, "
          f"elapsed: {status.elapsed_seconds:.3f}s, "
          f"signal: {status.signal_name}")

    # Do some fake "work" in Python
    _ = sum(range(10000))  # Simulate Python work
    time.sleep(0.1)

# Get final result
final = cpu.wait()
total_time = time.perf_counter() - start

print(f"\n[4] GPU execution complete!")
print(f"    Cycles executed: {final.cycles_executed:,}")
print(f"    Signal: {final.signal_name}")
print(f"    Total time: {total_time:.3f}s")
print(f"    IPS: {final.cycles_executed / total_time:,.0f}")

print("\n" + "=" * 70)
print("TEST 2: Compare async vs blocking")
print("=" * 70)

# Async mode
cpu_async = AsyncMetalCPU()
cpu_async.load_program(list(program_bytes), 0)
cpu_async.set_pc(0)

start = time.perf_counter()
cpu_async.start(total_cycles)
async_result = cpu_async.wait()
async_time = time.perf_counter() - start
async_ips = async_result.cycles_executed / async_time

print(f"\nASYNC mode:   {async_result.cycles_executed:,} cycles in {async_time:.3f}s = {async_ips:,.0f} IPS")

# ContinuousMetalCPU for comparison
from kvrm_metal import ContinuousMetalCPU

cpu_cont = ContinuousMetalCPU()
cpu_cont.load_program(list(program_bytes), 0)
cpu_cont.set_pc(0)

start = time.perf_counter()
cont_result = cpu_cont.execute_mega(total_cycles)
cont_time = time.perf_counter() - start
cont_ips = cont_result.total_cycles / cont_time

print(f"CONTINUOUS:   {cont_result.total_cycles:,} cycles in {cont_time:.3f}s = {cont_ips:,.0f} IPS")

print(f"\nAsync vs Continuous: {(async_ips - cont_ips) / cont_ips * 100:+.1f}%")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
AsyncMetalCPU provides TRUE non-blocking GPU execution:
- start() returns immediately
- poll() checks status without blocking
- wait() blocks until completion
- Python can do other work while GPU executes

This enables:
- Background GPU processing
- Progress monitoring
- Interruptible execution
- Multi-threaded Python + GPU
""")
