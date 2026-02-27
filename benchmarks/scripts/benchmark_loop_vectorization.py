#!/usr/bin/env python3
"""
Benchmark: Loop Vectorization vs Serial Execution

The key insight: We can't parallelize DEPENDENT instructions,
but we CAN vectorize LOOPS. Instead of:
    X0 = X0 + 1  (1000 times)
We compute:
    X0 = X0 + 1000  (one operation)

This benchmark tests the existing loop detection and vectorization.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from neural_cpu import NeuralCPU, device

def create_simple_loop(iterations: int) -> list:
    """
    Create a simple loop that adds 1 to X0, iterations times.

    loop:
        ADD X0, X0, #1      ; X0 += 1
        SUB X1, X1, #1      ; X1 -= 1 (counter)
        CBNZ X1, loop       ; if X1 != 0, goto loop
    """
    program = []
    for _ in range(iterations):
        program.append(0x91000400)  # ADD X0, X0, #1
        program.append(0xD1000421)  # SUB X1, X1, #1
        # CBNZ would branch back, but for simplicity we unroll

    return program


def create_vectorizable_loop() -> tuple:
    """
    Create a loop that CAN be vectorized: memset-style.

    Returns (program, loop_start_pc, expected_iterations)

    Loop:
        0x00: STRB WZR, [X0], #1    ; *X0++ = 0
        0x04: SUB X1, X1, #1        ; counter--
        0x08: CBNZ X1, 0x00         ; if counter != 0, loop
        0x0C: RET                   ; done
    """
    program = [
        0x3800141F,  # STRB WZR, [X0], #1 (post-index store zero)
        0xD1000421,  # SUB X1, X1, #1
        0xB5FFFFE1,  # CBNZ X1, -8 (back to start)
        0xD65F03C0,  # RET
    ]
    return program, 0, None  # iterations determined by X1


def benchmark_simple_serial():
    """Benchmark simple serial execution."""
    print("─" * 70)
    print("Test: Simple Serial Execution (unrolled loop)")
    print("─" * 70)

    cpu = NeuralCPU(memory_size=1024*1024)

    # Create 10000 ADD instructions (unrolled loop)
    iterations = 10000
    program = create_simple_loop(iterations)

    # Write to memory
    for i, inst in enumerate(program):
        addr = i * 4
        for j in range(4):
            cpu.memory[addr + j] = (inst >> (j * 8)) & 0xFF

    cpu.pc.fill_(0)
    cpu.regs.zero_()
    cpu.halted = False

    start = time.perf_counter()
    executed, elapsed = cpu.run(len(program))
    total = time.perf_counter() - start

    ips = executed / elapsed if elapsed > 0 else 0
    print(f"  Unrolled iterations: {iterations}")
    print(f"  Executed: {executed:,} instructions")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  IPS: {ips:,.0f}")
    print(f"  Final X0: {int(cpu.regs[0].item())} (expected {iterations})")
    print()

    return executed, elapsed, ips


def benchmark_loop_vectorization():
    """Benchmark with loop that should be vectorized."""
    print("─" * 70)
    print("Test: Vectorizable Memset Loop")
    print("─" * 70)

    cpu = NeuralCPU(memory_size=1024*1024)

    program, loop_start, _ = create_vectorizable_loop()

    # Write to memory
    for i, inst in enumerate(program):
        addr = i * 4
        for j in range(4):
            cpu.memory[addr + j] = (inst >> (j * 8)) & 0xFF

    # Set up: X0 = destination (0x10000), X1 = count (1000)
    iterations = 1000
    dest_addr = 0x10000
    cpu.pc.fill_(0)
    cpu.regs.zero_()
    cpu.regs[0] = dest_addr  # X0 = destination
    cpu.regs[1] = iterations  # X1 = count
    cpu.regs[30] = 0x20000  # LR = return address (arbitrary)
    cpu.halted = False

    start = time.perf_counter()
    executed, elapsed = cpu.run(iterations * 3 + 10)  # 3 instr/iter + safety
    total = time.perf_counter() - start

    ips = executed / elapsed if elapsed > 0 else 0
    print(f"  Loop iterations: {iterations}")
    print(f"  Executed: {executed:,} instructions")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  IPS: {ips:,.0f}")
    print(f"  Final X0: 0x{int(cpu.regs[0].item()):X} (expected 0x{dest_addr + iterations:X})")
    print(f"  Final X1: {int(cpu.regs[1].item())} (expected 0)")

    # Check if memory was zeroed
    zeroed = True
    for i in range(min(10, iterations)):
        if cpu.memory[dest_addr + i] != 0:
            zeroed = False
            break
    print(f"  Memory zeroed: {'✅' if zeroed else '❌'}")
    print()

    return executed, elapsed, ips


def benchmark_counter_loop():
    """
    Benchmark a counter loop that SHOULD be vectorized.

    Loop:
        0x00: ADD X0, X0, #1        ; counter++
        0x04: SUB X1, X1, #1        ; remaining--
        0x08: CBNZ X1, 0x00         ; if remaining != 0, loop
        0x0C: (halt)
    """
    print("─" * 70)
    print("Test: Counter Loop (ADD + SUB + CBNZ)")
    print("─" * 70)

    cpu = NeuralCPU(memory_size=1024*1024)

    program = [
        0x91000400,  # ADD X0, X0, #1
        0xD1000421,  # SUB X1, X1, #1
        0xB5FFFFC1,  # CBNZ X1, -2 (back 2 instructions)
        0x00000000,  # HALT
    ]

    # Write to memory
    for i, inst in enumerate(program):
        addr = i * 4
        for j in range(4):
            cpu.memory[addr + j] = (inst >> (j * 8)) & 0xFF

    iterations = 10000
    cpu.pc.fill_(0)
    cpu.regs.zero_()
    cpu.regs[1] = iterations  # X1 = count
    cpu.halted = False

    start = time.perf_counter()
    executed, elapsed = cpu.run(iterations * 3 + 100)
    total = time.perf_counter() - start

    ips = executed / elapsed if elapsed > 0 else 0
    expected_instr = iterations * 3  # 3 instructions per iteration

    print(f"  Loop iterations: {iterations}")
    print(f"  Executed: {executed:,} instructions")
    print(f"  Expected: {expected_instr:,} instructions (3 per iter)")

    if executed < expected_instr:
        speedup = expected_instr / max(1, executed)
        print(f"  🚀 VECTORIZED! {speedup:.1f}x fewer instructions executed")
    else:
        print(f"  Serial execution (no vectorization detected)")

    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  IPS: {ips:,.0f}")
    print(f"  Final X0: {int(cpu.regs[0].item())} (expected {iterations})")
    print(f"  Final X1: {int(cpu.regs[1].item())} (expected 0)")
    print()

    return executed, elapsed, ips


def main():
    print("=" * 70)
    print("  LOOP VECTORIZATION BENCHMARK")
    print("=" * 70)
    print()
    print("  Key insight: Vectorize LOOPS, not individual instructions")
    print("  Instead of X0++ (1000x), compute X0 += 1000 (1x)")
    print()

    results = {}

    results['serial'] = benchmark_simple_serial()
    results['memset'] = benchmark_loop_vectorization()
    results['counter'] = benchmark_counter_loop()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Serial unrolled:     {results['serial'][2]:>10,.0f} IPS")
    print(f"  Memset loop:         {results['memset'][2]:>10,.0f} IPS")
    print(f"  Counter loop:        {results['counter'][2]:>10,.0f} IPS")
    print("=" * 70)


if __name__ == "__main__":
    main()
