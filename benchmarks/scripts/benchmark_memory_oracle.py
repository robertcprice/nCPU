#!/usr/bin/env python3
"""
Benchmark: Memory Oracle Performance Analysis

Tests the LSTM-based memory access predictor and prefetcher to measure:
1. Stride detection accuracy
2. Prefetch hit rate
3. Pattern recognition
4. Overall performance impact

This is Phase 1 of the "Intelligent Dispatcher" architecture.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


def create_test_binary_sequential():
    """Create a binary that does sequential memory access."""
    # Simple program that:
    # 1. Loads X0 with base address (0x1000)
    # 2. Loops: LDR X1, [X0], #8 then SUB X2, X2, #1 then CBNZ X2, loop
    code = bytearray()

    # MOVZ X0, #0x1000 (base address)
    code += bytes([0x00, 0x00, 0x82, 0xD2])  # movz x0, #0x1000

    # MOVZ X2, #100 (loop counter)
    code += bytes([0x82, 0x0C, 0x80, 0xD2])  # movz x2, #100

    # loop: LDR X1, [X0], #8 (post-increment)
    code += bytes([0x01, 0x84, 0x40, 0xF8])  # ldr x1, [x0], #8

    # SUB X2, X2, #1
    code += bytes([0x42, 0x04, 0x00, 0xD1])  # sub x2, x2, #1

    # CBNZ X2, loop (-8 = 0xFFFFFFFE, imm19 = 0x7FFFE)
    code += bytes([0xE2, 0xFF, 0xFF, 0xB5])  # cbnz x2, -8

    # HLT
    code += bytes([0x00, 0x00, 0x40, 0xD4])

    return bytes(code)


def create_test_binary_strided():
    """Create a binary that does strided memory access (struct simulation)."""
    code = bytearray()

    # MOVZ X0, #0x2000 (base address)
    code += bytes([0x00, 0x00, 0x84, 0xD2])  # movz x0, #0x2000

    # MOVZ X2, #50 (loop counter)
    code += bytes([0x42, 0x06, 0x80, 0xD2])  # movz x2, #50

    # loop: LDR X1, [X0], #24 (struct stride of 24 bytes)
    code += bytes([0x01, 0x84, 0x41, 0xF8])  # ldr x1, [x0], #24

    # SUB X2, X2, #1
    code += bytes([0x42, 0x04, 0x00, 0xD1])  # sub x2, x2, #1

    # CBNZ X2, loop
    code += bytes([0xE2, 0xFF, 0xFF, 0xB5])  # cbnz x2, -8

    # HLT
    code += bytes([0x00, 0x00, 0x40, 0xD4])

    return bytes(code)


def create_test_binary_random():
    """Create a binary that simulates random memory access (pointer chasing)."""
    code = bytearray()

    # This simulates linked list traversal
    # Load pointer, follow it, repeat

    # MOVZ X0, #0x3000 (start of linked list)
    code += bytes([0x00, 0x00, 0x86, 0xD2])  # movz x0, #0x3000

    # MOVZ X2, #20 (max iterations)
    code += bytes([0x82, 0x02, 0x80, 0xD2])  # movz x2, #20

    # loop: LDR X0, [X0] (follow pointer - X0 = *X0)
    code += bytes([0x00, 0x00, 0x40, 0xF9])  # ldr x0, [x0]

    # SUB X2, X2, #1
    code += bytes([0x42, 0x04, 0x00, 0xD1])  # sub x2, x2, #1

    # CBNZ X2, loop
    code += bytes([0xE2, 0xFF, 0xFF, 0xB5])  # cbnz x2, -8

    # HLT
    code += bytes([0x00, 0x00, 0x40, 0xD4])

    return bytes(code)


def setup_linked_list(cpu, base=0x3000, num_nodes=30):
    """Setup a linked list in memory for pointer chasing test."""
    # Create a chain of pointers: base -> base+256 -> base+512 -> ...
    for i in range(num_nodes):
        addr = base + i * 256
        next_addr = base + (i + 1) * 256 if i < num_nodes - 1 else 0

        # Write next pointer (little-endian 64-bit)
        for j in range(8):
            cpu.memory[addr + j] = (next_addr >> (j * 8)) & 0xFF


def run_benchmark(test_name, binary_func, setup_func=None, iterations=3):
    """Run a benchmark and return statistics."""
    from neural_cpu import NeuralCPU as NeuralARM64

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {test_name}")
    print(f"{'='*60}")

    # Create CPU
    cpu = NeuralARM64()

    # Setup test data if needed
    if setup_func:
        setup_func(cpu)

    # Load binary
    binary = binary_func()
    cpu.load_binary(binary, addr=0x1000)
    cpu.pc = torch.tensor(0x1000, dtype=torch.int64, device=cpu.device)
    cpu.regs[31] = torch.tensor(0xFFF00, dtype=torch.int64, device=cpu.device)  # SP

    # Reset stats
    cpu.memory_oracle.reset_stats()

    # Run execution
    print(f"  Running {iterations} iterations...")
    total_time = 0
    total_instructions = 0

    for i in range(iterations):
        # Reset PC
        cpu.pc = torch.tensor(0x1000, dtype=torch.int64, device=cpu.device)
        cpu.halted = False
        cpu.memory_oracle.reset_stats()

        start = time.perf_counter()
        executed, elapsed = cpu.run_gpu_microbatch(max_instructions=10000)
        total_time += elapsed
        total_instructions += int(executed.item()) if hasattr(executed, 'item') else executed

        print(f"    Iteration {i+1}: {int(executed.item()) if hasattr(executed, 'item') else executed} instructions in {elapsed:.4f}s")

    # Get stats
    stats = cpu.get_memory_oracle_stats()

    print(f"\n  Results:")
    print(f"    Total Instructions: {total_instructions:,}")
    print(f"    Total Time: {total_time:.4f}s")
    print(f"    Average IPS: {total_instructions/total_time:,.0f}")
    print(f"\n  Memory Oracle Stats:")
    print(f"    Memory Accesses: {stats['total_accesses']:,}")
    print(f"    Predictions: {stats['predictions_made']:,}")
    print(f"    Prefetch Hits: {stats['prefetch_hits']:,}")
    print(f"    Hit Rate: {stats['prefetch_hit_rate']:.2%}")
    print(f"    Detected Pattern: {stats['detected_pattern']}")
    print(f"    Pattern Confidence: {stats['pattern_confidence']:.2%}")

    return {
        'name': test_name,
        'instructions': total_instructions,
        'time': total_time,
        'ips': total_instructions / total_time,
        'memory_stats': stats
    }


def benchmark_oracle_disabled(test_name, binary_func, setup_func=None):
    """Run same test with Oracle disabled for comparison."""
    from neural_cpu import NeuralCPU as NeuralARM64

    print(f"\n  [Baseline - Oracle Disabled]")

    cpu = NeuralARM64()
    cpu.memory_oracle_enabled = False  # Disable Oracle

    if setup_func:
        setup_func(cpu)

    binary = binary_func()
    cpu.load_binary(binary, addr=0x1000)
    cpu.pc = torch.tensor(0x1000, dtype=torch.int64, device=cpu.device)
    cpu.regs[31] = torch.tensor(0xFFF00, dtype=torch.int64, device=cpu.device)

    start = time.perf_counter()
    executed, elapsed = cpu.run_gpu_microbatch(max_instructions=10000)
    total = int(executed.item()) if hasattr(executed, 'item') else executed

    print(f"    Instructions: {total:,}")
    print(f"    Time: {elapsed:.4f}s")
    print(f"    IPS (no oracle): {total/elapsed:,.0f}")

    return {'ips_baseline': total / elapsed}


def main():
    print("=" * 70)
    print("  MEMORY ORACLE BENCHMARK SUITE")
    print("  Phase 1: Intelligent Dispatcher Architecture")
    print("=" * 70)

    results = []

    # Test 1: Sequential access
    result = run_benchmark(
        "Sequential Memory Access",
        create_test_binary_sequential
    )
    baseline = benchmark_oracle_disabled(
        "Sequential",
        create_test_binary_sequential
    )
    result['baseline'] = baseline
    results.append(result)

    # Test 2: Strided access
    result = run_benchmark(
        "Strided Memory Access (Struct)",
        create_test_binary_strided
    )
    baseline = benchmark_oracle_disabled(
        "Strided",
        create_test_binary_strided
    )
    result['baseline'] = baseline
    results.append(result)

    # Test 3: Random/pointer chasing
    result = run_benchmark(
        "Pointer Chasing (Linked List)",
        create_test_binary_random,
        setup_func=setup_linked_list
    )
    baseline = benchmark_oracle_disabled(
        "Pointer Chasing",
        create_test_binary_random,
        setup_func=setup_linked_list
    )
    result['baseline'] = baseline
    results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Test':<30} {'Oracle IPS':<15} {'Baseline IPS':<15} {'Improvement':<15}")
    print("-" * 70)

    for r in results:
        oracle_ips = r['ips']
        baseline_ips = r['baseline']['ips_baseline']
        improvement = (oracle_ips / baseline_ips - 1) * 100 if baseline_ips > 0 else 0
        print(f"{r['name']:<30} {oracle_ips:>12,.0f} {baseline_ips:>14,.0f} {improvement:>13.1f}%")

    print("\n" + "=" * 70)
    print("  MEMORY ORACLE BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
