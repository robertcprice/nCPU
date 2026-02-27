#!/usr/bin/env python3
"""
Benchmark: Memory Oracle Performance on REALISTIC Memory Patterns

Tests the Memory Oracle on patterns that CANNOT be loop-vectorized:
1. Linked list traversal (pointer chasing - address depends on data)
2. Hash table lookup (unpredictable access patterns)
3. Binary search (data-dependent branching)
4. String processing (null-terminated, unpredictable length)

These are the patterns where the Memory Oracle SHOULD help by predicting
and prefetching memory based on learned patterns.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


def setup_longer_linked_list(cpu, base=0x10000, num_nodes=100, stride=64):
    """
    Setup a longer linked list for more realistic pointer chasing.
    Each node contains:
      - [0:7]  next pointer (8 bytes)
      - [8:15] data value (8 bytes)
    """
    for i in range(num_nodes):
        addr = base + i * stride
        next_addr = base + (i + 1) * stride if i < num_nodes - 1 else 0

        # Write next pointer (little-endian 64-bit)
        for j in range(8):
            cpu.memory[addr + j] = (next_addr >> (j * 8)) & 0xFF

        # Write data value (just i for verification)
        data_val = i * 17  # Some pattern
        for j in range(8):
            cpu.memory[addr + 8 + j] = (data_val >> (j * 8)) & 0xFF

    return num_nodes


def create_linked_list_sum_binary():
    """
    Create a binary that traverses a linked list and sums the data values.
    This is the CLASSIC unpredictable memory access pattern.

    Program:
      X0 = base of linked list
      X1 = accumulator (sum)
      X2 = current node pointer

    Loop:
      LDR X3, [X2, #8]   ; Load data value
      ADD X1, X1, X3     ; Accumulate
      LDR X2, [X2]       ; Follow next pointer (UNPREDICTABLE!)
      CBNZ X2, Loop      ; Continue if not NULL
      HLT
    """
    code = bytearray()

    # MOVZ X0, #0x1000 (shifted by 4 to get 0x10000)
    # Actually use movz + movk for larger address
    code += bytes([0x00, 0x02, 0x80, 0xD2])  # movz x0, #0x10 (will shift)
    code += bytes([0x00, 0x00, 0xA0, 0xF2])  # movk x0, #0, lsl #16 -> x0 = 0x10
    code += bytes([0x00, 0x00, 0x20, 0xD3])  # lsl x0, x0, #12 -> x0 = 0x10000

    # MOVZ X1, #0 (accumulator)
    code += bytes([0x01, 0x00, 0x80, 0xD2])  # movz x1, #0

    # MOV X2, X0 (current = base)
    code += bytes([0xE2, 0x03, 0x00, 0xAA])  # mov x2, x0

    # Loop start (offset 20 = 5 instructions)
    # LDR X3, [X2, #8] - Load data value (unsigned offset)
    code += bytes([0x43, 0x04, 0x40, 0xF9])  # ldr x3, [x2, #8]

    # ADD X1, X1, X3 - Accumulate
    code += bytes([0x21, 0x00, 0x03, 0x8B])  # add x1, x1, x3

    # LDR X2, [X2] - Follow next pointer (THIS IS UNPREDICTABLE)
    code += bytes([0x42, 0x00, 0x40, 0xF9])  # ldr x2, [x2]

    # CBNZ X2, loop (-12 bytes = -3 instructions)
    # imm19 = -3 = 0x7FFFD, encoding = 0xB5 | (0x7FFFD << 5) | Rt
    # = 0xB5000000 | 0x0FFFFA0 | 0x02 = 0xB5FFFFA2
    code += bytes([0xA2, 0xFF, 0xFF, 0xB5])  # cbnz x2, -12

    # HLT
    code += bytes([0x00, 0x00, 0x40, 0xD4])

    return bytes(code)


def setup_array_for_binary_search(cpu, base=0x20000, size=256):
    """
    Setup a sorted array for binary search.
    Each element is 8 bytes (64-bit value).
    Values are spaced evenly: 0, 4, 8, 12, ...
    """
    for i in range(size):
        addr = base + i * 8
        val = i * 4  # Sorted values
        for j in range(8):
            cpu.memory[addr + j] = (val >> (j * 8)) & 0xFF

    # Set the target value we're searching for (in middle-ish area)
    target = 127 * 4  # This will require log2(256) = 8 comparisons
    cpu.memory[base - 8] = target & 0xFF
    for j in range(1, 8):
        cpu.memory[base - 8 + j] = (target >> (j * 8)) & 0xFF

    return size


def create_binary_search_binary():
    """
    Create a binary search program.
    This has DATA-DEPENDENT branching - cannot be vectorized.

    Searches sorted array for target value.
    Memory accesses depend on comparison results.
    """
    code = bytearray()

    # Setup pointers
    # X0 = array base (0x20000)
    code += bytes([0x00, 0x04, 0x80, 0xD2])  # movz x0, #0x20 (will shift)
    code += bytes([0x00, 0x00, 0x20, 0xD3])  # lsl x0, x0, #12 -> x0 = 0x20000

    # X1 = low = 0
    code += bytes([0x01, 0x00, 0x80, 0xD2])  # movz x1, #0

    # X2 = high = 255
    code += bytes([0xE2, 0x1F, 0x80, 0xD2])  # movz x2, #255

    # X3 = target value (load from memory at base - 8)
    code += bytes([0x03, 0xFC, 0x5F, 0xF8])  # ldur x3, [x0, #-8]

    # Loop: binary search
    # While (low <= high):

    # CMP X1, X2 (low vs high)
    code += bytes([0x3F, 0x00, 0x02, 0xEB])  # cmp x1, x2

    # B.GT done (if low > high, exit)
    # Will patch this offset
    code += bytes([0x0C, 0x01, 0x00, 0x54])  # b.gt +24 (to HLT)

    # mid = (low + high) / 2
    code += bytes([0x24, 0x00, 0x02, 0x8B])  # add x4, x1, x2
    code += bytes([0x84, 0xFC, 0x41, 0xD3])  # lsr x4, x4, #1

    # Load array[mid] into X5
    # addr = base + mid * 8
    code += bytes([0x85, 0x78, 0x64, 0xF8])  # ldr x5, [x4, x0, lsl #3]

    # CMP X5, X3 (array[mid] vs target)
    code += bytes([0xBF, 0x00, 0x03, 0xEB])  # cmp x5, x3

    # B.EQ found
    code += bytes([0x80, 0x00, 0x00, 0x54])  # b.eq +16 (to found)

    # B.LT go_high (if array[mid] < target, search upper half)
    code += bytes([0x4B, 0x00, 0x00, 0x54])  # b.lt +8

    # Else go_low: high = mid - 1
    code += bytes([0x82, 0x04, 0x00, 0xD1])  # sub x2, x4, #1
    code += bytes([0xEC, 0xFF, 0xFF, 0x17])  # b -20 (back to CMP)

    # go_high: low = mid + 1
    code += bytes([0x81, 0x04, 0x00, 0x91])  # add x1, x4, #1
    code += bytes([0xEA, 0xFF, 0xFF, 0x17])  # b -24 (back to CMP)

    # found: X4 contains the index
    # (fall through to HLT)

    # HLT
    code += bytes([0x00, 0x00, 0x40, 0xD4])

    return bytes(code)


def run_benchmark_no_vectorization(test_name, binary_func, setup_func=None, iterations=5):
    """
    Run benchmark with loop vectorization DISABLED.
    This forces instruction-by-instruction execution so we can properly
    measure Memory Oracle effectiveness.
    """
    from neural_cpu import NeuralCPU as NeuralARM64

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {test_name}")
    print(f"{'='*60}")

    # Create CPU
    cpu = NeuralARM64()

    # DISABLE loop vectorization to test Memory Oracle properly
    cpu.loop_vectorization_enabled = False

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
    print(f"  Running {iterations} iterations (loop vectorization DISABLED)...")
    total_time = 0
    total_instructions = 0

    for i in range(iterations):
        # Reset PC
        cpu.pc = torch.tensor(0x1000, dtype=torch.int64, device=cpu.device)
        cpu.halted = False
        cpu.memory_oracle.reset_stats()

        start = time.perf_counter()
        executed, elapsed = cpu.run_gpu_microbatch(max_instructions=100000)
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


def run_baseline_no_vectorization(test_name, binary_func, setup_func=None):
    """Run same test with Oracle disabled for comparison."""
    from neural_cpu import NeuralCPU as NeuralARM64

    print(f"\n  [Baseline - Oracle Disabled, No Vectorization]")

    cpu = NeuralARM64()
    cpu.memory_oracle_enabled = False
    cpu.loop_vectorization_enabled = False

    if setup_func:
        setup_func(cpu)

    binary = binary_func()
    cpu.load_binary(binary, addr=0x1000)
    cpu.pc = torch.tensor(0x1000, dtype=torch.int64, device=cpu.device)
    cpu.regs[31] = torch.tensor(0xFFF00, dtype=torch.int64, device=cpu.device)

    start = time.perf_counter()
    executed, elapsed = cpu.run_gpu_microbatch(max_instructions=100000)
    total = int(executed.item()) if hasattr(executed, 'item') else executed

    print(f"    Instructions: {total:,}")
    print(f"    Time: {elapsed:.4f}s")
    print(f"    IPS (no oracle): {total/elapsed:,.0f}")

    return {'ips_baseline': total / elapsed}


def main():
    print("=" * 70)
    print("  MEMORY ORACLE REALISTIC BENCHMARK")
    print("  Testing on NON-VECTORIZABLE Memory Patterns")
    print("=" * 70)

    results = []

    # Test 1: Linked list traversal (pointer chasing)
    result = run_benchmark_no_vectorization(
        "Linked List Sum (100 nodes)",
        create_linked_list_sum_binary,
        setup_func=lambda cpu: setup_longer_linked_list(cpu, num_nodes=100)
    )
    baseline = run_baseline_no_vectorization(
        "Linked List",
        create_linked_list_sum_binary,
        setup_func=lambda cpu: setup_longer_linked_list(cpu, num_nodes=100)
    )
    result['baseline'] = baseline
    results.append(result)

    # Test 2: Binary search (data-dependent branching)
    result = run_benchmark_no_vectorization(
        "Binary Search (256 elements)",
        create_binary_search_binary,
        setup_func=setup_array_for_binary_search
    )
    baseline = run_baseline_no_vectorization(
        "Binary Search",
        create_binary_search_binary,
        setup_func=setup_array_for_binary_search
    )
    result['baseline'] = baseline
    results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY - Realistic Memory Patterns")
    print("=" * 70)
    print(f"{'Test':<35} {'Oracle IPS':<15} {'Baseline IPS':<15} {'Change':<15}")
    print("-" * 70)

    for r in results:
        oracle_ips = r['ips']
        baseline_ips = r['baseline']['ips_baseline']
        change = (oracle_ips / baseline_ips - 1) * 100 if baseline_ips > 0 else 0
        print(f"{r['name']:<35} {oracle_ips:>12,.0f} {baseline_ips:>14,.0f} {change:>13.1f}%")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("  Note: These patterns CANNOT be loop-vectorized,")
    print("  so Memory Oracle predictions are the only speedup path.")
    print("=" * 70)


if __name__ == "__main__":
    main()
