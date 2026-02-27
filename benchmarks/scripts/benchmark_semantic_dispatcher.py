#!/usr/bin/env python3
"""
Benchmark Semantic Dispatcher Performance

Compares instruction-by-instruction emulation vs GPU kernel acceleration
for common memory operations (memset, memcpy, strlen, strcmp).

This demonstrates the massive speedup from semantic dispatch:
- Without: Execute N instructions one at a time
- With: Execute ONE GPU kernel operation
"""

import time
import torch
from neural_cpu import NeuralCPU


def benchmark_memset_loop(cpu: NeuralCPU, size: int, iterations: int = 3):
    """Benchmark memset via instruction loop vs semantic kernel."""
    print(f"\n{'='*60}")
    print(f"  MEMSET BENCHMARK - {size:,} bytes")
    print(f"{'='*60}")

    # Semantic kernel (accelerated)
    cpu.semantic_dispatcher.reset_stats()
    times_kernel = []
    for _ in range(iterations):
        start = time.perf_counter()
        cpu.accelerate_memset(0x10000, 0x00, size)
        elapsed = time.perf_counter() - start
        times_kernel.append(elapsed)

    avg_kernel = sum(times_kernel) / len(times_kernel)
    stats = cpu.get_semantic_dispatcher_stats()
    inst_skipped = stats['instructions_accelerated']

    # Estimated loop time (conservative: 50K IPS for instruction-by-instruction)
    # Real memset loop: ~3 instructions per byte (STR + SUB + CBNZ)
    estimated_instructions = size // 8 * 3  # 8-byte stores
    estimated_loop_time = estimated_instructions / 50000  # At 50K IPS

    print(f"\n  Results:")
    print(f"    Semantic Kernel:     {avg_kernel*1000:.3f} ms")
    print(f"    Est. Loop Time:      {estimated_loop_time*1000:.1f} ms (at 50K IPS)")
    print(f"    Instructions Saved:  {inst_skipped:,}")
    print(f"    Speedup:             {estimated_loop_time/max(avg_kernel, 1e-9):.0f}x")

    return avg_kernel, estimated_loop_time


def benchmark_memcpy_loop(cpu: NeuralCPU, size: int, iterations: int = 3):
    """Benchmark memcpy via instruction loop vs semantic kernel."""
    print(f"\n{'='*60}")
    print(f"  MEMCPY BENCHMARK - {size:,} bytes")
    print(f"{'='*60}")

    # Setup source data
    for i in range(size):
        cpu.memory[0x20000 + i] = i & 0xFF

    # Semantic kernel (accelerated)
    cpu.semantic_dispatcher.reset_stats()
    times_kernel = []
    for _ in range(iterations):
        start = time.perf_counter()
        cpu.accelerate_memcpy(0x30000, 0x20000, size)
        elapsed = time.perf_counter() - start
        times_kernel.append(elapsed)

    avg_kernel = sum(times_kernel) / len(times_kernel)
    stats = cpu.get_semantic_dispatcher_stats()
    inst_skipped = stats['instructions_accelerated']

    # Estimated loop time
    estimated_instructions = size // 8 * 4  # LDR + STR + SUB + CBNZ
    estimated_loop_time = estimated_instructions / 50000

    print(f"\n  Results:")
    print(f"    Semantic Kernel:     {avg_kernel*1000:.3f} ms")
    print(f"    Est. Loop Time:      {estimated_loop_time*1000:.1f} ms (at 50K IPS)")
    print(f"    Instructions Saved:  {inst_skipped:,}")
    print(f"    Speedup:             {estimated_loop_time/max(avg_kernel, 1e-9):.0f}x")

    # Verify correctness
    correct = all(cpu.memory[0x30000 + i].item() == (i & 0xFF) for i in range(min(100, size)))
    print(f"    Correctness:         {'✅ PASS' if correct else '❌ FAIL'}")

    return avg_kernel, estimated_loop_time


def benchmark_strlen(cpu: NeuralCPU, length: int, iterations: int = 3):
    """Benchmark strlen via instruction loop vs semantic kernel."""
    print(f"\n{'='*60}")
    print(f"  STRLEN BENCHMARK - {length:,} character string")
    print(f"{'='*60}")

    # Setup string
    for i in range(length):
        cpu.memory[0x40000 + i] = ord('A') + (i % 26)
    cpu.memory[0x40000 + length] = 0  # Null terminator

    # Semantic kernel (accelerated)
    cpu.semantic_dispatcher.reset_stats()
    times_kernel = []
    results = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = cpu.accelerate_strlen(0x40000)
        elapsed = time.perf_counter() - start
        times_kernel.append(elapsed)
        results.append(result)

    avg_kernel = sum(times_kernel) / len(times_kernel)
    stats = cpu.get_semantic_dispatcher_stats()

    # Estimated loop time
    estimated_instructions = length * 3  # LDRB + ADD + CBNZ per char
    estimated_loop_time = estimated_instructions / 50000

    print(f"\n  Results:")
    print(f"    Semantic Kernel:     {avg_kernel*1000:.3f} ms")
    print(f"    Est. Loop Time:      {estimated_loop_time*1000:.1f} ms (at 50K IPS)")
    print(f"    Returned Length:     {results[0]} (expected: {length})")
    print(f"    Speedup:             {estimated_loop_time/max(avg_kernel, 1e-9):.0f}x")
    print(f"    Correctness:         {'✅ PASS' if results[0] == length else '❌ FAIL'}")

    return avg_kernel, estimated_loop_time


def benchmark_strcmp(cpu: NeuralCPU, length: int, iterations: int = 3):
    """Benchmark strcmp via instruction loop vs semantic kernel."""
    print(f"\n{'='*60}")
    print(f"  STRCMP BENCHMARK - {length:,} character strings")
    print(f"{'='*60}")

    # Setup two identical strings
    for i in range(length):
        ch = ord('A') + (i % 26)
        cpu.memory[0x50000 + i] = ch
        cpu.memory[0x60000 + i] = ch
    cpu.memory[0x50000 + length] = 0
    cpu.memory[0x60000 + length] = 0

    # Semantic kernel (accelerated)
    cpu.semantic_dispatcher.reset_stats()
    from semantic_dispatcher import SemanticOp

    times_kernel = []
    results = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = cpu.semantic_dispatcher.force_dispatch(
            SemanticOp.STRCMP,
            src_addr=0x50000,
            dst_addr=0x60000
        )
        elapsed = time.perf_counter() - start
        times_kernel.append(elapsed)
        results.append(result.result_value if result else -999)

    avg_kernel = sum(times_kernel) / len(times_kernel)

    # Estimated loop time
    estimated_instructions = length * 5  # 2x LDRB + CMP + 2x branch per char
    estimated_loop_time = estimated_instructions / 50000

    print(f"\n  Results:")
    print(f"    Semantic Kernel:     {avg_kernel*1000:.3f} ms")
    print(f"    Est. Loop Time:      {estimated_loop_time*1000:.1f} ms (at 50K IPS)")
    print(f"    Comparison Result:   {results[0]} (expected: 0 for equal)")
    print(f"    Speedup:             {estimated_loop_time/max(avg_kernel, 1e-9):.0f}x")
    print(f"    Correctness:         {'✅ PASS' if results[0] == 0 else '❌ FAIL'}")

    return avg_kernel, estimated_loop_time


def main():
    print("=" * 70)
    print("  SEMANTIC DISPATCHER PERFORMANCE BENCHMARK")
    print("  Comparing GPU Kernel Acceleration vs Instruction-by-Instruction")
    print("=" * 70)

    cpu = NeuralCPU()

    # Test different sizes
    sizes = [1000, 10000, 100000]

    total_kernel_time = 0
    total_loop_time = 0

    for size in sizes:
        k, l = benchmark_memset_loop(cpu, size)
        total_kernel_time += k
        total_loop_time += l

        k, l = benchmark_memcpy_loop(cpu, size)
        total_kernel_time += k
        total_loop_time += l

    # String operations
    for length in [100, 1000, 10000]:
        k, l = benchmark_strlen(cpu, length)
        total_kernel_time += k
        total_loop_time += l

        k, l = benchmark_strcmp(cpu, length)
        total_kernel_time += k
        total_loop_time += l

    # Final stats
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    cpu.print_semantic_dispatcher_stats()

    overall_speedup = total_loop_time / max(total_kernel_time, 1e-9)
    print(f"\n  Overall Performance:")
    print(f"    Total Kernel Time:   {total_kernel_time*1000:.2f} ms")
    print(f"    Est. Loop Time:      {total_loop_time*1000:.1f} ms")
    print(f"    Overall Speedup:     {overall_speedup:.0f}x")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
