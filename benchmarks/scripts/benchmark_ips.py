#!/usr/bin/env python3
"""
IPS (Instructions Per Second) Benchmark for Neural Metal CPU

Measures the performance of the Rust Metal GPU kernel executing ARM64 instructions.
"""

import time
import sys

def benchmark_continuous_cpu():
    """Benchmark ContinuousMetalCPU with various workloads."""
    from kvrm_metal import ContinuousMetalCPU

    print("=" * 70)
    print("NEURAL CPU IPS BENCHMARK - Metal GPU Execution")
    print("=" * 70)

    results = []

    # Test 1: Simple loop (ADD X0, X0, #1; B -1)
    print("\n[Test 1] Simple increment loop (ADD X0, X0, #1)")
    cpu = ContinuousMetalCPU()
    program = bytes([
        0x00, 0x04, 0x00, 0x91,  # ADD X0, X0, #1
        0xFF, 0xFF, 0xFF, 0x17,  # B -1 (loop back)
    ])
    cpu.load_program(list(program), 0)
    cpu.set_pc(0)
    cpu.set_register(0, 0)

    # Warm up
    cpu.execute_continuous(1000)
    cpu.set_register(0, 0)
    cpu.set_pc(0)

    # Timed run
    start = time.perf_counter()
    result = cpu.execute_continuous(10_000_000)  # 10M max instructions
    elapsed = time.perf_counter() - start

    x0_val = cpu.get_register(0)
    ips = x0_val / elapsed if elapsed > 0 else 0
    results.append(("Simple ADD loop", ips, x0_val, elapsed))
    print(f"   Instructions executed: {x0_val:,}")
    print(f"   Time: {elapsed:.4f}s")
    print(f"   IPS: {ips:,.0f} ({ips/1e6:.2f}M IPS)")

    # Test 2: Mixed arithmetic (ADD, SUB alternating)
    print("\n[Test 2] Mixed arithmetic (ADD/SUB alternating)")
    cpu2 = ContinuousMetalCPU()
    program2 = bytes([
        0x00, 0x08, 0x00, 0x91,  # ADD X0, X0, #2
        0x00, 0x04, 0x00, 0xD1,  # SUB X0, X0, #1
        0xFE, 0xFF, 0xFF, 0x17,  # B -2 (loop back)
    ])
    cpu2.load_program(list(program2), 0)
    cpu2.set_pc(0)
    cpu2.set_register(0, 0)

    start = time.perf_counter()
    result2 = cpu2.execute_continuous(10_000_000)
    elapsed = time.perf_counter() - start

    x0_val = cpu2.get_register(0)
    # Each iteration does +2, -1 = net +1, so x0_val = iterations
    ips = (x0_val * 3) / elapsed if elapsed > 0 else 0  # 3 instructions per iteration
    results.append(("Mixed ADD/SUB loop", ips, x0_val * 3, elapsed))
    print(f"   Iterations: {x0_val:,}")
    print(f"   Instructions: {x0_val * 3:,}")
    print(f"   Time: {elapsed:.4f}s")
    print(f"   IPS: {ips:,.0f} ({ips/1e6:.2f}M IPS)")

    # Test 3: Memory-intensive (LDR/STR loop)
    print("\n[Test 3] Memory-intensive (LDR/STR loop)")
    cpu3 = ContinuousMetalCPU()
    # X1 = address, X0 = counter
    # Loop: LDR X2, [X1]; ADD X2, X2, #1; STR X2, [X1]; ADD X0, X0, #1; B loop
    program3 = bytes([
        0x22, 0x00, 0x40, 0xF9,  # LDR X2, [X1]
        0x42, 0x04, 0x00, 0x91,  # ADD X2, X2, #1
        0x22, 0x00, 0x00, 0xF9,  # STR X2, [X1]
        0x00, 0x04, 0x00, 0x91,  # ADD X0, X0, #1
        0xFC, 0xFF, 0xFF, 0x17,  # B -4 (loop back)
    ])
    cpu3.load_program(list(program3), 0)
    cpu3.set_pc(0)
    cpu3.set_register(0, 0)
    cpu3.set_register(1, 0x10000)  # Memory address for load/store

    start = time.perf_counter()
    result3 = cpu3.execute_continuous(10_000_000)
    elapsed = time.perf_counter() - start

    x0_val = cpu3.get_register(0)
    ips = (x0_val * 5) / elapsed if elapsed > 0 else 0  # 5 instructions per iteration
    results.append(("Memory LDR/STR loop", ips, x0_val * 5, elapsed))
    print(f"   Iterations: {x0_val:,}")
    print(f"   Instructions: {x0_val * 5:,}")
    print(f"   Time: {elapsed:.4f}s")
    print(f"   IPS: {ips:,.0f} ({ips/1e6:.2f}M IPS)")

    # Test 4: MUL-heavy workload
    print("\n[Test 4] Multiply-heavy workload")
    cpu4 = ContinuousMetalCPU()
    # X0 = counter, X1 = value, X2 = multiplier (2)
    # Loop: MUL X1, X1, X2 (will overflow quickly, but measures MUL speed)
    program4 = bytes([
        0x21, 0x7C, 0x02, 0x9B,  # MUL X1, X1, X2
        0x00, 0x04, 0x00, 0x91,  # ADD X0, X0, #1
        0xFE, 0xFF, 0xFF, 0x17,  # B -2
    ])
    cpu4.load_program(list(program4), 0)
    cpu4.set_pc(0)
    cpu4.set_register(0, 0)
    cpu4.set_register(1, 1)
    cpu4.set_register(2, 2)

    start = time.perf_counter()
    result4 = cpu4.execute_continuous(10_000_000)
    elapsed = time.perf_counter() - start

    x0_val = cpu4.get_register(0)
    ips = (x0_val * 3) / elapsed if elapsed > 0 else 0  # 3 instructions per iteration
    results.append(("MUL-heavy loop", ips, x0_val * 3, elapsed))
    print(f"   Iterations: {x0_val:,}")
    print(f"   Instructions: {x0_val * 3:,}")
    print(f"   Time: {elapsed:.4f}s")
    print(f"   IPS: {ips:,.0f} ({ips/1e6:.2f}M IPS)")

    # Test 5: Logical operations (AND, ORR, EOR)
    print("\n[Test 5] Logical operations (AND, ORR, EOR)")
    cpu5 = ContinuousMetalCPU()
    program5 = bytes([
        0x21, 0x00, 0x02, 0x8A,  # AND X1, X1, X2
        0x21, 0x00, 0x03, 0xAA,  # ORR X1, X1, X3
        0x21, 0x00, 0x04, 0xCA,  # EOR X1, X1, X4
        0x00, 0x04, 0x00, 0x91,  # ADD X0, X0, #1
        0xFC, 0xFF, 0xFF, 0x17,  # B -4
    ])
    cpu5.load_program(list(program5), 0)
    cpu5.set_pc(0)
    cpu5.set_register(0, 0)
    cpu5.set_register(1, 0xFFFFFFFF)
    cpu5.set_register(2, 0xF0F0F0F0)
    cpu5.set_register(3, 0x0F0F0F0F)
    cpu5.set_register(4, 0xAAAAAAAA)

    start = time.perf_counter()
    result5 = cpu5.execute_continuous(10_000_000)
    elapsed = time.perf_counter() - start

    x0_val = cpu5.get_register(0)
    ips = (x0_val * 5) / elapsed if elapsed > 0 else 0  # 5 instructions per iteration
    results.append(("Logical ops loop", ips, x0_val * 5, elapsed))
    print(f"   Iterations: {x0_val:,}")
    print(f"   Instructions: {x0_val * 5:,}")
    print(f"   Time: {elapsed:.4f}s")
    print(f"   IPS: {ips:,.0f} ({ips/1e6:.2f}M IPS)")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Test':<25} {'IPS':>15} {'M IPS':>10} {'Time':>10}")
    print("-" * 60)

    total_ips = 0
    for name, ips, insts, time_s in results:
        print(f"{name:<25} {ips:>15,.0f} {ips/1e6:>10.2f} {time_s:>10.4f}s")
        total_ips += ips

    avg_ips = total_ips / len(results)
    print("-" * 60)
    print(f"{'AVERAGE':<25} {avg_ips:>15,.0f} {avg_ips/1e6:>10.2f}")
    print("=" * 70)

    return results


def benchmark_peak_throughput():
    """Measure peak GPU throughput with maximum batch size."""
    from kvrm_metal import ContinuousMetalCPU

    print("\n" + "=" * 70)
    print("PEAK THROUGHPUT TEST")
    print("=" * 70)

    cpu = ContinuousMetalCPU()

    # Simple increment loop
    program = bytes([
        0x00, 0x04, 0x00, 0x91,  # ADD X0, X0, #1
        0xFF, 0xFF, 0xFF, 0x17,  # B -1
    ])
    cpu.load_program(list(program), 0)
    cpu.set_pc(0)
    cpu.set_register(0, 0)

    # Run for 1 second to measure sustained throughput
    print("\nRunning 1-second sustained throughput test...")
    start = time.perf_counter()
    target_end = start + 1.0

    total_instructions = 0
    batches = 0

    while time.perf_counter() < target_end:
        result = cpu.execute_continuous(100_000_000)  # 100M per batch
        current_x0 = cpu.get_register(0)
        total_instructions = current_x0
        batches += 1

    elapsed = time.perf_counter() - start
    peak_ips = total_instructions / elapsed

    print(f"\n   Sustained run time: {elapsed:.4f}s")
    print(f"   Total instructions: {total_instructions:,}")
    print(f"   GPU batches: {batches}")
    print(f"   PEAK IPS: {peak_ips:,.0f}")
    print(f"   PEAK MIPS: {peak_ips/1e6:.2f}")
    print(f"   PEAK GIPS: {peak_ips/1e9:.4f}")

    return peak_ips


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  KVRM Neural CPU - Metal GPU Performance Benchmark")
    print("  Running on Apple Silicon with Metal Shaders")
    print("=" * 70)

    # Run benchmarks
    results = benchmark_continuous_cpu()
    peak_ips = benchmark_peak_throughput()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Device: Apple M4 Pro (Metal GPU)")
    print(f"  Peak Performance: {peak_ips/1e6:.2f} MIPS")
    print(f"  Execution Mode: 100% GPU (zero .item() calls)")
    print("=" * 70)
