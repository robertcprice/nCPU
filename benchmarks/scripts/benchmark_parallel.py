#!/usr/bin/env python3
"""
ParallelMetalCPU Benchmark - Comprehensive testing of parallel GPU execution.

Tests performance with various workloads:
1. Tight loop (synthetic)
2. DOOM frame rendering
3. Linux boot process
"""

import sys
import importlib.util
import time

# Load the shared library directly
spec = importlib.util.spec_from_file_location(
    'kvrm_metal',
    '/Users/bobbyprice/projects/.venv/lib/python3.13/site-packages/kvrm_metal/kvrm_metal.cpython-313-darwin.so'
)
kvrm_metal = importlib.util.module_from_spec(spec)
sys.modules['kvrm_metal'] = kvrm_metal
spec.loader.exec_module(kvrm_metal)

ParallelMetalCPU = kvrm_metal.ParallelMetalCPU


def benchmark_tight_loop(num_lanes=32, cycles=10_000_000):
    """Benchmark: Tight computation loop (synthetic workload)"""
    print(f"\n{'='*60}")
    print(f"TEST: Tight Loop Benchmark ({num_lanes} lanes, {cycles:,} cycles)")
    print(f"{'='*60}")

    cpu = ParallelMetalCPU(num_lanes=num_lanes)

    # ARM64: loop 100M times: MOV X0, #100000000; loop: SUB X0, X0, #1; CBNZ X0, loop; HLT
    program = bytes([
        0x00, 0x1C, 0x80, 0xD2,  # MOV X0, #100000000
        0x20, 0x00, 0x00, 0x71,  # SUBS X0, X0, #1
        0x20, 0x00, 0x00, 0xB5,  # CBNZ X0, -4 (loop back)
        0x00, 0x00, 0x40, 0xD4,  # HLT #0
    ])

    cpu.load_program(list(program), 0)
    cpu.set_pc_all(0)

    start = time.time()
    result = cpu.execute(cycles)
    elapsed = time.time() - start

    print(f"  Total Cycles: {result.total_cycles:,}")
    print(f"  Avg IPS: {result.avg_ips():,.0f}")
    print(f"  Min/Max cycles per lane: {result.min_cycles():,}/{result.max_cycles():,}")
    print(f"  Lane Efficiency: {result.lane_efficiency():.1%}")
    print(f"  Wall Time: {elapsed:.4f}s")

    return result


def benchmark_lane_scaling(max_lanes=128, cycles=1_000_000):
    """Benchmark: Lane scaling (test performance vs number of lanes)"""
    print(f"\n{'='*60}")
    print(f"TEST: Lane Scaling Benchmark")
    print(f"{'='*60}")

    results = []
    test_lanes = [1, 8, 16, 32, 64, 128]

    # Simple tight loop program
    program = bytes([
        0x00, 0x1C, 0x80, 0xD2,  # MOV X0, #100000000
        0x20, 0x00, 0x00, 0x71,  # SUBS X0, X0, #1
        0x20, 0x00, 0x00, 0xB5,  # CBNZ X0, -4 (loop back)
        0x00, 0x00, 0x40, 0xD4,  # HLT #0
    ])

    for num_lanes in test_lanes:
        if num_lanes > max_lanes:
            continue

        cpu = ParallelMetalCPU(num_lanes=num_lanes)
        cpu.load_program(list(program), 0)
        cpu.set_pc_all(0)

        result = cpu.execute(cycles)

        print(f"  {num_lanes:3d} lanes: {result.avg_ips():>12,.0f} IPS | Efficiency: {result.lane_efficiency():>6.1%}")

        results.append({
            'lanes': num_lanes,
            'ips': result.avg_ips(),
            'efficiency': result.lane_efficiency(),
        })

    return results


def benchmark_doom_frame():
    """Benchmark: DOOM frame rendering (if DOOM binary available)"""
    print(f"\n{'='*60}")
    print(f"TEST: DOOM Frame Rendering")
    print(f"{'='*60}")

    # Try to load DOOM binary
    try:
        with open('doom', 'rb') as f:
            doom_binary = f.read()

        cpu = ParallelMetalCPU(num_lanes=32)

        # Load DOOM at address 0x10000 (typical ARM64 load address)
        cpu.load_program(list(doom_binary), 0x10000)

        # Set PC to DOOM entry point
        cpu.set_pc_all(0x10000)

        # Run for 10M cycles
        result = cpu.execute(10_000_000)

        print(f"  DOOM loaded: {len(doom_binary):,} bytes")
        print(f"  Total Cycles: {result.total_cycles:,}")
        print(f"  Avg IPS: {result.avg_ips():,.0f}")
        print(f"  Final PCs (first 4 lanes): {result.pcs_per_lane[:4]}")

        return result
    except FileNotFoundError:
        print("  SKIPPED: DOOM binary not found")
        return None


def print_summary(tight_loop_result, scaling_results):
    """Print benchmark summary"""
    print(f"\n{'='*60}")
    print(f"SUMMARY: ParallelMetalCPU Performance")
    print(f"{'='*60}")

    if tight_loop_result:
        print(f"\nTight Loop (32 lanes):")
        print(f"  Throughput: {tight_loop_result.avg_ips():,.0f} IPS")
        print(f"  Speedup vs 1 lane: ~{tight_loop_result.num_lanes}x theoretical")

    if scaling_results:
        print(f"\nLane Scaling:")
        best = max(scaling_results, key=lambda r: r['ips'])
        print(f"  Best Configuration: {best['lanes']} lanes @ {best['ips']:,.0f} IPS")
        print(f"  Efficiency at {best['lanes']} lanes: {best['efficiency']:.1%}")


def main():
    """Run all benchmarks"""
    print("="*60)
    print("ParallelMetalCPU Benchmark Suite")
    print("="*60)

    # 1. Tight loop benchmark
    tight_loop_result = benchmark_tight_loop(num_lanes=32, cycles=10_000_000)

    # 2. Lane scaling benchmark
    scaling_results = benchmark_lane_scaling(max_lanes=128, cycles=1_000_000)

    # 3. DOOM frame benchmark
    doom_result = benchmark_doom_frame()

    # 4. Print summary
    print_summary(tight_loop_result, scaling_results)

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
