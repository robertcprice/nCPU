#!/usr/bin/env python3
"""
Neural Weight Optimization via Direct IPS Measurement

Instead of training on synthetic data, directly optimize the 4 neural weights
(RAW, WAW, FLAG, BIAS) to maximize IPS on actual benchmark programs.
"""

import numpy as np
import itertools
from typing import List, Tuple
import kvrm_metal as metal


def create_test_program() -> bytes:
    """Create a test program with mixed dependencies"""
    program = []

    # Pattern 1: Independent adds (should parallelize)
    for i in range(10):
        reg_d = i % 20
        reg_n = (i + 1) % 20
        # ADD Xd, Xn, #1
        inst = 0x91000400 | reg_d | (reg_n << 5)
        program.extend(inst.to_bytes(4, 'little'))

    # Pattern 2: Dependent chain (RAW hazards)
    for i in range(10):
        # ADD X0, X0, #1 (each depends on previous)
        inst = 0x91000400 | 0 | (0 << 5)
        program.extend(inst.to_bytes(4, 'little'))

    # Pattern 3: Loop with counter (WAW hazard)
    # MOVZ X1, #100
    program.extend((0xD2800C81).to_bytes(4, 'little'))

    # Loop:
    # SUB X1, X1, #1
    program.extend((0xD1000421).to_bytes(4, 'little'))
    # SUBS XZR, X1, #0 (compare)
    program.extend((0xF100003F).to_bytes(4, 'little'))
    # B.NE loop
    program.extend((0x54FFFFC1).to_bytes(4, 'little'))

    # HLT
    program.extend((0xD4400000).to_bytes(4, 'little'))

    return bytes(program)


def benchmark_weights(weights: List[float], iterations: int = 5) -> float:
    """Benchmark a specific weight configuration"""
    try:
        cpu = metal.NeuralOoOCPU(memory_size=4*1024*1024, cycles_per_batch=100000)
        program = create_test_program()
        cpu.load_program(program, 0)
        cpu.load_weights(weights)

        # Run multiple times and average
        ips_values = []
        for _ in range(iterations):
            cpu.reset()
            cpu.load_program(program, 0)
            result = cpu.execute()
            if result.total_cycles > 0:
                ips_values.append(result.ips)

        if ips_values:
            return np.mean(ips_values)
        return 0.0
    except Exception as e:
        print(f"  Error: {e}")
        return 0.0


def grid_search():
    """Grid search over weight space"""
    print("=" * 60)
    print("Neural Weight Grid Search")
    print("=" * 60)
    print("Weights: [RAW, WAW, FLAG, BIAS]")
    print()

    # Define search ranges
    raw_range = [0.0, 2.0, 5.0, 10.0]
    waw_range = [0.0, 2.0, 5.0]
    flag_range = [0.0, 3.0, 5.0]
    bias_range = [-3.0, -1.0, 0.0, 1.0]

    best_ips = 0.0
    best_weights = [5.0, 3.0, 5.0, -2.0]  # Default weights

    # Get baseline
    print("Baseline (default weights)...")
    baseline_ips = benchmark_weights(best_weights, iterations=3)
    print(f"  Baseline IPS: {baseline_ips:,.0f}")
    print()

    print("Searching...")
    total_configs = len(raw_range) * len(waw_range) * len(flag_range) * len(bias_range)
    current = 0

    for raw, waw, flag, bias in itertools.product(raw_range, waw_range, flag_range, bias_range):
        current += 1
        weights = [raw, waw, flag, bias]
        ips = benchmark_weights(weights, iterations=2)

        if ips > best_ips:
            best_ips = ips
            best_weights = weights
            print(f"  [{current}/{total_configs}] New best: {weights} -> {ips:,.0f} IPS")

        if current % 20 == 0:
            print(f"  [{current}/{total_configs}] Progress... Best so far: {best_ips:,.0f} IPS")

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Best weights: {best_weights}")
    print(f"Best IPS: {best_ips:,.0f}")
    print(f"Improvement vs baseline: {(best_ips / baseline_ips - 1) * 100:.1f}%")

    # Verify result
    print()
    print("Verification run...")
    verify_ips = benchmark_weights(best_weights, iterations=5)
    print(f"Verified IPS: {verify_ips:,.0f}")

    return best_weights, best_ips


def compare_with_baselines(optimal_weights: List[float]):
    """Compare optimal neural weights with other engines"""
    print()
    print("=" * 60)
    print("Comparison with Other Engines")
    print("=" * 60)

    program = create_test_program()

    # BBCache
    try:
        bb_cpu = metal.BBCacheMetalCPU(memory_size=4*1024*1024, cycles_per_batch=100000)
        bb_cpu.load_program(program, 0)
        bb_result = bb_cpu.execute()
        print(f"BBCache:    {bb_result.total_cycles:>10,} cycles | {bb_result.ips:>12,.0f} IPS")
    except Exception as e:
        print(f"BBCache: Error - {e}")

    # Rule-based OoO
    try:
        ooo_cpu = metal.OoOMetalCPU(memory_size=4*1024*1024, cycles_per_batch=100000)
        ooo_cpu.load_program(program, 0)
        ooo_result = ooo_cpu.execute()
        print(f"OoO:        {ooo_result.total_cycles:>10,} cycles | {ooo_result.ips:>12,.0f} IPS")
    except Exception as e:
        print(f"OoO: Error - {e}")

    # Neural OoO with optimal weights
    try:
        neural_cpu = metal.NeuralOoOCPU(memory_size=4*1024*1024, cycles_per_batch=100000)
        neural_cpu.load_program(program, 0)
        neural_cpu.load_weights(optimal_weights)
        neural_result = neural_cpu.execute()
        print(f"Neural:     {neural_result.total_cycles:>10,} cycles | {neural_result.ips:>12,.0f} IPS")
    except Exception as e:
        print(f"Neural: Error - {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize neural weights via grid search")
    parser.add_argument("--quick", action="store_true", help="Quick search with fewer options")
    args = parser.parse_args()

    if args.quick:
        # Quick test with default weights
        print("Quick benchmark with default weights...")
        weights = [5.0, 3.0, 5.0, -2.0]
        ips = benchmark_weights(weights, iterations=5)
        print(f"Default weights IPS: {ips:,.0f}")
        compare_with_baselines(weights)
    else:
        optimal_weights, best_ips = grid_search()
        compare_with_baselines(optimal_weights)
