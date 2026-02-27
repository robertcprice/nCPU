#!/usr/bin/env python3
"""
BENCHMARK: ACTUAL LOOP OPTIMIZATION SPEEDUP
============================================

Honest before/after benchmark measuring REAL speedup from loop optimization.
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_neural_rtos_v2 import FullyNeuralCPU, load_elf


def run_rtos(cpu, num_instructions=50000, label=""):
    """Run RTOS and measure execution time."""

    print(f"\033[36m[RUNNING]\033[0m {label}")
    print(f"  Instructions: {num_instructions:,}")

    start = time.time()
    for i in range(num_instructions):
        cpu.step()

        if i > 0 and i % 10000 == 0:
            elapsed = time.time() - start
            ips = cpu.inst_count / elapsed if elapsed > 0 else 0
            print(f"  [{i:6d}] PC=0x{cpu.pc:x} IPS={ips:,.0f}")

    elapsed = time.time() - start
    ips = cpu.inst_count / elapsed if elapsed > 0 else 0

    print(f"\033[32m[DONE]\033[0m Time: {elapsed:.2f}s, IPS: {ips:,.0f}")

    return elapsed, ips, cpu


def benchmark_optimization():
    """
    Compare: Baseline vs Optimized execution.

    This is an HONEST benchmark - no theoretical estimates!
    """

    print("="*70)
    print(" ACTUAL LOOP OPTIMIZATION BENCHMARK")
    print("="*70)
    print()
    print("Measuring REAL speedup from loop optimization")
    print("No theoretical estimates - actual execution time!")
    print()
    print("="*70)
    print()

    # =====================================================================
    # BASELINE: Run without optimization
    # =====================================================================
    print("\033[33m[BASELINE]\033[0m Running WITHOUT loop optimization...")

    cpu_baseline = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)
    entry = load_elf(cpu_baseline, 'arm64_doom/neural_rtos.elf')
    cpu_baseline.pc = entry
    cpu_baseline.predecode_code_segment(0x10000, 0x2000)

    time_baseline, ips_baseline, _ = run_rtos(
        cpu_baseline,
        num_instructions=50000,
        label="Baseline (no optimization)"
    )

    # =====================================================================
    # OPTIMIZED: Run WITH loop optimization
    # =====================================================================
    print()
    print("\033[33m[OPTIMIZED]\033[0m Running WITH loop optimization...")

    cpu_optimized = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)
    entry = load_elf(cpu_optimized, 'arm64_doom/neural_rtos.elf')
    cpu_optimized.pc = entry
    cpu_optimized.predecode_code_segment(0x10000, 0x2000)

    # ENABLE loop optimization
    cpu_optimized.enable_optimization(enable=True)

    time_optimized, ips_optimized, _ = run_rtos(
        cpu_optimized,
        num_instructions=50000,
        label="Optimized (loop detection + skipping)"
    )

    # =====================================================================
    # RESULTS
    # =====================================================================
    print()
    print("="*70)
    print(" BENCHMARK RESULTS")
    print("="*70)
    print()
    print(f"Baseline:")
    print(f"  Time: {time_baseline:.2f}s")
    print(f"  IPS: {ips_baseline:,.0f}")
    print()
    print(f"Optimized:")
    print(f"  Time: {time_optimized:.2f}s")
    print(f"  IPS: {ips_optimized:,.0f}")
    print()

    if cpu_optimized.loop_optimizer:
        stats = cpu_optimized.loop_optimizer.get_stats()
        print(f"Loops detected: {stats['loops_detected']}")
        print(f"Loops optimized: {stats['loops_optimized']}")
        print(f"Loops rejected: {stats['loops_rejected']}")
        print(f"False positives: {stats['false_positives']}")
        print(f"Iterations saved: {stats['iterations_saved']:,}")

        # Print pattern discovery statistics
        if hasattr(cpu_optimized.loop_optimizer, 'pattern_discoverer'):
            cpu_optimized.loop_optimizer.pattern_discoverer.print_stats()
            # Save pattern library to disk
            cpu_optimized.loop_optimizer.pattern_discoverer.save_pattern_library()

    print()
    print("="*70)
    print(" SPEEDUP")
    print("="*70)
    print()

    # Calculate actual speedup
    speedup = ips_optimized / ips_baseline if ips_baseline > 0 else 1.0

    print(f"Actual speedup: {speedup:.2f}x")
    print()

    if speedup > 1.1:
        print("\033[32m✅ OPTIMIZATION WORKED!\033[0m")
        print(f"   Loop optimization achieved {speedup:.1f}x speedup")
    elif speedup < 0.9:
        print("\033[33m⚠️  OPTIMIZATION OVERHEAD\033[0m")
        print("   Loop detection added overhead without benefit")
    else:
        print("\033[33m⚠️  NO SIGNIFICANT CHANGE\033[0m")
        print("   Loop optimization didn't affect performance")

    print()
    print("="*70)
    print(" This is REAL speedup from ACTUAL optimization!")
    print("="*70)


if __name__ == "__main__":
    benchmark_optimization()
