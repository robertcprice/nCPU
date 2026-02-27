#!/usr/bin/env python3
"""DOOM-like Rendering Loop Benchmark"""

import time
import sys
sys.path.insert(0, '.')
from run_neural_rtos_v2 import FullyNeuralCPU, load_elf

def run_doom(cpu, max_instructions=200000, label=""):
    """Run DOOM benchmark."""
    print(f"\033[36m[RUNNING]\033[0m {label}")
    print(f"  Instructions: {max_instructions:,}")

    start = time.time()

    for i in range(max_instructions):
        cpu.step()

        if i > 0 and i % 25000 == 0:
            elapsed = time.time() - start
            ips = cpu.inst_count / elapsed if elapsed > 0 else 0
            print(f"  [{i:6d}] PC=0x{cpu.pc:x} IPS={ips:,.0f}")

        if cpu.pc == 0:
            break

    elapsed = time.time() - start
    ips = cpu.inst_count / elapsed if elapsed > 0 else 0

    print(f"\033[32m[DONE]\033[0m Time: {elapsed:.2f}s, IPS: {ips:,.0f}")
    return elapsed, ips, cpu

def benchmark_doom():
    """Benchmark DOOM rendering loops."""

    print("="*70)
    print(" DOOM RENDERING LOOP BENCHMARK")
    print("="*70)
    print()

    # Baseline
    print("\033[33m[BASELINE]\033[0m Running WITHOUT loop optimization...")
    cpu_baseline = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)
    entry = load_elf(cpu_baseline, 'doom_benchmark.elf')
    cpu_baseline.pc = entry  # IMPORTANT: Set PC to entry point!
    print(f"   Entry point: 0x{entry:x}")
    cpu_baseline.enable_optimization(enable=False)
    time_baseline, ips_baseline, _ = run_doom(
        cpu_baseline,
        max_instructions=200000,
        label="Baseline (no optimization)"
    )

    print()
    print("\033[33m[OPTIMIZED]\033[0m Running WITH loop optimization...")
    cpu_optimized = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)
    entry = load_elf(cpu_optimized, 'doom_benchmark.elf')
    cpu_optimized.pc = entry  # IMPORTANT: Set PC to entry point!
    print(f"   Entry point: 0x{entry:x}")
    cpu_optimized.enable_optimization(enable=True)
    time_optimized, ips_optimized, cpu_optimized = run_doom(
        cpu_optimized,
        max_instructions=200000,
        label="Optimized (loop detection + skipping)"
    )

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
        print(f"Iterations saved: {stats['iterations_saved']:,}")

        if hasattr(cpu_optimized.loop_optimizer, 'pattern_discoverer'):
            cpu_optimized.loop_optimizer.pattern_discoverer.print_stats()

    print()
    print("="*70)
    print(" SPEEDUP")
    print("="*70)
    print()

    speedup = ips_optimized / ips_baseline if ips_baseline > 0 else 1.0
    print(f"Actual speedup: {speedup:.2f}x")
    print()

    if speedup > 10:
        print("\033[32m✅ MASSIVE SPEEDUP!\033[0m")
        print(f"   DOOM rendering achieved {speedup:.1f}x speedup")
    elif speedup > 3:
        print("\033[32m✅ EXCELLENT SPEEDUP!\033[0m")
        print(f"   DOOM rendering achieved {speedup:.1f}x speedup")
    elif speedup > 1.1:
        print("\033[32m✅ OPTIMIZATION WORKED!\033[0m")
        print(f"   Loop optimization achieved {speedup:.1f}x speedup")

    print()
    print("="*70)

if __name__ == "__main__":
    benchmark_doom()
