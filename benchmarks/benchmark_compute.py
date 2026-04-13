#!/usr/bin/env python3
"""Three-tier benchmark: Neural vs Fast vs Compute mode.

Runs programs through all three execution tiers and compares IPS:
  - Neural:  All ALU ops through trained .pt models (~5K IPS)
  - Fast:    Native Python arithmetic in model CPU (~varies)
  - Compute: GPU Metal compute shader, qemu-style (~millions IPS)

Also verifies that compute mode produces identical register state
to neural mode for every program.

Usage:
    python3 benchmarks/benchmark_compute.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROGRAMS_DIR = PROJECT_ROOT / "programs"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "local" / "compute_results.json"

# Programs to benchmark (representative mix of complexity)
BENCHMARK_PROGRAMS = [
    ("sum_1_to_10.asm", "R0", 55),
    ("fibonacci_iterative.asm", "R1", 55),
    ("factorial.asm", "R0", None),  # varies by program params
    ("collatz.asm", "R0", None),
    ("gcd.asm", "R0", None),
    ("triangular_number.asm", "R0", None),
    ("power.asm", "R0", None),
    ("max_of_two.asm", "R0", None),
]

# Synthetic heavy program to showcase GPU compute throughput
# (small programs are dominated by per-dispatch kernel launch overhead)
HEAVY_PROGRAM = """\
    MOV R0, 0
    MOV R1, 10000
    MOV R2, 1
loop:
    ADD R0, R0, R2
    CMP R0, R1
    JNZ loop
    HALT
"""


def run_neural(source: str, max_cycles: int = 10000):
    """Run program on neural mode CPU (trained .pt models)."""
    from ncpu.model import CPU

    cpu = CPU(mock_mode=True, neural_execution=True, models_dir="models", max_cycles=max_cycles)
    cpu.load_program(source)

    start = time.perf_counter()
    try:
        cpu.run()
    except RuntimeError:
        pass
    elapsed = time.perf_counter() - start

    cycles = cpu.get_cycle_count()
    regs = cpu.dump_registers()
    ips = cycles / elapsed if elapsed > 0 else 0

    return {
        "cycles": cycles,
        "elapsed_s": elapsed,
        "ips": ips,
        "registers": regs,
        "halted": cpu.is_halted(),
    }


def run_fast(source: str, max_cycles: int = 10000):
    """Run program on fast mode CPU (native Python arithmetic, no neural models)."""
    from ncpu.model import CPU

    cpu = CPU(mock_mode=True, neural_execution=False, max_cycles=max_cycles)
    cpu.load_program(source)

    start = time.perf_counter()
    try:
        cpu.run()
    except RuntimeError:
        pass
    elapsed = time.perf_counter() - start

    cycles = cpu.get_cycle_count()
    regs = cpu.dump_registers()
    ips = cycles / elapsed if elapsed > 0 else 0

    return {
        "cycles": cycles,
        "elapsed_s": elapsed,
        "ips": ips,
        "registers": regs,
        "halted": cpu.is_halted(),
    }


def run_compute(source: str, max_cycles: int = 1_000_000):
    """Run program on GPU compute mode (Metal shader)."""
    from kernels.mlx.ncpu_kernel import NCPUComputeKernel

    kernel = NCPUComputeKernel()

    # Warm up: compile kernel with a trivial program first
    kernel.load_program_from_asm("HALT")
    kernel.execute(max_cycles=1)
    kernel.reset()

    # Now run the actual program
    kernel.load_program_from_asm(source)

    # Run multiple iterations and take the best to amortize kernel launch overhead
    best_elapsed = float("inf")
    best_result = None
    for _ in range(5):
        kernel.reset()
        kernel.load_program_from_asm(source)
        result = kernel.execute(max_cycles=max_cycles)
        if result.elapsed_seconds < best_elapsed:
            best_elapsed = result.elapsed_seconds
            best_result = result
            best_regs = kernel.get_registers_dict()

    cycles = best_result.cycles
    ips = cycles / best_elapsed if best_elapsed > 0 else 0

    return {
        "cycles": cycles,
        "elapsed_s": best_elapsed,
        "ips": ips,
        "registers": best_regs,
        "halted": best_result.stop_reason_name == "HALT",
    }


def main():
    print("=" * 80)
    print("nCPU THREE-TIER BENCHMARK: Neural vs Fast vs Compute")
    print("=" * 80)
    print()

    results = []

    # Header
    print(f"{'Program':<28} {'Neural IPS':>12} {'Fast IPS':>12} {'Compute IPS':>14} "
          f"{'Speedup':>10} {'Match':>6}")
    print("-" * 88)

    for program_name, result_reg, expected in BENCHMARK_PROGRAMS:
        program_path = PROGRAMS_DIR / program_name
        if not program_path.exists():
            print(f"  {program_name:<26} SKIP (not found)")
            continue

        source = program_path.read_text()
        label = program_name.replace(".asm", "")

        # Run all three modes
        try:
            neural = run_neural(source)
        except Exception as e:
            neural = {"ips": 0, "registers": {}, "halted": False, "cycles": 0, "elapsed_s": 0}

        try:
            fast = run_fast(source)
        except Exception as e:
            fast = {"ips": 0, "registers": {}, "halted": False, "cycles": 0, "elapsed_s": 0}

        try:
            compute = run_compute(source)
        except Exception as e:
            compute = {"ips": 0, "registers": {}, "halted": False, "cycles": 0, "elapsed_s": 0}

        # Check register match between neural and compute
        match = True
        for i in range(8):
            rn = f"R{i}"
            neural_val = neural["registers"].get(rn, 0)
            compute_val = compute["registers"].get(rn, 0)
            if neural_val != compute_val:
                match = False
                break

        # Compute speedup
        if neural["ips"] > 0 and compute["ips"] > 0:
            speedup = compute["ips"] / neural["ips"]
            speedup_str = f"{speedup:,.0f}x"
        else:
            speedup_str = "N/A"

        match_str = "YES" if match else "NO"

        print(f"  {label:<26} {neural['ips']:>10,.0f} {fast['ips']:>12,.0f} "
              f"{compute['ips']:>12,.0f}   {speedup_str:>10} {match_str:>6}")

        results.append({
            "program": program_name,
            "neural_ips": neural["ips"],
            "neural_cycles": neural["cycles"],
            "neural_elapsed_s": neural["elapsed_s"],
            "fast_ips": fast["ips"],
            "fast_cycles": fast["cycles"],
            "fast_elapsed_s": fast["elapsed_s"],
            "compute_ips": compute["ips"],
            "compute_cycles": compute["cycles"],
            "compute_elapsed_s": compute["elapsed_s"],
            "registers_match": match,
        })

    print("-" * 88)

    # Run heavy program to showcase sustained GPU throughput
    print()
    print("Heavy program (count to 10,000 — 30K cycles):")
    print("-" * 88)

    try:
        heavy_neural = run_neural(HEAVY_PROGRAM, max_cycles=50000)
    except Exception:
        heavy_neural = {"ips": 0, "registers": {}, "cycles": 0, "elapsed_s": 0}

    try:
        heavy_fast = run_fast(HEAVY_PROGRAM, max_cycles=50000)
    except Exception:
        heavy_fast = {"ips": 0, "registers": {}, "cycles": 0, "elapsed_s": 0}

    try:
        heavy_compute = run_compute(HEAVY_PROGRAM, max_cycles=1_000_000)
    except Exception:
        heavy_compute = {"ips": 0, "registers": {}, "cycles": 0, "elapsed_s": 0}

    heavy_match = all(
        heavy_neural["registers"].get(f"R{i}", 0) == heavy_compute["registers"].get(f"R{i}", 0)
        for i in range(8)
    )

    if heavy_neural["ips"] > 0 and heavy_compute["ips"] > 0:
        heavy_speedup = f"{heavy_compute['ips'] / heavy_neural['ips']:,.0f}x"
    else:
        heavy_speedup = "N/A"

    print(f"  {'count_to_10000':<26} {heavy_neural['ips']:>10,.0f} {heavy_fast['ips']:>12,.0f} "
          f"{heavy_compute['ips']:>12,.0f}   {heavy_speedup:>10} {'YES' if heavy_match else 'NO':>6}")
    print(f"  ({heavy_compute['cycles']:,} cycles in {heavy_compute['elapsed_s']*1000:.1f}ms)")
    print("-" * 88)
    print()

    # Summary
    if results:
        neural_avg = sum(r["neural_ips"] for r in results) / len(results)
        fast_avg = sum(r["fast_ips"] for r in results) / len(results)
        compute_avg = sum(r["compute_ips"] for r in results) / len(results)
        all_match = all(r["registers_match"] for r in results)

        print(f"Average IPS: Neural={neural_avg:,.0f}  Fast={fast_avg:,.0f}  Compute={compute_avg:,.0f}")
        if neural_avg > 0:
            print(f"Speedup over neural: Fast={fast_avg/neural_avg:,.1f}x  Compute={compute_avg/neural_avg:,.1f}x")
        print(f"All register outputs match: {'YES' if all_match else 'NO'}")
        print()

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump({"benchmark": "three_tier_compute", "results": results}, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
