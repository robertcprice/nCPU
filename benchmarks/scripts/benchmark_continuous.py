#!/usr/bin/env python3
"""Comprehensive benchmark: Continuous vs Standard GPU execution."""

import time
import sys
from kvrm_metal import ContinuousMetalCPU, MetalCPU

print("=" * 70)
print("CONTINUOUS VS STANDARD GPU EXECUTION BENCHMARK")
print("=" * 70)

def encode_movz(rd, imm16, hw=0):
    return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

def encode_add_imm(rd, rn, imm12):
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd

def encode_str_64(rt, rn, imm12=0):
    return 0xF9000000 | (imm12 << 10) | (rn << 5) | rt

def encode_b(offset_words):
    imm26 = offset_words & 0x3FFFFFF
    return 0x14000000 | imm26

def benchmark_continuous(cycles_per_batch, total_cycles, program_bytes, label):
    """Benchmark continuous execution."""
    num_batches = total_cycles // cycles_per_batch
    if num_batches < 1:
        num_batches = 1

    cpu = ContinuousMetalCPU(cycles_per_batch=cycles_per_batch)
    cpu.load_program(list(program_bytes), 0)
    cpu.set_pc(0)

    start = time.perf_counter()
    result = cpu.execute_continuous(max_batches=num_batches, timeout_seconds=120.0)
    elapsed = time.perf_counter() - start

    return result.total_cycles, elapsed, result.ips

def benchmark_standard(total_cycles, program_bytes, label):
    """Benchmark standard execution."""
    cpu = MetalCPU()
    cpu.load_program(list(program_bytes), 0)
    cpu.set_pc(0)

    start = time.perf_counter()
    result = cpu.execute(total_cycles)
    elapsed = time.perf_counter() - start

    ips = result.cycles / elapsed if elapsed > 0 else 0
    return result.cycles, elapsed, ips

# Test programs
programs = {
    "tight_loop": [
        encode_movz(0, 0),           # x0 = 0
        encode_add_imm(0, 0, 1),     # x0++
        encode_b(-1 & 0x3FFFFFF),    # loop
    ],
    "loop_with_str": [
        encode_movz(0, 0),           # Counter
        encode_movz(1, 0x8000),      # Address
        encode_add_imm(0, 0, 1),     # counter++
        encode_str_64(0, 1, 0),      # STR to memory
        encode_b(-2 & 0x3FFFFFF),    # Loop back
    ],
}

cycle_counts = [10_000_000, 50_000_000, 100_000_000]
batch_sizes = [1_000_000, 5_000_000, 10_000_000]

results = []

print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

for prog_name, program in programs.items():
    program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program)
    print(f"\n### Program: {prog_name} ###\n")

    for total in cycle_counts:
        print(f"\n--- {total:,} cycles ---")

        # Standard benchmark
        std_cycles, std_elapsed, std_ips = benchmark_standard(total, program_bytes, "standard")
        print(f"Standard:    {std_cycles:>12,} cycles in {std_elapsed:.3f}s = {std_ips:>12,.0f} IPS")

        # Continuous with different batch sizes
        for batch in batch_sizes:
            cont_cycles, cont_elapsed, cont_ips = benchmark_continuous(batch, total, program_bytes, f"batch_{batch}")
            improvement = ((cont_ips - std_ips) / std_ips) * 100 if std_ips > 0 else 0
            print(f"Continuous (batch={batch:>9,}): {cont_cycles:>12,} cycles in {cont_elapsed:.3f}s = {cont_ips:>12,.0f} IPS ({improvement:+.1f}%)")

            results.append({
                "program": prog_name,
                "total_cycles": total,
                "mode": f"continuous_batch_{batch}",
                "cycles": cont_cycles,
                "elapsed": cont_elapsed,
                "ips": cont_ips,
                "improvement": improvement,
            })

        results.append({
            "program": prog_name,
            "total_cycles": total,
            "mode": "standard",
            "cycles": std_cycles,
            "elapsed": std_elapsed,
            "ips": std_ips,
            "improvement": 0,
        })

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

cont_results = [r for r in results if r["mode"].startswith("continuous")]
if cont_results:
    avg_improvement = sum(r["improvement"] for r in cont_results) / len(cont_results)
    max_improvement = max(r["improvement"] for r in cont_results)
    max_ips = max(r["ips"] for r in cont_results)

    print(f"\nContinuous Execution Results:")
    print(f"  Average improvement over standard: {avg_improvement:+.1f}%")
    print(f"  Maximum improvement: {max_improvement:+.1f}%")
    print(f"  Peak IPS achieved: {max_ips:,.0f}")

std_results = [r for r in results if r["mode"] == "standard"]
if std_results:
    std_max_ips = max(r["ips"] for r in std_results)
    print(f"\nStandard Execution Results:")
    print(f"  Peak IPS: {std_max_ips:,.0f}")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
