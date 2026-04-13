#!/usr/bin/env python3
"""Benchmark suite for the nCPU neural ALU and program execution.

Measures per-operation latency of every neural ALU operation through the
NeuralALUBridge, then runs each .asm program in programs/ through the
full neural execution pipeline.

Usage:
    python3 benchmarks/benchmark_neural.py

Output:
    - Human-readable tables to stdout
    - JSON results to benchmarks/results.json
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve project root (one level above benchmarks/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROGRAMS_DIR = PROJECT_ROOT / "programs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "local" / "results.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ITERATIONS = 1000
WARMUP_ITERATIONS = 50


# ===========================================================================
# Utility helpers
# ===========================================================================

def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list (0-100 scale)."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def _format_us(value: float) -> str:
    """Format a microsecond value to a fixed-width string."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:>10.2f}s "
    if value >= 1_000:
        return f"{value / 1_000:>10.2f}ms"
    return f"{value:>10.1f}us"


def _collect_system_info() -> dict[str, Any]:
    """Gather system and environment metadata."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch_version": None,
        "torch_device": "cpu",
        "gpu_name": None,
        "gpu_memory_mb": None,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["torch_device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_mem
            info["gpu_memory_mb"] = round(total / (1024 * 1024))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["torch_device"] = "mps"
            info["gpu_name"] = "Apple Silicon (MPS)"
    except ImportError:
        pass

    return info


# ===========================================================================
# 1. Per-operation micro-benchmarks
# ===========================================================================

def _generate_operands(rng: random.Random) -> tuple[int, int]:
    """Generate a pair of random 32-bit signed ints."""
    a = rng.randint(-(2**30), 2**30)
    b = rng.randint(-(2**30), 2**30)
    return a, b


def _generate_shift_operands(rng: random.Random) -> tuple[int, int]:
    """Generate a value and a small shift amount (0-31)."""
    value = rng.randint(0, 2**31 - 1)
    amount = rng.randint(0, 31)
    return value, amount


def _generate_fixed_point(rng: random.Random, lo: float, hi: float) -> int:
    """Generate a fixed-point int (value / 1000 in [lo, hi])."""
    return int(rng.uniform(lo, hi) * 1000)


def _benchmark_op(
    fn,
    operand_gen,
    n_iter: int = N_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> dict[str, float]:
    """Time a callable over random operands. Returns stats in microseconds."""
    rng = random.Random(42)

    # Pre-generate all operand sets to avoid measuring RNG cost
    operands = [operand_gen(rng) for _ in range(warmup + n_iter)]

    # Warmup (primes caches, JIT, etc.)
    for i in range(warmup):
        args = operands[i]
        if isinstance(args, tuple):
            fn(*args)
        else:
            fn(args)

    # Timed iterations
    timings_us: list[float] = []
    for i in range(warmup, warmup + n_iter):
        args = operands[i]
        t0 = time.perf_counter_ns()
        if isinstance(args, tuple):
            fn(*args)
        else:
            fn(args)
        elapsed_ns = time.perf_counter_ns() - t0
        timings_us.append(elapsed_ns / 1_000.0)

    timings_us.sort()
    return {
        "mean_us": statistics.mean(timings_us),
        "median_us": statistics.median(timings_us),
        "p99_us": _percentile(timings_us, 99),
        "min_us": timings_us[0],
        "max_us": timings_us[-1],
        "stdev_us": statistics.stdev(timings_us) if len(timings_us) > 1 else 0.0,
    }


def run_operation_benchmarks(bridge) -> dict[str, dict[str, float]]:
    """Benchmark every NeuralALUBridge operation."""
    results: dict[str, dict[str, float]] = {}

    # -- Arithmetic ----------------------------------------------------------
    arith_ops = {
        "add": bridge.add,
        "sub": bridge.sub,
        "mul": bridge.mul,
    }
    for name, fn in arith_ops.items():
        results[name] = _benchmark_op(fn, _generate_operands)

    # -- Bitwise -------------------------------------------------------------
    bitwise_ops = {
        "and_": bridge.and_,
        "or_": bridge.or_,
        "xor_": bridge.xor_,
    }
    for name, fn in bitwise_ops.items():
        results[name] = _benchmark_op(fn, _generate_operands)

    # -- Shifts --------------------------------------------------------------
    shift_ops = {
        "shl": bridge.shl,
        "shr": bridge.shr,
    }
    for name, fn in shift_ops.items():
        results[name] = _benchmark_op(fn, _generate_shift_operands)

    # -- Comparison ----------------------------------------------------------
    results["cmp"] = _benchmark_op(bridge.cmp, _generate_operands)

    # -- Math (fixed-point) --------------------------------------------------
    math_unary = {
        "sin": (bridge.sin, lambda rng: (_generate_fixed_point(rng, -math.pi, math.pi),)),
        "cos": (bridge.cos, lambda rng: (_generate_fixed_point(rng, -math.pi, math.pi),)),
        "sqrt": (bridge.sqrt, lambda rng: (_generate_fixed_point(rng, 0.0, 100.0),)),
        "exp_": (bridge.exp_, lambda rng: (_generate_fixed_point(rng, -5.0, 5.0),)),
        "log_": (bridge.log_, lambda rng: (_generate_fixed_point(rng, 0.01, 100.0),)),
    }
    for name, (fn, gen) in math_unary.items():
        results[name] = _benchmark_op(fn, gen)

    # atan2 is binary
    def _atan2_gen(rng: random.Random) -> tuple[int, int]:
        y = _generate_fixed_point(rng, -10.0, 10.0)
        x = _generate_fixed_point(rng, -10.0, 10.0)
        # Avoid both zero
        if y == 0 and x == 0:
            x = 1000
        return y, x

    results["atan2"] = _benchmark_op(bridge.atan2, _atan2_gen)

    return results


# ===========================================================================
# 2. Program execution benchmarks
# ===========================================================================

def run_program_benchmarks() -> list[dict[str, Any]]:
    """Run each .asm program through the neural CPU, measuring wall time and cycles."""
    from ncpu.model import CPU

    program_files = sorted(PROGRAMS_DIR.rglob("*.asm"))
    if not program_files:
        print(f"  WARNING: No .asm files found in {PROGRAMS_DIR}")
        return []

    results: list[dict[str, Any]] = []

    for asm_path in program_files:
        source = asm_path.read_text()
        entry: dict[str, Any] = {
            "program": asm_path.name,
            "instructions": 0,
            "cycles": 0,
            "wall_time_ms": 0.0,
            "status": "ok",
            "error": None,
        }

        try:
            cpu = CPU(neural_execution=True, models_dir=str(MODELS_DIR), max_cycles=10_000)
            cpu.load_program(source)

            # Count non-empty, non-comment source lines for instruction count
            instr_lines = [
                line
                for line in source.splitlines()
                if line.strip() and not line.strip().startswith(";") and not line.strip().endswith(":")
            ]
            entry["instructions"] = len(instr_lines)

            t0 = time.perf_counter()
            cpu.run()
            elapsed = time.perf_counter() - t0

            entry["wall_time_ms"] = round(elapsed * 1000, 3)
            entry["cycles"] = cpu.get_cycle_count()

            # Capture final register state for verification
            entry["final_registers"] = cpu.dump_registers()

        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)

        results.append(entry)

    return results


# ===========================================================================
# 3. Report formatting
# ===========================================================================

HEADER_LINE = "=" * 80
SECTION_LINE = "-" * 80


def print_system_info(info: dict[str, Any]) -> None:
    """Print system information block."""
    print(HEADER_LINE)
    print("  nCPU Neural ALU Benchmark")
    print(HEADER_LINE)
    print()
    print("  System Information")
    print(SECTION_LINE)
    print(f"    Python:     {info['python_version']}")
    print(f"    Platform:   {info['platform']}")
    print(f"    Processor:  {info['processor']}")
    print(f"    Torch:      {info['torch_version'] or 'NOT INSTALLED'}")
    print(f"    Device:     {info['torch_device']}")
    if info["gpu_name"]:
        print(f"    GPU:        {info['gpu_name']}")
    if info["gpu_memory_mb"]:
        print(f"    GPU Memory: {info['gpu_memory_mb']} MB")
    print()


def print_model_availability(available: dict[str, bool]) -> None:
    """Print which neural models loaded successfully."""
    print("  Neural Model Availability")
    print(SECTION_LINE)
    for op, loaded in sorted(available.items()):
        status = "loaded" if loaded else "MISSING"
        marker = "  [+]" if loaded else "  [-]"
        print(f"    {marker} {op:<8} {status}")
    print()


def print_operation_results(op_results: dict[str, dict[str, float]]) -> None:
    """Print per-operation timing table."""
    print("  Per-Operation Latency ({} iterations each)".format(N_ITERATIONS))
    print(SECTION_LINE)
    print(f"    {'Operation':<10} {'Mean':>10}  {'Median':>10}  {'P99':>10}  {'Min':>10}  {'Max':>10}  {'StdDev':>10}")
    print(f"    {'-'*10:<10} {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}")

    categories = [
        ("Arithmetic", ["add", "sub", "mul"]),
        ("Bitwise", ["and_", "or_", "xor_"]),
        ("Shifts", ["shl", "shr"]),
        ("Compare", ["cmp"]),
        ("Math", ["sin", "cos", "sqrt", "exp_", "log_", "atan2"]),
    ]

    for cat_name, ops in categories:
        first = True
        for op_name in ops:
            if op_name not in op_results:
                continue
            stats = op_results[op_name]
            prefix = f"  {cat_name}" if first else "          "
            first = False
            print(
                f"    {prefix:<10}"
                f" {op_name:<8}"
                f" {_format_us(stats['mean_us'])} "
                f" {_format_us(stats['median_us'])} "
                f" {_format_us(stats['p99_us'])} "
                f" {_format_us(stats['min_us'])} "
                f" {_format_us(stats['max_us'])} "
                f" {_format_us(stats['stdev_us'])}"
            )
        if not first:
            print()


def print_program_results(prog_results: list[dict[str, Any]]) -> None:
    """Print program execution timing table."""
    print("  Program Execution (neural_execution=True)")
    print(SECTION_LINE)
    if not prog_results:
        print("    No programs found.")
        print()
        return

    print(f"    {'Program':<30} {'Status':<8} {'Cycles':>8} {'Wall Time':>12} {'us/cycle':>10}")
    print(f"    {'-'*30:<30} {'-'*8:<8} {'-'*8:>8} {'-'*12:>12} {'-'*10:>10}")

    for entry in prog_results:
        name = entry["program"]
        status = entry["status"]
        cycles = entry["cycles"]
        wall_ms = entry["wall_time_ms"]

        if status == "ok" and cycles > 0:
            us_per_cycle = (wall_ms * 1000) / cycles
            print(
                f"    {name:<30} {'OK':<8} {cycles:>8} {wall_ms:>10.3f}ms {us_per_cycle:>9.1f}"
            )
        elif status == "ok":
            print(f"    {name:<30} {'OK':<8} {cycles:>8} {wall_ms:>10.3f}ms {'N/A':>10}")
        else:
            err_short = (entry["error"] or "unknown")[:40]
            print(f"    {name:<30} {'FAIL':<8} {'':>8} {'':>12} {err_short}")

    print()


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    """Run the full benchmark suite and save results."""

    # -- System info ---------------------------------------------------------
    sys_info = _collect_system_info()
    print_system_info(sys_info)

    # -- Load neural models --------------------------------------------------
    print("  Loading neural models...")
    t0 = time.perf_counter()

    from ncpu.neural.neural_alu_bridge import NeuralALUBridge

    bridge = NeuralALUBridge(models_dir=str(MODELS_DIR))
    available = bridge.load()
    load_time_ms = (time.perf_counter() - t0) * 1000

    print(f"  Models loaded in {load_time_ms:.1f}ms")
    print()

    print_model_availability(available)

    loaded_count = sum(1 for v in available.values() if v)
    if loaded_count == 0:
        print("  ERROR: No neural models loaded. Cannot benchmark.")
        print(f"  Looked in: {MODELS_DIR}")
        return 1

    # -- Per-operation benchmarks --------------------------------------------
    print("  Running per-operation benchmarks ({} iterations, {} warmup)...".format(
        N_ITERATIONS, WARMUP_ITERATIONS
    ))
    print()
    op_results = run_operation_benchmarks(bridge)
    print_operation_results(op_results)

    # -- Program execution benchmarks ----------------------------------------
    print("  Running program execution benchmarks...")
    print()
    prog_results = run_program_benchmarks()
    print_program_results(prog_results)

    # -- Summary -------------------------------------------------------------
    print(HEADER_LINE)

    # Compute aggregate stats
    total_ops = len(op_results)
    if total_ops > 0:
        avg_mean = statistics.mean(s["mean_us"] for s in op_results.values())
        fastest_op = min(op_results, key=lambda k: op_results[k]["mean_us"])
        slowest_op = max(op_results, key=lambda k: op_results[k]["mean_us"])
        print(f"  Summary: {total_ops} operations benchmarked")
        print(f"    Average mean latency:  {_format_us(avg_mean)}")
        print(f"    Fastest operation:     {fastest_op} ({_format_us(op_results[fastest_op]['mean_us'])})")
        print(f"    Slowest operation:     {slowest_op} ({_format_us(op_results[slowest_op]['mean_us'])})")

    if prog_results:
        ok_count = sum(1 for p in prog_results if p["status"] == "ok")
        fail_count = len(prog_results) - ok_count
        print(f"    Programs executed:     {ok_count} OK, {fail_count} failed")

    print(HEADER_LINE)
    print()

    # -- Save JSON results ---------------------------------------------------
    full_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "n_iterations": N_ITERATIONS,
            "warmup_iterations": WARMUP_ITERATIONS,
            "models_dir": str(MODELS_DIR),
            "programs_dir": str(PROGRAMS_DIR),
        },
        "system": sys_info,
        "model_availability": available,
        "model_load_time_ms": round(load_time_ms, 1),
        "operation_benchmarks": op_results,
        "program_benchmarks": prog_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"  Results saved to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
