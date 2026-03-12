#!/usr/bin/env python3
"""
ARM64 Compiled C Benchmark — Compile and run C programs on Metal GPU.

Benchmarks 4 programs: fibonacci, factorial, bubble_sort, matrix_multiply.
Each compiled with aarch64-elf-gcc, run on the MLX Metal GPU kernel,
with cycle counts and IPS measured.

Usage:
    python benchmarks/benchmark_arm64.py
"""

import sys
import os
import time
import struct
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK C PROGRAMS (inline source)
# ═══════════════════════════════════════════════════════════════════════════════

PROGRAMS = {
    "fibonacci(30)": r"""
#include "arm64_syscalls.h"

int main(void) {
    long a = 0, b = 1;
    for (int i = 0; i < 30; i++) {
        long tmp = a + b;
        a = b;
        b = tmp;
    }
    /* Write result to fd=3 (custom signal) */
    sys_write(3, (const char *)&a, 8);
    return 0;
}
""",

    "factorial(20)": r"""
#include "arm64_syscalls.h"

int main(void) {
    long result = 1;
    for (int i = 2; i <= 20; i++) {
        result *= i;
    }
    sys_write(3, (const char *)&result, 8);
    return 0;
}
""",

    "bubble_sort(100)": r"""
#include "arm64_syscalls.h"

static long arr[100];

int main(void) {
    int n = 100;

    /* Initialize with descending values */
    for (int i = 0; i < n; i++) {
        arr[i] = n - i;
    }

    /* Bubble sort */
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                long tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }

    /* Write first and last element as verification */
    long results[2];
    results[0] = arr[0];
    results[1] = arr[n - 1];
    sys_write(3, (const char *)results, 16);
    return 0;
}
""",

    "matrix_multiply(8x8)": r"""
#include "arm64_syscalls.h"

#define N 8

static long A[N * N];
static long B[N * N];
static long C[N * N];

int main(void) {
    /* Initialize matrices */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = i * j + 1;
        }
    }

    /* Matrix multiply C = A * B */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            long sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    /* Write C[0][0] and C[7][7] as verification */
    long results[2];
    results[0] = C[0];
    results[1] = C[N * N - 1];
    sys_write(3, (const char *)results, 16);
    return 0;
}
""",
}

# Expected results for verification
EXPECTED = {
    "fibonacci(30)": 832040,
    "factorial(20)": 2432902008176640000,
    "bubble_sort(100)": (1, 100),  # first and last after sort
    "matrix_multiply(8x8)": None,  # just check it completes
}


def run_benchmark(name: str, source: str, iterations: int = 5) -> dict:
    """Compile and run a benchmark program multiple times."""

    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as cf:
        cf.write(source)
        c_path = cf.name

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as bf:
        bin_path = bf.name

    try:
        if not compile_c(c_path, bin_path, quiet=True):
            print(f"  FAILED to compile {name}")
            return None

        binary = Path(bin_path).read_bytes()
        results_list = []
        result_value = None

        for i in range(iterations):
            cpu = MLXKernelCPUv2()
            cpu.load_program(binary, address=0x10000)
            cpu.set_pc(0x10000)

            # Capture result written to fd=3
            captured = {"data": None}

            def on_write(fd, data):
                if fd == 3:
                    captured["data"] = data
                    return True
                return False

            handler = make_syscall_handler(on_write=on_write)
            result = run(cpu, handler, max_cycles=50_000_000, quiet=True)
            results_list.append(result)

            if captured["data"]:
                result_value = captured["data"]

        # Parse result
        parsed_result = None
        if result_value:
            if len(result_value) == 8:
                parsed_result = struct.unpack("<q", result_value)[0]
            elif len(result_value) >= 16:
                parsed_result = struct.unpack("<qq", result_value[:16])

        avg_cycles = sum(r["total_cycles"] for r in results_list) / len(results_list)
        avg_elapsed = sum(r["elapsed"] for r in results_list) / len(results_list)
        avg_ips = avg_cycles / avg_elapsed if avg_elapsed > 0 else 0

        return {
            "name": name,
            "avg_cycles": avg_cycles,
            "avg_elapsed": avg_elapsed,
            "avg_ips": avg_ips,
            "iterations": iterations,
            "binary_size": len(binary),
            "result": parsed_result,
        }

    finally:
        if os.path.exists(c_path):
            os.unlink(c_path)
        if os.path.exists(bin_path):
            os.unlink(bin_path)


def verify_result(name: str, result) -> str:
    """Check if result matches expected value."""
    expected = EXPECTED.get(name)
    if expected is None:
        return "OK (no check)"

    if isinstance(expected, tuple):
        if isinstance(result, tuple) and result == expected:
            return "PASS"
        return f"FAIL (got {result})"

    if result == expected:
        return "PASS"
    return f"FAIL (got {result}, expected {expected})"


def main():
    print("=" * 72)
    print("  ARM64 Compiled C Benchmark — Metal GPU Execution")
    print("  C → aarch64-elf-gcc → raw binary → Metal compute shader")
    print("=" * 72)
    print()

    results = []
    for name, source in PROGRAMS.items():
        print(f"Benchmarking: {name}")
        r = run_benchmark(name, source)
        if r:
            status = verify_result(name, r["result"])
            print(f"  Cycles: {r['avg_cycles']:,.0f}  "
                  f"Time: {r['avg_elapsed']*1000:.1f}ms  "
                  f"IPS: {r['avg_ips']:,.0f}  "
                  f"Binary: {r['binary_size']:,}B  "
                  f"Result: {status}")
            results.append(r)
        print()

    # Summary table
    print("=" * 72)
    print(f"{'Program':<25} {'Cycles':>10} {'Time (ms)':>10} {'IPS':>12} {'Binary':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:<25} {r['avg_cycles']:>10,.0f} "
              f"{r['avg_elapsed']*1000:>10.1f} "
              f"{r['avg_ips']:>12,.0f} "
              f"{r['binary_size']:>7,}B")
    print("=" * 72)

    if results:
        total_cycles = sum(r["avg_cycles"] for r in results)
        total_time = sum(r["avg_elapsed"] for r in results)
        overall_ips = total_cycles / total_time if total_time > 0 else 0
        print(f"{'Overall':.<25} {total_cycles:>10,.0f} "
              f"{total_time*1000:>10.1f} "
              f"{overall_ips:>12,.0f}")


if __name__ == "__main__":
    main()
