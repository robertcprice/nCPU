#!/usr/bin/env python3
"""
Self-Hosting C Compiler Benchmark — Measure compilation and execution performance
on the Metal GPU.

The benchmark compiles cc.c (the self-hosting C compiler) with the host GCC
cross-compiler, then uses it on the GPU to compile a suite of C test programs.
Each GPU-compiled binary is then executed on the GPU. We measure:

  - Compilation cycles and wall time (compiler running on GPU)
  - Execution cycles and wall time (compiled binary running on GPU)
  - Binary size produced by the GPU compiler
  - Correctness (exit code matches expected value)

Usage:
    python benchmarks/benchmark_selfhost.py
"""

import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2

RESULTS_PATH = Path(__file__).resolve().parent / "selfhost_results.json"
CC_SOURCE = PROJECT_ROOT / "ncpu" / "os" / "gpu" / "programs" / "tools" / "cc.c"

MAX_COMPILE_CYCLES = 200_000_000
MAX_EXEC_CYCLES = 10_000_000


# =============================================================================
# TEST PROGRAMS
# =============================================================================

TEST_PROGRAMS = {
    "arithmetic": {
        "source": """\
int main(void) {
    int a = 42;
    int b = 13;
    int sum = a + b;
    return sum;
}
""",
        "expected": 55,
    },

    "fibonacci": {
        "source": """\
int fib(int n) {
    if (n <= 1) return n;
    int a = 0;
    int b = 1;
    int i = 2;
    while (i <= n) {
        int tmp = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}

int main(void) {
    return fib(10);
}
""",
        "expected": 55,
    },

    "factorial": {
        "source": """\
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main(void) {
    return factorial(5);
}
""",
        "expected": 120,
    },

    "array": {
        "source": """\
int main(void) {
    int arr[5];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;
    int sum = 0;
    int i = 0;
    while (i < 5) {
        sum = sum + arr[i];
        i = i + 1;
    }
    return sum;
}
""",
        "expected": 150,
    },

    "forloop": {
        "source": """\
int main(void) {
    int sum = 0;
    for (int i = 1; i <= 10; i = i + 1) {
        sum = sum + i;
    }
    return sum;
}
""",
        "expected": 55,
    },

    "pointers": {
        "source": """\
int main(void) {
    int x = 100;
    int *p = &x;
    *p = *p + 23;
    return x;
}
""",
        "expected": 123,
    },

    "nested_calls": {
        "source": """\
int add(int a, int b) {
    return a + b;
}

int mul(int a, int b) {
    return a * b;
}

int main(void) {
    return add(3, mul(4, 5));
}
""",
        "expected": 23,
    },

    "control_flow": {
        "source": """\
int main(void) {
    int x = 0;
    int i = 0;
    while (i < 20) {
        if (i > 10) {
            break;
        }
        x = x + i;
        i = i + 1;
    }
    return x;
}
""",
        "expected": 55,
    },
}


# =============================================================================
# SYSTEM INFO
# =============================================================================

def collect_system_info() -> dict:
    """Collect system information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    try:
        import mlx.core as mx
        info["mlx_version"] = mx.__version__
    except (ImportError, AttributeError):
        info["mlx_version"] = "unknown"
    try:
        import torch
        info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "N/A"
    return info


# =============================================================================
# COMPILER BUILD
# =============================================================================

def build_compiler() -> bytes | None:
    """Compile cc.c with host GCC and return the raw binary bytes."""
    if not CC_SOURCE.exists():
        print(f"  ERROR: cc.c not found at {CC_SOURCE}")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    tmp.close()
    bin_path = tmp.name

    try:
        ok = compile_c(str(CC_SOURCE), bin_path, quiet=True)
        if not ok:
            print("  ERROR: Failed to compile cc.c with host GCC")
            return None
        binary = Path(bin_path).read_bytes()
        return binary
    finally:
        if os.path.exists(bin_path):
            os.unlink(bin_path)


# =============================================================================
# BENCHMARK CORE
# =============================================================================

def benchmark_program(
    name: str,
    source: str,
    expected: int,
    compiler_binary: bytes,
) -> dict | None:
    """
    Compile a test program on the GPU using cc.c, then execute the result.

    Returns a dict with compile/exec metrics, or None on total failure.
    """
    src_path = f"/tmp/{name}.c"
    out_path = f"/bin/{name}"

    # -- Phase A: Compile on GPU --
    fs = GPUFilesystem()
    fs.mkdir("/tmp")
    fs.mkdir("/bin")
    fs.write_file(src_path, source.encode())
    fs.write_file("/tmp/.cc_args", f"{src_path}\n{out_path}\n".encode())

    cpu = MLXKernelCPUv2()
    cpu.load_program(compiler_binary, address=0x10000)
    cpu.set_pc(0x10000)

    handler = make_syscall_handler(filesystem=fs)

    compile_start = time.perf_counter()
    compile_result = run(cpu, handler, max_cycles=MAX_COMPILE_CYCLES, quiet=True)
    compile_wall = time.perf_counter() - compile_start

    compile_cycles = compile_result["total_cycles"]

    if not fs.exists(out_path):
        return {
            "name": name,
            "compiled": False,
            "compile_cycles": compile_cycles,
            "compile_wall_ms": compile_wall * 1000,
            "exec_cycles": 0,
            "exec_wall_ms": 0.0,
            "binary_size": 0,
            "exit_code": -1,
            "expected": expected,
            "correct": False,
        }

    compiled_bin = fs.read_file(out_path)
    binary_size = len(compiled_bin)

    # -- Phase B: Execute the GPU-compiled binary on GPU --
    cpu2 = MLXKernelCPUv2()
    cpu2.load_program(compiled_bin, address=0x10000)
    cpu2.set_pc(0x10000)

    handler2 = make_syscall_handler()

    exec_start = time.perf_counter()
    exec_result = run(cpu2, handler2, max_cycles=MAX_EXEC_CYCLES, quiet=True)
    exec_wall = time.perf_counter() - exec_start

    exec_cycles = exec_result["total_cycles"]
    exit_code = cpu2.get_register(0)
    correct = exit_code == expected

    return {
        "name": name,
        "compiled": True,
        "compile_cycles": compile_cycles,
        "compile_wall_ms": compile_wall * 1000,
        "exec_cycles": exec_cycles,
        "exec_wall_ms": exec_wall * 1000,
        "binary_size": binary_size,
        "exit_code": exit_code,
        "expected": expected,
        "correct": correct,
    }


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_table(results: list[dict], compiler_size: int):
    """Print a formatted results table."""
    w = 90

    print()
    print("=" * w)
    print("  Self-Hosting C Compiler Benchmark -- Metal GPU Execution")
    print("  Host GCC -> cc.c binary -> GPU compiles C -> GPU executes result")
    print("=" * w)
    print()
    print(f"  Compiler binary: {compiler_size:,} bytes")
    print(f"  Max compile cycles: {MAX_COMPILE_CYCLES:,}")
    print(f"  Max exec cycles:    {MAX_EXEC_CYCLES:,}")
    print()

    # Header
    print(f"  {'Program':<16} {'CompCyc':>12} {'CompMs':>9} {'ExecCyc':>10} "
          f"{'ExecMs':>9} {'BinSz':>7} {'Exit':>5} {'Status':>8}")
    print("  " + "-" * (w - 4))

    for r in results:
        if not r:
            continue

        name = r["name"]
        if r["compiled"]:
            status = "PASS" if r["correct"] else "FAIL"
        else:
            status = "NO BIN"

        print(f"  {name:<16} {r['compile_cycles']:>12,} {r['compile_wall_ms']:>8.1f}s "
              f"{r['exec_cycles']:>10,} {r['exec_wall_ms']:>8.1f}s "
              f"{r['binary_size']:>7,} {r['exit_code']:>5} {status:>8}")

    print("  " + "-" * (w - 4))

    # Summary stats
    compiled = [r for r in results if r and r["compiled"]]
    correct = [r for r in results if r and r["correct"]]

    if compiled:
        mean_cc = sum(r["compile_cycles"] for r in compiled) / len(compiled)
        mean_cw = sum(r["compile_wall_ms"] for r in compiled) / len(compiled)
        mean_ec = sum(r["exec_cycles"] for r in compiled) / len(compiled)
        mean_ew = sum(r["exec_wall_ms"] for r in compiled) / len(compiled)
        mean_bs = sum(r["binary_size"] for r in compiled) / len(compiled)

        print()
        print(f"  Compiled:   {len(compiled)}/{len(results)}")
        print(f"  Correct:    {len(correct)}/{len(results)}")
        print(f"  Mean compile: {mean_cc:,.0f} cycles  ({mean_cw:.1f}ms)")
        print(f"  Mean exec:    {mean_ec:,.0f} cycles  ({mean_ew:.1f}ms)")
        print(f"  Mean binary:  {mean_bs:,.0f} bytes")

    print()
    print("=" * w)


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def build_json_output(
    results: list[dict],
    compiler_size: int,
    system_info: dict,
) -> dict:
    """Build the JSON output structure."""
    compiled = [r for r in results if r and r["compiled"]]
    correct = [r for r in results if r and r["correct"]]

    programs = []
    for r in results:
        if not r:
            continue
        programs.append({
            "name": r["name"],
            "compile_cycles": r["compile_cycles"],
            "compile_wall_ms": round(r["compile_wall_ms"], 2),
            "exec_cycles": r["exec_cycles"],
            "exec_wall_ms": round(r["exec_wall_ms"], 2),
            "binary_size": r["binary_size"],
            "exit_code": r["exit_code"],
            "expected": r["expected"],
            "correct": r["correct"],
        })

    summary = {
        "total_programs": len(results),
        "compiled": len(compiled),
        "correct": len(correct),
    }

    if compiled:
        summary["mean_compile_cycles"] = round(
            sum(r["compile_cycles"] for r in compiled) / len(compiled)
        )
        summary["mean_compile_wall_ms"] = round(
            sum(r["compile_wall_ms"] for r in compiled) / len(compiled), 2
        )
        summary["mean_exec_cycles"] = round(
            sum(r["exec_cycles"] for r in compiled) / len(compiled)
        )
        summary["mean_exec_wall_ms"] = round(
            sum(r["exec_wall_ms"] for r in compiled) / len(compiled), 2
        )
        summary["mean_binary_size"] = round(
            sum(r["binary_size"] for r in compiled) / len(compiled)
        )
    else:
        summary["mean_compile_cycles"] = 0
        summary["mean_compile_wall_ms"] = 0.0
        summary["mean_exec_cycles"] = 0
        summary["mean_exec_wall_ms"] = 0.0
        summary["mean_binary_size"] = 0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": system_info,
        "config": {
            "max_compile_cycles": MAX_COMPILE_CYCLES,
            "max_exec_cycles": MAX_EXEC_CYCLES,
        },
        "compiler": {
            "source": "cc.c",
            "binary_size": compiler_size,
        },
        "programs": programs,
        "summary": summary,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("  Self-Hosting C Compiler Benchmark")
    print("  cc.c compiled by host GCC, running on Metal GPU")
    print("=" * 60)
    print()

    # Collect system info
    system_info = collect_system_info()
    print(f"  Platform: {system_info['platform']}")
    print(f"  Python:   {system_info['python_version']}")
    print()

    # Step 1: Build the compiler
    print("[1] Compiling cc.c with host GCC...")
    build_start = time.perf_counter()
    compiler_binary = build_compiler()
    build_elapsed = time.perf_counter() - build_start

    if compiler_binary is None:
        print("  FATAL: Cannot build compiler. Aborting.")
        return 1

    compiler_size = len(compiler_binary)
    print(f"    Compiler binary: {compiler_size:,} bytes ({build_elapsed:.2f}s)")
    print()

    # Step 2: Benchmark each test program
    print(f"[2] Benchmarking {len(TEST_PROGRAMS)} test programs...")
    print()

    results = []
    for i, (name, info) in enumerate(TEST_PROGRAMS.items(), 1):
        source = info["source"]
        expected = info["expected"]

        print(f"  [{i}/{len(TEST_PROGRAMS)}] {name} (expected={expected})")

        result = benchmark_program(name, source, expected, compiler_binary)
        results.append(result)

        if result:
            if result["compiled"]:
                status = "PASS" if result["correct"] else f"FAIL (got {result['exit_code']})"
                print(f"        compile: {result['compile_cycles']:,} cycles  "
                      f"({result['compile_wall_ms']:.1f}ms)")
                print(f"        exec:    {result['exec_cycles']:,} cycles  "
                      f"({result['exec_wall_ms']:.1f}ms)")
                print(f"        binary:  {result['binary_size']:,} bytes  [{status}]")
            else:
                print(f"        COMPILE FAILED ({result['compile_cycles']:,} cycles)")
        else:
            print("        ERROR: benchmark returned no result")

        print()

    # Step 3: Print formatted table
    print_table(results, compiler_size)

    # Step 4: Save results
    json_output = build_json_output(results, compiler_size, system_info)

    with open(RESULTS_PATH, "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"\n  Results saved to {RESULTS_PATH}")

    # Return exit code based on correctness
    correct_count = sum(1 for r in results if r and r["correct"])
    return 0 if correct_count == len(TEST_PROGRAMS) else 1


if __name__ == "__main__":
    sys.exit(main())
