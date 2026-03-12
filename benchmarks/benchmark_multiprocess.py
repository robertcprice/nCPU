#!/usr/bin/env python3
"""
Multi-Process Benchmark — Measure fork, pipe, context switch, and IPC performance
on the Metal GPU multi-process runtime.

Benchmarks:
  1. Fork Latency:              Cycles for fork() + exit(0) + wait()
  2. Context Switch Overhead:   Cycles per context switch (parent/child alternation)
  3. Pipe Throughput:           Cycles per byte for pipe I/O at various payload sizes
  4. Process Create/Teardown:   Cycles to fork N children and wait for all
  5. Single vs Multi-process:   IPS comparison for identical compute workloads
  6. Memory Swap Inference:     Context switch cost minus pure compute cost

Usage:
    python benchmarks/benchmark_multiprocess.py
"""

import sys
import os
import json
import time
import tempfile
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import (
    compile_c_from_string, run, make_syscall_handler,
    ProcessManager, run_multiprocess, HEAP_BASE,
)
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2

RESULTS_PATH = Path(__file__).resolve().parent / "multiprocess_results.json"


# ═══════════════════════════════════════════════════════════════════════════════
# C PROGRAM SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Fork Latency: fork + child exit + parent wait
FORK_LATENCY_SRC = r'''
#include "arm64_libc.h"

int main(void) {
    int pid = fork();
    if (pid == 0) {
        /* Child: exit immediately */
        return 0;
    }
    /* Parent: wait for child */
    int status = 0;
    waitpid(pid, &status, 0);
    printf("fork+wait done\n");
    return 0;
}
'''

# 2. Context Switch Overhead: child does a small amount of work, parent waits
CONTEXT_SWITCH_SRC = r'''
#include "arm64_libc.h"

int main(void) {
    int pid = fork();
    if (pid == 0) {
        /* Child: do 100 iterations of trivial work */
        volatile int sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += i;
        }
        return 0;
    }
    /* Parent: wait for child to finish */
    int status = 0;
    waitpid(pid, &status, 0);
    printf("ctx switch done\n");
    return 0;
}
'''

# 3. Pipe Throughput: child writes N bytes via pipe, parent reads them
def make_pipe_throughput_src(payload_size: int) -> str:
    """Generate C source for pipe throughput benchmark with given payload size."""
    return r'''
#include "arm64_libc.h"

#define PAYLOAD_SIZE ''' + str(payload_size) + r'''
#define CHUNK_SIZE 64

int main(void) {
    int pipefd[2];
    pipe(pipefd);

    int pid = fork();
    if (pid == 0) {
        /* Child: close read end, write payload */
        close(pipefd[0]);
        char buf[CHUNK_SIZE];
        for (int i = 0; i < CHUNK_SIZE; i++) {
            buf[i] = (char)(i & 0x7F);
        }
        int written = 0;
        while (written < PAYLOAD_SIZE) {
            int to_write = PAYLOAD_SIZE - written;
            if (to_write > CHUNK_SIZE) to_write = CHUNK_SIZE;
            sys_write(pipefd[1], buf, to_write);
            written += to_write;
        }
        close(pipefd[1]);
        return 0;
    }

    /* Parent: close write end, read all data */
    close(pipefd[1]);
    char rbuf[CHUNK_SIZE];
    int total_read = 0;
    while (total_read < PAYLOAD_SIZE) {
        int n = sys_read(pipefd[0], rbuf, CHUNK_SIZE);
        if (n <= 0) break;
        total_read += n;
    }
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    printf("pipe %d bytes done\n", total_read);
    return 0;
}
'''

# 4. Process Creation/Teardown: fork N children, each exits, parent waits for all
def make_fork_n_src(n_children: int) -> str:
    """Generate C source that forks N children, each exits immediately."""
    return r'''
#include "arm64_libc.h"

#define N_CHILDREN ''' + str(n_children) + r'''

int main(void) {
    int i;
    for (i = 0; i < N_CHILDREN; i++) {
        int pid = fork();
        if (pid == 0) {
            /* Child: exit immediately */
            return 0;
        }
    }
    /* Parent: wait for all children */
    int status = 0;
    for (i = 0; i < N_CHILDREN; i++) {
        waitpid(-1, &status, 0);
    }
    printf("forked %d children done\n", N_CHILDREN);
    return 0;
}
'''

# 5. Simple compute loop (used for both single and multi-process IPS comparison)
COMPUTE_LOOP_SRC = r'''
#include "arm64_libc.h"

int main(void) {
    volatile long sum = 0;
    for (int i = 0; i < 1000; i++) {
        sum += i;
    }
    printf("sum=%ld\n", sum);
    return 0;
}
'''


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compile_source(source: str, label: str) -> str | None:
    """Compile a C source string to a temp binary. Returns path or None."""
    bin_path = tempfile.mktemp(suffix=".bin")
    ok = compile_c_from_string(source, bin_path, quiet=True)
    if not ok:
        print(f"  FAILED to compile: {label}")
        return None
    return bin_path


def run_multiprocess_benchmark(
    binary_path: str,
    max_cycles: int = 10_000_000,
    time_slice: int = 100_000,
    quiet: bool = True,
) -> dict | None:
    """Run a binary in multi-process mode and return results."""
    binary = Path(binary_path).read_bytes()
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    fs = GPUFilesystem()
    proc_mgr = ProcessManager(cpu, fs)
    proc_mgr.create_init_process(binary, fd_table={}, cwd="/")

    base_handler = make_syscall_handler(filesystem=fs)
    result = run_multiprocess(
        proc_mgr, base_handler,
        max_total_cycles=max_cycles,
        time_slice=time_slice,
        quiet=quiet,
    )
    return result


def run_single_process_benchmark(
    binary_path: str,
    max_cycles: int = 10_000_000,
    quiet: bool = True,
) -> dict | None:
    """Run a binary in single-process mode and return results."""
    binary = Path(binary_path).read_bytes()
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    handler = make_syscall_handler()
    result = run(cpu, handler, max_cycles=max_cycles, quiet=quiet)
    return result


def cleanup(*paths):
    """Remove temp files."""
    for p in paths:
        if p and os.path.exists(p):
            os.unlink(p)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_fork_latency(iterations: int = 5) -> dict:
    """Measure fork + exit + wait cycle cost."""
    print("  [1/6] Fork Latency")

    bin_path = compile_source(FORK_LATENCY_SRC, "fork_latency")
    if not bin_path:
        return {}

    cycle_counts = []
    elapsed_times = []

    for i in range(iterations):
        result = run_multiprocess_benchmark(bin_path)
        if result:
            cycle_counts.append(result["total_cycles"])
            elapsed_times.append(result["elapsed"])

    cleanup(bin_path)

    if not cycle_counts:
        return {}

    mean_cycles = statistics.mean(cycle_counts)
    stdev_cycles = statistics.stdev(cycle_counts) if len(cycle_counts) > 1 else 0.0
    mean_elapsed = statistics.mean(elapsed_times)

    print(f"        {mean_cycles:,.0f} cycles (stdev {stdev_cycles:,.0f}), "
          f"{mean_elapsed * 1000:.1f}ms")

    return {
        "mean_cycles": mean_cycles,
        "stdev_cycles": stdev_cycles,
        "mean_elapsed_ms": mean_elapsed * 1000,
        "samples": cycle_counts,
        "iterations": iterations,
    }


def benchmark_context_switch(iterations: int = 5) -> dict:
    """Measure context switch overhead via parent/child alternation."""
    print("  [2/6] Context Switch Overhead")

    bin_path = compile_source(CONTEXT_SWITCH_SRC, "context_switch")
    if not bin_path:
        return {}

    cycle_counts = []
    switch_counts = []
    elapsed_times = []

    for i in range(iterations):
        result = run_multiprocess_benchmark(bin_path)
        if result:
            cycle_counts.append(result["total_cycles"])
            switch_counts.append(result["total_context_switches"])
            elapsed_times.append(result["elapsed"])

    cleanup(bin_path)

    if not cycle_counts:
        return {}

    mean_cycles = statistics.mean(cycle_counts)
    stdev_cycles = statistics.stdev(cycle_counts) if len(cycle_counts) > 1 else 0.0
    mean_switches = statistics.mean(switch_counts)
    mean_elapsed = statistics.mean(elapsed_times)
    cycles_per_switch = mean_cycles / mean_switches if mean_switches > 0 else 0

    print(f"        {mean_cycles:,.0f} cycles, {mean_switches:.1f} switches, "
          f"{cycles_per_switch:,.0f} cycles/switch, {mean_elapsed * 1000:.1f}ms")

    return {
        "mean_cycles": mean_cycles,
        "stdev_cycles": stdev_cycles,
        "mean_switches": mean_switches,
        "cycles_per_switch": cycles_per_switch,
        "mean_elapsed_ms": mean_elapsed * 1000,
        "samples": cycle_counts,
        "iterations": iterations,
    }


def benchmark_pipe_throughput(iterations: int = 3) -> dict:
    """Measure pipe I/O throughput for different payload sizes."""
    print("  [3/6] Pipe Throughput")

    payload_sizes = [64, 256, 1024]
    results_by_size = {}

    for size in payload_sizes:
        src = make_pipe_throughput_src(size)
        bin_path = compile_source(src, f"pipe_{size}")
        if not bin_path:
            continue

        cycle_counts = []
        elapsed_times = []

        for i in range(iterations):
            result = run_multiprocess_benchmark(bin_path, max_cycles=20_000_000)
            if result:
                cycle_counts.append(result["total_cycles"])
                elapsed_times.append(result["elapsed"])

        cleanup(bin_path)

        if not cycle_counts:
            continue

        mean_cycles = statistics.mean(cycle_counts)
        mean_elapsed = statistics.mean(elapsed_times)
        cycles_per_byte = mean_cycles / size
        # Wall-clock throughput (dominated by dispatch overhead for small payloads)
        bytes_per_ms = size / (mean_elapsed * 1000) if mean_elapsed > 0 else 0

        print(f"        {size:>5d}B: {mean_cycles:>10,.0f} cycles, "
              f"{cycles_per_byte:>8,.1f} cycles/byte, "
              f"{mean_elapsed * 1000:.1f}ms wall")

        results_by_size[str(size)] = {
            "payload_bytes": size,
            "mean_cycles": mean_cycles,
            "cycles_per_byte": cycles_per_byte,
            "bytes_per_ms": bytes_per_ms,
            "mean_elapsed_ms": mean_elapsed * 1000,
            "samples": cycle_counts,
        }

    return results_by_size


def benchmark_process_creation(iterations: int = 3) -> dict:
    """Measure cost of forking N children and waiting for all."""
    print("  [4/6] Process Creation/Teardown")

    child_counts = [2, 4, 8]
    results_by_n = {}

    for n in child_counts:
        src = make_fork_n_src(n)
        bin_path = compile_source(src, f"fork_{n}")
        if not bin_path:
            continue

        cycle_counts = []
        elapsed_times = []

        for i in range(iterations):
            result = run_multiprocess_benchmark(
                bin_path,
                max_cycles=50_000_000,
                time_slice=100_000,
            )
            if result:
                cycle_counts.append(result["total_cycles"])
                elapsed_times.append(result["elapsed"])

        cleanup(bin_path)

        if not cycle_counts:
            continue

        mean_cycles = statistics.mean(cycle_counts)
        mean_elapsed = statistics.mean(elapsed_times)
        cycles_per_process = mean_cycles / n

        print(f"        N={n:>2d}: {mean_cycles:>10,.0f} cycles total, "
              f"{cycles_per_process:>10,.0f} cycles/process, "
              f"{mean_elapsed * 1000:.1f}ms")

        results_by_n[str(n)] = {
            "n_children": n,
            "mean_cycles": mean_cycles,
            "cycles_per_process": cycles_per_process,
            "mean_elapsed_ms": mean_elapsed * 1000,
            "samples": cycle_counts,
        }

    return results_by_n


def benchmark_ips_comparison(iterations: int = 5) -> dict:
    """Compare IPS for same workload in single vs multi-process mode."""
    print("  [5/6] Single vs Multi-process IPS")

    bin_path = compile_source(COMPUTE_LOOP_SRC, "compute_loop")
    if not bin_path:
        return {}

    single_ips_list = []
    single_cycles_list = []
    multi_ips_list = []
    multi_cycles_list = []

    for i in range(iterations):
        # Single-process
        sr = run_single_process_benchmark(bin_path)
        if sr:
            single_ips_list.append(sr["ips"])
            single_cycles_list.append(sr["total_cycles"])

        # Multi-process (same binary, but routed through ProcessManager)
        mr = run_multiprocess_benchmark(bin_path, time_slice=500_000)
        if mr:
            multi_ips_list.append(mr["ips"])
            multi_cycles_list.append(mr["total_cycles"])

    cleanup(bin_path)

    if not single_ips_list or not multi_ips_list:
        return {}

    mean_single_ips = statistics.mean(single_ips_list)
    mean_multi_ips = statistics.mean(multi_ips_list)
    mean_single_cycles = statistics.mean(single_cycles_list)
    mean_multi_cycles = statistics.mean(multi_cycles_list)
    overhead_pct = ((mean_multi_cycles - mean_single_cycles) / mean_single_cycles * 100
                    if mean_single_cycles > 0 else 0)

    print(f"        Single:  {mean_single_ips:>10,.0f} IPS ({mean_single_cycles:,.0f} cycles)")
    print(f"        Multi:   {mean_multi_ips:>10,.0f} IPS ({mean_multi_cycles:,.0f} cycles)")
    print(f"        Overhead: {overhead_pct:+.1f}% cycles")

    return {
        "single_process": {
            "mean_ips": mean_single_ips,
            "mean_cycles": mean_single_cycles,
            "samples_ips": single_ips_list,
        },
        "multi_process": {
            "mean_ips": mean_multi_ips,
            "mean_cycles": mean_multi_cycles,
            "samples_ips": multi_ips_list,
        },
        "overhead_pct": overhead_pct,
    }


def benchmark_memory_swap_inference(
    fork_result: dict,
    ctx_switch_result: dict,
) -> dict:
    """Infer memory swap time from context switch overhead vs pure fork latency."""
    print("  [6/6] Memory Swap Inference")

    if not fork_result or not ctx_switch_result:
        print("        Skipped (missing prerequisite results)")
        return {}

    fork_cycles = fork_result.get("mean_cycles", 0)
    ctx_cycles = ctx_switch_result.get("mean_cycles", 0)
    ctx_switches = ctx_switch_result.get("mean_switches", 0)

    # The difference between total context-switch benchmark cycles and
    # the fork-only benchmark gives us the overhead attributable to
    # memory swapping and scheduling during the additional switches.
    delta_cycles = ctx_cycles - fork_cycles
    if ctx_switches > 0:
        swap_per_switch = delta_cycles / ctx_switches if delta_cycles > 0 else 0
    else:
        swap_per_switch = 0

    fork_elapsed = fork_result.get("mean_elapsed_ms", 0)
    ctx_elapsed = ctx_switch_result.get("mean_elapsed_ms", 0)
    delta_ms = ctx_elapsed - fork_elapsed

    print(f"        Fork-only:      {fork_cycles:>10,.0f} cycles ({fork_elapsed:.1f}ms)")
    print(f"        With switches:  {ctx_cycles:>10,.0f} cycles ({ctx_elapsed:.1f}ms)")
    print(f"        Delta:          {delta_cycles:>10,.0f} cycles ({delta_ms:.1f}ms)")
    if ctx_switches > 0:
        print(f"        Inferred swap:  {swap_per_switch:>10,.0f} cycles/switch")

    return {
        "fork_only_cycles": fork_cycles,
        "with_switches_cycles": ctx_cycles,
        "delta_cycles": delta_cycles,
        "delta_ms": delta_ms,
        "inferred_swap_per_switch": swap_per_switch,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_number(n: float) -> str:
    """Format a number with comma separators."""
    if n >= 1_000_000:
        return f"{n:,.0f}"
    elif n >= 1000:
        return f"{n:,.0f}"
    elif n >= 1:
        return f"{n:.1f}"
    else:
        return f"{n:.3f}"


def print_summary(results: dict):
    """Print a formatted summary table."""

    # Extract key metrics
    fork_cycles = results.get("fork_latency", {}).get("mean_cycles", 0)
    fork_ms = results.get("fork_latency", {}).get("mean_elapsed_ms", 0)

    ctx_per_switch = results.get("context_switch", {}).get("cycles_per_switch", 0)
    ctx_ms = results.get("context_switch", {}).get("mean_elapsed_ms", 0)
    ctx_switches = results.get("context_switch", {}).get("mean_switches", 0)

    # Pipe throughput: use the largest payload (best amortization of overhead)
    pipe_results = results.get("pipe_throughput", {})
    best_cpb = 0
    for size_key, data in pipe_results.items():
        cpb = data.get("cycles_per_byte", 0)
        if cpb > 0 and (best_cpb == 0 or cpb < best_cpb):
            best_cpb = cpb

    # Process creation: use N=4 as representative
    proc_create = results.get("process_creation", {})
    cycles_per_proc = proc_create.get("4", {}).get("cycles_per_process", 0)

    # IPS comparison
    ips_data = results.get("ips_comparison", {})
    single_ips = ips_data.get("single_process", {}).get("mean_ips", 0)
    multi_ips = ips_data.get("multi_process", {}).get("mean_ips", 0)
    overhead = ips_data.get("overhead_pct", 0)

    # Swap inference
    swap_data = results.get("memory_swap", {})
    swap_per_switch = swap_data.get("inferred_swap_per_switch", 0)

    w = 60
    print()
    print("=" * w)
    print()

    # Build content lines, then pad to uniform width
    bw = 56  # content width between pipes (excluding pipe chars and outer indent)

    def box_line(content: str) -> str:
        """Format a content line padded to box width."""
        return f"  |  {content:<{bw}}  |"

    bar = f"  +{'-' * (bw + 4)}+"
    dbar = f"  +{'=' * (bw + 4)}+"
    blank = box_line("")

    content_rows = [
        f"Fork latency:            {fork_cycles:>10,.0f} cycles ({fork_ms:.1f}ms)",
        f"Context switch:          {ctx_per_switch:>10,.0f} cycles/switch",
        f"Pipe throughput:         {best_cpb:>10.1f} cycles/byte",
        f"Process create+teardown: {cycles_per_proc:>10,.0f} cycles/process",
        f"Single-process IPS:      {single_ips:>10,.0f}",
    ]
    if overhead != 0:
        content_rows.append(
            f"Multi-process IPS:       {multi_ips:>10,.0f} ({overhead:+.1f}%)")
    else:
        content_rows.append(
            f"Multi-process IPS:       {multi_ips:>10,.0f}")
    if swap_per_switch > 0:
        content_rows.append(
            f"Memory swap (inferred):  {swap_per_switch:>10,.0f} cycles/switch")

    print(dbar)
    title = "Multi-Process Benchmark Results"
    print(f"  |{title:^{bw + 4}}|")
    print(dbar)
    print(blank)
    for row in content_rows:
        print(box_line(row))
    print(blank)
    print(dbar)

    # Detail breakdown
    if pipe_results:
        print()
        print("  Detail: Pipe Throughput by Payload Size")
        print(f"  {'Size':>8}  {'Cycles':>12}  {'Cycles/Byte':>12}  {'Wall (ms)':>10}")
        print(f"  {'----':>8}  {'------':>12}  {'-----------':>12}  {'---------':>10}")
        for size_key in sorted(pipe_results.keys(), key=int):
            data = pipe_results[size_key]
            print(f"  {data['payload_bytes']:>7}B  {data['mean_cycles']:>12,.0f}  "
                  f"{data['cycles_per_byte']:>12.1f}  {data['mean_elapsed_ms']:>10.1f}")

    if proc_create:
        print()
        print("  Detail: Process Creation Scaling")
        print(f"  {'Children':>8}  {'Total Cycles':>14}  {'Cycles/Process':>14}  {'Time (ms)':>10}")
        print(f"  {'--------':>8}  {'-----------':>14}  {'--------------':>14}  {'---------':>10}")
        for n_key in sorted(proc_create.keys(), key=int):
            data = proc_create[n_key]
            print(f"  {data['n_children']:>8}  {data['mean_cycles']:>14,.0f}  "
                  f"{data['cycles_per_process']:>14,.0f}  {data['mean_elapsed_ms']:>10.1f}")

    print()
    print("=" * w)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Multi-Process Benchmark -- Metal GPU Execution")
    print("  fork / pipe / wait / context-switch performance")
    print("=" * 60)
    print()

    results = {}

    # 1. Fork Latency
    results["fork_latency"] = benchmark_fork_latency(iterations=5)
    print()

    # 2. Context Switch Overhead
    results["context_switch"] = benchmark_context_switch(iterations=5)
    print()

    # 3. Pipe Throughput
    results["pipe_throughput"] = benchmark_pipe_throughput(iterations=3)
    print()

    # 4. Process Creation/Teardown
    results["process_creation"] = benchmark_process_creation(iterations=3)
    print()

    # 5. Single vs Multi-process IPS
    results["ips_comparison"] = benchmark_ips_comparison(iterations=5)
    print()

    # 6. Memory Swap Inference
    results["memory_swap"] = benchmark_memory_swap_inference(
        results.get("fork_latency", {}),
        results.get("context_switch", {}),
    )

    # Print summary table
    print_summary(results)

    # Save results to JSON
    # Convert any non-serializable types
    def sanitize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return str(obj)

    clean_results = sanitize(results)
    clean_results["_meta"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": "Apple Silicon Metal GPU",
        "description": "Multi-process benchmark: fork, pipe, context switch, IPS",
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
