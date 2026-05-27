#!/usr/bin/env python3
"""
Baseline Comparison: Direct A/B of conventional vs neural-enhanced GPU OS.

Runs the EXACT same workload through two configurations:
  A. Conventional — demo.py path: zero neural models, pure Metal GPU
  B. Neural-enhanced — neural_demo.py path: all 9 neural models active

Measures:
  - IPS (raw and GPU-only, excluding GCC cross-compilation time)
  - Wall-clock time
  - Peak RSS memory usage
  - Total neural inferences
  - Output correctness (both configurations produce identical shell output)

This is the definitive "how much does neural enhancement cost?" experiment
for the nCPU paper.

Usage:
    python benchmarks/baseline_comparison.py
    python benchmarks/baseline_comparison.py --trials 3
"""

import sys
import os
import io
import time
import json
import math
import argparse
import logging
import tempfile
import resource
from pathlib import Path
from typing import Optional, Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.getLogger("ncpu.neural.metal_inference").setLevel(logging.ERROR)

from ncpu.os.gpu.runner import (
    compile_c, run, make_syscall_handler, read_string_from_gpu,
    HEAP_BASE,
)
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2
from ncpu.utils.provenance import collect_provenance

# ---------------------------------------------------------------------------
# Identical workload for both configurations
# ---------------------------------------------------------------------------

DEMO_COMMANDS = [
    "cat /etc/motd",
    "cat /etc/os-release",
    "ls /home/user",
    "echo Baseline comparison test > /tmp/test.txt",
    "cat /tmp/test.txt",
    "wc /tmp/test.txt",
    "echo === System Log === > /tmp/syslog.txt",
    "echo Boot: models loaded >> /tmp/syslog.txt",
    "echo Cache: active >> /tmp/syslog.txt",
    "echo Watchdog: monitoring >> /tmp/syslog.txt",
    "echo GIC: dispatching >> /tmp/syslog.txt",
    "cat /tmp/syslog.txt",
    "wc /tmp/syslog.txt",
    "ls /home/user | grep .c",
    "ls /home/user | sort",
    "cat /etc/motd",
    "cat /etc/os-release",
    "cat /home/user/README.txt",
    "cc hello.c",
    "cc fib.c",
    "cc sieve.c",
    "ls /bin",
    "cat /home/user/hello.c",
    "cat /home/user/fib.c",
    "cat /home/user/sieve.c",
    "mkdir /tmp/results",
    "echo test1 > /tmp/results/a.txt",
    "echo test2 > /tmp/results/b.txt",
    "echo test3 > /tmp/results/c.txt",
    "ls /tmp/results",
    "cat /tmp/results/a.txt",
    "cat /tmp/results/b.txt",
    "cat /tmp/syslog.txt",
    "ps",
    "echo Baseline comparison complete.",
    "run /bin/fib",
]


# ---------------------------------------------------------------------------
# Filesystem bootstrap
# ---------------------------------------------------------------------------

def bootstrap_filesystem() -> GPUFilesystem:
    """Create the standard filesystem (identical for both configs)."""
    fs = GPUFilesystem()
    for d in ["/home/user", "/var/log", "/usr/lib", "/tmp"]:
        fs.mkdir(d)

    fs.write_file("/etc/motd",
        "Welcome to GPU-Native UNIX OS v3.1 - Neural Enhanced Edition\n"
        "Running on Apple Silicon Metal with 8 neural models active\n"
        "Type 'help' for commands.\n"
    )
    fs.write_file("/etc/hostname", "gpu0\n")
    fs.write_file("/etc/os-release",
        "NAME=\"GPU-Native UNIX\"\n"
        "VERSION=\"3.1-neural\"\n"
        "ARCH=\"ARM64 Metal\"\n"
        "FEATURES=\"neural-display neural-cache neural-scheduler neural-watchdog "
        "neural-gic neural-compiler-opt neural-syscall-predict neural-prefetch\"\n"
    )
    fs.write_file("/home/user/hello.c",
        '#include "arm64_libc.h"\n\n'
        'int main(void) {\n'
        '    printf("Hello from GPU-compiled C!\\n");\n'
        '    printf("Running on Metal silicon with neural OS.\\n");\n'
        '    return 0;\n'
        '}\n'
    )
    fs.write_file("/home/user/fib.c",
        '#include "arm64_libc.h"\n\n'
        'int main(void) {\n'
        '    printf("Fibonacci sequence:\\n");\n'
        '    long a = 0, b = 1;\n'
        '    for (int i = 0; i < 20; i++) {\n'
        '        printf("  fib(%d) = %ld\\n", i, a);\n'
        '        long tmp = a + b;\n'
        '        a = b;\n'
        '        b = tmp;\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    fs.write_file("/home/user/fork_test.c",
        '#include "arm64_libc.h"\n\n'
        'int main(void) {\n'
        '    printf("Parent PID: %d\\n", getpid());\n'
        '    int pid = fork();\n'
        '    if (pid == 0) {\n'
        '        printf("Child process (PID %d, parent %d)\\n", getpid(), getppid());\n'
        '        exit(0);\n'
        '    } else if (pid > 0) {\n'
        '        printf("Forked child PID: %d\\n", pid);\n'
        '        int status;\n'
        '        waitpid(pid, &status, 0);\n'
        '        printf("Child exited, parent done\\n");\n'
        '    } else {\n'
        '        printf("Fork failed!\\n");\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    fs.write_file("/home/user/sieve.c",
        '#include "arm64_libc.h"\n\n'
        'int main(void) {\n'
        '    printf("Sieve of Eratosthenes (primes < 100):\\n");\n'
        '    int sieve[100];\n'
        '    for (int i = 0; i < 100; i++) sieve[i] = 1;\n'
        '    sieve[0] = sieve[1] = 0;\n'
        '    for (int i = 2; i < 10; i++) {\n'
        '        if (sieve[i]) {\n'
        '            for (int j = i*i; j < 100; j += i)\n'
        '                sieve[j] = 0;\n'
        '        }\n'
        '    }\n'
        '    int count = 0;\n'
        '    for (int i = 2; i < 100; i++) {\n'
        '        if (sieve[i]) {\n'
        '            printf("  %d", i);\n'
        '            count++;\n'
        '            if (count % 10 == 0) printf("\\n");\n'
        '        }\n'
        '    }\n'
        '    printf("\\n  Total: %d primes\\n", count);\n'
        '    return 0;\n'
        '}\n'
    )
    fs.write_file("/home/user/README.txt",
        "GPU-Native UNIX OS v3.1 - Neural Enhanced Edition\n"
        "==================================================\n"
        "This shell is compiled C running on Apple Silicon Metal GPU.\n"
        "Neural models enhance display, caching, scheduling, and monitoring.\n"
    )
    fs.chdir("/home/user")
    return fs


def make_demo_reader(commands: list):
    idx = [0]
    def reader(fd, max_len):
        if fd != 0:
            return None
        if idx[0] < len(commands):
            cmd = commands[idx[0]]
            idx[0] += 1
            return (cmd + "\n").encode("ascii")[:max_len]
        return b""
    return reader


GPU_OS_DIR = PROJECT_ROOT / "ncpu" / "os" / "gpu"
MODELS_DIR = PROJECT_ROOT / "models"


def compile_shell() -> bytes:
    c_file = GPU_OS_DIR / "src" / "arm64_unix_shell.c"
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name
    if not compile_c(str(c_file), bin_path, quiet=True):
        raise RuntimeError("Shell compilation failed")
    binary = Path(bin_path).read_bytes()
    os.unlink(bin_path)
    return binary


# ---------------------------------------------------------------------------
# Output capture — capture all shell output text for correctness comparison
# ---------------------------------------------------------------------------

class OutputCapture:
    """Capture all text written to stdout by the shell for comparison."""

    def __init__(self):
        self.lines = []
        self._buf = []

    def write(self, text):
        self._buf.append(text)

    def flush(self):
        pass

    def get_output(self) -> str:
        return "".join(self._buf)


# ---------------------------------------------------------------------------
# Run conventional (no neural models)
# ---------------------------------------------------------------------------

def run_conventional(binary: bytes) -> Dict[str, Any]:
    """Run with zero neural models — pure Metal GPU execution."""
    fs = bootstrap_filesystem()
    initial_file_count = len(fs.files)

    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    def on_exec(bin_path_str):
        resolved = fs.resolve_path(bin_path_str)
        binary_data = fs.read_file(resolved)
        if binary_data:
            cpu.load_program(binary_data, address=0x10000)
            cpu.set_pc(0x10000)
            return True
        return False

    compile_time_accum = [0.0]

    # Wrap handler to track compilation time
    from ncpu.os.gpu.runner import SYS_COMPILE
    base_handler = make_syscall_handler(
        filesystem=fs,
        on_exec=on_exec,
        on_read=make_demo_reader(DEMO_COMMANDS),
    )

    def timed_handler(cpu_inst):
        syscall_num = cpu_inst.get_register(8)
        if syscall_num == SYS_COMPILE:
            t0 = time.perf_counter()
            result = base_handler(cpu_inst)
            compile_time_accum[0] += time.perf_counter() - t0
            return result
        return base_handler(cpu_inst)

    # Capture output
    capture = OutputCapture()
    old_stdout = sys.stdout
    sys.stdout = capture

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()

    results = run(
        cpu, timed_handler,
        max_cycles=500_000_000,
        quiet=True,
    )

    elapsed = time.perf_counter() - start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    sys.stdout = old_stdout
    output_text = capture.get_output()

    total_cycles = results.get("total_cycles", 0)
    gpu_time = max(elapsed - compile_time_accum[0], 0.001)
    gpu_ips = total_cycles / gpu_time

    return {
        "config": "conventional",
        "total_cycles": total_cycles,
        "wall_time_s": elapsed,
        "compile_time_s": compile_time_accum[0],
        "gpu_time_s": gpu_time,
        "ips_raw": results.get("ips", 0),
        "ips_gpu_only": gpu_ips,
        "neural_inferences": 0,
        "models_active": 0,
        "peak_rss_kb": rss_after,
        "rss_delta_kb": rss_after - rss_before,
        "stop_reason": str(results.get("stop_reason", "unknown")),
        "files_created": len(fs.files) - initial_file_count,
        "output_text": output_text,
    }


# ---------------------------------------------------------------------------
# Run neural-enhanced (all 9 models)
# ---------------------------------------------------------------------------

def run_neural(binary: bytes) -> Dict[str, Any]:
    """Run with all 9 neural models active."""
    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    models_loaded = []
    total_inferences = [0]
    compile_time_accum = [0.0]

    # Load all neural models
    display = None
    try:
        from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
        display = NeuralDisplayV2()
        models_loaded.append("Display")
    except Exception:
        pass

    neural_cache = None
    try:
        from ncpu.os.neuros.cache import NeuralCache
        neural_cache = NeuralCache(device=device)
        optimal_path = MODELS_DIR / "os" / "cache_replace_optimal.pt"
        original_path = MODELS_DIR / "os" / "cache_replace.pt"
        replace_path = str(optimal_path) if optimal_path.exists() else str(original_path)
        result = neural_cache.load(
            replace_path=replace_path,
            prefetch_path=str(MODELS_DIR / "os" / "prefetch.pt"),
        )
        if result.get("replacer"):
            models_loaded.append("Cache")
        if result.get("prefetcher"):
            models_loaded.append("Prefetch")
    except Exception:
        neural_cache = None

    watchdog = None
    try:
        from ncpu.os.neuros.watchdog import NeuralWatchdog
        watchdog = NeuralWatchdog(device=device)
        if watchdog.load(str(MODELS_DIR / "os" / "watchdog.pt")):
            models_loaded.append("Watchdog")
        else:
            watchdog = None
    except Exception:
        watchdog = None

    gic = None
    try:
        from ncpu.os.neuros.interrupts import NeuralGIC
        gic = NeuralGIC(device=device)
        if gic.load(str(MODELS_DIR / "os" / "gic.pt")):
            models_loaded.append("GIC")
        else:
            gic = None
    except Exception:
        gic = None

    compiler_opt = None
    try:
        from ncpu.os.neuros.compiler import PeepholeOptimizerNet
        compiler_opt = PeepholeOptimizerNet().to(device)
        path = MODELS_DIR / "os" / "compiler_optimizer.pt"
        if path.exists():
            state_dict = torch.load(str(path), map_location=device, weights_only=True)
            compiler_opt.load_state_dict(state_dict)
            compiler_opt.eval()
            models_loaded.append("Compiler")
        else:
            compiler_opt = None
    except Exception:
        compiler_opt = None

    from ncpu.os.gpu.neural_demo import (
        NeuralSyscallPredictor, NeuralCommandSuggestor,
        NeuralCacheFS, NeuralMemoryAccessAnalyzer, WatchdogMonitor,
        NeuralGICWrapper, NeuralCompilationAdvisor,
    )

    syscall_predictor = NeuralSyscallPredictor()
    models_loaded.append("SyscallPred")
    command_suggestor = NeuralCommandSuggestor()
    models_loaded.append("CmdSuggest")

    # Setup filesystem
    fs = bootstrap_filesystem()
    initial_file_count = len(fs.files)

    # Cache wrapper
    cache_fs = None
    if neural_cache is not None:
        cache_fs = NeuralCacheFS(fs, neural_cache)
        original_read_file = fs.read_file
        original_write_file = fs.write_file

        def tracked_read_file(path):
            cache_fs.on_file_read(path)
            return original_read_file(path)

        def tracked_write_file(path, data):
            cache_fs.on_file_write(path)
            return original_write_file(path, data)

        fs.read_file = tracked_read_file
        fs.write_file = tracked_write_file

    mem_analyzer = None
    if neural_cache is not None:
        mem_analyzer = NeuralMemoryAccessAnalyzer(neural_cache)
        models_loaded.append("MemAnalyzer")

    watchdog_monitor = None
    if watchdog is not None:
        watchdog_monitor = WatchdogMonitor(
            watchdog, cache_fs=cache_fs,
            syscall_predictor=syscall_predictor,
            check_interval=50_000,
        )

    gic_wrapper = None
    if gic is not None:
        gic_wrapper = NeuralGICWrapper(gic, device)

    compile_advisor = None
    if compiler_opt is not None:
        compile_advisor = NeuralCompilationAdvisor(compiler_opt, device)

    # GPU
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    def on_exec(bin_path_str):
        resolved = fs.resolve_path(bin_path_str)
        binary_data = fs.read_file(resolved)
        if binary_data:
            cpu.load_program(binary_data, address=0x10000)
            cpu.set_pc(0x10000)
            return True
        return False

    base_handler = make_syscall_handler(
        filesystem=fs,
        on_exec=on_exec,
        neural_display=display,
        on_read=make_demo_reader(DEMO_COMMANDS),
    )

    _call_count = [0]
    _compile_count = [0]
    from ncpu.os.gpu.runner import SYS_COMPILE
    from ncpu.os.gpu.neural_demo import DEFAULT_SAMPLING_CONFIG
    _sampling_cfg = DEFAULT_SAMPLING_CONFIG

    def neural_handler(cpu_inst):
        _call_count[0] += 1
        syscall_num = cpu_inst.get_register(8)

        syscall_predictor.observe(syscall_num)
        if gic_wrapper and _call_count[0] % _sampling_cfg.gic_interval == 0:
            gic_wrapper.on_syscall(syscall_num)
            total_inferences[0] += 1

        if mem_analyzer:
            try:
                pc_val = cpu_inst.pc
                if pc_val > 0:
                    mem_analyzer.record_access(pc_val)
            except Exception:
                pass

        if watchdog_monitor and _call_count[0] % _sampling_cfg.watchdog_interval == 0:
            watchdog_monitor.maybe_check(_call_count[0] * 1000)
            total_inferences[0] += 1

        is_compile = (syscall_num == SYS_COMPILE)
        compile_src = None
        if is_compile and compile_advisor:
            _compile_count[0] += 1
            if _compile_count[0] % _sampling_cfg.compiler_interval == 0:
                try:
                    compile_src = read_string_from_gpu(cpu_inst, cpu_inst.get_register(0))
                except Exception:
                    pass

        if is_compile:
            t0 = time.perf_counter()
            result = base_handler(cpu_inst)
            compile_time_accum[0] += time.perf_counter() - t0
        else:
            result = base_handler(cpu_inst)

        if is_compile and compile_advisor and compile_src:
            ret_val = cpu_inst.get_register(0)
            if ret_val == 0:
                compile_advisor.on_compile(compile_src, 2048)
                total_inferences[0] += 1

        return result

    # Capture output
    capture = OutputCapture()
    old_stdout = sys.stdout
    sys.stdout = capture

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()

    results = run(
        cpu, neural_handler,
        max_cycles=500_000_000,
        quiet=True,
        neural_display=display,
    )

    elapsed = time.perf_counter() - start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    sys.stdout = old_stdout
    output_text = capture.get_output()

    total_cycles = results.get("total_cycles", 0)
    gpu_time = max(elapsed - compile_time_accum[0], 0.001)
    gpu_ips = total_cycles / gpu_time

    # Count all inferences
    if cache_fs:
        total_inferences[0] += cache_fs.read_count + cache_fs.write_count
    total_inferences[0] += syscall_predictor.total_observed
    total_inferences[0] += command_suggestor.total_commands
    if display:
        total_inferences[0] += 1

    if watchdog_monitor:
        watchdog_monitor.run_session_checks(total_cycles, 1, n_checks=5)
        total_inferences[0] += watchdog_monitor.watchdog.total_checks

    return {
        "config": "neural-enhanced",
        "total_cycles": total_cycles,
        "wall_time_s": elapsed,
        "compile_time_s": compile_time_accum[0],
        "gpu_time_s": gpu_time,
        "ips_raw": results.get("ips", 0),
        "ips_gpu_only": gpu_ips,
        "neural_inferences": total_inferences[0],
        "models_active": len(models_loaded),
        "models_loaded": models_loaded,
        "peak_rss_kb": rss_after,
        "rss_delta_kb": rss_after - rss_before,
        "stop_reason": str(results.get("stop_reason", "unknown")),
        "files_created": len(fs.files) - initial_file_count,
        "output_text": output_text,
    }


# ---------------------------------------------------------------------------
# Output correctness comparison
# ---------------------------------------------------------------------------

def compare_outputs(conv_text: str, neural_text: str) -> Dict[str, Any]:
    """Compare shell output between conventional and neural runs.

    Both should produce identical text since the neural models are
    side-channel enhancements that do not modify execution semantics.
    The neural display captures text but does not change what the shell prints.
    """
    conv_lines = conv_text.strip().splitlines()
    neural_lines = neural_text.strip().splitlines()

    # Filter out ANSI escape sequences for comparison (neural display might inject)
    import re
    ansi_re = re.compile(r'\033\[[0-9;]*m')

    def clean(line):
        return ansi_re.sub('', line).strip()

    conv_clean = [clean(l) for l in conv_lines if clean(l)]
    neural_clean = [clean(l) for l in neural_lines if clean(l)]

    matching = 0
    mismatched = []
    max_len = max(len(conv_clean), len(neural_clean))

    for i in range(min(len(conv_clean), len(neural_clean))):
        if conv_clean[i] == neural_clean[i]:
            matching += 1
        else:
            mismatched.append({
                "line": i + 1,
                "conventional": conv_clean[i][:80],
                "neural": neural_clean[i][:80],
            })

    total = max(len(conv_clean), len(neural_clean), 1)
    match_pct = matching / total * 100

    return {
        "conv_lines": len(conv_clean),
        "neural_lines": len(neural_clean),
        "matching_lines": matching,
        "match_percentage": match_pct,
        "mismatches": mismatched[:10],  # Show first 10
        "identical": len(mismatched) == 0 and len(conv_clean) == len(neural_clean),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="nCPU Baseline Comparison")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per configuration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks",
        help="Directory for JSON outputs (default: benchmarks/)",
    )
    args = parser.parse_args()
    n_trials = args.trials
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  nCPU Baseline Comparison — Conventional vs Neural-Enhanced GPU OS")
    print("=" * 78)
    print()
    print(f"  Workload:   {len(DEMO_COMMANDS)} shell commands (compile, run, pipes, file I/O)")
    print(f"  Trials:     {n_trials}")
    print()

    # Compile shell once
    print("[setup] Compiling shell binary (one-time)...")
    shell_binary = compile_shell()
    print(f"[setup] Shell binary: {len(shell_binary):,} bytes")

    # Warm the CompileCache for in-shell cc commands (hello.c, fib.c, sieve.c).
    # Without this, the first configuration pays ~500ms GCC overhead that later
    # configs skip via cache hit, creating an unfair measurement.
    print("[setup] Warming compile cache...")
    _warmup = run_conventional(shell_binary)
    print(f"[setup] Cache warm (compile overhead: {_warmup['compile_time_s']:.3f}s absorbed)")
    print()

    # --------------- Run conventional ---------------
    print("=" * 60)
    print("  Configuration A: Conventional (0 neural models)")
    print("=" * 60)

    conv_results = []
    for trial in range(n_trials):
        label = f"  trial {trial+1}/{n_trials}" if n_trials > 1 else "  running"
        print(f"{label}...", end="", flush=True)
        r = run_conventional(shell_binary)
        conv_results.append(r)
        print(f" {r['ips_gpu_only']:,.0f} IPS (GPU-only), {r['gpu_time_s']:.3f}s")

    print()

    # --------------- Run neural ---------------
    print("=" * 60)
    print("  Configuration B: Neural-Enhanced (all models)")
    print("=" * 60)

    neural_results = []
    for trial in range(n_trials):
        label = f"  trial {trial+1}/{n_trials}" if n_trials > 1 else "  running"
        print(f"{label}...", end="", flush=True)
        r = run_neural(shell_binary)
        neural_results.append(r)
        print(f" {r['ips_gpu_only']:,.0f} IPS (GPU-only), {r['neural_inferences']} inferences, {r['gpu_time_s']:.3f}s")

    print()

    # --------------- Average results with confidence intervals ---------------
    def _avg(results, key):
        return sum(r[key] for r in results) / len(results)

    def _std(results, key):
        if len(results) < 2:
            return 0.0
        m = _avg(results, key)
        return math.sqrt(sum((r[key] - m) ** 2 for r in results) / (len(results) - 1))

    def _ci95(results, key):
        n = len(results)
        if n < 2:
            return 0.0
        s = _std(results, key)
        t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                  7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
        t = t_crit.get(n, 1.96)
        return t * s / math.sqrt(n)

    conv_avg = {
        "config": "conventional",
        "total_cycles": int(_avg(conv_results, "total_cycles")),
        "wall_time_s": _avg(conv_results, "wall_time_s"),
        "compile_time_s": _avg(conv_results, "compile_time_s"),
        "gpu_time_s": _avg(conv_results, "gpu_time_s"),
        "ips_raw": _avg(conv_results, "ips_raw"),
        "ips_gpu_only": _avg(conv_results, "ips_gpu_only"),
        "ips_gpu_only_ci95": _ci95(conv_results, "ips_gpu_only"),
        "ips_gpu_only_std": _std(conv_results, "ips_gpu_only"),
        "neural_inferences": 0,
        "models_active": 0,
        "peak_rss_kb": int(_avg(conv_results, "peak_rss_kb")),
    }

    neural_avg = {
        "config": "neural-enhanced",
        "total_cycles": int(_avg(neural_results, "total_cycles")),
        "wall_time_s": _avg(neural_results, "wall_time_s"),
        "compile_time_s": _avg(neural_results, "compile_time_s"),
        "gpu_time_s": _avg(neural_results, "gpu_time_s"),
        "ips_raw": _avg(neural_results, "ips_raw"),
        "ips_gpu_only": _avg(neural_results, "ips_gpu_only"),
        "ips_gpu_only_ci95": _ci95(neural_results, "ips_gpu_only"),
        "ips_gpu_only_std": _std(neural_results, "ips_gpu_only"),
        "neural_inferences": int(_avg(neural_results, "neural_inferences")),
        "models_active": neural_results[0]["models_active"],
        "models_loaded": neural_results[0].get("models_loaded", []),
        "peak_rss_kb": int(_avg(neural_results, "peak_rss_kb")),
    }

    # --------------- Comparison table ---------------
    print()
    print("=" * 78)
    print("  BASELINE COMPARISON RESULTS")
    print("=" * 78)
    print()

    header = f"{'Metric':<30} {'Conventional':>20} {'Neural-Enhanced':>20} {'Delta':>12}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Models Active", f"{conv_avg['models_active']}", f"{neural_avg['models_active']}", f"+{neural_avg['models_active']}"),
        ("Total Cycles", f"{conv_avg['total_cycles']:,}", f"{neural_avg['total_cycles']:,}", ""),
        ("Wall Time (s)", f"{conv_avg['wall_time_s']:.3f}", f"{neural_avg['wall_time_s']:.3f}",
         f"{(neural_avg['wall_time_s'] - conv_avg['wall_time_s']) / conv_avg['wall_time_s'] * 100:+.1f}%"),
        ("Compile Time (s)", f"{conv_avg['compile_time_s']:.3f}", f"{neural_avg['compile_time_s']:.3f}", ""),
        ("GPU Time (s)", f"{conv_avg['gpu_time_s']:.3f}", f"{neural_avg['gpu_time_s']:.3f}",
         f"{(neural_avg['gpu_time_s'] - conv_avg['gpu_time_s']) / conv_avg['gpu_time_s'] * 100:+.1f}%"),
        ("IPS (raw)", f"{conv_avg['ips_raw']:,.0f}", f"{neural_avg['ips_raw']:,.0f}",
         f"{(neural_avg['ips_raw'] - conv_avg['ips_raw']) / conv_avg['ips_raw'] * 100:+.1f}%"),
        ("IPS (GPU-only)", f"{conv_avg['ips_gpu_only']:,.0f}", f"{neural_avg['ips_gpu_only']:,.0f}",
         f"{(neural_avg['ips_gpu_only'] - conv_avg['ips_gpu_only']) / conv_avg['ips_gpu_only'] * 100:+.1f}%"),
        ("IPS 95% CI", f"+/-{conv_avg['ips_gpu_only_ci95']:,.0f}" if conv_avg['ips_gpu_only_ci95'] > 0 else "---",
         f"+/-{neural_avg['ips_gpu_only_ci95']:,.0f}" if neural_avg['ips_gpu_only_ci95'] > 0 else "---", ""),
        ("Neural Inferences", f"{conv_avg['neural_inferences']:,}", f"{neural_avg['neural_inferences']:,}", ""),
        ("Peak RSS (KB)", f"{conv_avg['peak_rss_kb']:,}", f"{neural_avg['peak_rss_kb']:,}",
         f"+{neural_avg['peak_rss_kb'] - conv_avg['peak_rss_kb']:,}"),
    ]

    for label, conv_val, neural_val, delta in rows:
        print(f"{label:<30} {conv_val:>20} {neural_val:>20} {delta:>12}")

    # Overhead summary
    if conv_avg['ips_gpu_only'] > 0:
        overhead = (conv_avg['ips_gpu_only'] - neural_avg['ips_gpu_only']) / conv_avg['ips_gpu_only'] * 100
    else:
        overhead = 0

    print()
    print(f"  Neural overhead (GPU-only IPS): {overhead:.1f}%")
    if n_trials >= 3:
        conv_ci = conv_avg['ips_gpu_only_ci95']
        neur_ci = neural_avg['ips_gpu_only_ci95']
        print(f"  Conventional IPS: {conv_avg['ips_gpu_only']:,.0f} +/- {conv_ci:,.0f} (95% CI, {n_trials} trials)")
        print(f"  Neural IPS:       {neural_avg['ips_gpu_only']:,.0f} +/- {neur_ci:,.0f} (95% CI, {n_trials} trials)")
    print(f"  This is the cost of running {neural_avg['models_active']} neural models as")
    print(f"  side-channel enhancements alongside Metal GPU ARM64 execution.")
    print()

    # --------------- Output correctness ---------------
    print("=" * 78)
    print("  OUTPUT CORRECTNESS CHECK")
    print("=" * 78)
    print()

    # Use the last trial's outputs for comparison
    comparison = compare_outputs(
        conv_results[-1]["output_text"],
        neural_results[-1]["output_text"],
    )

    print(f"  Conventional output lines:  {comparison['conv_lines']}")
    print(f"  Neural output lines:        {comparison['neural_lines']}")
    print(f"  Matching lines:             {comparison['matching_lines']}")
    print(f"  Match percentage:           {comparison['match_percentage']:.1f}%")

    if comparison['identical']:
        print(f"  Verdict:                    IDENTICAL -- neural models do not alter execution")
    else:
        print(f"  Verdict:                    DIVERGED ({len(comparison['mismatches'])} mismatches)")
        if comparison['mismatches']:
            print()
            for m in comparison['mismatches'][:5]:
                print(f"    Line {m['line']}:")
                print(f"      conv:   {m['conventional']}")
                print(f"      neural: {m['neural']}")

    print()

    # --------------- Publication-quality summary ---------------
    print("=" * 78)
    print("  PUBLICATION SUMMARY")
    print("=" * 78)
    print()
    print(f"  The neural-enhanced GPU OS runs the same workload ({len(DEMO_COMMANDS)} commands,")
    print(f"  including in-shell C compilation and execution) with {neural_avg['models_active']} neural")
    print(f"  models active. The neural models add {overhead:.1f}% overhead to GPU-only IPS")
    print(f"  ({conv_avg['ips_gpu_only']:,.0f} -> {neural_avg['ips_gpu_only']:,.0f}), performing")
    print(f"  ~{neural_avg['neural_inferences']:,} neural inferences during the session.")
    print(f"  Output correctness: {comparison['match_percentage']:.0f}% identical lines.")
    print()

    # --------------- Save JSON ---------------
    output_path = output_dir / "baseline_comparison_results.json"
    save_data = {
        "metadata": {
            "trials": n_trials,
            "workload_commands": len(DEMO_COMMANDS),
        },
        "provenance": collect_provenance(
            PROJECT_ROOT,
            argv=[sys.argv[0], *sys.argv[1:]],
            extra={"benchmark": "baseline_comparison"},
        ),
        "conventional": conv_avg,
        "neural_enhanced": neural_avg,
        "overhead_pct": overhead,
        "output_comparison": {k: v for k, v in comparison.items() if k != "mismatches"},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"[output] Results saved to {output_path}")

    return conv_avg, neural_avg, comparison


if __name__ == "__main__":
    main()
