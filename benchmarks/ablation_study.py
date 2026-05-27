#!/usr/bin/env python3
"""
Ablation Study: Measure the impact of each neural model on GPU OS execution.

Runs the EXACT same workload (same shell commands, same programs compiled) under
progressively richer neural configurations:

  0. Baseline   — conventional GPU OS, zero neural models
  1. +Display   — add NeuralDisplayV2 (glyph MLP, 390K params)
  2. +Cache     — add neural LSTM cache replacement + prefetch
  3. +5 models  — add watchdog, GIC, compiler optimizer
  4. All 9      — add syscall predictor, command suggestor, memory analyzer

For each configuration we measure:
  - Wall-clock time and IPS (raw + GPU-only, excluding GCC compile time)
  - Total neural inferences performed
  - Qualitative model impact

The workload is deterministic: the same DEMO_COMMANDS list is fed to stdin.

Usage:
    python benchmarks/ablation_study.py
    python benchmarks/ablation_study.py --trials 3   # average over N trials
"""

import sys
import os
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

# Suppress noisy loggers
logging.getLogger("ncpu.neural.metal_inference").setLevel(logging.ERROR)

from ncpu.os.gpu.runner import (
    compile_c, run, make_syscall_handler, read_string_from_gpu,
    HEAP_BASE,
)
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2
from ncpu.utils.provenance import collect_provenance

# ---------------------------------------------------------------------------
# Shared workload: identical command sequence for every configuration
# ---------------------------------------------------------------------------

DEMO_COMMANDS = [
    # System info
    "cat /etc/motd",
    "cat /etc/os-release",
    # File operations
    "ls /home/user",
    "echo Neural OS ablation test > /tmp/test.txt",
    "cat /tmp/test.txt",
    "wc /tmp/test.txt",
    # Multi-line file creation (exercises sequential writes)
    "echo === System Log === > /tmp/syslog.txt",
    "echo Boot: models loaded >> /tmp/syslog.txt",
    "echo Cache: active >> /tmp/syslog.txt",
    "echo Watchdog: monitoring >> /tmp/syslog.txt",
    "echo GIC: dispatching >> /tmp/syslog.txt",
    "cat /tmp/syslog.txt",
    "wc /tmp/syslog.txt",
    # Pipe operations
    "ls /home/user | grep .c",
    "ls /home/user | sort",
    # Re-read files (exercises cache)
    "cat /etc/motd",
    "cat /etc/os-release",
    "cat /home/user/README.txt",
    # Compilation (triggers compiler advisor if present)
    "cc hello.c",
    "cc fib.c",
    "cc sieve.c",
    "ls /bin",
    # Re-read compiled sources (cache locality)
    "cat /home/user/hello.c",
    "cat /home/user/fib.c",
    "cat /home/user/sieve.c",
    # Directory operations
    "mkdir /tmp/results",
    "echo test1 > /tmp/results/a.txt",
    "echo test2 > /tmp/results/b.txt",
    "echo test3 > /tmp/results/c.txt",
    "ls /tmp/results",
    "cat /tmp/results/a.txt",
    "cat /tmp/results/b.txt",
    # Temporal locality
    "cat /tmp/syslog.txt",
    "ps",
    "echo Ablation test complete.",
    # Execute compiled program (replaces shell — must be last)
    "run /bin/fib",
]


# ---------------------------------------------------------------------------
# Filesystem bootstrap (identical for all configs — from neural_demo.py)
# ---------------------------------------------------------------------------

def bootstrap_filesystem() -> GPUFilesystem:
    """Create the identical filesystem used by neural_demo.py."""
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


# ---------------------------------------------------------------------------
# Demo reader (identical to neural_demo.py)
# ---------------------------------------------------------------------------

def make_demo_reader(commands: list):
    """Create an on_read callback that feeds commands sequentially."""
    idx = [0]
    def reader(fd: int, max_len: int) -> Optional[bytes]:
        if fd != 0:
            return None
        if idx[0] < len(commands):
            cmd = commands[idx[0]]
            idx[0] += 1
            return (cmd + "\n").encode("ascii")[:max_len]
        return b""
    return reader


# ---------------------------------------------------------------------------
# Shell compilation (shared binary — compiled once)
# ---------------------------------------------------------------------------

GPU_OS_DIR = PROJECT_ROOT / "ncpu" / "os" / "gpu"
MODELS_DIR = PROJECT_ROOT / "models"


def compile_shell() -> bytes:
    """Compile the UNIX shell binary once for all configurations."""
    c_file = GPU_OS_DIR / "src" / "arm64_unix_shell.c"
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name
    if not compile_c(str(c_file), bin_path, quiet=True):
        raise RuntimeError("Shell compilation failed")
    binary = Path(bin_path).read_bytes()
    os.unlink(bin_path)
    return binary


# ---------------------------------------------------------------------------
# Configuration runner
# ---------------------------------------------------------------------------

def run_configuration(
    name: str,
    binary: bytes,
    enable_display: bool = False,
    enable_cache: bool = False,
    enable_watchdog: bool = False,
    enable_gic: bool = False,
    enable_compiler: bool = False,
    enable_syscall_pred: bool = False,
    enable_command_sug: bool = False,
    enable_mem_analyzer: bool = False,
    suppress_output: bool = True,
) -> Dict[str, Any]:
    """Run one ablation configuration and return metrics."""
    import torch

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Track model loading
    models_loaded = []
    total_inferences = [0]  # mutable from closures
    compile_time_accum = [0.0]

    # --------------- Load neural models conditionally ---------------

    display = None
    if enable_display:
        try:
            from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
            display = NeuralDisplayV2()
            models_loaded.append("Display")
        except Exception:
            pass

    neural_cache = None
    if enable_cache:
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
            if result.get("replacer", False):
                models_loaded.append("Cache")
            if result.get("prefetcher", False):
                models_loaded.append("Prefetch")
        except Exception:
            neural_cache = None

    watchdog = None
    if enable_watchdog:
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
    if enable_gic:
        try:
            from ncpu.os.neuros.interrupts import NeuralGIC, IRQ_SYSCALL, IRQ_TIMER, IRQ_DISK
            gic = NeuralGIC(device=device)
            if gic.load(str(MODELS_DIR / "os" / "gic.pt")):
                models_loaded.append("GIC")
            else:
                gic = None
        except Exception:
            gic = None

    compiler_opt = None
    if enable_compiler:
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

    # Online models (no .pt)
    syscall_predictor = None
    if enable_syscall_pred:
        from ncpu.os.gpu.neural_demo import NeuralSyscallPredictor
        syscall_predictor = NeuralSyscallPredictor()
        models_loaded.append("SyscallPred")

    command_suggestor = None
    if enable_command_sug:
        from ncpu.os.gpu.neural_demo import NeuralCommandSuggestor
        command_suggestor = NeuralCommandSuggestor()
        models_loaded.append("CmdSuggest")

    # --------------- Set up filesystem ---------------
    fs = bootstrap_filesystem()
    initial_file_count = len(fs.files)

    # --------------- Cache wrapper ---------------
    cache_fs = None
    if neural_cache is not None:
        from ncpu.os.gpu.neural_demo import NeuralCacheFS
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

    # --------------- Memory analyzer ---------------
    mem_analyzer = None
    if enable_mem_analyzer and neural_cache is not None:
        from ncpu.os.gpu.neural_demo import NeuralMemoryAccessAnalyzer
        mem_analyzer = NeuralMemoryAccessAnalyzer(neural_cache)
        models_loaded.append("MemAnalyzer")

    # --------------- Watchdog monitor ---------------
    watchdog_monitor = None
    if watchdog is not None:
        from ncpu.os.gpu.neural_demo import WatchdogMonitor
        watchdog_monitor = WatchdogMonitor(
            watchdog, cache_fs=cache_fs,
            syscall_predictor=syscall_predictor,
            check_interval=50_000,
        )

    # --------------- GIC wrapper ---------------
    gic_wrapper = None
    if gic is not None:
        from ncpu.os.gpu.neural_demo import NeuralGICWrapper
        gic_wrapper = NeuralGICWrapper(gic, device)

    # --------------- Compile advisor ---------------
    compile_advisor = None
    if compiler_opt is not None:
        from ncpu.os.gpu.neural_demo import NeuralCompilationAdvisor
        compile_advisor = NeuralCompilationAdvisor(compiler_opt, device)

    # --------------- Build GPU CPU + handler ---------------
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    def on_exec(bin_path_str: str) -> bool:
        resolved = fs.resolve_path(bin_path_str)
        binary_data = fs.read_file(resolved)
        if binary_data:
            cpu.load_program(binary_data, address=0x10000)
            cpu.set_pc(0x10000)
            return True
        return False

    handler_kwargs = dict(
        filesystem=fs,
        on_exec=on_exec,
        neural_display=display,
        on_read=make_demo_reader(DEMO_COMMANDS),
    )

    base_handler = make_syscall_handler(**handler_kwargs)

    # --------------- Neural wrapper (if any neural models active) ---------------
    from ncpu.os.gpu.neural_demo import DEFAULT_SAMPLING_CONFIG
    _call_count = [0]
    _compile_count = [0]
    _sampling_cfg = DEFAULT_SAMPLING_CONFIG

    def neural_handler(cpu_inst):
        _call_count[0] += 1
        syscall_num = cpu_inst.get_register(8)

        if syscall_predictor:
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

        # Time compilations separately
        from ncpu.os.gpu.runner import SYS_COMPILE
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

    # Choose handler: use neural wrapper if any models are active, else plain
    use_handler = neural_handler if models_loaded else base_handler

    # --------------- Suppress stdout during execution ---------------
    if suppress_output:
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull

    # --------------- Run ---------------
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()

    results = run(
        cpu, use_handler,
        max_cycles=500_000_000,
        quiet=True,
        neural_display=display,
    )

    elapsed = time.perf_counter() - start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if suppress_output:
        sys.stdout = old_stdout
        devnull.close()

    # --------------- Collect metrics ---------------
    total_cycles = results.get("total_cycles", 0)
    raw_ips = results.get("ips", 0)
    gpu_time = max(elapsed - compile_time_accum[0], 0.001)
    gpu_ips = total_cycles / gpu_time

    # Count inferences from cache
    if cache_fs:
        total_inferences[0] += cache_fs.read_count + cache_fs.write_count
    if syscall_predictor:
        total_inferences[0] += syscall_predictor.total_observed
    if command_suggestor:
        total_inferences[0] += command_suggestor.total_commands
    if display:
        total_inferences[0] += 1  # single render

    # Watchdog + final checks
    if watchdog_monitor:
        watchdog_monitor.run_session_checks(total_cycles, 1, n_checks=5)
        total_inferences[0] += watchdog_monitor.watchdog.total_checks

    # Peak RSS delta (in KB on macOS, bytes on Linux)
    rss_delta = rss_after - rss_before

    return {
        "name": name,
        "models_loaded": models_loaded,
        "n_models": len(models_loaded),
        "total_cycles": total_cycles,
        "wall_time_s": elapsed,
        "compile_time_s": compile_time_accum[0],
        "gpu_time_s": gpu_time,
        "ips_raw": raw_ips,
        "ips_gpu_only": gpu_ips,
        "neural_inferences": total_inferences[0],
        "peak_rss_delta_kb": rss_delta,
        "stop_reason": results.get("stop_reason", "unknown"),
        "files_created": len(fs.files) - initial_file_count,
    }


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "baseline (0 models)",
        "desc": "Pure conventional GPU OS — zero neural models",
        "kwargs": {},
    },
    {
        "name": "+display (1 model)",
        "desc": "Add NeuralDisplayV2 — every pixel rendered by glyph MLPs",
        "kwargs": {"enable_display": True},
    },
    {
        "name": "+display +cache (3 models)",
        "desc": "Add neural LSTM cache replacement + prefetch",
        "kwargs": {
            "enable_display": True,
            "enable_cache": True,
        },
    },
    {
        "name": "+5 models (core)",
        "desc": "Add watchdog, GIC, compiler optimizer",
        "kwargs": {
            "enable_display": True,
            "enable_cache": True,
            "enable_watchdog": True,
            "enable_gic": True,
            "enable_compiler": True,
        },
    },
    {
        "name": "all 9 models",
        "desc": "Full neural OS — all models active",
        "kwargs": {
            "enable_display": True,
            "enable_cache": True,
            "enable_watchdog": True,
            "enable_gic": True,
            "enable_compiler": True,
            "enable_syscall_pred": True,
            "enable_command_sug": True,
            "enable_mem_analyzer": True,
        },
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="nCPU Ablation Study")
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
    print("  nCPU Ablation Study — Neural Model Impact on GPU OS Execution")
    print("=" * 78)
    print()
    print(f"  Workload:   {len(DEMO_COMMANDS)} shell commands (compile, run, pipes, file I/O)")
    print(f"  Trials:     {n_trials}")
    print(f"  Configs:    {len(CONFIGS)}")
    print()

    # Compile shell once (warm the cache)
    print("[setup] Compiling shell binary (one-time)...")
    shell_binary = compile_shell()
    print(f"[setup] Shell binary: {len(shell_binary):,} bytes")

    # Warm the CompileCache for in-shell cc commands (hello.c, fib.c, sieve.c).
    # Without this, the first configuration pays ~500ms GCC overhead that later
    # configs skip (cache hit), creating a measurement artifact where the first
    # config appears slower than it really is.
    print("[setup] Warming compile cache (running baseline once)...")
    _warmup = run_configuration("warmup", shell_binary, suppress_output=True)
    print(f"[setup] Compile cache warm ({_warmup['compile_time_s']:.3f}s compile overhead absorbed)")
    print()

    all_results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        desc = cfg["desc"]
        kwargs = cfg["kwargs"]

        trial_results = []
        print(f"[{name}] {desc}")

        for trial in range(n_trials):
            trial_label = f"  trial {trial+1}/{n_trials}" if n_trials > 1 else "  running"
            print(f"{trial_label}...", end="", flush=True)

            result = run_configuration(name, shell_binary, **kwargs)
            trial_results.append(result)
            print(f" {result['ips_gpu_only']:,.0f} IPS (GPU-only), "
                  f"{result['neural_inferences']} inferences, "
                  f"{result['gpu_time_s']:.3f}s GPU time")

        # Average across trials with confidence intervals
        def _mean(vals):
            return sum(vals) / len(vals)

        def _std(vals):
            if len(vals) < 2:
                return 0.0
            m = _mean(vals)
            return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

        def _ci95(vals):
            """95% confidence interval half-width using t-distribution approximation."""
            n = len(vals)
            if n < 2:
                return 0.0
            s = _std(vals)
            # t-distribution critical values for 95% CI (two-tailed)
            t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                      7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
            t = t_crit.get(n, 1.96)  # fall back to z for large n
            return t * s / math.sqrt(n)

        ips_vals = [r["ips_gpu_only"] for r in trial_results]

        avg = {
            "name": name,
            "desc": desc,
            "n_models": trial_results[0]["n_models"],
            "models_loaded": trial_results[0]["models_loaded"],
            "total_cycles": _mean([r["total_cycles"] for r in trial_results]),
            "wall_time_s": _mean([r["wall_time_s"] for r in trial_results]),
            "compile_time_s": _mean([r["compile_time_s"] for r in trial_results]),
            "gpu_time_s": _mean([r["gpu_time_s"] for r in trial_results]),
            "ips_raw": _mean([r["ips_raw"] for r in trial_results]),
            "ips_gpu_only": _mean(ips_vals),
            "ips_gpu_only_std": _std(ips_vals),
            "ips_gpu_only_ci95": _ci95(ips_vals),
            "neural_inferences": _mean([r["neural_inferences"] for r in trial_results]),
            "stop_reason": trial_results[0]["stop_reason"],
            "n_trials": n_trials,
        }
        all_results.append(avg)
        print()

    # --------------- Results table ---------------
    print()
    print("=" * 78)
    print("  ABLATION RESULTS")
    print("=" * 78)
    print()

    # Table header
    header = f"{'Configuration':<30} {'Models':>6} {'GPU IPS':>16} {'95% CI':>14} {'Inferences':>12} {'Overhead':>10} {'GPU Time':>10}"
    print(header)
    print("-" * len(header))

    baseline_ips = all_results[0]["ips_gpu_only"] if all_results else 1

    for r in all_results:
        ips = r["ips_gpu_only"]
        ci = r.get("ips_gpu_only_ci95", 0)
        overhead = ((baseline_ips - ips) / baseline_ips * 100) if baseline_ips > 0 else 0
        overhead_str = f"{overhead:+.1f}%" if r["n_models"] > 0 else "0.0%"
        inferences = int(r["neural_inferences"])
        gpu_time = r["gpu_time_s"]
        ci_str = f"+/-{ci:,.0f}" if ci > 0 else "---"

        print(f"{r['name']:<30} {r['n_models']:>6} {ips:>16,.0f} {ci_str:>14} {inferences:>12,} {overhead_str:>10} {gpu_time:>9.3f}s")

    print()

    # --------------- Detailed per-model impact ---------------
    print("=" * 78)
    print("  PER-MODEL INCREMENTAL IMPACT")
    print("=" * 78)
    print()

    for i in range(1, len(all_results)):
        prev = all_results[i - 1]
        curr = all_results[i]
        delta_ips = curr["ips_gpu_only"] - prev["ips_gpu_only"]
        delta_inf = curr["neural_inferences"] - prev["neural_inferences"]
        new_models = set(curr["models_loaded"]) - set(prev["models_loaded"])
        if new_models:
            models_str = ", ".join(sorted(new_models))
        else:
            models_str = "(online models)"

        print(f"  {prev['name']} -> {curr['name']}")
        print(f"    Added:        {models_str}")
        print(f"    IPS change:   {delta_ips:+,.0f}")
        print(f"    Inferences:   +{int(delta_inf):,}")
        overhead_pct = (abs(delta_ips) / baseline_ips * 100) if baseline_ips > 0 else 0
        print(f"    Overhead:     {overhead_pct:.1f}% of baseline")
        print()

    # --------------- Save JSON results ---------------
    output_path = output_dir / "ablation_results.json"
    serializable = []
    for r in all_results:
        s = dict(r)
        s["total_cycles"] = int(s["total_cycles"])
        s["neural_inferences"] = int(s["neural_inferences"])
        s["ips_gpu_only_ci95"] = round(s.get("ips_gpu_only_ci95", 0), 1)
        s["ips_gpu_only_std"] = round(s.get("ips_gpu_only_std", 0), 1)
        serializable.append(s)

    save_data = {
        "metadata": {
            "trials": n_trials,
            "workload_commands": len(DEMO_COMMANDS),
            "configurations": [cfg["name"] for cfg in CONFIGS],
        },
        "provenance": collect_provenance(
            PROJECT_ROOT,
            argv=[sys.argv[0], *sys.argv[1:]],
            extra={"benchmark": "ablation_study"},
        ),
        "results": serializable,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"[output] Results saved to {output_path}")
    if n_trials >= 3:
        print(f"[output] 95% CI computed from {n_trials} trials (t-distribution)")
    elif n_trials < 3:
        print(f"[note]   Use --trials 3 (or more) for meaningful confidence intervals")

    # Return results for programmatic use
    return all_results


if __name__ == "__main__":
    main()
