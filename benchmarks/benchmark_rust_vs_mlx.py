#!/usr/bin/env python3
"""
Rust Metal vs Python MLX — Side-by-Side ARM64 GPU Kernel Benchmark.

Compares the two Metal GPU backends for nCPU's ARM64 emulator:
  - Rust Metal (kvrm_metal): Rust + Metal, StorageModeShared, zero-copy SVC
  - Python MLX (MLXKernelCPUv2): Python + MLX, double-buffer, shadow numpy

Both backends run identical ARM64 programs on the Metal GPU and should produce
identical register states. This benchmark measures IPS (instructions per second)
and reports the speedup ratio.

Usage:
    python3 benchmarks/benchmark_rust_vs_mlx.py
    python3 benchmarks/benchmark_rust_vs_mlx.py --json-only
"""

import argparse
import contextlib
import io
import json
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = Path("/tmp/benchmark_rust_vs_mlx.json")
MEMORY_SIZE = 64 * 1024  # 64 KB


# =============================================================================
# ARM64 Instruction Encoders
# =============================================================================

def movz_x(rd: int, imm16: int) -> bytes:
    """MOVZ Xd, #imm16"""
    return struct.pack('<I', 0xD2800000 | (imm16 << 5) | rd)


def movz_w(rd: int, imm16: int) -> bytes:
    """MOVZ Wd, #imm16"""
    return struct.pack('<I', 0x52800000 | (imm16 << 5) | rd)


def subs_x_imm(rd: int, rn: int, imm12: int) -> bytes:
    """SUBS Xd, Xn, #imm12"""
    return struct.pack('<I', 0xF1000000 | (imm12 << 10) | (rn << 5) | rd)


def subs_w_imm(rd: int, rn: int, imm12: int) -> bytes:
    """SUBS Wd, Wn, #imm12"""
    return struct.pack('<I', 0x71000000 | (imm12 << 10) | (rn << 5) | rd)


def add_x_reg(rd: int, rn: int, rm: int) -> bytes:
    """ADD Xd, Xn, Xm"""
    return struct.pack('<I', 0x8B000000 | (rm << 16) | (rn << 5) | rd)


def mov_x_reg(rd: int, rn: int) -> bytes:
    """MOV Xd, Xn (alias: ORR Xd, XZR, Xn)"""
    return struct.pack('<I', 0xAA000000 | (rn << 16) | (31 << 5) | rd)


def b_ne(offset: int) -> bytes:
    """B.NE offset (offset in instructions from this instruction)"""
    return struct.pack('<I', 0x54000001 | ((offset & 0x7FFFF) << 5))


def cmp_x_zero(rn: int) -> bytes:
    """CMP Xn, #0 (alias: SUBS XZR, Xn, #0)"""
    return struct.pack('<I', 0xF100001F | (rn << 5))


def hlt() -> bytes:
    """HLT #0"""
    return struct.pack('<I', 0xD4400000)


def add_x_imm(rd: int, rn: int, imm12: int) -> bytes:
    """ADD Xd, Xn, #imm12"""
    return struct.pack('<I', 0x91000000 | (imm12 << 10) | (rn << 5) | rd)


def subs_x_reg(rd: int, rn: int, rm: int) -> bytes:
    """SUBS Xd, Xn, Xm"""
    return struct.pack('<I', 0xEB000000 | (rm << 16) | (rn << 5) | rd)


def str_x_imm0(rt: int, rn: int) -> bytes:
    """STR Xt, [Xn, #0] (unsigned offset, imm12=0)"""
    return struct.pack('<I', 0xF9000000 | (rn << 5) | rt)


def ldr_x_imm0(rt: int, rn: int) -> bytes:
    """LDR Xt, [Xn, #0] (unsigned offset, imm12=0)"""
    return struct.pack('<I', 0xF9400000 | (rn << 5) | rt)


def svc_0() -> bytes:
    """SVC #0"""
    return struct.pack('<I', 0xD4000001)


# Memory size for SVC-heavy benchmarks (need SVC buffer at 0x3F0000 + heap at 0x60000)
SVC_MEMORY_SIZE = 16 * 1024 * 1024  # 16 MB


# =============================================================================
# Benchmark Programs
# =============================================================================

def build_fibonacci_20() -> tuple[bytes, str, int]:
    """
    Compute fibonacci(20) = 6765.

    X0 = n (20)
    X1 = fib_prev (starts 0)
    X2 = fib_curr (starts 1)
    X3 = temp

    Loop:
      X3 = X1 + X2       (temp = prev + curr)
      X1 = X2             (prev = curr)
      X2 = X3             (curr = temp)
      X0 = X0 - 1         (n--)
      CMP X0, #0
      B.NE loop

    Result: X2 = fib(20) = 6765 (but fib(20) with this algo where we start
    with fib_prev=0, fib_curr=1 and loop 20 times gives fib(21)=10946 in X2
    and fib(20)=6765 in X1). We return X1.
    """
    program = b''
    program += movz_x(0, 20)       # X0 = 20 (counter)
    program += movz_x(1, 0)        # X1 = 0 (fib_prev)
    program += movz_x(2, 1)        # X2 = 1 (fib_curr)
    # loop (offset 3):
    program += add_x_reg(3, 1, 2)  # X3 = X1 + X2
    program += mov_x_reg(1, 2)     # X1 = X2
    program += mov_x_reg(2, 3)     # X2 = X3
    program += subs_x_imm(0, 0, 1) # X0 = X0 - 1, set flags
    # B.NE back to loop: offset = -4 instructions
    offset_ne = -4 & 0x7FFFF       # Two's complement 19-bit
    program += b_ne(offset_ne)     # B.NE loop
    program += hlt()               # Halt
    return program, "fibonacci", 6765, "X1", None


def build_countdown() -> tuple[bytes, str, int]:
    """
    Count down from 1,000,000 to 0.

    X0 = 1,000,000
    Loop:
      X0 = X0 - 1
      CMP X0, #0
      B.NE loop
    HLT

    1,000,000 exceeds MOVZ imm16 range (max 65535), so we build it:
      MOVZ X0, #1000       (X0 = 1000)
      MOVZ X1, #1000       (X1 = 1000)
      MUL X0, X0, X1       -- but MUL encoding is complex.

    Simpler: use MOVZ with shift. MOVZ Xd, #imm16, LSL #16 = 0xD2A00000.
    1,000,000 = 0xF4240 = 0x000F << 16 | 0x4240
      MOVZ X0, #0x4240
      MOVK X0, #0x000F, LSL #16

    MOVK Xd, #imm16, LSL #16: 0xF2A00000 | (imm16 << 5) | rd

    Actually let's just count down from 50,000 to keep it simple and fast.
    50,000 fits in MOVZ imm16.
    """
    count = 50_000
    program = b''
    program += movz_x(0, count)        # X0 = 50,000
    # loop (offset 1):
    program += subs_x_imm(0, 0, 1)     # X0 = X0 - 1, set flags
    # B.NE back to loop: offset = -1 instruction
    offset_ne = -1 & 0x7FFFF
    program += b_ne(offset_ne)         # B.NE loop
    program += hlt()                   # Halt
    return program, "countdown", 0, "X0", None


def build_nested_loop() -> tuple[bytes, str, int]:
    """
    Nested loop: outer 500 x inner 500 = 250,000 iterations.
    Increments X2 each inner iteration.

    X0 = 500 (outer counter)
    outer_loop:
      X1 = 500 (inner counter)
    inner_loop:
      X2 = X2 + 1
      X1 = X1 - 1
      CMP X1, #0
      B.NE inner_loop
      X0 = X0 - 1
      CMP X0, #0
      B.NE outer_loop
    HLT

    Result: X2 = 250,000
    """
    program = b''
    program += movz_x(0, 500)           # 0: X0 = 500
    program += movz_x(2, 0)             # 1: X2 = 0
    # outer_loop (offset 2):
    program += movz_x(1, 500)           # 2: X1 = 500
    # inner_loop (offset 3):
    # ADD X2, X2, #1
    program += struct.pack('<I', 0x91000000 | (1 << 10) | (2 << 5) | 2)
    program += subs_x_imm(1, 1, 1)     # 4: X1 = X1 - 1
    # B.NE inner_loop: offset = -2
    program += b_ne(-2 & 0x7FFFF)      # 5: B.NE inner_loop
    program += subs_x_imm(0, 0, 1)     # 6: X0 = X0 - 1
    # B.NE outer_loop: offset = -5
    program += b_ne(-5 & 0x7FFFF)      # 7: B.NE outer_loop
    program += hlt()                    # 8: Halt

    return program, "nested_loop", 250_000, "X2", None


def build_memory_intensive():
    """
    Store 1000 values to memory starting at 0x2000, then load them all back
    and sum them. Exercises the memory subsystem where Rust's StorageModeShared
    zero-copy advantage over MLX's double-buffer pattern is most pronounced.

    Store loop:
      X0 = counter (0..999)
      X1 = write pointer (starts at 0x2000, incremented by 8)
      X2 = 1000 (limit)
      X3 = running sum

      STR X0, [X1]           store counter value
      ADD X3, X3, X0         sum += counter
      ADD X1, X1, #8         pointer += 8
      ADD X0, X0, #1         counter++
      SUBS X7, X0, X2        set flags: counter - limit
      B.NE store_loop

    Load loop:
      X0 = counter (reset to 0)
      X1 = read pointer (reset to 0x2000)
      X4 = loaded sum

      LDR X5, [X1]           load value
      ADD X4, X4, X5         sum += loaded value
      ADD X1, X1, #8         pointer += 8
      ADD X0, X0, #1         counter++
      SUBS X7, X0, X2        set flags
      B.NE load_loop

    Expected: X3 = X4 = sum(0..999) = 499,500
    """
    program = b''
    # --- Store loop setup ---
    program += movz_x(0, 0)              # 0: X0 = 0 (counter)
    program += movz_x(1, 0x2000)         # 1: X1 = 0x2000 (write pointer)
    program += movz_x(2, 1000)           # 2: X2 = 1000 (limit)
    program += movz_x(3, 0)             # 3: X3 = 0 (running sum)
    # --- store_loop (offset 4) ---
    program += str_x_imm0(0, 1)          # 4: STR X0, [X1]
    program += add_x_reg(3, 3, 0)        # 5: X3 += X0
    program += add_x_imm(1, 1, 8)        # 6: X1 += 8
    program += add_x_imm(0, 0, 1)        # 7: X0 += 1
    program += subs_x_reg(7, 0, 2)       # 8: X7 = X0 - X2, set flags
    program += b_ne(-5 & 0x7FFFF)        # 9: B.NE store_loop (back to offset 4)
    # --- Load loop setup ---
    program += movz_x(0, 0)              # 10: X0 = 0 (counter)
    program += movz_x(1, 0x2000)         # 11: X1 = 0x2000 (read pointer)
    program += movz_x(4, 0)             # 12: X4 = 0 (loaded sum)
    # --- load_loop (offset 13) ---
    program += ldr_x_imm0(5, 1)          # 13: LDR X5, [X1]
    program += add_x_reg(4, 4, 5)        # 14: X4 += X5
    program += add_x_imm(1, 1, 8)        # 15: X1 += 8
    program += add_x_imm(0, 0, 1)        # 16: X0 += 1
    program += subs_x_reg(7, 0, 2)       # 17: X7 = X0 - X2, set flags
    program += b_ne(-5 & 0x7FFFF)        # 18: B.NE load_loop (back to offset 13)
    program += hlt()                     # 19: Halt

    return program, "mem_intensive", 499_500, "X3", None


def build_brk_loop():
    """
    200 SYS_BRK system calls via SVC #0.

    SYS_BRK (214) is handled ON-GPU by both the Rust Metal and Python MLX
    backends, but the round-trip cost differs dramatically:
      - Rust: StorageModeShared (zero-copy), SVC handled in-shader
      - MLX:  double-buffer pattern, shadow numpy arrays, per-SVC sync overhead

    This benchmark isolates the SVC dispatch path performance.

    Logic:
      X9 = 200             (loop counter)
      X0 = 0               (get current brk)
      X8 = 214             (SYS_BRK)
      SVC #0               (initial brk query)
      loop:
        ADD X0, X0, #256   (grow heap by 256 bytes)
        X8 = 214           (SYS_BRK)
        SVC #0
        SUBS X9, X9, #1
        B.NE loop
      HLT

    Requires 16 MB memory (SVC buffer at 0x3F0000).
    """
    program = b''
    program += movz_x(9, 200)             # 0: X9 = 200 (counter)
    program += movz_x(0, 0)               # 1: X0 = 0 (get current brk)
    program += movz_x(8, 214)             # 2: X8 = SYS_BRK
    program += svc_0()                     # 3: SVC #0 (query brk)
    # --- loop (offset 4) ---
    program += add_x_imm(0, 0, 256)       # 4: X0 += 256
    program += movz_x(8, 214)             # 5: X8 = SYS_BRK
    program += svc_0()                     # 6: SVC #0
    program += subs_x_imm(9, 9, 1)        # 7: X9--
    program += b_ne(-4 & 0x7FFFF)         # 8: B.NE loop (back to offset 4)
    program += hlt()                      # 9: Halt

    def setup(cpu):
        if hasattr(cpu, 'init_svc_buffer'):
            cpu.init_svc_buffer()

    return program, "brk_200_svc", 0, "X9", setup


def build_svc_write_loop():
    """
    50 SYS_WRITE system calls writing "X\\n" to stdout via SVC #0.

    SYS_WRITE to stdout (fd=1) is handled ON-GPU by both backends via
    GPU-side SVC buffering, but the cost profile differs:
      - Rust: StorageModeShared, write buffer accessed via shared pointer
      - MLX:  shadow numpy array, buffer synced at dispatch boundaries

    Requires pre-writing "X\\n" at address 0x1000 in memory before execution.
    Requires 16 MB memory (SVC buffer at 0x3F0000).

    Logic:
      X9 = 50              (loop counter)
      X7 = 0x1000          (buffer address of "X\\n")
      loop:
        X8 = 64            (SYS_WRITE)
        X0 = 1             (fd = stdout)
        X1 = X7            (buf)
        X2 = 2             (len)
        SVC #0
        SUBS X9, X9, #1
        B.NE loop
      HLT
    """
    program = b''
    program += movz_x(9, 50)              # 0: X9 = 50 (counter)
    program += movz_x(7, 0x1000)          # 1: X7 = 0x1000 (buf addr)
    # --- loop (offset 2) ---
    program += movz_x(8, 64)              # 2: X8 = SYS_WRITE
    program += movz_x(0, 1)               # 3: X0 = 1 (stdout)
    program += mov_x_reg(1, 7)            # 4: X1 = X7 (buf)
    program += movz_x(2, 2)               # 5: X2 = 2 (len)
    program += svc_0()                     # 6: SVC #0
    program += subs_x_imm(9, 9, 1)        # 7: X9--
    program += b_ne(-6 & 0x7FFFF)         # 8: B.NE loop (back to offset 2)
    program += hlt()                      # 9: Halt

    def setup(cpu):
        cpu.write_memory(0x1000, b"X\n")
        if hasattr(cpu, 'init_svc_buffer'):
            cpu.init_svc_buffer()

    return program, "svc_write_50", 0, "X9", setup


PROGRAMS = [
    build_fibonacci_20,
    build_countdown,
    build_nested_loop,
    build_memory_intensive,
    build_brk_loop,
    build_svc_write_loop,
]


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_rust(program: bytes, max_cycles: int = 2_000_000,
             memory_size: int = MEMORY_SIZE, setup_fn=None) -> dict:
    """Run program on Rust Metal backend and return timing results."""
    from kernels.mlx.rust_runner import RustMetalCPU

    cpu = RustMetalCPU(memory_size=memory_size, quiet=True)
    cpu.load_program(program, address=0)
    cpu.set_pc(0)

    if setup_fn is not None:
        setup_fn(cpu)

    result = cpu.execute(max_cycles=max_cycles)

    regs = {}
    for i in range(10):
        val = cpu.get_register(i)
        if val < 0:
            val += (1 << 64)
        regs[f"X{i}"] = val

    return {
        "cycles": result.cycles,
        "elapsed_s": result.elapsed_seconds,
        "ips": result.ips,
        "stop_reason": result.stop_reason_name,
        "registers": regs,
    }


def run_mlx(program: bytes, max_cycles: int = 2_000_000,
            memory_size: int = MEMORY_SIZE, setup_fn=None) -> dict:
    """Run program on Python MLX backend and return timing results."""
    from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2

    # Suppress MLX load_program print output
    with contextlib.redirect_stdout(io.StringIO()):
        cpu = MLXKernelCPUv2(memory_size=memory_size, quiet=True)
        cpu.load_program(program, address=0)
    cpu.set_pc(0)

    if setup_fn is not None:
        setup_fn(cpu)

    result = cpu.execute(max_cycles=max_cycles)

    regs = {}
    for i in range(10):
        val = cpu.get_register(i)
        if val < 0:
            val += (1 << 64)
        regs[f"X{i}"] = val

    return {
        "cycles": result.cycles,
        "elapsed_s": result.elapsed_seconds,
        "ips": result.ips,
        "stop_reason": result.stop_reason_name,
        "registers": regs,
    }


def format_ips(ips: float) -> str:
    """Format IPS with commas and appropriate suffix."""
    if ips >= 1_000_000:
        return f"{ips:,.0f}"
    elif ips >= 1_000:
        return f"{ips:,.0f}"
    else:
        return f"{ips:,.1f}"


def measure_compile_speed(json_only: bool = False) -> dict:
    """
    Measure Metal shader / Rust library initialization time.

    This captures the one-time cost of creating a new CPU instance, which
    includes Metal shader compilation (MLX) or Rust + Metal pipeline setup.
    """
    if not json_only:
        print(f"  [compile_speed] Measuring backend initialization time...")

    # Rust Metal init
    t0 = time.perf_counter()
    from kernels.mlx.rust_runner import RustMetalCPU
    _cpu_r = RustMetalCPU(memory_size=MEMORY_SIZE, quiet=True)
    rust_time = time.perf_counter() - t0
    del _cpu_r

    if not json_only:
        print(f"  [compile_speed] Rust Metal init: {rust_time:.4f}s")

    # Python MLX init (suppress output)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        _cpu_m = MLXKernelCPUv2(memory_size=MEMORY_SIZE, quiet=True)
    mlx_time = time.perf_counter() - t0
    del _cpu_m

    if not json_only:
        print(f"  [compile_speed] Python MLX init: {mlx_time:.4f}s")

    speedup = mlx_time / rust_time if rust_time > 0 else float("inf")

    if not json_only:
        print(f"  [compile_speed] Rust is {speedup:.1f}x faster to initialize")
        print()

    return {
        "program": "compile_speed",
        "rust_elapsed_s": rust_time,
        "mlx_elapsed_s": mlx_time,
        "speedup": speedup,
        "correct": True,
        "note": "Measures CPU instance creation (shader compilation / Rust init)",
    }


def run_benchmarks(json_only: bool = False) -> dict:
    """Run all benchmarks and return results dict."""
    results = []

    if not json_only:
        print()
        print("=" * 72)
        print("  Rust Metal vs Python MLX -- ARM64 GPU Kernel Benchmark")
        print("=" * 72)
        print()

    for builder in PROGRAMS:
        built = builder()
        program, name, expected, check_reg, setup_fn = built

        # Determine memory size: SVC programs need 16 MB
        needs_svc = setup_fn is not None
        mem_size = SVC_MEMORY_SIZE if needs_svc else MEMORY_SIZE

        if not json_only:
            print(f"  [{name}] Assembling {len(program)} bytes "
                  f"({len(program) // 4} instructions)"
                  f"{' [SVC, 16MB]' if needs_svc else ''}...")

        # -- Warmup: run once on each backend to JIT / compile kernels --
        if not json_only:
            print(f"  [{name}] Warming up backends...")
        _ = run_rust(program, max_cycles=100, memory_size=mem_size,
                     setup_fn=setup_fn)
        _ = run_mlx(program, max_cycles=100, memory_size=mem_size,
                    setup_fn=setup_fn)

        # -- Benchmark runs --
        if not json_only:
            print(f"  [{name}] Running Rust Metal...", end="", flush=True)
        rust = run_rust(program, memory_size=mem_size, setup_fn=setup_fn)
        if not json_only:
            print(f" {rust['cycles']:,} cycles in {rust['elapsed_s']:.4f}s")

        if not json_only:
            print(f"  [{name}] Running Python MLX...", end="", flush=True)
        mlx = run_mlx(program, memory_size=mem_size, setup_fn=setup_fn)
        if not json_only:
            print(f" {mlx['cycles']:,} cycles in {mlx['elapsed_s']:.4f}s")

        # -- Verify correctness --
        correct = True
        if expected is not None:
            rust_val = rust["registers"].get(check_reg, None)
            mlx_val = mlx["registers"].get(check_reg, None)

            if rust_val != expected:
                correct = False
                if not json_only:
                    print(f"  *** MISMATCH: Rust {check_reg}={rust_val}, "
                          f"expected {expected}")
            if mlx_val != expected:
                correct = False
                if not json_only:
                    print(f"  *** MISMATCH: MLX {check_reg}={mlx_val}, "
                          f"expected {expected}")

        speedup = rust["ips"] / mlx["ips"] if mlx["ips"] > 0 else float("inf")

        results.append({
            "program": name,
            "rust_cycles": rust["cycles"],
            "rust_elapsed_s": rust["elapsed_s"],
            "rust_ips": rust["ips"],
            "rust_stop": rust["stop_reason"],
            "mlx_cycles": mlx["cycles"],
            "mlx_elapsed_s": mlx["elapsed_s"],
            "mlx_ips": mlx["ips"],
            "mlx_stop": mlx["stop_reason"],
            "speedup": speedup,
            "correct": correct,
            "expected": expected,
            "check_reg": check_reg,
            "rust_regs": rust["registers"],
            "mlx_regs": mlx["registers"],
        })

        if not json_only:
            print()

    # -- Compile speed (special, not an ARM64 program) --
    compile_result = measure_compile_speed(json_only=json_only)
    compile_results = [compile_result]

    # -- Summary table --
    if not json_only:
        print_table(results, compile_results)

    return {
        "benchmarks": results,
        "compile_speed": compile_result,
        "memory_size": MEMORY_SIZE,
        "svc_memory_size": SVC_MEMORY_SIZE,
    }


def print_table(results: list[dict], compile_results: list[dict] = None) -> None:
    """Print a formatted comparison table."""
    # Column widths
    name_w = max(len(r["program"]) for r in results)
    name_w = max(name_w, len("Program"))

    rust_ips_strs = [format_ips(r["rust_ips"]) for r in results]
    mlx_ips_strs = [format_ips(r["mlx_ips"]) for r in results]
    speedup_strs = [f"{r['speedup']:.1f}x" for r in results]
    check_strs = ["PASS" if r["correct"] else "FAIL" for r in results]

    rust_w = max(len(s) for s in rust_ips_strs)
    rust_w = max(rust_w, len("Rust (IPS)"))
    mlx_w = max(len(s) for s in mlx_ips_strs)
    mlx_w = max(mlx_w, len("MLX (IPS)"))
    speed_w = max(len(s) for s in speedup_strs)
    speed_w = max(speed_w, len("Speedup"))
    check_w = max(len(s) for s in check_strs)
    check_w = max(check_w, len("Check"))

    # Build table
    def sep(left, mid, right, fill):
        return (f"{left}{fill * (name_w + 2)}{mid}{fill * (rust_w + 2)}"
                f"{mid}{fill * (mlx_w + 2)}{mid}{fill * (speed_w + 2)}"
                f"{mid}{fill * (check_w + 2)}{right}")

    print()
    print(sep("\u250C", "\u252C", "\u2510", "\u2500"))
    print(f"\u2502 {'Program':<{name_w}} "
          f"\u2502 {'Rust (IPS)':>{rust_w}} "
          f"\u2502 {'MLX (IPS)':>{mlx_w}} "
          f"\u2502 {'Speedup':>{speed_w}} "
          f"\u2502 {'Check':>{check_w}} \u2502")
    print(sep("\u251C", "\u253C", "\u2524", "\u2500"))

    for i, r in enumerate(results):
        print(f"\u2502 {r['program']:<{name_w}} "
              f"\u2502 {rust_ips_strs[i]:>{rust_w}} "
              f"\u2502 {mlx_ips_strs[i]:>{mlx_w}} "
              f"\u2502 {speedup_strs[i]:>{speed_w}} "
              f"\u2502 {check_strs[i]:>{check_w}} \u2502")

    print(sep("\u2514", "\u2534", "\u2518", "\u2500"))

    # Summary
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    all_correct = all(r["correct"] for r in results)
    print()
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print(f"  Correctness:     {'ALL PASS' if all_correct else 'FAILURES DETECTED'}")

    # Compile speed table (separate)
    if compile_results:
        cr = compile_results[0]
        print()
        print("  Backend Initialization (Metal shader / Rust pipeline):")
        print(f"    Rust Metal: {cr['rust_elapsed_s']:.4f}s")
        print(f"    Python MLX: {cr['mlx_elapsed_s']:.4f}s")
        print(f"    Speedup:    {cr['speedup']:.1f}x faster (Rust)")

    print()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Rust Metal vs Python MLX ARM64 GPU kernels"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Suppress table output, print only JSON",
    )
    args = parser.parse_args()

    results = run_benchmarks(json_only=args.json_only)

    # Save JSON
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    if args.json_only:
        print(json.dumps(results, indent=2))
    else:
        print(f"  Results saved to {RESULTS_PATH}")
        print()
